//! Pass/fail evaluation of scenario results.
//!
//! Key types:
//! - [`AssertResult`] -- pass/fail status with diagnostics and statistics
//! - [`Assert`] -- composable assertion config (worker + monitor checks)
//! - [`ScenarioStats`] / [`CgroupStats`] -- aggregated telemetry
//! - [`NumaMapsEntry`] -- parsed `/proc/self/numa_maps` VMA entry
//! - [`Verdict`] -- pointwise-claim accumulator (built via
//!   [`Assert::verdict`] / [`Verdict::new`]; comparators routed through
//!   [`ClaimBuilder`] / [`SetClaim`] / [`SeqClaim`])
//!
//! NUMA assertion functions:
//! - [`parse_numa_maps`] -- parse numa_maps content into per-VMA entries
//! - [`page_locality`] -- compute page locality fraction from entries
//! - [`parse_vmstat_numa_pages_migrated`] -- extract vmstat migration counter
//! - [`assert_page_locality`] / [`assert_cross_node_migration`] -- threshold checks
//!
//! Assertion uses a three-layer merge: [`Assert::default_checks()`] ->
//! `Scheduler.assert` -> per-test `assert`.
//!
//! # Statistical conventions
//!
//! - **Percentiles / medians**: nearest-rank (see [`percentile`]),
//!   value at index `ceil(n * p) - 1`. Unlike interpolated
//!   percentiles, every reported p99 is an actual observed sample,
//!   not a synthetic midpoint. Consistent across every
//!   [`CgroupStats`] and [`ScenarioStats`] latency field.
//! - **CV (coefficient of variation)** is stddev/mean computed over
//!   the pooled latency samples, not as a mean of per-worker CVs —
//!   see [`CgroupStats::wake_latency_cv`] for the masking caveat.
//!
//! See the [Checking](https://likewhatevs.github.io/ktstr/guide/concepts/checking.html)
//! chapter of the guide.

use crate::workload::WorkerReport;
use std::collections::{BTreeMap, BTreeSet};

/// Per-VMA entry parsed from `/proc/self/numa_maps`.
#[derive(Debug, Clone, Default)]
pub struct NumaMapsEntry {
    /// Virtual address of the VMA.
    pub addr: u64,
    /// Per-node page counts (node_id -> page_count).
    pub node_pages: BTreeMap<usize, u64>,
}

/// Parse `/proc/self/numa_maps` content into per-VMA entries.
///
/// Each line has the format:
///   `<hex_addr> <policy> [key=val ...]`
/// where per-node page counts appear as `N<node>=<count>`.
pub fn parse_numa_maps(content: &str) -> Vec<NumaMapsEntry> {
    let mut entries = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let addr = match parts.next().and_then(|s| u64::from_str_radix(s, 16).ok()) {
            Some(a) => a,
            None => continue,
        };
        // Skip policy field.
        let _ = parts.next();

        let mut entry = NumaMapsEntry {
            addr,
            ..Default::default()
        };

        for token in parts {
            if let Some(rest) = token.strip_prefix('N')
                && let Some((node_str, count_str)) = rest.split_once('=')
                && let (Ok(node), Ok(count)) = (node_str.parse::<usize>(), count_str.parse::<u64>())
            {
                *entry.node_pages.entry(node).or_insert(0) += count;
            }
        }

        if !entry.node_pages.is_empty() {
            entries.push(entry);
        }
    }
    entries
}

/// Compute page locality fraction from parsed numa_maps entries.
///
/// Returns the fraction of pages residing on any node in
/// `expected_nodes` (0.0-1.0). Returns 0.0 when no pages are observed
/// — a zero-allocation workload is not vacuously local; reporting 1.0
/// would let `min_page_locality` thresholds silently pass on broken
/// runs that produced no NUMA signal. The expected node set is
/// derived from the worker's
/// [`MemPolicy`](crate::workload::MemPolicy) at evaluation time.
pub fn page_locality(entries: &[NumaMapsEntry], expected_nodes: &BTreeSet<usize>) -> f64 {
    let mut total: u64 = 0;
    let mut local: u64 = 0;
    for entry in entries {
        for (&node, &count) in &entry.node_pages {
            total += count;
            if expected_nodes.contains(&node) {
                local += count;
            }
        }
    }
    if total > 0 {
        local as f64 / total as f64
    } else {
        0.0
    }
}

/// Extract `numa_pages_migrated` from `/proc/vmstat` content.
///
/// Returns `None` if the counter is not present. The counter is
/// cumulative; callers diff pre- and post-workload snapshots to
/// get migration count during the test.
pub fn parse_vmstat_numa_pages_migrated(content: &str) -> Option<u64> {
    for line in content.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("numa_pages_migrated") {
            let rest = rest.trim();
            if let Ok(v) = rest.parse::<u64>() {
                return Some(v);
            }
        }
    }
    None
}

fn gap_threshold_ms() -> u64 {
    // Unoptimized debug builds have higher scheduling overhead.
    if cfg!(debug_assertions) { 3000 } else { 2000 }
}

fn spread_threshold_pct() -> f64 {
    // Debug builds in small VMs (especially under EEVDF) show higher
    // spread than optimized builds under sched_ext schedulers.
    if cfg!(debug_assertions) { 35.0 } else { 15.0 }
}

/// Category tag for an [`AssertDetail`]. Enables structural filtering
/// (e.g. by [`AssertPlan`]) without matching on substrings of
/// human-readable messages, which is fragile if wording changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DetailKind {
    /// A worker made zero progress.
    Starved,
    /// A worker was stuck off-CPU longer than the gap threshold.
    Stuck,
    /// Spread between best and worst worker exceeded the fairness threshold.
    Unfair,
    /// A worker ran on a CPU outside its expected cpuset.
    Isolation,
    /// Throughput / benchmarking threshold failure (p99, CV, rate).
    Benchmark,
    /// Migration-ratio threshold failure (migrations per iteration).
    Migration,
    /// NUMA page locality threshold failure.
    PageLocality,
    /// Cross-node migration threshold failure.
    CrossNodeMigration,
    /// Slow-tier (memory tier) threshold failure.
    SlowTier,
    /// Monitor-subsystem anomaly (imbalance, DSQ depth, rq_clock stall).
    /// Use `DetailKind::SchedulerDied` for scheduler-liveness failures.
    Monitor,
    /// Scheduler process observed to have died (via `sched_pid`
    /// probe returning ESRCH or wait on the leader). Covers
    /// post-ops liveness probes and inter-step liveness checks;
    /// the vocabulary was unified from "exited" / "no longer
    /// running" onto "died" so every scheduler-liveness failure
    /// lands under a single structural tag. Consumers filter on
    /// this variant directly — `test_support::eval`'s console-dump
    /// gate matches on `kind == SchedulerDied` rather than
    /// scanning message text.
    SchedulerDied,
    /// SCX event-counter threshold failure. An error-class
    /// `SCX_EV_*` counter (e.g. `enq_skip_exiting`,
    /// `enq_skip_migration_disabled`, `dispatch_offline`) crossed
    /// the configured bound. Distinct from
    /// [`DetailKind::SchedulerDied`] (process-liveness) and
    /// [`DetailKind::Monitor`] (imbalance / DSQ-depth /
    /// rq_clock-stall): this kind flags individual event-counter
    /// regressions surfaced by [`assert_scx_events_clean`]. The
    /// counters themselves originate in the kernel's per-task
    /// `scx_event_stats` (see `kernel/sched/ext.c` —
    /// `SCX_EV_*` macros); ktstr reads aggregated deltas via
    /// `monitor::ScxEventDeltas` and presents them to the
    /// assertion as `(name, count)` pairs.
    SchedulerEvent,
    /// Temporal assertion failure on a periodic-capture
    /// [`SampleSeries`](crate::scenario::sample::SampleSeries).
    /// One of the six built-in patterns
    /// (`nondecreasing` / `strictly_increasing`, `rate_within`,
    /// `steady_within`, `converges_to`, `always_true`,
    /// `ratio_within`) or a per-sample scalar comparator
    /// invoked via `.each(...)` reported a violation. The
    /// detail message names the pattern, the offending sample
    /// tag(s), and the observed-vs-expected values; the
    /// stdout `--- temporal assertions ---` summary in
    /// `test_support::output` aggregates the same kind into
    /// per-assertion pass/fail rows.
    Temporal,
    /// Skip notification (scenario could not run under this topology/flags).
    Skip,
    /// Informational annotation that does NOT contribute to the
    /// failure verdict. Use when a scenario wants to surface
    /// observed values, environment context, or measured numbers
    /// alongside the verdict — e.g. "max_wchar=12345" attached to
    /// a passing IO_ACCOUNTING reachability check, or "psi.cpu.some
    /// total_usec=0 (kernel did not accumulate)" surfaced from a
    /// pass that intentionally allows the zero case.
    ///
    /// Producers should keep `AssertResult::passed = true` when
    /// adding a `Note` detail; the kind is purely structural so
    /// downstream consumers (sidecar parsers, stats tooling, the
    /// `evaluate_vm_result` failure path) can filter informational
    /// rows from genuine failures without scanning message text.
    /// `AssertResult::note` and `AssertResult::with_note` are the
    /// note-emitting helpers — neither enforces the invariant via
    /// debug_assert or otherwise; they just append the detail and
    /// leave `passed` untouched. Hand-constructed results that
    /// flip `passed = false` while pushing a `Note` detail still
    /// produce a valid (failing) result, but the kind is meant
    /// to read as "context", not "what failed".
    Note,
    /// Uncategorized — falls through when a detail has no specific kind.
    Other,
}

/// Message prefix emitted by every scenario-runner site that
/// detects the scheduler process has died — whether through a
/// post-ops liveness probe or an inter-step liveness check. Both
/// paths share this single prefix as the operator-visible
/// message format so someone grepping stderr for the canonical
/// "scheduler process died" string hits every emission site.
/// Structural routing (the console-dump gate in
/// `test_support::eval`) goes through [`DetailKind::SchedulerDied`],
/// NOT this prefix — the prefix is a human-readability contract,
/// not a detection mechanism. Exposed as `pub(crate)` so emitters
/// reference the same literal; renaming the prefix is a one-site
/// edit instead of a grep-and-hope across `scenario::*`.
///
/// Vocabulary history: prior versions of this module used two
/// prefixes (`SCHED_EXITED_PREFIX` = "scheduler process exited"
/// and `SCHED_NO_LONGER_RUNNING_PREFIX` = "scheduler process no
/// longer running") for in-workload vs post-ops detection. The
/// distinction carried no downstream semantics — every consumer
/// treated both as equivalent scheduler-death signals — so the
/// wording was unified onto "died" (shorter, matches the
/// `SchedulerDied` variant name, and closes a class of "which
/// wording does this site use?" drift bugs).
pub(crate) const SCHED_DIED_PREFIX: &str = "scheduler process died";

/// Format the scheduler-died detail message for an inter-step
/// liveness-probe failure (the scheduler was alive after step
/// `step_idx - 1` but ESRCH'd before step `step_idx` ran).
///
/// Begins with [`SCHED_DIED_PREFIX`] verbatim, followed by
/// "unexpectedly after completing step N of M (X.Xs into test)".
/// The prefix is the operator-visible stderr anchor (see the
/// prefix doc); structural routing is via
/// [`DetailKind::SchedulerDied`] on the emitted `AssertDetail`.
/// Centralized so ops.rs and any future emitter share a single
/// format.
pub(crate) fn format_sched_died_after_step(
    step_idx: usize,
    total_steps: usize,
    elapsed_s: f64,
) -> String {
    format!(
        "{SCHED_DIED_PREFIX} unexpectedly after completing step {step_idx} of {total_steps} ({elapsed_s:.1}s into test)",
    )
}

/// Format the scheduler-died detail message for the post-loop
/// liveness probe (the scheduler was alive throughout the step loop
/// but ESRCH'd after the last step completed).
///
/// Begins with [`SCHED_DIED_PREFIX`] verbatim; shares the prefix
/// invariant documented on [`format_sched_died_after_step`].
/// Structural routing is via [`DetailKind::SchedulerDied`] on the
/// emitted detail.
pub(crate) fn format_sched_died_after_all_steps(total_steps: usize, elapsed_s: f64) -> String {
    format!(
        "{SCHED_DIED_PREFIX} unexpectedly (detected after all {total_steps} steps completed, {elapsed_s:.1}s elapsed)",
    )
}

/// Format the scheduler-died detail message for the in-step
/// liveness probe (the scheduler ESRCH'd during a step's hold-period
/// sleep, before the step completed).
///
/// Begins with [`SCHED_DIED_PREFIX`] verbatim; shares the prefix
/// invariant documented on [`format_sched_died_after_step`].
/// Structural routing is via [`DetailKind::SchedulerDied`] on the
/// emitted detail. Emitted by `run_scenario` when the
/// liveness-poll inside `run_step`'s hold sleep observes
/// `process_alive(sched_pid) == false`, replacing the prior
/// behavior that waited for the post-loop probe to fire (which
/// stamped the message with the full scenario duration even when
/// the scheduler had died seconds earlier).
pub(crate) fn format_sched_died_during_workload(elapsed_s: f64) -> String {
    format!("{SCHED_DIED_PREFIX} unexpectedly during workload ({elapsed_s:.1}s into test)")
}

/// A single diagnostic message from an assertion, paired with a
/// structural [`DetailKind`] so filtering is robust to wording changes.
///
/// `Deref<Target = str>` and `Display` forward to `message` so existing
/// string-based probes (`d.contains("...")`, `format!("{d}")`) keep
/// working; new code that needs to filter by category should match on
/// `kind`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AssertDetail {
    pub kind: DetailKind,
    pub message: String,
}

impl PartialEq<&str> for AssertDetail {
    fn eq(&self, other: &&str) -> bool {
        self.message == *other
    }
}

impl PartialEq<str> for AssertDetail {
    fn eq(&self, other: &str) -> bool {
        self.message == *other
    }
}

impl PartialEq<String> for AssertDetail {
    fn eq(&self, other: &String) -> bool {
        self.message == *other
    }
}

impl AsRef<str> for AssertDetail {
    fn as_ref(&self) -> &str {
        &self.message
    }
}

impl AssertDetail {
    pub fn new(kind: DetailKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// Borrow this detail as a kind-prefixed [`std::fmt::Display`]
    /// adapter. The default [`Display`](std::fmt::Display) impl on
    /// `AssertDetail` writes only `message` so terminal output reads
    /// as bare prose; structured-log consumers that want to bucket
    /// failures by category without re-checking [`Self::kind`] reach
    /// for this helper instead.
    ///
    /// Renders as `[<DetailKind variant name>] <message>` — debug-form
    /// for the kind so the variant token is grep-stable across renames
    /// (a regression that drops a `DetailKind` variant breaks the
    /// match arms that produce it; the rendered token follows). Zero-
    /// allocation: the wrapper holds a `&AssertDetail` and writes
    /// straight into the formatter.
    ///
    /// ```
    /// # use ktstr::assert::{AssertDetail, DetailKind};
    /// let d = AssertDetail::new(DetailKind::Stuck, "tid 7 stuck 1500ms on cpu3");
    /// assert_eq!(d.to_string(), "tid 7 stuck 1500ms on cpu3");
    /// assert_eq!(
    ///     d.display_with_kind().to_string(),
    ///     "[Stuck] tid 7 stuck 1500ms on cpu3",
    /// );
    /// ```
    pub fn display_with_kind(&self) -> AssertDetailWithKind<'_> {
        AssertDetailWithKind { detail: self }
    }
}

/// `Display` adapter returned by [`AssertDetail::display_with_kind`].
/// Renders the detail as `[<kind>] <message>`. Held by reference so
/// the helper allocates nothing on the formatting path; the lifetime
/// is the borrow of the source `AssertDetail`.
#[must_use = "AssertDetailWithKind only renders when formatted"]
pub struct AssertDetailWithKind<'a> {
    detail: &'a AssertDetail,
}

impl std::fmt::Display for AssertDetailWithKind<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:?}] {}", self.detail.kind, self.detail.message)
    }
}

impl From<String> for AssertDetail {
    /// Conversion for uncategorized messages; defaults `kind` to
    /// [`DetailKind::Other`]. Prefer [`AssertDetail::new`] when the
    /// detail has a meaningful category — the `DetailKind` is serialized
    /// into the sidecar JSON and consumed by stats tooling to bucket
    /// failures, so losing the category bucket makes post-run
    /// categorization rely on free-text regex against `message`.
    fn from(message: String) -> Self {
        Self {
            kind: DetailKind::Other,
            message,
        }
    }
}

impl From<&str> for AssertDetail {
    /// Conversion for uncategorized messages; defaults `kind` to
    /// [`DetailKind::Other`]. Prefer [`AssertDetail::new`] when the
    /// detail has a meaningful category — the `DetailKind` is serialized
    /// into the sidecar JSON and consumed by stats tooling to bucket
    /// failures, so losing the category bucket makes post-run
    /// categorization rely on free-text regex against `message`.
    fn from(s: &str) -> Self {
        Self {
            kind: DetailKind::Other,
            message: s.to_string(),
        }
    }
}

impl std::ops::Deref for AssertDetail {
    type Target = str;
    fn deref(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for AssertDetail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// Result of checking a scenario run.
///
/// Contains pass/fail status, human-readable detail messages, and
/// aggregated statistics. Multiple results can be combined with
/// [`merge()`](AssertResult::merge).
///
/// ```
/// # use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
/// let mut a = AssertResult::pass();
/// assert!(a.passed);
///
/// let mut b = AssertResult::pass();
/// b.passed = false;
/// b.details.push(AssertDetail::new(DetailKind::Starved, "worker starved"));
///
/// a.merge(b);
/// assert!(!a.passed);
/// assert!(a.details.iter().any(|d| d.kind == DetailKind::Starved));
/// ```
/// Structured measurement value attached via
/// [`AssertResult::note_value`] / [`Verdict::note_value`].
///
/// The variants cover every primitive shape stats tooling consumes:
/// signed and unsigned 64-bit ints, 64-bit floats, booleans, and
/// owned strings. A test that wants to surface "max_wchar=12345"
/// alongside a passing IO_ACCOUNTING reachability check writes
/// `verdict.note_value("max_wchar", 12345i64)` and downstream stats
/// tooling reads `result.measurements["max_wchar"]` as
/// `NoteValue::Int(12345)`.
///
/// Distinct from [`AssertDetail`]'s free-form `Note` message: the
/// `Display` impl on `AssertDetail` is for human readers; the
/// structured map is for programmatic consumption (sidecar parsers,
/// `stats compare`, regression dashboards). Producers can call BOTH
/// `note(msg)` and `note_value(key, val)` on the same result — they
/// occupy independent buffers.
///
/// Conversion via the `From` impls below: any
/// `i64`/`u64`/`f64`/`bool`/`String`/`&str` literal flows into
/// `note_value` without explicit variant naming. Integer types
/// narrower than 64-bit (`i32`, `u32`, etc.) need an explicit cast
/// at the call site rather than a blanket impl, so the call site
/// reads honestly about the value's resolution.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum NoteValue {
    /// 64-bit signed integer — pid_t, exit codes, signed counters.
    Int(i64),
    /// 64-bit unsigned integer — work_units, byte counts, durations.
    Uint(u64),
    /// 64-bit float — ratios, rates, percentiles in microseconds.
    Float(f64),
    /// Boolean — completion flags, feature-detect results.
    Bool(bool),
    /// Owned string — categorical labels, environment tokens.
    Text(String),
}

impl From<i64> for NoteValue {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}
impl From<u64> for NoteValue {
    fn from(v: u64) -> Self {
        Self::Uint(v)
    }
}
impl From<f64> for NoteValue {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}
impl From<bool> for NoteValue {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}
impl From<String> for NoteValue {
    fn from(v: String) -> Self {
        Self::Text(v)
    }
}
impl From<&str> for NoteValue {
    fn from(v: &str) -> Self {
        Self::Text(v.to_string())
    }
}

#[must_use = "test verdict is lost if not checked"]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AssertResult {
    /// Whether all checks passed.
    pub passed: bool,
    /// True when the scenario was skipped (e.g. topology mismatch,
    /// missing resource). `passed` stays `true` so gate callers that
    /// treat skip as "not a failure" continue to work; stats tooling
    /// must subtract skipped runs from pass counts so they don't
    /// count as successful executions.
    pub skipped: bool,
    /// Human-readable diagnostic messages (failures, warnings), each
    /// tagged with a [`DetailKind`] for structural filtering.
    pub details: Vec<AssertDetail>,
    /// Aggregated stats from all workers in this scenario.
    pub stats: ScenarioStats,
    /// Structured measurements attached via [`Self::note_value`] /
    /// [`Verdict::note_value`]. Distinct from [`Self::details`] —
    /// `details` carries human-readable `String`s for operator
    /// triage, `measurements` carries typed `(key, NoteValue)` pairs
    /// for programmatic consumption (sidecar parsers, `stats
    /// compare`, regression dashboards).
    #[serde(default)]
    pub measurements: std::collections::BTreeMap<String, NoteValue>,
}

/// Per-cgroup statistics from worker telemetry.
///
/// # Percentile convention
///
/// `p99_wake_latency_us` and `median_wake_latency_us` are computed
/// by [`percentile`] using the NEAREST-RANK (Type 1) definition:
/// the value at `ceil(n * p) - 1` in sorted order. No interpolation
/// between samples. This matches the percentile convention used
/// throughout schbench and the BPF latency histograms the project
/// cross-references, so a `ktstr` p99 reading aligns with a
/// schbench `lat99` without adjustment. For small `n` (wake
/// reservoirs cap at `MAX_WAKE_SAMPLES = 100_000` per worker —
/// see `workload.rs`) nearest-rank is also numerically stable —
/// interpolation between the two nearest ranks would be
/// implementation-defined at sample-set boundaries.
///
/// # CV pooling scope
///
/// `wake_latency_cv` is POOLED across every sample from every
/// worker in the cgroup, not a per-worker CV averaged back. That
/// collapses per-worker dispersion into the cgroup-wide signal:
/// two workers with uniformly low jitter but different means
/// produce a high pooled CV (mean-shift between workers inflates
/// stddev), while per-worker CV would show neither worker as
/// bad. This is intentional for the fairness threshold
/// (`max_wake_latency_cv`): a scheduler that gives worker A
/// 10µs wakes and worker B 1ms wakes is failing fairness even if
/// each worker on its own is tight. Tests comparing single-worker
/// behavior should scope their assertions to per-worker data
/// rather than this aggregate.
///
/// # Derived ratios
///
/// Two metrics are DERIVED rather than measured and live as
/// `&self` methods, NOT as serde-serialized fields:
/// [`Self::wake_latency_tail_ratio`] (= p99/median) and
/// [`Self::iterations_per_worker`] (= total_iterations/num_workers).
/// Pre-1.0 cleanup eliminated the prior stored-field shadow and
/// `derive_ratios` stamper. Consumers always recompute on read,
/// so a hand-constructed fixture or a deserialized sidecar from an
/// older build cannot silently carry a stale ratio. The roll-up
/// fields on [`ScenarioStats::worst_wake_latency_tail_ratio`] /
/// [`ScenarioStats::worst_iterations_per_worker`] aggregate these
/// methods over per-cgroup [`Self`] entries during
/// [`AssertResult::merge`].
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, crate::Claim)]
pub struct CgroupStats {
    /// Number of workers in this cgroup.
    pub num_workers: usize,
    /// Distinct CPUs used across all workers in this cgroup.
    pub num_cpus: usize,
    /// Mean off-CPU percentage across workers (off_cpu_ns / wall_time_ns * 100).
    pub avg_off_cpu_pct: f64,
    /// Minimum off-CPU percentage across workers.
    pub min_off_cpu_pct: f64,
    /// Maximum off-CPU percentage across workers.
    pub max_off_cpu_pct: f64,
    /// max_off_cpu_pct - min_off_cpu_pct. Measures scheduling fairness within the cgroup.
    pub spread: f64,
    /// Longest scheduling gap across all workers (ms).
    pub max_gap_ms: u64,
    /// CPU where the longest scheduling gap occurred.
    pub max_gap_cpu: usize,
    /// Sum of CPU migration counts across all workers.
    pub total_migrations: u64,
    /// Migrations per iteration (total_migrations / total_iterations).
    pub migration_ratio: f64,
    /// 99th percentile wake latency across all workers (microseconds).
    pub p99_wake_latency_us: f64,
    /// Median wake latency across all workers (microseconds).
    pub median_wake_latency_us: f64,
    /// Coefficient of variation (stddev / mean) of wake latencies.
    ///
    /// Computed over the POOLED latency samples from every worker in
    /// the cgroup, not as a mean of per-worker CVs. Per-worker
    /// dispersion is therefore masked: a cgroup with one tight
    /// worker and one wildly variable worker can report a moderate
    /// pooled CV that looks healthier than either constituent. Use
    /// [`WorkerReport::resume_latencies_ns`] directly if per-worker
    /// CV is needed.
    pub wake_latency_cv: f64,
    /// Sum of iteration counts across all workers.
    pub total_iterations: u64,
    /// Mean schedstat run delay across workers (microseconds).
    pub mean_run_delay_us: f64,
    /// Worst schedstat run delay across workers (microseconds).
    pub worst_run_delay_us: f64,
    /// Fraction of pages on the expected NUMA node(s) (0.0-1.0).
    /// Derived from `/proc/self/numa_maps` and the worker's
    /// [`MemPolicy`](crate::workload::MemPolicy).
    pub page_locality: f64,
    /// Cross-node page migration ratio from `/proc/vmstat`
    /// `numa_pages_migrated` delta divided by total allocated pages.
    pub cross_node_migration_ratio: f64,
    /// Extensible metrics for the generic comparison pipeline.
    #[serde(default)]
    pub ext_metrics: BTreeMap<String, f64>,
}

impl CgroupStats {
    /// Wake-latency tail amplification:
    /// `p99_wake_latency_us / median_wake_latency_us`. Returns `0.0`
    /// when `median_wake_latency_us <= 0.0` so the result never
    /// propagates `NaN` / `Infinity` into downstream
    /// `finite_or_zero` filters. Method-only access (no stored
    /// shadow) — recomputed every call from the raw fields.
    ///
    /// Unitless; ≥1.0 by definition of order statistics (p99 cannot
    /// undershoot the median on the same sample set). Values far
    /// above 1.0 signal a long tail — the scheduler wakes most
    /// workers promptly but occasionally stalls some, a regression
    /// axis that neither `median_*` nor `p99_*` exposes in
    /// isolation.
    pub fn wake_latency_tail_ratio(&self) -> f64 {
        if self.median_wake_latency_us > 0.0 {
            self.p99_wake_latency_us / self.median_wake_latency_us
        } else {
            0.0
        }
    }

    /// Throughput per parallel degree:
    /// `total_iterations / num_workers`. Returns `0.0` when
    /// `num_workers == 0` so the result never propagates
    /// `NaN` / `Infinity`. Method-only access (no stored shadow) —
    /// recomputed every call from the raw fields.
    ///
    /// Only meaningful across runs of the SAME variant (equal
    /// scenario duration): cross-variant comparison is misleading
    /// because this metric is NOT rate-normalized — a longer-
    /// running scenario racks up more iterations per worker even if
    /// the scheduler is identical. `stats compare`-style
    /// comparisons hold scenario, topology, and work_type constant
    /// before reading this method.
    pub fn iterations_per_worker(&self) -> f64 {
        if self.num_workers > 0 {
            self.total_iterations as f64 / self.num_workers as f64
        } else {
            0.0
        }
    }
}

/// Aggregated statistics across all cgroups in a scenario.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, crate::Claim)]
pub struct ScenarioStats {
    /// Per-cgroup stats, one entry per cgroup.
    pub cgroups: Vec<CgroupStats>,
    /// Sum of workers across all cgroups.
    pub total_workers: usize,
    /// Sum of per-cgroup distinct CPU counts (not deduplicated across cgroups).
    pub total_cpus: usize,
    /// Sum of migration counts across all cgroups.
    pub total_migrations: u64,
    /// Worst spread across any cgroup (highest).
    pub worst_spread: f64,
    /// Worst gap across any cgroup (highest, ms). Paired with
    /// `worst_gap_cpu` — both come from the same cgroup.
    pub worst_gap_ms: u64,
    /// CPU where the worst gap occurred across all cgroups. Paired
    /// with `worst_gap_ms` — both come from the same cgroup.
    pub worst_gap_cpu: usize,
    /// Worst migration ratio across any cgroup (highest).
    pub worst_migration_ratio: f64,
    /// Worst p99 wake latency across all cgroups (highest, microseconds).
    pub worst_p99_wake_latency_us: f64,
    /// Worst median wake latency across all cgroups (highest, microseconds).
    pub worst_median_wake_latency_us: f64,
    /// Worst wake latency coefficient of variation across all cgroups (highest).
    pub worst_wake_latency_cv: f64,
    /// Sum of iteration counts across all cgroups.
    pub total_iterations: u64,
    /// Worst mean schedstat run delay across all cgroups (highest, microseconds).
    pub worst_mean_run_delay_us: f64,
    /// Worst schedstat run delay across all cgroups (highest, microseconds).
    pub worst_run_delay_us: f64,
    /// Worst page locality fraction across cgroups (lowest non-zero).
    pub worst_page_locality: f64,
    /// Worst cross-node migration ratio across cgroups (highest).
    pub worst_cross_node_migration_ratio: f64,
    /// Worst wake-latency tail amplification across cgroups
    /// (highest). Higher is worse — it is the ratio of p99 to
    /// median, so a cgroup with a severe long tail drives this up.
    /// Zero when every cgroup has `median_wake_latency_us == 0.0`
    /// (no samples). Pairs with
    /// [`CgroupStats::wake_latency_tail_ratio`] — see that field
    /// for the unit/semantics rationale.
    ///
    /// Routed through `GauntletRow` and the `METRICS` registry;
    /// `stats compare` surfaces this axis in its comparison rows.
    pub worst_wake_latency_tail_ratio: f64,
    /// Worst per-worker iteration count across cgroups (LOWEST
    /// non-zero).
    ///
    /// Per-cgroup [`CgroupStats::iterations_per_worker`] is a
    /// throughput metric; the worst-case (regression-detecting)
    /// roll-up across cgroups is the lowest non-zero value — a
    /// cgroup that fell behind surfaces as the lowest per-worker
    /// throughput. The fold in [`AssertResult::merge`] uses
    /// `fold_lowest_nonzero` rather than plain `min`: the
    /// accumulator pattern `AssertResult::pass().merge(real)`
    /// starts at 0.0 from `Default`, and a plain min would let
    /// that sentinel destroy any positive reading folded in. 0.0
    /// is treated as "not reported," matching the sentinel
    /// convention shared with [`Self::worst_page_locality`].
    ///
    /// Only meaningful across runs of the SAME variant — see
    /// [`CgroupStats::iterations_per_worker`] for the cross-
    /// variant caveat. Routed through `GauntletRow` and the
    /// `METRICS` registry; `stats compare` surfaces this axis
    /// in its comparison rows.
    pub worst_iterations_per_worker: f64,
    /// Extensible metrics for the generic comparison pipeline.
    /// Populated from per-cgroup ext_metrics (worst value across cgroups).
    #[serde(default)]
    pub ext_metrics: BTreeMap<String, f64>,
}

impl AssertResult {
    /// Empty passing result with no details and default stats. Use
    /// when a scenario completed successfully with nothing interesting
    /// to report.
    pub fn pass() -> Self {
        Self {
            passed: true,
            skipped: false,
            details: vec![],
            stats: Default::default(),
            measurements: std::collections::BTreeMap::new(),
        }
    }
    /// Pass result with a skip reason. Used when a scenario cannot run
    /// under the current topology or flag combination but is not a failure.
    pub fn skip(reason: impl Into<String>) -> Self {
        Self {
            passed: true,
            skipped: true,
            details: vec![AssertDetail::new(DetailKind::Skip, reason)],
            stats: Default::default(),
            measurements: std::collections::BTreeMap::new(),
        }
    }
    /// Failing result carrying a single [`AssertDetail`]. Mirrors
    /// [`Self::pass`] / [`Self::skip`] for the failure axis so callers
    /// don't hand-roll the struct-literal shape (`passed: false,
    /// skipped: false, details: vec![d], stats: Default::default()`)
    /// at every diagnostic-only failure site.
    pub fn fail(detail: AssertDetail) -> Self {
        Self {
            passed: false,
            skipped: false,
            details: vec![detail],
            stats: Default::default(),
            measurements: std::collections::BTreeMap::new(),
        }
    }
    /// Failing result carrying a single diagnostic message with
    /// [`DetailKind::Other`]. Shortcut for the common three-deep
    /// nesting `AssertResult::fail(AssertDetail::new(DetailKind::Other,
    /// msg))` at call sites where the failure is a diagnostic
    /// message and the kind is always `Other`. Named `fail_msg`
    /// rather than `fail_other` so the call site reads "failing
    /// result with a message" without leaking the `DetailKind`
    /// variant name into the API surface; external callers that do
    /// want a specific `kind` still reach for `AssertResult::fail`
    /// + `AssertDetail::new(kind, msg)`.
    pub fn fail_msg(msg: impl Into<String>) -> Self {
        Self::fail(AssertDetail::new(DetailKind::Other, msg))
    }
    /// Append an informational annotation tagged
    /// [`DetailKind::Note`]. Does NOT alter [`Self::passed`] or
    /// [`Self::skipped`] — a note is context, not a verdict.
    /// Use to surface observed values alongside a passing or
    /// failing result so the sidecar carries the diagnostic
    /// context an operator needs without forcing every test to
    /// hand-format a `format!` and push it onto `details`
    /// directly. The kind tag lets sidecar consumers filter
    /// informational rows from genuine failures structurally.
    pub fn note(&mut self, msg: impl Into<String>) -> &mut Self {
        self.details.push(AssertDetail::new(DetailKind::Note, msg));
        self
    }
    /// Builder-style sibling of [`Self::note`] returning the
    /// owned result so a scenario can chain
    /// `AssertResult::pass().with_note("max_wchar=12345")` at
    /// the return site. Equivalent to calling
    /// [`Self::note`] on a mutable binding.
    pub fn with_note(mut self, msg: impl Into<String>) -> Self {
        self.note(msg);
        self
    }
    /// Convenience accessor returning [`Self::skipped`]. Stats tooling
    /// uses this to subtract non-executions from pass counts so
    /// "topology mismatch" runs don't inflate the pass rate.
    pub fn is_skipped(&self) -> bool {
        self.skipped
    }
    /// Convenience accessor returning the negation of [`Self::passed`].
    /// Mirrors [`Self::is_skipped`] so short-circuit branches reading
    /// "did this claim fail?" don't have to negate `.passed` inline.
    pub fn is_failed(&self) -> bool {
        !self.passed
    }
    /// Fold `other` into `self`. `passed` is conjoined (any failure
    /// wins), `details` concatenate, and aggregate stats adopt the
    /// worst-case value per dimension so the merged result represents
    /// the union of all checks applied.
    pub fn merge(&mut self, other: AssertResult) {
        /// Lowest-non-zero fold: `*self_field` becomes `other_field`
        /// when `other_field` is strictly positive AND either
        /// `*self_field` is zero (uninitialized sentinel) or
        /// `other_field` is strictly smaller than `*self_field`.
        ///
        /// This is NOT `f64::min` — a plain min would let an
        /// unreported cgroup (`0.0` sentinel) clobber a real
        /// reading from another cgroup, treating "no data yet" as
        /// "worst possible." The accumulator pattern
        /// `AssertResult::pass().merge(real)` starts with 0.0 from
        /// `Default`, and a plain min would destroy any positive
        /// reading folded in — so every lowest-is-worse rollup
        /// uses this fold to treat 0.0 as a sentinel rather than a
        /// real measurement.
        fn fold_lowest_nonzero(self_field: &mut f64, other_field: f64) {
            if other_field > 0.0 && (*self_field == 0.0 || other_field < *self_field) {
                *self_field = other_field;
            }
        }

        if !other.passed {
            self.passed = false;
        }
        // skip + skip = skipped (nothing executed); skip + pass/fail =
        // NOT skipped (real work ran). Equivalent to logical AND of
        // the two `skipped` flags.
        self.skipped = self.skipped && other.skipped;
        self.details.extend(other.details);
        let s = &mut self.stats;
        let o = &other.stats;
        s.total_workers += o.total_workers;
        s.total_cpus += o.total_cpus;
        s.total_migrations += o.total_migrations;
        s.total_iterations += o.total_iterations;
        s.worst_spread = s.worst_spread.max(o.worst_spread);
        s.worst_migration_ratio = s.worst_migration_ratio.max(o.worst_migration_ratio);
        s.worst_p99_wake_latency_us = s.worst_p99_wake_latency_us.max(o.worst_p99_wake_latency_us);
        s.worst_median_wake_latency_us = s
            .worst_median_wake_latency_us
            .max(o.worst_median_wake_latency_us);
        s.worst_wake_latency_cv = s.worst_wake_latency_cv.max(o.worst_wake_latency_cv);
        s.worst_run_delay_us = s.worst_run_delay_us.max(o.worst_run_delay_us);
        s.worst_mean_run_delay_us = s.worst_mean_run_delay_us.max(o.worst_mean_run_delay_us);
        s.worst_cross_node_migration_ratio = s
            .worst_cross_node_migration_ratio
            .max(o.worst_cross_node_migration_ratio);
        // Tail ratio is higher-is-worse: max across cgroups surfaces
        // the worst long-tail amplification.
        s.worst_wake_latency_tail_ratio = s
            .worst_wake_latency_tail_ratio
            .max(o.worst_wake_latency_tail_ratio);
        // Per-worker throughput is lower-is-worse: take the
        // lowest non-zero reading across cgroups so a cgroup
        // falling behind wins the "worst" bucket. 0.0 is the
        // unreported sentinel — the accumulator pattern
        // `AssertResult::pass().merge(real)` starts at 0.0 from
        // `Default`, so a plain min would let that sentinel
        // destroy real measurements. See `fold_lowest_nonzero`
        // above for the policy.
        fold_lowest_nonzero(
            &mut s.worst_iterations_per_worker,
            o.worst_iterations_per_worker,
        );
        // Coupled fields: `worst_gap_cpu` must come from the same
        // cgroup that posted the new worst `worst_gap_ms`.
        if o.worst_gap_ms > s.worst_gap_ms {
            s.worst_gap_ms = o.worst_gap_ms;
            s.worst_gap_cpu = o.worst_gap_cpu;
        }
        // NUMA page locality: lowest-non-zero fold — see
        // `fold_lowest_nonzero` above for the sentinel convention.
        fold_lowest_nonzero(&mut s.worst_page_locality, o.worst_page_locality);
        // Merge extensible metrics: take worst per key according to
        // each metric's polarity in the MetricDef registry. For
        // `higher_is_worse: true` the worst is max; for
        // `higher_is_worse: false` the worst is min.
        //
        // Unregistered metric names fall through to
        // [`crate::stats::infer_higher_is_worse`], which derives the
        // polarity from name substrings (e.g. `*_iops`,
        // `*_latency_us`). Without the inference, a payload-author
        // throughput metric — e.g. `jobs.0.read.iops` from
        // `OutputFormat::LlmExtract` — would fold with `max`,
        // keeping the BETTER (higher) value across cgroups and
        // masking a cgroup that fell behind. The inference returns a
        // higher-is-worse default when no token matches, so genuinely
        // unknown names still surface their max (the safer side of
        // the regression-vs-improvement misclassification).
        //
        // `or_insert(*v)` rather than `or_insert(0.0)`: the old sentinel
        // clobbered real-but-small values for min-polarity metrics on
        // first merge, making the subsequent min comparison meaningless.
        for (k, v) in &other.stats.ext_metrics {
            let higher_is_worse = crate::stats::metric_def(k)
                .map(|m| m.higher_is_worse())
                .unwrap_or_else(|| crate::stats::infer_higher_is_worse(k));
            let entry = self.stats.ext_metrics.entry(k.clone()).or_insert(*v);
            *entry = if higher_is_worse {
                entry.max(*v)
            } else {
                entry.min(*v)
            };
        }
        // Append per-cgroup stats last: moving `other.stats.cgroups`
        // here consumes `other.stats`, so every scalar/map access
        // above goes through the `&other.stats` reference first.
        self.stats.cgroups.extend(other.stats.cgroups);

        // Fold structured measurements. Keys from `other` overwrite
        // existing keys from `self` because the merge protocol treats
        // the right-hand side as a more recent observation; a
        // duplicate-key write is a producer bug (two cgroups
        // measuring the same global metric) but the "later wins"
        // policy keeps the result deterministic for tests pinning
        // merge order. Producers that need additive accumulation
        // should use `stats.ext_metrics` (which has explicit polarity
        // semantics) rather than `measurements`.
        for (k, v) in other.measurements {
            self.measurements.insert(k, v);
        }
    }

    /// Attach a structured `(key, value)` measurement to the result.
    /// Writes into [`Self::measurements`] without altering
    /// [`Self::passed`] / [`Self::skipped`] / [`Self::details`] —
    /// pure context for stats tooling.
    ///
    /// Distinct from [`Self::note`]: `note` carries a free-form
    /// `String` for operator triage; `note_value` carries a typed
    /// `(key, NoteValue)` pair for programmatic consumption (sidecar
    /// parsers, `stats compare` regression dashboards). Producers
    /// commonly call BOTH — they occupy independent buffers and
    /// neither overwrites the other.
    ///
    /// Key collision policy: a second write with the same `key`
    /// overwrites the first. The intended call site shape is "one
    /// producer per key" (one site computes `max_wchar`, one site
    /// computes `psi_some_total_usec`); accidental key collision
    /// indicates a producer bug. The test
    /// `note_value_overwrites_on_duplicate_key` pins this last-
    /// write-wins semantics.
    ///
    /// ```
    /// # use ktstr::assert::{AssertResult, NoteValue};
    /// let mut r = AssertResult::pass();
    /// r.note_value("max_wchar", 12345i64);
    /// r.note_value("psi_available", true);
    /// assert_eq!(r.measurements["max_wchar"], NoteValue::Int(12345));
    /// assert_eq!(r.measurements["psi_available"], NoteValue::Bool(true));
    /// ```
    pub fn note_value(&mut self, key: impl Into<String>, value: impl Into<NoteValue>) -> &mut Self {
        self.measurements.insert(key.into(), value.into());
        self
    }

    /// Fold a sequence of [`AssertResult`]s with OR semantics: the
    /// returned result passes iff at least one branch passes. Use
    /// when a test author expresses "either of these two checks
    /// suffices" — a kernel-version-fork case where one path is
    /// expected on 6.16 and another on 7.1, or a topology probe
    /// where any of several detection methods landing is enough.
    ///
    /// Outcomes:
    /// - **At least one branch passes**: returned result is passing.
    ///   `details` carries the [`DetailKind::Note`]-tagged "branch N
    ///   chosen" annotation pointing at the first passing branch.
    ///   Failed-branch details are dropped (they would only confuse
    ///   the operator with messages from the not-taken paths).
    ///   `stats` adopts the first passing branch's `stats`.
    ///   `measurements` union all passing branches' measurements
    ///   (last write wins on key collision, matching `merge`).
    ///   `skipped` follows the first passing branch.
    /// - **All branches fail**: returned result is failing. Every
    ///   branch's `details` are concatenated, with each detail's
    ///   message prefixed by `"any_of[<branch-idx>]: "` so an
    ///   operator can identify which branch produced which failure.
    ///   `stats` and `measurements` adopt the FIRST branch's values
    ///   (an arbitrary choice but deterministic; a smarter
    ///   "best-failing-branch" pick would require comparing
    ///   `details.len()`, which is policy not mechanism).
    ///   `skipped` is `false`.
    /// - **Empty input**: returned result is failing with a single
    ///   detail explaining the empty `any_of`. An empty disjunction
    ///   is logically false; this surfaces a producer bug as a
    ///   nameable failure rather than a vacuous pass.
    ///
    /// Doc: a trivial two-branch test with the second branch passing
    /// and the first branch failing — pinning that the verdict
    /// chooses the passer.
    ///
    /// ```
    /// # use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
    /// let r = AssertResult::any_of([
    ///     {
    ///         let mut a = AssertResult::pass();
    ///         a.passed = false;
    ///         a.details.push(AssertDetail::new(DetailKind::Other, "branch 0 boom"));
    ///         a
    ///     },
    ///     AssertResult::pass(),
    /// ]);
    /// assert!(r.passed);
    /// ```
    pub fn any_of(branches: impl IntoIterator<Item = AssertResult>) -> AssertResult {
        let mut branches: Vec<AssertResult> = branches.into_iter().collect();
        if branches.is_empty() {
            return AssertResult::fail(AssertDetail::new(
                DetailKind::Other,
                "any_of: empty branch list — a disjunction of zero alternatives is logically false",
            ));
        }

        let first_pass_idx = branches.iter().position(|b| b.passed && !b.skipped);
        if let Some(idx) = first_pass_idx {
            // At least one branch passes. Take the first passing
            // branch's stats / skipped, union measurements across
            // every passing branch, and drop failed-branch details.
            let mut chosen = branches.swap_remove(idx);
            // After swap_remove(idx) we no longer iterate the
            // original branches in their original order. Use the
            // remaining `branches` to scoop up `measurements` from
            // any other passing branches before discarding their
            // `details`.
            for b in branches {
                if b.passed && !b.skipped {
                    for (k, v) in b.measurements {
                        chosen.measurements.insert(k, v);
                    }
                }
            }
            chosen.details.push(AssertDetail::new(
                DetailKind::Note,
                format!("any_of: branch {idx} satisfied the disjunction"),
            ));
            chosen
        } else {
            // All branches failed. Concatenate details with branch-
            // index prefixes; adopt the first branch's stats /
            // measurements / skipped (deterministic but arbitrary).
            let total_branches = branches.len();
            let mut iter = branches.into_iter().enumerate();
            let (_, first) = iter.next().expect("non-empty checked above");
            let mut acc = AssertResult {
                passed: false,
                skipped: false,
                details: Vec::new(),
                stats: first.stats,
                measurements: first.measurements,
            };
            // Re-prefix the first branch's details and feed in.
            for d in first.details {
                acc.details.push(AssertDetail::new(
                    d.kind,
                    format!("any_of[0]: {}", d.message),
                ));
            }
            for (idx, b) in iter {
                for d in b.details {
                    acc.details.push(AssertDetail::new(
                        d.kind,
                        format!("any_of[{idx}]: {}", d.message),
                    ));
                }
            }
            acc.details.push(AssertDetail::new(
                DetailKind::Other,
                format!("any_of: all {total_branches} branches failed"),
            ));
            acc
        }
    }

    /// Fold a sequence of [`AssertResult`]s with AND semantics:
    /// equivalent to `branches.into_iter().fold(pass(),
    /// |acc, b| { acc.merge(b); acc })`. Returns a passing result iff
    /// every branch passes.
    ///
    /// Distinct from [`Self::merge`] in API shape only: `merge`
    /// folds one external result into an existing accumulator;
    /// `all_of` folds an iterator of branches into a fresh result.
    /// Same semantics for `passed` (conjoined), `details`
    /// (concatenated), `stats` (worst-per-dimension), `measurements`
    /// (union with last-write-wins). An empty input yields the
    /// passing identity (`AssertResult::pass()`) — the AND of an
    /// empty set is logically true, mirroring `Iterator::all`.
    ///
    /// Use when the test reads more naturally as "every check
    /// must hold" than as a merge chain — e.g. when the checks
    /// are dynamically generated from a slice and the call site
    /// would otherwise need an explicit `for` loop with `merge`.
    ///
    /// ```
    /// # use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
    /// let r = AssertResult::all_of([
    ///     AssertResult::pass(),
    ///     AssertResult::pass(),
    /// ]);
    /// assert!(r.passed);
    ///
    /// let r = AssertResult::all_of([
    ///     AssertResult::pass(),
    ///     AssertResult::fail(AssertDetail::new(DetailKind::Other, "boom")),
    /// ]);
    /// assert!(!r.passed);
    /// ```
    pub fn all_of(branches: impl IntoIterator<Item = AssertResult>) -> AssertResult {
        let mut acc = AssertResult::pass();
        for b in branches {
            acc.merge(b);
        }
        acc
    }
}

/// Worker-side assertion plan (crate-internal). Specifies which checks
/// to run on worker reports after collection.
///
/// External users should use [`Assert`] and its `assert_cgroup()` method
/// instead.
#[derive(Clone, Debug)]
pub(crate) struct AssertPlan {
    pub(crate) not_starved: bool,
    pub(crate) isolation: bool,
    pub(crate) max_gap_ms: Option<u64>,
    pub(crate) max_spread_pct: Option<f64>,
    pub(crate) max_throughput_cv: Option<f64>,
    pub(crate) min_work_rate: Option<f64>,
    pub(crate) max_p99_wake_latency_ns: Option<u64>,
    pub(crate) max_wake_latency_cv: Option<f64>,
    pub(crate) min_iteration_rate: Option<f64>,
    pub(crate) max_migration_ratio: Option<f64>,
    pub(crate) min_page_locality: Option<f64>,
    pub(crate) max_cross_node_migration_ratio: Option<f64>,
    pub(crate) max_slow_tier_ratio: Option<f64>,
}

impl AssertPlan {
    pub(crate) fn new() -> Self {
        Self {
            not_starved: false,
            isolation: false,
            max_gap_ms: None,
            max_spread_pct: None,
            max_throughput_cv: None,
            min_work_rate: None,
            max_p99_wake_latency_ns: None,
            max_wake_latency_cv: None,
            min_iteration_rate: None,
            max_migration_ratio: None,
            min_page_locality: None,
            max_cross_node_migration_ratio: None,
            max_slow_tier_ratio: None,
        }
    }

    /// Run all configured checks against one cgroup's reports.
    ///
    /// `cpuset` is the expected CPU set for isolation checks. Pass `None`
    /// when there is no cpuset constraint (isolation check is skipped).
    ///
    /// `numa_nodes` is the NUMA node IDs covered by the cpuset (derived
    /// via `TestTopology::numa_nodes_for_cpuset`). Used for page locality
    /// and slow-tier ratio checks. Pass `None` when NUMA checks are not
    /// applicable.
    pub(crate) fn assert_cgroup(
        &self,
        reports: &[WorkerReport],
        cpuset: Option<&BTreeSet<usize>>,
        numa_nodes: Option<&BTreeSet<usize>>,
    ) -> AssertResult {
        let mut r = AssertResult::pass();
        if self.not_starved {
            let mut cgroup_result = assert_not_starved(reports);
            // Apply custom spread threshold if set.
            if let Some(spread_limit) = self.max_spread_pct {
                // Re-check spread against custom threshold. The default
                // assert_not_starved uses spread_threshold_pct(); clear
                // those failures and re-evaluate.
                cgroup_result
                    .details
                    .retain(|d| d.kind != DetailKind::Unfair);
                if let Some(cg) = cgroup_result.stats.cgroups.first() {
                    if cg.spread > spread_limit && cg.num_workers >= 2 {
                        cgroup_result.passed = false;
                        cgroup_result.details.push(AssertDetail::new(
                            DetailKind::Unfair,
                            format!(
                                "unfair cgroup: spread={:.0}% ({:.0}-{:.0}%) {} workers on {} cpus (threshold {:.0}%)",
                                cg.spread, cg.min_off_cpu_pct, cg.max_off_cpu_pct,
                                cg.num_workers, cg.num_cpus, spread_limit
                            ),
                        ));
                    } else {
                        // Re-derive passed: only non-spread failures matter.
                        cgroup_result.passed = !cgroup_result
                            .details
                            .iter()
                            .any(|d| matches!(d.kind, DetailKind::Starved | DetailKind::Stuck));
                    }
                }
            }
            // Apply custom gap threshold if set.
            if let Some(threshold) = self.max_gap_ms {
                // Re-check gaps against custom threshold. The default
                // assert_not_starved uses gap_threshold_ms() (2000ms
                // release, 3000ms debug); clear those failures and
                // re-evaluate.
                cgroup_result
                    .details
                    .retain(|d| d.kind != DetailKind::Stuck);
                let had_gap_failure = reports.iter().any(|w| w.max_gap_ms > threshold);
                if had_gap_failure {
                    cgroup_result.passed = false;
                    for w in reports {
                        if w.max_gap_ms > threshold {
                            cgroup_result.details.push(AssertDetail::new(
                                DetailKind::Stuck,
                                format!(
                                    "tid {} stuck {}ms on cpu{} at +{}ms (threshold {}ms)",
                                    w.tid, w.max_gap_ms, w.max_gap_cpu, w.max_gap_at_ms, threshold,
                                ),
                            ));
                        }
                    }
                } else {
                    // Re-derive passed: only non-gap failures matter.
                    cgroup_result.passed = !cgroup_result
                        .details
                        .iter()
                        .any(|d| matches!(d.kind, DetailKind::Starved | DetailKind::Unfair));
                }
            }
            r.merge(cgroup_result);
        }
        if self.isolation
            && let Some(cs) = cpuset
        {
            r.merge(assert_isolation(reports, cs));
        }
        if self.max_throughput_cv.is_some() || self.min_work_rate.is_some() {
            r.merge(assert_throughput_parity(
                reports,
                self.max_throughput_cv,
                self.min_work_rate,
            ));
        }
        if self.max_p99_wake_latency_ns.is_some()
            || self.max_wake_latency_cv.is_some()
            || self.min_iteration_rate.is_some()
        {
            r.merge(assert_benchmarks(
                reports,
                self.max_p99_wake_latency_ns,
                self.max_wake_latency_cv,
                self.min_iteration_rate,
            ));
        }
        if let Some(max_ratio) = self.max_migration_ratio {
            let total_mig: u64 = reports.iter().map(|w| w.migration_count).sum();
            let total_iters: u64 = reports.iter().map(|w| w.iterations).sum();
            let ratio = if total_iters > 0 {
                total_mig as f64 / total_iters as f64
            } else {
                0.0
            };
            if ratio > max_ratio {
                r.passed = false;
                r.details.push(AssertDetail::new(
                    DetailKind::Migration,
                    format!(
                        "migration ratio {:.4} exceeds threshold {:.4} ({} migrations / {} iterations)",
                        ratio, max_ratio, total_mig, total_iters,
                    ),
                ));
            }
        }
        if let Some(min_locality) = self.min_page_locality
            && let Some(nodes) = numa_nodes
        {
            // Aggregate NUMA pages across the cgroup so the locality
            // check evaluates the cgroup as a whole rather than
            // skipping workers with empty numa_pages or summing
            // misleading per-worker fractions. Skipping zero-page
            // workers lets a cgroup with no NUMA signal silently
            // pass `min_page_locality`.
            let mut total: u64 = 0;
            let mut local: u64 = 0;
            for w in reports {
                for (&node, &count) in &w.numa_pages {
                    total += count;
                    if nodes.contains(&node) {
                        local += count;
                    }
                }
            }
            let locality = if total > 0 {
                local as f64 / total as f64
            } else {
                // Zero observed pages across the cgroup is treated
                // as zero locality so the threshold surfaces a
                // workload that produced no NUMA allocations.
                0.0
            };
            r.merge(assert_page_locality(
                locality,
                Some(min_locality),
                total,
                local,
            ));
        }
        if let Some(max_ratio) = self.max_cross_node_migration_ratio {
            // `vmstat_numa_pages_migrated` is the delta of the
            // system-wide `/proc/vmstat numa_pages_migrated` counter
            // captured by each worker over its own work loop. With
            // concurrent workers the deltas overlap heavily — every
            // worker observes roughly the same system-wide migration
            // count, so summing them inflates the numerator by the
            // worker count. Take the maximum delta across the cgroup
            // as the closest approximation of total migrations
            // observed during the run, then divide once by the
            // cgroup-wide total of allocated pages.
            let total_pages: u64 = reports
                .iter()
                .map(|w| w.numa_pages.values().sum::<u64>())
                .sum();
            let migrated_pages: u64 = reports
                .iter()
                .map(|w| w.vmstat_numa_pages_migrated)
                .max()
                .unwrap_or(0);
            r.merge(assert_cross_node_migration(
                migrated_pages,
                total_pages,
                Some(max_ratio),
            ));
        }
        if let Some(max_ratio) = self.max_slow_tier_ratio
            && numa_nodes.is_some()
        {
            for w in reports {
                if w.numa_pages.is_empty() {
                    continue;
                }
                let total: u64 = w.numa_pages.values().sum();
                if total > 0 {
                    r.merge(assert_slow_tier_ratio(
                        &w.numa_pages,
                        max_ratio,
                        total,
                        numa_nodes,
                    ));
                }
            }
        }
        r
    }
}

/// Check slow-tier page ratio against threshold.
///
/// "Slow tier" nodes are NUMA nodes NOT in the cpuset's NUMA node set.
/// For CXL memory-only nodes, these are the nodes without CPUs.
fn assert_slow_tier_ratio(
    numa_pages: &BTreeMap<usize, u64>,
    max_ratio: f64,
    total_pages: u64,
    numa_nodes: Option<&BTreeSet<usize>>,
) -> AssertResult {
    let mut r = AssertResult::pass();
    let Some(cpu_nodes) = numa_nodes else {
        return r;
    };
    let slow_pages: u64 = numa_pages
        .iter()
        .filter(|(node, _)| !cpu_nodes.contains(node))
        .map(|(_, count)| count)
        .sum();
    let ratio = slow_pages as f64 / total_pages as f64;
    if ratio > max_ratio {
        r.passed = false;
        r.details.push(AssertDetail::new(
            DetailKind::SlowTier,
            format!(
                "slow-tier page ratio {ratio:.4} ({pct:.2}%) exceeds threshold {max_ratio:.4} ({thr_pct:.2}%) \
                 ({slow_pages}/{total_pages} pages on non-CPU nodes)",
                pct = ratio * 100.0,
                thr_pct = max_ratio * 100.0,
            ),
        ));
    }
    r
}

/// Check NUMA page locality against threshold.
///
/// `observed` is the fraction of pages on expected nodes (0.0-1.0).
/// `total_pages` and `local_pages` are included in diagnostics.
pub fn assert_page_locality(
    observed: f64,
    min_locality: Option<f64>,
    total_pages: u64,
    local_pages: u64,
) -> AssertResult {
    let mut r = AssertResult::pass();
    if let Some(threshold) = min_locality
        && observed < threshold
    {
        r.passed = false;
        r.details.push(AssertDetail::new(
            DetailKind::PageLocality,
            format!(
                "page locality {observed:.4} ({pct:.2}%) below threshold {threshold:.4} ({thr_pct:.2}%) ({local_pages}/{total_pages} pages local)",
                pct = observed * 100.0,
                thr_pct = threshold * 100.0,
            ),
        ));
    }
    r
}

/// Check cross-node page migration ratio against threshold.
///
/// `migrated_pages` is the delta of `/proc/vmstat` `numa_pages_migrated`
/// between pre- and post-workload snapshots. `total_pages` is the total
/// allocated pages from numa_maps.
///
/// Inconsistent inputs (`migrated_pages > 0` while `total_pages == 0`)
/// fail loudly: vmstat saw migrations the workload's numa_maps did not
/// account for, which is either a measurement gap or an instrumentation
/// bug, and silently coercing the ratio to 0.0 would let the assertion
/// pass on data the operator should not trust.
pub fn assert_cross_node_migration(
    migrated_pages: u64,
    total_pages: u64,
    max_ratio: Option<f64>,
) -> AssertResult {
    let mut r = AssertResult::pass();
    if let Some(threshold) = max_ratio {
        if total_pages == 0 {
            if migrated_pages > 0 {
                r.passed = false;
                r.details.push(AssertDetail::new(
                    DetailKind::CrossNodeMigration,
                    format!(
                        "cross-node migration inconsistent: {migrated_pages} pages migrated but 0 pages observed in numa_maps (threshold {threshold:.4})",
                    ),
                ));
            }
            return r;
        }
        let ratio = migrated_pages as f64 / total_pages as f64;
        if ratio > threshold {
            r.passed = false;
            r.details.push(AssertDetail::new(
                DetailKind::CrossNodeMigration,
                format!(
                    "cross-node migration ratio {ratio:.4} ({pct:.2}%) exceeds threshold {threshold:.4} ({thr_pct:.2}%) ({migrated_pages}/{total_pages} pages migrated)",
                    pct = ratio * 100.0,
                    thr_pct = threshold * 100.0,
                ),
            ));
        }
    }
    r
}

impl Default for AssertPlan {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
impl AssertPlan {
    fn check_not_starved(mut self) -> Self {
        self.not_starved = true;
        self
    }

    fn check_isolation(mut self) -> Self {
        self.isolation = true;
        self
    }

    fn max_gap_ms(mut self, ms: u64) -> Self {
        self.max_gap_ms = Some(ms);
        self
    }
}

/// Unified assertion configuration. Carries both worker checks and
/// monitor thresholds as a single composable type. Each `Option` field
/// acts as an override — `None` means "inherit from parent layer".
///
/// Merge order: `Assert::default_checks()` -> `Scheduler.assert` -> per-test `assert`.
/// `default_checks()` is `NO_OVERRIDES` — all assertions are opt-in.
///
/// ```
/// # use ktstr::assert::Assert;
/// // Scheduler opts into imbalance checking.
/// let sched_assert = Assert::NO_OVERRIDES.max_imbalance_ratio(5.0);
///
/// // Merge: defaults <- scheduler <- test.
/// let merged = Assert::default_checks()
///     .merge(&sched_assert)
///     .merge(&Assert::NO_OVERRIDES.max_gap_ms(5000));
///
/// assert_eq!(merged.not_starved, None);              // not opted in
/// assert_eq!(merged.max_imbalance_ratio, Some(5.0)); // from sched
/// assert_eq!(merged.max_gap_ms, Some(5000));         // from test
/// ```
#[must_use = "builder methods return a new Assert; discard means config is lost"]
#[derive(Clone, Copy, Debug)]
pub struct Assert {
    // Worker checks
    /// Enable starvation, fairness spread, and gap checks across
    /// worker reports. `Some(true)` enables, `Some(false)` explicitly
    /// disables (overriding any enabling merge from a lower layer),
    /// `None` inherits from the merge parent.
    pub not_starved: Option<bool>,
    /// Enable per-worker CPU isolation checks (ensure workers remain
    /// within their assigned cpuset). Same tri-state semantics as
    /// `not_starved`.
    pub isolation: Option<bool>,
    /// Max per-worker scheduling gap in milliseconds. Fails the
    /// assertion if any worker's longest off-CPU stretch exceeds this.
    pub max_gap_ms: Option<u64>,
    /// Max per-cgroup fairness spread as a percentage. Fails if the
    /// range between the most- and least-served workers exceeds this
    /// fraction of their mean.
    pub max_spread_pct: Option<f64>,

    // Throughput checks
    /// Max coefficient of variation for work_units/cpu_time across workers.
    /// Catches placement unfairness where some workers get less CPU than others.
    pub max_throughput_cv: Option<f64>,
    /// Minimum work_units per CPU-second. Catches cases where all workers
    /// are equally slow (CV passes but absolute throughput is too low).
    pub min_work_rate: Option<f64>,

    // Benchmarking checks
    /// Max p99 wake latency in NANOSECONDS. Fails if the pooled
    /// p99 across every worker's `resume_latencies_ns` exceeds this.
    ///
    /// # Unit-name gotcha
    ///
    /// The threshold is `_ns`, but the paired reporting field on
    /// [`CgroupStats::p99_wake_latency_us`] and the roll-up
    /// [`ScenarioStats::worst_p99_wake_latency_us`] are
    /// MICROSECONDS. The two surfaces are intentionally split:
    ///   - the threshold uses NS for precision (typical scheduler
    ///     wake latencies are single-digit µs, so sub-µs resolution
    ///     matters for regression gates);
    ///   - the reporting fields use US for readability in
    ///     `stats compare` / dashboard output.
    ///
    /// Both are computed from the same underlying
    /// [`WorkerReport::resume_latencies_ns`] samples — see
    /// [`assert_benchmarks`] for the threshold path and
    /// [`assert_not_starved`] for the reporting path. A bare
    /// comparison of `max_p99_wake_latency_ns` against
    /// `CgroupStats::p99_wake_latency_us` is a unit-mismatch bug;
    /// `assert_benchmarks` never does this — it consumes the raw
    /// `resume_latencies_ns` directly — and
    /// `assert_p99_ns_threshold_compares_against_ns_latencies` pins
    /// that contract.
    pub max_p99_wake_latency_ns: Option<u64>,
    /// Max wake latency coefficient of variation. Fails if CV exceeds this.
    pub max_wake_latency_cv: Option<f64>,
    /// Minimum iterations per wall-clock second. Fails if any worker is below.
    pub min_iteration_rate: Option<f64>,
    /// Max migration ratio (migrations/iterations). Fails if any cgroup exceeds this.
    pub max_migration_ratio: Option<f64>,

    // Monitor checks
    /// Max `nr_running` / LLC imbalance ratio observed by the monitor.
    /// Fails if the worst sample's imbalance exceeds this.
    pub max_imbalance_ratio: Option<f64>,
    /// Max local DSQ depth observed by the monitor. Fails if any
    /// sampled CPU's local DSQ grew beyond this.
    pub max_local_dsq_depth: Option<u32>,
    /// Treat a stall verdict from the monitor as a hard failure. Same
    /// tri-state semantics as `not_starved`.
    pub fail_on_stall: Option<bool>,
    /// Minimum number of consecutive samples that must exceed the
    /// monitor threshold before a verdict is raised. Smooths out
    /// single-sample spikes.
    pub sustained_samples: Option<usize>,
    /// Max `select_cpu_fallback` rate (events/sec). Fails if the
    /// scx event counter delta over the run exceeds this rate.
    pub max_fallback_rate: Option<f64>,
    /// Max `keep_last` rate (events/sec). Fails if the scx event
    /// counter delta over the run exceeds this rate.
    pub max_keep_last_rate: Option<f64>,

    // NUMA checks
    /// Minimum fraction of pages on the expected NUMA node(s) (0.0-1.0).
    /// Expected nodes are derived from the worker's
    /// [`MemPolicy`](crate::workload::MemPolicy) at evaluation time.
    /// Fails if the observed locality fraction falls below this.
    pub min_page_locality: Option<f64>,
    /// Maximum ratio of NUMA-node-migrated pages to total allocated
    /// pages (0.0-1.0). Distinct from [`max_migration_ratio`](Self::max_migration_ratio)
    /// which measures CPU migrations per iteration. Fails if the
    /// observed migration ratio exceeds this.
    pub max_cross_node_migration_ratio: Option<f64>,
    /// Maximum fraction of pages on slow-tier (memory-only) NUMA nodes
    /// (0.0-1.0). For CXL memory tiering tests: fails if more than
    /// this fraction of pages land on memory-only nodes. Requires
    /// `slow_tier_nodes` to be set at evaluation time.
    pub max_slow_tier_ratio: Option<f64>,
}

impl Assert {
    /// Human-readable multi-line dump of every threshold field. Each
    /// field renders as `  name: value` (`none` when the option is
    /// `None`, i.e. inherited or unset). Used by
    /// `cargo ktstr show-thresholds <test>` to expose the resolved
    /// merged `Assert` (`default_checks().merge(entry.scheduler.assert()).
    /// merge(&entry.assert)`) without forcing the operator to read
    /// the Debug impl or source. Output is a sequence of indented
    /// `row` lines ending with a newline; the caller owns any
    /// outer section header (the `show-thresholds` CLI already
    /// prints `Test: ...` / `Scheduler: ...` lines above the
    /// threshold block, which together establish context — an
    /// additional `Resolved assertion thresholds:` banner here
    /// would be a redundant third header).
    pub fn format_human(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        fn row<T: std::fmt::Display>(out: &mut String, name: &str, v: &Option<T>) {
            match v {
                Some(x) => writeln!(out, "  {name:<38}: {x}").unwrap(),
                None => writeln!(out, "  {name:<38}: none").unwrap(),
            }
        }
        row(&mut out, "not_starved", &self.not_starved);
        row(&mut out, "isolation", &self.isolation);
        row(&mut out, "max_gap_ms", &self.max_gap_ms);
        row(&mut out, "max_spread_pct", &self.max_spread_pct);
        row(&mut out, "max_throughput_cv", &self.max_throughput_cv);
        row(&mut out, "min_work_rate", &self.min_work_rate);
        row(
            &mut out,
            "max_p99_wake_latency_ns",
            &self.max_p99_wake_latency_ns,
        );
        row(&mut out, "max_wake_latency_cv", &self.max_wake_latency_cv);
        row(&mut out, "min_iteration_rate", &self.min_iteration_rate);
        row(&mut out, "max_migration_ratio", &self.max_migration_ratio);
        row(&mut out, "max_imbalance_ratio", &self.max_imbalance_ratio);
        row(&mut out, "max_local_dsq_depth", &self.max_local_dsq_depth);
        row(&mut out, "fail_on_stall", &self.fail_on_stall);
        row(&mut out, "sustained_samples", &self.sustained_samples);
        row(&mut out, "max_fallback_rate", &self.max_fallback_rate);
        row(&mut out, "max_keep_last_rate", &self.max_keep_last_rate);
        row(&mut out, "min_page_locality", &self.min_page_locality);
        row(
            &mut out,
            "max_cross_node_migration_ratio",
            &self.max_cross_node_migration_ratio,
        );
        row(&mut out, "max_slow_tier_ratio", &self.max_slow_tier_ratio);
        out
    }

    /// Identity element for [`Assert::merge`]: every field is `None`,
    /// so neither side of a merge with `NO_OVERRIDES` is altered.
    /// Equivalent to [`Self::default_checks`].
    pub const NO_OVERRIDES: Assert = Assert {
        not_starved: None,
        isolation: None,
        max_gap_ms: None,
        max_spread_pct: None,
        max_throughput_cv: None,
        min_work_rate: None,
        max_p99_wake_latency_ns: None,
        max_wake_latency_cv: None,
        min_iteration_rate: None,
        max_migration_ratio: None,
        max_imbalance_ratio: None,
        max_local_dsq_depth: None,
        fail_on_stall: None,
        sustained_samples: None,
        max_fallback_rate: None,
        max_keep_last_rate: None,
        min_page_locality: None,
        max_cross_node_migration_ratio: None,
        max_slow_tier_ratio: None,
    };

    /// Baseline of the runtime merge chain
    /// `default_checks().merge(&scheduler.assert).merge(&entry.assert)`.
    ///
    /// All checks are off by default — tests opt in to the assertions
    /// they care about via scheduler-level or per-test `Assert`
    /// overrides.
    pub const fn default_checks() -> Assert {
        Self::NO_OVERRIDES
    }

    /// Build a fresh [`Verdict`] under this `Assert`'s threshold
    /// config. The returned accumulator carries no claim records; call
    /// the typed `claim_<field>` methods generated by
    /// [`#[derive(Claim)]`](ktstr_macros::Claim) on stats structs as
    /// `stats.claim_<field>(&mut verdict)`, or use the
    /// [`claim!`](crate::claim) macro on a local/expression, then
    /// call [`Verdict::into_result`] to produce the final
    /// [`AssertResult`].
    ///
    /// This is the entry point of the pointwise-claim API. The
    /// `Assert` itself remains pure threshold config and stays
    /// `Copy`; per-test claims accumulate on the returned `Verdict`,
    /// which owns its own buffers (details, stats).
    ///
    /// ```
    /// # use ktstr::assert::Assert;
    /// let r = Assert::defaults().verdict().into_result();
    /// assert!(r.passed, "no claims means passing verdict");
    /// ```
    pub fn verdict(self) -> Verdict {
        Verdict::with_assert(self)
    }

    /// Identity-element constructor (equivalent to [`Self::NO_OVERRIDES`]).
    ///
    /// Replaces `NO_OVERRIDES` as the canonical name in the new
    /// claim-API surface — the constant remains available for
    /// merge-chain composition; `empty()` is the method-style entry
    /// point that pairs naturally with `.verdict()` for tests that
    /// don't want any threshold defaults to fire.
    pub const fn empty() -> Self {
        Self::NO_OVERRIDES
    }

    /// Default-checks constructor (equivalent to [`Self::default_checks`]).
    ///
    /// Method-style alias for the existing const fn; pairs with
    /// `.verdict()` so the canonical entry point reads
    /// `Assert::defaults().verdict()`.
    pub const fn defaults() -> Self {
        Self::default_checks()
    }

    pub const fn check_not_starved(mut self) -> Self {
        self.not_starved = Some(true);
        self
    }

    pub const fn check_isolation(mut self) -> Self {
        self.isolation = Some(true);
        self
    }

    pub const fn max_gap_ms(mut self, ms: u64) -> Self {
        self.max_gap_ms = Some(ms);
        self
    }

    pub const fn max_spread_pct(mut self, pct: f64) -> Self {
        self.max_spread_pct = Some(pct);
        self
    }

    pub const fn max_throughput_cv(mut self, v: f64) -> Self {
        self.max_throughput_cv = Some(v);
        self
    }

    pub const fn min_work_rate(mut self, v: f64) -> Self {
        self.min_work_rate = Some(v);
        self
    }

    pub const fn max_p99_wake_latency_ns(mut self, v: u64) -> Self {
        self.max_p99_wake_latency_ns = Some(v);
        self
    }

    pub const fn max_wake_latency_cv(mut self, v: f64) -> Self {
        self.max_wake_latency_cv = Some(v);
        self
    }

    pub const fn min_iteration_rate(mut self, v: f64) -> Self {
        self.min_iteration_rate = Some(v);
        self
    }

    pub const fn max_migration_ratio(mut self, v: f64) -> Self {
        self.max_migration_ratio = Some(v);
        self
    }

    pub const fn max_imbalance_ratio(mut self, v: f64) -> Self {
        self.max_imbalance_ratio = Some(v);
        self
    }

    pub const fn max_local_dsq_depth(mut self, v: u32) -> Self {
        self.max_local_dsq_depth = Some(v);
        self
    }

    /// Control whether a monitor stall verdict fails the assertion.
    pub const fn fail_on_stall(mut self, v: bool) -> Self {
        self.fail_on_stall = Some(v);
        self
    }

    /// Set the number of consecutive over-threshold samples required
    /// before the monitor raises a verdict.
    pub const fn sustained_samples(mut self, v: usize) -> Self {
        self.sustained_samples = Some(v);
        self
    }

    pub const fn max_fallback_rate(mut self, v: f64) -> Self {
        self.max_fallback_rate = Some(v);
        self
    }

    pub const fn max_keep_last_rate(mut self, v: f64) -> Self {
        self.max_keep_last_rate = Some(v);
        self
    }

    pub const fn min_page_locality(mut self, v: f64) -> Self {
        self.min_page_locality = Some(v);
        self
    }

    pub const fn max_cross_node_migration_ratio(mut self, v: f64) -> Self {
        self.max_cross_node_migration_ratio = Some(v);
        self
    }

    pub const fn max_slow_tier_ratio(mut self, v: f64) -> Self {
        self.max_slow_tier_ratio = Some(v);
        self
    }

    /// True when any worker-level check field is `Some`.
    pub const fn has_worker_checks(&self) -> bool {
        self.not_starved.is_some()
            || self.isolation.is_some()
            || self.max_gap_ms.is_some()
            || self.max_spread_pct.is_some()
            || self.max_throughput_cv.is_some()
            || self.min_work_rate.is_some()
            || self.max_p99_wake_latency_ns.is_some()
            || self.max_wake_latency_cv.is_some()
            || self.min_iteration_rate.is_some()
            || self.max_migration_ratio.is_some()
            || self.min_page_locality.is_some()
            || self.max_cross_node_migration_ratio.is_some()
            || self.max_slow_tier_ratio.is_some()
    }

    /// Merge `other` on top of `self`. Each `Some` field in `other`
    /// overrides the corresponding field in `self`; `None` fields
    /// inherit from `self`.
    ///
    /// [`Assert::NO_OVERRIDES`] is the two-sided identity:
    /// `x.merge(&NO_OVERRIDES)` and `NO_OVERRIDES.merge(&x)` both yield
    /// `x`. The runtime composes scheduler- and test-level overrides as
    /// `Assert::default_checks().merge(&scheduler.assert).merge(&test.assert)`,
    /// so a `NO_OVERRIDES` at either override layer leaves the defaults
    /// untouched -- which means "no override," not "no checks."
    pub const fn merge(&self, other: &Assert) -> Assert {
        // `Option::or` is not yet const-stable, so each field expands
        // a match rather than calling `other.x.or(self.x)`. Keep it
        // this way until `const fn` can call `Option::or`; at that
        // point the 19 match blocks collapse to 19 `.or()` calls.
        Assert {
            not_starved: match other.not_starved {
                Some(v) => Some(v),
                None => self.not_starved,
            },
            isolation: match other.isolation {
                Some(v) => Some(v),
                None => self.isolation,
            },
            max_gap_ms: match other.max_gap_ms {
                Some(v) => Some(v),
                None => self.max_gap_ms,
            },
            max_spread_pct: match other.max_spread_pct {
                Some(v) => Some(v),
                None => self.max_spread_pct,
            },
            max_throughput_cv: match other.max_throughput_cv {
                Some(v) => Some(v),
                None => self.max_throughput_cv,
            },
            min_work_rate: match other.min_work_rate {
                Some(v) => Some(v),
                None => self.min_work_rate,
            },
            max_p99_wake_latency_ns: match other.max_p99_wake_latency_ns {
                Some(v) => Some(v),
                None => self.max_p99_wake_latency_ns,
            },
            max_wake_latency_cv: match other.max_wake_latency_cv {
                Some(v) => Some(v),
                None => self.max_wake_latency_cv,
            },
            min_iteration_rate: match other.min_iteration_rate {
                Some(v) => Some(v),
                None => self.min_iteration_rate,
            },
            max_migration_ratio: match other.max_migration_ratio {
                Some(v) => Some(v),
                None => self.max_migration_ratio,
            },
            max_imbalance_ratio: match other.max_imbalance_ratio {
                Some(v) => Some(v),
                None => self.max_imbalance_ratio,
            },
            max_local_dsq_depth: match other.max_local_dsq_depth {
                Some(v) => Some(v),
                None => self.max_local_dsq_depth,
            },
            fail_on_stall: match other.fail_on_stall {
                Some(v) => Some(v),
                None => self.fail_on_stall,
            },
            sustained_samples: match other.sustained_samples {
                Some(v) => Some(v),
                None => self.sustained_samples,
            },
            max_fallback_rate: match other.max_fallback_rate {
                Some(v) => Some(v),
                None => self.max_fallback_rate,
            },
            max_keep_last_rate: match other.max_keep_last_rate {
                Some(v) => Some(v),
                None => self.max_keep_last_rate,
            },
            min_page_locality: match other.min_page_locality {
                Some(v) => Some(v),
                None => self.min_page_locality,
            },
            max_cross_node_migration_ratio: match other.max_cross_node_migration_ratio {
                Some(v) => Some(v),
                None => self.max_cross_node_migration_ratio,
            },
            max_slow_tier_ratio: match other.max_slow_tier_ratio {
                Some(v) => Some(v),
                None => self.max_slow_tier_ratio,
            },
        }
    }

    /// Extract an `AssertPlan` for worker-side checks.
    pub(crate) fn worker_plan(&self) -> AssertPlan {
        AssertPlan {
            not_starved: self.not_starved.unwrap_or(false),
            isolation: self.isolation.unwrap_or(false),
            max_gap_ms: self.max_gap_ms,
            max_spread_pct: self.max_spread_pct,
            max_throughput_cv: self.max_throughput_cv,
            min_work_rate: self.min_work_rate,
            max_p99_wake_latency_ns: self.max_p99_wake_latency_ns,
            max_wake_latency_cv: self.max_wake_latency_cv,
            min_iteration_rate: self.min_iteration_rate,
            max_migration_ratio: self.max_migration_ratio,
            min_page_locality: self.min_page_locality,
            max_cross_node_migration_ratio: self.max_cross_node_migration_ratio,
            max_slow_tier_ratio: self.max_slow_tier_ratio,
        }
    }

    /// Run the configured worker checks against one cgroup's reports.
    ///
    /// `cpuset` is the CPU set for isolation checks. `numa_nodes` is
    /// the NUMA node IDs covered by the cpuset (for page locality and
    /// slow-tier checks). Derive via
    /// [`TestTopology::numa_nodes_for_cpuset`](crate::topology::TestTopology::numa_nodes_for_cpuset).
    pub fn assert_cgroup(
        &self,
        reports: &[crate::workload::WorkerReport],
        cpuset: Option<&BTreeSet<usize>>,
    ) -> AssertResult {
        self.worker_plan().assert_cgroup(reports, cpuset, None)
    }

    /// Run worker checks with explicit NUMA node set for page locality.
    pub fn assert_cgroup_with_numa(
        &self,
        reports: &[crate::workload::WorkerReport],
        cpuset: Option<&BTreeSet<usize>>,
        numa_nodes: Option<&BTreeSet<usize>>,
    ) -> AssertResult {
        self.worker_plan()
            .assert_cgroup(reports, cpuset, numa_nodes)
    }

    /// Run NUMA page locality check.
    ///
    /// `observed` is the fraction of pages on expected nodes (0.0-1.0).
    /// `total_pages` and `local_pages` are for diagnostics.
    pub fn assert_page_locality(
        &self,
        observed: f64,
        total_pages: u64,
        local_pages: u64,
    ) -> AssertResult {
        assert_page_locality(observed, self.min_page_locality, total_pages, local_pages)
    }

    /// Run cross-node migration ratio check.
    ///
    /// `migrated_pages` is the `/proc/vmstat` `numa_pages_migrated` delta.
    /// `total_pages` is total allocated pages from numa_maps.
    pub fn assert_cross_node_migration(
        &self,
        migrated_pages: u64,
        total_pages: u64,
    ) -> AssertResult {
        assert_cross_node_migration(
            migrated_pages,
            total_pages,
            self.max_cross_node_migration_ratio,
        )
    }

    /// Extract `MonitorThresholds` for monitor-side evaluation.
    pub(crate) fn has_monitor_thresholds(&self) -> bool {
        self.max_imbalance_ratio.is_some()
            || self.max_local_dsq_depth.is_some()
            || self.fail_on_stall.is_some()
            || self.sustained_samples.is_some()
            || self.max_fallback_rate.is_some()
            || self.max_keep_last_rate.is_some()
    }

    pub(crate) fn monitor_thresholds(&self) -> crate::monitor::MonitorThresholds {
        use crate::monitor::MonitorThresholds;
        let d = MonitorThresholds::DEFAULT;
        MonitorThresholds {
            max_imbalance_ratio: self.max_imbalance_ratio.unwrap_or(d.max_imbalance_ratio),
            max_local_dsq_depth: self.max_local_dsq_depth.unwrap_or(d.max_local_dsq_depth),
            fail_on_stall: self.fail_on_stall.unwrap_or(d.fail_on_stall),
            sustained_samples: self.sustained_samples.unwrap_or(d.sustained_samples),
            max_fallback_rate: self.max_fallback_rate.unwrap_or(d.max_fallback_rate),
            max_keep_last_rate: self.max_keep_last_rate.unwrap_or(d.max_keep_last_rate),
        }
    }
}

pub mod claim;
pub mod temporal;

pub use claim::{ClaimBuilder, SeqClaim, SetClaim, Verdict};
pub use temporal::{EachClaim, SeriesField};

/// Check that workers only ran on CPUs in `expected`.
///
/// Any worker that used a CPU outside the expected set produces a
/// failure with the unexpected CPU IDs listed.
///
/// ```
/// # use ktstr::assert::assert_isolation;
/// # use ktstr::workload::WorkerReport;
/// # use std::collections::BTreeSet;
/// # let report = WorkerReport {
/// #     tid: 1, cpus_used: [0, 1].into_iter().collect(),
/// #     work_units: 100, cpu_time_ns: 1_000_000, wall_time_ns: 2_000_000,
/// #     off_cpu_ns: 1_000_000, migration_count: 0, migrations: vec![],
/// #     max_gap_ms: 0, max_gap_cpu: 0, max_gap_at_ms: 0,
/// #     resume_latencies_ns: vec![], wake_sample_total: 0,
/// #     iteration_costs_ns: vec![], iteration_cost_sample_total: 0,
/// #     iterations: 0,
/// #     schedstat_run_delay_ns: 0, schedstat_run_count: 0,
/// #     schedstat_cpu_time_ns: 0,
/// #     completed: true,
/// #     numa_pages: std::collections::BTreeMap::new(),
/// #     vmstat_numa_pages_migrated: 0,
/// #     exit_info: None,
/// #     is_messenger: false,
/// #     ..Default::default()
/// # };
/// let expected: BTreeSet<usize> = [0, 1, 2].into_iter().collect();
/// assert!(assert_isolation(&[report], &expected).passed);
/// ```
pub fn assert_isolation(reports: &[WorkerReport], expected: &BTreeSet<usize>) -> AssertResult {
    let mut r = AssertResult::pass();
    for w in reports {
        let bad: BTreeSet<usize> = w.cpus_used.difference(expected).copied().collect();
        if !bad.is_empty() {
            r.passed = false;
            r.details.push(AssertDetail::new(
                DetailKind::Isolation,
                format!("tid {} ran on unexpected CPUs {:?}", w.tid, bad),
            ));
        }
    }
    r
}

/// Check one cgroup's workers. Returns per-cgroup stats.
///
/// ```
/// # use ktstr::assert::assert_not_starved;
/// # use ktstr::workload::WorkerReport;
/// # let report = WorkerReport {
/// #     tid: 1, cpus_used: [0].into_iter().collect(),
/// #     work_units: 100, cpu_time_ns: 1_000_000, wall_time_ns: 5_000_000_000,
/// #     off_cpu_ns: 500_000_000, migration_count: 0, migrations: vec![],
/// #     max_gap_ms: 50, max_gap_cpu: 0, max_gap_at_ms: 1000,
/// #     resume_latencies_ns: vec![], wake_sample_total: 0,
/// #     iteration_costs_ns: vec![], iteration_cost_sample_total: 0,
/// #     iterations: 0,
/// #     schedstat_run_delay_ns: 0, schedstat_run_count: 0,
/// #     schedstat_cpu_time_ns: 0,
/// #     completed: true,
/// #     numa_pages: std::collections::BTreeMap::new(),
/// #     vmstat_numa_pages_migrated: 0,
/// #     exit_info: None,
/// #     is_messenger: false,
/// #     ..Default::default()
/// # };
/// let r = assert_not_starved(&[report]);
/// assert!(r.passed);
/// assert_eq!(r.stats.total_workers, 1);
/// ```
/// Nearest-rank percentile of a sorted slice (`p` in `[0.0, 1.0]`).
///
/// Returns the value at index `ceil(n * p) - 1`, clamped into
/// `[0, n-1]`. For `n = 100` and `p = 0.99` this is `sorted[98]` (the
/// 99th element in 1-indexed order), not `sorted[99]` (the max). The
/// previous formulation, `ceil(n * 0.99)` without the `-1`, was
/// off-by-one and returned the max for `n = 100`.
///
/// # Preconditions
///
/// `sorted` must be non-decreasing. The function indexes by rank
/// without checking order, so an unsorted input silently returns
/// the value at the computed index — a meaningless number. A
/// `debug_assert!` enforces this in debug builds; release builds
/// skip the check (the production callers sort immediately upstream
/// — `assert_not_starved` and `assert_benchmarks` both
/// `sorted.sort_unstable()` before this call — so the runtime
/// guard is unnecessary in production paths).
///
/// An empty slice yields `0` (the caller should short-circuit
/// before invoking).
fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    debug_assert!(
        sorted.windows(2).all(|w| w[0] <= w[1]),
        "percentile() requires sorted input; got slice with out-of-order pair",
    );
    let n = sorted.len();
    let idx = ((n as f64 * p).ceil() as usize)
        .saturating_sub(1)
        .min(n - 1);
    sorted[idx]
}

pub fn assert_not_starved(reports: &[WorkerReport]) -> AssertResult {
    let mut r = AssertResult::pass();
    if reports.is_empty() {
        return r;
    }

    let cpus: BTreeSet<usize> = reports
        .iter()
        .flat_map(|w| w.cpus_used.iter().copied())
        .collect();
    let mut pcts: Vec<f64> = Vec::new();

    for w in reports {
        if w.work_units == 0 {
            r.passed = false;
            r.details.push(AssertDetail::new(
                DetailKind::Starved,
                format!("tid {} starved (0 work units)", w.tid),
            ));
        }
        if w.wall_time_ns > 0 {
            pcts.push(w.off_cpu_ns as f64 / w.wall_time_ns as f64 * 100.0);
        }
    }

    let min = pcts.iter().cloned().reduce(f64::min).unwrap_or(0.0);
    let max = pcts.iter().cloned().reduce(f64::max).unwrap_or(0.0);
    let avg = if pcts.is_empty() {
        0.0
    } else {
        pcts.iter().sum::<f64>() / pcts.len() as f64
    };
    let spread = max - min;

    let worst_gap = reports.iter().max_by_key(|w| w.max_gap_ms);
    let (gap_ms, gap_cpu) = worst_gap
        .map(|w| (w.max_gap_ms, w.max_gap_cpu))
        .unwrap_or((0, 0));

    // Compute benchmarking stats from worker reports.
    let all_latencies: Vec<u64> = reports
        .iter()
        .flat_map(|w| w.resume_latencies_ns.iter().copied())
        .collect();
    let (p99_us, median_us, lat_cv) = if all_latencies.is_empty() {
        (0.0, 0.0, 0.0)
    } else {
        let mut sorted = all_latencies.clone();
        sorted.sort_unstable();
        let p99 = percentile(&sorted, 0.99) as f64 / 1000.0;
        // Median routes through `percentile(sorted, 0.5)` so the
        // nearest-rank algorithm matches every other percentile in
        // the project (p99, schbench's `lat99`, the BPF latency
        // histograms). A bare `sorted[n/2]` would pick the upper of
        // the two middle samples for even `n`, while `percentile`
        // returns the value at `ceil(n * 0.5) - 1` — the lower of
        // the two middles — and that lower-bound convention is what
        // the docs on [`CgroupStats::median_wake_latency_us`] and
        // the schbench cross-reference promise.
        let median = percentile(&sorted, 0.5) as f64 / 1000.0;
        let n = all_latencies.len() as f64;
        let mean_ns = all_latencies.iter().sum::<u64>() as f64 / n;
        let cv = if mean_ns > 0.0 {
            let variance = all_latencies
                .iter()
                .map(|&v| (v as f64 - mean_ns).powi(2))
                .sum::<f64>()
                / n;
            variance.sqrt() / mean_ns
        } else {
            0.0
        };
        (p99, median, cv)
    };

    let total_iters: u64 = reports.iter().map(|w| w.iterations).sum();
    let run_delays: Vec<f64> = reports
        .iter()
        .map(|w| w.schedstat_run_delay_ns as f64 / 1000.0)
        .collect();
    let mean_run_delay = if run_delays.is_empty() {
        0.0
    } else {
        run_delays.iter().sum::<f64>() / run_delays.len() as f64
    };
    let worst_run_delay = run_delays.iter().cloned().reduce(f64::max).unwrap_or(0.0);

    let total_mig: u64 = reports.iter().map(|w| w.migration_count).sum();
    let mig_ratio = if total_iters > 0 {
        total_mig as f64 / total_iters as f64
    } else {
        0.0
    };

    let cg = CgroupStats {
        num_workers: reports.len(),
        num_cpus: cpus.len(),
        avg_off_cpu_pct: avg,
        min_off_cpu_pct: min,
        max_off_cpu_pct: max,
        spread,
        max_gap_ms: gap_ms,
        max_gap_cpu: gap_cpu,
        total_migrations: total_mig,
        migration_ratio: mig_ratio,
        p99_wake_latency_us: p99_us,
        median_wake_latency_us: median_us,
        wake_latency_cv: lat_cv,
        total_iterations: total_iters,
        mean_run_delay_us: mean_run_delay,
        worst_run_delay_us: worst_run_delay,
        page_locality: 0.0,
        cross_node_migration_ratio: 0.0,
        ext_metrics: BTreeMap::new(),
    };

    // Per-cgroup fairness: spread above threshold means unequal scheduling within a cgroup.
    // Threshold is appended to the message so the detail carries the exact bound the
    // observed spread crossed, matching the AssertPlan custom-spread path's format
    // and giving the operator the gate value without re-grepping `show-thresholds`.
    let spread_limit = spread_threshold_pct();
    if spread > spread_limit && pcts.len() >= 2 {
        r.passed = false;
        r.details.push(AssertDetail::new(
            DetailKind::Unfair,
            format!(
                "unfair cgroup: spread={:.0}% ({:.0}-{:.0}%) {} workers on {} cpus (threshold {:.0}%)",
                spread,
                min,
                max,
                reports.len(),
                cpus.len(),
                spread_limit,
            ),
        ));
    }

    // Scheduling gap: >threshold = dispatch failure. The tid is included so an
    // operator triaging a multi-worker cgroup can identify the affected worker
    // without cross-referencing CPU placement; matches the `tid X starved` /
    // `tid X ran on unexpected CPUs` shape used by the sibling diagnostics.
    // Threshold is appended for parity with the AssertPlan custom-gap path.
    let gap_limit = gap_threshold_ms();
    for w in reports {
        if w.max_gap_ms > gap_limit {
            r.passed = false;
            r.details.push(AssertDetail::new(
                DetailKind::Stuck,
                format!(
                    "tid {} stuck {}ms on cpu{} at +{}ms (threshold {}ms)",
                    w.tid, w.max_gap_ms, w.max_gap_cpu, w.max_gap_at_ms, gap_limit,
                ),
            ));
        }
    }

    // Store this cgroup's stats - merge accumulates cgroups
    r.stats = ScenarioStats {
        total_workers: reports.len(),
        total_cpus: cpus.len(),
        total_migrations: reports.iter().map(|w| w.migration_count).sum(),
        worst_spread: spread,
        worst_gap_ms: gap_ms,
        worst_gap_cpu: gap_cpu,
        worst_migration_ratio: cg.migration_ratio,
        worst_p99_wake_latency_us: cg.p99_wake_latency_us,
        worst_median_wake_latency_us: cg.median_wake_latency_us,
        worst_wake_latency_cv: cg.wake_latency_cv,
        total_iterations: cg.total_iterations,
        worst_mean_run_delay_us: cg.mean_run_delay_us,
        worst_run_delay_us: cg.worst_run_delay_us,
        worst_page_locality: 0.0,
        worst_cross_node_migration_ratio: 0.0,
        worst_wake_latency_tail_ratio: cg.wake_latency_tail_ratio(),
        // `iterations_per_worker()` returns the per-worker
        // throughput for this cgroup. The merge fold treats 0.0
        // as the unreported sentinel — the accumulator pattern
        // `AssertResult::pass().merge(real)` starts at 0.0 from
        // `Default`, so any positive reading from a real
        // measurement must override the sentinel rather than be
        // masked by a plain min.
        worst_iterations_per_worker: cg.iterations_per_worker(),
        ext_metrics: cg.ext_metrics.clone(),
        cgroups: vec![cg],
    };

    r
}

/// Check throughput parity across workers: coefficient of variation and
/// minimum work rate.
///
/// `max_cv`: maximum allowed coefficient of variation (stddev/mean) for
/// work_units / cpu_time_ns across workers. `None` skips the CV check.
///
/// `min_rate`: minimum work_units per CPU-second. `None` skips the floor check.
///
/// ```
/// # use ktstr::assert::assert_throughput_parity;
/// # use ktstr::workload::WorkerReport;
/// # let mk = |units, cpu_ns| WorkerReport {
/// #     tid: 1, cpus_used: [0].into_iter().collect(),
/// #     work_units: units, cpu_time_ns: cpu_ns, wall_time_ns: cpu_ns,
/// #     off_cpu_ns: cpu_ns, migration_count: 0, migrations: vec![],
/// #     max_gap_ms: 0, max_gap_cpu: 0, max_gap_at_ms: 0,
/// #     resume_latencies_ns: vec![], wake_sample_total: 0,
/// #     iteration_costs_ns: vec![], iteration_cost_sample_total: 0,
/// #     iterations: 0,
/// #     schedstat_run_delay_ns: 0, schedstat_run_count: 0,
/// #     schedstat_cpu_time_ns: 0,
/// #     completed: true,
/// #     numa_pages: std::collections::BTreeMap::new(),
/// #     vmstat_numa_pages_migrated: 0,
/// #     exit_info: None,
/// #     is_messenger: false,
/// #     ..Default::default()
/// # };
/// // Equal throughput -> low CV -> passes.
/// let reports = [mk(1000, 1_000_000_000), mk(1000, 1_000_000_000)];
/// assert!(assert_throughput_parity(&reports, Some(0.5), None).passed);
/// ```
pub fn assert_throughput_parity(
    reports: &[WorkerReport],
    max_cv: Option<f64>,
    min_rate: Option<f64>,
) -> AssertResult {
    let mut r = AssertResult::pass();
    if reports.is_empty() {
        return r;
    }

    // Compute per-worker throughput: work_units / cpu_seconds
    let rates: Vec<f64> = reports
        .iter()
        .map(|w| {
            if w.cpu_time_ns == 0 {
                0.0
            } else {
                w.work_units as f64 / (w.cpu_time_ns as f64 / 1e9)
            }
        })
        .collect();

    let n = rates.len() as f64;
    let mean = rates.iter().sum::<f64>() / n;

    if let Some(cv_limit) = max_cv {
        // Guard zero-mean explicitly: a CV is undefined when every
        // rate is zero, and silently passing the check would let a
        // run where every worker recorded zero cpu_time look "in
        // parity" when in fact no worker accumulated any CPU time
        // at all. Surface the broken state so the operator sees it
        // instead of letting `max_throughput_cv` look green.
        let all_zero_cpu = reports.iter().all(|w| w.cpu_time_ns == 0);
        if all_zero_cpu {
            r.passed = false;
            r.details.push(AssertDetail::new(
                DetailKind::Benchmark,
                format!(
                    "throughput CV undefined: all {} workers recorded zero cpu_time_ns (limit {cv_limit:.3})",
                    reports.len()
                ),
            ));
        } else if mean > 0.0 && rates.len() >= 2 {
            let variance = rates.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
            let stddev = variance.sqrt();
            let cv = stddev / mean;
            if cv > cv_limit {
                r.passed = false;
                r.details.push(AssertDetail::new(
                    DetailKind::Benchmark,
                    format!(
                        "throughput CV {cv:.3} exceeds limit {cv_limit:.3} (mean={mean:.0} work/cpu_s)"
                    ),
                ));
            }
        }
    }

    if let Some(floor) = min_rate {
        for (i, &rate) in rates.iter().enumerate() {
            if rate < floor {
                r.passed = false;
                r.details.push(AssertDetail::new(
                    DetailKind::Benchmark,
                    format!(
                        "worker {} throughput {rate:.0} work/cpu_s below floor {floor:.0}",
                        reports[i].tid
                    ),
                ));
            }
        }
    }

    r
}

/// Check benchmarking metrics: p99 wake latency, wake latency CV,
/// and minimum iteration rate.
///
/// ```
/// # use ktstr::assert::assert_benchmarks;
/// # use ktstr::workload::WorkerReport;
/// # let report = WorkerReport {
/// #     tid: 1, cpus_used: [0].into_iter().collect(),
/// #     work_units: 1000, cpu_time_ns: 2_500_000_000,
/// #     wall_time_ns: 5_000_000_000, off_cpu_ns: 2_500_000_000,
/// #     migration_count: 0, migrations: vec![],
/// #     max_gap_ms: 50, max_gap_cpu: 0, max_gap_at_ms: 1000,
/// #     resume_latencies_ns: vec![100, 200, 300, 400, 500],
/// #     wake_sample_total: 5,
/// #     iteration_costs_ns: vec![], iteration_cost_sample_total: 0,
/// #     iterations: 1000,
/// #     schedstat_run_delay_ns: 0, schedstat_run_count: 0,
/// #     schedstat_cpu_time_ns: 0,
/// #     completed: true,
/// #     numa_pages: std::collections::BTreeMap::new(),
/// #     vmstat_numa_pages_migrated: 0,
/// #     exit_info: None,
/// #     is_messenger: false,
/// #     ..Default::default()
/// # };
/// // p99 = 500ns, well under 10000ns limit.
/// assert!(assert_benchmarks(&[report], Some(10000), None, None).passed);
/// ```
pub fn assert_benchmarks(
    reports: &[WorkerReport],
    max_p99_ns: Option<u64>,
    max_cv: Option<f64>,
    min_iter_rate: Option<f64>,
) -> AssertResult {
    let mut r = AssertResult::pass();
    if reports.is_empty() {
        // No worker reports means nothing to measure — any benchmark
        // threshold the caller supplied cannot be evaluated. A silent
        // pass would let thresholds look "green" on a broken run that
        // never produced signal; surface it as skip so the operator
        // knows the benchmark was not actually exercised.
        return AssertResult::skip("no worker reports — benchmark skipped");
    }

    // Collect all wake latencies across workers.
    let all_latencies: Vec<u64> = reports
        .iter()
        .flat_map(|w| w.resume_latencies_ns.iter().copied())
        .collect();

    if let Some(p99_limit) = max_p99_ns
        && !all_latencies.is_empty()
    {
        let mut sorted = all_latencies.clone();
        sorted.sort_unstable();
        let p99 = percentile(&sorted, 0.99);
        if p99 > p99_limit {
            r.passed = false;
            r.details.push(AssertDetail::new(
                DetailKind::Benchmark,
                format!(
                    "p99 wake latency {p99}ns exceeds limit {p99_limit}ns ({} samples)",
                    sorted.len()
                ),
            ));
        }
    }

    if let Some(cv_limit) = max_cv
        && all_latencies.len() >= 2
    {
        let n = all_latencies.len() as f64;
        let mean = all_latencies.iter().sum::<u64>() as f64 / n;
        if mean > 0.0 {
            let variance = all_latencies
                .iter()
                .map(|&v| (v as f64 - mean).powi(2))
                .sum::<f64>()
                / n;
            let cv = variance.sqrt() / mean;
            if cv > cv_limit {
                r.passed = false;
                r.details.push(AssertDetail::new(
                    DetailKind::Benchmark,
                    format!(
                        "wake latency CV {cv:.3} exceeds limit {cv_limit:.3} (mean={mean:.0}ns)"
                    ),
                ));
            }
        }
    }

    if let Some(rate_floor) = min_iter_rate {
        for w in reports {
            if w.wall_time_ns == 0 {
                continue;
            }
            let rate = w.iterations as f64 / (w.wall_time_ns as f64 / 1e9);
            if rate < rate_floor {
                r.passed = false;
                r.details.push(AssertDetail::new(
                    DetailKind::Benchmark,
                    format!(
                        "worker {} iteration rate {rate:.1}/s below floor {rate_floor:.1}/s",
                        w.tid
                    ),
                ));
            }
        }
    }

    r
}

/// Assert that every SCX event counter in `events` is at or below
/// `max_count`. `events` is a slice of `(name, count)` pairs sourced
/// from the kernel's per-task `scx_event_stats` (see `kernel/sched/ext.c`,
/// `SCX_EV_*` macros) — typically aggregated and surfaced via
/// `monitor::ScxEventDeltas` or sidecar `GauntletRow.fallback_count` /
/// `keep_last_count` fields. Pass `None` for `max_count` to require zero
/// (the strict default — error-class events should not fire under a
/// healthy scheduler).
///
/// The assertion is decoupled from the `monitor` module on purpose:
/// callers harvest the counters they care about (via the live monitor
/// path or by reading sidecar JSON post-hoc) and feed name/count
/// pairs in. This keeps the assert API surface decoupled from the
/// kernel-side counter inventory, which evolves across kernel
/// versions — adding a new `SCX_EV_*` does not force an API change
/// here.
///
/// Returns a passing result if every counter is within bound; failures
/// concatenate one [`AssertDetail`] per offending counter under
/// [`DetailKind::SchedulerEvent`] so an operator can identify which
/// events fired without scanning the full counter set.
///
/// ```
/// # use ktstr::assert::assert_scx_events_clean;
/// // Strict default — every counter must be zero.
/// let r = assert_scx_events_clean(&[("enq_skip_exiting", 0), ("dispatch_offline", 0)], None);
/// assert!(r.passed);
///
/// // A non-zero error-class counter fails.
/// let r = assert_scx_events_clean(&[("enq_skip_exiting", 7)], None);
/// assert!(!r.passed);
///
/// // Caller-supplied bound tolerates small counts.
/// let r = assert_scx_events_clean(&[("dispatch_keep_last", 3)], Some(10));
/// assert!(r.passed);
/// ```
pub fn assert_scx_events_clean(events: &[(&str, i64)], max_count: Option<i64>) -> AssertResult {
    let mut r = AssertResult::pass();
    for (name, count) in events {
        // Kernel `scx_event_stats` counters are monotonic u64 — a
        // negative i64 here means the source data is corrupted
        // (counter reset, wraparound on a signed conversion, or
        // sidecar JSON bit-loss). Treat negatives as failures rather
        // than letting them silently pass `*count > bound` for any
        // non-negative bound.
        let failed = match max_count {
            // Strict default: every counter must be exactly zero.
            // `*count > 0` would let -5 slip through.
            None => *count != 0,
            // Bounded: reject negatives explicitly, then enforce
            // the upper bound.
            Some(bound) => *count < 0 || *count > bound,
        };
        if failed {
            r.passed = false;
            let bound_desc = match max_count {
                None => "0".to_string(),
                Some(b) => b.to_string(),
            };
            r.details.push(AssertDetail::new(
                DetailKind::SchedulerEvent,
                format!("scx event `{name}` count {count} exceeds bound {bound_desc}",),
            ));
        }
    }
    r
}

/// Threshold-preset bundle for [`assert_baseline`]. Captures the
/// guarantees a scheduler-under-test should meet on a healthy run:
/// wake latency stays within bound, per-iteration compute cost stays
/// within bound, CPU migrations stay within bound, and every worker
/// makes some forward progress.
///
/// Each `Option` field is independent — `None` skips that check. A
/// `SchedulerBaseline` with every field `None` is a no-op (the
/// returned [`AssertResult`] always passes), useful as a starting
/// point for builder-style composition. Use [`Self::strict`] for the
/// "every check enabled with sane defaults" preset.
///
/// Distinct from [`Assert`]: `Assert` is the merge-tree threshold
/// config consumed by the worker-side `AssertPlan`; `SchedulerBaseline`
/// is a flat preset designed for direct invocation in test bodies
/// where the test author wants a one-call multi-field check without
/// engaging the merge chain. The two surfaces compose — a test can
/// run `assert_baseline` against a worker-report slice AND merge the
/// `Assert`-derived result into the same accumulator via
/// [`AssertResult::merge`].
#[must_use = "SchedulerBaseline only takes effect when passed to assert_baseline"]
#[derive(Debug, Clone, Copy, Default)]
pub struct SchedulerBaseline {
    /// Maximum acceptable p99 wake latency (nanoseconds). Compared
    /// against the pooled p99 across every worker's
    /// [`WorkerReport::resume_latencies_ns`]. `None` skips the check.
    /// Same units / semantics as [`Assert::max_p99_wake_latency_ns`].
    pub max_p99_wake_latency_ns: Option<u64>,
    /// Maximum acceptable p99 per-iteration compute cost (nanoseconds).
    /// Compared against the pooled p99 across every worker's
    /// [`WorkerReport::iteration_costs_ns`]. `None` skips the check.
    /// Only meaningful for compute work types that populate the
    /// reservoir (`AluHot`, `SmtSiblingSpin`, `IpcVariance`); blocking
    /// variants report empty `iteration_costs_ns` and the check is a
    /// no-op for those.
    pub max_iteration_cost_p99_ns: Option<u64>,
    /// Maximum acceptable total CPU migrations across every worker.
    /// Compared against the sum of [`WorkerReport::migration_count`].
    /// `None` skips the check. Distinct from
    /// [`Assert::max_migration_ratio`] (migrations per iteration) —
    /// this is an absolute count, useful when the test pins a known
    /// workload size and migrations should stay below a fixed ceiling
    /// regardless of how many iterations completed.
    pub max_migrations: Option<u64>,
    /// Minimum acceptable per-worker work_units. Every worker must
    /// have completed at least this many work units; one starved
    /// worker fails the check. `None` skips. Distinct from
    /// [`assert_not_starved`]'s zero-work-units check, which gates
    /// only against literal zero — this gate accepts a non-zero
    /// floor so a test can reject "barely made progress" runs that
    /// pass the strict starvation gate.
    pub min_work_units: Option<u64>,
}

impl SchedulerBaseline {
    /// Identity baseline — every field `None`, so [`assert_baseline`]
    /// returns a passing result with no checks performed. Useful as a
    /// starting point for builder-style composition.
    pub const EMPTY: SchedulerBaseline = SchedulerBaseline {
        max_p99_wake_latency_ns: None,
        max_iteration_cost_p99_ns: None,
        max_migrations: None,
        min_work_units: None,
    };

    /// Sane-default preset: p99 wake latency under 10ms, p99
    /// iteration cost under 1ms, total migrations under 1000, every
    /// worker completes ≥1 work unit. The defaults are deliberately
    /// loose — a baseline tight enough to catch egregious regressions
    /// without flagging every routine scheduler perturbation. Tests
    /// that need tighter bounds should set the fields explicitly via
    /// the `with_*` builder methods rather than tuning these constants.
    pub const fn strict() -> Self {
        Self {
            max_p99_wake_latency_ns: Some(10_000_000),
            max_iteration_cost_p99_ns: Some(1_000_000),
            max_migrations: Some(1000),
            min_work_units: Some(1),
        }
    }

    /// Builder setter for [`Self::max_p99_wake_latency_ns`].
    pub const fn with_max_p99_wake_latency_ns(mut self, v: u64) -> Self {
        self.max_p99_wake_latency_ns = Some(v);
        self
    }

    /// Builder setter for [`Self::max_iteration_cost_p99_ns`].
    pub const fn with_max_iteration_cost_p99_ns(mut self, v: u64) -> Self {
        self.max_iteration_cost_p99_ns = Some(v);
        self
    }

    /// Builder setter for [`Self::max_migrations`].
    pub const fn with_max_migrations(mut self, v: u64) -> Self {
        self.max_migrations = Some(v);
        self
    }

    /// Builder setter for [`Self::min_work_units`].
    pub const fn with_min_work_units(mut self, v: u64) -> Self {
        self.min_work_units = Some(v);
        self
    }
}

/// Run every check in `baseline` against `reports`, merging results
/// into a single [`AssertResult`]. A `None` field on the baseline
/// skips that check.
///
/// An empty `reports` slice short-circuits to a skip (`"no worker
/// reports to evaluate"`) regardless of baseline content — silently
/// passing a baseline against zero samples would let thresholds look
/// "green" on a run that produced no measurement.
///
/// Field-to-check mapping:
/// - `max_p99_wake_latency_ns` -> pooled p99 across every worker's
///   `resume_latencies_ns`; tagged [`DetailKind::Benchmark`].
/// - `max_iteration_cost_p99_ns` -> pooled p99 across every worker's
///   `iteration_costs_ns`; tagged [`DetailKind::Benchmark`].
/// - `max_migrations` -> sum of `migration_count` across workers;
///   tagged [`DetailKind::Migration`].
/// - `min_work_units` -> per-worker `work_units >= floor`; tagged
///   [`DetailKind::Starved`] when a worker is below the floor.
///
/// The wake-latency check delegates to [`assert_benchmarks`] for the
/// percentile path so the same nearest-rank algorithm applies; the
/// iteration-cost check uses an inline percentile call against the
/// pooled `iteration_costs_ns` reservoir.
///
/// ```
/// # use ktstr::assert::{SchedulerBaseline, assert_baseline};
/// # use ktstr::workload::WorkerReport;
/// # let report = WorkerReport {
/// #     tid: 1, cpus_used: [0].into_iter().collect(),
/// #     work_units: 1000, cpu_time_ns: 2_500_000_000,
/// #     wall_time_ns: 5_000_000_000, off_cpu_ns: 2_500_000_000,
/// #     migration_count: 5, migrations: vec![],
/// #     max_gap_ms: 50, max_gap_cpu: 0, max_gap_at_ms: 1000,
/// #     resume_latencies_ns: vec![100, 200, 300, 400, 500],
/// #     wake_sample_total: 5,
/// #     iteration_costs_ns: vec![1000, 2000, 3000, 4000, 5000],
/// #     iteration_cost_sample_total: 5,
/// #     iterations: 1000,
/// #     schedstat_run_delay_ns: 0, schedstat_run_count: 0,
/// #     schedstat_cpu_time_ns: 0,
/// #     completed: true,
/// #     numa_pages: std::collections::BTreeMap::new(),
/// #     vmstat_numa_pages_migrated: 0,
/// #     exit_info: None,
/// #     is_messenger: false,
/// #     group_idx: 0,
/// # };
/// // Strict preset on a healthy run — passes.
/// let r = assert_baseline(&[report], &SchedulerBaseline::strict());
/// assert!(r.passed);
/// ```
pub fn assert_baseline(reports: &[WorkerReport], baseline: &SchedulerBaseline) -> AssertResult {
    // Empty `reports` means nothing was measured. Returning a fresh
    // `pass()` here would silently green-light a broken run that
    // produced no signal; delegating to `assert_benchmarks` and
    // merging its skip would lose the skip flag (`AssertResult::merge`
    // ANDs `skipped`, so `pass.merge(skip) == passed-not-skipped`).
    // Surface the skip directly so the operator sees the baseline
    // wasn't actually exercised.
    if reports.is_empty() {
        return AssertResult::skip("no worker reports to evaluate");
    }

    let mut r = AssertResult::pass();

    // Wake-latency p99: reuse the existing `assert_benchmarks` path
    // so the percentile algorithm stays unified. With `reports`
    // non-empty here, `assert_benchmarks` cannot return a skip —
    // the merge sees only pass/fail, preserving baseline semantics.
    if baseline.max_p99_wake_latency_ns.is_some() {
        r.merge(assert_benchmarks(
            reports,
            baseline.max_p99_wake_latency_ns,
            None,
            None,
        ));
    }

    // Iteration-cost p99: pooled across every worker's reservoir.
    // Skipped when no samples are present — compute work types that
    // populate `iteration_costs_ns` are sparse, so an empty pooled
    // set is the common case for blocking variants and not a failure.
    if let Some(cost_limit) = baseline.max_iteration_cost_p99_ns {
        let all_costs: Vec<u64> = reports
            .iter()
            .flat_map(|w| w.iteration_costs_ns.iter().copied())
            .collect();
        if !all_costs.is_empty() {
            let mut sorted = all_costs.clone();
            sorted.sort_unstable();
            let p99 = percentile(&sorted, 0.99);
            if p99 > cost_limit {
                r.passed = false;
                r.details.push(AssertDetail::new(
                    DetailKind::Benchmark,
                    format!(
                        "p99 iteration cost {p99}ns exceeds limit {cost_limit}ns ({} samples)",
                        sorted.len(),
                    ),
                ));
            }
        }
    }

    // Total migrations across all workers: absolute-count gate
    // (distinct from migration_ratio which is a per-iteration rate).
    if let Some(max_mig) = baseline.max_migrations {
        let total_mig: u64 = reports.iter().map(|w| w.migration_count).sum();
        if total_mig > max_mig {
            r.passed = false;
            r.details.push(AssertDetail::new(
                DetailKind::Migration,
                format!(
                    "total migrations {total_mig} exceeds limit {max_mig} ({} workers)",
                    reports.len(),
                ),
            ));
        }
    }

    // Per-worker work_units floor: every worker must have completed
    // at least `min` work units. One starved worker fails the check.
    if let Some(min_units) = baseline.min_work_units {
        for w in reports {
            if w.work_units < min_units {
                r.passed = false;
                r.details.push(AssertDetail::new(
                    DetailKind::Starved,
                    format!(
                        "tid {} work_units {} below floor {min_units}",
                        w.tid, w.work_units,
                    ),
                ));
            }
        }
    }

    r
}

// (The legacy `Expect` / `Checks` / `CheckBuilder` types previously
// living here were replaced by the [`Verdict`]-based claim API
// (defined further up in this file). The new flow is
// `Assert::defaults().verdict().claim_<field>(stats).at_most(N)` for
// stats-struct-derived accessors, or `claim!(verdict, expr)` for
// expression-labeled claims. Both produce
// [`ClaimBuilder`]/[`SetClaim`]/[`SeqClaim`] under the hood and
// record outcomes onto the same [`AssertResult`] envelope that
// `assert_not_starved` / `assert_isolation` produce, so the two
// paths compose via [`Verdict::merge`].)

#[cfg(test)]
mod tests_assert;
#[cfg(test)]
mod tests_benchmarks;
#[cfg(test)]
mod tests_common;
#[cfg(test)]
mod tests_merge;
#[cfg(test)]
mod tests_note;
#[cfg(test)]
mod tests_numa;
#[cfg(test)]
mod tests_percentile;
#[cfg(test)]
mod tests_plan;
#[cfg(test)]
mod tests_sched_died;
#[cfg(test)]
mod tests_serde;
#[cfg(test)]
mod tests_stats;
#[cfg(test)]
mod tests_verdict;
#[cfg(test)]
mod tests_worker;
