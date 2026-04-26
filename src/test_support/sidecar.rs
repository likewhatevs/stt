//! Per-run sidecar JSON — the durable record of a ktstr test outcome.
//!
//! Every test (pass, fail, or skip) writes a [`SidecarResult`] to a
//! JSON file under the run's sidecar directory; downstream analysis
//! (`cargo ktstr stats`, CI dashboards) aggregates those files to
//! compute pass/fail rates, verifier stats, callback profiles, and
//! KVM stats across gauntlet variants.
//!
//! Responsibilities owned by this module:
//! - [`SidecarResult`]: the on-disk schema. Writer-side: every field
//!   is always emitted — `null` for `None`, `[]` for empty `Vec` —
//!   with no `skip_serializing_if` and no `serde(default)`. Reader-
//!   side: serde's native `Option<T>` deserialize tolerates absence
//!   (a missing key parses as `None`); non-`Option` fields (e.g.
//!   `test_name`, `passed`, `stats`) are hard-required and a missing
//!   key fails deserialize. The contract is intentionally asymmetric
//!   so a future producer that drops an `Option` field still parses
//!   on older readers, while the current writer guarantees full
//!   round-trip symmetry. Pre-1.0: old sidecar JSON is disposable;
//!   regenerate by re-running the test rather than relying on the
//!   reader-side tolerance for migration.
//! - [`collect_sidecars`]: load every `*.ktstr.json` under a directory
//!   (one level of subdirectories for per-job gauntlet layouts).
//! - [`write_sidecar`] / [`write_skip_sidecar`]: serialize one run to
//!   disk; variant-hash the discriminating fields so gauntlet variants
//!   don't clobber each other.
//! - [`sidecar_dir`], [`runs_root`], [`newest_run_dir`]: resolve where
//!   sidecars live (env override, or
//!   `{target}/ktstr/{kernel}-{project_commit}` where
//!   `{project_commit}` is the project tree's HEAD short hex from
//!   [`detect_project_commit`], suffixed `-dirty` when the
//!   worktree differs).
//! - [`format_verifier_stats`], [`format_callback_profile`],
//!   [`format_kvm_stats`]: human-readable summaries from a
//!   `Vec<SidecarResult>` for CLI output.
//! - [`detect_kernel_version`]: read the kernel version from
//!   `KTSTR_KERNEL` cache metadata for sidecar-dir naming and the
//!   `kernel_version` field, with fallback to
//!   `include/config/kernel.release` in the kernel source tree
//!   when the cache metadata is absent or does not carry a
//!   version (e.g. a raw source-tree path set in `KTSTR_KERNEL`
//!   rather than a cache key).
//! - [`detect_kernel_commit`]: read the kernel SOURCE TREE's git
//!   HEAD short hex (with `-dirty` suffix when worktree differs
//!   from the index or HEAD differs from the index) for the
//!   `kernel_commit` field. Distinct from `kernel_version`
//!   (release string from `kernel.release`) and `project_commit`
//!   (ktstr framework HEAD): this records "what kernel commit
//!   produced this run" so two runs of the same `kernel_version`
//!   but different WIP source trees compare distinctly.

use std::path::PathBuf;

use anyhow::Context;

use crate::assert::{AssertResult, ScenarioStats};
use crate::monitor::MonitorSummary;
use crate::test_support::PayloadMetrics;
use crate::timeline::StimulusEvent;
use crate::vmm;

use super::entry::KtstrTestEntry;
use super::timefmt::{generate_run_id, now_iso8601};

/// Test result sidecar written to KTSTR_SIDECAR_DIR for post-run analysis.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SidecarResult {
    /// Fully qualified test name (matches `KtstrTestEntry::name`,
    /// the bare function name without the `ktstr/` nextest prefix).
    pub test_name: String,
    /// Rendered topology label (e.g. `1n2l4c1t`) for the variant this
    /// sidecar describes.
    pub topology: String,
    /// Scheduler name (matches `Scheduler::name`); `"eevdf"` for
    /// tests run without an scx scheduler.
    pub scheduler: String,
    /// Best-effort git commit of the scheduler binary used for this
    /// run. Currently ALWAYS `None` for every `SchedulerSpec`
    /// variant — no variant today has a reliable commit source.
    /// The field is reserved on the schema so stats tooling can
    /// enrich it once a reliable source exists (e.g. a
    /// `--version` probe or ELF-note read on the resolved
    /// scheduler binary). See
    /// [`crate::test_support::SchedulerSpec::scheduler_commit`]
    /// for the full per-variant rationale.
    ///
    /// Writer always emits (`"scheduler_commit": null` on absence).
    /// Reader-side: serde's native `Option<T>` deserialize tolerates
    /// absence (a missing key parses as `None`); see the module-level
    /// doc for the full asymmetric contract that governs every
    /// nullable on this struct.
    pub scheduler_commit: Option<String>,
    /// Best-effort git HEAD of the ktstr project tree at sidecar-
    /// write time. Captured by [`detect_project_commit`] via
    /// `gix::discover` from the test process's current working
    /// directory; walks up to find the enclosing repo and reads
    /// HEAD short-hex, suffixing `-dirty` when index-vs-HEAD or
    /// worktree-vs-index changes are observed (submodules ignored,
    /// matching the [`crate::fetch::local_source`] dirty-detection
    /// pattern). `None` when cwd is not inside any git repo, or
    /// when the gix probe fails for any reason — this is metadata,
    /// not a gate, so probe failure must not abort the run.
    ///
    /// Distinct from [`SidecarResult::scheduler_commit`]: that
    /// field tracks the userspace scheduler binary's commit
    /// (currently always `None` per its own doc); this field
    /// tracks the ktstr framework / test-runner commit, so the
    /// stats CLI can answer "which version of the harness produced
    /// this sidecar?" without inspecting the scheduler.
    ///
    /// Writer always emits (`"project_commit": null` on absence).
    /// Reader-side: serde's native `Option<T>` deserialize tolerates
    /// absence (a missing key parses as `None`) — see the module-
    /// level doc for the full asymmetric contract. Excluded from
    /// [`sidecar_variant_hash`] for the same cross-host grouping
    /// reason `scheduler_commit` is excluded: two runs of the same
    /// semantic variant on different ktstr commits must still bucket
    /// together so `stats compare` can diff them; the commit-drift
    /// detection inspects this field directly via `--project-commit`
    /// / `--a-project-commit` / `--b-project-commit`.
    pub project_commit: Option<String>,
    /// Binary payload name (matches `Payload::name` when
    /// `entry.payload` is set). `None` when the test declared no
    /// binary payload. Writer always emits (`"payload": null` on
    /// absence); reader-side, serde's native `Option<T>` deserialize
    /// tolerates absence — see the module-level doc for the full
    /// asymmetric contract.
    pub payload: Option<String>,
    /// Per-payload extracted metrics collected from `ctx.payload(X).run()`
    /// / `.spawn().wait()` call sites during the test body.
    ///
    /// One [`PayloadMetrics`] per invocation, in the order the calls
    /// ran. Empty when no payload calls were made (scheduler-only
    /// tests, or a binary-only test where the body bailed before
    /// running the payload). Writer always emits (`"metrics": []` in
    /// that case); reader-side, this `Vec` field is hard-required —
    /// non-`Option` fields fail deserialize on absence. See the
    /// module-level doc for the full contract.
    pub metrics: Vec<PayloadMetrics>,
    /// Overall pass/fail verdict for this run.
    pub passed: bool,
    /// True when the test was skipped (e.g. topology mismatch,
    /// missing resource). A skipped test has `passed == true`
    /// (to keep the verdict gate simple) but downstream stats
    /// tooling must subtract `skipped` runs from "pass count" to
    /// avoid reporting non-executions as passes.
    pub skipped: bool,
    /// Aggregate per-cgroup statistics merged across every worker.
    pub stats: ScenarioStats,
    /// Monitor summary. `None` means the monitor loop did not run
    /// (host-only tests, early VM failure) or sample collection
    /// produced no valid data. Writer always emits (`"monitor": null`
    /// on absence); reader-side, serde's native `Option<T>`
    /// deserialize tolerates absence — see the module-level doc.
    pub monitor: Option<MonitorSummary>,
    /// Ordered stimulus events published by the guest step executor
    /// while the scenario ran.
    pub stimulus_events: Vec<StimulusEvent>,
    /// Work type label used for post-hoc filtering and A/B comparison
    /// (distinct from the `WorkType` enum — this is the text name).
    pub work_type: String,
    /// Scheduler flag names active for this gauntlet variant. Empty
    /// for the default (no-flags) profile. Participates in the
    /// sidecar variant-hash so flag-only variants don't clobber.
    pub active_flags: Vec<String>,
    /// Per-BPF-program verifier statistics captured from the VM's
    /// scheduler (when one was loaded). Empty when no scheduler
    /// programs were inspected. Writer always emits as
    /// `"verifier_stats": []` in that case; reader-side, this `Vec`
    /// field is hard-required (non-`Option` fields fail deserialize
    /// on absence). See the module-level doc.
    pub verifier_stats: Vec<crate::monitor::bpf_prog::ProgVerifierStats>,
    /// Aggregate per-vCPU KVM stats read after VM exit. `None` when
    /// the VM did not run (host-only tests) or KVM stats were
    /// unavailable. Writer always emits (`"kvm_stats": null` on
    /// absence); reader-side, serde's native `Option<T>` deserialize
    /// tolerates absence — see the module-level doc.
    pub kvm_stats: Option<crate::vmm::KvmStatsTotals>,
    /// Effective sysctls active during this test run, recorded as raw
    /// `sysctl.key=value` cmdline strings. Writer always emits as
    /// `"sysctls": []` when none; reader-side, this `Vec` field is
    /// hard-required (non-`Option` fields fail deserialize on
    /// absence). See the module-level doc.
    pub sysctls: Vec<String>,
    /// Effective kernel command-line args active during this test run.
    /// Writer always emits as `"kargs": []` when none; reader-side,
    /// this `Vec` field is hard-required (non-`Option` fields fail
    /// deserialize on absence). See the module-level doc.
    pub kargs: Vec<String>,
    /// Kernel version of the VM under test (from cache metadata,
    /// e.g. `"6.14.2"`). Populated from the cache entry's
    /// `metadata.json` version field, with fallback to the kernel
    /// source tree's `include/config/kernel.release` when
    /// `KTSTR_KERNEL` points at a raw source path rather than a
    /// cache key; `None` for host-only tests or when neither
    /// source yields a version string. The host's running kernel
    /// release is carried separately in `host.kernel_release`.
    /// Writer always emits (`"kernel_version": null` on absence);
    /// reader-side, serde's native `Option<T>` deserialize tolerates
    /// absence — see the module-level doc for the full asymmetric
    /// contract.
    pub kernel_version: Option<String>,
    /// Kernel SOURCE TREE git HEAD short hex (7 chars via
    /// `oid::to_hex_with_len(7)`), with `-dirty` suffix appended
    /// when HEAD-vs-index or index-vs-worktree changes are
    /// observed. Probes via `gix::open` against the kernel
    /// directory resolved from `KTSTR_KERNEL` (not `gix::discover`
    /// — the kernel dir is explicit, not walked-up). Captured by
    /// [`detect_kernel_commit`] at sidecar-write time.
    ///
    /// Distinct from sibling fields:
    /// - [`SidecarResult::kernel_version`] — release string read
    ///   from cache metadata or `include/config/kernel.release`,
    ///   e.g. `"6.14.2"`. Two runs of `6.14.2` from a clean
    ///   tree and a `-dirty` worktree at the same HEAD share
    ///   `kernel_version` but differ on `kernel_commit`.
    /// - [`SidecarResult::project_commit`] — ktstr framework
    ///   HEAD captured from the test process's cwd. Tracks
    ///   "what version of the harness produced this sidecar?"
    ///   independently of the kernel under test.
    /// - [`SidecarResult::scheduler_commit`] — userspace
    ///   scheduler binary's commit (currently always `None`).
    ///
    /// `None` when:
    /// - `KTSTR_KERNEL` is unset or empty;
    /// - the resolved `KernelId` is `Version` / `CacheKey` whose
    ///   underlying source is `Tarball` / `Git` (no source tree
    ///   on disk to probe);
    /// - the resolved kernel directory is not a git repository
    ///   (`gix::open` fails);
    /// - HEAD cannot be read (unborn HEAD on a fresh `git init`
    ///   with zero commits);
    /// - any other gix probe failure — metadata, not a gate.
    ///
    /// Writer always emits (`"kernel_commit": null` on absence);
    /// reader-side, serde's native `Option<T>` deserialize tolerates
    /// absence — see the module-level doc for the full asymmetric
    /// contract. Excluded from [`sidecar_variant_hash`] for the same
    /// cross-host grouping reason `scheduler_commit` and
    /// `project_commit` are excluded: two runs of the same semantic
    /// variant on different kernel-source HEADs must still bucket
    /// together so `stats compare` can diff them; the commit-drift
    /// detection inspects this field directly via the
    /// `--kernel-commit` filter.
    pub kernel_commit: Option<String>,
    /// ISO 8601 timestamp of when this test run started.
    pub timestamp: String,
    /// Unique identifier for the test run. Composed as
    /// `{run_id_timestamp}-{counter}` — the `YYYYMMDDTHHMMSSZ`
    /// process-start stamp followed by a process-local monotonic
    /// counter. Every sidecar produced in one `cargo ktstr test`
    /// invocation shares the same timestamp prefix; the counter
    /// distinguishes concurrent gauntlet variants within that
    /// invocation. Distinct from the run DIRECTORY name (keyed
    /// `{kernel}-{project_commit}`, see [`sidecar_dir`]) — the
    /// directory groups runs by what they tested, the `run_id`
    /// groups sidecars by which process emitted them.
    pub run_id: String,
    /// Host context — static-ish runtime state (CPU model,
    /// memory size, THP policy, kernel release, host cmdline,
    /// scheduler tunables). Populated by production sidecar
    /// writers.
    ///
    /// `None` causes:
    /// - **test-fixture path**: not the production sidecar
    ///   writer (production writers always populate `host`).
    /// - **pre-enrichment archive**: sidecar predates the
    ///   host-context landing — re-run the test to regenerate
    ///   under the current schema (no migration shim exists
    ///   per the pre-1.0 disposable-data contract).
    ///
    /// Deliberately excluded from the variant hash so
    /// gauntlet variants on different hosts collapse into the same
    /// hash bucket.
    ///
    /// No serde attributes: writer always emits (`"host": null` when
    /// `None`); reader-side, serde's native `Option<T>` deserialize
    /// tolerates absence (a missing key parses as `None`). The
    /// asymmetric contract is crate-wide — see the module-level doc.
    /// Pre-1.0, sidecar data is disposable, so regenerate by
    /// re-running the test rather than carrying a compat shim for
    /// older JSON; the reader-side tolerance exists so an in-flight
    /// schema rename of an `Option` field does not break parsing of
    /// older sidecars during the same producer-version, not as a
    /// long-term migration story.
    pub host: Option<crate::host_context::HostContext>,
    /// Wall-clock milliseconds spent in
    /// [`KtstrVm::collect_results`](crate::vmm::KtstrVm) — the host-side
    /// teardown window from BSP exit through SHM drain (mirrors
    /// [`VmResult::cleanup_duration`](crate::vmm::VmResult::cleanup_duration);
    /// `Duration` is converted to `u64` ms here because every other
    /// timing field on this struct that lands in a sidecar-comparison
    /// CLI uses integer ms or seconds, and JSON has no native
    /// `Duration`). `None` when the run was killed by the watchdog
    /// before `collect_results` returned, or for the `host_only` /
    /// host-only-stub paths that never boot a VM. Writer always emits
    /// (`"cleanup_duration_ms": null` on absence); reader-side,
    /// serde's native `Option<T>` deserialize tolerates absence — see
    /// the module-level doc for the full asymmetric contract.
    pub cleanup_duration_ms: Option<u64>,
    /// Provenance tag for this sidecar — distinguishes a developer's
    /// local run from a CI run so cross-environment comparisons in
    /// `stats compare` can narrow on (or contrast across) the run
    /// environment without inferring it from `host`.
    ///
    /// Recorded by [`detect_run_source`] at sidecar-write time:
    /// - `Some("ci")` when [`KTSTR_CI_ENV`] is set non-empty (CI runner
    ///   scripts export it before invoking the test binary; local
    ///   runs never set it).
    /// - `Some("local")` otherwise — the default for any sidecar
    ///   produced by a developer-driven invocation.
    /// - The third documented value (`"archive"`) is NEVER written
    ///   here: a sidecar cannot know it will later be archived. The
    ///   stats CLI applies the `"archive"` tag at LOAD time when its
    ///   `--dir` flag points at a non-default pool root, overriding
    ///   whatever was on disk via [`apply_archive_source_override`].
    ///
    /// `Option<String>` (rather than an enum) keeps the schema
    /// extensible without a serde-version bump if a future producer
    /// wants a new tag (e.g. `"benchmark"`); the consumer side
    /// treats unknown values the same as known ones — they are
    /// strings the operator can pass via `--run-source` to filter on.
    /// Writer always emits (`"run_source": null` on absence);
    /// reader-side, serde's native `Option<T>` deserialize tolerates
    /// absence — see the module-level doc for the full asymmetric
    /// contract. Excluded from [`sidecar_variant_hash`] for the same
    /// cross-host grouping reason `host` is excluded — two runs of
    /// the same semantic variant from different environments must
    /// still bucket together so `stats compare` can diff them;
    /// `--run-source` and `--a-run-source` / `--b-run-source` are the
    /// explicit knobs for source-aware narrowing.
    ///
    /// Field name `run_source` (renamed from `source`) disambiguates
    /// from [`crate::cache::KernelSource`] / `KernelMetadata.source`
    /// — those describe the kernel build's input (tarball / git /
    /// local), this describes the run-environment provenance.
    ///
    /// **On-disk JSON key changed from `"source"` to `"run_source"`
    /// in the field rename.** No `#[serde(alias = "source")]` is
    /// in place: archived sidecars written before the rename carry
    /// the `"source"` key, which the current schema treats as an
    /// unknown field. Because `SidecarResult`'s derive does NOT
    /// set `deny_unknown_fields`, the deserialize does not fail
    /// outright — instead serde silently DROPS the stale `"source"`
    /// payload and lands `run_source = None` (since `Option<T>`'s
    /// "tolerate absence" rule kicks in for the missing
    /// `"run_source"` field). The data is lost, not preserved. This
    /// is deliberate per the project's pre-1.0 disposable-data
    /// contract: re-running tests regenerates sidecars under the
    /// new key rather than carrying compat shims forward. Consumers
    /// who need the run-source classification on archived JSON
    /// must either rename the key in-place before deserialize, or
    /// re-run the test to regenerate the sidecar with the new
    /// schema. Tooling that runs against the renamed schema and
    /// observes a `None` `run_source` cannot distinguish "sidecar
    /// pre-dates the field" from "sidecar pre-dates the rename and
    /// lost its tag" — both lower-bound at `None` for filter
    /// purposes.
    pub run_source: Option<String>,
}

#[cfg(test)]
impl SidecarResult {
    /// Populated [`SidecarResult`] for unit tests. Every field has a
    /// reasonable default so call sites only spell out what they want
    /// to vary via struct-update syntax:
    ///
    /// ```ignore
    /// let sc = SidecarResult {
    ///     test_name: "my_test".to_string(),
    ///     passed: false,
    ///     ..SidecarResult::test_fixture()
    /// };
    /// ```
    ///
    /// Defaults model a passing EEVDF run on a minimal `1n1l1c1t`
    /// topology with no payload and no VM telemetry: `test_name="t"`,
    /// `topology="1n1l1c1t"`, `scheduler="eevdf"`, `work_type="CpuSpin"`,
    /// `passed=true`, `skipped=false`, every [`Option`] `None`, every
    /// [`Vec`] empty, `stats` is `ScenarioStats::default()`, and both
    /// `timestamp`/`run_id` are empty strings.
    ///
    /// **Prefer this over local `base = || SidecarResult { ... }`
    /// closures.** A local closure duplicates the default set and
    /// drifts the moment [`SidecarResult`] grows a field; this fixture
    /// is the single place those defaults live.
    ///
    /// **Hash-stability tests must not rely on these defaults for
    /// hash-participating fields** (`topology`, `scheduler`, `payload`,
    /// `work_type`, `active_flags`, `sysctls`, `kargs`). Tests that pin
    /// a [`sidecar_variant_hash`] output against a literal constant
    /// must spell every hash-participating field out explicitly so a
    /// future change to these defaults cannot silently shift the
    /// pinned value.
    pub(crate) fn test_fixture() -> SidecarResult {
        SidecarResult {
            test_name: "t".to_string(),
            topology: "1n1l1c1t".to_string(),
            scheduler: "eevdf".to_string(),
            scheduler_commit: None,
            project_commit: None,
            payload: None,
            metrics: Vec::new(),
            passed: true,
            skipped: false,
            stats: crate::assert::ScenarioStats::default(),
            monitor: None,
            stimulus_events: Vec::new(),
            work_type: "CpuSpin".to_string(),
            active_flags: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            sysctls: Vec::new(),
            kargs: Vec::new(),
            kernel_version: None,
            kernel_commit: None,
            timestamp: String::new(),
            run_id: String::new(),
            host: None,
            cleanup_duration_ms: None,
            run_source: None,
        }
    }
}

/// Predicate: is `path` a ktstr sidecar JSON filename?
///
/// True iff the path's extension is `json` AND the path's
/// FILENAME COMPONENT (`Path::file_name`) contains `.ktstr.` —
/// matching the on-disk shape produced by [`write_sidecar`]
/// (`<test>-<variant_hash>.ktstr.json`). Both gates are required:
/// bare `*.json` files (cargo cache, stray fixtures) and non-json
/// files whose name happens to contain `.ktstr.` (e.g. a log)
/// are excluded.
///
/// The filename-component check (rather than full-path string)
/// is load-bearing: a parent directory like
/// `target/foo.ktstr.bar/extra.json` would falsely match a
/// whole-path `contains(".ktstr.")` while NOT being a sidecar.
/// `Path::file_name()` returns only the trailing component, so
/// `.ktstr.` in any ancestor segment cannot trigger the predicate.
///
/// Single source of truth for "is this file a sidecar?" — used
/// by [`collect_sidecars_with_errors`]'s parsing walker and by
/// [`crate::cli::count_sidecar_files`]'s file-count walker. Both
/// walkers MUST agree on the predicate so `walked` (count) and
/// `valid + errors` (parse outcomes) reconcile against each
/// other; a divergence would let a file count toward `walked`
/// without contributing to either bucket, manifesting as a
/// silent-drop count that has no source.
pub(crate) fn is_sidecar_filename(path: &std::path::Path) -> bool {
    path.extension().and_then(|e| e.to_str()) == Some("json")
        && path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.contains(".ktstr."))
}

/// Scan a directory for ktstr sidecar JSON files. Recurses one level
/// into subdirectories to handle per-job gauntlet layouts.
///
/// Convenience wrapper over [`collect_sidecars_with_errors`] for
/// callers that only need the parsed sidecars and not the
/// per-file parse-failure list. The eprintln-driven diagnostic
/// path is preserved unchanged inside the underlying walker.
pub(crate) fn collect_sidecars(dir: &std::path::Path) -> Vec<SidecarResult> {
    collect_sidecars_with_errors(dir).0
}

/// Per-file parse-failure record returned by
/// [`collect_sidecars_with_errors`] and threaded through
/// [`crate::cli::WalkStats::errors`] to the renderers.
///
/// Named-field struct (rather than a `(PathBuf, String,
/// Option<String>)` tuple) so call sites read fields by name —
/// pattern-matching `for err in errors` and accessing
/// `err.path` / `err.raw_error` / `err.enriched_message`
/// resists the tuple-position-swap class of bug where positional
/// fields could destructure in either order without compiler help.
pub(crate) struct SidecarParseError {
    /// On-disk path of the sidecar JSON that failed to parse.
    pub path: std::path::PathBuf,
    /// Verbatim serde-error string. Kept raw for
    /// grep-friendly parse-error tracking and surfaced through
    /// the JSON channel as the `error` key.
    pub raw_error: String,
    /// Operator-facing remediation prose computed by
    /// [`enriched_parse_error_message`]. `Some(...)` for known
    /// schema-drift cases (currently the `host` missing-field
    /// pattern), `None` otherwise. Surfaced through the JSON
    /// channel as `enriched_message`.
    pub enriched_message: Option<String>,
}

/// Per-file IO-failure record returned by
/// [`collect_sidecars_with_errors`] and threaded through
/// [`crate::cli::WalkStats::io_errors`] to the renderers.
///
/// Captures files where the filename predicate matched but
/// `std::fs::read_to_string` failed before parsing could begin —
/// permission denied, mid-rotate truncation, broken symlink,
/// etc. Distinct from [`SidecarParseError`] (which represents
/// "file read OK but JSON parse failed"); separating the two
/// lets dashboard consumers triage filesystem incidents apart
/// from schema drift.
///
/// Named-field struct mirroring [`SidecarParseError`]'s shape so
/// the renderer side can iterate by field name without tuple-
/// position fragility. No `enriched_message` field — there is no
/// remediation catalog for IO failures (causes vary per host:
/// fix permissions, fix the filesystem, retry the test).
pub(crate) struct SidecarIoError {
    /// On-disk path the predicate matched as a sidecar candidate.
    pub path: std::path::PathBuf,
    /// Verbatim `std::io::Error` Display string. Surfaced through
    /// the JSON channel as the `error` key on
    /// [`crate::cli::WalkIoError`] entries and through the text
    /// channel as the `error: ...` line under the `io errors`
    /// trailing block.
    pub raw_error: String,
}

/// Test-only re-export of [`enriched_parse_error_message`] so
/// `cli::tests` can verify the enrichment-pattern logic
/// directly against synthetic error strings. The helper itself
/// stays private so production code routes through
/// [`collect_sidecars_with_errors`].
#[cfg(test)]
pub(crate) fn enriched_parse_error_message_for_test(
    path: &std::path::Path,
    raw_error: &str,
) -> Option<String> {
    enriched_parse_error_message(path, raw_error)
}

/// Compute the operator-prose enrichment for a serde parse-error
/// message, when one applies. Today the only enriched case is the
/// `host` missing-field schema-drift diagnostic; the function
/// returns `None` for any other shape so consumers can branch on
/// "enrichment exists" without re-implementing the match.
///
/// Pulled out of [`collect_sidecars_with_errors`]'s hot path so
/// the eprintln-side prose and the structured-channel
/// `enriched` carry identical text.
///
/// Matching on the Display text is deliberate: serde's typed-error
/// surface for `missing field "X"` is not stable across
/// serde_json versions, but the rendered message is — a
/// forward-compat regression-resilient check costs one string
/// search.
fn enriched_parse_error_message(path: &std::path::Path, raw_error: &str) -> Option<String> {
    let is_missing_host = raw_error.contains("missing field") && raw_error.contains("`host`");
    if is_missing_host {
        Some(format!(
            "ktstr_test: skipping {}: {raw_error} — the `host` field \
             was added to SidecarResult; pre-1.0 policy is \
             disposable-sidecar: re-run the test to regenerate this \
             file under the current schema (no migration shim exists)",
            path.display(),
        ))
    } else {
        None
    }
}

/// Scan a directory for ktstr sidecar JSON files, returning the
/// parsed sidecars, a [`SidecarParseError`] record (named fields
/// `path`, `raw_error`, `enriched_message`) for every file that
/// passed the filename predicate but failed to deserialize, and a
/// [`SidecarIoError`] record (named fields `path`, `raw_error`)
/// for every file that passed the predicate but whose
/// `read_to_string` failed before parsing could begin. Recurses
/// one level into subdirectories to handle per-job gauntlet
/// layouts.
///
/// Surfaces parse failures in two channels:
/// - `eprintln!` to stderr (preserved for the operator-facing
///   pre-1.0 disposable-sidecar diagnostic — emits the enriched
///   prose for the host-missing schema-drift case, the raw serde
///   message otherwise).
/// - The returned parse-errors vec, capturing a
///   [`SidecarParseError`] record (named fields `path`,
///   `raw_error`, `enriched_message`) for structured callers
///   (`explain-sidecar`'s walker output). Both raw and enriched
///   are exposed so dashboard consumers can pick: raw for
///   parse-error grepping, enriched for human-facing remediation
///   prose.
///
/// IO failures (third return) get a single eprintln line plus a
/// structured [`SidecarIoError`] record. Distinguished from
/// parse failures so dashboard consumers can triage filesystem
/// incidents (permission denied, mid-rotate truncation, broken
/// symlink) apart from schema drift. With this third channel,
/// every predicate-matching file lands in exactly one of the
/// three returned vecs — the prior implicit
/// `walked - valid - parse_errors.len()` silent-drop count is
/// now zero by construction.
///
/// Callers that don't need structured errors should use
/// [`collect_sidecars`].
pub(crate) fn collect_sidecars_with_errors(
    dir: &std::path::Path,
) -> (
    Vec<SidecarResult>,
    Vec<SidecarParseError>,
    Vec<SidecarIoError>,
) {
    let mut sidecars = Vec::new();
    let mut parse_errors: Vec<SidecarParseError> = Vec::new();
    let mut io_errors: Vec<SidecarIoError> = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return (sidecars, parse_errors, io_errors),
    };
    let mut subdirs = Vec::new();
    let try_load = |path: &std::path::Path,
                    out: &mut Vec<SidecarResult>,
                    parse_errs: &mut Vec<SidecarParseError>,
                    io_errs: &mut Vec<SidecarIoError>| {
        if !is_sidecar_filename(path) {
            return;
        }
        let data = match std::fs::read_to_string(path) {
            Ok(d) => d,
            Err(e) => {
                let raw = e.to_string();
                eprintln!("ktstr_test: cannot read {}: {raw}", path.display());
                io_errs.push(SidecarIoError {
                    path: path.to_path_buf(),
                    raw_error: raw,
                });
                return;
            }
        };
        match serde_json::from_str::<SidecarResult>(&data) {
            Ok(sc) => out.push(sc),
            Err(e) => {
                let raw = e.to_string();
                let enriched = enriched_parse_error_message(path, &raw);
                // eprintln channel: emit the enriched prose when
                // it applies, the raw serde message otherwise.
                // Identical text flows through both channels —
                // both go through `enriched_parse_error_message`.
                match &enriched {
                    Some(prose) => eprintln!("{prose}"),
                    None => eprintln!("ktstr_test: skipping {}: {raw}", path.display()),
                }
                parse_errs.push(SidecarParseError {
                    path: path.to_path_buf(),
                    raw_error: raw,
                    enriched_message: enriched,
                });
            }
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            subdirs.push(path);
            continue;
        }
        try_load(&path, &mut sidecars, &mut parse_errors, &mut io_errors);
    }
    for sub in subdirs {
        if let Ok(entries) = std::fs::read_dir(&sub) {
            for entry in entries.flatten() {
                try_load(
                    &entry.path(),
                    &mut sidecars,
                    &mut parse_errors,
                    &mut io_errors,
                );
            }
        }
    }
    (sidecars, parse_errors, io_errors)
}

/// Pool every sidecar JSON under every run directory at `root`.
///
/// Walks each immediate subdirectory of `root` (one per run, named
/// `{kernel}-{project_commit}` by [`sidecar_dir`] where
/// `{project_commit}` is the project tree's HEAD short hex with
/// `-dirty` suffix when the worktree differs from HEAD) and
/// concatenates the sidecars each
/// one yields via [`collect_sidecars`]. The result is a flat
/// `Vec<SidecarResult>` covering every recorded run on disk —
/// `cargo ktstr stats compare`'s pool-driven sourcing reads it
/// once, applies the typed `--a-*` / `--b-*` filters in memory,
/// and partitions the survivors into A/B sides.
///
/// `root` is typically [`runs_root`]; pass an alternate path when
/// comparing archived sidecar trees copied off a CI host (the
/// `--dir` escape hatch on `stats compare`).
///
/// Returns an empty Vec when `root` does not exist or contains no
/// run directories. Per-run failure (a corrupt sidecar, a partial
/// directory) prints a per-file `eprintln!` from
/// [`collect_sidecars`] and continues — pool-collection never
/// aborts on a single bad file.
///
/// Performance: this is a full filesystem walk over `root`. On a
/// host with many archived runs (dozens to hundreds), each
/// invocation re-reads every sidecar JSON. The cost is acceptable
/// for the current operator workflow (one comparison per
/// session) but is taskifyable if it becomes a hot path — a
/// directory-name fast-path could skip runs whose
/// `{kernel}-{project_commit}` prefix does not match the active
/// `--a-kernel` / `--b-kernel` filter.
pub fn collect_pool(root: &std::path::Path) -> Vec<SidecarResult> {
    let entries = match std::fs::read_dir(root) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    let mut pool = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            // `collect_sidecars` already handles "one level of
            // subdirectories for per-job gauntlet layouts" inside
            // each run directory, so the two-level
            // `{root}/{run_dir}/{job_subdir}` shape works without
            // a third walker level.
            pool.extend(collect_sidecars(&path));
        }
    }
    pool
}

/// BPF verifier complexity limit (BPF_COMPLEXITY_LIMIT_INSNS).
const VERIFIER_INSN_LIMIT: u32 = 1_000_000;

/// Percentage of the verifier limit that triggers a warning.
const VERIFIER_WARN_PCT: f64 = 75.0;

/// Aggregate BPF verifier stats across sidecars into a summary table.
///
/// verified_insns is deterministic for a given binary, so per-program
/// values are deduplicated (max across observations). Flags programs
/// using >=75% of the 1M verifier complexity limit.
pub(crate) fn format_verifier_stats(sidecars: &[SidecarResult]) -> String {
    use std::collections::BTreeMap;

    let mut by_name: BTreeMap<&str, u32> = BTreeMap::new();
    for sc in sidecars {
        for info in &sc.verifier_stats {
            let entry = by_name.entry(&info.name).or_insert(0);
            *entry = (*entry).max(info.verified_insns);
        }
    }

    if by_name.is_empty() {
        return String::new();
    }

    let mut out = String::from("\n=== BPF VERIFIER STATS ===\n\n");
    out.push_str(&format!(
        "  {:<24} {:>12} {:>8}\n",
        "program", "verified", "limit%"
    ));
    out.push_str(&format!("  {:-<24} {:-<12} {:-<8}\n", "", "", ""));

    let mut warnings = Vec::new();
    let mut total: u64 = 0;

    for (&name, &verified_insns) in &by_name {
        let pct = (verified_insns as f64 / VERIFIER_INSN_LIMIT as f64) * 100.0;
        let flag = if pct >= VERIFIER_WARN_PCT { " !" } else { "" };
        out.push_str(&format!(
            "  {:<24} {:>12} {:>7.1}%{flag}\n",
            name, verified_insns, pct,
        ));
        if pct >= VERIFIER_WARN_PCT {
            warnings.push(format!(
                "  {name}: {pct:.1}% of 1M limit ({verified_insns} verified insns)",
            ));
        }
        total += verified_insns as u64;
    }

    out.push_str(&format!("\n  total verified insns: {total}\n"));

    if !warnings.is_empty() {
        out.push_str("\nWARNING: programs near verifier complexity limit:\n");
        for w in &warnings {
            out.push_str(w);
            out.push('\n');
        }
    }

    out
}

/// Per-test BPF callback profile from monitor prog_stats_deltas.
///
/// Shows per-program invocation count, total CPU time, and average
/// nanoseconds per call. Each test's profile is printed independently.
pub(crate) fn format_callback_profile(sidecars: &[SidecarResult]) -> String {
    let mut out = String::new();

    for sc in sidecars {
        let deltas = match sc
            .monitor
            .as_ref()
            .and_then(|m| m.prog_stats_deltas.as_ref())
        {
            Some(d) if !d.is_empty() => d,
            _ => continue,
        };

        if out.is_empty() {
            out.push_str("\n=== BPF CALLBACK PROFILE ===\n");
        }
        out.push_str(&format!("\n  {} ({}):\n", sc.test_name, sc.topology));
        out.push_str(&format!(
            "    {:<24} {:>12} {:>14} {:>12}\n",
            "program", "cnt", "total_ns", "avg_ns"
        ));
        out.push_str(&format!(
            "    {:-<24} {:-<12} {:-<14} {:-<12}\n",
            "", "", "", ""
        ));
        for d in deltas {
            out.push_str(&format!(
                "    {:<24} {:>12} {:>14} {:>12.0}\n",
                d.name, d.cnt, d.nsecs, d.nsecs_per_call,
            ));
        }
    }

    out
}

/// Aggregate KVM stats across sidecars into a compact summary.
///
/// Averages each stat across all tests that returned `Some(KvmStatsTotals)`.
/// Tests without KVM stats (non-VM tests, old kernels) are excluded
/// from the denominator.
pub(crate) fn format_kvm_stats(sidecars: &[SidecarResult]) -> String {
    let with_stats: Vec<&crate::vmm::KvmStatsTotals> = sidecars
        .iter()
        .filter_map(|sc| sc.kvm_stats.as_ref())
        .collect();

    if with_stats.is_empty() {
        return String::new();
    }

    let n_vms = with_stats.len();

    // Compute cross-VM averages for each stat.
    let vm_avg = |name: &str| -> u64 {
        let sum: u64 = with_stats.iter().map(|d| d.avg(name)).sum();
        sum / n_vms as u64
    };

    let exits = vm_avg("exits");
    let halt = vm_avg("halt_exits");
    let halt_wait_ns = vm_avg("halt_wait_ns");
    let preempted = vm_avg("preemption_reported");
    let signal = vm_avg("signal_exits");
    let hypercalls = vm_avg("hypercalls");

    // Halt poll efficiency across all vCPUs and VMs.
    let total_poll_ok: u64 = with_stats
        .iter()
        .map(|d| d.sum("halt_successful_poll"))
        .sum();
    let total_poll_try: u64 = with_stats
        .iter()
        .map(|d| d.sum("halt_attempted_poll"))
        .sum();

    if exits == 0 {
        return String::new();
    }

    let halt_wait_ms = halt_wait_ns as f64 / 1_000_000.0;
    let poll_pct = if total_poll_try > 0 {
        (total_poll_ok as f64 / total_poll_try as f64) * 100.0
    } else {
        0.0
    };

    let mut out = format!("\n=== KVM STATS (avg across {n_vms} VMs) ===\n\n");
    out.push_str(&format!(
        "  exits/vcpu  {:>7}   halt/vcpu     {:>5}   halt_wait_ms {:>7.1}\n",
        exits, halt, halt_wait_ms,
    ));
    out.push_str(&format!(
        "  poll_ok%    {:>6.1}%   preempted/vcpu {:>4}   signal/vcpu  {:>7}\n",
        poll_pct, preempted, signal,
    ));
    if hypercalls > 0 {
        out.push_str(&format!("  hypercalls/vcpu {:>4}\n", hypercalls));
    }

    // Trust warnings.
    if preempted > 0 {
        let total: u64 = with_stats
            .iter()
            .map(|d| d.sum("preemption_reported"))
            .sum();
        out.push_str(&format!(
            "\n  WARNING: {total} host preemptions detected \
             -- timing results may be unreliable\n",
        ));
    }

    out
}

/// Resolve the sidecar output directory for the current test process.
///
/// Override: `KTSTR_SIDECAR_DIR` (used as-is when non-empty). When
/// the override is set, `serialize_and_write_sidecar` ALSO skips
/// the per-directory pre-clear so any pre-existing sidecars in
/// the operator-chosen directory are preserved verbatim — see
/// [`sidecar_dir_override`].
///
/// Default: `{CARGO_TARGET_DIR or "target"}/ktstr/{kernel}-{project_commit}/`,
/// where `{kernel}` is the version detected from `KTSTR_KERNEL`'s
/// metadata (or `"unknown"` when no kernel is set / detection fails)
/// and `{project_commit}` is the project-tree HEAD short hex from
/// [`detect_project_commit`] (with `-dirty` suffix when the worktree
/// differs from HEAD), or `"unknown"` when the test process is not
/// running inside a git repository or the probe fails. Every sidecar
/// written from the same `cargo ktstr test` invocation lands in the
/// same directory; two runs sharing the same kernel + project commit
/// (e.g. re-running the same suite without committing changes) reuse
/// the same directory, with the second run pre-clearing any
/// `*.ktstr.json` files left by the first via
/// [`pre_clear_run_dir_once`] — the directory is a last-writer-wins
/// snapshot keyed on (kernel, project commit), not an append-only
/// archive of every invocation.
pub(crate) fn sidecar_dir() -> PathBuf {
    sidecar_dir_override().unwrap_or_else(resolve_default_sidecar_dir)
}

/// Compute the default-path sidecar directory:
/// `{runs_root}/{kernel}-{project_commit}` where `{kernel}` and
/// `{project_commit}` come from [`detect_kernel_version`] and
/// [`detect_project_commit`] respectively, with `"unknown"`
/// substituted via [`format_run_dirname`] when either probe
/// returns `None`. Emits the one-shot
/// [`warn_unknown_project_commit_once`] stderr warning when the
/// project commit probe falls back to `"unknown"` (operators in
/// this state lose the per-commit run-directory discriminator).
///
/// Shared by [`sidecar_dir`] and the default-path branch of
/// [`serialize_and_write_sidecar`] so both call sites resolve the
/// same kernel/commit/warn/format chain through one place.
/// `serialize_and_write_sidecar` cannot call [`sidecar_dir`]
/// directly because it needs a single-read of
/// [`sidecar_dir_override`] (gated against the env-var flipping
/// mid-call between the dir-resolve and the pre-clear gate); the
/// helper supplies the default-branch body so the override read
/// stays at one site.
fn resolve_default_sidecar_dir() -> PathBuf {
    let kernel = detect_kernel_version();
    let commit = detect_project_commit();
    if commit.is_none() {
        warn_unknown_project_commit_once();
    }
    runs_root().join(format_run_dirname(kernel.as_deref(), commit.as_deref()))
}

/// Build the run-directory leaf name from optional kernel and commit
/// components. `None` collapses to the literal `"unknown"` sentinel
/// in either slot, so a non-git cwd produces `"{kernel}-unknown"`
/// and a missing kernel produces `"unknown-{project_commit}"`. Pure
/// function over the two inputs — no I/O — so unit tests can pin
/// every shape (clean, dirty, missing-kernel, missing-commit, both
/// missing) without driving the [`detect_kernel_version`] /
/// [`detect_project_commit`] OnceLocks.
///
/// SENTINEL ASYMMETRY: the on-disk dirname uses `"unknown"` for
/// missing values, but the in-memory [`SidecarResult::project_commit`]
/// / [`SidecarResult::kernel_version`] fields stay `None` (`null`
/// in JSON). `cargo ktstr stats compare --project-commit unknown`
/// will NOT match a sidecar whose `project_commit` is `None` —
/// omit the filter to include `None`-commit rows. The asymmetry
/// is deliberate: the dirname needs a filesystem-safe sentinel,
/// while the JSON field preserves the original probe outcome for
/// downstream tooling that distinguishes "no probe ran" from
/// "probe ran but found nothing."
fn format_run_dirname(kernel: Option<&str>, commit: Option<&str>) -> String {
    let kernel = kernel.unwrap_or("unknown");
    let commit = commit.unwrap_or("unknown");
    format!("{kernel}-{commit}")
}

/// Resolve the parent directory that holds all test-run subdirectories.
///
/// `{CARGO_TARGET_DIR or "target"}/ktstr/`. Used by `cargo ktstr stats`
/// to enumerate runs without needing to reconstruct a specific run key.
pub fn runs_root() -> PathBuf {
    let target = std::env::var("CARGO_TARGET_DIR")
        .ok()
        .filter(|d| !d.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target"));
    target.join("ktstr")
}

/// Predicate: is `entry` a candidate run directory under
/// [`runs_root`]?
///
/// True iff `entry`'s path is a directory AND its filename does
/// NOT begin with a `.` byte. The dotfile filter excludes the
/// flock sentinel subdirectory ([`crate::flock::LOCK_DIR_NAME`] =
/// `.locks`) plus any other operator-created or filesystem-
/// reserved dotfile directories from run-listing walkers
/// ([`newest_run_dir`] here, `sorted_run_entries` in
/// `crate::stats`) so the lock infrastructure does not pollute
/// `cargo ktstr stats list` output or claim the "most recent
/// run" bucket. Checking the first byte directly via
/// `as_encoded_bytes` is OS-string-safe (no UTF-8 round-trip)
/// and short-circuits cleanly on non-UTF-8 names that would
/// confuse a `to_str().starts_with('.')` chain.
///
/// Single source of truth for "is this a run-dir entry?" — both
/// run-listing call sites must pipe through this predicate so a
/// future relocation of `.locks/` (or any other added reserved
/// dotfile) updates one place.
pub(crate) fn is_run_directory(entry: &std::fs::DirEntry) -> bool {
    let path = entry.path();
    if !path.is_dir() {
        return false;
    }
    path.file_name()
        .and_then(|n| n.as_encoded_bytes().first().copied())
        .is_none_or(|b| b != b'.')
}

/// Find the most recently modified run directory under [`runs_root`].
///
/// Used by bare `cargo ktstr stats` (no subcommand) when
/// `KTSTR_SIDECAR_DIR` isn't set: the stats command doesn't itself
/// run a kernel, so it can't reconstruct the
/// `{kernel}-{project_commit}` key that the test process used.
/// Picking the newest subdirectory by mtime mirrors "show me the
/// report from my last test run."
///
/// Dotfile-prefixed entries (notably the flock sentinel
/// subdirectory `.locks/`) are excluded via [`is_run_directory`]
/// so the lock infrastructure cannot claim the "most recent
/// run" bucket — `.locks/`'s mtime tracks per-write flock
/// activity and would otherwise eclipse the actual newest run
/// dir on every default-path sidecar write.
pub fn newest_run_dir() -> Option<PathBuf> {
    let root = runs_root();
    let entries = std::fs::read_dir(&root).ok()?;
    entries
        .filter_map(|e| e.ok())
        .filter(is_run_directory)
        .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok())
        .map(|e| e.path())
}

/// Detect the kernel version associated with the current test run.
///
/// Routes through [`crate::ktstr_kernel_env`] for the raw env value
/// and [`crate::kernel_path::KernelId`] for variant dispatch so the
/// three [`KernelId`] variants are honoured symmetrically:
///
/// - `KernelId::Path(dir)`: read `metadata.json` (cache entry
///   layout) or `include/config/kernel.release` (source tree
///   layout). Unchanged from the previous behaviour.
/// - `KernelId::Version(ver)`: the user asked for a specific
///   version — return it directly. No cache access needed; a
///   version string IS a version string.
/// - `KernelId::CacheKey(key)`: look up the cache entry and
///   return `entry.metadata.version`. The previous code path
///   silently treated the key as a directory name and read
///   `<cwd>/<key>/metadata.json`, which never matched — producing
///   `None` + `sidecar_dir()` using the `"unknown"` fallback even
///   though the cache metadata already carried the version.
///
/// Returns `None` when the env var is unset, or when the env
/// resolves to a variant whose underlying source doesn't yield a
/// version string (e.g. a Path whose metadata.json / kernel.release
/// are both absent, or a CacheKey with no cache hit).
pub(crate) fn detect_kernel_version() -> Option<String> {
    use crate::kernel_path::KernelId;
    let raw = crate::ktstr_kernel_env()?;
    match KernelId::parse(&raw) {
        KernelId::Path(_) => {
            let p = std::path::Path::new(&raw);
            let meta_path = p.join("metadata.json");
            if let Ok(data) = std::fs::read_to_string(&meta_path)
                && let Ok(meta) = serde_json::from_str::<crate::cache::KernelMetadata>(&data)
            {
                return meta.version;
            }
            let ver_path = p.join("include/config/kernel.release");
            if let Ok(v) = std::fs::read_to_string(ver_path) {
                let v = v.trim();
                if !v.is_empty() {
                    return Some(v.to_string());
                }
            }
            None
        }
        KernelId::Version(ver) => Some(ver),
        KernelId::CacheKey(key) => {
            let cache = crate::cache::CacheDir::new().ok()?;
            let entry = cache.lookup(&key)?;
            entry.metadata.version
        }
        // Multi-kernel specs in KTSTR_KERNEL never reach this
        // function in production — `find_kernel`'s env reader bails
        // before sidecar writing happens. This arm is defensive: if
        // the env value is somehow a range or git spec, return
        // `None` rather than guessing one endpoint, and the sidecar
        // record will leave `kernel_version` as null.
        KernelId::Range { .. } | KernelId::Git { .. } => None,
    }
}

/// Detect the ktstr project's git HEAD at sidecar-write time.
///
/// Walks up from the test process's current working directory via
/// `gix::discover` to find an enclosing repository, then reads HEAD
/// short-hex (7 chars via `oid::to_hex_with_len(7)`) and appends
/// `-dirty` when index-vs-HEAD or worktree-vs-index changes are
/// observed. Submodules are ignored
/// (`Submodule::Given { ignore: All }`).
///
/// Dirt-detection runs through the shared [`repo_is_dirty`]
/// helper (peel HEAD to its tree, diff tree-vs-index, then
/// `status()` for worktree-vs-index, submodules skipped); see its
/// doc for cascade details. The cascade is similar in spirit to
/// [`crate::fetch::local_source`]'s dirt probe but deliberately
/// diverges in missing-index handling: the sidecar path silently
/// degrades a missing index leg to "treat as clean" so metadata
/// probes never gate sidecar writes, whereas `local_source`'s
/// cache-key path treats every leg as load-bearing. The HASH
/// REPRESENTATION also DIFFERS: `fetch::local_source` DROPS the
/// short hash entirely on dirty (returns `None`) because the
/// commit no longer describes the build input the cache key
/// embeds — publishing a stale hash there would misidentify the
/// build. This helper KEEPS the hash with a `-dirty` suffix
/// instead because the sidecar's `project_commit` is a debugging
/// breadcrumb (operator-readable identity, not a cache-key input);
/// the hash plus dirty flag carries strictly more information
/// than `None` for the operator's "which ktstr commit did this
/// sidecar come from?" question.
///
/// Returns `None` when:
/// - `current_dir()` cannot be resolved (process has no valid
///   cwd — extremely rare; happens only for processes whose cwd
///   was rmdir'd while alive);
/// - cwd is not inside any git repository (`gix::discover` fails);
/// - HEAD cannot be read (an unborn HEAD on a fresh `git init`
///   with zero commits, or a corrupt repository).
///
/// Returns `Some(short_hash)` (without the `-dirty` suffix) when
/// the HEAD read succeeds but a downstream dirt-detection call
/// fails — including a missing index, an unreadable working tree,
/// or `head_tree()` failure. Each failed leg degrades to "treat
/// as clean" rather than aborting the probe, because metadata
/// must not gate sidecar writes.
///
/// `None` is the documented fallback — sidecar writing must not
/// abort because of a metadata probe failure. Stats tooling that
/// reads `project_commit` already tolerates `None` rows by
/// treating them as wildcards (no `--project-commit` filter narrowing
/// applies).
///
/// `gix::discover` is preferred over `gix::open` because tests can
/// be launched from a subdirectory of the repo (e.g.
/// `cd src && cargo test`); `discover` walks parents until it
/// finds the `.git` marker, while `open` requires the exact root
/// path. The walk is cheap — a few stat() calls bounded by the
/// depth of the cwd inside the repo.
///
/// `env!("CARGO_MANIFEST_DIR")` is deliberately NOT used here:
/// `env!` resolves at compile time and bakes the build-host's
/// absolute manifest path into the binary's read-only data
/// segment, leaking the build environment into every published
/// artifact. Resolving cwd at runtime instead means the recorded
/// commit reflects the project tree the test was launched FROM —
/// for a scheduler crate using ktstr as a dev-dependency, this is
/// the scheduler crate's commit, not ktstr's. That is the more
/// accurate semantic anyway: "what code produced this sidecar"
/// depends on the cwd at test launch (which crate is exercising
/// ktstr), not the build host.
pub(crate) fn detect_project_commit() -> Option<String> {
    // Per-process memoization: the cwd is stable for the lifetime
    // of a test process (no caller mutates it), and the project
    // tree's HEAD plus dirty state cannot change underneath us
    // without an explicit user action that's outside the scope
    // of any individual sidecar write. Gauntlet runs invoke this
    // function once per sidecar — thousands of times per process
    // — so caching the result behind a `OnceLock` collapses every
    // post-first call to a `Clone`. The probe itself does
    // ~3 syscalls (gix discover + head_id + status) which dominate
    // the sidecar-write critical path; eliminating that cost is
    // the only meaningful perf win available here.
    //
    // The cache is `Option<String>` so `None` (probe failure: no
    // git repo, unborn HEAD, etc.) also memoizes — repeating the
    // failing probe yields the same `None`, no point re-running.
    //
    // CACHE DOES NOT INVALIDATE: a user who commits / amends /
    // resets the project tree mid-run and expects the new HEAD
    // to surface in subsequent sidecars will see stale values.
    // This is acceptable per CLAUDE.md guidance — the project
    // tree is treated as stable-enough for a single suite run;
    // callers mutating the tree during a run own the consequences.
    static PROJECT_COMMIT: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    PROJECT_COMMIT
        .get_or_init(|| {
            let cwd = std::env::current_dir().ok()?;
            detect_commit_at(&cwd)
        })
        .clone()
}

/// Path-taking core of [`detect_project_commit`]. Factored out so
/// unit tests can drive the full branch matrix (clean repo, dirty
/// repo, non-git directory, unborn HEAD, concurrent calls) against
/// `gix::init`-built fixtures in tempdirs without mutating the
/// process-wide `current_dir`. The public entry point reads `cwd`
/// once and delegates here.
///
/// `gix::discover` walks parents until it finds a `.git` marker —
/// tests can be launched from a subdirectory of the repo (e.g.
/// `cd src && cargo test`); the parent walk handles that, where
/// `gix::open` would require the exact root. The
/// open-vs-discover distinction is the ONLY difference between
/// this function and [`detect_kernel_commit`]; the post-open
/// "read HEAD, format short hex, append `-dirty` on dirt" body
/// lives in the shared [`commit_with_dirty_suffix`] helper.
fn detect_commit_at(path: &std::path::Path) -> Option<String> {
    let repo = gix::discover(path).ok()?;
    commit_with_dirty_suffix(&repo)
}

/// Shared post-open body for [`detect_commit_at`] and
/// [`detect_kernel_commit`]: read `repo.head_id()`, format the
/// 7-char short hex, and append `-dirty` when [`repo_is_dirty`]
/// returns `Some(true)`.
///
/// Returns `None` when `head_id()` fails (unborn HEAD on a fresh
/// `gix::init` with zero commits, or a corrupt repository) — the
/// short-hex cannot be formed.
///
/// Returns `Some(short_hash)` (without `-dirty`) when the HEAD
/// read succeeds but the [`repo_is_dirty`] probe returns `None`
/// (HEAD-tree peel failure). This matches the documented "treat
/// as clean on probe failure" degradation: metadata probes must
/// not gate sidecar writes, so a probe failure flows through as
/// "clean" rather than aborting.
///
/// `to_hex_with_len(7)` produces a `HexDisplay` that formats 7
/// hex chars without the 40-char intermediate `format!("{}")`
/// allocation. `Id` derefs to `oid` (gix-hash) which owns the
/// method.
///
/// CALL SITES diverge ONLY on the open mode (`gix::discover` for
/// the project commit, `gix::open` for the kernel commit). The
/// helper takes a `&Repository` so each caller picks the open
/// strategy that matches its semantics: project commit walks
/// parents (cwd may be inside a subdir of the repo); kernel
/// commit demands the explicit root (the kernel directory is
/// not walked-up to avoid resolving the parent ktstr repo).
fn commit_with_dirty_suffix(repo: &gix::Repository) -> Option<String> {
    let head = repo.head_id().ok()?;
    let short_hash = head.to_hex_with_len(7).to_string();
    if repo_is_dirty(repo).unwrap_or(false) {
        Some(format!("{short_hash}-dirty"))
    } else {
        Some(short_hash)
    }
}

/// Probe whether a gix repository's working tree differs from its
/// HEAD commit, ignoring submodules.
///
/// Returns `Some(true)` when the index differs from the HEAD tree
/// or the worktree differs from the index for any tracked file;
/// `Some(false)` when neither leg observed a difference; `None`
/// when the HEAD-tree peel itself failed (HEAD points at something
/// that cannot be read as a tree).
///
/// Callers in [`detect_commit_at`] / [`detect_kernel_commit`]
/// degrade `None` to "treat as clean" via `unwrap_or(false)` so
/// metadata probes never gate sidecar writes.
///
/// PROBE LEGS:
/// - tree-vs-index: peel HEAD to its tree, then `tree_index_status`
///   diff against the on-disk index. `repo.index()` returning Err
///   (missing index — partially-checked-out clones, or fresh
///   `git init` before the first commit) silently leaves the
///   index-dirty leg false. `index_or_empty()` is deliberately
///   NOT used because it would substitute an empty index and the
///   diff would flag every tracked file as "deleted from index",
///   tripping false-dirty.
/// - index-vs-worktree: `repo.status()` configured with
///   `Submodule::Given { ignore: All }` so submodule worktree
///   state is skipped. Short-circuited when the tree-vs-index leg
///   already flipped dirty: the result only needs one positive
///   signal, so a known-dirty index makes the worktree walk
///   redundant. Matches the equivalent short-circuit in
///   [`crate::fetch::local_source`].
///
/// FAILURE DEGRADATION: any individual leg failure (missing index,
/// `repo.status()` failure, `into_index_worktree_iter()` failure)
/// silently degrades that leg to "no signal" rather than aborting.
/// The function only returns `None` when the HEAD-tree peel
/// fails, because at that point neither leg can run at all.
///
/// `pub` (not `pub(crate)`) because `cargo-ktstr.rs` is a
/// separate `[[bin]]` crate that consumes `ktstr` as an
/// external dependency and needs this helper to compute the
/// `-dirty` suffix in
/// `cargo ktstr stats compare --project-commit HEAD`. Hidden
/// from rustdoc via `#[doc(hidden)]` because it is a probe-
/// style helper without a stable API contract — external
/// consumers should not depend on it.
#[doc(hidden)]
pub fn repo_is_dirty(repo: &gix::Repository) -> Option<bool> {
    let head_tree_id = repo.head_tree().ok()?.id;

    let mut index_dirty = false;
    if let Ok(index) = repo.index() {
        let _ = repo.tree_index_status(
            &head_tree_id,
            &index,
            None,
            gix::status::tree_index::TrackRenames::Disabled,
            |_, _, _| {
                index_dirty = true;
                Ok::<_, std::convert::Infallible>(std::ops::ControlFlow::Break(()))
            },
        );
    }

    let worktree_dirty = if index_dirty {
        false
    } else {
        repo.status(gix::progress::Discard)
            .ok()
            .and_then(|s| {
                s.index_worktree_rewrites(None)
                    .index_worktree_submodules(gix::status::Submodule::Given {
                        ignore: gix::submodule::config::Ignore::All,
                        check_dirty: false,
                    })
                    .index_worktree_options_mut(|opts| {
                        opts.dirwalk_options = None;
                    })
                    .into_index_worktree_iter(Vec::new())
                    .ok()
                    .map(|mut iter| iter.next().is_some())
            })
            .unwrap_or(false)
    };

    Some(index_dirty || worktree_dirty)
}

/// Detect the kernel SOURCE TREE's git HEAD at sidecar-write time.
///
/// `kernel_dir` is the explicit kernel source directory — typically
/// resolved from `KTSTR_KERNEL` for `KernelId::Path`, or from the
/// cache entry's `KernelSource::Local::source_tree_path` when
/// `KTSTR_KERNEL` is a Version / CacheKey whose underlying build
/// recorded a local tree. Uses `gix::open(kernel_dir)` (NOT
/// `gix::discover`) because the kernel directory is explicit, not
/// walked-up: the parent walk that `discover` performs would
/// resolve to whichever ancestor `.git` it found first, which
/// might be the ktstr project's repo when `kernel_dir` is a
/// non-git subdirectory inside it. `open` requires `kernel_dir`
/// itself to be the repo root, which is the documented invariant
/// for kernel checkouts.
///
/// Reads HEAD short-hex (7 chars via `oid::to_hex_with_len(7)`)
/// and appends `-dirty` when index-vs-HEAD or worktree-vs-index
/// changes are observed. Dirt-detection runs through the shared
/// [`repo_is_dirty`] helper (submodules skipped via
/// `Submodule::Given { ignore: All }`); see its doc for cascade
/// details. The cascade matches [`detect_project_commit`] and is
/// similar in spirit to [`crate::fetch::local_source`] but
/// deliberately diverges in missing-index handling: the sidecar
/// path silently degrades a missing index leg to "treat as
/// clean" so metadata probes never gate sidecar writes, whereas
/// `local_source`'s cache-key path treats every leg as
/// load-bearing. Same "treat as clean on probe failure"
/// degradation rules apply otherwise: a missing index, an
/// unreadable worktree, or `head_tree()` failure each fall
/// through as "clean" rather than aborting the probe — metadata
/// must not gate sidecar writes.
///
/// HASH REPRESENTATION matches [`detect_project_commit`]: keeps
/// the hash with `-dirty` appended (operator-readable identity).
/// Distinct from [`crate::fetch::local_source`], which DROPS the
/// hash on dirty because the commit no longer describes the
/// build INPUT for cache-key purposes.
///
/// Returns `None` when:
/// - `kernel_dir` is not a git repository (`gix::open` fails);
/// - HEAD cannot be read (unborn HEAD on a fresh `git init` with
///   zero commits, or a corrupt repository).
///
/// Returns `Some(short_hash)` (without the `-dirty` suffix) when
/// the HEAD read succeeds but a downstream dirt-detection call
/// fails — including a missing index, an unreadable working
/// tree, or `head_tree()` failure. Each failed leg degrades to
/// "treat as clean" rather than aborting the probe, because
/// metadata must not gate sidecar writes.
pub(crate) fn detect_kernel_commit(kernel_dir: &std::path::Path) -> Option<String> {
    // Per-process, path-keyed memoization. Same rationale as
    // `detect_project_commit`: gauntlet runs invoke this function
    // once per sidecar — thousands of times — and the kernel
    // tree's HEAD plus dirty state cannot change underneath us
    // mid-suite without an explicit user action outside any
    // sidecar's control. The path key handles the fixture-test
    // case where unit tests rotate through synthetic
    // `tempfile::TempDir` kernel paths in the same process; each
    // distinct path memoizes independently.
    //
    // `Mutex<HashMap>` rather than `OnceLock` because the input
    // is parameterized on `kernel_dir` — a `OnceLock` collapses
    // every input to one cached result, which would conflate
    // different kernel directories into a single value.
    // Contention is bounded: post-warm reads are O(1) hash
    // lookups against a near-empty map (in production typically
    // ONE kernel per process), and the mutex is held only for
    // the duration of the lookup + insert.
    //
    // Mutex poisoning recovery: a panic mid-probe could poison
    // the lock; the `unwrap_or_else(|e| e.into_inner())` pattern
    // recovers the guard so a future caller doesn't fail
    // catastrophically. The cached map is just a HashMap of
    // owned types; no invariant beyond "key→value mapping" can
    // be broken by an interrupted probe.
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::{Mutex, OnceLock};
    static KERNEL_COMMIT_CACHE: OnceLock<Mutex<HashMap<PathBuf, Option<String>>>> = OnceLock::new();
    // Canonicalize the cache key so two paths that resolve to the
    // same on-disk directory share one entry. Without this, a
    // symlinked alias (`./linux` symlinked to `/abs/.../linux`)
    // and the resolved target would each populate their own slot,
    // re-running the gix-open + dirt-walk on every alias and
    // defeating the memoization. `canonicalize` resolves symlinks,
    // collapses `..` / `.`, and yields the absolute path the
    // kernel actually lives at. Falls back to the raw path on
    // canonicalize failure (e.g. caller passed a non-existent
    // `kernel_dir`) — gix::open will fail downstream and the
    // cache entry will memoize the `None` result against the raw
    // path, which is the correct behavior for a path that doesn't
    // exist (no symlink alias is possible).
    let cache_key = kernel_dir
        .canonicalize()
        .unwrap_or_else(|_| kernel_dir.to_path_buf());
    let cache = KERNEL_COMMIT_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(cached) = guard.get(&cache_key) {
        return cached.clone();
    }
    // `gix::open` (NOT `gix::discover`) — `kernel_dir` must BE the
    // repo root. Without this the parent walk could resolve to the
    // ktstr project's own `.git` when `kernel_dir` is a non-git
    // subdirectory inside the ktstr checkout. The
    // open-vs-discover distinction is the ONLY difference between
    // this function and [`detect_commit_at`]; the post-open
    // "read HEAD, format short hex, append `-dirty` on dirt" body
    // lives in the shared [`commit_with_dirty_suffix`] helper.
    //
    // Open against `kernel_dir` (the caller-supplied path) rather
    // than `cache_key`. The two paths point at the same on-disk
    // repo by construction (canonicalize resolves to the same
    // place), so gix opens the same repository either way; passing
    // the original keeps any user-facing diagnostics (gix's
    // internal error chain) consistent with the input shape.
    let result = gix::open(kernel_dir)
        .ok()
        .and_then(|repo| commit_with_dirty_suffix(&repo));
    guard.insert(cache_key, result.clone());
    result
}

/// Environment variable CI runners set to mark sidecars they produce
/// as `"ci"`-source. Any non-empty value flips the tag; empty string
/// is treated as unset so a defensively-cleared variable does not
/// accidentally classify a developer run as CI.
///
/// Read at sidecar-write time by [`detect_run_source`]; matches the
/// `KTSTR_KERNEL` / `KTSTR_CACHE_DIR` env-name convention so the
/// full set of ktstr-controlled env vars is `KTSTR_*`-prefixed.
pub const KTSTR_CI_ENV: &str = "KTSTR_CI";

/// Tag value written to [`SidecarResult::run_source`] for sidecars
/// produced under [`KTSTR_CI_ENV`].
pub const SIDECAR_RUN_SOURCE_CI: &str = "ci";

/// Tag value written to [`SidecarResult::run_source`] for sidecars
/// produced without [`KTSTR_CI_ENV`] — the developer-machine
/// default.
pub const SIDECAR_RUN_SOURCE_LOCAL: &str = "local";

/// Tag value applied to [`SidecarResult::run_source`] /
/// [`GauntletRow::run_source`](crate::stats::GauntletRow::run_source)
/// at LOAD time when the consumer pulls sidecars from a non-default
/// pool root via `cargo ktstr stats compare --dir` /
/// `cargo ktstr stats list-values --dir`. NEVER written by
/// [`write_sidecar`] — the writer cannot know the file will later
/// be moved off-host. See [`apply_archive_source_override`].
pub const SIDECAR_RUN_SOURCE_ARCHIVE: &str = "archive";

/// Read [`KTSTR_CI_ENV`] and classify the run as `"ci"` (when the
/// env var is set non-empty) or `"local"` (the default for any
/// developer-driven invocation). Empty-string env values count as
/// unset — see [`KTSTR_CI_ENV`] for rationale.
///
/// Returns `Some(_)` unconditionally because every sidecar producer
/// is, by construction, either local or CI; an `Option` return
/// keeps the field shape symmetric with the other nullable
/// `SidecarResult` fields and reserves room for a future "unknown"
/// arm without a serde-version bump.
pub(crate) fn detect_run_source() -> Option<String> {
    match std::env::var(KTSTR_CI_ENV) {
        Ok(v) if !v.is_empty() => Some(SIDECAR_RUN_SOURCE_CI.to_string()),
        _ => Some(SIDECAR_RUN_SOURCE_LOCAL.to_string()),
    }
}

/// Override every sidecar's `run_source` field to
/// [`SIDECAR_RUN_SOURCE_ARCHIVE`] when the consumer pulled the pool
/// from a non-default root via `--dir`. Called at the boundary
/// between [`collect_pool`] and the downstream stats pipeline so
/// on-disk values stay untouched while the in-memory pool reflects
/// the operator's intent: "these sidecars were copied off another
/// host; treat them as archives, not as the local-machine record."
///
/// Mutation strategy is in-place rewrite of the entire `run_source`
/// field — the `"local"` / `"ci"` distinction is meaningful on the
/// PRODUCING host but irrelevant once the sidecars have been
/// moved off, where the only useful classification is "archived
/// elsewhere." Operators who need to retain the producer-side
/// distinction inside an archive bucket can keep `--dir`
/// untargeted (read from the default root) and let the on-disk
/// values pass through.
pub(crate) fn apply_archive_source_override(pool: &mut [SidecarResult]) {
    for sc in pool {
        sc.run_source = Some(SIDECAR_RUN_SOURCE_ARCHIVE.to_string());
    }
}

/// Resolve the kernel source-tree path for [`detect_kernel_commit`]
/// from the [`crate::KTSTR_KERNEL_ENV`] env var.
///
/// Routes through [`crate::ktstr_kernel_env`] for the raw env
/// value and [`crate::kernel_path::KernelId`] for variant
/// dispatch:
///
/// - `KernelId::Path(p)`: returns the raw path verbatim. The
///   typical case: `KTSTR_KERNEL=/path/to/linux` points at a
///   working source tree that may be a git repo.
/// - `KernelId::Version(ver)`: looks for a Local cache entry
///   whose `metadata.version == ver` carrying a
///   `source_tree_path`. The tarball-shaped key (`{ver}-tarball-
///   {arch}-kc{suffix}`) is checked first because it is the
///   most-common form a Version-shaped env points at; on miss
///   (or hit yielding `Tarball` / `Git` source, both of which
///   are transient with no on-disk tree to probe), the function
///   falls back to scanning every valid cache entry for a Local
///   match on version. The fallback is the bug fix for #58:
///   without it, a cache populated by `kernel build --kernel
///   /path/to/linux` (a Local entry with source_tree_path) is
///   never found by a sidecar writer that has
///   `KTSTR_KERNEL=6.14.2`, even though the local tree is
///   exactly what the kernel_commit field needs to probe.
/// - `KernelId::CacheKey(k)`: uses `k` verbatim — the cache key
///   already carries every detail (source-type prefix, arch,
///   kconfig hash). On hit, returns
///   `KernelSource::Local::source_tree_path` if set, else
///   `None` (Tarball / Git entries are transient and have no
///   persisted source tree).
/// - `KernelId::Range { .. }` / `KernelId::Git { .. }`:
///   multi-kernel specs in `KTSTR_KERNEL` never reach this
///   helper in production (find_kernel's env reader bails
///   before sidecar writing). Defensive: returns `None`.
///
/// Returns `None` when the env var is unset, when no source
/// tree path is recoverable, or when the cache lookup fails.
fn resolve_kernel_source_dir() -> Option<std::path::PathBuf> {
    use crate::kernel_path::KernelId;
    let raw = crate::ktstr_kernel_env()?;
    let id = KernelId::parse(&raw);
    match id {
        KernelId::Path(_) => Some(std::path::PathBuf::from(&raw)),
        KernelId::Version(_) | KernelId::CacheKey(_) => {
            let cache = crate::cache::CacheDir::new().ok()?;
            resolve_kernel_source_dir_with_cache(&id, &cache)
        }
        KernelId::Range { .. } | KernelId::Git { .. } => None,
    }
}

/// Pure helper for [`resolve_kernel_source_dir`] that takes the
/// parsed `KernelId` and an opened `CacheDir`, returning the source
/// tree path if recoverable.
///
/// Split out from [`resolve_kernel_source_dir`] so tests can pin a
/// `CacheDir` at a tempdir root without mutating env vars (which
/// would race other tests reading `KTSTR_KERNEL` /
/// `KTSTR_CACHE_DIR`).
///
/// Lookup order for [`KernelId::Version`]:
/// 1. Tarball-shaped cache key (`{ver}-tarball-{arch}-kc{suffix}`),
///    direct lookup. Returns `Some` only if the entry is a
///    `KernelSource::Local` carrying a `source_tree_path`.
/// 2. Fallback scan: every valid cache entry whose
///    `metadata.version == ver`. First match with
///    `KernelSource::Local::source_tree_path` set wins. Handles
///    the case where the user built `--kernel /path/to/linux`
///    (a Local cache entry without the tarball cache-key prefix)
///    but later set `KTSTR_KERNEL=6.14.2` for the test run —
///    without this fallback, the local source tree would be
///    invisible to the sidecar writer.
///
/// `KernelSource::Tarball` and `KernelSource::Git` entries are
/// skipped at every step because their source trees are transient
/// (deleted by the cache pipeline after build), so probing them
/// for a `kernel_commit` would always fail.
///
/// For [`KernelId::CacheKey`], performs a single direct lookup —
/// the cache key already encodes every detail (source-type
/// prefix, arch, kconfig hash) so no fallback scan is needed.
fn resolve_kernel_source_dir_with_cache(
    id: &crate::kernel_path::KernelId,
    cache: &crate::cache::CacheDir,
) -> Option<std::path::PathBuf> {
    use crate::kernel_path::KernelId;
    match id {
        KernelId::Version(ver) => {
            let arch = std::env::consts::ARCH;
            let tarball_key = format!("{ver}-tarball-{arch}-kc{}", crate::cache_key_suffix());
            if let Some(entry) = cache.lookup(&tarball_key)
                && let crate::cache::KernelSource::Local {
                    source_tree_path: Some(p),
                    ..
                } = &entry.metadata.source
            {
                return Some(p.clone());
            }
            let entries = cache.list().ok()?;
            for listed in entries {
                let crate::cache::ListedEntry::Valid(entry) = listed else {
                    continue;
                };
                if entry.metadata.version.as_deref() != Some(ver.as_str()) {
                    continue;
                }
                if let crate::cache::KernelSource::Local {
                    source_tree_path: Some(p),
                    ..
                } = &entry.metadata.source
                {
                    return Some(p.clone());
                }
            }
            None
        }
        KernelId::CacheKey(k) => {
            let entry = cache.lookup(k)?;
            match entry.metadata.source {
                crate::cache::KernelSource::Local {
                    source_tree_path: Some(ref p),
                    ..
                } => Some(p.clone()),
                _ => None,
            }
        }
        // Path / Range / Git callers do not reach this helper —
        // resolve_kernel_source_dir handles them inline. Defensive
        // None covers any future caller that adds a new arm.
        _ => None,
    }
}

/// Compute a stable 64-bit discriminator over the fields that
/// distinguish gauntlet variants of the same test. Used to suffix
/// the sidecar filename so concurrent variants do not clobber each
/// other's output.
///
/// Uses [`siphasher::sip::SipHasher13`] with zero keys for the same
/// stability reason as the initramfs cache keys — the discriminator
/// must be the same across Rust toolchain versions or downstream
/// tooling that groups variants by filename breaks.
///
/// # Host-state collision caveat
///
/// The hash is over test-identity fields (topology, scheduler,
/// payload, work_type, flags, sysctls, kargs) — NOT over
/// [`HostContext`], NOT over `scheduler_commit`, NOT over
/// `project_commit`, NOT over `kernel_commit`, and NOT over
/// `run_source`. The [`HostContext`] exclusion is pinned by
/// [`sidecar_variant_hash_excludes_host_context`]; the
/// `scheduler_commit` exclusion by
/// [`sidecar_variant_hash_excludes_scheduler_commit`]; the
/// `project_commit` exclusion by
/// [`sidecar_variant_hash_excludes_project_commit`]; the
/// `kernel_commit` exclusion by
/// [`sidecar_variant_hash_excludes_kernel_commit`]; the
/// `run_source` exclusion by
/// [`sidecar_variant_hash_excludes_run_source`]. All five are
/// deliberate for the same cross-host grouping reason — a
/// gauntlet rebuilt against a different userspace scheduler
/// commit, a bumped ktstr checkout, a kernel source tree at a
/// different HEAD, or a different CI runner / developer
/// machine must still bucket with the same-named variant so
/// `compare_partitions` can diff two runs of the "same" test
/// without the commit hash or run-source tag shattering them
/// into one-row-per-commit islands. Callers that want to detect
/// a commit drift or compare across run environments inspect
/// [`SidecarResult::scheduler_commit`] /
/// [`SidecarResult::project_commit`] /
/// [`SidecarResult::kernel_commit`] /
/// [`SidecarResult::run_source`] directly (the latter three via
/// `--project-commit` / `--kernel-commit` / `--run-source` on
/// `stats compare`); the filename stays stable across commits
/// and run environments by design.
///
/// The corollary of the HostContext exclusion: if the host's
/// observable state mutates mid-suite — NUMA hotplug, hugepage
/// reconfiguration, a `sysctl -w` from a parallel process — two
/// runs of the same test will produce the same sidecar filename
/// and the later write clobbers the earlier. ktstr treats host
/// state as stable-enough for a single suite run; callers
/// mutating host state during a run own the ordering themselves
/// (e.g. by writing to a different `KTSTR_SIDECAR_DIR` per host
/// snapshot).
pub(crate) fn sidecar_variant_hash(sidecar: &SidecarResult) -> u64 {
    use siphasher::sip::SipHasher13;
    use std::hash::Hasher;
    let mut h = SipHasher13::new_with_keys(0, 0);
    h.write(sidecar.topology.as_bytes());
    h.write(&[0]);
    h.write(sidecar.scheduler.as_bytes());
    h.write(&[0]);
    // Binary payload name — two tests that differ only in the
    // primary payload (e.g. scheduler=EEVDF + payload=FIO vs
    // scheduler=EEVDF + payload=STRESS_NG) must produce distinct
    // sidecar filenames. `None` emits a single separator byte so the
    // absent-payload variant doesn't collide with a payload name that
    // happens to hash-chain into the next field.
    h.write(&[0xfc]);
    if let Some(name) = &sidecar.payload {
        h.write(name.as_bytes());
    }
    h.write(&[0]);
    h.write(sidecar.work_type.as_bytes());
    h.write(&[0]);
    h.write(&[0xfe]);
    for f in &sidecar.active_flags {
        h.write(f.as_bytes());
        h.write(&[0]);
    }
    // Sysctls and kargs are canonicalized at hash time — NOT at
    // write time like `active_flags` — so the on-disk sidecar
    // preserves the scheduler-declared order (useful for humans
    // reading the JSON) while the filename suffix stays a pure
    // function of the SET, not the sequence. Sorting lexically
    // here means two schedulers that declare the same sysctls in
    // different source-code orders fold to the same filename,
    // matching the order-insensitivity contract documented on
    // `canonicalize_active_flags`. Two small `Vec<&str>` per
    // call — acceptable because `sidecar_variant_hash` runs
    // once per `write_sidecar`, not on a hot path.
    h.write(&[0xfd]);
    let mut sorted_sysctls: Vec<&str> = sidecar.sysctls.iter().map(String::as_str).collect();
    sorted_sysctls.sort_unstable();
    for s in &sorted_sysctls {
        h.write(s.as_bytes());
        h.write(&[0]);
    }
    h.write(&[0xff]);
    let mut sorted_kargs: Vec<&str> = sidecar.kargs.iter().map(String::as_str).collect();
    sorted_kargs.sort_unstable();
    for k in &sorted_kargs {
        h.write(k.as_bytes());
        h.write(&[0]);
    }
    h.finish()
}

/// Entry-derived scheduler metadata that every sidecar carries
/// regardless of pass/fail/skip.
///
/// Both write paths ([`write_sidecar`] and [`write_skip_sidecar`])
/// thread the same materialized fields through to their
/// `SidecarResult` constructors; keeping the derivation in a
/// named struct (rather than a 4-tuple) means a new
/// scheduler-level field shows up as a named field at both
/// writer sites and in every call-site binding, instead of as
/// an additional anonymous tuple slot that readers have to
/// remember the ordering of.
///
/// `pub(crate)` rather than `pub`: the intermediate struct is a
/// write-path detail, not a public API surface. No serde — this
/// is not a persisted shape, just a grouped return value.
///
/// Derives `Debug` for `assert_eq!` diagnostics, `Clone` so tests
/// can materialize a fixture once and reuse it across assertions,
/// and `PartialEq`/`Eq` so tests can compare whole fingerprints
/// in one statement rather than destructuring and asserting on
/// each field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SchedulerFingerprint {
    /// Pretty scheduler name (matches `SidecarResult::scheduler`),
    /// e.g. `"eevdf"` or a scheduler-kind payload's declared name.
    pub(crate) scheduler: String,
    /// Best-effort userspace scheduler commit; `None` for every
    /// current variant per
    /// [`crate::test_support::SchedulerSpec::scheduler_commit`].
    pub(crate) scheduler_commit: Option<String>,
    /// Formatted `sysctl.<key>=<value>` lines derived from the
    /// scheduler's declared `sysctls()`.
    pub(crate) sysctls: Vec<String>,
    /// Kernel command-line args declared by the scheduler,
    /// forwarded verbatim.
    pub(crate) kargs: Vec<String>,
}

/// Materialize the [`SchedulerFingerprint`] for a test entry.
///
/// A change to the sidecar schema (e.g. a new scheduler-level
/// field) extends this function + [`SchedulerFingerprint`] in
/// one place and every writer picks it up automatically.
fn scheduler_fingerprint(entry: &KtstrTestEntry) -> SchedulerFingerprint {
    let scheduler = entry.scheduler.scheduler_name().to_string();
    // `entry.scheduler` is a `&Payload` wrapper, not a `&Scheduler`
    // directly — routing through `scheduler_binary()` returns the
    // underlying `Option<&SchedulerSpec>` (None for binary-kind
    // payloads). Flatten with `and_then` so a binary-kind payload
    // naturally yields `None` without duplicating the
    // binary-vs-scheduler dispatch logic here.
    let scheduler_commit = entry
        .scheduler
        .scheduler_binary()
        .and_then(|s| s.scheduler_commit())
        .map(|s| s.to_string());
    let sysctls: Vec<String> = entry
        .scheduler
        .sysctls()
        .iter()
        .map(|s| format!("sysctl.{}={}", s.key, s.value))
        .collect();
    let kargs: Vec<String> = entry
        .scheduler
        .kargs()
        .iter()
        .map(|s| s.to_string())
        .collect();
    SchedulerFingerprint {
        scheduler,
        scheduler_commit,
        sysctls,
        kargs,
    }
}

/// Compute the per-variant sidecar path and serialize + write the
/// result to disk.
///
/// Gauntlet variants of the same test differ by work_type, flags
/// (via scheduler args → sysctls/kargs), scheduler, and topology. A
/// filename of just `{test_name}.ktstr.json` causes variants to
/// overwrite each other, erasing all but the last-written result.
/// `sidecar_variant_hash` hashes the discriminating fields into a
/// short stable suffix so each variant gets its own sidecar file.
///
/// On the first call PER UNIQUE DIRECTORY within a process,
/// [`pre_clear_run_dir_once`] removes any pre-existing
/// `*.ktstr.json` files in the resolved directory so the run is a
/// clean snapshot rather than a mosaic of sidecars carried over
/// from a prior invocation that shared the same
/// `{kernel}-{project_commit}` key (e.g. re-running the suite
/// without committing changes).
/// Subsequent writes within the same process to the same directory
/// append into the cleared directory.
///
/// Pre-clear is SKIPPED when `KTSTR_SIDECAR_DIR` is set: the
/// operator chose that directory and owns its contents — silent
/// data loss is not acceptable on an explicit override. When the
/// override is unset (the default-path branch),
/// `std::fs::create_dir_all` materializes the directory BEFORE
/// pre-clear runs so the helper's canonicalize step always sees
/// an existing on-disk path; without this ordering, a missing
/// dir on the very first call would key the cache against the
/// raw path while a later call (after the dir exists) would key
/// against the canonicalized absolute path, splitting the cache
/// and causing the second call to re-fire pre-clear and wipe the
/// first call's sidecars.
///
/// CROSS-PROCESS SERIALIZATION: on the default path (override
/// unset), the call acquires advisory `LOCK_EX` on a per-run-key
/// sentinel file (`{runs_root}/.locks/{key}.lock`) before
/// pre-clear runs and holds it for the duration of the
/// pre-clear + serialize + write cycle. The lock prevents
/// process B's `pre_clear_run_dir_once` from interleaving with
/// process A's mid-write `std::fs::write` — the kernel-flock
/// critical section makes the (read_dir + remove_file) +
/// (serialize + write) sequence atomic with respect to peer
/// processes targeting the same `{kernel}-{project_commit}`
/// directory. Without the lock, two concurrent CI jobs sharing
/// the same key could (a) tear partially-written sidecars
/// (write fd open while pre-clear's `remove_file` runs) or
/// (b) interleave pre-clear + write phases, leaving the dir
/// in a state neither process intended. The override path
/// skips the lock for the same reason it skips pre-clear:
/// operator-chosen directories are owned by the operator and
/// out of scope for the cross-process gate.
///
/// `label` is a caller-supplied noun for the context message ("skip
/// sidecar" / "sidecar") so the error chain points at the right call
/// site.
fn serialize_and_write_sidecar(sidecar: &SidecarResult, label: &str) -> anyhow::Result<()> {
    // Read the override ONCE. The two branches below carry the
    // result through structurally so neither leg re-reads
    // `KTSTR_SIDECAR_DIR` — preventing the override from flipping
    // mid-call (which would otherwise let an external mutation
    // between the dir resolve and the pre-clear gate either skip
    // the wipe on a default-path dir or fire a wipe on an
    // operator-chosen one).
    let (dir, do_pre_clear) = match sidecar_dir_override() {
        Some(path) => (path, false),
        None => (resolve_default_sidecar_dir(), true),
    };
    // Materialize the directory FIRST so `pre_clear_run_dir_once`
    // can canonicalize a path that exists on disk. Without this,
    // the very first invocation in a process resolves the cache
    // key against the raw relative path (canonicalize fails on a
    // missing dir, falls back to raw); subsequent invocations
    // resolve against the canonicalized absolute path because the
    // dir now exists. Two distinct keys for the same logical dir
    // → second invocation re-fires pre-clear and wipes the first
    // invocation's sidecars. Materializing pre-pre-clear closes
    // the relative-vs-absolute split.
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("create sidecar dir {}", dir.display()))?;
    // Acquire the per-run-key cross-process flock for the duration
    // of the pre-clear + write cycle. The override branch (operator-
    // chosen directory) skips the lock for the same reason it skips
    // pre-clear — see the function-level doc. `_run_dir_lock` is
    // scoped to this function body so the kernel-side flock releases
    // via `OwnedFd::drop` when the function returns (success or
    // error path), making the lock RAII-managed without an explicit
    // unlock call.
    let _run_dir_lock = if do_pre_clear {
        Some(acquire_run_dir_flock(&dir)?)
    } else {
        None
    };
    if do_pre_clear {
        pre_clear_run_dir_once(&dir);
    }
    let variant_hash = sidecar_variant_hash(sidecar);
    let path = dir.join(format!(
        "{}-{:016x}.ktstr.json",
        sidecar.test_name, variant_hash
    ));
    let json = serde_json::to_string_pretty(sidecar)
        .with_context(|| format!("serialize {label} for '{}'", sidecar.test_name))?;
    std::fs::write(&path, json).with_context(|| format!("write {label} {}", path.display()))?;
    Ok(())
}

/// `Some(path)` when `KTSTR_SIDECAR_DIR` is set non-empty,
/// returning the override path verbatim; `None` when the env
/// var is unset or empty (default-path branch). Single source
/// of truth for the override read so [`sidecar_dir`] and
/// [`serialize_and_write_sidecar`] (which gates pre-clear on
/// the override's presence) share one env-read site rather
/// than each calling `std::env::var` independently.
///
/// The `is_empty()` filter is deliberate: a defensively-cleared
/// `KTSTR_SIDECAR_DIR=""` must NOT be treated as an override
/// (joining an empty path onto the run-root would silently
/// alias the runs-root itself, contaminating the listing).
/// Empty-string aliases unset, matching the
/// `if let Ok(d) ... && !d.is_empty()` predicate the function
/// replaced.
///
/// `serialize_and_write_sidecar` interprets `Some(_)` as the
/// "operator chose this dir, do not pre-clear" gate — silent
/// data loss is unacceptable on an explicit override (the
/// override is for users who want exact control over where
/// sidecars land: test isolation, archival capture, custom CI
/// layouts).
fn sidecar_dir_override() -> Option<PathBuf> {
    std::env::var("KTSTR_SIDECAR_DIR")
        .ok()
        .filter(|d| !d.is_empty())
        .map(PathBuf::from)
}

/// Emit a one-shot stderr warning when [`detect_project_commit`]
/// resolves to `None` and the run directory therefore lands at
/// `{kernel}-unknown`. Operators in this state lose the
/// `{project_commit}` discriminator on the run-directory name —
/// every non-git invocation at the same kernel collides on a
/// single directory, with the latest run pre-clearing the
/// previous one's sidecars. The warning surfaces this loss-of-isolation
/// risk so the operator can either set `KTSTR_SIDECAR_DIR` to
/// disambiguate per-run, or place the project tree under git
/// so each run carries its own commit hash.
///
/// `OnceLock<()>` gates the warning to fire EXACTLY ONCE per
/// process: every gauntlet variant resolves a sidecar directory
/// independently (via [`sidecar_dir`] and
/// [`serialize_and_write_sidecar`]), so without the gate the
/// operator would see thousands of duplicate warnings interleaved
/// with test output. Called via [`resolve_default_sidecar_dir`] —
/// which is the shared default-path body that both [`sidecar_dir`]
/// and [`serialize_and_write_sidecar`] funnel through — so the
/// warning fires only on the default-path branch. The override
/// branch in either caller returns before
/// [`resolve_default_sidecar_dir`] is reached, so an operator who
/// set `KTSTR_SIDECAR_DIR` to disambiguate non-git runs does not
/// see a misleading "commit unknown" warning that does not apply
/// to their effective directory layout.
///
/// Implementation is split into a public-facing wrapper
/// (this function) that owns the process-global `OnceLock` and
/// targets stderr, and a pure inner helper
/// [`warn_unknown_project_commit_inner`] that takes the
/// `&OnceLock<()>` gate and the `&mut dyn Write` sink as
/// parameters. The split lets tests drive the warning logic
/// against a local `OnceLock` and a `Vec<u8>` sink without
/// fighting the process-global gate or the global stderr fd —
/// the wrapper's behavior is what the inner does, just with
/// the static gate and stderr supplied.
fn warn_unknown_project_commit_once() {
    static WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    let mut sink = std::io::stderr();
    warn_unknown_project_commit_inner(&WARNED, &mut sink);
}

/// Pure helper for [`warn_unknown_project_commit_once`]: gate the
/// warning on `gate` and write the warning text to `sink` exactly
/// once across the gate's lifetime. Both parameters are taken by
/// reference so call sites supply ownership semantics that match
/// their gating story:
/// - The production wrapper passes a `'static` `OnceLock<()>` so
///   the gate spans the whole process and a stderr handle so the
///   warning lands in the operator's terminal.
/// - Tests pass a local `OnceLock<()>` so each test gets a fresh
///   gate (no cross-test contamination via a process-global)
///   and a `Vec<u8>` sink so the test can read back the emitted
///   bytes and assert on the warning text.
///
/// Errors from `writeln!` are ignored via `let _ =`: a metadata
/// probe warning must not gate sidecar writes. This DEPARTS from
/// the previous `eprintln!` semantics (which panic on stderr
/// write failure per the std docs) — here we drop the write
/// error silently because a metadata probe warning must not gate
/// sidecar writes.
fn warn_unknown_project_commit_inner(
    gate: &std::sync::OnceLock<()>,
    sink: &mut dyn std::io::Write,
) {
    gate.get_or_init(|| {
        let _ = writeln!(
            sink,
            "ktstr: WARNING: project commit unavailable (cwd not in a git \
             repo, or HEAD unreadable); runs at this kernel overwrite \
             each other in target/ktstr/{{kernel}}-unknown/. Set \
             KTSTR_SIDECAR_DIR=<unique-path> per run, or run from inside a \
             git repo with at least one commit."
        );
    });
}

/// Remove any pre-existing `*.ktstr.json` files in the resolved
/// run directory, exactly once per unique directory per process.
///
/// The run-key format is `{kernel}-{project_commit}` (see
/// [`sidecar_dir`]), so two `cargo ktstr test` invocations sharing
/// the same kernel and project commit (the typical "re-run the
/// suite without committing changes" loop) resolve to the same
/// directory. Without
/// pre-clearing, each subsequent run would land its sidecars next
/// to the previous run's, leaving downstream `cargo ktstr stats`
/// readers to see a mosaic of two distinct test outcomes for the
/// same variant — the variant-hash suffix on each filename
/// prevents overwrites within a single run, but ALSO prevents the
/// next run from naturally clobbering the previous one's files
/// when the test set or pass/fail mix changes. Wiping
/// `*.ktstr.json` once at first-write makes each run a clean
/// snapshot of (kernel, project commit) — the last-writer-wins
/// semantics the directory naming implies.
///
/// PER-DIRECTORY KEYING: the cache is a `Mutex<HashSet<PathBuf>>`
/// keyed on the canonicalized `dir` (with raw `dir` as fallback
/// when canonicalize fails — e.g. the directory does not yet
/// exist). A `OnceLock<()>` would fire once for the FIRST
/// directory only, leaving subsequent writes to other directories
/// unprotected. The HashSet ensures every distinct directory the
/// process writes to gets pre-cleared exactly once, regardless of
/// ordering. Canonicalization collapses symlink aliases so two
/// path spellings of the same on-disk dir share one entry.
///
/// In production today only the default-path
/// `runs_root().join({kernel}-{project_commit})` is fed into this
/// function (the override path skips pre-clear entirely via
/// [`sidecar_dir_override`]), so per-process cache size
/// stays at exactly 1 entry. The HashSet shape is the
/// future-proof keying for direct unit-test fixtures (which
/// rotate tempdir paths through this helper) and any future
/// production code path that writes default-path sidecars from
/// multiple distinct (kernel, commit) pairs in one process.
///
/// SCOPE: only `*.ktstr.json` files in the immediate directory
/// are removed. Subdirectories (per-job gauntlet layouts written
/// by external orchestrators) and non-sidecar files are left
/// untouched — pre-clear is shallow. Note that `collect_sidecars`
/// walks one level of subdirectories, so stale sidecars left in
/// subdirectories from a prior run will still appear in
/// `cargo ktstr stats` output until the operator removes them.
/// The function never deletes the directory itself; production
/// callers (`serialize_and_write_sidecar`) materialize the
/// directory via `create_dir_all` BEFORE invoking this helper, so
/// the only file-deletion side effect is the `*.ktstr.json`
/// wipe inside an existing dir.
///
/// CONCURRENT WRITERS: the per-process `Mutex<HashSet>` guards
/// against multiple writes within a single process re-clearing
/// the same directory. The cache mutex is held ACROSS the
/// `read_dir` walk and per-file removals — releasing it after
/// the cache insert but before the walk would open a TOCTOU
/// window where a sibling thread observes the cached entry,
/// skips its own pre-clear, writes a sidecar, and then the
/// original thread's still-pending walk deletes that sibling's
/// fresh file. Holding the lock across the bounded walk closes
/// the window. Two concurrent test PROCESSES that both resolve
/// to the same `{kernel}-{project_commit}` run dir will both
/// pre-clear; that cross-process race is out of scope here
/// (tracked separately under the concurrent-write collision
/// protection backlog item) and would corrupt each other's
/// outputs even without pre-clearing.
///
/// FAILURE: `read_dir` errors are silently ignored — defensive
/// behavior for direct callers (e.g. unit tests probing the
/// missing-dir edge); production callers materialize the
/// directory before invoking this helper, so the missing-dir
/// branch is unreachable in production today. Metadata probes
/// must not gate sidecar writes. Per-file `remove_file`
/// errors are also silently ignored — a partial pre-clear leaves
/// either an overwrite (when the new run reproduces a stale
/// file's exact `{test_name}-{variant_hash}.ktstr.json` name —
/// the desired outcome) or a coexistence (when the new run's
/// variant set differs from the prior run's, leaving stale
/// sidecars next to fresh ones — the undesired outcome that
/// pre-clear was meant to prevent). Coexistence is the acceptable
/// degradation here: a noisy pre-clear failure should not abort
/// the test run.
fn pre_clear_run_dir_once(dir: &std::path::Path) {
    use std::collections::HashSet;
    use std::path::PathBuf;
    use std::sync::{Mutex, OnceLock};
    static PRE_CLEARED: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();
    // Canonicalize so two spellings of the same on-disk dir share
    // one cache entry. Falls back to the raw path when canonicalize
    // fails (the directory may not exist yet on the very first
    // write, in which case the raw path keys the entry; subsequent
    // calls with the same raw path also miss canonicalize the
    // same way and share the entry).
    let cache_key = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    let cache = PRE_CLEARED.get_or_init(|| Mutex::new(HashSet::new()));
    let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
    if !guard.insert(cache_key) {
        return;
    }
    // First time this directory has been seen — wipe sidecars while
    // the cache mutex is still held. Releasing the guard before the
    // read_dir walk would open a TOCTOU window: a sibling thread that
    // observes the now-cached entry would skip its own pre-clear,
    // proceed to write a sidecar, and the original thread's walk
    // (running after the drop) would then delete that sibling's
    // freshly-written file. The walk is one read_dir + a bounded
    // number of `*.ktstr.json` removals, so holding the lock across
    // it is brief; concurrent calls against DIFFERENT directories
    // serialize through this critical section but each does a small,
    // bounded amount of I/O, which is acceptable for a metadata
    // probe call pattern. `guard` is dropped at end-of-scope so the
    // lock release happens after the loop completes.
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && is_sidecar_filename(&path) {
                let _ = std::fs::remove_file(&path);
            }
        }
    }
    drop(guard);
}

/// Wall-clock timeout for [`acquire_run_dir_flock`] before it gives
/// up and returns an error. 30 s is generous for the per-write
/// critical section: each peer writer holds the lock for at most
/// one (read_dir + bounded removes) + one (serialize + write)
/// cycle, all measured in milliseconds. A holder that does not
/// release within 30 s has stalled (a stuck filesystem, a panic
/// inside the locked section that somehow survived the RAII
/// drop, etc.) and surfacing that as an actionable error beats
/// hanging the test run indefinitely. The timeout is asymmetric
/// with the cache-store 60 s timeout because cache-store waits
/// for tens of test runs to drain whereas this lock waits for
/// at most one peer write.
const RUN_DIR_LOCK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Compute the per-run-key flock sentinel path for `dir`.
///
/// Layout: `{dir.parent()}/.locks/{dir.file_name()}.lock`. When
/// `dir = {runs_root}/{key}` (the production default-path shape),
/// this resolves to `{runs_root}/.locks/{key}.lock`. Sourced from
/// [`crate::flock::LOCK_DIR_NAME`] so a relocation of the lock
/// subdirectory updates one place across both this surface and
/// the cache module.
///
/// Returns `None` when `dir` has no parent (root) or no
/// `file_name` component (current dir, root) — neither case is
/// reachable on the production default path
/// ([`runs_root`] always returns a non-root multi-component
/// path), but the function is total over the input domain so a
/// future caller passing an unusual path surfaces a clean `None`
/// rather than panicking on `unwrap`.
///
/// Pure function over the input path — no I/O. The caller is
/// responsible for materializing the parent `.locks/`
/// subdirectory before opening the lockfile —
/// [`crate::flock::acquire_flock_with_timeout`] handles that
/// lazily.
fn run_dir_lock_path(dir: &std::path::Path) -> Option<PathBuf> {
    let parent = dir.parent()?;
    let leaf = dir.file_name()?;
    let mut filename = std::ffi::OsString::from(leaf);
    filename.push(".lock");
    Some(parent.join(crate::flock::LOCK_DIR_NAME).join(filename))
}

/// Acquire `LOCK_EX` on the per-run-key flock sentinel for `dir`.
/// Default-timeout wrapper over [`acquire_run_dir_flock_with_timeout`];
/// see that helper's doc for the full behavior contract. The
/// timeout split exists so tests can exercise the contention /
/// timeout path with a sub-second deadline rather than waiting
/// 30 s of real time per assertion.
fn acquire_run_dir_flock(dir: &std::path::Path) -> anyhow::Result<std::os::fd::OwnedFd> {
    acquire_run_dir_flock_with_timeout(dir, RUN_DIR_LOCK_TIMEOUT)
}

/// Test-parametrizable inner of [`acquire_run_dir_flock`].
///
/// Resolves the per-run-key lockfile path via [`run_dir_lock_path`]
/// then delegates to [`crate::flock::acquire_flock_with_timeout`],
/// which handles parent-directory creation, the poll loop, the
/// `tracing::debug!` contention log, and the formatted timeout
/// error. The `context` argument names the run directory and the
/// `remediation` argument supplies the operator-facing recovery
/// hint about peer cargo ktstr test processes that the shared
/// helper appends to the timeout error.
///
/// Returns `Err` on:
/// - `run_dir_lock_path(dir)` returning `None` (no parent / no
///   file_name — production default path always satisfies both,
///   so this is a defensive arm),
/// - any error from [`crate::flock::acquire_flock_with_timeout`]
///   (parent directory creation failure, `try_flock` error, or
///   wall-clock `timeout` elapsing).
///
/// Returns `Ok(OwnedFd)` on successful acquire. Caller drops the
/// fd to release the kernel-side flock; the OFD-bound semantics
/// of `flock(2)` mean no explicit unlock call is required —
/// `OwnedFd::drop` runs `close(2)` which releases the lock when
/// no other fd refers to the same OFD (the fresh `try_flock`
/// open guarantees uniqueness).
fn acquire_run_dir_flock_with_timeout(
    dir: &std::path::Path,
    timeout: std::time::Duration,
) -> anyhow::Result<std::os::fd::OwnedFd> {
    let lock_path = run_dir_lock_path(dir).ok_or_else(|| {
        anyhow::anyhow!(
            "cannot derive run-dir lock path from {} (no parent or no file_name component)",
            dir.display(),
        )
    })?;
    let context = format!("run-dir {}", dir.display());
    crate::flock::acquire_flock_with_timeout(
        &lock_path,
        crate::flock::FlockMode::Exclusive,
        timeout,
        &context,
        Some(
            "A peer cargo ktstr test process is writing sidecars to the \
             same {kernel}-{project_commit} directory; wait for it to \
             finish or kill it, then retry.",
        ),
    )
}

/// Return `active_flags` sorted into canonical
/// [`crate::scenario::flags::ALL`] order. Both sidecar writers
/// pipe their caller-supplied flag slice through this helper so
/// the persisted ordering is a pure function of the flag SET,
/// not the order the caller happened to accumulate them in.
///
/// Why this matters: [`sidecar_variant_hash`] walks
/// `active_flags` in-order and folds each byte into a SipHasher
/// state (see sibling site that hashes `for f in
/// &sidecar.active_flags`). Two runs of the same semantic variant
/// that differ only in flag accumulation order — e.g. a gauntlet
/// path that inserts `llc` then `steal` versus one that inserts
/// `steal` then `llc` — would otherwise produce distinct hashes,
/// distinct sidecar filenames, and end up as two separate rows in
/// `compare_partitions` even though they describe the same variant. By
/// canonicalizing at write time against the canonical
/// [`crate::scenario::flags::ALL`] positional ordering (shared
/// with `compute_flag_profiles` at scenario/mod.rs, which sorts
/// the same way), the on-disk representation is
/// order-insensitive by construction.
///
/// Flags not found in [`crate::scenario::flags::ALL`] are kept
/// and sorted to the end in lexical order. Sort key is composite:
/// positional for known flags (so the canonical ALL order leads),
/// then `&str` comparison as a tiebreaker. The lexical secondary
/// matters because two unknown flags both collide on the fallback
/// `usize::MAX` positional key — without the tiebreak, a caller
/// that supplies `["zzz_unknown", "aaa_unknown"]` versus the
/// reverse would share identical positional keys yet produce
/// different on-disk orderings under a stable sort, once again
/// breaking the "variant hash is a pure function of the flag
/// SET" invariant. The lexical secondary collapses them to one
/// canonical order so future or ad-hoc flag names are handled
/// without data loss AND without order sensitivity.
fn canonicalize_active_flags(flags: &[String]) -> Vec<String> {
    let mut v: Vec<String> = flags.to_vec();
    v.sort_by(|a, b| {
        let ka = crate::scenario::flags::ALL
            .iter()
            .position(|x| *x == a.as_str())
            .unwrap_or(usize::MAX);
        let kb = crate::scenario::flags::ALL
            .iter()
            .position(|x| *x == b.as_str())
            .unwrap_or(usize::MAX);
        ka.cmp(&kb).then_with(|| a.as_str().cmp(b.as_str()))
    });
    v
}

/// Emit a minimal sidecar for a PRE-VM-BOOT skip path.
///
/// Stats tooling enumerates sidecars to compute pass/skip/fail
/// rates; when a test bails before `run_ktstr_test_inner` reaches
/// the VM-run site that calls [`write_sidecar`], the skip is
/// invisible to post-run analysis — it shows up as a missing
/// result rather than a recorded skip.
///
/// This helper writes a sidecar flagged `skipped: true, passed: true`
/// with empty VM telemetry (no monitor, no stimulus events, no
/// verifier stats, no kvm stats, no payload metrics). Stats tooling
/// that subtracts skipped runs from the pass count treats the entry
/// correctly.
///
/// # Distinction from in-VM `AssertResult::skip` paths
///
/// There are TWO classes of skip, each with its own sidecar writer:
///
/// 1. **Pre-VM-boot skips** route through this helper
///    (`write_skip_sidecar`). Examples:
///    - `performance_mode` gated off via `KTSTR_NO_PERF_MODE`
///      (see `run_ktstr_test_inner`),
///    - `ResourceContention` at `builder.build()` or `vm.run()`
///      (topology-level unavailability — the VM never booted).
///
///    These paths write a MINIMAL sidecar: empty VM telemetry,
///    `work_type = "skipped"`, and `payload` pinned to the entry's
///    declared payload so stats can still attribute the skip to
///    the correct gauntlet variant. There is no VmResult to drain
///    because the VM didn't boot.
///
/// 2. **In-VM `AssertResult::skip` returns** — e.g. the
///    empty-cpuset skip in `scenario::run_scenario`
///    (`AssertResult::skip("not enough CPUs/LLCs")`), or the
///    `need >= 4 CPUs` checks in `scenario::dynamic::*` — route
///    through [`write_sidecar`] at `run_ktstr_test_inner`'s end.
///    The guest VM fully booted, ran through scenario setup,
///    discovered the topology couldn't accommodate the test, and
///    returned early. The resulting sidecar carries REAL VM
///    telemetry (monitor, kvm_stats, verifier_stats) alongside
///    `skipped: true` — not a blind spot, just a richer record
///    than what this helper emits.
///
/// The asymmetry is intentional: pre-VM-boot skips have no
/// telemetry to record, while in-VM skips do. Stats tooling that
/// wants to uniformly discount skipped runs filters on
/// [`SidecarResult::skipped == true`] regardless of which writer
/// produced the entry — both set the field identically.
///
/// Returns `Err` when the sidecar directory cannot be created, the
/// JSON cannot be serialized, or the file write fails. Callers that
/// ignore the Result accept the risk of stats-tooling blind spots on
/// this run.
pub(crate) fn write_skip_sidecar(
    entry: &KtstrTestEntry,
    active_flags: &[String],
) -> anyhow::Result<()> {
    let SchedulerFingerprint {
        scheduler,
        scheduler_commit,
        sysctls,
        kargs,
    } = scheduler_fingerprint(entry);
    let sidecar = SidecarResult {
        test_name: entry.name.to_string(),
        topology: entry.topology.to_string(),
        scheduler,
        scheduler_commit,
        project_commit: detect_project_commit(),
        // A skip never runs the payload. Still record the declared
        // payload name so stats tooling can attribute the skip to
        // the payload-gauntlet variant rather than losing the
        // association.
        payload: entry.payload.map(|p| p.name.to_string()),
        metrics: Vec::new(),
        passed: true,
        skipped: true,
        stats: Default::default(),
        monitor: None,
        stimulus_events: Vec::new(),
        // Skip paths never ran a workload; work_type is "skipped"
        // so stats tooling that groups by work_type puts these in a
        // distinguishable bucket.
        work_type: "skipped".to_string(),
        active_flags: canonicalize_active_flags(active_flags),
        verifier_stats: Vec::new(),
        kvm_stats: None,
        sysctls,
        kargs,
        kernel_version: detect_kernel_version(),
        kernel_commit: resolve_kernel_source_dir().and_then(|d| detect_kernel_commit(&d)),
        timestamp: now_iso8601(),
        run_id: generate_run_id(),
        host: Some(crate::host_context::collect_host_context()),
        // Skip paths never reach `collect_results`, so cleanup
        // duration is undefined. Emit `null` per the sidecar's
        // symmetric serialize/deserialize contract.
        cleanup_duration_ms: None,
        run_source: detect_run_source(),
    };
    serialize_and_write_sidecar(&sidecar, "skip sidecar")
}

/// Write a sidecar JSON file for post-run analysis.
///
/// Output goes to the current run's sidecar directory
/// (`KTSTR_SIDECAR_DIR` override, or
/// `{CARGO_TARGET_DIR or "target"}/ktstr/{kernel}-{project_commit}/`,
/// where `{project_commit}` is the project HEAD short hex with
/// `-dirty` when the worktree differs).
///
/// `payload_metrics` is the accumulated per-invocation output from
/// `ctx.payload(X).run()` / `.spawn().wait()` calls made in the
/// test body. Empty vec when the test body never called
/// `Ctx::payload` (scheduler-only tests, host-only probes).
///
/// Returns `Err` when the sidecar directory cannot be created, the
/// JSON cannot be serialized, or the file write fails. Callers that
/// ignore the Result accept the risk of stats-tooling blind spots on
/// this run.
pub(crate) fn write_sidecar(
    entry: &KtstrTestEntry,
    vm_result: &vmm::VmResult,
    stimulus_events: &[StimulusEvent],
    check_result: &AssertResult,
    work_type: &str,
    active_flags: &[String],
    payload_metrics: &[PayloadMetrics],
) -> anyhow::Result<()> {
    let SchedulerFingerprint {
        scheduler,
        scheduler_commit,
        sysctls,
        kargs,
    } = scheduler_fingerprint(entry);
    let sidecar = SidecarResult {
        test_name: entry.name.to_string(),
        topology: entry.topology.to_string(),
        scheduler,
        scheduler_commit,
        project_commit: detect_project_commit(),
        payload: entry.payload.map(|p| p.name.to_string()),
        metrics: payload_metrics.to_vec(),
        passed: check_result.passed,
        skipped: check_result.is_skipped(),
        stats: check_result.stats.clone(),
        monitor: vm_result.monitor.as_ref().map(|m| m.summary.clone()),
        stimulus_events: stimulus_events.to_vec(),
        work_type: work_type.to_string(),
        active_flags: canonicalize_active_flags(active_flags),
        verifier_stats: vm_result.verifier_stats.clone(),
        kvm_stats: vm_result.kvm_stats.clone(),
        sysctls,
        kargs,
        kernel_version: detect_kernel_version(),
        kernel_commit: resolve_kernel_source_dir().and_then(|d| detect_kernel_commit(&d)),
        timestamp: now_iso8601(),
        run_id: generate_run_id(),
        host: Some(crate::host_context::collect_host_context()),
        cleanup_duration_ms: vm_result.cleanup_duration.map(|d| d.as_millis() as u64),
        run_source: detect_run_source(),
    };
    serialize_and_write_sidecar(&sidecar, "sidecar")
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::{EnvVarGuard, lock_env};
    use super::*;
    use crate::assert::{AssertResult, CgroupStats};
    use crate::scenario::Ctx;
    use anyhow::Result;

    /// Collect every sidecar file in `dir` whose name starts with
    /// `prefix` and ends with `.ktstr.json`. Returns paths in
    /// filesystem iteration order; non-UTF-8 filenames are skipped.
    ///
    /// Call sites that write a single sidecar take the first match
    /// via `.into_iter().next().expect(..)` (the variant-hash suffix
    /// is opaque to the test so prefix match is how the file is
    /// recovered); tests that assert on the number of gauntlet
    /// variants use `.len()`.
    ///
    /// **Prefer this over hand-rolling read_dir/filter_map in new
    /// write_sidecar tests** — the 7 pre-existing call sites were
    /// near-identical inline blocks; funneling new tests through
    /// this helper keeps the lookup contract in one place.
    ///
    /// The `.ktstr.json` suffix filter is an intentional tightening
    /// relative to two of the original inline patterns
    /// (`write_sidecar_variant_hash_distinguishes_active_flags` and
    /// `_work_types`), which filtered only by prefix. The write-side
    /// tests only ever produce `.ktstr.json` files in their temp
    /// dirs, so the tightening is safe and rules out future stray
    /// files (a `.json.tmp` atomic-write residue, for instance) from
    /// inflating the count assertions.
    fn find_sidecars_by_prefix(dir: &std::path::Path, prefix: &str) -> Vec<std::path::PathBuf> {
        std::fs::read_dir(dir)
            .expect("sidecar dir must exist for lookup")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(prefix) && n.ends_with(".ktstr.json"))
            })
            .collect()
    }

    /// Single-file variant of [`find_sidecars_by_prefix`] for tests
    /// that exercise one variant per run. Asserts exactly one match
    /// and returns the owned path.
    ///
    /// What the length assertion catches: a test producing MORE than
    /// one sidecar under the given prefix — typically a stray
    /// leftover from a prior run (if the temp-dir cleanup is stale),
    /// or a call-site bug that invokes the writer twice. A
    /// variant-hash collision on its own would overwrite the file
    /// in place (same hash → same filename → single file), so this
    /// assertion is NOT a collision detector; it's a
    /// "one-call-one-file" invariant for single-variant tests.
    /// Centralizes the pattern so the 5 single-variant writer tests
    /// share one length check + error message.
    fn find_single_sidecar_by_prefix(dir: &std::path::Path, prefix: &str) -> std::path::PathBuf {
        let paths = find_sidecars_by_prefix(dir, prefix);
        assert_eq!(
            paths.len(),
            1,
            "single-variant test must produce exactly one sidecar under \
             prefix {prefix:?}; got {paths:?}",
        );
        paths
            .into_iter()
            .next()
            .expect("length-1 vec yields Some on first next()")
    }

    // -- find_sidecars_by_prefix self-tests --
    //
    // Pin the helper's filter behavior so changes to its logic
    // surface as failures here rather than as behavior shifts in
    // call sites.

    /// The `.ktstr.json` suffix filter must exclude files that share
    /// the prefix but carry a different extension. Without the
    /// suffix check, an atomic-write residue (`.json.tmp`) or a
    /// non-ktstr `.json` written into the same directory would
    /// inflate the match count.
    #[test]
    fn find_sidecars_by_prefix_filters_suffix() {
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let tmp = tmp_dir.path();
        std::fs::write(tmp.join("foo-0001.ktstr.json"), b"{}").unwrap();
        std::fs::write(tmp.join("foo-0002.ktstr.json.tmp"), b"{}").unwrap();
        std::fs::write(tmp.join("foo-0003.json"), b"{}").unwrap();
        std::fs::write(tmp.join("foo-0004.ktstr.txt"), b"{}").unwrap();
        let paths = find_sidecars_by_prefix(tmp, "foo-");
        assert_eq!(
            paths.len(),
            1,
            "only the .ktstr.json file must match, got {paths:?}",
        );
    }

    /// The prefix filter must reject filenames whose prefix does
    /// not match, so the count-based gauntlet-variant tests
    /// (`write_sidecar_variant_hash_distinguishes_*`) can coexist
    /// safely with sidecars from unrelated tests that happen to
    /// share a parent directory.
    #[test]
    fn find_sidecars_by_prefix_filters_prefix() {
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let tmp = tmp_dir.path();
        std::fs::write(tmp.join("foo-0001.ktstr.json"), b"{}").unwrap();
        std::fs::write(tmp.join("bar-0002.ktstr.json"), b"{}").unwrap();
        std::fs::write(tmp.join("foobar-0003.ktstr.json"), b"{}").unwrap();
        let paths = find_sidecars_by_prefix(tmp, "foo-");
        assert_eq!(
            paths.len(),
            1,
            "only files starting with 'foo-' must match (not 'foobar-'), got {paths:?}",
        );
    }

    /// A directory that contains nothing matching the `prefix` +
    /// `.ktstr.json` contract must yield an empty `Vec`, not panic.
    /// Call sites that use `.into_iter().next().expect(..)` rely on
    /// this — an empty Vec lets them surface a descriptive "sidecar
    /// file ... should be written" error rather than an opaque
    /// helper-internal panic.
    #[test]
    fn find_sidecars_by_prefix_empty_when_no_match() {
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let tmp = tmp_dir.path();
        std::fs::write(tmp.join("bar-0001.ktstr.json"), b"{}").unwrap();
        let paths = find_sidecars_by_prefix(tmp, "foo-");
        assert!(
            paths.is_empty(),
            "no prefix match must yield empty Vec, got {paths:?}",
        );
    }

    // -- test_fixture self-tests --
    //
    // Guard the fixture's observable shape so call-site tests can rely
    // on these defaults without re-asserting them.

    /// Serializing the fixture and parsing the result back must
    /// succeed — proves every field is serde-compatible and no default
    /// produces a value that fails to round-trip (e.g. a NaN float or
    /// an invalid Option combination).
    #[test]
    fn test_fixture_round_trips_clean() {
        let sc = SidecarResult::test_fixture();
        let json = serde_json::to_string(&sc).expect("fixture must serialize");
        let _loaded: SidecarResult =
            serde_json::from_str(&json).expect("fixture JSON must parse back");
    }

    /// `passed=true, skipped=false` is the fixture's verdict default
    /// so tests that only care about the success path don't need to
    /// spell either field out. A silent flip of either bit would
    /// invert the meaning of every unmodified call-site test.
    #[test]
    fn test_fixture_is_pass_not_skip() {
        let sc = SidecarResult::test_fixture();
        assert!(sc.passed, "fixture must default to passed=true");
        assert!(!sc.skipped, "fixture must default to skipped=false");
    }

    /// `host=None` is the fixture's host default so
    /// [`sidecar_variant_hash_excludes_host_context`] and every test
    /// that asserts the JSON does not carry a host key can rely on
    /// the default rather than spelling it out. Production writers
    /// populate host explicitly (see `write_sidecar` /
    /// `write_skip_sidecar`).
    #[test]
    fn test_fixture_host_is_none() {
        let sc = SidecarResult::test_fixture();
        assert!(sc.host.is_none(), "fixture must default to host=None");
    }

    /// `payload=None, metrics=empty` is the fixture's default so
    /// tests that verify the serde always-emit contract
    /// (e.g. [`sidecar_payload_and_metrics_always_emit_when_empty`])
    /// can rely on these defaults rather than re-spelling them.
    #[test]
    fn test_fixture_payload_and_metrics_empty() {
        let sc = SidecarResult::test_fixture();
        assert!(sc.payload.is_none(), "fixture must default to payload=None");
        assert!(
            sc.metrics.is_empty(),
            "fixture must default to metrics=empty"
        );
    }

    /// Summary guard on every empty-collection / None-Option /
    /// empty-String default. A silent flip of any of these defaults
    /// breaks every test that depends on "unset → serialized as
    /// null / []" via the symmetric always-emit contract — and
    /// there are many such tests across this file. One tripwire
    /// here catches the flip in one place rather than fanning out
    /// to per-default pins.
    ///
    /// Hash-participating string defaults (`test_name`,
    /// `topology`, `scheduler`, `work_type`) are intentionally NOT
    /// re-asserted here — their drift is caught by
    /// `test_fixture_variant_hash_is_stable` which pins the hash.
    #[test]
    fn test_fixture_all_collections_empty_by_default() {
        let sc = SidecarResult::test_fixture();
        assert!(sc.metrics.is_empty(), "metrics must default empty");
        assert!(
            sc.active_flags.is_empty(),
            "active_flags must default empty"
        );
        assert!(
            sc.stimulus_events.is_empty(),
            "stimulus_events must default empty"
        );
        assert!(
            sc.verifier_stats.is_empty(),
            "verifier_stats must default empty"
        );
        assert!(sc.sysctls.is_empty(), "sysctls must default empty");
        assert!(sc.kargs.is_empty(), "kargs must default empty");
        assert!(sc.payload.is_none(), "payload must default None");
        assert!(sc.monitor.is_none(), "monitor must default None");
        assert!(sc.kvm_stats.is_none(), "kvm_stats must default None");
        assert!(
            sc.kernel_version.is_none(),
            "kernel_version must default None"
        );
        assert!(
            sc.kernel_commit.is_none(),
            "kernel_commit must default None"
        );
        assert!(sc.host.is_none(), "host must default None");
        assert!(
            sc.timestamp.is_empty(),
            "timestamp must default empty String"
        );
        assert!(sc.run_id.is_empty(), "run_id must default empty String");
        assert!(
            sc.stats.cgroups.is_empty(),
            "stats.cgroups must default empty (ScenarioStats::default)",
        );
        // Overlaps deliberately with `test_fixture_is_pass_not_skip`
        // so this single summary test is sufficient to catch a
        // verdict-default flip even if callers forget the other
        // self-test exists. Cheap belt + suspenders.
        assert!(sc.passed, "passed must default true");
        assert!(!sc.skipped, "skipped must default false");
    }

    /// Two fresh fixtures must hash to the same value and that value
    /// must match the pinned constant. Protects against a change to
    /// fixture defaults that would silently shift every call-site
    /// test that passes the fixture straight into
    /// [`sidecar_variant_hash`] (e.g. `sidecar_variant_hash_distinguishes_payload`'s
    /// `none` handle). If this constant needs to move, every such
    /// call site must be re-read to confirm the shift is intentional.
    #[test]
    fn test_fixture_variant_hash_is_stable() {
        let a = sidecar_variant_hash(&SidecarResult::test_fixture());
        let b = sidecar_variant_hash(&SidecarResult::test_fixture());
        assert_eq!(a, b, "two fresh fixtures must hash identically");
        assert_eq!(
            a, 0x55f6b9881e152f8c,
            "fixture hash drifted — update only if the fixture default \
             change is intentional; verify every call site that passes \
             the fixture straight into sidecar_variant_hash still expresses \
             the intent it had before",
        );
    }

    /// Full literal intentional: exercises every field through serde so
    /// a future addition is caught by a compile error here.
    #[test]
    fn sidecar_result_roundtrip() {
        let sc = SidecarResult {
            test_name: "my_test".to_string(),
            topology: "1n2l4c2t".to_string(),
            scheduler: "scx_mitosis".to_string(),
            scheduler_commit: Some("abc123".to_string()),
            project_commit: Some("def4567".to_string()),
            payload: None,
            metrics: vec![],
            passed: true,
            skipped: false,
            stats: crate::assert::ScenarioStats {
                cgroups: vec![CgroupStats {
                    num_workers: 4,
                    num_cpus: 2,
                    avg_off_cpu_pct: 50.0,
                    min_off_cpu_pct: 40.0,
                    max_off_cpu_pct: 60.0,
                    spread: 20.0,
                    max_gap_ms: 100,
                    max_gap_cpu: 1,
                    total_migrations: 5,
                    ..Default::default()
                }],
                total_workers: 4,
                total_cpus: 2,
                total_migrations: 5,
                worst_spread: 20.0,
                worst_gap_ms: 100,
                worst_gap_cpu: 1,
                ..Default::default()
            },
            monitor: Some(MonitorSummary {
                prog_stats_deltas: None,
                total_samples: 10,
                max_imbalance_ratio: 1.5,
                max_local_dsq_depth: 3,
                stall_detected: false,
                event_deltas: Some(crate::monitor::ScxEventDeltas {
                    total_fallback: 7,
                    fallback_rate: 0.5,
                    max_fallback_burst: 2,
                    total_dispatch_offline: 0,
                    total_dispatch_keep_last: 3,
                    keep_last_rate: 0.2,
                    total_enq_skip_exiting: 0,
                    total_enq_skip_migration_disabled: 0,
                    ..Default::default()
                }),
                schedstat_deltas: None,
                ..Default::default()
            }),
            stimulus_events: vec![crate::timeline::StimulusEvent {
                elapsed_ms: 500,
                label: "StepStart[0]".to_string(),
                op_kind: Some("SetCpuset".to_string()),
                detail: Some("4 cpus".to_string()),
                total_iterations: None,
            }],
            work_type: "CpuSpin".to_string(),
            active_flags: Vec::new(),
            verifier_stats: vec![],
            kvm_stats: None,
            sysctls: vec![],
            kargs: vec![],
            kernel_version: None,
            kernel_commit: Some("kabcde7".to_string()),
            timestamp: String::new(),
            run_id: String::new(),
            host: None,
            cleanup_duration_ms: Some(123),
            run_source: Some(SIDECAR_RUN_SOURCE_LOCAL.to_string()),
        };
        let json = serde_json::to_string_pretty(&sc).unwrap();
        let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
        // Exhaustive destructure — `SidecarResult` is `non_exhaustive`
        // only across crates, but in-crate destructure still requires
        // every field to appear by name. Adding a field to
        // `SidecarResult` without extending this pattern fails to
        // compile here, forcing the author to make an explicit
        // roundtrip-coverage decision at the same time they introduce
        // the field. See sibling
        // [`sidecar_payload_and_metrics_always_emit_when_empty`] for
        // the empty-collection variant of this pin.
        let SidecarResult {
            test_name,
            topology,
            scheduler,
            scheduler_commit,
            project_commit,
            payload,
            metrics,
            passed,
            skipped,
            stats,
            monitor,
            stimulus_events,
            work_type,
            active_flags,
            verifier_stats,
            kvm_stats,
            sysctls,
            kargs,
            kernel_version,
            kernel_commit,
            timestamp,
            run_id,
            host,
            cleanup_duration_ms,
            run_source,
        } = loaded;
        // Hash-participating string fields round-trip verbatim.
        assert_eq!(test_name, "my_test");
        assert_eq!(topology, "1n2l4c2t");
        assert_eq!(scheduler, "scx_mitosis");
        assert_eq!(work_type, "CpuSpin");
        // Nullable string metadata fields.
        assert_eq!(scheduler_commit.as_deref(), Some("abc123"));
        assert_eq!(project_commit.as_deref(), Some("def4567"));
        assert_eq!(
            kernel_commit.as_deref(),
            Some("kabcde7"),
            "kernel_commit must round-trip the literal string \
             populated on the write side, including the 7-char \
             hex shape `detect_kernel_commit` produces. The \
             fixture uses `kabcde7` (hex-only) to make accidental \
             field-swap regressions with project_commit / \
             scheduler_commit obvious — each commit field carries \
             a distinct token.",
        );
        assert_eq!(payload, None, "fixture declared no payload");
        assert_eq!(kvm_stats, None, "fixture declared no kvm_stats");
        assert_eq!(kernel_version, None, "fixture declared no kernel_version");
        assert_eq!(host, None, "fixture declared no host context");
        assert_eq!(timestamp, "", "fixture used empty-string timestamp");
        assert_eq!(run_id, "", "fixture used empty-string run_id");
        // Verdict bits — passed true + skipped false pinned.
        assert!(passed);
        assert!(!skipped, "fixture declared skipped=false");
        // Empty-Vec collections — regression guard against a serde
        // regression that dropped `[]` on round-trip.
        assert!(metrics.is_empty(), "fixture declared empty metrics");
        assert!(
            active_flags.is_empty(),
            "fixture declared empty active_flags",
        );
        assert!(
            verifier_stats.is_empty(),
            "fixture declared empty verifier_stats",
        );
        assert!(sysctls.is_empty(), "fixture declared empty sysctls");
        assert!(kargs.is_empty(), "fixture declared empty kargs");
        // Populated nested structs.
        assert_eq!(stats.total_workers, 4);
        assert_eq!(stats.cgroups.len(), 1);
        assert_eq!(stats.cgroups[0].num_workers, 4);
        assert_eq!(stats.worst_spread, 20.0);
        let mon = monitor.unwrap();
        assert_eq!(mon.total_samples, 10);
        assert_eq!(mon.max_imbalance_ratio, 1.5);
        assert_eq!(mon.max_local_dsq_depth, 3);
        assert!(!mon.stall_detected);
        let deltas = mon.event_deltas.unwrap();
        assert_eq!(deltas.total_fallback, 7);
        assert_eq!(deltas.total_dispatch_keep_last, 3);
        assert_eq!(stimulus_events.len(), 1);
        assert_eq!(stimulus_events[0].label, "StepStart[0]");
        assert_eq!(
            cleanup_duration_ms,
            Some(123),
            "cleanup_duration_ms round-tripped",
        );
        assert_eq!(
            run_source.as_deref(),
            Some(SIDECAR_RUN_SOURCE_LOCAL),
            "run_source must round-trip the literal `local` populated on \
             the write side, including the absent-vs-populated distinction",
        );
    }

    /// Exhaustive schema-audit gate for `SidecarResult`'s serde
    /// round-trip. Every field is populated with a value that is
    /// distinct from the `test_fixture` default AND every field is
    /// asserted individually after serialization + deserialization.
    /// A new field added to `SidecarResult` triggers failure at two
    /// independent sites for `SidecarResult` top-level fields; nested
    /// structs use `..Default::default()` and rely on their own
    /// per-type tests:
    /// 1. The construction literal below fails to compile (Rust
    ///    requires every field in a struct literal without
    ///    `..Default::default()`).
    /// 2. The per-field assertion block below misses the new field,
    ///    so the audit surfaces as a reviewer note.
    ///
    /// Nested struct literals inside the construction (e.g.
    /// `MonitorSummary`, `ScenarioStats`, `HostContext`,
    /// `PayloadMetrics`) use `..Default::default()` to remain
    /// resilient to unrelated nested-type growth — adding a field
    /// to one of those nested types does NOT trip this test. Fields
    /// of those nested types that should trigger a similar audit
    /// must grow their own all-fields round-trip test in their
    /// owning module (e.g.
    /// `host_context_populated_round_trips_via_json` for
    /// `HostContext`).
    ///
    /// Complements the structurally-populated
    /// [`sidecar_result_roundtrip`] which exercises nested-struct
    /// shapes but only asserts on a subset of fields. Leaving both
    /// is intentional: the structural test proves deep trees survive
    /// serde; this test proves every scalar and Option round-trips.
    ///
    /// Distinct non-default values used:
    /// - `test_name="audit"` (vs fixture `"t"`).
    /// - `topology="8n8l16c2t"` (vs fixture `"1n1l1c1t"`).
    /// - `scheduler="scx_audit"` (vs fixture `"eevdf"`).
    /// - `work_type="AuditWork"` (vs fixture `"CpuSpin"`).
    /// - `passed=false, skipped=true` (vs fixture `true`, `false`).
    /// - Non-empty collections for every `Vec<_>` field.
    /// - `Some(…)` for every `Option<_>` field.
    /// - Non-empty Strings for `timestamp`, `run_id`.
    #[test]
    fn sidecar_result_roundtrip_all_fields_round_trip() {
        use crate::assert::{CgroupStats, ScenarioStats};
        use crate::host_context::HostContext;
        use crate::monitor::MonitorSummary;
        use crate::monitor::bpf_prog::ProgVerifierStats;
        use crate::test_support::{Metric, MetricSource, MetricStream, PayloadMetrics, Polarity};
        use crate::timeline::StimulusEvent;

        let sc = SidecarResult {
            test_name: "audit".to_string(),
            topology: "8n8l16c2t".to_string(),
            scheduler: "scx_audit".to_string(),
            scheduler_commit: Some("deadbeef1234567890abcdef".to_string()),
            project_commit: Some("cafebab-dirty".to_string()),
            payload: Some("audit_payload".to_string()),
            metrics: vec![PayloadMetrics {
                payload_index: 0,
                metrics: vec![Metric {
                    name: "audit_metric".to_string(),
                    value: 42.0,
                    polarity: Polarity::HigherBetter,
                    unit: "audits".to_string(),
                    source: MetricSource::Json,
                    stream: MetricStream::Stdout,
                }],
                exit_code: 7,
            }],
            passed: false,
            skipped: true,
            stats: ScenarioStats {
                cgroups: vec![CgroupStats {
                    num_workers: 3,
                    ..Default::default()
                }],
                total_workers: 3,
                ..Default::default()
            },
            monitor: Some(MonitorSummary {
                total_samples: 17,
                ..Default::default()
            }),
            stimulus_events: vec![StimulusEvent {
                elapsed_ms: 123,
                label: "audit_event".to_string(),
                op_kind: None,
                detail: None,
                total_iterations: None,
            }],
            work_type: "AuditWork".to_string(),
            active_flags: vec!["flag_a".to_string(), "flag_b".to_string()],
            verifier_stats: vec![ProgVerifierStats {
                name: "audit_prog".to_string(),
                verified_insns: 999,
            }],
            kvm_stats: Some(crate::vmm::KvmStatsTotals::default()),
            sysctls: vec!["sysctl.kernel.audit_sysctl=1".to_string()],
            kargs: vec!["audit_karg".to_string()],
            kernel_version: Some("6.99.0".to_string()),
            kernel_commit: Some("kabcde7-dirty".to_string()),
            timestamp: "audit-timestamp".to_string(),
            run_id: "audit-run-id".to_string(),
            host: Some(HostContext {
                kernel_name: Some("AuditLinux".to_string()),
                ..Default::default()
            }),
            cleanup_duration_ms: Some(987),
            run_source: Some(SIDECAR_RUN_SOURCE_CI.to_string()),
        };

        let json = serde_json::to_string(&sc).expect("serialize");
        let loaded: SidecarResult = serde_json::from_str(&json).expect("deserialize");

        // Every field asserted, in struct-declaration order.
        assert_eq!(loaded.test_name, "audit");
        assert_eq!(loaded.topology, "8n8l16c2t");
        assert_eq!(loaded.scheduler, "scx_audit");
        assert_eq!(
            loaded.scheduler_commit.as_deref(),
            Some("deadbeef1234567890abcdef"),
            "scheduler_commit must round-trip the literal string \
             populated on the write side — not collapse to None via \
             a missing serde attribute or default fallback",
        );
        assert_eq!(
            loaded.project_commit.as_deref(),
            Some("cafebab-dirty"),
            "project_commit must round-trip the literal string \
             populated on the write side, including the `-dirty` \
             suffix that `detect_project_commit` appends — a \
             regression that stripped the suffix or substituted \
             None for a populated value would surface here. \
             Fixture uses 7-char hex (`cafebab`) to match the \
             `oid::to_hex_with_len(7)` shape `detect_project_commit` \
             produces in production.",
        );
        assert_eq!(loaded.payload.as_deref(), Some("audit_payload"));
        assert_eq!(loaded.metrics.len(), 1);
        assert_eq!(loaded.metrics[0].exit_code, 7);
        assert_eq!(loaded.metrics[0].metrics.len(), 1);
        assert_eq!(loaded.metrics[0].metrics[0].name, "audit_metric");
        assert_eq!(loaded.metrics[0].metrics[0].value, 42.0);
        assert!(!loaded.passed, "passed must survive as false");
        assert!(loaded.skipped, "skipped must survive as true");
        assert_eq!(loaded.stats.total_workers, 3);
        assert_eq!(loaded.stats.cgroups.len(), 1);
        assert_eq!(loaded.stats.cgroups[0].num_workers, 3);
        let mon = loaded.monitor.expect("monitor round-trips");
        assert_eq!(mon.total_samples, 17);
        assert_eq!(loaded.stimulus_events.len(), 1);
        assert_eq!(loaded.stimulus_events[0].label, "audit_event");
        assert_eq!(loaded.stimulus_events[0].elapsed_ms, 123);
        assert_eq!(loaded.work_type, "AuditWork");
        assert_eq!(loaded.active_flags, vec!["flag_a", "flag_b"]);
        assert_eq!(loaded.verifier_stats.len(), 1);
        assert_eq!(loaded.verifier_stats[0].name, "audit_prog");
        assert_eq!(loaded.verifier_stats[0].verified_insns, 999);
        assert!(
            loaded.kvm_stats.is_some(),
            "kvm_stats must round-trip as Some"
        );
        assert_eq!(loaded.sysctls, vec!["sysctl.kernel.audit_sysctl=1"]);
        assert_eq!(loaded.kargs, vec!["audit_karg"]);
        assert_eq!(loaded.kernel_version.as_deref(), Some("6.99.0"));
        assert_eq!(
            loaded.kernel_commit.as_deref(),
            Some("kabcde7-dirty"),
            "kernel_commit must round-trip the literal string \
             populated on the write side, including the `-dirty` \
             suffix that `detect_kernel_commit` appends. Fixture \
             uses 7-char hex (`kabcde7`) to match the \
             `oid::to_hex_with_len(7)` shape `detect_kernel_commit` \
             produces in production. The leading `k` in the fixture \
             token makes a project_commit / kernel_commit field-swap \
             regression visible — each commit field carries a \
             distinct token in the audit fixture.",
        );
        assert_eq!(loaded.timestamp, "audit-timestamp");
        assert_eq!(loaded.run_id, "audit-run-id");
        let host = loaded.host.expect("host round-trips");
        assert_eq!(host.kernel_name.as_deref(), Some("AuditLinux"));
        assert_eq!(loaded.cleanup_duration_ms, Some(987));
        assert_eq!(
            loaded.run_source.as_deref(),
            Some(SIDECAR_RUN_SOURCE_CI),
            "run_source must round-trip the literal `ci` populated on \
             the write side. Audit fixture uses `ci` (vs `local` in \
             the sibling roundtrip) so a write-vs-read field-swap \
             regression that mapped one tag onto another would \
             surface in this audit pass even if the sibling test \
             did not detect it.",
        );
    }

    #[test]
    fn sidecar_result_roundtrip_no_monitor() {
        let sc = SidecarResult {
            test_name: "eevdf_test".to_string(),
            topology: "1n1l2c1t".to_string(),
            passed: false,
            ..SidecarResult::test_fixture()
        };
        let json = serde_json::to_string(&sc).unwrap();
        let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.test_name, "eevdf_test");
        assert!(!loaded.passed);
        assert!(loaded.monitor.is_none());
        assert!(loaded.stimulus_events.is_empty());
        // `monitor` is emitted as `"monitor":null` when absent — the
        // writer side guarantees full symmetry by always emitting
        // every field. (The reader side tolerates absence on `Option`
        // fields per serde's native rule; non-`Option` fields remain
        // hard-required.) Pinning the emission pattern prevents a
        // drift back to the old asymmetric `skip_serializing_if` form
        // that omitted None-produced fields entirely.
        assert!(
            json.contains("\"monitor\":null"),
            "monitor=None must serialize as `\"monitor\":null`, not be omitted: {json}",
        );
    }

    /// Strict-schema rejection for non-`Option` fields: a sidecar
    /// JSON that omits any required (non-`Option`) top-level field
    /// must fail deserialization, not silently default to the empty
    /// string / empty Vec / similar. The SidecarResult policy —
    /// `serde(default)` removed crate-wide, no `skip_serializing_if`
    /// — is stated in the module doc; this test pins the parser-side
    /// half by construction. A regression that reintroduces
    /// `#[serde(default)]` on any non-`Option` SidecarResult field
    /// would cause the `from_str` calls below to succeed instead of
    /// error.
    ///
    /// `Option` fields are deliberately excluded: serde's native
    /// `Option<T>` deserialize rule treats absence as `None`, and
    /// that tolerance is part of the asymmetric contract documented
    /// at the module level — writer always emits, reader tolerates
    /// absence on `Option`s. The sibling
    /// `serialize_always_emits_option_keys` tests pin the writer
    /// side; this loop pins the reader side for non-`Option` fields
    /// only.
    #[test]
    fn sidecar_result_missing_required_field_rejected_by_deserialize() {
        // Table-driven expansion covering every non-`Option` field of
        // `SidecarResult`. Each must fail deserialize when absent with
        // a missing-field error naming the removed key.
        //
        // **Why Option fields are excluded**: serde treats
        // `Option<T>` as tolerant-of-absence natively (no explicit
        // `#[serde(default)]` needed — it's a builtin rule), so
        // removing e.g. `payload: Option<String>` from the JSON
        // yields `None` on the parsed struct rather than a rejection.
        // The module doc at src/test_support/sidecar.rs promises
        // "required on deserialize" for Option fields, but that's
        // enforced at the writer (always-emitted) side, not the
        // parser side. The `serialize_always_emits_option_keys`
        // sibling tests pin the writer half; this test pins the
        // parser-side strictness for every non-Option field.
        //
        // Old single-field-sentinel form (checking only `test_name`)
        // would pass silently if e.g. a regression added
        // `#[serde(default)]` to `run_id` alone — this loop catches
        // that class of softening across every non-Option field.
        const REQUIRED_NON_OPTION_FIELDS: &[&str] = &[
            "test_name",
            "topology",
            "scheduler",
            "metrics",
            "passed",
            "skipped",
            "stats",
            "stimulus_events",
            "work_type",
            "active_flags",
            "verifier_stats",
            "sysctls",
            "kargs",
            "timestamp",
            "run_id",
        ];

        let fixture = SidecarResult::test_fixture();
        let full = match serde_json::to_value(&fixture).unwrap() {
            serde_json::Value::Object(m) => m,
            other => panic!("expected object, got {other:?}"),
        };

        for field in REQUIRED_NON_OPTION_FIELDS {
            let mut obj = full.clone();
            assert!(
                obj.remove(*field).is_some(),
                "SidecarResult test fixture must emit `{field}` for its \
                 rejection case to be meaningful — the required-fields \
                 list has drifted from the struct definition",
            );
            let json = serde_json::Value::Object(obj).to_string();
            let err = serde_json::from_str::<SidecarResult>(&json)
                .err()
                .unwrap_or_else(|| {
                    panic!(
                        "deserialize must reject SidecarResult with `{field}` removed, \
                     but succeeded — a regression may have added \
                     `#[serde(default)]` to this field",
                    )
                });
            let msg = format!("{err}");
            assert!(
                msg.contains(field),
                "missing-field error for `{field}` must name the field; got: {msg}",
            );
        }
    }

    /// Rename contract pin for the `source` → `run_source`
    /// schema change. Per the doc on
    /// [`SidecarResult::run_source`], no `#[serde(alias =
    /// "source")]` is in place, so an archived sidecar carrying
    /// the old `"source": "ci"` key deserializes to
    /// `run_source: None` (serde silently drops the unknown
    /// `"source"` field, then `Option<T>`'s "tolerate absence"
    /// rule fires for the missing `"run_source"` key).
    ///
    /// This is the documented data-loss behavior — pre-1.0
    /// disposable schema, re-running the test regenerates the
    /// sidecar under the new key. The test pins:
    ///
    /// 1. Old key (`"source": "ci"`) → `run_source: None` (the
    ///    payload IS dropped, not preserved). A regression that
    ///    added `#[serde(alias = "source")]` would surface here
    ///    as `Some("ci")`.
    /// 2. New key (`"run_source": "ci"`) → `Some("ci")` (the
    ///    canonical deserialize path under the post-rename
    ///    schema). A regression that broke the new-key path
    ///    would surface here as `None`.
    /// 3. Old key + new key both present → new key wins (sanity
    ///    check that the rename did not silently route the new
    ///    key through the old field's deserialize logic). Pins
    ///    the post-rename canonical-key precedence.
    #[test]
    fn sidecar_result_rename_contract_old_source_key_lands_run_source_none() {
        let fixture = SidecarResult::test_fixture();
        let full = match serde_json::to_value(&fixture).unwrap() {
            serde_json::Value::Object(m) => m,
            other => panic!("expected object, got {other:?}"),
        };

        // Arm 1: old `"source"` key only — the new schema has
        // no alias, so this is the documented data-loss path.
        let mut obj_old = full.clone();
        obj_old.remove("run_source");
        obj_old.insert(
            "source".to_string(),
            serde_json::Value::String("ci".to_string()),
        );
        let json_old = serde_json::Value::Object(obj_old).to_string();
        let parsed_old: SidecarResult = serde_json::from_str(&json_old).expect(
            "old-key sidecar must still deserialize — \
             SidecarResult does not set deny_unknown_fields, \
             so the unrecognised `\"source\"` key is silently dropped",
        );
        assert_eq!(
            parsed_old.run_source, None,
            "old `\"source\": \"ci\"` key must land run_source = None \
             per the documented data-loss contract; a regression that \
             added `#[serde(alias = \"source\")]` would yield Some(\"ci\") here",
        );

        // Arm 2: new `"run_source"` key only — the canonical
        // post-rename deserialize path.
        let mut obj_new = full.clone();
        obj_new.insert(
            "run_source".to_string(),
            serde_json::Value::String("ci".to_string()),
        );
        let json_new = serde_json::Value::Object(obj_new).to_string();
        let parsed_new: SidecarResult =
            serde_json::from_str(&json_new).expect("new-key sidecar must deserialize cleanly");
        assert_eq!(
            parsed_new.run_source.as_deref(),
            Some("ci"),
            "new `\"run_source\": \"ci\"` key must populate \
             run_source — a regression breaking the new-key path \
             would yield None here",
        );

        // Arm 3: BOTH keys present — the new key wins because
        // the old `"source"` is unknown and silently dropped.
        // Pins that the rename did not accidentally route the
        // new key through the old field's logic (which would
        // make this case ambiguous).
        let mut obj_both = full.clone();
        obj_both.insert(
            "run_source".to_string(),
            serde_json::Value::String("ci".to_string()),
        );
        obj_both.insert(
            "source".to_string(),
            serde_json::Value::String("local".to_string()),
        );
        let json_both = serde_json::Value::Object(obj_both).to_string();
        let parsed_both: SidecarResult =
            serde_json::from_str(&json_both).expect("both-keys sidecar must deserialize cleanly");
        assert_eq!(
            parsed_both.run_source.as_deref(),
            Some("ci"),
            "with both keys present, new `\"run_source\"` must win \
             — the old `\"source\"` is silently dropped, NOT used \
             as a fallback. A regression that processed `\"source\"` \
             as an alias would surface here as Some(\"local\")",
        );
    }

    // -- collect_sidecars tests --

    #[test]
    fn collect_sidecars_empty_dir() {
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let results = collect_sidecars(tmp_dir.path());
        assert!(results.is_empty());
    }

    #[test]
    fn collect_sidecars_nonexistent_dir() {
        let results = collect_sidecars(std::path::Path::new("/nonexistent/path"));
        assert!(results.is_empty());
    }

    #[test]
    fn collect_sidecars_reads_json() {
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let tmp = tmp_dir.path();
        let sc = SidecarResult {
            test_name: "test_x".to_string(),
            topology: "1n1l2c1t".to_string(),
            ..SidecarResult::test_fixture()
        };
        let json = serde_json::to_string(&sc).unwrap();
        std::fs::write(tmp.join("test_x.ktstr.json"), &json).unwrap();
        // Non-ktstr JSON should be ignored.
        std::fs::write(tmp.join("other.json"), r#"{"key":"val"}"#).unwrap();
        let results = collect_sidecars(tmp);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].test_name, "test_x");
    }

    #[test]
    fn collect_sidecars_recurses_one_level() {
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let tmp = tmp_dir.path();
        let sub = tmp.join("job-0");
        std::fs::create_dir_all(&sub).unwrap();
        let sc = SidecarResult {
            test_name: "nested_test".to_string(),
            topology: "1n2l4c2t".to_string(),
            scheduler: "scx_mitosis".to_string(),
            passed: false,
            ..SidecarResult::test_fixture()
        };
        let json = serde_json::to_string(&sc).unwrap();
        std::fs::write(sub.join("nested_test.ktstr.json"), &json).unwrap();
        let results = collect_sidecars(tmp);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].test_name, "nested_test");
        assert!(!results[0].passed);
    }

    #[test]
    fn collect_sidecars_does_not_recurse_past_one_level() {
        // Companion to `collect_sidecars_recurses_one_level`: pin the
        // "exactly one level, no deeper" contract. A sidecar two
        // directories deep must be ignored. If a future change
        // switches collect_sidecars to a depth-unbounded walk, this
        // test catches the schema-scope regression before stats
        // tooling starts double-counting results from unrelated
        // sub-runs under the same `runs_root`.
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let tmp = tmp_dir.path();
        let top_sub = tmp.join("job-0");
        let deep_sub = top_sub.join("replay-0");
        std::fs::create_dir_all(&deep_sub).unwrap();

        let sc = |name: &str| SidecarResult {
            test_name: name.to_string(),
            ..SidecarResult::test_fixture()
        };
        // One level: should be collected.
        std::fs::write(
            top_sub.join("top_level.ktstr.json"),
            serde_json::to_string(&sc("top_level")).unwrap(),
        )
        .unwrap();
        // Two levels: must NOT be collected.
        std::fs::write(
            deep_sub.join("deep_level.ktstr.json"),
            serde_json::to_string(&sc("deep_level")).unwrap(),
        )
        .unwrap();

        let results = collect_sidecars(tmp);
        let names: Vec<&str> = results.iter().map(|r| r.test_name.as_str()).collect();
        assert_eq!(
            names,
            vec!["top_level"],
            "collect_sidecars must see only the one-level-deep sidecar, not the two-level one"
        );
    }

    #[test]
    fn collect_sidecars_skips_invalid_json() {
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let tmp = tmp_dir.path();
        std::fs::write(tmp.join("bad.ktstr.json"), "not json").unwrap();
        let results = collect_sidecars(tmp);
        assert!(results.is_empty());
    }

    #[test]
    fn collect_sidecars_skips_non_ktstr_json() {
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let tmp = tmp_dir.path();
        // File ends in .json but does NOT contain ".ktstr." in the name
        std::fs::write(tmp.join("other.json"), r#"{"test":"val"}"#).unwrap();
        let results = collect_sidecars(tmp);
        assert!(results.is_empty());
    }

    #[test]
    fn sidecar_result_work_type_field() {
        let sc = SidecarResult {
            work_type: "Bursty".to_string(),
            ..SidecarResult::test_fixture()
        };
        let json = serde_json::to_string(&sc).unwrap();
        let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.work_type, "Bursty");
    }

    #[test]
    fn write_sidecar_defaults_to_target_dir_without_env() {
        let _lock = lock_env();
        let target_dir = tempfile::TempDir::new().unwrap();
        let _env_target = EnvVarGuard::set("CARGO_TARGET_DIR", target_dir.path());
        let _env_sidecar = EnvVarGuard::remove("KTSTR_SIDECAR_DIR");
        let _env_kernel = EnvVarGuard::remove("KTSTR_KERNEL");

        let dir = sidecar_dir();
        // Expected layout: `{CARGO_TARGET_DIR}/ktstr/{kernel}-{project_commit}`.
        // `KTSTR_KERNEL` is unset so kernel resolves to `"unknown"`.
        // `{project_commit}` is whatever `detect_project_commit()`
        // resolves on this machine (`Some(hex7)` when cwd is inside
        // a git repo, `None` -> `"unknown"` otherwise). Compute the
        // expected via `runs_root` + `format_run_dirname` so the
        // assertion matches the production path symmetrically and
        // does not depend on the cwd's git state.
        let kernel = detect_kernel_version();
        let commit = detect_project_commit();
        let expected = runs_root().join(format_run_dirname(kernel.as_deref(), commit.as_deref()));
        assert_eq!(dir, expected);

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "__sidecar_default_dir__",
            func: dummy,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        };
        let vm_result = crate::vmm::VmResult::test_fixture();
        let check_result = AssertResult::pass();
        write_sidecar(&entry, &vm_result, &[], &check_result, "CpuSpin", &[], &[]).unwrap();

        // The actual on-disk filename embeds a variant-hash suffix
        // (see `serialize_and_write_sidecar`), so a fixed
        // `test_name + ".ktstr.json"` path never matches — use the
        // prefix-scan helper the sibling tests use. The tempdir's
        // Drop wipes everything when this scope ends, so no manual
        // cleanup is required.
        let paths = find_sidecars_by_prefix(&dir, "__sidecar_default_dir__-");
        // One call to `write_sidecar` above must produce exactly
        // one sidecar under this test's unique prefix. A count
        // above 1 exposes either a variant-hash collision (two
        // distinct test_name + variant-hash pairs hashing to the
        // same filename suffix) or a regression in
        // `pre_clear_run_dir_once` (which is now keyed per-directory
        // via `Mutex<HashSet<PathBuf>>` — every distinct dir gets
        // exactly one pre-clear per process — so a stale file from
        // a prior crashed run should be wiped on the very first
        // call into this dir, regardless of which test runs first).
        assert_eq!(
            paths.len(),
            1,
            "single `write_sidecar` call against prefix \
             `__sidecar_default_dir__-` must produce exactly one \
             file; got {} ({paths:?}). If >1, either the variant \
             hash collided for this test's variant-field tuple or \
             `pre_clear_run_dir_once`'s per-directory keying failed \
             to wipe a stale sidecar from a prior crashed run.",
            paths.len(),
        );
    }

    // -- KTSTR_SIDECAR_DIR override: empty-string falls back to default --

    /// `KTSTR_SIDECAR_DIR=""` (defensively-cleared empty string)
    /// must NOT activate the override branch — `sidecar_dir`
    /// must compute the default
    /// `runs_root().join({kernel}-{project_commit})` path instead
    /// of returning an empty path. Pins the
    /// `is_empty()` filter on the override read in
    /// [`sidecar_dir_override`]: a regression that dropped the
    /// filter (e.g. simplified to `std::env::var("...").ok().map(PathBuf::from)`)
    /// would surface here as `sidecar_dir()` returning `PathBuf::from("")`
    /// — a path that joins onto runs-root as a no-op alias and
    /// silently contaminates the runs listing.
    ///
    /// The override branch SHORT-CIRCUITS on a non-empty value
    /// (returns the override verbatim, skipping the format-run-dirname
    /// computation), so the assertion below — comparing
    /// `sidecar_dir()` against the manually-computed default — is
    /// proof that the empty-string DID NOT take the short-circuit
    /// path. A regression that activated the override on empty
    /// would surface as `dir == PathBuf::from("")`, not equal to
    /// the computed default.
    #[test]
    fn sidecar_dir_empty_override_falls_back_to_default() {
        let _lock = lock_env();
        let target_dir = tempfile::TempDir::new().unwrap();
        let _env_target = EnvVarGuard::set("CARGO_TARGET_DIR", target_dir.path());
        // EnvVarGuard::set with an empty path covers the
        // defensively-cleared `KTSTR_SIDECAR_DIR=""` operator
        // pattern. EnvVarGuard accepts AsRef<OsStr>, and a
        // zero-length `&str` ("") satisfies that bound.
        let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", "");
        let _env_kernel = EnvVarGuard::remove("KTSTR_KERNEL");

        let dir = sidecar_dir();
        // Compute the expected default the same way `sidecar_dir`
        // does on its default branch. With KTSTR_KERNEL unset the
        // kernel resolves to "unknown"; commit comes from the
        // OnceLock-cached project probe (Some(hash) when running
        // inside the ktstr repo).
        let kernel = detect_kernel_version();
        let commit = detect_project_commit();
        let expected = runs_root().join(format_run_dirname(kernel.as_deref(), commit.as_deref()));
        assert_eq!(
            dir, expected,
            "empty KTSTR_SIDECAR_DIR must fall back to the default \
             `runs_root().join(format_run_dirname(...))` path, NOT \
             return PathBuf::from(\"\"). A regression that dropped \
             the `is_empty()` filter on the override read would \
             surface here as `dir == PathBuf::from(\"\")`.",
        );
        assert_ne!(
            dir,
            std::path::PathBuf::new(),
            "sidecar_dir must never return an empty path",
        );
    }

    // -- format_run_dirname (pure function, no OnceLock dependency) --

    /// Clean commit shape: `{kernel}-{hex7}` — the standard happy
    /// path. Pinning the format here means a regression that adds
    /// extra punctuation, swaps the order, or drops a component
    /// surfaces as a unit-test failure rather than as a downstream
    /// stats-tooling miss.
    #[test]
    fn format_run_dirname_clean_commit() {
        assert_eq!(
            format_run_dirname(Some("6.14.2"), Some("abc1234")),
            "6.14.2-abc1234",
            "clean dirname must be `{{kernel}}-{{project_commit}}`",
        );
    }

    /// Dirty commit shape: the `-dirty` suffix flows through verbatim
    /// because `format_run_dirname` does not interpret the commit
    /// string — it simply joins. The suffix is appended upstream by
    /// `commit_with_dirty_suffix`. This test pins the verbatim
    /// pass-through.
    #[test]
    fn format_run_dirname_dirty_commit() {
        assert_eq!(
            format_run_dirname(Some("6.14.2"), Some("abc1234-dirty")),
            "6.14.2-abc1234-dirty",
            "dirty dirname must pass the `-dirty` suffix through verbatim",
        );
    }

    /// Missing commit (non-git cwd or probe failure) collapses to
    /// the literal `"unknown"` sentinel in the commit slot, so the
    /// dirname is `{kernel}-unknown`. This is the documented
    /// dirname-vs-JSON asymmetry: in-memory the
    /// `SidecarResult::project_commit` field stays `None`, but the
    /// dirname uses a filesystem-safe sentinel.
    #[test]
    fn format_run_dirname_unknown_commit() {
        assert_eq!(
            format_run_dirname(Some("6.14.2"), None),
            "6.14.2-unknown",
            "missing commit must collapse to `{{kernel}}-unknown` sentinel",
        );
    }

    /// Missing kernel mirrors the missing-commit shape: `unknown-{project_commit}`.
    /// Captures the `KTSTR_KERNEL` unset / detection-failed path
    /// so a regression in the unwrap_or fallback surfaces here.
    #[test]
    fn format_run_dirname_unknown_kernel() {
        assert_eq!(
            format_run_dirname(None, Some("abc1234")),
            "unknown-abc1234",
            "missing kernel must collapse to `unknown-{{project_commit}}` sentinel",
        );
    }

    /// Both components missing: every run from a non-git cwd with no
    /// `KTSTR_KERNEL` set lands in the same `unknown-unknown`
    /// directory. Documented collision: the operator must set
    /// `KTSTR_SIDECAR_DIR` or place the project tree under git to
    /// disambiguate concurrent test runs.
    #[test]
    fn format_run_dirname_both_unknown_collide() {
        assert_eq!(
            format_run_dirname(None, None),
            "unknown-unknown",
            "both-missing case must produce `unknown-unknown` — the documented \
             collision the operator must disambiguate via KTSTR_SIDECAR_DIR or git",
        );
    }

    // -- pre_clear_run_dir_once tests --
    //
    // Pin the four behavioral invariants the doc on
    // `pre_clear_run_dir_once` claims:
    // 1. *.ktstr.json files in the immediate dir are removed.
    // 2. Subdirectories and non-sidecar files are left untouched.
    // 3. A missing dir is silent (no panic).
    // 4. Per-directory keying via Mutex<HashSet<PathBuf>>: a second
    //    call for the SAME dir is a no-op, but a call for a NEW dir
    //    fires its own pre-clear.
    //
    // Each test uses a fresh tempdir so the per-process cache never
    // collides across tests; tests do NOT need `lock_env` because
    // they do not touch any environment variable — pre_clear is
    // env-independent.

    /// `pre_clear_run_dir_once` removes every `*.ktstr.json` file in
    /// the immediate directory on its first call against that dir.
    /// Pins the wipe-on-first-call invariant.
    #[test]
    fn pre_clear_run_dir_once_wipes_existing_sidecars() {
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let tmp = tmp_dir.path();
        std::fs::write(tmp.join("test_a-0000.ktstr.json"), b"{}").unwrap();
        std::fs::write(tmp.join("test_b-1111.ktstr.json"), b"{}").unwrap();
        assert_eq!(
            std::fs::read_dir(tmp).unwrap().count(),
            2,
            "fixture precondition: tempdir must contain two sidecars",
        );

        pre_clear_run_dir_once(tmp);

        let remaining: Vec<_> = std::fs::read_dir(tmp)
            .unwrap()
            .flatten()
            .map(|e| e.file_name())
            .collect();
        assert!(
            remaining.is_empty(),
            "every *.ktstr.json file must be wiped; got {remaining:?}",
        );
    }

    /// `pre_clear_run_dir_once` does NOT recurse — subdirectories
    /// and any non-sidecar files in the immediate dir are left
    /// untouched. Pins the shallow-scope invariant: an external
    /// orchestrator that writes per-job subdirectories under the
    /// run dir does not lose its fixture state to a sibling
    /// invocation's pre-clear.
    #[test]
    fn pre_clear_run_dir_once_skips_subdirs_and_non_sidecars() {
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let tmp = tmp_dir.path();
        // Top-level sidecar: should be wiped.
        std::fs::write(tmp.join("victim-0000.ktstr.json"), b"{}").unwrap();
        // Top-level non-sidecar files: should survive.
        std::fs::write(tmp.join("README.md"), b"keep").unwrap();
        std::fs::write(tmp.join("other.json"), b"{}").unwrap();
        std::fs::write(tmp.join("partial.ktstr.json.tmp"), b"{}").unwrap();
        // Subdirectory with a sidecar inside: subdir AND its
        // contents should survive (pre-clear does not recurse).
        let sub = tmp.join("job-1");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(sub.join("nested-0000.ktstr.json"), b"{}").unwrap();

        pre_clear_run_dir_once(tmp);

        assert!(
            !tmp.join("victim-0000.ktstr.json").exists(),
            "top-level *.ktstr.json file must be wiped",
        );
        assert!(
            tmp.join("README.md").exists(),
            "non-sidecar file must survive",
        );
        assert!(
            tmp.join("other.json").exists(),
            "bare *.json (no .ktstr. infix) must survive",
        );
        assert!(
            tmp.join("partial.ktstr.json.tmp").exists(),
            "non-`.json` extension must survive even with .ktstr. infix",
        );
        assert!(sub.exists(), "subdirectory must survive");
        assert!(
            sub.join("nested-0000.ktstr.json").exists(),
            "sidecar inside subdirectory must survive (pre-clear is shallow)",
        );
    }

    /// `pre_clear_run_dir_once` is silent when the target directory
    /// does not yet exist — `read_dir` errors are swallowed. Pins
    /// the helper's API contract that a missing dir is a no-op
    /// rather than a panic. The production caller
    /// (`serialize_and_write_sidecar`) materializes the dir via
    /// `create_dir_all` BEFORE feeding it to this helper, so the
    /// missing-dir branch is unreachable in production today; the
    /// invariant is preserved for defensive correctness against
    /// future direct callers and to keep the helper safe to call
    /// from unit tests that probe the missing-dir edge.
    #[test]
    fn pre_clear_run_dir_once_silent_on_missing_dir() {
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let nonexistent = tmp_dir.path().join("does_not_exist_yet");
        assert!(
            !nonexistent.exists(),
            "fixture precondition: dir must not exist"
        );

        // Must not panic. The function returns `()`, so the only
        // observable failure mode is a panic — the call returning
        // normally is the test's pass condition.
        pre_clear_run_dir_once(&nonexistent);

        // Sanity: pre_clear must not have created the dir as a
        // side effect either. `serialize_and_write_sidecar`'s
        // create_dir_all is the only path that materializes the
        // directory.
        assert!(
            !nonexistent.exists(),
            "pre_clear must not create the dir as a side effect",
        );
    }

    /// Per-directory keying via `Mutex<HashSet<PathBuf>>`: a second
    /// call against the SAME dir is a no-op (newly-written sidecars
    /// after the first pre-clear must NOT be wiped on the second
    /// call), but a call against a DIFFERENT dir fires its own
    /// pre-clear. Pins both halves of the per-dir contract:
    /// idempotent for repeats, fresh for novel paths.
    ///
    /// The two tempdirs share a process-global `OnceLock<Mutex<HashSet<...>>>`,
    /// so the test order is incidental — what matters is that the
    /// HashSet has separate entries per dir.
    #[test]
    fn pre_clear_run_dir_once_keys_per_directory() {
        let tmp_a = tempfile::TempDir::new().unwrap();
        let tmp_b = tempfile::TempDir::new().unwrap();

        // Phase 1: prime dir A. Populate with a sidecar, call
        // pre_clear, verify wiped. The HashSet now contains A's
        // canonicalized path.
        std::fs::write(tmp_a.path().join("a-0000.ktstr.json"), b"{}").unwrap();
        pre_clear_run_dir_once(tmp_a.path());
        assert!(
            !tmp_a.path().join("a-0000.ktstr.json").exists(),
            "first call against A must wipe A's sidecar",
        );

        // Phase 2: write a new sidecar to A (modeling the writer
        // populating the dir AFTER pre-clear), then call pre_clear
        // against A again. The cache hit must short-circuit the
        // wipe — the new sidecar must SURVIVE.
        std::fs::write(tmp_a.path().join("a-1111.ktstr.json"), b"{}").unwrap();
        pre_clear_run_dir_once(tmp_a.path());
        assert!(
            tmp_a.path().join("a-1111.ktstr.json").exists(),
            "second call against A must be a no-op (cache hit) — \
             the post-prime sidecar must survive. A regression to \
             OnceLock<()> or a HashSet that ignores the key would \
             leak this assertion.",
        );

        // Phase 3: prime dir B (new path). The HashSet has no
        // entry for B yet, so this call must wipe B's sidecar
        // — proving the cache distinguishes paths rather than
        // collapsing every call after the first.
        std::fs::write(tmp_b.path().join("b-0000.ktstr.json"), b"{}").unwrap();
        pre_clear_run_dir_once(tmp_b.path());
        assert!(
            !tmp_b.path().join("b-0000.ktstr.json").exists(),
            "first call against B must wipe B's sidecar — proves the \
             per-dir keying distinguishes A from B (a OnceLock<()> \
             that fired once for A would leak this assertion).",
        );
    }

    // -- warn_unknown_project_commit_inner tests --
    //
    // Pin the three behavioral invariants the inner helper exposes:
    // 1. Calling once writes the warning text to the sink.
    // 2. The emitted text contains the operator-actionable substring
    //    pointing at `KTSTR_SIDECAR_DIR` so a future doc-drift on the
    //    warning prose surfaces here rather than silently changing
    //    operator-facing remediation.
    // 3. A second call against the SAME `OnceLock<()>` is a no-op —
    //    the second call must NOT append additional bytes to the sink.
    //
    // Each test owns a local `OnceLock<()>` so the tests are
    // independent of any other test (or the production wrapper) that
    // might already have initialized the process-global gate. No
    // `lock_env` needed: the inner helper does not touch any env var
    // or any shared global state beyond the gate the caller supplies.

    /// First call against a fresh `OnceLock<()>` writes the warning
    /// text to the sink. Pins the emit-once invariant on initial
    /// invocation and proves the inner helper emits via the
    /// caller-provided sink rather than fd 2.
    #[test]
    fn warn_unknown_project_commit_inner_emits_on_first_call() {
        let gate = std::sync::OnceLock::new();
        let mut sink: Vec<u8> = Vec::new();
        warn_unknown_project_commit_inner(&gate, &mut sink);
        assert!(
            !sink.is_empty(),
            "first call must emit bytes to the sink; got empty",
        );
    }

    /// Pin the operator-actionable substring of the warning. The
    /// test does NOT pin the entire prose verbatim — that would
    /// make every wording tweak break here — but it DOES pin the
    /// single load-bearing remediation hint (`KTSTR_SIDECAR_DIR`)
    /// so a future edit that drops the recommended env var loses
    /// this assertion. The `WARNING:` marker is also pinned so a
    /// downgrade from warning to info changes the severity tag
    /// observably.
    #[test]
    fn warn_unknown_project_commit_inner_emits_expected_substring() {
        let gate = std::sync::OnceLock::new();
        let mut sink: Vec<u8> = Vec::new();
        warn_unknown_project_commit_inner(&gate, &mut sink);
        let captured = String::from_utf8(sink).expect("warning text must be UTF-8");
        assert!(
            captured.contains("WARNING:"),
            "warning must carry the WARNING severity tag; got: {captured:?}",
        );
        assert!(
            captured.contains("KTSTR_SIDECAR_DIR"),
            "warning must reference KTSTR_SIDECAR_DIR as the remediation \
             knob — operators rely on this hint to disambiguate \
             non-git runs; got: {captured:?}",
        );
    }

    /// A second call against the SAME `OnceLock<()>` is a no-op —
    /// the gate has already been initialized by the first call, so
    /// `get_or_init`'s closure does not fire and no additional bytes
    /// land in the sink. Pins the once-per-gate contract that
    /// gauntlet variants rely on (otherwise the operator would see
    /// thousands of duplicate warnings interleaved with test output).
    ///
    /// The assertion compares the sink's length AFTER the second
    /// call against its length AFTER the first call. A regression
    /// that re-fires the warning would extend the sink and break
    /// this equality.
    #[test]
    fn warn_unknown_project_commit_inner_second_call_is_no_op() {
        let gate = std::sync::OnceLock::new();
        let mut sink: Vec<u8> = Vec::new();
        warn_unknown_project_commit_inner(&gate, &mut sink);
        let after_first = sink.len();
        assert!(
            after_first > 0,
            "fixture precondition: first call must emit bytes",
        );
        warn_unknown_project_commit_inner(&gate, &mut sink);
        assert_eq!(
            sink.len(),
            after_first,
            "second call against the same gate must NOT append bytes — \
             the OnceLock<()> gating is the load-bearing invariant; got \
             len {} (expected {after_first})",
            sink.len(),
        );
    }

    // -- newest_run_dir tests --
    //
    // Pin the dotfile filter so the flock sentinel subdirectory
    // (`.locks/`) cannot eclipse a real run dir as the "most
    // recent run" — `.locks/`'s mtime tracks per-write flock
    // activity and would otherwise advance past the run dir's
    // own mtime on the most recent sidecar write, claiming the
    // newest-run bucket.

    /// `newest_run_dir` must pick a real run directory in
    /// preference to a NEWER `.locks/` directory at the same
    /// runs root. Mtime ordering is stamped via filesystem
    /// create order with a sleep between calls so the test
    /// deterministically distinguishes "newer .locks ignored"
    /// from "older real run picked up because it happened to
    /// have the largest mtime."
    #[test]
    fn newest_run_dir_skips_dotfile_subdirectories() {
        use std::thread::sleep;
        use std::time::Duration;
        let _lock = lock_env();
        let target_dir = tempfile::TempDir::new().unwrap();
        let _env_target = EnvVarGuard::set("CARGO_TARGET_DIR", target_dir.path());
        // `runs_root()` returns `{CARGO_TARGET_DIR}/ktstr/`, so
        // create that intermediate before populating run subdirs.
        let runs = target_dir.path().join("ktstr");
        std::fs::create_dir(&runs).expect("mkdir runs root");
        // Real run dir created first, so its mtime is OLDER.
        let real = runs.join("real-run");
        std::fs::create_dir(&real).expect("mkdir real run dir");
        sleep(Duration::from_millis(50));
        // .locks/ created second, so its mtime is NEWER. Without
        // the dotfile filter, this entry would win the
        // max_by_key contest and `newest_run_dir` would return
        // `.locks/` — the regression that this test guards.
        std::fs::create_dir(runs.join(".locks")).expect("mkdir .locks");
        let got = newest_run_dir().expect("non-empty runs root must yield Some");
        assert_eq!(
            got, real,
            "newest_run_dir must pick the real run dir even when \
             .locks/ has a newer mtime — a regression that drops \
             the dotfile filter would surface here as `.locks/` \
             winning the mtime contest",
        );
    }

    /// `newest_run_dir` returns `None` when only dotfile-prefixed
    /// subdirectories exist under the runs root. Pins the
    /// post-filter empty case: even if the runs root itself is
    /// non-empty, a fresh repo state (only `.locks/` lives there
    /// because no test has ever produced a sidecar) must not
    /// surface `.locks/` as a stand-in run.
    #[test]
    fn newest_run_dir_yields_none_when_only_dotfiles_exist() {
        let _lock = lock_env();
        let target_dir = tempfile::TempDir::new().unwrap();
        let _env_target = EnvVarGuard::set("CARGO_TARGET_DIR", target_dir.path());
        let runs = target_dir.path().join("ktstr");
        std::fs::create_dir(&runs).expect("mkdir runs root");
        std::fs::create_dir(runs.join(".locks")).expect("mkdir .locks");
        std::fs::create_dir(runs.join(".cache")).expect("mkdir .cache");
        let got = newest_run_dir();
        assert!(
            got.is_none(),
            "runs root with only dotfile subdirs must yield None; got {got:?}",
        );
    }

    // -- is_run_directory predicate tests --
    //
    // Direct unit tests over the predicate that backs both
    // `newest_run_dir` and `sorted_run_entries`'s filter. Pure
    // shape contract, no I/O beyond a tempdir to materialize
    // DirEntries the predicate can consume.

    /// A regular subdirectory whose name does not start with `.`
    /// passes the predicate.
    #[test]
    fn is_run_directory_accepts_non_dotfile_subdir() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::create_dir(tmp.path().join("real-run")).unwrap();
        let entry = std::fs::read_dir(tmp.path())
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        assert!(
            super::is_run_directory(&entry),
            "non-dotfile subdir must be accepted",
        );
    }

    /// A subdirectory whose name starts with `.` is rejected.
    /// Pins the dotfile filter — the load-bearing rule for the
    /// `.locks/` exclusion.
    #[test]
    fn is_run_directory_rejects_dotfile_subdir() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::create_dir(tmp.path().join(".locks")).unwrap();
        let entry = std::fs::read_dir(tmp.path())
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        assert!(
            !super::is_run_directory(&entry),
            "dotfile subdir must be rejected",
        );
    }

    /// A regular file (not a directory) is rejected, regardless
    /// of name — the `is_dir()` short-circuit must precede the
    /// dotfile check.
    #[test]
    fn is_run_directory_rejects_regular_files() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("regular-file"), b"x").unwrap();
        let entry = std::fs::read_dir(tmp.path())
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        assert!(
            !super::is_run_directory(&entry),
            "regular file must be rejected",
        );
    }

    // -- run_dir_lock_path / acquire_run_dir_flock tests --
    //
    // Pin the cross-process flock contract added for the
    // concurrent-write collision fix:
    //
    // 1. `run_dir_lock_path` derives the canonical
    //    `{parent}/.locks/{leaf}.lock` shape so two callers
    //    keying off the same `dir` agree on the lockfile.
    // 2. `acquire_run_dir_flock_with_timeout` materializes the
    //    parent `.locks/` subdirectory on first call and returns
    //    an `OwnedFd` whose Drop releases the lock — so a second
    //    call after the first returns can acquire successfully.
    // 3. While a peer holds `LOCK_EX` on the lockfile, the helper
    //    times out with an actionable error. (Different `OwnedFd`s
    //    in the same process are distinct OFDs, so flock(2)
    //    serializes them the same way it would two processes —
    //    no fork required to exercise the contention path.)
    //
    // No `lock_env` needed: the helpers don't touch any env var.
    // Each test owns a tempdir so the per-test lockfile namespace
    // is isolated.

    /// `run_dir_lock_path({parent}/{key})` returns
    /// `{parent}/.locks/{key}.lock`. Pins the layout so a future
    /// edit to [`crate::flock::LOCK_DIR_NAME`] or the join shape
    /// surfaces here rather than as a silent cross-call divergence.
    #[test]
    fn run_dir_lock_path_returns_expected_shape() {
        let dir = std::path::Path::new("/runs-root/6.14.2-deadbee");
        let lock = super::run_dir_lock_path(dir).expect("non-root dir must yield Some");
        assert_eq!(
            lock,
            std::path::PathBuf::from("/runs-root/.locks/6.14.2-deadbee.lock"),
        );
    }

    /// A path with no parent (root `/`) has no canonical lockfile
    /// location — the helper returns `None` rather than constructing
    /// an unsafe sentinel. Pins the defensive arm so a regression
    /// that unwraps `parent()` surfaces here.
    #[test]
    fn run_dir_lock_path_no_parent_returns_none() {
        let lock = super::run_dir_lock_path(std::path::Path::new("/"));
        assert!(
            lock.is_none(),
            "root path must yield None (no parent), got {lock:?}",
        );
    }

    /// First call against a fresh `dir` materializes the parent
    /// `.locks/` subdirectory on demand and returns an `OwnedFd`
    /// holding `LOCK_EX`. The lockfile itself persists after the
    /// fd is dropped (only the kernel-side lock is released);
    /// that's what `try_flock`'s own contract guarantees.
    #[test]
    fn acquire_run_dir_flock_creates_locks_subdir_lazily() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path().join("6.14.2-deadbee");
        // `acquire_run_dir_flock_with_timeout` doesn't require the
        // run-dir itself to exist — the production caller
        // materializes it via `create_dir_all` BEFORE this point.
        // Mirror that pattern.
        std::fs::create_dir_all(&dir).unwrap();

        let fd = super::acquire_run_dir_flock_with_timeout(&dir, std::time::Duration::from_secs(1))
            .expect("first acquire must succeed against an uncontended dir");
        assert!(
            tmp.path().join(".locks").exists(),
            ".locks/ subdirectory must be created lazily on first acquire",
        );
        assert!(
            tmp.path().join(".locks/6.14.2-deadbee.lock").exists(),
            "lockfile must exist on disk after acquire",
        );
        // Drop the fd — releases the kernel-side flock. The
        // sentinel file persists (released, but not unlinked).
        drop(fd);
        assert!(
            tmp.path().join(".locks/6.14.2-deadbee.lock").exists(),
            "lockfile sentinel must persist after fd drop — \
             try_flock's contract is fd-bound release, not file unlink",
        );
    }

    /// A second `acquire_run_dir_flock_with_timeout` against the
    /// same dir AFTER the first fd was dropped must succeed —
    /// proves the kernel-side release happens via `OwnedFd::drop`
    /// (no leaked OFD blocking subsequent acquires).
    #[test]
    fn acquire_run_dir_flock_releases_on_drop() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path().join("key");
        std::fs::create_dir_all(&dir).unwrap();

        let fd1 =
            super::acquire_run_dir_flock_with_timeout(&dir, std::time::Duration::from_secs(1))
                .expect("first acquire");
        drop(fd1);
        let fd2 =
            super::acquire_run_dir_flock_with_timeout(&dir, std::time::Duration::from_secs(1))
                .expect(
                    "second acquire after drop must succeed — a regression that \
             fails to release the kernel flock on OwnedFd::drop would \
             leak this assertion",
                );
        drop(fd2);
    }

    /// While a peer holds `LOCK_EX` on the same dir's lockfile,
    /// `acquire_run_dir_flock_with_timeout` waits and eventually
    /// fails with an actionable error message. Pins the
    /// cross-process serialization contract.
    ///
    /// In-process collision: two `try_flock` calls open distinct
    /// OFDs against the same lockfile, and `flock(2)` serializes
    /// them the same way it would two processes — so this test
    /// exercises the production contention path without spawning
    /// a child.
    #[test]
    fn acquire_run_dir_flock_times_out_when_peer_holds_lock() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path().join("contended-key");
        std::fs::create_dir_all(&dir).unwrap();

        // Peer: acquire the lock through the same machinery and
        // hold the fd alive for the duration of the test. Any
        // sibling acquire must time out behind this hold.
        let lock_path = super::run_dir_lock_path(&dir).unwrap();
        std::fs::create_dir_all(lock_path.parent().unwrap()).unwrap();
        let _peer_fd = crate::flock::try_flock(&lock_path, crate::flock::FlockMode::Exclusive)
            .expect("peer flock attempt")
            .expect("peer must acquire on a fresh lockfile");

        let start = std::time::Instant::now();
        let err =
            super::acquire_run_dir_flock_with_timeout(&dir, std::time::Duration::from_millis(300))
                .expect_err("acquire must fail while peer holds LOCK_EX");
        let elapsed = start.elapsed();
        // Sanity: the helper waited at least roughly the requested
        // timeout before erroring — proves it polled rather than
        // returning EWOULDBLOCK on the first try.
        assert!(
            elapsed >= std::time::Duration::from_millis(250),
            "acquire must wait ~timeout before erroring; elapsed={elapsed:?}",
        );
        let msg = format!("{err:#}");
        assert!(
            msg.contains("timed out"),
            "error must surface the timeout cause; got: {msg}",
        );
        assert!(
            msg.contains("LOCK_EX"),
            "error must name the flock mode for operator triage; got: {msg}",
        );
    }

    // -- write_sidecar reuse-dir behavior (fix #5) --

    /// Two `write_sidecar` invocations against the same effective
    /// run directory (same `KTSTR_SIDECAR_DIR` here, simulating two
    /// invocations from the same kernel + project commit) must
    /// produce a directory containing only the second invocation's
    /// sidecars — the first invocation's outputs are pre-cleared
    /// before the second writes. Pins the last-writer-wins
    /// semantics the documented `{kernel}-{project_commit}` keying implies.
    ///
    /// CAVEAT: this test exercises the OVERRIDE path
    /// (`KTSTR_SIDECAR_DIR` is set), where pre-clear is currently
    /// SKIPPED per the fix-#2 contract. To exercise pre-clear in
    /// the env-overridden context, the test directly calls
    /// `pre_clear_run_dir_once` BETWEEN the two writes — modeling
    /// what `serialize_and_write_sidecar` does on the default path
    /// (env unset). Both writes go through the override path so
    /// the test does not depend on the OnceLock-cached cwd.
    #[test]
    fn write_sidecar_same_dir_is_last_writer_wins_after_pre_clear() {
        let _lock = lock_env();
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let tmp = tmp_dir.path();
        let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", tmp);

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        // First invocation: write a sidecar for entry A.
        let entry_a = KtstrTestEntry {
            name: "__reuse_first_run__",
            func: dummy,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        };
        let vm_result = crate::vmm::VmResult::test_fixture();
        let ok = AssertResult::pass();
        write_sidecar(&entry_a, &vm_result, &[], &ok, "CpuSpin", &[], &[]).unwrap();
        // Confirm the first invocation's sidecar is on disk.
        assert_eq!(
            find_sidecars_by_prefix(tmp, "__reuse_first_run__-").len(),
            1,
            "first invocation must write its sidecar",
        );

        // Simulate the second invocation: pre-clear the dir (which
        // is what `serialize_and_write_sidecar` does on the default
        // path), then write a sidecar for entry B.
        pre_clear_run_dir_once(tmp);
        // The first invocation's sidecar must be wiped by pre-clear.
        assert_eq!(
            find_sidecars_by_prefix(tmp, "__reuse_first_run__-").len(),
            0,
            "pre-clear must wipe the first invocation's sidecar before \
             the second invocation writes — this is the last-writer-wins \
             contract",
        );

        // Second invocation: distinct entry name to prove the
        // dir-state after pre-clear contains ONLY the second
        // invocation's sidecars.
        let entry_b = KtstrTestEntry {
            name: "__reuse_second_run__",
            func: dummy,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        };
        write_sidecar(&entry_b, &vm_result, &[], &ok, "CpuSpin", &[], &[]).unwrap();

        // Final state: only the second invocation's sidecar is
        // present. The first invocation is gone, the second is
        // intact.
        assert_eq!(
            find_sidecars_by_prefix(tmp, "__reuse_first_run__-").len(),
            0,
            "first invocation's sidecar must remain wiped after second invocation writes",
        );
        assert_eq!(
            find_sidecars_by_prefix(tmp, "__reuse_second_run__-").len(),
            1,
            "second invocation's sidecar must be the only sidecar in the dir",
        );
    }

    // -- KTSTR_SIDECAR_DIR override skips pre-clear (fix #2) --

    /// When `KTSTR_SIDECAR_DIR` is set, `serialize_and_write_sidecar`
    /// must NOT call `pre_clear_run_dir_once` against the override
    /// dir. Pins the contract that operator-chosen directories are
    /// preserved verbatim — silent data loss on an explicit env
    /// override is unacceptable.
    ///
    /// The test populates the override dir with a pre-existing
    /// sidecar (from a hypothetical sibling run or a manual
    /// fixture), runs `write_sidecar`, and verifies BOTH the
    /// pre-existing sidecar AND the newly-written one are present.
    /// A regression that pre-cleared on the override path would
    /// leak this assertion (the pre-existing sidecar would be
    /// wiped).
    #[test]
    fn write_sidecar_override_does_not_pre_clear() {
        let _lock = lock_env();
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let tmp = tmp_dir.path();
        let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", tmp);

        // Pre-existing sidecar in the override dir — modeling a
        // run the operator wants to preserve.
        std::fs::write(tmp.join("__preserved__-0000.ktstr.json"), b"{}").unwrap();

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "__override_skips_preclear__",
            func: dummy,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        };
        let vm_result = crate::vmm::VmResult::test_fixture();
        let ok = AssertResult::pass();
        write_sidecar(&entry, &vm_result, &[], &ok, "CpuSpin", &[], &[]).unwrap();

        // The pre-existing sidecar must still be there. A regression
        // that fired pre_clear on the override path would have
        // wiped it.
        assert!(
            tmp.join("__preserved__-0000.ktstr.json").exists(),
            "pre-existing sidecar in override dir must NOT be pre-cleared — \
             operator-chosen directories are owned by the operator and \
             must not lose data on `write_sidecar`",
        );
        // Sanity: the new sidecar landed too.
        assert_eq!(
            find_sidecars_by_prefix(tmp, "__override_skips_preclear__-").len(),
            1,
            "new sidecar must be written alongside the preserved one",
        );
    }

    // -- B3 regression: relative-path canonicalize cache split (fix #1 in pass 2 ruling) --

    /// Two sequential `write_sidecar` calls in the same process
    /// against the DEFAULT path (no `KTSTR_SIDECAR_DIR` override)
    /// must both survive: the second call must NOT wipe the first.
    ///
    /// Pins the regression caught in pass-2 review: when
    /// `serialize_and_write_sidecar` invoked `pre_clear_run_dir_once`
    /// BEFORE `create_dir_all`, the first call resolved the
    /// pre-clear cache key against the raw path because
    /// `canonicalize` failed on a missing dir. The first call then
    /// created the dir via `create_dir_all` and wrote sidecar 1.
    /// On the second call, `canonicalize` SUCCEEDED against the
    /// now-existing dir, producing an absolute path that DIFFERED
    /// from the cache key inserted by the first call — so the
    /// second call missed the cache, fired pre-clear, and wiped
    /// sidecar 1.
    ///
    /// The fix moves `create_dir_all` before `pre_clear_run_dir_once`
    /// so canonicalize sees the same on-disk dir on both calls and
    /// produces the same canonicalized cache key. With the fix,
    /// the second call hits the cache and pre-clear is a no-op,
    /// so sidecar 1 survives.
    ///
    /// ISOLATION: the test sets `CARGO_TARGET_DIR` to a unique
    /// tempdir so the resolved sidecar dir is
    /// `{tempdir}/ktstr/{kernel}-{project_commit}/` — uncrossable by
    /// sibling test processes that share the workspace's
    /// `target/ktstr/`. Without this isolation, a concurrent
    /// nextest worker writing to the SAME shared default dir could
    /// fire pre-clear for that dir, race with this test's writes,
    /// and surface as a flaky `__b3_first__-` count = 0. The test
    /// still exercises the REAL default-path flow (sidecar_dir
    /// computes from runs_root + format_run_dirname,
    /// serialize_and_write_sidecar runs create_dir_all then
    /// pre_clear) — the only thing CARGO_TARGET_DIR redirects is
    /// the runs-root parent.
    ///
    /// `KTSTR_KERNEL` and `KTSTR_SIDECAR_DIR` are explicitly
    /// removed: kernel resolves to `"unknown"` (deterministic),
    /// override is unset (so the default-path branch runs).
    /// Project commit comes from the test process's
    /// OnceLock-cached cwd probe and is shared with every other
    /// default-path test in the same process — irrelevant here
    /// since the tempdir-scoped runs-root parent is unique to this
    /// test, so no other test's pre-clear cache entry collides
    /// with ours.
    #[test]
    fn write_sidecar_default_path_two_writes_both_survive() {
        let _lock = lock_env();
        let target_dir = tempfile::TempDir::new().unwrap();
        let _env_target = EnvVarGuard::set("CARGO_TARGET_DIR", target_dir.path());
        let _env_sidecar = EnvVarGuard::remove("KTSTR_SIDECAR_DIR");
        let _env_kernel = EnvVarGuard::remove("KTSTR_KERNEL");

        // Resolve the default dir AFTER the env mutations so it
        // reflects the tempdir-scoped target. With KTSTR_KERNEL
        // unset and KTSTR_SIDECAR_DIR unset, this is
        // `{tempdir}/ktstr/unknown-{cached_project_commit}/`.
        let dir = sidecar_dir();

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry_first = KtstrTestEntry {
            name: "__b3_first__",
            func: dummy,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        };
        let entry_second = KtstrTestEntry {
            name: "__b3_second__",
            func: dummy,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        };
        let vm_result = crate::vmm::VmResult::test_fixture();
        let ok = AssertResult::pass();

        // First write: under the buggy ordering, this resolved
        // canonicalize-fails (dir missing) → cache key under raw
        // path → wipe was a no-op (dir didn't exist) → created
        // dir → wrote sidecar 1.
        write_sidecar(&entry_first, &vm_result, &[], &ok, "CpuSpin", &[], &[]).unwrap();
        // Confirm sidecar 1 lands.
        assert_eq!(
            find_sidecars_by_prefix(&dir, "__b3_first__-").len(),
            1,
            "first write must produce its sidecar",
        );

        // Second write: under the buggy ordering, this resolved
        // canonicalize-succeeds (dir now exists) → cache key under
        // absolute canonicalized path → DIFFERENT key than first
        // call → cache MISS → wipe ran → DELETED sidecar 1 → wrote
        // sidecar 2. Under the fix, create_dir_all runs first on
        // both calls, both canonicalize against an existing dir,
        // both produce the same canonicalized key, and the second
        // call hits the cache → no wipe → both survive.
        write_sidecar(&entry_second, &vm_result, &[], &ok, "CpuSpin", &[], &[]).unwrap();

        // Both sidecars must be present. A regression to the buggy
        // ordering would surface here as `__b3_first__-` count = 0.
        let first_count = find_sidecars_by_prefix(&dir, "__b3_first__-").len();
        let second_count = find_sidecars_by_prefix(&dir, "__b3_second__-").len();
        assert_eq!(
            first_count, 1,
            "first sidecar must survive the second write — a count of 0 \
             reveals the canonicalize-cache-split regression: pre-clear \
             ran a second time and wiped sidecar 1. Move `create_dir_all` \
             before `pre_clear_run_dir_once` so canonicalize sees the \
             same dir on both calls.",
        );
        assert_eq!(second_count, 1, "second sidecar must land normally",);

        // No explicit cleanup: the TempDir's Drop removes the
        // entire tempdir tree, including the sidecars and any
        // pre-clear residue under `{tempdir}/ktstr/`.
    }

    #[test]
    fn write_sidecar_writes_file() {
        let _lock = lock_env();
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let tmp = tmp_dir.path();
        let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", tmp);

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "__sidecar_write_test__",
            func: dummy,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        };
        let vm_result = crate::vmm::VmResult::test_fixture();
        let check_result = AssertResult::pass();
        write_sidecar(&entry, &vm_result, &[], &check_result, "CpuSpin", &[], &[]).unwrap();

        // Sidecar filename now includes a variant hash suffix so
        // gauntlet variants don't clobber each other. Use the
        // single-match helper, which also guards against stray
        // leftover files from prior runs or double-writer bugs.
        let path = find_single_sidecar_by_prefix(tmp, "__sidecar_write_test__-");
        let data = std::fs::read_to_string(&path).unwrap();
        let loaded: SidecarResult = serde_json::from_str(&data).unwrap();
        assert_eq!(loaded.test_name, "__sidecar_write_test__");
        assert!(loaded.passed);
        assert!(!loaded.skipped, "pass result is not a skip");
        // write_sidecar must populate the host-context snapshot so
        // downstream `stats compare --runs a b` can diff hosts.
        // Without this assertion, a regression that dropped the
        // `host: Some(collect_host_context())` builder line would
        // land silently. `kernel_name` is always `Some("Linux")`
        // on a running Linux process (uname syscall, no filesystem
        // dependency), matching the baseline asserted by
        // `host_context::tests::collect_host_context_returns_populated_struct_on_linux`.
        let host = loaded
            .host
            .as_ref()
            .expect("write_sidecar must populate host field from collect_host_context");
        assert_eq!(host.kernel_name.as_deref(), Some("Linux"));
        // Pair the uname check with a field that `HostContext::default()`
        // leaves None. A regression that swapped the full
        // `collect_host_context()` call for `HostContext { kernel_name:
        // Some("Linux".into()), ..Default::default() }` would pass the
        // uname assertion but drop every other captured field —
        // `kernel_cmdline` is present on every live Linux process
        // (/proc/cmdline is always readable; see host_context::tests:
        // collect_host_context_captures_cmdline_on_linux) so
        // `kernel_cmdline.is_some()` catches the default-substitution
        // regression.
        assert!(
            host.kernel_cmdline.is_some(),
            "write_sidecar must capture full HostContext, not Default::default() — \
             /proc/cmdline is always readable on Linux (see host_context tests)",
        );
        // Second Default-distinguishing field: `kernel_release` is
        // populated by the uname() syscall on any live Linux host
        // (filesystem-independent — no /proc/sys dependency), so a
        // `None` here would indicate the default-substitution
        // regression reached the uname path. Pairing cmdline
        // (filesystem-sourced) with kernel_release (syscall-sourced)
        // gives two independent capture paths, so a regression that
        // broke only one collection site is still caught.
        assert!(
            host.kernel_release.is_some(),
            "write_sidecar must capture kernel_release — uname() is \
             filesystem-independent; a None here means the default \
             substitution bypassed the full collect_host_context()",
        );
    }

    #[test]
    fn write_sidecar_variant_hash_distinguishes_active_flags() {
        // Two gauntlet variants differing ONLY in active_flags must
        // produce distinct sidecar filenames so neither clobbers the
        // other. A hash of work_type/sysctls/kargs alone would miss
        // this difference.
        let _lock = lock_env();
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let tmp = tmp_dir.path();
        let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", tmp);

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "__flagvariant_test__",
            func: dummy,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        };
        let vm_result = crate::vmm::VmResult::test_fixture();
        let ok = AssertResult::pass();
        let flags_a = vec!["llc".to_string()];
        let flags_b = vec!["llc".to_string(), "steal".to_string()];
        write_sidecar(&entry, &vm_result, &[], &ok, "CpuSpin", &flags_a, &[]).unwrap();
        write_sidecar(&entry, &vm_result, &[], &ok, "CpuSpin", &flags_b, &[]).unwrap();

        let paths = find_sidecars_by_prefix(tmp, "__flagvariant_test__-");
        assert_eq!(
            paths.len(),
            2,
            "two active_flags variants must produce two distinct files, got {paths:?}"
        );
    }

    /// Two `write_sidecar` calls differing ONLY in the ORDER their
    /// caller accumulated `active_flags` — same semantic variant,
    /// same flag SET — must produce identical sidecar filenames.
    /// Filenames are keyed on [`sidecar_variant_hash`], which walks
    /// `active_flags` in-order and folds each byte into the hash
    /// state. Without canonicalization at the write site, a caller
    /// that happened to collect `["steal", "llc"]` would hash to
    /// a different bucket than one that collected `["llc",
    /// "steal"]` for the same run — `stats compare` would then see
    /// two rows for one semantic variant and mark one as "new" or
    /// "removed" on a re-run that only changed flag accumulation
    /// order.
    ///
    /// This test pins the canonicalization done by
    /// `canonicalize_active_flags` (applied in both
    /// `write_sidecar` and `write_skip_sidecar`): two writes with
    /// reversed flag order collapse to a single file via normal
    /// overwrite. A regression that dropped the sort (reverting to
    /// `active_flags.to_vec()`) would make the second write land
    /// at a different hash → two files, caught here. Pair with
    /// `write_sidecar_variant_hash_distinguishes_active_flags`
    /// above, which pins the complementary property: different
    /// flag SETS must still hash distinctly.
    #[test]
    fn write_sidecar_variant_hash_is_order_invariant_for_active_flags() {
        let _lock = lock_env();
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let tmp = tmp_dir.path();
        let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", tmp);

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "__flagorder_test__",
            func: dummy,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        };
        let vm_result = crate::vmm::VmResult::test_fixture();
        let ok = AssertResult::pass();
        // Same set of flags in reversed accumulation order. `llc` is
        // `ALL_DECLS[0].name` and `steal` is `ALL_DECLS[2].name`, so
        // the canonical order is ["llc","steal"] regardless of
        // which order the caller supplied them.
        let forward = vec!["llc".to_string(), "steal".to_string()];
        let reversed = vec!["steal".to_string(), "llc".to_string()];
        write_sidecar(&entry, &vm_result, &[], &ok, "CpuSpin", &forward, &[]).unwrap();
        write_sidecar(&entry, &vm_result, &[], &ok, "CpuSpin", &reversed, &[]).unwrap();

        let paths = find_sidecars_by_prefix(tmp, "__flagorder_test__-");
        assert_eq!(
            paths.len(),
            1,
            "reversed-order writes of the same flag SET must \
             collapse to a single canonical sidecar filename \
             (overwrite); got {paths:?}. If this fails with \
             `paths.len() == 2`, the write path has regressed to \
             hashing caller-order flags — re-sort via \
             `canonicalize_active_flags` in both write_sidecar \
             and write_skip_sidecar.",
        );

        // Defensive: the single surviving file must carry the
        // canonical order on disk, not whichever order the last
        // caller passed. Deserialize and check.
        let path = &paths[0];
        let data = std::fs::read_to_string(path).expect("read canonical sidecar");
        let loaded: SidecarResult =
            serde_json::from_str(&data).expect("deserialize canonical sidecar");
        assert_eq!(
            loaded.active_flags,
            vec!["llc".to_string(), "steal".to_string()],
            "on-disk active_flags must be sorted in \
             `scenario::flags::ALL` positional order; got: {:?}",
            loaded.active_flags,
        );
    }

    /// `sidecar_variant_hash` is order-insensitive for `sysctls`
    /// and `kargs` — same contract as `active_flags`, but
    /// canonicalized at hash time (local sort inside
    /// `sidecar_variant_hash`) rather than at write time. Pinning
    /// the invariant directly against the hash function catches a
    /// regression that drops the sort block (reverts to iterating
    /// `&sidecar.sysctls` / `&sidecar.kargs` in-order) even if all
    /// existing stability pins continue to pass — those pins use
    /// single-element collections where sorting is a no-op, so
    /// they cannot detect this regression by themselves.
    ///
    /// Calls the hash function directly rather than going through
    /// `write_sidecar` because the sysctls/kargs come from
    /// `entry.scheduler.sysctls()` / `kargs()` — static slices the
    /// caller cannot reorder. The only path for a reordered input
    /// is a direct `SidecarResult` construction with reordered
    /// fields, which this test exercises.
    #[test]
    fn sidecar_variant_hash_is_order_invariant_for_sysctls_and_kargs() {
        let forward = SidecarResult {
            sysctls: vec![
                "sysctl.a=1".to_string(),
                "sysctl.b=2".to_string(),
                "sysctl.c=3".to_string(),
            ],
            kargs: vec![
                "karg_alpha".to_string(),
                "karg_beta".to_string(),
                "karg_gamma".to_string(),
            ],
            ..SidecarResult::test_fixture()
        };
        let reversed = SidecarResult {
            sysctls: vec![
                "sysctl.c=3".to_string(),
                "sysctl.b=2".to_string(),
                "sysctl.a=1".to_string(),
            ],
            kargs: vec![
                "karg_gamma".to_string(),
                "karg_beta".to_string(),
                "karg_alpha".to_string(),
            ],
            ..SidecarResult::test_fixture()
        };
        assert_eq!(
            sidecar_variant_hash(&forward),
            sidecar_variant_hash(&reversed),
            "reversed-order sysctls/kargs must hash identically — \
             the hash sorts both collections lexically before \
             folding bytes in, matching the set-determines-hash \
             contract documented on `sidecar_variant_hash`. A \
             regression that dropped the sort block would produce \
             distinct hashes and duplicate sidecar files for the \
             same semantic variant.",
        );

        // Permutation check: a partial reorder (sysctls same,
        // kargs reversed) must also collapse. Guards against a
        // partial revert that drops the sort in only one of the
        // two collections.
        let partial = SidecarResult {
            sysctls: forward.sysctls.clone(),
            kargs: reversed.kargs.clone(),
            ..SidecarResult::test_fixture()
        };
        assert_eq!(
            sidecar_variant_hash(&forward),
            sidecar_variant_hash(&partial),
            "kargs-only reversal must still hash identically — \
             partial revert (one of the two sorts dropped) must \
             fail this assertion. Got distinct hashes for: \
             sysctls={:?}, kargs={:?} vs sysctls={:?}, kargs={:?}",
            forward.sysctls,
            forward.kargs,
            partial.sysctls,
            partial.kargs,
        );
    }

    /// `write_skip_sidecar` sibling of
    /// `write_sidecar_variant_hash_is_order_invariant_for_active_flags`.
    /// The canonicalization path is applied at BOTH write sites
    /// (`write_sidecar` for run-to-completion results,
    /// `write_skip_sidecar` for pre-VM-boot skips), so both need
    /// order-invariance coverage — a partial revert that dropped
    /// `canonicalize_active_flags` in just the skip path would
    /// leave the run path covered by the sibling test yet leave
    /// skip-variant hashes order-sensitive, producing duplicate
    /// skip-sidecar files for the same semantic variant under
    /// `stats list` / `stats compare`.
    ///
    /// Pins the same two invariants as the sibling: (1) reversed
    /// flag-order inputs collapse to a single file via normal
    /// overwrite, (2) the surviving on-disk `active_flags` is in
    /// canonical `scenario::flags::ALL` order. Uses a distinct
    /// entry-name prefix (`__skipflagorder_test__`) so the
    /// `find_sidecars_by_prefix` scan doesn't overlap with the
    /// run-path test's fixtures.
    #[test]
    fn write_skip_sidecar_variant_hash_is_order_invariant_for_active_flags() {
        let _lock = lock_env();
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let tmp = tmp_dir.path();
        let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", tmp);

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "__skipflagorder_test__",
            func: dummy,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        };

        // Same flag SET, reversed accumulation order. Mirrors the
        // `llc` / `steal` choice from the run-path sibling so the
        // canonical order (index 0, index 2 in `ALL_DECLS`) is
        // unambiguous.
        let forward = vec!["llc".to_string(), "steal".to_string()];
        let reversed = vec!["steal".to_string(), "llc".to_string()];
        write_skip_sidecar(&entry, &forward).unwrap();
        write_skip_sidecar(&entry, &reversed).unwrap();

        let paths = find_sidecars_by_prefix(tmp, "__skipflagorder_test__-");
        assert_eq!(
            paths.len(),
            1,
            "reversed-order skip-sidecar writes of the same flag \
             SET must collapse to a single canonical filename \
             (overwrite); got {paths:?}. If this fails with \
             `paths.len() == 2`, canonicalization was removed from \
             `write_skip_sidecar` even if the run-path test above \
             still passes — apply `canonicalize_active_flags` in \
             both write sites, not just one.",
        );

        let path = &paths[0];
        let data = std::fs::read_to_string(path).expect("read canonical skip sidecar");
        let loaded: SidecarResult =
            serde_json::from_str(&data).expect("deserialize canonical skip sidecar");
        assert_eq!(
            loaded.active_flags,
            vec!["llc".to_string(), "steal".to_string()],
            "on-disk active_flags of a skip sidecar must be sorted \
             in `scenario::flags::ALL` positional order; got: {:?}",
            loaded.active_flags,
        );
    }

    /// Directly exercises `canonicalize_active_flags` on a mixed
    /// input: known canonical flags AND ad-hoc unknown flags. The
    /// sibling
    /// `write_sidecar_variant_hash_is_order_invariant_for_active_flags`
    /// test pins the known-flag-only case through the full
    /// write-and-read round trip; this unit-level test pins the
    /// composite sort-key contract in isolation so a regression in
    /// the tiebreaker (e.g. dropping the secondary lexical
    /// comparator, reverting to `sort_by_key` with a bare
    /// positional key) fails here with a precise diagnostic,
    /// rather than going undetected until a user trips it with
    /// ad-hoc flags.
    ///
    /// Invariants pinned:
    /// 1. Known flags (members of `scenario::flags::ALL`) always
    ///    appear before unknown flags, regardless of input order.
    /// 2. Known flags are ordered by their position in ALL
    ///    (positional key as primary sort).
    /// 3. Unknown flags are ordered lexically among themselves
    ///    (secondary `&str` comparator). Without the secondary,
    ///    two unknown flags share `usize::MAX` as their positional
    ///    key and stable-sort preserves input order — so reversed
    ///    unknown-flag input would produce reversed output and
    ///    the variant hash would still depend on caller order.
    #[test]
    fn canonicalize_active_flags_orders_unknown_lexically_after_known() {
        // `llc` is `ALL[0]`, so it always wins against unknown
        // flags on the positional key. The two `*_unknown` flags
        // collide at `usize::MAX` and must then be ordered
        // lexically (`aaa_` < `zzz_`).
        let input = vec![
            "zzz_unknown".to_string(),
            "llc".to_string(),
            "aaa_unknown".to_string(),
        ];
        let got = canonicalize_active_flags(&input);
        assert_eq!(
            got,
            vec![
                "llc".to_string(),
                "aaa_unknown".to_string(),
                "zzz_unknown".to_string(),
            ],
            "known flags must sort first by ALL position, unknown \
             flags must sort lexically after; got: {got:?}",
        );

        // Invariance check: reversing the input must produce the
        // same output. Without the lexical secondary the two
        // unknowns would swap, breaking the set-determines-hash
        // property for any variant carrying ad-hoc flags.
        let reversed: Vec<String> = input.into_iter().rev().collect();
        let got_rev = canonicalize_active_flags(&reversed);
        assert_eq!(
            got_rev, got,
            "reversed input must canonicalize to the same output; \
             got: {got_rev:?}, expected: {got:?}",
        );
    }

    #[test]
    fn write_sidecar_variant_hash_distinguishes_work_types() {
        // Two gauntlet variants differing only in work_type must
        // produce distinct sidecar filenames so neither clobbers the
        // other.
        let _lock = lock_env();
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let tmp = tmp_dir.path();
        let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", tmp);

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "__variant_test__",
            func: dummy,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        };
        let vm_result = crate::vmm::VmResult::test_fixture();
        let ok = AssertResult::pass();
        write_sidecar(&entry, &vm_result, &[], &ok, "CpuSpin", &[], &[]).unwrap();
        write_sidecar(&entry, &vm_result, &[], &ok, "YieldHeavy", &[], &[]).unwrap();

        let paths = find_sidecars_by_prefix(tmp, "__variant_test__-");
        assert_eq!(
            paths.len(),
            2,
            "two work_type variants must produce two distinct files, got {paths:?}"
        );
    }

    /// Freeze the `sidecar_variant_hash` wire format to the exact 64-bit
    /// value produced for a representative populated SidecarResult.
    ///
    /// Sidecar filenames embed this hash as a hex suffix; gauntlet
    /// tooling groups variants by it. A silent change — e.g. bumping
    /// `siphasher`, switching keys, or reordering fields fed into the
    /// hasher — would let old-version tooling mis-group new-version
    /// sidecars and vice versa. Pinning the output against a
    /// pre-computed constant catches that drift before it ships.
    ///
    /// Every currently hash-participating field (topology, scheduler,
    /// payload, work_type, active_flags, sysctls, kargs) is set
    /// explicitly; non-participating fields come from
    /// [`SidecarResult::test_fixture`] so unrelated schema growth does
    /// not disturb the constant. If a future change adds a new
    /// hash-participating field to [`sidecar_variant_hash`], add it
    /// here too — otherwise this test silently degrades into a
    /// same-defaults check.
    #[test]
    fn sidecar_variant_hash_stability_populated() {
        // Every currently hash-participating field is spelled out
        // explicitly so a change to `test_fixture` defaults cannot
        // silently shift the pinned constant. If you add a new
        // hash-participating field to `sidecar_variant_hash`, add
        // it here and recompute the expected constant.
        let sc = SidecarResult {
            topology: "1n2l4c1t".to_string(),
            scheduler: "scx-ktstr".to_string(),
            payload: None,
            work_type: "CpuSpin".to_string(),
            active_flags: vec!["llc".to_string(), "steal".to_string()],
            sysctls: vec!["sysctl.kernel.sched_cfs_bandwidth_slice_us=1000".to_string()],
            kargs: vec!["nosmt".to_string()],
            ..SidecarResult::test_fixture()
        };
        // If this assertion trips, the wire format changed. Bumping
        // the expected value is the wrong fix unless you also plan
        // for old sidecars to be regenerated — see the contract on
        // `sidecar_variant_hash`.
        assert_eq!(
            sidecar_variant_hash(&sc),
            0xbc0f38005915a09f,
            "sidecar_variant_hash output drifted — regenerate expected only if \
             the wire format change is intentional and old sidecars are \
             disposable (which they are per ktstr's pre-1.0 stance)",
        );
    }

    /// Pair to [`sidecar_variant_hash_stability_populated`] covering
    /// the empty-collections path. If the inter-collection separator
    /// bytes (0xfe / 0xfd / 0xff) disappear or change, an empty-
    /// flags variant could collide with an empty-sysctls variant
    /// whose kargs start with bytes that happen to match the dropped
    /// separator. Pinning the empty-inputs hash catches separator
    /// regressions.
    #[test]
    fn sidecar_variant_hash_stability_empty_collections() {
        // Every currently hash-participating field is spelled out
        // explicitly so a change to `test_fixture` defaults cannot
        // silently shift the pinned constant. If you add a new
        // hash-participating field to `sidecar_variant_hash`, add
        // it here and recompute the expected constant.
        let sc = SidecarResult {
            topology: "1n1l1c1t".to_string(),
            scheduler: "eevdf".to_string(),
            payload: None,
            work_type: String::new(),
            active_flags: Vec::new(),
            sysctls: Vec::new(),
            kargs: Vec::new(),
            ..SidecarResult::test_fixture()
        };
        assert_eq!(sidecar_variant_hash(&sc), 0x1b61394511b42e01);
    }

    /// Two sidecars that differ only in `payload` must produce
    /// distinct variant hashes so gauntlet runs composing the same
    /// scheduler with different primary payloads (FIO vs STRESS_NG)
    /// don't clobber each other's files.
    #[test]
    fn sidecar_variant_hash_distinguishes_payload() {
        // `none` relies on [`SidecarResult::test_fixture`] defaulting
        // `payload` to `None`. If that default changes, the absent-vs-
        // present comparison below collapses — the assertion below
        // and this comment are intentionally load-bearing.
        let base = SidecarResult::test_fixture;
        let none = base();
        assert!(
            none.payload.is_none(),
            "fixture default for payload must remain None"
        );
        let fio = SidecarResult {
            payload: Some("fio".to_string()),
            ..base()
        };
        let stress = SidecarResult {
            payload: Some("stress-ng".to_string()),
            ..base()
        };
        let h_none = sidecar_variant_hash(&none);
        let h_fio = sidecar_variant_hash(&fio);
        let h_stress = sidecar_variant_hash(&stress);
        assert_ne!(
            h_none, h_fio,
            "absent vs present payload must hash differently",
        );
        assert_ne!(
            h_fio, h_stress,
            "different payload names must hash differently",
        );
    }

    // -- format_verifier_stats tests --

    #[test]
    fn format_verifier_stats_empty() {
        assert!(format_verifier_stats(&[]).is_empty());
    }

    #[test]
    fn format_verifier_stats_no_data() {
        let sc = SidecarResult::test_fixture();
        assert!(format_verifier_stats(&[sc]).is_empty());
    }

    #[test]
    fn format_verifier_stats_table() {
        let sc = SidecarResult {
            verifier_stats: vec![
                crate::monitor::bpf_prog::ProgVerifierStats {
                    name: "dispatch".to_string(),
                    verified_insns: 50000,
                },
                crate::monitor::bpf_prog::ProgVerifierStats {
                    name: "enqueue".to_string(),
                    verified_insns: 30000,
                },
            ],
            ..SidecarResult::test_fixture()
        };
        let result = format_verifier_stats(&[sc]);
        assert!(result.contains("BPF VERIFIER STATS"));
        assert!(result.contains("dispatch"));
        assert!(result.contains("enqueue"));
        assert!(result.contains("50000"));
        assert!(result.contains("30000"));
        assert!(result.contains("total verified insns: 80000"));
        assert!(!result.contains("WARNING"));
    }

    #[test]
    fn format_verifier_stats_warning() {
        let sc = SidecarResult {
            verifier_stats: vec![crate::monitor::bpf_prog::ProgVerifierStats {
                name: "heavy".to_string(),
                verified_insns: 800000,
            }],
            ..SidecarResult::test_fixture()
        };
        let result = format_verifier_stats(&[sc]);
        assert!(result.contains("WARNING"));
        assert!(result.contains("heavy"));
        assert!(result.contains("80.0%"));
    }

    #[test]
    fn sidecar_verifier_stats_serde_roundtrip() {
        let sc = SidecarResult {
            verifier_stats: vec![crate::monitor::bpf_prog::ProgVerifierStats {
                name: "init".to_string(),
                verified_insns: 5000,
            }],
            ..SidecarResult::test_fixture()
        };
        let json = serde_json::to_string(&sc).unwrap();
        assert!(json.contains("verifier_stats"));
        let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.verifier_stats.len(), 1);
        assert_eq!(loaded.verifier_stats[0].name, "init");
        assert_eq!(loaded.verifier_stats[0].verified_insns, 5000);
    }

    /// Every `Vec` field emits as `"x":[]` when empty rather than
    /// being omitted. Pin the always-emit contract so a regression
    /// that re-adds `skip_serializing_if` on `verifier_stats` is
    /// caught before it ships.
    #[test]
    fn sidecar_verifier_stats_empty_emits_as_empty_array() {
        let sc = SidecarResult::test_fixture();
        let json = serde_json::to_string(&sc).unwrap();
        assert!(
            json.contains("\"verifier_stats\":[]"),
            "empty verifier_stats must emit as `\"verifier_stats\":[]`: {json}",
        );
    }

    #[test]
    fn format_verifier_stats_deduplicates() {
        let vstats = vec![crate::monitor::bpf_prog::ProgVerifierStats {
            name: "dispatch".to_string(),
            verified_insns: 50000,
        }];
        let sc1 = SidecarResult {
            verifier_stats: vstats.clone(),
            ..SidecarResult::test_fixture()
        };
        let sc2 = SidecarResult {
            verifier_stats: vstats,
            ..SidecarResult::test_fixture()
        };
        let result = format_verifier_stats(&[sc1, sc2]);
        // Deduplicated: total should be 50000, not 100000.
        assert!(result.contains("total verified insns: 50000"));
    }

    // -- scheduler_fingerprint --

    #[test]
    fn scheduler_fingerprint_eevdf_empty_extras() {
        // Default scheduler (EEVDF) has no sysctls/kargs; fingerprint
        // returns the display name and two empty vecs.
        let entry = KtstrTestEntry {
            name: "eevdf_test",
            ..KtstrTestEntry::DEFAULT
        };
        let SchedulerFingerprint {
            scheduler: name,
            scheduler_commit: commit,
            sysctls,
            kargs,
        } = scheduler_fingerprint(&entry);
        assert_eq!(name, "eevdf");
        assert!(
            commit.is_none(),
            "Eevdf variant has no userspace binary; \
             scheduler_commit must be None. Got: {commit:?}",
        );
        assert!(sysctls.is_empty());
        assert!(kargs.is_empty());
    }

    #[test]
    fn scheduler_fingerprint_formats_sysctls_with_prefix() {
        use super::super::entry::Sysctl;
        static SYSCTLS: &[Sysctl] = &[
            Sysctl::new("kernel.foo", "1"),
            Sysctl::new("kernel.bar", "yes"),
        ];
        static SCHED: super::super::entry::Scheduler =
            super::super::entry::Scheduler::new("s").sysctls(SYSCTLS);
        static SCHED_PAYLOAD: super::super::payload::Payload =
            super::super::payload::Payload::from_scheduler(&SCHED);
        let entry = KtstrTestEntry {
            name: "s_test",
            scheduler: &SCHED_PAYLOAD,
            ..KtstrTestEntry::DEFAULT
        };
        let SchedulerFingerprint {
            scheduler: name,
            scheduler_commit: _,
            sysctls,
            kargs,
        } = scheduler_fingerprint(&entry);
        assert_eq!(name, "s");
        assert_eq!(
            sysctls,
            vec![
                "sysctl.kernel.foo=1".to_string(),
                "sysctl.kernel.bar=yes".to_string(),
            ]
        );
        assert!(kargs.is_empty());
    }

    #[test]
    fn scheduler_fingerprint_forwards_kargs_verbatim() {
        static SCHED: super::super::entry::Scheduler =
            super::super::entry::Scheduler::new("s").kargs(&["quiet", "splash"]);
        static SCHED_PAYLOAD: super::super::payload::Payload =
            super::super::payload::Payload::from_scheduler(&SCHED);
        let entry = KtstrTestEntry {
            name: "s_test",
            scheduler: &SCHED_PAYLOAD,
            ..KtstrTestEntry::DEFAULT
        };
        let SchedulerFingerprint {
            scheduler: _,
            scheduler_commit: _,
            sysctls,
            kargs,
        } = scheduler_fingerprint(&entry);
        assert_eq!(kargs, vec!["quiet".to_string(), "splash".to_string()]);
        assert!(sysctls.is_empty());
    }

    #[test]
    fn scheduler_fingerprint_uses_display_name_for_discover() {
        use super::super::entry::SchedulerSpec;
        static SCHED: super::super::entry::Scheduler =
            super::super::entry::Scheduler::new("s").binary(SchedulerSpec::Discover("scx_relaxed"));
        static SCHED_PAYLOAD: super::super::payload::Payload =
            super::super::payload::Payload::from_scheduler(&SCHED);
        let entry = KtstrTestEntry {
            name: "rel_test",
            scheduler: &SCHED_PAYLOAD,
            ..KtstrTestEntry::DEFAULT
        };
        let SchedulerFingerprint {
            scheduler: name,
            scheduler_commit: commit,
            sysctls: _,
            kargs: _,
        } = scheduler_fingerprint(&entry);
        assert_eq!(name, "s");
        assert!(
            commit.is_none(),
            "Discover variant currently returns None via \
             `SchedulerSpec::scheduler_commit` — \
             `resolve_scheduler`'s cascade does not guarantee a \
             fresh build, so there is no authoritative source for \
             the scheduler binary's commit and `scheduler_commit` \
             reports None honestly. Got: {commit:?}",
        );
    }

    /// `scheduler_fingerprint` on a binary-kind `Payload`
    /// (constructed via `Payload::binary`) must produce
    /// `commit: None`. The `and_then` chain in `scheduler_fingerprint`
    /// (`entry.scheduler.scheduler_binary().and_then(|s|
    /// s.scheduler_commit())`) relies on `Payload::scheduler_binary`
    /// returning `None` for `PayloadKind::Binary` to short-circuit
    /// the commit lookup — a regression that accidentally returned
    /// `Some(&some_default)` from `scheduler_binary` for
    /// binary-kind payloads would skip this short-circuit and
    /// populate `scheduler_commit` with a value that has nothing
    /// to do with a scheduler. This test pins that short-circuit
    /// end-to-end.
    ///
    /// Complements the `scheduler_commit_*` variant tests on
    /// `SchedulerSpec` itself (which cover the scheduler-kind
    /// branches) by exercising the binary-kind fallthrough that
    /// never touches `SchedulerSpec` at all.
    #[test]
    fn scheduler_fingerprint_binary_payload_has_no_commit() {
        static BINARY_PAYLOAD: super::super::payload::Payload =
            super::super::payload::Payload::binary("bin_test", "some_binary");
        let entry = KtstrTestEntry {
            name: "bin_test",
            scheduler: &BINARY_PAYLOAD,
            ..KtstrTestEntry::DEFAULT
        };
        let SchedulerFingerprint {
            scheduler: name,
            scheduler_commit: commit,
            sysctls,
            kargs,
        } = scheduler_fingerprint(&entry);
        // Per `Payload::scheduler_name`, binary-kind payloads
        // carry the intent-level label `"kernel_default"` — pinning
        // this alongside the None-commit keeps the binary-kind
        // contract visible in one place.
        assert_eq!(
            name, "kernel_default",
            "binary-kind payload must report the intent-level \
             scheduler label; got: {name:?}",
        );
        assert!(
            commit.is_none(),
            "binary-kind payload has no scheduler binary at all — \
             scheduler_commit must be None via the `and_then` \
             short-circuit on `scheduler_binary() == None`. Got: \
             {commit:?}",
        );
        assert!(
            sysctls.is_empty(),
            "binary-kind payload reports no sysctls; got: {sysctls:?}",
        );
        assert!(
            kargs.is_empty(),
            "binary-kind payload reports no kargs; got: {kargs:?}",
        );
    }

    // -- write_skip_sidecar --

    /// `write_skip_sidecar` is the path covered by the ResourceContention
    /// skip branch and any early-exit that bails before `run_ktstr_test_inner`
    /// reaches the VM-run call site. The sidecar must be flagged
    /// `skipped: true, passed: true` so stats tooling that subtracts
    /// skipped runs from pass counts sees a recorded skip instead of
    /// a missing file. This regression guards that contract against a
    /// future change that forgets the passed-true flag or drops skip
    /// sidecars entirely for non-VM early exits.
    #[test]
    fn write_skip_sidecar_records_passed_true_skipped_true() {
        let _lock = lock_env();
        let tmp = std::env::temp_dir().join("ktstr-sidecar-skip-writes-test");
        let _ = std::fs::remove_dir_all(&tmp);
        let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", &tmp);

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "__skip_sidecar_test__",
            func: dummy,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        };
        let active_flags: Vec<String> = vec!["llc".to_string()];
        write_skip_sidecar(&entry, &active_flags).expect("skip sidecar must write");

        let path = find_single_sidecar_by_prefix(&tmp, "__skip_sidecar_test__-");
        let data = std::fs::read_to_string(&path).unwrap();
        let loaded: SidecarResult = serde_json::from_str(&data).unwrap();
        assert_eq!(loaded.test_name, "__skip_sidecar_test__");
        assert!(
            loaded.passed,
            "skip sidecar must set passed=true so the verdict gate does not flip fail",
        );
        assert!(
            loaded.skipped,
            "skip sidecar must set skipped=true so stats tooling excludes from pass count",
        );
        assert_eq!(
            loaded.work_type, "skipped",
            "skip path uses the 'skipped' work_type bucket so grouping keeps the skip distinguishable",
        );
        assert_eq!(loaded.active_flags, active_flags);
        // write_skip_sidecar shares the host-context capture with
        // write_sidecar (same `collect_host_context()` builder line)
        // so skip paths still give `stats compare --runs` a host
        // baseline. A regression that dropped the skip-path capture
        // would leave `host: None` in only the skip bucket, producing
        // silent per-run partial data.
        let host = loaded
            .host
            .as_ref()
            .expect("write_skip_sidecar must populate host field from collect_host_context");
        assert_eq!(host.kernel_name.as_deref(), Some("Linux"));
        // Pair the uname check with a Default-distinguishing field —
        // see `write_sidecar_writes_file` for the rationale. Keeps
        // both the happy-path writer and the skip-path writer guarded
        // against the same default-substitution regression.
        assert!(
            host.kernel_cmdline.is_some(),
            "write_skip_sidecar must capture full HostContext, not Default::default()",
        );
        // Syscall-sourced companion to the filesystem-sourced
        // `kernel_cmdline` check — see `write_sidecar_writes_file`
        // for the two-independent-paths rationale.
        assert!(
            host.kernel_release.is_some(),
            "write_skip_sidecar must capture kernel_release (syscall-sourced)",
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// When the sidecar directory cannot be created (path collision
    /// with a regular file), `write_skip_sidecar` must return `Err`
    /// rather than silently eating the failure. Stats tooling relies
    /// on the error chain to diagnose missing sidecars; a swallowed
    /// error would make skips invisible to post-run analysis.
    #[test]
    fn write_skip_sidecar_returns_err_when_dir_cannot_be_created() {
        let _lock = lock_env();

        // Create a regular file, then try to use it as the sidecar
        // directory. `create_dir_all` fails because the path exists
        // but is not a directory.
        let blocker = std::env::temp_dir().join("ktstr-sidecar-skip-blocker");
        let _ = std::fs::remove_file(&blocker);
        let _ = std::fs::remove_dir_all(&blocker);
        std::fs::write(&blocker, b"not a dir").unwrap();
        let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", &blocker);

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "__skip_sidecar_err_test__",
            func: dummy,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        };
        let result = write_skip_sidecar(&entry, &[]);
        assert!(
            result.is_err(),
            "skip sidecar write must return Err when the target is a regular file",
        );

        let _ = std::fs::remove_file(&blocker);
    }

    // -- sidecar payload + metrics fields --

    /// Empty `payload` / `metrics` serialize as `"payload":null` /
    /// `"metrics":[]` (always-emit symmetric with `host`) rather than
    /// being omitted. Pin the wire shape so a regression that re-adds
    /// `skip_serializing_if` on either field is caught before it
    /// ships, and verify the None/empty round-trip remains correct
    /// under the deserialize-requires contract.
    #[test]
    fn sidecar_payload_and_metrics_always_emit_when_empty() {
        let sc = SidecarResult::test_fixture();
        let json = serde_json::to_string(&sc).unwrap();
        assert!(
            json.contains("\"payload\":null"),
            "empty payload must emit as `\"payload\":null`: {json}",
        );
        assert!(
            json.contains("\"metrics\":[]"),
            "empty metrics must emit as `\"metrics\":[]`: {json}",
        );
        assert!(
            json.contains("\"project_commit\":null"),
            "absent project_commit must emit as `\"project_commit\":null`, \
             not be omitted via `skip_serializing_if`: {json}",
        );
        assert!(
            json.contains("\"kernel_commit\":null"),
            "absent kernel_commit must emit as `\"kernel_commit\":null`, \
             not be omitted via `skip_serializing_if`: {json}",
        );
        let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
        // Exhaustive destructure so a new `Option<_>` / `Vec<_>`
        // field on `SidecarResult` that defaults to `None` / empty
        // forces this test to spell it out and make an
        // always-emit-vs-skip decision at the same time. See
        // [`sidecar_result_roundtrip`] for the same pattern on the
        // populated side — the two together pin the wire contract
        // at both extremes of the default distribution.
        let SidecarResult {
            test_name: _,
            topology: _,
            scheduler: _,
            scheduler_commit,
            project_commit,
            payload,
            metrics,
            passed: _,
            skipped: _,
            stats: _,
            monitor,
            stimulus_events,
            work_type: _,
            active_flags,
            verifier_stats,
            kvm_stats,
            sysctls,
            kargs,
            kernel_version,
            kernel_commit,
            timestamp: _,
            run_id: _,
            host,
            cleanup_duration_ms,
            run_source,
        } = loaded;
        assert!(payload.is_none());
        assert!(metrics.is_empty());
        // The sibling-field defaults on the empty fixture — every
        // nullable must be None and every Vec empty, matching the
        // always-emit invariants that the JSON shape above pins.
        assert!(scheduler_commit.is_none());
        assert!(project_commit.is_none());
        assert!(monitor.is_none());
        assert!(stimulus_events.is_empty());
        assert!(active_flags.is_empty());
        assert!(verifier_stats.is_empty());
        assert!(kvm_stats.is_none());
        assert!(sysctls.is_empty());
        assert!(kargs.is_empty());
        assert!(kernel_version.is_none());
        assert!(kernel_commit.is_none());
        assert!(host.is_none());
        assert!(cleanup_duration_ms.is_none());
        assert!(
            run_source.is_none(),
            "absent run_source must round-trip as None, \
             matching the symmetric serialize/deserialize \
             contract enforced for every other nullable field",
        );
    }

    /// Populated `payload` + `metrics` survive round-trip with the
    /// exact shape stats tooling will consume — one entry per
    /// `ctx.payload(X).run()` call, each carrying its exit code and
    /// any extracted metrics. Regression guard against a future
    /// schema shift that flattens metrics across payloads (which
    /// would lose the per-payload provenance the design requires).
    #[test]
    fn sidecar_payload_and_metrics_roundtrip_populated() {
        use crate::test_support::{Metric, MetricSource, MetricStream, PayloadMetrics, Polarity};
        let pm = PayloadMetrics {
            payload_index: 0,
            metrics: vec![Metric {
                name: "iops".to_string(),
                value: 5000.0,
                polarity: Polarity::HigherBetter,
                unit: "iops".to_string(),
                source: MetricSource::Json,
                stream: MetricStream::Stdout,
            }],
            exit_code: 0,
        };
        let sc = SidecarResult {
            test_name: "fio_run".to_string(),
            topology: "1n1l2c1t".to_string(),
            payload: Some("fio".to_string()),
            metrics: vec![pm],
            ..SidecarResult::test_fixture()
        };
        let json = serde_json::to_string(&sc).unwrap();
        assert!(json.contains("\"payload\":\"fio\""));
        assert!(json.contains("\"metrics\""));
        assert!(json.contains("\"iops\""));
        let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.payload.as_deref(), Some("fio"));
        assert_eq!(loaded.metrics.len(), 1);
        assert_eq!(loaded.metrics[0].exit_code, 0);
        assert_eq!(loaded.metrics[0].metrics.len(), 1);
        assert_eq!(loaded.metrics[0].metrics[0].name, "iops");
        assert_eq!(loaded.metrics[0].metrics[0].value, 5000.0);
        assert_eq!(
            loaded.metrics[0].metrics[0].stream,
            MetricStream::Stdout,
            "metric stream tag must round-trip through sidecar \
             serde; a regression that lost `stream` serialization \
             or deserialized it to a different variant would break \
             review-tooling's stdout-vs-stderr attribution",
        );
    }

    /// `write_sidecar` must populate `payload` from `entry.payload`
    /// so a test declaring a binary payload writes the payload name
    /// into the sidecar even when no payload-metrics have been
    /// threaded in yet. This pins the half-wired state the
    /// follow-up WOs will extend: stats tooling that already groups
    /// by payload name sees the grouping key on the sidecar
    /// immediately.
    #[test]
    fn write_sidecar_records_entry_payload_name() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};

        let _lock = lock_env();
        let tmp = std::env::temp_dir().join("ktstr-sidecar-payload-name-test");
        let _ = std::fs::remove_dir_all(&tmp);
        let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", &tmp);

        static FIO: Payload = Payload {
            name: "fio",
            kind: PayloadKind::Binary("fio"),
            output: OutputFormat::Json,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
            metric_bounds: None,
        };

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "__payload_name_test__",
            func: dummy,
            auto_repro: false,
            payload: Some(&FIO),
            ..KtstrTestEntry::DEFAULT
        };
        let vm_result = crate::vmm::VmResult::test_fixture();
        let ok = AssertResult::pass();
        write_sidecar(&entry, &vm_result, &[], &ok, "CpuSpin", &[], &[]).unwrap();

        let path = find_single_sidecar_by_prefix(&tmp, "__payload_name_test__-");
        let data = std::fs::read_to_string(&path).unwrap();
        let loaded: SidecarResult = serde_json::from_str(&data).unwrap();
        assert_eq!(loaded.payload.as_deref(), Some("fio"));
        assert!(
            loaded.metrics.is_empty(),
            "metrics stay empty until a Ctx-level accumulator lands",
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// `write_sidecar` must forward the `payload_metrics` slice
    /// into `SidecarResult.metrics` unmodified — once the
    /// follow-up Ctx-accumulator WO lands, stats tooling will see
    /// every `ctx.payload(X).run()` invocation's output in order.
    #[test]
    fn write_sidecar_forwards_payload_metrics_slice() {
        use crate::test_support::{Metric, MetricSource, MetricStream, PayloadMetrics, Polarity};

        let _lock = lock_env();
        let tmp = std::env::temp_dir().join("ktstr-sidecar-metrics-slice-test");
        let _ = std::fs::remove_dir_all(&tmp);
        let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", &tmp);

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "__metrics_slice_test__",
            func: dummy,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        };
        let vm_result = crate::vmm::VmResult::test_fixture();
        let ok = AssertResult::pass();
        let metrics = vec![
            PayloadMetrics {
                payload_index: 0,
                metrics: vec![Metric {
                    name: "iops".to_string(),
                    value: 1200.0,
                    polarity: Polarity::HigherBetter,
                    unit: "iops".to_string(),
                    source: MetricSource::Json,
                    stream: MetricStream::Stdout,
                }],
                exit_code: 0,
            },
            PayloadMetrics {
                payload_index: 1,
                metrics: vec![],
                exit_code: 2,
            },
        ];
        write_sidecar(&entry, &vm_result, &[], &ok, "CpuSpin", &[], &metrics).unwrap();

        let path = find_single_sidecar_by_prefix(&tmp, "__metrics_slice_test__-");
        let data = std::fs::read_to_string(&path).unwrap();
        let loaded: SidecarResult = serde_json::from_str(&data).unwrap();
        assert_eq!(loaded.metrics.len(), 2);
        assert_eq!(loaded.metrics[0].exit_code, 0);
        assert_eq!(loaded.metrics[0].metrics.len(), 1);
        assert_eq!(loaded.metrics[0].metrics[0].name, "iops");
        assert_eq!(loaded.metrics[1].exit_code, 2);
        assert!(loaded.metrics[1].metrics.is_empty());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// `write_skip_sidecar` must also carry `entry.payload` through
    /// so a ResourceContention or early-skip on a payload-carrying
    /// test still records the payload name. Missing this would
    /// drop skipped runs out of payload-grouped stats.
    #[test]
    fn write_skip_sidecar_records_entry_payload_name() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};

        let _lock = lock_env();
        let tmp = std::env::temp_dir().join("ktstr-sidecar-skip-payload-test");
        let _ = std::fs::remove_dir_all(&tmp);
        let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", &tmp);

        static STRESS: Payload = Payload {
            name: "stress-ng",
            kind: PayloadKind::Binary("stress-ng"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
            metric_bounds: None,
        };

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "__skip_payload_name_test__",
            func: dummy,
            auto_repro: false,
            payload: Some(&STRESS),
            ..KtstrTestEntry::DEFAULT
        };
        write_skip_sidecar(&entry, &[]).unwrap();

        let path = find_single_sidecar_by_prefix(&tmp, "__skip_payload_name_test__-");
        let data = std::fs::read_to_string(&path).unwrap();
        let loaded: SidecarResult = serde_json::from_str(&data).unwrap();
        assert_eq!(loaded.payload.as_deref(), Some("stress-ng"));
        assert!(loaded.skipped);
        assert!(
            loaded.metrics.is_empty(),
            "skip path never accumulates metrics"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// `host` is deliberately excluded from `sidecar_variant_hash`:
    /// two gauntlet variants run on different hosts must collapse
    /// into the same hash bucket so downstream stats tooling groups
    /// them together. If a future change accidentally folds
    /// `HostContext` into the hash, this test catches it before
    /// the run-key split reaches on-disk sidecars.
    #[test]
    fn sidecar_variant_hash_excludes_host_context() {
        use crate::host_context::HostContext;
        let populated = HostContext {
            cpu_model: Some("Example CPU".to_string()),
            cpu_vendor: Some("GenuineExample".to_string()),
            total_memory_kb: Some(16_384_000),
            hugepages_total: Some(0),
            hugepages_free: Some(0),
            hugepages_size_kb: Some(2048),
            thp_enabled: Some("always [madvise] never".to_string()),
            thp_defrag: Some("[always] defer madvise never".to_string()),
            sched_tunables: None,
            online_cpus: Some(8),
            numa_nodes: Some(2),
            cpufreq_governor: std::collections::BTreeMap::new(),
            kernel_name: Some("Linux".to_string()),
            kernel_release: Some("6.11.0".to_string()),
            arch: Some("x86_64".to_string()),
            kernel_cmdline: Some("preempt=lazy".to_string()),
            heap_state: None,
        };
        let without_host = SidecarResult {
            topology: "1n1l2c1t".to_string(),
            ..SidecarResult::test_fixture()
        };
        let with_host = SidecarResult {
            topology: "1n1l2c1t".to_string(),
            host: Some(populated),
            ..SidecarResult::test_fixture()
        };
        assert_eq!(
            sidecar_variant_hash(&without_host),
            sidecar_variant_hash(&with_host),
            "host context must not influence variant hash",
        );
    }

    /// `scheduler_commit` is metadata, not a variant discriminator:
    /// two gauntlet runs differing only in the recorded scheduler
    /// commit (e.g. same variant re-run after a scheduler rebuild)
    /// must share one hash bucket so `stats compare` treats them as
    /// the same semantic variant. If a future change folds
    /// `scheduler_commit` into `sidecar_variant_hash`, this test
    /// catches it before the run-key split reaches on-disk sidecars
    /// and splits previously-comparable runs. Mirrors
    /// `sidecar_variant_hash_excludes_host_context`.
    #[test]
    fn sidecar_variant_hash_excludes_scheduler_commit() {
        let without_commit = SidecarResult {
            topology: "1n1l2c1t".to_string(),
            scheduler_commit: None,
            ..SidecarResult::test_fixture()
        };
        let with_commit = SidecarResult {
            topology: "1n1l2c1t".to_string(),
            scheduler_commit: Some("0000000000000000000000000000000000000000".to_string()),
            ..SidecarResult::test_fixture()
        };
        assert_eq!(
            sidecar_variant_hash(&without_commit),
            sidecar_variant_hash(&with_commit),
            "scheduler_commit must not influence variant hash — \
             runs of the same semantic variant on different \
             scheduler-binary builds must remain comparable by \
             `stats compare`",
        );
    }

    /// `project_commit` is metadata, not a variant discriminator:
    /// two gauntlet runs differing only in the recorded ktstr
    /// project commit (e.g. same variant re-run after a `git pull`
    /// of the harness, or run from two ktstr clones at different
    /// HEADs) must share one hash bucket so `stats compare`
    /// treats them as the same semantic variant. If a future
    /// change folds `project_commit` into `sidecar_variant_hash`,
    /// this test catches it before the run-key split reaches
    /// on-disk sidecars and splits previously-comparable runs.
    /// Mirrors `sidecar_variant_hash_excludes_scheduler_commit` —
    /// the same exclusion rationale applies to both metadata
    /// fields.
    ///
    /// Three cases pinned: (1) None vs Some, (2) two distinct
    /// populated values, (3) clean Some vs `-dirty` Some. Without
    /// the populated×populated case, a regression that XOR'd
    /// project_commit's bytes into the hash would still pass the
    /// None vs Some case if the empty-input contribution happened
    /// to be zero; the third case guards specifically against a
    /// change that distinguished only the dirty bit.
    #[test]
    fn sidecar_variant_hash_excludes_project_commit() {
        let without_commit = SidecarResult {
            topology: "1n1l2c1t".to_string(),
            project_commit: None,
            ..SidecarResult::test_fixture()
        };
        let with_commit = SidecarResult {
            topology: "1n1l2c1t".to_string(),
            project_commit: Some("abcdef1-dirty".to_string()),
            ..SidecarResult::test_fixture()
        };
        assert_eq!(
            sidecar_variant_hash(&without_commit),
            sidecar_variant_hash(&with_commit),
            "project_commit must not influence variant hash — \
             None vs Some(...) case",
        );

        let with_commit_a = SidecarResult {
            topology: "1n1l2c1t".to_string(),
            project_commit: Some("abc1234".to_string()),
            ..SidecarResult::test_fixture()
        };
        let with_commit_b = SidecarResult {
            topology: "1n1l2c1t".to_string(),
            project_commit: Some("def5678".to_string()),
            ..SidecarResult::test_fixture()
        };
        assert_eq!(
            sidecar_variant_hash(&with_commit_a),
            sidecar_variant_hash(&with_commit_b),
            "project_commit must not influence variant hash — \
             two distinct populated commits case (catches XOR-style \
             regressions where None and one specific Some happen to \
             collide)",
        );

        let with_commit_clean = SidecarResult {
            topology: "1n1l2c1t".to_string(),
            project_commit: Some("abc1234".to_string()),
            ..SidecarResult::test_fixture()
        };
        let with_commit_dirty = SidecarResult {
            topology: "1n1l2c1t".to_string(),
            project_commit: Some("abc1234-dirty".to_string()),
            ..SidecarResult::test_fixture()
        };
        assert_eq!(
            sidecar_variant_hash(&with_commit_clean),
            sidecar_variant_hash(&with_commit_dirty),
            "project_commit must not influence variant hash — \
             clean vs `-dirty` of the same hex case (catches a \
             regression that distinguished only the dirty bit)",
        );
    }

    /// `kernel_commit` is metadata, not a variant discriminator:
    /// two gauntlet runs differing only in the recorded kernel
    /// source-tree commit (e.g. same variant re-run after a
    /// `git pull` of the kernel tree, or the same release rebuilt
    /// on top of a WIP patch) must share one hash bucket so
    /// `stats compare` treats them as the same semantic variant.
    /// If a future change folds `kernel_commit` into
    /// `sidecar_variant_hash`, this test catches it before the
    /// run-key split reaches on-disk sidecars and splits
    /// previously-comparable runs. Mirrors
    /// `sidecar_variant_hash_excludes_project_commit` /
    /// `sidecar_variant_hash_excludes_scheduler_commit` — the
    /// same exclusion rationale applies to all three metadata
    /// commit fields.
    ///
    /// Three cases pinned: (1) None vs Some, (2) two distinct
    /// populated values, (3) clean Some vs `-dirty` Some. Without
    /// the populated×populated case, a regression that XOR'd
    /// kernel_commit's bytes into the hash would still pass the
    /// None vs Some case if the empty-input contribution happened
    /// to be zero; the third case guards specifically against a
    /// change that distinguished only the dirty bit.
    #[test]
    fn sidecar_variant_hash_excludes_kernel_commit() {
        let without_commit = SidecarResult {
            topology: "1n1l2c1t".to_string(),
            kernel_commit: None,
            ..SidecarResult::test_fixture()
        };
        let with_commit = SidecarResult {
            topology: "1n1l2c1t".to_string(),
            kernel_commit: Some("abcdef1-dirty".to_string()),
            ..SidecarResult::test_fixture()
        };
        assert_eq!(
            sidecar_variant_hash(&without_commit),
            sidecar_variant_hash(&with_commit),
            "kernel_commit must not influence variant hash — \
             None vs Some(...) case",
        );

        let with_commit_a = SidecarResult {
            topology: "1n1l2c1t".to_string(),
            kernel_commit: Some("abc1234".to_string()),
            ..SidecarResult::test_fixture()
        };
        let with_commit_b = SidecarResult {
            topology: "1n1l2c1t".to_string(),
            kernel_commit: Some("def5678".to_string()),
            ..SidecarResult::test_fixture()
        };
        assert_eq!(
            sidecar_variant_hash(&with_commit_a),
            sidecar_variant_hash(&with_commit_b),
            "kernel_commit must not influence variant hash — \
             two distinct populated commits case (catches XOR-style \
             regressions where None and one specific Some happen to \
             collide)",
        );

        let with_commit_clean = SidecarResult {
            topology: "1n1l2c1t".to_string(),
            kernel_commit: Some("abc1234".to_string()),
            ..SidecarResult::test_fixture()
        };
        let with_commit_dirty = SidecarResult {
            topology: "1n1l2c1t".to_string(),
            kernel_commit: Some("abc1234-dirty".to_string()),
            ..SidecarResult::test_fixture()
        };
        assert_eq!(
            sidecar_variant_hash(&with_commit_clean),
            sidecar_variant_hash(&with_commit_dirty),
            "kernel_commit must not influence variant hash — \
             clean vs `-dirty` of the same hex case (catches a \
             regression that distinguished only the dirty bit)",
        );
    }

    /// `run_source` (the run-environment provenance tag) must not
    /// influence the variant hash. Two runs of the same semantic
    /// variant — one from a developer machine (`run_source: "local"`)
    /// and one from a CI runner (`run_source: "ci"`) — must produce
    /// the same sidecar filename so `compare_partitions` can diff them
    /// across the CI/local boundary without the run-source tag
    /// shattering them into per-environment buckets. Mirrors the
    /// commit-exclusion tests: covers `None` vs `Some("local")`,
    /// `Some("local")` vs `Some("ci")`, and `Some("ci")` vs
    /// `Some("archive")` so a regression that distinguished only
    /// one specific tag pair would still be caught.
    #[test]
    fn sidecar_variant_hash_excludes_run_source() {
        let none = SidecarResult {
            topology: "1n1l2c1t".to_string(),
            run_source: None,
            ..SidecarResult::test_fixture()
        };
        let local = SidecarResult {
            topology: "1n1l2c1t".to_string(),
            run_source: Some(SIDECAR_RUN_SOURCE_LOCAL.to_string()),
            ..SidecarResult::test_fixture()
        };
        assert_eq!(
            sidecar_variant_hash(&none),
            sidecar_variant_hash(&local),
            "run_source must not influence variant hash — None vs \
             Some(\"local\") case",
        );

        let ci = SidecarResult {
            topology: "1n1l2c1t".to_string(),
            run_source: Some(SIDECAR_RUN_SOURCE_CI.to_string()),
            ..SidecarResult::test_fixture()
        };
        assert_eq!(
            sidecar_variant_hash(&local),
            sidecar_variant_hash(&ci),
            "run_source must not influence variant hash — \
             Some(\"local\") vs Some(\"ci\") case (catches XOR-style \
             regressions where two specific tags happen to collide)",
        );

        let archive = SidecarResult {
            topology: "1n1l2c1t".to_string(),
            run_source: Some(SIDECAR_RUN_SOURCE_ARCHIVE.to_string()),
            ..SidecarResult::test_fixture()
        };
        assert_eq!(
            sidecar_variant_hash(&ci),
            sidecar_variant_hash(&archive),
            "run_source must not influence variant hash — \
             Some(\"ci\") vs Some(\"archive\") case",
        );
    }

    /// `detect_run_source` reads `KTSTR_CI` and returns `"ci"`
    /// when set non-empty, `"local"` otherwise. Empty-string env
    /// values count as unset so a defensively-cleared variable
    /// does not accidentally classify a developer run as CI.
    #[test]
    fn detect_run_source_routes_on_ktstr_ci_env() {
        let _lock = lock_env();
        let _restore = EnvVarGuard::remove(KTSTR_CI_ENV);
        assert_eq!(
            detect_run_source(),
            Some(SIDECAR_RUN_SOURCE_LOCAL.to_string()),
            "unset KTSTR_CI must classify as `local`",
        );
        let _set_empty = EnvVarGuard::set(KTSTR_CI_ENV, std::path::Path::new(""));
        assert_eq!(
            detect_run_source(),
            Some(SIDECAR_RUN_SOURCE_LOCAL.to_string()),
            "empty-string KTSTR_CI must classify as `local` so a \
             defensively-cleared variable does not accidentally \
             flip the tag",
        );
        drop(_set_empty);
        let _set_one = EnvVarGuard::set(KTSTR_CI_ENV, std::path::Path::new("1"));
        assert_eq!(
            detect_run_source(),
            Some(SIDECAR_RUN_SOURCE_CI.to_string()),
            "non-empty KTSTR_CI must classify as `ci`",
        );
    }

    /// `apply_archive_source_override` rewrites every sidecar's
    /// `run_source` to `"archive"` regardless of the prior value, so
    /// that `--dir`-loaded pools surface uniformly under the
    /// archive bucket. Pin both branches: a populated `run_source`
    /// (`"local"` / `"ci"`) is overwritten, and `None` is
    /// rewritten to `Some("archive")` rather than left as `None`.
    #[test]
    fn apply_archive_source_override_rewrites_every_entry() {
        let mut pool = vec![
            SidecarResult {
                run_source: Some(SIDECAR_RUN_SOURCE_LOCAL.to_string()),
                ..SidecarResult::test_fixture()
            },
            SidecarResult {
                run_source: Some(SIDECAR_RUN_SOURCE_CI.to_string()),
                ..SidecarResult::test_fixture()
            },
            SidecarResult {
                run_source: None,
                ..SidecarResult::test_fixture()
            },
        ];
        apply_archive_source_override(&mut pool);
        for sc in &pool {
            assert_eq!(
                sc.run_source.as_deref(),
                Some(SIDECAR_RUN_SOURCE_ARCHIVE),
                "every sidecar in a --dir pool must surface as \
                 archive after override",
            );
        }
    }

    /// A `SidecarResult` carrying a fully populated `HostContext`
    /// round-trips through serde_json without losing fields.
    /// Struct-level `PartialEq` on `HostContext` makes one
    /// `assert_eq!(host, ctx)` cover every field, so a future
    /// change that breaks composition between the outer
    /// `SidecarResult` and the embedded `HostContext` is caught at
    /// the seam without needing a per-field assertion.
    #[test]
    fn sidecar_result_roundtrip_with_populated_host_context() {
        use crate::host_context::HostContext;
        let mut tunables = std::collections::BTreeMap::new();
        tunables.insert("sched_migration_cost_ns".to_string(), "500000".to_string());
        let ctx = HostContext {
            cpu_model: Some("Example CPU".to_string()),
            cpu_vendor: Some("GenuineExample".to_string()),
            total_memory_kb: Some(16_384_000),
            hugepages_total: Some(4),
            hugepages_free: Some(2),
            hugepages_size_kb: Some(2048),
            thp_enabled: Some("always [madvise] never".to_string()),
            thp_defrag: Some("[always] defer madvise never".to_string()),
            sched_tunables: Some(tunables),
            online_cpus: Some(8),
            numa_nodes: Some(2),
            cpufreq_governor: std::collections::BTreeMap::new(),
            kernel_name: Some("Linux".to_string()),
            kernel_release: Some("6.11.0".to_string()),
            arch: Some("x86_64".to_string()),
            kernel_cmdline: Some("preempt=lazy isolcpus=1-3".to_string()),
            heap_state: Some(crate::host_heap::HostHeapState::test_fixture()),
        };
        let sc = SidecarResult {
            topology: "1n1l2c1t".to_string(),
            host: Some(ctx.clone()),
            ..SidecarResult::test_fixture()
        };
        let json = serde_json::to_string(&sc).unwrap();
        let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
        let host = loaded.host.expect("host must round-trip");
        assert_eq!(host, ctx);
    }

    /// Every sidecar produced within a single ktstr run records the
    /// SAME host context — all writers call
    /// [`crate::host_context::collect_host_context`], which
    /// memoises the static subset in a process-global `OnceLock`
    /// (`STATIC_HOST_INFO`) and re-reads the dynamic subset from
    /// the same `/proc` / `/sys` sources on every call. Runtime
    /// drift in the captured struct across sidecars in one run
    /// would mean one of two bad outcomes:
    ///   - a regression in the static memoisation (cache key / init
    ///     closure), producing per-call distinct values for fields
    ///     that cannot change across a process lifetime (uname,
    ///     CPU model, NUMA topology);
    ///   - a test concurrently mutating a dynamic field
    ///     (`thp_enabled`, `sched_tunables`, hugepage reservations)
    ///     while another test writes a sidecar, which would be a
    ///     test-isolation bug — every in-tree test treats host
    ///     tunables as read-only.
    ///
    /// This test runs a deterministic N-iteration loop (NOT a
    /// proptest-style property sampler — there is no input-space
    /// shrinker and no random seed; the same N calls with the same
    /// ordering produce the same comparisons every run) of
    /// back-to-back `collect_host_context()` calls simulating the
    /// per-test sidecar drumbeat of a gauntlet run. Every resulting
    /// `host` field must compare equal across all N samples. The
    /// sibling [`crate::host_context`] tests already pin
    /// `collect_host_context` internal stability; this test pins
    /// the SIDECAR surface so a regression that threaded a partial
    /// context through `write_sidecar` / `write_skip_sidecar`
    /// would fail here even if `collect_host_context` itself
    /// stayed stable.
    ///
    /// Bounded N=8: enough iterations to catch intermittent drift
    /// without bloating the test runtime — `collect_host_context`
    /// does ~20 sysfs/procfs reads per call, so the cost scales
    /// linearly and must stay modest.
    ///
    /// `#[cfg(target_os = "linux")]`: `collect_host_context` only
    /// reads meaningful data on Linux — on other hosts every field
    /// is `None` and the equality would trivially hold without
    /// exercising the contract.
    #[cfg(target_os = "linux")]
    #[test]
    fn sidecars_in_a_run_carry_identical_host_context() {
        const N: usize = 8;
        let samples: Vec<crate::host_context::HostContext> = (0..N)
            .map(|_| crate::host_context::collect_host_context())
            .collect();
        let first = samples
            .first()
            .expect("N > 0 samples must produce at least one host context");

        // Fields expected to stay STRICTLY equal — either memoised
        // in STATIC_HOST_INFO (uname, CPU, memory, topology) or
        // effectively reboot-static (kernel_cmdline). A regression
        // that broke the cache or mis-read /proc would diverge here.
        for (i, s) in samples.iter().enumerate() {
            assert_eq!(
                s.kernel_name, first.kernel_name,
                "sidecar {i}: kernel_name drifted from first sample",
            );
            assert_eq!(
                s.kernel_release, first.kernel_release,
                "sidecar {i}: kernel_release drifted — STATIC_HOST_INFO cache broken?",
            );
            assert_eq!(
                s.arch, first.arch,
                "sidecar {i}: arch drifted — STATIC_HOST_INFO cache broken?",
            );
            assert_eq!(
                s.cpu_model, first.cpu_model,
                "sidecar {i}: cpu_model drifted — STATIC_HOST_INFO cache broken?",
            );
            assert_eq!(
                s.cpu_vendor, first.cpu_vendor,
                "sidecar {i}: cpu_vendor drifted — STATIC_HOST_INFO cache broken?",
            );
            assert_eq!(
                s.total_memory_kb, first.total_memory_kb,
                "sidecar {i}: total_memory_kb drifted — STATIC_HOST_INFO cache broken?",
            );
            assert_eq!(
                s.hugepages_size_kb, first.hugepages_size_kb,
                "sidecar {i}: hugepages_size_kb drifted — STATIC_HOST_INFO cache broken?",
            );
            assert_eq!(
                s.online_cpus, first.online_cpus,
                "sidecar {i}: online_cpus drifted — STATIC_HOST_INFO cache broken?",
            );
            assert_eq!(
                s.numa_nodes, first.numa_nodes,
                "sidecar {i}: numa_nodes drifted — STATIC_HOST_INFO cache broken?",
            );
            assert_eq!(
                s.kernel_cmdline, first.kernel_cmdline,
                "sidecar {i}: kernel_cmdline drifted — only a reboot can change it",
            );
        }

        // Dynamic fields are allowed to vary in value under
        // concurrent sysctl/THP/hugepage twiddles (see the sibling
        // `collect_host_context_dynamic_subset_is_stable_across_calls`
        // test for the rationale), but the PRESENCE of each field
        // must stay consistent — a sidecar that suddenly loses the
        // THP row means the collector silently degraded, which
        // stats tooling would read as "no THP data on that host"
        // rather than the truth ("collector broke").
        for (i, s) in samples.iter().enumerate() {
            assert_eq!(
                s.hugepages_total.is_some(),
                first.hugepages_total.is_some(),
                "sidecar {i}: hugepages_total presence flipped across sidecars",
            );
            assert_eq!(
                s.hugepages_free.is_some(),
                first.hugepages_free.is_some(),
                "sidecar {i}: hugepages_free presence flipped across sidecars",
            );
            assert_eq!(
                s.thp_enabled.is_some(),
                first.thp_enabled.is_some(),
                "sidecar {i}: thp_enabled presence flipped across sidecars",
            );
            assert_eq!(
                s.thp_defrag.is_some(),
                first.thp_defrag.is_some(),
                "sidecar {i}: thp_defrag presence flipped across sidecars",
            );
            assert_eq!(
                s.sched_tunables.is_some(),
                first.sched_tunables.is_some(),
                "sidecar {i}: sched_tunables presence flipped across sidecars",
            );
        }
    }

    // -- detect_project_commit branch coverage --
    //
    // The five branches probed below cover every shape `detect_commit_at`
    // can produce: a clean repo (Some(hex)), a dirty tracked-file
    // worktree (Some(hex-dirty)), a non-git directory (None), an unborn
    // HEAD (None), and a submodule-entry tree with the submodule
    // unchecked-out (Some(hex), no -dirty). Fixtures use gix directly
    // — no `git` shell-out — so the tests reflect the same library the
    // production probe uses.

    /// Construct a single-blob tree at `dir`, populate the index from it,
    /// write the file content into the worktree, and return the new
    /// HEAD commit's id. After this helper the repo is fully clean:
    /// HEAD-tree == index == worktree.
    ///
    /// `committer_or_set_generic_fallback` is invoked because the test
    /// process inherits no `user.name|email` git config and the
    /// commit/ref-edit path requires a non-empty signature; the
    /// fallback writes "no name configured" / "noEmailAvailable@…"
    /// into the in-memory config snapshot, which is sufficient to
    /// produce a syntactically valid commit object.
    fn init_clean_repo_with_file(dir: &std::path::Path) -> gix::ObjectId {
        let mut repo = gix::init(dir).expect("gix::init");
        let _ = repo
            .committer_or_set_generic_fallback()
            .expect("committer fallback");
        let blob_id: gix::ObjectId = repo.write_blob(b"original\n").expect("write blob").detach();
        let tree = gix::objs::Tree {
            entries: vec![gix::objs::tree::Entry {
                mode: gix::objs::tree::EntryKind::Blob.into(),
                filename: "file.txt".into(),
                oid: blob_id,
            }],
        };
        let tree_id: gix::ObjectId = repo.write_object(&tree).expect("write tree").detach();
        let commit_id: gix::ObjectId = repo
            .commit("HEAD", "init", tree_id, std::iter::empty::<gix::ObjectId>())
            .expect("commit")
            .detach();
        // Populate the index from the tree and persist it so the
        // tree-vs-index check sees no staged drift, then write the
        // worktree file so the index-vs-worktree check sees no
        // unstaged drift.
        let mut idx = repo.index_from_tree(&tree_id).expect("index_from_tree");
        idx.write(gix::index::write::Options::default())
            .expect("write index");
        std::fs::write(dir.join("file.txt"), b"original\n").expect("write worktree file");
        commit_id
    }

    /// Clean repo: HEAD reachable, no staged or worktree diffs. The
    /// short-hash matches `head.to_hex_with_len(7)`, exactly the same
    /// shape `detect_commit_at` formats with — pinning the literal
    /// also confirms the 7-char truncation is honored end-to-end (a
    /// future refactor that swapped to `format!("{}").chars().take(8)`
    /// would silently break the cross-run grouping that stats tooling
    /// relies on).
    #[test]
    fn detect_project_commit_clean_repo_returns_short_hash() {
        let tmp = tempfile::TempDir::new().unwrap();
        let head = init_clean_repo_with_file(tmp.path());
        let result = super::detect_commit_at(tmp.path()).expect("clean repo must yield Some");
        assert_eq!(
            result.len(),
            7,
            "clean result must be a 7-char hex hash, got {result:?}"
        );
        assert!(
            !result.contains('-'),
            "clean result must not carry a -dirty suffix, got {result:?}"
        );
        assert!(
            result.chars().all(|c| c.is_ascii_hexdigit()),
            "clean result must be pure hex, got {result:?}"
        );
        assert_eq!(
            result,
            head.to_hex_with_len(7).to_string(),
            "clean result must match the HEAD short hash exactly"
        );
    }

    /// Dirty tracked-file worktree: HEAD reachable, index matches
    /// HEAD, but worktree diverges from the index. The result must
    /// carry the `-dirty` suffix per the `index_worktree` leg of the
    /// dirt probe.
    #[test]
    fn detect_project_commit_dirty_repo_appends_dirty_suffix() {
        let tmp = tempfile::TempDir::new().unwrap();
        let head = init_clean_repo_with_file(tmp.path());
        // Mutate the tracked file so index-vs-worktree diverges.
        std::fs::write(tmp.path().join("file.txt"), b"modified\n").unwrap();
        let result = super::detect_commit_at(tmp.path()).expect("dirty repo must yield Some");
        let expected_prefix = head.to_hex_with_len(7).to_string();
        assert_eq!(
            result,
            format!("{expected_prefix}-dirty"),
            "dirty result must be {expected_prefix:?} + -dirty suffix"
        );
    }

    /// `repo_is_dirty` returns `Some(false)` for a clean repo. Pins
    /// the contract that the helper distinguishes "I checked, it's
    /// clean" from "I couldn't check" (`None`), so future callers
    /// that need that distinction get reliable signal. The
    /// callthrough from `detect_commit_at` collapses both via
    /// `unwrap_or(false)`, so this test covers a branch the
    /// end-to-end `detect_project_commit_clean_repo_returns_short_hash`
    /// test cannot pin.
    #[test]
    fn repo_is_dirty_clean_repo_returns_some_false() {
        let tmp = tempfile::TempDir::new().unwrap();
        init_clean_repo_with_file(tmp.path());
        let repo = gix::open(tmp.path()).expect("gix::open clean repo");
        assert_eq!(
            super::repo_is_dirty(&repo),
            Some(false),
            "clean repo must yield Some(false)"
        );
    }

    /// `repo_is_dirty` returns `Some(true)` when the worktree
    /// diverges from the index. Pins the index-vs-worktree leg of
    /// the cascade independently of the suffix-formatting logic in
    /// `detect_commit_at`.
    #[test]
    fn repo_is_dirty_dirty_worktree_returns_some_true() {
        let tmp = tempfile::TempDir::new().unwrap();
        init_clean_repo_with_file(tmp.path());
        std::fs::write(tmp.path().join("file.txt"), b"modified\n").unwrap();
        let repo = gix::open(tmp.path()).expect("gix::open dirty repo");
        assert_eq!(
            super::repo_is_dirty(&repo),
            Some(true),
            "dirty worktree must yield Some(true)"
        );
    }

    /// Non-git directory: `gix::discover` walks the parent chain of
    /// the tempdir all the way up; if a parent happens to be a git
    /// repo (e.g. tests run from inside a checkout), `discover`
    /// resolves to that ancestor. To pin the "no repo" branch we have
    /// to break the parent walk, which is impossible from inside a
    /// tempdir nested under a git checkout — the test instead asserts
    /// that no repo is discoverable from the system temp root, which
    /// is reliably outside any project repo.
    #[test]
    fn detect_project_commit_non_git_returns_none() {
        // Use a fresh tempdir directly under /tmp so no parent in the
        // walk is itself a git repo. The TempDir's path is unique per
        // run, so concurrent tests do not collide.
        let tmp = tempfile::TempDir::new_in(std::env::temp_dir()).unwrap();
        // Sanity: discover from this path must fail before we trust
        // the test's None expectation. If a future change makes
        // /tmp/* sit inside a discoverable repo (extremely unlikely
        // on POSIX hosts) the assert here surfaces the violation
        // before the function-under-test assertion below.
        assert!(
            gix::discover(tmp.path()).is_err(),
            "tempdir {} must not resolve to any ancestor git repo",
            tmp.path().display()
        );
        let result = super::detect_commit_at(tmp.path());
        assert!(
            result.is_none(),
            "non-git directory must yield None, got {result:?}"
        );
    }

    /// Unborn HEAD: `gix::init` produces a repo whose HEAD points at a
    /// branch that has not been written to yet. `head_id()` returns
    /// Err on this state; `detect_commit_at` returns None.
    #[test]
    fn detect_project_commit_unborn_head_returns_none() {
        let tmp = tempfile::TempDir::new().unwrap();
        let _repo = gix::init(tmp.path()).expect("gix::init");
        let result = super::detect_commit_at(tmp.path());
        assert!(
            result.is_none(),
            "unborn HEAD must yield None, got {result:?}"
        );
    }

    /// Concurrent invocation stability: the probe is read-only across
    /// the gix layer, so N parallel calls against the same repo must
    /// all return the same result. Failure here would indicate a
    /// thread-safety regression in either gix or our usage of it.
    #[test]
    fn detect_project_commit_concurrent_calls_agree() {
        let tmp = tempfile::TempDir::new().unwrap();
        init_clean_repo_with_file(tmp.path());
        let path = tmp.path();
        let baseline =
            super::detect_commit_at(path).expect("baseline single-thread call must yield Some");

        const THREADS: usize = 8;
        let results = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(THREADS);
            for _ in 0..THREADS {
                handles.push(scope.spawn(|| super::detect_commit_at(path)));
            }
            handles
                .into_iter()
                .map(|h| h.join().expect("thread join"))
                .collect::<Vec<_>>()
        });
        for (i, r) in results.iter().enumerate() {
            assert_eq!(
                r.as_deref(),
                Some(baseline.as_str()),
                "thread {i} disagreed with baseline {baseline:?}: got {r:?}"
            );
        }
    }

    /// Submodule false-positive guard: an uninitialized submodule (a
    /// gitlinks tree+index entry whose checked-out subdirectory has no
    /// `.git` artifact yet) must NOT trip the dirty probe.
    /// `detect_commit_at` configures `Submodule::Given { ignore: All,
    /// .. }` precisely so a parent repo cloned without
    /// `--recurse-submodules` does not get erroneously tagged `-dirty`
    /// for every sidecar.
    ///
    /// The fixture writes a tree containing a `.gitmodules` blob (the
    /// submodule registration gix needs to recognise the gitlinks
    /// entry as a submodule rather than a phantom directory) plus a
    /// `Commit`-mode tree entry pointing at an arbitrary OID. The
    /// worktree contains the `.gitmodules` file and an EMPTY `submod/`
    /// directory — modelling a parent that was cloned without
    /// `--recurse-submodules`. The probe must still report clean.
    #[test]
    fn detect_project_commit_submodule_uninit_is_clean() {
        let tmp = tempfile::TempDir::new().unwrap();
        let mut repo = gix::init(tmp.path()).expect("gix::init");
        let _ = repo
            .committer_or_set_generic_fallback()
            .expect("committer fallback");

        // A submodule reference needs both a `.gitmodules` registration
        // (so gix recognises the gitlinks entry as a submodule, not a
        // phantom file) and the gitlinks tree entry itself. The
        // submodule directory is INTENTIONALLY left absent from the
        // worktree, which is the "uninitialized" state the production
        // probe must tolerate.
        let gitmodules_content = b"\
[submodule \"submod\"]\n\
\tpath = submod\n\
\turl = https://example.invalid/submod.git\n";
        let gitmodules_blob: gix::ObjectId = repo
            .write_blob(gitmodules_content)
            .expect("write .gitmodules blob")
            .detach();
        // Any 20-byte OID is a syntactically valid commit reference
        // from the tree's perspective. The null id keeps the fixture
        // self-contained — no dependency on an actual submodule commit
        // having been written.
        let null_commit_id = gix::ObjectId::null(gix::hash::Kind::Sha1);
        let tree = gix::objs::Tree {
            entries: vec![
                gix::objs::tree::Entry {
                    mode: gix::objs::tree::EntryKind::Blob.into(),
                    filename: ".gitmodules".into(),
                    oid: gitmodules_blob,
                },
                gix::objs::tree::Entry {
                    mode: gix::objs::tree::EntryKind::Commit.into(),
                    filename: "submod".into(),
                    oid: null_commit_id,
                },
            ],
        };
        let tree_id: gix::ObjectId = repo.write_object(&tree).expect("write tree").detach();
        let head: gix::ObjectId = repo
            .commit("HEAD", "init", tree_id, std::iter::empty::<gix::ObjectId>())
            .expect("commit")
            .detach();
        let mut idx = repo.index_from_tree(&tree_id).expect("index_from_tree");
        idx.write(gix::index::write::Options::default())
            .expect("write index");
        // Materialize the .gitmodules blob in the worktree so the
        // index-vs-worktree leg sees no diff for that file. Create an
        // empty `submod/` directory to model a parent that was cloned
        // with `--no-recurse-submodules`: the gitlinks entry is in the
        // tree and index, the directory exists in the worktree, but no
        // `.git` artifact lives inside it (the submodule is
        // unintialized).
        std::fs::write(tmp.path().join(".gitmodules"), gitmodules_content)
            .expect("write .gitmodules worktree");
        std::fs::create_dir(tmp.path().join("submod")).expect("create submod dir");

        let result =
            super::detect_commit_at(tmp.path()).expect("submodule repo must still yield Some");
        assert_eq!(
            result,
            head.to_hex_with_len(7).to_string(),
            "uninitialized submodule must not trigger -dirty suffix"
        );
    }

    // -- detect_kernel_commit branch coverage --
    //
    // Mirror the `detect_project_commit` branch matrix for the
    // kernel-tree probe. The implementations are nearly identical
    // except for `gix::open` (NOT `gix::discover`) so the parent
    // walk does not surface; the tests pin the open-vs-discover
    // shape explicitly.

    /// Clean kernel repo: HEAD reachable, no staged or worktree
    /// diffs. `detect_kernel_commit` returns the 7-char short
    /// hash.
    #[test]
    fn detect_kernel_commit_clean_repo_returns_short_hash() {
        let tmp = tempfile::TempDir::new().unwrap();
        let head = init_clean_repo_with_file(tmp.path());
        let result = super::detect_kernel_commit(tmp.path()).expect("clean repo must yield Some");
        assert_eq!(
            result.len(),
            7,
            "clean result must be a 7-char hex hash, got {result:?}"
        );
        assert!(
            !result.contains('-'),
            "clean result must not carry a -dirty suffix, got {result:?}"
        );
        assert!(
            result.chars().all(|c| c.is_ascii_hexdigit()),
            "clean result must be pure hex, got {result:?}"
        );
        assert_eq!(
            result,
            head.to_hex_with_len(7).to_string(),
            "clean result must match the HEAD short hash exactly"
        );
    }

    /// Dirty tracked-file worktree: HEAD reachable, index matches
    /// HEAD, but worktree diverges. The result must carry the
    /// `-dirty` suffix.
    #[test]
    fn detect_kernel_commit_dirty_repo_appends_dirty_suffix() {
        let tmp = tempfile::TempDir::new().unwrap();
        let head = init_clean_repo_with_file(tmp.path());
        std::fs::write(tmp.path().join("file.txt"), b"modified\n").unwrap();
        let result = super::detect_kernel_commit(tmp.path()).expect("dirty repo must yield Some");
        let expected_prefix = head.to_hex_with_len(7).to_string();
        assert_eq!(
            result,
            format!("{expected_prefix}-dirty"),
            "dirty result must be {expected_prefix:?} + -dirty suffix"
        );
    }

    /// Non-git directory: `detect_kernel_commit` uses `gix::open`,
    /// NOT `gix::discover`. Open requires `kernel_dir` to BE the
    /// repo root, so a non-git directory yields None even when an
    /// ancestor IS a git repo. This is the critical behavioural
    /// difference from `detect_project_commit`: the kernel
    /// directory is explicit, not walked-up.
    ///
    /// Reproduces the failure mode `gix::discover` would trip
    /// (parent walk resolves to ktstr's repo when the user passes
    /// a non-git subdir as KTSTR_KERNEL): a literal subdirectory
    /// of a real git tempdir, NOT initialized as its own repo,
    /// must still yield None for the kernel probe.
    #[test]
    fn detect_kernel_commit_non_git_directory_returns_none() {
        let parent = tempfile::TempDir::new().unwrap();
        // Parent IS a git repo — discover() would walk up and find
        // it from any subdir.
        init_clean_repo_with_file(parent.path());
        let nested = parent.path().join("not_a_repo");
        std::fs::create_dir(&nested).expect("create nested non-git subdir");
        // Pin the precondition: discover() WOULD succeed from this
        // path because the parent is a git repo. If `detect_kernel_commit`
        // accidentally used discover instead of open, it would
        // surface the parent's HEAD here — which is exactly the
        // wrong-kernel-commit-recorded bug we want to prevent.
        assert!(
            gix::discover(&nested).is_ok(),
            "gix::discover must succeed from the nested path (parent IS a repo) — \
             this precondition validates that detect_kernel_commit's open-vs-discover \
             choice is the correct one for the test scenario",
        );
        let result = super::detect_kernel_commit(&nested);
        assert!(
            result.is_none(),
            "non-git directory must yield None — `detect_kernel_commit` uses \
             `gix::open` (NOT `gix::discover`), so the parent's HEAD must \
             NOT leak through. Got {result:?}",
        );
    }

    /// Unborn HEAD: `gix::init` produces a repo whose HEAD points at a
    /// branch that has not been written to yet. `head_id()` returns
    /// Err on this state; `detect_kernel_commit` returns None.
    #[test]
    fn detect_kernel_commit_unborn_head_returns_none() {
        let tmp = tempfile::TempDir::new().unwrap();
        let _repo = gix::init(tmp.path()).expect("gix::init");
        let result = super::detect_kernel_commit(tmp.path());
        assert!(
            result.is_none(),
            "unborn HEAD must yield None, got {result:?}"
        );
    }

    /// Submodule false-positive guard, mirroring
    /// `detect_project_commit_submodule_uninit_is_clean` — an
    /// uninitialized submodule must NOT trip the dirty probe in
    /// the kernel-tree shape either. Kernel trees commonly carry
    /// submodules (e.g. `.git` worktrees pointing to lib stubs)
    /// without those subdirectories being checked out, and a
    /// false-positive `-dirty` would shatter every sidecar's
    /// kernel_commit into a unique bucket.
    #[test]
    fn detect_kernel_commit_submodule_uninit_is_clean() {
        let tmp = tempfile::TempDir::new().unwrap();
        let mut repo = gix::init(tmp.path()).expect("gix::init");
        let _ = repo
            .committer_or_set_generic_fallback()
            .expect("committer fallback");

        let gitmodules_content = b"\
[submodule \"submod\"]\n\
\tpath = submod\n\
\turl = https://example.invalid/submod.git\n";
        let gitmodules_blob: gix::ObjectId = repo
            .write_blob(gitmodules_content)
            .expect("write .gitmodules blob")
            .detach();
        let null_commit_id = gix::ObjectId::null(gix::hash::Kind::Sha1);
        let tree = gix::objs::Tree {
            entries: vec![
                gix::objs::tree::Entry {
                    mode: gix::objs::tree::EntryKind::Blob.into(),
                    filename: ".gitmodules".into(),
                    oid: gitmodules_blob,
                },
                gix::objs::tree::Entry {
                    mode: gix::objs::tree::EntryKind::Commit.into(),
                    filename: "submod".into(),
                    oid: null_commit_id,
                },
            ],
        };
        let tree_id: gix::ObjectId = repo.write_object(&tree).expect("write tree").detach();
        let head: gix::ObjectId = repo
            .commit("HEAD", "init", tree_id, std::iter::empty::<gix::ObjectId>())
            .expect("commit")
            .detach();
        let mut idx = repo.index_from_tree(&tree_id).expect("index_from_tree");
        idx.write(gix::index::write::Options::default())
            .expect("write index");
        std::fs::write(tmp.path().join(".gitmodules"), gitmodules_content)
            .expect("write .gitmodules worktree");
        std::fs::create_dir(tmp.path().join("submod")).expect("create submod dir");

        let result =
            super::detect_kernel_commit(tmp.path()).expect("submodule repo must still yield Some");
        assert_eq!(
            result,
            head.to_hex_with_len(7).to_string(),
            "uninitialized submodule must not trigger -dirty suffix"
        );
    }

    /// `detect_project_commit` memoizes its probe behind a
    /// process-wide [`std::sync::OnceLock`] (declared on the
    /// function body). Two consecutive calls in the same process
    /// must therefore return identical [`Option<String>`] results
    /// — the first call seeds the cache with a probe of cwd; the
    /// second collapses to a `Clone` of the cached entry.
    ///
    /// The OnceLock is process-global and writes during the FIRST
    /// call observed by the test process — that may be this test
    /// or any sibling that ran earlier, since the cache survives
    /// across test functions in a single binary. Either way, the
    /// public-API contract this test pins is "consecutive calls
    /// agree", which holds whether the cache is hot from a
    /// previous test or warmed by the first call here.
    ///
    /// `Option<String>::None` (cwd outside any git repo) memoizes
    /// the same way as `Some` per the function's own commentary
    /// — repeating the failing probe yields the same `None`. The
    /// test does not constrain whether the result is Some or None
    /// because the cwd at test-runner launch is environmental;
    /// equality across the two calls is the testable contract.
    #[test]
    fn detect_project_commit_memoizes_across_consecutive_calls() {
        let first = super::detect_project_commit();
        let second = super::detect_project_commit();
        assert_eq!(
            first, second,
            "consecutive detect_project_commit calls must return \
             identical Option<String> via the OnceLock cache; \
             got first={first:?}, second={second:?}",
        );
        // Also pin against a third call to catch a regression that
        // re-probes on every non-first call (e.g. one that read
        // the OnceLock but bypassed it on the return path).
        let third = super::detect_project_commit();
        assert_eq!(
            first, third,
            "third detect_project_commit call must still match the \
             first; got first={first:?}, third={third:?}",
        );
    }

    /// `detect_kernel_commit` memoizes its probe behind a
    /// process-wide [`std::sync::Mutex<HashMap<PathBuf,
    /// Option<String>>>`] keyed on the input path. Two
    /// consecutive calls with the SAME path must return identical
    /// results — the first call seeds the cache; the second
    /// returns a clone of the cached entry without re-probing.
    ///
    /// Uses a fresh tempdir so the cache key is unique to this
    /// test (no collision with other test functions in the same
    /// binary). The hashmap key is `PathBuf::to_path_buf()` of
    /// `&Path`, so a stable path argument across calls produces
    /// a cache hit on the second invocation.
    #[test]
    fn detect_kernel_commit_memoizes_across_consecutive_calls_same_path() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let head = init_clean_repo_with_file(tmp.path());
        let expected = head.to_hex_with_len(7).to_string();

        let first = super::detect_kernel_commit(tmp.path());
        let second = super::detect_kernel_commit(tmp.path());
        let third = super::detect_kernel_commit(tmp.path());

        assert_eq!(
            first.as_deref(),
            Some(expected.as_str()),
            "first call must return the clean short hash {expected:?}; \
             got {first:?}",
        );
        assert_eq!(
            first, second,
            "consecutive detect_kernel_commit calls with the same \
             path must agree via the Mutex<HashMap> cache; got \
             first={first:?}, second={second:?}",
        );
        assert_eq!(
            first, third,
            "third detect_kernel_commit call with the same path must \
             still match; got first={first:?}, third={third:?}",
        );
    }

    /// `detect_kernel_commit`'s path-keyed cache must not
    /// cross-contaminate between distinct kernel directories. Two
    /// fresh tempdirs with different HEADs (different blob
    /// content → different tree → different commit OID) must each
    /// return their OWN HEAD short hash. A regression that, e.g.,
    /// keyed the cache on a prefix or a hash-collision-prone
    /// derivation would surface here as one of the two paths
    /// returning the OTHER path's HEAD.
    ///
    /// Mixed call interleaving (`a`, `b`, `a`, `b`) catches a
    /// regression that overwrites the entry on every call rather
    /// than inserting per-key.
    #[test]
    fn detect_kernel_commit_distinct_paths_do_not_cross_contaminate() {
        let tmp_a = tempfile::TempDir::new().expect("tempdir A");
        let tmp_b = tempfile::TempDir::new().expect("tempdir B");

        // Distinct HEADs: write a different blob in each repo so
        // the resulting commit OIDs differ. The blob bytes alone
        // determine the tree OID via gix; identical blobs would
        // yield identical commit OIDs and defeat the test.
        let head_a = init_clean_repo_with_file(tmp_a.path());
        // Overwrite the helper's "original\n" content with a
        // distinct payload, then re-commit so HEAD diverges from
        // tmp_b's. We can't reuse `init_clean_repo_with_file`
        // verbatim because that would commit the same content.
        let mut repo_b = gix::init(tmp_b.path()).expect("gix::init B");
        let _ = repo_b
            .committer_or_set_generic_fallback()
            .expect("committer fallback B");
        let blob_id_b: gix::ObjectId = repo_b
            .write_blob(b"different\n")
            .expect("write blob B")
            .detach();
        let tree_b = gix::objs::Tree {
            entries: vec![gix::objs::tree::Entry {
                mode: gix::objs::tree::EntryKind::Blob.into(),
                filename: "file.txt".into(),
                oid: blob_id_b,
            }],
        };
        let tree_id_b: gix::ObjectId = repo_b.write_object(&tree_b).expect("write tree B").detach();
        let head_b: gix::ObjectId = repo_b
            .commit(
                "HEAD",
                "init B",
                tree_id_b,
                std::iter::empty::<gix::ObjectId>(),
            )
            .expect("commit B")
            .detach();
        let mut idx_b = repo_b
            .index_from_tree(&tree_id_b)
            .expect("index_from_tree B");
        idx_b
            .write(gix::index::write::Options::default())
            .expect("write index B");
        std::fs::write(tmp_b.path().join("file.txt"), b"different\n")
            .expect("write worktree file B");

        let expected_a = head_a.to_hex_with_len(7).to_string();
        let expected_b = head_b.to_hex_with_len(7).to_string();
        assert_ne!(
            expected_a, expected_b,
            "fixture precondition: the two repos must have distinct \
             HEADs for this test to mean anything; got a={expected_a} \
             b={expected_b}",
        );

        // Interleave the calls: a, b, a, b. A regression that
        // overwrote the cache on each insert (instead of inserting
        // per-key) would surface here as the second `a` call
        // returning B's hash, or the second `b` returning A's.
        let a1 = super::detect_kernel_commit(tmp_a.path());
        let b1 = super::detect_kernel_commit(tmp_b.path());
        let a2 = super::detect_kernel_commit(tmp_a.path());
        let b2 = super::detect_kernel_commit(tmp_b.path());

        assert_eq!(
            a1.as_deref(),
            Some(expected_a.as_str()),
            "first call against path A must return A's HEAD short \
             hash {expected_a:?}; got {a1:?}",
        );
        assert_eq!(
            b1.as_deref(),
            Some(expected_b.as_str()),
            "first call against path B must return B's HEAD short \
             hash {expected_b:?}; got {b1:?}",
        );
        assert_eq!(
            a1, a2,
            "second call against path A must match the first \
             (cache hit on the A entry); got a1={a1:?}, a2={a2:?}",
        );
        assert_eq!(
            b1, b2,
            "second call against path B must match the first \
             (cache hit on the B entry, NOT contaminated by A); \
             got b1={b1:?}, b2={b2:?}",
        );
        assert_ne!(
            a2, b2,
            "after interleaved calls, A and B must STILL hold \
             distinct values — a regression that lost per-key \
             distinction would equate them; got a2={a2:?}, b2={b2:?}",
        );
    }

    /// `detect_kernel_commit` canonicalizes its cache key so two
    /// path spellings that resolve to the same on-disk repo share
    /// one cache entry. Without canonicalization a symlink alias
    /// would re-run the gix-open + dirt-walk on every call,
    /// defeating the memoization the cache exists to provide.
    ///
    /// Behavioral proof: prime the cache against the canonical
    /// (real) path of a CLEAN repo, then mutate the worktree so a
    /// re-probe would surface `-dirty`, then call via a symlink
    /// alias. With canonicalization the alias canonicalizes to
    /// the real path, hits the cached CLEAN entry, and returns
    /// the no-`-dirty` value. Without canonicalization the alias
    /// keys the cache under its literal path, misses, re-probes,
    /// and surfaces the new dirt as `*-dirty`.
    ///
    /// The cache deliberately does NOT invalidate mid-process
    /// (per the `KERNEL_COMMIT_CACHE` doc-comment); the
    /// stale-on-purpose cached return is the load-bearing signal
    /// that proves the symlink hit the canonicalized entry.
    ///
    /// Unix-only — `std::os::unix::fs::symlink` is gated.
    #[cfg(unix)]
    #[test]
    fn detect_kernel_commit_canonicalizes_symlink_aliases() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let real = tmp.path().join("real");
        std::fs::create_dir(&real).expect("mkdir real");
        let head = init_clean_repo_with_file(&real);

        // Sibling symlink pointing at `real`. Both paths live
        // inside `tmp` so TempDir's drop cleans up everything.
        let alias = tmp.path().join("alias");
        std::os::unix::fs::symlink(&real, &alias).expect("symlink alias -> real");

        // Prime the cache via the canonical path. The entry is
        // now memoized under `real.canonicalize()` with the clean
        // short hash.
        let real_clean =
            super::detect_kernel_commit(&real).expect("clean canonical-path probe must yield Some");
        assert_eq!(
            real_clean,
            head.to_hex_with_len(7).to_string(),
            "fixture precondition: canonical-path probe must return \
             the clean 7-char head hash; got {real_clean:?}",
        );

        // Introduce dirt — any cache-bypass re-probe would now
        // observe `-dirty`. The cached entry deliberately does
        // not invalidate, so the symlink call (if it canonicalizes
        // correctly) returns the stale clean value.
        std::fs::write(real.join("file.txt"), b"modified-after-prime\n")
            .expect("dirty the worktree");

        // Call via the symlink alias. With canonicalization, the
        // alias canonicalizes to the real path and hits the
        // cached clean entry. Without it, the alias misses, re-
        // probes, and surfaces the new dirt as `*-dirty`.
        let alias_result =
            super::detect_kernel_commit(&alias).expect("alias-path probe must yield Some");
        assert!(
            !alias_result.ends_with("-dirty"),
            "alias call must hit the cached pre-dirt entry — a \
             `-dirty` suffix proves the alias bypassed the cache \
             and re-probed the now-dirty repo, which is the \
             regression this test guards against. got {alias_result:?}",
        );
        assert_eq!(
            alias_result, real_clean,
            "alias call must return the EXACT cached clean value \
             from the canonical-path probe; got alias={alias_result:?}, \
             cached={real_clean:?}",
        );
    }

    /// Helper for `resolve_kernel_source_dir_with_cache` tests:
    /// build a [`KernelMetadata`] for a Local-source entry. Used
    /// across the fallback-scan and tarball-priority tests.
    fn local_metadata_with_source_tree(
        version: &str,
        source_tree_path: std::path::PathBuf,
    ) -> crate::cache::KernelMetadata {
        crate::cache::KernelMetadata::new(
            crate::cache::KernelSource::Local {
                source_tree_path: Some(source_tree_path),
                git_hash: None,
            },
            std::env::consts::ARCH.to_string(),
            "bzImage".to_string(),
            "2026-04-26T00:00:00Z".to_string(),
        )
        .with_version(Some(version.to_string()))
        .with_config_hash(Some("abc123".to_string()))
        .with_ktstr_kconfig_hash(Some("def456".to_string()))
    }

    /// Helper: build a fake kernel image file under `dir` and
    /// return its path. Cache `store()` requires an existing image
    /// file to copy into the entry directory.
    fn create_fake_image_in(dir: &std::path::Path) -> std::path::PathBuf {
        let image = dir.join("bzImage");
        std::fs::write(&image, b"fake kernel image").expect("write fake image");
        image
    }

    /// Tarball-shaped lookup hit yields the entry's source_tree_path
    /// directly when the entry is a Local source. Pins the fast
    /// path of the Version arm before exercising the fallback scan.
    #[test]
    fn resolve_kernel_source_dir_with_cache_version_tarball_key_local_source() {
        let cache_root = tempfile::TempDir::new().expect("cache tempdir");
        let cache = crate::cache::CacheDir::with_root(cache_root.path().to_path_buf());
        let src = tempfile::TempDir::new().expect("src tempdir");
        let image_dir = tempfile::TempDir::new().expect("image tempdir");
        let image = create_fake_image_in(image_dir.path());

        let arch = std::env::consts::ARCH;
        let key = format!("6.14.2-tarball-{arch}-kc{}", crate::cache_key_suffix());
        let meta = local_metadata_with_source_tree("6.14.2", src.path().to_path_buf());
        cache
            .store(&key, &crate::cache::CacheArtifacts::new(&image), &meta)
            .expect("store cache entry");

        let id = crate::kernel_path::KernelId::Version("6.14.2".to_string());
        let resolved = super::resolve_kernel_source_dir_with_cache(&id, &cache);
        assert_eq!(
            resolved.as_deref(),
            Some(src.path()),
            "tarball-shaped Local entry must resolve via direct lookup",
        );
    }

    /// Fallback scan: the tarball-shaped key is absent, but a
    /// non-tarball cache entry (e.g. one stored under a `local-`
    /// or git-shaped key) carries a matching version + Local
    /// source_tree_path. The Version arm must find it via the
    /// list-and-match fallback. This is the bug fix for #58.
    #[test]
    fn resolve_kernel_source_dir_with_cache_version_falls_back_to_scan_for_local() {
        let cache_root = tempfile::TempDir::new().expect("cache tempdir");
        let cache = crate::cache::CacheDir::with_root(cache_root.path().to_path_buf());
        let src = tempfile::TempDir::new().expect("src tempdir");
        let image_dir = tempfile::TempDir::new().expect("image tempdir");
        let image = create_fake_image_in(image_dir.path());

        // Store under a non-tarball key shape — mimics a build
        // driven by `--kernel /path/to/linux`.
        let key = format!(
            "local-deadbee-{arch}-kc{suffix}",
            arch = std::env::consts::ARCH,
            suffix = crate::cache_key_suffix(),
        );
        let meta = local_metadata_with_source_tree("6.14.2", src.path().to_path_buf());
        cache
            .store(&key, &crate::cache::CacheArtifacts::new(&image), &meta)
            .expect("store cache entry");

        let id = crate::kernel_path::KernelId::Version("6.14.2".to_string());
        let resolved = super::resolve_kernel_source_dir_with_cache(&id, &cache);
        assert_eq!(
            resolved.as_deref(),
            Some(src.path()),
            "fallback scan must find a Local entry by version when \
             the tarball-shaped lookup misses",
        );
    }

    /// Fallback scan must SKIP non-Local entries even when the
    /// version matches. A Tarball or Git entry has no on-disk
    /// source tree to probe, so iterating past it to find the
    /// Local sibling (or returning `None` when no Local exists)
    /// is the correct behavior.
    #[test]
    fn resolve_kernel_source_dir_with_cache_version_skips_non_local_in_fallback() {
        let cache_root = tempfile::TempDir::new().expect("cache tempdir");
        let cache = crate::cache::CacheDir::with_root(cache_root.path().to_path_buf());
        let image_dir = tempfile::TempDir::new().expect("image tempdir");
        let image = create_fake_image_in(image_dir.path());

        // Store ONE entry: a tarball-source entry under a non-
        // tarball cache-key shape (so the direct lookup misses and
        // we hit the fallback scan). Version matches the query but
        // source is Tarball, so resolve must yield None.
        let key = format!(
            "weird-key-{arch}-kc{suffix}",
            arch = std::env::consts::ARCH,
            suffix = crate::cache_key_suffix(),
        );
        let meta = crate::cache::KernelMetadata::new(
            crate::cache::KernelSource::Tarball,
            std::env::consts::ARCH.to_string(),
            "bzImage".to_string(),
            "2026-04-26T00:00:00Z".to_string(),
        )
        .with_version(Some("6.14.2".to_string()))
        .with_config_hash(Some("abc123".to_string()))
        .with_ktstr_kconfig_hash(Some("def456".to_string()));
        cache
            .store(&key, &crate::cache::CacheArtifacts::new(&image), &meta)
            .expect("store cache entry");

        let id = crate::kernel_path::KernelId::Version("6.14.2".to_string());
        let resolved = super::resolve_kernel_source_dir_with_cache(&id, &cache);
        assert!(
            resolved.is_none(),
            "non-Local entries are transient and must not be returned by the fallback scan; got {resolved:?}",
        );
    }

    /// Version mismatch: even a Local entry with source_tree_path
    /// is skipped when its `metadata.version` differs from the
    /// queried version. Pinning this prevents a regression where
    /// the fallback scan returns the first Local entry regardless
    /// of version (collapsing every Version query to the same
    /// path).
    #[test]
    fn resolve_kernel_source_dir_with_cache_version_skips_mismatched_version_in_fallback() {
        let cache_root = tempfile::TempDir::new().expect("cache tempdir");
        let cache = crate::cache::CacheDir::with_root(cache_root.path().to_path_buf());
        let src = tempfile::TempDir::new().expect("src tempdir");
        let image_dir = tempfile::TempDir::new().expect("image tempdir");
        let image = create_fake_image_in(image_dir.path());

        let key = format!(
            "local-deadbee-{arch}-kc{suffix}",
            arch = std::env::consts::ARCH,
            suffix = crate::cache_key_suffix(),
        );
        let meta = local_metadata_with_source_tree("6.13.0", src.path().to_path_buf());
        cache
            .store(&key, &crate::cache::CacheArtifacts::new(&image), &meta)
            .expect("store cache entry");

        let id = crate::kernel_path::KernelId::Version("6.14.2".to_string());
        let resolved = super::resolve_kernel_source_dir_with_cache(&id, &cache);
        assert!(
            resolved.is_none(),
            "Local entry with mismatched version must not be returned; got {resolved:?}",
        );
    }

    /// `KernelId::CacheKey` resolves via direct cache.lookup — no
    /// fallback scan needed because the key already encodes every
    /// detail (source-type prefix, arch, kconfig hash). Pinning
    /// the CacheKey arm against a Local entry stored under that
    /// exact key.
    #[test]
    fn resolve_kernel_source_dir_with_cache_cache_key_direct_lookup_local() {
        let cache_root = tempfile::TempDir::new().expect("cache tempdir");
        let cache = crate::cache::CacheDir::with_root(cache_root.path().to_path_buf());
        let src = tempfile::TempDir::new().expect("src tempdir");
        let image_dir = tempfile::TempDir::new().expect("image tempdir");
        let image = create_fake_image_in(image_dir.path());

        let key = format!(
            "local-deadbee-{arch}-kc{suffix}",
            arch = std::env::consts::ARCH,
            suffix = crate::cache_key_suffix(),
        );
        let meta = local_metadata_with_source_tree("6.14.2", src.path().to_path_buf());
        cache
            .store(&key, &crate::cache::CacheArtifacts::new(&image), &meta)
            .expect("store cache entry");

        let id = crate::kernel_path::KernelId::CacheKey(key);
        let resolved = super::resolve_kernel_source_dir_with_cache(&id, &cache);
        assert_eq!(resolved.as_deref(), Some(src.path()));
    }

    /// CacheKey lookup against a non-Local entry yields None — no
    /// transient source tree to probe.
    #[test]
    fn resolve_kernel_source_dir_with_cache_cache_key_non_local_yields_none() {
        let cache_root = tempfile::TempDir::new().expect("cache tempdir");
        let cache = crate::cache::CacheDir::with_root(cache_root.path().to_path_buf());
        let image_dir = tempfile::TempDir::new().expect("image tempdir");
        let image = create_fake_image_in(image_dir.path());

        let key = format!(
            "main-git-deadbee-{arch}-kc{suffix}",
            arch = std::env::consts::ARCH,
            suffix = crate::cache_key_suffix(),
        );
        let meta = crate::cache::KernelMetadata::new(
            crate::cache::KernelSource::Git {
                git_hash: Some("deadbee".to_string()),
                git_ref: Some("main".to_string()),
            },
            std::env::consts::ARCH.to_string(),
            "bzImage".to_string(),
            "2026-04-26T00:00:00Z".to_string(),
        )
        .with_version(Some("6.14.2".to_string()))
        .with_config_hash(Some("abc123".to_string()))
        .with_ktstr_kconfig_hash(Some("def456".to_string()));
        cache
            .store(&key, &crate::cache::CacheArtifacts::new(&image), &meta)
            .expect("store cache entry");

        let id = crate::kernel_path::KernelId::CacheKey(key);
        let resolved = super::resolve_kernel_source_dir_with_cache(&id, &cache);
        assert!(
            resolved.is_none(),
            "Git source has no persisted source tree; got {resolved:?}",
        );
    }

    /// Empty cache + Version query yields None. Sanity check
    /// against a regression that crashes on an empty entries list.
    #[test]
    fn resolve_kernel_source_dir_with_cache_version_empty_cache_yields_none() {
        let cache_root = tempfile::TempDir::new().expect("cache tempdir");
        let cache = crate::cache::CacheDir::with_root(cache_root.path().to_path_buf());
        let id = crate::kernel_path::KernelId::Version("6.14.2".to_string());
        let resolved = super::resolve_kernel_source_dir_with_cache(&id, &cache);
        assert!(resolved.is_none());
    }
}
