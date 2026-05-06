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
//! - [`format_run_dirname`]: render the
//!   `{kernel}-{project_commit}` leaf name from the resolved
//!   kernel + commit slots, substituting the literal `unknown`
//!   when either probe returned `None` so the dirname stays
//!   filesystem-safe (see the unknown-commit collision
//!   semantics in the runs guide).
//! - [`is_run_directory`]: predicate consumed by run-listing
//!   walkers ([`newest_run_dir`] here, `sorted_run_entries` in
//!   `crate::stats`). Filters non-directories and dotfile
//!   subdirectories (notably the `.locks/` flock-sentinel
//!   subdirectory) so the lock infrastructure cannot pollute
//!   `cargo ktstr stats list` output or claim the "most recent
//!   run" bucket.
//! - [`pre_clear_run_dir_once`]: shallow-wipe `*.ktstr.json` files
//!   in the run directory at the FIRST write of each test
//!   process so a re-run at the same `{kernel}-{project_commit}`
//!   key produces a last-writer-wins snapshot rather than an
//!   append-only archive. Subsequent writes in the same process
//!   are gated by an internal `Mutex<HashSet<PathBuf>>` so only
//!   the first call per key per process clears.
//! - [`acquire_run_dir_flock`]: cross-process `LOCK_EX` on the
//!   per-run-key sentinel
//!   (`{runs_root}/.locks/{key}.lock`) held for the duration of
//!   the pre-clear + serialize + write cycle. Two concurrent
//!   ktstr processes targeting the same key serialize through
//!   this lock so neither tears the other's mid-write
//!   sidecars. The override branch (operator-chosen
//!   `KTSTR_SIDECAR_DIR`) skips the flock for the same reason
//!   it skips pre-clear: the operator owns the directory's
//!   contents.
//! - [`warn_unknown_project_commit_once`]: one-shot stderr warning
//!   on first sidecar write when `detect_project_commit` returns
//!   `None` (test process not in a git repo) so concurrent or
//!   successive non-git runs colliding on `{kernel}-unknown`
//!   surface the disambiguation hint
//!   (`KTSTR_SIDECAR_DIR=…` or place the tree under git) at
//!   first invocation rather than as a silent collision.
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
    /// WorkSpec type label used for post-hoc filtering and A/B comparison
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
    /// `topology="1n1l1c1t"`, `scheduler="eevdf"`, `work_type="SpinWait"`,
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
            work_type: "SpinWait".to_string(),
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
        Err(e) => {
            tracing::warn!(
                dir = %dir.display(),
                error = %e,
                "ktstr_test: collect_sidecars_with_errors cannot read root dir",
            );
            return (sidecars, parse_errors, io_errors);
        }
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
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(
                    dir = %dir.display(),
                    error = %e,
                    "ktstr_test: skipping unreadable DirEntry while collecting sidecars",
                );
                continue;
            }
        };
        let path = entry.path();
        if path.is_dir() {
            subdirs.push(path);
            continue;
        }
        try_load(&path, &mut sidecars, &mut parse_errors, &mut io_errors);
    }
    for sub in subdirs {
        let sub_entries = match std::fs::read_dir(&sub) {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(
                    subdir = %sub.display(),
                    error = %e,
                    "ktstr_test: skipping unreadable subdirectory while collecting sidecars",
                );
                continue;
            }
        };
        for entry in sub_entries {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!(
                        subdir = %sub.display(),
                        error = %e,
                        "ktstr_test: skipping unreadable DirEntry in sidecar subdirectory",
                    );
                    continue;
                }
            };
            try_load(
                &entry.path(),
                &mut sidecars,
                &mut parse_errors,
                &mut io_errors,
            );
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
        Err(e) => {
            tracing::warn!(
                root = %root.display(),
                error = %e,
                "ktstr_test: collect_pool cannot read root; returning empty pool",
            );
            return Vec::new();
        }
    };
    let mut pool = Vec::new();
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(
                    root = %root.display(),
                    error = %e,
                    "ktstr_test: skipping unreadable DirEntry while collecting pool",
                );
                continue;
            }
        };
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
pub fn sidecar_dir() -> PathBuf {
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
    // Per-process memoization of the SUCCESS case only.
    //
    // The cwd is stable for the lifetime of a test process (no
    // caller mutates it), and the project tree's HEAD plus dirty
    // state cannot change underneath us without an explicit user
    // action that's outside the scope of any individual sidecar
    // write. Gauntlet runs invoke this function once per sidecar —
    // thousands of times per process — so caching the resolved
    // hash collapses every post-first successful call to a
    // `Clone`. The probe itself does ~3 syscalls (gix discover +
    // head_id + status) which dominate the sidecar-write critical
    // path; eliminating that cost on the hot path is the only
    // meaningful perf win available here.
    //
    // FAILURE IS NOT CACHED: a `None` probe outcome (no git repo
    // discoverable from cwd, unborn HEAD, transient FS / gix open
    // failure) does NOT seed the OnceLock. A FIRST call from a
    // momentarily-broken context (e.g. a test that swapped CWD via
    // some indirect path before ever calling
    // `detect_project_commit`, or a transient I/O hiccup during
    // `gix::discover`) would otherwise lock in `None` for the
    // rest of the process — every subsequent sidecar would land
    // under `target/ktstr/{kernel}-unknown/` even though the
    // commit IS resolvable from a healthy cwd. Retrying on failure
    // costs the same ~3 syscalls the success case pays once; the
    // re-probe only fires while the answer is still unknown.
    //
    // CACHE DOES NOT INVALIDATE on success: a user who commits /
    // amends / resets the project tree mid-run and expects the
    // new HEAD to surface in subsequent sidecars will see stale
    // values. This is acceptable — the
    // project tree is treated as stable-enough for a single suite
    // run; callers mutating the tree during a run own the
    // consequences.
    static PROJECT_COMMIT: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    if let Some(cached) = PROJECT_COMMIT.get() {
        return Some(cached.clone());
    }
    let cwd = std::env::current_dir().ok()?;
    let probed = detect_commit_at(&cwd)?;
    // `set` on a hot OnceLock is a no-op `Err` — safe to ignore.
    // First successful caller wins; a second concurrent caller's
    // identical hash discards harmlessly.
    let _ = PROJECT_COMMIT.set(probed.clone());
    Some(probed)
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
    // Per-process, path-keyed memoization of the SUCCESS case
    // only. Same rationale as `detect_project_commit`: gauntlet
    // runs invoke this function once per sidecar — thousands of
    // times — and the kernel tree's HEAD plus dirty state cannot
    // change underneath us mid-suite without an explicit user
    // action outside any sidecar's control. The path key handles
    // the fixture-test case where unit tests rotate through
    // synthetic `tempfile::TempDir` kernel paths in the same
    // process; each distinct path memoizes independently.
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
    // FAILURE IS NOT CACHED: a `None` probe outcome (kernel_dir
    // is not a git repo, unborn HEAD, transient `gix::open`
    // failure) does NOT seed the cache. Caching `None` would lock
    // in `unknown` for every subsequent sidecar even after the
    // condition resolves (e.g. a kernel directory that becomes a
    // valid checkout mid-suite, or a flaky FS that recovers).
    // Re-probing on failure costs the same gix-open + dirt-walk
    // the success case pays once; the re-probe only fires while
    // the answer is still unknown for that path.
    //
    // Mutex poisoning recovery: a panic mid-probe could poison
    // the lock; the `unwrap_or_else(|e| e.into_inner())` pattern
    // recovers the guard so a future caller doesn't fail
    // catastrophically. The cached map is just a HashMap of
    // owned strings; no invariant beyond "key→value mapping" can
    // be broken by an interrupted probe.
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::{Mutex, OnceLock};
    static KERNEL_COMMIT_CACHE: OnceLock<Mutex<HashMap<PathBuf, String>>> = OnceLock::new();
    // Canonicalize the cache key so two paths that resolve to the
    // same on-disk directory share one entry. Without this, a
    // symlinked alias (`./linux` symlinked to `/abs/.../linux`)
    // and the resolved target would each populate their own slot,
    // re-running the gix-open + dirt-walk on every alias and
    // defeating the memoization. `canonicalize` resolves symlinks,
    // collapses `..` / `.`, and yields the absolute path the
    // kernel actually lives at. Falls back to the raw path on
    // canonicalize failure (e.g. caller passed a non-existent
    // `kernel_dir`) — gix::open will fail downstream and re-probe
    // each call until the path becomes resolvable.
    let cache_key = kernel_dir
        .canonicalize()
        .unwrap_or_else(|_| kernel_dir.to_path_buf());
    let cache = KERNEL_COMMIT_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let guard = cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(cached) = guard.get(&cache_key) {
            return Some(cached.clone());
        }
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
    if let Some(ref hash) = result {
        let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
        // First successful caller wins; a concurrent caller's
        // identical hash would overwrite harmlessly because
        // success is deterministic for a given (canonicalized
        // path, HEAD, dirty state) tuple.
        guard.insert(cache_key, hash.clone());
    }
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
/// - `KernelId::Path(p)`: probes the path's `metadata.json` first
///   — `cargo-ktstr`'s `--kernel /path/to/linux` resolver routes
///   clean source trees through the cache pipeline (see
///   [`crate::cli::resolve_kernel_dir_to_entry`]) and exports the
///   CACHE ENTRY directory through `KTSTR_KERNEL`, not the
///   literal source tree. When `metadata.json` parses and carries
///   a `KernelSource::Local::source_tree_path`, that path is the
///   underlying source tree and is returned. When parsing fails
///   (the path IS the source tree, the dirty-tree path that
///   skipped the cache store), falls back to using the raw env
///   value verbatim — that path is itself the source tree.
/// - `KernelId::Version(ver)`: looks for a Local cache entry
///   whose `metadata.version == ver` carrying a
///   `source_tree_path`. The tarball-shaped key (`{ver}-tarball-
///   {arch}-kc{suffix}`) is checked first because it is the
///   most-common form a Version-shaped env points at; on miss
///   (or hit yielding `Tarball` / `Git` source, both of which
///   are transient with no on-disk tree to probe), the function
///   falls back to scanning every valid cache entry for a Local
///   match on version. Without this fallback,
///   a cache populated by `kernel build --kernel
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
        KernelId::Path(_) => {
            let p = std::path::Path::new(&raw);
            // Cache-entry layout: `metadata.json` carries the
            // `KernelSource::Local::source_tree_path` recorded at
            // build time. Source-tree layout (dirty path that
            // skipped cache store): no metadata, so the env value
            // IS the source tree. The shared helper handles both.
            crate::cache::recover_local_source_tree(p)
                .or_else(|| Some(std::path::PathBuf::from(&raw)))
        }
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
    let mut sorted_sysctls = sidecar.sysctls.clone();
    sorted_sysctls.sort();
    let mut sorted_kargs = sidecar.kargs.clone();
    sorted_kargs.sort();
    let canonical = serde_json::json!({
        "topology": sidecar.topology,
        "scheduler": sidecar.scheduler,
        "payload": sidecar.payload,
        "work_type": sidecar.work_type,
        "active_flags": sidecar.active_flags,
        "sysctls": sorted_sysctls,
        "kargs": sorted_kargs,
    });
    let bytes = serde_json::to_vec(&canonical).expect("json serialization cannot fail for strings");
    let mut h = SipHasher13::new_with_keys(0, 0);
    h.write(&bytes);
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
/// directory. The override path skips the lock for the same
/// reason it skips pre-clear: operator-chosen directories are
/// owned by the operator, so we do not place a `.locks/` sibling
/// inside (or above) their custom layout.
///
/// PER-FILE ATOMICITY (both branches): the JSON is written to a
/// `<final>.tmp.<pid>.<run_id>` sibling and then `rename(2)`'d into
/// place. POSIX `rename` is atomic for same-directory destinations,
/// so a peer reader (`collect_sidecars`) never observes a partial
/// JSON payload — either the old contents stay or the new contents
/// replace them in one filesystem step. Two concurrent writers that
/// both target the same `{test_name}-{variant_hash}.ktstr.json`
/// (override path: two CI jobs sharing one operator-chosen dir;
/// default path: a torn-write window inside the flock body that the
/// flock would otherwise have to cover) cannot leave a half-written
/// JSON behind — last-rename-wins, both files are individually
/// well-formed. The `.tmp.<pid>.<run_id>` discriminator on the
/// staging name keeps two writers from racing on the same staging
/// path even when their final destinations collide. The flock on
/// the default path remains load-bearing for the pre-clear leg
/// (atomic write only protects the write itself, not the
/// `read_dir + remove_file` walk that pre-clear runs).
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
    // Atomic write: stage into a `.tmp.<pid>.<run_id>` sibling and
    // rename(2) into the final path. `rename` is atomic for
    // same-directory destinations on every filesystem ktstr supports
    // (ext4, btrfs, xfs, tmpfs, overlayfs); a peer reader never
    // observes a partial payload. The staging name carries the pid
    // AND the unique sidecar `run_id` so two writers in the same
    // process targeting identical final paths (e.g. two threads in
    // the budget-test stdout-capture path) cannot stomp each other's
    // staging file before either rename lands. On rename failure the
    // staging file is removed so a partial sidecar does not survive
    // as garbage in the run dir; rename success consumes the staging
    // entry and there is nothing to clean up.
    let pid = std::process::id();
    let staging = dir.join(format!(
        "{}-{:016x}.ktstr.json.tmp.{pid}.{}",
        sidecar.test_name, variant_hash, sidecar.run_id,
    ));
    std::fs::write(&staging, &json)
        .with_context(|| format!("write {label} staging {}", staging.display()))?;
    if let Err(e) = std::fs::rename(&staging, &path) {
        // Best-effort cleanup of the staged payload; ignore the
        // unlink error so the rename failure is what surfaces
        // (the rename error names the actual problem).
        let _ = std::fs::remove_file(&staging);
        return Err(anyhow::Error::from(e).context(format!(
            "rename {label} staging {} -> {}",
            staging.display(),
            path.display(),
        )));
    }
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
    if guard.contains(&cache_key) {
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
    // probe call pattern. The cache insert happens AFTER the wipe
    // completes (rather than before) so a panic mid-wipe does not
    // poison the cache with an entry whose wipe never actually ran.
    // The mutex itself enforces serialization across threads; the
    // entry only records "wipe completed for this dir" and must
    // never be observed without the wipe having succeeded. `guard`
    // is dropped at end-of-scope so the lock release happens after
    // the loop completes.
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            // Two file shapes are wiped per directory entry:
            // - `<test>-<hash>.ktstr.json` (live sidecars from a
            //   prior run sharing this `{kernel}-{project_commit}`
            //   key — see the function-level doc for why
            //   coexistence is the bug pre-clear prevents);
            // - `<test>-<hash>.ktstr.json.tmp.<pid>.<run_id>`
            //   (orphaned staging files from a writer that died
            //   between `write` and `rename` in
            //   `serialize_and_write_sidecar`'s atomic-write path).
            //   Without the staging sweep, every crash mid-write
            //   would leak a `.tmp.…` artifact that
            //   `is_sidecar_filename` excludes (extension is
            //   `<run_id>`, not `json`), so neither
            //   `collect_sidecars` nor the next pre-clear pass
            //   would ever reap them. The flock on the default
            //   path makes wiping in-flight staging files
            //   impossible — a peer writer either holds the
            //   lock (we wait) or is between locks (no in-flight
            //   stage can exist).
            if is_sidecar_filename(&path) || is_sidecar_staging_filename(&path) {
                let _ = std::fs::remove_file(&path);
            }
        }
    }
    // Record completion AFTER the wipe finishes, not before. If a
    // panic interrupts the loop above, the cache remains empty so
    // a subsequent call retries the wipe rather than skipping it
    // on the assumption that a prior call already cleared the dir.
    guard.insert(cache_key);
    drop(guard);
}

/// Predicate: is `path` an atomic-write staging file produced by
/// [`serialize_and_write_sidecar`]?
///
/// True iff the filename matches the `<test>-<hash>.ktstr.json.tmp.…`
/// shape — `is_sidecar_filename` rejects these because the
/// extension is `<run_id>` rather than `json`, so a separate
/// predicate is needed for the [`pre_clear_run_dir_once`] sweep
/// that reaps orphaned staging files. Filename-component check
/// (rather than full-path string) for the same load-bearing reason
/// `is_sidecar_filename` uses `Path::file_name()`: a `.ktstr.json.tmp.`
/// substring inside an ancestor segment must not match.
fn is_sidecar_staging_filename(path: &std::path::Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.contains(".ktstr.json.tmp."))
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
mod tests;
