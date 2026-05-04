//! Per-sidecar `Option`-field absence diagnostic surface.
//!
//! Holds [`explain_sidecar`] (the `cargo ktstr stats explain` entry
//! point), the static catalog ([`SIDECAR_NONE_CATALOG`]) cataloging
//! every `Option<T>` field on
//! [`crate::test_support::SidecarResult`] with cause prose +
//! actionable fix wording, the [`NoneClassification`] tag splitting
//! "expected" from "actionable" gaps, the run-directory walker
//! ([`walk_run_with_stats`]) plus its file-count helper
//! ([`count_sidecar_files`]) and per-walk [`WalkStats`], and both
//! renderers ([`render_explain_sidecar_text`] /
//! [`render_explain_sidecar_json`] with the schema-versioned
//! [`ExplainOutput`] / [`WalkStatsJson`] / [`WalkError`] /
//! [`WalkIoError`] / [`FieldDiagnostic`] shapes).

use std::path::Path;

use anyhow::{Result, bail};

use super::dispatch::suggest_closest_run_key;

/// Whether a `None` value on a [`crate::test_support::SidecarResult`]
/// `Option` field is the expected steady-state shape (e.g. `payload`
/// for a scheduler-only test) or signals a recoverable gap an
/// operator can remediate (e.g. `kernel_commit` from a tarball-cache
/// kernel that has no on-disk source tree to probe).
///
/// Used by [`explain_sidecar`] to label every diagnostic block; the
/// `JSON` shape exposes this as a `"classification"` string per
/// field so dashboards can color-code "expected" vs "actionable"
/// blocks without re-deriving the rule from causes prose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NoneClassification {
    /// `None` is the expected steady state for this field — no
    /// operator action recovers it (the source data does not
    /// exist or has not been wired yet).
    Expected,
    /// `None` indicates a recoverable gap — re-running the test in
    /// a different environment (in-repo cwd, non-tarball kernel,
    /// non-host-only test) would populate the field.
    Actionable,
}

impl NoneClassification {
    /// Stable string token for the JSON `classification` key on
    /// [`explain_sidecar`]'s machine-readable output. The text
    /// renderer uses the same token so a human reading the
    /// terminal output sees the same label they would scrape from
    /// JSON.
    fn as_str(self) -> &'static str {
        match self {
            Self::Expected => "expected",
            Self::Actionable => "actionable",
        }
    }
}

/// Catalog entry for one [`crate::test_support::SidecarResult`]
/// `Option` field — wired into [`SIDECAR_NONE_CATALOG`] so the
/// diagnostic surface stays in lockstep with the on-disk schema.
///
/// Each entry was derived from the rustdoc on the corresponding
/// field in `src/test_support/sidecar.rs` — see the per-field
/// references at the catalog site for the source of truth. This
/// catalog is static (no live probing): explain-sidecar's purpose
/// is post-hoc archive diagnosis, not live debugging, and dynamic
/// probes against an absent host (e.g. checking `KTSTR_KERNEL` on
/// a CI runner that produced an archived sidecar) would return
/// nonsense.
struct NoneCatalogEntry {
    /// Field name as serialized on disk (matches the struct field
    /// identifier verbatim — serde uses the field name without
    /// rename attributes for [`crate::test_support::SidecarResult`]).
    field: &'static str,
    /// Classification when this field is `None`. See
    /// [`NoneClassification`].
    classification: NoneClassification,
    /// Human-readable cause prose, one entry per documented cause
    /// from the field's rustdoc. The text renderer prints each
    /// cause on its own bulleted line; the JSON shape emits them
    /// as a JSON string array verbatim.
    causes: &'static [&'static str],
    /// Operator-actionable remediation, when one applies.
    /// `Some(...)` for fields where re-running in a different
    /// configuration would populate the value (e.g.
    /// `kernel_commit` recovers when `KTSTR_KERNEL` points at a
    /// local source tree); `None` for fields whose `None` is the
    /// steady-state shape with no recourse (e.g.
    /// `scheduler_commit` is reserved on the schema for future
    /// enrichment).
    ///
    /// One fix per entry: this is the most-common-case
    /// remediation, picked when a field has multiple causes that
    /// all converge on the same operator action. The current
    /// shape covers the typical case without forcing a deeper
    /// data-model change; a per-cause split is possible if the
    /// catalog grows fields whose causes legitimately diverge
    /// in their remediation.
    fix: Option<&'static str>,
}

/// Static catalog covering every `Option<T>` field on
/// [`crate::test_support::SidecarResult`]. Order matches the on-
/// disk schema declaration order so a human diff against
/// `SidecarResult` reads top-to-bottom.
///
/// Causes prose is sourced FROM the per-field rustdoc on
/// `SidecarResult` — see `src/test_support/sidecar.rs` for the
/// single source of truth on what each `None` means. A future
/// schema change that adds, removes, or renames an `Option`
/// field MUST update this catalog; the
/// `none_catalog_covers_every_option_field` test enforces
/// `SIDECAR_NONE_CATALOG.len() == EXPECTED_OPTION_FIELD_COUNT`
/// (a hand-coded `10`) and asserts the projection helper
/// enumerates the same field names in the same order. A new
/// `Option` field on `SidecarResult` requires bumping the
/// constant, extending [`project_optional_fields`]'s array
/// (which has compile-checked length `10` and will fail to
/// compile on a missing entry), and adding a catalog row.
const SIDECAR_NONE_CATALOG: &[NoneCatalogEntry] = &[
    NoneCatalogEntry {
        field: "scheduler_commit",
        classification: NoneClassification::Expected,
        causes: &["no SchedulerSpec variant currently exposes a reliable \
             commit source — reserved on the schema for future \
             enrichment (e.g. --version probe or ELF-note read on \
             the resolved scheduler binary)"],
        fix: None,
    },
    NoneCatalogEntry {
        field: "project_commit",
        classification: NoneClassification::Actionable,
        causes: &[
            "current_dir() could not be resolved at sidecar-write \
             time (process cwd was rmdir'd while alive)",
            "test process cwd was not inside any git repository",
            "HEAD could not be read (unborn HEAD on a fresh \
             `git init` with zero commits, or a corrupt repository)",
        ],
        fix: Some(
            "run from inside a git-tracked source tree with at \
             least one commit",
        ),
    },
    NoneCatalogEntry {
        field: "payload",
        classification: NoneClassification::Expected,
        causes: &["test declared no binary payload (scheduler-only test \
             or pure-scenario test that never invokes \
             ctx.payload(...))"],
        fix: None,
    },
    NoneCatalogEntry {
        field: "monitor",
        classification: NoneClassification::Actionable,
        causes: &[
            "host-only test path: monitor loop never started",
            "early VM failure: monitor loop terminated before \
             producing samples",
            "sample collection produced no valid data",
        ],
        fix: None,
    },
    NoneCatalogEntry {
        field: "kvm_stats",
        classification: NoneClassification::Actionable,
        causes: &[
            "host-only test path: VM did not run",
            "KVM stats were unavailable on this host (e.g. KVM \
             module not loaded, /dev/kvm permissions, or kernel \
             missing the stats interface)",
        ],
        fix: None,
    },
    NoneCatalogEntry {
        field: "kernel_version",
        classification: NoneClassification::Actionable,
        causes: &[
            "host-only test path: no kernel under test",
            "neither cache metadata nor `include/config/kernel.release` \
             yielded a version string",
        ],
        fix: None,
    },
    NoneCatalogEntry {
        field: "kernel_commit",
        classification: NoneClassification::Actionable,
        causes: &[
            "KTSTR_KERNEL is unset or empty",
            "kernel source is a Tarball or Git transient cache \
             entry (no on-disk source tree to probe)",
            "resolved kernel directory is not a git repository \
             (gix::open failed)",
            "HEAD cannot be read (unborn HEAD on a fresh `git init` \
             with zero commits)",
            "gix probe failed for another reason — metadata, not \
             a gate",
        ],
        fix: Some(
            "set KTSTR_KERNEL to a local kernel source tree that \
             is a git repository (e.g. a git clone of the kernel)",
        ),
    },
    NoneCatalogEntry {
        field: "host",
        classification: NoneClassification::Actionable,
        causes: &[
            "test-fixture path: not the production sidecar \
             writer (production writers always populate `host`)",
            "pre-enrichment archive: sidecar predates the \
             host-context landing — re-run the test to \
             regenerate under the current schema",
        ],
        fix: Some(
            "for pre-enrichment archives, re-run the test to \
             regenerate under the current schema; test-fixture \
             sidecars are not production runs and cannot be \
             recovered by re-running",
        ),
    },
    NoneCatalogEntry {
        field: "cleanup_duration_ms",
        classification: NoneClassification::Actionable,
        causes: &[
            "host-only / host-only-stub test path: no VM teardown \
             window to time",
            "run was killed by the watchdog before \
             `KtstrVm::collect_results` returned",
        ],
        fix: None,
    },
    NoneCatalogEntry {
        field: "run_source",
        classification: NoneClassification::Actionable,
        causes: &["pre-rename archive: sidecar carries the old `source` \
             key which the current schema drops as an unknown \
             field, leaving `run_source` to fall back to None via \
             serde's tolerate-absence rule. Re-run the test to \
             regenerate under the new schema, or rename the key \
             in-place before deserialize"],
        fix: Some(
            "re-run the test to regenerate, or rename the on-disk \
             `source` key to `run_source`",
        ),
    },
];

/// Project one [`crate::test_support::SidecarResult`] onto its
/// `Option` fields, returning `(field_name, is_some)` pairs in the
/// same order as [`SIDECAR_NONE_CATALOG`].
///
/// Hand-written rather than derived because:
/// - Only the 10 `Option<T>` fields are diagnostic surface; the
///   non-`Option` fields (`test_name`, `passed`, `stats`, etc.)
///   are always populated by deserialize and would clutter the
///   output without adding signal.
/// - The order MUST match the catalog so the field-by-field
///   lookup in [`render_explain_sidecar_text`] resolves
///   correctly.
///
/// A future schema addition that introduces a new `Option<T>`
/// field on `SidecarResult` MUST update this projection; the
/// `[(_, _); 10]` array literal makes the length compile-checked
/// — adding an entry without updating the length is a compile
/// error — and the `none_catalog_covers_every_option_field` test
/// asserts the catalog and projection enumerate the same names
/// in the same order.
fn project_optional_fields(sc: &crate::test_support::SidecarResult) -> [(&'static str, bool); 10] {
    [
        ("scheduler_commit", sc.scheduler_commit.is_some()),
        ("project_commit", sc.project_commit.is_some()),
        ("payload", sc.payload.is_some()),
        ("monitor", sc.monitor.is_some()),
        ("kvm_stats", sc.kvm_stats.is_some()),
        ("kernel_version", sc.kernel_version.is_some()),
        ("kernel_commit", sc.kernel_commit.is_some()),
        ("host", sc.host.is_some()),
        ("cleanup_duration_ms", sc.cleanup_duration_ms.is_some()),
        ("run_source", sc.run_source.is_some()),
    ]
}

/// File-walk statistics for the run directory under
/// [`explain_sidecar`]. Used to drive the `walked / valid` header
/// and the corrupt-sidecars footer; the steady-state invariant is
/// `walked == valid + errors.len() + io_errors.len()`.
struct WalkStats {
    walked: usize,
    valid: usize,
    errors: Vec<crate::test_support::SidecarParseError>,
    io_errors: Vec<crate::test_support::SidecarIoError>,
}

/// Count `.ktstr.json` files under `run_dir` using
/// [`crate::test_support::collect_sidecars`]'s walk shape (flat
/// files plus one level of subdirectories). Pure file-existence
/// pass — no parsing, no `serde_json` work — used purely to
/// derive the `walked` count for [`WalkStats`].
fn count_sidecar_files(run_dir: &Path) -> usize {
    let mut count = 0usize;
    let entries = match std::fs::read_dir(run_dir) {
        Ok(e) => e,
        Err(_) => return 0,
    };
    let mut subdirs = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            subdirs.push(path);
            continue;
        }
        if crate::test_support::is_sidecar_filename(&path) {
            count += 1;
        }
    }
    for sub in subdirs {
        if let Ok(entries) = std::fs::read_dir(&sub) {
            for entry in entries.flatten() {
                if crate::test_support::is_sidecar_filename(&entry.path()) {
                    count += 1;
                }
            }
        }
    }
    count
}

/// Load sidecars under `run_dir` and report file-walk statistics.
fn walk_run_with_stats(run_dir: &Path) -> (Vec<crate::test_support::SidecarResult>, WalkStats) {
    let walked = count_sidecar_files(run_dir);
    let (sidecars, errors, io_errors) = crate::test_support::collect_sidecars_with_errors(run_dir);
    let valid = sidecars.len();
    (
        sidecars,
        WalkStats {
            walked,
            valid,
            errors,
            io_errors,
        },
    )
}

/// Diagnose `Option`-field absences for a run's sidecars. Mirrors
/// `show_run_host`'s shape (`--run` + optional `--dir`,
/// printable string return). See the original rustdoc on the
/// public surface for the full JSON / text contract and exit-code
/// policy.
pub fn explain_sidecar(run: &str, dir: Option<&Path>, json: bool) -> Result<String> {
    if run.is_empty() {
        bail!(
            "run argument must not be empty. The run argument is \
             joined onto the run-root via `Path::join` and must \
             contain at least one `Normal` path component — i.e. \
             must not be empty, `.`, `..`, or absolute (e.g. a \
             typical run key shape: `6.14-abc1234` or \
             `6.14-abc1234-dirty`). To point at a different pool \
             root, use `--dir`. Run `cargo ktstr stats list` to \
             enumerate available run keys.",
        );
    }
    for component in std::path::Path::new(run).components() {
        match component {
            std::path::Component::CurDir
            | std::path::Component::ParentDir
            | std::path::Component::RootDir
            | std::path::Component::Prefix(_) => {
                bail!(
                    "run '{run}' contains pool-root-aliasing or \
                     path-traversal components (`.`, `..`, or absolute \
                     path). The run argument is joined onto the \
                     run-root via `Path::join` and must contain only \
                     `Normal` path components — no `.`, `..`, or \
                     absolute prefix (e.g. a typical run key shape: \
                     `6.14-abc1234` or `6.14-abc1234-dirty`; \
                     multi-component paths like `gauntlet/job-1` are \
                     also accepted). To point at a different pool \
                     root, use `--dir`. Run `cargo ktstr stats list` \
                     to enumerate available run keys.",
                );
            }
            std::path::Component::Normal(_) => {}
        }
    }
    let root: std::path::PathBuf = match dir {
        Some(d) => d.to_path_buf(),
        None => crate::test_support::runs_root(),
    };
    let run_dir = root.join(run);
    if !run_dir.exists() {
        let suggestion = suggest_closest_run_key(run, &root)
            .map(|name| format!(" Did you mean `{name}`?"))
            .unwrap_or_default();
        bail!(
            "run '{run}' not found under {}.{suggestion} \
             Run `cargo ktstr stats list` to enumerate available run keys.",
            root.display(),
        );
    }
    let (sidecars, walk_stats) = walk_run_with_stats(&run_dir);
    if walk_stats.walked == 0 {
        bail!(
            "run '{run}' has no sidecar data (searched {})",
            run_dir.display(),
        );
    }
    if json {
        Ok(render_explain_sidecar_json(&sidecars, &walk_stats))
    } else {
        Ok(render_explain_sidecar_text(&sidecars, &walk_stats))
    }
}

/// Render the per-sidecar text block for [`explain_sidecar`].
fn render_explain_sidecar_text(
    sidecars: &[crate::test_support::SidecarResult],
    walk_stats: &WalkStats,
) -> String {
    use std::fmt::Write as _;
    let mut sorted: Vec<&crate::test_support::SidecarResult> = sidecars.iter().collect();
    sorted.sort_by(|a, b| {
        a.test_name
            .cmp(&b.test_name)
            .then_with(|| a.run_id.cmp(&b.run_id))
    });
    let mut out = String::new();
    let _ = writeln!(
        out,
        "walked {} sidecar file(s), parsed {} valid\n",
        walk_stats.walked, walk_stats.valid,
    );
    for sc in &sorted {
        let _ = writeln!(out, "test: {}", sc.test_name);
        let _ = writeln!(out, "  topology: {}", sc.topology);
        let _ = writeln!(out, "  scheduler: {}", sc.scheduler);
        let _ = writeln!(out, "  run_id: {}", sc.run_id);
        let arch = sc
            .host
            .as_ref()
            .and_then(|h| h.arch.as_deref())
            .unwrap_or("-");
        let _ = writeln!(out, "  arch: {arch}");
        let projected = project_optional_fields(sc);
        let populated: Vec<&'static str> = projected
            .iter()
            .filter(|(_, b)| *b)
            .map(|(n, _)| *n)
            .collect();
        let none_fields: Vec<&'static str> = projected
            .iter()
            .filter(|(_, b)| !*b)
            .map(|(n, _)| *n)
            .collect();
        let populated_text = if populated.is_empty() {
            "<none>".to_string()
        } else {
            populated.join(", ")
        };
        let _ = writeln!(
            out,
            "  populated optional fields ({}): {populated_text}",
            populated.len(),
        );
        if none_fields.is_empty() {
            let _ = writeln!(out, "  none fields: <all populated>\n");
            continue;
        }
        let _ = writeln!(out, "  none fields ({}):", none_fields.len());
        for field in none_fields {
            let entry = SIDECAR_NONE_CATALOG
                .iter()
                .find(|e| e.field == field)
                .expect(
                    "catalog must cover every projected field — \
                     guarded by none_catalog_covers_every_option_field",
                );
            let _ = writeln!(
                out,
                "    {} [{}]",
                entry.field,
                entry.classification.as_str(),
            );
            for cause in entry.causes {
                let _ = writeln!(out, "      - {cause}");
            }
            if let Some(fix) = entry.fix {
                let _ = writeln!(out, "      fix: {fix}");
            }
        }
        out.push('\n');
    }
    if !walk_stats.errors.is_empty() {
        let _ = writeln!(out, "corrupt sidecars ({}):", walk_stats.errors.len());
        for err in &walk_stats.errors {
            let _ = writeln!(out, "  {}", err.path.display());
            let _ = writeln!(out, "    error: {}", err.raw_error);
            if let Some(prose) = &err.enriched_message {
                let _ = writeln!(out, "    enriched: {prose}");
            }
        }
        out.push('\n');
    }
    if !walk_stats.io_errors.is_empty() {
        let _ = writeln!(out, "io errors ({}):", walk_stats.io_errors.len());
        for err in &walk_stats.io_errors {
            let _ = writeln!(out, "  {}", err.path.display());
            let _ = writeln!(out, "    error: {}", err.raw_error);
        }
        out.push('\n');
    }
    out
}

/// JSON schema version stamp emitted on
/// [`ExplainOutput::_schema_version`]. Bumped on any incompatible
/// shape change.
const EXPLAIN_SIDECAR_SCHEMA_VERSION: &str = "1";

#[derive(serde::Serialize)]
struct ExplainOutput<'a> {
    _schema_version: &'a str,
    _walk: WalkStatsJson<'a>,
    fields: std::collections::BTreeMap<&'a str, FieldDiagnostic<'a>>,
}

#[derive(serde::Serialize)]
struct WalkStatsJson<'a> {
    walked: usize,
    valid: usize,
    errors: Vec<WalkError<'a>>,
    io_errors: Vec<WalkIoError<'a>>,
}

#[derive(serde::Serialize)]
struct WalkError<'a> {
    path: String,
    error: &'a str,
    enriched_message: Option<&'a str>,
}

#[derive(serde::Serialize)]
struct WalkIoError<'a> {
    path: String,
    error: &'a str,
}

#[derive(serde::Serialize)]
struct FieldDiagnostic<'a> {
    none_count: usize,
    some_count: usize,
    classification: &'a str,
    causes: &'a [&'a str],
    fix: Option<&'a str>,
}

/// Render the aggregate JSON shape for [`explain_sidecar`].
fn render_explain_sidecar_json(
    sidecars: &[crate::test_support::SidecarResult],
    walk_stats: &WalkStats,
) -> String {
    let fields: std::collections::BTreeMap<&str, FieldDiagnostic<'_>> = SIDECAR_NONE_CATALOG
        .iter()
        .map(|entry| {
            let none_count = sidecars
                .iter()
                .filter(|sc| {
                    project_optional_fields(sc)
                        .iter()
                        .any(|(n, b)| *n == entry.field && !*b)
                })
                .count();
            let some_count = sidecars.len().saturating_sub(none_count);
            (
                entry.field,
                FieldDiagnostic {
                    none_count,
                    some_count,
                    classification: entry.classification.as_str(),
                    causes: entry.causes,
                    fix: entry.fix,
                },
            )
        })
        .collect();
    let errors: Vec<WalkError<'_>> = walk_stats
        .errors
        .iter()
        .map(|err| WalkError {
            path: err.path.display().to_string(),
            error: &err.raw_error,
            enriched_message: err.enriched_message.as_deref(),
        })
        .collect();
    let io_errors: Vec<WalkIoError<'_>> = walk_stats
        .io_errors
        .iter()
        .map(|err| WalkIoError {
            path: err.path.display().to_string(),
            error: &err.raw_error,
        })
        .collect();
    let output = ExplainOutput {
        _schema_version: EXPLAIN_SIDECAR_SCHEMA_VERSION,
        _walk: WalkStatsJson {
            walked: walk_stats.walked,
            valid: walk_stats.valid,
            errors,
            io_errors,
        },
        fields,
    };
    serde_json::to_string_pretty(&output).expect(
        "static-shape JSON serialization is infallible — every \
         field in ExplainOutput / WalkStatsJson / WalkError / WalkIoError / \
         FieldDiagnostic is a primitive, &str, or Vec/BTreeMap \
         of those — no NaN, no non-string keys, no unsupported \
         types",
    )
}

#[cfg(test)]
mod tests {
    use super::super::super::testing::{
        SIDECAR_VEC_FIELDS, make_test_run, write_corrupt_sidecar, write_sidecar,
    };
    use super::*;

    /// Drift guard: every `Option<T>` field on `SidecarResult` must
    /// have a matching catalog entry in `SIDECAR_NONE_CATALOG`, and
    /// the projected-fields helper must enumerate the same set.
    #[test]
    fn none_catalog_covers_every_option_field() {
        const EXPECTED_OPTION_FIELD_COUNT: usize = 10;
        assert_eq!(
            SIDECAR_NONE_CATALOG.len(),
            EXPECTED_OPTION_FIELD_COUNT,
            "SIDECAR_NONE_CATALOG must cover every Option<T> field on \
             SidecarResult; expected {EXPECTED_OPTION_FIELD_COUNT}, got \
             {}. A schema change must update the catalog in lockstep.",
            SIDECAR_NONE_CATALOG.len(),
        );
        let sc = crate::test_support::SidecarResult::test_fixture();
        let projected = project_optional_fields(&sc);
        assert_eq!(
            projected.len(),
            EXPECTED_OPTION_FIELD_COUNT,
            "project_optional_fields must enumerate every Option<T> \
             field; expected {EXPECTED_OPTION_FIELD_COUNT}, got {}. Co-update \
             with the catalog when adding a new Option field.",
            projected.len(),
        );
        for (i, (name, _)) in projected.iter().enumerate() {
            let catalog = &SIDECAR_NONE_CATALOG[i];
            assert_eq!(
                *name, catalog.field,
                "projected field {i} ({name:?}) must match catalog \
                 entry at the same index ({:?}) — order drift breaks \
                 the renderer's catalog-lookup expectation",
                catalog.field,
            );
        }
    }

    /// Catalog `causes` arrays must be non-empty for every entry.
    #[test]
    fn none_catalog_every_entry_has_causes() {
        for entry in SIDECAR_NONE_CATALOG {
            assert!(
                !entry.causes.is_empty(),
                "catalog entry for {} has no causes — every field's \
                 None case must document at least one cause",
                entry.field,
            );
        }
    }

    /// Expected-classified entries (steady-state None) must NOT
    /// carry a `fix:` — there is no operator action that recovers
    /// an Expected None, so emitting one would mislead.
    #[test]
    fn none_catalog_expected_entries_have_no_fix() {
        for entry in SIDECAR_NONE_CATALOG {
            if matches!(entry.classification, NoneClassification::Expected) {
                assert!(
                    entry.fix.is_none(),
                    "Expected-classified field {} must not carry a `fix:` \
                     — there is no operator action that recovers a \
                     steady-state None",
                    entry.field,
                );
            }
        }
    }

    /// `fix:` assignment policy: must-fix for fields with a single
    /// concrete recovery action; must-not-fix for Actionable fields
    /// whose cause set spans multiple unrelated remedies (no single
    /// operator action covers them).
    #[test]
    fn none_catalog_fix_assignments_match_policy() {
        let by_field: std::collections::HashMap<&'static str, Option<&'static str>> =
            SIDECAR_NONE_CATALOG
                .iter()
                .map(|e| (e.field, e.fix))
                .collect();
        let must_fix = ["project_commit", "kernel_commit", "host", "run_source"];
        let must_not_fix = [
            "scheduler_commit",
            "payload",
            "monitor",
            "kvm_stats",
            "kernel_version",
            "cleanup_duration_ms",
        ];
        assert_eq!(
            must_fix.len() + must_not_fix.len(),
            SIDECAR_NONE_CATALOG.len(),
            "every catalog entry must be classified as either \
             must-fix or must-not-fix; expected sum = catalog len \
             ({}), got must_fix={} + must_not_fix={}",
            SIDECAR_NONE_CATALOG.len(),
            must_fix.len(),
            must_not_fix.len(),
        );
        for field in &must_fix {
            let fix = by_field.get(field).copied().flatten();
            assert!(
                fix.is_some(),
                "field {field} has a single concrete recovery action and must carry a `fix:`",
            );
        }
        for field in &must_not_fix {
            let fix = by_field.get(field).copied().flatten();
            assert!(
                fix.is_none(),
                "field {field} must NOT carry a `fix:` (multi-cause or \
                 steady-state None) — got: {fix:?}",
            );
        }
    }

    /// Error path: the named run directory does not exist.
    #[test]
    fn explain_sidecar_missing_run_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let err = explain_sidecar("nonexistent-run", Some(tmp.path()), false).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("run 'nonexistent-run' not found"),
            "missing-run error must name the run: {msg}",
        );
        assert!(
            msg.contains("cargo ktstr stats list"),
            "missing-run error must name the discovery command: {msg}",
        );
    }

    /// Error path: run directory exists but is empty.
    #[test]
    fn explain_sidecar_empty_run_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-empty");
        std::fs::create_dir(&run_dir).unwrap();
        let err = explain_sidecar("run-empty", Some(tmp.path()), false).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no sidecar data"),
            "empty-run error must use the canonical message: {msg}",
        );
        assert!(
            msg.contains("searched"),
            "empty-run error must name the searched directory: {msg}",
        );
        assert!(
            msg.contains(&run_dir.display().to_string()),
            "empty-run error must include the resolved run_dir path \
             ({}): {msg}",
            run_dir.display(),
        );
    }

    /// All-corrupt run is NOT a hard error — text rendering surfaces
    /// every parse failure under the trailing `corrupt sidecars`
    /// block; per-sidecar `test:` blocks must NOT appear.
    #[test]
    fn explain_sidecar_all_corrupt_renders_structured_diagnostic() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-corrupt");
        std::fs::create_dir(&run_dir).unwrap();
        std::fs::write(run_dir.join("a-0000000000000000.ktstr.json"), "not json {").unwrap();
        std::fs::write(
            run_dir.join("b-0000000000000000.ktstr.json"),
            "{\"missing\": \"required-fields\"}",
        )
        .unwrap();
        let out = explain_sidecar("run-corrupt", Some(tmp.path()), false)
            .expect("all-corrupt is no longer a hard error — must render");
        assert!(
            out.contains("walked 2"),
            "header must name the walked count: {out}",
        );
        assert!(
            out.contains("parsed 0 valid"),
            "header must distinguish walked-vs-parsed (zero valid): {out}",
        );
        assert!(
            out.contains("corrupt sidecars (2):"),
            "all-corrupt run must surface the corrupt-sidecars \
             block listing every parse failure: {out}",
        );
        assert!(
            !out.contains("test:"),
            "no sidecar parsed — must not emit any per-sidecar \
             block: {out}",
        );
    }

    /// Happy path: one fixture sidecar (every Option None). Text
    /// output must list ALL ten fields under "none fields" with
    /// classifications + at least one cause string per entry.
    #[test]
    fn explain_sidecar_text_lists_all_none_fields_for_fixture() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-all-none");
        std::fs::create_dir(&run_dir).unwrap();
        let sc = crate::test_support::SidecarResult::test_fixture();
        std::fs::write(
            run_dir.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc).unwrap(),
        )
        .unwrap();
        let out = explain_sidecar("run-all-none", Some(tmp.path()), false).unwrap();
        assert!(out.contains("walked 1"), "header must report walked: {out}");
        assert!(out.contains("parsed 1"), "header must report parsed: {out}");
        assert!(
            out.contains("none fields (10)"),
            "fixture has every Option as None — count must be 10: {out}",
        );
        for entry in SIDECAR_NONE_CATALOG {
            assert!(
                out.contains(entry.field),
                "output must mention field {}: {out}",
                entry.field,
            );
        }
        assert!(
            out.contains("[expected]"),
            "expected-class fields must surface their tag: {out}",
        );
        assert!(
            out.contains("[actionable]"),
            "actionable-class fields must surface their tag: {out}",
        );
        let project_commit_fix = SIDECAR_NONE_CATALOG
            .iter()
            .find(|e| e.field == "project_commit")
            .and_then(|e| e.fix)
            .expect("project_commit has a single concrete recovery action and must carry a fix");
        assert!(
            out.contains(&format!("fix: {project_commit_fix}")),
            "project_commit's fix: line must render its catalog \
             prose verbatim ({project_commit_fix:?}): {out}",
        );
        let fix_line_count = out.matches("\n      fix:").count();
        let expected_fix_count = SIDECAR_NONE_CATALOG
            .iter()
            .filter(|e| e.fix.is_some())
            .count();
        assert_eq!(
            fix_line_count, expected_fix_count,
            "exactly {expected_fix_count} entries carry a fix: in \
             the catalog; output emitted {fix_line_count}: {out}",
        );
    }

    /// JSON shape: aggregate per-field with `none_count`,
    /// `some_count`, `classification`, `causes`, `fix`. With one
    /// fixture sidecar (every Option None), every field reports
    /// none_count=1, some_count=0, and the two sum to _walk.valid.
    #[test]
    fn explain_sidecar_json_shape_aggregates_none_counts() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-json");
        std::fs::create_dir(&run_dir).unwrap();
        let sc = crate::test_support::SidecarResult::test_fixture();
        std::fs::write(
            run_dir.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc).unwrap(),
        )
        .unwrap();
        let out = explain_sidecar("run-json", Some(tmp.path()), true).unwrap();
        let parsed: serde_json::Value =
            serde_json::from_str(&out).expect("json output must round-trip parse");
        let walk = parsed.get("_walk").expect("must have _walk key");
        assert_eq!(walk.get("walked").and_then(|v| v.as_u64()), Some(1));
        assert_eq!(walk.get("valid").and_then(|v| v.as_u64()), Some(1));
        let fields = parsed.get("fields").expect("must have fields key");
        for entry in SIDECAR_NONE_CATALOG {
            let f = fields
                .get(entry.field)
                .unwrap_or_else(|| panic!("missing field {}", entry.field));
            let none_count = f
                .get("none_count")
                .and_then(|v| v.as_u64())
                .unwrap_or_else(|| panic!("missing none_count for {}", entry.field));
            let some_count = f
                .get("some_count")
                .and_then(|v| v.as_u64())
                .unwrap_or_else(|| panic!("missing some_count for {}", entry.field));
            assert_eq!(
                none_count, 1,
                "fixture: none_count must be 1 for {}",
                entry.field
            );
            assert_eq!(
                some_count, 0,
                "fixture: some_count must be 0 for {}",
                entry.field
            );
            assert_eq!(
                none_count + some_count,
                1,
                "sum invariant for {}",
                entry.field
            );
            assert_eq!(
                f.get("classification").and_then(|v| v.as_str()),
                Some(entry.classification.as_str()),
                "classification must round-trip for {}",
                entry.field,
            );
            let causes = f
                .get("causes")
                .and_then(|v| v.as_array())
                .unwrap_or_else(|| panic!("missing causes for {}", entry.field));
            assert_eq!(
                causes.len(),
                entry.causes.len(),
                "causes array length must match catalog for {}",
                entry.field,
            );
            let fix_value = f
                .get("fix")
                .unwrap_or_else(|| panic!("missing fix for {}", entry.field));
            match entry.fix {
                Some(expected) => {
                    assert_eq!(
                        fix_value.as_str(),
                        Some(expected),
                        "fix string must round-trip for {}",
                        entry.field,
                    );
                }
                None => {
                    assert!(
                        fix_value.is_null(),
                        "fix must be JSON null for fix=None entry {}: \
                         got {fix_value:?}",
                        entry.field,
                    );
                }
            }
        }
    }

    /// Mixed populated/None: text output splits "populated optional
    /// fields (N)" from "none fields (M)" with the right counts.
    #[test]
    fn explain_sidecar_text_distinguishes_populated_from_none() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-mixed");
        std::fs::create_dir(&run_dir).unwrap();
        let mut sc = crate::test_support::SidecarResult::test_fixture();
        sc.payload = Some("ipc_pingpong".to_string());
        sc.kernel_version = Some("6.14.2".to_string());
        sc.run_source = Some("local".to_string());
        std::fs::write(
            run_dir.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc).unwrap(),
        )
        .unwrap();
        let out = explain_sidecar("run-mixed", Some(tmp.path()), false).unwrap();
        assert!(
            out.contains("populated optional fields (3)"),
            "must report 3 populated: {out}",
        );
        assert!(
            out.contains("payload"),
            "populated `payload` must appear: {out}",
        );
        assert!(out.contains("none fields (7)"), "must report 7 None: {out}",);
    }

    /// Per-sidecar text output: `arch:` line surfaces under each
    /// sidecar's block, sourced from `host.arch`.
    #[test]
    fn explain_sidecar_text_renders_arch_line() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-arch");
        std::fs::create_dir(&run_dir).unwrap();
        let mut sc = crate::test_support::SidecarResult::test_fixture();
        sc.host = Some(crate::host_context::HostContext::test_fixture());
        std::fs::write(
            run_dir.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc).unwrap(),
        )
        .unwrap();
        let out = explain_sidecar("run-arch", Some(tmp.path()), false).unwrap();
        assert!(
            out.contains("arch: x86_64"),
            "host-populated sidecar must surface `arch: x86_64`: {out}",
        );
    }

    /// Per-sidecar text output: when `host` is `None`, `arch:` line
    /// still emits with `-` sentinel for uniform shape.
    #[test]
    fn explain_sidecar_text_arch_line_falls_back_to_dash_when_host_none() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-arch-none");
        std::fs::create_dir(&run_dir).unwrap();
        let sc = crate::test_support::SidecarResult::test_fixture();
        std::fs::write(
            run_dir.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc).unwrap(),
        )
        .unwrap();
        let out = explain_sidecar("run-arch-none", Some(tmp.path()), false).unwrap();
        assert!(
            out.contains("arch: -"),
            "host-None sidecar must surface `arch: -`: {out}",
        );
    }

    /// Per-sidecar text output: two sidecars in the same run with
    /// different None patterns must each get their own block.
    #[test]
    fn explain_sidecar_text_emits_one_block_per_sidecar() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-two");
        std::fs::create_dir(&run_dir).unwrap();
        let mut a = crate::test_support::SidecarResult::test_fixture();
        a.test_name = "test_a".to_string();
        let mut b = crate::test_support::SidecarResult::test_fixture();
        b.test_name = "test_b".to_string();
        b.payload = Some("ipc_pingpong".to_string());
        std::fs::write(
            run_dir.join("a-0000000000000000.ktstr.json"),
            serde_json::to_string(&a).unwrap(),
        )
        .unwrap();
        std::fs::write(
            run_dir.join("b-0000000000000000.ktstr.json"),
            serde_json::to_string(&b).unwrap(),
        )
        .unwrap();
        let out = explain_sidecar("run-two", Some(tmp.path()), false).unwrap();
        assert!(out.contains("test: test_a"), "test_a block missing: {out}");
        assert!(out.contains("test: test_b"), "test_b block missing: {out}");
        assert!(out.contains("walked 2"), "walked count must be 2: {out}");
        assert!(out.contains("parsed 2"), "parsed count must be 2: {out}");
    }

    /// JSON aggregation across multiple sidecars: partial None on a
    /// per-field basis surfaces both none_count and some_count.
    #[test]
    fn explain_sidecar_json_aggregates_partial_none_correctly() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-partial");
        std::fs::create_dir(&run_dir).unwrap();
        let a = crate::test_support::SidecarResult::test_fixture();
        let mut b = crate::test_support::SidecarResult::test_fixture();
        b.payload = Some("ipc_pingpong".to_string());
        std::fs::write(
            run_dir.join("a-0000000000000000.ktstr.json"),
            serde_json::to_string(&a).unwrap(),
        )
        .unwrap();
        std::fs::write(
            run_dir.join("b-0000000000000000.ktstr.json"),
            serde_json::to_string(&b).unwrap(),
        )
        .unwrap();
        let out = explain_sidecar("run-partial", Some(tmp.path()), true).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        let payload = parsed
            .get("fields")
            .and_then(|f| f.get("payload"))
            .expect("payload field must be present");
        assert_eq!(payload.get("none_count").and_then(|v| v.as_u64()), Some(1));
        assert_eq!(payload.get("some_count").and_then(|v| v.as_u64()), Some(1));
        let host = parsed
            .get("fields")
            .and_then(|f| f.get("host"))
            .expect("host field must be present");
        assert_eq!(host.get("none_count").and_then(|v| v.as_u64()), Some(2));
        assert_eq!(host.get("some_count").and_then(|v| v.as_u64()), Some(0));
    }

    /// Walker counts both valid and corrupt `.ktstr.json` files.
    #[test]
    fn explain_sidecar_walks_corrupt_files_into_count() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-mixed-parse");
        std::fs::create_dir(&run_dir).unwrap();
        let valid = crate::test_support::SidecarResult::test_fixture();
        std::fs::write(
            run_dir.join("a-0000000000000000.ktstr.json"),
            serde_json::to_string(&valid).unwrap(),
        )
        .unwrap();
        std::fs::write(run_dir.join("b-0000000000000000.ktstr.json"), "garbage{").unwrap();
        let out = explain_sidecar("run-mixed-parse", Some(tmp.path()), false).unwrap();
        assert!(
            out.contains("walked 2"),
            "walker must visit both files: {out}"
        );
        assert!(
            out.contains("parsed 1"),
            "only the valid file parses: {out}"
        );
    }

    /// Walker recurses one level into subdirectories.
    #[test]
    fn explain_sidecar_walks_one_level_subdirectory() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-sub");
        let sub = run_dir.join("job-x");
        std::fs::create_dir_all(&sub).unwrap();
        let sc = crate::test_support::SidecarResult::test_fixture();
        std::fs::write(
            sub.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc).unwrap(),
        )
        .unwrap();
        let out = explain_sidecar("run-sub", Some(tmp.path()), false).unwrap();
        assert!(out.contains("walked 1"), "must walk into job-x: {out}");
        assert!(
            out.contains("parsed 1"),
            "must parse the nested file: {out}"
        );
    }

    /// Walker MUST ignore non-`.ktstr.json` files.
    #[test]
    fn explain_sidecar_ignores_non_ktstr_json() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-with-other-json");
        std::fs::create_dir(&run_dir).unwrap();
        let sc = crate::test_support::SidecarResult::test_fixture();
        std::fs::write(
            run_dir.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc).unwrap(),
        )
        .unwrap();
        std::fs::write(run_dir.join("metadata.json"), "{}").unwrap();
        let out = explain_sidecar("run-with-other-json", Some(tmp.path()), false).unwrap();
        assert!(
            out.contains("walked 1"),
            "non-ktstr JSON must not inflate the walked count: {out}",
        );
    }

    /// JSON output must be a single valid JSON document.
    #[test]
    fn explain_sidecar_json_is_valid_document() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-roundtrip");
        std::fs::create_dir(&run_dir).unwrap();
        let sc = crate::test_support::SidecarResult::test_fixture();
        std::fs::write(
            run_dir.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc).unwrap(),
        )
        .unwrap();
        let out = explain_sidecar("run-roundtrip", Some(tmp.path()), true).unwrap();
        let _: serde_json::Value = serde_json::from_str(&out).expect("output must be valid JSON");
    }

    /// Partial population: 7 of 10 Options populated; report shows
    /// "populated optional fields (7)" + "none fields (3)".
    #[test]
    fn explain_sidecar_text_handles_partial_population() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-partial-pop");
        std::fs::create_dir(&run_dir).unwrap();
        let mut sc = crate::test_support::SidecarResult::test_fixture();
        sc.scheduler_commit = Some("aaaa111".to_string());
        sc.project_commit = Some("bbbb222".to_string());
        sc.payload = Some("payload".to_string());
        sc.kernel_version = Some("6.14.2".to_string());
        sc.kernel_commit = Some("cccc333".to_string());
        sc.cleanup_duration_ms = Some(123);
        sc.run_source = Some("local".to_string());
        std::fs::write(
            run_dir.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc).unwrap(),
        )
        .unwrap();
        let out = explain_sidecar("run-partial-pop", Some(tmp.path()), false).unwrap();
        assert!(
            out.contains("populated optional fields (7)"),
            "7 of 10 Options populated must be reflected in the count: {out}",
        );
        assert!(
            out.contains("none fields (3)"),
            "3 of 10 Options remain None — must report (3): {out}",
        );
    }

    /// Classification labels are stable strings.
    #[test]
    fn none_classification_as_str_returns_stable_tokens() {
        assert_eq!(NoneClassification::Expected.as_str(), "expected");
        assert_eq!(NoneClassification::Actionable.as_str(), "actionable");
    }

    /// `kernel_commit` rustdoc enumerates 5 None causes; catalog
    /// must mirror that.
    #[test]
    fn kernel_commit_catalog_lists_five_causes() {
        let entry = SIDECAR_NONE_CATALOG
            .iter()
            .find(|e| e.field == "kernel_commit")
            .expect("kernel_commit must be in the catalog");
        assert_eq!(
            entry.causes.len(),
            5,
            "kernel_commit rustdoc enumerates 5 None causes; catalog \
             must mirror that",
        );
    }

    /// Schema version stamp is "1".
    #[test]
    fn explain_sidecar_schema_version_constant_is_one() {
        assert_eq!(EXPLAIN_SIDECAR_SCHEMA_VERSION, "1");
    }

    /// JSON output stamps `_schema_version` at top level.
    #[test]
    fn explain_sidecar_json_includes_schema_version() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-schema");
        std::fs::create_dir(&run_dir).unwrap();
        let sc = crate::test_support::SidecarResult::test_fixture();
        std::fs::write(
            run_dir.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc).unwrap(),
        )
        .unwrap();
        let out = explain_sidecar("run-schema", Some(tmp.path()), true).unwrap();
        let parsed: serde_json::Value =
            serde_json::from_str(&out).expect("json output must round-trip parse");
        assert_eq!(
            parsed.get("_schema_version").and_then(|v| v.as_str()),
            Some(EXPLAIN_SIDECAR_SCHEMA_VERSION),
            "JSON output must stamp _schema_version: {out}",
        );
    }

    /// JSON `_walk.errors` is empty array on the all-clean path.
    #[test]
    fn explain_sidecar_json_walk_errors_empty_when_all_valid() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-clean-walk");
        std::fs::create_dir(&run_dir).unwrap();
        let sc = crate::test_support::SidecarResult::test_fixture();
        std::fs::write(
            run_dir.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc).unwrap(),
        )
        .unwrap();
        let out = explain_sidecar("run-clean-walk", Some(tmp.path()), true).unwrap();
        let parsed: serde_json::Value =
            serde_json::from_str(&out).expect("json output must round-trip parse");
        let errors = parsed
            .get("_walk")
            .and_then(|w| w.get("errors"))
            .and_then(|e| e.as_array())
            .expect("_walk.errors must be a JSON array");
        assert!(
            errors.is_empty(),
            "no parse failures — _walk.errors must be empty: {out}",
        );
    }

    /// JSON `_walk.errors` lists `{path, error, enriched_message}`
    /// triples for every parse failure. enriched_message is null
    /// for generic parse failures.
    #[test]
    fn explain_sidecar_json_walk_errors_lists_corrupt_files() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-mixed-errs-json");
        std::fs::create_dir(&run_dir).unwrap();
        let valid = crate::test_support::SidecarResult::test_fixture();
        std::fs::write(
            run_dir.join("a-0000000000000000.ktstr.json"),
            serde_json::to_string(&valid).unwrap(),
        )
        .unwrap();
        let corrupt_path = run_dir.join("b-0000000000000000.ktstr.json");
        std::fs::write(&corrupt_path, "garbage{").unwrap();
        let out = explain_sidecar("run-mixed-errs-json", Some(tmp.path()), true).unwrap();
        let parsed: serde_json::Value =
            serde_json::from_str(&out).expect("json output must round-trip parse");
        let walk = parsed.get("_walk").expect("must have _walk key");
        assert_eq!(walk.get("walked").and_then(|v| v.as_u64()), Some(2));
        assert_eq!(walk.get("valid").and_then(|v| v.as_u64()), Some(1));
        let errors = walk
            .get("errors")
            .and_then(|e| e.as_array())
            .expect("_walk.errors must be a JSON array");
        assert_eq!(errors.len(), 1);
        let entry = &errors[0];
        let path = entry.get("path").and_then(|v| v.as_str()).expect("path");
        assert_eq!(path, corrupt_path.display().to_string());
        let error = entry.get("error").and_then(|v| v.as_str()).expect("error");
        assert!(!error.is_empty());
        let enriched = entry
            .get("enriched_message")
            .expect("each error entry must carry an enriched_message key");
        assert!(
            enriched.is_null(),
            "generic parse failure has no schema-drift remediation; \
             enriched_message must be JSON null: {enriched:?}",
        );
    }

    /// `enriched_parse_error_message` returns operator-facing prose
    /// for the host-missing schema-drift pattern.
    #[test]
    fn enriched_parse_error_message_returns_prose_for_host_missing_pattern() {
        let raw = "missing field `host` at line 1 column 100";
        let path = std::path::Path::new("/tmp/example-run/sidecar.ktstr.json");
        let enriched = crate::test_support::enriched_parse_error_message_for_test(path, raw)
            .expect("host-missing pattern must produce enrichment prose");
        assert!(
            enriched.contains("host"),
            "enrichment must mention host: {enriched}"
        );
        assert!(
            enriched.contains("re-run"),
            "enrichment must point at the re-run remediation: {enriched}",
        );
        assert!(
            enriched.contains("disposable-sidecar"),
            "enrichment must reference the pre-1.0 disposable-sidecar \
             policy: {enriched}",
        );
        let raw_generic = "expected ident at line 1 column 2";
        let no_enrichment =
            crate::test_support::enriched_parse_error_message_for_test(path, raw_generic);
        assert!(
            no_enrichment.is_none(),
            "generic parse error must produce no enrichment"
        );
    }

    /// All-corrupt run renders structured JSON with `valid: 0`,
    /// every parse failure under `_walk.errors`, and every field's
    /// counts at zero.
    #[test]
    fn explain_sidecar_all_corrupt_json_renders_structured_diagnostic() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-all-corrupt-json");
        std::fs::create_dir(&run_dir).unwrap();
        std::fs::write(run_dir.join("a-0000000000000000.ktstr.json"), "{").unwrap();
        std::fs::write(run_dir.join("b-0000000000000000.ktstr.json"), "garbage{").unwrap();
        let out = explain_sidecar("run-all-corrupt-json", Some(tmp.path()), true)
            .expect("all-corrupt JSON must render, not bail");
        let parsed: serde_json::Value =
            serde_json::from_str(&out).expect("json output must round-trip parse");
        let walk = parsed.get("_walk").expect("must have _walk key");
        assert_eq!(walk.get("walked").and_then(|v| v.as_u64()), Some(2));
        assert_eq!(
            walk.get("valid").and_then(|v| v.as_u64()),
            Some(0),
            "all-corrupt run must report valid=0: {out}",
        );
        let errors = walk
            .get("errors")
            .and_then(|e| e.as_array())
            .expect("_walk.errors must be present");
        assert_eq!(errors.len(), 2);
        let fields = parsed
            .get("fields")
            .and_then(|f| f.as_object())
            .expect("fields must be present");
        for entry in SIDECAR_NONE_CATALOG {
            let f = fields
                .get(entry.field)
                .unwrap_or_else(|| panic!("field {} must be present", entry.field));
            assert_eq!(f.get("none_count").and_then(|v| v.as_u64()), Some(0));
            assert_eq!(f.get("some_count").and_then(|v| v.as_u64()), Some(0));
        }
        assert_eq!(
            parsed.get("_schema_version").and_then(|v| v.as_str()),
            Some(EXPLAIN_SIDECAR_SCHEMA_VERSION),
        );
    }

    /// Generic parse failures emit `error:` line in corrupt block
    /// but NOT `enriched:` line.
    #[test]
    fn explain_sidecar_text_omits_enriched_line_for_generic_failure() {
        let (tmp, run_dir) = make_test_run("run-generic-fail-text");
        write_corrupt_sidecar(&run_dir, "a-0000000000000000", "garbage{");
        let out = explain_sidecar("run-generic-fail-text", Some(tmp.path()), false).unwrap();
        assert!(
            out.contains("corrupt sidecars (1):"),
            "generic parse failure must surface in the corrupt block: {out}",
        );
        assert!(
            out.contains("    error:"),
            "generic parse failure must emit raw `error:` line: {out}",
        );
        assert!(
            !out.contains("    enriched:"),
            "generic parse failure has no enrichment — `enriched:` \
             line must NOT appear: {out}",
        );
    }

    /// Text output appends trailing `corrupt sidecars (N):` block
    /// with positional invariant: header → per-sidecar blocks →
    /// trailing corrupt block.
    #[test]
    fn explain_sidecar_text_appends_corrupt_sidecars_block() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-text-corrupt");
        std::fs::create_dir(&run_dir).unwrap();
        let mut valid = crate::test_support::SidecarResult::test_fixture();
        valid.test_name = "valid_test".to_string();
        std::fs::write(
            run_dir.join("a-0000000000000000.ktstr.json"),
            serde_json::to_string(&valid).unwrap(),
        )
        .unwrap();
        let corrupt_path = run_dir.join("b-0000000000000000.ktstr.json");
        std::fs::write(&corrupt_path, "garbage{").unwrap();
        let out = explain_sidecar("run-text-corrupt", Some(tmp.path()), false).unwrap();
        assert!(out.contains("corrupt sidecars (1):"));
        assert!(out.contains(&corrupt_path.display().to_string()));
        assert!(out.contains("    error:"));
        let header_pos = out.find("walked 2 sidecar file(s)").unwrap();
        let test_block_pos = out.find("test: valid_test").unwrap();
        let corrupt_pos = out.find("corrupt sidecars (1):").unwrap();
        assert!(
            header_pos < test_block_pos,
            "header must precede per-sidecar blocks"
        );
        assert!(
            test_block_pos < corrupt_pos,
            "per-sidecar blocks must precede trailing corrupt block"
        );
    }

    /// Corrupt-sidecars block is suppressed when zero parse failures.
    #[test]
    fn explain_sidecar_text_omits_corrupt_block_when_no_errors() {
        let (tmp, run_dir) = make_test_run("run-text-clean");
        let sc = crate::test_support::SidecarResult::test_fixture();
        write_sidecar(&run_dir, "t-0000000000000000", &sc);
        let out = explain_sidecar("run-text-clean", Some(tmp.path()), false).unwrap();
        assert!(
            !out.contains("corrupt sidecars"),
            "no parse failures — corrupt-sidecars block must be \
             suppressed: {out}",
        );
    }

    /// Vec fields on `SidecarResult` (metrics, stimulus_events, etc.)
    /// are hard-required, NOT Option<T>. Catalog must NEVER name a
    /// Vec field as None.
    #[test]
    fn explain_sidecar_does_not_flag_empty_vec_fields_as_none() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-vecs");
        std::fs::create_dir(&run_dir).unwrap();
        let mut sc = crate::test_support::SidecarResult::test_fixture();
        sc.scheduler_commit = Some("aaaa111".to_string());
        sc.project_commit = Some("bbbb222".to_string());
        sc.payload = Some("payload".to_string());
        sc.kernel_version = Some("6.14.2".to_string());
        sc.kernel_commit = Some("cccc333".to_string());
        sc.cleanup_duration_ms = Some(123);
        sc.run_source = Some("local".to_string());
        sc.monitor = Some(crate::monitor::MonitorSummary::default());
        sc.kvm_stats = Some(crate::vmm::KvmStatsTotals::default());
        sc.host = Some(crate::host_context::HostContext::test_fixture());
        std::fs::write(
            run_dir.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc).unwrap(),
        )
        .unwrap();
        let out = explain_sidecar("run-vecs", Some(tmp.path()), false).unwrap();
        assert!(
            out.contains("none fields: <all populated>"),
            "all Options populated — must report no None fields: {out}",
        );
        for vec_field in SIDECAR_VEC_FIELDS {
            assert!(
                !out.contains(vec_field),
                "Vec field '{vec_field}' is hard-required (not Option) and \
                 must never appear in explain-sidecar output: {out}",
            );
        }
    }

    /// Pre-rename archive: on-disk `source` key → `run_source` None
    /// via serde tolerate-absence; diagnostic must surface with
    /// "rename" cause prose.
    #[test]
    fn explain_sidecar_handles_old_source_key_sidecar() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-old-source-key");
        std::fs::create_dir(&run_dir).unwrap();
        let sc = crate::test_support::SidecarResult::test_fixture();
        let mut value = serde_json::to_value(&sc).expect("fixture must serialize");
        let obj = value.as_object_mut().expect("fixture is an Object");
        obj.remove("run_source");
        obj.insert(
            "source".to_string(),
            serde_json::Value::String("archive".to_string()),
        );
        std::fs::write(
            run_dir.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&value).unwrap(),
        )
        .unwrap();
        let out = explain_sidecar("run-old-source-key", Some(tmp.path()), false).unwrap();
        assert!(
            out.contains("run_source"),
            "explain-sidecar must surface run_source as None for \
             pre-rename archive: {out}",
        );
        assert!(
            out.contains("rename"),
            "run_source None cause must mention the rename: {out}",
        );
    }

    /// `dir=None` defaults to `runs_root` derived from `CARGO_TARGET_DIR`.
    #[test]
    fn explain_sidecar_resolves_dir_default_to_runs_root() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let tmp = tempfile::tempdir().unwrap();
        let _env_target = EnvVarGuard::set("CARGO_TARGET_DIR", tmp.path());
        let _env_sidecar = EnvVarGuard::remove("KTSTR_SIDECAR_DIR");
        let runs_root = tmp.path().join("ktstr");
        let run_dir = runs_root.join("run-default-root");
        std::fs::create_dir_all(&run_dir).unwrap();
        let sc = crate::test_support::SidecarResult::test_fixture();
        std::fs::write(
            run_dir.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc).unwrap(),
        )
        .unwrap();
        let out = explain_sidecar("run-default-root", None, false)
            .expect("dir=None must resolve via runs_root() and succeed");
        assert!(out.contains("walked 1"));
        assert!(out.contains("parsed 1 valid"));
    }

    /// 0-byte `.ktstr.json` is a parse failure (serde_json rejects
    /// empty input); walker counts it in walked + emits parse error.
    #[test]
    fn explain_sidecar_handles_zero_byte_file() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-zero-byte");
        std::fs::create_dir(&run_dir).unwrap();
        let valid = crate::test_support::SidecarResult::test_fixture();
        std::fs::write(
            run_dir.join("a-0000000000000000.ktstr.json"),
            serde_json::to_string(&valid).unwrap(),
        )
        .unwrap();
        std::fs::write(run_dir.join("b-0000000000000000.ktstr.json"), "").unwrap();
        let out = explain_sidecar("run-zero-byte", Some(tmp.path()), false).unwrap();
        assert!(out.contains("walked 2"));
        assert!(out.contains("parsed 1"));
        assert!(
            out.contains("corrupt sidecars (1):"),
            "zero-byte file must surface in the corrupt-sidecars \
             block as a parse failure, not be silently dropped: {out}",
        );
    }

    /// `SidecarResult` does NOT set `deny_unknown_fields`, so a
    /// future-schema sidecar must still deserialize cleanly.
    #[test]
    fn explain_sidecar_tolerates_unknown_extra_fields() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-extra-fields");
        std::fs::create_dir(&run_dir).unwrap();
        let sc = crate::test_support::SidecarResult::test_fixture();
        let mut value = serde_json::to_value(&sc).expect("fixture must serialize");
        let obj = value.as_object_mut().expect("fixture is an Object");
        obj.insert(
            "future_field".to_string(),
            serde_json::Value::String("hypothetical".to_string()),
        );
        std::fs::write(
            run_dir.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&value).unwrap(),
        )
        .unwrap();
        let out = explain_sidecar("run-extra-fields", Some(tmp.path()), false).unwrap();
        assert!(out.contains("walked 1"));
        assert!(out.contains("parsed 1 valid"));
        assert!(out.contains("test: t"));
    }

    /// Per-field classification mapping is operator-visible as a
    /// stable tag. HashMap dedup guard catches catalog duplicate
    /// field names.
    #[test]
    fn explain_sidecar_classification_accuracy_per_field() {
        let by_field: std::collections::HashMap<&'static str, NoneClassification> =
            SIDECAR_NONE_CATALOG
                .iter()
                .map(|e| (e.field, e.classification))
                .collect();
        assert_eq!(
            by_field.len(),
            SIDECAR_NONE_CATALOG.len(),
            "SIDECAR_NONE_CATALOG must have unique `field` values \
             — HashMap collected {} entries, catalog has {}.",
            by_field.len(),
            SIDECAR_NONE_CATALOG.len(),
        );
        let expected_pairs: &[(&str, NoneClassification)] = &[
            ("scheduler_commit", NoneClassification::Expected),
            ("payload", NoneClassification::Expected),
            ("project_commit", NoneClassification::Actionable),
            ("monitor", NoneClassification::Actionable),
            ("kvm_stats", NoneClassification::Actionable),
            ("kernel_version", NoneClassification::Actionable),
            ("kernel_commit", NoneClassification::Actionable),
            ("host", NoneClassification::Actionable),
            ("cleanup_duration_ms", NoneClassification::Actionable),
            ("run_source", NoneClassification::Actionable),
        ];
        assert_eq!(
            expected_pairs.len(),
            SIDECAR_NONE_CATALOG.len(),
            "every catalog entry must have a pinned classification",
        );
        for (field, expected) in expected_pairs {
            let actual = by_field
                .get(field)
                .copied()
                .unwrap_or_else(|| panic!("catalog must contain field {field}"));
            assert_eq!(
                actual, *expected,
                "field {field}: classification mismatch — expected \
                 {expected:?}, got {actual:?}",
            );
        }
    }

    /// IO failures (sidecar predicate matched but read_to_string
    /// failed) surface in the trailing `io errors` text block AND
    /// `_walk.io_errors` JSON array. Trigger via a directory named
    /// like a sidecar (read returns EISDIR).
    #[test]
    fn explain_sidecar_io_errors_surface_in_text_block_and_json() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-io-err");
        std::fs::create_dir(&run_dir).unwrap();
        let sub = run_dir.join("sub");
        std::fs::create_dir(&sub).unwrap();
        std::fs::create_dir(sub.join("eisdir.ktstr.json")).unwrap();

        let text_out = explain_sidecar("run-io-err", Some(tmp.path()), false).unwrap();
        assert!(text_out.contains("walked 1"));
        assert!(text_out.contains("parsed 0 valid"));
        assert!(text_out.contains("io errors (1):"));
        assert!(text_out.contains("eisdir.ktstr.json"));
        assert!(!text_out.contains("corrupt sidecars"));

        let json_out = explain_sidecar("run-io-err", Some(tmp.path()), true).unwrap();
        let parsed: serde_json::Value =
            serde_json::from_str(&json_out).expect("json output must round-trip parse");
        let walk = parsed.get("_walk").expect("must have _walk");
        assert_eq!(walk.get("walked").and_then(|v| v.as_u64()), Some(1));
        assert_eq!(walk.get("valid").and_then(|v| v.as_u64()), Some(0));
        let parse_errs = walk.get("errors").and_then(|e| e.as_array()).unwrap();
        assert!(parse_errs.is_empty());
        let io_errs = walk.get("io_errors").and_then(|e| e.as_array()).unwrap();
        assert_eq!(io_errs.len(), 1);
        let entry = &io_errs[0];
        let path = entry.get("path").and_then(|v| v.as_str()).unwrap();
        assert!(path.ends_with("eisdir.ktstr.json"));
        let error = entry.get("error").and_then(|v| v.as_str()).unwrap();
        assert!(!error.is_empty());
        assert!(
            entry.get("enriched_message").is_none(),
            "io-error entries must NOT have enriched_message: {json_out}",
        );
    }

    /// `walked == valid + errors.len() + io_errors.len()` — every
    /// predicate-matching file lands in exactly one bucket.
    #[test]
    fn explain_sidecar_walk_counts_reconcile_across_outcomes() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-mixed-outcomes");
        std::fs::create_dir(&run_dir).unwrap();
        let valid = crate::test_support::SidecarResult::test_fixture();
        std::fs::write(
            run_dir.join("a-0000000000000000.ktstr.json"),
            serde_json::to_string(&valid).unwrap(),
        )
        .unwrap();
        std::fs::write(run_dir.join("b-0000000000000000.ktstr.json"), "garbage{").unwrap();
        let sub = run_dir.join("sub");
        std::fs::create_dir(&sub).unwrap();
        std::fs::create_dir(sub.join("c-0000000000000000.ktstr.json")).unwrap();

        let json_out = explain_sidecar("run-mixed-outcomes", Some(tmp.path()), true).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json_out).unwrap();
        let walk = parsed.get("_walk").unwrap();
        let walked = walk.get("walked").and_then(|v| v.as_u64()).unwrap();
        let valid_n = walk.get("valid").and_then(|v| v.as_u64()).unwrap();
        let parse_errs = walk.get("errors").and_then(|e| e.as_array()).unwrap().len() as u64;
        let io_errs = walk
            .get("io_errors")
            .and_then(|e| e.as_array())
            .unwrap()
            .len() as u64;
        assert_eq!(
            walked,
            valid_n + parse_errs + io_errs,
            "walked must equal valid + errors + io_errors. \
             walked={walked}, valid={valid_n}, errors={parse_errs}, \
             io_errors={io_errs}",
        );
        assert_eq!(walked, 3);
        assert_eq!(valid_n, 1);
        assert_eq!(parse_errs, 1);
        assert_eq!(io_errs, 1);
    }

    /// `_walk.io_errors` is empty array on the all-clean happy path.
    #[test]
    fn explain_sidecar_json_walk_io_errors_empty_when_no_io_failures() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-clean-io");
        std::fs::create_dir(&run_dir).unwrap();
        let sc = crate::test_support::SidecarResult::test_fixture();
        std::fs::write(
            run_dir.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc).unwrap(),
        )
        .unwrap();
        let out = explain_sidecar("run-clean-io", Some(tmp.path()), true).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        let io_errs = parsed
            .get("_walk")
            .and_then(|w| w.get("io_errors"))
            .and_then(|e| e.as_array())
            .unwrap();
        assert!(io_errs.is_empty());
    }

    /// E2E renderer test: synthetic [`WalkStats`] with one enriched
    /// parse error renders both `error:` and `enriched:` lines in
    /// the corrupt block, in that order.
    #[test]
    fn explain_sidecar_text_e2e_enrichment_renders_in_corrupt_block() {
        let parse_err = crate::test_support::SidecarParseError {
            path: std::path::PathBuf::from("/tmp/example-run/sidecar.ktstr.json"),
            raw_error: "missing field `host` at line 1 column 100".to_string(),
            enriched_message: Some(
                "ktstr_test: skipping /tmp/example-run/sidecar.ktstr.json: \
                 missing field `host` ... — re-run the test"
                    .to_string(),
            ),
        };
        let walk = WalkStats {
            walked: 1,
            valid: 0,
            errors: vec![parse_err],
            io_errors: Vec::new(),
        };
        let out = render_explain_sidecar_text(&[], &walk);
        assert!(out.contains("corrupt sidecars (1):"));
        assert!(out.contains("    error: missing field `host`"));
        assert!(out.contains("    enriched: "));
        let error_pos = out.find("    error: ").unwrap();
        let enriched_pos = out.find("    enriched: ").unwrap();
        assert!(
            error_pos < enriched_pos,
            "raw `error:` line must precede `enriched:` line",
        );
    }

    /// JSON-channel mirror: synthetic [`WalkStats`] with one enriched
    /// parse error renders enriched_message as JSON string (not null).
    #[test]
    fn explain_sidecar_json_e2e_enrichment_renders_in_walk_errors() {
        let prose = "ktstr_test: skipping path: missing field `host` \
                     — re-run the test to regenerate";
        let parse_err = crate::test_support::SidecarParseError {
            path: std::path::PathBuf::from("/tmp/example-run/sidecar.ktstr.json"),
            raw_error: "missing field `host` at line 1 column 100".to_string(),
            enriched_message: Some(prose.to_string()),
        };
        let walk = WalkStats {
            walked: 1,
            valid: 0,
            errors: vec![parse_err],
            io_errors: Vec::new(),
        };
        let out = render_explain_sidecar_json(&[], &walk);
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        let errors = parsed
            .get("_walk")
            .and_then(|w| w.get("errors"))
            .and_then(|e| e.as_array())
            .unwrap();
        assert_eq!(errors.len(), 1);
        let entry = &errors[0];
        let enriched = entry
            .get("enriched_message")
            .and_then(|v| v.as_str())
            .expect("enriched_message must be a JSON string");
        assert_eq!(enriched, prose);
        let raw = entry.get("error").and_then(|v| v.as_str()).unwrap();
        assert!(raw.contains("missing field"));
    }

    /// `--run` with `..` segments must bail before path resolution.
    #[test]
    fn explain_sidecar_rejects_parent_dir_traversal_in_run() {
        let tmp = tempfile::tempdir().unwrap();
        for traversal in ["../escape", "subdir/../../escape"] {
            let err = explain_sidecar(traversal, Some(tmp.path()), false)
                .expect_err("path-traversal `..` in --run must be rejected");
            let msg = format!("{err:#}");
            assert!(
                msg.contains("path-traversal"),
                "rejection message must name the cause for {traversal}: \
                 {msg}",
            );
            assert!(msg.contains(traversal));
        }
    }

    /// Absolute paths in `--run` must bail.
    #[test]
    fn explain_sidecar_rejects_absolute_path_in_run() {
        let tmp = tempfile::tempdir().unwrap();
        let err = explain_sidecar("/etc/passwd", Some(tmp.path()), false)
            .expect_err("absolute path in --run must be rejected");
        let msg = format!("{err:#}");
        assert!(msg.contains("path-traversal"));
    }

    /// Empty `--run` must bail.
    #[test]
    fn explain_sidecar_rejects_empty_run() {
        let tmp = tempfile::tempdir().unwrap();
        let err =
            explain_sidecar("", Some(tmp.path()), false).expect_err("empty --run must be rejected");
        let msg = format!("{err:#}");
        assert!(msg.contains("must not be empty"));
    }

    /// `--run .` must bail (CurDir aliases pool root).
    #[test]
    fn explain_sidecar_rejects_curdir_run() {
        let tmp = tempfile::tempdir().unwrap();
        let err =
            explain_sidecar(".", Some(tmp.path()), false).expect_err("`.` --run must be rejected");
        let msg = format!("{err:#}");
        assert!(msg.contains("path-traversal"));
    }

    /// Bare run keys with Normal-only components pass the traversal
    /// validator and reach the not-found gate.
    #[test]
    fn explain_sidecar_accepts_bare_run_key_after_traversal_check() {
        let tmp = tempfile::tempdir().unwrap();
        let err = explain_sidecar("6.14-abc1234", Some(tmp.path()), false)
            .expect_err("non-existent run must surface the not-found error");
        let msg = format!("{err:#}");
        assert!(msg.contains("not found"));
        assert!(!msg.contains("path-traversal"));
    }
}
