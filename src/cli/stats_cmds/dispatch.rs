//! `stats` subcommand dispatch: thin wrappers over the
//! [`crate::stats`] surface plus host-context render entry points.
//!
//! Holds [`print_stats_report`], [`list_runs`], [`list_metrics`],
//! [`list_values`], [`compare_partitions`], [`show_host`],
//! [`show_run_host`], [`show_thresholds`] and the per-test /
//! per-run-key fuzzy-match helpers
//! ([`suggest_closest_test_name`], [`suggest_closest_run_key`]).

use std::path::Path;

use anyhow::{Result, anyhow, bail};

use crate::stats::{ComparisonPolicy, RowFilter};

/// Read sidecar JSON files and return the gauntlet analysis report.
///
/// Source directory:
/// - `KTSTR_SIDECAR_DIR` if set, else
/// - the most recently modified subdirectory under
///   `{CARGO_TARGET_DIR or "target"}/ktstr/`.
///
/// `cargo ktstr stats` doesn't itself run a kernel, so it can't
/// reconstruct the `{kernel}-{project_commit}` key the test process
/// used; the mtime fallback mirrors "show me the report from my
/// last test run."
///
/// Returns `None` with a warning on stderr when no sidecars are found.
/// This is not an error -- regular test runs that skip gauntlet tests
/// produce no sidecar files.
pub fn print_stats_report() -> Option<String> {
    let dir = match std::env::var("KTSTR_SIDECAR_DIR") {
        Ok(d) if !d.is_empty() => Some(std::path::PathBuf::from(d)),
        _ => crate::test_support::newest_run_dir(),
    };
    let report = dir
        .as_deref()
        .map(|d| crate::test_support::analyze_sidecars(Some(d)))
        .filter(|r| !r.is_empty());
    if report.is_none() {
        eprintln!("cargo ktstr: no sidecar data found (skipped)");
    }
    report
}

/// List test runs under `{CARGO_TARGET_DIR or "target"}/ktstr/`.
pub fn list_runs() -> Result<()> {
    crate::stats::list_runs()
}

/// Render the metric registry for `cargo ktstr stats list-metrics`.
///
/// Thin wrapper over [`crate::stats::list_metrics`] — exposed through
/// `cli::` to match the `list_runs` / `compare_partitions` / `show_host`
/// convention where every stats-subcommand dispatch arm lands on a
/// `cli::*` helper before reaching the private `stats` module. The
/// returned `String` is printed verbatim by the dispatch site.
pub fn list_metrics(json: bool) -> Result<String> {
    crate::stats::list_metrics(json)
}

/// Render the distinct-value catalogue for the sidecar pool, for
/// `cargo ktstr stats list-values`.
///
/// Thin wrapper over [`crate::stats::list_values`] — exposed
/// through `cli::` for the same surface-stability reason as
/// [`list_metrics`]. The returned `String` is printed verbatim by
/// the dispatch site.
pub fn list_values(json: bool, dir: Option<&Path>) -> Result<String> {
    crate::stats::list_values(json, dir)
}

/// Compare two filter-defined partitions of the sidecar pool and
/// report regressions across slicing dimensions. See
/// [`crate::stats::compare_partitions`] for the full contract.
pub fn compare_partitions(
    filter_a: &RowFilter,
    filter_b: &RowFilter,
    filter: Option<&str>,
    policy: &ComparisonPolicy,
    dir: Option<&Path>,
    no_average: bool,
) -> Result<i32> {
    crate::stats::compare_partitions(filter_a, filter_b, filter, policy, dir, no_average)
}

/// Collect the current host context via
/// [`crate::host_context::collect_host_context`] and render it as
/// a human-readable multi-line report via
/// [`crate::host_context::HostContext::format_human`]. The output
/// ends with a newline; callers print it verbatim.
pub fn show_host() -> String {
    crate::host_context::collect_host_context().format_human()
}

/// Return the run-directory leaf name under `root` whose Levenshtein
/// edit distance from `query` is smallest AND within the closeness
/// threshold, or `None` if no candidate is close enough (or if
/// `root` cannot be enumerated).
///
/// Threshold is `max(3, query.len() / 3)` — same shape as
/// [`suggest_closest_test_name`] / [`suggest_closest_scenario_name`]
/// so the "did you mean?" UX stays uniform across the test-name,
/// scenario-name, and run-key surfaces. The absolute-3 floor lets
/// short keys (e.g. `6.14`) tolerate small typos while the
/// proportional `len/3` lets longer keys (e.g.
/// `6.14-abcdef1-dirty`) tolerate roughly one bit-flip per 3
/// chars.
///
/// Ties resolve to the FIRST name encountered in `read_dir`
/// iteration order — non-deterministic across filesystems but
/// consistent within a single invocation. The returned `String`
/// owns the leaf name (heap allocation per match) because
/// `read_dir` yields `OsString` filenames that the suggestion
/// outlives.
///
/// `read_dir` failure (root doesn't exist, permission denied)
/// silently degrades to `None` — the caller's primary diagnostic
/// is "run not found"; the "did you mean?" hint is best-effort
/// gravy and must not gate the bail path.
///
/// Filters via [`crate::test_support::is_run_directory`] so the
/// flock sentinel subdirectory (`.locks/`) and any other
/// dotfile-prefixed entry under `runs_root` cannot surface as
/// a "did you mean?" suggestion — the same predicate that
/// `newest_run_dir` and `sorted_run_entries` use, so all three
/// run-listing surfaces agree on what counts as a run dir.
pub(super) fn suggest_closest_run_key(query: &str, root: &Path) -> Option<String> {
    let threshold = std::cmp::max(3, query.len() / 3);
    let entries = std::fs::read_dir(root).ok()?;
    let mut best: Option<(usize, String)> = None;
    for entry in entries.flatten() {
        if !crate::test_support::is_run_directory(&entry) {
            continue;
        }
        let name = match entry.file_name().to_str() {
            Some(s) => s.to_string(),
            None => continue,
        };
        let d = strsim::levenshtein(query, &name);
        if d > threshold {
            continue;
        }
        match best {
            Some((best_d, _)) if best_d <= d => continue,
            _ => best = Some((d, name)),
        }
    }
    best.map(|(_, name)| name)
}

/// Render the archived host context for the named run, resolved
/// against `dir` (or `test_support::runs_root()` when `dir` is
/// `None`). Loads sidecars under the run directory and returns the
/// `HostContext::format_human` of the first sidecar that has a
/// populated `host` field — every sidecar in a single run captures
/// the same host, so first-wins is adequate.
///
/// Returns `Err` when:
/// - The run directory does not exist (actionable message names
///   the expected root),
/// - The run directory exists but has no sidecar data (matches
///   the `compare_partitions` error shape),
/// - Every sidecar carried `host: None` (older pre-enrichment
///   runs won't have the field).
pub fn show_run_host(run: &str, dir: Option<&Path>) -> Result<String> {
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
    let sidecars = crate::test_support::collect_sidecars(&run_dir);
    if sidecars.is_empty() {
        bail!("run '{run}' has no sidecar data");
    }
    // First sidecar with a populated host wins. Every sidecar in a
    // single run captures the same host; pre-enrichment sidecars
    // may have `host: None`. Scan forward rather than take the
    // first entry so older data doesn't force a "no host context"
    // error when newer sidecars in the same run DO have it.
    let host = sidecars
        .iter()
        .find_map(|sc| sc.host.as_ref())
        .ok_or_else(|| {
            anyhow!(
                "run '{run}' has {} sidecar(s) but none carries a populated \
                 host context; this usually means the run predates host-context \
                 enrichment. Re-run the test to produce a sidecar with the \
                 current schema.",
                sidecars.len(),
            )
        })?;
    Ok(host.format_human())
}

/// Return the registered test name whose Levenshtein edit distance
/// from `query` is smallest AND within the closeness threshold, or
/// `None` if no candidate is close enough.
pub(super) fn suggest_closest_test_name(query: &str) -> Option<&'static str> {
    let threshold = std::cmp::max(3, query.len() / 3);
    let mut best: Option<(usize, &'static str)> = None;
    for entry in crate::test_support::KTSTR_TESTS.iter() {
        let d = strsim::levenshtein(query, entry.name);
        if d > threshold {
            continue;
        }
        match best {
            Some((best_d, _)) if best_d <= d => continue,
            _ => best = Some((d, entry.name)),
        }
    }
    best.map(|(_, name)| name)
}

/// Render the resolved, merged `Assert` thresholds for the named
/// test — the same merge chain evaluated at run time in
/// `run_ktstr_test_inner`:
/// `Assert::default_checks().merge(entry.scheduler.assert()).merge(&entry.assert)`.
///
/// Returns `Err` when no registered test matches `test_name`. The
/// CLI wiring (`cargo ktstr show-thresholds <test>`) surfaces this
/// to the operator without requiring them to read the source, the
/// nextest `--list` output, or the Debug impl of `Assert`.
pub fn show_thresholds(test_name: &str) -> Result<String> {
    let entry = crate::test_support::find_test(test_name).ok_or_else(|| {
        let suggestion = suggest_closest_test_name(test_name)
            .map(|s| format!(" Did you mean `{s}`?"))
            .unwrap_or_default();
        anyhow!(
            "no registered ktstr test named '{test_name}'.{suggestion} \
             Run `cargo nextest list` to see the available test names \
             — then pass just the function-name component to \
             `show-thresholds`, not the `<binary>::` prefix that \
             nextest prepends to each line."
        )
    })?;
    let merged = crate::assert::Assert::default_checks()
        .merge(entry.scheduler.assert())
        .merge(&entry.assert);
    let mut out = format!("Test: {}\n", entry.name);
    out.push_str(&format!(
        "Scheduler: {}\n",
        entry.scheduler.scheduler_name(),
    ));
    out.push_str("Resolved assertion thresholds:\n");
    out.push_str(&merged.format_human());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn show_host_returns_populated_report() {
        let out = show_host();
        assert!(!out.is_empty(), "show_host must return non-empty output");
        assert!(
            out.ends_with('\n'),
            "show_host output must end with a newline for print! use: {out:?}",
        );
        assert!(
            out.contains("kernel_name"),
            "show_host must surface the kernel_name field: {out}",
        );
    }

    /// Error path: the named run directory does not exist.
    #[test]
    fn show_run_host_missing_run_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let err = show_run_host("nonexistent-run", Some(tmp.path())).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("run 'nonexistent-run' not found"),
            "missing-run error must name the run: {msg}",
        );
        assert!(
            msg.contains("cargo ktstr stats list"),
            "missing-run error must name the `stats list` discovery \
             command so operators can enumerate available run keys \
             without extra lookups: {msg}",
        );
    }

    /// Error path: run dir exists but has no sidecars.
    #[test]
    fn show_run_host_empty_run_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join("run-empty")).unwrap();
        let err = show_run_host("run-empty", Some(tmp.path())).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no sidecar data"),
            "empty-run error must name the condition: {msg}",
        );
    }

    /// Error path: every sidecar carries `host: None`.
    #[test]
    fn show_run_host_all_host_none_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-no-host");
        std::fs::create_dir(&run_dir).unwrap();
        let sc = crate::test_support::SidecarResult::test_fixture();
        let json = serde_json::to_string(&sc).unwrap();
        std::fs::write(run_dir.join("t-0000000000000000.ktstr.json"), json).unwrap();
        let err = show_run_host("run-no-host", Some(tmp.path())).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no sidecar with a populated host")
                || msg.contains("none carries a populated host context"),
            "all-host-None error must name the pre-enrichment likely cause: {msg}",
        );
    }

    /// Happy path: a run with a populated host returns format_human.
    #[test]
    fn show_run_host_populated_sidecar_returns_format_human() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-with-host");
        std::fs::create_dir(&run_dir).unwrap();
        let mut sc = crate::test_support::SidecarResult::test_fixture();
        sc.host = Some(crate::host_context::HostContext::test_fixture());
        let json = serde_json::to_string(&sc).unwrap();
        std::fs::write(run_dir.join("t-0000000000000000.ktstr.json"), json).unwrap();

        let out = show_run_host("run-with-host", Some(tmp.path())).unwrap();
        assert!(
            out.contains("kernel_name"),
            "populated host output must include the kernel_name row: {out}",
        );
        assert!(
            out.ends_with('\n'),
            "output must end with newline for print!: {out:?}",
        );
    }

    /// Happy path forward-scan: first sidecar has host:None, later
    /// has populated host. `iter().find_map` must skip the first.
    #[test]
    fn show_run_host_forward_scans_past_none_sidecars() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-mixed");
        std::fs::create_dir(&run_dir).unwrap();
        let sc_none = crate::test_support::SidecarResult::test_fixture();
        std::fs::write(
            run_dir.join("a-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc_none).unwrap(),
        )
        .unwrap();
        let mut sc_host = crate::test_support::SidecarResult::test_fixture();
        sc_host.host = Some(crate::host_context::HostContext::test_fixture());
        std::fs::write(
            run_dir.join("b-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc_host).unwrap(),
        )
        .unwrap();

        let out = show_run_host("run-mixed", Some(tmp.path()))
            .expect("forward scan must find the populated sidecar");
        assert!(
            out.contains("kernel_name"),
            "output from populated sidecar must include kernel_name: {out}",
        );
    }

    /// Happy path: a registered test name resolves through the
    /// merge chain (default → scheduler → entry assert) and renders
    /// the three-section report (Test:, Scheduler:, Resolved
    /// assertion thresholds:) with the header preceding the
    /// threshold dump.
    #[test]
    fn show_thresholds_known_test_returns_populated_report() {
        let Some(entry) = crate::test_support::KTSTR_TESTS.iter().next() else {
            eprintln!(
                "ktstr: SKIP: show_thresholds_known_test_returns_populated_report — \
                 no entries in KTSTR_TESTS",
            );
            return;
        };
        let out = show_thresholds(entry.name).expect("show_thresholds must resolve known test");
        assert!(out.contains("Test:"), "output missing `Test:` header: {out}");
        assert!(out.contains("Scheduler:"), "output missing `Scheduler:` header: {out}");
        assert!(
            out.contains("Resolved assertion thresholds:"),
            "output missing thresholds section: {out}",
        );
        let test_idx = out.find("Test:").unwrap();
        let thresholds_idx = out.find("Resolved assertion thresholds:").unwrap();
        assert!(
            test_idx < thresholds_idx,
            "`Test:` header must precede threshold dump",
        );
    }

    /// Unknown test name surfaces actionable error: missing-name
    /// diagnostic + nextest pointer + binary:: prefix caveat.
    #[test]
    fn show_thresholds_unknown_test_returns_actionable_error() {
        let err = show_thresholds("definitely_not_a_registered_test_xyz123").unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no registered ktstr test named"),
            "error must name the missing-test condition: {msg}",
        );
        assert!(
            msg.contains("cargo nextest list"),
            "error must point at the discovery command: {msg}",
        );
        assert!(
            msg.contains("function-name component"),
            "error must flag the nextest binary:: prefix caveat: {msg}",
        );
    }

    /// `suggest_closest_test_name` — positive: distance-1 typo of a
    /// long registered name returns the registered name.
    #[test]
    fn suggest_closest_test_name_finds_near_match() {
        let Some(entry) = crate::test_support::KTSTR_TESTS.iter().find(|e| {
            e.name.len() >= 10 && !(e.name.starts_with("__unit_test_") && e.name.ends_with("__"))
        }) else {
            skip!(
                "no registered non-sentinel test with name >= 10 chars \
                 — cannot construct a positive strsim probe"
            );
        };
        let mut mutated: Vec<u8> = entry.name.bytes().collect();
        mutated[0] = if mutated[0] == b'z' { b'a' } else { b'z' };
        let query = std::str::from_utf8(&mutated).expect("ASCII mutation stays UTF-8");
        let suggestion = suggest_closest_test_name(query)
            .expect("distance-1 typo on a registered name must yield a suggestion");
        assert_eq!(
            suggestion, entry.name,
            "a single-byte typo must suggest the exact name it was derived from",
        );
    }

    /// Negative: 40-char unrelated query returns None.
    #[test]
    fn suggest_closest_test_name_returns_none_for_unrelated_query() {
        let unrelated = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
        assert_eq!(
            suggest_closest_test_name(unrelated),
            None,
            "a query with no lexical relationship to any registered \
             test name must yield no suggestion (not an over-reach)",
        );
    }

    /// `suggest_closest_run_key` — distance-1 planted dir is found.
    #[test]
    fn suggest_closest_run_key_finds_near_match() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join("6.14-abc1234")).expect("plant run dir");
        let suggestion = suggest_closest_run_key("6.14-abc1235", tmp.path())
            .expect("distance-1 typo on a planted run dir must yield a suggestion");
        assert_eq!(suggestion, "6.14-abc1234");
    }

    /// `suggest_closest_run_key` — distant query returns None.
    #[test]
    fn suggest_closest_run_key_returns_none_for_distant_query() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join("6.14-abc1234")).expect("plant run dir");
        assert_eq!(
            suggest_closest_run_key("xxxxxxxxxxxxx", tmp.path()),
            None,
        );
    }

    /// `suggest_closest_run_key` — empty root returns None.
    #[test]
    fn suggest_closest_run_key_returns_none_for_empty_root() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(
            suggest_closest_run_key("6.14-abc1234", tmp.path()),
            None,
            "empty root must yield None — no candidates to match against",
        );
    }

    /// `suggest_closest_run_key` — file entries are skipped via the
    /// `is_run_directory` filter.
    #[test]
    fn suggest_closest_run_key_skips_files() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("6.14-abc1234"), b"not a dir").expect("plant file");
        std::fs::create_dir(tmp.path().join("6.14-abc1235")).expect("plant dir");
        let suggestion = suggest_closest_run_key("6.14-abc1234", tmp.path())
            .expect("the planted directory must yield a suggestion despite the same-name file");
        assert_eq!(
            suggestion, "6.14-abc1235",
            "a regression that drops the is_dir() filter would surface \
             here as `Some(\"6.14-abc1234\")` (the file at distance 0) \
             instead of `Some(\"6.14-abc1235\")` (the dir at distance 1)",
        );
    }
}
