//! Tests for `super::columns` (Phase F.2 per-module redistribution).

#![allow(unused_imports)]
#![allow(clippy::field_reassign_with_default)]

use std::collections::BTreeMap;
use std::path::Path;

use super::*;
use super::aggregate::{format_cpu_range, merge_aggregated_into};
use super::cgroup_merge::{
    merge_cgroup_cpu, merge_cgroup_memory, merge_cgroup_pids, merge_kv_counters,
    merge_max_option, merge_memory_stat, merge_min_option, merge_psi,
};
use super::columns::{compare_columns_for, format_cgroup_only_section_warning};
use super::compare::sort_diff_rows_by_keys;
use super::groups::build_row;
use super::pattern::{
    Segment, apply_systemd_template, cgroup_normalize_skeleton, cgroup_skeleton_tokens,
    classify_token, is_token_separator, pattern_counts_union, pattern_key, split_into_segments,
    tighten_group,
};
use super::render::psi_pair_has_data;
use super::scale::{auto_scale, format_delta_cell};
use super::tests_fixtures::*;
use crate::ctprof::{CgroupStats, CtprofSnapshot, Psi, ThreadState};
use crate::metric_types::{
    Bytes, CategoricalString, CpuSet, MonotonicCount, MonotonicNs, OrdinalI32, PeakNs,
};
use regex::Regex;

/// Default DisplayFormat is `Full`. Pinned via `Default`
/// derive so a future enum reorder cannot silently shift
/// the default.
#[test]
fn display_format_default_is_full() {
    assert_eq!(DisplayFormat::default(), DisplayFormat::Full);
}

/// Round-trip the `arrow` form on its own. In a
/// user-supplied `--columns` spec, `arrow` is mutually
/// exclusive with `baseline` / `candidate` only (the arrow
/// cell visually replaces those columns; pairing them
/// would render the same data twice). `arrow + delta + %`
/// is allowed and mirrors the format-default for
/// `DisplayFormat::Arrow`.
#[test]
fn parse_columns_round_trips_arrow_form() {
    let spec = "group,threads,metric,arrow";
    let cols = parse_columns(spec, true).expect("valid arrow-form spec");
    assert_eq!(
        cols,
        vec![
            Column::Group,
            Column::Threads,
            Column::Metric,
            Column::Arrow,
        ]
    );
}

/// `parse_columns` rejects an unknown name with a list of
/// valid alternatives.
#[test]
fn parse_columns_rejects_unknown_name() {
    let err = parse_columns("not_a_column", true).unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("not_a_column"), "error must cite name: {msg}",);
}

/// `parse_columns` rejects duplicate names.
#[test]
fn parse_columns_rejects_duplicate() {
    let err = parse_columns("metric,delta,metric", true).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("duplicate"),
        "error must mention duplicates: {msg}"
    );
}

/// `parse_columns` rejects empty entries between commas.
#[test]
fn parse_columns_rejects_empty_entry() {
    let err = parse_columns("metric,,delta", true).unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("empty"), "error must mention empty: {msg}");
}

/// Empty `--columns` parses to an empty Vec — caller falls
/// back to the format default.
#[test]
fn parse_columns_empty_returns_empty_vec() {
    let cols = parse_columns("", true).expect("empty parses");
    assert!(cols.is_empty());
    let cols = parse_columns("   ", true).expect("whitespace-only parses as empty");
    assert!(cols.is_empty());
}

/// Compare-side `parse_columns` rejects `arrow` paired with
/// `baseline` or `candidate` — the arrow cell visually
/// replaces those two columns, so pairing them would render
/// the same data twice. Pairing `arrow` with `delta` or `%`
/// is allowed and mirrors the format-default column set for
/// `DisplayFormat::Arrow`. The error message names the
/// constraint so the operator can recover.
#[test]
fn parse_columns_rejects_arrow_with_redundant_columns() {
    for redundant in &["baseline", "candidate"] {
        let spec = format!("arrow,{redundant}");
        let res = parse_columns(&spec, true);
        let err = res
            .err()
            .unwrap_or_else(|| panic!("arrow+{redundant} must be rejected"));
        let msg = format!("{err:#}");
        assert!(
            msg.contains("arrow") && msg.contains("mutually exclusive"),
            "error must name arrow's mutual exclusivity for spec {spec:?}: {msg}"
        );
    }
}

/// `arrow + delta + %` round-trips cleanly through
/// [`parse_columns`] — it matches the default column set
/// for `DisplayFormat::Arrow` and must be expressible from a
/// user-supplied `--columns` spec.
#[test]
fn parse_columns_accepts_arrow_with_delta_and_pct() {
    let spec = "group,threads,metric,arrow,delta,%";
    let cols = parse_columns(spec, true).expect("arrow + delta + % must parse");
    assert_eq!(
        cols,
        vec![
            Column::Group,
            Column::Threads,
            Column::Metric,
            Column::Arrow,
            Column::Delta,
            Column::Pct,
        ],
    );
}

/// Empty / whitespace-only `--sections` parses to an empty
/// `Vec` — caller treats that as "all sections render" via
/// [`DisplayOptions::is_section_enabled`]'s empty-input
/// short-circuit. Mirror of [`parse_columns_empty_returns_empty_vec`].
#[test]
fn parse_sections_empty_returns_empty_vec() {
    let secs = parse_sections("").expect("empty parses");
    assert!(secs.is_empty());
    let secs = parse_sections("   ").expect("whitespace-only parses as empty");
    assert!(secs.is_empty());
}

/// Round-trip every [`Section::ALL`] entry through its
/// [`Section::cli_name`] and back through [`parse_sections`].
/// Exhaustively pins the cli_name table and the parser's
/// recognition logic against drift — adding a new variant
/// without updating cli_name would surface here as a
/// nonexistent name in the comma-joined spec.
#[test]
fn parse_sections_round_trips_every_name() {
    let spec = Section::ALL
        .iter()
        .map(|s| s.cli_name())
        .collect::<Vec<_>>()
        .join(",");
    let parsed = parse_sections(&spec).expect("every cli_name must round-trip");
    assert_eq!(
        parsed,
        Section::ALL.to_vec(),
        "round-trip must preserve order and identity"
    );
}

/// Unknown section name must surface a diagnostic that
/// names the offending token and lists every valid name —
/// the operator should be able to recover from the error
/// alone without reading the source.
#[test]
fn parse_sections_rejects_unknown_name() {
    let err = parse_sections("not_a_section").unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("not_a_section"),
        "error must cite the offending name: {msg}"
    );
    // Sample a couple of valid names so a future cli_name
    // rename surfaces here too.
    assert!(
        msg.contains("primary"),
        "error must list valid names: {msg}"
    );
    assert!(
        msg.contains("host-pressure"),
        "error must list valid names: {msg}"
    );
}

/// Duplicate name across two entries must reject — same
/// section appearing twice carries no extra information and
/// signals a typo.
#[test]
fn parse_sections_rejects_duplicate() {
    let err = parse_sections("primary,derived,primary").unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("duplicate"),
        "error must mention duplicates: {msg}"
    );
}

/// Empty token between commas (`primary,,derived`) must
/// reject. Mirrors `parse_columns_rejects_empty_entry` —
/// surfacing the typo at parse time beats silently
/// dropping an empty slot.
#[test]
fn parse_sections_rejects_empty_entry() {
    let err = parse_sections("primary,,derived").unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("empty"), "error must mention empty: {msg}");
}

/// Multiple non-overlapping names parse in input order —
/// the resolved Vec preserves the operator-supplied
/// sequence rather than re-sorting into [`Section::ALL`]
/// order. Pins that the parser does not stealthily
/// reorder.
#[test]
fn parse_sections_accepts_multiple_in_input_order() {
    let secs =
        parse_sections("derived,primary,host-pressure").expect("multi-section spec parses");
    assert_eq!(
        secs,
        vec![Section::Derived, Section::Primary, Section::HostPressure],
        "input order must be preserved",
    );
}

/// Whitespace around each entry is trimmed before lookup —
/// `--sections "primary , derived"` must parse identically
/// to `--sections primary,derived`. Pins the trim() call in
/// the parser body.
#[test]
fn parse_sections_trims_whitespace_around_entries() {
    let secs =
        parse_sections("  primary , derived  ").expect("whitespace-tolerant spec parses");
    assert_eq!(secs, vec![Section::Primary, Section::Derived]);
}

/// [`Section::ALL`] must list every variant exactly once.
/// Walks ALL, round-trips each through `parse_sections`,
/// and enforces uniqueness via the parser's duplicate
/// rejection — a future variant added without an `ALL`
/// entry would fail the round-trip; a duplicate in `ALL`
/// would fail the BTreeSet uniqueness check below.
/// Pinning this invariant in the test surface lets
/// `parse_sections` stay the single source of truth and
/// catches drift between the enum and the constant.
#[test]
fn section_all_is_exhaustive_and_unique() {
    let mut names: std::collections::BTreeSet<&'static str> = std::collections::BTreeSet::new();
    for s in Section::ALL {
        assert!(
            names.insert(s.cli_name()),
            "duplicate cli_name in Section::ALL: {}",
            s.cli_name()
        );
        // Each name must round-trip individually so a
        // future variant whose `cli_name` collides with
        // another's is caught by the BTreeSet insert
        // above, AND its absence from `parse_sections`'s
        // recognition would surface here as a parse
        // failure.
        let parsed = parse_sections(s.cli_name())
            .unwrap_or_else(|e| panic!("cli_name {} failed parse: {e:#}", s.cli_name()));
        assert_eq!(parsed, vec![*s]);
    }
    assert_eq!(
        names.len(),
        Section::ALL.len(),
        "ALL count must match the unique-names count",
    );
}

/// Empty `sections` Vec on [`DisplayOptions`] means "every
/// section is enabled" — the no-filter default. Pins the
/// short-circuit in `is_section_enabled` so a regression
/// that flipped the empty case to "no section enabled"
/// surfaces here.
#[test]
fn is_section_enabled_empty_treats_all_as_on() {
    let opts = DisplayOptions::default();
    for s in Section::ALL {
        assert!(
            opts.is_section_enabled(*s),
            "empty filter must enable {} (default = all-on)",
            s.cli_name()
        );
    }
}

/// Non-empty `sections` Vec restricts rendering to the
/// listed entries — every variant not in the filter must
/// be disabled, every variant in the filter enabled. Pins
/// the `contains` membership check.
#[test]
fn is_section_enabled_non_empty_restricts_to_listed() {
    let mut opts = DisplayOptions::default();
    opts.sections = vec![Section::Primary, Section::HostPressure];
    for s in Section::ALL {
        let in_filter = matches!(s, Section::Primary | Section::HostPressure);
        assert_eq!(
            opts.is_section_enabled(*s),
            in_filter,
            "is_section_enabled({}) under {{Primary, HostPressure}} \
             must be {in_filter}",
            s.cli_name(),
        );
    }
}

/// [`Section::requires_cgroup_grouping`] returns true for
/// the five sections behind the `GroupBy::Cgroup` outer
/// gate (`CgroupStats`, `Limits`, `MemoryStat`,
/// `MemoryEvents`, `Pressure`) and false for every other
/// variant. Pins the closed-set so a future variant
/// addition that lives behind the cgroup gate has to
/// update this match arm.
#[test]
fn section_requires_cgroup_grouping_classifies_correctly() {
    for s in Section::ALL {
        let expected = matches!(
            s,
            Section::CgroupStats
                | Section::Limits
                | Section::MemoryStat
                | Section::MemoryEvents
                | Section::Pressure
        );
        assert_eq!(
            s.requires_cgroup_grouping(),
            expected,
            "Section::{s:?}.requires_cgroup_grouping() must be {expected}",
        );
    }
}

/// Empty / whitespace-only `--metrics` parses to an empty
/// `Vec<&str>` — caller treats that as "every metric
/// renders" via [`DisplayOptions::is_metric_enabled`]'s
/// empty-input short-circuit.
#[test]
fn parse_metrics_empty_returns_empty_vec() {
    assert!(parse_metrics("").expect("empty parses").is_empty());
    assert!(
        parse_metrics("   ")
            .expect("whitespace-only parses as empty")
            .is_empty()
    );
}

/// Every primary registry name round-trips through
/// `parse_metrics`. Walks `CTPROF_METRICS` exhaustively
/// — adding a new metric to the registry without re-running
/// its name through this parser would surface here only if
/// the parser silently dropped it; the linear-scan match in
/// `parse_metrics` accepts any `name` field, so the test
/// is a sanity rail rather than a drift detector.
#[test]
fn parse_metrics_round_trips_every_primary_registry_name() {
    for m in CTPROF_METRICS {
        let parsed = parse_metrics(m.name)
            .unwrap_or_else(|e| panic!("metric name {} failed parse: {e:#}", m.name));
        assert_eq!(parsed, vec![m.name]);
    }
}

/// Derived metric names round-trip identically to primary
/// metric names — the parser accepts both registries and
/// returns the registry's `&'static str` either way. Pins
/// the union-of-registries lookup contract.
#[test]
fn parse_metrics_round_trips_every_derived_registry_name() {
    for d in CTPROF_DERIVED_METRICS {
        let parsed = parse_metrics(d.name)
            .unwrap_or_else(|e| panic!("derived name {} failed parse: {e:#}", d.name));
        assert_eq!(parsed, vec![d.name]);
    }
}

/// Mixed primary + derived metrics in one spec parse in
/// input order. Pins that the parser does not stealthily
/// segregate by registry, and that the input-order contract
/// matches `parse_sections`.
#[test]
fn parse_metrics_accepts_primary_and_derived_in_input_order() {
    // `run_time_ns` is a primary metric, `cpu_efficiency`
    // is a derived metric — both well-known names that
    // exist in the live registry.
    let parsed = parse_metrics("cpu_efficiency,run_time_ns")
        .expect("mixed primary+derived spec must parse");
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0], "cpu_efficiency");
    assert_eq!(parsed[1], "run_time_ns");
}

/// Unknown metric name surfaces a diagnostic that names the
/// offending token and points at `ctprof metric-list`.
/// The error must mention BOTH registries so the operator
/// knows the lookup spans primary + derived.
#[test]
fn parse_metrics_rejects_unknown_name() {
    let err = parse_metrics("not_a_real_metric").unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("not_a_real_metric"),
        "error must cite the offending name: {msg}"
    );
    assert!(
        msg.contains("metric-list"),
        "error must point operator at the discovery command: {msg}"
    );
}

/// Duplicate metric across two entries rejects.
#[test]
fn parse_metrics_rejects_duplicate() {
    let err = parse_metrics("run_time_ns,wait_sum,run_time_ns").unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("duplicate"),
        "error must mention duplicates: {msg}"
    );
}

/// Empty token between commas (`run_time_ns,,wait_sum`)
/// rejects.
#[test]
fn parse_metrics_rejects_empty_entry() {
    let err = parse_metrics("run_time_ns,,wait_sum").unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("empty"), "error must mention empty: {msg}");
}

/// Whitespace around each entry is trimmed before lookup.
#[test]
fn parse_metrics_trims_whitespace_around_entries() {
    let parsed =
        parse_metrics("  run_time_ns , wait_sum  ").expect("whitespace-tolerant spec parses");
    assert_eq!(parsed, vec!["run_time_ns", "wait_sum"]);
}

/// [`format_cgroup_only_section_warning`] renders a
/// diagnostic that names the offending section, the
/// `--group-by cgroup` requirement, AND the operator's
/// chosen group-by spelling. Pins all three load-bearing
/// elements of the warning text against drift.
#[test]
fn format_cgroup_only_section_warning_names_all_three_elements() {
    let msg = format_cgroup_only_section_warning(Section::Pressure, GroupBy::Pcomm);
    assert!(
        msg.contains("'pressure'"),
        "warning must quote the section cli_name: {msg}",
    );
    assert!(
        msg.contains("--group-by cgroup"),
        "warning must name the cgroup requirement: {msg}",
    );
    assert!(
        msg.contains("pcomm"),
        "warning must echo the operator's --group-by axis: {msg}",
    );
}

/// [`format_cgroup_only_section_warning`] echoes the
/// `comm-exact` spelling (not `CommExact`) so the warning
/// matches the value-enum the operator typed at the CLI.
/// Pins [`group_by_cli_name`]'s mapping for the
/// hyphenated variant — clap's value-enum derive renames
/// `CommExact` to `comm-exact`, and a regression that
/// stringified the variant via `Debug` would surface
/// `CommExact` instead.
#[test]
fn format_cgroup_only_section_warning_uses_comm_exact_spelling() {
    let msg = format_cgroup_only_section_warning(Section::CgroupStats, GroupBy::CommExact);
    assert!(
        msg.contains("comm-exact"),
        "warning must use the clap value-enum spelling: {msg}",
    );
    assert!(
        !msg.contains("CommExact"),
        "warning must not surface the rust variant name: {msg}",
    );
}

/// `--columns` overrides `--display-format`'s default.
/// Resolved column set comes from `columns` when non-empty.
#[test]
fn columns_override_wins_over_display_format() {
    let mut opts = DisplayOptions::default();
    opts.format = DisplayFormat::Full;
    opts.columns = vec![Column::Metric, Column::Delta];
    let resolved = opts.resolved_compare_columns();
    assert_eq!(resolved, vec![Column::Metric, Column::Delta]);
}

/// `DisplayFormat::DeltaOnly` end-to-end: rendered diff
/// table omits the `baseline` and `candidate` columns.
#[test]
fn write_diff_delta_only_omits_baseline_candidate_columns() {
    let (a, b) = snap_pair_for_display();
    let diff = compare(&a, &b, &CompareOptions::default());
    let mut display = DisplayOptions::default();
    display.format = DisplayFormat::DeltaOnly;
    let mut out = String::new();
    write_diff(
        &mut out,
        &diff,
        Path::new("a"),
        Path::new("b"),
        GroupBy::Pcomm,
        &display,
    )
    .unwrap();
    // The first line of `out` is the section heading
    // `## Primary metrics`; the table column header is the
    // first line containing the `metric` token.
    let header_line = out
        .lines()
        .find(|line| line.contains("metric") && !line.starts_with("##"))
        .unwrap_or("");
    assert!(
        !header_line.contains("baseline"),
        "delta-only header must drop baseline column:\n{header_line}"
    );
    assert!(
        !header_line.contains("candidate"),
        "delta-only header must drop candidate column:\n{header_line}"
    );
    assert!(
        header_line.contains("delta"),
        "delta column must remain:\n{header_line}"
    );
}

/// `DisplayFormat::NoPct` drops the `%` column.
#[test]
fn write_diff_no_pct_omits_pct_column() {
    let (a, b) = snap_pair_for_display();
    let diff = compare(&a, &b, &CompareOptions::default());
    let mut display = DisplayOptions::default();
    display.format = DisplayFormat::NoPct;
    let mut out = String::new();
    write_diff(
        &mut out,
        &diff,
        Path::new("a"),
        Path::new("b"),
        GroupBy::Pcomm,
        &display,
    )
    .unwrap();
    // The first line of `out` is the section heading
    // `## Primary metrics`; the table column header is the
    // first line containing the `metric` token.
    let header_line = out
        .lines()
        .find(|line| line.contains("metric") && !line.starts_with("##"))
        .unwrap_or("");
    // `%` is the literal column name; assert it is absent
    // as a stand-alone token. The header is whitespace-padded
    // by comfy_table; check there's no bare " % " run.
    assert!(
        !header_line.contains(" % "),
        "no-pct header must drop percent column:\n{header_line}"
    );
}

/// `--columns metric,delta` overrides `--display-format
/// full` and emits exactly those two columns plus their
/// labels.
#[test]
fn write_diff_columns_override_emits_only_selected_columns() {
    let (a, b) = snap_pair_for_display();
    let diff = compare(&a, &b, &CompareOptions::default());
    let mut display = DisplayOptions::default();
    display.format = DisplayFormat::Full; // would normally emit 7 columns
    display.columns = vec![Column::Metric, Column::Delta];
    let mut out = String::new();
    write_diff(
        &mut out,
        &diff,
        Path::new("a"),
        Path::new("b"),
        GroupBy::Pcomm,
        &display,
    )
    .unwrap();
    // The first line of `out` is the section heading
    // `## Primary metrics`; the table column header is the
    // first line containing the `metric` token.
    let header_line = out
        .lines()
        .find(|line| line.contains("metric") && !line.starts_with("##"))
        .unwrap_or("");
    assert!(
        header_line.contains("metric"),
        "metric column must appear:\n{header_line}"
    );
    assert!(
        header_line.contains("delta"),
        "delta column must appear:\n{header_line}"
    );
    assert!(
        !header_line.contains("baseline"),
        "baseline must NOT appear when --columns excludes it:\n{header_line}"
    );
    assert!(
        !header_line.contains("candidate"),
        "candidate must NOT appear when --columns excludes it:\n{header_line}"
    );
}

/// Render integration: write_diff emits the `## Derived
/// metrics` section with one row per derivation per matched
/// group. Pin the section header and a representative row.
#[test]
fn write_diff_emits_derived_section() {
    let mut t = make_thread("p", "w");
    t.run_time_ns = MonotonicNs(1000);
    t.timeslices = MonotonicCount(4);
    let diff = compare(
        &snap_with(vec![t.clone()]),
        &snap_with(vec![t]),
        &CompareOptions::default(),
    );
    let mut out = String::new();
    write_diff(
        &mut out,
        &diff,
        Path::new("a"),
        Path::new("b"),
        GroupBy::Pcomm,
        &DisplayOptions::default(),
    )
    .unwrap();
    assert!(
        out.contains("## Derived metrics"),
        "missing derived section header:\n{out}",
    );
    assert!(
        out.contains("avg_slice_ns"),
        "missing avg_slice_ns row in derived section:\n{out}",
    );
}

#[test]
fn write_diff_emits_expected_column_headers() {
    let diff = compare(
        &snap_with(vec![make_thread("p", "w")]),
        &snap_with(vec![make_thread("p", "w")]),
        &CompareOptions::default(),
    );
    let mut out = String::new();
    write_diff(
        &mut out,
        &diff,
        Path::new("a"),
        Path::new("b"),
        GroupBy::Pcomm,
        &DisplayOptions::default(),
    )
    .unwrap();
    for h in [
        "pcomm",
        "threads",
        "metric",
        "baseline",
        "candidate",
        "delta",
        "%",
    ] {
        assert!(out.contains(h), "missing header {h}:\n{out}");
    }
}

#[test]
fn write_diff_prints_only_baseline_section() {
    let diff = CtprofDiff {
        only_baseline: vec!["missing_proc".into()],
        ..CtprofDiff::default()
    };
    let mut out = String::new();
    write_diff(
        &mut out,
        &diff,
        Path::new("/tmp/a.ctprof.zst"),
        Path::new("/tmp/b.ctprof.zst"),
        GroupBy::Pcomm,
        &DisplayOptions::default(),
    )
    .unwrap();
    assert!(out.contains("only in baseline"));
    assert!(out.contains("missing_proc"));
    assert!(out.contains("/tmp/a.ctprof.zst"));
}

#[test]
fn write_diff_prints_only_candidate_section() {
    let diff = CtprofDiff {
        only_candidate: vec!["new_proc".into()],
        ..CtprofDiff::default()
    };
    let mut out = String::new();
    write_diff(
        &mut out,
        &diff,
        Path::new("/tmp/a.ctprof.zst"),
        Path::new("/tmp/b.ctprof.zst"),
        GroupBy::Pcomm,
        &DisplayOptions::default(),
    )
    .unwrap();
    assert!(out.contains("only in candidate"));
    assert!(out.contains("new_proc"));
    assert!(out.contains("/tmp/b.ctprof.zst"));
}

#[test]
fn write_diff_cgroup_enrichment_section_for_cgroup_mode() {
    let mut diff = CtprofDiff::default();
    diff.cgroup_stats_a
        .insert("/app".into(), simple_cgroup_stats(10, 0, 0, 100));
    diff.cgroup_stats_b
        .insert("/app".into(), simple_cgroup_stats(50, 0, 0, 200));
    let mut out = String::new();
    write_diff(
        &mut out,
        &diff,
        Path::new("a"),
        Path::new("b"),
        GroupBy::Cgroup,
        &DisplayOptions::default(),
    )
    .unwrap();
    assert!(
        out.contains("cpu_usage_usec"),
        "missing enrichment header:\n{out}"
    );
    // Cell renders as a contiguous `baseline → candidate
    // (delta)` triple via `cgroup_cell`. Both 10 µs and 50 µs
    // are below the 1000-µs ms-step threshold, so they keep
    // the base unit (`10µs`, `50µs`); delta +40 likewise.
    // Asserting on the contiguous string (rather than three
    // bare integer substrings) defends against a regression
    // where one cell's render drifts — bare `out.contains("10")`
    // would silently pass even if the µs cell were dropped
    // entirely (the substring "10" appears in the larger
    // surrounding format).
    assert!(
        out.contains("10µs → 50µs (+40µs)"),
        "missing contiguous scaled triple `10µs → 50µs (+40µs)`:\n{out}",
    );
    // Memory_current went 100 → 200 — both below the 1024 KiB
    // threshold so they render as bare bytes with the `B`
    // unit. Pin the contiguous form here too so the byte
    // family's no-step-up path is covered.
    assert!(
        out.contains("100B → 200B (+100B)"),
        "missing contiguous scaled triple `100B → 200B (+100B)`:\n{out}",
    );
}

/// Show-side `parse_columns` accepts the `metric,value`
/// pair — the show-only allowed vocabulary. Pins that the
/// show-side path actually parses both names rather than
/// silently rejecting `value` as if it were compare-only.
#[test]
fn parse_columns_accepts_show_side_metric_value() {
    let cols = parse_columns("metric,value", false).expect("metric,value is show-side valid");
    assert_eq!(cols, vec![Column::Metric, Column::Value]);
}

