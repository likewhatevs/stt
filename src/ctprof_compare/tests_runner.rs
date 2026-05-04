//! Tests for `super::runner` (Phase F.2 per-module redistribution).

#![allow(unused_imports)]
#![allow(clippy::field_reassign_with_default)]

use std::collections::BTreeMap;
use std::path::Path;

use super::aggregate::{format_cpu_range, merge_aggregated_into};
use super::cgroup_merge::{
    merge_cgroup_cpu, merge_cgroup_memory, merge_cgroup_pids, merge_kv_counters, merge_max_option,
    merge_memory_stat, merge_min_option, merge_psi,
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
use super::*;
use crate::ctprof::{CgroupStats, CtprofSnapshot, Psi, ThreadState};
use crate::metric_types::{
    Bytes, CategoricalString, CpuSet, MonotonicCount, MonotonicNs, OrdinalI32, PeakNs,
};
use regex::Regex;

/// `--sort-by` rejects unknown names with a hint that lists
/// derived names alongside primary registry names.
#[test]
fn parse_sort_by_unknown_lists_derived_names() {
    let err = parse_sort_by("not_a_real_metric").unwrap_err();
    let msg = format!("{err:#}");
    // Lists at least one derived metric name.
    assert!(
        msg.contains("affine_success_ratio")
            || msg.contains("cpu_efficiency")
            || msg.contains("avg_wait_ns"),
        "error must list derived metric names alongside primary; got: {msg}",
    );
}

/// `new_constrained_table` sets a dummy header to force
/// column allocation, then attaches per-column width
/// constraints, on the assumption that comfy_table
/// preserves those constraints when the caller later
/// replaces the header via `set_header`. This test pins
/// that contract empirically: build a constrained table,
/// replace its header, render data wider than the
/// constraint, and assert the rendered cell is bounded by
/// the constraint width. A comfy_table upgrade that
/// breaks header-replacement-preserves-constraints surfaces
/// here as cells exceeding the configured width.
#[test]
fn new_constrained_table_constraints_survive_header_replacement() {
    let display = DisplayOptions::default();
    // 5-character upper bound on a single column.
    let max_widths: Vec<u16> = vec![5];
    let mut t = display.new_constrained_table(&max_widths);
    // Replace the dummy header with the real one (single
    // column, matching the dummy's count).
    t.set_header(vec!["col"]);
    // Add a row whose data is wider than the constraint —
    // any preservation regression would let the rendered
    // line exceed the bound.
    t.add_row(vec!["aaaaaaaaaaaaaaaaaaaa"]);
    let rendered = t.to_string();
    // Each line of the rendered table must be no wider than
    // the constraint plus comfy_table's borders/padding.
    // The 5-char data cell with default padding (1 char on
    // each side) plus 2 border chars = 9. Fail if we see a
    // line wider than 16 (generous cap that catches an
    // unconstrained 20-char cell).
    for line in rendered.lines() {
        assert!(
            line.chars().count() <= 16,
            "rendered line exceeds constrained width of 5; constraints \
             may not survive set_header replacement: \n{rendered}"
        );
    }
}

/// Empty `--sort-by` value parses to an empty Vec — caller
/// then falls back to the default delta_pct sort.
#[test]
fn parse_sort_by_empty_returns_empty_vec() {
    let keys = parse_sort_by("").expect("empty parses");
    assert!(keys.is_empty());
}

/// Single field with no direction defaults to descending
/// (largest delta first, matching operator default).
#[test]
fn parse_sort_by_single_field_defaults_to_desc() {
    let keys = parse_sort_by("wait_sum").expect("parse");
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0].metric, "wait_sum");
    assert!(keys[0].descending);
}

/// Explicit `:asc` and `:desc` directions parse correctly.
/// Mixed-direction multi-key spec round-trips fine.
#[test]
fn parse_sort_by_explicit_directions() {
    let keys = parse_sort_by("wait_sum:asc,run_time_ns:desc").expect("parse");
    assert_eq!(keys.len(), 2);
    assert_eq!(keys[0].metric, "wait_sum");
    assert!(!keys[0].descending);
    assert_eq!(keys[1].metric, "run_time_ns");
    assert!(keys[1].descending);
}

/// Whitespace is trimmed at every layer — entry-level
/// (between commas) AND inside the metric:direction split.
/// Both `  wait_sum:desc  ` and `wait_sum : desc` (spaces
/// around the `:`) parse to the same key because the metric
/// and direction are independently trimmed after
/// `split_once(':')`.
#[test]
fn parse_sort_by_trims_whitespace_between_entries() {
    let keys = parse_sort_by("  wait_sum:desc  ,  run_time_ns:asc  ").expect("parse");
    assert_eq!(keys.len(), 2);
    assert_eq!(keys[0].metric, "wait_sum");
    assert!(keys[0].descending);
    assert_eq!(keys[1].metric, "run_time_ns");
    assert!(!keys[1].descending);
}

/// Whitespace around the `:` separator is tolerated:
/// `wait_sum : desc` parses as if the spaces were absent.
/// Pin both metric- and direction-side trimming. A regression
/// that drops the direction-side trim would surface as an
/// "invalid direction \" desc\"" error.
#[test]
fn parse_sort_by_trims_whitespace_around_colon() {
    let keys = parse_sort_by("wait_sum : desc").expect("trimmed colon parse");
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0].metric, "wait_sum");
    assert!(keys[0].descending);
    // Asymmetric whitespace is also fine.
    let keys2 = parse_sort_by("run_time_ns:  asc  ").expect("trimmed asc-side parse");
    assert_eq!(keys2.len(), 1);
    assert_eq!(keys2[0].metric, "run_time_ns");
    assert!(!keys2[0].descending);
}

/// Direction matching is case-insensitive: `:DESC`, `:Desc`,
/// `:Asc`, and `:ASC` all map to the canonical `desc` /
/// `asc` semantics. Pin the lowercase normalization so an
/// operator who typed in caps doesn't get an
/// "invalid direction" error.
#[test]
fn parse_sort_by_direction_is_case_insensitive() {
    for spec in ["wait_sum:DESC", "wait_sum:Desc", "wait_sum:dEsC"] {
        let keys = parse_sort_by(spec).unwrap_or_else(|e| panic!("{spec} must parse: {e}"));
        assert_eq!(keys.len(), 1, "{spec}");
        assert!(keys[0].descending, "{spec}");
    }
    for spec in ["wait_sum:ASC", "wait_sum:Asc", "wait_sum:aSc"] {
        let keys = parse_sort_by(spec).unwrap_or_else(|e| panic!("{spec} must parse: {e}"));
        assert_eq!(keys.len(), 1, "{spec}");
        assert!(!keys[0].descending, "{spec}");
    }
}

/// Unknown metric name is rejected with a parse error
/// citing the offending name.
#[test]
fn parse_sort_by_rejects_unknown_metric() {
    let err = parse_sort_by("not_a_real_metric").unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("not_a_real_metric"),
        "error must cite offending metric name, got: {msg}"
    );
    // Also pin the "must be one of" preamble + at least one
    // canonical valid name so an operator who hits a typo
    // can recover from the diagnostic alone (without reading
    // the source). `parse_sort_by_unknown_metric_lists_valid_names_sorted`
    // pins the alphabetical order; this lighter test just
    // pins that the list rendering itself fired.
    assert!(
        msg.contains("must be one of"),
        "error must include the 'must be one of' preamble that introduces the valid-name list, got: {msg}"
    );
    assert!(
        msg.contains("run_time_ns"),
        "error must list at least one canonical metric name from the registry, got: {msg}"
    );
    // Pin the bare-metric-name hint: rendered cells now carry
    // `[tag]` suffixes (e.g. `wait_sum [non-ext] [SCHEDSTATS]`),
    // and an operator pasting the rendered cell verbatim into
    // `--sort-by` would land here. The error must redirect them
    // to the bare name.
    assert!(
        msg.contains("bare metric name"),
        "error must hint at bare-metric-name usage, got: {msg}"
    );
}

/// Pasting a tagged cell verbatim into --sort-by produces an
/// error that carries the bare-metric-name hint. Pins the
/// hint as actionable for the most likely operator failure
/// mode after the tag-suffix change.
#[test]
fn parse_sort_by_unknown_with_tag_suffix_carries_hint() {
    let err = parse_sort_by("wait_sum [non-ext] [SCHEDSTATS]").unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("bare metric name"),
        "tagged-cell paste must produce the bare-name hint, got: {msg}",
    );
}

/// Invalid direction string (anything other than `asc` /
/// `desc`) is rejected with an actionable error.
#[test]
fn parse_sort_by_rejects_invalid_direction() {
    let err = parse_sort_by("wait_sum:sideways").unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("sideways"),
        "error must cite offending direction, got: {msg}"
    );
}

/// Empty entry between commas (`a,,b`) is rejected.
#[test]
fn parse_sort_by_rejects_empty_entry() {
    let err = parse_sort_by("wait_sum,,run_time_ns").unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("empty entry"),
        "error must mention empty entry, got: {msg}"
    );
}

/// Trailing comma (`"wait_sum,"`) yields an empty token at
/// the tail and is rejected with the same diagnostic as
/// `"a,,b"`. Pins that `split(',')` semantics produce an
/// empty trailing entry rather than silently dropping it.
#[test]
fn parse_sort_by_rejects_trailing_comma() {
    let err = parse_sort_by("wait_sum,").unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("empty entry"),
        "trailing comma must surface as empty-entry error, got: {msg}"
    );
}

/// Leading comma (`",wait_sum"`) yields an empty token at
/// the head — same shape as the trailing-comma case. Pins
/// the symmetric behavior so an operator who pastes a stray
/// `,` at either end of the spec gets a consistent error.
#[test]
fn parse_sort_by_rejects_leading_comma() {
    let err = parse_sort_by(",wait_sum").unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("empty entry"),
        "leading comma must surface as empty-entry error, got: {msg}"
    );
}

/// Bare colon (`":"`) splits to an empty metric and the
/// empty string as direction. The empty direction matches
/// neither `desc` nor `asc`, so the bad-direction arm fires
/// citing the empty token. Pins this branch over the
/// alternative interpretation ("metric is empty") so the
/// diagnostic stays operator-actionable.
#[test]
fn parse_sort_by_rejects_bare_colon() {
    let err = parse_sort_by(":").unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("invalid direction"),
        "bare colon must surface as invalid-direction error, got: {msg}"
    );
}

/// A categorical metric — one whose [`AggRule`] is any
/// `Mode*` variant: [`AggRule::Mode`] (`policy`, string),
/// [`AggRule::ModeChar`] (`state`, char), or
/// [`AggRule::ModeBool`] (`ext_enabled`, bool) — has no
/// scalar to sort by. `parse_sort_by` rejects it at the CLI
/// boundary so the operator gets an actionable error rather
/// than silent fall-through to alphabetical group order.
/// Pin the canonical `policy` entry from the registry.
#[test]
fn parse_sort_by_rejects_categorical_metric() {
    // Sanity: policy is currently registered with AggRule::Mode
    // (the CategoricalString variant — distinct from
    // ModeChar/ModeBool).
    let policy_def = CTPROF_METRICS
        .iter()
        .find(|m| m.name == "policy")
        .expect("policy must be in CTPROF_METRICS");
    assert!(
        matches!(policy_def.rule, AggRule::Mode(_)),
        "test premise drift: policy is no longer Mode-aggregated; \
         pick a different categorical metric for this test",
    );
    let err = parse_sort_by("policy").unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("categorical"),
        "categorical metric error must label the failure mode, got: {msg}"
    );
    assert!(
        msg.contains("policy"),
        "categorical metric error must name the offending metric, got: {msg}"
    );
}

/// Duplicate metric name across two entries
/// (`--sort-by wait_sum,wait_sum` or `wait_sum:asc,wait_sum:desc`)
/// is rejected. The second key never contributes to the lex
/// ordering (the first key already disambiguated every
/// non-tied case, and the second key would tie identically
/// on the same metric), so it's an operator typo rather
/// than a meaningful spec.
#[test]
fn parse_sort_by_rejects_duplicate_metric() {
    let err = parse_sort_by("wait_sum,wait_sum").unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("duplicate"),
        "duplicate-metric error must label the failure mode, got: {msg}"
    );
    assert!(
        msg.contains("wait_sum"),
        "duplicate-metric error must name the offending metric, got: {msg}"
    );
    // Different directions on the same metric still count
    // as duplicate — the second entry can't change the
    // ordering, so it's still a typo.
    let err2 = parse_sort_by("wait_sum:asc,wait_sum:desc").unwrap_err();
    let msg2 = format!("{err2:#}");
    assert!(
        msg2.contains("duplicate"),
        "duplicate metric across different directions must still reject, got: {msg2}"
    );
}

/// Multi-key sort spec preserves entry order in the
/// returned Vec (left-to-right). Pins the documented
/// "lexicographic in input order" contract — a reordering
/// regression would silently rank by the second key first.
#[test]
fn parse_sort_by_multi_key_preserves_order() {
    // Three keys, distinct names — pick one each from the
    // ns / unitless / count axes so the entries are visibly
    // distinct.
    let keys = parse_sort_by("run_time_ns:desc,nr_wakeups:asc,wait_time_ns:desc").expect("parse");
    assert_eq!(keys.len(), 3);
    assert_eq!(keys[0].metric, "run_time_ns");
    assert!(keys[0].descending);
    assert_eq!(keys[1].metric, "nr_wakeups");
    assert!(!keys[1].descending);
    assert_eq!(keys[2].metric, "wait_time_ns");
    assert!(keys[2].descending);
}

/// Empty `metrics` Vec on [`DisplayOptions`] means "every
/// metric is enabled" — the no-filter default. Pins the
/// short-circuit in `is_metric_enabled` so a regression
/// that flipped the empty case to "no metric enabled"
/// surfaces here.
#[test]
fn is_metric_enabled_empty_treats_all_as_on() {
    let opts = DisplayOptions::default();
    // Sample a primary and a derived metric — both must
    // be enabled under the empty default.
    assert!(opts.is_metric_enabled("run_time_ns"));
    assert!(opts.is_metric_enabled("cpu_efficiency"));
    // Even a name not in any registry returns true under
    // the empty filter. is_metric_enabled is the gate at
    // render time; parse_metrics enforces validity at CLI
    // parse time, so these two checks compose to "filter
    // restricts only when populated."
    assert!(opts.is_metric_enabled("anything_under_empty_filter"));
}

/// Non-empty `metrics` Vec restricts rendering to the
/// listed names — names IN the filter return true, names
/// NOT in the filter return false. Pins the contains
/// membership check.
#[test]
fn is_metric_enabled_non_empty_restricts_to_listed() {
    let mut opts = DisplayOptions::default();
    opts.metrics = vec!["run_time_ns", "wait_sum"];
    assert!(opts.is_metric_enabled("run_time_ns"));
    assert!(opts.is_metric_enabled("wait_sum"));
    assert!(!opts.is_metric_enabled("nr_wakeups"));
    assert!(!opts.is_metric_enabled("cpu_efficiency"));
}

/// `--sort-by` accepts derived metric names. Three groups
/// with distinct cpu_efficiency values: sort descending puts
/// the highest first.
#[test]
fn parse_sort_by_accepts_derived_metric_name() {
    let keys = parse_sort_by("cpu_efficiency").expect("derived name parses");
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0].metric, "cpu_efficiency");
    assert!(keys[0].descending);
}

/// Bare metric name surrounded by whitespace (no colon, no
/// direction) parses as a single descending key. Pins the
/// metric-side trim path on the `None` arm of the
/// `split_once(':')` match — `entry.trim()` runs first to
/// strip the entry-level whitespace, then the `None` arm
/// passes the trimmed string straight through. A regression
/// that dropped either trim layer would surface here as a
/// failed registry lookup on the literal `"  wait_sum  "`.
#[test]
fn parse_sort_by_bare_metric_with_whitespace_no_colon() {
    let keys = parse_sort_by("  wait_sum  ").expect("bare-metric whitespace must parse");
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0].metric, "wait_sum");
    assert!(keys[0].descending);
}

/// Metric name with trailing colon and no direction
/// (`"wait_sum:"`) splits to (`"wait_sum"`, `""`). The
/// empty direction is not `asc` or `desc`, so the
/// bad-direction arm fires. A regression that treated empty
/// direction as the default `desc` would silently accept
/// the typo.
#[test]
fn parse_sort_by_rejects_metric_colon_no_direction() {
    let err = parse_sort_by("wait_sum:").unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("invalid direction"),
        "metric-colon-no-direction must surface as invalid-direction error, got: {msg}"
    );
}

/// Unknown-metric error message lists the valid registry
/// entries as a sorted comma-separated list (not a
/// `BTreeSet` debug dump). Pins the operator-facing shape:
/// the diagnostic is copy-pasteable and the names appear in
/// alphabetical order so the operator can scan for the one
/// they meant.
#[test]
fn parse_sort_by_unknown_metric_lists_valid_names_sorted() {
    let err = parse_sort_by("not_a_real_metric").unwrap_err();
    let msg = format!("{err:#}");
    // The list is comma-separated. Find two known-adjacent
    // names from the sorted set and pin their relative
    // order in the diagnostic.
    let nice_at = msg
        .find("nice")
        .expect("error must list 'nice' from the registry");
    let policy_at = msg
        .find("policy")
        .expect("error must list 'policy' from the registry");
    let run_time_at = msg
        .find("run_time_ns")
        .expect("error must list 'run_time_ns' from the registry");
    assert!(
        nice_at < policy_at,
        "names must appear in alphabetical order: \
         nice@{nice_at} < policy@{policy_at}\nmsg: {msg}",
    );
    assert!(
        policy_at < run_time_at,
        "names must appear in alphabetical order: \
         policy@{policy_at} < run_time_ns@{run_time_at}\nmsg: {msg}",
    );
    // Format must be comma-separated, not BTreeSet debug
    // (`{...}`). Pin the absence of the debug-set delimiters.
    assert!(
        !msg.contains("{\""),
        "error must use comma-separated list, not BTreeSet debug dump:\n{msg}"
    );
}
