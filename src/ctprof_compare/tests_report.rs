//! Tests for `super::report` (Phase F.2 per-module redistribution).

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

/// Per-row gate: a cgroup with counter data but no
/// caps / weight / pids accounting must NOT contribute a
/// row to the "## Cgroup limits / knobs" sub-table. The
/// cgroup-stats primary table still mentions it, but the
/// limits table is exclusive to cgroups exposing those
/// knobs.
#[test]
fn write_diff_limits_table_skips_cgroups_without_caps() {
    let mut diff = CtprofDiff::default();
    // /counters-only carries pure counter data — no
    // cpu.max/weight, no memory.max/high, no pids.
    diff.cgroup_stats_a.insert(
        "/counters-only".into(),
        simple_cgroup_stats(100, 0, 0, 1024),
    );
    diff.cgroup_stats_b.insert(
        "/counters-only".into(),
        simple_cgroup_stats(200, 0, 0, 2048),
    );
    // /capped sets a memory.max and a cpu.weight, so it
    // SHOULD appear in the limits table.
    let mut capped_a = CgroupStats::default();
    capped_a.memory.max = Some(1 << 30);
    capped_a.cpu.weight = Some(150);
    let mut capped_b = CgroupStats::default();
    capped_b.memory.max = Some(1 << 30);
    capped_b.cpu.weight = Some(150);
    diff.cgroup_stats_a.insert("/capped".into(), capped_a);
    diff.cgroup_stats_b.insert("/capped".into(), capped_b);

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

    // Header is rendered (at least one cgroup carries
    // limits data).
    assert!(
        out.contains("## Cgroup limits / knobs"),
        "limits header missing:\n{out}",
    );
    // Find the section bounds — between the limits header
    // and the next `##` header (or EOF).
    let header_pos = out.find("## Cgroup limits / knobs").unwrap();
    let after_header = &out[header_pos..];
    let next_section = after_header
        .find("\n## ")
        .map(|p| p + 1)
        .unwrap_or(after_header.len());
    let limits_section = &after_header[..next_section];
    // /capped appears (has caps), /counters-only does not.
    assert!(
        limits_section.contains("/capped"),
        "capped cgroup should appear in limits table:\n{limits_section}",
    );
    assert!(
        !limits_section.contains("/counters-only"),
        "counters-only cgroup should NOT appear (no caps/weight/pids):\n{limits_section}",
    );
}

/// memory.stat unchanged-row suppression: a key that
/// carries the same value on both sides must NOT appear in
/// the rendered memory.stat sub-table; a key that changed
/// MUST appear. Pins the baseline-vs-candidate equality
/// gate that cuts output ~10x for typical runs.
#[test]
fn write_diff_memory_stat_skips_unchanged_rows() {
    let mut diff = CtprofDiff::default();
    let mut a = CgroupStats::default();
    a.memory.stat.insert("pgfault".into(), 100);
    a.memory.stat.insert("anon".into(), 1_000_000);
    let mut b = CgroupStats::default();
    b.memory.stat.insert("pgfault".into(), 250);
    b.memory.stat.insert("anon".into(), 1_000_000);
    diff.cgroup_stats_a.insert("/app".into(), a);
    diff.cgroup_stats_b.insert("/app".into(), b);

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

    let header_pos = out
        .find("## memory.stat")
        .expect("memory.stat header missing");
    let after_header = &out[header_pos..];
    let next_section = after_header
        .find("\n## ")
        .map(|p| p + 1)
        .unwrap_or(after_header.len());
    let stat_section = &after_header[..next_section];
    assert!(
        stat_section.contains("pgfault"),
        "changed key (pgfault: 100 → 250) must appear:\n{stat_section}",
    );
    assert!(
        !stat_section.contains("anon"),
        "unchanged gauge key (anon: 1M = 1M) must be suppressed:\n{stat_section}",
    );
}

/// memory.events unchanged-row suppression: same pattern
/// as memory.stat — only changed events surface.
#[test]
fn write_diff_memory_events_skips_unchanged_rows() {
    let mut diff = CtprofDiff::default();
    let mut a = CgroupStats::default();
    a.memory.events.insert("low".into(), 5);
    a.memory.events.insert("oom_kill".into(), 0);
    let mut b = CgroupStats::default();
    b.memory.events.insert("low".into(), 12);
    b.memory.events.insert("oom_kill".into(), 0);
    diff.cgroup_stats_a.insert("/app".into(), a);
    diff.cgroup_stats_b.insert("/app".into(), b);

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

    let header_pos = out
        .find("## memory.events")
        .expect("memory.events header missing");
    let after_header = &out[header_pos..];
    let next_section = after_header
        .find("\n## ")
        .map(|p| p + 1)
        .unwrap_or(after_header.len());
    let events_section = &after_header[..next_section];
    assert!(
        events_section.contains("low"),
        "changed event (low: 5 → 12) must appear:\n{events_section}",
    );
    // `oom_kill` 0→0 should be suppressed. Use a
    // word-boundary check: `low` is a prefix of `low` but
    // distinct from `oom_kill`, so just check the literal
    // substring is absent.
    assert!(
        !events_section.contains("oom_kill"),
        "unchanged event (oom_kill: 0 = 0) must be suppressed:\n{events_section}",
    );
}

/// `DisplayFormat::Arrow` collapses baseline → candidate
/// into a single arrow cell, with the Delta column
/// alongside (not fused into the arrow). Pin the arrow cell
/// shape AND the adjacent Delta column rendering.
#[test]
fn write_diff_arrow_renders_combined_cell() {
    let (a, b) = snap_pair_for_display();
    let diff = compare(&a, &b, &CompareOptions::default());
    let mut display = DisplayOptions::default();
    display.format = DisplayFormat::Arrow;
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
    // run_time_ns row with 100 -> 200 (+100). Auto-scale
    // ladder leaves these as ns since they're below 1000.
    // Arrow cell renders `100ns -> 200ns`; the Delta column
    // (alongside Arrow under DisplayFormat::Arrow) renders
    // `+100ns`. The arrow glyph is U+2192.
    assert!(
        out.contains("\u{2192}"),
        "arrow glyph must appear in output:\n{out}"
    );
    assert!(
        out.contains("100ns") && out.contains("200ns"),
        "baseline and candidate values must surface in arrow cell:\n{out}"
    );
    assert!(
        out.contains("+100ns"),
        "delta must appear in adjacent Delta column:\n{out}"
    );
}

/// `DisplayFormat::Arrow` for derived rows: rendered
/// derived row also collapses to a single arrow cell.
#[test]
fn write_diff_arrow_renders_derived_arrow_cell() {
    let (a, b) = snap_pair_for_display();
    let diff = compare(&a, &b, &CompareOptions::default());
    let mut display = DisplayOptions::default();
    display.format = DisplayFormat::Arrow;
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
    // The avg_wait_ns derived row shows up. wait_sum 1000/4
    // = 250.00ns baseline; 2000/4 = 500.00ns candidate.
    assert!(
        out.contains("avg_wait_ns"),
        "derived metric must appear in arrow rendering:\n{out}"
    );
    // Both values should appear in the arrow form.
    assert!(
        out.contains("250.00ns") || out.contains("250ns"),
        "baseline derived value must appear in arrow cell:\n{out}"
    );
}

/// `DisplayFormat::PctOnly` drops baseline / candidate /
/// delta — only the % column carries data.
#[test]
fn write_diff_pct_only_keeps_only_pct() {
    let (a, b) = snap_pair_for_display();
    let diff = compare(&a, &b, &CompareOptions::default());
    let mut display = DisplayOptions::default();
    display.format = DisplayFormat::PctOnly;
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
        "pct-only header must drop baseline:\n{header_line}"
    );
    assert!(
        !header_line.contains("candidate"),
        "pct-only header must drop candidate:\n{header_line}"
    );
    assert!(
        !header_line.contains("delta"),
        "pct-only header must drop delta:\n{header_line}"
    );
    // The `%` column header is just the literal `%` glyph,
    // which is hard to match unambiguously in a wide
    // table. Pin the data instead — run_time_ns 100 → 200
    // is +100% so the cell renders `+100.0%`.
    assert!(
        out.contains("+100.0%"),
        "pct-only must render percent cell:\n{out}",
    );
}

/// Ratio rows render with absolute delta in the delta column
/// and `-` in the % column (suppressed for ratios per design
/// call: 0.5 → 0.6 reads as +0.100 absolute = +10pp; the
/// fraction +0.2 = +20% of baseline is misleading).
#[test]
fn write_diff_derived_ratio_suppresses_pct() {
    let mut a = make_thread("p", "w");
    a.nr_wakeups_affine = MonotonicCount(50);
    a.nr_wakeups_affine_attempts = MonotonicCount(100); // ratio = 0.5
    let mut b = make_thread("p", "w");
    b.nr_wakeups_affine = MonotonicCount(60);
    b.nr_wakeups_affine_attempts = MonotonicCount(100); // ratio = 0.6
    let diff = compare(
        &snap_with(vec![a]),
        &snap_with(vec![b]),
        &CompareOptions::default(),
    );
    let row = diff
        .derived_rows
        .iter()
        .find(|r| r.metric_name == "affine_success_ratio")
        .expect("affine_success_ratio present");
    let delta = row.delta.expect("delta present when both sides defined");
    assert!(
        (delta - 0.1).abs() < 1e-10,
        "expected delta ~0.1 (0.6 - 0.5 in f64), got {delta}",
    );
    assert!(
        row.delta_pct.is_none(),
        "ratio row must suppress delta_pct, got {:?}",
        row.delta_pct
    );
}

/// Non-ratio (ns/B) derivations keep delta_pct populated.
#[test]
fn write_diff_derived_ns_keeps_pct() {
    let mut a = make_thread("p", "w");
    a.wait_sum = MonotonicNs(1000);
    a.wait_count = MonotonicCount(10); // avg = 100ns
    let mut b = make_thread("p", "w");
    b.wait_sum = MonotonicNs(1500);
    b.wait_count = MonotonicCount(10); // avg = 150ns
    let diff = compare(
        &snap_with(vec![a]),
        &snap_with(vec![b]),
        &CompareOptions::default(),
    );
    let row = diff
        .derived_rows
        .iter()
        .find(|r| r.metric_name == "avg_wait_ns")
        .expect("avg_wait_ns present");
    assert_eq!(row.baseline, Some(DerivedValue::Scalar(100.0)));
    assert_eq!(row.candidate, Some(DerivedValue::Scalar(150.0)));
    assert_eq!(row.delta, Some(50.0));
    // delta_pct = 50/100 = 0.5
    assert!(row.delta_pct.is_some());
    let pct = row.delta_pct.unwrap();
    assert!(
        (pct - 0.5).abs() < 1e-9,
        "expected delta_pct ~0.5, got {pct}"
    );
}

#[test]
fn write_diff_header_switches_on_group_by() {
    // Minimal thread pair so write_diff has data to render
    // under `Cgroup` (hierarchical, table-per-parent) — the
    // hierarchical layout only opens a table when at least
    // one parent has rows, so an empty diff renders nothing.
    // The Comm branch always emits the column header (its
    // table is created and printed even with zero rows), so
    // the data plumbing here is for the Cgroup side. We pass
    // the same single-thread snapshot through both axes for
    // brevity.
    let mut t = make_thread("p", "w");
    t.cgroup = "/app".into();
    let snap = snap_with(vec![t]);

    let cg_opts = CompareOptions {
        group_by: GroupBy::Cgroup.into(),
        cgroup_flatten: vec![],
        no_thread_normalize: false,
        no_cg_normalize: false,
        sort_by: Vec::new(),
    };
    let cg_diff = compare(&snap, &snap, &cg_opts);
    let mut out = String::new();
    write_diff(
        &mut out,
        &cg_diff,
        Path::new("a"),
        Path::new("b"),
        GroupBy::Cgroup,
        &DisplayOptions::default(),
    )
    .unwrap();
    assert!(
        out.contains("cgroup"),
        "cgroup column header missing:\n{out}"
    );

    let comm_opts = CompareOptions {
        group_by: GroupBy::Comm.into(),
        cgroup_flatten: vec![],
        no_thread_normalize: false,
        no_cg_normalize: false,
        sort_by: Vec::new(),
    };
    let comm_diff = compare(&snap, &snap, &comm_opts);
    let mut out = String::new();
    write_diff(
        &mut out,
        &comm_diff,
        Path::new("a"),
        Path::new("b"),
        GroupBy::Comm,
        &DisplayOptions::default(),
    )
    .unwrap();
    assert!(out.contains("comm"), "comm column header missing:\n{out}");
    // "comm" must render as the column header, not as a
    // substring of "pcomm" left over from the Pcomm variant.
    assert!(
        !out.contains("pcomm"),
        "pcomm leak under Comm group_by:\n{out}"
    );
}

#[test]
fn write_diff_delta_cell_has_plus_minus_sign() {
    let mut ta = make_thread("app", "w");
    ta.run_time_ns = MonotonicNs(100);
    let mut tb = make_thread("app", "w");
    tb.run_time_ns = MonotonicNs(50);
    let diff = compare(
        &snap_with(vec![ta]),
        &snap_with(vec![tb]),
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
    // 50 - 100 = -50 ns → integer delta below the µs
    // threshold → bare signed-integer render via
    // `format_delta_cell`'s short-circuit (no `.000` noise).
    assert!(
        out.contains("-50ns"),
        "missing signed delta with unit:\n{out}",
    );
    assert!(out.contains("-50.0%"), "missing signed pct:\n{out}");
}

#[test]
fn write_diff_categorical_delta_labels_same_or_differs() {
    let mut ta = make_thread("app", "w");
    ta.policy = "SCHED_OTHER".into();
    let mut tb = make_thread("app", "w");
    tb.policy = "SCHED_FIFO".into();
    let diff = compare(
        &snap_with(vec![ta]),
        &snap_with(vec![tb]),
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
    assert!(out.contains("differs"), "missing 'differs' label:\n{out}");
}

/// Enrichment renderer must union `cgroup_stats_a` and
/// `cgroup_stats_b` keys so a cgroup that appeared in only one
/// run still surfaces a row. Drives the one-sided paths of
/// `cgroup_cell` through `write_diff` so the rendered output
/// carries the `"X → -"` / `"- → Y"` strings.
#[test]
fn write_diff_enrichment_handles_one_sided_cgroup_keys() {
    let mut diff = CtprofDiff::default();
    diff.cgroup_stats_a
        .insert("/only-baseline".into(), simple_cgroup_stats(111, 0, 0, 0));
    diff.cgroup_stats_b
        .insert("/only-candidate".into(), simple_cgroup_stats(222, 0, 0, 0));
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
    // Both keys present.
    assert!(
        out.contains("/only-baseline"),
        "baseline-only key missing:\n{out}",
    );
    assert!(
        out.contains("/only-candidate"),
        "candidate-only key missing:\n{out}",
    );
    // Each one-sided row emits the en-dash placeholder for
    // the absent side (per `cgroup_cell`'s Some/None branch).
    // cpu_usage_usec carries the "µs" unit; 111 µs is below
    // the ms threshold (1000), so it renders verbatim with
    // the base unit suffix.
    assert!(
        out.contains("111µs → -"),
        "baseline-only row missing '111µs → -' cell:\n{out}",
    );
    assert!(
        out.contains("- → 222µs"),
        "candidate-only row missing '- → 222µs' cell:\n{out}",
    );
}

/// Rows with equal `sort_key()` break ties by ascending
/// `group_key`. Build two groups that move the same metric by
/// the same percentage (so their sort keys are identical) and
/// verify the output order is alphabetical.
#[test]
fn write_diff_stable_sort_tie_breaks_by_group_key_ascending() {
    // Same percentage swing, distinct group keys "alpha" and
    // "bravo". Both rise 1_000 → 2_000 (+100%).
    let mut a1 = make_thread("alpha", "w");
    a1.run_time_ns = MonotonicNs(1_000);
    let mut a2 = make_thread("bravo", "w");
    a2.run_time_ns = MonotonicNs(1_000);
    let mut b1 = make_thread("alpha", "w");
    b1.run_time_ns = MonotonicNs(2_000);
    let mut b2 = make_thread("bravo", "w");
    b2.run_time_ns = MonotonicNs(2_000);
    let diff = compare(
        &snap_with(vec![a1, a2]),
        &snap_with(vec![b1, b2]),
        &CompareOptions::default(),
    );
    // Filter to run_time_ns rows across the two groups; the
    // tie-break must put "alpha" before "bravo".
    let run_rows: Vec<&DiffRow> = diff
        .rows
        .iter()
        .filter(|r| r.metric_name == "run_time_ns")
        .collect();
    assert_eq!(run_rows.len(), 2);
    assert!(
        (run_rows[0].delta_pct.unwrap() - 1.0).abs() < 1e-9
            && (run_rows[1].delta_pct.unwrap() - 1.0).abs() < 1e-9,
        "test fixture must produce identical delta_pct for both groups",
    );
    assert_eq!(
        run_rows[0].group_key, "alpha",
        "ascending group_key tie-break expected alpha first",
    );
    assert_eq!(run_rows[1].group_key, "bravo");
}

/// Sort by total Rss desc: the smaps render iterates process
/// keys ranked by max(baseline_rss, candidate_rss) descending
/// so the heaviest mover appears first. Pin that the rendered
/// table places `heavy` ahead of `light` regardless of
/// alphabetical key order. Without the sort, BTreeSet
/// iteration would put `heavy` after `light`.
#[test]
fn write_diff_smaps_orders_processes_by_rss_desc() {
    let mut diff = CtprofDiff::default();
    let mut heavy = BTreeMap::new();
    heavy.insert("Rss".to_string(), 100 * 1024 * 1024); // 100 MiB
    heavy.insert("Pss".to_string(), 50 * 1024 * 1024);
    let mut heavy_b = BTreeMap::new();
    heavy_b.insert("Rss".to_string(), 200 * 1024 * 1024);
    heavy_b.insert("Pss".to_string(), 100 * 1024 * 1024);
    let mut light = BTreeMap::new();
    light.insert("Rss".to_string(), 1024); // 1 KiB
    light.insert("Pss".to_string(), 512);
    let mut light_b = BTreeMap::new();
    light_b.insert("Rss".to_string(), 2048);
    light_b.insert("Pss".to_string(), 1024);
    // Keys ordered alphabetically (`a_light` before `b_heavy`
    // when sorted) so a regression that fell back to BTreeSet
    // iteration would put a_light first.
    diff.smaps_rollup_a.insert("a_light[1]".to_string(), light);
    diff.smaps_rollup_b
        .insert("a_light[1]".to_string(), light_b);
    diff.smaps_rollup_a.insert("b_heavy[2]".to_string(), heavy);
    diff.smaps_rollup_b
        .insert("b_heavy[2]".to_string(), heavy_b);

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

    let smaps_at = out
        .find("## smaps_rollup")
        .expect("smaps section must render");
    let after_header = &out[smaps_at..];
    let heavy_pos = after_header
        .find("b_heavy[2]")
        .expect("b_heavy must appear");
    let light_pos = after_header
        .find("a_light[1]")
        .expect("a_light must appear");
    assert!(
        heavy_pos < light_pos,
        "process with larger Rss must render first; \
         b_heavy@{heavy_pos} must precede a_light@{light_pos}",
    );
}

/// Render-side process-name pin: both process keys appear
/// in the smaps section body, not just the headers. Pins
/// that the row-emission loop reaches both keys — a future
/// regression that broke iteration after the first match
/// would surface here as a missing process.
#[test]
fn write_diff_smaps_emits_row_for_each_process_key() {
    let mut diff = CtprofDiff::default();
    let mut firefox_a = BTreeMap::new();
    firefox_a.insert("Rss".to_string(), 100 * 1024 * 1024);
    let mut firefox_b = BTreeMap::new();
    firefox_b.insert("Rss".to_string(), 200 * 1024 * 1024);
    let mut bash_a = BTreeMap::new();
    bash_a.insert("Rss".to_string(), 1024);
    let mut bash_b = BTreeMap::new();
    bash_b.insert("Rss".to_string(), 2048);
    diff.smaps_rollup_a.insert("firefox".into(), firefox_a);
    diff.smaps_rollup_b.insert("firefox".into(), firefox_b);
    diff.smaps_rollup_a.insert("bash".into(), bash_a);
    diff.smaps_rollup_b.insert("bash".into(), bash_b);

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
    let smaps_at = out
        .find("## smaps_rollup")
        .expect("smaps section must render");
    let smaps_section = &out[smaps_at..];
    assert!(
        smaps_section.contains("firefox"),
        "process key `firefox` must appear in smaps section body:\n{smaps_section}",
    );
    assert!(
        smaps_section.contains("bash"),
        "process key `bash` must appear in smaps section body:\n{smaps_section}",
    );
}

/// Smaps render max-Rss tiebreaker: when two processes
/// report equal absolute Rss delta, the secondary sort key
/// is descending max-Rss across baseline and candidate. Pin
/// that the larger-max-Rss process appears ahead of the
/// smaller-max-Rss process when the absolute delta is tied.
/// Choose pcomm names such that alphabetical fallback would
/// place the lower-max-Rss process first — that way the test
/// distinguishes "max-Rss tiebreak fired" from "alpha
/// fallback fired."
#[test]
fn write_diff_smaps_max_rss_breaks_tie_when_delta_equal() {
    let mut diff = CtprofDiff::default();
    // Two processes with identical absolute Rss delta
    // (+20 MiB on each side); one carries higher max-Rss.
    // The higher-max-Rss process must render first.
    let mut a = BTreeMap::new();
    a.insert("Rss".to_string(), 100 * 1024 * 1024);
    a.insert("Pss".to_string(), 30 * 1024 * 1024);
    let mut a_b = BTreeMap::new();
    a_b.insert("Rss".to_string(), 120 * 1024 * 1024);
    a_b.insert("Pss".to_string(), 35 * 1024 * 1024);
    // Same +20 MiB delta as alpha_proc, but max_Rss is
    // 240 MiB vs 120 MiB.
    let mut z = BTreeMap::new();
    z.insert("Rss".to_string(), 220 * 1024 * 1024);
    z.insert("Pss".to_string(), 80 * 1024 * 1024);
    let mut z_b = BTreeMap::new();
    z_b.insert("Rss".to_string(), 240 * 1024 * 1024);
    z_b.insert("Pss".to_string(), 90 * 1024 * 1024);
    // Alphabetical pcomm names: "alpha_proc" < "zoomed".
    // Under pure alpha, alpha_proc would come first — which
    // here is the LOWER-max-Rss process. The max-Rss
    // tiebreaker must place "zoomed" first to pass.
    diff.smaps_rollup_a.insert("alpha_proc".into(), a);
    diff.smaps_rollup_b.insert("alpha_proc".into(), a_b);
    diff.smaps_rollup_a.insert("zoomed".into(), z);
    diff.smaps_rollup_b.insert("zoomed".into(), z_b);

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
    let smaps_at = out
        .find("## smaps_rollup")
        .expect("smaps section must render");
    let after = &out[smaps_at..];
    let zoomed_pos = after.find("zoomed").expect("zoomed key must appear");
    let alpha_pos = after
        .find("alpha_proc")
        .expect("alpha_proc key must appear");
    assert!(
        zoomed_pos < alpha_pos,
        "max-Rss tiebreaker must place higher-max-Rss process (zoomed) \
         ahead of lower-max-Rss process (alpha_proc) when delta ties; got \
         zoomed@{zoomed_pos} alpha_proc@{alpha_pos}",
    );
}

/// End-to-end literal-mode smaps render: drive
/// `compare(..., no_thread_normalize: true)` with populated
/// smaps_rollup data, then `write_diff` it, and assert the
/// rendered section shows the literal `pcomm[tgid]` keys
/// (NOT `pattern_key(pcomm)`). The same process instance
/// must run on both sides for the row to join — that's the
/// price of literal mode and the row only appears when both
/// snapshots reference the same PID. Pin the full pipeline
/// from `collect_smaps_rollup` literal-key construction
/// through `write_diff`'s smaps section emission.
#[test]
fn write_diff_smaps_literal_mode_renders_pcomm_tgid_keys() {
    let mut leader_a = make_thread("worker", "worker");
    leader_a.tid = 4242;
    leader_a.tgid = 4242;
    leader_a.smaps_rollup_kb.insert("Rss".into(), 4096);
    leader_a.smaps_rollup_kb.insert("Pss".into(), 1024);
    let snap_a = snap_with(vec![leader_a]);

    let mut leader_b = make_thread("worker", "worker");
    leader_b.tid = 4242;
    leader_b.tgid = 4242;
    leader_b.smaps_rollup_kb.insert("Rss".into(), 4096);
    leader_b.smaps_rollup_kb.insert("Pss".into(), 2048);
    let snap_b = snap_with(vec![leader_b]);

    let opts = CompareOptions {
        group_by: GroupBy::Pcomm.into(),
        cgroup_flatten: vec![],
        no_thread_normalize: true,
        no_cg_normalize: false,
        sort_by: Vec::new(),
    };
    let diff = compare(&snap_a, &snap_b, &opts);

    // Diff struct must carry the literal `worker[4242]` key,
    // not the normalized `worker` form.
    assert!(
        diff.smaps_rollup_a.contains_key("worker[4242]"),
        "literal-mode baseline key must be `worker[4242]`; got {:?}",
        diff.smaps_rollup_a.keys().collect::<Vec<_>>(),
    );
    assert!(
        diff.smaps_rollup_b.contains_key("worker[4242]"),
        "literal-mode candidate key must be `worker[4242]`; got {:?}",
        diff.smaps_rollup_b.keys().collect::<Vec<_>>(),
    );
    // No normalized `worker` key under literal mode.
    assert!(
        !diff.smaps_rollup_a.contains_key("worker"),
        "literal-mode must NOT carry the normalized `worker` key",
    );

    // Rendered section text shows the literal key.
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
    let smaps_at = out
        .find("## smaps_rollup")
        .expect("smaps section must render");
    let after = &out[smaps_at..];
    assert!(
        after.contains("worker[4242]"),
        "literal-mode rendered table must show `worker[4242]` key:\n{after}",
    );
}

/// Smaps render primary sort key: absolute Rss delta
/// (descending). When two processes have different absolute
/// Rss deltas, the larger-delta process must render first
/// regardless of max-Rss or alphabetical order. Pin the
/// primary-key contract by constructing a case where
/// max-Rss and alpha both prefer the WRONG order, isolating
/// the abs-delta primary sort.
#[test]
fn write_diff_smaps_abs_rss_delta_is_primary_sort_key() {
    let mut diff = CtprofDiff::default();
    // alpha_proc: small delta, BIG max-Rss.
    // The alpha-sort fallback would put alpha_proc first
    // (alphabetically before zoomed); the max-Rss tiebreak
    // would also prefer alpha_proc (240 MiB > 60 MiB max).
    // Only the abs-delta primary sort places zoomed first.
    let mut a = BTreeMap::new();
    a.insert("Rss".to_string(), 200 * 1024 * 1024);
    let mut a_b = BTreeMap::new();
    a_b.insert("Rss".to_string(), 240 * 1024 * 1024);
    // zoomed: HUGE delta (+50 MiB), small max-Rss (60 MiB).
    let mut z = BTreeMap::new();
    z.insert("Rss".to_string(), 10 * 1024 * 1024);
    let mut z_b = BTreeMap::new();
    z_b.insert("Rss".to_string(), 60 * 1024 * 1024);
    diff.smaps_rollup_a.insert("alpha_proc".into(), a);
    diff.smaps_rollup_b.insert("alpha_proc".into(), a_b);
    diff.smaps_rollup_a.insert("zoomed".into(), z);
    diff.smaps_rollup_b.insert("zoomed".into(), z_b);

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
    let smaps_at = out
        .find("## smaps_rollup")
        .expect("smaps section must render");
    let after = &out[smaps_at..];
    let zoomed_pos = after.find("zoomed").expect("zoomed key must appear");
    let alpha_pos = after
        .find("alpha_proc")
        .expect("alpha_proc key must appear");
    assert!(
        zoomed_pos < alpha_pos,
        "abs-delta primary sort must place larger-delta process (zoomed: \
         +50 MiB) ahead of smaller-delta process (alpha_proc: +40 MiB), \
         regardless of max-Rss or alpha; got zoomed@{zoomed_pos} \
         alpha_proc@{alpha_pos}",
    );
}
