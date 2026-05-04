//! Per-process `smaps_rollup` table emitter for
//! [`super::write_diff`].
//!
//! Iterates the union of process keys across both snapshots;
//! one row per `(process, key)` pair carrying baseline →
//! candidate kB values rendered through the existing "B"
//! auto-scale ladder after kB → bytes conversion. Suppressed
//! when neither side has any smaps_rollup data; per-row gate
//! skips rows where baseline equals candidate (treats absent
//! and 0 as equal). Mirrors the memory.stat compare layout.
//!
//! Process iteration order: descending by absolute Rss delta,
//! tiebreak by descending max-Rss across baseline and
//! candidate, final tiebreak alphabetical. Memory-heavy
//! processes that moved between snapshots surface first; the
//! renderer then walks each process's per-field smaps keys in
//! BTreeSet (alphabetical) order so within-process row
//! ordering stays deterministic.
//!
//! Rss is the primary "how much memory does this process
//! hold" signal; max-Rss carries the high-watermark and breaks
//! ties when two processes report equal absolute Rss delta.
//! Processes missing both keys sort last under
//! `unwrap_or(0)` and preserve alphabetical order under
//! `BTreeSet`'s ordering.

use std::collections::BTreeSet;
use std::fmt;

use super::super::columns::{Column, Section};
use super::super::diff_types::CtprofDiff;
use super::super::options::GroupBy;
use super::super::render::{color_diff_cell, colored_header_with_sort};
use super::super::runner::DisplayOptions;
use super::super::scale::{format_delta_cell, format_scaled_u64, ScaleLadder};
use super::primary::depth_color;

pub(super) fn write_smaps_section<W: fmt::Write>(
    w: &mut W,
    diff: &CtprofDiff,
    group_by: GroupBy,
    columns: &[Column],
    display: &DisplayOptions,
    global_max_widths: &[u16],
) -> fmt::Result {
    if !display.is_section_enabled(Section::Smaps) {
        return Ok(());
    }
    if diff.smaps_rollup_a.is_empty() && diff.smaps_rollup_b.is_empty() {
        return Ok(());
    }

    let mut process_keys: BTreeSet<&String> = diff.smaps_rollup_a.keys().collect();
    process_keys.extend(diff.smaps_rollup_b.keys());

    let max_field_for = |pkey: &&String, field: &str| -> u64 {
        let a = diff
            .smaps_rollup_a
            .get(*pkey)
            .and_then(|m| m.get(field).copied())
            .unwrap_or(0);
        let b = diff
            .smaps_rollup_b
            .get(*pkey)
            .and_then(|m| m.get(field).copied())
            .unwrap_or(0);
        a.max(b)
    };
    let abs_rss_delta = |pkey: &&String| -> u64 {
        let a = diff
            .smaps_rollup_a
            .get(*pkey)
            .and_then(|m| m.get("Rss").copied())
            .unwrap_or(0);
        let b = diff
            .smaps_rollup_b
            .get(*pkey)
            .and_then(|m| m.get("Rss").copied())
            .unwrap_or(0);
        (b as i128 - a as i128).unsigned_abs() as u64
    };
    let mut sorted_process_keys: Vec<&String> = process_keys.iter().copied().collect();
    sorted_process_keys.sort_by(|a, b| {
        abs_rss_delta(b)
            .cmp(&abs_rss_delta(a))
            .then_with(|| max_field_for(b, "Rss").cmp(&max_field_for(a, "Rss")))
            .then_with(|| a.cmp(b))
    });

    // Pre-pass: any (process, key) pair with a non-equal
    // delta? Scan the FULL key set BEFORE truncating to the
    // section line limit so a top-N-without-deltas case
    // (movers below the truncation line) still surfaces the
    // section. Truncating first would suppress the header
    // when the top-N happen to carry equal baseline/candidate
    // values but lower-ranked keys carry real deltas.
    let any_delta = sorted_process_keys.iter().any(|pkey| {
        let a = diff.smaps_rollup_a.get(*pkey);
        let b = diff.smaps_rollup_b.get(*pkey);
        let mut keys: BTreeSet<&String> =
            a.map(|m| m.keys().collect()).unwrap_or_default();
        if let Some(m) = b {
            keys.extend(m.keys());
        }
        keys.iter().any(|k| {
            let av = a.and_then(|m| m.get(*k).copied());
            let bv = b.and_then(|m| m.get(*k).copied());
            av != bv
        })
    });
    if !any_delta {
        return Ok(());
    }

    // Apply the line-limit truncation AFTER the delta scan
    // so the section-suppression heuristic sees the full set.
    if display.section_line_limit > 0 {
        sorted_process_keys.truncate(display.section_line_limit);
    }

    writeln!(w)?;
    writeln!(w, "## smaps_rollup")?;
    let mut st = if global_max_widths.is_empty() {
        display.new_table()
    } else {
        display.new_constrained_table(global_max_widths)
    };
    // Header: under GroupBy::All the key shape is compound
    // (cgroup\x00pcomm) and the row column carries the
    // hierarchical comm cell, matching the primary table's
    // "comm" header at primary.rs:298 and 426. Under any
    // non-All group_by the keys are bare pcomm strings and
    // the column carries that.
    let is_compound = group_by == GroupBy::All;
    let group_header = if is_compound { "comm" } else { "pcomm" };
    st.set_header(colored_header_with_sort(columns, group_header, diff.sort_metric_name));

    // For All mode, re-sort by cgroup hierarchy (keys are
    // compound cgroup\x00pcomm). Track segments for tree headings.
    let mut sorted_keys = sorted_process_keys.clone();
    if is_compound {
        sorted_keys.sort();
    }

    let mut last_segs: Vec<&str> = Vec::new();

    for pkey in &sorted_keys {
        let (cg_part, display_process) = if is_compound {
            pkey.split_once('\x00').unwrap_or(("", pkey))
        } else {
            ("", pkey.as_str())
        };

        if is_compound {
            let segs: Vec<&str> = cg_part.split('/').filter(|s| !s.is_empty()).collect();
            let common = segs
                .iter()
                .zip(last_segs.iter())
                .take_while(|(a, b)| a == b)
                .count();
            if common < last_segs.len() || segs.len() > last_segs.len() {
                for (depth, seg) in segs.iter().enumerate().skip(common) {
                    let indent = "  ".repeat(depth);
                    let label = format!("{indent}{seg}");
                    let hcells: Vec<comfy_table::Cell> = columns
                        .iter()
                        .map(|c| {
                            if *c == Column::Group {
                                comfy_table::Cell::new(&label)
                                    .fg(depth_color(depth))
                                    .add_attribute(comfy_table::Attribute::Bold)
                            } else {
                                comfy_table::Cell::new("")
                            }
                        })
                        .collect();
                    st.add_row(hcells);
                }
                last_segs = segs;
            }
        }

        let a = diff.smaps_rollup_a.get(*pkey);
        let b = diff.smaps_rollup_b.get(*pkey);
        let mut keys_union: BTreeSet<&String> =
            a.map(|m| m.keys().collect()).unwrap_or_default();
        if let Some(m) = b {
            keys_union.extend(m.keys());
        }
        for sk in keys_union {
            let av = a.and_then(|m| m.get(sk).copied());
            let bv = b.and_then(|m| m.get(sk).copied());
            if av == bv {
                continue;
            }
            let a_cell = av
                .map(|v| format_scaled_u64(v, ScaleLadder::Bytes))
                .unwrap_or_else(|| "-".to_string());
            let b_cell = bv
                .map(|v| format_scaled_u64(v, ScaleLadder::Bytes))
                .unwrap_or_else(|| "-".to_string());
            let value_cell = format!("{a_cell} \u{2192} {b_cell}");
            let a_val = av.unwrap_or(0);
            let b_val = bv.unwrap_or(0);
            let delta = b_val as i128 - a_val as i128;
            let delta_cell = if av.is_none() || bv.is_none() {
                "-".to_string()
            } else {
                format_delta_cell(delta as f64, ScaleLadder::Bytes)
            };
            let pct_cell = if a_val == 0 || av.is_none() || bv.is_none() {
                "-".to_string()
            } else {
                let pct = (delta as f64 / a_val as f64) * 100.0;
                format!("{pct:+.1}%")
            };
            let cg_depth = last_segs.len();
            let group_label = format!("{}  {}", "  ".repeat(cg_depth + 1), display_process);
            let delta_pct_opt: Option<f64> = if a_val > 0 && av.is_some() && bv.is_some() {
                Some(delta as f64 / a_val as f64)
            } else {
                None
            };
            let delta_opt: Option<f64> = if av.is_some() && bv.is_some() {
                Some(delta as f64)
            } else {
                None
            };
            let string_cells: Vec<String> = columns
                .iter()
                .map(|c| match c {
                    Column::Group => group_label.clone(),
                    Column::Threads => String::new(),
                    Column::Metric => sk.clone(),
                    Column::Baseline => a_cell.clone(),
                    Column::Candidate => b_cell.clone(),
                    Column::Arrow => value_cell.clone(),
                    Column::Delta => delta_cell.clone(),
                    Column::Pct => pct_cell.clone(),
                    Column::Uptime => String::new(),
                    _ => String::new(),
                })
                .collect();
            let cells: Vec<comfy_table::Cell> = string_cells
                .into_iter()
                .zip(columns.iter())
                .map(|(s, col)| color_diff_cell(s, *col, delta_opt, delta_pct_opt, None, None))
                .collect();
            st.add_row(cells);
        }
    }
    writeln!(w, "{st}")?;
    Ok(())
}
