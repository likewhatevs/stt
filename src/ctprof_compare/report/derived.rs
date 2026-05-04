//! Derived-metrics table emitter for [`super::write_diff`].
//!
//! Renders the `## Derived metrics` table following the
//! primary-metrics block. The outer gate mirrors
//! [`super::primary`]: open the table when EITHER
//! [`Section::Derived`] or [`Section::TaskstatsDelay`] is
//! enabled, then per-row gating restricts the visible subset.
//!
//! Reuses [`super::primary::build_primary_hier`] for the
//! `GroupBy::All` cgroup + pcomm + comm hierarchy so primary
//! and derived sub-tables share the tree heading discipline
//! and rank-by-sort-key ordering.

use std::collections::BTreeMap;
use std::fmt;

use super::super::CTPROF_DERIVED_METRICS;
use super::super::columns::{Column, Section};
use super::super::diff_types::{CtprofDiff, DerivedRow};
use super::super::options::GroupBy;
use super::super::render::{color_diff_cell, colored_header_with_sort, render_derived_row_cells};
use super::super::runner::DisplayOptions;
use super::primary::{build_primary_hier, emit_cgroup_segments, emit_pcomm_heading};

/// Render the `## Derived metrics` table when either
/// [`Section::Derived`] or [`Section::TaskstatsDelay`] is
/// enabled and the diff carries at least one derived row.
pub(super) fn write_derived_section<W: fmt::Write>(
    w: &mut W,
    diff: &CtprofDiff,
    group_by: GroupBy,
    group_header: &'static str,
    columns: &[Column],
    display: &DisplayOptions,
    global_max_widths: &[u16],
) -> fmt::Result {
    if !(display.is_section_enabled(Section::Derived)
        || display.is_section_enabled(Section::TaskstatsDelay))
        || diff.derived_rows.is_empty()
    {
        return Ok(());
    }

    let derived_rows: Vec<&DerivedRow> = diff
        .derived_rows
        .iter()
        .filter(|row| {
            if !display.is_metric_enabled(row.metric_name) {
                return false;
            }
            let metric = CTPROF_DERIVED_METRICS
                .iter()
                .find(|d| d.name == row.metric_name)
                .expect("derived metric_name from CTPROF_DERIVED_METRICS");
            display.is_section_enabled(metric.section)
        })
        .collect();

    // Build uptime lookup from primary rows for derived rendering.
    let uptime_map: BTreeMap<&str, Option<f64>> = diff
        .rows
        .iter()
        .map(|r| (r.group_key.as_str(), r.uptime_pct))
        .collect();

    if group_by == GroupBy::All {
        write_derived_all(
            w,
            &derived_rows,
            columns,
            display,
            global_max_widths,
            &uptime_map,
            diff.sort_metric_name,
        )?;
    } else {
        write_derived_flat(
            w,
            &derived_rows,
            columns,
            display,
            group_header,
            &uptime_map,
            diff.sort_metric_name,
        )?;
    }
    Ok(())
}

fn write_derived_all<W: fmt::Write>(
    w: &mut W,
    derived_rows: &[&DerivedRow],
    columns: &[Column],
    display: &DisplayOptions,
    global_max_widths: &[u16],
    uptime_map: &BTreeMap<&str, Option<f64>>,
    sort_metric_name: Option<&'static str>,
) -> fmt::Result {
    // Hierarchical derived rendering — same tree as primary.
    //
    // Pin fudged rows (display_key starting with `[fudged`)
    // outside the truncation budget so the operator-requested
    // N:1 merges always render. Mirrors the primary-section
    // partition at primary.rs:276-289 — without the partition,
    // the line limit could silently bury fudged rows whenever
    // the top-N normal movers consumed the budget first.
    let (fudged_rows, normal_rows): (Vec<&DerivedRow>, Vec<&DerivedRow>) = derived_rows
        .iter()
        .copied()
        .partition(|r| r.display_key.starts_with("[fudged"));
    let limited: Vec<&DerivedRow> = if display.section_line_limit > 0 {
        let mut out: Vec<&DerivedRow> = fudged_rows.clone();
        let budget = display.section_line_limit.saturating_sub(fudged_rows.len());
        out.extend(normal_rows.into_iter().take(budget));
        out
    } else {
        let mut out = fudged_rows;
        out.extend(normal_rows);
        out
    };

    let hier = build_primary_hier(limited.iter().copied(), |row: &DerivedRow| {
        row.group_key.as_str()
    });

    writeln!(w)?;
    writeln!(w, "## Derived metrics")?;
    let mut dt = display.new_constrained_table(global_max_widths);
    dt.set_header(colored_header_with_sort(columns, "comm", sort_metric_name));
    let mut last_segs: Vec<&str> = Vec::new();
    let mut last_pc = "";
    for h in &hier {
        if let Some(new_segs) = emit_cgroup_segments(&mut dt, h.cgroup, &last_segs, columns) {
            last_segs = new_segs;
            last_pc = "";
        }
        if h.pcomm != last_pc {
            emit_pcomm_heading(&mut dt, h.pcomm, last_segs.len(), columns);
            last_pc = h.pcomm;
        }
        let mut cells = render_derived_row_cells(h.row, columns);
        if let Some(pos) = columns.iter().position(|c| *c == Column::Group) {
            let cg_depth = last_segs.len();
            cells[pos] = format!("{}  {}", "  ".repeat(cg_depth + 1), h.comm);
        }
        let colored: Vec<comfy_table::Cell> = cells
            .into_iter()
            .zip(columns.iter())
            .map(|(s, col)| {
                let up = uptime_map.get(h.row.group_key.as_str()).copied().flatten();
                if *col == Column::Uptime {
                    let text = match up {
                        Some(pct) => format!("{pct:.0}%"),
                        None => "-".to_string(),
                    };
                    color_diff_cell(
                        text,
                        *col,
                        h.row.delta,
                        h.row.delta_pct,
                        up,
                        h.row.sort_by_delta,
                    )
                } else {
                    color_diff_cell(
                        s,
                        *col,
                        h.row.delta,
                        h.row.delta_pct,
                        up,
                        h.row.sort_by_delta,
                    )
                }
            })
            .collect();
        dt.add_row(colored);
    }
    writeln!(w, "{dt}")?;
    Ok(())
}

fn write_derived_flat<W: fmt::Write>(
    w: &mut W,
    derived_rows: &[&DerivedRow],
    columns: &[Column],
    display: &DisplayOptions,
    group_header: &'static str,
    uptime_map: &BTreeMap<&str, Option<f64>>,
    sort_metric_name: Option<&'static str>,
) -> fmt::Result {
    writeln!(w)?;
    writeln!(w, "## Derived metrics")?;
    let mut dt = display.new_table();
    dt.set_header(colored_header_with_sort(
        columns,
        group_header,
        sort_metric_name,
    ));
    let d_limit = if display.section_line_limit > 0 {
        &derived_rows[..derived_rows.len().min(display.section_line_limit)]
    } else {
        derived_rows
    };
    for row in d_limit {
        let string_cells = render_derived_row_cells(row, columns);
        let cells: Vec<comfy_table::Cell> = string_cells
            .into_iter()
            .zip(columns.iter())
            .map(|(s, col)| {
                let up = uptime_map.get(row.group_key.as_str()).copied().flatten();
                if *col == Column::Uptime {
                    let text = match up {
                        Some(pct) => format!("{pct:.0}%"),
                        None => "-".to_string(),
                    };
                    color_diff_cell(text, *col, row.delta, row.delta_pct, up, row.sort_by_delta)
                } else {
                    color_diff_cell(s, *col, row.delta, row.delta_pct, up, row.sort_by_delta)
                }
            })
            .collect();
        dt.add_row(cells);
    }
    writeln!(w, "{dt}")?;
    Ok(())
}
