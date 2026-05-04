//! Primary-metrics table emitter for [`super::write_diff`].
//!
//! Renders the per-thread primary table (52 non-taskstats rows
//! plus 34 taskstats genetlink rows). Two [`Section`] flags
//! share the table: [`Section::Primary`] gates the non-taskstats
//! rows; [`Section::TaskstatsDelay`] gates the genetlink rows.
//! Either enabled keeps the table open and the per-row gate
//! filters the matching subset.
//!
//! Three rendering branches — [`GroupBy::All`] (cgroup +
//! pcomm + comm hierarchy with ranked tree headings),
//! [`GroupBy::Cgroup`] (table-per-parent), and the
//! flat-table default — share the global column-width vector
//! computed by [`super::write_diff`] so column widths align
//! across primary and derived sub-tables.
//!
//! See [`super::write_diff`] for the orchestrator that calls
//! into [`write_primary_section`] before emitting derived,
//! smaps, cgroup, host-PSI, sched_ext, and the orphan / fudge
//! lists.
//!
//! Helper [`build_primary_hier`] is shared with
//! [`super::derived`] which performs the same cgroup +
//! pcomm grouping over `DerivedRow`s.

use std::collections::BTreeMap;
use std::fmt;

use super::super::CTPROF_METRICS;
use super::super::columns::{Column, Section};
use super::super::diff_types::{CtprofDiff, DiffRow};
use super::super::options::GroupBy;
use super::super::render::{
    cgroup_parent_leaf, color_diff_cell, colored_header_with_sort, render_diff_row_cells,
};
use super::super::runner::DisplayOptions;

/// Cgroup-tree heading color by depth — green at the root,
/// cyan at level 1, dark grey thereafter. Shared with
/// [`super::derived`] so the primary and derived hierarchies
/// match visually.
pub(super) fn depth_color(depth: usize) -> comfy_table::Color {
    match depth {
        0 => comfy_table::Color::Green,
        1 => comfy_table::Color::Cyan,
        _ => comfy_table::Color::DarkGrey,
    }
}

/// Render the `## Primary metrics` section (and its taskstats
/// subset when [`Section::TaskstatsDelay`] is enabled). The
/// outer gate keeps the table open while EITHER section is
/// enabled — `--sections taskstats-delay` alone still emits
/// the table containing only the 34 taskstats rows;
/// `--sections primary` alone emits the table containing only
/// the 52 non-taskstats rows; either combined or the empty
/// default ("all on") emits all rows.
pub(super) fn write_primary_section<W: fmt::Write>(
    w: &mut W,
    diff: &CtprofDiff,
    group_by: GroupBy,
    group_header: &'static str,
    columns: &[Column],
    display: &DisplayOptions,
    global_max_widths: &[u16],
) -> fmt::Result {
    if !display.is_section_enabled(Section::Primary)
        && !display.is_section_enabled(Section::TaskstatsDelay)
    {
        return Ok(());
    }

    // Filter rows first.
    let primary_rows: Vec<&DiffRow> = diff
        .rows
        .iter()
        .filter(|row| {
            if !display.is_metric_enabled(row.metric_name) {
                return false;
            }
            let metric = CTPROF_METRICS
                .iter()
                .find(|m| m.name == row.metric_name)
                .expect("metric_name comes from CTPROF_METRICS via build_row");
            display.is_section_enabled(metric.section)
        })
        .collect();

    if group_by == GroupBy::All {
        write_primary_all(
            w,
            &primary_rows,
            columns,
            display,
            global_max_widths,
            diff.sort_metric_name,
        )?;
    } else if group_by == GroupBy::Cgroup {
        write_primary_cgroup(w, &primary_rows, columns, display, diff.sort_metric_name)?;
    } else {
        write_primary_flat(
            w,
            &primary_rows,
            columns,
            display,
            group_header,
            diff.sort_metric_name,
        )?;
    }

    Ok(())
}

/// One row in the cgroup + pcomm + comm hierarchy used by the
/// `GroupBy::All` rendering of primary and (analogously) derived
/// rows. Stored references all point into the underlying
/// [`DiffRow::group_key`] string, which uses NUL bytes
/// (`\x00`) as cgroup / pcomm / comm field separators.
pub(super) struct HierRow<'a, R> {
    pub(super) cgroup: &'a str,
    pub(super) pcomm: &'a str,
    pub(super) comm: &'a str,
    pub(super) row: &'a R,
}

/// Build a sorted hierarchy from a slice of rows for
/// [`GroupBy::All`] rendering. Uses pre-sorted row index as
/// the rank within (cgroup, pcomm) and (cgroup) buckets so the
/// chosen `--sort-by` metric drives the tree heading order:
/// the cgroup containing the biggest mover floats to the top,
/// inside it the pcomm containing the biggest mover floats to
/// the top, and within that bucket rows render in sort order.
pub(super) fn build_primary_hier<'a, R>(
    rows: impl IntoIterator<Item = &'a R>,
    key_fn: impl Fn(&R) -> &str,
) -> Vec<HierRow<'a, R>>
where
    R: 'a,
{
    let mut hier: Vec<HierRow<'a, R>> = rows
        .into_iter()
        .map(|row| {
            let mut parts = key_fn(row).splitn(3, '\x00');
            let cgroup = parts.next().unwrap_or("");
            let pcomm = parts.next().unwrap_or("");
            let comm = parts.next().unwrap_or(pcomm);
            HierRow {
                cgroup,
                pcomm,
                comm,
                row,
            }
        })
        .collect();
    let row_rank: BTreeMap<*const R, usize> = hier
        .iter()
        .enumerate()
        .map(|(i, h)| (h.row as *const R, i))
        .collect();
    let mut leaf_rank: BTreeMap<(&str, &str), usize> = BTreeMap::new();
    let mut cg_rank: BTreeMap<&str, usize> = BTreeMap::new();
    for h in &hier {
        let rank = row_rank[&(h.row as *const R)];
        let le = leaf_rank.entry((h.cgroup, h.pcomm)).or_insert(usize::MAX);
        if rank < *le {
            *le = rank;
        }
        let ce = cg_rank.entry(h.cgroup).or_insert(usize::MAX);
        if rank < *ce {
            *ce = rank;
        }
    }
    hier.sort_by(|a, b| {
        let cga = cg_rank.get(a.cgroup).copied().unwrap_or(usize::MAX);
        let cgb = cg_rank.get(b.cgroup).copied().unwrap_or(usize::MAX);
        cga.cmp(&cgb)
            .then_with(|| {
                let sa = leaf_rank
                    .get(&(a.cgroup, a.pcomm))
                    .copied()
                    .unwrap_or(usize::MAX);
                let sb = leaf_rank
                    .get(&(b.cgroup, b.pcomm))
                    .copied()
                    .unwrap_or(usize::MAX);
                sa.cmp(&sb)
            })
            .then_with(|| {
                let ra = row_rank[&(a.row as *const R)];
                let rb = row_rank[&(b.row as *const R)];
                ra.cmp(&rb)
            })
    });
    hier
}

/// Emit a cgroup-segment heading row. The leaf segment of the
/// path renders at depth-colored bold; ancestor segments seen
/// in `last_segments` are skipped. Returns the new
/// `last_segments` vector for the caller to track.
pub(super) fn emit_cgroup_segments<'a>(
    table: &mut comfy_table::Table,
    cgroup: &'a str,
    last_segments: &[&'a str],
    columns: &[Column],
) -> Option<Vec<&'a str>> {
    let segments: Vec<&str> = cgroup.split('/').filter(|s| !s.is_empty()).collect();
    let common = segments
        .iter()
        .zip(last_segments.iter())
        .take_while(|(a, b)| a == b)
        .count();
    let cg_changed = common < last_segments.len() || segments.len() > last_segments.len();
    if cg_changed {
        for (depth, seg) in segments.iter().enumerate().skip(common) {
            let indent = "  ".repeat(depth);
            let label = format!("{indent}{seg}");
            let heading_cells: Vec<comfy_table::Cell> = columns
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
            table.add_row(heading_cells);
        }
        Some(segments)
    } else {
        None
    }
}

/// Emit a pcomm heading row at the current cgroup-depth indent.
pub(super) fn emit_pcomm_heading(
    table: &mut comfy_table::Table,
    pcomm: &str,
    cg_depth: usize,
    columns: &[Column],
) {
    let indent = "  ".repeat(cg_depth);
    let label = format!("{indent}{pcomm}");
    let heading_cells: Vec<comfy_table::Cell> = columns
        .iter()
        .map(|c| {
            if *c == Column::Group {
                comfy_table::Cell::new(&label)
                    .fg(comfy_table::Color::White)
                    .add_attribute(comfy_table::Attribute::Bold)
            } else {
                comfy_table::Cell::new("")
            }
        })
        .collect();
    table.add_row(heading_cells);
}

fn write_primary_all<W: fmt::Write>(
    w: &mut W,
    primary_rows: &[&DiffRow],
    columns: &[Column],
    display: &DisplayOptions,
    global_max_widths: &[u16],
    sort_metric_name: Option<&'static str>,
) -> fmt::Result {
    // Sort + truncate BEFORE organizing into the cgroup
    // tree. primary_rows are already delta-sorted from
    // compare(). Apply the line limit here so the tree
    // only contains the top movers.
    //
    // Pin fudged rows: rows whose display_key starts
    // with `[fudged` represent N:1 merges that the
    // operator specifically asked the fudge stage to
    // produce. Truncating them silently buries the
    // most informative output. Partition first so
    // every fudged row survives the truncation, then
    // fill the remaining budget with the top non-fudged
    // movers in delta order.
    let (fudged_rows, normal_rows): (Vec<&DiffRow>, Vec<&DiffRow>) = primary_rows
        .iter()
        .copied()
        .partition(|r| r.display_key.starts_with("[fudged"));
    let limited_rows: Vec<&DiffRow> = if display.section_line_limit > 0 {
        let mut out: Vec<&DiffRow> = fudged_rows.clone();
        let budget = display.section_line_limit.saturating_sub(fudged_rows.len());
        out.extend(normal_rows.into_iter().take(budget));
        out
    } else {
        let mut out = fudged_rows;
        out.extend(normal_rows);
        out
    };

    let hier = build_primary_hier(limited_rows.iter().copied(), |row: &DiffRow| {
        row.group_key.as_str()
    });

    // Two-pass: first measure data-only widths, then build
    // the real table with heading rows constrained to those
    // widths so headings can't inflate columns.
    writeln!(w, "## Primary metrics")?;
    let mut last_segments: Vec<&str> = Vec::new();
    let mut last_pcomm = "";
    let mut table = display.new_constrained_table(global_max_widths);
    table.set_header(colored_header_with_sort(columns, "comm", sort_metric_name));

    for h in &hier {
        if let Some(new_segments) =
            emit_cgroup_segments(&mut table, h.cgroup, &last_segments, columns)
        {
            last_segments = new_segments;
            last_pcomm = "";
        }

        if h.pcomm != last_pcomm {
            emit_pcomm_heading(&mut table, h.pcomm, last_segments.len(), columns);
            last_pcomm = h.pcomm;
        }

        let mut string_cells = render_diff_row_cells(h.row, columns);
        if let Some(pos) = columns.iter().position(|c| *c == Column::Group) {
            let cg_depth = last_segments.len();
            // Preserve the [fudged] marker for rows that
            // came from the N:1 fudge merge — without
            // this preservation the comm-overwrite below
            // would silently strip the indicator and
            // fudged rows would render identically to
            // naturally-matched rows.
            let fudge_marker = if h.row.display_key.starts_with("[fudged") {
                h.row.display_key.as_str()
            } else {
                ""
            };
            // Insert a single space between marker and
            // comm; empty marker leaves the cell as
            // just `<indent>  <comm>` (matches
            // pre-fudge layout).
            let fudge_separator = if fudge_marker.is_empty() { "" } else { " " };
            string_cells[pos] = format!(
                "{}  {}{}{}",
                "  ".repeat(cg_depth + 1),
                fudge_marker,
                fudge_separator,
                h.comm,
            );
        }
        let cells: Vec<comfy_table::Cell> = string_cells
            .into_iter()
            .zip(columns.iter())
            .map(|(s, col)| {
                color_diff_cell(
                    s,
                    *col,
                    h.row.delta,
                    h.row.delta_pct,
                    h.row.uptime_pct,
                    h.row.sort_by_delta,
                )
            })
            .collect();
        table.add_row(cells);
    }
    writeln!(w, "{table}")?;
    Ok(())
}

fn write_primary_cgroup<W: fmt::Write>(
    w: &mut W,
    primary_rows: &[&DiffRow],
    columns: &[Column],
    display: &DisplayOptions,
    sort_metric_name: Option<&'static str>,
) -> fmt::Result {
    // Hierarchical cgroup rendering: group rows by parent
    // path, emit a sub-heading per parent, show only the
    // leaf segment in the group column.
    let mut by_parent: BTreeMap<&str, Vec<&DiffRow>> = BTreeMap::new();
    for row in primary_rows {
        let (parent, _) = cgroup_parent_leaf(&row.display_key);
        by_parent.entry(parent).or_default().push(row);
    }
    for (parent, rows) in &by_parent {
        writeln!(w)?;
        writeln!(w, "\x1b[1;32m## {parent}\x1b[0m")?;
        let mut table = display.new_table();
        table.set_header(colored_header_with_sort(
            columns,
            "cgroup",
            sort_metric_name,
        ));
        let cg_limit = if display.section_line_limit > 0 {
            &rows[..rows.len().min(display.section_line_limit)]
        } else {
            &rows[..]
        };
        for row in cg_limit {
            let (_, leaf) = cgroup_parent_leaf(&row.display_key);
            let mut string_cells = render_diff_row_cells(row, columns);
            // Replace group cell with leaf segment.
            if let Some(pos) = columns.iter().position(|c| *c == Column::Group) {
                string_cells[pos] = leaf.to_string();
            }
            let cells: Vec<comfy_table::Cell> = string_cells
                .into_iter()
                .zip(columns.iter())
                .map(|(s, col)| {
                    color_diff_cell(
                        s,
                        *col,
                        row.delta,
                        row.delta_pct,
                        row.uptime_pct,
                        row.sort_by_delta,
                    )
                })
                .collect();
            table.add_row(cells);
        }
        writeln!(w, "{table}")?;
    }
    Ok(())
}

fn write_primary_flat<W: fmt::Write>(
    w: &mut W,
    primary_rows: &[&DiffRow],
    columns: &[Column],
    display: &DisplayOptions,
    group_header: &'static str,
    sort_metric_name: Option<&'static str>,
) -> fmt::Result {
    writeln!(w, "## Primary metrics")?;
    let mut table = display.new_table();
    table.set_header(colored_header_with_sort(
        columns,
        group_header,
        sort_metric_name,
    ));
    let limit_iter = if display.section_line_limit > 0 {
        &primary_rows[..primary_rows.len().min(display.section_line_limit)]
    } else {
        primary_rows
    };
    for row in limit_iter {
        let string_cells = render_diff_row_cells(row, columns);
        let cells: Vec<comfy_table::Cell> = string_cells
            .into_iter()
            .zip(columns.iter())
            .map(|(s, col)| {
                color_diff_cell(
                    s,
                    *col,
                    row.delta,
                    row.delta_pct,
                    row.uptime_pct,
                    row.sort_by_delta,
                )
            })
            .collect();
        table.add_row(cells);
    }
    writeln!(w, "{table}")?;
    Ok(())
}
