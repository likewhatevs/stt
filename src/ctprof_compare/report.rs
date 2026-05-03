//! `write_diff` — primary diff renderer for `CtprofDiff`.
//!
//! This is the single multi-section emitter: per-thread primary
//! table, derived-metric table, fudged-pair list, host-PSI
//! summary, per-cgroup summary, per-cgroup PSI, and the
//! baseline-/candidate-only key lists. Each sub-section is gated
//! on [`super::DisplayOptions::is_section_enabled`] so
//! `--sections` always wins over the data-availability
//! suppression heuristic, and column widths are measured once
//! across primary + derived tables so every sub-table aligns.
//!
//! Cell rendering is delegated to [`super::render`]; metric
//! lookup to [`super::metrics`]; column resolution to
//! [`super::columns`]; the table builder lives on
//! [`super::DisplayOptions`]. Splitting the body further (per
//! Section emitter) is taskified as #586.

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use super::columns::Column;
use super::diff_types::{CtprofDiff, DerivedRow, DiffRow};
use super::options::GroupBy;
use super::render::{
    cgroup_cell, cgroup_parent_leaf, color_diff_cell, colored_header_with_sort,
    format_psi_avg_cell, psi_pair_has_data, psi_resource_accessors, psi_resource_has_data,
    render_derived_row_cells, render_diff_row_cells,
};
use super::runner::DisplayOptions;
use super::scale::{
    ScaleLadder, cgroup_limits_cell, cgroup_optional_limit_cell, format_delta_cell,
    format_scaled_u64,
};
use super::{CTPROF_DERIVED_METRICS, CTPROF_METRICS, columns::Section};
use crate::ctprof::CgroupStats;

/// Render [`CtprofDiff`] into `w`. The formatter layer lives
/// here so tests can inspect exactly what `print_diff` would
/// emit without shelling through stdout capture. Write errors
/// propagate as [`std::fmt::Error`] — callers that write into an
/// infallible sink (`String`) can unwrap or ignore.
///
/// `display` controls per-row column layout, terminal-width
/// wrapping, and per-section filtering: see [`DisplayFormat`] /
/// [`Column`] / [`Section`] / [`DisplayOptions`] for the
/// resolution rules. Each sub-table emission below is gated on
/// [`DisplayOptions::is_section_enabled`] before its
/// data-availability check, so `--sections` always wins over
/// the per-section zero-suppression heuristic.
pub fn write_diff<W: fmt::Write>(
    w: &mut W,
    diff: &CtprofDiff,
    baseline_path: &Path,
    candidate_path: &Path,
    group_by: GroupBy,
    display: &DisplayOptions,
) -> fmt::Result {
    let group_header = match group_by {
        GroupBy::Pcomm => "pcomm",
        GroupBy::Cgroup => "cgroup",
        GroupBy::Comm => "comm-pattern",
        GroupBy::CommExact => "comm",
        GroupBy::All => "comm",
    };

    let mut columns = display.resolved_compare_columns();
    let has_sort_col = diff.rows.first().is_some_and(|r| r.sort_by_cell.is_some());
    if has_sort_col {
        columns.push(Column::SortBy);
    }

    // Compute column widths from ALL data rows (primary + derived)
    // so every table in every section shares the same widths.
    // Heading rows are constrained to these widths via
    // new_constrained_table so they can't inflate columns.
    let global_max_widths: Vec<u16> = if group_by == GroupBy::All {
        let mut measure = display.new_table();
        measure.set_header(colored_header_with_sort(
            &columns,
            group_header,
            diff.sort_metric_name,
        ));
        for row in &diff.rows {
            let mut cells = render_diff_row_cells(row, &columns);
            if let Some(pos) = columns.iter().position(|c| *c == Column::Group) {
                let comm = row.group_key.splitn(3, '\x00').nth(2).unwrap_or("");
                cells[pos] = comm.to_string();
            }
            measure.add_row(cells);
        }
        for row in &diff.derived_rows {
            let mut cells = render_derived_row_cells(row, &columns);
            if let Some(pos) = columns.iter().position(|c| *c == Column::Group) {
                let comm = row.group_key.splitn(3, '\x00').nth(2).unwrap_or("");
                cells[pos] = comm.to_string();
            }
            measure.add_row(cells);
        }
        measure.column_max_content_widths()
    } else {
        Vec::new()
    };

    // The primary table renders rows whose metric.section is
    // enabled. Two sections share the table:
    //   - Section::Primary: the 52 non-taskstats rows.
    //   - Section::TaskstatsDelay: the 34 taskstats genetlink rows.
    // The outer gate keeps the table open while EITHER section
    // is enabled — `--sections taskstats-delay` alone still emits
    // the table containing only the 34 taskstats rows;
    // `--sections primary` alone emits the table containing only
    // the 52 non-taskstats rows; either combined or the empty
    // default ("all on") emits all rows.
    if display.is_section_enabled(Section::Primary)
        || display.is_section_enabled(Section::TaskstatsDelay)
    {
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

            struct HierRow<'a> {
                cgroup: &'a str,
                pcomm: &'a str,
                comm: &'a str,
                row: &'a DiffRow,
            }
            let mut hier: Vec<HierRow<'_>> = limited_rows
                .iter()
                .map(|row| {
                    let mut parts = row.group_key.splitn(3, '\x00');
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
            // Use pre-sorted row index as score so hierarchy honors
            // --sort-by. Rows arrive sorted by the chosen metric.
            // Lower index = higher priority (biggest mover first).
            let row_rank: BTreeMap<*const DiffRow, usize> = hier
                .iter()
                .enumerate()
                .map(|(i, h)| (h.row as *const DiffRow, i))
                .collect();
            let mut leaf_rank: BTreeMap<(&str, &str), usize> = BTreeMap::new();
            let mut cg_rank: BTreeMap<&str, usize> = BTreeMap::new();
            for h in &hier {
                let rank = row_rank[&(h.row as *const DiffRow)];
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
                        let ra = row_rank[&(a.row as *const DiffRow)];
                        let rb = row_rank[&(b.row as *const DiffRow)];
                        ra.cmp(&rb)
                    })
            });

            // Two-pass: first measure data-only widths, then build
            // the real table with heading rows constrained to those
            // widths so headings can't inflate columns.
            writeln!(w, "## Primary metrics")?;
            let mut last_segments: Vec<&str> = Vec::new();
            let mut last_pcomm = "";
            let mut table = display.new_constrained_table(&global_max_widths);
            table.set_header(colored_header_with_sort(
                &columns,
                "comm",
                diff.sort_metric_name,
            ));

            let depth_color = |depth: usize| -> comfy_table::Color {
                match depth {
                    0 => comfy_table::Color::Green,
                    1 => comfy_table::Color::Cyan,
                    _ => comfy_table::Color::DarkGrey,
                }
            };

            for h in &hier {
                let segments: Vec<&str> = h.cgroup.split('/').filter(|s| !s.is_empty()).collect();

                let common = segments
                    .iter()
                    .zip(last_segments.iter())
                    .take_while(|(a, b)| a == b)
                    .count();

                let cg_changed =
                    common < last_segments.len() || segments.len() > last_segments.len();
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
                    last_segments = segments;
                    last_pcomm = "";
                }

                if h.pcomm != last_pcomm {
                    let cg_depth = last_segments.len();
                    let indent = "  ".repeat(cg_depth);
                    let label = format!("{indent}{}", h.pcomm);
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
                    last_pcomm = h.pcomm;
                }

                let mut string_cells = render_diff_row_cells(h.row, &columns);
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
        } else if group_by == GroupBy::Cgroup {
            // Hierarchical cgroup rendering: group rows by parent
            // path, emit a sub-heading per parent, show only the
            // leaf segment in the group column.
            let mut by_parent: BTreeMap<&str, Vec<&DiffRow>> = BTreeMap::new();
            for row in &primary_rows {
                let (parent, _) = cgroup_parent_leaf(&row.display_key);
                by_parent.entry(parent).or_default().push(row);
            }
            for (parent, rows) in &by_parent {
                writeln!(w)?;
                writeln!(w, "\x1b[1;32m## {parent}\x1b[0m")?;
                let mut table = display.new_table();
                table.set_header(colored_header_with_sort(
                    &columns,
                    "cgroup",
                    diff.sort_metric_name,
                ));
                let cg_limit = if display.section_line_limit > 0 {
                    &rows[..rows.len().min(display.section_line_limit)]
                } else {
                    &rows[..]
                };
                for row in cg_limit {
                    let (_, leaf) = cgroup_parent_leaf(&row.display_key);
                    let mut string_cells = render_diff_row_cells(row, &columns);
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
        } else {
            writeln!(w, "## Primary metrics")?;
            let mut table = display.new_table();
            table.set_header(colored_header_with_sort(
                &columns,
                group_header,
                diff.sort_metric_name,
            ));
            let limit_iter = if display.section_line_limit > 0 {
                &primary_rows[..primary_rows.len().min(display.section_line_limit)]
            } else {
                &primary_rows[..]
            };
            for row in limit_iter {
                let string_cells = render_diff_row_cells(row, &columns);
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
    }

    // Derived-table outer gate mirrors the primary-table pattern:
    // open the table when EITHER `Section::Derived` (the eight
    // pre-existing derivations) OR `Section::TaskstatsDelay` (the
    // nine taskstats-derived rollups) is enabled. Per-row gating
    // below keeps `--sections taskstats-delay` from leaking
    // unrelated derivations into the table.
    if (display.is_section_enabled(Section::Derived)
        || display.is_section_enabled(Section::TaskstatsDelay))
        && !diff.derived_rows.is_empty()
    {
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
            // Hierarchical derived rendering — same tree as primary.
            let limited: Vec<&DerivedRow> = if display.section_line_limit > 0 {
                derived_rows
                    .iter()
                    .copied()
                    .take(display.section_line_limit)
                    .collect()
            } else {
                derived_rows
            };
            struct DHier<'a> {
                cgroup: &'a str,
                pcomm: &'a str,
                comm: &'a str,
                row: &'a DerivedRow,
            }
            let mut hier: Vec<DHier<'_>> = limited
                .iter()
                .map(|row| {
                    let mut parts = row.group_key.splitn(3, '\x00');
                    let cg = parts.next().unwrap_or("");
                    let pc = parts.next().unwrap_or("");
                    let cm = parts.next().unwrap_or(pc);
                    DHier {
                        cgroup: cg,
                        pcomm: pc,
                        comm: cm,
                        row,
                    }
                })
                .collect();
            // Use pre-sorted row index as score so hierarchy honors
            // --sort-by. Rows arrive sorted by the chosen metric.
            // Lower index = higher priority (biggest mover first).
            let row_rank: BTreeMap<*const DerivedRow, usize> = hier
                .iter()
                .enumerate()
                .map(|(i, h)| (h.row as *const DerivedRow, i))
                .collect();
            let mut leaf_rank: BTreeMap<(&str, &str), usize> = BTreeMap::new();
            let mut cg_rank: BTreeMap<&str, usize> = BTreeMap::new();
            for h in &hier {
                let rank = row_rank[&(h.row as *const DerivedRow)];
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
                        let ra = row_rank[&(a.row as *const DerivedRow)];
                        let rb = row_rank[&(b.row as *const DerivedRow)];
                        ra.cmp(&rb)
                    })
            });

            writeln!(w)?;
            writeln!(w, "## Derived metrics")?;
            let mut dt = display.new_constrained_table(&global_max_widths);
            dt.set_header(colored_header_with_sort(
                &columns,
                "comm",
                diff.sort_metric_name,
            ));
            let mut last_segs: Vec<&str> = Vec::new();
            let mut last_pc = "";
            let depth_color = |d: usize| -> comfy_table::Color {
                match d {
                    0 => comfy_table::Color::Green,
                    1 => comfy_table::Color::Cyan,
                    _ => comfy_table::Color::DarkGrey,
                }
            };
            for h in &hier {
                let segs: Vec<&str> = h.cgroup.split('/').filter(|s| !s.is_empty()).collect();
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
                        dt.add_row(hcells);
                    }
                    last_segs = segs;
                    last_pc = "";
                }
                if h.pcomm != last_pc {
                    let cg_depth = last_segs.len();
                    let indent = "  ".repeat(cg_depth);
                    let label = format!("{indent}{}", h.pcomm);
                    let hcells: Vec<comfy_table::Cell> = columns
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
                    dt.add_row(hcells);
                    last_pc = h.pcomm;
                }
                let mut cells = render_derived_row_cells(h.row, &columns);
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
        } else {
            writeln!(w)?;
            writeln!(w, "## Derived metrics")?;
            let mut dt = display.new_table();
            dt.set_header(colored_header_with_sort(
                &columns,
                group_header,
                diff.sort_metric_name,
            ));
            let d_limit = if display.section_line_limit > 0 {
                &derived_rows[..derived_rows.len().min(display.section_line_limit)]
            } else {
                &derived_rows[..]
            };
            for row in d_limit {
                let string_cells = render_derived_row_cells(row, &columns);
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
                            color_diff_cell(
                                text,
                                *col,
                                row.delta,
                                row.delta_pct,
                                up,
                                row.sort_by_delta,
                            )
                        } else {
                            color_diff_cell(
                                s,
                                *col,
                                row.delta,
                                row.delta_pct,
                                up,
                                row.sort_by_delta,
                            )
                        }
                    })
                    .collect();
                dt.add_row(cells);
            }
            writeln!(w, "{dt}")?;
        }
    }

    // within the equal cluster.
    if display.is_section_enabled(Section::Smaps)
        && (!diff.smaps_rollup_a.is_empty() || !diff.smaps_rollup_b.is_empty())
    {
        let mut process_keys: std::collections::BTreeSet<&String> =
            diff.smaps_rollup_a.keys().collect();
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
        if display.section_line_limit > 0 {
            sorted_process_keys.truncate(display.section_line_limit);
        }

        // Pre-pass: any (process, key) pair with a non-equal
        // delta? Suppresses the section header when nothing
        // moved, even if both maps are populated.
        let any_delta = sorted_process_keys.iter().any(|pkey| {
            let a = diff.smaps_rollup_a.get(*pkey);
            let b = diff.smaps_rollup_b.get(*pkey);
            let mut keys: std::collections::BTreeSet<&String> =
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
        if any_delta {
            writeln!(w)?;
            writeln!(w, "## smaps_rollup")?;
            let mut st = if global_max_widths.is_empty() {
                display.new_table()
            } else {
                display.new_constrained_table(&global_max_widths)
            };
            st.set_header(colored_header_with_sort(
                &columns,
                "pcomm",
                diff.sort_metric_name,
            ));

            // For All mode, re-sort by cgroup hierarchy (keys are
            // compound cgroup\x00pcomm). Track segments for tree headings.
            let is_compound = group_by == GroupBy::All;
            let mut sorted_keys = sorted_process_keys.clone();
            if is_compound {
                sorted_keys.sort();
            }

            let mut last_segs: Vec<&str> = Vec::new();
            let depth_color = |d: usize| -> comfy_table::Color {
                match d {
                    0 => comfy_table::Color::Green,
                    1 => comfy_table::Color::Cyan,
                    _ => comfy_table::Color::DarkGrey,
                }
            };

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
                let mut keys_union: std::collections::BTreeSet<&String> =
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
                        .map(|(s, col)| {
                            color_diff_cell(s, *col, delta_opt, delta_pct_opt, None, None)
                        })
                        .collect();
                    st.add_row(cells);
                }
            }
            writeln!(w, "{st}")?;
        }
    }

    if group_by == GroupBy::Cgroup
        && (!diff.cgroup_stats_a.is_empty() || !diff.cgroup_stats_b.is_empty())
    {
        // CgroupStats / Limits / MemoryStat / MemoryEvents /
        // Pressure all live behind the GroupBy::Cgroup gate
        // because their data only exists when the diff was
        // computed with cgroup grouping. The `--sections` filter
        // is checked again per sub-table below so a user can
        // request, e.g., `--sections pressure` and get only the
        // PSI rollups even though the cgroup-stats prefix is
        // present in the diff.
        let mut all_keys: std::collections::BTreeSet<&String> =
            diff.cgroup_stats_a.keys().collect();
        all_keys.extend(diff.cgroup_stats_b.keys());

        if display.is_section_enabled(Section::CgroupStats) {
            writeln!(w)?;
            let mut ct = display.new_table();
            ct.set_header(vec![
                "cgroup",
                "cpu_usage_usec",
                "nr_throttled",
                "throttled_usec",
                "memory_current",
            ]);
            for key in &all_keys {
                let a = diff.cgroup_stats_a.get(*key);
                let b = diff.cgroup_stats_b.get(*key);
                ct.add_row(vec![
                    key.to_string(),
                    cgroup_cell(
                        a.map(|s| s.cpu.usage_usec),
                        b.map(|s| s.cpu.usage_usec),
                        ScaleLadder::Us,
                    ),
                    cgroup_cell(
                        a.map(|s| s.cpu.nr_throttled),
                        b.map(|s| s.cpu.nr_throttled),
                        ScaleLadder::Unitless,
                    ),
                    cgroup_cell(
                        a.map(|s| s.cpu.throttled_usec),
                        b.map(|s| s.cpu.throttled_usec),
                        ScaleLadder::Us,
                    ),
                    cgroup_cell(
                        a.map(|s| s.memory.current),
                        b.map(|s| s.memory.current),
                        ScaleLadder::Bytes,
                    ),
                ]);
            }
            writeln!(w, "{ct}")?;
        }

        // Per-cgroup limits / knobs sub-table — operator-set
        // configuration: cpu.max, cpu.weight, memory.max,
        // memory.high, pids.current/max. Cells render as
        // baseline → candidate. `Option<u64>` limits show "max"
        // when None per [`format_optional_limit`]. Suppressed
        // when no cgroup in either snapshot exposes any of these
        // (root cgroup, controllers not enabled, etc.).
        if display.is_section_enabled(Section::Limits) {
            let any_limits = all_keys.iter().any(|key| {
                let has_limits = |s: &CgroupStats| {
                    s.cpu.max_quota_us.is_some()
                        || s.cpu.weight.is_some()
                        || s.memory.max.is_some()
                        || s.memory.high.is_some()
                        || s.pids.current.is_some()
                        || s.pids.max.is_some()
                };
                diff.cgroup_stats_a.get(*key).is_some_and(has_limits)
                    || diff.cgroup_stats_b.get(*key).is_some_and(has_limits)
            });
            if any_limits {
                writeln!(w)?;
                writeln!(w, "## Cgroup limits / knobs")?;
                let mut lt = display.new_table();
                lt.set_header(vec![
                    "cgroup",
                    "cpu.max",
                    "cpu.weight",
                    "memory.max",
                    "memory.high",
                    "pids.current",
                    "pids.max",
                ]);
                for key in &all_keys {
                    let a = diff.cgroup_stats_a.get(*key);
                    let b = diff.cgroup_stats_b.get(*key);
                    // Per-row gate: skip rows where every column is
                    // unset on BOTH sides (the cgroup has no caps,
                    // no weight, no pids accounting on either
                    // baseline or candidate).
                    let row_has_data = |s: &CgroupStats| {
                        s.cpu.max_quota_us.is_some()
                            || s.cpu.weight.is_some()
                            || s.memory.max.is_some()
                            || s.memory.high.is_some()
                            || s.pids.current.is_some()
                            || s.pids.max.is_some()
                    };
                    if !a.is_some_and(row_has_data) && !b.is_some_and(row_has_data) {
                        continue;
                    }
                    lt.add_row(vec![
                        key.to_string(),
                        cgroup_limits_cell(
                            a.map(|s| (s.cpu.max_quota_us, s.cpu.max_period_us)),
                            b.map(|s| (s.cpu.max_quota_us, s.cpu.max_period_us)),
                        ),
                        cgroup_cell(
                            a.and_then(|s| s.cpu.weight),
                            b.and_then(|s| s.cpu.weight),
                            ScaleLadder::Unitless,
                        ),
                        cgroup_optional_limit_cell(
                            a.and_then(|s| s.memory.max),
                            b.and_then(|s| s.memory.max),
                            ScaleLadder::Bytes,
                        ),
                        cgroup_optional_limit_cell(
                            a.and_then(|s| s.memory.high),
                            b.and_then(|s| s.memory.high),
                            ScaleLadder::Bytes,
                        ),
                        cgroup_cell(
                            a.and_then(|s| s.pids.current),
                            b.and_then(|s| s.pids.current),
                            ScaleLadder::Unitless,
                        ),
                        cgroup_optional_limit_cell(
                            a.and_then(|s| s.pids.max),
                            b.and_then(|s| s.pids.max),
                            ScaleLadder::Unitless,
                        ),
                    ]);
                }
                writeln!(w, "{lt}")?;
            }
        }

        // Per-cgroup memory.stat sub-table — kernel-emitted
        // memory counters per cgroup. Up to 71 keys per cgroup.
        // Long-table layout: one row per (cgroup, key) pair
        // with baseline → candidate cells.
        if display.is_section_enabled(Section::MemoryStat)
            && all_keys.iter().any(|key| {
                let has_stat = |s: &CgroupStats| !s.memory.stat.is_empty();
                diff.cgroup_stats_a.get(*key).is_some_and(has_stat)
                    || diff.cgroup_stats_b.get(*key).is_some_and(has_stat)
            })
        {
            writeln!(w)?;
            writeln!(w, "## memory.stat")?;
            let mut mt = display.new_table();
            mt.set_header(vec!["cgroup", "key", "value"]);
            for key in &all_keys {
                let a = diff.cgroup_stats_a.get(*key);
                let b = diff.cgroup_stats_b.get(*key);
                let mut keys_union: std::collections::BTreeSet<&String> = a
                    .map(|s| s.memory.stat.keys().collect())
                    .unwrap_or_default();
                if let Some(s) = b {
                    keys_union.extend(s.memory.stat.keys());
                }
                for stat_key in keys_union {
                    let av = a.and_then(|s| s.memory.stat.get(stat_key).copied());
                    let bv = b.and_then(|s| s.memory.stat.get(stat_key).copied());
                    // Compare-side zero-row suppression: skip
                    // rows where baseline equals candidate. With
                    // 71 keys × N cgroups the table is dominated
                    // by unchanged values; surfacing only the
                    // movers cuts output ~10x for typical runs.
                    // Treats absent and explicit 0 as equal
                    // (both render as "0" / "-").
                    if av == bv {
                        continue;
                    }
                    mt.add_row(vec![
                        key.to_string(),
                        stat_key.clone(),
                        cgroup_cell(av, bv, ScaleLadder::Unitless),
                    ]);
                }
            }
            writeln!(w, "{mt}")?;
        }

        // Per-cgroup memory.events sub-table — pressure-event
        // counters. Same long-table layout as memory.stat with
        // the same baseline-vs-candidate zero-row suppression.
        if display.is_section_enabled(Section::MemoryEvents)
            && all_keys.iter().any(|key| {
                let has_events = |s: &CgroupStats| !s.memory.events.is_empty();
                diff.cgroup_stats_a.get(*key).is_some_and(has_events)
                    || diff.cgroup_stats_b.get(*key).is_some_and(has_events)
            })
        {
            writeln!(w)?;
            writeln!(w, "## memory.events")?;
            let mut et = display.new_table();
            et.set_header(vec!["cgroup", "event", "count"]);
            for key in &all_keys {
                let a = diff.cgroup_stats_a.get(*key);
                let b = diff.cgroup_stats_b.get(*key);
                let mut keys_union: std::collections::BTreeSet<&String> = a
                    .map(|s| s.memory.events.keys().collect())
                    .unwrap_or_default();
                if let Some(s) = b {
                    keys_union.extend(s.memory.events.keys());
                }
                for event_key in keys_union {
                    let av = a.and_then(|s| s.memory.events.get(event_key).copied());
                    let bv = b.and_then(|s| s.memory.events.get(event_key).copied());
                    if av == bv {
                        continue;
                    }
                    et.add_row(vec![
                        key.to_string(),
                        event_key.clone(),
                        cgroup_cell(av, bv, ScaleLadder::Unitless),
                    ]);
                }
            }
            writeln!(w, "{et}")?;
        }

        // Per-cgroup PSI sub-tables — one per resource, each
        // with `some`+`full` rows × `avg10/60/300/total` columns.
        // Mirrors the show-side layout but with
        // baseline→candidate→delta cells rather than single
        // values. Suppressed when every cell on both sides is
        // zero — synthetic fixtures and PSI-disabled hosts both
        // hit this case and there's nothing useful to render.
        if display.is_section_enabled(Section::Pressure) {
            for (resource_name, accessor) in psi_resource_accessors() {
                let any_data = all_keys.iter().any(|key| {
                    let a = diff.cgroup_stats_a.get(*key).map(|s| accessor(&s.psi));
                    let b = diff.cgroup_stats_b.get(*key).map(|s| accessor(&s.psi));
                    a.as_ref().is_some_and(psi_resource_has_data)
                        || b.as_ref().is_some_and(psi_resource_has_data)
                });
                if !any_data {
                    continue;
                }
                writeln!(w)?;
                writeln!(w, "## Pressure / {resource_name}")?;
                let mut pt = display.new_table();
                pt.set_header(vec!["cgroup", "row", "avg10", "avg60", "avg300", "total"]);
                for key in &all_keys {
                    let a = diff.cgroup_stats_a.get(*key).map(|s| accessor(&s.psi));
                    let b = diff.cgroup_stats_b.get(*key).map(|s| accessor(&s.psi));
                    pt.add_row(vec![
                        key.to_string(),
                        "some".into(),
                        format_psi_avg_cell(a.map(|r| r.some.avg10), b.map(|r| r.some.avg10)),
                        format_psi_avg_cell(a.map(|r| r.some.avg60), b.map(|r| r.some.avg60)),
                        format_psi_avg_cell(a.map(|r| r.some.avg300), b.map(|r| r.some.avg300)),
                        cgroup_cell(
                            a.map(|r| r.some.total_usec),
                            b.map(|r| r.some.total_usec),
                            ScaleLadder::Us,
                        ),
                    ]);
                    pt.add_row(vec![
                        key.to_string(),
                        "full".into(),
                        format_psi_avg_cell(a.map(|r| r.full.avg10), b.map(|r| r.full.avg10)),
                        format_psi_avg_cell(a.map(|r| r.full.avg60), b.map(|r| r.full.avg60)),
                        format_psi_avg_cell(a.map(|r| r.full.avg300), b.map(|r| r.full.avg300)),
                        cgroup_cell(
                            a.map(|r| r.full.total_usec),
                            b.map(|r| r.full.total_usec),
                            ScaleLadder::Us,
                        ),
                    ]);
                }
                writeln!(w, "{pt}")?;
            }
        }
    }

    // Host-level PSI compare — one sub-table per resource. Runs
    // independent of `GroupBy` because host pressure is the
    // primary scheduler-health signal regardless of which axis
    // the user grouped per-thread metrics by. Suppressed when
    // both snapshots' host PSI is all-zero.
    if display.is_section_enabled(Section::HostPressure)
        && psi_pair_has_data(&diff.host_psi_a, &diff.host_psi_b)
    {
        for (resource_name, accessor) in psi_resource_accessors() {
            let a = accessor(&diff.host_psi_a);
            let b = accessor(&diff.host_psi_b);
            if !psi_resource_has_data(&a) && !psi_resource_has_data(&b) {
                continue;
            }
            writeln!(w)?;
            writeln!(w, "## Host pressure / {resource_name}")?;
            let mut pt = display.new_table();
            pt.set_header(vec!["row", "avg10", "avg60", "avg300", "total"]);
            pt.add_row(vec![
                "some".into(),
                format_psi_avg_cell(Some(a.some.avg10), Some(b.some.avg10)),
                format_psi_avg_cell(Some(a.some.avg60), Some(b.some.avg60)),
                format_psi_avg_cell(Some(a.some.avg300), Some(b.some.avg300)),
                cgroup_cell(
                    Some(a.some.total_usec),
                    Some(b.some.total_usec),
                    ScaleLadder::Us,
                ),
            ]);
            pt.add_row(vec![
                "full".into(),
                format_psi_avg_cell(Some(a.full.avg10), Some(b.full.avg10)),
                format_psi_avg_cell(Some(a.full.avg60), Some(b.full.avg60)),
                format_psi_avg_cell(Some(a.full.avg300), Some(b.full.avg300)),
                cgroup_cell(
                    Some(a.full.total_usec),
                    Some(b.full.total_usec),
                    ScaleLadder::Us,
                ),
            ]);
            writeln!(w, "{pt}")?;
        }
    }

    // Per-process smaps_rollup compare. Iterates the union of
    // process keys across both snapshots; one row per
    // (process, key) pair carrying baseline → candidate kB
    // values rendered through the existing "B" auto-scale
    // ladder after kB → bytes conversion. Suppressed when
    // neither side has any smaps_rollup data; per-row gate
    // skips rows where baseline equals candidate (treats
    // absent and 0 as equal). Mirrors the memory.stat compare
    // layout.
    //
    // Process iteration order: descending by max(Rss baseline,
    // Rss candidate), tiebreak by descending max-Pss, final
    // tiebreak alphabetical. Memory-heavy processes that moved
    // between snapshots surface first; the renderer then walks
    // each process's per-field smaps keys in BTreeSet
    // (alphabetical) order so within-process row ordering stays
    // deterministic. Rss is the primary "how much memory does
    // this process hold" signal; Pss carries proportional set
    // size (per `fs/proc/task_mmu.c::smap_account`) and breaks
    // ties when two processes report equal Rss but differ in
    // shared-page accounting. Processes missing both keys sort
    // last under `unwrap_or(0)` and preserve alphabetical order
    // Global sched_ext sysfs compare. Suppressed when both
    // sides are None (CONFIG_SCHED_CLASS_EXT=n on both kernels)
    // OR when both sides are Some and every field is identical
    // across baseline and candidate (no signal to surface). When
    // exactly one side is Some, surface the configuration delta
    // — that's a load-bearing signal that the host kernel
    // changed between snapshots.
    let scx_emit = match (&diff.sched_ext_a, &diff.sched_ext_b) {
        (None, None) => false,
        (Some(_), None) | (None, Some(_)) => true,
        (Some(a), Some(b)) => {
            a.state != b.state
                || a.switch_all != b.switch_all
                || a.nr_rejected != b.nr_rejected
                || a.hotplug_seq != b.hotplug_seq
                || a.enable_seq != b.enable_seq
        }
    };
    if display.is_section_enabled(Section::SchedExt) && scx_emit {
        writeln!(w)?;
        writeln!(w, "## sched_ext")?;
        let mut at = display.new_table();
        at.set_header(vec!["attr", "value"]);
        // state cell: render "-" for both absent (Option=None)
        // AND for the empty-string-but-Some case (file
        // unreadable but directory present). The "-" placeholder
        // makes "no observation" visually distinct from a real
        // sched_ext_state_str[] value.
        fn state_cell_for(s: Option<&crate::ctprof::SchedExtSysfs>) -> String {
            match s {
                None => "-".to_string(),
                Some(scx) if scx.state.is_empty() => "-".to_string(),
                Some(scx) => scx.state.clone(),
            }
        }
        let state_a = state_cell_for(diff.sched_ext_a.as_ref());
        let state_b = state_cell_for(diff.sched_ext_b.as_ref());
        let state_cell = if state_a == state_b {
            state_a
        } else {
            format!("{state_a} → {state_b}")
        };
        at.add_row(vec!["state".into(), state_cell]);
        at.add_row(vec![
            "switch_all".into(),
            cgroup_cell(
                diff.sched_ext_a.as_ref().map(|s| s.switch_all),
                diff.sched_ext_b.as_ref().map(|s| s.switch_all),
                ScaleLadder::Unitless,
            ),
        ]);
        at.add_row(vec![
            "nr_rejected".into(),
            cgroup_cell(
                diff.sched_ext_a.as_ref().map(|s| s.nr_rejected),
                diff.sched_ext_b.as_ref().map(|s| s.nr_rejected),
                ScaleLadder::Unitless,
            ),
        ]);
        at.add_row(vec![
            "hotplug_seq".into(),
            cgroup_cell(
                diff.sched_ext_a.as_ref().map(|s| s.hotplug_seq),
                diff.sched_ext_b.as_ref().map(|s| s.hotplug_seq),
                ScaleLadder::Unitless,
            ),
        ]);
        at.add_row(vec![
            "enable_seq".into(),
            cgroup_cell(
                diff.sched_ext_a.as_ref().map(|s| s.enable_seq),
                diff.sched_ext_b.as_ref().map(|s| s.enable_seq),
                ScaleLadder::Unitless,
            ),
        ]);
        writeln!(w, "{at}")?;
    }

    let write_only_list = |w: &mut W, label: &str, path: &Path, keys: &[String]| -> fmt::Result {
        if keys.is_empty() {
            return Ok(());
        }
        writeln!(
            w,
            "\n{} group(s) only in {label} ({}):",
            keys.len(),
            path.display()
        )?;
        if group_by == GroupBy::All {
            let mut sorted: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
            sorted.sort();
            let mut last_segs: Vec<&str> = Vec::new();
            for k in &sorted {
                let (cg, pc) = k.split_once('\x00').unwrap_or(("", k));
                let segs: Vec<&str> = cg.split('/').filter(|s| !s.is_empty()).collect();
                let common = segs
                    .iter()
                    .zip(last_segs.iter())
                    .take_while(|(a, b)| a == b)
                    .count();
                if common < last_segs.len() || segs.len() > last_segs.len() {
                    for (depth, seg) in segs.iter().enumerate().skip(common) {
                        let indent = "  ".repeat(depth + 1);
                        writeln!(w, "{indent}{seg}")?;
                    }
                    last_segs = segs;
                }
                let indent = "  ".repeat(last_segs.len() + 1);
                writeln!(w, "{indent}{pc}")?;
            }
        } else {
            for k in keys {
                writeln!(w, "  {k}")?;
            }
        }
        Ok(())
    };
    // Render the Fudged-cgroup-matches section BEFORE the
    // only-baseline / only-candidate lists. Fudge pairs are
    // matched cgroups that the operator should see together;
    // putting them after the orphan lists buries the most
    // informative section under noise.
    if !diff.fudged_pairs.is_empty() {
        writeln!(
            w,
            "\n\x1b[1;33m## Fudged cgroup matches ({} pair(s))\x1b[0m",
            diff.fudged_pairs.len()
        )?;
        for fp in &diff.fudged_pairs {
            writeln!(w, "\n  \x1b[36mbaseline:\x1b[0m {}", fp.baseline_cgroup)?;
            writeln!(w, "  \x1b[36mcandidate:\x1b[0m {}", fp.candidate_cgroup)?;
            // Surface cascade roots when they differ from the
            // matched baseline / candidate paths — operators
            // need to see the longest-common-suffix root that
            // governs how cascaded children get joined.
            if fp.baseline_root != fp.baseline_cgroup || fp.candidate_root != fp.candidate_cgroup {
                writeln!(
                    w,
                    "  cascade roots: baseline={} candidate={}",
                    fp.baseline_root, fp.candidate_root,
                )?;
            }
            writeln!(
                w,
                "  overlap: {} thread types, Jaccard: {:.1}%, cascaded children: {}",
                fp.overlap,
                fp.jaccard * 100.0,
                fp.cascaded_children
            )?;
            if !fp.baseline_residual.is_empty() {
                writeln!(
                    w,
                    "  residual (baseline only): {}",
                    fp.baseline_residual.join(", ")
                )?;
            }
            if !fp.candidate_residual.is_empty() {
                writeln!(
                    w,
                    "  residual (candidate only): {}",
                    fp.candidate_residual.join(", ")
                )?;
            }
        }
    }

    write_only_list(w, "baseline", baseline_path, &diff.only_baseline)?;
    write_only_list(w, "candidate", candidate_path, &diff.only_candidate)?;

    Ok(())
}
