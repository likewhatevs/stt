//! `write_diff` — primary diff renderer for [`CtprofDiff`].
//!
//! Multi-section emitter for the `ctprof compare` output.
//! Each section emits through a dedicated submodule so the
//! orchestrator below stays a slim sequencer:
//!
//! - [`primary`] — `## Primary metrics` table (52 non-taskstats
//!   rows + 34 taskstats rows). Three layouts based on
//!   [`GroupBy`]: hierarchical cgroup + pcomm + comm tree
//!   under [`GroupBy::All`], one-table-per-parent under
//!   [`GroupBy::Cgroup`], flat under [`GroupBy::Pcomm`] /
//!   [`GroupBy::Comm`] / [`GroupBy::CommExact`].
//! - [`derived`] — `## Derived metrics` table; reuses the
//!   primary tree builder for [`GroupBy::All`] so primary and
//!   derived rows share heading order.
//! - [`smaps`] — `## smaps_rollup` per-process memory
//!   compare; suppressed when neither side carries data.
//! - [`cgroup`] — five sub-tables that exist only under
//!   [`GroupBy::Cgroup`]: cgroup-stats, limits/knobs,
//!   memory.stat, memory.events, per-cgroup PSI.
//! - [`host_psi`] — `## Host pressure / <resource>` tables;
//!   independent of group_by because host pressure is the
//!   primary scheduler-health signal.
//! - [`sched_ext`] — `## sched_ext` global sysfs compare.
//! - [`orphans`] — fudged-cgroup pairs (rendered first so
//!   they don't get buried), only-baseline list,
//!   only-candidate list.
//!
//! Each emitter is gated on
//! [`DisplayOptions::is_section_enabled`] so `--sections`
//! always wins over the data-availability suppression
//! heuristic, and the column widths used by [`primary`],
//! [`derived`], and [`smaps`] are measured ONCE across all
//! data rows (primary + derived) so every table in every
//! section shares column widths under [`GroupBy::All`].
//!
//! Cell rendering is delegated to [`super::render`]; metric
//! lookup to [`super::metrics`]; column resolution to
//! [`super::columns`]; the table builder lives on
//! [`super::DisplayOptions`].

mod cgroup;
mod derived;
mod host_psi;
mod orphans;
mod primary;
mod sched_ext;
mod smaps;

use std::fmt;
use std::path::Path;

use super::columns::Column;
use super::diff_types::CtprofDiff;
use super::options::GroupBy;
use super::render::{colored_header_with_sort, render_derived_row_cells, render_diff_row_cells};
use super::runner::DisplayOptions;

/// Render [`CtprofDiff`] into `w`. The formatter layer lives
/// here so tests can inspect exactly what `print_diff` would
/// emit without shelling through stdout capture. Write errors
/// propagate as [`std::fmt::Error`] — callers that write into an
/// infallible sink (`String`) can unwrap or ignore.
///
/// `display` controls per-row column layout, terminal-width
/// wrapping, and per-section filtering: see [`DisplayFormat`] /
/// [`Column`] / [`Section`] / [`DisplayOptions`] for the
/// resolution rules. Each sub-emitter is gated on
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
        compute_global_max_widths(diff, &columns, display)
    } else {
        Vec::new()
    };

    primary::write_primary_section(
        w,
        diff,
        group_by,
        group_header,
        &columns,
        display,
        &global_max_widths,
    )?;
    derived::write_derived_section(
        w,
        diff,
        group_by,
        group_header,
        &columns,
        display,
        &global_max_widths,
    )?;
    smaps::write_smaps_section(w, diff, group_by, &columns, display, &global_max_widths)?;
    cgroup::write_cgroup_sections(w, diff, group_by, display)?;
    host_psi::write_host_psi_section(w, diff, display)?;
    sched_ext::write_sched_ext_section(w, diff, display)?;
    orphans::write_orphans_section(w, diff, baseline_path, candidate_path, group_by)?;

    Ok(())
}

/// Two-pass column-width measurement: build a throwaway table,
/// add every primary + derived data row through the same cell
/// path the real renderers use, and read back per-column max
/// widths. Heading rows are constrained to these widths in the
/// real tables so they can't inflate columns.
fn compute_global_max_widths(
    diff: &CtprofDiff,
    columns: &[Column],
    display: &DisplayOptions,
) -> Vec<u16> {
    let mut measure = display.new_table();
    measure.set_header(colored_header_with_sort(
        columns,
        "comm",
        diff.sort_metric_name,
    ));
    for row in &diff.rows {
        let mut cells = render_diff_row_cells(row, columns);
        if let Some(pos) = columns.iter().position(|c| *c == Column::Group) {
            let comm = row.group_key.splitn(3, '\x00').nth(2).unwrap_or("");
            cells[pos] = comm.to_string();
        }
        measure.add_row(cells);
    }
    for row in &diff.derived_rows {
        let mut cells = render_derived_row_cells(row, columns);
        if let Some(pos) = columns.iter().position(|c| *c == Column::Group) {
            let comm = row.group_key.splitn(3, '\x00').nth(2).unwrap_or("");
            cells[pos] = comm.to_string();
        }
        measure.add_row(cells);
    }
    measure.column_max_content_widths()
}
