//! Host-level PSI compare emitter for [`super::write_diff`].
//!
//! One sub-table per resource (cpu / memory / io / irq).
//! Runs independent of [`GroupBy`] because host pressure is
//! the primary scheduler-health signal regardless of which
//! axis the user grouped per-thread metrics by. Suppressed
//! when both snapshots' host PSI is all-zero.

use std::fmt;

use super::super::columns::Section;
use super::super::diff_types::CtprofDiff;
use super::super::render::{
    cgroup_cell, format_psi_avg_cell, psi_pair_has_data, psi_resource_accessors,
    psi_resource_has_data,
};
use super::super::runner::DisplayOptions;
use super::super::scale::ScaleLadder;

pub(super) fn write_host_psi_section<W: fmt::Write>(
    w: &mut W,
    diff: &CtprofDiff,
    display: &DisplayOptions,
) -> fmt::Result {
    if !display.is_section_enabled(Section::HostPressure) {
        return Ok(());
    }
    if !psi_pair_has_data(&diff.host_psi_a, &diff.host_psi_b) {
        return Ok(());
    }
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
    Ok(())
}
