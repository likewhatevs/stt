//! Global sched_ext sysfs compare for [`super::write_diff`].
//!
//! Suppressed when both sides are None
//! (CONFIG_SCHED_CLASS_EXT=n on both kernels) OR when both
//! sides are Some and every field is identical across baseline
//! and candidate (no signal to surface). When exactly one side
//! is Some, surface the configuration delta — that's a
//! load-bearing signal that the host kernel changed between
//! snapshots.

use std::fmt;

use super::super::columns::Section;
use super::super::diff_types::CtprofDiff;
use super::super::render::cgroup_cell;
use super::super::runner::DisplayOptions;
use super::super::scale::ScaleLadder;

pub(super) fn write_sched_ext_section<W: fmt::Write>(
    w: &mut W,
    diff: &CtprofDiff,
    display: &DisplayOptions,
) -> fmt::Result {
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
    if !display.is_section_enabled(Section::SchedExt) || !scx_emit {
        return Ok(());
    }

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
    Ok(())
}
