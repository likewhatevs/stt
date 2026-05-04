//! Per-cgroup secondary tables for [`super::write_diff`].
//!
//! Five sub-tables share this module because their data only
//! exists when the diff was computed with [`GroupBy::Cgroup`]
//! grouping:
//!
//! - **CgroupStats** — the four headline counters
//!   (cpu_usage_usec, nr_throttled, throttled_usec,
//!   memory_current).
//! - **Limits** — operator-set configuration: cpu.max,
//!   cpu.weight, memory.max, memory.high, pids.current/max.
//! - **MemoryStat** — kernel-emitted memory.stat counters
//!   (~71 keys per cgroup).
//! - **MemoryEvents** — pressure-event counters from
//!   memory.events.
//! - **Pressure** — per-cgroup PSI (`some` + `full` rows ×
//!   `avg10/60/300/total` columns).
//!
//! The `--sections` filter is checked again per sub-table so a
//! user can request, e.g., `--sections pressure` and get only
//! the PSI rollups even though the cgroup-stats prefix is
//! present in the diff.

use std::collections::BTreeSet;
use std::fmt;

use super::super::columns::Section;
use super::super::diff_types::CtprofDiff;
use super::super::options::GroupBy;
use super::super::render::{
    cgroup_cell, format_psi_avg_cell, psi_resource_accessors, psi_resource_has_data,
};
use super::super::runner::DisplayOptions;
use super::super::scale::{cgroup_limits_cell, cgroup_optional_limit_cell, ScaleLadder};
use crate::ctprof::CgroupStats;

pub(super) fn write_cgroup_sections<W: fmt::Write>(
    w: &mut W,
    diff: &CtprofDiff,
    group_by: GroupBy,
    display: &DisplayOptions,
) -> fmt::Result {
    if group_by != GroupBy::Cgroup {
        return Ok(());
    }
    if diff.cgroup_stats_a.is_empty() && diff.cgroup_stats_b.is_empty() {
        return Ok(());
    }

    let mut all_keys: BTreeSet<&String> = diff.cgroup_stats_a.keys().collect();
    all_keys.extend(diff.cgroup_stats_b.keys());

    write_cgroup_stats_table(w, diff, &all_keys, display)?;
    write_limits_table(w, diff, &all_keys, display)?;
    write_memory_stat_table(w, diff, &all_keys, display)?;
    write_memory_events_table(w, diff, &all_keys, display)?;
    write_pressure_tables(w, diff, &all_keys, display)?;
    Ok(())
}

fn write_cgroup_stats_table<W: fmt::Write>(
    w: &mut W,
    diff: &CtprofDiff,
    all_keys: &BTreeSet<&String>,
    display: &DisplayOptions,
) -> fmt::Result {
    if !display.is_section_enabled(Section::CgroupStats) {
        return Ok(());
    }
    writeln!(w)?;
    writeln!(w, "## CgroupStats")?;
    let mut ct = display.new_table();
    ct.set_header(vec![
        "cgroup",
        "cpu_usage_usec",
        "nr_throttled",
        "throttled_usec",
        "memory_current",
    ]);
    for key in all_keys {
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
    Ok(())
}

/// Per-cgroup limits / knobs sub-table — operator-set
/// configuration: cpu.max, cpu.weight, memory.max,
/// memory.high, pids.current/max. Cells render as
/// baseline → candidate. `Option<u64>` limits show "max"
/// when None per [`super::super::scale::format_optional_limit`].
/// Suppressed when no cgroup in either snapshot exposes any of
/// these (root cgroup, controllers not enabled, etc.).
fn write_limits_table<W: fmt::Write>(
    w: &mut W,
    diff: &CtprofDiff,
    all_keys: &BTreeSet<&String>,
    display: &DisplayOptions,
) -> fmt::Result {
    if !display.is_section_enabled(Section::Limits) {
        return Ok(());
    }
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
    if !any_limits {
        return Ok(());
    }
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
    for key in all_keys {
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
    Ok(())
}

/// Per-cgroup memory.stat sub-table — kernel-emitted memory
/// counters per cgroup. Up to 71 keys per cgroup. Long-table
/// layout: one row per `(cgroup, key)` pair with baseline →
/// candidate cells.
fn write_memory_stat_table<W: fmt::Write>(
    w: &mut W,
    diff: &CtprofDiff,
    all_keys: &BTreeSet<&String>,
    display: &DisplayOptions,
) -> fmt::Result {
    if !display.is_section_enabled(Section::MemoryStat) {
        return Ok(());
    }
    let any_stats = all_keys.iter().any(|key| {
        let has_stat = |s: &CgroupStats| !s.memory.stat.is_empty();
        diff.cgroup_stats_a.get(*key).is_some_and(has_stat)
            || diff.cgroup_stats_b.get(*key).is_some_and(has_stat)
    });
    if !any_stats {
        return Ok(());
    }
    writeln!(w)?;
    writeln!(w, "## memory.stat")?;
    let mut mt = display.new_table();
    mt.set_header(vec!["cgroup", "key", "value"]);
    for key in all_keys {
        let a = diff.cgroup_stats_a.get(*key);
        let b = diff.cgroup_stats_b.get(*key);
        let mut keys_union: BTreeSet<&String> = a
            .map(|s| s.memory.stat.keys().collect())
            .unwrap_or_default();
        if let Some(s) = b {
            keys_union.extend(s.memory.stat.keys());
        }
        for stat_key in keys_union {
            let av = a.and_then(|s| s.memory.stat.get(stat_key).copied());
            let bv = b.and_then(|s| s.memory.stat.get(stat_key).copied());
            // Compare-side zero-row suppression: skip rows
            // where baseline equals candidate. With 71 keys ×
            // N cgroups the table is dominated by unchanged
            // values; surfacing only the movers cuts output
            // ~10x for typical runs. Treats absent and
            // explicit 0 as equal (both render as "0" / "-").
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
    Ok(())
}

/// Per-cgroup memory.events sub-table — pressure-event
/// counters. Same long-table layout as memory.stat with the
/// same baseline-vs-candidate zero-row suppression.
fn write_memory_events_table<W: fmt::Write>(
    w: &mut W,
    diff: &CtprofDiff,
    all_keys: &BTreeSet<&String>,
    display: &DisplayOptions,
) -> fmt::Result {
    if !display.is_section_enabled(Section::MemoryEvents) {
        return Ok(());
    }
    let any_events = all_keys.iter().any(|key| {
        let has_events = |s: &CgroupStats| !s.memory.events.is_empty();
        diff.cgroup_stats_a.get(*key).is_some_and(has_events)
            || diff.cgroup_stats_b.get(*key).is_some_and(has_events)
    });
    if !any_events {
        return Ok(());
    }
    writeln!(w)?;
    writeln!(w, "## memory.events")?;
    let mut et = display.new_table();
    et.set_header(vec!["cgroup", "event", "count"]);
    for key in all_keys {
        let a = diff.cgroup_stats_a.get(*key);
        let b = diff.cgroup_stats_b.get(*key);
        let mut keys_union: BTreeSet<&String> = a
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
    Ok(())
}

/// Per-cgroup PSI sub-tables — one per resource, each with
/// `some`+`full` rows × `avg10/60/300/total` columns. Mirrors
/// the show-side layout but with baseline→candidate→delta
/// cells rather than single values. Suppressed when every
/// cell on both sides is zero — synthetic fixtures and
/// PSI-disabled hosts both hit this case and there's nothing
/// useful to render.
fn write_pressure_tables<W: fmt::Write>(
    w: &mut W,
    diff: &CtprofDiff,
    all_keys: &BTreeSet<&String>,
    display: &DisplayOptions,
) -> fmt::Result {
    if !display.is_section_enabled(Section::Pressure) {
        return Ok(());
    }
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
        for key in all_keys {
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
    Ok(())
}
