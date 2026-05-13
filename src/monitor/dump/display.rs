//! Human-readable [`std::fmt::Display`] impls for the failure-dump
//! types.
//!
//! All `Display` impls for [`super::FailureDumpReport`],
//! [`super::DualFailureDumpReport`], [`super::FailureDumpReportAny`],
//! [`super::FailureDumpMap`], [`super::FailureDumpEntry`],
//! [`super::FailureDumpPercpuEntry`], and
//! [`super::FailureDumpPercpuHashEntry`] live here so the type
//! definitions in [`super`] stay focused on the data shape and the
//! formatting concerns are isolated in one place.
//!
//! JSON remains the programmatic form via `serde_json`; these impls
//! are the default presentation used in test-failure output.

use super::super::btf_render::{
    RenderedMember, RenderedValue, is_inline_scalar, is_zero, write_value_at_depth,
};
use super::{
    DegradedFailureDumpReport, DualFailureDumpReport, EventCounterSample, FailureDumpEntry,
    FailureDumpMap, FailureDumpPercpuEntry, FailureDumpPercpuHashEntry, FailureDumpReport,
    FailureDumpReportAny, render_sparkline_i64,
};

/// Minimum entry count for [`FailureDumpMap`] table rendering. Two
/// entries are still meaningful as a table (column headers vs two
/// per-entry blocks is denser); below that the table header
/// overhead exceeds the savings.
const TABLE_MIN_ENTRIES: usize = 2;

/// Try to render a [`FailureDumpMap`]'s `entries` as a homogeneous
/// table. Returns `Ok(true)` when the table was rendered (caller
/// must skip per-entry rendering); `Ok(false)` when the entries
/// don't qualify (caller falls through to per-entry rendering).
///
/// Table eligibility (every condition required):
///   - at least [`TABLE_MIN_ENTRIES`] entries,
///   - every entry has BOTH `key.is_some()` AND `value.is_some()`,
///   - no entry has a `payload` (typed sdt_alloc payloads need
///     block rendering below the entry; the table format can't
///     represent them),
///   - every key is a `RenderedValue::Struct` with the same
///     `type_name` and same member names AND every member is an
///     inline scalar (matches [`is_inline_scalar`]'s definition),
///   - every value is a `RenderedValue::Struct` with the same
///     `type_name` and same member names AND every member is an
///     inline scalar.
///
/// Output shape, with `|` separating key columns from value columns
/// and numeric columns right-aligned:
/// ```text
///   cgrp_id  llc_id | llcx
///         1       5 | 17592186046336
///        61       3 | 17592186047616
/// ```
fn try_write_entry_table(
    f: &mut std::fmt::Formatter<'_>,
    entries: &[FailureDumpEntry],
) -> Result<bool, std::fmt::Error> {
    if entries.len() < TABLE_MIN_ENTRIES {
        return Ok(false);
    }
    // Reject if any entry has a payload — the typed payload renders
    // in a block below the entry; the table can't carry it without
    // breaking the row layout.
    if entries.iter().any(|e| e.payload.is_some()) {
        return Ok(false);
    }
    // Collect every entry's key + value as Struct references. Any
    // missing render (None key or None value) bails immediately —
    // a hex-only entry can't be a table cell.
    let pairs: Option<Vec<(&Vec<RenderedMember>, &Vec<RenderedMember>)>> = entries
        .iter()
        .map(|e| match (&e.key, &e.value) {
            (
                Some(RenderedValue::Struct { members: k, .. }),
                Some(RenderedValue::Struct { members: v, .. }),
            ) => Some((k, v)),
            _ => None,
        })
        .collect();
    let Some(pairs) = pairs else {
        return Ok(false);
    };
    if pairs.is_empty() {
        return Ok(false);
    }

    // Type names + member names must match across every entry. The
    // first entry sets the template; subsequent entries must agree.
    let (first_key_name, first_value_name) = match (&entries[0].key, &entries[0].value) {
        (
            Some(RenderedValue::Struct { type_name: kn, .. }),
            Some(RenderedValue::Struct { type_name: vn, .. }),
        ) => (kn.clone(), vn.clone()),
        _ => return Ok(false),
    };
    for e in &entries[1..] {
        match (&e.key, &e.value) {
            (
                Some(RenderedValue::Struct { type_name: kn, .. }),
                Some(RenderedValue::Struct { type_name: vn, .. }),
            ) => {
                if *kn != first_key_name || *vn != first_value_name {
                    return Ok(false);
                }
            }
            _ => return Ok(false),
        }
    }

    let (k0, v0) = pairs[0];
    let key_names: Vec<&str> = k0.iter().map(|m| m.name.as_str()).collect();
    let value_names: Vec<&str> = v0.iter().map(|m| m.name.as_str()).collect();

    // Member names + counts must match across every entry. A
    // mismatch means the structs aren't actually homogeneous (the
    // BTF rendered different fields per entry — possible if the
    // entry value type id changed mid-iteration, or a torn read
    // produced a Truncated partial inside the Struct).
    for (k, v) in &pairs {
        if k.len() != k0.len() || v.len() != v0.len() {
            return Ok(false);
        }
        for (a, b) in k.iter().zip(k0.iter()) {
            if a.name != b.name {
                return Ok(false);
            }
        }
        for (a, b) in v.iter().zip(v0.iter()) {
            if a.name != b.name {
                return Ok(false);
            }
        }
    }

    // Every member in every entry's key + value must be an inline
    // scalar. A composite member (Struct, Array, CpuList, etc.)
    // breaks the single-line-per-row contract.
    for (k, v) in &pairs {
        if !k.iter().all(|m| is_inline_scalar(&m.value)) {
            return Ok(false);
        }
        if !v.iter().all(|m| is_inline_scalar(&m.value)) {
            return Ok(false);
        }
    }

    // Pre-render every cell so column widths can be measured. A
    // numeric cell uses the same Display impl that produces
    // "<value>" (e.g. "1024") so widths reflect the rendered
    // form, not the raw integer.
    let key_rows: Vec<Vec<String>> = pairs
        .iter()
        .map(|(k, _)| k.iter().map(|m| format!("{}", m.value)).collect())
        .collect();
    let value_rows: Vec<Vec<String>> = pairs
        .iter()
        .map(|(_, v)| v.iter().map(|m| format!("{}", m.value)).collect())
        .collect();

    // Per-column width: max of header (member name) and any cell
    // in the column. Header is the member name; cells come from
    // the pre-rendered row vectors.
    let key_widths: Vec<usize> = (0..key_names.len())
        .map(|c| {
            let cell_max = key_rows.iter().map(|r| r[c].len()).max().unwrap_or(0);
            key_names[c].len().max(cell_max)
        })
        .collect();
    let value_widths: Vec<usize> = (0..value_names.len())
        .map(|c| {
            let cell_max = value_rows.iter().map(|r| r[c].len()).max().unwrap_or(0);
            value_names[c].len().max(cell_max)
        })
        .collect();

    // Header row: key names | value names.
    f.write_str("\n  ")?;
    for (i, name) in key_names.iter().enumerate() {
        if i > 0 {
            f.write_str("  ")?;
        }
        write!(f, "{:>width$}", name, width = key_widths[i])?;
    }
    f.write_str(" | ")?;
    for (i, name) in value_names.iter().enumerate() {
        if i > 0 {
            f.write_str("  ")?;
        }
        write!(f, "{:>width$}", name, width = value_widths[i])?;
    }

    // Data rows. Right-align every cell to the column width — the
    // values are scalar Display output (integers, hex pointers,
    // booleans), and right-alignment makes a vertical tens/hundreds
    // alignment immediately readable.
    for (key_row, value_row) in key_rows.iter().zip(value_rows.iter()) {
        f.write_str("\n  ")?;
        for (i, cell) in key_row.iter().enumerate() {
            if i > 0 {
                f.write_str("  ")?;
            }
            write!(f, "{:>width$}", cell, width = key_widths[i])?;
        }
        f.write_str(" | ")?;
        for (i, cell) in value_row.iter().enumerate() {
            if i > 0 {
                f.write_str("  ")?;
            }
            write!(f, "{:>width$}", cell, width = value_widths[i])?;
        }
    }

    Ok(true)
}

impl std::fmt::Display for DualFailureDumpReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Summary header: a one-line at-a-glance description so an
        // operator scanning logs sees the shape (early present /
        // absent, late map + vcpu_regs counts, plus the trigger
        // metric and threshold when early fired) before paging
        // through the full body.
        let n_maps = self.late.maps.len();
        let m_vcpu_regs = self.late.vcpu_regs.len();
        if self.early.is_some() {
            // Both jiffies fields zero means the early-snapshot
            // bookkeeping never recorded a trigger metric (e.g. the
            // snapshot was attached without the runnable_at scan
            // populating the threshold). Render a distinct line so
            // operators don't read "max_age=0j, threshold=0j" as a
            // legitimate sub-tick trigger.
            if self.early_max_age_jiffies == 0 && self.early_threshold_jiffies == 0 {
                write!(
                    f,
                    "DualFailureDumpReport: early=present (jiffies not captured), \
                     late=({n_maps} maps, {m_vcpu_regs} vcpu_regs)\n\n",
                )?;
            } else {
                write!(
                    f,
                    "DualFailureDumpReport: early=present (max_age={}j, threshold={}j), \
                     late=({n_maps} maps, {m_vcpu_regs} vcpu_regs)\n\n",
                    self.early_max_age_jiffies, self.early_threshold_jiffies,
                )?;
            }
        } else if let Some(reason) = self.early_skipped_reason.as_deref() {
            // Structured reason populated by the freeze coordinator
            // (one of: scan prerequisites unavailable, max_age never
            // crossed threshold, scx_tick stall — see
            // `DualFailureDumpReport::early_skipped_reason` doc for
            // the full set). Surface it directly so the operator
            // does not have to re-run with RUST_LOG=ktstr=debug to
            // recover the cause.
            write!(
                f,
                "DualFailureDumpReport: early=absent ({reason}), \
                 late=({n_maps} maps, {m_vcpu_regs} vcpu_regs)\n\n",
            )?;
        } else {
            // Legacy generic message. Reached only on dumps written
            // before the freeze coordinator started populating
            // `early_skipped_reason` (no field on the JSON, deserialised
            // as None). Keep the RUST_LOG hint so old dumps remain
            // actionable; new dumps take the structured branch above.
            write!(
                f,
                "DualFailureDumpReport: early=absent, late=({n_maps} maps, \
                 {m_vcpu_regs} vcpu_regs)\n\n",
            )?;
        }
        match &self.early {
            Some(early) => {
                f.write_str("early snapshot (sched_ext watchdog half-way):\n")?;
                std::fmt::Display::fmt(early, f)?;
                f.write_str("\n\nlate snapshot (error-exit):\n")?;
                std::fmt::Display::fmt(&self.late, f)
            }
            None => {
                if let Some(reason) = self.early_skipped_reason.as_deref() {
                    writeln!(
                        f,
                        "late snapshot (error-exit; early snapshot absent: \
                         {reason}):",
                    )?;
                } else {
                    f.write_str(
                        "late snapshot (error-exit; early snapshot absent \
                         (stall fired before half-way threshold, or runnable_at \
                         scan setup failed) — re-run with RUST_LOG=ktstr=debug \
                         for scan resolution diagnostics):\n",
                    )?;
                }
                std::fmt::Display::fmt(&self.late, f)
            }
        }
    }
}

impl std::fmt::Display for FailureDumpReportAny {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Single(r) => std::fmt::Display::fmt(r, f),
            Self::Dual(r) => std::fmt::Display::fmt(r.as_ref(), f),
            Self::Degraded(r) => std::fmt::Display::fmt(r.as_ref(), f),
        }
    }
}

impl std::fmt::Display for DegradedFailureDumpReport {
    /// Renders the degraded report as a short operator-oriented
    /// banner: schema label, the human reason, the per-vCPU
    /// `parked` / `not_parked` pattern that identifies which vCPUs
    /// stalled, the watchpoint + bss latch state, the optional
    /// live `exit_kind`, and the elapsed-ms budget the coordinator
    /// spent before giving up. Designed to fit a single terminal
    /// scroll without paging — the full diagnostic surface lives
    /// in the structured fields.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("degraded failure dump:\n")?;
        writeln!(f, "  reason: {}", self.reason)?;
        if !self.vcpu_regs.is_empty() {
            let parked = self.vcpu_regs.iter().filter(|s| s.is_some()).count();
            let total = self.vcpu_regs.len();
            writeln!(f, "  vcpus_parked: {parked}/{total}")?;
            for (i, slot) in self.vcpu_regs.iter().enumerate() {
                match slot {
                    Some(s) => writeln!(f, "    vcpu {i}: parked, {s}")?,
                    None => writeln!(f, "    vcpu {i}: not_parked")?,
                }
            }
        }
        writeln!(f, "  watchpoint_hit: {}", self.watchpoint_hit)?;
        writeln!(f, "  bss_latch_state: {}", self.bss_latch_state)?;
        if let Some(kind) = self.exit_kind {
            writeln!(f, "  exit_kind: {kind}")?;
        }
        if self.elapsed_ms != 0 {
            writeln!(f, "  elapsed_ms: {}", self.elapsed_ms)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for FailureDumpReport {
    /// Human-readable rendering of every map plus per-vCPU register
    /// snapshots, per-program runtime stats, per-CPU CPU-time /
    /// softirq / IRQ counters, per-node NUMA stats, per-task
    /// enrichments, scx walker output (rq->scx, DSQ, scx_sched
    /// state), and event-counter timeline. JSON remains the
    /// programmatic form via `serde_json`; this Display is the
    /// default presentation used in test-failure output.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.maps.is_empty()
            && self.vcpu_regs.is_empty()
            && self.sdt_allocations.is_empty()
            && self.scx_static_ranges.is_empty()
            && self.prog_runtime_stats.is_empty()
            && self.prog_runtime_stats_unavailable.is_none()
            && self.per_cpu_time.is_empty()
            && self.per_node_numa.is_empty()
            && self.per_node_numa_unavailable.is_none()
            && self.task_enrichments.is_empty()
            && self.task_enrichments_unavailable.is_none()
            && self.event_counter_timeline.is_empty()
            && self.rq_scx_states.is_empty()
            && self.dsq_states.is_empty()
            && self.scx_sched_state.is_none()
            && self.scx_walker_unavailable.is_none()
            && self.vcpu_perf_at_freeze.is_empty()
        {
            return f.write_str("(empty failure dump)");
        }
        use rayon::prelude::*;
        let rendered_maps: Vec<String> = self.maps.par_iter().map(|m| format!("{m}")).collect();
        let mut first = true;
        for s in &rendered_maps {
            if !first {
                f.write_str("\n\n")?;
            }
            first = false;
            f.write_str(s)?;
        }
        if !self.vcpu_regs.is_empty() {
            if !first {
                f.write_str("\n\n")?;
            }
            first = false;
            f.write_str("vcpu_regs:")?;
            for (i, slot) in self.vcpu_regs.iter().enumerate() {
                f.write_str("\n  ")?;
                match slot {
                    Some(s) => write!(f, "vcpu {i}: {s}")?,
                    None => write!(f, "vcpu {i}: <unavailable>")?,
                }
            }
        }
        for snap in &self.sdt_allocations {
            if !first {
                f.write_str("\n\n")?;
            }
            first = false;
            std::fmt::Display::fmt(snap, f)?;
        }
        if !self.scx_static_ranges.is_empty() {
            if !first {
                f.write_str("\n\n")?;
            }
            first = false;
            std::fmt::Display::fmt(&self.scx_static_ranges, f)?;
        }
        if !self.prog_runtime_stats.is_empty() {
            if !first {
                f.write_str("\n\n")?;
            }
            first = false;
            f.write_str("prog_runtime_stats:")?;
            for stats in &self.prog_runtime_stats {
                f.write_str("\n  ")?;
                std::fmt::Display::fmt(stats, f)?;
            }
        }
        if let Some(reason) = &self.prog_runtime_stats_unavailable {
            if !first {
                f.write_str("\n\n")?;
            }
            first = false;
            write!(f, "prog_runtime_stats: <unavailable: {reason}>")?;
        }
        if !self.per_cpu_time.is_empty() {
            if !first {
                f.write_str("\n\n")?;
            }
            first = false;
            // Per-CPU CPU-time / softirq / IRQ summary. JSON carries
            // the full per-CPU breakdown; this Display surfaces
            // counts so the test-failure log shows what was captured
            // without paging through every CPU.
            write!(f, "per_cpu_time: {} CPUs captured", self.per_cpu_time.len())?;
        }
        if !self.per_node_numa.is_empty() {
            if !first {
                f.write_str("\n\n")?;
            }
            first = false;
            write!(
                f,
                "per_node_numa: {} nodes captured",
                self.per_node_numa.len()
            )?;
        }
        if let Some(reason) = &self.per_node_numa_unavailable {
            if !first {
                f.write_str("\n\n")?;
            }
            first = false;
            write!(f, "per_node_numa: <unavailable: {reason}>")?;
        }
        if !self.task_enrichments.is_empty() {
            if !first {
                f.write_str("\n\n")?;
            }
            first = false;
            write!(
                f,
                "task_enrichments: {} tasks captured",
                self.task_enrichments.len(),
            )?;
        }
        if let Some(reason) = &self.task_enrichments_unavailable {
            if !first {
                f.write_str("\n\n")?;
            }
            first = false;
            write!(f, "task_enrichments: <unavailable: {reason}>")?;
        }
        if !self.rq_scx_states.is_empty()
            || !self.dsq_states.is_empty()
            || self.scx_sched_state.is_some()
        {
            if !first {
                f.write_str("\n\n")?;
            }
            first = false;
            // scx walker section: counts of each sub-walk's output.
            // JSON carries the full per-CPU rq->scx state, per-DSQ
            // task lists, and scx_sched scalars; the Display surface
            // tells the operator what the walker reached.
            write!(
                f,
                "scx_walker: rq_scx={} dsq={} sched={}",
                self.rq_scx_states.len(),
                self.dsq_states.len(),
                if self.scx_sched_state.is_some() {
                    "captured"
                } else {
                    "absent"
                },
            )?;
        }
        if let Some(reason) = &self.scx_walker_unavailable {
            if !first {
                f.write_str("\n\n")?;
            }
            first = false;
            write!(f, "scx_walker: <unavailable: {reason}>")?;
        }
        if !self.event_counter_timeline.is_empty() {
            if !first {
                f.write_str("\n\n")?;
            }
            // Trailing section before vcpu_perf_at_freeze; mirrors
            // the vcpu_perf_at_freeze pattern below — `first` is no
            // longer consulted after this block (the per-counter
            // sparkline rows do their own `\n` writes).
            write!(
                f,
                "event_counter_timeline: {} samples ({}–{}ms)",
                self.event_counter_timeline.len(),
                self.event_counter_timeline
                    .first()
                    .map(|s| s.elapsed_ms)
                    .unwrap_or(0),
                self.event_counter_timeline
                    .last()
                    .map(|s| s.elapsed_ms)
                    .unwrap_or(0),
            )?;
            // Per-counter sparkline. Each row is one of the 13
            // SCX_EV_* counters across all samples in the
            // timeline. Skips counters that stayed at zero across
            // every sample to keep the rendering compact (a
            // counter at zero everywhere has no signal worth
            // surfacing in the human-readable view).
            type EventCounterExtract = (&'static str, fn(&EventCounterSample) -> i64);
            let extract: [EventCounterExtract; 13] = [
                ("select_cpu_fallback", |s| s.select_cpu_fallback),
                ("dispatch_local_dsq_offline", |s| {
                    s.dispatch_local_dsq_offline
                }),
                ("dispatch_keep_last", |s| s.dispatch_keep_last),
                ("enq_skip_exiting", |s| s.enq_skip_exiting),
                ("enq_skip_migration_disabled", |s| {
                    s.enq_skip_migration_disabled
                }),
                ("reenq_immed", |s| s.reenq_immed),
                ("reenq_local_repeat", |s| s.reenq_local_repeat),
                ("refill_slice_dfl", |s| s.refill_slice_dfl),
                ("bypass_duration", |s| s.bypass_duration),
                ("bypass_dispatch", |s| s.bypass_dispatch),
                ("bypass_activate", |s| s.bypass_activate),
                ("insert_not_owned", |s| s.insert_not_owned),
                ("sub_bypass_dispatch", |s| s.sub_bypass_dispatch),
            ];
            for (name, ext) in extract {
                let series: Vec<i64> = self.event_counter_timeline.iter().map(ext).collect();
                if series.iter().all(|&v| v == 0) {
                    continue;
                }
                let line = render_sparkline_i64(&series);
                let last = series.last().copied().unwrap_or(0);
                write!(f, "\n  {name:>30}  {line}  (last={last})")?;
            }
        }
        if !self.vcpu_perf_at_freeze.is_empty() {
            if !first {
                f.write_str("\n\n")?;
            }
            // Trailing section; mirrors the event_counter_timeline
            // pattern — `first` is no longer consulted after this
            // block.
            f.write_str("vcpu_perf_at_freeze:")?;
            for (i, slot) in self.vcpu_perf_at_freeze.iter().enumerate() {
                f.write_str("\n  ")?;
                match slot {
                    Some(s) => write!(
                        f,
                        "vcpu {i}: cycles={} insns={} ipc={:.3} cache_misses={} branch_misses={} (en/ru={}/{} ns)",
                        s.cycles,
                        s.instructions,
                        s.ipc(),
                        s.cache_misses,
                        s.branch_misses,
                        s.time_enabled_ns,
                        s.time_running_ns,
                    )?,
                    None => write!(f, "vcpu {i}: <unavailable>")?,
                }
            }
        }
        Ok(())
    }
}

impl std::fmt::Display for FailureDumpMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Render `map_type` as the symbolic `BPF_MAP_TYPE_<NAME>`
        // suffix when known; fall through to the raw integer for
        // forward-compatibility with kernels newer than this dump
        // renderer. Operators see "type=ringbuf" instead of
        // "type=27"; the rare unknown discriminant still surfaces
        // a stable numeric handle.
        let type_str: std::borrow::Cow<'_, str> =
            match super::render_map::map_type_name(self.map_type) {
                Some(name) => std::borrow::Cow::Borrowed(name),
                None => std::borrow::Cow::Owned(format!("{}", self.map_type)),
            };
        write!(
            f,
            "map {} (type={}, value_size={}, max_entries={})",
            self.name, type_str, self.value_size, self.max_entries
        )?;
        if let Some(err) = &self.error {
            write!(f, " [error: {err}]")?;
        }
        if let Some(value) = &self.value {
            f.write_str("\n")?;
            std::fmt::Display::fmt(value, f)?;
        }
        // Try table layout for homogeneous entries: when every
        // entry has BTF-rendered key + value, both sides are
        // structs of the same shape, every member is an inline
        // scalar, and no entry has a typed payload, render as a
        // compact table. The table form replaces the per-entry
        // `entry { ... }` blocks for the qualifying batch — a
        // 30-entry hash map renders as 31 lines (header + 30 rows)
        // instead of 30 four-line entry blocks.
        if !try_write_entry_table(f, &self.entries)? {
            for entry in &self.entries {
                f.write_str("\n")?;
                std::fmt::Display::fmt(entry, f)?;
            }
        }
        for entry in &self.percpu_entries {
            f.write_str("\n")?;
            std::fmt::Display::fmt(entry, f)?;
        }
        for entry in &self.percpu_hash_entries {
            f.write_str("\n")?;
            std::fmt::Display::fmt(entry, f)?;
        }
        if let Some(arena) = &self.arena {
            let total_pages = arena.pages.len();
            let nonzero = arena
                .pages
                .iter()
                .filter(|p| p.bytes.iter().any(|&b| b != 0))
                .count();
            write!(
                f,
                "\narena: {total_pages} pages captured ({} KiB), \
                 {nonzero} non-zero (see sdt_alloc section + JSON for typed data)",
                total_pages * 4,
            )?;
            if arena.truncated {
                write!(f, " (truncated, {} declared)", arena.declared_pages)?;
            }
        }
        if let Some(rb) = &self.ringbuf {
            // Show capacity, pending bytes (consumer lag), and the
            // four position counters. The pending_pos vs producer_pos
            // gap signals a producer mid-reserve; the pending_bytes
            // vs capacity ratio signals consumer-stall pressure.
            let pct = if rb.capacity == 0 {
                0
            } else {
                (rb.pending_bytes.saturating_mul(100) / rb.capacity).min(100)
            };
            write!(
                f,
                "\nringbuf: capacity={}B, pending={}B ({pct}%), \
                 consumer_pos={}, producer_pos={}, pending_pos={}",
                rb.capacity, rb.pending_bytes, rb.consumer_pos, rb.producer_pos, rb.pending_pos,
            )?;
        }
        if let Some(st) = &self.stack_trace {
            write!(
                f,
                "\nstack_trace: {} of {} buckets populated",
                st.entries.len(),
                st.n_buckets,
            )?;
            if st.truncated {
                f.write_str(" (truncated)")?;
            }
            for entry in &st.entries {
                if entry.pcs.is_empty() {
                    write!(f, "\n  bucket {}: nr={}", entry.bucket_id, entry.nr)?;
                } else {
                    // Show first up to 8 PCs as hex; full list is
                    // in JSON. Write directly to the formatter
                    // with a manual comma separator — no
                    // intermediate Vec<String> + join allocation
                    // per bucket. Stack-trace dumps with hundreds
                    // of buckets compounded the per-bucket Vec
                    // alloc into a measurable overhead.
                    write!(f, "\n  bucket {}: nr={} pcs=[", entry.bucket_id, entry.nr)?;
                    for (i, pc) in entry.pcs.iter().take(8).enumerate() {
                        if i > 0 {
                            f.write_str(", ")?;
                        }
                        write!(f, "{pc:#x}")?;
                    }
                    let extra = entry.pcs.len().saturating_sub(8);
                    if extra > 0 {
                        write!(f, ", +{extra} more")?;
                    }
                    f.write_str("]")?;
                }
            }
        }
        if let Some(fda) = &self.fd_array {
            write!(
                f,
                "\nfd_array: {} of {} slots populated",
                fda.populated, fda.scanned,
            )?;
            if fda.truncated {
                f.write_str(" (slots truncated)")?;
            }
            if !fda.indices.is_empty() {
                // Same pattern as stack-trace: stream directly
                // to the formatter rather than allocating a Vec
                // and joining.
                f.write_str(" indices=[")?;
                for (i, idx) in fda.indices.iter().take(16).enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{idx}")?;
                }
                let extra = fda.indices.len().saturating_sub(16);
                if extra > 0 {
                    write!(f, ", +{extra} more")?;
                }
                f.write_str("]")?;
            }
        }
        Ok(())
    }
}

impl std::fmt::Display for FailureDumpEntry {
    /// Render an entry using the indent-based format:
    ///
    /// ```text
    /// entry: key=<rendered key>
    ///   value: <rendered value>
    ///   payload TypeName:
    ///     field=val   field=val   field=val
    /// ```
    ///
    /// `entry:` is a label and `key=` is a field assignment; the
    /// `=` follows the field-assignment convention used elsewhere
    /// in the dump output. `value:` is also a label introducing
    /// the rendered value (which carries its own `TypeName{...}`
    /// or breadcrumb form). The optional payload follows the
    /// breadcrumb pattern: `payload <rendered>` where the value's
    /// own Type breadcrumb completes the line.
    ///
    /// The renderer is invoked with `depth = 1` for the value and
    /// payload positions so any multi-line struct / array body
    /// indents one level deeper than the entry's own `  ` prefix.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Header: `entry: key=<rendered>` on a single line. Small
        // struct keys collapse to `Type{field=value}` via the
        // btf_render inline path; non-struct keys (Uint, etc.)
        // render as their own scalar form. The hex fallback path
        // (no BTF render) emits the hex string with a trailing
        // `(raw)` marker so the operator distinguishes "no BTF"
        // from a parsed value.
        f.write_str("entry: key=")?;
        match &self.key {
            Some(k) => write_value_at_depth(f, k, 1)?,
            None => write!(f, "{} (raw)", self.key_hex)?,
        }
        // Value: `value: <rendered>` indented one level. The
        // rendered value carries its own Type breadcrumb /
        // braces form depending on size; depth=1 ensures any
        // multi-line body indents two levels deep (4 spaces).
        f.write_str("\n  value: ")?;
        match &self.value {
            Some(v) => write_value_at_depth(f, v, 1)?,
            None => write!(f, "{} (raw)", self.value_hex)?,
        }
        // Typed sdt_alloc payload, when the entry value carried a
        // `struct sdt_data __arena *` field that resolved into a
        // captured arena page. Surfaced AFTER the surface value so
        // the operator reads the surface struct first (with its
        // tid / tptr / data fields) and then the typed payload —
        // matching the order a kernel-side debugger would inspect:
        // chase the pointer, then read the dereferenced struct.
        // The space after `payload` lets the rendered value's own
        // `TypeName:` breadcrumb (or inline `Type{...}` form) read
        // as `payload TypeName:` on the same line.
        if let Some(p) = &self.payload {
            f.write_str("\n  .data ")?;
            write_value_at_depth(f, p, 1)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for FailureDumpPercpuEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let structs: Vec<(usize, &[RenderedMember])> = self
            .per_cpu
            .iter()
            .enumerate()
            .filter_map(|(cpu, slot)| match slot {
                Some(RenderedValue::Struct { members, .. }) => Some((cpu, members.as_slice())),
                _ => None,
            })
            .collect();

        if structs.len() < 2 || structs.iter().any(|(_, m)| m.is_empty()) {
            write!(f, "key {}:", self.key)?;
            for (cpu, slot) in self.per_cpu.iter().enumerate() {
                f.write_str("\n")?;
                match slot {
                    Some(v) => {
                        write!(f, "  cpu {cpu}: ")?;
                        std::fmt::Display::fmt(v, f)?;
                    }
                    None => write!(f, "  cpu {cpu}: <unmapped>")?,
                }
            }
            return Ok(());
        }

        let type_name = match &self.per_cpu.iter().flatten().next() {
            Some(RenderedValue::Struct { type_name, .. }) => type_name.clone(),
            _ => None,
        };
        let n_cpus = self.per_cpu.len();
        match &type_name {
            Some(name) => write!(f, "key {}: struct {name} ({n_cpus} CPUs)", self.key)?,
            None => write!(f, "key {}: ({n_cpus} CPUs)", self.key)?,
        }

        // Skip the cross-CPU dedup pass on hosts with >64 CPUs:
        // the existing dedup loop is O(n²) in the number of
        // groups, so a 256-CPU host with mostly-unique values
        // ends up doing several million RenderedValue equality
        // comparisons (each itself a deep walk of the struct
        // tree). Above the threshold we just emit one row per
        // CPU — at scale, the table form would not fit in any
        // reasonable terminal anyway, and the per-CPU rows are
        // grep-friendly.
        const PERCPU_DEDUP_CPU_LIMIT: usize = 64;
        if n_cpus > PERCPU_DEDUP_CPU_LIMIT {
            for (cpu, slot) in self.per_cpu.iter().enumerate() {
                f.write_str("\n  ")?;
                match slot {
                    Some(v) => {
                        write!(f, "cpu {cpu}: ")?;
                        std::fmt::Display::fmt(v, f)?;
                    }
                    None => write!(f, "cpu {cpu}: <unmapped>")?,
                }
            }
            return Ok(());
        }

        // Group CPUs by identical struct content. Show each unique
        // value once with its CPU list. The find() is O(n) per
        // CPU, which at 64 CPUs is bounded; the >64 fallback above
        // protects callers against the quadratic scaling.
        let mut groups: Vec<(Vec<usize>, &RenderedValue)> = Vec::new();
        let mut unmapped: Vec<usize> = Vec::new();
        for (cpu, slot) in self.per_cpu.iter().enumerate() {
            match slot {
                Some(val) => {
                    if let Some(g) = groups.iter_mut().find(|(_, v)| *v == val) {
                        g.0.push(cpu);
                    } else {
                        groups.push((vec![cpu], val));
                    }
                }
                None => unmapped.push(cpu),
            }
        }
        // Template detection: if every CPU has a unique struct but
        // most fields are identical, show the struct once with a
        // per-CPU table of varying fields.
        if groups.len() >= 3 && groups.iter().all(|(cpus, _)| cpus.len() == 1) {
            let all_structs: Vec<(usize, &[RenderedMember])> = groups
                .iter()
                .filter_map(|(cpus, val)| match val {
                    RenderedValue::Struct { members, .. } => Some((cpus[0], members.as_slice())),
                    _ => None,
                })
                .collect();
            if all_structs.len() == groups.len()
                && all_structs
                    .iter()
                    .all(|(_, m)| m.len() == all_structs[0].1.len())
            {
                let first = all_structs[0].1;
                let mut varying: Vec<usize> = Vec::new();
                for i in 0..first.len() {
                    if all_structs[1..]
                        .iter()
                        .any(|(_, m)| m[i].value != first[i].value)
                    {
                        varying.push(i);
                    }
                }
                if !varying.is_empty() && varying.len() < 8 {
                    // Show common fields once. Zero fields are
                    // suppressed silently — no count line.
                    f.write_str("\n  common:")?;
                    for (i, m) in first.iter().enumerate() {
                        if varying.contains(&i) {
                            continue;
                        }
                        if is_zero(&m.value) {
                            continue;
                        }
                        write!(f, "\n    {}: ", m.name)?;
                        std::fmt::Display::fmt(&m.value, f)?;
                    }
                    // Show varying fields as per-CPU table.
                    f.write_str("\n  per-cpu:")?;
                    f.write_str("\n    cpu")?;
                    for &vi in &varying {
                        write!(f, " | {}", first[vi].name)?;
                    }
                    for (cpu, members) in &all_structs {
                        write!(f, "\n    {cpu:>3}")?;
                        for &vi in &varying {
                            write!(f, " | {}", members[vi].value)?;
                        }
                    }
                    if !unmapped.is_empty() {
                        write!(f, "\n  cpus {unmapped:?}: <unmapped>")?;
                    }
                    return Ok(());
                }
            }
        }
        // Fallback: show each group with its CPU list.
        for (cpus, val) in &groups {
            let cpu_list = if cpus.len() == n_cpus {
                "all CPUs".to_string()
            } else if cpus.len() == 1 {
                format!("cpu {}", cpus[0])
            } else if cpus.windows(2).all(|w| w[1] == w[0] + 1) {
                // Contiguity: every adjacent pair differs by 1.
                // The endpoint-only `last - first + 1 == len` check
                // would falsely accept e.g. [0, 2, 4, 6] (span 7,
                // len 4 — never contiguous) if a duplicate or
                // gap somehow slipped past collection; `windows`
                // is robust to construction errors.
                format!("cpus {}-{}", cpus[0], cpus.last().unwrap())
            } else {
                format!("cpus {:?}", cpus)
            };
            write!(f, "\n  {cpu_list}: ")?;
            std::fmt::Display::fmt(val, f)?;
        }
        if !unmapped.is_empty() {
            write!(f, "\n  cpus {unmapped:?}: <unmapped>")?;
        }
        Ok(())
    }
}

impl std::fmt::Display for FailureDumpPercpuHashEntry {
    /// Match [`FailureDumpEntry`]'s `entry: key=...` header so an
    /// operator scanning the human-readable failure dump sees the
    /// same shape regardless of whether the underlying map is a
    /// plain HASH or a PERCPU_HASH variant. Each per-CPU slot
    /// renders on its own indented line as `cpu N: <value>`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("entry: key=")?;
        match &self.key {
            Some(k) => write_value_at_depth(f, k, 1)?,
            None => write!(f, "{} (raw)", self.key_hex)?,
        }
        // Per-CPU values: list each CPU's slot. Matches the
        // FailureDumpPercpuEntry simple branch (no group folding) for
        // readability — typical PERCPU_HASH maps have small
        // num_cpus * num_keys, so the verbose listing isn't a
        // problem in practice. CPU rows use `cpu N:` as a label
        // (the cpu id is metadata, not a struct field assignment).
        for (cpu, slot) in self.per_cpu.iter().enumerate() {
            f.write_str("\n  ")?;
            match slot {
                Some(v) => {
                    write!(f, "cpu {cpu}: ")?;
                    write_value_at_depth(f, v, 1)?;
                }
                None => write!(f, "cpu {cpu}: <unmapped>")?,
            }
        }
        Ok(())
    }
}
