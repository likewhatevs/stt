//! User-facing display configuration: columns, sections, and
//! display-format shorthands.
//!
//! Three layers, all consumed by the renderer in
//! [`super::write_diff`] / `write_show` paths and by the CLI
//! parsers that map operator-supplied strings to typed values:
//!
//! 1. [`DisplayFormat`] — closed enum of layout shorthands
//!    (Full, DeltaOnly, NoPct, Arrow, PctOnly) that resolve to
//!    a fixed [`Column`] vec via [`compare_columns_for`].
//!    [`show_columns_default`] mirrors the same shape for the
//!    `ctprof show` single-snapshot path.
//!
//! 2. [`Column`] / [`parse_columns`] — closed enumeration of the
//!    rendered cell slots (Group, Threads, Metric, Baseline,
//!    Candidate, Delta, Pct, Arrow, Uptime, SortBy). The
//!    [`parse_columns`] CLI parser takes a comma-separated spec
//!    and rejects unknown names + show-incompatible columns at
//!    parse time so the renderer never sees a mismatched column
//!    set. Order in the resolved vec is the rendered order; the
//!    renderer never re-sorts.
//!
//! 3. [`Section`] / [`parse_sections`] / [`parse_metrics`] —
//!    user-facing filter dimensions. Section is the per-row
//!    section-tag dimension (Primary, TaskstatsDelay, Derived,
//!    CgroupStats, Limits, MemoryStat, MemoryEvents, Pressure,
//!    HostPressure, Smaps, SchedExt) used to scope `--sections
//!    <list>` and `--metrics <list>`. The
//!    [`warn_cgroup_only_sections_under_non_cgroup`] helper
//!    emits a stderr warning at run time when an explicit
//!    `--sections` filter names a cgroup-only section while
//!    `--group-by` is not [`super::GroupBy::Cgroup`] — without
//!    the warning, the section would silently render zero rows.
//!
//! All three layers are CLI-input parsers; none of them dispatch
//! on rendered data shape. The renderer consumes the resolved
//! `Vec<Column>` / `Vec<Section>` / `Vec<&'static str>` directly
//! without re-validating.

use super::{CTPROF_DERIVED_METRICS, CTPROF_METRICS, GroupBy};

/// Per-row display layout for [`write_diff`].
///
/// `Full` (default) emits the seven-column form
/// `(group | threads | metric | baseline | candidate | delta | %)`.
/// The remaining variants are compact shortcuts for common
/// operator workflows; each resolves to a fixed [`Column`] set
/// before the renderer runs. A `--columns` override on the same
/// invocation wins over the format's default column set.
///
/// [`Arrow`] collapses baseline / candidate into a single cell
/// shaped `<baseline> -> <candidate>` so a narrow display still
/// surfaces directionality. The arrow column is paired with the
/// dedicated Delta + Pct + Uptime columns (not fused into the
/// arrow cell itself), so the renderer keeps the deltas
/// readable on either side of the arrow form. The arrow cell
/// shape mirrors [`cgroup_cell`]'s so primary and cgroup tables
/// stay visually consistent.
///
/// [`Arrow`]: DisplayFormat::Arrow
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
#[non_exhaustive]
pub enum DisplayFormat {
    /// Default — emit baseline, candidate, delta, and pct
    /// columns alongside group / threads / metric.
    #[default]
    Full,
    /// Drop baseline + candidate columns; keep delta + pct.
    DeltaOnly,
    /// Drop pct column; keep baseline + candidate + delta.
    NoPct,
    /// Arrow form: `<baseline> -> <candidate>` cell paired with
    /// the dedicated Delta + Pct + Uptime columns. Compact view
    /// for "show me direction at a glance" while still carrying
    /// the deltas on the same row.
    Arrow,
    /// Drop baseline / candidate / delta; keep pct only.
    PctOnly,
}

/// One column slot in the rendered diff/show table. The renderer
/// iterates the resolved [`Column`] vec to build both the
/// header row and each data row, dispatching cell construction
/// per variant. Order in the slice is the rendered order — the
/// renderer never re-sorts.
///
/// Column variants are uniform across compare and show even
/// though show's [`Column::Baseline`], [`Column::Candidate`],
/// [`Column::Delta`], [`Column::Pct`], [`Column::Arrow`] are
/// meaningless for a single snapshot. The show entry point
/// rejects those names at CLI parse time so an operator never
/// reaches the renderer with a mismatched column set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Column {
    /// The group-by axis label (rendered header is "pcomm",
    /// "cgroup", "comm-pattern", or "comm" per `GroupBy`).
    Group,
    /// Thread-count cell (`N` when the count matches across
    /// snapshots, `A->B` arrow form otherwise).
    Threads,
    /// Metric name with bracketed tag suffix.
    Metric,
    /// Baseline value (compare only).
    Baseline,
    /// Candidate value (compare only).
    Candidate,
    /// Signed delta (compare only).
    Delta,
    /// Percentage delta (compare only).
    Pct,
    /// Single-cell `<baseline> -> <candidate> (<delta>)`
    /// (compare only). Mutually exclusive with
    /// `Baseline`/`Candidate`/`Delta`/`Pct` — the arrow form
    /// fuses them.
    Arrow,
    /// Aggregated value cell (show only).
    Value,
    /// Bracketed tag suffix (sched_class + config_gates + dead).
    /// Off by default — opt in with `--columns ...,tags`.
    Tags,
    /// Relative uptime: group age as a percentage of the oldest
    /// thread in the snapshot. 100% = as old as the oldest
    /// thread, 0% = just spawned. Color gradient: green ≥50%,
    /// red <50% (>2x younger than the oldest).
    Uptime,
    /// Sort-by metric summary column. Shows the --sort-by metric's
    /// baseline→candidate (delta%) per group. Only present when
    /// --sort-by is set.
    SortBy,
}

impl Column {
    /// Canonical CLI name. Round-trips through
    /// [`parse_columns`].
    pub fn cli_name(self) -> &'static str {
        match self {
            Column::Group => "group",
            Column::Threads => "threads",
            Column::Metric => "metric",
            Column::Baseline => "baseline",
            Column::Candidate => "candidate",
            Column::Delta => "delta",
            Column::Pct => "%",
            Column::Arrow => "arrow",
            Column::Value => "value",
            Column::Tags => "tags",
            Column::Uptime => "uptime",
            Column::SortBy => "sort-by", // overridden dynamically in colored_header
        }
    }

    /// Header cell text. The group axis carries a per-`GroupBy`
    /// label (`pcomm`, `cgroup`, etc.); other columns echo
    /// [`Self::cli_name`].
    pub fn header(self, group_header: &'static str) -> &'static str {
        match self {
            Column::Group => group_header,
            Column::Threads => "threads",
            Column::Metric => "metric",
            Column::Baseline => "baseline",
            Column::Candidate => "candidate",
            Column::Delta => "delta",
            Column::Pct => "%",
            Column::Arrow => "value",
            Column::Value => "value",
            Column::Tags => "tags",
            Column::Uptime => "%uptime",
            Column::SortBy => "sort-by", // overridden dynamically in colored_header
        }
    }
}

/// Resolve a [`DisplayFormat`] to its default column set
/// (compare-side). Returns the full ordered column slice
/// including the group / threads / metric prefix.
pub(super) fn compare_columns_for(format: DisplayFormat) -> Vec<Column> {
    let mut cols = vec![Column::Group, Column::Threads, Column::Metric];
    let trailing: &[Column] = match format {
        DisplayFormat::Full => &[
            Column::Baseline,
            Column::Candidate,
            Column::Delta,
            Column::Pct,
        ],
        DisplayFormat::DeltaOnly => &[Column::Delta, Column::Pct],
        DisplayFormat::NoPct => &[Column::Baseline, Column::Candidate, Column::Delta],
        DisplayFormat::Arrow => &[Column::Arrow, Column::Delta, Column::Pct, Column::Uptime],
        DisplayFormat::PctOnly => &[Column::Pct],
    };
    cols.extend_from_slice(trailing);
    cols
}

/// Resolve the show-side default column set (no
/// baseline/candidate/delta/pct — show is single-snapshot).
pub(super) fn show_columns_default() -> Vec<Column> {
    vec![
        Column::Group,
        Column::Threads,
        Column::Metric,
        Column::Value,
    ]
}

/// Parse a CLI `--columns` spec into a typed [`Column`] vec.
/// Format: comma-separated names matching [`Column::cli_name`].
/// Whitespace around each name is trimmed. Empty input parses
/// to an empty Vec — caller falls back to the format default.
///
/// `compare_side` controls which subset is allowed:
/// - `true` accepts every variant except [`Column::Value`]
///   (show-only).
/// - `false` accepts every variant except
///   [`Column::Baseline`], [`Column::Candidate`],
///   [`Column::Delta`], [`Column::Pct`], [`Column::Arrow`]
///   (compare-only).
///
/// Errors:
/// - Unknown name (cite the offending token; list valid names).
/// - Wrong-side name (e.g. `value` on compare or `baseline`
///   on show).
/// - Duplicate name across two entries.
/// - Empty token between commas.
/// - `arrow` paired with `baseline` or `candidate` (the arrow
///   cell already shows `baseline -> candidate`; pairing those
///   would render the same data twice). `arrow + delta + %` is
///   allowed and matches the format-default for
///   [`DisplayFormat::Arrow`].
pub fn parse_columns(spec: &str, compare_side: bool) -> anyhow::Result<Vec<Column>> {
    if spec.trim().is_empty() {
        return Ok(Vec::new());
    }
    let allowed: &[Column] = if compare_side {
        &[
            Column::Group,
            Column::Threads,
            Column::Metric,
            Column::Baseline,
            Column::Candidate,
            Column::Delta,
            Column::Pct,
            Column::Arrow,
            Column::Tags,
            Column::Uptime,
        ]
    } else {
        &[
            Column::Group,
            Column::Threads,
            Column::Metric,
            Column::Value,
            Column::Tags,
            Column::Uptime,
        ]
    };
    let valid_names = allowed
        .iter()
        .map(|c| c.cli_name())
        .collect::<Vec<_>>()
        .join(", ");
    let mut out: Vec<Column> = Vec::new();
    let mut seen: std::collections::BTreeSet<&'static str> = std::collections::BTreeSet::new();
    for entry in spec.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            anyhow::bail!(
                "empty entry in --columns spec {spec:?}; \
                 entries are comma-separated and must be non-empty"
            );
        }
        let normalized = entry.to_ascii_lowercase();
        let Some(col) = allowed.iter().copied().find(|c| c.cli_name() == normalized) else {
            anyhow::bail!(
                "unknown column {entry:?} in --columns spec {spec:?}; \
                 must be one of: {valid_names}",
            );
        };
        if !seen.insert(col.cli_name()) {
            anyhow::bail!(
                "duplicate column {entry:?} in --columns spec {spec:?}; \
                 each column may appear at most once"
            );
        }
        out.push(col);
    }
    // The arrow cell renders `<baseline> -> <candidate>`, which
    // visually replaces the separate Baseline / Candidate
    // columns; pairing arrow with either of those names asks
    // the renderer to emit the same data twice. Reject at parse
    // time. Delta and Pct, by contrast, are NOT visually
    // duplicated by the arrow cell — they remain as numeric
    // columns alongside Arrow under the format-default
    // (`compare_columns_for(DisplayFormat::Arrow)` produces
    // `[Group, Threads, Metric, Arrow, Delta, Pct, Uptime]`),
    // so user-supplied `--columns` specs can also include them.
    let has_arrow = out.iter().any(|c| matches!(c, Column::Arrow));
    let has_redundant_with_arrow = out
        .iter()
        .any(|c| matches!(c, Column::Baseline | Column::Candidate));
    if has_arrow && has_redundant_with_arrow {
        anyhow::bail!(
            "column 'arrow' is mutually exclusive with baseline/candidate \
             — the arrow cell already shows baseline -> candidate. \
             Pair arrow with delta/% (or use it alone) instead."
        );
    }
    Ok(out)
}

/// One sub-table emitted by [`write_diff`] / `write_show`.
/// `--sections` filters which sub-tables render — every section
/// not in the filter is suppressed before its emission gate
/// (zero-suppression, group-by-cgroup gating, etc.) runs, so a
/// section that would otherwise emit when its data is present
/// stays silent when omitted from the filter.
///
/// Variant order tracks the rendering order in [`write_diff`]
/// and `write_show` so iteration over [`Section::ALL`] walks
/// the table in the order the operator sees it. The
/// [`Self::cli_name`] tokens are the spelling accepted by
/// [`parse_sections`] — round-trip through that parser pins the
/// vocabulary against drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Section {
    /// Per-thread metric table — the primary rows produced by
    /// `build_row` / `aggregate`, EXCLUDING the taskstats
    /// genetlink-sourced rows which carry their own
    /// [`Section::TaskstatsDelay`] tag for separate filtering.
    /// Always rendered first.
    Primary,
    /// `## Derived metrics` section emitted from
    /// [`CTPROF_DERIVED_METRICS`].
    Derived,
    /// Cgroup-enrichment table (`cpu_usage_usec`,
    /// `nr_throttled`, `throttled_usec`, `memory_current`).
    /// Compare- and show-side both gate on `GroupBy::Cgroup`
    /// plus a non-empty `cgroup_stats` map; the `--sections`
    /// filter runs ahead of that gate.
    CgroupStats,
    /// `## Cgroup limits / knobs` table — operator-set
    /// configuration (`cpu.max`, `cpu.weight`, `memory.max`,
    /// `memory.high`, `pids.current`, `pids.max`).
    Limits,
    /// `## memory.stat` long-table — kernel-emitted memory
    /// counters per cgroup.
    MemoryStat,
    /// `## memory.events` long-table — pressure-event counters
    /// per cgroup.
    MemoryEvents,
    /// `## Pressure / <resource>` per-cgroup PSI sub-tables
    /// (cpu / memory / io / irq).
    Pressure,
    /// `## Host pressure / <resource>` host-level PSI
    /// sub-tables.
    HostPressure,
    /// `## smaps_rollup` memory-mapping summary. Compare-side
    /// rows are keyed per pcomm pattern under default
    /// normalization (matching the [`GroupBy::Pcomm`] join key)
    /// or per literal `pcomm[tgid]` PID under
    /// [`CompareOptions::no_thread_normalize`]; show-side rows
    /// are emitted per-PID directly off each captured leader
    /// thread.
    Smaps,
    /// `## sched_ext` global sysfs section (`state`,
    /// `switch_all`, `nr_rejected`, `hotplug_seq`,
    /// `enable_seq`).
    SchedExt,
    /// Taskstats genetlink-sourced rows in the primary table —
    /// the 34 fields covering the eight delay-accounting
    /// categories (`cpu_delay_*`, `blkio_delay_*`,
    /// `swapin_delay_*`, `freepages_delay_*`,
    /// `thrashing_delay_*`, `compact_delay_*`, `wpcopy_delay_*`,
    /// `irq_delay_*`) plus the two memory watermarks
    /// (`hiwater_rss_bytes`, `hiwater_vm_bytes`). Renders inside
    /// the primary table alongside [`Section::Primary`] rows;
    /// each [`CtprofMetricDef`] carries a [`Self`] tag in its
    /// [`CtprofMetricDef::section`] field, and the primary
    /// table emitter checks
    /// [`DisplayOptions::is_section_enabled`] per row so
    /// `--sections taskstats-delay` shows only the taskstats
    /// rows, `--sections primary` excludes them, and either
    /// alone keeps the primary table open. Captured via the
    /// kernel's TASKSTATS family in [`crate::taskstats`].
    TaskstatsDelay,
}

impl Section {
    /// Every variant in rendering order. Single source of
    /// truth — `parse_sections` walks this slice to validate
    /// names and the [`DisplayOptions::is_section_enabled`]
    /// default-empty case treats it as "all on."
    pub const ALL: &'static [Section] = &[
        Section::Primary,
        Section::TaskstatsDelay,
        Section::Derived,
        Section::CgroupStats,
        Section::Limits,
        Section::MemoryStat,
        Section::MemoryEvents,
        Section::Pressure,
        Section::HostPressure,
        Section::Smaps,
        Section::SchedExt,
    ];

    /// Canonical CLI name. Round-trips through
    /// [`parse_sections`].
    pub fn cli_name(self) -> &'static str {
        match self {
            Section::Primary => "primary",
            Section::TaskstatsDelay => "taskstats-delay",
            Section::Derived => "derived",
            Section::CgroupStats => "cgroup-stats",
            Section::Limits => "cgroup-limits",
            Section::MemoryStat => "memory-stat",
            Section::MemoryEvents => "memory-events",
            Section::Pressure => "pressure",
            Section::HostPressure => "host-pressure",
            Section::Smaps => "smaps-rollup",
            Section::SchedExt => "sched-ext",
        }
    }

    /// Returns `true` when this section's data only exists
    /// under [`GroupBy::Cgroup`] grouping. Five sections live
    /// behind the cgroup outer-gate in `write_diff` /
    /// `write_show`: [`CgroupStats`](Section::CgroupStats),
    /// [`Limits`](Section::Limits),
    /// [`MemoryStat`](Section::MemoryStat),
    /// [`MemoryEvents`](Section::MemoryEvents), and
    /// [`Pressure`](Section::Pressure). Naming any of them
    /// under `--sections` while using a non-cgroup
    /// `--group-by` would silently produce zero rows for that
    /// section — the framework warns the operator instead via
    /// [`warn_cgroup_only_sections_under_non_cgroup`].
    pub fn requires_cgroup_grouping(self) -> bool {
        matches!(
            self,
            Section::CgroupStats
                | Section::Limits
                | Section::MemoryStat
                | Section::MemoryEvents
                | Section::Pressure
        )
    }
}

/// Emit a stderr warning when an explicit `--sections` filter
/// names a cgroup-only section while `--group-by` is not
/// [`GroupBy::Cgroup`]. Without the warning, the section would
/// silently render zero rows (its outer-gate suppresses it),
/// leaving the operator wondering whether their snapshot lacked
/// the data or their flag was misconfigured.
///
/// Only fires when the filter is explicitly populated — the
/// default-empty case ("render every section that has data")
/// is already self-correcting and emits no warning. Non-cgroup
/// sections in the same explicit filter are not flagged; only
/// the cgroup-only entries are called out.
///
/// Diagnostic shape per cgroup-only entry: one line of the
/// form `section 'X' requires --group-by cgroup; omitted under
/// --group-by Y`. The text is pinned by
/// [`format_cgroup_only_section_warning`] so a wording drift
/// surfaces in unit tests rather than at the operator's
/// terminal.
pub fn warn_cgroup_only_sections_under_non_cgroup(sections: &[Section], group_by: GroupBy) {
    if sections.is_empty() || group_by == GroupBy::Cgroup {
        return;
    }
    for section in sections {
        if section.requires_cgroup_grouping() {
            eprintln!("{}", format_cgroup_only_section_warning(*section, group_by));
        }
    }
}

/// Render the per-section "requires --group-by cgroup" warning
/// text. Split from [`warn_cgroup_only_sections_under_non_cgroup`]
/// so the wording can be unit-tested without capturing stderr.
/// The `--group-by` axis is rendered via [`group_by_cli_name`]
/// so the operator-facing label matches the clap value-enum
/// spelling they typed (`pcomm` / `cgroup` / `comm` /
/// `comm-exact`).
pub(crate) fn format_cgroup_only_section_warning(section: Section, group_by: GroupBy) -> String {
    format!(
        "section '{}' requires --group-by cgroup; omitted under --group-by {}",
        section.cli_name(),
        group_by_cli_name(group_by),
    )
}

/// Operator-facing spelling of a [`GroupBy`] variant — matches
/// the clap value-enum tokens accepted on the CLI. Centralized
/// here so the warning surface and any future diagnostic site
/// share one source of truth.
fn group_by_cli_name(group_by: GroupBy) -> &'static str {
    match group_by {
        GroupBy::Pcomm => "pcomm",
        GroupBy::Cgroup => "cgroup",
        GroupBy::Comm => "comm",
        GroupBy::CommExact => "comm-exact",
        GroupBy::All => "all",
    }
}

/// Parse a CLI `--sections` spec into a typed [`Section`] vec.
/// Format: comma-separated names matching [`Section::cli_name`].
/// Whitespace around each name is trimmed. Empty input parses
/// to an empty `Vec` — caller treats that as "every section
/// renders" via [`DisplayOptions::is_section_enabled`].
///
/// Errors (mirrored from [`parse_columns`] so the two CLI
/// surfaces report drift identically):
/// - Unknown name (cite the offending token; list valid names).
/// - Duplicate name across two entries.
/// - Empty token between commas.
pub fn parse_sections(spec: &str) -> anyhow::Result<Vec<Section>> {
    if spec.trim().is_empty() {
        return Ok(Vec::new());
    }
    let valid_names = Section::ALL
        .iter()
        .map(|s| s.cli_name())
        .collect::<Vec<_>>()
        .join(", ");
    let mut out: Vec<Section> = Vec::new();
    let mut seen: std::collections::BTreeSet<&'static str> = std::collections::BTreeSet::new();
    for entry in spec.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            anyhow::bail!(
                "empty entry in --sections spec {spec:?}; \
                 entries are comma-separated and must be non-empty"
            );
        }
        let normalized = entry.to_ascii_lowercase();
        let Some(section) = Section::ALL
            .iter()
            .copied()
            .find(|s| s.cli_name() == normalized)
        else {
            anyhow::bail!(
                "unknown section {entry:?} in --sections spec {spec:?}; \
                 must be one of: {valid_names}",
            );
        };
        if !seen.insert(section.cli_name()) {
            anyhow::bail!(
                "duplicate section {entry:?} in --sections spec {spec:?}; \
                 each section may appear at most once"
            );
        }
        out.push(section);
    }
    Ok(out)
}

/// Parse a CLI `--metrics` spec into a typed
/// `Vec<&'static str>` of registry names. Format:
/// comma-separated names that must each match a `name` field
/// from either [`CTPROF_METRICS`] or
/// [`CTPROF_DERIVED_METRICS`]. Whitespace around each name
/// is trimmed. Empty input parses to an empty `Vec` — caller
/// treats that as "every metric renders" via
/// [`DisplayOptions::is_metric_enabled`], mirroring
/// [`parse_sections`]'s empty-input semantic.
///
/// The returned `&'static str`s point into the registry's own
/// `name` fields (not into the input `spec`), so the parsed
/// vector survives the input string going out of scope and
/// equality checks against registry names are pointer-stable.
///
/// Errors (mirrored from [`parse_sections`] / [`parse_columns`]
/// so the three CLI surfaces report drift identically):
/// - Unknown name (cite the offending token).
/// - Duplicate name across two entries.
/// - Empty token between commas.
pub fn parse_metrics(spec: &str) -> anyhow::Result<Vec<&'static str>> {
    if spec.trim().is_empty() {
        return Ok(Vec::new());
    }
    let mut out: Vec<&'static str> = Vec::new();
    let mut seen: std::collections::BTreeSet<&'static str> = std::collections::BTreeSet::new();
    for entry in spec.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            anyhow::bail!(
                "empty entry in --metrics spec {spec:?}; \
                 entries are comma-separated and must be non-empty"
            );
        }
        // Linear scan over both registries — name lookup is
        // not on a hot path. Returns the registry's own
        // `&'static str` so the parsed vec is pointer-stable
        // and survives the input string's lifetime.
        let primary = CTPROF_METRICS
            .iter()
            .find(|m| m.name == entry)
            .map(|m| m.name);
        let derived = CTPROF_DERIVED_METRICS
            .iter()
            .find(|d| d.name == entry)
            .map(|d| d.name);
        let Some(name) = primary.or(derived) else {
            anyhow::bail!(
                "unknown metric {entry:?} in --metrics spec {spec:?}; \
                 must be one of the names from `ctprof metric-list` \
                 (CTPROF_METRICS or CTPROF_DERIVED_METRICS)",
            );
        };
        if !seen.insert(name) {
            anyhow::bail!(
                "duplicate metric {entry:?} in --metrics spec {spec:?}; \
                 each metric may appear at most once"
            );
        }
        out.push(name);
    }
    Ok(out)
}
