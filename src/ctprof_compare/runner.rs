//! CLI entry-point orchestration: parsers, top-level command
//! dispatchers, args struct, and the `DisplayOptions` shape that
//! `write_diff` consumes.
//!
//! Three layers:
//!
//! 1. CLI parsers — [`parse_sort_by`] (the sole runner-side
//!    parser; [`super::parse_columns`] / [`super::parse_sections`] /
//!    [`super::parse_metrics`] live in `columns.rs` because they
//!    parse user-visible filter dimensions).
//!
//! 2. CLI args + display options — [`CtprofCompareArgs`]
//!    (`ctprof compare` flags surface) and [`DisplayOptions`]
//!    (the renderer's plumbed-as-one-struct config bundle).
//!
//! 3. Top-level dispatch — [`run_compare`] / [`run_metric_list`]
//!    drive the `ctprof compare` and `ctprof list-metrics` paths
//!    end-to-end. [`write_metric_list`] / [`print_metric_list`]
//!    emit the metric registry. [`print_diff`] is the post-compare
//!    print convenience that wraps [`super::write_diff`] +
//!    [`limit_sections`] / [`flush_section`].

use std::fmt;
use std::path::Path;

use anyhow::Context;

use crate::ctprof::CtprofSnapshot;

use super::{
    AggRule, CTPROF_DERIVED_METRICS, CTPROF_METRICS, Column, CompareOptions, CtprofDiff,
    CtprofMetricDef, DisplayFormat, GroupBy, Section, SortKey,
    columns::{compare_columns_for, show_columns_default},
    compare, metric_tags, parse_columns, parse_metrics, parse_sections,
    warn_cgroup_only_sections_under_non_cgroup, write_diff,
};

/// Parse a `--sort-by` CLI value into a list of [`SortKey`]s.
/// Spec format: `metric1[:dir1],metric2[:dir2],...` where each
/// `metric` is a name from [`CTPROF_METRICS`] or
/// [`CTPROF_DERIVED_METRICS`] and `dir` is `asc` or `desc`
/// (case-insensitive — `:DESC`, `:Asc`, `:asc` all work).
/// Direction defaults to `desc` (largest delta first — operator
/// "show me the largest changes" default).
///
/// Whitespace around the metric name and around the direction
/// is trimmed independently, so `wait_sum : desc` and
/// `wait_sum:desc` produce identical [`SortKey`] values.
///
/// Each parsed [`SortKey`] stores the matched registry name as
/// `&'static str` (not a copy of the user's input), so downstream
/// equality with [`CtprofMetricDef::name`] or
/// [`DerivedMetricDef::name`] is a content-equality check
/// (`str::eq`) over the same registry-owned bytes — no per-key
/// allocation outlives this call. The two registries are
/// disjoint, so a name resolves unambiguously to one or the
/// other.
///
/// Sorts groups by their aggregated metric values under whatever
/// `--group-by` axis is in effect. The same spec works under
/// every grouping (pcomm / cgroup / comm / comm-exact) — group
/// rank reflects the per-group aggregate (sum, max, etc. per
/// the metric's [`AggRule`]) of the named metric, OR the
/// per-group derived value for entries from
/// [`CTPROF_DERIVED_METRICS`].
///
/// Examples:
/// - `"wait_sum"` → one key, descending.
/// - `"wait_sum:asc"` → one key, ascending.
/// - `"wait_sum:desc,run_time_ns:desc"` → two keys, both
///   descending; lexicographic.
/// - `"avg_wait_ns:desc"` → one key referencing a derived
///   metric, descending.
/// - `""` → empty Vec (caller falls back to default sort).
///
/// Errors:
/// - Unknown metric name (not in [`CTPROF_METRICS`] AND not
///   in [`CTPROF_DERIVED_METRICS`]).
/// - Categorical metric name (one whose [`AggRule`] is
///   [`AggRule::Mode`] / [`AggRule::ModeChar`] /
///   [`AggRule::ModeBool`] — string- / char- / bool-valued, no
///   scalar to sort by). The default sort already places mode
///   rows last under the `delta_pct` ladder; sorting BY a mode
///   metric would silently degrade to alphabetical group order.
/// - Duplicate metric name across two entries (e.g.
///   `--sort-by wait_sum,wait_sum`). The second key would never
///   contribute to the lex ordering, so it's an operator typo
///   rather than a meaningful spec.
/// - Direction string other than `asc` / `desc`.
/// - Empty token between commas (e.g. `"a,,b"`).
pub fn parse_sort_by(spec: &str) -> anyhow::Result<Vec<SortKey>> {
    if spec.is_empty() {
        return Ok(Vec::new());
    }
    // Build a `name → &'static CtprofMetricDef` index so the
    // lookup returns the canonical registry pointer (for storing
    // in SortKey) AND the AggRule (for the categorical-reject
    // check).
    let registry: std::collections::BTreeMap<&'static str, &'static CtprofMetricDef> =
        CTPROF_METRICS.iter().map(|m| (m.name, m)).collect();
    let mut out: Vec<SortKey> = Vec::new();
    let mut seen: std::collections::BTreeSet<&'static str> = std::collections::BTreeSet::new();
    for entry in spec.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            anyhow::bail!(
                "empty entry in --sort-by spec {spec:?}; \
                 entries are comma-separated and must be non-empty"
            );
        }
        let (metric, descending) = match entry.split_once(':') {
            Some((m, dir)) => {
                // Trim both sides (`"wait_sum : DESC"` → metric
                // `"wait_sum"` and direction `"DESC"`) and lowercase
                // the direction so `:DESC` / `:Asc` / `:asc` are
                // accepted equivalently. Operator-typed CLI input
                // is forgiving about case; the canonical form
                // stored in [`SortKey`] is still derived from the
                // matched ascii literal.
                let dir_norm = dir.trim().to_ascii_lowercase();
                match dir_norm.as_str() {
                    "desc" => (m, true),
                    "asc" => (m, false),
                    _ => anyhow::bail!(
                        "invalid direction {dir:?} in --sort-by entry \
                         {entry:?}; expected `asc` or `desc`"
                    ),
                }
            }
            None => (entry, true),
        };
        let metric = metric.trim();
        // Resolve the input name against either the primary
        // registry or the derived registry. The two namespaces
        // are disjoint (the registry_and_derived_names_disjoint
        // test pins this), so a name resolves unambiguously.
        // The categorical-reject check applies only to primary
        // metrics (derived metrics never go through AggRule).
        let resolved_name: Option<&'static str> = if let Some(def) = registry.get(metric).copied() {
            if matches!(
                def.rule,
                AggRule::Mode(_) | AggRule::ModeChar(_) | AggRule::ModeBool(_),
            ) {
                anyhow::bail!(
                    "metric {metric:?} is categorical (no numeric value to sort by); \
                     --sort-by accepts only metrics whose AggRule yields a scalar \
                     (Sum*, Max*, Range*, or Affinity)"
                );
            }
            Some(def.name)
        } else {
            CTPROF_DERIVED_METRICS
                .iter()
                .find(|d| d.name == metric)
                .map(|d| d.name)
        };
        let Some(canonical) = resolved_name else {
            // Sorted comma-separated list keeps the diagnostic
            // copy-pasteable — operator can grep the names
            // without parsing BTreeSet debug syntax. The
            // rendered table cells append `[tag]` suffixes (e.g.
            // `wait_sum [non-ext] [SCHEDSTATS]`), but
            // `--sort-by` accepts only the bare metric name; if
            // the operator pasted the rendered cell verbatim
            // the trailing bracket would land here, hence the
            // explicit hint.
            let mut valid: Vec<&'static str> = registry.keys().copied().collect();
            for d in CTPROF_DERIVED_METRICS {
                valid.push(d.name);
            }
            valid.sort();
            let valid = valid.join(", ");
            anyhow::bail!(
                "unknown metric {metric:?} in --sort-by spec {spec:?}; \
                 use the bare metric name, not the rendered cell with \
                 [tag] suffixes; must be one of: {valid}",
            );
        };
        if !seen.insert(canonical) {
            anyhow::bail!(
                "duplicate metric {metric:?} in --sort-by spec {spec:?}; \
                 each metric may appear at most once across all sort keys"
            );
        }
        out.push(SortKey {
            metric: canonical,
            descending,
        });
    }
    Ok(out)
}

/// Aggregate display options for the renderer. Plumbed as a
/// single struct through [`write_diff`] so a future addition
/// lands in one place without growing every signature. The
/// show-side entry (`write_show` in `src/bin/ktstr.rs`) keeps
/// a flatter signature for historical reasons but mirrors the
/// same field semantics — `--wrap`, `--sections`, `--metrics`
/// reach show via `wrap` / `sections` / `metrics` parameters
/// that share the same
/// helpers (`new_wrapped_table`, [`Section::cli_name`]).
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct DisplayOptions {
    /// Format shorthand. Default [`DisplayFormat::Full`].
    pub format: DisplayFormat,
    /// User-supplied column override; empty Vec means "use the
    /// format's default column set." Set via [`parse_columns`].
    pub columns: Vec<Column>,
    /// When `true`, tables render with comfy-table's
    /// [`comfy_table::ContentArrangement::Dynamic`] layout
    /// (terminal-width-aware cell wrapping). When `false`,
    /// tables render with `ContentArrangement::Disabled` — the
    /// prior shape, where columns can spill past the terminal
    /// edge. Default `false` keeps existing operator workflows
    /// untouched until the flag is opted into.
    pub wrap: bool,
    /// User-supplied section filter; empty Vec means "render
    /// every section that has data" — the unfiltered default.
    /// Non-empty restricts the rendered output to the listed
    /// sections only. Set via [`parse_sections`].
    pub sections: Vec<Section>,
    /// User-supplied per-metric row filter; empty Vec means
    /// "render every metric in the primary + derived sections"
    /// — the unfiltered default. Non-empty restricts the
    /// rendered rows to the listed metric names (which must
    /// be in [`CTPROF_METRICS`] or
    /// [`CTPROF_DERIVED_METRICS`]). Set via
    /// [`parse_metrics`].
    ///
    /// Distinct from [`Self::sections`]: sections gate whole
    /// sub-tables (primary, derived, cgroup-stats, …);
    /// metrics gate individual ROWS within the primary and
    /// derived sub-tables. The two compose — naming
    /// `--sections primary` and `--metrics run_time_ns` shows
    /// a single primary row.
    pub metrics: Vec<&'static str>,
    /// Maximum rendered lines per section. Sections whose table
    /// output exceeds this limit are truncated with a notice.
    /// `0` means unlimited (no truncation). Default `750`.
    pub section_line_limit: usize,
}

/// Arguments for the `ktstr ctprof compare` subcommand.
#[derive(Debug, clap::Args)]
pub struct CtprofCompareArgs {
    /// Baseline snapshot (`.ctprof.zst`) from `ktstr ctprof capture -o`.
    pub baseline: std::path::PathBuf,
    /// Candidate snapshot (`.ctprof.zst`) from `ktstr ctprof capture -o`.
    pub candidate: std::path::PathBuf,
    /// Grouping key. `pcomm` (default) aggregates per process
    /// name with token-based pattern normalization (so
    /// `worker-{0..N}` parent processes cluster into one
    /// `worker-{N}` bucket); `cgroup` per cgroup path; `comm`
    /// aggregates threads by NAME PATTERN under the same
    /// token-based normalizer (digits, hex,
    /// alpha-prefix-digits collapse into placeholders so
    /// `tokio-worker-{0..N}` and `kworker/u8:7` cluster); use
    /// `--no-thread-normalize` to disable that collapse and group
    /// by literal `comm` / `pcomm` instead. `comm-exact` is a
    /// synonym for `comm --no-thread-normalize`.
    ///
    /// Under `all` (default): also activates fudging — pairs
    /// of cgroups with renamed-but-identical thread populations
    /// (Jaccard similarity ≥ 0.90 over (pcomm, comm) thread
    /// types, both sides ≥ 10 distinct types) are joined for
    /// diffing instead of surfacing as orphans. Fudged rows
    /// render with a `[fudged: <leaf>]` marker; the
    /// `## Fudged cgroup matches` section at the bottom of the
    /// output details the matched pairs and their
    /// jaccard / overlap / cascade roots / residuals.
    #[arg(long, value_enum, default_value_t = GroupBy::All, help_heading = "Grouping")]
    pub group_by: GroupBy,
    /// Glob patterns that collapse dynamic cgroup path segments
    /// so structurally-equivalent cgroups across runs group
    /// together. Example:
    /// `--cgroup-flatten '/kubepods/*/workload'` treats different
    /// pod IDs as the same group. Repeatable. Independent of
    /// `--no-cg-normalize`: explicit globs apply first, then
    /// auto-normalize runs unless disabled.
    #[arg(long, help_heading = "Grouping")]
    pub cgroup_flatten: Vec<String>,
    /// Disable token-based pattern normalization across every
    /// name-family axis: `--group-by comm`, `--group-by pcomm`,
    /// AND the `## smaps_rollup` per-process keying (which
    /// normalizes by the pcomm pattern by default — see
    /// `collect_smaps_rollup`). With this flag set:
    /// threads / processes group by their literal name; smaps
    /// rows preserve their per-PID identity (`pcomm[tgid]`
    /// instead of the normalized pcomm pattern). The
    /// digit/hex/alpha-prefix placeholders are bypassed on every
    /// axis. Has no effect under `--group-by comm-exact`
    /// (already literal) or `--group-by cgroup`. Mirror of
    /// `--no-cg-normalize` for the cgroup axis.
    #[arg(long, help_heading = "Grouping")]
    pub no_thread_normalize: bool,
    /// Disable token-based pattern normalization for the cgroup
    /// axis (`--group-by cgroup`). Cgroup paths group by literal
    /// post-`--cgroup-flatten` path — Layer 1 (systemd template
    /// `@<id>.service` → `@{I}.service`), Layer 2 (token
    /// normalization), and Layer 3 (tighten) are all bypassed.
    /// Has no effect under any other grouping.
    #[arg(long, help_heading = "Grouping")]
    pub no_cg_normalize: bool,
    /// Multi-key sort spec for the diff rows. Format:
    /// `metric1[:dir1],metric2[:dir2],...` where each `metric` is
    /// one of the primary or derived metric names (run
    /// `ctprof metric-list` for the full vocabulary) and
    /// `dir` is `asc` or `desc` (default `desc`). Groups rank by
    /// the tuple (`metric1_delta`, `metric2_delta`, ...) under
    /// lexicographic order with per-key direction; rows within a
    /// group keep registry order. Empty (the default) keeps the
    /// "biggest |delta_pct|" sort. Examples:
    /// - `--sort-by wait_sum:desc,run_time_ns:desc` — rank by
    ///   the largest scheduler-wait deltas first, breaking ties
    ///   by run-time delta.
    /// - `--sort-by hiwater_rss_bytes:desc` — rank by the
    ///   largest peak-RSS growth across the snapshot. Useful
    ///   for memory-leak investigations.
    /// - `--sort-by avg_wait_ns:asc` — rank by smallest average
    ///   wait time first; surfaces the most-improved processes.
    ///
    /// Affects only the per-thread metric table and the
    /// derived-metrics section. The `## smaps_rollup`
    /// sub-table sorts process rows independently by total Rss
    /// descending (its own built-in default; see
    /// [`write_diff`]); a future flag could expose that knob,
    /// but `--sort-by` does not propagate to it today.
    ///
    /// Parsed by [`parse_sort_by`] into [`CompareOptions::sort_by`].
    #[arg(long, default_value = "", help_heading = "Display")]
    pub sort_by: String,
    /// Per-row column layout. `full` (default) emits the
    /// seven-column form; `delta-only` drops baseline +
    /// candidate; `no-pct` drops the percentage column;
    /// `arrow` collapses baseline / candidate into one
    /// `baseline -> candidate` cell paired with separate Delta
    /// and Pct columns; `pct-only` keeps just the percentage.
    /// `--columns` (below) overrides the format's default
    /// column set when both are present.
    #[arg(long, value_enum, default_value_t = DisplayFormat::Arrow, help_heading = "Display")]
    pub display_format: DisplayFormat,
    /// Comma-separated column names to render. Empty (the
    /// default) means "use the column set selected by
    /// --display-format." Valid names: `group`, `threads`,
    /// `metric`, `baseline`, `candidate`, `delta`, `%`,
    /// `arrow`. Order in the spec is the rendered order.
    /// Example: `--columns metric,delta,%`. Applies to the
    /// `primary` section's per-metric table only; secondary
    /// tables (cgroup-stats, smaps-rollup, etc.) have fixed
    /// column shapes and ignore this flag.
    #[arg(long, default_value = "", help_heading = "Display")]
    pub columns: String,
    /// Comma-separated section names to render. Empty (the
    /// default) renders every section that has data. When
    /// non-empty, restricts output to the listed sub-tables —
    /// every section not named is suppressed before its
    /// data-availability gate runs. Valid names: `primary`,
    /// `taskstats-delay`, `derived`, `cgroup-stats`,
    /// `cgroup-limits`, `memory-stat`, `memory-events`,
    /// `pressure`, `host-pressure`, `smaps-rollup`,
    /// `sched-ext`. Useful for narrowing a wide compare to one
    /// area of interest. Example:
    /// `--sections primary,host-pressure`.
    #[arg(long, default_value = "", help_heading = "Filter")]
    pub sections: String,
    /// Comma-separated metric names to render. Empty (the
    /// default) renders every metric in the primary and
    /// derived sub-tables. When non-empty, restricts the
    /// rendered ROWS to the listed names — names must come
    /// from the `ctprof metric-list` vocabulary
    /// (CTPROF_METRICS or CTPROF_DERIVED_METRICS).
    /// Useful for zooming on a specific counter family
    /// without computing every metric: `--metrics
    /// run_time_ns,wait_sum,affine_success_ratio`. Composes
    /// with `--sections` — naming `--sections primary
    /// --metrics run_time_ns` shows a single primary row.
    #[arg(long, default_value = "", help_heading = "Filter")]
    pub metrics: String,
    /// Wrap table cells to fit the terminal width. Off by
    /// default — wide tables can spill past the terminal edge,
    /// matching the prior shell-pipeline-friendly layout. When
    /// set, cells too wide for the available width wrap inside
    /// the cell rather than overflowing, at the cost of taller
    /// rows. The wrap kicks in only when stdout is a tty (the
    /// terminal width is unknown otherwise); when piped to a
    /// file or another command, the flag is silently dropped
    /// and output stays unwrapped so awk/grep pipelines see
    /// the same byte sequence as without the flag.
    #[arg(long, help_heading = "Display")]
    pub wrap: bool,
    /// Maximum rendered lines per section. Sections whose table
    /// output exceeds this limit are truncated with a notice
    /// showing the number of hidden lines. Applies independently
    /// to each sub-table (primary, derived, smaps-rollup, etc.).
    /// `0` disables truncation entirely. Default `500`.
    #[arg(long, default_value_t = 500, help_heading = "Display")]
    pub limit: usize,
}

/// Entry point for the compare CLI. Parses `--sort-by` first,
/// then loads both snapshots, computes the diff, prints the
/// table, and returns `0` on success. Exits non-zero only on
/// I/O or parse errors; a non-empty diff is data, not a
/// failure.
///
/// Order is deliberate: `parse_sort_by` runs before the
/// snapshot loads so an operator typo in the spec (`--sort-by
/// not_a_real_metric`) fails fast without waiting on disk I/O.
/// Without this ordering the operator pays for two snapshot
/// loads only to hit the parser error after — and an
/// integration test driving a malformed spec against
/// non-existent snapshot paths would surface the load failure
/// instead of the parser failure (the path the test actually
/// pins).
pub fn run_compare(args: &CtprofCompareArgs) -> anyhow::Result<i32> {
    let sort_by = parse_sort_by(&args.sort_by)
        .with_context(|| format!("parse --sort-by {:?}", args.sort_by))?;
    // Parse --columns alongside --sort-by so a malformed spec
    // surfaces before the snapshot loads. compare_side: true
    // for the diff renderer. --sections / --metrics share the
    // same fail-fast contract — an unknown name should not pay
    // for two snapshot loads before failing.
    let columns = parse_columns(&args.columns, true)
        .with_context(|| format!("parse --columns {:?}", args.columns))?;
    let sections = parse_sections(&args.sections)
        .with_context(|| format!("parse --sections {:?}", args.sections))?;
    let metrics = parse_metrics(&args.metrics)
        .with_context(|| format!("parse --metrics {:?}", args.metrics))?;

    // Warn the operator if any explicitly-named section is
    // cgroup-only but the requested grouping isn't cgroup —
    // those sections would silently render zero rows under the
    // outer GroupBy::Cgroup gate in `write_diff` otherwise.
    // The warning fires before snapshot load so the operator
    // sees it immediately, not after a long disk-I/O wait.
    warn_cgroup_only_sections_under_non_cgroup(&sections, args.group_by);

    let baseline = CtprofSnapshot::load(&args.baseline)
        .with_context(|| format!("load baseline {}", args.baseline.display()))?;
    let candidate = CtprofSnapshot::load(&args.candidate)
        .with_context(|| format!("load candidate {}", args.candidate.display()))?;

    let display = DisplayOptions {
        format: args.display_format,
        columns,
        wrap: args.wrap,
        sections,
        metrics,
        section_line_limit: args.limit,
    };

    let opts = CompareOptions {
        group_by: args.group_by.into(),
        cgroup_flatten: args.cgroup_flatten.clone(),
        no_thread_normalize: args.no_thread_normalize,
        no_cg_normalize: args.no_cg_normalize,
        sort_by,
    };
    let diff = compare(&baseline, &candidate, &opts);
    print_diff(
        &diff,
        &args.baseline,
        &args.candidate,
        args.group_by,
        &display,
    );
    Ok(0)
}

/// Render the metric-list discovery output: a tag legend
/// (sched_class / config_gates / `[dead]`) followed by a per-metric
/// table whose rows show `name | tags | description`. Tag legend
/// is keyed off the closed-set vocabulary the registry pin test
/// guards (`registry_tag_vocabulary_is_closed`), so adding a new
/// allowed class or gate fails the test until both the legend
/// and the closed-set table are updated together.
///
/// Splits rendering from I/O so tests can drive the formatter
/// into a `String` buffer; the public [`run_metric_list`] entry
/// point is the print wrapper.
pub fn write_metric_list<W: fmt::Write>(w: &mut W) -> fmt::Result {
    writeln!(w, "## Tag legend")?;
    writeln!(w)?;
    writeln!(w, "sched_class:")?;
    writeln!(
        w,
        "  [cfs-only]    metric increments only inside CFS-class call paths (kernel/sched/fair.c);"
    )?;
    writeln!(w, "                zero under sched_ext / RT / DL / IDLE.")?;
    writeln!(
        w,
        "  [non-ext]     metric is written by the schedstat sleep/wait family wrappers"
    )?;
    writeln!(
        w,
        "                (kernel/sched/stats.c); CFS / RT / DL accumulate, sched_ext bypasses."
    )?;
    writeln!(
        w,
        "  [fair-policy] metric emits only when fair_policy(p->policy) is true:"
    )?;
    writeln!(
        w,
        "                SCHED_NORMAL, SCHED_BATCH, AND SCHED_EXT under CONFIG_SCHED_CLASS_EXT."
    )?;
    writeln!(w)?;
    writeln!(
        w,
        "config_gates (compact form; full kconfig symbol prefixed with CONFIG_):"
    )?;
    writeln!(
        w,
        "  [SCHED_INFO]            requires CONFIG_SCHED_INFO; gates the sched_info_* counters"
    )?;
    writeln!(
        w,
        "                          surfaced via /proc/<tid>/schedstat (run_time_ns, wait_time_ns,"
    )?;
    writeln!(w, "                          timeslices).")?;
    writeln!(
        w,
        "  [SCHEDSTATS]            requires CONFIG_SCHEDSTATS; gates every __schedstat_* /"
    )?;
    writeln!(
        w,
        "                          schedstat_* macro call (kernel/sched/stats.h:75-82)."
    )?;
    writeln!(
        w,
        "  [SCHED_CORE]            requires CONFIG_SCHED_CORE; gates the core-scheduling"
    )?;
    writeln!(
        w,
        "                          subsystem (core_forceidle_sum)."
    )?;
    writeln!(
        w,
        "  [SCHED_CLASS_EXT]       requires CONFIG_SCHED_CLASS_EXT; without it no task can"
    )?;
    writeln!(w, "                          land on the sched_ext class.")?;
    writeln!(
        w,
        "  [TASK_DELAY_ACCT]       requires CONFIG_TASK_DELAY_ACCT AND runtime delayacct=on"
    )?;
    writeln!(
        w,
        "                          (boot param or kernel.task_delayacct sysctl)."
    )?;
    writeln!(
        w,
        "  [TASK_IO_ACCOUNTING]    requires CONFIG_TASK_IO_ACCOUNTING; gates /proc/<tid>/io."
    )?;
    writeln!(
        w,
        "  [TASKSTATS]             requires CONFIG_TASKSTATS; gates the netlink TASKSTATS family"
    )?;
    writeln!(
        w,
        "                          (kernel/taskstats.c) used by the taskstats delay-accounting"
    )?;
    writeln!(
        w,
        "                          and hiwater_rss/hiwater_vm capture path. Calls also need"
    )?;
    writeln!(w, "                          CAP_NET_ADMIN.")?;
    writeln!(
        w,
        "  [TASK_XACCT]            requires CONFIG_TASK_XACCT; gates extended accounting fields"
    )?;
    writeln!(
        w,
        "                          (hiwater_rss, hiwater_vm) populated by xacct_add_tsk."
    )?;
    writeln!(w)?;
    writeln!(w, "status:")?;
    writeln!(
        w,
        "  [dead]        kernel exposes the counter via /proc but never increments it; always"
    )?;
    writeln!(
        w,
        "                reads zero. Surfaced for forward-compat parity with the kernel's"
    )?;
    writeln!(w, "                exposure surface.")?;
    writeln!(w)?;

    // Sections vocabulary table — discovery companion to the
    // `--sections` CLI flag. Lists every Section variant in
    // rendering order with its CLI name and a short description
    // of what it renders. Operators reading the rendered table
    // see `--sections primary,host-pressure` (or whatever) in
    // their compare/show invocation and need a way to learn
    // which sub-tables those tokens correspond to without
    // jumping to source. This section closes that loop.
    writeln!(w, "## Sections")?;
    writeln!(w)?;
    let mut sections_table = crate::cli::new_table();
    sections_table.set_header(vec!["section", "rendered heading", "description"]);
    for section in Section::ALL {
        let (heading, desc) = match section {
            Section::Primary => (
                "(no heading; first table)",
                "Per-thread metric table — the primary aggregated rows EXCLUDING the taskstats genetlink rows (those carry the `taskstats-delay` tag).",
            ),
            Section::TaskstatsDelay => (
                "(rendered inside the primary table)",
                "Taskstats genetlink-sourced rows — eight delay-accounting categories (cpu/blkio/swapin/freepages/thrashing/compact/wpcopy/irq × count/total/max/min) plus hiwater_rss_bytes / hiwater_vm_bytes. Per-row filter inside the primary table.",
            ),
            Section::Derived => (
                "## Derived metrics",
                "Computed metrics derived from the primary registry (ratios, averages, signed differences).",
            ),
            Section::CgroupStats => (
                "(no heading; cgroup-stats table)",
                "Per-cgroup CPU + memory enrichment from cpu.stat / memory.current. Requires --group-by cgroup.",
            ),
            Section::Limits => (
                "## Cgroup limits / knobs",
                "Operator-set cgroup configuration — cpu.max, cpu.weight, memory.max, memory.high, pids.*. Requires --group-by cgroup.",
            ),
            Section::MemoryStat => (
                "## memory.stat",
                "Kernel-emitted memory.stat counters per cgroup. Requires --group-by cgroup.",
            ),
            Section::MemoryEvents => (
                "## memory.events",
                "Pressure-event counters from memory.events per cgroup. Requires --group-by cgroup.",
            ),
            Section::Pressure => (
                "## Pressure / <resource>",
                "Per-cgroup PSI sub-tables — one per resource (cpu / memory / io / irq). Requires --group-by cgroup.",
            ),
            Section::HostPressure => (
                "## Host pressure / <resource>",
                "System-level PSI sub-tables from /proc/pressure/<resource>.",
            ),
            Section::Smaps => (
                "## smaps_rollup",
                "Per-process memory-mapping summary from /proc/<pid>/smaps_rollup (Rss / Pss / private / shared / swap). Compare-side keys default to per-pcomm-pattern aggregates (`worker-{N}`); pass `--no-thread-normalize` to switch back to literal `pcomm[tgid]` per-PID rows. Under default normalization, byte counts per (pcomm-pattern, key) pair are field-summed across all PIDs sharing the same pcomm skeleton.",
            ),
            Section::SchedExt => (
                "## sched_ext",
                "Global sched_ext sysfs state — state, switch_all, nr_rejected, hotplug_seq, enable_seq.",
            ),
        };
        sections_table.add_row(vec![
            section.cli_name().to_string(),
            heading.to_string(),
            desc.to_string(),
        ]);
    }
    writeln!(w, "{sections_table}")?;
    writeln!(w)?;

    writeln!(w, "## Metrics")?;
    writeln!(w)?;
    let mut table = crate::cli::new_table();
    table.set_header(vec!["metric", "tags", "description"]);
    for m in CTPROF_METRICS {
        // Strip the bare metric name off the rendered display
        // form so the `tags` column carries only the bracketed
        // suffixes — keeps the table scannable. When the metric
        // has no tags, the cell is empty.
        let tags = metric_tags(m);
        table.add_row(vec![m.name.to_string(), tags, m.description.to_string()]);
    }
    writeln!(w, "{table}")?;
    writeln!(w)?;
    writeln!(w, "## Derived metrics")?;
    writeln!(w)?;
    let mut dt = crate::cli::new_table();
    dt.set_header(vec!["metric", "unit", "inputs", "description"]);
    for d in CTPROF_DERIVED_METRICS {
        // Phase 4: ladder is the source of truth — `ratio` and
        // unit suffixes both fall out of `ScaleLadder::base_unit`
        // (with an explicit override for ratio rows where
        // is_ratio is true and the ladder is None).
        let unit_cell = if d.is_ratio {
            "ratio".to_string()
        } else {
            d.ladder.base_unit().to_string()
        };
        dt.add_row(vec![
            d.name.to_string(),
            unit_cell,
            d.inputs.join(", "),
            d.description.to_string(),
        ]);
    }
    writeln!(w, "{dt}")?;
    Ok(())
}

/// Print the metric-list discovery output to stdout. Thin
/// wrapper over [`write_metric_list`] so the CLI keeps the
/// one-line call ergonomics; tests drive the writer into a
/// `String` buffer.
pub fn print_metric_list() {
    let mut out = String::new();
    // Infallible: writing into a String cannot fail.
    let _ = write_metric_list(&mut out);
    print!("{out}");
}

/// Entry point for the `ctprof metric-list` subcommand.
/// Always returns `Ok(0)` — discovery output is informational
/// and never fails.
pub fn run_metric_list() -> anyhow::Result<i32> {
    print_metric_list();
    Ok(0)
}

/// Render [`CtprofDiff`] as a table on stdout. Thin wrapper
/// over [`write_diff`] so the non-test caller keeps the
/// ergonomics of a one-line call; tests drive [`write_diff`]
/// into a `String` buffer.
pub fn print_diff(
    diff: &CtprofDiff,
    baseline_path: &Path,
    candidate_path: &Path,
    group_by: GroupBy,
    display: &DisplayOptions,
) {
    let mut out = String::new();
    // Infallible: writing into a String cannot fail.
    let _ = write_diff(
        &mut out,
        diff,
        baseline_path,
        candidate_path,
        group_by,
        display,
    );
    if display.section_line_limit > 0 {
        print!("{}", limit_sections(&out, display.section_line_limit));
    } else {
        print!("{out}");
    }
}

/// Truncate each `## <heading>` section to at most `limit` lines.
/// Sections are delimited by lines starting with `## `. Content
/// before the first section header passes through untruncated
/// (typically the file-path header row).
pub fn limit_sections(output: &str, limit: usize) -> String {
    let mut result = String::with_capacity(output.len());
    let mut section_lines: Vec<&str> = Vec::new();
    let mut section_header: Option<&str> = None;

    for line in output.lines() {
        if line.starts_with("## ") {
            flush_section(&mut result, section_header, &section_lines, limit);
            section_lines.clear();
            section_header = Some(line);
        } else if section_header.is_some() {
            section_lines.push(line);
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }
    flush_section(&mut result, section_header, &section_lines, limit);
    result
}

fn flush_section(result: &mut String, header: Option<&str>, lines: &[&str], limit: usize) {
    let Some(header) = header else { return };
    result.push_str(header);
    result.push('\n');
    if lines.len() <= limit {
        for line in lines {
            result.push_str(line);
            result.push('\n');
        }
    } else {
        for line in &lines[..limit] {
            result.push_str(line);
            result.push('\n');
        }
        result.push_str(&format!(
            "... {} more lines truncated (use --limit 0 for unlimited)\n",
            lines.len() - limit,
        ));
    }
}

impl DisplayOptions {
    /// Resolved compare-side column set: `columns` if
    /// non-empty, otherwise [`compare_columns_for`] over
    /// `format`. `--columns` always wins over the format
    /// shorthand (explicit > shorthand) per the design call.
    pub fn resolved_compare_columns(&self) -> Vec<Column> {
        if self.columns.is_empty() {
            compare_columns_for(self.format)
        } else {
            self.columns.clone()
        }
    }

    /// Resolved show-side column set: `columns` if non-empty,
    /// otherwise [`show_columns_default`].
    pub fn resolved_show_columns(&self) -> Vec<Column> {
        if self.columns.is_empty() {
            show_columns_default()
        } else {
            self.columns.clone()
        }
    }

    /// Returns `true` when `section` should render under the
    /// current filter. Empty `sections` means "every section
    /// renders" (the default — no filter applied), matching
    /// [`parse_sections`]'s empty-input semantic. Non-empty
    /// `sections` restricts rendering to the named entries.
    pub fn is_section_enabled(&self, section: Section) -> bool {
        self.sections.is_empty() || self.sections.contains(&section)
    }

    /// Returns `true` when the metric named `name` should
    /// render under the current row-level filter. Empty
    /// `metrics` means "every metric renders" — the
    /// unfiltered default mirroring
    /// [`Self::is_section_enabled`]. Non-empty restricts
    /// rendering to the listed names. The comparison is on
    /// the metric's `&'static str` name (so a registry-name
    /// pointer or any byte-equal string both match).
    pub fn is_metric_enabled(&self, name: &str) -> bool {
        self.metrics.is_empty() || self.metrics.contains(&name)
    }

    /// Construct a comfy-table builder honouring the
    /// [`wrap`](Self::wrap) flag: terminal-width-aware
    /// `Dynamic` arrangement when `wrap` is true, otherwise the
    /// existing borderless, disabled-arrangement layout via
    /// [`crate::cli::new_table`]. Single source of truth so
    /// every section in [`write_diff`] honours `--wrap` without
    /// per-call-site `if` branching. The show-side renderer
    /// (`write_show` in `src/bin/ktstr.rs`) calls the underlying
    /// helpers directly through the same branch.
    pub fn new_table(&self) -> comfy_table::Table {
        if self.wrap {
            crate::cli::new_wrapped_table()
        } else {
            crate::cli::new_table()
        }
    }

    /// Create a table constrained to the given max content widths.
    /// Heading rows wider than data get auto-truncated by comfy_table
    /// with its built-in "..." indicator.
    pub fn new_constrained_table(&self, max_widths: &[u16]) -> comfy_table::Table {
        let mut t = self.new_table();
        // Create dummy columns so constraints can be set.
        // Columns are auto-created when the header is added later,
        // but we need them NOW for set_constraint. Adding a dummy
        // header row with the right column count, then replacing
        // it when the real header is set.
        let dummy: Vec<&str> = (0..max_widths.len()).map(|_| "").collect();
        t.set_header(dummy);
        for (i, &w) in max_widths.iter().enumerate() {
            if let Some(col) = t.column_mut(i) {
                col.set_constraint(comfy_table::ColumnConstraint::UpperBoundary(
                    comfy_table::Width::Fixed(w),
                ));
            }
        }
        t
    }
}
