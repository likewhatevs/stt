//! Group, aggregate, and render the comparison between two
//! [`CtprofSnapshot`]s.
//!
//! Design summary: the per-thread profiler emits
//! one snapshot per run. Comparison groups threads within each
//! snapshot by a single axis (pcomm, cgroup, comm, or
//! comm-exact — see [`GroupBy`]), aggregates every metric per
//! the rule on its [`CtprofMetricDef`], then matches groups
//! across the two snapshots and emits one row per
//! `(group, metric)` pair. Groups present on only one side
//! surface as unmatched entries rather than imaginary
//! zero-valued rows — a row is missing because the process did
//! not exist, not because it did zero work.
//!
//! No judgment labels. The comparison prints raw numbers and
//! percent delta; interpretation (regression vs improvement) is
//! scheduler-specific and left to the user. This mirrors the
//! no-label principle for the broader stats comparison pipeline
//! (see the `stats.rs` module doc).

mod pattern;
pub use pattern::{pattern_display_label, pattern_key};

mod scale;
pub use scale::{
    ScaleLadder, cgroup_limits_cell, cgroup_optional_limit_cell, format_cpu_max,
    format_derived_delta_cell, format_derived_value_cell, format_optional_limit, format_scaled_u64,
    format_value_cell,
};

mod options;
pub use options::{AggRule, CompareOptions, GroupBy, GroupByOrDefault, SortKey};

mod metrics;
pub use metrics::{
    CTPROF_DERIVED_METRICS, CTPROF_METRICS, CtprofMetricDef, DerivedMetricDef, DerivedValue,
    metric_display_name, metric_tags,
};

mod aggregate;
pub use aggregate::{AffinitySummary, Aggregated};

mod diff_types;
pub use diff_types::{CtprofDiff, DerivedRow, DiffRow, FudgedPair, ThreadGroup};

mod cgroup_merge;

mod groups;
pub use groups::{
    aggregate, build_cgroup_key_map, build_groups, collect_smaps_rollup,
    collect_smaps_rollup_hierarchical, compile_flatten_patterns, flatten_cgroup_path,
};

mod compare;
pub use compare::{compare, flatten_cgroup_stats};

mod columns;
pub use columns::{
    Column, DisplayFormat, Section, parse_columns, parse_metrics, parse_sections,
    warn_cgroup_only_sections_under_non_cgroup,
};
mod runner;
pub use runner::{
    CtprofCompareArgs, DisplayOptions, limit_sections, parse_sort_by, print_diff,
    print_metric_list, run_compare, run_metric_list, write_metric_list,
};

mod render;
pub use render::{
    cgroup_cell, color_derived_cells, color_diff_cell, colored_header, colored_header_with_sort,
    format_psi_avg_cell, format_psi_avg_centi_percent,
};

mod report;
pub use report::write_diff;

#[cfg(test)]
mod tests_fixtures;
#[cfg(test)]
mod tests_aggregate;
#[cfg(test)]
mod tests_cgroup_merge;
#[cfg(test)]
mod tests_columns;
#[cfg(test)]
mod tests_compare;
#[cfg(test)]
mod tests_diff_types;
#[cfg(test)]
mod tests_groups;
#[cfg(test)]
mod tests_metrics;
#[cfg(test)]
mod tests_metrics2;
#[cfg(test)]
mod tests_options;
#[cfg(test)]
mod tests_pattern;
#[cfg(test)]
mod tests_render;
#[cfg(test)]
mod tests_report;
#[cfg(test)]
mod tests_runner;
#[cfg(test)]
mod tests_scale;
