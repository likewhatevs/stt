//! Per-cell and per-row rendering helpers for the compare output.
//!
//! This module is the formatting layer between
//! [`crate::ctprof_compare::CtprofDiff`] and
//! [`comfy_table::Table`]: every cell that the diff emitter
//! writes flows through one of the helpers here. Splits cleanly
//! into three responsibilities:
//!
//! - **Row builders** — [`render_diff_row_cells`] /
//!   [`render_derived_row_cells`] take a single [`DiffRow`] /
//!   [`DerivedRow`] plus the resolved [`Column`] set and emit
//!   the column-by-column `Vec<String>` that the table adapter
//!   consumes.
//! - **Cell formatters** — [`format_arrow_cell`],
//!   [`cgroup_cell`], [`format_psi_avg_cell`] and friends render
//!   `baseline → candidate (delta)` triples for the various
//!   units (raw `Aggregated`, raw `u64` cgroup counters,
//!   centi-percent PSI averages).
//! - **Color / chrome** — [`color_diff_cell`],
//!   [`color_derived_cells`], [`colored_header`] /
//!   [`colored_header_with_sort`] wrap rendered strings in
//!   [`comfy_table::Cell`]s with the foreground color encoding
//!   regression vs improvement (yellow / magenta), uptime
//!   gradient (green / yellow / red), or the cyan header / blue
//!   derived-row palette.
//!
//! All cell formatters delegate scalar formatting to
//! [`super::scale`] (auto-scale ladders) and metric-name lookup
//! to [`super::metrics`]; the comfy-table builder lives in
//! [`super::runner::DisplayOptions`].

use super::aggregate::Aggregated;
use super::columns::Column;
use super::diff_types::{DerivedRow, DiffRow};
use super::scale::{
    ScaleLadder, format_delta_cell, format_derived_delta_cell, format_derived_value_cell,
    format_scaled_u64, format_value_cell,
};
use super::{CTPROF_METRICS, metric_display_name, metric_tags};
use crate::ctprof::{Psi, PsiHalf, PsiResource};

/// Format the arrow cell `<baseline> -> <candidate> (<delta>)`
/// for primary diff rows. Mirrors [`cgroup_cell`]'s shape so
/// the visual style stays consistent across primary and cgroup
/// tables. When `delta` is `None` (categorical Mode rule), the
/// parenthetical drops to "same"/"differs"; for non-Mode rows
/// without a numeric projection the parenthetical is "-".
pub(super) fn format_arrow_cell(
    baseline: &Aggregated,
    candidate: &Aggregated,
    delta: Option<f64>,
    ladder: ScaleLadder,
) -> String {
    let baseline_cell = format_value_cell(baseline, ladder);
    let candidate_cell = format_value_cell(candidate, ladder);
    let _ = delta;
    format!("{baseline_cell} \u{2192} {candidate_cell}")
}

/// Format the arrow cell for derived rows. Same shape as
/// [`format_arrow_cell`] but pulls from typed
/// [`super::DerivedValue`]s and routes through the derived
/// formatters so ratios / ns / B units pick up their
/// auto-scale ladders.
pub(super) fn format_arrow_cell_derived(row: &DerivedRow) -> String {
    let baseline_cell = match row.baseline {
        Some(v) => format_derived_value_cell(v, row.metric_ladder, row.is_ratio),
        None => "-".to_string(),
    };
    let candidate_cell = match row.candidate {
        Some(v) => format_derived_value_cell(v, row.metric_ladder, row.is_ratio),
        None => "-".to_string(),
    };
    format!("{baseline_cell} \u{2192} {candidate_cell}")
}

/// Helper: shared threads-cell rendering — the `N` form when
/// counts match across snapshots, `A->B` arrow form otherwise.
pub(super) fn render_threads_cell(a: usize, b: usize) -> String {
    if a == b {
        a.to_string()
    } else {
        format!("{}\u{2192}{}", a, b)
    }
}

/// Render a [`DiffRow`] into the column-by-column cell vector
/// per the resolved [`Column`] set. Caller emits the resulting
/// `Vec<String>` straight into a comfy_table row.
pub(super) fn render_diff_row_cells(row: &DiffRow, columns: &[Column]) -> Vec<String> {
    let metric_def = CTPROF_METRICS
        .iter()
        .find(|m| m.name == row.metric_name)
        .expect("metric_name comes from CTPROF_METRICS via build_row");
    let metric_cell = metric_display_name(metric_def).to_string();
    let mut cells = Vec::with_capacity(columns.len());
    for col in columns {
        let cell = match col {
            Column::Group => row.display_key.clone(),
            Column::Threads => render_threads_cell(row.thread_count_a, row.thread_count_b),
            Column::Metric => metric_cell.clone(),
            Column::Baseline => format_value_cell(&row.baseline, row.metric_ladder),
            Column::Candidate => format_value_cell(&row.candidate, row.metric_ladder),
            Column::Delta => match row.delta {
                Some(d) => format_delta_cell(d, row.metric_ladder),
                None => match (&row.baseline, &row.candidate) {
                    (Aggregated::Mode { .. }, Aggregated::Mode { .. }) => {
                        if row.baseline.mode_value() == row.candidate.mode_value() {
                            "same".to_string()
                        } else {
                            "differs".to_string()
                        }
                    }
                    _ => "-".to_string(),
                },
            },
            Column::Pct => match row.delta_pct {
                Some(p) => format!("{:+.1}%", p * 100.0),
                None => "-".to_string(),
            },
            Column::Arrow => {
                format_arrow_cell(&row.baseline, &row.candidate, row.delta, row.metric_ladder)
            }
            // Show-only columns. The compare-side parse_columns
            // gate rejects Value at CLI parse time, so reaching
            // this arm requires constructing a column set
            // directly through the Rust API. Surface a `-`
            // rather than panic.
            Column::Value => "-".to_string(),
            Column::Tags => metric_tags(metric_def),
            Column::Uptime => match row.uptime_pct {
                Some(pct) => format!("{pct:.0}%"),
                None => "-".to_string(),
            },
            Column::SortBy => row.sort_by_cell.clone().unwrap_or_else(|| "-".to_string()),
        };
        cells.push(cell);
    }
    cells
}

/// Color a diff-table cell based on its column type, the row's
/// raw delta (sign of color), and the row's delta_pct (fraction
/// for the bold threshold). Delta/% cells: yellow for positive
/// (increase), magenta for negative (decrease). Uptime:
/// green/yellow/red gradient. Other columns: default.
///
/// `delta` carries the raw metric delta — used only for color
/// sign (positive vs negative). `delta_pct` carries the
/// fractional delta (Δ / baseline) — used as the bold threshold
/// on the Pct column. Without the split, the bold check
/// |raw_delta| > 0.5 fired on every non-zero ns/byte change
/// (ns and bytes are large; 0.5 trivially exceeded).
pub fn color_diff_cell(
    text: String,
    col: Column,
    delta: Option<f64>,
    delta_pct: Option<f64>,
    uptime_pct: Option<f64>,
    sort_by_delta: Option<f64>,
) -> comfy_table::Cell {
    use comfy_table::{Attribute, Color};
    match col {
        Column::Pct => {
            let color = match delta {
                Some(d) if d > 0.0 => Color::Yellow,
                Some(d) if d < 0.0 => Color::Magenta,
                _ => Color::White,
            };
            let mut cell = comfy_table::Cell::new(text).fg(color);
            if matches!(delta_pct, Some(p) if p.abs() > 0.5) {
                cell = cell.add_attribute(Attribute::Bold);
            }
            cell
        }
        Column::Delta => {
            let color = match delta {
                Some(d) if d > 0.0 => Color::Yellow,
                Some(d) if d < 0.0 => Color::Magenta,
                _ => Color::White,
            };
            comfy_table::Cell::new(text).fg(color)
        }
        Column::Uptime => {
            let color = match uptime_pct {
                Some(p) if p >= 75.0 => Color::Green,
                Some(p) if p >= 50.0 => Color::Yellow,
                Some(_) => Color::Red,
                None => Color::White,
            };
            let mut cell = comfy_table::Cell::new(text).fg(color);
            if matches!(uptime_pct, Some(p) if p < 50.0) {
                cell = cell.add_attribute(Attribute::Bold);
            }
            cell
        }
        Column::SortBy => {
            let color = match sort_by_delta {
                Some(d) if d > 0.0 => Color::Yellow,
                Some(d) if d < 0.0 => Color::Magenta,
                _ => Color::Cyan,
            };
            comfy_table::Cell::new(text).fg(color)
        }
        _ => comfy_table::Cell::new(text),
    }
}

/// Extract the parent directory and leaf segment of a cgroup path.
/// `/system.slice/foo.service` → (`/system.slice`, `foo.service`).
/// `/` → (`/`, `/`). Empty → (``, ``).
pub(super) fn cgroup_parent_leaf(path: &str) -> (&str, &str) {
    match path.rfind('/') {
        Some(0) => ("/", &path[1..]),
        Some(i) => (&path[..i], &path[i + 1..]),
        None => ("", path),
    }
}

/// Build a colored header row — cyan foreground so headers are
/// visually distinct from data rows.
pub fn colored_header(columns: &[Column], group_header: &'static str) -> Vec<comfy_table::Cell> {
    colored_header_with_sort(columns, group_header, None)
}

pub fn colored_header_with_sort(
    columns: &[Column],
    group_header: &'static str,
    sort_metric: Option<&str>,
) -> Vec<comfy_table::Cell> {
    columns
        .iter()
        .map(|c| {
            let label = if *c == Column::SortBy {
                sort_metric.unwrap_or("sort-by")
            } else {
                c.header(group_header)
            };
            comfy_table::Cell::new(label).fg(comfy_table::Color::Cyan)
        })
        .collect()
}

/// Wrap a string-cell row in [`comfy_table::Cell`]s with blue
/// foreground so derived-metric rows render visually distinct
/// from the per-thread primary table when stdout is a TTY.
/// Operators scanning a long compare or show output can locate
/// the `## Derived metrics` rows at a glance instead of relying
/// on the section header alone.
///
/// On a non-TTY stdout the comfy-table builder calls
/// [`comfy_table::Table::force_no_tty`] (see
/// [`crate::cli::new_table`]) which strips the ANSI escape
/// sequences; the rendered output is byte-identical to the
/// pre-color baseline for shell-pipeline consumers.
///
/// Color choice: blue contrasts with both the unstyled primary
/// table and the stats compare verdict palette
/// (`Color::Red` / `Color::Green` for REGRESSION /
/// improvement) — derived rows do not carry a regression
/// verdict of their own, so reusing the verdict colors here
/// would conflict with the established convention.
pub fn color_derived_cells(cells: Vec<String>) -> Vec<comfy_table::Cell> {
    cells
        .into_iter()
        .map(|c| comfy_table::Cell::new(c).fg(comfy_table::Color::Blue))
        .collect()
}

/// Render a [`DerivedRow`] into the column-by-column cell
/// vector. Mirrors [`render_diff_row_cells`] but routes
/// numeric cells through the typed-derived formatters.
pub(super) fn render_derived_row_cells(row: &DerivedRow, columns: &[Column]) -> Vec<String> {
    let mut cells = Vec::with_capacity(columns.len());
    for col in columns {
        let cell = match col {
            Column::Group => row.display_key.clone(),
            Column::Threads => render_threads_cell(row.thread_count_a, row.thread_count_b),
            Column::Metric => row.metric_name.to_string(),
            Column::Baseline => match row.baseline {
                Some(v) => format_derived_value_cell(v, row.metric_ladder, row.is_ratio),
                None => "-".to_string(),
            },
            Column::Candidate => match row.candidate {
                Some(v) => format_derived_value_cell(v, row.metric_ladder, row.is_ratio),
                None => "-".to_string(),
            },
            Column::Delta => match row.delta {
                Some(d) => format_derived_delta_cell(d, row.metric_ladder, row.is_ratio),
                None => "-".to_string(),
            },
            Column::Pct => match row.delta_pct {
                Some(p) => format!("{:+.1}%", p * 100.0),
                None => "-".to_string(),
            },
            Column::Arrow => format_arrow_cell_derived(row),
            Column::Value => "-".to_string(),
            Column::Tags => String::new(),
            Column::Uptime => "-".to_string(),
            Column::SortBy => row.sort_by_cell.clone().unwrap_or_else(|| "-".to_string()),
        };
        cells.push(cell);
    }
    cells
}

/// Render a `(baseline, candidate, delta)` cell for the
/// cgroup-enrichment secondary table emitted under
/// [`super::GroupBy::Cgroup`]. The `ladder` parameter routes
/// each scalar through `auto_scale` (private to this module) so
/// a 7.5 GiB `memory_current` row reads
/// `7.500GiB → 8.250GiB (+768.000MiB)` instead of
/// `8053063680 → 8858370048 (+805306368)`. Each cell scales
/// independently — baseline, candidate, and delta may pick
/// different prefixes when their magnitudes cross thresholds.
///
/// See [`ScaleLadder`] for the closed enumeration of supported
/// ladder families and per-variant step-up rules. The variants
/// most relevant to cgroup-render call sites:
/// - [`ScaleLadder::Us`]: cgroup `cpu_usage_usec` /
///   `throttled_usec` / PSI `total_usec`.
/// - [`ScaleLadder::Bytes`]: `memory_current` / `memory.max` /
///   `memory.high` (IEC binary, B → KiB → MiB → GiB → TiB).
/// - [`ScaleLadder::Unitless`]: `nr_throttled` / `cpu.weight` /
///   `pids.current` / sched_ext attribute counters (decimal
///   SI, "" → K → M → G).
pub fn cgroup_cell(baseline: Option<u64>, candidate: Option<u64>, ladder: ScaleLadder) -> String {
    match (baseline, candidate) {
        (Some(baseline), Some(candidate)) => {
            let baseline_cell = format_scaled_u64(baseline, ladder);
            let candidate_cell = format_scaled_u64(candidate, ladder);
            let d = candidate as i128 - baseline as i128;
            // Delta is signed; route via format_delta_cell so the
            // sign is rendered explicitly and the auto-scale step
            // applies. i128 → f64 cast is lossy at extreme
            // magnitudes (>2^53) but cgroup counters on typical
            // hosts stay well under that ceiling.
            let delta_cell = format_delta_cell(d as f64, ladder);
            format!("{baseline_cell} → {candidate_cell} ({delta_cell})")
        }
        (Some(baseline), None) => format!("{} → -", format_scaled_u64(baseline, ladder)),
        (None, Some(candidate)) => format!("- → {}", format_scaled_u64(candidate, ladder)),
        (None, None) => "-".to_string(),
    }
}

/// Render a baseline→candidate→delta cell for a PSI average
/// field. `baseline` and `candidate` are centi-percent (0..=10000
/// covering 0.00..=100.00 %); the cell renders each as `N.NN%`
/// and computes a signed delta `(+|-D.DD%)`. Mirrors
/// [`cgroup_cell`]'s structure but does NOT route through the
/// auto-scale ladder — a pressure percentage is dimensionless
/// and topping out at 100 means there's nothing to scale.
pub fn format_psi_avg_cell(baseline: Option<u16>, candidate: Option<u16>) -> String {
    match (baseline, candidate) {
        (Some(b), Some(c)) => {
            let baseline_cell = format_psi_avg_centi_percent(b);
            let candidate_cell = format_psi_avg_centi_percent(c);
            let d = c as i32 - b as i32;
            let sign = if d >= 0 { "+" } else { "-" };
            let abs = d.unsigned_abs();
            let delta_int = abs / 100;
            let delta_frac = abs % 100;
            format!("{baseline_cell} → {candidate_cell} ({sign}{delta_int}.{delta_frac:02}%)")
        }
        (Some(b), None) => format!("{} → -", format_psi_avg_centi_percent(b)),
        (None, Some(c)) => format!("- → {}", format_psi_avg_centi_percent(c)),
        (None, None) => "-".to_string(),
    }
}

/// Convert a centi-percent value (0..=10000) to its display
/// form `N.NN%`. The centi-percent representation is 1:1 with
/// the kernel's `LOAD_INT.LOAD_FRAC` 2-decimal-digit emission at
/// `kernel/sched/psi.c:1284` — preserve that precision on
/// display.
pub fn format_psi_avg_centi_percent(v: u16) -> String {
    let int = v / 100;
    let frac = v % 100;
    format!("{int}.{frac:02}%")
}

/// One entry in the [`psi_resource_accessors`] table — a
/// display name paired with the accessor that pulls one
/// [`PsiResource`] out of a [`Psi`] bundle.
type PsiAccessor = (&'static str, fn(&Psi) -> PsiResource);

/// Returns the four PSI resource accessors paired with their
/// display names. Single source of truth for compare-side
/// rendering — adding a fifth resource means one edit here.
pub(super) fn psi_resource_accessors() -> [PsiAccessor; 4] {
    [
        ("cpu", |p| p.cpu),
        ("memory", |p| p.memory),
        ("io", |p| p.io),
        ("irq", |p| p.irq),
    ]
}

/// Returns true when either side of a [`Psi`] pair has any
/// non-zero data. Used to suppress a host-pressure or
/// per-cgroup-pressure section when both sides are flat zero.
pub(super) fn psi_pair_has_data(a: &Psi, b: &Psi) -> bool {
    psi_has_data(a) || psi_has_data(b)
}

pub(super) fn psi_has_data(p: &Psi) -> bool {
    [p.cpu, p.memory, p.io, p.irq]
        .iter()
        .any(psi_resource_has_data)
}

pub(super) fn psi_resource_has_data(r: &PsiResource) -> bool {
    let h = |h: &PsiHalf| h.avg10 != 0 || h.avg60 != 0 || h.avg300 != 0 || h.total_usec != 0;
    h(&r.some) || h(&r.full)
}
