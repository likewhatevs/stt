//! `stats` subcommand surface.
//!
//! Split into two submodules:
//! - [`dispatch`] — thin wrappers over [`crate::stats`] +
//!   host-context render entry points (list_runs, list_metrics,
//!   list_values, compare_partitions, show_host, show_run_host,
//!   show_thresholds, print_stats_report) and the per-test /
//!   per-run-key fuzzy-match helpers.
//! - [`explain_sidecar`] — the per-sidecar `Option`-field absence
//!   diagnostic ([`explain_sidecar`]) with its static catalog,
//!   walk-stats helpers, and JSON / text renderers.

mod dispatch;
mod explain_sidecar;

pub use dispatch::{
    compare_partitions, list_metrics, list_runs, list_values, print_stats_report, show_host,
    show_run_host, show_thresholds,
};
pub use explain_sidecar::explain_sidecar;
