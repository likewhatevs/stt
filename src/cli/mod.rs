//! CLI support functions shared between `ktstr` and `cargo-ktstr`.
//!
//! Validation, configuration, and kernel/KVM resolution logic used
//! by both binaries.

mod kernel_build;
mod kernel_cmd;
mod kernel_list;
mod locks;
mod parse;
mod resolve;
mod stats_cmds;
mod util;

#[cfg(test)]
mod testing;

pub use kernel_cmd::{
    CPU_CAP_HELP, DIRTY_TREE_CACHE_SKIP_HINT, EMBEDDED_KCONFIG, EOL_EXPLANATION,
    EXTRA_KCONFIG_HELP, KERNEL_HELP_NO_RAW, KERNEL_HELP_RAW_OK, KERNEL_LIST_LONG_ABOUT,
    KernelCommand, NON_GIT_TREE_CACHE_SKIP_HINT, STALE_KCONFIG_EXPLANATION,
    UNTRACKED_KCONFIG_EXPLANATION, embedded_kconfig_hash,
};

pub use kernel_list::{format_entry_row, kernel_clean, kernel_list, kernel_list_range_preview};

pub use kernel_build::{
    KernelBuildResult, append_extra_kconfig_suffix, configure_kernel, has_sched_ext,
    kernel_build_pipeline, make_kernel_with_output, read_extra_kconfig, run_make,
    run_make_with_output, validate_kernel_config,
};

pub use parse::{DISK_HELP, parse_disk_arg, parse_disk_size_mib, parse_topology_string};

pub use resolve::{
    KernelDirCacheHit, KernelDirOutcome, KernelResolvePolicy, auto_download_kernel, cache_lookup,
    check_kvm, download_and_cache_version, expand_kernel_range, resolve_cached_kernel,
    resolve_git_kernel, resolve_include_files, resolve_kernel_dir, resolve_kernel_dir_to_entry,
    resolve_kernel_image, resolve_kernel_parallelism,
};

pub use stats_cmds::{
    compare_partitions, explain_sidecar, list_metrics, list_runs, list_values, print_stats_report,
    show_host, show_run_host, show_thresholds,
};

pub use locks::list_locks;

pub use util::{
    Spinner, new_table, new_wrapped_table, restore_sigpipe_default, stderr_color, stdout_color,
};

/// Re-export of the internal `vmm::host_topology::CpuCap` type so
/// the `ktstr` and `cargo-ktstr` CLI binaries (which import this
/// module through the `pub mod cli` surface) can resolve
/// `--cpu-cap N` without depending on the `pub(crate)` `vmm`
/// module. Keeping the canonical definition in `vmm::host_topology`
/// (so the `acquire_llc_plan` internal call site consumes its own
/// type without needing `cli`) and re-exporting here — versus
/// inverting the dependency — avoids pulling the CLI module into
/// the VMM internals.
pub use crate::vmm::host_topology::CpuCap;

/// Re-exports of the dimensional-slicing types used by
/// `cargo-ktstr`'s `BuildCompareFilters::build()` plumbing. The
/// `stats` module is `pub(crate)` (its tabular reporting types
/// have no stable surface yet), but the `cargo-ktstr` binary needs
/// `Dimension` and `derive_slicing_dims` to construct compare
/// requests and to unit-test the filter-builder shape. Same
/// pattern as `CpuCap` above: keep the canonical definitions in
/// `stats` (where the comparison plumbing consumes them
/// internally) and re-export the slim slicing surface through
/// `cli` so the binaries reach them through the public `cli`
/// module.
pub use crate::stats::{Dimension, derive_slicing_dims};

/// Re-export of the comparison-policy types so downstream crates
/// using `ktstr::cli` as their public surface don't need to reach
/// into the internal `ktstr::stats` module (which is `pub(crate)` —
/// see `lib.rs` — and therefore not a stable public path). The
/// policy is the only item in `stats` that a CLI or external
/// consumer constructs directly; every other item is internal
/// plumbing reached via `cli::compare_partitions`.
pub use crate::stats::{AveragedGroup, ComparisonPolicy, RowFilter};
