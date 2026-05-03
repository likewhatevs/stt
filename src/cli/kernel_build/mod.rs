//! Kernel build pipeline: configure, validate, build, cache-store.
//!
//! Split into three submodules:
//! - [`make`] — `make` subprocess wrappers ([`run_make`],
//!   [`run_make_with_output`], [`make_kernel_with_output`]) plus
//!   the byte-oriented line drain and timeout-poll loop they share.
//! - [`kconfig`] — fragment merging ([`configure_kernel`]),
//!   `--extra-kconfig` parsing ([`read_extra_kconfig`],
//!   [`append_extra_kconfig_suffix`]), pre/post warning passes
//!   over user fragments, and the post-build critical-options
//!   check ([`validate_kernel_config`], [`has_sched_ext`]).
//! - [`build`] — top-level orchestrator ([`kernel_build_pipeline`])
//!   and its two-phase reservation acquisition
//!   ([`acquire_build_reservation`], [`acquire_source_tree_lock`]).

mod build;
mod kconfig;
mod make;

pub use build::{KernelBuildResult, kernel_build_pipeline};
pub use kconfig::{
    append_extra_kconfig_suffix, configure_kernel, has_sched_ext, read_extra_kconfig,
    validate_kernel_config,
};
pub use make::{make_kernel_with_output, run_make, run_make_with_output};
