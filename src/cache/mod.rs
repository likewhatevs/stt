//! Kernel image cache for ktstr.
//!
//! Manages a local cache of built kernel images under an XDG-compliant
//! directory. Each cached kernel is a directory containing the boot
//! image, optionally a stripped vmlinux ELF (symbol table, BTF, and
//! the section headers that monitor/probe code reads), and a
//! `metadata.json` descriptor. `CONFIG_HZ` is recovered from the
//! embedded IKCONFIG blob in the stripped vmlinux (ktstr.kconfig
//! forces `CONFIG_IKCONFIG=y`), so no separate `.config` sidecar is
//! cached.
//!
//! # Cache location
//!
//! Resolved in order:
//! 1. `KTSTR_CACHE_DIR` environment variable
//! 2. `$XDG_CACHE_HOME/ktstr/kernels/`
//! 3. `$HOME/.cache/ktstr/kernels/`
//!
//! # Submodule layout
//!
//! - [`metadata`] — public types: [`KernelSource`], [`KernelMetadata`],
//!   [`CacheArtifacts`], [`KconfigStatus`], [`CacheEntry`],
//!   [`ListedEntry`], plus the internal `classify_corrupt_reason`
//!   dispatcher.
//! - [`cache_dir`] — [`CacheDir`] handle, lock guards
//!   ([`SharedLockGuard`], [`ExclusiveLockGuard`]), store/lookup/list/
//!   clean lifecycle, and reader/writer-asymmetric lock policy.
//! - [`housekeeping`] — atomic-rename install primitives, cache-key
//!   and image-name validators, `read_metadata` decoder, and the
//!   `clean_orphaned_tmp_dirs` cross-PID sweep.
//! - [`vmlinux_strip`] — ELF strip pipeline ([`strip_vmlinux_debug`],
//!   `neutralize_relocs`, `strip_keep_list`, `strip_debug_prefix`)
//!   plus the keep-list / zero-data section-name unions.
//! - [`resolve`] — env-cascade root resolution
//!   (`resolve_cache_root_with_suffix`, `validate_home_for_cache`,
//!   [`path_inside_cache_root`]) and source-tree path helpers
//!   ([`prefer_source_tree_for_dwarf`], [`recover_local_source_tree`]).
//!
//! Each submodule owns its tests (collocated under the same file in
//! a `#[cfg(test)] mod tests` block); shared test fixtures used by
//! more than one submodule's tests live in
//! [`shared_test_helpers`].

use crate::flock::LOCK_DIR_NAME;

mod cache_dir;
mod housekeeping;
mod metadata;
mod resolve;
mod vmlinux_strip;

#[cfg(test)]
pub(crate) mod shared_test_helpers;

// Public API re-exports — preserve every `crate::cache::*` path that
// external callers (lib.rs, cli.rs, fetch.rs, monitor/*, probe/btf.rs,
// vmm/disk_template.rs, test_support/*, remote_cache.rs, stats.rs,
// flock.rs) rely on.

pub use cache_dir::{CacheDir, ExclusiveLockGuard, SharedLockGuard};
pub use metadata::{
    CacheArtifacts, CacheEntry, KconfigStatus, KernelMetadata, KernelSource, ListedEntry,
};
pub use resolve::{prefer_source_tree_for_dwarf, recover_local_source_tree};

// Re-export KernelId from kernel_path (canonical definition, std-only).
pub use crate::kernel_path::KernelId;

// Crate-internal API re-exports for callers in other modules
// (test_support/model, vmm/disk_template, monitor/btf_offsets).
pub(crate) use resolve::{path_inside_cache_root, resolve_cache_root_with_suffix, resolve_lock_dir};
// Re-exported for rustdoc cross-link resolution (cache/mod.rs:31's
// `[`strip_vmlinux_debug`]` link). No `crate::cache::strip_vmlinux_debug`
// code call sites today; intra-cache callers (cache_dir.rs, tests)
// reach the function via `super::vmlinux_strip::strip_vmlinux_debug`.
#[allow(unused_imports)]
pub(crate) use vmlinux_strip::strip_vmlinux_debug;

/// Filename prefix that marks an in-progress atomic-store directory
/// under the cache root. Format: `{TMP_DIR_PREFIX}{cache_key}-{pid}`.
/// Centralized here so the four roles — emitter
/// ([`cache_dir::CacheDir::store`]), scanner
/// ([`housekeeping::clean_orphaned_tmp_dirs`]), validator
/// ([`housekeeping::validate_cache_key`]), and listing filter
/// ([`cache_dir::CacheDir::list`]'s skip-tmp-dirs branch) —
/// cannot drift.
pub(crate) const TMP_DIR_PREFIX: &str = ".tmp-";
