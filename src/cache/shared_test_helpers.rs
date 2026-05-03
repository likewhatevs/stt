//! Test fixtures shared across the cache submodule test files.
//!
//! [`test_metadata`] and [`create_fake_image`] are referenced from
//! tests in `cache_dir.rs`, `housekeeping.rs`, and `metadata.rs`.
//! Keeping them here in a single module prevents drift if the
//! [`KernelMetadata`] field set or the in-test image content changes.

use std::fs;
use std::path::{Path, PathBuf};

use super::metadata::{KernelMetadata, KernelSource};

/// Build a default-shaped [`KernelMetadata`] with the given version.
/// Mirrors the minimal happy-path metadata `cache_dir::store` emits.
pub(crate) fn test_metadata(version: &str) -> KernelMetadata {
    KernelMetadata {
        version: Some(version.to_string()),
        source: KernelSource::Tarball,
        arch: "x86_64".to_string(),
        image_name: "bzImage".to_string(),
        config_hash: Some("abc123".to_string()),
        built_at: "2026-04-12T10:00:00Z".to_string(),
        ktstr_kconfig_hash: Some("def456".to_string()),
        extra_kconfig_hash: None,
        has_vmlinux: false,
        vmlinux_stripped: false,
        source_vmlinux_size: None,
        source_vmlinux_mtime_secs: None,
    }
}

/// Materialize a fake `bzImage` file under `dir` and return its path.
pub(crate) fn create_fake_image(dir: &Path) -> PathBuf {
    let image = dir.join("bzImage");
    fs::write(&image, b"fake kernel image").unwrap();
    image
}
