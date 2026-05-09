//! Locate the `vmlinux` ELF that pairs with a guest kernel image.
//!
//! Used by the host monitor and BPF reader to resolve symbols and
//! BTF offsets against the running guest kernel.

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};

/// Process-global cache of vmlinux ELF bytes keyed by canonical path.
///
/// `collect_verifier_stats` is called once per failure-dump cycle; in a
/// nextest test process running many `#[ktstr_test]` cases that boot
/// fresh VMs against the same kernel, repeating the file read for every
/// VM costs 50-340 MB of disk I/O on the freeze-coord cleanup critical
/// path. Caching the bytes once per canonical path collapses every
/// subsequent VM's read to a hash lookup + `Arc::clone`.
///
/// Same-process invariant: vmlinux content is immutable per process.
/// A user that rebuilds vmlinux must restart the test process for the
/// new bytes to take effect — same as every other on-disk artifact
/// the host pre-loads at boot. The cache key is the canonicalized path
/// so symlinks across cache / source-tree layouts collapse to one
/// entry. A `canonicalize` failure (EACCES, missing target) skips the
/// cache and falls through to the direct read.
static VMLINUX_BYTES_CACHE: OnceLock<RwLock<std::collections::HashMap<PathBuf, Arc<Vec<u8>>>>> =
    OnceLock::new();

/// Return the cached vmlinux ELF bytes for `path`, populating the cache
/// on first read.
///
/// Returns `None` when `path` is unreadable. The error case is not
/// cached: a transient EACCES (e.g. a half-written cache entry whose
/// permissions arrive on the next ms) should not poison the cache for
/// the rest of the process.
pub(crate) fn cached_vmlinux_bytes(path: &Path) -> Option<Arc<Vec<u8>>> {
    let canon = std::fs::canonicalize(path).ok().unwrap_or_else(|| path.to_path_buf());
    let slot = VMLINUX_BYTES_CACHE.get_or_init(|| RwLock::new(std::collections::HashMap::new()));
    {
        let read = slot.read().unwrap_or_else(|e| e.into_inner());
        if let Some(bytes) = read.get(&canon) {
            return Some(Arc::clone(bytes));
        }
    }
    // Read outside the write lock so a slow read doesn't block other
    // canonical paths' lookups. A racing second reader will pay the
    // same read once each — acceptable: the bytes are immutable so
    // both inserts produce the same value.
    let bytes = std::fs::read(&canon).ok()?;
    let arc = Arc::new(bytes);
    let mut write = slot.write().unwrap_or_else(|e| e.into_inner());
    // `or_insert` consumes `arc` only on the miss path; on the racing-
    // win-then-lose path the existing entry's `Arc` is returned and
    // our `arc` is dropped (one wasted file read; the bytes match).
    Some(Arc::clone(write.entry(canon).or_insert(arc)))
}

/// Find the vmlinux ELF next to a kernel image path.
///
/// Shared across x86_64 and aarch64. Both architectures follow the
/// kernel build's `<root>/arch/<arch>/boot/<image>` layout, so
/// stepping 3 directories up from `kernel_path` lands on `<root>`
/// where `vmlinux` sits. Distro paths diverge: x86_64 ships debug
/// vmlinux at `/usr/lib/debug/boot/vmlinux-<version>`, aarch64 splits
/// between `/boot/vmlinux-<version>` and
/// `/lib/modules/<version>/build/vmlinux`. Both distro layouts are
/// probed regardless of arch — the arch-specific filename prefix
/// (`bzImage` vs `Image`) only tells us where to look, not which
/// layout owns the match.
pub(crate) fn find_vmlinux(kernel_path: &Path) -> Option<PathBuf> {
    let dir = kernel_path.parent()?;
    let candidate = dir.join("vmlinux");
    if candidate.exists() {
        return Some(candidate);
    }
    // Kernel build tree: <root>/arch/<arch>/boot/<image> -> <root>/vmlinux.
    if let Ok(root) = dir.join("../../..").canonicalize() {
        let candidate = root.join("vmlinux");
        if candidate.exists() {
            return Some(candidate);
        }
    }
    // Distro layouts keyed by the image's version suffix
    // (`vmlinuz-<version>`).
    if let Some(name) = kernel_path.file_name().and_then(|n| n.to_str()) {
        let version = name.strip_prefix("vmlinuz-").unwrap_or(name);
        for candidate in [
            PathBuf::from(format!("/usr/lib/debug/boot/vmlinux-{version}")),
            PathBuf::from(format!("/boot/vmlinux-{version}")),
            PathBuf::from(format!("/lib/modules/{version}/build/vmlinux")),
        ] {
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    // `/lib/modules/<version>/vmlinuz` layout: version is the parent
    // directory name, and the sibling `build/vmlinux` is the target.
    if let Some(parent_name) = dir.file_name().and_then(|n| n.to_str()) {
        for candidate in [
            dir.join("build/vmlinux"),
            PathBuf::from(format!("/boot/vmlinux-{parent_name}")),
        ] {
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn find_vmlinux_from_bzimage_path() {
        // Create a temp dir simulating <root>/arch/x86/boot/bzImage with vmlinux at <root>.
        let tmp = std::env::temp_dir().join("ktstr-find-vmlinux-test");
        let boot_dir = tmp.join("arch/x86/boot");
        std::fs::create_dir_all(&boot_dir).unwrap();
        let vmlinux = tmp.join("vmlinux");
        std::fs::write(&vmlinux, b"ELF").unwrap();
        let bzimage = boot_dir.join("bzImage");
        std::fs::write(&bzimage, b"kernel").unwrap();

        let found = find_vmlinux(&bzimage);
        assert_eq!(found, Some(vmlinux));

        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn find_vmlinux_sibling() {
        // vmlinux in the same directory as the kernel image.
        let tmp = std::env::temp_dir().join("ktstr-find-vmlinux-sibling");
        std::fs::create_dir_all(&tmp).unwrap();
        let vmlinux = tmp.join("vmlinux");
        std::fs::write(&vmlinux, b"ELF").unwrap();
        let kernel = tmp.join("bzImage");
        std::fs::write(&kernel, b"kernel").unwrap();

        let found = find_vmlinux(&kernel);
        assert_eq!(found, Some(vmlinux));

        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn find_vmlinux_bare_filename() {
        // A bare filename — parent is "" so no vmlinux sibling found.
        assert_eq!(find_vmlinux(Path::new("vmlinuz")), None);
    }

    #[test]
    fn find_vmlinux_root_parent() {
        // /vmlinuz has parent "/" — no vmlinux there (or if there is, fine).
        // The function should not panic.
        let result = find_vmlinux(Path::new("/vmlinuz"));
        // /vmlinux almost certainly doesn't exist; if it does, that's still valid.
        if !Path::new("/vmlinux").exists() {
            assert_eq!(result, None);
        }
    }

    #[test]
    fn find_vmlinux_missing_returns_none() {
        let tmp = std::env::temp_dir().join("ktstr-find-vmlinux-none");
        std::fs::create_dir_all(&tmp).unwrap();
        let kernel = tmp.join("bzImage");
        std::fs::write(&kernel, b"kernel").unwrap();

        assert_eq!(find_vmlinux(&kernel), None);

        std::fs::remove_dir_all(&tmp).unwrap();
    }

    /// First call reads from disk; second call returns a clone of the
    /// cached `Arc<Vec<u8>>`, proving the cache hit path does not re-
    /// read. `Arc::ptr_eq` is the load-bearing assertion: the bytes
    /// would compare equal even from a re-read, but only the cache
    /// hit returns the same allocation.
    #[test]
    fn cached_vmlinux_bytes_hits_on_second_call() {
        let tmp = std::env::temp_dir().join("ktstr-cached-vmlinux-bytes");
        std::fs::create_dir_all(&tmp).unwrap();
        let vmlinux = tmp.join("vmlinux-test-cache");
        std::fs::write(&vmlinux, b"FAKE_VMLINUX_BYTES").unwrap();

        let first = cached_vmlinux_bytes(&vmlinux).expect("first read populates cache");
        let second = cached_vmlinux_bytes(&vmlinux).expect("second read hits cache");
        assert_eq!(first.as_slice(), b"FAKE_VMLINUX_BYTES");
        assert!(
            Arc::ptr_eq(&first, &second),
            "cache hit must return the same Arc; got fresh allocations on each call"
        );

        std::fs::remove_file(&vmlinux).ok();
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// Unreadable path returns `None` without populating the cache;
    /// a subsequent successful path is unaffected.
    #[test]
    fn cached_vmlinux_bytes_missing_returns_none() {
        let nonexistent = std::env::temp_dir().join("ktstr-cached-vmlinux-bytes-missing-xyzzy");
        // Defensive: ensure the file does not exist from a prior run.
        std::fs::remove_file(&nonexistent).ok();
        assert!(cached_vmlinux_bytes(&nonexistent).is_none());
    }
}
