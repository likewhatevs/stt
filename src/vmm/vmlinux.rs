//! Locate the `vmlinux` ELF that pairs with a guest kernel image.
//!
//! Used by the host monitor and BPF reader to resolve symbols and
//! BTF offsets against the running guest kernel.

use std::path::{Path, PathBuf};

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
}
