// Shared kernel directory resolution.
//
// Used by both build.rs (via include!) and lib.rs (via mod).
// Only uses std — no external crate dependencies.
// All functions are pure: callers supply inputs, handle caching.

/// Resolve a kernel source/build directory.
///
/// `kernel_dir`: value of `KTSTR_KERNEL` env var (if set).
///
/// Search order:
/// 1. `kernel_dir` parameter (from env var)
/// 2. `./linux` (workspace-local build tree)
/// 3. `../linux` (sibling directory)
/// 4. `/lib/modules/{release}/build` (installed kernel headers)
///
/// Returns the directory path if a kernel tree is found.
#[allow(dead_code)]
pub fn resolve_kernel(kernel_dir: Option<&str>) -> Option<std::path::PathBuf> {
    // 1. Explicit directory.
    if let Some(dir) = kernel_dir {
        let p = std::path::PathBuf::from(dir);
        if p.is_dir() {
            return Some(p);
        }
    }

    // 2-3. Local build trees.
    for rel in &["./linux", "../linux"] {
        let p = std::path::PathBuf::from(rel);
        if p.is_dir() && _has_kernel_artifacts(&p) {
            return Some(p);
        }
    }

    // 4. Installed kernel build dir.
    if let Some(rel) = _kernel_release() {
        let p = std::path::PathBuf::from(format!("/lib/modules/{rel}/build"));
        if p.is_dir() {
            return Some(p);
        }
    }

    None
}

/// Find a bootable kernel image within a directory.
#[allow(dead_code)]
fn _find_image_in_dir(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    #[cfg(target_arch = "x86_64")]
    {
        let p = dir.join("arch/x86/boot/bzImage");
        if p.exists() {
            return Some(p);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let p = dir.join("arch/arm64/boot/Image");
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Find a bootable kernel image on the host.
///
/// `kernel_dir`: explicit kernel directory (e.g. from `KTSTR_KERNEL`).
///
/// Checks the resolved kernel directory for arch-specific boot images,
/// then versioned paths (`/lib/modules/$(uname -r)/vmlinuz`,
/// `/boot/vmlinuz-$(uname -r)`), then `/boot/vmlinuz`.
#[allow(dead_code)]
pub fn find_image(kernel_dir: Option<&str>) -> Option<std::path::PathBuf> {
    if let Some(dir) = resolve_kernel(kernel_dir)
        && let Some(img) = _find_image_in_dir(&dir)
    {
        return Some(img);
    }

    if let Some(rel) = _kernel_release() {
        let p = std::path::PathBuf::from(format!("/lib/modules/{rel}/vmlinuz"));
        if std::fs::File::open(&p).is_ok() {
            return Some(p);
        }
        let p = std::path::PathBuf::from(format!("/boot/vmlinuz-{rel}"));
        if std::fs::File::open(&p).is_ok() {
            return Some(p);
        }
    }

    let p = std::path::PathBuf::from("/boot/vmlinuz");
    if std::fs::File::open(&p).is_ok() {
        return Some(p);
    }

    None
}

/// Resolve the BTF source file for vmlinux.h generation.
///
/// `kernel_dir`: explicit kernel directory (e.g. from `KTSTR_KERNEL`).
///
/// Prefers `{resolved_dir}/vmlinux`, then `/sys/kernel/btf/vmlinux`.
#[allow(dead_code)]
pub fn resolve_btf(kernel_dir: Option<&str>) -> Option<std::path::PathBuf> {
    if let Some(dir) = resolve_kernel(kernel_dir) {
        let vmlinux = dir.join("vmlinux");
        if vmlinux.exists() {
            return Some(vmlinux);
        }
    }
    let sysfs = std::path::Path::new("/sys/kernel/btf/vmlinux");
    if sysfs.exists() {
        return Some(sysfs.to_path_buf());
    }
    None
}

/// Check if a directory contains kernel build artifacts.
#[allow(dead_code)]
fn _has_kernel_artifacts(dir: &std::path::Path) -> bool {
    if dir.join("vmlinux").exists() {
        return true;
    }
    #[cfg(target_arch = "x86_64")]
    if dir.join("arch/x86/boot/bzImage").exists() {
        return true;
    }
    #[cfg(target_arch = "aarch64")]
    if dir.join("arch/arm64/boot/Image").exists() {
        return true;
    }
    false
}

/// Get the running kernel release string (equivalent to `uname -r`).
#[allow(dead_code)]
fn _kernel_release() -> Option<String> {
    std::process::Command::new("uname")
        .arg("-r")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::TempDir;

    // -- resolve_kernel --

    #[test]
    fn kernel_path_resolve_explicit_dir_exists() {
        let tmp = TempDir::new().unwrap();
        let result = resolve_kernel(Some(tmp.path().to_str().unwrap()));
        assert_eq!(result, Some(tmp.path().to_path_buf()));
    }

    #[test]
    fn kernel_path_resolve_explicit_dir_not_exists() {
        let result = resolve_kernel(Some("/nonexistent/kernel/dir/that/does/not/exist"));
        // The explicit dir doesn't exist, so resolve_kernel skips it.
        // It may still find a kernel via fallback paths (./linux, ../linux,
        // /lib/modules). The key invariant: the nonexistent path must never
        // be returned.
        assert_ne!(
            result,
            Some(PathBuf::from("/nonexistent/kernel/dir/that/does/not/exist"))
        );
    }

    #[test]
    fn kernel_path_resolve_none_falls_through() {
        // With None, resolve_kernel skips the explicit branch and tries
        // ./linux, ../linux, then /lib/modules. The result depends on
        // the host, but the function must not panic.
        let _ = resolve_kernel(None);
    }

    #[test]
    fn kernel_path_resolve_empty_string() {
        // Empty string creates a PathBuf("") which is_dir() returns false,
        // so it falls through to search paths.
        let result = resolve_kernel(Some(""));
        // "" is not a directory, so it must not be returned as the explicit path.
        assert_ne!(result, Some(PathBuf::from("")));
    }

    // -- _has_kernel_artifacts --

    #[test]
    fn kernel_path_has_artifacts_vmlinux() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("vmlinux"), b"fake").unwrap();
        assert!(_has_kernel_artifacts(tmp.path()));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn kernel_path_has_artifacts_bzimage() {
        let tmp = TempDir::new().unwrap();
        let boot = tmp.path().join("arch/x86/boot");
        std::fs::create_dir_all(&boot).unwrap();
        std::fs::write(boot.join("bzImage"), b"fake").unwrap();
        assert!(_has_kernel_artifacts(tmp.path()));
    }

    #[test]
    fn kernel_path_has_artifacts_empty_dir() {
        let tmp = TempDir::new().unwrap();
        assert!(!_has_kernel_artifacts(tmp.path()));
    }

    // -- _find_image_in_dir --

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn kernel_path_find_image_in_dir_bzimage() {
        let tmp = TempDir::new().unwrap();
        let boot = tmp.path().join("arch/x86/boot");
        std::fs::create_dir_all(&boot).unwrap();
        std::fs::write(boot.join("bzImage"), b"fake").unwrap();
        let result = _find_image_in_dir(tmp.path());
        assert_eq!(result, Some(boot.join("bzImage")));
    }

    #[test]
    fn kernel_path_find_image_in_dir_empty() {
        let tmp = TempDir::new().unwrap();
        assert!(_find_image_in_dir(tmp.path()).is_none());
    }

    // -- resolve_btf --

    #[test]
    fn kernel_path_resolve_btf_with_vmlinux_in_dir() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("vmlinux"), b"fake").unwrap();
        let result = resolve_btf(Some(tmp.path().to_str().unwrap()));
        assert_eq!(result, Some(tmp.path().join("vmlinux")));
    }

    #[test]
    fn kernel_path_resolve_btf_dir_without_vmlinux() {
        let tmp = TempDir::new().unwrap();
        // No vmlinux in dir; falls through to /sys/kernel/btf/vmlinux check.
        let result = resolve_btf(Some(tmp.path().to_str().unwrap()));
        // Result depends on host: either /sys/kernel/btf/vmlinux exists or None.
        if let Some(ref p) = result {
            assert!(p.exists());
        }
    }

    #[test]
    fn kernel_path_resolve_btf_nonexistent_dir() {
        let result = resolve_btf(Some("/nonexistent/btf/dir/xyz"));
        // Dir doesn't exist so resolve_kernel returns None; falls to sysfs.
        if let Some(ref p) = result {
            assert!(p.exists());
        }
    }

    // -- find_image --

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn kernel_path_find_image_explicit_dir_with_bzimage() {
        let tmp = TempDir::new().unwrap();
        let boot = tmp.path().join("arch/x86/boot");
        std::fs::create_dir_all(&boot).unwrap();
        std::fs::write(boot.join("bzImage"), b"fake").unwrap();
        let result = find_image(Some(tmp.path().to_str().unwrap()));
        assert_eq!(result, Some(boot.join("bzImage")));
    }

    #[test]
    fn kernel_path_find_image_nonexistent_dir() {
        // Nonexistent explicit dir: falls through to kernel_release paths.
        // Must not panic regardless of host state.
        let _ = find_image(Some("/nonexistent/image/dir/xyz"));
    }

    // -- _kernel_release --

    #[test]
    fn kernel_path_kernel_release_returns_string() {
        let rel = _kernel_release();
        // uname -r should succeed on any Linux host.
        assert!(rel.is_some());
        let s = rel.unwrap();
        assert!(!s.is_empty());
        assert!(!s.contains('\n'));
    }
}
