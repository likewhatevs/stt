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
