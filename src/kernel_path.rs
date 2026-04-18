// Shared kernel directory resolution.
//
// Used by both build.rs (via include!) and lib.rs (via mod).
// Only uses std — no external crate dependencies.
// All functions are pure: callers supply inputs, handle caching.

/// Kernel identifier: filesystem path, version string, or cache key.
///
/// Parsing heuristic (see [`KernelId::parse`]):
/// - Contains `/` or starts with `.` or `~`: [`KernelId::Path`]
/// - Matches `MAJOR.MINOR[.PATCH][-rcN]`: [`KernelId::Version`]
/// - Otherwise: [`KernelId::CacheKey`]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelId {
    /// Filesystem path to kernel source/build directory.
    Path(std::path::PathBuf),
    /// Kernel version string (e.g. "6.14.2", "6.15-rc3").
    Version(String),
    /// Cache key (e.g. "6.14.2-tarball-x86_64-kc...").
    CacheKey(String),
}

impl KernelId {
    /// Parse a string into a kernel identifier.
    pub fn parse(s: &str) -> Self {
        if s.contains('/') || s.starts_with('.') || s.starts_with('~') {
            return KernelId::Path(std::path::PathBuf::from(s));
        }
        if _is_version_string(s) {
            return KernelId::Version(s.to_string());
        }
        KernelId::CacheKey(s.to_string())
    }
}

impl std::fmt::Display for KernelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelId::Path(p) => write!(f, "{}", p.display()),
            KernelId::Version(v) => write!(f, "{v}"),
            KernelId::CacheKey(k) => write!(f, "{k}"),
        }
    }
}

/// Check if a string matches a kernel version pattern.
///
/// Matches: `6.14`, `6.14.2`, `6.15-rc3`, `6.14.2-rc1`.
/// Does not match: `v6.14` (git tag prefix), `6` (no minor),
/// `6.14.2-tarball-x86_64-kc...` (cache key with extra segments).
fn _is_version_string(s: &str) -> bool {
    let (version_part, rc_part) = match s.split_once("-rc") {
        Some((v, rc)) => (v, Some(rc)),
        None => (s, None),
    };

    // The part after -rc must be a non-empty digit string.
    if let Some(rc) = rc_part
        && (rc.is_empty() || !rc.bytes().all(|b| b.is_ascii_digit()))
    {
        return false;
    }

    let mut parts = version_part.split('.');

    // Major: required, non-empty digits.
    match parts.next() {
        Some(p) if !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()) => {}
        _ => return false,
    }
    // Minor: required, non-empty digits.
    match parts.next() {
        Some(p) if !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()) => {}
        _ => return false,
    }
    // Patch: optional, non-empty digits.
    if let Some(patch) = parts.next()
        && (patch.is_empty() || !patch.bytes().all(|b| b.is_ascii_digit()))
    {
        return false;
    }
    // No more segments allowed (rejects `1.2.3.4`).
    parts.next().is_none()
}

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

/// Derive the kernel directory (holding `vmlinux` and related build
/// artifacts) from a kernel image path.
///
/// Recognizes two layouts:
///
/// - **Build tree**: `<root>/arch/x86/boot/bzImage` (or
///   `arch/arm64/boot/Image`) → `<root>`. Suffix match on the
///   canonical path.
/// - **Cache entry**: `<cache_dir>/bzImage` (or `Image`) with a
///   sibling `vmlinux` → `<cache_dir>`. Lets probe source-location
///   resolution walk a cached kernel's stripped ELF.
///
/// Returns `None` when neither layout matches or the input path
/// doesn't canonicalize.
///
/// Cache entries carry stripped vmlinux (no DWARF) — `strip_vmlinux_debug`
/// drops `.debug_*` on every cache entry regardless of source type.
/// file:line resolution works only for build-tree paths where the
/// unstripped vmlinux is still present, or when the caller layers
/// `cache::prefer_source_tree_for_dwarf` on top to re-route
/// `cache::KernelSource::Local` entries at their original source tree.
#[allow(dead_code)]
pub fn derive_kernel_dir(image: &std::path::Path) -> Option<std::path::PathBuf> {
    let canon = std::fs::canonicalize(image).ok()?;

    #[cfg(target_arch = "x86_64")]
    let build_suffix = "/arch/x86/boot/bzImage";
    #[cfg(target_arch = "aarch64")]
    let build_suffix = "/arch/arm64/boot/Image";

    if let Some(canon_str) = canon.to_str()
        && let Some(root) = canon_str.strip_suffix(build_suffix)
    {
        return Some(std::path::PathBuf::from(root));
    }

    let parent = canon.parent()?;
    // is_file (not exists) matches cache::prefer_source_tree_for_dwarf's
    // sibling probe, so a `vmlinux` directory or symlink-to-directory
    // cannot satisfy either check.
    if parent.join("vmlinux").is_file() {
        return Some(parent.to_path_buf());
    }

    None
}

/// Find a bootable kernel image within a directory.
///
/// Checks the arch-specific build tree path first (`arch/x86/boot/bzImage`
/// or `arch/arm64/boot/Image`), then falls back to the directory root
/// (for cache entries that store the boot image directly).
#[allow(dead_code)]
pub fn find_image_in_dir(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    // Build tree layout: arch-specific subdirectory.
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
    // Cache entry layout: boot image at directory root.
    #[cfg(target_arch = "x86_64")]
    {
        let p = dir.join("bzImage");
        if p.exists() {
            return Some(p);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let p = dir.join("Image");
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Find a bootable kernel image on the host.
///
/// `kernel_dir`: explicit kernel directory (e.g. from `KTSTR_KERNEL`).
/// When set, only that directory is searched — no fallback to local
/// build trees or host paths.
///
/// `release`: kernel release string (e.g. from `uname -r`). When
/// `None`, falls back to running `uname -r` via `Command`.
///
/// Without `kernel_dir`, searches local build trees (`./linux`,
/// `../linux`), `/lib/modules/{release}/build`, then host paths
/// (`/lib/modules/{release}/vmlinuz`, `/boot/vmlinuz-{release}`,
/// `/boot/vmlinuz`).
#[allow(dead_code)]
pub fn find_image(kernel_dir: Option<&str>, release: Option<&str>) -> Option<std::path::PathBuf> {
    // When kernel_dir is explicit, only check that directory.
    if let Some(dir_str) = kernel_dir {
        let dir = std::path::PathBuf::from(dir_str);
        if !dir.is_dir() {
            return None;
        }
        return find_image_in_dir(&dir);
    }

    // No explicit dir: search local build trees via resolve_kernel.
    if let Some(dir) = resolve_kernel(None)
        && let Some(img) = find_image_in_dir(&dir)
    {
        return Some(img);
    }

    // Host fallback paths.
    let owned_release;
    let rel = match release {
        Some(r) => Some(r),
        None => {
            owned_release = _kernel_release();
            owned_release.as_deref()
        }
    };

    if let Some(rel) = rel {
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
///
/// Checks both build tree layout (arch subdirectories) and cache
/// entry layout (boot image at directory root).
#[allow(dead_code)]
fn _has_kernel_artifacts(dir: &std::path::Path) -> bool {
    if dir.join("vmlinux").exists() {
        return true;
    }
    #[cfg(target_arch = "x86_64")]
    if dir.join("arch/x86/boot/bzImage").exists() || dir.join("bzImage").exists() {
        return true;
    }
    #[cfg(target_arch = "aarch64")]
    if dir.join("arch/arm64/boot/Image").exists() || dir.join("Image").exists() {
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

    // -- find_image_in_dir --

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn kernel_path_find_image_in_dir_bzimage() {
        let tmp = TempDir::new().unwrap();
        let boot = tmp.path().join("arch/x86/boot");
        std::fs::create_dir_all(&boot).unwrap();
        std::fs::write(boot.join("bzImage"), b"fake").unwrap();
        let result = find_image_in_dir(tmp.path());
        assert_eq!(result, Some(boot.join("bzImage")));
    }

    #[test]
    fn kernel_path_find_image_in_dir_empty() {
        let tmp = TempDir::new().unwrap();
        assert!(find_image_in_dir(tmp.path()).is_none());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn kernel_path_find_image_in_dir_cache_layout() {
        // Cache entries store bzImage at directory root (no arch/ subdir).
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("bzImage"), b"fake").unwrap();
        let result = find_image_in_dir(tmp.path());
        assert_eq!(result, Some(tmp.path().join("bzImage")));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn kernel_path_find_image_in_dir_prefers_build_tree() {
        // When both arch/ and root-level bzImage exist, prefer arch/.
        let tmp = TempDir::new().unwrap();
        let boot = tmp.path().join("arch/x86/boot");
        std::fs::create_dir_all(&boot).unwrap();
        std::fs::write(boot.join("bzImage"), b"build-tree").unwrap();
        std::fs::write(tmp.path().join("bzImage"), b"root-level").unwrap();
        let result = find_image_in_dir(tmp.path());
        assert_eq!(result, Some(boot.join("bzImage")));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn kernel_path_has_artifacts_root_bzimage() {
        // Cache entry layout: bzImage at directory root.
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("bzImage"), b"fake").unwrap();
        assert!(_has_kernel_artifacts(tmp.path()));
    }

    // -- derive_kernel_dir --

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn derive_kernel_dir_build_tree_x86() {
        let tmp = TempDir::new().unwrap();
        let boot = tmp.path().join("arch/x86/boot");
        std::fs::create_dir_all(&boot).unwrap();
        let image = boot.join("bzImage");
        std::fs::write(&image, b"fake").unwrap();

        let canon_root = std::fs::canonicalize(tmp.path()).unwrap();
        assert_eq!(derive_kernel_dir(&image), Some(canon_root));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn derive_kernel_dir_build_tree_aarch64() {
        let tmp = TempDir::new().unwrap();
        let boot = tmp.path().join("arch/arm64/boot");
        std::fs::create_dir_all(&boot).unwrap();
        let image = boot.join("Image");
        std::fs::write(&image, b"fake").unwrap();

        let canon_root = std::fs::canonicalize(tmp.path()).unwrap();
        assert_eq!(derive_kernel_dir(&image), Some(canon_root));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn derive_kernel_dir_cache_entry_x86_with_vmlinux() {
        let tmp = TempDir::new().unwrap();
        let image = tmp.path().join("bzImage");
        std::fs::write(&image, b"fake").unwrap();
        std::fs::write(tmp.path().join("vmlinux"), b"fake-elf").unwrap();

        let canon = std::fs::canonicalize(tmp.path()).unwrap();
        assert_eq!(derive_kernel_dir(&image), Some(canon));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn derive_kernel_dir_cache_entry_aarch64_with_vmlinux() {
        let tmp = TempDir::new().unwrap();
        let image = tmp.path().join("Image");
        std::fs::write(&image, b"fake").unwrap();
        std::fs::write(tmp.path().join("vmlinux"), b"fake-elf").unwrap();

        let canon = std::fs::canonicalize(tmp.path()).unwrap();
        assert_eq!(derive_kernel_dir(&image), Some(canon));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn derive_kernel_dir_cache_entry_without_vmlinux() {
        // bzImage at root with no vmlinux sibling — neither layout
        // applies, return None.
        let tmp = TempDir::new().unwrap();
        let image = tmp.path().join("bzImage");
        std::fs::write(&image, b"fake").unwrap();
        assert_eq!(derive_kernel_dir(&image), None);
    }

    #[test]
    fn derive_kernel_dir_nonexistent_path() {
        // canonicalize fails on a nonexistent path.
        let p = std::path::Path::new("/nonexistent/kernel/bzImage");
        assert_eq!(derive_kernel_dir(p), None);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn derive_kernel_dir_arbitrary_image_no_vmlinux_sibling() {
        // A file named bzImage but in a dir without a vmlinux sibling
        // and not under arch/x86/boot — no match.
        let tmp = TempDir::new().unwrap();
        let sub = tmp.path().join("somewhere/else");
        std::fs::create_dir_all(&sub).unwrap();
        let image = sub.join("bzImage");
        std::fs::write(&image, b"fake").unwrap();
        assert_eq!(derive_kernel_dir(&image), None);
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
        let result = find_image(Some(tmp.path().to_str().unwrap()), None);
        assert_eq!(result, Some(boot.join("bzImage")));
    }

    #[test]
    fn kernel_path_find_image_nonexistent_dir() {
        // Nonexistent explicit dir: is_dir() is false, returns None
        // immediately with no fallthrough.
        let _ = find_image(Some("/nonexistent/image/dir/xyz"), None);
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

    // -- KernelId parsing --

    #[test]
    fn kernel_id_parse_path_with_slash() {
        assert_eq!(
            KernelId::parse("../linux"),
            KernelId::Path(PathBuf::from("../linux"))
        );
        assert_eq!(
            KernelId::parse("/boot/vmlinuz"),
            KernelId::Path(PathBuf::from("/boot/vmlinuz"))
        );
    }

    #[test]
    fn kernel_id_parse_path_dot_prefix() {
        assert_eq!(
            KernelId::parse("./linux"),
            KernelId::Path(PathBuf::from("./linux"))
        );
        assert_eq!(KernelId::parse("."), KernelId::Path(PathBuf::from(".")));
    }

    #[test]
    fn kernel_id_parse_path_tilde_prefix() {
        assert_eq!(
            KernelId::parse("~/linux"),
            KernelId::Path(PathBuf::from("~/linux"))
        );
    }

    #[test]
    fn kernel_id_parse_version_stable() {
        assert_eq!(
            KernelId::parse("6.14.2"),
            KernelId::Version("6.14.2".to_string())
        );
        assert_eq!(
            KernelId::parse("6.14"),
            KernelId::Version("6.14".to_string())
        );
    }

    #[test]
    fn kernel_id_parse_version_rc() {
        assert_eq!(
            KernelId::parse("6.15-rc3"),
            KernelId::Version("6.15-rc3".to_string())
        );
    }

    #[test]
    fn kernel_id_parse_version_patch_rc() {
        assert_eq!(
            KernelId::parse("6.14.2-rc1"),
            KernelId::Version("6.14.2-rc1".to_string())
        );
    }

    #[test]
    fn kernel_id_parse_cache_key() {
        assert_eq!(
            KernelId::parse("6.14.2-tarball-x86_64"),
            KernelId::CacheKey("6.14.2-tarball-x86_64".to_string())
        );
        assert_eq!(
            KernelId::parse("local-deadbeef-x86_64"),
            KernelId::CacheKey("local-deadbeef-x86_64".to_string())
        );
    }

    #[test]
    fn kernel_id_parse_v_prefix_not_version() {
        // "v6.14" starts with 'v', not a digit -- cache key.
        assert_eq!(
            KernelId::parse("v6.14"),
            KernelId::CacheKey("v6.14".to_string())
        );
    }

    #[test]
    fn kernel_id_parse_bare_major_not_version() {
        // "6" alone has no minor component -- cache key.
        assert_eq!(KernelId::parse("6"), KernelId::CacheKey("6".to_string()));
    }

    #[test]
    fn kernel_id_display() {
        assert_eq!(
            KernelId::Version("6.14.2".to_string()).to_string(),
            "6.14.2"
        );
        assert_eq!(
            KernelId::Path(PathBuf::from("../linux")).to_string(),
            "../linux"
        );
        assert_eq!(
            KernelId::CacheKey("my-key".to_string()).to_string(),
            "my-key"
        );
    }

    // -- _is_version_string --

    #[test]
    fn kernel_id_is_version_string_valid() {
        assert!(_is_version_string("6.14"));
        assert!(_is_version_string("6.14.2"));
        assert!(_is_version_string("6.15-rc3"));
        assert!(_is_version_string("6.14.0-rc1"));
        assert!(_is_version_string("5.0"));
        assert!(_is_version_string("5.0.0"));
        assert!(_is_version_string("5.4.0"));
    }

    #[test]
    fn kernel_id_is_version_string_invalid() {
        assert!(!_is_version_string("6"));
        assert!(!_is_version_string("v6.14"));
        assert!(!_is_version_string(""));
        assert!(!_is_version_string("6.14.2-tarball-x86_64"));
        assert!(!_is_version_string("6.14.2.3"));
        assert!(!_is_version_string("6.14-rc"));
        assert!(!_is_version_string("6.14-rcX"));
        // rc_part contains non-digits after splitting on "-rc".
        assert!(!_is_version_string("6.14-rc3-tarball-x86_64"));
        assert!(!_is_version_string("abc"));
        assert!(!_is_version_string(".14"));
        assert!(!_is_version_string("6."));
        assert!(!_is_version_string("linux"));
        assert!(!_is_version_string(".6"));
    }

    // -- proptest --

    proptest::proptest! {
        #[test]
        fn prop_kernel_id_parse_never_panics(s in "\\PC{0,30}") {
            let _ = KernelId::parse(&s);
        }

        #[test]
        fn prop_kernel_id_path_on_slash(
            prefix in "[a-z]{1,5}",
            suffix in "[a-z]{1,5}",
        ) {
            let s = format!("{prefix}/{suffix}");
            assert!(matches!(KernelId::parse(&s), KernelId::Path(_)));
        }

        #[test]
        fn prop_kernel_id_path_on_dot_prefix(s in "\\.[a-z]{1,10}") {
            assert!(matches!(KernelId::parse(&s), KernelId::Path(_)));
        }

        #[test]
        fn prop_kernel_id_version_roundtrip(
            major in 1u32..20,
            minor in 0u32..50,
            patch in 0u32..100,
        ) {
            let v = format!("{major}.{minor}.{patch}");
            assert_eq!(KernelId::parse(&v), KernelId::Version(v.clone()));
        }

        #[test]
        fn prop_kernel_id_version_rc(major in 1u32..20, minor in 0u32..50, rc in 1u32..10) {
            let v = format!("{major}.{minor}-rc{rc}");
            assert_eq!(KernelId::parse(&v), KernelId::Version(v.clone()));
        }
    }
}
