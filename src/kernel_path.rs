// Shared kernel directory resolution.
//
// Used by both build.rs (via `include!("src/kernel_path.rs")`) and
// lib.rs (via `pub mod kernel_path`). The `include!` is deliberate:
// build.rs runs before the crate compiles, so it cannot `use ktstr::...`.
// Duplicating the resolution logic would drift between build-time
// BTF discovery (vmlinux.h generation) and run-time kernel selection.
//
// Constraints that every edit to this file must satisfy — breaking
// any of them surfaces as either a cryptic build-script error or a
// runtime/build-time behaviour mismatch:
//
// 1. **No non-std imports.** build.rs has its own dependency graph
//    (`libbpf-cargo`, `tempfile`, etc.). A `use foo::bar` here would
//    compile inside lib.rs (via the `pub mod` path) but fail inside
//    build.rs because build.rs hasn't declared `foo` as a build-dep.
// 2. **No `pub(crate)` items.** `pub(crate)` is meaningless inside
//    an `include!`'d fragment — build.rs isn't a crate, so the item
//    resolves at crate-root visibility there. Use `pub` for items
//    build.rs needs, `fn` (private) for items lib.rs alone uses.
// 3. **`#[cfg(test)]` blocks may use non-std test helpers freely.**
//    Cargo does not set `cfg(test)` when compiling build scripts, so
//    `#[cfg(test)]` items inside this file are simply elided from the
//    build.rs view of the fragment — `tempfile`, `proptest`, etc. are
//    safe to import inside `#[cfg(test)] mod tests { ... }`. The
//    std-only rule (#1 above) applies to non-`cfg(test)` items only.
// 4. **All functions are pure.** Callers supply inputs and handle
//    caching — no global state, no `std::env::set_var`, no FS
//    writes outside the caller-provided paths. Pure is what makes
//    the double-consumer (build + runtime) safe.

/// Kernel identifier: filesystem path, version string, cache key,
/// stable-release range, or git source.
///
/// Parsing heuristic (see [`KernelId::parse`]):
/// - Contains `/` (without a `git+` prefix) or starts with `.` or `~`:
///   [`KernelId::Path`]
/// - Starts with `git+`: [`KernelId::Git`] (form `git+URL#REF`)
/// - Contains `..` between two version-shaped tokens:
///   [`KernelId::Range`] (inclusive on both endpoints)
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
    /// Inclusive range of stable kernel versions, expanded against
    /// kernel.org's release index at resolve time. `start` and `end`
    /// are both [`KernelId::Version`]-shaped strings (e.g. "6.10",
    /// "6.13"); the resolver fans this out to every release in
    /// [start, end] inclusive on both endpoints. A version present in
    /// the range but missing from the upstream index is a hard error
    /// before any boot — partial expansions are not silently dropped.
    Range {
        /// Inclusive lower bound, version-shaped.
        start: String,
        /// Inclusive upper bound, version-shaped.
        end: String,
    },
    /// Git source: clone `url`, check out `git_ref`. `git_ref` may be
    /// a branch, tag, or sha — current `KernelId::parse` stores it
    /// verbatim with no remote contact. Branches will be resolved to
    /// a sha at cache-resolution time (Stage 3 wiring; the resolver
    /// will call `git ls-remote` once per branch ref to anchor
    /// cached builds to a content-addressable
    /// `(URL, resolved_sha)` cache key so identical underlying
    /// commits collapse to one cache entry regardless of which
    /// spelling produced them).
    Git {
        /// Remote URL (https or git@).
        url: String,
        /// Branch name, tag, or sha. Stored verbatim by `parse`;
        /// branches will be resolved to a sha at cache-resolution
        /// time so cached builds remain content-addressed.
        git_ref: String,
    },
}

impl KernelId {
    /// Parse a string into a kernel identifier.
    ///
    /// Recognizes (in order):
    /// - `git+URL#REF` → [`KernelId::Git`] (the `git+` prefix takes
    ///   precedence over the `/`-contains test below, since URLs
    ///   contain `/`).
    /// - `START..END` where both endpoints are version-shaped →
    ///   [`KernelId::Range`] with both endpoints inclusive. The `..`
    ///   spelling is fixed regardless of inclusivity (Rust's
    ///   exclusive-`..` / inclusive-`..=` distinction does not apply
    ///   here — the range is always closed).
    /// - `/`-containing or `.`/`~`-prefixed → [`KernelId::Path`].
    /// - Version-shaped → [`KernelId::Version`].
    /// - Anything else → [`KernelId::CacheKey`].
    pub fn parse(s: &str) -> Self {
        if let Some(rest) = s.strip_prefix("git+")
            && let Some((url, git_ref)) = rest.rsplit_once('#')
            && !url.is_empty()
            && !git_ref.is_empty()
        {
            return KernelId::Git {
                url: url.to_string(),
                git_ref: git_ref.to_string(),
            };
        }
        if let Some((start, end)) = s.split_once("..")
            && _is_version_string(start)
            && _is_version_string(end)
        {
            return KernelId::Range {
                start: start.to_string(),
                end: end.to_string(),
            };
        }
        if s.contains('/') || s.starts_with('.') || s.starts_with('~') {
            return KernelId::Path(std::path::PathBuf::from(s));
        }
        if _is_version_string(s) {
            return KernelId::Version(s.to_string());
        }
        KernelId::CacheKey(s.to_string())
    }

    /// Parse a comma-separated list of kernel specs into a vector of
    /// identifiers. Empty entries are silently skipped (so trailing
    /// commas or repeated separators are forgiving). Each non-empty
    /// segment is fed through [`KernelId::parse`] verbatim — so
    /// `parse_list("6.10,git+URL#main,/srv/linux")` returns three
    /// distinct variants. Deduplication is the resolver's
    /// responsibility (after canonicalization to a cache key); this
    /// function preserves order and duplicates as written.
    pub fn parse_list(s: &str) -> Vec<KernelId> {
        s.split(',')
            .map(str::trim)
            .filter(|seg| !seg.is_empty())
            .map(KernelId::parse)
            .collect()
    }

    /// Validate a parsed `KernelId` for resolve-time legality. Returns
    /// `Err(message)` when the identifier carries a structural problem
    /// the parser couldn't catch on its own — currently:
    ///
    /// - [`KernelId::Range`] with `start > end` after numeric
    ///   component-wise comparison. The parser cannot reject this at
    ///   parse time because both endpoints are valid version strings
    ///   in isolation; the inversion only surfaces when the two are
    ///   compared.
    ///
    /// All other variants always return `Ok(())` — this is a hook for
    /// future per-variant invariants, not a general-purpose validator.
    /// Use `Result<(), String>` rather than `anyhow::Result` because
    /// this file is included from `build.rs` (see file header rule
    /// #1, no non-std imports outside `cfg(test)`).
    ///
    /// Comparison semantics: each endpoint decomposes to a
    /// `(major, minor, patch, rc)` tuple where missing patch maps to
    /// `0` and missing `-rc` maps to `u64::MAX` so a release
    /// (`6.10`) sorts strictly above any pre-release (`6.10-rc3`) of
    /// the same major.minor.patch. Inverted ranges include
    /// `7.0..6.99`, `6.10..6.5`, `6.10..6.10-rc3` (release > rc), and
    /// `6.10-rc3..6.10-rc1`. Equal endpoints (`6.10..6.10`) pass
    /// validation as a single-element range.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            KernelId::Range { start, end } => {
                let start_key = decompose_version_for_compare(start).ok_or_else(|| {
                    format!(
                        "kernel range start `{start}` is not a parseable version \
                         (version components must fit u64). Direct callers that \
                         construct `KernelId::Range` outside `KernelId::parse` are \
                         responsible for endpoint validity; the parser admits a \
                         strict subset.",
                    )
                })?;
                let end_key = decompose_version_for_compare(end).ok_or_else(|| {
                    format!(
                        "kernel range end `{end}` is not a parseable version \
                         (version components must fit u64). Direct callers that \
                         construct `KernelId::Range` outside `KernelId::parse` are \
                         responsible for endpoint validity; the parser admits a \
                         strict subset.",
                    )
                })?;
                if start_key > end_key {
                    return Err(format!(
                        "inverted kernel range `{start}..{end}`: start version is greater \
                         than end version. Swap the endpoints (`{end}..{start}`) or omit \
                         the range to pass a single version.",
                    ));
                }
                Ok(())
            }
            KernelId::Path(_)
            | KernelId::Version(_)
            | KernelId::CacheKey(_)
            | KernelId::Git { .. } => Ok(()),
        }
    }
}

impl std::fmt::Display for KernelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelId::Path(p) => write!(f, "{}", p.display()),
            KernelId::Version(v) => write!(f, "{v}"),
            KernelId::CacheKey(k) => write!(f, "{k}"),
            KernelId::Range { start, end } => write!(f, "{start}..{end}"),
            KernelId::Git { url, git_ref } => write!(f, "git+{url}#{git_ref}"),
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

/// Decompose a version-shaped string into a `(major, minor, patch,
/// rc)` tuple suitable for `Ord` comparison. Returns `None` when the
/// input doesn't match the kernel-version grammar — same predicate as
/// [`_is_version_string`] but extracting numeric components rather
/// than just yes/no.
///
/// Comparison semantics:
/// - Missing patch defaults to `0` so `6.10` and `6.10.0` compare
///   equal.
/// - Missing `-rcN` defaults to `u64::MAX` so a release
///   (`6.10`, `6.10.5`) sorts strictly above any pre-release
///   (`6.10-rc3`, `6.10.5-rc1`) of the same `major.minor.patch`. A
///   future major/minor/patch bump still dominates because the tuple
///   is compared in declaration order — the rc-as-MAX trick only
///   resolves ties on the leading three components.
///
/// Used by [`KernelId::validate`] to detect inverted ranges
/// (`6.16..6.12`, `6.10..6.10-rc3`, `7.0..6.99`).
fn decompose_version_for_compare(s: &str) -> Option<(u64, u64, u64, u64)> {
    let (version_part, rc_part) = match s.split_once("-rc") {
        Some((v, rc)) => (v, Some(rc)),
        None => (s, None),
    };
    // rc must be a non-empty digit string when present.
    let rc: u64 = match rc_part {
        Some(rc) if rc.is_empty() || !rc.bytes().all(|b| b.is_ascii_digit()) => return None,
        Some(rc) => rc.parse().ok()?,
        None => u64::MAX,
    };
    let mut parts = version_part.split('.');
    let major: u64 = parts.next()?.parse().ok()?;
    let minor: u64 = parts.next()?.parse().ok()?;
    let patch: u64 = match parts.next() {
        Some("") => return None,
        Some(p) => p.parse().ok()?,
        None => 0,
    };
    // Reject `1.2.3.4` and similar — only major.minor[.patch] is grammar.
    if parts.next().is_some() {
        return None;
    }
    Some((major, minor, patch, rc))
}

/// Read the running kernel release from `/proc/sys/kernel/osrelease`.
///
/// Returns `None` if the procfs entry is unreadable, empty, or missing.
/// Callers that need the release string for `/lib/modules/{release}/…`
/// fallbacks use this rather than shelling out to `uname -r`: the
/// procfs entry exposes the same value the kernel returns from the
/// uname(2) syscall (see linux/kernel/sys.c: `override_release`) and
/// only costs a small read.
fn kernel_release_from_procfs() -> Option<String> {
    std::fs::read_to_string("/proc/sys/kernel/osrelease")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
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
        if p.is_dir() && has_kernel_artifacts(&p) {
            return Some(p);
        }
    }

    // 4. Installed kernel build dir — use the running release from
    // procfs to locate `/lib/modules/{release}/build`.
    if let Some(rel) = kernel_release_from_procfs() {
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
/// `None`, falls back to reading `/proc/sys/kernel/osrelease` — the
/// same value the kernel exposes via the `uname(2)` syscall, without
/// the shell-out cost.
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

    // Host fallback paths. When `release` is not supplied, pull the
    // running kernel release from procfs via
    // [`kernel_release_from_procfs`].
    let owned_release;
    let rel = match release {
        Some(r) => Some(r),
        None => {
            owned_release = kernel_release_from_procfs();
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
fn has_kernel_artifacts(dir: &std::path::Path) -> bool {
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
    fn kernel_path_resolve_none_returns_osrelease_build_dir_when_present() {
        // resolve_kernel(None) reads `/proc/sys/kernel/osrelease` and
        // checks `/lib/modules/{rel}/build` as its last fallback. The
        // earlier branches (`./linux`, `../linux`) cannot be controlled
        // from a parallel-safe unit test (`set_current_dir` is process-
        // wide), so this test is strong only when those local trees are
        // absent. When `/lib/modules/{rel}/build` is absent on the host
        // (typical CI without installed kernel headers), skip via early
        // return — the panic-free contract is already covered by
        // `kernel_path_resolve_none_falls_through`.
        let release = std::fs::read_to_string("/proc/sys/kernel/osrelease")
            .expect("host /proc/sys/kernel/osrelease must be readable for this test")
            .trim()
            .to_string();
        let expected = std::path::PathBuf::from(format!("/lib/modules/{release}/build"));
        if !expected.is_dir() {
            return;
        }

        let resolved = resolve_kernel(None).unwrap_or_else(|| {
            panic!(
                "resolve_kernel(None) must return Some when {} exists",
                expected.display(),
            )
        });
        assert!(
            resolved.is_dir(),
            "resolved path must be a directory, got {}",
            resolved.display(),
        );
        // Strong pin only when no earlier branch (`./linux`, `../linux`)
        // shadowed the osrelease path. When an earlier branch matched,
        // the panic-free + valid-dir contract above is what we get.
        let local_shadowed = std::path::PathBuf::from("./linux").is_dir()
            || std::path::PathBuf::from("../linux").is_dir();
        if !local_shadowed {
            assert_eq!(
                resolved, expected,
                "with no local trees, resolve_kernel(None) must return the osrelease build dir",
            );
        }
    }

    #[test]
    fn kernel_path_resolve_empty_string() {
        // Empty string creates a PathBuf("") which is_dir() returns false,
        // so it falls through to search paths.
        let result = resolve_kernel(Some(""));
        // "" is not a directory, so it must not be returned as the explicit path.
        assert_ne!(result, Some(PathBuf::from("")));
    }

    // -- has_kernel_artifacts --

    #[test]
    fn kernel_path_has_artifacts_vmlinux() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("vmlinux"), b"fake").unwrap();
        assert!(has_kernel_artifacts(tmp.path()));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn kernel_path_has_artifacts_bzimage() {
        let tmp = TempDir::new().unwrap();
        let boot = tmp.path().join("arch/x86/boot");
        std::fs::create_dir_all(&boot).unwrap();
        std::fs::write(boot.join("bzImage"), b"fake").unwrap();
        assert!(has_kernel_artifacts(tmp.path()));
    }

    #[test]
    fn kernel_path_has_artifacts_empty_dir() {
        let tmp = TempDir::new().unwrap();
        assert!(!has_kernel_artifacts(tmp.path()));
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
        assert!(has_kernel_artifacts(tmp.path()));
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

    #[test]
    fn kernel_path_find_image_release_none_matches_osrelease() {
        // The `/proc/sys/kernel/osrelease` path is hardcoded in
        // find_image and cannot be mocked, so the fallback can only
        // be verified by equivalence: read osrelease the way the
        // function does, then assert that find_image(None, None)
        // equals find_image(None, Some(<that value>)). Identical
        // post-`rel` logic in both calls means equal outputs prove
        // the None branch derived `rel` from osrelease (or both
        // short-circuited via resolve_kernel(None), which is also
        // a contract — no panic, no divergence).
        let host_release = std::fs::read_to_string("/proc/sys/kernel/osrelease")
            .expect("host /proc/sys/kernel/osrelease must be readable for this test")
            .trim()
            .to_string();
        assert!(
            !host_release.is_empty(),
            "/proc/sys/kernel/osrelease must be non-empty for this test",
        );

        let derived = find_image(None, None);
        let explicit = find_image(None, Some(&host_release));
        assert_eq!(
            derived, explicit,
            "find_image(None, None) must equal find_image(None, Some(osrelease)); fallback diverged",
        );
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
        assert_eq!(
            KernelId::Range {
                start: "6.10".to_string(),
                end: "6.13".to_string(),
            }
            .to_string(),
            "6.10..6.13",
        );
        assert_eq!(
            KernelId::Git {
                url: "https://example.com/r.git".to_string(),
                git_ref: "main".to_string(),
            }
            .to_string(),
            "git+https://example.com/r.git#main",
        );
    }

    // -- KernelId::parse — Range arm --

    #[test]
    fn kernel_id_parse_range_versions() {
        assert_eq!(
            KernelId::parse("6.10..6.15"),
            KernelId::Range {
                start: "6.10".to_string(),
                end: "6.15".to_string(),
            },
        );
    }

    #[test]
    fn kernel_id_parse_range_patch_versions() {
        assert_eq!(
            KernelId::parse("6.10.5..6.10.10"),
            KernelId::Range {
                start: "6.10.5".to_string(),
                end: "6.10.10".to_string(),
            },
        );
    }

    #[test]
    fn kernel_id_parse_range_rc() {
        assert_eq!(
            KernelId::parse("6.10..6.10-rc3"),
            KernelId::Range {
                start: "6.10".to_string(),
                end: "6.10-rc3".to_string(),
            },
        );
    }

    /// Both endpoints non-version: not a Range. The `/`-contains
    /// test fails too, so this falls to the version-shaped check
    /// (also fails on the `..`) and lands as CacheKey.
    #[test]
    fn kernel_id_parse_range_non_version_falls_through() {
        assert_eq!(
            KernelId::parse("foo..bar"),
            KernelId::CacheKey("foo..bar".to_string()),
        );
    }

    /// One endpoint version-shaped, the other not: the Range arm
    /// requires BOTH endpoints to pass `_is_version_string`, so
    /// `6.10..foo` falls through to CacheKey.
    #[test]
    fn kernel_id_parse_range_one_non_version() {
        assert_eq!(
            KernelId::parse("6.10..foo"),
            KernelId::CacheKey("6.10..foo".to_string()),
        );
    }

    /// Trailing `..` with no second endpoint: `_is_version_string("")`
    /// is false, so the Range arm doesn't fire. Falls to CacheKey
    /// (the version-shaped check also fails because the trailing `..`
    /// means a parts-iter sees an empty patch component).
    #[test]
    fn kernel_id_parse_range_empty_endpoint() {
        assert_eq!(
            KernelId::parse("6.10.."),
            KernelId::CacheKey("6.10..".to_string()),
        );
    }

    // -- KernelId::parse — Git arm --

    #[test]
    fn kernel_id_parse_git_branch() {
        assert_eq!(
            KernelId::parse("git+https://example.com/r.git#main"),
            KernelId::Git {
                url: "https://example.com/r.git".to_string(),
                git_ref: "main".to_string(),
            },
        );
    }

    #[test]
    fn kernel_id_parse_git_sha() {
        assert_eq!(
            KernelId::parse("git+https://example.com/r.git#abc1234"),
            KernelId::Git {
                url: "https://example.com/r.git".to_string(),
                git_ref: "abc1234".to_string(),
            },
        );
    }

    /// The Git arm splits on the LAST `#` so URL fragments survive
    /// inside the URL slot — `git+https://x#frag#main` parses as
    /// url=`https://x#frag`, git_ref=`main`. A future regression
    /// that swapped to `split_once('#')` would land here as a
    /// flipped URL/ref.
    #[test]
    fn kernel_id_parse_git_multi_hash_url() {
        assert_eq!(
            KernelId::parse("git+https://x#frag#main"),
            KernelId::Git {
                url: "https://x#frag".to_string(),
                git_ref: "main".to_string(),
            },
        );
    }

    /// Empty git_ref after the `#`: the Git arm requires a non-empty
    /// ref so it skips. The string still contains `/` (the URL's
    /// scheme separator and path), so the `/`-contains Path arm
    /// fires next and the value lands as a Path holding the literal
    /// `git+` spelling. That's a degenerate Path — the auto-build
    /// step will reject `git+...` as a non-existent directory at
    /// resolve time, surfacing the typo with a clear filesystem
    /// error instead of letting the Git arm swallow an empty ref.
    #[test]
    fn kernel_id_parse_git_empty_ref_falls_through() {
        assert_eq!(
            KernelId::parse("git+https://example.com/r.git#"),
            KernelId::Path(PathBuf::from("git+https://example.com/r.git#")),
        );
    }

    /// Empty URL before the `#`: the Git arm fails on the empty url
    /// check. Unlike the empty-ref case above, this string contains
    /// no `/`, so the Path arm doesn't fire either. `_is_version_string`
    /// fails on the leading `git+`, so it lands as CacheKey holding
    /// the literal spelling — which a downstream cache lookup will
    /// reject as a missing entry.
    #[test]
    fn kernel_id_parse_git_empty_url_falls_through() {
        assert_eq!(
            KernelId::parse("git+#main"),
            KernelId::CacheKey("git+#main".to_string()),
        );
    }

    /// `git+` prefix takes precedence over the `/`-contains Path
    /// test. A user pointing at a local clone via `git+/local/repo#v1`
    /// should get a Git, not a Path. This pins the parse-arm
    /// ordering — flipping the Path check above the Git check would
    /// land here as KernelId::Path("git+/local/repo#v1").
    #[test]
    fn kernel_id_parse_git_beats_path() {
        assert_eq!(
            KernelId::parse("git+/local/repo#v1"),
            KernelId::Git {
                url: "/local/repo".to_string(),
                git_ref: "v1".to_string(),
            },
        );
    }

    // -- KernelId::parse_list --

    #[test]
    fn kernel_id_parse_list_basic() {
        let list = KernelId::parse_list("6.10,6.13");
        assert_eq!(
            list,
            vec![
                KernelId::Version("6.10".to_string()),
                KernelId::Version("6.13".to_string()),
            ],
        );
    }

    #[test]
    fn kernel_id_parse_list_mixed() {
        let list = KernelId::parse_list("6.10,git+url#main,/srv/linux");
        assert_eq!(list.len(), 3, "expected 3 entries, got {list:?}");
        assert!(matches!(list[0], KernelId::Version(ref v) if v == "6.10"));
        assert!(matches!(
            list[1],
            KernelId::Git { ref url, ref git_ref } if url == "url" && git_ref == "main"
        ));
        assert!(matches!(list[2], KernelId::Path(ref p) if p == &PathBuf::from("/srv/linux")));
    }

    #[test]
    fn kernel_id_parse_list_empty() {
        assert_eq!(KernelId::parse_list(""), Vec::<KernelId>::new());
    }

    /// Trailing / leading / repeated commas are forgiving — empty
    /// segments are silently dropped so `,6.10,,` yields just one
    /// entry. Spec says: defer dedup to the resolver but do not
    /// inject empty Cache-key entries from an operator typo.
    #[test]
    fn kernel_id_parse_list_trailing_comma() {
        assert_eq!(
            KernelId::parse_list(",6.10,,"),
            vec![KernelId::Version("6.10".to_string())],
        );
    }

    /// Whitespace around comma-separated entries gets trimmed before
    /// `parse` runs so `"6.10 , 6.13"` produces clean Version variants
    /// rather than CacheKey entries with embedded spaces.
    #[test]
    fn kernel_id_parse_list_whitespace() {
        assert_eq!(
            KernelId::parse_list("6.10 , 6.13"),
            vec![
                KernelId::Version("6.10".to_string()),
                KernelId::Version("6.13".to_string()),
            ],
        );
    }

    /// A single-entry list with no commas falls through `split(',')`
    /// as one segment and produces the same Variant `parse` would
    /// have produced directly. Pins the parse_list/parse equivalence
    /// for the trivial case so a future regression that special-cased
    /// "must contain comma" lands here.
    #[test]
    fn kernel_id_parse_list_single() {
        assert_eq!(
            KernelId::parse_list("6.10"),
            vec![KernelId::Version("6.10".to_string())],
        );
    }

    /// Duplicate entries are PRESERVED at parse time — `parse_list`
    /// is a pure splitter, and dedup is the resolver's job (after
    /// canonicalization to a cache key, since `6.10` and `v6.10` and
    /// a tag pointing at the same sha all collapse). Pin the count
    /// AND the index of each occurrence so a future regression that
    /// added an early dedup at parse time (which would silently
    /// collapse `6.10,6.10` to one entry and lose the operator's
    /// "run twice" intent if they later added that semantic) lands
    /// here.
    #[test]
    fn kernel_id_parse_list_preserves_dups() {
        let list = KernelId::parse_list("6.10,6.10,6.13");
        assert_eq!(list.len(), 3, "expected 3 entries, got {list:?}");
        assert_eq!(list[0], KernelId::Version("6.10".to_string()));
        assert_eq!(list[1], KernelId::Version("6.10".to_string()));
        assert_eq!(list[2], KernelId::Version("6.13".to_string()));
    }

    // -- KernelId::validate — inverted-range rejection --

    /// Forward range `6.10..6.13` validates fine — the most common
    /// happy-path case, here as a baseline for the failure tests
    /// below.
    #[test]
    fn kernel_id_validate_range_forward_ok() {
        let id = KernelId::parse("6.10..6.13");
        assert!(id.validate().is_ok(), "forward range must validate: {id:?}");
    }

    /// Equal endpoints `6.10..6.10` validate fine — degenerate
    /// single-element range, not inverted.
    #[test]
    fn kernel_id_validate_range_equal_endpoints_ok() {
        let id = KernelId::parse("6.10..6.10");
        assert!(
            id.validate().is_ok(),
            "equal endpoints must validate: {id:?}"
        );
    }

    /// `6.16..6.12` — same major, minor decreases. Reject. The error
    /// message must name both endpoints AND suggest the swapped
    /// spelling so the operator can fix the typo without re-reading
    /// the help.
    #[test]
    fn kernel_id_validate_range_inverted_minor() {
        let id = KernelId::parse("6.16..6.12");
        let err = id.validate().unwrap_err();
        assert!(
            err.contains("inverted kernel range"),
            "error must say 'inverted kernel range', got: {err}",
        );
        assert!(
            err.contains("6.16..6.12"),
            "error must cite the spec, got: {err}"
        );
        assert!(
            err.contains("6.12..6.16"),
            "error must suggest the swapped form, got: {err}",
        );
    }

    /// `7.0..6.99` — major decreases. Reject.
    #[test]
    fn kernel_id_validate_range_inverted_major() {
        let id = KernelId::parse("7.0..6.99");
        assert!(id.validate().is_err(), "inverted major must reject: {id:?}");
    }

    /// `6.10.5..6.10.3` — same major.minor, patch decreases. Reject.
    #[test]
    fn kernel_id_validate_range_inverted_patch() {
        let id = KernelId::parse("6.10.5..6.10.3");
        assert!(id.validate().is_err(), "inverted patch must reject: {id:?}");
    }

    /// `6.10..6.10-rc3` — release > rc per the rc-as-MAX rule, so
    /// pre-release on the upper end is inverted. Reject. Catches the
    /// common operator mistake of "I want 6.10 latest stable up
    /// through the rc series" written in reverse order.
    #[test]
    fn kernel_id_validate_range_inverted_rc_below_release() {
        let id = KernelId::parse("6.10..6.10-rc3");
        assert!(
            id.validate().is_err(),
            "release > rc — `6.10..6.10-rc3` must reject: {id:?}",
        );
    }

    /// `6.10-rc3..6.10` — pre-release < release. Forward direction;
    /// validate passes. The companion to `inverted_rc_below_release`.
    #[test]
    fn kernel_id_validate_range_rc_below_release_forward_ok() {
        let id = KernelId::parse("6.10-rc3..6.10");
        assert!(
            id.validate().is_ok(),
            "rc < release — `6.10-rc3..6.10` must validate: {id:?}",
        );
    }

    /// `6.10-rc3..6.10-rc1` — same major.minor.patch but rc decreases.
    /// Reject. Pre-release ordering must follow numeric rcN order.
    #[test]
    fn kernel_id_validate_range_inverted_rc_to_rc() {
        let id = KernelId::parse("6.10-rc3..6.10-rc1");
        assert!(id.validate().is_err(), "rc3..rc1 must reject: {id:?}");
    }

    /// `6.10..6.10.5` — `6.10` decomposes to (6,10,0,MAX), `6.10.5`
    /// to (6,10,5,MAX). Forward direction. Validates.
    #[test]
    fn kernel_id_validate_range_missing_patch_treated_as_zero() {
        let id = KernelId::parse("6.10..6.10.5");
        assert!(
            id.validate().is_ok(),
            "missing patch defaults to 0, so `6.10..6.10.5` is forward: {id:?}",
        );
    }

    /// All non-Range variants validate trivially — Path, Version,
    /// CacheKey, Git all return Ok. Pins the "validate is currently
    /// only meaningful for Range" contract: a future field with its
    /// own resolve-time invariant should add an arm here, not slip
    /// through silently.
    #[test]
    fn kernel_id_validate_non_range_variants_ok() {
        assert!(KernelId::Version("6.14.2".to_string()).validate().is_ok());
        assert!(KernelId::CacheKey("my-key".to_string()).validate().is_ok());
        assert!(KernelId::Path(PathBuf::from("../linux")).validate().is_ok(),);
        assert!(
            KernelId::Git {
                url: "https://example.com/r.git".to_string(),
                git_ref: "main".to_string(),
            }
            .validate()
            .is_ok(),
        );
    }

    /// Direct construction with an unparseable `start` endpoint
    /// (callers that build `KernelId::Range` outside `KernelId::parse`
    /// can put any string in either slot — the Display round-trip
    /// gives them the spelling back, but `validate()` is the safety
    /// net for resolve-time legality). Asserts the error names the
    /// "not a parseable version" condition so a downstream tool can
    /// distinguish this from the inverted-range message above.
    #[test]
    fn kernel_id_validate_range_unparseable_start() {
        let id = KernelId::Range {
            start: "garbage".to_string(),
            end: "6.10".to_string(),
        };
        let err = id.validate().unwrap_err();
        assert!(
            err.contains("not a parseable version"),
            "error must say 'not a parseable version', got: {err}",
        );
        assert!(
            err.contains("garbage"),
            "error must cite the bad endpoint, got: {err}"
        );
    }

    /// Companion to `unparseable_start` for the `end` slot.
    #[test]
    fn kernel_id_validate_range_unparseable_end() {
        let id = KernelId::Range {
            start: "6.10".to_string(),
            end: "garbage".to_string(),
        };
        let err = id.validate().unwrap_err();
        assert!(
            err.contains("not a parseable version"),
            "error must say 'not a parseable version', got: {err}",
        );
        assert!(
            err.contains("garbage"),
            "error must cite the bad endpoint, got: {err}"
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

    use proptest::prop_assert;

    proptest::proptest! {
        /// Arbitrary input must parse into a `KernelId` variant whose
        /// payload round-trips to the original string where a
        /// round-trip is defined (Path / Version / CacheKey). Bumped
        /// the input range from 30 to 120 characters to exercise long
        /// paths and pathological multi-dot strings.
        #[test]
        fn prop_kernel_id_parse_never_panics(s in "\\PC{0,120}") {
            match KernelId::parse(&s) {
                KernelId::Path(p) => prop_assert!(p == s, "Path payload drift for {s:?}"),
                KernelId::Version(v) => prop_assert!(v == s, "Version payload drift for {s:?}"),
                KernelId::CacheKey(k) => prop_assert!(k == s, "CacheKey payload drift for {s:?}"),
                KernelId::Range { start, end } => {
                    // Range is constructed only when both endpoints
                    // are version-shaped, so the payload round-trips
                    // through the `start..end` rendering. Display
                    // emits the same separator the parser consumed.
                    prop_assert!(
                        format!("{start}..{end}") == s,
                        "Range payload drift for {s:?}",
                    );
                }
                KernelId::Git { url, git_ref } => {
                    // Git is constructed only on the `git+URL#REF`
                    // prefix branch with non-empty url and git_ref;
                    // round-trip the full prefix/separator shape.
                    prop_assert!(
                        format!("git+{url}#{git_ref}") == s,
                        "Git payload drift for {s:?}",
                    );
                }
            }
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
