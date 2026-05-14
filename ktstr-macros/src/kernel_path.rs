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
    /// - `START..=END` or `START..END` where both endpoints are
    ///   version-shaped → [`KernelId::Range`]. The endpoints are
    ///   ALWAYS inclusive — both `..` and `..=` spellings produce a
    ///   closed range, regardless of Rust's exclusive-`..` /
    ///   inclusive-`..=` distinction. Both forms are accepted so test
    ///   authors and CLI users can write whichever feels natural.
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
        if let Some((start, end)) = s.split_once("..=")
            && _is_version_string(start)
            && _is_version_string(end)
        {
            return KernelId::Range {
                start: start.to_string(),
                end: end.to_string(),
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
            return KernelId::Path(expand_tilde(s));
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
/// (`6.16..6.12`, `6.10..6.10-rc3`, `7.0..6.99`), and by
/// [`crate::cli`]'s range-expansion helper to filter and sort
/// kernel.org release rows that fall inside a `start..end` interval.
pub(crate) fn decompose_version_for_compare(s: &str) -> Option<(u64, u64, u64, u64)> {
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

/// Expand a leading `~` or `~/...` in `s` against `$HOME` and
/// return the resulting [`PathBuf`]. Any other shape (no leading
/// `~`, `~user/...` for a different user, `$HOME` unset or empty)
/// passes through verbatim — the caller's downstream `is_dir()`
/// surfaces a regular "no such directory" error instead of being
/// silently rewritten.
///
/// Cases handled:
/// - `"~"` → `$HOME`
/// - `"~/"` → `$HOME/`
/// - `"~/linux"` → `$HOME/linux`
/// - `"~user/..."` → unchanged (std has no `getpwnam`; a
///   different-user expansion would require shelling out, which
///   the file's "no non-std imports outside cfg(test)" rule
///   forbids; the operator who wants a peer's home dir can spell
///   it absolutely)
/// - any input not starting with `~` → unchanged
/// - `~`-prefix with `$HOME` unset / empty → unchanged (the
///   downstream `is_dir()` failure is the clearest error path
///   we can produce without a logging dep)
///
/// Pure with respect to filesystem writes; reads `$HOME` once. Env
/// reads are consistent with the existing
/// [`kernel_release_from_procfs`] pattern (FS read at resolve time)
/// and explicitly outside the file-header `std::env::set_var` ban.
///
/// Called from [`KernelId::parse`]'s Path arm so the Path variant
/// stores an absolute (or filesystem-resolvable) path. Without this,
/// `KernelId::parse("~/linux")` stores the literal `"~/linux"`,
/// which `is_dir()` rejects unconditionally — there is no shell to
/// perform the standard tilde expansion on the operator's behalf at
/// CLI invocation time.
fn expand_tilde(s: &str) -> std::path::PathBuf {
    // Bare `~` and `~/...` are the only shapes we expand. Anything
    // else falls through verbatim.
    if s != "~" && !s.starts_with("~/") {
        return std::path::PathBuf::from(s);
    }
    // `$HOME` empty or unset is treated identically to "no
    // expansion possible" — the caller's `is_dir()` check will
    // surface the missing-path error normally. We do NOT panic
    // here because `KernelId::parse` is `pub` and on a hot CLI
    // path; failing to expand a single arg is not a fatal
    // condition for the whole CLI.
    let home = match std::env::var("HOME") {
        Ok(h) if !h.is_empty() => h,
        _ => return std::path::PathBuf::from(s),
    };
    if s == "~" {
        return std::path::PathBuf::from(home);
    }
    // s starts with "~/", so the suffix we want to splice on is
    // the slice starting AFTER the `/` separator. Joining `home`
    // with `&s[1..]` would land an absolute path inside `home`
    // (PathBuf::push of an absolute path RESETS the buffer to that
    // absolute path), so we strip the leading `/` first. Doubled
    // separators in the rest portion (`~//foo` → `s[2..] = "/foo"`)
    // would also reset the buffer; loop the strip so any run of
    // leading `/`s is consumed before the push.
    let mut rest = &s[2..]; // skip "~/"
    while let Some(stripped) = rest.strip_prefix('/') {
        rest = stripped;
    }
    let mut p = std::path::PathBuf::from(home);
    if !rest.is_empty() {
        p.push(rest);
    }
    p
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
