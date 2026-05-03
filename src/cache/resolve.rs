//! Cache root resolution and source-tree path helpers.
//!
//! Two responsibilities live here:
//!
//! 1. **Cache-root resolution.** [`resolve_cache_root_with_suffix`]
//!    runs the env cascade (`KTSTR_CACHE_DIR` → `$XDG_CACHE_HOME` →
//!    `$HOME/.cache`) that turns a per-cache `suffix` (e.g. `kernels`,
//!    `models`) into an absolute cache directory path. HOME validation
//!    ([`validate_home_for_cache`]) gates the third fallback so a
//!    suid-stripped or root-but-no-HOME process produces a clear error
//!    rather than writing into `/root/.cache/...` by accident.
//!    [`path_inside_cache_root`] is the cache-membership predicate used
//!    by callers (notably the model resolver and BTF probe) to decide
//!    whether a path on disk is "ours" before applying cache-aware
//!    invalidation.
//!
//! 2. **Source-tree DWARF re-routing.** [`prefer_source_tree_for_dwarf`]
//!    short-circuits the cache when the operator's cwd happens to be a
//!    kernel source tree whose vmlinux is full-fat (matching the cached
//!    one's identity). Combined with [`recover_local_source_tree`]
//!    (which reads `metadata.json` to find the canonical source path
//!    for a cached entry), this lets `cargo ktstr test` reuse the
//!    operator's tree directly and avoid stripping debug info from a
//!    perfectly good local vmlinux.
//!
//! Tests cover the env-cascade arms, HOME-rejection cases, the
//! cache-membership predicate's symlink-resolution semantics, the
//! DWARF preference policy, and metadata-driven source-tree recovery.

use std::fs;
use std::path::{Path, PathBuf};

use super::housekeeping::read_metadata;
use super::metadata::KernelSource;

/// Resolve the cache root directory path with a per-cache `suffix`
/// (`"kernels"` for the kernel cache, `"models"` for the model cache).
///
/// Resolution cascade:
/// 1. `KTSTR_CACHE_DIR` (with non-UTF-8 bail). The override returns
///    the path verbatim — no `suffix` is appended.
/// 2. `XDG_CACHE_HOME/ktstr/{suffix}` when set and non-empty.
/// 3. `$HOME/.cache/ktstr/{suffix}` after HOME validation.
///
/// Does not create the directory.
pub(crate) fn resolve_cache_root_with_suffix(suffix: &str) -> anyhow::Result<PathBuf> {
    match std::env::var("KTSTR_CACHE_DIR") {
        Ok(dir) if !dir.is_empty() => return Ok(PathBuf::from(dir)),
        Ok(_) => { /* empty string -> fall through to fallbacks */ }
        Err(std::env::VarError::NotPresent) => { /* unset -> fall through */ }
        Err(std::env::VarError::NotUnicode(raw)) => {
            anyhow::bail!(
                "KTSTR_CACHE_DIR contains non-UTF-8 bytes ({} bytes): {:?}. \
                 ktstr requires a UTF-8 cache path — set KTSTR_CACHE_DIR \
                 to an ASCII/UTF-8 directory (e.g. `/tmp/ktstr-cache`) or \
                 unset it to fall back to $XDG_CACHE_HOME/$HOME.",
                raw.len(),
                raw,
            );
        }
    }
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME")
        && !xdg.is_empty()
    {
        return Ok(PathBuf::from(xdg).join("ktstr").join(suffix));
    }
    let home = validate_home_for_cache()?;
    Ok(home.join(".cache").join("ktstr").join(suffix))
}

/// Read `HOME` from the environment, reject values that produce a
/// guaranteed-junk cache path, and return the validated `PathBuf`.
///
/// The function exists because the `$HOME/.cache/ktstr/...` fallback
/// is the LAST stage of [`resolve_cache_root_with_suffix`]'s cascade
/// — by the time we get here, neither `KTSTR_CACHE_DIR` nor
/// `XDG_CACHE_HOME` was set, so `HOME` is the only remaining input
/// that can name a cache root. A bad `HOME` here means the operator
/// has no working cache location at all, so we fail loudly with a
/// remediation hint rather than silently writing into `/.cache/...`
/// or a relative-to-cwd directory that breaks every subsequent
/// invocation from a different cwd.
///
/// # Rejection cases
///
/// 1. **HOME missing** (unset OR empty string). Both arms of the
///    `std::env::var("HOME")` match — `Err(NotPresent)` and
///    `Ok("")` — bail with the same family of remediation. Empty
///    is just as broken as unset because every PathBuf join gives
///    a relative path. Common cause: a Dockerfile / login init
///    that emits `export HOME=` or `ENV HOME=` with no value.
/// 2. **HOME = `/`.** Joining `/.cache/ktstr/<suffix>` aliases a
///    kernel-root path rather than a per-user cache. Common cause:
///    a root login that never set HOME, leaving it at the libc
///    default. Bail with a suggestion to use `KTSTR_CACHE_DIR` or
///    `XDG_CACHE_HOME`.
/// 3. **HOME is relative.** Anything that doesn't start with `/`.
///    A relative HOME would resolve against the operator's cwd
///    at every invocation, so the cache root would silently move
///    around — every test run would miss the previous build's
///    output. Bail with the same env-override hint.
///
/// On success, returns the validated absolute path as a `PathBuf`
/// for the caller to extend with `.cache/ktstr/<suffix>`.
pub(crate) fn validate_home_for_cache() -> anyhow::Result<PathBuf> {
    let home = match std::env::var("HOME") {
        Ok(v) if !v.is_empty() => v,
        Ok(_) => {
            anyhow::bail!(
                "HOME is set to the empty string; cannot resolve cache directory. \
                 An empty HOME usually means a Dockerfile or shell rc has \
                 `export HOME=` or `ENV HOME=` with no value. Either set HOME \
                 to a real absolute path, or set KTSTR_CACHE_DIR to an absolute \
                 path (e.g. /tmp/ktstr-cache) or XDG_CACHE_HOME to specify a \
                 cache location explicitly."
            );
        }
        Err(_) => {
            anyhow::bail!(
                "HOME is unset; cannot resolve cache directory. \
                 The container init or login shell did not assign HOME — set \
                 it to an absolute path, or set KTSTR_CACHE_DIR to an absolute \
                 path (e.g. /tmp/ktstr-cache) or XDG_CACHE_HOME to specify a \
                 cache location explicitly."
            );
        }
    };
    if home == "/" {
        anyhow::bail!(
            "HOME is `/`; the resulting cache path /.cache/ktstr would alias the \
             root filesystem rather than naming a user cache. This usually means \
             the process inherited HOME from a container init or root login that \
             did not set a real home. Set KTSTR_CACHE_DIR to an absolute path \
             (e.g. /tmp/ktstr-cache) or XDG_CACHE_HOME to bypass HOME entirely."
        );
    }
    if !home.starts_with('/') {
        anyhow::bail!(
            "HOME={home:?} is not an absolute path; ktstr requires HOME to start \
             with `/` so the cache root resolves consistently regardless of the \
             current working directory. Set HOME to an absolute path, or set \
             KTSTR_CACHE_DIR / XDG_CACHE_HOME to a specific cache location."
        );
    }
    Ok(PathBuf::from(home))
}

/// Resolve the kernel cache root directory path.
pub(crate) fn resolve_cache_root() -> anyhow::Result<PathBuf> {
    resolve_cache_root_with_suffix("kernels")
}

/// Re-route a cache-entry directory to its original source tree when
/// blazesym DWARF access is required. Validates the source-tree
/// vmlinux's current size and mtime against the values captured at
/// cache-store time.
pub fn prefer_source_tree_for_dwarf(dir: &Path) -> Option<PathBuf> {
    let metadata = read_metadata(dir).ok()?;
    let want_size = metadata.source_vmlinux_size?;
    let want_mtime = metadata.source_vmlinux_mtime_secs?;
    let KernelSource::Local {
        source_tree_path, ..
    } = metadata.source
    else {
        return None;
    };
    let src_path = source_tree_path?;
    let vmlinux = src_path.join("vmlinux");
    let stat = std::fs::metadata(&vmlinux).ok()?;
    if !stat.is_file() {
        return None;
    }
    if stat.len() != want_size {
        return None;
    }
    let cur_mtime = stat.modified().ok().and_then(|t| {
        t.duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .ok()
            .or_else(|| {
                std::time::UNIX_EPOCH
                    .duration_since(t)
                    .ok()
                    .map(|d| -(d.as_secs() as i64))
            })
    })?;
    if cur_mtime != want_mtime {
        return None;
    }
    Some(src_path)
}

/// Read `dir/metadata.json` and return the persisted source-tree
/// path when the entry was built from a local source tree.
pub fn recover_local_source_tree(dir: &Path) -> Option<PathBuf> {
    let metadata = read_metadata(dir).ok()?;
    if let KernelSource::Local {
        source_tree_path: Some(p),
        ..
    } = metadata.source
    {
        return Some(p);
    }
    None
}

/// Is `p` (a file path) located inside the kernel cache root?
pub(crate) fn path_inside_cache_root(p: &Path) -> bool {
    let root = match resolve_cache_root() {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!(
                err = %e,
                "cache root unresolvable; treating path as outside cache",
            );
            return false;
        }
    };
    let canon_root = match fs::canonicalize(&root) {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!(
                root = %root.display(),
                err = %e,
                "cache root canonicalize failed; treating path as outside cache",
            );
            return false;
        }
    };
    let parent = match p.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => return false,
    };
    let canon_parent = match fs::canonicalize(parent) {
        Ok(p) => p,
        Err(e) => {
            tracing::debug!(
                parent = %parent.display(),
                err = %e,
                "input path parent canonicalize failed; treating as outside cache",
            );
            return false;
        }
    };
    canon_parent.starts_with(&canon_root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
    use std::fs;
    use tempfile::TempDir;

    // -- resolve_cache_root --

    #[test]
    fn cache_resolve_root_ktstr_cache_dir() {
        let _lock = lock_env();
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("custom-cache");
        let _guard = EnvVarGuard::set("KTSTR_CACHE_DIR", &dir);
        let root = resolve_cache_root().unwrap();
        assert_eq!(root, dir);
    }

    #[test]
    fn cache_resolve_root_xdg_cache_home() {
        let _lock = lock_env();
        let tmp = TempDir::new().unwrap();
        let _guard1 = EnvVarGuard::remove("KTSTR_CACHE_DIR");
        let _guard2 = EnvVarGuard::set("XDG_CACHE_HOME", tmp.path());
        let root = resolve_cache_root().unwrap();
        assert_eq!(root, tmp.path().join("ktstr").join("kernels"));
    }

    #[test]
    fn cache_resolve_root_empty_ktstr_cache_dir_falls_through() {
        let _lock = lock_env();
        let tmp = TempDir::new().unwrap();
        let _guard1 = EnvVarGuard::set("KTSTR_CACHE_DIR", "");
        let _guard2 = EnvVarGuard::set("XDG_CACHE_HOME", tmp.path());
        let root = resolve_cache_root().unwrap();
        assert_eq!(root, tmp.path().join("ktstr").join("kernels"));
    }

    #[test]
    fn cache_resolve_root_empty_xdg_falls_to_home() {
        let _lock = lock_env();
        let tmp = TempDir::new().unwrap();
        let _guard1 = EnvVarGuard::remove("KTSTR_CACHE_DIR");
        let _guard2 = EnvVarGuard::set("XDG_CACHE_HOME", "");
        let _guard3 = EnvVarGuard::set("HOME", tmp.path());
        let root = resolve_cache_root().unwrap();
        assert_eq!(
            root,
            tmp.path().join(".cache").join("ktstr").join("kernels")
        );
    }

    // -- resolve_cache_root error paths --

    #[test]
    fn cache_resolve_root_home_unset_error() {
        let _lock = lock_env();
        let _guard1 = EnvVarGuard::remove("KTSTR_CACHE_DIR");
        let _guard2 = EnvVarGuard::remove("XDG_CACHE_HOME");
        let _guard3 = EnvVarGuard::remove("HOME");
        let err = resolve_cache_root().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("HOME is unset"),
            "expected HOME-unset error, got: {msg}"
        );
        assert!(
            !msg.contains("HOME is set to the empty string"),
            "unset HOME must NOT use the empty-string diagnostic — the two \
             cases are distinct now (NotPresent vs Ok(\"\")), got: {msg}",
        );
        assert!(
            msg.contains("KTSTR_CACHE_DIR"),
            "error should suggest KTSTR_CACHE_DIR, got: {msg}"
        );
    }

    #[test]
    fn cache_resolve_root_home_root_slash_error() {
        let _lock = lock_env();
        let _guard1 = EnvVarGuard::remove("KTSTR_CACHE_DIR");
        let _guard2 = EnvVarGuard::remove("XDG_CACHE_HOME");
        let _guard3 = EnvVarGuard::set("HOME", "/");
        let err = resolve_cache_root().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("HOME is `/`"),
            "expected HOME=/ specific error, got: {msg}"
        );
        assert!(
            msg.contains("/.cache/ktstr"),
            "diagnostic must cite the offending cache path, got: {msg}"
        );
        assert!(
            msg.contains("KTSTR_CACHE_DIR"),
            "error should suggest KTSTR_CACHE_DIR, got: {msg}"
        );
    }

    #[test]
    fn cache_resolve_root_home_empty_error() {
        let _lock = lock_env();
        let _guard1 = EnvVarGuard::remove("KTSTR_CACHE_DIR");
        let _guard2 = EnvVarGuard::remove("XDG_CACHE_HOME");
        let _guard3 = EnvVarGuard::set("HOME", "");
        let err = resolve_cache_root().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("HOME is set to the empty string"),
            "empty-HOME bail must use the empty-string diagnostic, got: {msg}",
        );
        assert!(
            !msg.contains("HOME is unset"),
            "empty-HOME must NOT use the unset diagnostic — the two \
             cases are distinct now, got: {msg}",
        );
    }

    #[test]
    fn cache_resolve_root_home_relative_path_error() {
        let _lock = lock_env();
        let _guard1 = EnvVarGuard::remove("KTSTR_CACHE_DIR");
        let _guard2 = EnvVarGuard::remove("XDG_CACHE_HOME");
        let _guard3 = EnvVarGuard::set("HOME", "relative/dir");
        let err = resolve_cache_root().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("not an absolute path"),
            "expected relative-path-specific error, got: {msg}"
        );
        assert!(
            msg.contains("relative/dir"),
            "diagnostic must cite the offending HOME value, got: {msg}"
        );
        assert!(
            msg.contains("KTSTR_CACHE_DIR"),
            "error should suggest KTSTR_CACHE_DIR, got: {msg}"
        );
    }

    #[test]
    fn cache_resolve_root_home_bare_name_relative_error() {
        let _lock = lock_env();
        let _guard1 = EnvVarGuard::remove("KTSTR_CACHE_DIR");
        let _guard2 = EnvVarGuard::remove("XDG_CACHE_HOME");
        let _guard3 = EnvVarGuard::set("HOME", "tmp");
        let err = resolve_cache_root().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("not an absolute path"),
            "expected relative-path-specific error, got: {msg}"
        );
        assert!(
            msg.contains("\"tmp\""),
            "diagnostic must cite the offending HOME value via its Debug \
             representation, got: {msg}"
        );
    }

    #[test]
    fn cache_resolve_root_home_absolute_passes() {
        let _lock = lock_env();
        let _guard1 = EnvVarGuard::remove("KTSTR_CACHE_DIR");
        let _guard2 = EnvVarGuard::remove("XDG_CACHE_HOME");
        let tmp = TempDir::new().expect("tempdir");
        let _guard3 = EnvVarGuard::set("HOME", tmp.path());
        let resolved = resolve_cache_root().expect("absolute HOME must resolve");
        let expected = tmp.path().join(".cache").join("ktstr").join("kernels");
        assert_eq!(
            resolved, expected,
            "absolute HOME must produce $HOME/.cache/ktstr/kernels",
        );
    }

    #[test]
    #[cfg(unix)]
    fn cache_resolve_root_non_utf8_ktstr_cache_dir_bails() {
        let _lock = lock_env();
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        let bytes: &[u8] = b"/tmp/ktstr-\xFFcache";
        let value = OsStr::from_bytes(bytes);
        let _guard = EnvVarGuard::set("KTSTR_CACHE_DIR", value);
        let err = resolve_cache_root()
            .expect_err("non-UTF-8 KTSTR_CACHE_DIR must bail, not silently fall through");
        let msg = err.to_string();
        assert!(
            msg.contains("KTSTR_CACHE_DIR"),
            "error must name the offending variable, got: {msg}",
        );
        assert!(
            msg.contains("non-UTF-8"),
            "error must mention non-UTF-8 so the operator knows the encoding, \
             got: {msg}",
        );
        assert!(
            msg.contains("UTF-8") || msg.contains("unset") || msg.contains("ASCII"),
            "error must name a remediation (UTF-8 replacement or unset), \
             got: {msg}",
        );
    }

    // -- path_inside_cache_root direct unit tests --

    #[test]
    fn path_inside_cache_root_accepts_path_inside() {
        let _lock = lock_env();
        let tmp = TempDir::new().unwrap();
        let _guard = EnvVarGuard::set("KTSTR_CACHE_DIR", tmp.path());
        let entry = tmp.path().join("kentry");
        std::fs::create_dir_all(&entry).unwrap();
        let vmlinux = entry.join("vmlinux");
        std::fs::write(&vmlinux, b"placeholder").unwrap();
        assert!(
            path_inside_cache_root(&vmlinux),
            "vmlinux directly under cache root must be classified as in-cache",
        );
    }

    #[test]
    fn path_inside_cache_root_rejects_path_outside() {
        let _lock = lock_env();
        let cache_root = TempDir::new().unwrap();
        let _guard = EnvVarGuard::set("KTSTR_CACHE_DIR", cache_root.path());
        let source_tree = TempDir::new().unwrap();
        let vmlinux = source_tree.path().join("vmlinux");
        std::fs::write(&vmlinux, b"placeholder").unwrap();
        assert!(
            !path_inside_cache_root(&vmlinux),
            "vmlinux in a sibling tempdir must NOT be classified as in-cache",
        );
    }

    #[test]
    fn path_inside_cache_root_rejects_bare_filename() {
        let _lock = lock_env();
        let tmp = TempDir::new().unwrap();
        let _guard = EnvVarGuard::set("KTSTR_CACHE_DIR", tmp.path());
        let bare = std::path::Path::new("vmlinux");
        assert!(
            !path_inside_cache_root(bare),
            "bare filename (no parent) must short-circuit to false",
        );
    }

    #[test]
    fn path_inside_cache_root_false_when_unresolvable() {
        let _lock = lock_env();
        let _g1 = EnvVarGuard::remove("KTSTR_CACHE_DIR");
        let _g2 = EnvVarGuard::remove("XDG_CACHE_HOME");
        let _g3 = EnvVarGuard::remove("HOME");
        let dir = TempDir::new().unwrap();
        let f = dir.path().join("vmlinux");
        std::fs::write(&f, b"x").unwrap();
        assert!(
            !path_inside_cache_root(&f),
            "unresolvable cache root must classify as outside-cache (false)",
        );
    }

    #[test]
    fn path_inside_cache_root_false_when_parent_canonicalize_fails() {
        let _lock = lock_env();
        let tmp = TempDir::new().unwrap();
        let _guard = EnvVarGuard::set("KTSTR_CACHE_DIR", tmp.path());
        let nonexistent = std::path::Path::new("/this/parent/should/not/exist/vmlinux");
        assert!(
            !nonexistent.parent().unwrap().exists(),
            "precondition: parent must not exist for the canonicalize \
             failure path to be exercised",
        );
        assert!(
            !path_inside_cache_root(nonexistent),
            "nonexistent parent must surface as outside-cache, not panic",
        );
    }

    #[test]
    #[cfg(unix)]
    fn path_inside_cache_root_follows_symlink_into_cache() {
        let _lock = lock_env();
        let cache_root = TempDir::new().unwrap();
        let _guard = EnvVarGuard::set("KTSTR_CACHE_DIR", cache_root.path());
        let entry = cache_root.path().join("kentry");
        std::fs::create_dir_all(&entry).unwrap();
        let real = entry.join("vmlinux");
        std::fs::write(&real, b"placeholder").unwrap();
        let outside = TempDir::new().unwrap();
        let alias_parent = outside.path().join("alias");
        std::os::unix::fs::symlink(&entry, &alias_parent).unwrap();
        let through_alias = alias_parent.join("vmlinux");
        assert!(
            through_alias.exists(),
            "precondition: path through symlinked parent must be reachable",
        );
        assert!(
            path_inside_cache_root(&through_alias),
            "path whose parent symlink resolves into cache must classify as in-cache",
        );
    }

    #[test]
    #[cfg(unix)]
    fn path_inside_cache_root_follows_symlink_out_of_cache() {
        let _lock = lock_env();
        let cache_root = TempDir::new().unwrap();
        let _guard = EnvVarGuard::set("KTSTR_CACHE_DIR", cache_root.path());
        let outside = TempDir::new().unwrap();
        let real = outside.path().join("vmlinux");
        std::fs::write(&real, b"placeholder").unwrap();
        let alias_parent = cache_root.path().join("alias");
        std::os::unix::fs::symlink(outside.path(), &alias_parent).unwrap();
        let through_alias = alias_parent.join("vmlinux");
        assert!(
            through_alias.exists(),
            "precondition: path through symlinked parent must be reachable",
        );
        assert!(
            !path_inside_cache_root(&through_alias),
            "path whose parent symlink resolves OUT of cache must classify as outside-cache",
        );
    }

    #[test]
    fn path_inside_cache_root_empty_ktstr_cache_dir_falls_through() {
        let _lock = lock_env();
        let tmp = TempDir::new().unwrap();
        let _g1 = EnvVarGuard::set("KTSTR_CACHE_DIR", "");
        let _g2 = EnvVarGuard::set("XDG_CACHE_HOME", tmp.path());
        let resolved = tmp.path().join("ktstr").join("kernels");
        let entry = resolved.join("kentry");
        std::fs::create_dir_all(&entry).unwrap();
        let vmlinux = entry.join("vmlinux");
        std::fs::write(&vmlinux, b"placeholder").unwrap();
        assert!(
            path_inside_cache_root(&vmlinux),
            "with empty KTSTR_CACHE_DIR, the cascade must resolve via \
             XDG_CACHE_HOME and accept paths inside that resolved root",
        );
    }

    #[test]
    fn path_inside_cache_root_fresh_resolution_per_call() {
        let _lock = lock_env();
        let cache_a = TempDir::new().unwrap();
        let cache_b = TempDir::new().unwrap();
        let entry_a = cache_a.path().join("kentry");
        std::fs::create_dir_all(&entry_a).unwrap();
        let vmlinux_a = entry_a.join("vmlinux");
        std::fs::write(&vmlinux_a, b"placeholder").unwrap();
        {
            let _guard = EnvVarGuard::set("KTSTR_CACHE_DIR", cache_a.path());
            assert!(
                path_inside_cache_root(&vmlinux_a),
                "first call: vmlinux is inside cache_a (the active root)",
            );
        }
        {
            let _guard = EnvVarGuard::set("KTSTR_CACHE_DIR", cache_b.path());
            assert!(
                !path_inside_cache_root(&vmlinux_a),
                "second call: KTSTR_CACHE_DIR has moved to cache_b, so the \
                 vmlinux (still under cache_a) must be classified outside",
            );
        }
    }

    // -- prefer_source_tree_for_dwarf --

    #[test]
    fn prefer_source_tree_local_with_vmlinux() {
        let tmp = TempDir::new().unwrap();
        let cache_entry = tmp.path().join("cache");
        let src_tree = tmp.path().join("src");
        fs::create_dir_all(&cache_entry).unwrap();
        fs::create_dir_all(&src_tree).unwrap();
        let vmlinux = src_tree.join("vmlinux");
        fs::write(&vmlinux, b"fake-elf").unwrap();
        let stat = fs::metadata(&vmlinux).unwrap();
        let mtime_secs = stat
            .modified()
            .unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let meta = crate::cache::KernelMetadata {
            version: Some("6.14.2".to_string()),
            source: KernelSource::Local {
                source_tree_path: Some(src_tree.clone()),
                git_hash: None,
            },
            arch: "x86_64".to_string(),
            image_name: "bzImage".to_string(),
            config_hash: None,
            built_at: "2026-04-18T10:00:00Z".to_string(),
            ktstr_kconfig_hash: None,
            extra_kconfig_hash: None,
            has_vmlinux: true,
            vmlinux_stripped: true,
            source_vmlinux_size: Some(stat.len()),
            source_vmlinux_mtime_secs: Some(mtime_secs),
        };
        fs::write(
            cache_entry.join("metadata.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        assert_eq!(prefer_source_tree_for_dwarf(&cache_entry), Some(src_tree));
    }

    #[test]
    fn prefer_source_tree_local_without_vmlinux_in_tree() {
        let tmp = TempDir::new().unwrap();
        let cache_entry = tmp.path().join("cache");
        let src_tree = tmp.path().join("src");
        fs::create_dir_all(&cache_entry).unwrap();
        fs::create_dir_all(&src_tree).unwrap();

        let meta = crate::cache::KernelMetadata {
            version: None,
            source: KernelSource::Local {
                source_tree_path: Some(src_tree),
                git_hash: None,
            },
            arch: "x86_64".to_string(),
            image_name: "bzImage".to_string(),
            config_hash: None,
            built_at: "2026-04-18T10:00:00Z".to_string(),
            ktstr_kconfig_hash: None,
            extra_kconfig_hash: None,
            has_vmlinux: false,
            vmlinux_stripped: false,
            source_vmlinux_size: Some(42),
            source_vmlinux_mtime_secs: Some(1_700_000_000),
        };
        fs::write(
            cache_entry.join("metadata.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        assert_eq!(prefer_source_tree_for_dwarf(&cache_entry), None);
    }

    #[test]
    fn prefer_source_tree_tarball_source_returns_none() {
        let tmp = TempDir::new().unwrap();
        let cache_entry = tmp.path().join("cache");
        fs::create_dir_all(&cache_entry).unwrap();

        let meta = crate::cache::KernelMetadata {
            version: Some("6.14.2".to_string()),
            source: KernelSource::Tarball,
            arch: "x86_64".to_string(),
            image_name: "bzImage".to_string(),
            config_hash: None,
            built_at: "2026-04-18T10:00:00Z".to_string(),
            ktstr_kconfig_hash: None,
            extra_kconfig_hash: None,
            has_vmlinux: true,
            vmlinux_stripped: true,
            source_vmlinux_size: None,
            source_vmlinux_mtime_secs: None,
        };
        fs::write(
            cache_entry.join("metadata.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        assert_eq!(prefer_source_tree_for_dwarf(&cache_entry), None);
    }

    #[test]
    fn prefer_source_tree_no_metadata_returns_none() {
        let tmp = TempDir::new().unwrap();
        assert_eq!(prefer_source_tree_for_dwarf(tmp.path()), None);
    }

    #[test]
    fn prefer_source_tree_metadata_parse_failure_returns_none() {
        let tmp = TempDir::new().unwrap();
        let cache_entry = tmp.path().join("cache");
        fs::create_dir_all(&cache_entry).unwrap();
        fs::write(
            cache_entry.join("metadata.json"),
            br#"{"not_kernel_metadata": true}"#,
        )
        .unwrap();

        assert_eq!(
            prefer_source_tree_for_dwarf(&cache_entry),
            None,
            "malformed metadata.json must short-circuit to None, not bail",
        );

        let other_entry = tmp.path().join("other");
        fs::create_dir_all(&other_entry).unwrap();
        fs::write(other_entry.join("metadata.json"), b"not json at all {{{").unwrap();
        assert_eq!(
            prefer_source_tree_for_dwarf(&other_entry),
            None,
            "unparseable metadata.json must short-circuit to None, not bail",
        );
    }

    #[test]
    fn prefer_source_tree_local_with_none_source_tree_path_returns_none() {
        let tmp = TempDir::new().unwrap();
        let cache_entry = tmp.path().join("cache");
        fs::create_dir_all(&cache_entry).unwrap();

        let meta = crate::cache::KernelMetadata {
            version: Some("6.14.2".to_string()),
            source: KernelSource::Local {
                source_tree_path: None,
                git_hash: Some("abc123".to_string()),
            },
            arch: "x86_64".to_string(),
            image_name: "bzImage".to_string(),
            config_hash: None,
            built_at: "2026-04-18T10:00:00Z".to_string(),
            ktstr_kconfig_hash: None,
            extra_kconfig_hash: None,
            has_vmlinux: true,
            vmlinux_stripped: true,
            source_vmlinux_size: Some(42),
            source_vmlinux_mtime_secs: Some(1_700_000_000),
        };
        fs::write(
            cache_entry.join("metadata.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        assert_eq!(
            prefer_source_tree_for_dwarf(&cache_entry),
            None,
            "Local entry with source_tree_path=None must short-circuit \
             to None at the `let src_path = source_tree_path?;` line",
        );
    }

    #[test]
    fn prefer_source_tree_validates_matching_vmlinux_stat_and_returns_path() {
        let tmp = TempDir::new().unwrap();
        let cache_entry = tmp.path().join("cache");
        let src_tree = tmp.path().join("src");
        fs::create_dir_all(&cache_entry).unwrap();
        fs::create_dir_all(&src_tree).unwrap();
        let vmlinux = src_tree.join("vmlinux");
        fs::write(&vmlinux, b"fake-elf-bytes").unwrap();
        let stat = fs::metadata(&vmlinux).unwrap();
        let mtime_secs = stat
            .modified()
            .unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let meta = crate::cache::KernelMetadata {
            version: None,
            source: KernelSource::Local {
                source_tree_path: Some(src_tree.clone()),
                git_hash: None,
            },
            arch: "x86_64".to_string(),
            image_name: "bzImage".to_string(),
            config_hash: None,
            built_at: "2026-04-18T10:00:00Z".to_string(),
            ktstr_kconfig_hash: None,
            extra_kconfig_hash: None,
            has_vmlinux: true,
            vmlinux_stripped: true,
            source_vmlinux_size: Some(stat.len()),
            source_vmlinux_mtime_secs: Some(mtime_secs),
        };
        fs::write(
            cache_entry.join("metadata.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        assert_eq!(
            prefer_source_tree_for_dwarf(&cache_entry),
            Some(src_tree),
            "matching size + mtime must pass the validation gate"
        );
    }

    #[test]
    fn prefer_source_tree_size_mismatch_returns_none() {
        let tmp = TempDir::new().unwrap();
        let cache_entry = tmp.path().join("cache");
        let src_tree = tmp.path().join("src");
        fs::create_dir_all(&cache_entry).unwrap();
        fs::create_dir_all(&src_tree).unwrap();
        let vmlinux = src_tree.join("vmlinux");
        fs::write(&vmlinux, b"fake-elf-bytes").unwrap();
        let stat = fs::metadata(&vmlinux).unwrap();
        let mtime_secs = stat
            .modified()
            .unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let meta = crate::cache::KernelMetadata {
            version: None,
            source: KernelSource::Local {
                source_tree_path: Some(src_tree),
                git_hash: None,
            },
            arch: "x86_64".to_string(),
            image_name: "bzImage".to_string(),
            config_hash: None,
            built_at: "2026-04-18T10:00:00Z".to_string(),
            ktstr_kconfig_hash: None,
            extra_kconfig_hash: None,
            has_vmlinux: true,
            vmlinux_stripped: true,
            source_vmlinux_size: Some(stat.len() + 1),
            source_vmlinux_mtime_secs: Some(mtime_secs),
        };
        fs::write(
            cache_entry.join("metadata.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        assert_eq!(
            prefer_source_tree_for_dwarf(&cache_entry),
            None,
            "size mismatch must drop validation and return None"
        );
    }

    #[test]
    fn prefer_source_tree_mtime_mismatch_returns_none() {
        let tmp = TempDir::new().unwrap();
        let cache_entry = tmp.path().join("cache");
        let src_tree = tmp.path().join("src");
        fs::create_dir_all(&cache_entry).unwrap();
        fs::create_dir_all(&src_tree).unwrap();
        let vmlinux = src_tree.join("vmlinux");
        fs::write(&vmlinux, b"fake-elf-bytes").unwrap();
        let stat = fs::metadata(&vmlinux).unwrap();
        let mtime_secs = stat
            .modified()
            .unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let meta = crate::cache::KernelMetadata {
            version: None,
            source: KernelSource::Local {
                source_tree_path: Some(src_tree),
                git_hash: None,
            },
            arch: "x86_64".to_string(),
            image_name: "bzImage".to_string(),
            config_hash: None,
            built_at: "2026-04-18T10:00:00Z".to_string(),
            ktstr_kconfig_hash: None,
            extra_kconfig_hash: None,
            has_vmlinux: true,
            vmlinux_stripped: true,
            source_vmlinux_size: Some(stat.len()),
            source_vmlinux_mtime_secs: Some(mtime_secs - 3600),
        };
        fs::write(
            cache_entry.join("metadata.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        assert_eq!(
            prefer_source_tree_for_dwarf(&cache_entry),
            None,
            "mtime mismatch must drop validation and return None"
        );
    }

    // -- recover_local_source_tree --

    #[test]
    fn recover_local_source_tree_local_with_path_returns_source_tree() {
        let tmp = TempDir::new().unwrap();
        let cache_entry = tmp.path().join("cache");
        let src_tree = tmp.path().join("src");
        fs::create_dir_all(&cache_entry).unwrap();
        fs::create_dir_all(&src_tree).unwrap();

        let meta = crate::cache::KernelMetadata {
            version: Some("6.14.2".to_string()),
            source: KernelSource::Local {
                source_tree_path: Some(src_tree.clone()),
                git_hash: Some("abc1234".to_string()),
            },
            arch: "x86_64".to_string(),
            image_name: "bzImage".to_string(),
            config_hash: None,
            built_at: "2026-04-18T10:00:00Z".to_string(),
            ktstr_kconfig_hash: None,
            extra_kconfig_hash: None,
            has_vmlinux: false,
            vmlinux_stripped: false,
            source_vmlinux_size: None,
            source_vmlinux_mtime_secs: None,
        };
        fs::write(
            cache_entry.join("metadata.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        assert_eq!(recover_local_source_tree(&cache_entry), Some(src_tree));
    }

    #[test]
    fn recover_local_source_tree_no_metadata_returns_none() {
        let tmp = TempDir::new().unwrap();
        assert_eq!(recover_local_source_tree(tmp.path()), None);
    }

    #[test]
    fn recover_local_source_tree_tarball_source_returns_none() {
        let tmp = TempDir::new().unwrap();
        let cache_entry = tmp.path().join("cache");
        fs::create_dir_all(&cache_entry).unwrap();

        let meta = crate::cache::KernelMetadata {
            version: Some("6.14.2".to_string()),
            source: KernelSource::Tarball,
            arch: "x86_64".to_string(),
            image_name: "bzImage".to_string(),
            config_hash: None,
            built_at: "2026-04-18T10:00:00Z".to_string(),
            ktstr_kconfig_hash: None,
            extra_kconfig_hash: None,
            has_vmlinux: true,
            vmlinux_stripped: true,
            source_vmlinux_size: None,
            source_vmlinux_mtime_secs: None,
        };
        fs::write(
            cache_entry.join("metadata.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        assert_eq!(recover_local_source_tree(&cache_entry), None);
    }

    #[test]
    fn recover_local_source_tree_local_with_none_path_returns_none() {
        let tmp = TempDir::new().unwrap();
        let cache_entry = tmp.path().join("cache");
        fs::create_dir_all(&cache_entry).unwrap();

        let meta = crate::cache::KernelMetadata {
            version: Some("6.14.2".to_string()),
            source: KernelSource::Local {
                source_tree_path: None,
                git_hash: Some("abc1234".to_string()),
            },
            arch: "x86_64".to_string(),
            image_name: "bzImage".to_string(),
            config_hash: None,
            built_at: "2026-04-18T10:00:00Z".to_string(),
            ktstr_kconfig_hash: None,
            extra_kconfig_hash: None,
            has_vmlinux: true,
            vmlinux_stripped: true,
            source_vmlinux_size: None,
            source_vmlinux_mtime_secs: None,
        };
        fs::write(
            cache_entry.join("metadata.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        assert_eq!(recover_local_source_tree(&cache_entry), None);
    }

    #[test]
    fn recover_local_source_tree_malformed_metadata_returns_none() {
        let tmp = TempDir::new().unwrap();
        let cache_entry = tmp.path().join("cache");
        fs::create_dir_all(&cache_entry).unwrap();
        fs::write(
            cache_entry.join("metadata.json"),
            br#"{"not_kernel_metadata": true}"#,
        )
        .unwrap();
        assert_eq!(recover_local_source_tree(&cache_entry), None);
    }

    // -- validate_home_for_cache direct unit tests --

    #[test]
    fn validate_home_for_cache_rejects_unset() {
        let _env_lock = lock_env();
        let _home = EnvVarGuard::remove("HOME");
        let err = validate_home_for_cache().expect_err("unset HOME must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("HOME is unset"),
            "diagnostic must call out the unset case specifically: {msg}",
        );
        assert!(
            !msg.contains("HOME is set to the empty string"),
            "unset HOME must NOT use the empty-string diagnostic — the two \
             cases are distinct now (NotPresent vs Ok(\"\")): {msg}",
        );
    }

    #[test]
    fn validate_home_for_cache_rejects_empty() {
        let _env_lock = lock_env();
        let _home = EnvVarGuard::set("HOME", "");
        let err = validate_home_for_cache().expect_err("empty HOME must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("HOME is set to the empty string"),
            "diagnostic must call out the empty-string case specifically: {msg}",
        );
        assert!(
            !msg.contains("HOME is unset"),
            "empty HOME must NOT use the unset diagnostic — the two \
             cases are distinct now: {msg}",
        );
    }

    #[test]
    fn validate_home_for_cache_rejects_root_slash() {
        let _env_lock = lock_env();
        let _home = EnvVarGuard::set("HOME", "/");
        let err = validate_home_for_cache().expect_err("HOME=/ must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("HOME is `/`"),
            "diagnostic must call out the root-slash case specifically: {msg}",
        );
        assert!(
            msg.contains("/.cache/ktstr"),
            "diagnostic must explain why (/.cache/ktstr aliases root fs): {msg}",
        );
    }

    #[test]
    fn validate_home_for_cache_rejects_relative_path() {
        let _env_lock = lock_env();
        for rel in ["relative", "./relative", "home/user", "."] {
            let _home = EnvVarGuard::set("HOME", rel);
            let err = validate_home_for_cache()
                .expect_err(&format!("relative path '{rel}' must be rejected"));
            let msg = format!("{err:#}");
            assert!(
                msg.contains("not an absolute path"),
                "[rel={rel:?}] diagnostic must call out non-absolute: {msg}",
            );
            assert!(
                msg.contains(&format!("{rel:?}")),
                "[rel={rel:?}] diagnostic must echo the offending value verbatim: {msg}",
            );
        }
    }

    #[test]
    fn validate_home_for_cache_accepts_absolute_paths() {
        let _env_lock = lock_env();
        for ok in [
            "/home/user",
            "/var/empty",
            "/root",
            "/a",
            "/home/user with spaces",
            "/home/user/.local/share",
        ] {
            let _home = EnvVarGuard::set("HOME", ok);
            let got = validate_home_for_cache()
                .unwrap_or_else(|e| panic!("absolute path {ok:?} must be accepted; got: {e:#}"));
            assert_eq!(
                got,
                std::path::PathBuf::from(ok),
                "returned PathBuf must equal the HOME value verbatim — \
                 helper does not append the cache suffix or canonicalize",
            );
        }
    }

    #[test]
    fn validate_home_for_cache_does_not_canonicalize_dots_and_doubles() {
        let _env_lock = lock_env();
        for not_normalized in ["//", "/./", "/.", "/foo//bar", "/./home"] {
            let _home = EnvVarGuard::set("HOME", not_normalized);
            validate_home_for_cache().unwrap_or_else(|e| {
                panic!(
                    "non-normalized but absolute path {not_normalized:?} must \
                     pass the helper (downstream OS surfaces the diagnostic); \
                     got: {e:#}",
                )
            });
        }
    }
}
