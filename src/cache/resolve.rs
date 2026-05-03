//! Cache root resolution and source-tree path helpers.

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
