//! Atomic-rename install primitives, cache-key validators, and
//! orphan-tempdir sweep for the kernel image cache.

use std::fs;
use std::path::Path;

use super::TMP_DIR_PREFIX;
use super::metadata::KernelMetadata;

/// Rejects empty keys, whitespace-only keys, keys starting with
/// `.tmp-` (reserved for in-progress stores), and keys containing
/// path separators (`/`, `\`), parent-directory traversal (`..`),
/// or null bytes. Returns `Ok(())` on valid keys.
pub(crate) fn validate_cache_key(key: &str) -> anyhow::Result<()> {
    if key.is_empty() || key.trim().is_empty() {
        anyhow::bail!("cache key must not be empty or whitespace-only");
    }
    if key.contains('/') || key.contains('\\') {
        anyhow::bail!("cache key must not contain path separators: {key:?}");
    }
    if key == "." || key == ".." {
        anyhow::bail!("cache key must not be a directory reference: {key:?}");
    }
    if key.contains("..") {
        anyhow::bail!("cache key must not contain path traversal: {key:?}");
    }
    if key.contains('\0') {
        anyhow::bail!("cache key must not contain null bytes");
    }
    if key.starts_with(TMP_DIR_PREFIX) {
        anyhow::bail!("cache key must not start with {TMP_DIR_PREFIX} (reserved): {key:?}",);
    }
    Ok(())
}

/// Validate a filename (e.g. image_name in metadata).
pub(crate) fn validate_filename(name: &str) -> anyhow::Result<()> {
    if name.is_empty() {
        anyhow::bail!("image name must not be empty");
    }
    if name.contains('/') || name.contains('\\') {
        anyhow::bail!("image name must not contain path separators: {name:?}");
    }
    if name.contains("..") {
        anyhow::bail!("image name must not contain path traversal: {name:?}");
    }
    if name.contains('\0') {
        anyhow::bail!("image name must not contain null bytes");
    }
    Ok(())
}

/// RAII guard that removes a temporary directory on drop.
pub(crate) struct TmpDirGuard<'a>(pub(crate) &'a Path);

impl Drop for TmpDirGuard<'_> {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(self.0);
    }
}

/// Atomically swap two filesystem paths via renameat2(RENAME_EXCHANGE).
pub(crate) fn atomic_swap_dirs(src: &Path, dst: &Path) -> anyhow::Result<()> {
    rustix::fs::renameat_with(
        rustix::fs::CWD,
        src,
        rustix::fs::CWD,
        dst,
        rustix::fs::RenameFlags::EXCHANGE,
    )
    .map_err(|e| {
        anyhow::anyhow!(
            "renameat2(RENAME_EXCHANGE) {} <-> {}: {e}",
            src.display(),
            dst.display(),
        )
    })
}

/// Read and deserialize metadata.json from a cache entry directory.
///
/// On failure returns a human-readable reason with a distinct prefix
/// per failure mode (missing / unreadable / schema-drift / malformed
/// / truncated / parse error). Prefix consumers key on
/// [`super::metadata::classify_corrupt_reason`].
pub(crate) fn read_metadata(dir: &Path) -> Result<KernelMetadata, String> {
    let meta_path = dir.join("metadata.json");
    let contents = match fs::read_to_string(&meta_path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err("metadata.json missing".to_string());
        }
        Err(e) => return Err(format!("metadata.json unreadable: {e}")),
    };
    serde_json::from_str(&contents).map_err(|e| match e.classify() {
        serde_json::error::Category::Data => format!("metadata.json schema drift: {e}"),
        serde_json::error::Category::Syntax => format!("metadata.json malformed: {e}"),
        serde_json::error::Category::Eof => format!("metadata.json truncated: {e}"),
        serde_json::error::Category::Io => {
            tracing::error!(
                err = %e,
                "serde_json::from_str returned Category::Io — unexpected for in-memory input",
            );
            format!("metadata.json parse error: {e}")
        }
    })
}

/// Scan `cache_root` for `.tmp-{key}-{pid}` directories whose `{pid}`
/// is no longer a live process and remove them.
///
/// Cross-PID orphan sweep. `kill(pid, None)` returning `Err(ESRCH)`
/// is the only outcome that justifies removal; alive / EPERM
/// preserve.
pub(crate) fn clean_orphaned_tmp_dirs(cache_root: &Path) -> anyhow::Result<()> {
    if !cache_root.is_dir() {
        return Ok(());
    }
    let read_dir = match fs::read_dir(cache_root) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => anyhow::bail!("read cache root {}: {e}", cache_root.display()),
    };
    for dir_entry in read_dir {
        let dir_entry = match dir_entry {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(err = %format!("{e:#}"), "skip unreadable cache root entry");
                continue;
            }
        };
        let name = match dir_entry.file_name().into_string() {
            Ok(n) => n,
            Err(_) => continue,
        };
        if !name.starts_with(TMP_DIR_PREFIX) {
            continue;
        }
        let pid_str = match name.rsplit_once('-') {
            Some((_, suffix)) if !suffix.is_empty() => suffix,
            _ => continue,
        };
        let pid: i32 = match pid_str.parse() {
            Ok(p) => p,
            Err(_) => continue,
        };
        if pid <= 0 {
            continue;
        }
        let dead = matches!(
            nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None),
            Err(nix::errno::Errno::ESRCH),
        );
        if !dead {
            continue;
        }
        let path = dir_entry.path();
        match fs::remove_dir_all(&path) {
            Ok(()) => {
                tracing::info!(
                    path = %path.display(),
                    orphan_pid = pid,
                    "cleaned orphaned .tmp- dir from prior crashed process",
                );
            }
            Err(e) => {
                tracing::warn!(
                    err = %format!("{e:#}"),
                    path = %path.display(),
                    "failed to remove orphaned .tmp- dir; leaving in place",
                );
            }
        }
    }
    Ok(())
}
