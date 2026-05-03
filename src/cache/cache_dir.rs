//! [`CacheDir`] handle, lock guards, and cache-lock timeout policy.
//!
//! Reader/writer asymmetry: shared (reader) lock blocks 10 s,
//! exclusive (writer) lock blocks 60 s. Writer must outlast every
//! concurrent test reader; reader bails fast on a stuck writer.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Context;

use super::housekeeping::{
    TmpDirGuard, atomic_swap_dirs, clean_orphaned_tmp_dirs, read_metadata, validate_cache_key,
    validate_filename,
};
use super::metadata::{CacheArtifacts, CacheEntry, KernelMetadata, ListedEntry};
use super::resolve::resolve_cache_root;
use super::vmlinux_strip::strip_vmlinux_debug;
use super::{LOCK_DIR_NAME, TMP_DIR_PREFIX};
use crate::flock::{FlockMode, acquire_flock_with_timeout};

/// Default wall-clock timeout for [`CacheDir::acquire_shared_lock`].
const SHARED_LOCK_DEFAULT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Timeout for [`CacheDir::store`]'s internal `LOCK_EX` acquire.
const STORE_EXCLUSIVE_LOCK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// Handle to the kernel image cache directory.
#[derive(Debug)]
#[non_exhaustive]
pub struct CacheDir {
    root: PathBuf,
}

/// Emit a per-lookup warning when a cache entry was created with an
/// unstripped vmlinux.
fn warn_if_unstripped_vmlinux(entry: &CacheEntry) {
    if should_warn_unstripped(entry) {
        eprintln!(
            "cache: using unstripped vmlinux for {} (strip failed on a prior build; \
             re-run with a clean cache to retry)",
            entry.key,
        );
    }
}

/// Pure decision logic for [`warn_if_unstripped_vmlinux`].
pub(crate) fn should_warn_unstripped(entry: &CacheEntry) -> bool {
    entry.metadata.has_vmlinux() && !entry.metadata.vmlinux_stripped()
}

impl CacheDir {
    /// Open a cache directory at the resolved root path.
    pub fn new() -> anyhow::Result<Self> {
        let root = resolve_cache_root()?;
        Ok(CacheDir { root })
    }

    /// Open a cache directory at a specific path.
    pub fn with_root(root: PathBuf) -> Self {
        CacheDir { root }
    }

    /// Resolve the default cache root path without side effects.
    pub fn default_root() -> anyhow::Result<PathBuf> {
        resolve_cache_root()
    }

    /// Root directory this `CacheDir` is anchored at.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Look up a cached kernel by cache key.
    pub fn lookup(&self, cache_key: &str) -> Option<CacheEntry> {
        if let Err(e) = validate_cache_key(cache_key) {
            tracing::warn!("invalid cache key: {e}");
            return None;
        }
        let entry_dir = self.root.join(cache_key);
        if !entry_dir.is_dir() {
            return None;
        }
        let metadata = read_metadata(&entry_dir).ok()?;
        if !entry_dir.join(&metadata.image_name).exists() {
            return None;
        }
        let entry = CacheEntry {
            key: cache_key.to_string(),
            path: entry_dir,
            metadata,
        };
        warn_if_unstripped_vmlinux(&entry);
        Some(entry)
    }

    /// List all cached kernel entries, sorted by build time (newest
    /// first).
    pub fn list(&self) -> anyhow::Result<Vec<ListedEntry>> {
        let mut entries: Vec<ListedEntry> = Vec::new();
        let read_dir = match fs::read_dir(&self.root) {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(entries),
            Err(e) => return Err(e.into()),
        };
        for dir_entry in read_dir {
            let dir_entry = dir_entry?;
            let path = dir_entry.path();
            let file_name = dir_entry.file_name();
            let name_hint = file_name.to_string_lossy();
            // Skip dotfile children — `.locks/` and `.tmp-*` are
            // reserved for ktstr's own bookkeeping.
            if name_hint.starts_with('.') {
                continue;
            }
            if !path.is_dir() {
                continue;
            }
            let name = match dir_entry.file_name().into_string() {
                Ok(n) => n,
                Err(_) => continue,
            };
            if name.starts_with(TMP_DIR_PREFIX) {
                continue;
            }
            match read_metadata(&path) {
                Ok(metadata) => {
                    let image_path = path.join(&metadata.image_name);
                    if image_path.exists() {
                        entries.push(ListedEntry::Valid(Box::new(CacheEntry {
                            key: name,
                            path,
                            metadata,
                        })));
                    } else {
                        entries.push(ListedEntry::Corrupt {
                            key: name,
                            path,
                            reason: format!(
                                "image file {} missing from entry directory",
                                metadata.image_name
                            ),
                        });
                    }
                }
                Err(reason) => {
                    tracing::info!(
                        entry = %name,
                        path = %path.display(),
                        %reason,
                        "cache entry corrupt at list-time",
                    );
                    entries.push(ListedEntry::Corrupt {
                        key: name,
                        path,
                        reason,
                    });
                }
            }
        }
        entries.sort_by(|a, b| {
            let a_time = a.as_valid().map(|e| e.metadata.built_at.as_str());
            let b_time = b.as_valid().map(|e| e.metadata.built_at.as_str());
            b_time.cmp(&a_time)
        });
        Ok(entries)
    }

    /// Store a kernel image in the cache. Atomic install via temp
    /// directory + `renameat2(RENAME_EXCHANGE)`.
    pub fn store(
        &self,
        cache_key: &str,
        artifacts: &CacheArtifacts<'_>,
        metadata: &KernelMetadata,
    ) -> anyhow::Result<CacheEntry> {
        validate_cache_key(cache_key)?;
        validate_filename(&metadata.image_name)?;

        let _store_lock =
            self.acquire_exclusive_lock_blocking(cache_key, STORE_EXCLUSIVE_LOCK_TIMEOUT)?;

        let final_dir = self.root.join(cache_key);
        let tmp_dir = self.root.join(format!(
            "{TMP_DIR_PREFIX}{}-{}",
            cache_key,
            std::process::id(),
        ));

        if tmp_dir.exists() {
            fs::remove_dir_all(&tmp_dir)?;
        }
        if let Err(e) = clean_orphaned_tmp_dirs(&self.root) {
            tracing::warn!(err = %format!("{e:#}"), "clean_orphaned_tmp_dirs failed; continuing store");
        }
        fs::create_dir_all(&tmp_dir)?;

        let _guard = TmpDirGuard(&tmp_dir);

        let image_dest = tmp_dir.join(&metadata.image_name);
        fs::copy(artifacts.image, &image_dest)
            .map_err(|e| anyhow::anyhow!("copy kernel image to cache: {e}"))?;

        let (has_vmlinux, vmlinux_stripped) = if let Some(vmlinux) = artifacts.vmlinux {
            let vmlinux_dest = tmp_dir.join("vmlinux");
            match strip_vmlinux_debug(vmlinux) {
                Ok(stripped) => {
                    fs::copy(stripped.path(), &vmlinux_dest)
                        .map_err(|e| anyhow::anyhow!("copy stripped vmlinux to cache: {e}"))?;
                    (true, true)
                }
                Err(e) => {
                    eprintln!(
                        "cache: vmlinux strip failed for {cache_key} ({e:#}); \
                         caching unstripped (larger on-disk payload). \
                         See `ktstr cache list --json` vmlinux_stripped field.",
                    );
                    tracing::warn!(
                        cache_key = cache_key,
                        err = %format!("{e:#}"),
                        "vmlinux strip failed, caching unstripped",
                    );
                    fs::copy(vmlinux, &vmlinux_dest)
                        .map_err(|e| anyhow::anyhow!("copy vmlinux to cache: {e}"))?;
                    (true, false)
                }
            }
        } else {
            (false, false)
        };

        let mut meta = metadata.clone();
        meta.set_has_vmlinux(has_vmlinux);
        meta.set_vmlinux_stripped(vmlinux_stripped);
        let meta_json = serde_json::to_string_pretty(&meta)?;
        fs::write(tmp_dir.join("metadata.json"), meta_json)
            .map_err(|e| anyhow::anyhow!("write cache metadata: {e}"))?;

        match fs::rename(&tmp_dir, &final_dir) {
            Ok(()) => {}
            Err(e)
                if e.raw_os_error() == Some(libc::ENOTEMPTY)
                    || e.raw_os_error() == Some(libc::EEXIST) =>
            {
                atomic_swap_dirs(&tmp_dir, &final_dir)?;
            }
            Err(e) => {
                return Err(anyhow::anyhow!("atomic rename cache entry: {e}"));
            }
        }

        Ok(CacheEntry {
            key: cache_key.to_string(),
            path: final_dir,
            metadata: meta,
        })
    }

    /// Remove every cached entry. Returns the number of entries
    /// removed. Preserves the `.locks/` subdirectory.
    pub fn clean_all(&self) -> anyhow::Result<usize> {
        self.remove_entries(self.list()?)
    }

    /// Remove every cached entry except the `keep` most recent ones
    /// (by `built_at` timestamp). Preserves the `.locks/`
    /// subdirectory.
    pub fn clean_keep(&self, keep: usize) -> anyhow::Result<usize> {
        self.remove_entries(self.list()?.into_iter().skip(keep))
    }

    fn remove_entries<I: IntoIterator<Item = ListedEntry>>(
        &self,
        iter: I,
    ) -> anyhow::Result<usize> {
        let to_remove: Vec<_> = iter.into_iter().collect();
        let count = to_remove.len();
        for entry in &to_remove {
            fs::remove_dir_all(entry.path())?;
        }
        Ok(count)
    }

    // ---------------- Per-entry coordination locks ----------------

    /// Absolute path to the coordination lockfile for `cache_key`.
    pub(crate) fn lock_path(&self, cache_key: &str) -> PathBuf {
        self.root
            .join(LOCK_DIR_NAME)
            .join(format!("{cache_key}.lock"))
    }

    /// Create the `{cache_root}/.locks/` subdirectory if absent.
    pub(crate) fn ensure_lock_dir(&self) -> anyhow::Result<()> {
        let dir = self.root.join(LOCK_DIR_NAME);
        fs::create_dir_all(&dir)
            .with_context(|| format!("create lock subdirectory {}", dir.display()))
    }

    /// Acquire `LOCK_SH` on the cache-entry lockfile.
    pub fn acquire_shared_lock(&self, cache_key: &str) -> anyhow::Result<SharedLockGuard> {
        validate_cache_key(cache_key)?;
        let path = self.lock_path(cache_key);
        let fd = acquire_flock_with_timeout(
            &path,
            FlockMode::Shared,
            SHARED_LOCK_DEFAULT_TIMEOUT,
            &format!("cache entry {cache_key:?}"),
            None,
        )?;
        Ok(SharedLockGuard { fd })
    }

    /// Acquire `LOCK_EX` on the cache-entry lockfile, blocking up
    /// to `timeout`.
    pub fn acquire_exclusive_lock_blocking(
        &self,
        cache_key: &str,
        timeout: std::time::Duration,
    ) -> anyhow::Result<ExclusiveLockGuard> {
        validate_cache_key(cache_key)?;
        let path = self.lock_path(cache_key);
        let fd = acquire_flock_with_timeout(
            &path,
            FlockMode::Exclusive,
            timeout,
            &format!("cache entry {cache_key:?}"),
            None,
        )?;
        Ok(ExclusiveLockGuard { fd })
    }

    /// Non-blocking `LOCK_EX` attempt on the cache-entry lockfile.
    pub fn try_acquire_exclusive_lock(
        &self,
        cache_key: &str,
    ) -> anyhow::Result<ExclusiveLockGuard> {
        validate_cache_key(cache_key)?;
        self.ensure_lock_dir()?;
        let path = self.lock_path(cache_key);
        match crate::flock::try_flock(&path, crate::flock::FlockMode::Exclusive)? {
            Some(fd) => Ok(ExclusiveLockGuard { fd }),
            None => {
                let holders = crate::flock::read_holders(&path).unwrap_or_default();
                anyhow::bail!(
                    "cache entry {cache_key:?} is locked by active test runs \
                     (lockfile {lockfile}, holders: {holders}). Wait for \
                     those tests to finish, or kill them, then retry.",
                    lockfile = path.display(),
                    holders = crate::flock::format_holder_list(&holders),
                );
            }
        }
    }
}

/// RAII guard for a `LOCK_SH` hold on a cache-entry lockfile.
#[derive(Debug)]
pub struct SharedLockGuard {
    #[allow(dead_code)]
    fd: std::os::fd::OwnedFd,
}

/// RAII guard for a `LOCK_EX` hold on a cache-entry lockfile.
#[derive(Debug)]
pub struct ExclusiveLockGuard {
    #[allow(dead_code)]
    fd: std::os::fd::OwnedFd,
}
