//! [`CacheDir`] handle, lock guards, and cache-lock timeout policy.
//!
//! Public surface: [`CacheDir`] (the operator-facing handle exposed
//! via `crate::cache::CacheDir`), [`SharedLockGuard`] /
//! [`ExclusiveLockGuard`] (RAII wrappers around per-key flock
//! acquisitions), and the [`CacheDir::store`] /
//! [`CacheDir::lookup`] / [`CacheDir::list`] /
//! [`CacheDir::clean`] lifecycle methods. The internal
//! `warn_if_unstripped_vmlinux` and `should_warn_unstripped`
//! helpers gate a per-lookup warning on entries whose vmlinux
//! sidecar took the strip-failure fallback in
//! [`super::vmlinux_strip::strip_vmlinux_debug`].
//!
//! Sibling modules:
//! - [`super::metadata`] — pure types ([`KernelSource`],
//!   [`KernelMetadata`], [`CacheArtifacts`], [`KconfigStatus`],
//!   [`CacheEntry`], [`ListedEntry`]) plus the
//!   [`super::metadata::classify_corrupt_reason`] dispatcher and
//!   [`super::metadata::format_image_missing_reason`] helper that
//!   `list` uses to emit corrupt-entry reason strings.
//! - [`super::housekeeping`] — atomic-rename install primitives
//!   ([`super::housekeeping::atomic_swap_dirs`],
//!   [`super::housekeeping::TmpDirGuard`]), cache-key /
//!   filename validators, the JSON metadata reader
//!   ([`super::housekeeping::read_metadata`]), and the cross-PID
//!   orphan-tempdir sweep
//!   ([`super::housekeeping::clean_orphaned_tmp_dirs`]).
//! - [`super::vmlinux_strip`] — the ELF strip pipeline
//!   ([`super::vmlinux_strip::strip_vmlinux_debug`]) `store()`
//!   invokes when an artifact carries a vmlinux sidecar.
//! - [`super::resolve`] — env-cascade root resolution that
//!   `CacheDir::new` and `CacheDir::default_root` flow through.
//!
//! Reader/writer asymmetry: shared (reader) lock blocks 10 s,
//! exclusive (writer) lock blocks 5 minutes (overridable via
//! the [`STORE_EXCLUSIVE_LOCK_TIMEOUT_ENV`] environment variable).
//! Writer must outlast every concurrent test reader; reader bails
//! fast on a stuck writer. See [`SHARED_LOCK_DEFAULT_TIMEOUT`] and
//! [`STORE_EXCLUSIVE_LOCK_DEFAULT_TIMEOUT`] for the literal
//! durations and their rationale.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Context;

use super::housekeeping::{
    TmpDirGuard, atomic_swap_dirs, clean_orphaned_tmp_dirs, read_metadata, validate_cache_key,
    validate_filename,
};
use super::metadata::{
    CacheArtifacts, CacheEntry, KconfigStatus, KernelMetadata, ListedEntry,
    format_image_missing_reason,
};
use super::resolve::resolve_cache_root;
use super::vmlinux_strip::strip_vmlinux_debug;
use super::{LOCK_DIR_NAME, TMP_DIR_PREFIX};
use crate::flock::{FlockMode, acquire_flock_with_timeout};

/// Default wall-clock timeout for [`CacheDir::acquire_shared_lock`].
const SHARED_LOCK_DEFAULT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Default timeout for [`CacheDir::store`]'s internal `LOCK_EX`
/// acquire when [`STORE_EXCLUSIVE_LOCK_TIMEOUT_ENV`] is unset.
///
/// 5 minutes covers a `store` peer's full critical section in the
/// worst case: under heavy parallelism N concurrent runners may
/// contend on the SAME `cache_key`, where the head writer holds
/// `LOCK_EX` while it copies the boot image, runs the 3-stage
/// vmlinux strip pipeline ([`super::vmlinux_strip::strip_vmlinux_debug`]),
/// writes `metadata.json`, and finishes the
/// [`super::housekeeping::atomic_swap_dirs`] swap. A real vmlinux
/// strip on a debug-symbol-rich build can spend tens of seconds
/// inside the strip pipeline alone, and stacking N peers in series
/// behind that producer scales the wait linearly. 60 s was tight
/// enough that 5–10 contending peers reliably timed out before
/// the head writer finished. The new 5-minute default leaves
/// headroom for ~50 contending peers behind a slow strip without
/// losing the "fail loud rather than block forever" property of a
/// finite timeout.
const STORE_EXCLUSIVE_LOCK_DEFAULT_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(300);

/// Environment variable name that overrides
/// [`STORE_EXCLUSIVE_LOCK_DEFAULT_TIMEOUT`]. Parsed via
/// [`humantime::parse_duration`] so operators can tune with
/// human-readable units (`30s`, `2m`, `10min`, `1h`). An invalid
/// value falls back to the default with a `warn!` so a typo never
/// silently disables the lock — the operator can see the
/// fall-through in their tracing output and fix the setting.
const STORE_EXCLUSIVE_LOCK_TIMEOUT_ENV: &str = "KTSTR_CACHE_STORE_LOCK_TIMEOUT";

/// Resolve the per-store `LOCK_EX` acquire timeout, honoring the
/// [`STORE_EXCLUSIVE_LOCK_TIMEOUT_ENV`] override. Pure function so
/// tests can exercise the parse/fall-through branches without
/// driving a full `store()` cycle.
fn store_exclusive_lock_timeout() -> std::time::Duration {
    match std::env::var(STORE_EXCLUSIVE_LOCK_TIMEOUT_ENV) {
        Ok(v) if !v.is_empty() => match humantime::parse_duration(&v) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(
                    env = %STORE_EXCLUSIVE_LOCK_TIMEOUT_ENV,
                    value = %v,
                    err = %e,
                    "invalid cache-store lock timeout env value; \
                     falling back to default timeout",
                );
                STORE_EXCLUSIVE_LOCK_DEFAULT_TIMEOUT
            }
        },
        _ => STORE_EXCLUSIVE_LOCK_DEFAULT_TIMEOUT,
    }
}

/// Handle to the kernel image cache directory.
#[derive(Debug)]
#[non_exhaustive]
pub struct CacheDir {
    root: PathBuf,
}

/// Emit a per-lookup warning when a cache entry was created with an
/// unstripped vmlinux.
///
/// Uses [`tracing::warn!`] so the message routes through the same
/// observability pipeline as every other cache-layer diagnostic
/// (the cargo-ktstr binary's `tracing_subscriber::fmt` writes warns
/// to stderr; library consumers can subscribe a different layer).
/// `eprintln!` would bypass that pipeline and force every consumer
/// to live with raw-stderr output regardless of their tracing
/// configuration.
fn warn_if_unstripped_vmlinux(entry: &CacheEntry) {
    if should_warn_unstripped(entry) {
        tracing::warn!(
            cache_key = %entry.key,
            "cache: using unstripped vmlinux (strip failed on a prior build; \
             re-run with a clean cache to retry)",
        );
    }
}

/// Pure decision logic for [`warn_if_unstripped_vmlinux`].
pub(crate) fn should_warn_unstripped(entry: &CacheEntry) -> bool {
    entry.metadata.has_vmlinux() && !entry.metadata.vmlinux_stripped()
}

/// Whether the existing `cached` cache entry already satisfies a
/// caller's intent to `store` an artifact under the same cache key.
///
/// Pure decision logic for [`CacheDir::store`]'s in-lock re-lookup
/// (step 3 of the docs). When N concurrent peers race on the same
/// `cache_key` they all miss the pre-lock cache check, serialise
/// behind `LOCK_EX`, and would otherwise each repeat the head
/// writer's copy / strip / atomic-publish work. This predicate
/// answers the post-lock question: "is the head writer's output
/// byte-equivalent to what I'd publish?" If yes, the late peers
/// short-circuit — only the head writer pays the publish cost.
///
/// Compares only the metadata fields that drive the on-disk bytes
/// `store()` would write:
///
/// - `config_hash` (CRC32 of the final `.config`) — pins the
///   kernel image identity.
/// - `ktstr_kconfig_hash` (CRC32 of `ktstr.kconfig`) — kconfig
///   fragment that produced the build.
/// - `extra_kconfig_hash` (CRC32 of the user `--extra-kconfig`
///   fragment) — same.
/// - `caller_has_vmlinux` — whether the caller passed a vmlinux
///   sidecar in `CacheArtifacts`. This is the actual switch
///   `store()` keys on (it overwrites `metadata.has_vmlinux`
///   from the artifacts argument), so the predicate compares
///   against the artifacts shape, not the caller's metadata
///   field.
///
/// Excludes:
///
/// - `built_at` — wall-clock timestamp that drifts every build;
///   pinning it would break the early-return and serialise every
///   peer through a redundant publish.
/// - `version` — display-only string, not a byte-difference.
/// - `source` — acquire-time provenance (Tarball / Git / Local +
///   payload). Two peers may publish the same image under
///   different `source` payloads (e.g. one from a tarball mirror,
///   one from a git checkout) and still produce byte-equivalent
///   bytes. The kconfig hash is the authoritative content key.
/// - `arch`, `image_name` — fixed by the cache key shape.
/// - `vmlinux_stripped` — set by `store()` based on
///   strip pipeline success/failure, not caller intent. The head
///   writer either succeeded (stripped) or fell back (unstripped);
///   late peers would just observe the head writer's outcome.
/// - `source_vmlinux_size`, `source_vmlinux_mtime_secs` —
///   DWARF-routing hints, not cached content.
///
/// Pure function so a unit test can pin every accept/reject branch
/// without driving a full `store()` cycle through a temp cache.
pub(crate) fn cache_content_matches(
    cached: &KernelMetadata,
    caller: &KernelMetadata,
    caller_has_vmlinux: bool,
) -> bool {
    cached.config_hash == caller.config_hash
        && cached.ktstr_kconfig_hash == caller.ktstr_kconfig_hash
        && cached.extra_kconfig_hash == caller.extra_kconfig_hash
        && cached.has_vmlinux() == caller_has_vmlinux
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
                            reason: format_image_missing_reason(&metadata.image_name),
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

    /// Store a kernel image (and optional vmlinux sidecar) in the
    /// cache under `cache_key`. Atomic install via temp directory +
    /// `renameat2(RENAME_EXCHANGE)`, so a concurrent reader never
    /// observes a partially-written entry.
    ///
    /// # Steps (in order)
    ///
    /// 1. **Validate inputs.** [`validate_cache_key`] rejects
    ///    `..`, slashes, NUL, and the `TMP_DIR_PREFIX` reservation;
    ///    [`validate_filename`] rejects path-separator characters in
    ///    the image basename. Invalid input fails before any I/O.
    /// 2. **Acquire the per-key store lock.** `LOCK_EX` on
    ///    `<root>/.locks/<cache_key>.lock`. Timeout defaults to
    ///    [`STORE_EXCLUSIVE_LOCK_DEFAULT_TIMEOUT`] (5 minutes) and
    ///    can be overridden via [`STORE_EXCLUSIVE_LOCK_TIMEOUT_ENV`]
    ///    for environments where a slow vmlinux strip stacks many
    ///    contending peers behind the head writer. The lock
    ///    excludes other writers for the same key while letting
    ///    readers and writers for unrelated keys proceed. Timeout
    ///    produces an error rather than blocking forever — a hung
    ///    writer cannot indefinitely block a fresh rebuild attempt.
    /// 3. **Double-checked re-lookup inside the lock.** After
    ///    acquiring `LOCK_EX`, re-run [`Self::lookup`] for
    ///    `cache_key`. When N peers race to publish the same key
    ///    they all miss the pre-lock cache check, queue on
    ///    `LOCK_EX`, and serialise behind the head writer. Without
    ///    this recheck, every peer re-runs the full copy + strip +
    ///    publish steps in series even though the head writer's
    ///    output already satisfies them. The recheck early-returns
    ///    when the existing cached entry's content-defining metadata
    ///    fields ([`cache_content_matches`] — config_hash,
    ///    ktstr_kconfig_hash, extra_kconfig_hash, has_vmlinux) match
    ///    the caller's intent for this publish, so only the head
    ///    writer pays the strip/copy/rename cost. Cache-relevant
    ///    differences (a fresh kconfig hash, a different vmlinux
    ///    presence) bypass the early-return and proceed to a real
    ///    overwrite-publish. Cache-irrelevant differences (a fresh
    ///    `built_at` timestamp, a different `version` display
    ///    string) trigger the early-return — the on-disk bytes the
    ///    overwrite would write are byte-equivalent to what's
    ///    already cached, so the publish is redundant.
    /// 4. **Stage into a temp directory.** `<root>/.tmp-<key>-<pid>`
    ///    is created (or pruned and recreated if a previous attempt
    ///    by the same PID exists), with [`TmpDirGuard`] enrolling the
    ///    path for cleanup on any subsequent error. A best-effort
    ///    [`clean_orphaned_tmp_dirs`] pass also runs here so dead
    ///    sibling temp directories from crashed PIDs are GC'd before
    ///    we add another one.
    /// 5. **Copy the boot image.** `metadata.image_name` lands at
    ///    `tmp/<image_name>` via `fs::copy`.
    /// 6. **Strip and copy vmlinux (if supplied).** When
    ///    `artifacts.vmlinux` is `Some`, [`strip_vmlinux_debug`]
    ///    runs the 3-stage strip pipeline and the result is written
    ///    to `tmp/vmlinux`. **Strip-fallback rationale:** if the
    ///    strip pipeline returns an error (e.g. an unrecognised ELF
    ///    layout from a future toolchain or an exotic config), the
    ///    write does NOT abort — it falls back to copying the raw
    ///    unstripped vmlinux and records `vmlinux_stripped: false`
    ///    in metadata. The cache trades a much larger on-disk
    ///    payload for "still usable for monitoring/probes," and
    ///    `cargo ktstr kernel list --json` exposes the
    ///    `vmlinux_stripped` field so operators can spot entries
    ///    that need rebuilding once the strip-failure root cause is
    ///    fixed. A hard failure here would be worse: it would
    ///    effectively brick the cache for that build.
    /// 7. **Write `metadata.json`.** A pretty-printed serde dump of
    ///    `KernelMetadata` (with `has_vmlinux` and `vmlinux_stripped`
    ///    set from step 6) at `tmp/metadata.json`. Pretty-print is
    ///    intentional — operators inspect this file directly when
    ///    debugging cache state.
    /// 8. **Atomic publish.** `fs::rename(tmp → final)` if `final`
    ///    does not exist; otherwise [`atomic_swap_dirs`] uses
    ///    `renameat2(RENAME_EXCHANGE)` to swap the two directories
    ///    in a single atomic syscall. Either way, no reader observes
    ///    a partial entry; the swap path also cleans up the
    ///    now-stale prior version under the temp name.
    pub fn store(
        &self,
        cache_key: &str,
        artifacts: &CacheArtifacts<'_>,
        metadata: &KernelMetadata,
    ) -> anyhow::Result<CacheEntry> {
        validate_cache_key(cache_key)?;
        validate_filename(&metadata.image_name)?;

        let _store_lock =
            self.acquire_exclusive_lock_blocking(cache_key, store_exclusive_lock_timeout())?;

        // Double-checked re-lookup inside LOCK_EX: when N peers race
        // on the same cache_key they all miss the pre-lock cache
        // check, queue on the lock, and would otherwise repeat the
        // head writer's copy/strip/publish work in series. The
        // recheck early-returns when the existing entry's
        // content-defining metadata fields match what we'd publish
        // (see [`cache_content_matches`] for the predicate). The
        // matched entry is returned to the caller verbatim — its
        // on-disk bytes are byte-equivalent to what we would write,
        // so no overwrite-publish is needed.
        if let Some(existing) = self.lookup(cache_key)
            && cache_content_matches(&existing.metadata, metadata, artifacts.vmlinux.is_some())
        {
            tracing::debug!(
                cache_key = cache_key,
                "cache.store: in-lock recheck hit; skipping copy/strip/publish",
            );
            return Ok(existing);
        }

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
    /// to `timeout`. On timeout, the error message surfaces the
    /// [`STORE_EXCLUSIVE_LOCK_TIMEOUT_ENV`] override so an operator
    /// hitting a contended `store()` discovers the env-var
    /// remediation without reading the docs.
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
            Some("override the timeout via KTSTR_CACHE_STORE_LOCK_TIMEOUT (humantime: 30s, 2m, 1h)"),
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

#[cfg(test)]
mod tests {
    use super::super::shared_test_helpers::{create_fake_image, test_metadata};
    use super::*;
    use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

    // -- CacheDir --

    #[test]
    fn cache_dir_with_root_does_not_create_dir() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("kernels");
        assert!(!root.exists());
        let cache = CacheDir::with_root(root.clone());
        assert!(!root.exists());
        assert_eq!(cache.root(), root);
    }

    #[test]
    fn cache_dir_list_returns_empty_for_nonexistent_root() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("never-created");
        assert!(!root.exists());
        let cache = CacheDir::with_root(root);
        let entries = cache.list().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn cache_dir_store_creates_root_lazily() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("lazy-root");
        assert!(!root.exists());
        let cache = CacheDir::with_root(root.clone());
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");
        cache
            .store("key", &CacheArtifacts::new(&image), &meta)
            .unwrap();
        assert!(root.exists(), "store() must create the cache root");
    }

    #[test]
    fn cache_dir_default_root_returns_path() {
        let _lock = lock_env();
        let tmp = TempDir::new().unwrap();
        let _guard = EnvVarGuard::set("KTSTR_CACHE_DIR", tmp.path());
        let resolved = CacheDir::default_root().unwrap();
        assert_eq!(resolved, tmp.path());
    }

    #[test]
    fn cache_dir_list_empty() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        let entries = cache.list().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn cache_dir_store_and_lookup() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));

        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");

        let entry = cache
            .store("6.14.2-tarball-x86_64", &CacheArtifacts::new(&image), &meta)
            .unwrap();
        assert_eq!(entry.key, "6.14.2-tarball-x86_64");
        assert!(entry.path.join("bzImage").exists());
        assert!(entry.path.join("metadata.json").exists());

        let found = cache.lookup("6.14.2-tarball-x86_64");
        assert!(found.is_some());
        let found = found.unwrap();
        assert_eq!(found.key, "6.14.2-tarball-x86_64");
        assert_eq!(found.metadata.version.as_deref(), Some("6.14.2"));
    }

    #[test]
    fn cache_dir_lookup_missing() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        assert!(cache.lookup("nonexistent").is_none());
    }

    #[test]
    fn cache_dir_lookup_corrupt_metadata() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        let entry_dir = tmp.path().join("bad-entry");
        fs::create_dir_all(&entry_dir).unwrap();
        fs::write(entry_dir.join("bzImage"), b"fake").unwrap();
        fs::write(entry_dir.join("metadata.json"), b"not json").unwrap();
        let found = cache.lookup("bad-entry");
        assert!(found.is_none());
    }

    #[test]
    fn cache_dir_lookup_missing_image() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());

        let entry_dir = tmp.path().join("no-image");
        fs::create_dir_all(&entry_dir).unwrap();
        let meta = test_metadata("6.14.2");
        let json = serde_json::to_string(&meta).unwrap();
        fs::write(entry_dir.join("metadata.json"), json).unwrap();

        let found = cache.lookup("no-image");
        assert!(found.is_none());
    }

    #[test]
    fn cache_dir_store_overwrites_existing() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());

        let meta1 = KernelMetadata {
            built_at: "2026-04-12T10:00:00Z".to_string(),
            config_hash: Some("hash-v1".to_string()),
            ..test_metadata("6.14.2")
        };
        cache
            .store(
                "6.14.2-tarball-x86_64",
                &CacheArtifacts::new(&image),
                &meta1,
            )
            .unwrap();

        // Bump config_hash so the in-lock recheck classifies meta2's
        // intent as a real overwrite (different on-disk contents);
        // bumping only built_at would now early-return — see
        // cache_content_matches.
        let meta2 = KernelMetadata {
            built_at: "2026-04-12T11:00:00Z".to_string(),
            config_hash: Some("hash-v2".to_string()),
            ..test_metadata("6.14.2")
        };
        cache
            .store(
                "6.14.2-tarball-x86_64",
                &CacheArtifacts::new(&image),
                &meta2,
            )
            .unwrap();

        let found = cache.lookup("6.14.2-tarball-x86_64").unwrap();
        assert_eq!(found.metadata.built_at, "2026-04-12T11:00:00Z");
        assert_eq!(found.metadata.config_hash.as_deref(), Some("hash-v2"));
    }

    #[test]
    fn cache_dir_list_sorted_newest_first() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());

        let meta_old = KernelMetadata {
            built_at: "2026-04-10T10:00:00Z".to_string(),
            ..test_metadata("6.13.0")
        };
        let meta_new = KernelMetadata {
            built_at: "2026-04-12T10:00:00Z".to_string(),
            ..test_metadata("6.14.2")
        };
        let meta_mid = KernelMetadata {
            built_at: "2026-04-11T10:00:00Z".to_string(),
            ..test_metadata("6.14.0")
        };

        cache
            .store("old", &CacheArtifacts::new(&image), &meta_old)
            .unwrap();
        cache
            .store("new", &CacheArtifacts::new(&image), &meta_new)
            .unwrap();
        cache
            .store("mid", &CacheArtifacts::new(&image), &meta_mid)
            .unwrap();

        let entries = cache.list().unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].key(), "new");
        assert_eq!(entries[1].key(), "mid");
        assert_eq!(entries[2].key(), "old");
    }

    #[test]
    fn cache_dir_list_includes_corrupt_entries() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());

        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");
        cache
            .store("valid", &CacheArtifacts::new(&image), &meta)
            .unwrap();

        let bad_dir = tmp.path().join("corrupt");
        fs::create_dir_all(&bad_dir).unwrap();

        let entries = cache.list().unwrap();
        assert_eq!(entries.len(), 2);
        let valid = entries.iter().find(|e| e.key() == "valid").unwrap();
        assert!(valid.as_valid().is_some());
        let corrupt = entries.iter().find(|e| e.key() == "corrupt").unwrap();
        assert!(corrupt.as_valid().is_none());
        let ListedEntry::Corrupt { reason, .. } = corrupt else {
            panic!("expected Corrupt variant");
        };
        assert_eq!(
            reason, "metadata.json missing",
            "missing-metadata reason should be the exact missing-file label, got: {reason}",
        );
    }

    #[test]
    fn cache_dir_list_classifies_missing_image_as_corrupt() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");
        let entry = cache
            .store("missing-image", &CacheArtifacts::new(&image), &meta)
            .unwrap();

        fs::remove_file(entry.image_path()).unwrap();

        let entries = cache.list().unwrap();
        assert_eq!(entries.len(), 1);
        let listed = &entries[0];
        assert_eq!(listed.key(), "missing-image");
        assert!(
            listed.as_valid().is_none(),
            "entry with missing image must not surface as Valid",
        );
        let ListedEntry::Corrupt { reason, .. } = listed else {
            panic!("expected Corrupt variant for missing-image entry");
        };
        assert!(
            reason.contains("image file") && reason.contains("missing"),
            "reason should cite missing image file, got: {reason}",
        );
        assert!(
            reason.contains(&meta.image_name),
            "reason should name the specific image file, got: {reason}",
        );
    }

    #[test]
    fn cache_dir_list_classifies_unreadable_metadata_as_corrupt() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        let entry_dir = tmp.path().join("unreadable-metadata");
        fs::create_dir_all(entry_dir.join("metadata.json")).unwrap();

        let entries = cache.list().unwrap();
        assert_eq!(entries.len(), 1);
        let listed = &entries[0];
        assert_eq!(listed.key(), "unreadable-metadata");
        assert!(listed.as_valid().is_none());
        let ListedEntry::Corrupt { reason, .. } = listed else {
            panic!("expected Corrupt variant for entry with unreadable metadata");
        };
        assert!(
            reason.starts_with("metadata.json unreadable: "),
            "unreadable-metadata reason should carry the unreadable prefix distinct from the \
             missing / schema-drift / malformed / truncated prefixes, got: {reason}",
        );
    }

    #[test]
    fn cache_dir_list_classifies_malformed_json_as_corrupt() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        let entry_dir = tmp.path().join("malformed-json");
        fs::create_dir_all(&entry_dir).unwrap();
        fs::write(entry_dir.join("metadata.json"), b"not valid json {[").unwrap();

        let entries = cache.list().unwrap();
        assert_eq!(entries.len(), 1);
        let listed = &entries[0];
        assert_eq!(listed.key(), "malformed-json");
        assert!(listed.as_valid().is_none());
        let ListedEntry::Corrupt { reason, .. } = listed else {
            panic!("expected Corrupt variant for malformed-json entry");
        };
        assert!(
            reason.starts_with("metadata.json malformed: "),
            "malformed-JSON reason should carry the malformed prefix \
             (Category::Syntax route), got: {reason}",
        );
    }

    #[test]
    fn cache_dir_list_classifies_incomplete_metadata_as_corrupt() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        let entry_dir = tmp.path().join("incomplete-metadata");
        fs::create_dir_all(&entry_dir).unwrap();
        fs::write(entry_dir.join("metadata.json"), br#"{"version": "6.14"}"#).unwrap();

        let entries = cache.list().unwrap();
        assert_eq!(entries.len(), 1);
        let listed = &entries[0];
        assert_eq!(listed.key(), "incomplete-metadata");
        assert!(
            listed.as_valid().is_none(),
            "incomplete-metadata missing required fields must not deserialize as Valid",
        );
        let ListedEntry::Corrupt { reason, .. } = listed else {
            panic!("expected Corrupt variant for entry with incomplete metadata");
        };
        assert!(
            reason.starts_with("metadata.json schema drift: "),
            "incomplete-metadata reason should carry the schema-drift \
             prefix (Category::Data route), got: {reason}",
        );
        assert!(
            reason.contains("missing field `source`"),
            "incomplete-metadata reason should name the first missing required field, got: {reason}",
        );
    }

    #[test]
    fn cache_dir_list_classifies_truncated_json_as_corrupt() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        let entry_dir = tmp.path().join("truncated-json");
        fs::create_dir_all(&entry_dir).unwrap();
        fs::write(entry_dir.join("metadata.json"), br#"{"source":"#).unwrap();

        let entries = cache.list().unwrap();
        assert_eq!(entries.len(), 1);
        let listed = &entries[0];
        assert_eq!(listed.key(), "truncated-json");
        assert!(listed.as_valid().is_none());
        let ListedEntry::Corrupt { reason, .. } = listed else {
            panic!("expected Corrupt variant for truncated-json entry");
        };
        assert!(
            reason.starts_with("metadata.json truncated: "),
            "truncated-JSON reason should carry the truncated prefix \
             (Category::Eof route), got: {reason}",
        );
    }

    #[test]
    fn cache_dir_list_skips_tmp_dirs() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());

        let tmp_dir = tmp.path().join(".tmp-in-progress-12345");
        fs::create_dir_all(&tmp_dir).unwrap();

        let entries = cache.list().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn cache_dir_list_skips_regular_files() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());

        fs::write(tmp.path().join("stray-file.txt"), b"stray").unwrap();

        let entries = cache.list().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn cache_dir_clean_all() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());

        cache
            .store("a", &CacheArtifacts::new(&image), &test_metadata("6.14.0"))
            .unwrap();
        cache
            .store("b", &CacheArtifacts::new(&image), &test_metadata("6.14.1"))
            .unwrap();
        cache
            .store("c", &CacheArtifacts::new(&image), &test_metadata("6.14.2"))
            .unwrap();

        let removed = cache.clean_all().unwrap();
        assert_eq!(removed, 3);
        assert!(cache.list().unwrap().is_empty());
    }

    #[test]
    fn cache_dir_clean_keep_n() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());

        let meta_old = KernelMetadata {
            built_at: "2026-04-10T10:00:00Z".to_string(),
            ..test_metadata("6.13.0")
        };
        let meta_new = KernelMetadata {
            built_at: "2026-04-12T10:00:00Z".to_string(),
            ..test_metadata("6.14.2")
        };
        let meta_mid = KernelMetadata {
            built_at: "2026-04-11T10:00:00Z".to_string(),
            ..test_metadata("6.14.0")
        };

        cache
            .store("old", &CacheArtifacts::new(&image), &meta_old)
            .unwrap();
        cache
            .store("new", &CacheArtifacts::new(&image), &meta_new)
            .unwrap();
        cache
            .store("mid", &CacheArtifacts::new(&image), &meta_mid)
            .unwrap();

        let removed = cache.clean_keep(1).unwrap();
        assert_eq!(removed, 2);

        let remaining = cache.list().unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].key(), "new");
    }

    #[test]
    fn cache_dir_clean_keep_more_than_exist() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());

        cache
            .store(
                "only",
                &CacheArtifacts::new(&image),
                &test_metadata("6.14.2"),
            )
            .unwrap();

        let removed = cache.clean_keep(5).unwrap();
        assert_eq!(removed, 0);
        assert_eq!(cache.list().unwrap().len(), 1);
    }

    #[test]
    fn cache_dir_clean_empty_cache() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        let removed = cache.clean_all().unwrap();
        assert_eq!(removed, 0);
    }

    // -- image_name traversal via store --

    #[test]
    fn cache_dir_store_rejects_image_name_traversal() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let mut meta = test_metadata("6.14.2");
        meta.image_name = "../escape".to_string();

        let err = cache
            .store("valid-key", &CacheArtifacts::new(&image), &meta)
            .unwrap_err();
        assert!(
            err.to_string().contains("image name"),
            "expected image_name rejection, got: {err}"
        );
    }

    // -- .tmp- prefix via store/lookup --

    #[test]
    fn cache_dir_store_tmp_prefix_key_rejected() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");

        let err = cache
            .store(".tmp-sneaky", &CacheArtifacts::new(&image), &meta)
            .unwrap_err();
        assert!(
            err.to_string().contains(".tmp-"),
            "expected .tmp- rejection, got: {err}"
        );
    }

    #[test]
    fn cache_dir_lookup_tmp_prefix_returns_none() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        assert!(cache.lookup(".tmp-sneaky").is_none());
    }

    // -- cache key validation via store/lookup --

    #[test]
    fn cache_dir_store_empty_key_rejected() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");

        let err = cache
            .store("", &CacheArtifacts::new(&image), &meta)
            .unwrap_err();
        assert!(
            err.to_string().contains("empty"),
            "expected empty-key error, got: {err}"
        );
    }

    #[test]
    fn cache_dir_lookup_empty_key_returns_none() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        assert!(cache.lookup("").is_none());
    }

    #[test]
    fn cache_dir_store_path_traversal_rejected() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");

        let err = cache
            .store("../escape", &CacheArtifacts::new(&image), &meta)
            .unwrap_err();
        assert!(
            err.to_string().contains("path"),
            "expected path-traversal error, got: {err}"
        );
    }

    #[test]
    fn cache_dir_lookup_path_traversal_returns_none() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        assert!(cache.lookup("../escape").is_none());
        assert!(cache.lookup("foo/../bar").is_none());
    }

    #[test]
    fn cache_dir_store_slash_in_key_rejected() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");

        let err = cache
            .store("a/b", &CacheArtifacts::new(&image), &meta)
            .unwrap_err();
        assert!(
            err.to_string().contains("path separator"),
            "expected path-separator error, got: {err}"
        );
    }

    #[test]
    fn cache_dir_store_whitespace_only_key_rejected() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");

        let err = cache
            .store("   ", &CacheArtifacts::new(&image), &meta)
            .unwrap_err();
        assert!(
            err.to_string().contains("empty"),
            "expected empty/whitespace error, got: {err}"
        );
    }

    // -- clean with mixed valid + corrupt entries --

    #[test]
    fn cache_dir_clean_keep_n_with_mixed_entries() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());

        let meta_new = KernelMetadata {
            built_at: "2026-04-12T10:00:00Z".to_string(),
            ..test_metadata("6.14.2")
        };
        let meta_old = KernelMetadata {
            built_at: "2026-04-10T10:00:00Z".to_string(),
            ..test_metadata("6.13.0")
        };
        cache
            .store("new", &CacheArtifacts::new(&image), &meta_new)
            .unwrap();
        cache
            .store("old", &CacheArtifacts::new(&image), &meta_old)
            .unwrap();

        let corrupt_dir = tmp.path().join("cache").join("corrupt");
        fs::create_dir_all(&corrupt_dir).unwrap();

        let removed = cache.clean_keep(1).unwrap();
        assert_eq!(removed, 2);

        let remaining = cache.list().unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].key(), "new");
    }

    // -- atomic write safety --

    #[test]
    fn cache_dir_store_overwrites_existing_key_atomically() {
        let tmp = TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let cache = CacheDir::with_root(cache_root.clone());

        let src_a = TempDir::new().unwrap();
        let image_a = create_fake_image(src_a.path());
        fs::write(&image_a, b"version-a").unwrap();
        let mut meta_a = test_metadata("6.14.2");
        meta_a.built_at = "2026-04-10T00:00:00Z".to_string();
        meta_a.config_hash = Some("hash-a".to_string());
        let entry_a = cache
            .store("collide", &CacheArtifacts::new(&image_a), &meta_a)
            .unwrap();
        assert_eq!(
            fs::read(entry_a.path.join("bzImage")).unwrap(),
            b"version-a"
        );

        let src_b = TempDir::new().unwrap();
        let image_b = create_fake_image(src_b.path());
        fs::write(&image_b, b"version-b").unwrap();
        let mut meta_b = test_metadata("6.14.2");
        meta_b.built_at = "2026-04-18T00:00:00Z".to_string();
        // Distinct config_hash forces the in-lock recheck to bypass
        // the early-return and proceed through the real overwrite
        // path — the test exercises atomic publish, not recheck.
        meta_b.config_hash = Some("hash-b".to_string());
        let entry_b = cache
            .store("collide", &CacheArtifacts::new(&image_b), &meta_b)
            .unwrap();

        assert_eq!(
            fs::read(entry_b.path.join("bzImage")).unwrap(),
            b"version-b",
            "new content must replace old content atomically"
        );
        let installed_meta = read_metadata(&entry_b.path).expect("metadata.json");
        assert_eq!(installed_meta.built_at, "2026-04-18T00:00:00Z");
        assert_eq!(installed_meta.config_hash.as_deref(), Some("hash-b"));

        for dirent in fs::read_dir(&cache_root).unwrap() {
            let name = dirent.unwrap().file_name().to_string_lossy().into_owned();
            assert!(
                !name.starts_with(".evict-") && !name.starts_with(".tmp-"),
                "unexpected leftover directory under cache_root: {name}"
            );
        }
    }

    #[test]
    fn cache_dir_store_cleans_stale_tmp() {
        let tmp = TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let cache = CacheDir::with_root(cache_root.clone());

        let stale_tmp = cache_root.join(format!(".tmp-mykey-{}", std::process::id()));
        fs::create_dir_all(&stale_tmp).unwrap();
        fs::write(stale_tmp.join("junk"), b"leftover").unwrap();

        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");

        let entry = cache
            .store("mykey", &CacheArtifacts::new(&image), &meta)
            .unwrap();
        assert!(entry.path.join("bzImage").exists());
        assert!(!stale_tmp.exists());
    }

    #[test]
    fn cache_dir_store_atomic_under_concurrent_readers() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::thread;

        let tmp = TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let cache = Arc::new(CacheDir::with_root(cache_root.clone()));

        let src_a = TempDir::new().unwrap();
        let image_a = src_a.path().join("bzImage");
        let content_a = b"AAAAAAAA-image-version-a-AAAAAAAA".repeat(64);
        fs::write(&image_a, &content_a).unwrap();

        let src_b = TempDir::new().unwrap();
        let image_b = src_b.path().join("bzImage");
        let content_b = b"BBBBBBBB-image-version-b-BBBBBBBB".repeat(64);
        fs::write(&image_b, &content_b).unwrap();

        let meta_prime = test_metadata("6.14.2");
        cache
            .store("atomic-key", &CacheArtifacts::new(&image_a), &meta_prime)
            .unwrap();

        const WRITE_ITERATIONS: usize = 40;
        let stop = Arc::new(AtomicBool::new(false));
        let lookups_observed = Arc::new(AtomicUsize::new(0));
        let atomicity_violations = Arc::new(AtomicUsize::new(0));

        let reader_count = 4;
        let mut readers = Vec::with_capacity(reader_count);
        for _ in 0..reader_count {
            let cache = Arc::clone(&cache);
            let stop = Arc::clone(&stop);
            let lookups_observed = Arc::clone(&lookups_observed);
            let violations = Arc::clone(&atomicity_violations);
            let expected_a = content_a.clone();
            let expected_b = content_b.clone();
            readers.push(thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let Some(entry) = cache.lookup("atomic-key") else {
                        violations.fetch_add(1, Ordering::Relaxed);
                        continue;
                    };
                    let image_path = entry.image_path();
                    let Ok(bytes) = fs::read(&image_path) else {
                        violations.fetch_add(1, Ordering::Relaxed);
                        continue;
                    };
                    if bytes != expected_a && bytes != expected_b {
                        violations.fetch_add(1, Ordering::Relaxed);
                    }
                    lookups_observed.fetch_add(1, Ordering::Relaxed);
                }
            }));
        }

        for i in 0..WRITE_ITERATIONS {
            let (image, label) = if i % 2 == 0 {
                (&image_a, "a")
            } else {
                (&image_b, "b")
            };
            let mut meta = test_metadata("6.14.2");
            meta.built_at = format!("2026-04-18T00:00:{:02}Z", i % 60);
            meta.config_hash = Some(format!("iter-{i}-{label}"));
            cache
                .store("atomic-key", &CacheArtifacts::new(image), &meta)
                .expect("store under concurrent readers must not fail");
        }

        stop.store(true, Ordering::Relaxed);
        for r in readers {
            r.join().expect("reader thread panicked");
        }

        assert_eq!(
            atomicity_violations.load(Ordering::Relaxed),
            0,
            "lookup observed a missing or torn cache entry during concurrent store; \
             rename-to-staging swap is not atomic",
        );
        assert!(
            lookups_observed.load(Ordering::Relaxed) > 0,
            "readers never observed a successful lookup — test did not \
             actually exercise the concurrency window",
        );

        let final_entry = cache.lookup("atomic-key").expect("entry must exist");
        let final_bytes = fs::read(final_entry.image_path()).unwrap();
        assert!(
            final_bytes == content_a || final_bytes == content_b,
            "final image must match one of the writer's versions",
        );
        for dirent in fs::read_dir(&cache_root).unwrap() {
            let name = dirent.unwrap().file_name().to_string_lossy().into_owned();
            assert!(
                !name.starts_with(".evict-") && !name.starts_with(".tmp-"),
                "unexpected leftover directory under cache_root: {name}",
            );
        }
    }

    #[test]
    fn cache_dir_store_with_vmlinux() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let vmlinux = src_dir.path().join("vmlinux");
        fs::write(&vmlinux, b"fake vmlinux ELF").unwrap();
        let meta = test_metadata("6.14.2");

        let entry = cache
            .store(
                "with-vmlinux",
                &CacheArtifacts::new(&image).with_vmlinux(&vmlinux),
                &meta,
            )
            .unwrap();
        assert!(entry.path.join("bzImage").exists());
        assert!(entry.path.join("vmlinux").exists());
        assert!(entry.path.join("metadata.json").exists());
        assert!(entry.metadata.has_vmlinux);
        assert!(image.exists());
        assert!(vmlinux.exists());
    }

    #[test]
    fn cache_dir_store_without_vmlinux() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");

        let entry = cache
            .store("no-vmlinux", &CacheArtifacts::new(&image), &meta)
            .unwrap();
        assert!(entry.path.join("bzImage").exists());
        assert!(!entry.path.join("vmlinux").exists());
        assert!(entry.path.join("metadata.json").exists());
        assert!(!entry.metadata.has_vmlinux);
        assert!(!entry.metadata.vmlinux_stripped);
    }

    #[test]
    fn cache_dir_store_falls_back_when_strip_fails() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let vmlinux = src_dir.path().join("vmlinux");
        let raw = b"not an ELF file";
        fs::write(&vmlinux, raw).unwrap();
        let meta = test_metadata("6.14.2");

        let entry = cache
            .store(
                "strip-fallback",
                &CacheArtifacts::new(&image).with_vmlinux(&vmlinux),
                &meta,
            )
            .unwrap();
        let cached = fs::read(entry.path.join("vmlinux")).unwrap();
        assert_eq!(cached, raw, "fallback must copy raw bytes verbatim");
        assert!(entry.metadata.has_vmlinux);
        assert!(
            !entry.metadata.vmlinux_stripped,
            "raw-fallback path must set vmlinux_stripped = false"
        );
    }

    // -- should_warn_unstripped --

    fn make_warn_test_entry(has_vmlinux: bool, vmlinux_stripped: bool) -> CacheEntry {
        let mut meta = KernelMetadata::new(
            super::super::metadata::KernelSource::Tarball,
            "x86_64".to_string(),
            "bzImage".to_string(),
            "2026-04-24T12:00:00Z".to_string(),
        );
        meta.set_has_vmlinux(has_vmlinux);
        meta.set_vmlinux_stripped(vmlinux_stripped);
        CacheEntry {
            key: "test-key".to_string(),
            path: PathBuf::from("/nonexistent/entry"),
            metadata: meta,
        }
    }

    #[test]
    fn should_warn_unstripped_fires_when_vmlinux_present_and_unstripped() {
        let entry = make_warn_test_entry(true, false);
        assert!(
            should_warn_unstripped(&entry),
            "has_vmlinux=true + vmlinux_stripped=false must warn"
        );
    }

    #[test]
    fn should_warn_unstripped_silent_when_vmlinux_stripped() {
        let entry = make_warn_test_entry(true, true);
        assert!(
            !should_warn_unstripped(&entry),
            "has_vmlinux=true + vmlinux_stripped=true must not warn"
        );
    }

    #[test]
    fn should_warn_unstripped_silent_when_no_vmlinux() {
        let entry = make_warn_test_entry(false, false);
        assert!(
            !should_warn_unstripped(&entry),
            "has_vmlinux=false must not warn (no vmlinux to worry about)"
        );
    }

    #[test]
    fn cache_dir_store_preserves_original_image() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");

        cache
            .store("key", &CacheArtifacts::new(&image), &meta)
            .unwrap();

        assert!(image.exists());
    }

    // -- CacheEntry accessors --

    #[test]
    fn cache_entry_image_path_joins_key_with_image_name() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let entry = cache
            .store(
                "key",
                &CacheArtifacts::new(&image),
                &test_metadata("6.14.2"),
            )
            .unwrap();
        assert_eq!(entry.image_path(), entry.path.join("bzImage"));
        assert!(entry.image_path().exists());
    }

    #[test]
    fn cache_entry_vmlinux_path_none_when_not_stored() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let entry = cache
            .store(
                "no-vml",
                &CacheArtifacts::new(&image),
                &test_metadata("6.14.2"),
            )
            .unwrap();
        assert!(entry.vmlinux_path().is_none());
    }

    // -- KconfigStatus variants --

    #[test]
    fn kconfig_status_matches_when_hash_equal() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2").with_ktstr_kconfig_hash(Some("deadbeef".to_string()));
        let entry = cache
            .store("kc-match", &CacheArtifacts::new(&image), &meta)
            .unwrap();
        assert_eq!(entry.kconfig_status("deadbeef"), KconfigStatus::Matches);
    }

    #[test]
    fn kconfig_status_untracked_when_no_hash_in_entry() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = KernelMetadata {
            ktstr_kconfig_hash: None,
            ..test_metadata("6.14.2")
        };
        let entry = cache
            .store("kc-untracked", &CacheArtifacts::new(&image), &meta)
            .unwrap();
        assert_eq!(entry.kconfig_status("anything"), KconfigStatus::Untracked);
    }

    #[test]
    fn kconfig_status_stale_pins_cached_and_current_field_order() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2").with_ktstr_kconfig_hash(Some("old_cached".to_string()));
        let entry = cache
            .store("kc-stale", &CacheArtifacts::new(&image), &meta)
            .unwrap();
        match entry.kconfig_status("new_current") {
            KconfigStatus::Stale { cached, current } => {
                assert_eq!(
                    cached, "old_cached",
                    "`cached` must hold the hash recorded in the entry"
                );
                assert_eq!(
                    current, "new_current",
                    "`current` must hold the hash the caller passed in"
                );
            }
            other => panic!("expected KconfigStatus::Stale, got {other:?}"),
        }
    }

    // -- Cache-entry coordination locks --

    #[test]
    fn acquire_shared_lock_creates_lockfile_at_expected_path() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        let _guard = cache.acquire_shared_lock("some-key-123").unwrap();
        assert!(
            tmp.path().join(".locks").is_dir(),
            "parent .locks/ subdirectory must materialize on first acquire",
        );
        assert!(
            tmp.path().join(".locks").join("some-key-123.lock").exists(),
            "lockfile must materialize at {{cache_root}}/.locks/{{key}}.lock on first acquire",
        );
    }

    #[test]
    fn acquire_shared_lock_permits_concurrent_readers() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let tmp = TempDir::new().unwrap();
        let cache = Arc::new(CacheDir::with_root(tmp.path().to_path_buf()));
        let key = "concurrent-sh";
        let success = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..4 {
            let cache = Arc::clone(&cache);
            let success = Arc::clone(&success);
            handles.push(std::thread::spawn(move || {
                let _g = cache
                    .acquire_shared_lock(key)
                    .expect("LOCK_SH must succeed");
                success.fetch_add(1, Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(50));
            }));
        }
        for h in handles {
            h.join().expect("reader thread panicked");
        }
        assert_eq!(
            success.load(Ordering::SeqCst),
            4,
            "all 4 concurrent LOCK_SH acquires must succeed",
        );
    }

    #[test]
    fn try_acquire_exclusive_lock_fails_with_active_reader() {
        use std::sync::Arc;
        use std::sync::mpsc;
        let tmp = TempDir::new().unwrap();
        let cache = Arc::new(CacheDir::with_root(tmp.path().to_path_buf()));
        let key = "force-contended";
        let (ready_tx, ready_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let cache_reader = Arc::clone(&cache);
        let reader = std::thread::spawn(move || {
            let _g = cache_reader
                .acquire_shared_lock(key)
                .expect("reader LOCK_SH must succeed");
            ready_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });
        ready_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("reader thread did not signal ready in time");
        let err = cache.try_acquire_exclusive_lock(key).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("is locked by active test runs") || msg.contains("holders:"),
            "error must surface the contention diagnostic; got: {msg}",
        );
        assert!(
            msg.contains("lockfile"),
            "error must name the lockfile path: {msg}",
        );
        release_tx.send(()).unwrap();
        reader.join().expect("reader thread panicked");
    }

    #[test]
    fn acquire_exclusive_lock_blocking_times_out_on_contention() {
        use std::sync::Arc;
        use std::sync::mpsc;
        let tmp = TempDir::new().unwrap();
        let cache = Arc::new(CacheDir::with_root(tmp.path().to_path_buf()));
        let key = "blocking-timeout";
        let (ready_tx, ready_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let cache_reader = Arc::clone(&cache);
        let reader = std::thread::spawn(move || {
            let _g = cache_reader
                .acquire_shared_lock(key)
                .expect("reader LOCK_SH must succeed");
            ready_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });
        ready_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("reader did not signal ready in time");
        let start = std::time::Instant::now();
        let err = cache
            .acquire_exclusive_lock_blocking(key, std::time::Duration::from_millis(200))
            .unwrap_err();
        let elapsed = start.elapsed();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("timed out"),
            "error must mention the timeout: {msg}",
        );
        assert!(
            elapsed >= std::time::Duration::from_millis(150),
            "acquire should have waited ~timeout (150ms lower bound); \
             got {elapsed:?}",
        );
        assert!(
            msg.contains("KTSTR_CACHE_STORE_LOCK_TIMEOUT"),
            "timeout error must surface the env-var override so \
             operators discover the remediation without reading docs: {msg}",
        );
        release_tx.send(()).unwrap();
        reader.join().expect("reader thread panicked");
    }

    #[test]
    fn store_succeeds_under_internal_exclusive_lock() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");
        let entry = cache
            .store("internal-lock", &CacheArtifacts::new(&image), &meta)
            .expect("store must succeed when no readers contend");
        assert!(entry.path.join("bzImage").exists());
        assert!(
            tmp.path()
                .join("cache")
                .join(".locks")
                .join("internal-lock.lock")
                .exists(),
            "lockfile materialized during store must persist after store returns",
        );
    }

    #[test]
    fn store_blocks_while_reader_holds_shared_lock() {
        use std::sync::Arc;
        use std::sync::mpsc;
        let tmp = TempDir::new().unwrap();
        let cache = Arc::new(CacheDir::with_root(tmp.path().join("cache-block")));
        let key = "blocked-store";
        let (ready_tx, ready_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let cache_reader = Arc::clone(&cache);
        let reader = std::thread::spawn(move || {
            let _g = cache_reader
                .acquire_shared_lock(key)
                .expect("reader LOCK_SH must succeed");
            ready_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });
        ready_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("reader did not signal ready in time");

        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");
        let (store_done_tx, store_done_rx) = mpsc::channel();
        let cache_store = Arc::clone(&cache);
        let image_clone = image.clone();
        let store_thread = std::thread::spawn(move || {
            let _ = cache_store.store(key, &CacheArtifacts::new(&image_clone), &meta);
            store_done_tx.send(()).unwrap();
        });
        let early = store_done_rx.recv_timeout(std::time::Duration::from_millis(200));
        assert!(
            early.is_err(),
            "store() must block while reader holds LOCK_SH; got completion signal early",
        );
        release_tx.send(()).unwrap();
        let finish = store_done_rx.recv_timeout(std::time::Duration::from_secs(10));
        assert!(
            finish.is_ok(),
            "store() must complete after reader releases; got timeout",
        );
        reader.join().expect("reader thread panicked");
        store_thread.join().expect("store thread panicked");
    }

    #[test]
    fn lock_path_returns_expected_shape() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        let path = cache.lock_path("my-key-42");
        assert_eq!(path, tmp.path().join(".locks").join("my-key-42.lock"));
    }

    #[test]
    fn locks_subdir_persists_after_guard_drop() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        let locks_dir = tmp.path().join(".locks");
        {
            let _guard = cache
                .acquire_shared_lock("persist-test")
                .expect("acquire must succeed");
            assert!(locks_dir.is_dir(), "must exist during guard lifetime");
        }
        assert!(
            locks_dir.is_dir(),
            ".locks/ must persist after guard drop — next acquire \
             keys /proc/locks on the existing inode",
        );
    }

    #[test]
    fn list_skips_locks_dotfile_subdirectory() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        let _guard = cache.acquire_shared_lock("dummy").expect("acquire");
        drop(_guard);
        assert!(
            tmp.path().join(".locks").is_dir(),
            ".locks/ must exist after acquire drop",
        );
        let entries = cache.list().expect("list must succeed");
        let keys: Vec<&str> = entries
            .iter()
            .map(|e| match e {
                ListedEntry::Valid(entry) => entry.key.as_str(),
                ListedEntry::Corrupt { key, .. } => key.as_str(),
            })
            .collect();
        assert!(
            !keys.iter().any(|k| k.starts_with('.')),
            "list() must not return dotfile children: {keys:?}",
        );
    }

    #[test]
    fn acquire_on_empty_root_creates_locks_dir_lazily() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("pristine");
        std::fs::create_dir(&root).unwrap();
        let cache = CacheDir::with_root(root.clone());
        assert!(!root.join(".locks").exists());
        let _guard = cache
            .acquire_shared_lock("lazy-test")
            .expect("first acquire on empty root must succeed");
        assert!(
            root.join(".locks").is_dir(),
            "first acquire must materialize .locks/ lazily",
        );
    }

    #[test]
    fn cache_dir_clean_all_preserves_locks_subdir() {
        let tmp = TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let cache = CacheDir::with_root(cache_root.clone());
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());

        cache
            .store(
                "entry-a",
                &CacheArtifacts::new(&image),
                &test_metadata("6.14.0"),
            )
            .expect("store must succeed");
        let _guard = cache
            .acquire_shared_lock("entry-a")
            .expect("SH acquire must succeed");

        let locks_dir = cache_root.join(".locks");
        let lockfile = locks_dir.join("entry-a.lock");
        assert!(locks_dir.is_dir(), "precondition: .locks/ must exist");
        assert!(lockfile.exists(), "precondition: lockfile must exist");

        let removed = cache.clean_all().expect("clean_all must succeed");
        assert_eq!(removed, 1, "clean_all must remove exactly 1 entry");

        assert!(
            locks_dir.is_dir(),
            ".locks/ subdirectory must survive clean_all",
        );
        assert!(
            lockfile.exists(),
            "lockfile must still exist under .locks/ after clean_all",
        );

        assert!(
            !cache_root.join("entry-a").exists(),
            "cache entry must be removed by clean_all",
        );
    }

    #[test]
    fn cache_dir_acquire_rejects_path_traversal_key() {
        let tmp = TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let cache = CacheDir::with_root(cache_root.clone());

        let err = cache
            .acquire_shared_lock("../../etc/passwd")
            .expect_err("path-traversal key must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("path"),
            "error must mention path rejection: {msg}",
        );

        let etc_passwd_lock = tmp.path().join("etc").join("passwd.lock");
        assert!(
            !etc_passwd_lock.exists(),
            "path traversal must NOT create a lockfile outside .locks/",
        );
        assert!(
            !cache_root.join(".locks").exists()
                || cache_root
                    .join(".locks")
                    .read_dir()
                    .unwrap()
                    .next()
                    .is_none(),
            ".locks/ must be empty if it exists at all — validator \
             rejects before lockfile creation",
        );
    }

    // -- try_acquire_exclusive_lock happy path --

    /// Uncontended `try_acquire_exclusive_lock` returns the
    /// `ExclusiveLockGuard` and materializes the lockfile under
    /// `.locks/`.
    #[test]
    fn try_acquire_exclusive_lock_succeeds_when_uncontended() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());

        let guard = cache
            .try_acquire_exclusive_lock("happy-path-key")
            .expect("uncontended try_acquire_exclusive_lock must succeed");

        let lockfile = tmp.path().join(".locks").join("happy-path-key.lock");
        assert!(
            lockfile.exists(),
            "happy-path acquire must materialize the lockfile at \
             {} — without it, /proc/locks lookup of contention \
             diagnostics fails to attribute the holder",
            lockfile.display(),
        );
        assert!(
            tmp.path().join(".locks").is_dir(),
            ".locks/ subdirectory must exist after a happy-path \
             acquire (lazy materialization)",
        );

        drop(guard);

        let guard2 = cache
            .try_acquire_exclusive_lock("happy-path-key")
            .expect("second acquire on same key must succeed after the first guard drops");
        drop(guard2);
    }

    /// `try_acquire_exclusive_lock` rejects path-traversal keys
    /// before opening any lockfile, mirroring the
    /// `acquire_shared_lock` rejection contract.
    #[test]
    fn try_acquire_exclusive_lock_rejects_invalid_key() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        let err = cache
            .try_acquire_exclusive_lock("../escape")
            .expect_err("invalid key must be rejected before lockfile open");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("path"),
            "validator must surface a path-related diagnostic: {msg}",
        );
    }

    /// `try_acquire_exclusive_lock` succeeds against the same key on
    /// distinct `CacheDir` roots concurrently — the lock is keyed on
    /// the per-root `.locks/<key>.lock` inode, not the bare key
    /// string.
    #[test]
    fn try_acquire_exclusive_lock_distinct_roots_dont_contend() {
        let tmp_a = TempDir::new().unwrap();
        let tmp_b = TempDir::new().unwrap();
        let cache_a = CacheDir::with_root(tmp_a.path().to_path_buf());
        let cache_b = CacheDir::with_root(tmp_b.path().to_path_buf());

        let guard_a = cache_a
            .try_acquire_exclusive_lock("shared-name")
            .expect("acquire under root A must succeed");
        let guard_b = cache_b
            .try_acquire_exclusive_lock("shared-name")
            .expect(
                "acquire on the same key under root B must NOT \
                 contend with A — different lockfiles, different OFDs",
            );

        drop(guard_a);
        drop(guard_b);
    }

    // -- in-lock double-checked re-lookup (cache_content_matches) --
    //
    // Direct unit coverage of the predicate:
    // identical-content-different-built_at must hit, distinct
    // config_hash must miss, distinct ktstr_kconfig_hash must miss,
    // distinct extra_kconfig_hash must miss, mismatched
    // caller_has_vmlinux must miss. Plus an end-to-end test that
    // proves the in-lock recheck observably skips the publish step
    // by leaving the cached `built_at` intact when only `built_at`
    // differs.

    /// Identical hashes + identical vmlinux presence: predicate
    /// matches even when built_at and version differ.
    #[test]
    fn cache_content_matches_when_only_built_at_differs() {
        let mut cached = test_metadata("6.14.2");
        cached.built_at = "2026-04-12T10:00:00Z".to_string();
        let mut caller = test_metadata("6.14.2");
        caller.built_at = "2026-04-12T11:00:00Z".to_string();
        assert!(
            cache_content_matches(&cached, &caller, false),
            "identical content hashes (config_hash, ktstr_kconfig_hash, \
             extra_kconfig_hash) and identical vmlinux presence must \
             classify as content-equal — built_at is just a timestamp",
        );
    }

    /// Distinct config_hash → real overwrite intent → predicate misses.
    #[test]
    fn cache_content_matches_when_config_hash_differs() {
        let mut cached = test_metadata("6.14.2");
        cached.config_hash = Some("hash-cached".to_string());
        let mut caller = test_metadata("6.14.2");
        caller.config_hash = Some("hash-caller".to_string());
        assert!(
            !cache_content_matches(&cached, &caller, false),
            "distinct config_hash must classify as content-different \
             — the .config differs, so the boot image bytes differ",
        );
    }

    /// Distinct ktstr_kconfig_hash → real overwrite intent.
    #[test]
    fn cache_content_matches_when_ktstr_kconfig_hash_differs() {
        let mut cached = test_metadata("6.14.2");
        cached.ktstr_kconfig_hash = Some("kc-cached".to_string());
        let mut caller = test_metadata("6.14.2");
        caller.ktstr_kconfig_hash = Some("kc-caller".to_string());
        assert!(
            !cache_content_matches(&cached, &caller, false),
            "distinct ktstr_kconfig_hash means the kconfig fragment \
             changed → built differently → content-different",
        );
    }

    /// Distinct extra_kconfig_hash → real overwrite intent.
    #[test]
    fn cache_content_matches_when_extra_kconfig_hash_differs() {
        let mut cached = test_metadata("6.14.2");
        cached.extra_kconfig_hash = Some("xc-cached".to_string());
        let mut caller = test_metadata("6.14.2");
        caller.extra_kconfig_hash = Some("xc-caller".to_string());
        assert!(
            !cache_content_matches(&cached, &caller, false),
            "distinct extra_kconfig_hash means the user fragment \
             changed → built differently → content-different",
        );
    }

    /// Caller wants vmlinux but cached entry lacks it (or vice
    /// versa) → publish is required to add/remove the sidecar.
    #[test]
    fn cache_content_matches_when_vmlinux_presence_differs() {
        let cached_with = {
            let mut m = test_metadata("6.14.2");
            m.set_has_vmlinux(true);
            m
        };
        let caller = test_metadata("6.14.2");
        assert!(
            !cache_content_matches(&cached_with, &caller, false),
            "cached has vmlinux, caller lacks vmlinux artifact — \
             content-different (publish must drop the sidecar)",
        );

        let cached_without = test_metadata("6.14.2");
        assert!(
            !cache_content_matches(&cached_without, &caller, true),
            "cached lacks vmlinux, caller supplies one — \
             content-different (publish must add the sidecar)",
        );
    }

    /// End-to-end: a second `store()` that only bumps `built_at`
    /// must hit the in-lock recheck and short-circuit, leaving the
    /// FIRST publish's metadata intact on disk. Without the
    /// recheck the second publish would land and the assertion on
    /// `built_at` would flip to the second timestamp.
    #[test]
    fn store_in_lock_recheck_short_circuits_on_built_at_only_change() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());

        let meta1 = KernelMetadata {
            built_at: "2026-04-12T10:00:00Z".to_string(),
            ..test_metadata("6.14.2")
        };
        cache
            .store("recheck-key", &CacheArtifacts::new(&image), &meta1)
            .unwrap();

        // Same content (same hashes, no vmlinux), bumped built_at —
        // the recheck must classify this as content-equivalent and
        // skip the publish.
        let meta2 = KernelMetadata {
            built_at: "2026-04-13T10:00:00Z".to_string(),
            ..test_metadata("6.14.2")
        };
        let returned = cache
            .store("recheck-key", &CacheArtifacts::new(&image), &meta2)
            .unwrap();

        assert_eq!(
            returned.metadata.built_at, "2026-04-12T10:00:00Z",
            "the in-lock recheck must short-circuit and return the \
             EXISTING cached entry — the returned built_at must \
             match meta1, not meta2. If this flips to meta2, the \
             recheck did not fire and every concurrent peer is \
             redundantly republishing.",
        );

        let on_disk = cache.lookup("recheck-key").unwrap();
        assert_eq!(
            on_disk.metadata.built_at, "2026-04-12T10:00:00Z",
            "the on-disk metadata must also remain meta1 — the \
             recheck must skip the rename/swap step",
        );
    }

    /// End-to-end: when a second `store()` carries a real content
    /// change (distinct config_hash), the recheck miss-and-bypass
    /// must publish the new content. Pins the recheck does NOT
    /// silently lose legitimate overwrites.
    #[test]
    fn store_in_lock_recheck_bypasses_when_content_actually_differs() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());

        let meta1 = KernelMetadata {
            built_at: "2026-04-12T10:00:00Z".to_string(),
            config_hash: Some("hash-v1".to_string()),
            ..test_metadata("6.14.2")
        };
        cache
            .store("bypass-key", &CacheArtifacts::new(&image), &meta1)
            .unwrap();

        let meta2 = KernelMetadata {
            built_at: "2026-04-13T10:00:00Z".to_string(),
            config_hash: Some("hash-v2".to_string()),
            ..test_metadata("6.14.2")
        };
        let returned = cache
            .store("bypass-key", &CacheArtifacts::new(&image), &meta2)
            .unwrap();

        assert_eq!(
            returned.metadata.config_hash.as_deref(),
            Some("hash-v2"),
            "distinct config_hash must bypass the recheck and \
             publish meta2; the returned entry's config_hash must \
             be meta2's",
        );
        assert_eq!(
            returned.metadata.built_at, "2026-04-13T10:00:00Z",
            "with content actually changing, the publish must \
             land meta2's built_at",
        );
    }

    /// End-to-end: N concurrent peers race to `store()` the same
    /// content under the same key. With the recheck, only the head
    /// writer's publish lands; every late peer hits the in-lock
    /// re-lookup and short-circuits. Observable through the
    /// returned `CacheEntry::metadata.built_at` — every late peer
    /// sees the head writer's timestamp regardless of what they
    /// passed in.
    #[test]
    fn store_in_lock_recheck_serialises_concurrent_peers() {
        use std::sync::Arc;
        use std::sync::Barrier;
        use std::thread;

        let tmp = TempDir::new().unwrap();
        let cache = Arc::new(CacheDir::with_root(tmp.path().join("cache")));
        let src_dir = TempDir::new().unwrap();
        let image = src_dir.path().join("bzImage");
        std::fs::write(&image, b"shared image bytes").unwrap();

        const PEER_COUNT: usize = 8;
        let barrier = Arc::new(Barrier::new(PEER_COUNT));
        let mut handles = Vec::with_capacity(PEER_COUNT);
        for i in 0..PEER_COUNT {
            let cache = Arc::clone(&cache);
            let barrier = Arc::clone(&barrier);
            let image = image.clone();
            handles.push(thread::spawn(move || {
                let mut meta = test_metadata("6.14.2");
                // Each peer claims a distinct built_at — but
                // identical hashes → recheck-equivalent.
                meta.built_at = format!("2026-04-12T10:00:{i:02}Z");
                barrier.wait();
                cache
                    .store("race-key", &CacheArtifacts::new(&image), &meta)
                    .expect("every peer's store must succeed")
            }));
        }
        let entries: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // Exactly one distinct built_at across all returned
        // entries — the head writer's. If the recheck didn't fire,
        // each peer's publish would land in turn and we'd see N
        // distinct values.
        let timestamps: std::collections::BTreeSet<_> =
            entries.iter().map(|e| e.metadata.built_at.clone()).collect();
        assert_eq!(
            timestamps.len(),
            1,
            "every peer must observe the same head-writer timestamp \
             after the in-lock recheck short-circuits theirs; \
             distinct timestamps means the recheck didn't fire and \
             every peer redundantly republished. Got: {timestamps:?}",
        );

        // The on-disk entry must still match what every peer
        // observed — a sanity check that no half-publish landed.
        let final_entry = cache.lookup("race-key").expect("entry must exist");
        let head_timestamp = timestamps.iter().next().unwrap();
        assert_eq!(
            &final_entry.metadata.built_at, head_timestamp,
            "the cached entry's built_at must match what every peer \
             returned — proves the head writer's publish landed and \
             every late peer short-circuited to the same on-disk \
             state",
        );
    }

    // -- store_exclusive_lock_timeout env override --

    /// Unset env var → default timeout.
    #[test]
    fn store_exclusive_lock_timeout_returns_default_when_unset() {
        let _lock = lock_env();
        let _g = EnvVarGuard::remove(STORE_EXCLUSIVE_LOCK_TIMEOUT_ENV);
        assert_eq!(
            store_exclusive_lock_timeout(),
            STORE_EXCLUSIVE_LOCK_DEFAULT_TIMEOUT,
            "absent env var must return the default timeout",
        );
    }

    /// Empty env var → default timeout (mirrors KTSTR_CACHE_DIR's
    /// "empty falls through" cascade behaviour for consistency).
    #[test]
    fn store_exclusive_lock_timeout_returns_default_when_empty() {
        let _lock = lock_env();
        let _g = EnvVarGuard::set(STORE_EXCLUSIVE_LOCK_TIMEOUT_ENV, "");
        assert_eq!(
            store_exclusive_lock_timeout(),
            STORE_EXCLUSIVE_LOCK_DEFAULT_TIMEOUT,
            "empty env var must fall through to the default",
        );
    }

    /// Valid humantime string → parsed duration.
    #[test]
    fn store_exclusive_lock_timeout_parses_humantime() {
        let _lock = lock_env();
        for (input, want_secs) in [
            ("30s", 30),
            ("2m", 120),
            ("10min", 600),
            ("1h", 3600),
            ("90s", 90),
        ] {
            let _g = EnvVarGuard::set(STORE_EXCLUSIVE_LOCK_TIMEOUT_ENV, input);
            assert_eq!(
                store_exclusive_lock_timeout(),
                std::time::Duration::from_secs(want_secs),
                "input `{input}` must parse to {want_secs}s",
            );
        }
    }

    /// Invalid env var value → fall through to default (the warn!
    /// is emitted but the timeout is still safe). A typo never
    /// silently drops the lock entirely.
    #[test]
    fn store_exclusive_lock_timeout_falls_through_on_parse_error() {
        let _lock = lock_env();
        let _g = EnvVarGuard::set(STORE_EXCLUSIVE_LOCK_TIMEOUT_ENV, "not-a-duration");
        assert_eq!(
            store_exclusive_lock_timeout(),
            STORE_EXCLUSIVE_LOCK_DEFAULT_TIMEOUT,
            "unparseable env value must fall back to the default \
             rather than zero / disabled — a typo must not silently \
             remove the timeout",
        );
    }
}
