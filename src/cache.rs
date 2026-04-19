//! Kernel image cache for ktstr.
//!
//! Manages a local cache of built kernel images under an XDG-compliant
//! directory. Each cached kernel is a directory containing the boot
//! image, optionally a stripped vmlinux ELF (symbol table, BTF, and
//! the section headers that monitor/probe code reads), and a
//! `metadata.json` descriptor. `CONFIG_HZ` is recovered from the
//! embedded IKCONFIG blob in the stripped vmlinux (ktstr.kconfig
//! forces `CONFIG_IKCONFIG=y`), so no separate `.config` sidecar is
//! cached.
//!
//! # Cache location
//!
//! Resolved in order:
//! 1. `KTSTR_CACHE_DIR` environment variable
//! 2. `$XDG_CACHE_HOME/ktstr/kernels/`
//! 3. `$HOME/.cache/ktstr/kernels/`
//!
//! # Directory structure
//!
//! ```text
//! $CACHE_ROOT/
//!   6.14.2-tarball-x86_64-kc{kconfig_hash}/
//!     bzImage           # kernel boot image
//!     vmlinux           # stripped ELF (optional, see strip_vmlinux_debug)
//!     metadata.json     # KernelMetadata descriptor
//!   local-deadbee-x86_64-kc{kconfig_hash}/
//!     bzImage
//!     vmlinux
//!     metadata.json
//! ```
//!
//! # Atomic writes
//!
//! [`CacheDir::store`] writes to a temporary directory inside the cache
//! root, then atomically renames to the final path. Partial failures
//! never leave corrupt entries. The cache root directory is created
//! lazily on the first `store()`; `new()` and `with_root()` only
//! resolve the path.

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// How a cached kernel's source was acquired, with per-variant
/// payload (git details for `Git`, source-tree path for `Local`).
///
/// Serialized as `{"type": "tarball"}`, `{"type": "git", "hash": ..., "ref": ...}`,
/// or `{"type": "local", "source_tree_path": ...}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase", tag = "type")]
#[non_exhaustive]
pub enum KernelSource {
    /// Downloaded tarball from kernel.org (version / prefix / EOL
    /// probe paths).
    Tarball,
    /// Shallow clone of a git URL at a caller-specified ref.
    Git {
        /// Git commit hash of the kernel source (short form).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        hash: Option<String>,
        /// Git ref used for checkout (branch, tag, or ref spec).
        #[serde(default, rename = "ref", skip_serializing_if = "Option::is_none")]
        git_ref: Option<String>,
    },
    /// Build of a local on-disk kernel source tree.
    Local {
        /// Path to the source tree on disk. `None` when the tree has
        /// been sanitized for remote cache transport or is otherwise
        /// unavailable.
        #[serde(default)]
        source_tree_path: Option<PathBuf>,
    },
}

impl fmt::Display for KernelSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelSource::Tarball => f.write_str("tarball"),
            KernelSource::Git { .. } => f.write_str("git"),
            KernelSource::Local { .. } => f.write_str("local"),
        }
    }
}

/// Metadata stored alongside a cached kernel image.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct KernelMetadata {
    /// Kernel version string (e.g. "6.14.2", "6.15-rc3").
    /// `None` for local builds without a version tag.
    #[serde(default)]
    pub version: Option<String>,
    /// How the kernel source was acquired, with per-source payload.
    pub source: KernelSource,
    /// Target architecture (e.g. "x86_64", "aarch64").
    pub arch: String,
    /// Boot image filename (e.g. "bzImage", "Image").
    pub image_name: String,
    /// CRC32 of the final .config used for the build.
    #[serde(default)]
    pub config_hash: Option<String>,
    /// ISO 8601 timestamp of when the image was built.
    pub built_at: String,
    /// CRC32 of ktstr.kconfig at build time.
    #[serde(default)]
    pub ktstr_kconfig_hash: Option<String>,
    /// Whether a stripped vmlinux ELF was cached alongside the image.
    /// When true, the entry directory contains a `vmlinux` file; see
    /// [`strip_vmlinux_debug`] for the strip policy.
    #[serde(default)]
    pub has_vmlinux: bool,
}

impl KernelMetadata {
    /// Create a new KernelMetadata with required fields.
    ///
    /// Optional fields default to `None` / `false`. Use setter methods
    /// to populate them.
    pub fn new(source: KernelSource, arch: String, image_name: String, built_at: String) -> Self {
        KernelMetadata {
            version: None,
            source,
            arch,
            image_name,
            config_hash: None,
            built_at,
            ktstr_kconfig_hash: None,
            has_vmlinux: false,
        }
    }

    /// Set the kernel version.
    pub fn with_version(mut self, version: Option<String>) -> Self {
        self.version = version;
        self
    }

    /// Set the .config CRC32 hash.
    pub fn with_config_hash(mut self, hash: Option<String>) -> Self {
        self.config_hash = hash;
        self
    }

    /// Set the ktstr.kconfig CRC32 hash.
    pub fn with_ktstr_kconfig_hash(mut self, hash: Option<String>) -> Self {
        self.ktstr_kconfig_hash = hash;
        self
    }
}

/// Bundle of cache artifacts for [`CacheDir::store`].
///
/// Groups the image plus optional sidecars so the store signature
/// stays legible as new artifacts are added. Callers pass by
/// reference; the bundle is copy-out only — the cache reads paths
/// and copies contents.
///
/// The vmlinux path points at the raw (unstripped) ELF. `store()`
/// strips it internally via [`strip_vmlinux_debug`] and writes the
/// result, so callers do not need to run the strip pipeline
/// themselves.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CacheArtifacts<'a> {
    /// Path to the kernel boot image (bzImage or Image).
    pub image: &'a Path,
    /// Optional path to the raw (unstripped) vmlinux ELF. `store()`
    /// strips it internally before caching — see
    /// [`strip_vmlinux_debug`].
    pub vmlinux: Option<&'a Path>,
}

impl<'a> CacheArtifacts<'a> {
    /// Create an artifact bundle with only the required image.
    pub fn new(image: &'a Path) -> Self {
        CacheArtifacts {
            image,
            vmlinux: None,
        }
    }

    /// Attach the raw (unstripped) vmlinux ELF. [`CacheDir::store`]
    /// runs [`strip_vmlinux_debug`] on it before caching.
    pub fn with_vmlinux(mut self, vmlinux: &'a Path) -> Self {
        self.vmlinux = Some(vmlinux);
        self
    }
}

/// Comparison between a cache entry's kconfig hash and a current
/// reference hash. Returned by [`CacheEntry::kconfig_status`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum KconfigStatus {
    /// Entry was built with the current kconfig — nothing to do.
    Matches,
    /// Entry was built with a different kconfig. `cached` and
    /// `current` name the two hashes for diagnostics.
    Stale {
        /// Hash recorded in the cache entry.
        cached: String,
        /// Hash the caller compared against.
        current: String,
    },
    /// Entry has no kconfig hash recorded (pre-tracking cache
    /// format). Treat as unknown; do not assume stale.
    Untracked,
}

// Re-export KernelId from kernel_path (canonical definition, std-only).
pub use crate::kernel_path::KernelId;

/// A cached kernel entry returned by [`CacheDir::lookup`] and
/// [`CacheDir::store`]. Metadata is always present — entries with
/// missing or corrupt metadata surface as [`ListedEntry::Corrupt`]
/// via [`CacheDir::list`] instead.
#[derive(Debug)]
#[non_exhaustive]
pub struct CacheEntry {
    /// Cache key (directory name).
    pub key: String,
    /// Path to the cache entry directory.
    pub path: PathBuf,
    /// Deserialized metadata. Always present on a [`CacheEntry`].
    pub metadata: KernelMetadata,
}

impl CacheEntry {
    /// Absolute path to the cached boot image
    /// (`<entry>/<image_name>`).
    pub fn image_path(&self) -> PathBuf {
        self.path.join(&self.metadata.image_name)
    }

    /// Absolute path to the cached stripped vmlinux ELF, when one
    /// was stored alongside the image. Returns `None` for entries
    /// that were cached without a vmlinux (e.g. the source vmlinux
    /// could not be read at build time).
    pub fn vmlinux_path(&self) -> Option<PathBuf> {
        self.metadata.has_vmlinux.then(|| self.path.join("vmlinux"))
    }

    /// Compare this entry's kconfig hash against `current_hash`.
    pub fn kconfig_status(&self, current_hash: &str) -> KconfigStatus {
        match self.metadata.ktstr_kconfig_hash.as_deref() {
            None => KconfigStatus::Untracked,
            Some(h) if h == current_hash => KconfigStatus::Matches,
            Some(h) => KconfigStatus::Stale {
                cached: h.to_string(),
                current: current_hash.to_string(),
            },
        }
    }

    /// Convenience: true when [`kconfig_status`](Self::kconfig_status)
    /// is [`KconfigStatus::Stale`]. `Untracked` returns false.
    pub fn has_stale_kconfig(&self, current_hash: &str) -> bool {
        matches!(
            self.kconfig_status(current_hash),
            KconfigStatus::Stale { .. }
        )
    }
}

/// Entry yielded by [`CacheDir::list`]. Distinguishes valid entries
/// (with parsed metadata) from corrupt ones (unreadable or
/// unparseable metadata) so callers don't have to re-check `Option`.
#[derive(Debug)]
#[non_exhaustive]
pub enum ListedEntry {
    /// Valid cache entry with parsed metadata.
    Valid(CacheEntry),
    /// Entry directory exists but metadata.json is missing or
    /// fails to parse.
    Corrupt {
        /// Cache key (directory name).
        key: String,
        /// Path to the (corrupt) entry directory.
        path: PathBuf,
    },
}

impl ListedEntry {
    /// Cache key (directory name) for either variant.
    pub fn key(&self) -> &str {
        match self {
            ListedEntry::Valid(e) => &e.key,
            ListedEntry::Corrupt { key, .. } => key,
        }
    }

    /// Path to the entry directory for either variant.
    pub fn path(&self) -> &Path {
        match self {
            ListedEntry::Valid(e) => &e.path,
            ListedEntry::Corrupt { path, .. } => path,
        }
    }

    /// Borrow the valid [`CacheEntry`] payload, or `None` for
    /// [`ListedEntry::Corrupt`].
    pub fn as_valid(&self) -> Option<&CacheEntry> {
        match self {
            ListedEntry::Valid(e) => Some(e),
            ListedEntry::Corrupt { .. } => None,
        }
    }
}

/// Handle to the kernel image cache directory.
///
/// All operations are local filesystem operations via `std::fs`.
/// Thread safety: individual operations are atomic (rename-based
/// writes), but concurrent callers must coordinate externally.
#[derive(Debug)]
pub struct CacheDir {
    root: PathBuf,
}

impl CacheDir {
    /// Open a cache directory at the resolved root path.
    ///
    /// Resolution order:
    /// 1. `KTSTR_CACHE_DIR` environment variable
    /// 2. `$XDG_CACHE_HOME/ktstr/kernels/`
    /// 3. `$HOME/.cache/ktstr/kernels/`
    ///
    /// Only resolves the path; does not create the directory.
    /// [`store`](Self::store) creates the tree lazily on first write.
    /// [`list`](Self::list) and [`lookup`](Self::lookup) handle a
    /// nonexistent root as "no entries".
    pub fn new() -> anyhow::Result<Self> {
        let root = resolve_cache_root()?;
        Ok(CacheDir { root })
    }

    /// Open a cache directory at a specific path.
    ///
    /// Only sets the root path; does not create the directory — the
    /// constructor is infallible. Used by tests and callers that need
    /// an explicit cache location.
    pub fn with_root(root: PathBuf) -> Self {
        CacheDir { root }
    }

    /// Resolve the default cache root path without side effects.
    ///
    /// Returns the path that [`new`](Self::new) would use, without
    /// constructing a [`CacheDir`] or touching the filesystem.
    pub fn resolve_root() -> anyhow::Result<PathBuf> {
        resolve_cache_root()
    }

    /// Root directory of the cache.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Look up a cached kernel by cache key.
    ///
    /// Returns the cache entry if it exists, has valid metadata, and
    /// contains the expected kernel image file. Returns `None` if the
    /// key is invalid, the entry does not exist, or is corrupted.
    pub fn lookup(&self, cache_key: &str) -> Option<CacheEntry> {
        if let Err(e) = validate_cache_key(cache_key) {
            tracing::warn!("invalid cache key: {e}");
            return None;
        }
        let entry_dir = self.root.join(cache_key);
        if !entry_dir.is_dir() {
            return None;
        }
        let metadata = read_metadata(&entry_dir)?;
        // Entry must have a kernel image file.
        if !entry_dir.join(&metadata.image_name).exists() {
            return None;
        }
        Some(CacheEntry {
            key: cache_key.to_string(),
            path: entry_dir,
            metadata,
        })
    }

    /// List all cached kernel entries, sorted by build time (newest
    /// first). Entries with missing or corrupt metadata surface as
    /// [`ListedEntry::Corrupt`] at the end of the Vec.
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
            if !path.is_dir() {
                continue;
            }
            // Skip temp directories from in-progress stores.
            let name = match dir_entry.file_name().into_string() {
                Ok(n) => n,
                Err(_) => continue,
            };
            if name.starts_with(".tmp-") {
                continue;
            }
            match read_metadata(&path) {
                Some(metadata) => entries.push(ListedEntry::Valid(CacheEntry {
                    key: name,
                    path,
                    metadata,
                })),
                None => entries.push(ListedEntry::Corrupt { key: name, path }),
            }
        }
        // Sort by built_at descending (newest first). Corrupt entries
        // have no timestamp and sort last.
        entries.sort_by(|a, b| {
            let a_time = a.as_valid().map(|e| e.metadata.built_at.as_str());
            let b_time = b.as_valid().map(|e| e.metadata.built_at.as_str());
            b_time.cmp(&a_time)
        });
        Ok(entries)
    }

    /// Store a kernel image in the cache.
    ///
    /// `cache_key`: directory name for the entry (e.g.
    /// `6.14.2-tarball-x86_64-kc{kconfig_hash}`).
    ///
    /// `artifacts`: required image plus optional raw vmlinux. When
    /// `artifacts.vmlinux` is set, `store()` runs
    /// [`strip_vmlinux_debug`] internally; on strip failure it falls
    /// back to caching the unstripped vmlinux (logged via
    /// `tracing::warn!`). See [`CacheArtifacts`].
    ///
    /// `metadata`: descriptor to serialize as `metadata.json`. The
    /// `has_vmlinux` field is overwritten based on whether
    /// `artifacts.vmlinux` is present, so callers do not need to
    /// pre-populate it.
    ///
    /// Files are copied (not moved) so the caller retains the
    /// originals. Writes atomically via a temporary directory that is
    /// renamed into place on success. Creates the cache root lazily
    /// if it does not yet exist.
    pub fn store(
        &self,
        cache_key: &str,
        artifacts: &CacheArtifacts<'_>,
        metadata: &KernelMetadata,
    ) -> anyhow::Result<CacheEntry> {
        validate_cache_key(cache_key)?;
        validate_filename(&metadata.image_name)?;
        let final_dir = self.root.join(cache_key);
        let tmp_dir = self
            .root
            .join(format!(".tmp-{}-{}", cache_key, std::process::id()));

        // Clean up any stale temp dir from a prior crash. create_dir_all
        // on tmp_dir also creates self.root lazily on first store.
        if tmp_dir.exists() {
            fs::remove_dir_all(&tmp_dir)?;
        }
        fs::create_dir_all(&tmp_dir)?;

        // TmpGuard ensures the temp dir is cleaned up on any error
        // path, including serde serialization failures.
        let guard = TmpDirGuard(&tmp_dir);

        // Copy boot image.
        let image_dest = tmp_dir.join(&metadata.image_name);
        fs::copy(artifacts.image, &image_dest)
            .map_err(|e| anyhow::anyhow!("copy kernel image to cache: {e}"))?;

        // Strip the raw vmlinux and copy the stripped result into the
        // cache entry. If the strip pipeline errors (e.g. ELF too
        // exotic for object::build::Builder), fall back to caching
        // the unstripped vmlinux so downstream callers still get
        // symbols and BTF — the size penalty is preferable to losing
        // the vmlinux entirely.
        let has_vmlinux = if let Some(vmlinux) = artifacts.vmlinux {
            let vmlinux_dest = tmp_dir.join("vmlinux");
            match strip_vmlinux_debug(vmlinux) {
                Ok(stripped) => {
                    fs::copy(stripped.path(), &vmlinux_dest)
                        .map_err(|e| anyhow::anyhow!("copy stripped vmlinux to cache: {e}"))?;
                }
                Err(e) => {
                    tracing::warn!("vmlinux strip failed ({e:#}), caching unstripped",);
                    fs::copy(vmlinux, &vmlinux_dest)
                        .map_err(|e| anyhow::anyhow!("copy vmlinux to cache: {e}"))?;
                }
            }
            true
        } else {
            false
        };

        // Write metadata. has_vmlinux reflects whether we actually
        // stored a vmlinux sidecar, overriding whatever the caller set.
        let mut meta = metadata.clone();
        meta.has_vmlinux = has_vmlinux;
        let meta_json = serde_json::to_string_pretty(&meta)?;
        fs::write(tmp_dir.join("metadata.json"), meta_json)
            .map_err(|e| anyhow::anyhow!("write cache metadata: {e}"))?;

        // Rename-to-staging swap. A naive "remove_dir_all(final_dir)
        // then rename(tmp_dir, final_dir)" has a TOCTOU window: a
        // concurrent store of the same key could re-create final_dir
        // between the remove and the rename, and the second rename
        // would fail with ENOTEMPTY, leaving our fresh content
        // orphaned in tmp_dir.
        //
        // Instead, when final_dir already exists, atomically move the
        // old content sideways into a thread-unique staging dir and
        // then rename our fresh tmp_dir into place. Best-effort clean
        // the staging dir afterwards via RAII.
        match fs::rename(&tmp_dir, &final_dir) {
            Ok(()) => {}
            Err(e)
                if e.raw_os_error() == Some(libc::ENOTEMPTY)
                    || e.raw_os_error() == Some(libc::EEXIST) =>
            {
                // Thread id disambiguates same-process concurrent
                // stores of the same key; without it, two threads
                // would collide on the staging path.
                let staging_dir = self.root.join(format!(
                    ".evict-{}-{}-{:?}",
                    cache_key,
                    std::process::id(),
                    std::thread::current().id(),
                ));
                // A stale staging dir from a crashed prior store must
                // go before we try to rename into the slot.
                if staging_dir.exists() {
                    let _ = fs::remove_dir_all(&staging_dir);
                }
                fs::rename(&final_dir, &staging_dir)
                    .map_err(|e2| anyhow::anyhow!("atomic rename final_dir to staging: {e2}"))?;
                // Guard cleans up staging_dir on every exit from this
                // block. On successful rollback below, staging_dir no
                // longer exists and the guard's remove_dir_all is a
                // harmless no-op. On rollback failure we disarm so the
                // old content persists in staging rather than being
                // deleted.
                let stage_guard = TmpDirGuard(&staging_dir);
                if let Err(e2) = fs::rename(&tmp_dir, &final_dir) {
                    // Install failed. Try to restore the previous
                    // cache content so readers don't see a missing
                    // entry. If rollback also fails, preserve the
                    // staging dir so the old content isn't lost.
                    if let Err(rollback_err) = fs::rename(&staging_dir, &final_dir) {
                        tracing::warn!(
                            "cache rollback failed, preserving staging dir: {rollback_err}"
                        );
                        stage_guard.disarm();
                    }
                    return Err(anyhow::anyhow!(
                        "atomic rename tmp_dir to final_dir (retry): {e2}"
                    ));
                }
            }
            Err(e) => {
                return Err(anyhow::anyhow!("atomic rename cache entry: {e}"));
            }
        }

        // Rename succeeded — disarm the cleanup guard.
        guard.disarm();

        Ok(CacheEntry {
            key: cache_key.to_string(),
            path: final_dir,
            metadata: meta,
        })
    }

    /// Remove every cached entry. Returns the number of entries
    /// removed.
    pub fn clean_all(&self) -> anyhow::Result<usize> {
        self.remove_entries(self.list()?)
    }

    /// Remove every cached entry except the `keep` most recent ones
    /// (by `built_at` timestamp). `keep == 0` is equivalent to
    /// [`clean_all`](Self::clean_all). Returns the number of entries
    /// removed.
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
}

/// Validate a cache key.
///
/// Rejects empty keys, whitespace-only keys, keys starting with
/// `.tmp-` (reserved for in-progress stores), and keys containing
/// path separators (`/`, `\`), parent-directory traversal (`..`),
/// or null bytes. Returns `Ok(())` on valid keys.
fn validate_cache_key(key: &str) -> anyhow::Result<()> {
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
    if key.starts_with(".tmp-") {
        anyhow::bail!("cache key must not start with .tmp- (reserved): {key:?}");
    }
    Ok(())
}

/// Validate a filename (e.g. image_name in metadata).
///
/// Rejects empty names, path separators (`/`, `\`), parent-directory
/// traversal (`..`), and null bytes to prevent path traversal when
/// joining the filename to a directory path.
fn validate_filename(name: &str) -> anyhow::Result<()> {
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
///
/// Call [`disarm`](TmpDirGuard::disarm) after a successful rename to
/// prevent cleanup of the (now-moved) directory.
struct TmpDirGuard<'a>(&'a Path);

impl TmpDirGuard<'_> {
    /// Prevent cleanup. Call after the tmp dir has been renamed.
    fn disarm(self) {
        std::mem::forget(self);
    }
}

impl Drop for TmpDirGuard<'_> {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(self.0);
    }
}

/// Read and deserialize metadata.json from a cache entry directory.
fn read_metadata(dir: &Path) -> Option<KernelMetadata> {
    let meta_path = dir.join("metadata.json");
    let contents = fs::read_to_string(meta_path).ok()?;
    serde_json::from_str(&contents).ok()
}

/// Re-route a cache-entry directory to its original source tree when
/// blazesym DWARF access is required.
///
/// Cache entries carry vmlinux stripped of `.debug_*` sections
/// (see [`strip_vmlinux_debug`]), so pointing blazesym at the cache
/// directory gives no file:line. [`KernelSource::Local`] entries record
/// the original build tree in the variant's `source_tree_path` field;
/// when it is populated and the tree still has an unstripped vmlinux
/// on disk, this helper returns that path so callers can route
/// blazesym there instead.
///
/// Returns `None` when:
/// - `dir` has no `metadata.json` (not a cache entry, e.g. a build-tree root).
/// - Metadata parse fails.
/// - `metadata.source` is not [`KernelSource::Local`] (tarball/git entries
///   — no original source tree to route to).
/// - `source_tree_path` is `None` on a Local entry (tree location not recorded).
/// - `source_tree_path` is set but has no `vmlinux` file (tree deleted
///   or rebuilt without saving vmlinux).
///
/// In any of those cases callers should fall back to the cache
/// directory for symbol/BTF lookup — file:line is genuinely
/// unrecoverable without re-downloading sources.
pub fn prefer_source_tree_for_dwarf(dir: &Path) -> Option<PathBuf> {
    let metadata = read_metadata(dir)?;
    let KernelSource::Local { source_tree_path } = metadata.source else {
        return None;
    };
    let src_path = source_tree_path?;
    if src_path.join("vmlinux").is_file() {
        Some(src_path)
    } else {
        None
    }
}

/// Structural ELF sections that must survive any cache-time strip
/// so that downstream readers can parse the result. Not tied to any
/// specific consumer — independent of monitor or probe code.
const STRUCTURAL_KEEP_SECTIONS: &[&[u8]] = &[
    b"",          // null section (index 0) — required by ELF spec
    b".shstrtab", // section header string table
];

/// Data sections retained as SHT_NOBITS headers with no current
/// consumer. Kept defensively so that symbols future kernels might
/// place here survive `Builder::delete_orphans`; remove an entry
/// only if no in-tree or upstream kernel version places monitored
/// symbols in it.
const SPECULATIVE_ZERO_DATA_SECTIONS: &[&[u8]] = &[b".init.data"];

/// Union of consumer-declared keep-lists plus structural sections.
///
/// Each consumer module owns the list of ELF sections it reads —
/// see [`crate::monitor::symbols::VMLINUX_KEEP_SECTIONS`],
/// [`crate::monitor::VMLINUX_KEEP_SECTIONS`], and
/// [`crate::probe::btf::VMLINUX_KEEP_SECTIONS`]. `is_keep_section`
/// unions them with [`STRUCTURAL_KEEP_SECTIONS`] at strip time.
///
/// Keep-list (vs remove-list) is safer: new debug or data sections
/// added by future compiler / kernel versions are stripped
/// automatically. Sections not matched by `is_keep_section` or
/// `is_zero_data_section` are deleted outright (non-code) or have
/// their bytes dropped via SHT_NOBITS (code — see [`strip_keep_list`]).
fn is_keep_section(name: &[u8]) -> bool {
    STRUCTURAL_KEEP_SECTIONS.contains(&name)
        || crate::monitor::symbols::VMLINUX_KEEP_SECTIONS.contains(&name)
        || crate::monitor::VMLINUX_KEEP_SECTIONS.contains(&name)
        || crate::probe::btf::VMLINUX_KEEP_SECTIONS.contains(&name)
}

/// Union of consumer-declared zero-data lists plus the speculative
/// retention set.
///
/// Data sections whose bytes are dropped (converted to SHT_NOBITS +
/// zero-length data) while their headers and addresses are
/// preserved. Monitor code reads only symbol addresses (`st_value`)
/// from these, never the backing bytes. Keeping the header lets
/// symbols with `st_shndx` pointing at them survive
/// `Builder::delete_orphans`.
///
/// See [`crate::monitor::symbols::VMLINUX_ZERO_DATA_SECTIONS`] for
/// the current consumer list and [`SPECULATIVE_ZERO_DATA_SECTIONS`]
/// for retained-without-consumer entries.
fn is_zero_data_section(name: &[u8]) -> bool {
    SPECULATIVE_ZERO_DATA_SECTIONS.contains(&name)
        || crate::monitor::symbols::VMLINUX_ZERO_DATA_SECTIONS.contains(&name)
}

/// Stripped vmlinux written to a temporary file.
///
/// Owns the backing [`tempfile::TempDir`] so the file stays alive
/// until the caller drops the [`StrippedVmlinux`]. Callers read the
/// stripped-vmlinux path via [`path`](Self::path) to pass into
/// [`CacheDir::store`] (or equivalent consumer) before the handle
/// goes out of scope.
#[derive(Debug)]
pub(crate) struct StrippedVmlinux {
    _tmp: tempfile::TempDir,
    path: PathBuf,
}

impl StrippedVmlinux {
    /// Path to the stripped vmlinux file inside the owned temp dir.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Strip vmlinux for caching. Sections fall into three groups:
/// bytes preserved (matched by [`is_keep_section`] — symbol tables,
/// BTF, `.rodata` for IKCONFIG, `.bss`, `.shstrtab`), header-only
/// via SHT_NOBITS (matched by [`is_zero_data_section`] plus all
/// code sections, so symbols with `st_shndx` pointing at them
/// survive), and deleted (everything else — DWARF `.debug_*`,
/// relocations, etc.). See [`strip_keep_list`] for the dispatch
/// detail.
///
/// The cached vmlinux is consumed by monitor and probe code for
/// symbol addresses (`.symtab`) and BTF type info (`.BTF`). DWARF
/// from the build tree (not the cache) is used by blazesym for probe
/// source locations — stripping the cached copy does not affect that
/// path.
///
/// If the keep-list strip fails (e.g. `Builder::read` encounters
/// an unsupported ELF feature), falls back to removing only
/// `.debug_*` sections, which preserves all other sections
/// including those the symbol table references.
///
/// Returns a [`StrippedVmlinux`] handle that owns the backing temp
/// directory; the caller must keep it alive until consumers (e.g.
/// [`CacheDir::store`]) have copied the file.
pub(crate) fn strip_vmlinux_debug(vmlinux_path: &Path) -> anyhow::Result<StrippedVmlinux> {
    let raw =
        fs::read(vmlinux_path).map_err(|e| anyhow::anyhow!("read vmlinux for stripping: {e}"))?;
    let original_size = raw.len();
    let data = neutralize_alloc_relocs(&raw)
        .map_err(|e| anyhow::anyhow!("preprocess vmlinux ELF: {e}"))?;

    let out = match strip_keep_list(&data) {
        Ok(buf) => buf,
        Err(e) => {
            tracing::warn!("keep-list strip failed ({e:#}), falling back to debug-only strip");
            strip_debug_prefix(&data)?
        }
    };

    let stripped_size = out.len();
    let saved_mb = (original_size - stripped_size) as f64 / (1024.0 * 1024.0);
    tracing::debug!(
        original = original_size,
        stripped = stripped_size,
        saved_mb = format!("{saved_mb:.0}"),
        "strip_vmlinux_debug",
    );

    let tmp_dir = tempfile::TempDir::new()
        .map_err(|e| anyhow::anyhow!("create temp dir for stripped vmlinux: {e}"))?;
    let stripped_path = tmp_dir.path().join("vmlinux");
    fs::write(&stripped_path, &out).map_err(|e| anyhow::anyhow!("write stripped vmlinux: {e}"))?;
    Ok(StrippedVmlinux {
        _tmp: tmp_dir,
        path: stripped_path,
    })
}

/// Zero `sh_size` on every `SHT_REL`/`SHT_RELA` section that has the
/// `SHF_ALLOC` flag set, returning a modified copy of the bytes.
///
/// Workaround for `object::build::elf::Builder::read`: the Builder
/// treats any `SHF_ALLOC` relocation section as a dynamic-relocation
/// section and parses each entry against an empty (zero-length)
/// dynamic symbol table. Any entry referencing a non-null symbol
/// index then trips the bounds check at `read_relocations_impl` and
/// the whole read fails with `Invalid symbol index N in relocation
/// section at index M`. Kernels built with `CONFIG_RELOCATABLE` +
/// `CONFIG_RANDOMIZE_BASE` (any x86_64 defconfig + kASLR build) emit
/// such sections (e.g. `.rela.dyn`-style entries for kASLR /
/// static-call patching) so the Builder cannot parse the vmlinux
/// at all -- both the keep-list strip and the debug-only fallback
/// fail at parse time, and `strip_vmlinux_debug` returns an error
/// that the cache build path silently swallows (caching the
/// unstripped vmlinux), and the test path bubbles up as a panic.
///
/// Zeroing `sh_size` makes the Builder see these sections as empty,
/// so the relocation walk finds no entries and the parse succeeds.
/// The keep-list pass then deletes the sections by name like any
/// other non-kept section. The output is identical to what we would
/// have written if these sections had never been parsed.
///
/// No-op for ELFs that have no `SHF_ALLOC` relocation sections
/// (returns the original bytes copied into a new `Vec`).
fn neutralize_alloc_relocs(data: &[u8]) -> anyhow::Result<Vec<u8>> {
    let elf = goblin::elf::Elf::parse(data)
        .map_err(|e| anyhow::anyhow!("parse vmlinux ELF for preprocess: {e}"))?;
    let mut out = data.to_vec();
    let shoff = elf.header.e_shoff as usize;
    let shentsize = elf.header.e_shentsize as usize;
    // sh_size byte offset and width within a section header entry.
    // ELF64 section header layout: sh_name(4) sh_type(4) sh_flags(8)
    // sh_addr(8) sh_offset(8) sh_size(8) ... -> sh_size at offset 32.
    // ELF32 layout: sh_name(4) sh_type(4) sh_flags(4) sh_addr(4)
    // sh_offset(4) sh_size(4) ... -> sh_size at offset 20.
    let (sh_size_offset, sh_size_width) = if elf.is_64 { (32, 8) } else { (20, 4) };
    use goblin::elf::section_header::{SHF_ALLOC, SHT_REL, SHT_RELA};
    for (i, sh) in elf.section_headers.iter().enumerate() {
        let is_rela = sh.sh_type == SHT_RELA || sh.sh_type == SHT_REL;
        let is_alloc = sh.sh_flags & u64::from(SHF_ALLOC) != 0;
        if !(is_rela && is_alloc) {
            continue;
        }
        let entry_offset = shoff
            .checked_add(
                i.checked_mul(shentsize)
                    .ok_or_else(|| anyhow::anyhow!("section header table overflow at index {i}"))?,
            )
            .ok_or_else(|| anyhow::anyhow!("section header offset overflow at index {i}"))?;
        let size_offset = entry_offset
            .checked_add(sh_size_offset)
            .ok_or_else(|| anyhow::anyhow!("sh_size offset overflow at index {i}"))?;
        let size_end = size_offset
            .checked_add(sh_size_width)
            .ok_or_else(|| anyhow::anyhow!("sh_size end overflow at index {i}"))?;
        if size_end > out.len() {
            anyhow::bail!("sh_size at section header {i} extends past file end");
        }
        // Zero is endian-agnostic.
        out[size_offset..size_end].fill(0);
    }
    Ok(out)
}

/// Keep-list strip: three-way partition of ELF sections.
///
/// Sections matched by [`is_keep_section`] keep their bytes (symbol
/// tables, BTF, `.shstrtab`, `.rodata` for IKCONFIG, `.bss` already
/// SHT_NOBITS).
///
/// Sections matched by [`is_zero_data_section`] have their headers
/// preserved but bytes dropped via `SHT_NOBITS` + zero-length data.
/// Monitor code reads symbol addresses (`st_value`) from these, never
/// the backing bytes.
///
/// Code sections (`SHF_EXECINSTR`: `.text`, `.init.text`,
/// `.exit.text`, `.text.hot`, `.altinstr_replacement`, etc.) receive
/// the same SHT_NOBITS treatment so that ~115k function symbols
/// pointing into them (`schedule`, `__schedule`, etc.) survive
/// `Builder::delete_orphans` — the auto-pass at the top of
/// `Builder::write` that drops any symbol whose section was deleted.
/// Without this, `resolve_addrs_from_elf` (probe/output.rs) returns
/// an empty vec for any kernel function lookup.
///
/// Everything else is deleted outright (DWARF `.debug_*`, relocation
/// sections, etc.).
///
/// After stripping, verifies the result has a non-empty symbol table.
/// Returns an error to trigger the fallback to `strip_debug_prefix`
/// if the symbol table is empty.
fn strip_keep_list(data: &[u8]) -> anyhow::Result<Vec<u8>> {
    let mut builder = object::build::elf::Builder::read(data)
        .map_err(|e| anyhow::anyhow!("parse vmlinux ELF: {e}"))?;
    for section in builder.sections.iter_mut() {
        let name = section.name.as_slice();
        if is_keep_section(name) {
            continue;
        }
        if is_zero_data_section(name) {
            section.sh_type = object::elf::SHT_NOBITS;
            section.data = object::build::elf::SectionData::UninitializedData(0);
            continue;
        }
        let is_code = section.sh_flags & u64::from(object::elf::SHF_EXECINSTR) != 0;
        if is_code {
            section.sh_type = object::elf::SHT_NOBITS;
            section.data = object::build::elf::SectionData::UninitializedData(0);
        } else {
            section.delete = true;
        }
    }
    let mut out = Vec::new();
    builder
        .write(&mut out)
        .map_err(|e| anyhow::anyhow!("rewrite stripped vmlinux: {e}"))?;

    // Verify symtab survived. goblin always includes the null
    // symbol (index 0), so check for at least one symbol with a
    // non-empty name.
    let elf =
        goblin::elf::Elf::parse(&out).map_err(|e| anyhow::anyhow!("verify stripped ELF: {e}"))?;
    let named_syms = elf
        .syms
        .iter()
        .filter(|s| s.st_name != 0 && elf.strtab.get_at(s.st_name).is_some_and(|n| !n.is_empty()))
        .count();
    if named_syms == 0 {
        anyhow::bail!("keep-list strip emptied symbol table (0 named symbols)");
    }
    Ok(out)
}

/// Fallback strip: remove only .debug_* and .comment sections.
fn strip_debug_prefix(data: &[u8]) -> anyhow::Result<Vec<u8>> {
    let mut builder = object::build::elf::Builder::read(data)
        .map_err(|e| anyhow::anyhow!("parse vmlinux ELF (fallback): {e}"))?;
    for section in builder.sections.iter_mut() {
        let name = section.name.as_slice();
        if name.starts_with(b".debug_") || name == b".comment" {
            section.delete = true;
        }
    }
    let mut out = Vec::new();
    builder
        .write(&mut out)
        .map_err(|e| anyhow::anyhow!("rewrite stripped vmlinux (fallback): {e}"))?;
    Ok(out)
}

/// Resolve the cache root directory path.
///
/// Does not create the directory -- the caller is responsible for
/// ensuring it exists.
fn resolve_cache_root() -> anyhow::Result<PathBuf> {
    // 1. Explicit override.
    if let Ok(dir) = std::env::var("KTSTR_CACHE_DIR")
        && !dir.is_empty()
    {
        return Ok(PathBuf::from(dir));
    }
    // 2. XDG_CACHE_HOME/ktstr/kernels.
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME")
        && !xdg.is_empty()
    {
        return Ok(PathBuf::from(xdg).join("ktstr").join("kernels"));
    }
    // 3. $HOME/.cache/ktstr/kernels.
    let home = std::env::var("HOME").map_err(|_| {
        anyhow::anyhow!(
            "HOME not set; cannot resolve cache directory. \
             Set KTSTR_CACHE_DIR to specify a cache location."
        )
    })?;
    Ok(PathBuf::from(home)
        .join(".cache")
        .join("ktstr")
        .join("kernels"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_metadata(version: &str) -> KernelMetadata {
        KernelMetadata {
            version: Some(version.to_string()),
            source: KernelSource::Tarball,
            arch: "x86_64".to_string(),
            image_name: "bzImage".to_string(),
            config_hash: Some("abc123".to_string()),
            built_at: "2026-04-12T10:00:00Z".to_string(),
            ktstr_kconfig_hash: Some("def456".to_string()),
            has_vmlinux: false,
        }
    }

    fn create_fake_image(dir: &Path) -> PathBuf {
        let image = dir.join("bzImage");
        fs::write(&image, b"fake kernel image").unwrap();
        image
    }

    // -- KernelMetadata serde --

    #[test]
    fn cache_metadata_serde_roundtrip() {
        let meta = test_metadata("6.14.2");
        let json = serde_json::to_string_pretty(&meta).unwrap();
        let parsed: KernelMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.version.as_deref(), Some("6.14.2"));
        assert_eq!(parsed.source, KernelSource::Tarball);
        assert_eq!(parsed.arch, "x86_64");
        assert_eq!(parsed.image_name, "bzImage");
        assert_eq!(parsed.config_hash.as_deref(), Some("abc123"));
        assert_eq!(parsed.built_at, "2026-04-12T10:00:00Z");
        assert_eq!(parsed.ktstr_kconfig_hash.as_deref(), Some("def456"));
        assert!(!parsed.has_vmlinux);
    }

    #[test]
    fn cache_metadata_serde_git_with_payload() {
        let meta = KernelMetadata {
            version: Some("6.15-rc3".to_string()),
            source: KernelSource::Git {
                hash: Some("a1b2c3d".to_string()),
                git_ref: Some("v6.15-rc3".to_string()),
            },
            arch: "aarch64".to_string(),
            image_name: "Image".to_string(),
            config_hash: None,
            built_at: "2026-04-12T12:00:00Z".to_string(),
            ktstr_kconfig_hash: None,
            has_vmlinux: false,
        };
        let json = serde_json::to_string(&meta).unwrap();
        let parsed: KernelMetadata = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            parsed.source,
            KernelSource::Git {
                hash: Some(ref h),
                git_ref: Some(ref r),
            }
            if h == "a1b2c3d" && r == "v6.15-rc3"
        ));
    }

    #[test]
    fn cache_metadata_serde_local_with_source_tree() {
        let meta = KernelMetadata {
            version: Some("6.14.0".to_string()),
            source: KernelSource::Local {
                source_tree_path: Some(PathBuf::from("/tmp/linux")),
            },
            arch: "x86_64".to_string(),
            image_name: "bzImage".to_string(),
            config_hash: Some("fff000".to_string()),
            built_at: "2026-04-12T14:00:00Z".to_string(),
            ktstr_kconfig_hash: Some("aaa111".to_string()),
            has_vmlinux: true,
        };
        let json = serde_json::to_string(&meta).unwrap();
        let parsed: KernelMetadata = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            parsed.source,
            KernelSource::Local {
                source_tree_path: Some(ref p),
            }
            if p == &PathBuf::from("/tmp/linux")
        ));
        assert!(parsed.has_vmlinux);
    }

    #[test]
    fn cache_metadata_deserialize_tagged_tarball() {
        // Minimal Tarball entry with optional fields absent.
        let json = r#"{
            "version": "6.14.2",
            "source": {"type": "tarball"},
            "arch": "x86_64",
            "image_name": "bzImage",
            "built_at": "2026-04-12T10:00:00Z"
        }"#;
        let parsed: KernelMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.version.as_deref(), Some("6.14.2"));
        assert_eq!(parsed.source, KernelSource::Tarball);
        assert!(parsed.config_hash.is_none());
        assert!(parsed.ktstr_kconfig_hash.is_none());
        assert!(!parsed.has_vmlinux);
    }

    #[test]
    fn cache_metadata_deserialize_null_version() {
        let json = r#"{
            "version": null,
            "source": {"type": "local", "source_tree_path": null},
            "arch": "x86_64",
            "image_name": "bzImage",
            "config_hash": null,
            "built_at": "2026-04-12T10:00:00Z",
            "ktstr_kconfig_hash": null
        }"#;
        let parsed: KernelMetadata = serde_json::from_str(json).unwrap();
        assert!(parsed.version.is_none());
        assert!(matches!(
            parsed.source,
            KernelSource::Local {
                source_tree_path: None
            }
        ));
    }

    #[test]
    fn kernel_source_serde_tagged_representation() {
        // Verify the tagged JSON shape on each variant.
        let t = serde_json::to_string(&KernelSource::Tarball).unwrap();
        assert_eq!(t, r#"{"type":"tarball"}"#);
        let g = serde_json::to_string(&KernelSource::Git {
            hash: Some("abc".to_string()),
            git_ref: Some("main".to_string()),
        })
        .unwrap();
        assert!(g.contains(r#""type":"git""#));
        assert!(g.contains(r#""hash":"abc""#));
        assert!(g.contains(r#""ref":"main""#));
        let l = serde_json::to_string(&KernelSource::Local {
            source_tree_path: Some(PathBuf::from("/tmp/linux")),
        })
        .unwrap();
        assert!(l.contains(r#""type":"local""#));
        assert!(l.contains(r#""source_tree_path":"/tmp/linux""#));
    }

    // -- CacheDir --

    #[test]
    fn cache_dir_with_root_does_not_create_dir() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("kernels");
        assert!(!root.exists());
        let cache = CacheDir::with_root(root.clone());
        // Resolution must not create the directory — store() does it
        // lazily on first write.
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
    fn cache_dir_resolve_root_returns_path() {
        let tmp = TempDir::new().unwrap();
        let _guard = EnvVarGuard::set("KTSTR_CACHE_DIR", tmp.path().to_str().unwrap());
        let resolved = CacheDir::resolve_root().unwrap();
        assert_eq!(resolved, tmp.path());
        // Side-effect-free: calling resolve_root() must not create
        // any directories beyond what the env var already pointed at.
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

        // Create a fake kernel image.
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");

        // Store.
        let entry = cache
            .store("6.14.2-tarball-x86_64", &CacheArtifacts::new(&image), &meta)
            .unwrap();
        assert_eq!(entry.key, "6.14.2-tarball-x86_64");
        assert!(entry.path.join("bzImage").exists());
        assert!(entry.path.join("metadata.json").exists());

        // Lookup.
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

        // Create entry dir with image but corrupt metadata.
        let entry_dir = tmp.path().join("bad-entry");
        fs::create_dir_all(&entry_dir).unwrap();
        fs::write(entry_dir.join("bzImage"), b"fake").unwrap();
        fs::write(entry_dir.join("metadata.json"), b"not json").unwrap();

        // lookup returns None because metadata is corrupt and we
        // cannot determine the image_name field.
        let found = cache.lookup("bad-entry");
        assert!(found.is_none());
    }

    #[test]
    fn cache_dir_lookup_missing_image() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());

        // Create entry dir with valid metadata but no image file.
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
            ..test_metadata("6.14.2")
        };
        cache
            .store(
                "6.14.2-tarball-x86_64",
                &CacheArtifacts::new(&image),
                &meta1,
            )
            .unwrap();

        let meta2 = KernelMetadata {
            built_at: "2026-04-12T11:00:00Z".to_string(),
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

        // Store in non-chronological order.
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

        // Create a valid entry.
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");
        cache
            .store("valid", &CacheArtifacts::new(&image), &meta)
            .unwrap();

        // Create a corrupt entry (no metadata).
        let bad_dir = tmp.path().join("corrupt");
        fs::create_dir_all(&bad_dir).unwrap();

        let entries = cache.list().unwrap();
        assert_eq!(entries.len(), 2);
        // Valid entry surfaces as ListedEntry::Valid.
        let valid = entries.iter().find(|e| e.key() == "valid").unwrap();
        assert!(valid.as_valid().is_some());
        // Corrupt entry surfaces as ListedEntry::Corrupt.
        let corrupt = entries.iter().find(|e| e.key() == "corrupt").unwrap();
        assert!(corrupt.as_valid().is_none());
    }

    #[test]
    fn cache_dir_list_skips_tmp_dirs() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());

        // Create a .tmp- directory (in-progress store).
        let tmp_dir = tmp.path().join(".tmp-in-progress-12345");
        fs::create_dir_all(&tmp_dir).unwrap();

        let entries = cache.list().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn cache_dir_list_skips_regular_files() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());

        // Create a regular file in the cache root.
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

    // -- resolve_cache_root --

    #[test]
    fn cache_resolve_root_ktstr_cache_dir() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("custom-cache");
        // Temporarily set env var for this test.
        let _guard = EnvVarGuard::set("KTSTR_CACHE_DIR", dir.to_str().unwrap());
        let root = resolve_cache_root().unwrap();
        assert_eq!(root, dir);
    }

    #[test]
    fn cache_resolve_root_xdg_cache_home() {
        let tmp = TempDir::new().unwrap();
        let _guard1 = EnvVarGuard::remove("KTSTR_CACHE_DIR");
        let _guard2 = EnvVarGuard::set("XDG_CACHE_HOME", tmp.path().to_str().unwrap());
        let root = resolve_cache_root().unwrap();
        assert_eq!(root, tmp.path().join("ktstr").join("kernels"));
    }

    #[test]
    fn cache_resolve_root_empty_ktstr_cache_dir_falls_through() {
        let tmp = TempDir::new().unwrap();
        let _guard1 = EnvVarGuard::set("KTSTR_CACHE_DIR", "");
        let _guard2 = EnvVarGuard::set("XDG_CACHE_HOME", tmp.path().to_str().unwrap());
        let root = resolve_cache_root().unwrap();
        assert_eq!(root, tmp.path().join("ktstr").join("kernels"));
    }

    #[test]
    fn cache_resolve_root_empty_xdg_falls_to_home() {
        let tmp = TempDir::new().unwrap();
        let _guard1 = EnvVarGuard::remove("KTSTR_CACHE_DIR");
        let _guard2 = EnvVarGuard::set("XDG_CACHE_HOME", "");
        let _guard3 = EnvVarGuard::set("HOME", tmp.path().to_str().unwrap());
        let root = resolve_cache_root().unwrap();
        assert_eq!(
            root,
            tmp.path().join(".cache").join("ktstr").join("kernels")
        );
    }

    // -- resolve_cache_root error paths --

    #[test]
    fn cache_resolve_root_home_unset_error() {
        let _guard1 = EnvVarGuard::remove("KTSTR_CACHE_DIR");
        let _guard2 = EnvVarGuard::remove("XDG_CACHE_HOME");
        let _guard3 = EnvVarGuard::remove("HOME");
        let err = resolve_cache_root().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("HOME not set"),
            "expected HOME-unset error, got: {msg}"
        );
        assert!(
            msg.contains("KTSTR_CACHE_DIR"),
            "error should suggest KTSTR_CACHE_DIR, got: {msg}"
        );
    }

    // -- validate_cache_key unit tests --

    #[test]
    fn cache_validate_key_rejects_empty() {
        let err = validate_cache_key("").unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn cache_validate_key_rejects_whitespace_only() {
        let err = validate_cache_key("   ").unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn cache_validate_key_rejects_forward_slash() {
        let err = validate_cache_key("a/b").unwrap_err();
        assert!(err.to_string().contains("path separator"));
    }

    #[test]
    fn cache_validate_key_rejects_backslash() {
        let err = validate_cache_key("a\\b").unwrap_err();
        assert!(err.to_string().contains("path separator"));
    }

    #[test]
    fn cache_validate_key_rejects_dotdot() {
        // Use a key without slashes to specifically hit the ".." check
        // (slashes are rejected first by the separator check).
        let err = validate_cache_key("foo..bar").unwrap_err();
        assert!(err.to_string().contains("path traversal"));
    }

    #[test]
    fn cache_validate_key_rejects_null_byte() {
        let err = validate_cache_key("key\0evil").unwrap_err();
        assert!(err.to_string().contains("null"));
    }

    #[test]
    fn cache_validate_key_rejects_tmp_prefix() {
        let err = validate_cache_key(".tmp-in-progress").unwrap_err();
        assert!(
            err.to_string().contains(".tmp-"),
            "expected .tmp- rejection, got: {err}"
        );
    }

    #[test]
    fn cache_validate_key_rejects_dot() {
        let err = validate_cache_key(".").unwrap_err();
        assert!(
            err.to_string().contains("directory reference"),
            "expected dot rejection, got: {err}"
        );
    }

    #[test]
    fn cache_validate_key_rejects_dotdot_bare() {
        let err = validate_cache_key("..").unwrap_err();
        assert!(
            err.to_string().contains("directory reference"),
            "expected dotdot rejection, got: {err}"
        );
    }

    #[test]
    fn cache_validate_key_accepts_valid() {
        assert!(validate_cache_key("6.14.2-tarball-x86_64").is_ok());
        assert!(validate_cache_key("local-deadbeef-x86_64").is_ok());
        assert!(validate_cache_key("v6.14-git-a1b2c3d-aarch64").is_ok());
    }

    // -- validate_filename --

    #[test]
    fn cache_validate_filename_rejects_traversal() {
        assert!(validate_filename("../etc/passwd").is_err());
        assert!(validate_filename("foo/../bar").is_err());
    }

    #[test]
    fn cache_validate_filename_rejects_empty() {
        assert!(validate_filename("").is_err());
    }

    #[test]
    fn cache_validate_filename_accepts_valid() {
        assert!(validate_filename("bzImage").is_ok());
        assert!(validate_filename("Image").is_ok());
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

        // Two valid entries with different timestamps.
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

        // One corrupt entry (no metadata).
        let corrupt_dir = tmp.path().join("cache").join("corrupt");
        fs::create_dir_all(&corrupt_dir).unwrap();

        // list() returns 3 entries. Corrupt entries (no built_at) sort
        // last. keep=1 should keep the newest valid entry and remove
        // the old valid + corrupt entries.
        let removed = cache.clean_keep(1).unwrap();
        assert_eq!(removed, 2);

        let remaining = cache.list().unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].key(), "new");
    }

    // -- atomic write safety --

    /// Regression for the rename-TOCTOU fix: a second `store` with
    /// the same key must atomically replace the previous entry's
    /// content without leaving half-installed state — even though
    /// the underlying code path exercises the
    /// `final_dir-exists → swap` branch rather than the plain
    /// rename. The new content wins, the old content is gone, and
    /// no `.evict-*` staging dir lingers under cache_root.
    #[test]
    fn cache_dir_store_overwrites_existing_key_atomically() {
        let tmp = TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let cache = CacheDir::with_root(cache_root.clone());

        // First install.
        let src_a = TempDir::new().unwrap();
        let image_a = create_fake_image(src_a.path());
        fs::write(&image_a, b"version-a").unwrap();
        let mut meta_a = test_metadata("6.14.2");
        meta_a.built_at = "2026-04-10T00:00:00Z".to_string();
        let entry_a = cache
            .store("collide", &CacheArtifacts::new(&image_a), &meta_a)
            .unwrap();
        assert_eq!(
            fs::read(entry_a.path.join("bzImage")).unwrap(),
            b"version-a"
        );

        // Second install with the same key — exercises the rename-
        // to-staging branch. Different built_at so we can tell
        // which metadata won.
        let src_b = TempDir::new().unwrap();
        let image_b = create_fake_image(src_b.path());
        fs::write(&image_b, b"version-b").unwrap();
        let mut meta_b = test_metadata("6.14.2");
        meta_b.built_at = "2026-04-18T00:00:00Z".to_string();
        let entry_b = cache
            .store("collide", &CacheArtifacts::new(&image_b), &meta_b)
            .unwrap();

        // New content wins.
        assert_eq!(
            fs::read(entry_b.path.join("bzImage")).unwrap(),
            b"version-b",
            "new content must replace old content atomically"
        );
        let installed_meta = read_metadata(&entry_b.path).expect("metadata.json");
        assert_eq!(installed_meta.built_at, "2026-04-18T00:00:00Z");

        // No staging or tmp residue.
        for dirent in fs::read_dir(&cache_root).unwrap() {
            let name = dirent.unwrap().file_name().to_string_lossy().into_owned();
            assert!(
                !name.starts_with(".evict-") && !name.starts_with(".tmp-"),
                "unexpected leftover directory under cache_root: {name}"
            );
        }
    }

    /// If a stale `.evict-*` dir from a crashed prior store survived
    /// on disk, the next store of the same key must still succeed
    /// (the fresh rename reuses the staging name).
    #[test]
    fn cache_dir_store_cleans_stale_evict_dir() {
        let tmp = TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let cache = CacheDir::with_root(cache_root.clone());

        // Prime the cache so final_dir exists.
        let src_a = TempDir::new().unwrap();
        let image_a = create_fake_image(src_a.path());
        cache
            .store(
                "repeat",
                &CacheArtifacts::new(&image_a),
                &test_metadata("6.14.2"),
            )
            .unwrap();

        // Plant a stale staging dir from a simulated prior crash.
        // Match the exact staging path the current thread would
        // compute, since staging names are now pid+thread-scoped.
        let stale_evict = cache_root.join(format!(
            ".evict-repeat-{}-{:?}",
            std::process::id(),
            std::thread::current().id(),
        ));
        fs::create_dir_all(&stale_evict).unwrap();
        fs::write(stale_evict.join("leftover"), b"x").unwrap();

        // Second store must handle the stale evict dir and succeed.
        let src_b = TempDir::new().unwrap();
        let image_b = create_fake_image(src_b.path());
        fs::write(&image_b, b"fresh").unwrap();
        let entry = cache
            .store(
                "repeat",
                &CacheArtifacts::new(&image_b),
                &test_metadata("6.14.2"),
            )
            .unwrap();
        assert_eq!(fs::read(entry.path.join("bzImage")).unwrap(), b"fresh");
        assert!(!stale_evict.exists(), "stale evict dir must be cleaned");
    }

    #[test]
    fn cache_dir_store_cleans_stale_tmp() {
        let tmp = TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let cache = CacheDir::with_root(cache_root.clone());

        // Create a stale .tmp- directory simulating a prior crash.
        let stale_tmp = cache_root.join(format!(".tmp-mykey-{}", std::process::id()));
        fs::create_dir_all(&stale_tmp).unwrap();
        fs::write(stale_tmp.join("junk"), b"leftover").unwrap();

        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");

        // Store should succeed despite stale tmp dir.
        let entry = cache
            .store("mykey", &CacheArtifacts::new(&image), &meta)
            .unwrap();
        assert!(entry.path.join("bzImage").exists());
        // Stale tmp dir should be gone.
        assert!(!stale_tmp.exists());
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
        // Metadata records has_vmlinux.
        assert!(entry.metadata.has_vmlinux);
        // Original files still exist (copy, not move).
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
        // Metadata records absence of vmlinux.
        assert!(!entry.metadata.has_vmlinux);
    }

    #[test]
    fn cache_dir_store_strips_vmlinux_internally() {
        // Real ELF fixture: store() must run strip_vmlinux_debug and
        // the stored vmlinux must reflect the strip (smaller than
        // source, no .debug_* sections).
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let vmlinux = create_strip_test_fixture(src_dir.path());
        let source_size = fs::metadata(&vmlinux).unwrap().len();
        let meta = test_metadata("6.14.2");

        let entry = cache
            .store(
                "strip-in-store",
                &CacheArtifacts::new(&image).with_vmlinux(&vmlinux),
                &meta,
            )
            .unwrap();
        let cached_vmlinux = entry.path.join("vmlinux");
        let cached_size = fs::metadata(&cached_vmlinux).unwrap().len();
        assert!(
            cached_size < source_size,
            "stored vmlinux ({cached_size} bytes) should be smaller \
             than source ({source_size}) after internal strip"
        );
        let data = fs::read(&cached_vmlinux).unwrap();
        let elf = goblin::elf::Elf::parse(&data).unwrap();
        let section_names: Vec<&str> = elf
            .section_headers
            .iter()
            .filter_map(|s| elf.shdr_strtab.get_at(s.sh_name))
            .collect();
        assert!(
            !section_names.contains(&".debug_info"),
            "internal strip should have removed .debug_info"
        );
        assert!(entry.metadata.has_vmlinux);
    }

    #[test]
    fn cache_dir_store_falls_back_when_strip_fails() {
        // Unparseable vmlinux: strip errors, store() falls back to
        // copying the raw bytes. has_vmlinux stays true so consumers
        // still see it.
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
    }

    #[test]
    fn cache_dir_store_preserves_original_vmlinux() {
        // strip_vmlinux_debug reads the source path; verify the
        // source file is still there after store() (no move, no
        // truncate).
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let vmlinux = create_strip_test_fixture(src_dir.path());
        let source_size = fs::metadata(&vmlinux).unwrap().len();
        let meta = test_metadata("6.14.2");

        cache
            .store(
                "preserve-src",
                &CacheArtifacts::new(&image).with_vmlinux(&vmlinux),
                &meta,
            )
            .unwrap();
        assert!(vmlinux.exists(), "source vmlinux must survive store()");
        assert_eq!(
            fs::metadata(&vmlinux).unwrap().len(),
            source_size,
            "source vmlinux size must not change"
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

        // Original image must still exist (copy, not move).
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
    fn cache_entry_vmlinux_path_some_when_stored() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let vmlinux = create_strip_test_fixture(src_dir.path());
        let entry = cache
            .store(
                "with-vml",
                &CacheArtifacts::new(&image).with_vmlinux(&vmlinux),
                &test_metadata("6.14.2"),
            )
            .unwrap();
        let vml_path = entry.vmlinux_path().expect("vmlinux_path() should be Some");
        assert_eq!(vml_path, entry.path.join("vmlinux"));
        assert!(vml_path.exists());
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

    // -- prefer_source_tree_for_dwarf --

    #[test]
    fn prefer_source_tree_local_with_vmlinux() {
        // Local-source cache entry whose source tree is still on disk
        // and has a vmlinux: helper returns the source tree path.
        let tmp = TempDir::new().unwrap();
        let cache_entry = tmp.path().join("cache");
        let src_tree = tmp.path().join("src");
        fs::create_dir_all(&cache_entry).unwrap();
        fs::create_dir_all(&src_tree).unwrap();
        fs::write(src_tree.join("vmlinux"), b"fake-elf").unwrap();

        let meta = KernelMetadata {
            version: Some("6.14.2".to_string()),
            source: KernelSource::Local {
                source_tree_path: Some(src_tree.clone()),
            },
            arch: "x86_64".to_string(),
            image_name: "bzImage".to_string(),
            config_hash: None,
            built_at: "2026-04-18T10:00:00Z".to_string(),
            ktstr_kconfig_hash: None,
            has_vmlinux: true,
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
        // Local-source cache entry but source tree lacks vmlinux:
        // fall back to None so caller keeps the cache-entry path.
        let tmp = TempDir::new().unwrap();
        let cache_entry = tmp.path().join("cache");
        let src_tree = tmp.path().join("src");
        fs::create_dir_all(&cache_entry).unwrap();
        fs::create_dir_all(&src_tree).unwrap();
        // No vmlinux in src_tree.

        let meta = KernelMetadata {
            version: None,
            source: KernelSource::Local {
                source_tree_path: Some(src_tree),
            },
            arch: "x86_64".to_string(),
            image_name: "bzImage".to_string(),
            config_hash: None,
            built_at: "2026-04-18T10:00:00Z".to_string(),
            ktstr_kconfig_hash: None,
            has_vmlinux: false,
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
        // Tarball source entry has no source_tree_path — return None
        // so caller uses the cache-entry directory (symbol lookup only,
        // no file:line).
        let tmp = TempDir::new().unwrap();
        let cache_entry = tmp.path().join("cache");
        fs::create_dir_all(&cache_entry).unwrap();

        let meta = KernelMetadata {
            version: Some("6.14.2".to_string()),
            source: KernelSource::Tarball,
            arch: "x86_64".to_string(),
            image_name: "bzImage".to_string(),
            config_hash: None,
            built_at: "2026-04-18T10:00:00Z".to_string(),
            ktstr_kconfig_hash: None,
            has_vmlinux: true,
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
        // Directory without metadata.json (e.g. a build-tree root, not
        // a cache entry): return None, caller keeps its existing path.
        let tmp = TempDir::new().unwrap();
        assert_eq!(prefer_source_tree_for_dwarf(tmp.path()), None);
    }

    // -- strip_vmlinux_debug --

    /// Check whether `elf` has a defined symbol with the given name.
    /// Mirrors the `sym_addr` closure inside `KernelSymbols::from_vmlinux`
    /// by requiring `st_value != 0` to reject undefined/absent symbols.
    fn has_symbol(elf: &goblin::elf::Elf, name: &str) -> bool {
        elf.syms
            .iter()
            .any(|s| s.st_value != 0 && elf.strtab.get_at(s.st_name) == Some(name))
    }

    /// Build a minimal ELF covering every strip dispatch branch:
    /// `.text` (code, bytes dropped via SHT_NOBITS), `.BTF` and
    /// `.rodata` (kept whole via the keep-list predicate), `.bss`
    /// (keep-list, already SHT_NOBITS), `.BTF.ext` + `.debug_*`
    /// (deleted), and the zero-data sections (`.data`,
    /// `.data..percpu`, `.init.data`; bytes dropped via SHT_NOBITS).
    /// Each bytes-dropped section has a symbol pointing at an
    /// in-bounds offset so tests can assert the symbols survive
    /// `Builder::delete_orphans`.
    fn create_strip_test_fixture(dir: &Path) -> PathBuf {
        use object::write;
        let mut obj = write::Object::new(
            object::BinaryFormat::Elf,
            object::Architecture::X86_64,
            object::Endianness::Little,
        );
        // .text — loadable code (not in keep-list, bytes dropped by keep-list path).
        let text_id = obj.add_section(Vec::new(), b".text".to_vec(), object::SectionKind::Text);
        obj.append_section_data(text_id, &[0xCC; 64], 1);
        // Symbol so .symtab and .strtab are generated.
        let _ = obj.add_symbol(write::Symbol {
            name: b"test_text_symbol".to_vec(),
            value: 0x10,
            size: 8,
            kind: object::SymbolKind::Data,
            scope: object::SymbolScope::Compilation,
            weak: false,
            section: write::SymbolSection::Section(text_id),
            flags: object::SymbolFlags::None,
        });
        // .BTF — kept by both keep-list and fallback.
        let btf_id = obj.add_section(Vec::new(), b".BTF".to_vec(), object::SectionKind::Metadata);
        obj.append_section_data(btf_id, &[0xEB; 256], 1);
        // .rodata — kept by keep-list (IKCONFIG gzip blob at runtime).
        // Bytes are preserved verbatim so read_hz_from_ikconfig can scan
        // for the IKCFG_ST marker; fixture stores an opaque payload.
        let rodata_id = obj.add_section(
            Vec::new(),
            b".rodata".to_vec(),
            object::SectionKind::ReadOnlyData,
        );
        obj.append_section_data(rodata_id, &[0xCA; 512], 1);
        // .bss — kept by keep-list; already SHT_NOBITS on any real
        // kernel build. object::write emits it without backing bytes
        // when `kind = UninitializedData`, matching real-vmlinux layout.
        let bss_id = obj.add_section(
            Vec::new(),
            b".bss".to_vec(),
            object::SectionKind::UninitializedData,
        );
        obj.append_section_bss(bss_id, 256, 8);
        let _ = obj.add_symbol(write::Symbol {
            name: b"test_bss_symbol".to_vec(),
            value: 0x50,
            size: 8,
            kind: object::SymbolKind::Data,
            scope: object::SymbolScope::Compilation,
            weak: false,
            section: write::SymbolSection::Section(bss_id),
            flags: object::SymbolFlags::None,
        });
        // .BTF.ext — deleted by keep-list (no consumer).
        let btf_ext_id = obj.add_section(
            Vec::new(),
            b".BTF.ext".to_vec(),
            object::SectionKind::Metadata,
        );
        obj.append_section_data(btf_ext_id, &[0xE1; 128], 1);
        // .debug_info — always stripped.
        let debug_id = obj.add_section(
            Vec::new(),
            b".debug_info".to_vec(),
            object::SectionKind::Debug,
        );
        obj.append_section_data(debug_id, &[0xAA; 4096], 1);
        // .debug_str — always stripped.
        let debug_str_id = obj.add_section(
            Vec::new(),
            b".debug_str".to_vec(),
            object::SectionKind::Debug,
        );
        obj.append_section_data(debug_str_id, &[0xBB; 2048], 1);
        // .data — bytes dropped via SHT_NOBITS; symbol must survive.
        let data_id = obj.add_section(Vec::new(), b".data".to_vec(), object::SectionKind::Data);
        obj.append_section_data(data_id, &[0xDD; 512], 8);
        let _ = obj.add_symbol(write::Symbol {
            name: b"test_data_symbol".to_vec(),
            value: 0x20,
            size: 8,
            kind: object::SymbolKind::Data,
            scope: object::SymbolScope::Compilation,
            weak: false,
            section: write::SymbolSection::Section(data_id),
            flags: object::SymbolFlags::None,
        });
        // .data..percpu — bytes dropped via SHT_NOBITS; symbol must survive.
        let percpu_id = obj.add_section(
            Vec::new(),
            b".data..percpu".to_vec(),
            object::SectionKind::Data,
        );
        obj.append_section_data(percpu_id, &[0xCC; 256], 8);
        let _ = obj.add_symbol(write::Symbol {
            name: b"test_percpu_symbol".to_vec(),
            value: 0x30,
            size: 8,
            kind: object::SymbolKind::Data,
            scope: object::SymbolScope::Compilation,
            weak: false,
            section: write::SymbolSection::Section(percpu_id),
            flags: object::SymbolFlags::None,
        });
        // .init.data — bytes dropped via SHT_NOBITS; symbol must survive.
        let initdata_id = obj.add_section(
            Vec::new(),
            b".init.data".to_vec(),
            object::SectionKind::Data,
        );
        obj.append_section_data(initdata_id, &[0x11; 1024], 8);
        let _ = obj.add_symbol(write::Symbol {
            name: b"test_initdata_symbol".to_vec(),
            value: 0x40,
            size: 8,
            kind: object::SymbolKind::Data,
            scope: object::SymbolScope::Compilation,
            weak: false,
            section: write::SymbolSection::Section(initdata_id),
            flags: object::SymbolFlags::None,
        });

        let data = obj.write().unwrap();
        let path = dir.join("vmlinux");
        fs::write(&path, &data).unwrap();
        path
    }

    #[test]
    fn strip_vmlinux_debug_applies_keep_list() {
        let src = TempDir::new().unwrap();
        let vmlinux = create_strip_test_fixture(src.path());
        let original_size = fs::metadata(&vmlinux).unwrap().len();

        // Positive control: the fixture must actually carry the
        // sections this test asserts on. If object::write silently
        // renames or drops one, the post-strip absence assertions
        // would false-pass.
        let source_data = fs::read(&vmlinux).unwrap();
        let source_elf = goblin::elf::Elf::parse(&source_data).unwrap();
        let source_section_names: Vec<&str> = source_elf
            .section_headers
            .iter()
            .filter_map(|s| source_elf.shdr_strtab.get_at(s.sh_name))
            .collect();
        // Positive control covers every section the post-strip
        // assertions inspect — kept, dropped, or deleted. A future
        // fixture regression that silently omits any of these would
        // make the corresponding post-strip check vacuous.
        for name in [
            ".debug_info",
            ".debug_str",
            ".BTF.ext",
            ".BTF",
            ".rodata",
            ".bss",
            ".symtab",
            ".strtab",
        ] {
            assert!(
                source_section_names.contains(&name),
                "fixture missing expected section {name}"
            );
        }

        let stripped = strip_vmlinux_debug(&vmlinux).unwrap();
        let stripped_path = stripped.path();
        let stripped_size = fs::metadata(stripped_path).unwrap().len();

        assert!(
            stripped_size < original_size,
            "stripped ({stripped_size}) should be smaller than original ({original_size})"
        );

        let data = fs::read(stripped_path).unwrap();
        let elf = goblin::elf::Elf::parse(&data).unwrap();
        let section_names: Vec<&str> = elf
            .section_headers
            .iter()
            .filter_map(|s| elf.shdr_strtab.get_at(s.sh_name))
            .collect();
        // Debug sections removed.
        assert!(
            !section_names.contains(&".debug_info"),
            "should not contain .debug_info"
        );
        assert!(
            !section_names.contains(&".debug_str"),
            "should not contain .debug_str"
        );
        // .BTF.ext removed (no consumer).
        assert!(
            !section_names.contains(&".BTF.ext"),
            "should not contain .BTF.ext"
        );
        // Keep-list sections preserved (names from all three consumer
        // modules plus structural).
        for name in [".BTF", ".rodata", ".bss", ".symtab", ".strtab"] {
            assert!(section_names.contains(&name), "should preserve {name}");
        }
    }

    #[test]
    fn strip_vmlinux_debug_symtab_readable() {
        let src = TempDir::new().unwrap();
        let vmlinux = create_strip_test_fixture(src.path());

        let stripped = strip_vmlinux_debug(&vmlinux).unwrap();
        let stripped_path = stripped.path();
        let data = fs::read(stripped_path).unwrap();
        let elf = goblin::elf::Elf::parse(&data).unwrap();

        // Smoke check: stripping a synthetic ELF produces a readable
        // symbol table whose strtab still contains our test symbol
        // names. End-to-end symbol preservation on real vmlinuxes is
        // covered by the *_preserves_monitor_symbols tests below.
        assert!(
            has_symbol(&elf, "test_text_symbol"),
            "stripped ELF should contain test_text_symbol in symtab"
        );
        // test_bss_symbol anchors .bss against Builder::delete_orphans.
        // Queryable via has_symbol because its fixture st_value is
        // nonzero (an in-bounds offset within .bss).
        assert!(
            has_symbol(&elf, "test_bss_symbol"),
            "stripped ELF should contain test_bss_symbol in symtab"
        );
    }

    /// Data sections matched by [`is_zero_data_section`] and code
    /// sections must come out as SHT_NOBITS with sh_size == 0, and
    /// symbols pointing at them must survive. Runs on the synthetic
    /// fixture so it exercises the keep-list path in CI environments
    /// without a real vmlinux.
    #[test]
    fn strip_vmlinux_debug_zeros_data_sections() {
        let src = TempDir::new().unwrap();
        let vmlinux = create_strip_test_fixture(src.path());

        // Pre-strip positive control: every zero-data section must
        // start with a non-SHT_NOBITS type AND non-zero sh_size. If
        // the fixture ever emits them as empty or already-SHT_NOBITS,
        // the post-strip assertions below become tautological and
        // would pass even if the strip pipeline regressed.
        use goblin::elf::section_header::SHT_NOBITS;
        let source_data = fs::read(&vmlinux).unwrap();
        let source_elf = goblin::elf::Elf::parse(&source_data).unwrap();
        for name_bytes in crate::monitor::symbols::VMLINUX_ZERO_DATA_SECTIONS
            .iter()
            .chain(SPECULATIVE_ZERO_DATA_SECTIONS.iter())
        {
            let name = std::str::from_utf8(name_bytes).unwrap();
            let sh = source_elf
                .section_headers
                .iter()
                .find(|s| source_elf.shdr_strtab.get_at(s.sh_name) == Some(name))
                .unwrap_or_else(|| panic!("fixture missing expected {name}"));
            assert_ne!(
                sh.sh_type, SHT_NOBITS,
                "fixture {name} must start non-SHT_NOBITS so the strip is observable"
            );
            assert!(
                sh.sh_size > 0,
                "fixture {name} must start with nonzero sh_size"
            );
        }

        let stripped = strip_vmlinux_debug(&vmlinux).unwrap();
        let stripped_path = stripped.path();
        let data = fs::read(stripped_path).unwrap();
        let elf = goblin::elf::Elf::parse(&data).unwrap();

        let find_section = |name: &str| {
            elf.section_headers
                .iter()
                .find(|s| elf.shdr_strtab.get_at(s.sh_name) == Some(name))
                .unwrap_or_else(|| panic!("section {name} missing from stripped ELF"))
        };
        let assert_nobits_empty = |name: &str| {
            let sh = find_section(name);
            let sh_type = sh.sh_type;
            let sh_size = sh.sh_size;
            assert_eq!(
                sh_type, SHT_NOBITS,
                "section {name} should be SHT_NOBITS after strip, got sh_type={sh_type}",
            );
            assert_eq!(
                sh_size, 0,
                "section {name} should have sh_size == 0 after strip, got {sh_size}",
            );
        };

        // Iterate both the consumer-declared zero-data sections and
        // the speculative retention set so the test stays in sync
        // automatically when either source changes.
        for name_bytes in crate::monitor::symbols::VMLINUX_ZERO_DATA_SECTIONS
            .iter()
            .chain(SPECULATIVE_ZERO_DATA_SECTIONS.iter())
        {
            let name = std::str::from_utf8(name_bytes).unwrap();
            assert_nobits_empty(name);
        }

        // Code sections (`.text` in the fixture) receive the same
        // SHT_NOBITS treatment so function symbols survive
        // `Builder::delete_orphans`.
        assert_nobits_empty(".text");

        // Symbols pointing at the zeroed data sections must survive.
        // Fixture symbol values are nonzero (0x20/0x30/0x40, within
        // their section bounds), so has_symbol's st_value != 0 filter
        // matches them.
        assert!(
            has_symbol(&elf, "test_data_symbol"),
            "test_data_symbol dropped by strip"
        );
        assert!(
            has_symbol(&elf, "test_percpu_symbol"),
            "test_percpu_symbol dropped by strip"
        );
        assert!(
            has_symbol(&elf, "test_initdata_symbol"),
            "test_initdata_symbol dropped by strip"
        );
    }

    /// `strip_debug_prefix` is the fallback path `strip_vmlinux_debug`
    /// hits when the keep-list strip errors out. Exercise it
    /// directly on the synthetic fixture so the success path has
    /// coverage independent of the keep-list branch.
    #[test]
    fn strip_debug_prefix_removes_debug_and_preserves_rest() {
        let src = TempDir::new().unwrap();
        let vmlinux = create_strip_test_fixture(src.path());
        let raw = fs::read(&vmlinux).unwrap();
        let processed = neutralize_alloc_relocs(&raw).unwrap();

        let stripped = strip_debug_prefix(&processed).unwrap();
        let elf = goblin::elf::Elf::parse(&stripped).unwrap();
        let names: Vec<&str> = elf
            .section_headers
            .iter()
            .filter_map(|s| elf.shdr_strtab.get_at(s.sh_name))
            .collect();

        // .debug_* sections deleted. (.comment is also in the fallback's
        // delete set but this fixture does not emit one, so it is not
        // exercised here.)
        assert!(
            !names.contains(&".debug_info"),
            "fallback should remove .debug_info"
        );
        assert!(
            !names.contains(&".debug_str"),
            "fallback should remove .debug_str"
        );
        // Every other section the fixture carries survives — unlike
        // the keep-list path, the fallback does not partition by
        // consumer. In particular `.BTF.ext` (which keep-list would
        // delete) remains.
        for name in [".BTF", ".BTF.ext", ".text", ".data", ".rodata", ".symtab"] {
            assert!(
                names.contains(&name),
                "fallback must preserve {name}, got sections {names:?}"
            );
        }
    }

    #[test]
    fn strip_vmlinux_debug_nonexistent_file() {
        let result = strip_vmlinux_debug(Path::new("/nonexistent/vmlinux"));
        assert!(result.is_err());
    }

    #[test]
    fn strip_vmlinux_debug_non_elf_file() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("vmlinux");
        fs::write(&path, b"not an ELF file").unwrap();
        let result = strip_vmlinux_debug(&path);
        assert!(result.is_err());
    }

    #[test]
    fn strip_vmlinux_debug_preserves_monitor_symbols() {
        let Some(path) = crate::monitor::find_test_vmlinux() else {
            skip!("no vmlinux found; set KTSTR_KERNEL or place vmlinux in ./linux");
        };
        // find_test_vmlinux may return /sys/kernel/btf/vmlinux (raw BTF,
        // not an ELF), which strip_vmlinux_debug cannot parse.
        if path.starts_with("/sys/") {
            skip!("vmlinux is raw BTF (not ELF), cannot strip debug");
        }
        let stripped = strip_vmlinux_debug(&path).unwrap();
        let stripped_path = stripped.path();
        let syms = crate::monitor::symbols::KernelSymbols::from_vmlinux(stripped_path).unwrap();
        // `runqueues` and `per_cpu_offset` are required non-Option
        // fields on KernelSymbols; `from_vmlinux` bails via
        // `Context::context` if either symbol is absent or zero
        // (`sym_addr` filters `st_value != 0`). Reaching the unwrap
        // above therefore guarantees both are nonzero. These asserts
        // are defensive against a future regression that loosens the
        // sym_addr filter or adds a non-error-on-missing path.
        assert_ne!(
            syms.runqueues, 0,
            "runqueues symbol missing from stripped vmlinux"
        );
        assert_ne!(
            syms.per_cpu_offset, 0,
            "__per_cpu_offset symbol missing from stripped vmlinux"
        );
        // For every optional symbol KernelSymbols tracks: presence must
        // survive the strip. A symbol that is absent from the source
        // vmlinux stays absent (kernel-config-dependent); a symbol that
        // is present must still be present.
        let source_syms = crate::monitor::symbols::KernelSymbols::from_vmlinux(&path).unwrap();
        assert_eq!(
            source_syms.init_top_pgt.is_some(),
            syms.init_top_pgt.is_some(),
            "strip changed KernelSymbols init_top_pgt presence"
        );
        assert_eq!(
            source_syms.page_offset_base_kva.is_some(),
            syms.page_offset_base_kva.is_some(),
            "strip changed page_offset_base_kva presence"
        );
        assert_eq!(
            source_syms.scx_root.is_some(),
            syms.scx_root.is_some(),
            "strip changed scx_root presence"
        );
        assert_eq!(
            source_syms.pgtable_l5_enabled.is_some(),
            syms.pgtable_l5_enabled.is_some(),
            "strip changed pgtable_l5_enabled presence"
        );
        assert_eq!(
            source_syms.prog_idr.is_some(),
            syms.prog_idr.is_some(),
            "strip changed prog_idr presence"
        );
        assert_eq!(
            source_syms.scx_watchdog_timeout.is_some(),
            syms.scx_watchdog_timeout.is_some(),
            "strip changed scx_watchdog_timeout presence"
        );

        // KernelSymbols.init_top_pgt collapses init_top_pgt OR
        // swapper_pg_dir via or_else. Check both names directly against
        // the raw symbol table so a regression that keeps one while
        // dropping the other is caught.
        let source_data = fs::read(&path).unwrap();
        let source_elf = goblin::elf::Elf::parse(&source_data).unwrap();
        let stripped_data = fs::read(stripped_path).unwrap();
        let stripped_elf = goblin::elf::Elf::parse(&stripped_data).unwrap();
        assert_eq!(
            has_symbol(&source_elf, "init_top_pgt"),
            has_symbol(&stripped_elf, "init_top_pgt"),
            "strip changed raw-symtab init_top_pgt presence"
        );
        assert_eq!(
            has_symbol(&source_elf, "swapper_pg_dir"),
            has_symbol(&stripped_elf, "swapper_pg_dir"),
            "strip changed raw-symtab swapper_pg_dir presence"
        );
    }

    /// Guards against a regression where `strip_vmlinux_debug` returns
    /// `Ok` but produces output close to the source size — e.g. if
    /// `.debug_*` removal is silently skipped.
    ///
    /// Skipped when the source vmlinux carries no `.debug_info`,
    /// which is the signature of an already-stripped input: ktstr's
    /// own cache path caches pre-stripped vmlinuxes, and CI that
    /// points this test at a cache-produced vmlinux would see the
    /// DWARF sections already gone. Running strip over an
    /// already-stripped ELF produces output the same size as the
    /// input (the keep-list partition is idempotent once DWARF is
    /// gone), so the `<` inequality no longer observes the strip.
    /// Rebuild the source-tree vmlinux to exercise this test.
    #[test]
    fn strip_vmlinux_debug_shrinks_when_source_has_debug_info() {
        let Some(path) = crate::monitor::find_test_vmlinux() else {
            skip!("no vmlinux found; set KTSTR_KERNEL or place vmlinux in ./linux");
        };
        if path.starts_with("/sys/") {
            skip!("vmlinux is raw BTF (not ELF), cannot strip debug");
        }
        let source_data = fs::read(&path).unwrap();
        let source_elf = goblin::elf::Elf::parse(&source_data).unwrap();
        let source_has_debug = source_elf
            .section_headers
            .iter()
            .any(|sh| source_elf.shdr_strtab.get_at(sh.sh_name) == Some(".debug_info"));
        if !source_has_debug {
            skip!(
                "source vmlinux has no .debug_info — already stripped \
                 (cached copy or distro-stripped); rebuild source tree \
                 to exercise the size-shrink path"
            );
        }

        let stripped = strip_vmlinux_debug(&path).unwrap();
        let source_size = fs::metadata(&path).unwrap().len();
        let stripped_size = fs::metadata(stripped.path()).unwrap().len();
        assert!(
            stripped_size < source_size,
            "stripped vmlinux ({stripped_size} bytes) should be smaller than \
             source ({source_size} bytes)"
        );
    }

    #[test]
    fn strip_vmlinux_debug_preserves_bpf_idr_symbols() {
        let Some(path) = crate::monitor::find_test_vmlinux() else {
            skip!("no vmlinux found; set KTSTR_KERNEL or place vmlinux in ./linux");
        };
        if path.starts_with("/sys/") {
            skip!("vmlinux is raw BTF (not ELF), cannot strip debug");
        }
        let stripped = strip_vmlinux_debug(&path).unwrap();
        let stripped_path = stripped.path();
        let data = fs::read(stripped_path).unwrap();
        let elf = goblin::elf::Elf::parse(&data).unwrap();
        assert!(
            has_symbol(&elf, "map_idr"),
            "map_idr symbol missing from stripped vmlinux"
        );
        assert!(
            has_symbol(&elf, "prog_idr"),
            "prog_idr symbol missing from stripped vmlinux"
        );
    }

    /// Function symbols (in `.text` and friends) must survive the
    /// strip so `resolve_addrs_from_elf` can resolve event addresses
    /// from the cached vmlinux. The strip preserves code-section
    /// headers as `SHT_NOBITS` to keep these symbols from being
    /// dropped by `Builder::delete_orphans`.
    #[test]
    fn strip_vmlinux_debug_preserves_function_symbols() {
        let Some(path) = crate::monitor::find_test_vmlinux() else {
            skip!("no vmlinux found; set KTSTR_KERNEL or place vmlinux in ./linux");
        };
        if path.starts_with("/sys/") {
            skip!("vmlinux is raw BTF (not ELF), cannot strip debug");
        }
        // Skip if the source vmlinux has no `schedule` symbol -- that
        // means it was already stripped by an older build of ktstr
        // and no longer carries .text symbols. The test exercises
        // strip-preserves behavior, not whether a particular cache
        // entry was rebuilt.
        let source_data = fs::read(&path).unwrap();
        let source_elf = goblin::elf::Elf::parse(&source_data).unwrap();
        if !has_symbol(&source_elf, "schedule") {
            skip!(
                "source vmlinux has no `schedule` symbol \
                 (already stripped by older ktstr) -- rebuild the kernel \
                 cache to exercise this test"
            );
        }

        let stripped = strip_vmlinux_debug(&path).unwrap();
        let stripped_path = stripped.path();
        let data = fs::read(stripped_path).unwrap();
        let elf = goblin::elf::Elf::parse(&data).unwrap();
        assert!(
            has_symbol(&elf, "schedule"),
            "schedule function symbol dropped by strip"
        );
    }

    // -- EnvVarGuard for test isolation --

    /// RAII guard that sets/unsets an environment variable and restores
    /// the original value on drop. Not thread-safe -- tests using this
    /// must run serially (nextest runs each test in its own process).
    struct EnvVarGuard {
        key: String,
        original: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &str, value: &str) -> Self {
            let original = std::env::var(key).ok();
            // SAFETY: nextest runs each test in its own process, so
            // concurrent env var mutation cannot occur.
            unsafe { std::env::set_var(key, value) };
            EnvVarGuard {
                key: key.to_string(),
                original,
            }
        }

        fn remove(key: &str) -> Self {
            let original = std::env::var(key).ok();
            // SAFETY: nextest runs each test in its own process.
            unsafe { std::env::remove_var(key) };
            EnvVarGuard {
                key: key.to_string(),
                original,
            }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.original {
                // SAFETY: nextest runs each test in its own process.
                Some(val) => unsafe { std::env::set_var(&self.key, val) },
                None => unsafe { std::env::remove_var(&self.key) },
            }
        }
    }
}
