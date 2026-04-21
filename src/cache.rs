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
/// payload (git details for `Git`, source-tree path and git hash for
/// `Local`).
///
/// Serialized as `{"type": "tarball"}`, `{"type": "git", "git_hash": ..., "ref": ...}`,
/// or `{"type": "local", "source_tree_path": ..., "git_hash": ...}`.
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
        git_hash: Option<String>,
        /// Git ref used for checkout (branch, tag, or ref spec).
        #[serde(default, rename = "ref", skip_serializing_if = "Option::is_none")]
        git_ref: Option<String>,
    },
    /// Build of a local on-disk kernel source tree.
    Local {
        /// Path to the source tree on disk. `None` when the tree has
        /// been sanitized for remote cache transport or is otherwise
        /// unavailable.
        source_tree_path: Option<PathBuf>,
        /// Git commit hash of the source tree at build time (short
        /// form). `None` when the tree is not a git repository or
        /// the hash could not be read.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        git_hash: Option<String>,
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
    pub version: Option<String>,
    /// How the kernel source was acquired, with per-source payload.
    pub source: KernelSource,
    /// Target architecture (e.g. "x86_64", "aarch64").
    pub arch: String,
    /// Boot image filename (e.g. "bzImage", "Image").
    pub image_name: String,
    /// CRC32 of the final .config used for the build.
    pub config_hash: Option<String>,
    /// ISO 8601 timestamp of when the image was built.
    pub built_at: String,
    /// CRC32 of ktstr.kconfig at build time.
    pub ktstr_kconfig_hash: Option<String>,
    /// Whether a stripped vmlinux ELF was cached alongside the image.
    /// When true, the entry directory contains a `vmlinux` file; see
    /// [`strip_vmlinux_debug`] for the strip policy.
    ///
    /// Required in metadata.json — plain `bool` without
    /// `#[serde(default)]` must be present during deserialization, so
    /// entries that predate this field surface as Corrupt rather than
    /// defaulting to `false`.
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
}

/// Entry yielded by [`CacheDir::list`]. Distinguishes valid entries
/// (with parsed metadata AND a present image file) from corrupt ones
/// (unreadable metadata, unparseable metadata, or metadata that
/// references an image file that no longer exists) so callers don't
/// have to re-check `Option` or re-stat the image path.
#[derive(Debug)]
#[non_exhaustive]
pub enum ListedEntry {
    /// Valid cache entry with parsed metadata and an image file
    /// present on disk at the metadata-declared path.
    Valid(CacheEntry),
    /// Entry directory exists but is unusable. Common reasons:
    /// metadata.json missing, metadata.json unparseable, or metadata
    /// parsed cleanly but the declared image file is absent (partial
    /// download, manual deletion, failed strip+rename).
    Corrupt {
        /// Cache key (directory name).
        key: String,
        /// Path to the (corrupt) entry directory.
        path: PathBuf,
        /// Human-readable explanation of why the entry is classified
        /// as corrupt. Rendered by CLI consumers for the user.
        reason: String,
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
    /// first). Entries with missing metadata, unparseable metadata,
    /// or a missing image file surface as [`ListedEntry::Corrupt`] at
    /// the end of the Vec. Valid entries are guaranteed to have an
    /// image file present — callers can call
    /// [`CacheEntry::image_path`] without re-stat'ing.
    /// (Concurrent cache mutation can invalidate this — callers in
    /// multi-process contexts should handle ENOENT gracefully.)
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
                Some(metadata) => {
                    let image_path = path.join(&metadata.image_name);
                    if image_path.exists() {
                        entries.push(ListedEntry::Valid(CacheEntry {
                            key: name,
                            path,
                            metadata,
                        }));
                    } else {
                        // Metadata parsed but the image file declared
                        // inside it is gone — treat as corrupt so
                        // callers don't dispatch to image_path() and
                        // get a path that 404s downstream.
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
                None => entries.push(ListedEntry::Corrupt {
                    key: name,
                    path,
                    reason: "metadata.json missing or unparseable".to_string(),
                }),
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

        // TmpGuard ensures tmp_dir is cleaned up on any error path
        // (including serde serialization failures) and on the
        // success path — where tmp_dir either no longer exists
        // (plain rename moved it to final_dir) or now holds the
        // displaced old cache content (atomic swap rotated it in).
        let _guard = TmpDirGuard(&tmp_dir);

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

        // Atomic install. Two cases:
        //
        // 1. final_dir does not yet exist → plain rename(tmp_dir,
        //    final_dir). The first such rename to win races gets to
        //    install; a concurrent store that also thought final_dir
        //    was absent retries via the swap branch when its rename
        //    trips ENOTEMPTY/EEXIST against the just-installed dir.
        //
        // 2. final_dir already exists → renameat2 with
        //    RENAME_EXCHANGE atomically swaps tmp_dir and final_dir
        //    in a single syscall. Readers calling lookup() at any
        //    instant see either the old entry or the new one,
        //    never a missing final_dir. After the swap, the old
        //    content lives in tmp_dir and is cleaned by the guard.
        //
        // RENAME_EXCHANGE is Linux-specific (kernel ≥ 3.15) and
        // requires that both paths exist at swap time. POSIX
        // rename(old, new) on a non-empty directory returns ENOTEMPTY
        // and would otherwise force a two-syscall dance with an
        // observable gap where final_dir does not exist — breaking
        // reader atomicity.
        match fs::rename(&tmp_dir, &final_dir) {
            Ok(()) => {}
            Err(e)
                if e.raw_os_error() == Some(libc::ENOTEMPTY)
                    || e.raw_os_error() == Some(libc::EEXIST) =>
            {
                atomic_swap_dirs(&tmp_dir, &final_dir)?;
                // After the swap, tmp_dir points at the old cache
                // entry content; drop it so the guard's
                // remove_dir_all cleans the right tree.
            }
            Err(e) => {
                return Err(anyhow::anyhow!("atomic rename cache entry: {e}"));
            }
        }

        // Guard drops at end of scope, cleaning tmp_dir in both
        // cases: a plain rename left it nonexistent (remove_dir_all
        // no-ops on ENOENT); an atomic swap placed the displaced
        // old entry there for removal.
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

/// RAII guard that removes a temporary directory on drop. Used by
/// [`CacheDir::store`] to clean up both serialization-failure
/// remnants and post-swap old cache content.
struct TmpDirGuard<'a>(&'a Path);

impl Drop for TmpDirGuard<'_> {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(self.0);
    }
}

/// Atomically swap two filesystem paths.
///
/// Wraps Linux `renameat2(AT_FDCWD, src, AT_FDCWD, dst,
/// RENAME_EXCHANGE)`. Both paths must already exist. On success the
/// inodes they name are exchanged in a single syscall — observers
/// traversing either path see the old target or the new one, never
/// a missing or partial entry.
///
/// Fails if the kernel does not support `RENAME_EXCHANGE`
/// (pre-3.15, `ENOSYS`), if either path is missing (`ENOENT`), or
/// if the two paths live on different filesystems (`EXDEV`). The
/// cache keeps both tmp_dir and final_dir under the same root, so
/// `EXDEV` would only fire on exotic bind mounts.
fn atomic_swap_dirs(src: &Path, dst: &Path) -> anyhow::Result<()> {
    use std::os::unix::ffi::OsStrExt;
    let src_c = std::ffi::CString::new(src.as_os_str().as_bytes())
        .map_err(|e| anyhow::anyhow!("src path contains NUL: {e}"))?;
    let dst_c = std::ffi::CString::new(dst.as_os_str().as_bytes())
        .map_err(|e| anyhow::anyhow!("dst path contains NUL: {e}"))?;
    // SAFETY: both pointers come from CString::as_ptr and outlive
    // the syscall. AT_FDCWD interprets the paths relative to the
    // current working directory of the calling thread, matching
    // what fs::rename does.
    let rc = unsafe {
        libc::syscall(
            libc::SYS_renameat2,
            libc::AT_FDCWD,
            src_c.as_ptr(),
            libc::AT_FDCWD,
            dst_c.as_ptr(),
            libc::RENAME_EXCHANGE,
        )
    };
    if rc == 0 {
        return Ok(());
    }
    let err = std::io::Error::last_os_error();
    Err(anyhow::anyhow!(
        "renameat2(RENAME_EXCHANGE) {} <-> {}: {}",
        src.display(),
        dst.display(),
        err,
    ))
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
    let KernelSource::Local {
        source_tree_path, ..
    } = metadata.source
    else {
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

    // Verify symtab survivors BEFORE writing. The `Builder` state
    // already holds every parsed symbol; counting named, non-deleted
    // entries here lets us fail fast without the post-write
    // `goblin::elf::Elf::parse(&out)` we used to run. The null symbol
    // (index 0) is filtered via the empty-name check, matching the
    // semantics of the old verification pass.
    let named_syms = builder
        .symbols
        .iter()
        .filter(|s| !s.delete && !s.name.as_slice().is_empty())
        .count();
    if named_syms == 0 {
        anyhow::bail!("keep-list strip emptied symbol table (0 named symbols)");
    }

    let mut out = Vec::new();
    builder
        .write(&mut out)
        .map_err(|e| anyhow::anyhow!("rewrite stripped vmlinux: {e}"))?;
    Ok(out)
}

/// Fallback strip: remove only .debug_* and .comment sections. Uses
/// the shared [`crate::elf_strip::rewrite`] primitive.
fn strip_debug_prefix(data: &[u8]) -> anyhow::Result<Vec<u8>> {
    crate::elf_strip::rewrite(data, |name| {
        name.starts_with(b".debug_") || name == b".comment"
    })
    .map_err(|e| anyhow::anyhow!("rewrite stripped vmlinux (fallback): {e}"))
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

    /// Build a minimal ELF object with a single `.text` section (64
    /// bytes of 0xCC) anchored by one symbol. The anchor symbol is
    /// what drives `object::write` to emit `.symtab`/`.strtab`, and
    /// every `neutralize_alloc_relocs` test shares this base shape.
    /// Callers that need SHF_ALLOC relocation sections add them on
    /// top of the returned object before calling `.write()`.
    fn build_base_elf_with_text_symbol() -> object::write::Object<'static> {
        use object::write;
        let mut obj = write::Object::new(
            object::BinaryFormat::Elf,
            object::Architecture::X86_64,
            object::Endianness::Little,
        );
        let text_id = obj.add_section(Vec::new(), b".text".to_vec(), object::SectionKind::Text);
        obj.append_section_data(text_id, &[0xCC; 64], 1);
        let _ = obj.add_symbol(write::Symbol {
            name: b"test_text_symbol".to_vec(),
            value: 0x0,
            size: 8,
            kind: object::SymbolKind::Data,
            scope: object::SymbolScope::Compilation,
            weak: false,
            section: write::SymbolSection::Section(text_id),
            flags: object::SymbolFlags::None,
        });
        obj
    }

    // -- keep-list source disjointness --

    /// Every entry in `is_keep_section` comes from one of four source
    /// lists owned by independent modules. Overlap is harmless at
    /// strip time but masks drift: if two modules add `.foo` and one
    /// later removes it, the strip still preserves `.foo` via the
    /// other module — the "dead" reference outlives its reader.
    ///
    /// This test locks the four lists as disjoint sets so a removal
    /// in one module immediately drops `.foo` from
    /// `is_keep_section`, and the downstream consumer's loss of its
    /// declared section becomes a visible test break rather than a
    /// silent ALL-tests-pass-but-data-is-missing runtime surprise.
    #[test]
    fn keep_section_sources_are_disjoint() {
        use std::collections::HashMap;
        let mut origins: HashMap<&[u8], Vec<&str>> = HashMap::new();
        let sources: &[(&str, &[&[u8]])] = &[
            ("cache::STRUCTURAL_KEEP_SECTIONS", STRUCTURAL_KEEP_SECTIONS),
            (
                "monitor::symbols::VMLINUX_KEEP_SECTIONS",
                crate::monitor::symbols::VMLINUX_KEEP_SECTIONS,
            ),
            (
                "monitor::VMLINUX_KEEP_SECTIONS",
                crate::monitor::VMLINUX_KEEP_SECTIONS,
            ),
            (
                "probe::btf::VMLINUX_KEEP_SECTIONS",
                crate::probe::btf::VMLINUX_KEEP_SECTIONS,
            ),
        ];
        for (label, list) in sources {
            for name in *list {
                origins.entry(*name).or_default().push(label);
            }
        }
        let dupes: Vec<_> = origins
            .iter()
            .filter(|(_, lists)| lists.len() > 1)
            .collect();
        assert!(
            dupes.is_empty(),
            "keep-list entries declared by multiple source modules (drift hazard): {dupes:?}",
        );
    }

    /// Same disjointness contract for the two zero-data lists.
    /// Retained sections here keep symbols but drop bytes — duplicate
    /// declarations would mask the same drift the keep-list test
    /// guards against.
    #[test]
    fn zero_data_section_sources_are_disjoint() {
        use std::collections::HashSet;
        let speculative: HashSet<&[u8]> = SPECULATIVE_ZERO_DATA_SECTIONS.iter().copied().collect();
        let declared: HashSet<&[u8]> = crate::monitor::symbols::VMLINUX_ZERO_DATA_SECTIONS
            .iter()
            .copied()
            .collect();
        let overlap: Vec<_> = speculative.intersection(&declared).collect();
        assert!(
            overlap.is_empty(),
            "zero-data section declared by both SPECULATIVE and a consumer (drift hazard): {overlap:?}",
        );
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
                git_hash: Some("a1b2c3d".to_string()),
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
                git_hash: Some(ref h),
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
                git_hash: Some("deadbee".to_string()),
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
                git_hash: Some(ref h),
            }
            if p == &PathBuf::from("/tmp/linux") && h == "deadbee"
        ));
        assert!(parsed.has_vmlinux);
    }

    /// git_hash on KernelSource::Local uses serde(default) and
    /// skip_serializing_if = Option::is_none. When the field is absent
    /// in the JSON input, deserialization must fill `None` rather than
    /// erroring; when `None` on the value being serialized, the key
    /// must not appear in the emitted JSON.
    #[test]
    fn kernel_source_local_git_hash_serde_round_trip_none() {
        let src = KernelSource::Local {
            source_tree_path: Some(PathBuf::from("/tmp/linux")),
            git_hash: None,
        };
        let json = serde_json::to_string(&src).unwrap();
        assert!(
            !json.contains("git_hash"),
            "git_hash=None must be skipped during serialization, got {json}"
        );
        let parsed: KernelSource = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, KernelSource::Local { git_hash: None, .. }));
    }

    #[test]
    fn kernel_source_serde_tagged_representation() {
        // Verify the tagged JSON shape on each variant.
        let t = serde_json::to_string(&KernelSource::Tarball).unwrap();
        assert_eq!(t, r#"{"type":"tarball"}"#);
        let g = serde_json::to_string(&KernelSource::Git {
            git_hash: Some("abc".to_string()),
            git_ref: Some("main".to_string()),
        })
        .unwrap();
        assert!(g.contains(r#""type":"git""#));
        assert!(g.contains(r#""git_hash":"abc""#));
        assert!(g.contains(r#""ref":"main""#));
        let l = serde_json::to_string(&KernelSource::Local {
            source_tree_path: Some(PathBuf::from("/tmp/linux")),
            git_hash: Some("a1b2c3d".to_string()),
        })
        .unwrap();
        assert!(l.contains(r#""type":"local""#));
        assert!(l.contains(r#""source_tree_path":"/tmp/linux""#));
        assert!(l.contains(r#""git_hash":"a1b2c3d""#));
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
        let ListedEntry::Corrupt { reason, .. } = corrupt else {
            panic!("expected Corrupt variant");
        };
        assert!(
            reason.contains("metadata.json missing or unparseable"),
            "missing-metadata reason should cite metadata.json, got: {reason}",
        );
    }

    #[test]
    fn cache_dir_list_classifies_missing_image_as_corrupt() {
        // Metadata parses cleanly but the image file it references
        // has been deleted (partial download / manual cleanup).
        // list() must surface the entry as ListedEntry::Corrupt with
        // an image-missing reason, so callers don't dispatch to
        // image_path() and get a stale path.
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");
        let entry = cache
            .store("missing-image", &CacheArtifacts::new(&image), &meta)
            .unwrap();

        // Delete only the image file; leave metadata.json in place.
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
    fn cache_dir_list_classifies_malformed_json_as_corrupt() {
        // metadata.json exists but is not valid JSON. `read_metadata`
        // returns None on `serde_json::from_str` failure, so list()
        // must surface the entry as Corrupt with the
        // unparseable-metadata reason.
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
            reason.contains("metadata.json missing or unparseable"),
            "malformed-JSON reason should match the unparseable-metadata label, got: {reason}",
        );
    }

    #[test]
    fn cache_dir_list_classifies_incomplete_metadata_as_corrupt() {
        // metadata.json is valid JSON but omits fields the current
        // `KernelMetadata` schema requires: `source`, `arch`,
        // `image_name`, `built_at`, and `has_vmlinux`. These are
        // non-`Option`, non-`#[serde(default)]` fields, so
        // `serde_json::from_str` fails when they are absent. Note
        // `has_vmlinux: bool` is required even though it is not
        // wrapped in `Option` — a plain `bool` with no
        // `#[serde(default)]` attribute must still be present in the
        // JSON payload. serde_json reports the first missing required
        // field in declaration order (`source`) but the test asserts
        // only the Corrupt classification and the generic
        // unparseable-metadata label, not the specific field name.
        // list() must surface the entry as Corrupt with the
        // unparseable-metadata reason — both incomplete metadata and
        // malformed JSON surface as Corrupt with the same reason
        // string because read_metadata returns None for either
        // failure mode.
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
            reason.contains("metadata.json missing or unparseable"),
            "incomplete-metadata reason should match the unparseable-metadata label, got: {reason}",
        );
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

    /// Concurrent readers calling `lookup()` while a writer is
    /// rapidly overwriting the same cache key must never observe a
    /// half-installed entry. The atomic rename-to-staging swap in
    /// `store()` should make every successful lookup return an entry
    /// whose `image_path()` exists and whose contents match one of
    /// the writer's complete versions — never a missing file, never a
    /// truncated image.
    ///
    /// Pinning this behavior catches regressions where the swap
    /// sequence is reordered (e.g. removing `final_dir` before
    /// renaming the tmp dir into place) or replaced with a non-atomic
    /// copy. Such regressions would let a reader observe a cache
    /// entry with valid metadata but a missing `bzImage`, or a
    /// partially-written image with bytes from two generations.
    #[test]
    fn cache_dir_store_atomic_under_concurrent_readers() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::thread;

        let tmp = TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let cache = Arc::new(CacheDir::with_root(cache_root.clone()));

        // Writer sources: two distinct full versions with
        // recognizable, long-ish content so a torn read would be
        // detectable by byte comparison, not just length.
        let src_a = TempDir::new().unwrap();
        let image_a = src_a.path().join("bzImage");
        let content_a = b"AAAAAAAA-image-version-a-AAAAAAAA".repeat(64);
        fs::write(&image_a, &content_a).unwrap();

        let src_b = TempDir::new().unwrap();
        let image_b = src_b.path().join("bzImage");
        let content_b = b"BBBBBBBB-image-version-b-BBBBBBBB".repeat(64);
        fs::write(&image_b, &content_b).unwrap();

        // Prime the cache so lookup() has something to find from
        // iteration one onwards. Without priming, early readers would
        // legitimately see None until the writer lands the first
        // store — and we want to assert "never missing once present,"
        // which requires an initial present state.
        let meta_prime = test_metadata("6.14.2");
        cache
            .store("atomic-key", &CacheArtifacts::new(&image_a), &meta_prime)
            .unwrap();

        const WRITE_ITERATIONS: usize = 40;
        let stop = Arc::new(AtomicBool::new(false));
        let lookups_observed = Arc::new(AtomicUsize::new(0));
        let atomicity_violations = Arc::new(AtomicUsize::new(0));

        // Spawn reader threads. Each reader loops until `stop` is
        // set, calling lookup() and verifying that when Some(entry)
        // comes back, the image file exists and matches one of the
        // two known writer contents byte-for-byte.
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
                        // Once primed, lookup must always see an
                        // entry. A None here is a real atomicity
                        // violation: the writer briefly removed the
                        // final_dir without immediately replacing it.
                        violations.fetch_add(1, Ordering::Relaxed);
                        continue;
                    };
                    let image_path = entry.image_path();
                    let Ok(bytes) = fs::read(&image_path) else {
                        // Entry directory + metadata visible, but
                        // image file missing → non-atomic install.
                        violations.fetch_add(1, Ordering::Relaxed);
                        continue;
                    };
                    if bytes != expected_a && bytes != expected_b {
                        // Torn read: bytes don't match either
                        // complete version. Would indicate the image
                        // was observed mid-copy.
                        violations.fetch_add(1, Ordering::Relaxed);
                    }
                    lookups_observed.fetch_add(1, Ordering::Relaxed);
                }
            }));
        }

        // Writer: alternate between version A and version B,
        // exercising the rename-to-staging branch on every iteration
        // after the first (final_dir already exists from priming).
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

        // Post-condition: final state is intact and no staging or
        // tmp residue leaked out of the write loop.
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
                git_hash: None,
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
                git_hash: None,
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

        // .debug_* sections deleted. (.comment is exercised by the
        // dedicated `strip_debug_prefix_removes_dot_comment` test
        // against a fixture that emits one.)
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

    /// `strip_debug_prefix`'s delete filter matches on two distinct
    /// predicates: `name.starts_with(b".debug_")` and
    /// `name == b".comment"`. The `.debug_*` branch is exercised by
    /// `strip_debug_prefix_removes_debug_and_preserves_rest` against
    /// the shared fixture. This test covers the `.comment` branch
    /// against a focused fixture that specifically emits one — the
    /// shared fixture deliberately does not, to keep the keep-list
    /// assertions scoped.
    #[test]
    fn strip_debug_prefix_removes_dot_comment() {
        use object::write;
        // Minimal ELF: one loadable .text (fallback must preserve it)
        // plus a .comment section (fallback must delete it). A symbol
        // anchors .text so the `object` writer emits .symtab/.strtab.
        let mut obj = write::Object::new(
            object::BinaryFormat::Elf,
            object::Architecture::X86_64,
            object::Endianness::Little,
        );
        let text_id = obj.add_section(Vec::new(), b".text".to_vec(), object::SectionKind::Text);
        obj.append_section_data(text_id, &[0xCC; 64], 1);
        let _ = obj.add_symbol(write::Symbol {
            name: b"test_text_symbol".to_vec(),
            value: 0x0,
            size: 8,
            kind: object::SymbolKind::Data,
            scope: object::SymbolScope::Compilation,
            weak: false,
            section: write::SymbolSection::Section(text_id),
            flags: object::SymbolFlags::None,
        });
        // `.comment` is ELF's standard toolchain-producer string table
        // (`object::SectionKind::OtherString`).
        // Real kernel builds carry one stamped by GCC/Clang.
        let comment_id = obj.add_section(
            Vec::new(),
            b".comment".to_vec(),
            object::SectionKind::OtherString,
        );
        obj.append_section_data(comment_id, b"GCC: (GNU) 14.2.1 20250207\0", 1);
        let data = obj.write().unwrap();

        // Positive control: the fixture must actually carry `.comment`
        // and `.text` before strip. If `object::write` silently dropped
        // either (e.g. renaming, or treating OtherString non-standardly),
        // the post-strip absence assertion on `.comment` would
        // false-pass. Mirrors the positive-control pattern in
        // `strip_vmlinux_debug_applies_keep_list`.
        let source_elf = goblin::elf::Elf::parse(&data).unwrap();
        let source_names: Vec<&str> = source_elf
            .section_headers
            .iter()
            .filter_map(|s| source_elf.shdr_strtab.get_at(s.sh_name))
            .collect();
        for name in [".comment", ".text"] {
            assert!(
                source_names.contains(&name),
                "fixture missing expected section {name}; got {source_names:?}"
            );
        }

        // `neutralize_alloc_relocs` is a no-op on this fixture (no
        // SHF_ALLOC relocation sections) — run it anyway so the test
        // exercises the exact input pipeline `strip_vmlinux_debug` uses.
        let processed = neutralize_alloc_relocs(&data).unwrap();
        let stripped = strip_debug_prefix(&processed).unwrap();
        let elf = goblin::elf::Elf::parse(&stripped).unwrap();
        let names: Vec<&str> = elf
            .section_headers
            .iter()
            .filter_map(|s| elf.shdr_strtab.get_at(s.sh_name))
            .collect();

        assert!(
            !names.contains(&".comment"),
            "fallback must remove .comment, got sections {names:?}"
        );
        // Non-comment, non-debug sections survive untouched — guards
        // against an overly broad filter that accidentally drops
        // unrelated sections.
        assert!(
            names.contains(&".text"),
            "fallback must preserve .text, got sections {names:?}"
        );
    }

    /// `neutralize_alloc_relocs` rewrites the `sh_size` field to 0
    /// in every section header whose `sh_type` is `SHT_REL` or
    /// `SHT_RELA` AND whose `sh_flags` carries `SHF_ALLOC`. Pin the
    /// four observable invariants against a focused fixture:
    ///
    /// 1a. SHF_ALLOC + SHT_RELA section has sh_size zeroed post-call.
    /// 1b. SHF_ALLOC + SHT_REL section has sh_size zeroed post-call.
    /// 2. Non-ALLOC SHT_RELA section has sh_size preserved (guards
    ///    against over-matching — real kernel ELFs carry
    ///    non-ALLOC RELA sections like `.rela.debug_info` that must
    ///    survive untouched).
    /// 3. Non-RELA section (e.g. `.text`) has sh_size preserved
    ///    (guards against an accidentally-broader filter).
    ///
    /// Also pins content preservation: `neutralize_alloc_relocs`
    /// only mutates the section HEADER's sh_size, not the section's
    /// data bytes. Raw bytes at the original sh_offset must remain
    /// bit-identical post-call.
    #[test]
    fn neutralize_alloc_relocs_zeros_only_sh_size_of_alloc_reloc_sections() {
        use object::elf::{SHF_ALLOC, SHT_REL, SHT_RELA};

        // Base ELF with .text + anchor symbol (so object::write
        // emits .symtab/.strtab). Reloc sections are added below.
        let mut obj = build_base_elf_with_text_symbol();
        // .rela.kaslr — SHT_RELA + SHF_ALLOC. Shape matches what
        // CONFIG_RELOCATABLE + CONFIG_RANDOMIZE_BASE kernels emit
        // and motivates this pass per the fn docstring. sh_size
        // must be zeroed.
        let kaslr_id = obj.add_section(
            Vec::new(),
            b".rela.kaslr".to_vec(),
            object::SectionKind::Elf(SHT_RELA),
        );
        obj.append_section_data(kaslr_id, &[0xA5; 32], 1);
        obj.section_mut(kaslr_id).flags = object::SectionFlags::Elf {
            sh_flags: u64::from(SHF_ALLOC),
        };
        // .rel.foo — SHT_REL + SHF_ALLOC. The fn's match arm accepts
        // both SHT_REL and SHT_RELA; exercising only SHT_RELA would
        // let a regression that dropped SHT_REL ride unnoticed.
        let rel_id = obj.add_section(
            Vec::new(),
            b".rel.foo".to_vec(),
            object::SectionKind::Elf(SHT_REL),
        );
        obj.append_section_data(rel_id, &[0xC7; 24], 1);
        obj.section_mut(rel_id).flags = object::SectionFlags::Elf {
            sh_flags: u64::from(SHF_ALLOC),
        };
        // .rela.debug_info — SHT_RELA WITHOUT SHF_ALLOC. Negative
        // control: must be left untouched by neutralize_alloc_relocs.
        let rdbg_id = obj.add_section(
            Vec::new(),
            b".rela.debug_info".to_vec(),
            object::SectionKind::Elf(SHT_RELA),
        );
        obj.append_section_data(rdbg_id, &[0xB6; 16], 1);
        // flags left as SectionFlags::None — no SHF_ALLOC.

        let data = obj.write().unwrap();

        // Positive-control the fixture: the four sections we assert
        // on must actually exist in the produced ELF with the expected
        // sh_type/sh_flags/sh_size. If `object::write` renamed or
        // reshaped one, the post-call assertions would false-pass.
        let pre_elf = goblin::elf::Elf::parse(&data).unwrap();
        let mut pre_kaslr = None;
        let mut pre_rel = None;
        let mut pre_rdbg = None;
        let mut pre_text = None;
        for sh in pre_elf.section_headers.iter() {
            let name = pre_elf.shdr_strtab.get_at(sh.sh_name).unwrap_or("");
            match name {
                ".rela.kaslr" => pre_kaslr = Some(sh.clone()),
                ".rel.foo" => pre_rel = Some(sh.clone()),
                ".rela.debug_info" => pre_rdbg = Some(sh.clone()),
                ".text" => pre_text = Some(sh.clone()),
                _ => {}
            }
        }
        let pre_kaslr = pre_kaslr.expect("fixture must carry .rela.kaslr");
        let pre_rel = pre_rel.expect("fixture must carry .rel.foo");
        let pre_rdbg = pre_rdbg.expect("fixture must carry .rela.debug_info");
        let pre_text = pre_text.expect("fixture must carry .text");
        assert_eq!(
            pre_kaslr.sh_type, SHT_RELA,
            ".rela.kaslr sh_type must be SHT_RELA"
        );
        assert!(
            pre_kaslr.sh_flags & u64::from(SHF_ALLOC) != 0,
            ".rela.kaslr must carry SHF_ALLOC; got sh_flags={:#x}",
            pre_kaslr.sh_flags
        );
        assert_eq!(
            pre_kaslr.sh_size, 32,
            ".rela.kaslr sh_size must match 32-byte payload"
        );
        assert_eq!(pre_rel.sh_type, SHT_REL, ".rel.foo sh_type must be SHT_REL");
        assert!(
            pre_rel.sh_flags & u64::from(SHF_ALLOC) != 0,
            ".rel.foo must carry SHF_ALLOC; got sh_flags={:#x}",
            pre_rel.sh_flags
        );
        assert_eq!(
            pre_rel.sh_size, 24,
            ".rel.foo sh_size must match 24-byte payload"
        );
        assert_eq!(
            pre_rdbg.sh_type, SHT_RELA,
            ".rela.debug_info sh_type must be SHT_RELA"
        );
        assert_eq!(
            pre_rdbg.sh_flags & u64::from(SHF_ALLOC),
            0,
            ".rela.debug_info must NOT carry SHF_ALLOC; got sh_flags={:#x}",
            pre_rdbg.sh_flags
        );
        assert_eq!(
            pre_rdbg.sh_size, 16,
            ".rela.debug_info sh_size must match 16-byte payload"
        );
        assert_eq!(
            pre_text.sh_size, 64,
            ".text sh_size must match 64-byte payload"
        );

        // Snapshot the .rela.kaslr data bytes before the call so we
        // can assert they survive the sh_size rewrite.
        let kaslr_offset = pre_kaslr.sh_offset as usize;
        let kaslr_size = pre_kaslr.sh_size as usize;
        let kaslr_original_data = data[kaslr_offset..kaslr_offset + kaslr_size].to_vec();

        let processed = neutralize_alloc_relocs(&data).unwrap();
        assert_eq!(
            processed.len(),
            data.len(),
            "neutralize_alloc_relocs must not resize the ELF; only sh_size header fields are rewritten"
        );

        let post_elf = goblin::elf::Elf::parse(&processed).unwrap();
        let mut post_kaslr = None;
        let mut post_rel = None;
        let mut post_rdbg = None;
        let mut post_text = None;
        for sh in post_elf.section_headers.iter() {
            let name = post_elf.shdr_strtab.get_at(sh.sh_name).unwrap_or("");
            match name {
                ".rela.kaslr" => post_kaslr = Some(sh.clone()),
                ".rel.foo" => post_rel = Some(sh.clone()),
                ".rela.debug_info" => post_rdbg = Some(sh.clone()),
                ".text" => post_text = Some(sh.clone()),
                _ => {}
            }
        }
        let post_kaslr = post_kaslr.expect(".rela.kaslr must survive");
        let post_rel = post_rel.expect(".rel.foo must survive");
        let post_rdbg = post_rdbg.expect(".rela.debug_info must survive");
        let post_text = post_text.expect(".text must survive");

        // Invariant 1a: SHF_ALLOC + SHT_RELA section has sh_size zeroed.
        assert_eq!(
            post_kaslr.sh_size, 0,
            ".rela.kaslr sh_size must be zeroed; got {}",
            post_kaslr.sh_size
        );
        // Invariant 1b: SHF_ALLOC + SHT_REL section has sh_size zeroed
        // (the `|| sh.sh_type == SHT_REL` branch in the filter).
        assert_eq!(
            post_rel.sh_size, 0,
            ".rel.foo sh_size must be zeroed; got {}",
            post_rel.sh_size
        );
        // Invariant 2: Non-ALLOC SHT_RELA preserved.
        assert_eq!(
            post_rdbg.sh_size, pre_rdbg.sh_size,
            ".rela.debug_info sh_size must be preserved (no SHF_ALLOC)"
        );
        // Invariant 3: Non-RELA section preserved.
        assert_eq!(
            post_text.sh_size, pre_text.sh_size,
            ".text sh_size must be preserved (not a relocation section)"
        );

        // Content preservation: the raw bytes at the section's
        // sh_offset must be bit-identical to pre-call. Only the
        // sh_size header field was rewritten.
        assert_eq!(
            &processed[kaslr_offset..kaslr_offset + kaslr_size],
            &kaslr_original_data[..],
            ".rela.kaslr data bytes must be preserved; neutralize only rewrites sh_size"
        );

        // sh_offset, sh_type, sh_flags of the neutralized section
        // must also be preserved — the fn touches ONE field per
        // matching section header.
        assert_eq!(
            post_kaslr.sh_offset, pre_kaslr.sh_offset,
            "sh_offset must be preserved"
        );
        assert_eq!(
            post_kaslr.sh_type, pre_kaslr.sh_type,
            "sh_type must be preserved"
        );
        assert_eq!(
            post_kaslr.sh_flags, pre_kaslr.sh_flags,
            "sh_flags must be preserved"
        );
    }

    /// For ELFs that carry no SHF_ALLOC relocation sections,
    /// `neutralize_alloc_relocs` returns an unchanged copy —
    /// documented as the "no-op" branch in the fn docstring.
    #[test]
    fn neutralize_alloc_relocs_noop_when_no_alloc_reloc_sections() {
        // Base ELF carries only .text + anchor symbol — no reloc
        // sections at all, so the filter matches nothing.
        let data = build_base_elf_with_text_symbol().write().unwrap();

        let processed = neutralize_alloc_relocs(&data).unwrap();
        assert_eq!(
            processed, data,
            "neutralize_alloc_relocs must be a byte-identity no-op when no SHF_ALLOC reloc sections are present"
        );
    }

    /// `neutralize_alloc_relocs` must be byte-identity idempotent:
    /// `f(f(x)) == f(x)`. The production filter at cache.rs:937-960
    /// keys on `sh_type` and `sh_flags` — neither of which the
    /// function ever writes, only `sh_size` is rewritten. On the
    /// second pass every matching section header is re-matched (same
    /// sh_type and sh_flags) and `out[size_offset..size_end].fill(0)`
    /// rewrites already-zero bytes; every other byte is left untouched
    /// by `out = data.to_vec()`.
    ///
    /// Guards against a future "skip already-zero" optimization that
    /// drifts the filter predicate (e.g. stops matching headers whose
    /// `sh_size` is already 0), and against a future mutation that
    /// widens the rewrite to touch `sh_flags` or `sh_type` — which
    /// would break idempotence by changing what the second pass
    /// matches on.
    ///
    /// Uses the same 4-section fixture as
    /// `neutralize_alloc_relocs_zeros_only_sh_size_of_alloc_reloc_sections`
    /// so both positive (SHT_RELA+ALLOC, SHT_REL+ALLOC) and negative
    /// (SHT_RELA no-ALLOC, non-RELA .text) paths are re-walked on the
    /// second pass.
    #[test]
    fn neutralize_alloc_relocs_is_idempotent() {
        use object::elf::{SHF_ALLOC, SHT_REL, SHT_RELA};

        // Base .text + anchor symbol; the reloc sections added below
        // intentionally mirror the sibling zeros_only test's fixture
        // so both positive and negative filter paths re-walk on the
        // second pass.
        let mut obj = build_base_elf_with_text_symbol();
        // .rela.kaslr — SHT_RELA + SHF_ALLOC. Primary positive case.
        let kaslr_id = obj.add_section(
            Vec::new(),
            b".rela.kaslr".to_vec(),
            object::SectionKind::Elf(SHT_RELA),
        );
        obj.append_section_data(kaslr_id, &[0xA5; 32], 1);
        obj.section_mut(kaslr_id).flags = object::SectionFlags::Elf {
            sh_flags: u64::from(SHF_ALLOC),
        };
        // .rel.foo — SHT_REL + SHF_ALLOC. Exercises the second arm of
        // the `is_rela` predicate so a regression that special-cased
        // only SHT_RELA on re-entry would surface here.
        let rel_id = obj.add_section(
            Vec::new(),
            b".rel.foo".to_vec(),
            object::SectionKind::Elf(SHT_REL),
        );
        obj.append_section_data(rel_id, &[0xC7; 24], 1);
        obj.section_mut(rel_id).flags = object::SectionFlags::Elf {
            sh_flags: u64::from(SHF_ALLOC),
        };
        // .rela.debug_info — SHT_RELA without SHF_ALLOC. Negative
        // control on the second pass as well.
        let rdbg_id = obj.add_section(
            Vec::new(),
            b".rela.debug_info".to_vec(),
            object::SectionKind::Elf(SHT_RELA),
        );
        obj.append_section_data(rdbg_id, &[0xB6; 16], 1);
        // flags left as SectionFlags::None — no SHF_ALLOC.

        let data = obj.write().unwrap();

        let first_pass = neutralize_alloc_relocs(&data).unwrap();
        let second_pass = neutralize_alloc_relocs(&first_pass).unwrap();

        // Non-vacuous guard: the first call must actually modify bytes
        // on this fixture (which carries SHF_ALLOC reloc sections); a
        // degenerate no-op implementation of `neutralize_alloc_relocs`
        // would trivially satisfy idempotence and must not pass.
        assert_ne!(
            first_pass, data,
            "first call must modify bytes on a fixture with SHF_ALLOC reloc sections; \
             if this fails, neutralize_alloc_relocs is a no-op"
        );

        // Primary idempotence assertion: byte equality between passes.
        assert_eq!(
            second_pass, first_pass,
            "neutralize_alloc_relocs must be idempotent: a second pass over its own output produces byte-identical bytes"
        );

        // Length preservation across both passes — the function only
        // rewrites in-place `sh_size` fields, never resizes the buffer.
        assert_eq!(
            first_pass.len(),
            data.len(),
            "first pass must preserve ELF length"
        );
        assert_eq!(
            second_pass.len(),
            first_pass.len(),
            "second pass must preserve ELF length"
        );

        // Re-parse post-second-pass: the ELF header and section
        // header table must still be well-formed after two rewrites.
        let post_elf = goblin::elf::Elf::parse(&second_pass)
            .expect("second-pass output must remain parseable as ELF");

        let mut post_kaslr = None;
        let mut post_rel = None;
        let mut post_rdbg = None;
        for sh in post_elf.section_headers.iter() {
            let name = post_elf.shdr_strtab.get_at(sh.sh_name).unwrap_or("");
            match name {
                ".rela.kaslr" => post_kaslr = Some(sh.clone()),
                ".rel.foo" => post_rel = Some(sh.clone()),
                ".rela.debug_info" => post_rdbg = Some(sh.clone()),
                _ => {}
            }
        }
        let post_kaslr = post_kaslr.expect(".rela.kaslr must survive second pass");
        let post_rel = post_rel.expect(".rel.foo must survive second pass");
        let post_rdbg = post_rdbg.expect(".rela.debug_info must survive second pass");

        // SHF_ALLOC+reloc sections stay zeroed on the second pass.
        assert_eq!(
            post_kaslr.sh_size, 0,
            ".rela.kaslr sh_size must remain zero after the second pass"
        );
        assert_eq!(
            post_rel.sh_size, 0,
            ".rel.foo sh_size must remain zero after the second pass"
        );

        // SHF_ALLOC flag must still be set on the zeroed sections —
        // the function touches sh_size ONLY, never sh_flags.
        assert!(
            post_kaslr.sh_flags & u64::from(SHF_ALLOC) != 0,
            ".rela.kaslr SHF_ALLOC flag must survive both passes; got sh_flags={:#x}",
            post_kaslr.sh_flags
        );
        assert!(
            post_rel.sh_flags & u64::from(SHF_ALLOC) != 0,
            ".rel.foo SHF_ALLOC flag must survive both passes; got sh_flags={:#x}",
            post_rel.sh_flags
        );

        // Negative control survives both passes untouched: non-ALLOC
        // SHT_RELA section keeps its original sh_size.
        assert_eq!(
            post_rdbg.sh_size, 16,
            ".rela.debug_info sh_size must be preserved across both passes (no SHF_ALLOC)"
        );
    }

    /// `neutralize_alloc_relocs` fails loudly when fed bytes that do
    /// not parse as an ELF — the goblin parse returns Err and the
    /// function wraps it in an `anyhow::anyhow!("parse vmlinux ELF
    /// for preprocess: {e}")`. Pin only the stable "parse vmlinux ELF
    /// for preprocess" wrapper in `neutralize_alloc_relocs`; the
    /// goblin-side error text is version-dependent and not part of
    /// the contract.
    ///
    /// Exercises two distinct goblin failure paths through the same
    /// anyhow wrapper:
    ///
    /// 1. Bad magic: "not an ELF..." passes the 16-byte length gate but
    ///    its first four bytes do not match `\x7fELF`, so goblin's
    ///    `TryFromCtx` for `Header` bails with `Error::BadMagic` before
    ///    inspecting any later field (see goblin 0.10 `elf/header.rs`
    ///    `try_from_ctx` at the `ident[0..SELFMAG] != ELFMAG` branch).
    /// 2. Invalid EI_CLASS: a 16-byte prefix with the correct magic but
    ///    `ident[EI_CLASS] == 0` (ELFCLASSNONE) passes BOTH the length
    ///    and magic gates, and fails on the subsequent class-dispatch
    ///    match with `Error::Malformed("invalid ELF class 0")`. This is
    ///    the "passes magic, fails deeper" path.
    ///
    /// Either failure mode flows through the same `anyhow::anyhow!`
    /// wrapper, so the test pins the wrapper string for each input
    /// without pinning the goblin-side sub-error wording.
    #[test]
    fn neutralize_alloc_relocs_rejects_invalid_elf() {
        // Table-driven so a future goblin upgrade that changes either
        // sub-error's wording still surfaces both paths distinctly.
        let cases: &[(&str, &[u8])] = &[
            ("bad magic", b"not an ELF at all, just some bytes"),
            (
                "magic ok but invalid EI_CLASS",
                &[
                    0x7f, b'E', b'L', b'F', // magic
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // ident[4..16]: class=0, rest 0
                ],
            ),
        ];
        for (label, input) in cases {
            let err = neutralize_alloc_relocs(input).unwrap_err();
            let rendered = format!("{err:#}");
            assert!(
                rendered.contains("parse vmlinux ELF for preprocess"),
                "[{label}] expected error context to name the ELF parse step; got: {rendered}"
            );
        }
    }

    /// ELF32 counterpart of
    /// [`neutralize_alloc_relocs_zeros_only_sh_size_of_alloc_reloc_sections`].
    ///
    /// `neutralize_alloc_relocs` dispatches on `elf.is_64` at the
    /// `sh_size` offset/width pair — 32-byte offset + 8-byte field for
    /// ELF64, 20-byte offset + 4-byte field for ELF32 (per the
    /// ELF32/ELF64 section header layouts documented at the call site).
    /// The existing fixture-driven coverage is all ELF64 (Architecture::
    /// X86_64), so a regression that swapped the `if elf.is_64` branches
    /// or hardcoded the 64-bit offsets would silently corrupt 32-bit
    /// inputs without tripping any assertion.
    ///
    /// Uses `Architecture::I386` which `object` maps to
    /// `address_size == U32` and therefore emits ELFCLASS32 via the
    /// writer's is_64=false path. goblin then parses the output with
    /// `is_64 == false`, driving `neutralize_alloc_relocs` through the
    /// else `(20, 4)` branch.
    ///
    /// Exercises BOTH arms of the
    /// `sh.sh_type == SHT_RELA || sh.sh_type == SHT_REL` filter
    /// predicate on the ELF32 code path. A regression that special-
    /// cased SHT_REL only on ELF64 (e.g. wired it to a 64-bit offset
    /// table), or dropped SHT_REL from the ELF32 filter altogether,
    /// would leave one section un-zeroed here.
    ///
    /// Invariants pinned: SHF_ALLOC + SHT_RELA AND SHF_ALLOC + SHT_REL
    /// both have their sh_size zeroed, the output remains parseable
    /// as ELF32 (is_64 stays false), and the buffer length is
    /// preserved (the fn only rewrites an in-place header field,
    /// never resizes).
    #[test]
    fn neutralize_alloc_relocs_zeros_sh_size_in_elf32_fixture() {
        use object::elf::{SHF_ALLOC, SHT_REL, SHT_RELA};
        use object::write;

        let mut obj = write::Object::new(
            object::BinaryFormat::Elf,
            object::Architecture::I386,
            object::Endianness::Little,
        );
        // .text anchored by a symbol so object::write emits
        // .symtab/.strtab — mirrors the ELF64 fixture pattern.
        let text_id = obj.add_section(Vec::new(), b".text".to_vec(), object::SectionKind::Text);
        obj.append_section_data(text_id, &[0xCC; 64], 1);
        let _ = obj.add_symbol(write::Symbol {
            name: b"test_text_symbol".to_vec(),
            value: 0x0,
            size: 4,
            kind: object::SymbolKind::Data,
            scope: object::SymbolScope::Compilation,
            weak: false,
            section: write::SymbolSection::Section(text_id),
            flags: object::SymbolFlags::None,
        });
        // .rela.kaslr — SHT_RELA + SHF_ALLOC. A 16-byte payload is
        // large enough that a mis-targeted 4-byte write at offset 20
        // (the correct ELF32 sh_size location) is observable via
        // goblin's post-parse sh_size read.
        let kaslr_id = obj.add_section(
            Vec::new(),
            b".rela.kaslr".to_vec(),
            object::SectionKind::Elf(SHT_RELA),
        );
        obj.append_section_data(kaslr_id, &[0xA5; 16], 1);
        obj.section_mut(kaslr_id).flags = object::SectionFlags::Elf {
            sh_flags: u64::from(SHF_ALLOC),
        };
        // .rel.foo — SHT_REL + SHF_ALLOC. Exercises the second arm
        // of the `is_rela` predicate (`|| sh.sh_type == SHT_REL`) on
        // the ELF32 code path. A regression that dropped SHT_REL
        // from the filter on the 32-bit path would leave this
        // section's sh_size unchanged and trip the post-call
        // assertion below.
        let rel_id = obj.add_section(
            Vec::new(),
            b".rel.foo".to_vec(),
            object::SectionKind::Elf(SHT_REL),
        );
        obj.append_section_data(rel_id, &[0xC7; 12], 1);
        obj.section_mut(rel_id).flags = object::SectionFlags::Elf {
            sh_flags: u64::from(SHF_ALLOC),
        };

        let data = obj.write().unwrap();

        // Positive-control the fixture: the output must actually be
        // ELF32 (is_64 == false) — otherwise this test would false-pass
        // through the ELF64 code path the sibling test already covers.
        // A future object-crate change that remapped I386 to ELF64
        // would surface here rather than silently duplicating existing
        // coverage.
        let pre_elf = goblin::elf::Elf::parse(&data).unwrap();
        assert!(
            !pre_elf.is_64,
            "fixture must produce ELF32 (is_64 == false) to exercise the (20, 4) branch"
        );
        let pre_kaslr = pre_elf
            .section_headers
            .iter()
            .find(|sh| pre_elf.shdr_strtab.get_at(sh.sh_name) == Some(".rela.kaslr"))
            .expect("fixture must carry .rela.kaslr")
            .clone();
        let pre_rel = pre_elf
            .section_headers
            .iter()
            .find(|sh| pre_elf.shdr_strtab.get_at(sh.sh_name) == Some(".rel.foo"))
            .expect("fixture must carry .rel.foo")
            .clone();
        assert_eq!(
            pre_kaslr.sh_type, SHT_RELA,
            ".rela.kaslr sh_type must be SHT_RELA"
        );
        assert!(
            pre_kaslr.sh_flags & u64::from(SHF_ALLOC) != 0,
            ".rela.kaslr must carry SHF_ALLOC; got sh_flags={:#x}",
            pre_kaslr.sh_flags
        );
        assert_eq!(
            pre_kaslr.sh_size, 16,
            ".rela.kaslr sh_size must match 16-byte payload pre-call"
        );
        assert_eq!(pre_rel.sh_type, SHT_REL, ".rel.foo sh_type must be SHT_REL");
        assert!(
            pre_rel.sh_flags & u64::from(SHF_ALLOC) != 0,
            ".rel.foo must carry SHF_ALLOC; got sh_flags={:#x}",
            pre_rel.sh_flags
        );
        assert_eq!(
            pre_rel.sh_size, 12,
            ".rel.foo sh_size must match 12-byte payload pre-call"
        );

        let processed = neutralize_alloc_relocs(&data).unwrap();
        assert_eq!(
            processed.len(),
            data.len(),
            "neutralize_alloc_relocs must not resize the ELF32 buffer"
        );

        let post_elf = goblin::elf::Elf::parse(&processed).unwrap();
        assert!(
            !post_elf.is_64,
            "post-call parse must still be ELF32; the fn must not alter the e_ident class byte"
        );
        let post_kaslr = post_elf
            .section_headers
            .iter()
            .find(|sh| post_elf.shdr_strtab.get_at(sh.sh_name) == Some(".rela.kaslr"))
            .expect(".rela.kaslr must survive the neutralize pass")
            .clone();
        let post_rel = post_elf
            .section_headers
            .iter()
            .find(|sh| post_elf.shdr_strtab.get_at(sh.sh_name) == Some(".rel.foo"))
            .expect(".rel.foo must survive the neutralize pass")
            .clone();
        // Primary invariants: sh_size is zeroed in the ELF32 4-byte
        // slot at offset 20 within both section header entries — one
        // per arm of the SHT_RELA || SHT_REL filter predicate.
        assert_eq!(
            post_kaslr.sh_size, 0,
            "ELF32 .rela.kaslr sh_size must be zeroed (SHT_RELA arm); got {}",
            post_kaslr.sh_size
        );
        assert_eq!(
            post_rel.sh_size, 0,
            "ELF32 .rel.foo sh_size must be zeroed (SHT_REL arm); got {}",
            post_rel.sh_size
        );
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

    use crate::test_support::test_helpers::EnvVarGuard;
}
