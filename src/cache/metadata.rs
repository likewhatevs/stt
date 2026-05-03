//! Pure data types for the kernel image cache.
//!
//! Public shape of cache entries — [`KernelSource`] /
//! [`KernelMetadata`] / [`CacheArtifacts`] / [`KconfigStatus`] /
//! [`CacheEntry`] / [`ListedEntry`] — plus the internal
//! [`classify_corrupt_reason`] dispatcher that routes
//! `read_metadata`-emitted reason strings into stable `error_kind`
//! snake_case identifiers surfaced by `kernel list --json`. No I/O,
//! no syscalls — every entry point is a pure transformation over
//! already-loaded data.

use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

// serde(default) is stripped from cache types so an old metadata.json
// missing a required non-Option field (e.g. has_vmlinux) fails at
// deserialize time → ListedEntry::Corrupt. Option fields tolerate
// absent keys via serde_json's native handling.

/// How a cached kernel's source was acquired, with per-variant
/// payload (git details for `Git`, source-tree path and git hash for
/// `Local`).
///
/// Serialized as `{"type": "tarball"}`, `{"type": "git", "git_hash": ..., "ref": ...}`,
/// or `{"type": "local", "source_tree_path": ..., "git_hash": ...}`.
/// Every per-variant payload field is emitted explicitly — `Option`
/// fields serialize as `null` when `None` rather than being skipped,
/// so JSON consumers see stable keys across every variant regardless
/// of which optional payload values are set.
///
/// On deserialize, serde_json treats absent `Option` keys as `None`,
/// so an old `metadata.json` that drops `git_hash`, `ref`, or
/// `source_tree_path` still round-trips. Cache-integrity enforcement
/// (truncated `metadata.json` surfacing as [`ListedEntry::Corrupt`]
/// via [`crate::cache::CacheDir::list`]) rides on the required
/// non-`Option` fields of [`KernelMetadata`], not on the optional
/// payloads here.
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
        git_hash: Option<String>,
        /// Git ref used for checkout (branch, tag, or ref spec).
        #[serde(rename = "ref")]
        git_ref: Option<String>,
    },
    /// Build of a local on-disk kernel source tree.
    Local {
        /// Path to the source tree on disk. `None` when the tree has
        /// been sanitized for remote cache transport or is otherwise
        /// unavailable.
        source_tree_path: Option<PathBuf>,
        /// Git commit hash of the source tree at build time (short
        /// form). `None` when the tree is not a git repository, the
        /// hash could not be read, or the worktree is dirty — a
        /// HEAD hash does not describe a tree with uncommitted
        /// changes, so identifying it by that hash would mislead a
        /// reproducer.
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

impl KernelSource {
    /// Borrow the `git_hash` field on a [`KernelSource::Local`]
    /// variant. Returns `None` for any other variant or when the
    /// Local variant carries `git_hash: None` (dirty / non-git
    /// tree at acquire time).
    ///
    /// Mainly used by [`crate::cli::kernel_build_pipeline`]'s
    /// post-build dirty re-check, which compares the post-build
    /// HEAD hash against the acquire-time hash to detect mid-build
    /// commits or branch flips. Borrows rather than clones so the
    /// caller does not pay an allocation when only comparing.
    pub fn as_local_git_hash(&self) -> Option<&str> {
        match self {
            KernelSource::Local { git_hash, .. } => git_hash.as_deref(),
            _ => None,
        }
    }
}

/// Metadata stored alongside a cached kernel image.
///
/// Required fields (`source`, `arch`, `image_name`, `built_at`,
/// `has_vmlinux`, `vmlinux_stripped`) must be present in
/// `metadata.json` during deserialization; a truncated file that
/// drops any of them surfaces the entry as [`ListedEntry::Corrupt`]
/// via [`crate::cache::CacheDir::list`] rather than silently
/// defaulting. Optional fields tolerate absent keys as `None`.
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
    /// CRC32 of the user-supplied `--extra-kconfig` fragment (raw
    /// bytes) at build time. `None` for builds without
    /// `--extra-kconfig`.
    pub extra_kconfig_hash: Option<String>,
    /// Whether a vmlinux ELF was cached alongside the image.
    /// Required in metadata.json.
    pub(crate) has_vmlinux: bool,
    /// Whether the cached vmlinux ELF came from a successful strip
    /// pass (`true`) or the raw-fallback path (`false`).
    pub(crate) vmlinux_stripped: bool,
    /// Size in bytes of the SOURCE-TREE vmlinux at cache-store time.
    pub source_vmlinux_size: Option<u64>,
    /// Modification time (seconds since UNIX epoch) of the
    /// SOURCE-TREE vmlinux at cache-store time.
    pub source_vmlinux_mtime_secs: Option<i64>,
}

impl KernelMetadata {
    /// Create a new KernelMetadata with required fields.
    pub fn new(source: KernelSource, arch: String, image_name: String, built_at: String) -> Self {
        KernelMetadata {
            version: None,
            source,
            arch,
            image_name,
            config_hash: None,
            built_at,
            ktstr_kconfig_hash: None,
            extra_kconfig_hash: None,
            has_vmlinux: false,
            vmlinux_stripped: false,
            source_vmlinux_size: None,
            source_vmlinux_mtime_secs: None,
        }
    }

    /// Set the source-tree vmlinux size and mtime captured at cache
    /// store time.
    pub fn with_source_vmlinux_stat(mut self, size: u64, mtime_secs: i64) -> Self {
        self.source_vmlinux_size = Some(size);
        self.source_vmlinux_mtime_secs = Some(mtime_secs);
        self
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

    /// Set the `--extra-kconfig` fragment CRC32 hash.
    pub fn with_extra_kconfig_hash(mut self, hash: Option<String>) -> Self {
        self.extra_kconfig_hash = hash;
        self
    }

    /// Whether a vmlinux ELF was cached alongside the image.
    pub fn has_vmlinux(&self) -> bool {
        self.has_vmlinux
    }

    /// Crate-only mutator for `has_vmlinux`.
    pub(crate) fn set_has_vmlinux(&mut self, value: bool) {
        self.has_vmlinux = value;
    }

    /// Whether the cached vmlinux came from a successful strip pass.
    pub fn vmlinux_stripped(&self) -> bool {
        self.vmlinux_stripped
    }

    /// Crate-only mutator for `vmlinux_stripped`.
    pub(crate) fn set_vmlinux_stripped(&mut self, value: bool) {
        self.vmlinux_stripped = value;
    }
}

/// Bundle of cache artifacts for [`crate::cache::CacheDir::store`].
///
/// The vmlinux path points at the raw (unstripped) ELF. `store()`
/// strips it internally via [`crate::cache::strip_vmlinux_debug`]
/// and writes the result.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CacheArtifacts<'a> {
    /// Path to the kernel boot image (bzImage or Image).
    pub image: &'a Path,
    /// Optional path to the raw (unstripped) vmlinux ELF. `store()`
    /// strips it internally before caching.
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

    /// Attach the raw (unstripped) vmlinux ELF.
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
    /// Entry was built with a different kconfig.
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

impl KconfigStatus {
    /// `true` iff the entry is stale against the current kconfig.
    pub fn is_stale(&self) -> bool {
        matches!(self, Self::Stale { .. })
    }

    /// `true` iff the entry has no recorded kconfig hash.
    pub fn is_untracked(&self) -> bool {
        matches!(self, Self::Untracked)
    }
}

impl fmt::Display for KconfigStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KconfigStatus::Matches => f.write_str("matches"),
            KconfigStatus::Stale { .. } => f.write_str("stale"),
            KconfigStatus::Untracked => f.write_str("untracked"),
        }
    }
}

/// A cached kernel entry returned by
/// [`crate::cache::CacheDir::lookup`] and
/// [`crate::cache::CacheDir::store`].
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
    /// Absolute path to the cached boot image.
    pub fn image_path(&self) -> PathBuf {
        self.path.join(&self.metadata.image_name)
    }

    /// Absolute path to the cached stripped vmlinux ELF, when one
    /// was stored alongside the image.
    pub fn vmlinux_path(&self) -> Option<PathBuf> {
        self.metadata.has_vmlinux.then(|| self.path.join("vmlinux"))
    }

    /// Absolute path to the cached btrfs disk template
    /// (`<entry>/disk-template.img`).
    pub fn disk_template_path(&self) -> PathBuf {
        self.path.join("disk-template.img")
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

    /// Whether this cache entry was built with a user
    /// `--extra-kconfig` fragment.
    pub fn has_extra_kconfig(&self) -> bool {
        self.metadata.extra_kconfig_hash.is_some()
    }
}

/// Entry yielded by [`crate::cache::CacheDir::list`]. Distinguishes
/// valid entries from corrupt ones.
#[derive(Debug)]
#[non_exhaustive]
pub enum ListedEntry {
    /// Valid cache entry with parsed metadata and an image file
    /// present on disk at the metadata-declared path.
    Valid(Box<CacheEntry>),
    /// Entry directory exists but is unusable.
    Corrupt {
        /// Cache key (directory name).
        key: String,
        /// Path to the (corrupt) entry directory.
        path: PathBuf,
        /// Human-readable explanation of why the entry is classified
        /// as corrupt.
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
            ListedEntry::Valid(e) => Some(e.as_ref()),
            ListedEntry::Corrupt { .. } => None,
        }
    }

    /// Machine-readable classification of a corrupt entry's failure
    /// mode. Returns `None` on a `Valid` entry.
    pub fn error_kind(&self) -> Option<&'static str> {
        match self {
            ListedEntry::Valid(_) => None,
            ListedEntry::Corrupt { reason, .. } => Some(classify_corrupt_reason(reason)),
        }
    }
}

/// Shared prefix → `error_kind` classifier.
pub(crate) fn classify_corrupt_reason(reason: &str) -> &'static str {
    if reason == "metadata.json missing" {
        "missing"
    } else if reason.starts_with("metadata.json unreadable: ") {
        "unreadable"
    } else if reason.starts_with("metadata.json schema drift: ") {
        "schema_drift"
    } else if reason.starts_with("metadata.json malformed: ") {
        "malformed"
    } else if reason.starts_with("metadata.json truncated: ") {
        "truncated"
    } else if reason.starts_with("metadata.json parse error: ") {
        "parse_error"
    } else if reason.starts_with("image file ") && reason.contains("missing") {
        "image_missing"
    } else {
        "unknown"
    }
}
