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

use anyhow::Context;
use serde::{Deserialize, Serialize};

/// Filename prefix that marks an in-progress atomic-store directory
/// under the cache root. Format: `{TMP_DIR_PREFIX}{cache_key}-{pid}`.
/// Emitted by [`CacheDir::store`] when composing the tempdir path,
/// recognized by the list/lookup scanners and by the orphan-sweep
/// path in [`clean_orphaned_tmp_dirs`] so a scanner that sees this
/// prefix on a directory name knows to skip (in-progress), and the
/// validator in [`validate_cache_key`] rejects keys starting with it
/// so user input cannot shadow a real tempdir. Centralized here so
/// the three roles — emitter, scanner, validator — cannot drift.
const TMP_DIR_PREFIX: &str = ".tmp-";

/// Subdirectory name under the cache root that holds per-entry
/// coordination lockfiles. Files inside are named
/// `{cache_key}.lock` — full path is
/// `{cache_root}/.locks/{cache_key}.lock`.
///
/// # Why a subdirectory, not a sibling next to the entry
///
/// [`CacheDir::store`] installs new entries via
/// `renameat2(RENAME_EXCHANGE)` on the entry directory itself. Any
/// lockfile living INSIDE the entry directory would be swapped along
/// with the entry — reader processes holding the lockfile's flock
/// keep a stale inode, and the newly installed entry has a different
/// lockfile inode that fresh readers target instead. The two sides
/// stop coordinating.
///
/// Parking the lockfile outside every entry keeps its inode stable
/// across swaps. We pick a dedicated subdirectory rather than
/// sibling-by-prefix (`.lock-{key}`) so the lockfile namespace is
/// cleanly separated from the entry namespace. Cache-enumeration code
/// ([`CacheDir::list`], [`clean_orphaned_tmp_dirs`],
/// [`CacheDir::clean_all`]) walks first-level children looking for
/// entries — the `.locks/` dotfile subdirectory is skipped by the
/// dotfile filter in [`CacheDir::list`] and does not match
/// [`TMP_DIR_PREFIX`] so it bypasses the orphan sweep too. User
/// cache keys likewise cannot shadow lockfile paths: lockfiles are
/// children of `.locks/` rather than first-level entries, so no
/// [`validate_cache_key`] prefix rejection is needed.
///
/// The subdirectory is created lazily on first acquire and reused;
/// `CacheDir` never removes it. Lockfiles inside persist as empty
/// sentinels between runs — the flock itself releases on process
/// death, but the file stays for the next acquirer to reuse.
const LOCK_DIR_NAME: &str = ".locks";

// serde(default) is stripped from cache types so an old metadata.json
// missing a required non-Option field (e.g. has_vmlinux) fails at
// deserialize time → ListedEntry::Corrupt. Option fields tolerate
// absent keys via serde_json's native handling. Other modules use
// serde(default, skip_serializing_if) for producer-omitted fields
// (JSON minification, not compat shims); do not import that pattern here.

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
/// Only the serialization side is pinned here. On deserialize,
/// serde_json treats absent `Option` keys as `None`, so an old
/// `metadata.json` that drops `git_hash`, `ref`, or
/// `source_tree_path` still round-trips — see
/// [`tests::kernel_source_absent_option_keys_deserialize_as_none`].
/// Cache-integrity enforcement (truncated `metadata.json` surfacing
/// as [`ListedEntry::Corrupt`] via [`CacheDir::list`]) rides on the
/// required non-`Option` fields of [`KernelMetadata`], not on the
/// optional payloads here.
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

/// Metadata stored alongside a cached kernel image.
///
/// Required fields (`source`, `arch`, `image_name`, `built_at`,
/// `has_vmlinux`, `vmlinux_stripped`) must be present in
/// `metadata.json` during deserialization; a truncated file that
/// drops any of them surfaces the entry as [`ListedEntry::Corrupt`]
/// via [`CacheDir::list`] rather than silently defaulting. Optional
/// fields (`version`, `config_hash`, `ktstr_kconfig_hash`) and the
/// `Option`-typed payloads inside [`KernelSource`] variants tolerate
/// absent keys as `None` — they participate in the on-disk shape but
/// do not gate cache integrity.
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
    /// Whether a vmlinux ELF was cached alongside the image. When
    /// true, the entry directory contains a `vmlinux` file; see
    /// [`strip_vmlinux_debug`] for the strip policy and
    /// [`KernelMetadata::vmlinux_stripped`] for whether the cached
    /// bytes came from the strip pipeline or the raw fallback.
    ///
    /// Required in metadata.json — plain `bool` without
    /// `#[serde(default)]` must be present during deserialization, so
    /// entries that predate this field surface as Corrupt rather than
    /// defaulting to `false`.
    ///
    /// Visibility: private field, [`KernelMetadata::has_vmlinux`]
    /// reader is public and [`KernelMetadata::set_has_vmlinux`]
    /// mutator is `pub(crate)`. The authoritative writer is
    /// [`CacheDir::store`]; making the field crate-writable-only
    /// enforces that ownership statically instead of leaving the
    /// "don't set this directly" guidance purely advisory (see the
    /// note on `new()` below).
    has_vmlinux: bool,
    /// Whether the cached vmlinux ELF came from a successful strip
    /// pass (`true`) or the raw-fallback path (`false`) where both
    /// the keep-list strip and the debug-prefix strip failed and
    /// [`CacheDir::store`] copied the unstripped bytes.
    ///
    /// Meaningful only when [`has_vmlinux`](Self::has_vmlinux)
    /// returns `true`. When `has_vmlinux` is `false`, this field is
    /// always `false` and has no consumer (there is no vmlinux to
    /// describe).
    ///
    /// Required in metadata.json — plain `bool` without
    /// `#[serde(default)]` must be present during deserialization.
    /// Old entries without this field surface as Corrupt and must be
    /// regenerated (pre-1.0 policy: no serde compatibility shims —
    /// re-running the test that populates the cache rebuilds the
    /// entry with the new field).
    ///
    /// Visibility: private field with the same ownership rules as
    /// [`has_vmlinux`](Self::has_vmlinux) — the authoritative writer
    /// is [`CacheDir::store`].
    vmlinux_stripped: bool,
}

impl KernelMetadata {
    /// Create a new KernelMetadata with required fields.
    ///
    /// Optional fields default to `None` / `false`. Use setter methods
    /// to populate them.
    ///
    /// Note on `has_vmlinux`: builders leave this `false` on
    /// construction and the setter should not be called directly.
    /// The authoritative writer is [`CacheDir::store`], which
    /// inspects the artifacts it's asked to persist and sets
    /// `has_vmlinux = true` iff a stripped `vmlinux` ELF was
    /// actually written into the entry directory. Setting
    /// `has_vmlinux` anywhere else risks drift between the field
    /// and the on-disk contents; that drift stays hidden until a
    /// caller of [`vmlinux_path()`](CacheEntry::vmlinux_path)
    /// consumes a path that doesn't exist, or silently skips a
    /// vmlinux that's there but marked absent. `list()` does not
    /// verify vmlinux presence, so the field alone is the source
    /// of truth for lookups.
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
            vmlinux_stripped: false,
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

    /// Whether a vmlinux ELF was cached alongside the image.
    ///
    /// Returns `true` whether the cached vmlinux came from a
    /// successful strip pass OR from the raw-fallback path (both
    /// produce an entry-dir `vmlinux` file). Use
    /// [`vmlinux_stripped`](Self::vmlinux_stripped) to distinguish
    /// the two shapes.
    ///
    /// Public reader for the private `has_vmlinux` field; consumers
    /// outside this crate must use this method to observe the field
    /// (the field itself is no longer directly accessible). Lookups
    /// that consume the bool to decide whether to resolve
    /// `vmlinux_path()` should read through this accessor.
    pub fn has_vmlinux(&self) -> bool {
        self.has_vmlinux
    }

    /// Crate-only mutator for `has_vmlinux`.
    ///
    /// Exists to allow [`CacheDir::store`] (the authoritative
    /// writer) to record whether the stripped vmlinux was actually
    /// persisted into the entry directory. Not public: external
    /// callers mutating this bit would risk drift between the
    /// field and the on-disk contents, which is exactly what
    /// `store()` ownership is supposed to prevent.
    pub(crate) fn set_has_vmlinux(&mut self, value: bool) {
        self.has_vmlinux = value;
    }

    /// Whether the cached vmlinux came from a successful strip pass.
    ///
    /// Returns `true` when the cached bytes were produced by
    /// [`strip_vmlinux_debug`] (either the keep-list strip or the
    /// debug-prefix fallback inside it). Returns `false` when
    /// [`strip_vmlinux_debug`] itself errored and [`CacheDir::store`]
    /// fell back to copying the unstripped source — a much larger
    /// on-disk payload that still exposes symbols / BTF but indicates
    /// the strip pipeline hit a parse failure on this kernel.
    ///
    /// Meaningful only when [`has_vmlinux`](Self::has_vmlinux) is
    /// `true`; returns `false` otherwise.
    pub fn vmlinux_stripped(&self) -> bool {
        self.vmlinux_stripped
    }

    /// Crate-only mutator for `vmlinux_stripped`.
    ///
    /// Authoritative writer is [`CacheDir::store`]; same ownership
    /// discipline as [`set_has_vmlinux`](Self::set_has_vmlinux).
    pub(crate) fn set_vmlinux_stripped(&mut self, value: bool) {
        self.vmlinux_stripped = value;
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

impl KconfigStatus {
    /// Single source of truth for the "is the entry stale against
    /// the current kconfig?" predicate. Callers that previously
    /// open-coded `matches!(status, KconfigStatus::Stale { .. })`
    /// should use this method so future variants that also mean
    /// "stale" (e.g. a hypothetical version-mismatch variant) are
    /// picked up in one place.
    pub fn is_stale(&self) -> bool {
        matches!(self, Self::Stale { .. })
    }

    /// Single source of truth for the "does the entry lack a
    /// recorded kconfig hash?" predicate. Parallels [`is_stale`] so
    /// `kernel list`'s tag-aggregation loop can track the two
    /// non-Matches variants with one predicate each.
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

    /// Machine-readable classification of a corrupt entry's failure
    /// mode, for consumers that need to branch without parsing the
    /// free-form `reason`. Returns `None` on a `Valid` entry.
    ///
    /// Classifier keys the prefix the [`read_metadata`] and
    /// [`CacheDir::list`] producers actually emit, so a downstream
    /// consumer can switch on the returned `&str` instead of matching
    /// human text. Values are stable identifiers (snake_case) the
    /// `kernel list --json` schema surfaces as the `error_kind` field:
    ///
    /// | `error_kind`    | Producing prefix                        |
    /// |-----------------|-----------------------------------------|
    /// | `"missing"`     | `"metadata.json missing"`               |
    /// | `"unreadable"`  | `"metadata.json unreadable: ..."`       |
    /// | `"schema_drift"`| `"metadata.json schema drift: ..."`     |
    /// | `"malformed"`   | `"metadata.json malformed: ..."`        |
    /// | `"truncated"`   | `"metadata.json truncated: ..."`        |
    /// | `"parse_error"` | `"metadata.json parse error: ..."` (Io fallback) |
    /// | `"image_missing"` | `"image file <name> missing from entry directory"` |
    /// | `"unknown"`     | Any reason that does not match the above prefixes. Surfaces here instead of panicking so a future prefix addition degrades to "unclassified" rather than bailing the whole list. |
    ///
    /// A regression that introduces a new prefix in the producer
    /// without updating this classifier surfaces as `"unknown"` in
    /// the JSON output — consumer-side alerts on that value give
    /// maintainers a clear signal to extend the table.
    pub fn error_kind(&self) -> Option<&'static str> {
        match self {
            ListedEntry::Valid(_) => None,
            ListedEntry::Corrupt { reason, .. } => Some(classify_corrupt_reason(reason)),
        }
    }
}

/// Shared prefix → `error_kind` classifier. Public to the module so
/// tests can pin each prefix routes to its documented value without
/// constructing a whole `ListedEntry::Corrupt`.
fn classify_corrupt_reason(reason: &str) -> &'static str {
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

/// Handle to the kernel image cache directory.
///
/// All operations are local filesystem operations via `std::fs`.
/// Thread safety: individual operations are atomic (rename-based
/// writes), but concurrent callers must coordinate externally.
///
/// `#[non_exhaustive]` matches the sibling pub types in this module
/// ([`KernelMetadata`], [`KernelSource`], [`CacheArtifacts`],
/// [`KconfigStatus`], [`CacheEntry`], [`ListedEntry`]). Every field
/// today is private, so external struct-literal construction is
/// already impossible; the attribute is kept for consistency and to
/// pin the "no cross-crate struct-literal" contract against a future
/// change that promotes a field to `pub`. Use [`Self::new`] or
/// [`Self::with_root`] to construct.
#[derive(Debug)]
#[non_exhaustive]
pub struct CacheDir {
    root: PathBuf,
}

/// Emit a per-lookup warning when a cache entry was created with an
/// unstripped vmlinux — i.e. a prior `CacheDir::store` call took the
/// strip-fallback path. The entry is still usable (monitor and probe
/// symbol lookup works on the raw ELF) but the on-disk payload is
/// much larger than a successfully-stripped entry. Firing every
/// lookup gives the operator a persistent reminder until the cache
/// is rebuilt — complementing the one-shot eprintln at store time.
fn warn_if_unstripped_vmlinux(entry: &CacheEntry) {
    if should_warn_unstripped(entry) {
        eprintln!(
            "cache: using unstripped vmlinux for {} (strip failed on a prior build; \
             re-run with a clean cache to retry)",
            entry.key,
        );
    }
}

/// Pure decision logic for [`warn_if_unstripped_vmlinux`]: `true`
/// iff the entry has a cached vmlinux AND that vmlinux came from
/// the raw-fallback path (strip failure). Returns `false` when
/// `has_vmlinux` is false (no vmlinux was cached; the
/// `vmlinux_stripped` bit is meaningless in that shape and emitting
/// a warning would be noise).
///
/// Separated from the eprintln so the `!has_vmlinux → no warning`
/// and `has_vmlinux && vmlinux_stripped → no warning` branches can
/// be unit-tested without capturing stderr.
fn should_warn_unstripped(entry: &CacheEntry) -> bool {
    entry.metadata.has_vmlinux() && !entry.metadata.vmlinux_stripped()
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
    /// Renamed from `resolve_root` to `default_root` so the name
    /// describes what the returned path represents (the default
    /// the constructor *would* pick), while
    /// [`root`](Self::root) on an instance returns the root a
    /// particular `CacheDir` actually points at — the two names
    /// no longer share a noun.
    pub fn default_root() -> anyhow::Result<PathBuf> {
        resolve_cache_root()
    }

    /// Root directory this `CacheDir` is anchored at. Distinct from
    /// [`default_root`](Self::default_root) (static): `default_root`
    /// says "the default constructor would build here";
    /// `root` says "this specific instance lives here".
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Look up a cached kernel by cache key.
    ///
    /// Returns the cache entry if it exists, has valid metadata, and
    /// contains the expected kernel image file. Returns `None` if the
    /// key is invalid, the entry does not exist, or is corrupted.
    ///
    /// On a successful lookup of an entry whose vmlinux came from the
    /// raw-fallback path (`vmlinux_stripped == false` alongside
    /// `has_vmlinux == true`), emits a per-lookup eprintln warning so
    /// operators running with the default (no-tracing) subscriber see
    /// the signal every time they hit a bloated entry — not just once
    /// at store time when a prior build's strip pipeline failed.
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
        // Entry must have a kernel image file.
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
    /// first). Entries with missing metadata, unparseable metadata,
    /// or a missing image file surface as [`ListedEntry::Corrupt`] at
    /// the end of the Vec. Valid entries are observed to have an
    /// image file present at scan time — the presence check runs
    /// inside this function before classification, so callers do not
    /// need to re-stat for steady-state reads. The guarantee is
    /// best-effort against TOCTOU: concurrent cache mutation
    /// (another process calling `store`/`clean`, or manual rmdir)
    /// can invalidate the invariant between `list()` return and a
    /// subsequent [`CacheEntry::image_path`] open, so callers in
    /// multi-process contexts must still handle ENOENT gracefully.
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
            // Skip dotfile children BEFORE the is_dir check. The
            // cache root's dotfile-namespace is reserved for ktstr's
            // own bookkeeping: `.locks/` (per-entry coordination
            // lockfiles, see [`LOCK_DIR_NAME`]), `.tmp-*` (in-progress
            // store tempdirs, see [`TMP_DIR_PREFIX`]), and any future
            // bookkeeping directory. Cache keys validated by
            // [`validate_cache_key`] cannot begin with `.` — that
            // validator rejects only reserved prefixes, but POSIX
            // convention keeps application-visible data out of
            // dotfile names, and every bookkeeping surface in this
            // module follows the convention. Skipping at the first
            // filter stops `clean_all` / `clean_keep` (which remove
            // every ListedEntry this function returns) from ever
            // touching `.locks/` or any future bookkeeping
            // subdirectory even if `is_dir` would otherwise let it
            // through.
            if name_hint.starts_with('.') {
                continue;
            }
            if !path.is_dir() {
                continue;
            }
            // Skip temp directories from in-progress stores.
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
                Err(reason) => {
                    // Emit a `tracing::info` so operators reviewing
                    // logs can see per-entry corruption without
                    // scraping the CLI table. `reason` carries a
                    // self-classified prefix from `read_metadata` —
                    // `"metadata.json missing"`, `"... unreadable: "`,
                    // `"... schema drift: "`, `"... malformed: "`, or
                    // `"... truncated: "`. The log message itself
                    // stays neutral; downstream log parsers key on
                    // the `reason` prefix rather than the message.
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
    /// `has_vmlinux` and `vmlinux_stripped` fields are overwritten
    /// based on what `store()` actually persisted (whether a vmlinux
    /// was given, and whether the strip pipeline succeeded), so
    /// callers do not need to pre-populate either.
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

        // Serialize atomic-swap installs against every reader (test
        // VM holding LOCK_SH on this entry). The guard drops at the
        // end of this function, after `renameat2` has run and
        // `_guard` (TmpDirGuard) has cleaned the displaced content.
        // A 60 s blocking timeout catches pathologically stuck peers
        // while tolerating a single healthy test run draining. See
        // [`LOCK_DIR_NAME`] for the `.locks/` subdirectory placement
        // rationale.
        let _store_lock =
            self.acquire_exclusive_lock_blocking(cache_key, STORE_EXCLUSIVE_LOCK_TIMEOUT)?;

        let final_dir = self.root.join(cache_key);
        let tmp_dir = self.root.join(format!(
            "{TMP_DIR_PREFIX}{}-{}",
            cache_key,
            std::process::id(),
        ));

        // Clean up any stale temp dir from a prior crash. create_dir_all
        // on tmp_dir also creates self.root lazily on first store.
        if tmp_dir.exists() {
            fs::remove_dir_all(&tmp_dir)?;
        }
        // Scan-and-clean orphaned `.tmp-*` siblings from prior
        // panic/abort debris across OTHER PIDs. The same-PID cleanup
        // above only fires for the current PID's tmp dir; a prior
        // run that died with a different PID leaves debris behind,
        // which accumulates across crashes and takes disk space.
        // `clean_orphaned_tmp_dirs` reads the cache root, inspects
        // each `.tmp-{key}-{pid}` name, and removes entries whose
        // `{pid}` is no longer a live pid. Errors are logged
        // and swallowed — a failure to clean orphans must not block
        // a successful store.
        if let Err(e) = clean_orphaned_tmp_dirs(&self.root) {
            tracing::warn!(err = %format!("{e:#}"), "clean_orphaned_tmp_dirs failed; continuing store");
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
        //
        // `vmlinux_stripped` distinguishes the two success shapes:
        // `true` on the strip-ok path, `false` on the raw-fallback
        // path. Consumers (e.g. `ktstr cache list --json`) surface
        // this so operators notice when a newly-cached kernel hit
        // the fallback — a silent size regression that used to only
        // appear in tracing logs.
        let (has_vmlinux, vmlinux_stripped) = if let Some(vmlinux) = artifacts.vmlinux {
            let vmlinux_dest = tmp_dir.join("vmlinux");
            match strip_vmlinux_debug(vmlinux) {
                Ok(stripped) => {
                    fs::copy(stripped.path(), &vmlinux_dest)
                        .map_err(|e| anyhow::anyhow!("copy stripped vmlinux to cache: {e}"))?;
                    (true, true)
                }
                Err(e) => {
                    // eprintln in addition to tracing::warn so the
                    // failure is visible to operators running with
                    // the default (no-tracing) subscriber. Without
                    // this, the only signal was a tracing::warn
                    // swallowed by the default writer — the cached
                    // entry grew unboundedly (the uncompressed
                    // source vmlinux's full size) with no on-screen
                    // explanation.
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

        // Write metadata. has_vmlinux and vmlinux_stripped reflect
        // what we actually stored, overriding whatever the caller set.
        let mut meta = metadata.clone();
        meta.set_has_vmlinux(has_vmlinux);
        meta.set_vmlinux_stripped(vmlinux_stripped);
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

    // ------------------------------------------------------------------
    // Per-entry coordination locks
    // ------------------------------------------------------------------
    //
    // See [`LOCK_DIR_NAME`] for the on-disk shape rationale. A
    // shared (reader) lock is held by each test-VM run against the
    // cache entry it resolved its kernel image from; an exclusive
    // (writer) lock is held by `CacheDir::store` for the window
    // spanning temp-dir composition → atomic rename. flock(2) is
    // per-open-file-description; RAII guards drop the OwnedFd which
    // releases the lock.

    /// Absolute path to the coordination lockfile for `cache_key`.
    /// The lockfile lives at `{cache_root}/.locks/{cache_key}.lock`
    /// (see [`LOCK_DIR_NAME`] for the subdirectory placement
    /// rationale). Does not touch the filesystem — use
    /// [`CacheDir::ensure_lock_dir`] before opening the file to make
    /// sure the parent `.locks/` subdirectory exists.
    pub(crate) fn lock_path(&self, cache_key: &str) -> PathBuf {
        self.root
            .join(LOCK_DIR_NAME)
            .join(format!("{cache_key}.lock"))
    }

    /// Create the `{cache_root}/.locks/` subdirectory if absent.
    /// Callers invoke this before [`open_lockfile`] so the parent
    /// exists; `fs::create_dir_all` is idempotent (Ok on existing
    /// directory), mirroring the lazy-create semantics
    /// [`CacheDir::store`] uses for the cache root itself.
    fn ensure_lock_dir(&self) -> anyhow::Result<()> {
        let dir = self.root.join(LOCK_DIR_NAME);
        fs::create_dir_all(&dir)
            .with_context(|| format!("create lock subdirectory {}", dir.display()))
    }

    /// Acquire `LOCK_SH` on the cache-entry lockfile. Blocks until the
    /// lock is available or the default shared-lock timeout elapses.
    ///
    /// Test-VM runs call this on the entry they resolved their kernel
    /// image from. A concurrent `CacheDir::store` that holds `LOCK_EX`
    /// for an atomic-swap install is serialized behind every reader.
    /// Multiple readers coexist.
    ///
    /// The underlying lockfile is created on-demand when it does not
    /// exist (mode 0o666); the cache root is created if missing
    /// (matching [`CacheDir::store`]'s lazy root creation). Returns
    /// `Err` on filesystem errors or timeout.
    pub fn acquire_shared_lock(&self, cache_key: &str) -> anyhow::Result<SharedLockGuard> {
        validate_cache_key(cache_key)?;
        let fd = acquire_flock_with_timeout(
            self,
            cache_key,
            FlockMode::Shared,
            SHARED_LOCK_DEFAULT_TIMEOUT,
        )?;
        Ok(SharedLockGuard { fd })
    }

    /// Acquire `LOCK_EX` on the cache-entry lockfile. Blocks until the
    /// lock is available or `timeout` elapses.
    ///
    /// Called internally by [`CacheDir::store`] to serialize atomic-swap
    /// installs against concurrent readers (test runs holding
    /// `LOCK_SH`). On timeout, the error message lists the PIDs that
    /// are currently holding the lock (parsed from `/proc/locks`) so
    /// the operator can kill or wait on them deliberately.
    pub fn acquire_exclusive_lock_blocking(
        &self,
        cache_key: &str,
        timeout: std::time::Duration,
    ) -> anyhow::Result<ExclusiveLockGuard> {
        validate_cache_key(cache_key)?;
        let fd = acquire_flock_with_timeout(self, cache_key, FlockMode::Exclusive, timeout)?;
        Ok(ExclusiveLockGuard { fd })
    }

    /// Non-blocking `LOCK_EX` attempt on the cache-entry lockfile. Used
    /// as a pre-check by `--force` operator-driven rebuilds: fail fast
    /// with a PID list if tests are actively using the entry, instead
    /// of silently stomping on an in-flight run.
    ///
    /// Returns `Err` immediately when any reader or writer holds the
    /// lock; the error surfaces the holder PIDs parsed from
    /// `/proc/locks`.
    pub fn try_acquire_exclusive_lock(
        &self,
        cache_key: &str,
    ) -> anyhow::Result<ExclusiveLockGuard> {
        validate_cache_key(cache_key)?;
        self.ensure_lock_dir()?;
        let path = self.lock_path(cache_key);
        // `try_flock` folds `open(O_CLOEXEC) + flock(LOCK_NB)` into a
        // single call; on `EWOULDBLOCK` it returns `Ok(None)` with the
        // fd already dropped, so the /proc/locks scan below cannot
        // confuse our own new fd (just-opened, never flocked) with a
        // real holder.
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

// ---------------------------------------------------------------------------
// Cache-lock timeouts — reader 10s, writer 60s (INTENTIONAL asymmetry)
// ---------------------------------------------------------------------------
//
// Readers (test VMs) wait 10 s for a writer to finish. Writers
// (`CacheDir::store`) wait 60 s for all readers to drain. The
// asymmetry is deliberate, not a tuning artifact:
//
// - A writer's critical section is SHORT — temp-dir copy + metadata
//   write + `renameat2(RENAME_EXCHANGE)` — measured in low seconds
//   even for multi-GB vmlinux images. A reader waiting 10 s must
//   have caught a genuinely stuck writer, and surfacing that as an
//   error beats hanging the test run indefinitely.
// - A writer's contention is LONG — it must outlast every reader
//   currently running a test on this entry. A single test run takes
//   tens of seconds (VM boot + scenario + teardown); several parallel
//   runs serialize through their own nextest scheduling. 60 s is
//   empirically enough for the natural drain without rewarding
//   pathological test loops.
// - Pathological peers in either direction surface as actionable
//   errors (holder PIDs + cmdlines in the error text) instead of
//   silent hangs.

/// Default wall-clock timeout for [`CacheDir::acquire_shared_lock`].
/// See module-level rationale block above.
const SHARED_LOCK_DEFAULT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Internal poll interval for [`acquire_flock_with_timeout`]. flock(2)
/// has no native timed-wait variant; we emulate one by retrying the
/// non-blocking form with a short sleep. 100ms balances responsiveness
/// (contention clears in ≤1 poll under normal load) against CPU burn
/// (at most 10 wakes/s per waiter).
const FLOCK_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(100);

/// Timeout for [`CacheDir::store`]'s internal `LOCK_EX` acquire.
/// See module-level rationale block above for the reader/writer
/// asymmetry (10 s vs 60 s).
const STORE_EXCLUSIVE_LOCK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

// Flock primitive lives in [`crate::flock`], shared with
// `crate::vmm::host_topology`'s LLC / per-CPU acquires. LlcLockMode
// remains separate at the scheduler-intent layer (perf-mode vs
// no-perf-mode request, not a flock operation).
use crate::flock::FlockMode;

/// RAII guard for a `LOCK_SH` hold on a cache-entry lockfile. Dropping
/// the guard releases the advisory lock via `OwnedFd::drop` — no
/// explicit `unlock()` call needed. The `fd` field carries the
/// `OwnedFd` whose Drop releases the kernel-side flock; nothing reads
/// it after construction, but it must stay named (not `_fd`) so
/// grep-based audits and struct-literal tools can find the holder
/// field without a leading-underscore filter.
#[derive(Debug)]
pub struct SharedLockGuard {
    #[allow(dead_code)]
    fd: std::os::fd::OwnedFd,
}

/// RAII guard for a `LOCK_EX` hold on a cache-entry lockfile. Dropping
/// the guard releases the advisory lock via `OwnedFd::drop`.
#[derive(Debug)]
pub struct ExclusiveLockGuard {
    #[allow(dead_code)]
    fd: std::os::fd::OwnedFd,
}

/// Poll-acquire an advisory flock with a wall-clock timeout.
///
/// `flock(2)` has no native timed-wait form — the blocking variant
/// cannot be cancelled, so we loop on [`crate::flock::try_flock`]'s
/// non-blocking operation with [`FLOCK_POLL_INTERVAL`] between
/// attempts. `try_flock` folds `open(O_CLOEXEC)` + `flock(LOCK_NB)`
/// and already drops the fd on `Ok(None)`, so stacked
/// open-file-descriptions cannot build up across iterations.
///
/// Creates the `.locks/` subdirectory lazily when missing (matches
/// [`CacheDir::store`]'s lazy-root semantics for a freshly-resolved
/// cache path).
///
/// Returns `Err` on the wall-clock deadline or on unexpected flock
/// errors. Timeout errors surface the holder PID list from
/// `/proc/locks` via [`crate::flock::read_holders`] so operators can
/// identify the peer holding the lock.
fn acquire_flock_with_timeout(
    cache: &CacheDir,
    cache_key: &str,
    kind: FlockMode,
    timeout: std::time::Duration,
) -> anyhow::Result<std::os::fd::OwnedFd> {
    cache.ensure_lock_dir()?;
    let path = cache.lock_path(cache_key);

    let deadline = std::time::Instant::now() + timeout;
    loop {
        match crate::flock::try_flock(&path, kind)? {
            Some(fd) => return Ok(fd),
            None => {
                if std::time::Instant::now() >= deadline {
                    let holders = crate::flock::read_holders(&path).unwrap_or_default();
                    let kind_str = match kind {
                        FlockMode::Shared => "LOCK_SH",
                        FlockMode::Exclusive => "LOCK_EX",
                    };
                    anyhow::bail!(
                        "flock {kind_str} on cache entry {cache_key:?} timed \
                         out after {timeout:?} (lockfile {lockfile}, \
                         holders: {holders}).",
                        lockfile = path.display(),
                        holders = crate::flock::format_holder_list(&holders),
                    );
                }
                std::thread::sleep(FLOCK_POLL_INTERVAL);
            }
        }
    }
}

/// Scan `cache_root` for `.tmp-{key}-{pid}` directories whose `{pid}`
/// is no longer a live process and remove them.
///
/// Same-PID orphan cleanup lives in `store()`; this helper handles
/// the cross-PID case — debris left by a prior process that died
/// with a different PID. Without periodic cleanup, crashes
/// accumulate directories forever. Called from the start of every
/// `store()` so ordinary fetch traffic cleans up without a
/// dedicated CLI knob.
///
/// Discrimination: parse the trailing `-{digits}` suffix as a pid
/// and probe liveness via `kill(pid, None)` (the standard
/// signal-zero probe). `Err(ESRCH)` is the only outcome that
/// justifies removal; `Ok(())` and `Err(EPERM)` both indicate a
/// live pid whose debris we leave alone. A pid recycled between
/// orphan creation and this scan is treated as alive — false
/// negative, preserves debris — but never false positive.
///
/// # TOCTOU: pid reuse between the `kill(pid, None)` probe and
///   `remove_dir_all(path)`
///
/// There is an inherent race between the liveness probe above and
/// the filesystem unlink below: the kernel is free to reap the
/// embedded pid (actually dead → probe returns `ESRCH`), allocate
/// it to a fresh unrelated process, and schedule that process
/// before `remove_dir_all` starts. The cleanup then proceeds
/// because the probe snapshot said "dead" — it cannot observe the
/// subsequent reuse. The consequence is bounded and harmless: the
/// new pid-holder does NOT own the `.tmp-{key}-{pid}` directory
/// (it was created by the prior dead process), so
/// `remove_dir_all` operates on paths that no live process is
/// reading or writing. The reuse does not cross into the cleanup
/// target's data. The pid-reuse wraparound distance is the
/// runtime `/proc/sys/kernel/pid_max` sysctl (the default on
/// modern Linux is `2^22` == `4_194_304`, matching the
/// compile-time ceiling `include/linux/threads.h::PID_MAX_LIMIT`;
/// administrators can lower it via `sysctl kernel.pid_max=<N>`
/// to shrink the wraparound surface, or — less commonly — raise
/// it on 64-bit hosts where the 32-bit pid_t carries the higher
/// bound). So reuse is rare on a host with moderate fork rate
/// but not vanishingly so on CI runners that recycle pids
/// quickly, especially if the runner's base image has lowered
/// `pid_max` to fit a container namespace.
///
/// # TOCTOU: concurrent ktstr processes cannot collide on the
///   same `.tmp-` directory
///
/// Two ktstr processes running against the same cache root
/// cannot create conflicting `.tmp-{key}-{pid}` entries because
/// the path embeds the CREATOR's pid. Process A with pid=100
/// creates `.tmp-foo-100`; process B with pid=200 creates
/// `.tmp-foo-200`. The scan here classifies both entries
/// independently via the `kill(pid, None)` probe on the EMBEDDED
/// pid, so a live sibling's debris is never misclassified as
/// the scanner's own orphan — the scanner never writes a path
/// embedding its own pid AND holding liveness for a foreign pid
/// at the same time. A live ktstr whose pid happens to be
/// reused after death falls under the pid-reuse case above, not
/// this concurrent-process case, because the embedded pid
/// refers to the DEAD creator, not the current live process
/// occupying the same pid slot.
///
/// We accept the race rather than serialize behind an additional
/// lock because (a) the damage model is "delete a dead process's
/// leftover tempdir"— exactly the scan's intent — regardless of
/// what happens to the pid slot AFTER the probe, and (b) the only
/// alternative that closes the race (an exclusive `flock` on each
/// tempdir before remove) would compound its own failure modes in
/// exchange for eliminating an effectively-benign race. The
/// reverse race (pid reuse BEFORE the probe, producing a false
/// "alive" verdict) is already covered by the "false negative,
/// preserves debris" bullet above — debris staying one extra cycle
/// is acceptable; deleting a live process's state would not be.
///
/// Errors during individual entry walks are swallowed and logged
/// (each one would prevent a store that otherwise has nothing to
/// do with the unreadable entry). Filesystem-level read errors on
/// the cache root itself are propagated.
fn clean_orphaned_tmp_dirs(cache_root: &Path) -> anyhow::Result<()> {
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
            Err(_) => continue, // non-UTF-8, not a `.tmp-` we created
        };
        if !name.starts_with(TMP_DIR_PREFIX) {
            // Lockfile subdirectory (`.locks/`) and any future
            // dotfile bookkeeping sibling fall through here. They
            // don't start with `.tmp-`, so this prefix filter
            // excludes them from the tmp-dir sweep — exactly the
            // namespace-separation contract advertised in the
            // `LOCK_DIR_NAME` comment. No extra check needed.
            continue;
        }
        // Suffix parse: `.tmp-{key}-{pid}`. Key may itself contain
        // `-`; the PID is the last `-`-separated token.
        let pid_str = match name.rsplit_once('-') {
            Some((_, suffix)) if !suffix.is_empty() => suffix,
            _ => continue, // malformed — not our format
        };
        let pid: i32 = match pid_str.parse() {
            Ok(p) => p,
            Err(_) => continue, // non-numeric suffix — not our format
        };
        // Portable liveness probe via `kill(pid, signal=None)`:
        // `Ok(())` — signal could have been delivered, pid is alive
        // (same uid or we hold CAP_KILL). `Err(ESRCH)` — pid is
        // dead; the only case where removal is safe. `Err(EPERM)` —
        // pid is alive but owned by another uid; its debris is not
        // ours to touch. Any other errno treats the pid as alive and
        // preserves the debris (false negatives are recoverable;
        // false positives delete live state).
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
    if key.starts_with(TMP_DIR_PREFIX) {
        anyhow::bail!("cache key must not start with {TMP_DIR_PREFIX} (reserved): {key:?}",);
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
///
/// On failure returns a human-readable reason suitable for surfacing
/// through [`ListedEntry::Corrupt::reason`]. The reason carries a
/// distinct prefix per failure mode so `kernel list` output and CI
/// logs can point the user at the specific cause:
/// - `"metadata.json missing"` — the file is absent (ENOENT), which
///   typically means the directory is not a cache entry at all
///   (scanner stumbled onto an unrelated dir).
/// - `"metadata.json unreadable: {e}"` — other I/O error from
///   `fs::read_to_string` (permissions, dangling symlink, etc.).
/// - `"metadata.json schema drift: {e}"` — serde_json's
///   `Category::Data` branch: JSON parsed cleanly but does not match
///   the [`KernelMetadata`] shape (missing required field, wrong
///   type on a present field). Signals a cache written by a ktstr
///   version whose `KernelMetadata` schema has since changed.
/// - `"metadata.json malformed: {e}"` — serde_json's
///   `Category::Syntax` branch: the file is not valid JSON at all
///   (stray characters, unbalanced braces, etc.).
/// - `"metadata.json truncated: {e}"` — serde_json's `Category::Eof`
///   branch: the file ends mid-value. Typical cause is a
///   partially-written metadata from a crashed store.
///
/// `Category::Io` is not reachable from `from_str`, which only sees
/// an in-memory `&str` — the I/O branch exists for `from_reader`
/// callers. If a future serde_json promotes `Category::Io` onto the
/// `from_str` path, the fallback arm in the match below keeps the
/// entry classified (generic "parse error") rather than panicking.
fn read_metadata(dir: &Path) -> Result<KernelMetadata, String> {
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
    let metadata = read_metadata(dir).ok()?;
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
    let data =
        neutralize_relocs(&raw).map_err(|e| anyhow::anyhow!("preprocess vmlinux ELF: {e}"))?;

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

/// Rewrite every relocation section (`SHT_REL`, `SHT_RELA`, `SHT_RELR`,
/// `SHT_CREL`) to `SHT_PROGBITS` with `sh_size = 0`, regardless of the
/// `SHF_ALLOC` flag, returning a modified copy of the bytes.
///
/// Workaround for `object::build::elf::Builder::read`. The Builder
/// routes sections by `sh_type` through two mechanisms that each pose
/// failure modes on malformed reloc inputs.
///
/// **`SHT_REL` / `SHT_RELA`** go through `section.rel()` /
/// `section.rela()` (object-0.37.3/src/build/elf.rs:187,198) which
/// call `data_as_array::<Rel|Rela<Endianness>>` against the raw
/// bytes. Four independent failure modes trip that call:
///
/// - `sh_offset + sh_size` byte range exceeds the file (arm64 kernel
///   7.0's observed failure mode, matches `"Invalid ELF relocation
///   section offset or size"`).
/// - `sh_size` not divisible by the entry size (24 bytes for `Rela64`,
///   16 for `Rel64`) — `slice_from_all_bytes` rejects the leftover
///   tail.
/// - Under ktstr's feature set (`default-features = false,
///   features = ["build"]`) the runtime-endian `Rela<Endianness>`
///   uses the non-`unaligned` `aligned` module in
///   object-0.37.3/src/endian.rs with `align_of == 8`. Even with
///   `sh_size == 0`, `read_bytes_at` returns a literal `Ok(&[])`
///   whose dangling pointer (`0x1` on x86_64) is NOT 8-aligned, so
///   `slice_from_all_bytes::<Rela64>` fails the alignment check.
///   Zeroing `sh_size` alone is insufficient under this feature
///   configuration.
/// - For allocated relocs with `sh_link == 0`,
///   `read_relocations_impl` bounds-checks each entry's `r_info`
///   symbol index against an empty dynamic symtab
///   (`symtab_len == 0`); any non-null index trips "Invalid symbol
///   index N in relocation section at index M".
///
/// **`SHT_RELR` / `SHT_CREL`** short-circuit both `section.rel()`
/// (returns `Ok(None)` because `sh_type != SHT_REL`) and
/// `section.rela()` (`sh_type != SHT_RELA`) in the same
/// `Builder::read` loop, then reach the sh_type match at
/// build/elf.rs:221 which dispatches to `section.data()`
/// (line 225-232) — the opaque-data path. Failure mode there is
/// `sh_offset + sh_size` past the file end, surfacing as
/// `"Invalid ELF section size or offset"`. Empty-slice alignment
/// does not apply (opaque byte reads have no entry alignment
/// requirement).
///
/// Rewriting `sh_type` to `SHT_PROGBITS` routes every matching section
/// through the same opaque-data path regardless of its original type:
/// `section.rel()` / `section.rela()` / `section.relr()` /
/// `section.crel()` all return `Ok(None)` immediately at their
/// respective sh_type mismatch guards (section.rs:829, 849, 867, 886),
/// the sh_type match at build/elf.rs:221 falls through to the
/// `SHT_PROGBITS` arm, and `section.data(endian, data)` returns
/// `Ok(&[])` via the zero-size short-circuit in `read_bytes_at` (line
/// 132) without any alignment check on the empty slice's pointer.
/// Builder::read succeeds; Builder::write reserves 0 bytes for the
/// opaque data and emits a well-formed section header.
///
/// Zeroing `sh_size` in the same pass ensures that if a downstream
/// reader re-parses the output with a strict "sh_offset + sh_size <=
/// file_len" check, the zero-length section passes trivially. The
/// section still carries its original name and file offset so the
/// keep-list or debug-prefix strip that runs next can delete it by
/// name.
///
/// The output is indistinguishable from a file that never had the
/// offending relocation sections: their bytes (pre-existing entries)
/// are left in place but orphaned, and the strip passes that follow
/// remove their headers by name. Neither monitor nor probe walks
/// relocation entries — the output is semantically equivalent.
///
/// No-op for ELFs that have no matching relocation sections (returns
/// the original bytes copied into a new `Vec`).
fn neutralize_relocs(data: &[u8]) -> anyhow::Result<Vec<u8>> {
    // SHT_RELR and SHT_CREL are not exposed by goblin; use the
    // object-crate constants (19 and 0x40000014 respectively, matching
    // object-0.37.3/src/elf.rs).
    const SHT_RELR: u32 = object::elf::SHT_RELR;
    const SHT_CREL: u32 = object::elf::SHT_CREL;
    // SHT_PROGBITS (value 1) is the "program data" section type —
    // opaque bytes that the Builder reads as `SectionData::Data` and
    // writes back verbatim. Rewriting reloc sections to this type
    // routes them through the opaque path, avoiding the rel/rela/relr/
    // crel parse branches that the empty-slice alignment pathology
    // breaks.
    const SHT_PROGBITS: u32 = goblin::elf::section_header::SHT_PROGBITS;

    let elf = goblin::elf::Elf::parse(data)
        .map_err(|e| anyhow::anyhow!("parse vmlinux ELF for preprocess: {e}"))?;
    let mut out = data.to_vec();
    let shoff = elf.header.e_shoff as usize;
    let shentsize = elf.header.e_shentsize as usize;
    // Section header field offsets (same order in ELF32 and ELF64, width
    // differs on the 64-bit `sh_flags`/`sh_addr`/`sh_offset` path).
    // ELF64: sh_name(4) sh_type(4) sh_flags(8) sh_addr(8) sh_offset(8)
    //        sh_size(8) ... -> sh_size at offset 32.
    // ELF32: sh_name(4) sh_type(4) sh_flags(4) sh_addr(4) sh_offset(4)
    //        sh_size(4) ... -> sh_size at offset 20.
    // sh_type is at offset 4 with width 4 in both layouts (the ELF
    // spec fixes it at `Elf{32,64}_Word = u32`).
    let (sh_size_offset, sh_size_width) = if elf.is_64 { (32, 8) } else { (20, 4) };
    let sh_type_offset: usize = 4;
    let sh_type_width: usize = 4;
    // sh_type is stored little- or big-endian per the ELF header's
    // e_ident[EI_DATA]. goblin::elf::Elf's `little_endian` field
    // reports the observed endianness.
    let le = elf.little_endian;
    use goblin::elf::section_header::{SHT_REL, SHT_RELA};
    for (i, sh) in elf.section_headers.iter().enumerate() {
        let is_reloc = matches!(sh.sh_type, SHT_REL | SHT_RELA | SHT_RELR | SHT_CREL);
        if !is_reloc {
            continue;
        }
        let entry_offset = shoff
            .checked_add(
                i.checked_mul(shentsize)
                    .ok_or_else(|| anyhow::anyhow!("section header table overflow at index {i}"))?,
            )
            .ok_or_else(|| anyhow::anyhow!("section header offset overflow at index {i}"))?;
        let type_offset = entry_offset
            .checked_add(sh_type_offset)
            .ok_or_else(|| anyhow::anyhow!("sh_type offset overflow at index {i}"))?;
        let type_end = type_offset
            .checked_add(sh_type_width)
            .ok_or_else(|| anyhow::anyhow!("sh_type end overflow at index {i}"))?;
        let size_offset = entry_offset
            .checked_add(sh_size_offset)
            .ok_or_else(|| anyhow::anyhow!("sh_size offset overflow at index {i}"))?;
        let size_end = size_offset
            .checked_add(sh_size_width)
            .ok_or_else(|| anyhow::anyhow!("sh_size end overflow at index {i}"))?;
        if type_end > out.len() || size_end > out.len() {
            anyhow::bail!("section header {i} sh_type or sh_size field extends past file end");
        }
        // Write sh_type = SHT_PROGBITS (u32, per the ELF spec for
        // Elf{32,64}_Word) in the file's endianness.
        let type_bytes: [u8; 4] = if le {
            SHT_PROGBITS.to_le_bytes()
        } else {
            SHT_PROGBITS.to_be_bytes()
        };
        out[type_offset..type_end].copy_from_slice(&type_bytes);
        // Zero sh_size (endian-agnostic: all bytes are zero).
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
/// After stripping, checks the result has a non-empty symbol table.
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

    // Check symtab survivors BEFORE writing. The `Builder` state
    // already holds every parsed symbol; counting named, non-deleted
    // entries here lets us fail fast without the post-write
    // `goblin::elf::Elf::parse(&out)` we used to run. The null symbol
    // (index 0) is filtered via the empty-name check, matching the
    // semantics of the old check pass.
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

/// Fallback strip: remove `.debug_*`, `.comment`, and neutralized
/// relocation sections (`.rela.*`, `.rel.*`, `.relr.*`, `.crel.*`).
/// Uses the shared [`crate::elf_strip::rewrite`] primitive.
///
/// The reloc-prefix arms delete the header entries that
/// [`neutralize_relocs`] left behind with `sh_size = 0`. Without this
/// pass the fallback output carries one zero-size ghost section
/// header per reloc section — parseable but wasted section-header
/// bytes that consumers never walk (neither monitor nor probe reads
/// relocation entries).
fn strip_debug_prefix(data: &[u8]) -> anyhow::Result<Vec<u8>> {
    crate::elf_strip::rewrite(data, |name| {
        name.starts_with(b".debug_")
            || name == b".comment"
            || name.starts_with(b".rela.")
            || name.starts_with(b".rel.")
            || name.starts_with(b".relr.")
            || name.starts_with(b".crel.")
    })
    .map_err(|e| anyhow::anyhow!("rewrite stripped vmlinux (fallback): {e}"))
}

/// Resolve the cache root directory path with a per-cache `suffix`
/// (`"kernels"` for the kernel cache, `"models"` for the model cache).
///
/// Single source of truth for env-variable handling and HOME
/// validation across both ktstr cache flavors. Both
/// [`resolve_cache_root`] (kernel cache) and
/// [`crate::test_support::model::resolve_cache_root`] (model cache)
/// route through here so a future change to environment-variable
/// semantics or HOME validation lands once.
///
/// Resolution cascade:
/// 1. `KTSTR_CACHE_DIR` (with non-UTF-8 bail). The override returns
///    the path verbatim — no `suffix` is appended, since the
///    operator who sets `KTSTR_CACHE_DIR` is naming the literal
///    cache root they want. The kernel cache and the model cache
///    co-locate under the same root in this case but never collide
///    on filesystem paths because cache keys (kernel build hashes)
///    and GGUF filenames live in disjoint name spaces under the
///    root.
/// 2. `XDG_CACHE_HOME/ktstr/{suffix}` when set and non-empty.
/// 3. `$HOME/.cache/ktstr/{suffix}` after HOME validation (3 arms:
///    unset/empty, literal `/`, non-absolute path — see body
///    comments for the per-shape rationale).
///
/// Does not create the directory — the caller is responsible for
/// ensuring it exists.
///
/// Non-UTF-8 handling: `KTSTR_CACHE_DIR` is the explicit operator
/// override and a silent fall-through on a non-UTF-8 value would
/// leave the operator debugging why their override "didn't work"
/// when the fallback `$HOME/.cache/ktstr` took over instead.
/// [`crate::test_support::test_helpers::EnvVarGuard`] accepts
/// arbitrary `OsStr`, so a test fixture or a real-world
/// locale-encoded path can legally put non-UTF-8 bytes in this
/// variable. A non-UTF-8 `KTSTR_CACHE_DIR` surfaces as an
/// actionable error naming the variable and pointing at a UTF-8
/// replacement path — the override never fails silently.
pub(crate) fn resolve_cache_root_with_suffix(suffix: &str) -> anyhow::Result<PathBuf> {
    // 1. Explicit override.
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
    // 2. XDG_CACHE_HOME/ktstr/{suffix}.
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME")
        && !xdg.is_empty()
    {
        return Ok(PathBuf::from(xdg).join("ktstr").join(suffix));
    }
    // 3. $HOME/.cache/ktstr/{suffix}.
    //
    // HOME shape validation lives in [`validate_home_for_cache`]
    // below — it reads `HOME` and rejects unset/empty, literal
    // `/`, and non-absolute values, returning the validated
    // `PathBuf` for the caller to extend with the cache-flavor
    // suffix. The gate is INTENTIONALLY NARROW: it catches only
    // shapes where the resulting cache path is structurally wrong
    // before the filesystem has had a chance to surface its own
    // diagnostic. Other malformed HOME values (non-existent
    // directory, path pointing at a non-directory, kernel pseudo-
    // filesystem like /proc) pass through; downstream
    // open()/statvfs() surface the real error with the offending
    // path embedded so the operator sees what HOME actually
    // expanded to.
    //
    // Cases this gate does NOT catch (intentional):
    //
    // - `HOME=/nonexistent`: ENOENT at use-time, with the offending
    //   path in the error message. Pre-validating with `Path::exists`
    //   would race against an operator who creates the directory
    //   between our check and the use site, OR mask a stale-NFS
    //   timeout that would otherwise surface as a real error.
    // - `HOME=/dev/null`: ENOTDIR at use-time. Same race rationale
    //   as `/nonexistent`; pre-flighting with metadata() adds a
    //   syscall on every cache lookup for a vanishingly rare case.
    // - `HOME=/proc`, `HOME=/sys`, `HOME=/`-rooted pseudo-fs paths:
    //   these surface as EROFS or EACCES on write attempts. The
    //   error path is well-described by the OS-level diagnostic;
    //   adding a special case here would be churn without payoff.
    // - `HOME=//`, `HOME=/./`, `HOME=/.`: these expand to root via
    //   POSIX path normalization. PathBuf does not normalize at
    //   `from`, so the literal junk-path lands on the filesystem
    //   layer. We could canonicalize() here to catch them, but
    //   canonicalize is a syscall on every cache lookup and these
    //   shapes are rare enough (operator typo or shell quoting
    //   accident) that the OS-level "operation not permitted on
    //   /.cache" suffices.
    let home = validate_home_for_cache()?;
    Ok(home.join(".cache").join("ktstr").join(suffix))
}

/// Read `HOME` from the environment, reject values that produce a
/// guaranteed-junk cache path, and return the validated `PathBuf`.
///
/// On `Ok` the returned PathBuf is `PathBuf::from(<HOME value>)` —
/// the caller appends the cache-flavor suffix (e.g. `.cache/ktstr/kernels`).
/// On `Err` the diagnostic names the specific rejection case so the
/// operator can fix HOME or set `KTSTR_CACHE_DIR` / `XDG_CACHE_HOME`
/// instead.
///
/// Three rejected shapes (full rationale in the body comments
/// above [`resolve_cache_root_with_suffix`]'s HOME-fallback site,
/// repeated near the matching check below):
/// 1. Unset / empty (`std::env::var().unwrap_or_default()` collapses
///    both `Err(NotPresent)` and `Ok("")` into `""`).
/// 2. Literal `/` (root user / container init with no home).
/// 3. Non-absolute (CWD-relative cache state — silently relocates).
///
/// Cases this gate INTENTIONALLY does not catch (`HOME=/nonexistent`,
/// `HOME=/dev/null`, `HOME=/proc`, `HOME=//`, `HOME=/.`, etc.) are
/// deferred to filesystem-level errors at use time — see the call
/// site's comment block for the per-shape rationale.
///
/// `pub(crate)` because both the kernel cache (this module) and
/// the model cache ([`crate::test_support::model::resolve_cache_root`])
/// reach this helper transitively through
/// [`resolve_cache_root_with_suffix`]. There is now exactly one
/// place that defines what "valid HOME for ktstr cache derivation"
/// means; a future change to the validation policy lands once and
/// flows through both cache flavors.
pub(crate) fn validate_home_for_cache() -> anyhow::Result<PathBuf> {
    // Distinguish unset (`Err(NotPresent)` — HOME never assigned in
    // the process environment) from empty (`Ok("")` — HOME assigned
    // but to the empty string) so the operator's diagnostic names
    // the actual misconfiguration shape. The two failure modes
    // arise from different bugs in practice:
    //   - Unset: container init dropped HOME (`docker run` without
    //     `-e HOME=...`, systemd unit without `Environment=HOME=...`).
    //   - Empty: a Dockerfile `ENV HOME=` line, a shell rc that
    //     accidentally `unset`s HOME via assignment-expansion (e.g.
    //     `export HOME=$HOME_OVERRIDE` when `HOME_OVERRIDE` is unset
    //     under `set -u` semantics inverted).
    // Non-UTF-8 HOME (`Err(NotUnicode)`) is treated as unset for
    // diagnostic purposes — the operator's remediation is the same
    // (set KTSTR_CACHE_DIR / XDG_CACHE_HOME), and naming the raw
    // bytes here would surface environment-leakage in the error.
    let home = match std::env::var("HOME") {
        Ok(v) if !v.is_empty() => v,
        Ok(_) => {
            // Empty assignment.
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
            // Unset (NotPresent) or non-UTF-8 (NotUnicode) — both
            // surface as "no usable HOME for derivation" with the
            // same remediation.
            anyhow::bail!(
                "HOME is unset; cannot resolve cache directory. \
                 The container init or login shell did not assign HOME — set \
                 it to an absolute path, or set KTSTR_CACHE_DIR to an absolute \
                 path (e.g. /tmp/ktstr-cache) or XDG_CACHE_HOME to specify a \
                 cache location explicitly."
            );
        }
    };
    // 2. Literal `/`: a process-environment artifact (root user with
    //    no home dir, container init that didn't set HOME).
    //    `PathBuf::from("/").join(".cache")` yields `/.cache` —
    //    statvfs reports the root filesystem's free space, not a
    //    usable user-cache filesystem; cache writes also escalate
    //    to root-fs writes.
    if home == "/" {
        anyhow::bail!(
            "HOME is `/`; the resulting cache path /.cache/ktstr would alias the \
             root filesystem rather than naming a user cache. This usually means \
             the process inherited HOME from a container init or root login that \
             did not set a real home. Set KTSTR_CACHE_DIR to an absolute path \
             (e.g. /tmp/ktstr-cache) or XDG_CACHE_HOME to bypass HOME entirely."
        );
    }
    // 3. Relative path: `PathBuf::from("relative").join(".cache")`
    //    yields `relative/.cache` resolved against CWD at every
    //    call. Cache state would silently relocate as the operator
    //    moves between directories — a usability nightmare worse
    //    than the deferred-error case. POSIX HOME is documented as
    //    an absolute pathname.
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
///
/// Thin wrapper over [`resolve_cache_root_with_suffix`] with the
/// `"kernels"` suffix. The model cache uses the same helper with
/// the `"models"` suffix from
/// [`crate::test_support::model::resolve_cache_root`]; both share the
/// `KTSTR_CACHE_DIR` / `XDG_CACHE_HOME` / `HOME` cascade verbatim.
///
/// Does not create the directory -- the caller is responsible for
/// ensuring it exists.
fn resolve_cache_root() -> anyhow::Result<PathBuf> {
    resolve_cache_root_with_suffix("kernels")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Decode an ELF section `sh_type` integer to its `SHT_*` constant
    /// name. Strip-helper assertions embed the decoded name alongside
    /// the raw integer so a failing diagnostic like "left: 8 right: 1"
    /// reads as "sh_type=8 (SHT_NOBITS)" / "sh_type=1 (SHT_PROGBITS)"
    /// — immediately actionable instead of requiring the reader to
    /// look up the ELF spec table.
    fn sh_type_name(t: u32) -> &'static str {
        use goblin::elf::section_header::{
            SHT_DYNAMIC, SHT_DYNSYM, SHT_HASH, SHT_NOBITS, SHT_NOTE, SHT_NULL, SHT_PROGBITS,
            SHT_REL, SHT_RELA, SHT_SHLIB, SHT_STRTAB, SHT_SYMTAB,
        };
        match t {
            SHT_NULL => "SHT_NULL",
            SHT_PROGBITS => "SHT_PROGBITS",
            SHT_SYMTAB => "SHT_SYMTAB",
            SHT_STRTAB => "SHT_STRTAB",
            SHT_RELA => "SHT_RELA",
            SHT_HASH => "SHT_HASH",
            SHT_DYNAMIC => "SHT_DYNAMIC",
            SHT_NOTE => "SHT_NOTE",
            SHT_NOBITS => "SHT_NOBITS",
            SHT_REL => "SHT_REL",
            SHT_SHLIB => "SHT_SHLIB",
            SHT_DYNSYM => "SHT_DYNSYM",
            _ => "SHT_UNKNOWN",
        }
    }

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
            vmlinux_stripped: false,
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
    /// every `neutralize_relocs` test shares this base shape.
    /// Callers that need relocation sections (with or without
    /// `SHF_ALLOC`) add them on top of the returned object before
    /// calling `.write()`.
    ///
    /// `arch` selects the ELF class: `Architecture::X86_64` yields
    /// ELF64 (8-byte anchor symbol), `Architecture::I386` yields
    /// ELF32 (4-byte anchor symbol). The anchor-symbol size is the
    /// only shape difference between the two classes at this
    /// fixture level; everything downstream (section headers, the
    /// `is_reloc` predicate under test) is driven by the
    /// ELF32/ELF64 split `object::write` performs based on `arch`.
    fn build_base_elf_with_text_symbol(
        arch: object::Architecture,
    ) -> object::write::Object<'static> {
        use object::write;
        let sym_size = match arch {
            object::Architecture::X86_64 => 8,
            object::Architecture::I386 => 4,
            other => panic!(
                "build_base_elf_with_text_symbol: unsupported arch {other:?}; supported: X86_64, I386",
            ),
        };
        let mut obj =
            write::Object::new(object::BinaryFormat::Elf, arch, object::Endianness::Little);
        let text_id = obj.add_section(Vec::new(), b".text".to_vec(), object::SectionKind::Text);
        obj.append_section_data(text_id, &[0xCC; 64], 1);
        let _ = obj.add_symbol(write::Symbol {
            name: b"test_text_symbol".to_vec(),
            value: 0x0,
            size: sym_size,
            kind: object::SymbolKind::Data,
            scope: object::SymbolScope::Compilation,
            weak: false,
            section: write::SymbolSection::Section(text_id),
            flags: object::SymbolFlags::None,
        });
        obj
    }

    /// Regression pin for the explicit `other =>` arm in
    /// [`build_base_elf_with_text_symbol`]. Before the guard, an
    /// unsupported architecture silently fell through to `sym_size = 8`
    /// which is wrong for any future 32-bit arch (or any arch whose
    /// address width isn't 8 bytes). `Aarch64` is a supported object
    /// crate architecture that isn't on the helper's allow-list, so
    /// passing it triggers the panic and the `#[should_panic]`
    /// assertion confirms the guard fires.
    #[test]
    #[should_panic(expected = "unsupported arch")]
    fn build_base_elf_with_text_symbol_panics_on_unsupported_arch() {
        let _ = build_base_elf_with_text_symbol(object::Architecture::Aarch64);
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
        assert!(!parsed.vmlinux_stripped);
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
            vmlinux_stripped: false,
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
            vmlinux_stripped: true,
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
        assert!(parsed.vmlinux_stripped);
    }

    /// git_hash on KernelSource::Local is a plain Option<String> with
    /// no serde attributes — the compat shims (serde(default) +
    /// skip_serializing_if) were removed for pre-1.0, so `None`
    /// serializes as an explicit `null` key and deserialization
    /// accepts `null` back as `None`. This test pins only the
    /// None → null → None round trip; the absent-key branch is
    /// exercised separately by
    /// [`kernel_source_absent_option_keys_deserialize_as_none`].
    #[test]
    fn kernel_source_local_git_hash_serde_round_trip_none() {
        let src = KernelSource::Local {
            source_tree_path: Some(PathBuf::from("/tmp/linux")),
            git_hash: None,
        };
        let json = serde_json::to_string(&src).unwrap();
        assert!(
            json.contains(r#""git_hash":null"#),
            "git_hash=None must round-trip as explicit null, got {json}"
        );
        let parsed: KernelSource = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, KernelSource::Local { git_hash: None, .. }));
    }

    /// Pins the post-shim wire format: `Option` payload fields inside
    /// every [`KernelSource`] variant serialize as explicit `null`
    /// rather than being omitted. The `serde(default)` /
    /// `skip_serializing_if` compat shims were removed for pre-1.0;
    /// [`kernel_source_local_git_hash_serde_round_trip_none`] already
    /// covers the round-trip statement for the single Local.git_hash
    /// slot. This test extends that guarantee across every Option
    /// payload on both Git and Local so `cargo ktstr kernel list
    /// --json` consumers see stable key presence regardless of which
    /// optional values are set — absent keys would mean the emitted
    /// schema has silently regressed.
    #[test]
    fn kernel_source_option_fields_serialize_as_explicit_null() {
        let local = KernelSource::Local {
            source_tree_path: None,
            git_hash: None,
        };
        let local_json = serde_json::to_string(&local).unwrap();
        assert!(
            local_json.contains(r#""source_tree_path":null"#),
            "Local.source_tree_path=None must serialize as explicit null, got {local_json}"
        );
        assert!(
            local_json.contains(r#""git_hash":null"#),
            "Local.git_hash=None must serialize as explicit null, got {local_json}"
        );

        let git = KernelSource::Git {
            git_hash: None,
            git_ref: None,
        };
        let git_json = serde_json::to_string(&git).unwrap();
        assert!(
            git_json.contains(r#""git_hash":null"#),
            "Git.git_hash=None must serialize as explicit null, got {git_json}"
        );
        // The struct field is `git_ref` but `#[serde(rename = "ref")]`
        // renames the JSON key — check the renamed key, not the field.
        assert!(
            git_json.contains(r#""ref":null"#),
            "Git.git_ref=None must serialize as explicit null under the `ref` key, got {git_json}"
        );
    }

    /// Older `metadata.json` files written before `Option` fields
    /// were emitted as explicit `null` simply omit the keys. The
    /// [`KernelSource`] doc states absent `Option` keys must
    /// deserialize as `None` — cache-integrity enforcement rides on
    /// the required non-`Option` fields of [`KernelMetadata`], not
    /// on the optional payloads inside variants. Feed each variant
    /// a minimal JSON with every `Option` key omitted, deserialize,
    /// and assert the result carries `None` in every payload slot.
    #[test]
    fn kernel_source_absent_option_keys_deserialize_as_none() {
        // Git with both git_hash and ref omitted.
        let git_bare: KernelSource = serde_json::from_str(r#"{"type":"git"}"#)
            .expect("Git with absent Option keys must deserialize");
        assert!(matches!(
            git_bare,
            KernelSource::Git {
                git_hash: None,
                git_ref: None,
            }
        ));

        // Git with only git_hash present.
        let git_hash_only: KernelSource =
            serde_json::from_str(r#"{"type":"git","git_hash":"abc"}"#)
                .expect("Git with only git_hash must deserialize");
        assert!(matches!(
            git_hash_only,
            KernelSource::Git {
                git_hash: Some(ref h),
                git_ref: None,
            } if h == "abc"
        ));

        // Git with only ref present.
        let git_ref_only: KernelSource = serde_json::from_str(r#"{"type":"git","ref":"main"}"#)
            .expect("Git with only ref must deserialize");
        assert!(matches!(
            git_ref_only,
            KernelSource::Git {
                git_hash: None,
                git_ref: Some(ref r),
            } if r == "main"
        ));

        // Local with both source_tree_path and git_hash omitted.
        let local_bare: KernelSource = serde_json::from_str(r#"{"type":"local"}"#)
            .expect("Local with absent Option keys must deserialize");
        assert!(matches!(
            local_bare,
            KernelSource::Local {
                source_tree_path: None,
                git_hash: None,
            }
        ));

        // Local with only source_tree_path present.
        let local_path_only: KernelSource =
            serde_json::from_str(r#"{"type":"local","source_tree_path":"/tmp/linux"}"#)
                .expect("Local with only source_tree_path must deserialize");
        assert!(matches!(
            local_path_only,
            KernelSource::Local {
                source_tree_path: Some(ref p),
                git_hash: None,
            } if p.to_str() == Some("/tmp/linux")
        ));

        // Local with only git_hash present.
        let local_hash_only: KernelSource =
            serde_json::from_str(r#"{"type":"local","git_hash":"deadbeef"}"#)
                .expect("Local with only git_hash must deserialize");
        assert!(matches!(
            local_hash_only,
            KernelSource::Local {
                source_tree_path: None,
                git_hash: Some(ref h),
            } if h == "deadbeef"
        ));
    }

    #[test]
    fn kernel_source_serde_tagged_representation() {
        // Check the tagged JSON shape on each variant.
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
    fn cache_dir_default_root_returns_path() {
        // `lock_env()` serializes against every other env-touching
        // test in the crate (test_support/model.rs, test_helpers
        // siblings, the cache_resolve_root_* tests below). nextest
        // runs unit tests concurrently within a binary and
        // `std::env::set_var` is process-wide, so a sibling test
        // that mutates HOME / XDG_CACHE_HOME / KTSTR_CACHE_DIR
        // without the lock can race the save / mutate / restore
        // window of an `EnvVarGuard` here. Tester finding T1.
        let _lock = lock_env();
        let tmp = TempDir::new().unwrap();
        let _guard = EnvVarGuard::set("KTSTR_CACHE_DIR", tmp.path());
        let resolved = CacheDir::default_root().unwrap();
        assert_eq!(resolved, tmp.path());
        // Side-effect-free: calling default_root() must not create
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
        assert_eq!(
            reason, "metadata.json missing",
            "missing-metadata reason should be the exact missing-file label, got: {reason}",
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
    fn cache_dir_list_classifies_unreadable_metadata_as_corrupt() {
        // The `missing` and parse-family branches (schema drift,
        // malformed, truncated) of `read_metadata` are covered
        // elsewhere; the I/O-error branch — any `fs::read_to_string`
        // failure that is NOT `ErrorKind::NotFound` — is exercised
        // here. Forcing a non-ENOENT error without relying on
        // filesystem permissions (which vary across rootless
        // containers and CI sandboxes) is awkward, so we make
        // `metadata.json` a DIRECTORY: `read_to_string` then fails
        // with `EISDIR`, which `read_metadata` must map to the
        // `"metadata.json unreadable: "` prefix rather than the
        // missing or any parse-family label. This pins the
        // distinction so a future refactor that collapses the arms
        // back into a single generic "corrupt" reason breaks this
        // test before it ships a less-actionable diagnostic.
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
        // metadata.json exists but is not valid JSON at all (unbalanced
        // punctuation / stray characters). `read_metadata` must route
        // this through `serde_json::Error::classify() ==
        // Category::Syntax` to produce the
        // `"metadata.json malformed: {e}"` prefix — distinct from the
        // schema-drift prefix that fires when JSON parses but does
        // not match the `KernelMetadata` shape.
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
        // metadata.json is valid JSON but omits fields the current
        // `KernelMetadata` schema requires: `source`, `arch`,
        // `image_name`, `built_at`, `has_vmlinux`, and
        // `vmlinux_stripped`. These are non-`Option`,
        // non-`#[serde(default)]` fields, so `serde_json::from_str`
        // fails with `Category::Data` when they are absent. Note
        // `has_vmlinux: bool` and `vmlinux_stripped: bool` are
        // required even though they are not wrapped in `Option` — a
        // plain `bool` with no `#[serde(default)]` attribute must
        // still be present in the JSON payload. serde_json reports
        // the first missing required field in declaration order
        // (`source`), and `read_metadata` wraps it under the
        // schema-drift prefix so the user sees both the
        // classification ("schema drift") and
        // the specific missing field.
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
        // metadata.json ends mid-value: `{"source":` stops after the
        // colon with no value byte. serde_json surfaces this as
        // `Category::Eof`, which `read_metadata` wraps under the
        // `"metadata.json truncated: {e}"` prefix. Covers the Eof
        // branch of the classify() match — distinct from the schema-
        // drift (Data) and malformed (Syntax) branches exercised by
        // the sibling tests above.
        //
        // Typical real-world cause: a crashed `store()` whose atomic
        // rename never completed, leaving a partially-written
        // metadata.json in a surviving entry directory.
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

    /// Table-drive every prefix → `error_kind` classifier mapping.
    /// Pins each documented value independently so a regression in
    /// one arm surfaces with the specific prefix cited in the
    /// failure message, not as a blanket "classifier broken". The
    /// "unknown" fallback row is the safety net: a future producer
    /// prefix that falls through this table must surface as
    /// `"unknown"` to consumers rather than panic.
    #[test]
    fn classify_corrupt_reason_covers_every_documented_prefix() {
        let cases: &[(&str, &str)] = &[
            ("metadata.json missing", "missing"),
            (
                "metadata.json unreadable: Is a directory (os error 21)",
                "unreadable",
            ),
            (
                "metadata.json schema drift: missing field `source` at line 1 column 21",
                "schema_drift",
            ),
            (
                "metadata.json malformed: expected value at line 1 column 1",
                "malformed",
            ),
            (
                "metadata.json truncated: EOF while parsing a value at line 1 column 10",
                "truncated",
            ),
            (
                "metadata.json parse error: something unexpected",
                "parse_error",
            ),
            (
                "image file bzImage missing from entry directory",
                "image_missing",
            ),
            ("some future prefix nobody wrote yet", "unknown"),
        ];
        for (reason, expected) in cases {
            assert_eq!(
                classify_corrupt_reason(reason),
                *expected,
                "reason `{reason}` should classify as `{expected}`",
            );
        }
    }

    /// `ListedEntry::error_kind()` returns `None` on a Valid entry
    /// and the classifier result on a Corrupt entry. Pins the
    /// Valid → None contract so a consumer that dispatches on
    /// `error_kind().is_some()` can safely gate on the corrupt
    /// path.
    #[test]
    fn listed_entry_error_kind_dispatches_on_variant() {
        // Construct a Valid entry via the normal store path.
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");
        cache
            .store("valid-ek", &CacheArtifacts::new(&image), &meta)
            .unwrap();

        // And a Corrupt entry via a missing-metadata directory.
        let bad_dir = tmp.path().join("cache").join("corrupt-ek");
        fs::create_dir_all(&bad_dir).unwrap();

        let entries = cache.list().unwrap();
        assert_eq!(entries.len(), 2);
        let valid = entries
            .iter()
            .find(|e| e.key() == "valid-ek")
            .expect("valid entry must be listed");
        let corrupt = entries
            .iter()
            .find(|e| e.key() == "corrupt-ek")
            .expect("corrupt entry must be listed");
        assert_eq!(
            valid.error_kind(),
            None,
            "Valid entries must report no error_kind",
        );
        assert_eq!(
            corrupt.error_kind(),
            Some("missing"),
            "missing-metadata Corrupt entry must classify as `missing`",
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
        // `lock_env()` serializes against sibling env-touching tests
        // in test_support/model.rs and the cache_resolve_root_* group
        // below. See `cache_dir_default_root_returns_path` for the
        // long-form rationale (Tester finding T1).
        let _lock = lock_env();
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("custom-cache");
        // Temporarily set env var for this test.
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

    /// A HOME literal of `"/"` (legacy root convention, container
    /// init that forgot to override HOME) must NOT silently produce
    /// `/.cache/ktstr/kernels` — that path's statvfs reports the
    /// root filesystem's free space, which is typically a small
    /// constrained mount and never the user's intended cache
    /// location. Bail with a diagnostic that names the resulting
    /// junk path and points the operator at a remediation.
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

    /// A HOME literal of `""` (empty string) is just as broken as
    /// unset (`PathBuf::from("").join(".cache")` produces a
    /// relative `.cache` rooted at the process CWD instead of the
    /// user's home), but the diagnostic now distinguishes the two
    /// shapes: empty-string assignment hits the `Ok("")` arm of
    /// `validate_home_for_cache`, surfacing "HOME is set to the
    /// empty string" so an operator can identify a Dockerfile
    /// `ENV HOME=` or shell-rc `export HOME=` typo as the cause
    /// rather than a missing init-time assignment.
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

    /// A relative-path HOME (e.g. `HOME=relative/dir`) silently
    /// resolves the cache against CWD, which silently relocates
    /// the cache as the operator changes directories — a usability
    /// nightmare worse than a deferred error. Pin the explicit
    /// rejection so a regression that drops the absolute-path
    /// check surfaces here instead of as a hard-to-diagnose
    /// "cache contents disappeared" report from the operator.
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

    /// A bare-name HOME (no path separators at all, e.g. `HOME=tmp`)
    /// is also relative — `PathBuf::from("tmp").join(".cache")`
    /// yields `tmp/.cache` against CWD. Pin separately from the
    /// `relative/dir` case to confirm the absolute-path check
    /// isn't accidentally permissive on shapes that lack a `/`.
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

    /// Sanity check the happy path: an absolute HOME pointing at a
    /// real directory must resolve through the gate to the expected
    /// `$HOME/.cache/ktstr/kernels` path. Pins that the new
    /// validation does not over-reject — a regression that hardens
    /// the gate further (e.g. requires HOME to exist via metadata)
    /// would break this.
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

    /// A non-UTF-8 `KTSTR_CACHE_DIR` must fail fast with an
    /// actionable diagnostic rather than silently falling through to
    /// `$XDG_CACHE_HOME` / `$HOME`. Before the `NotUnicode` branch
    /// existed, `std::env::var` returned `Err` and the old `if let
    /// Ok(..)` guard dropped the override without a trace — an
    /// operator who set the variable would see ktstr write to a
    /// surprising directory under `$HOME` and have no clue why the
    /// override was ignored.
    ///
    /// `EnvVarGuard::set` accepts arbitrary `OsStr`, so the test can
    /// plant a lone 0xFF byte (valid on Unix filesystems, invalid as
    /// UTF-8) and observe the bail.
    #[test]
    #[cfg(unix)]
    fn cache_resolve_root_non_utf8_ktstr_cache_dir_bails() {
        // `lock_env()` for the same reason every other env-touching
        // cache.rs test holds it (Tester finding T1).
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

    // -- clean_orphaned_tmp_dirs unit tests --
    //
    // Parser/dispatcher coverage: the scan must remove directories
    // under `.tmp-{key}-{pid}` whose `{pid}` is verifiably dead,
    // must LEAVE malformed entries and non-`.tmp-` entries alone,
    // and must tolerate a nonexistent cache root.

    /// A `.tmp-{key}-{pid}` directory whose pid refers to a dead
    /// process is removed. Uses `libc::pid_t::MAX` — above
    /// `PID_MAX_LIMIT` (2^22), so no live process can ever claim it
    /// (same technique as `process_alive_nonexistent_pid` in
    /// scenario tests, removes the pid-reuse race from the test).
    #[test]
    fn clean_orphaned_tmp_dirs_removes_dead_pid_tempdir() {
        let tmp = TempDir::new().unwrap();
        let dead_pid = libc::pid_t::MAX;
        let orphan = tmp
            .path()
            .join(format!("{TMP_DIR_PREFIX}somekey-{dead_pid}"));
        std::fs::create_dir_all(&orphan).unwrap();
        // Plant a nested file so a regression that hand-rolled
        // `remove_dir` (non-recursive) instead of `remove_dir_all`
        // would fail with ENOTEMPTY and the dir would survive.
        std::fs::write(orphan.join("inner.txt"), b"data").unwrap();

        clean_orphaned_tmp_dirs(tmp.path()).unwrap();
        assert!(
            !orphan.exists(),
            "dead-pid tempdir must be removed by clean_orphaned_tmp_dirs",
        );
    }

    /// A `.tmp-{key}-{pid}` directory whose pid is LIVE (the test
    /// process itself) must be preserved. `kill(getpid(), None)`
    /// returns `Ok(())` inside `clean_orphaned_tmp_dirs`'s liveness
    /// probe, which routes to the `!dead` continue branch.
    #[test]
    fn clean_orphaned_tmp_dirs_preserves_live_pid_tempdir() {
        let tmp = TempDir::new().unwrap();
        let live_pid = unsafe { libc::getpid() };
        let keeper = tmp
            .path()
            .join(format!("{TMP_DIR_PREFIX}somekey-{live_pid}"));
        std::fs::create_dir_all(&keeper).unwrap();

        clean_orphaned_tmp_dirs(tmp.path()).unwrap();
        assert!(
            keeper.exists(),
            "live-pid tempdir must NOT be removed — its owner is still running",
        );
    }

    /// Entries whose suffix cannot be parsed as a pid (non-numeric
    /// or empty after the trailing `-`) must be left alone — they
    /// do not match our format and may belong to an unrelated
    /// tool. Covers the `rsplit_once` / `parse::<i32>` continue
    /// branches.
    #[test]
    fn clean_orphaned_tmp_dirs_leaves_malformed_suffix_alone() {
        let tmp = TempDir::new().unwrap();
        // Case A: non-numeric suffix.
        let nonnum = tmp.path().join(format!("{TMP_DIR_PREFIX}somekey-notapid"));
        std::fs::create_dir_all(&nonnum).unwrap();
        // Case B: empty suffix (name ends with `-`).
        let empty_suf = tmp.path().join(format!("{TMP_DIR_PREFIX}somekey-"));
        std::fs::create_dir_all(&empty_suf).unwrap();
        // Case C: no `-` at all after the prefix (rsplit_once
        // still finds the `-` inside the prefix itself, but
        // `.tmp` parses as non-numeric → continue).
        let no_dash = tmp.path().join(format!("{TMP_DIR_PREFIX}nokeyhere"));
        std::fs::create_dir_all(&no_dash).unwrap();

        clean_orphaned_tmp_dirs(tmp.path()).unwrap();
        assert!(nonnum.exists(), "non-numeric pid suffix must be left alone");
        assert!(empty_suf.exists(), "empty pid suffix must be left alone");
        assert!(no_dash.exists(), "no-pid-suffix entry must be left alone");
    }

    /// Directories that do not begin with [`TMP_DIR_PREFIX`] must
    /// never be touched. The cache root also holds real cache
    /// entries (hash-keyed directories), and an overbroad scan
    /// would wipe them out.
    #[test]
    fn clean_orphaned_tmp_dirs_leaves_unrelated_entries_alone() {
        let tmp = TempDir::new().unwrap();
        let real_entry = tmp.path().join("real-cache-entry");
        std::fs::create_dir_all(&real_entry).unwrap();
        let other = tmp.path().join("not-a-tempdir");
        std::fs::create_dir_all(&other).unwrap();

        clean_orphaned_tmp_dirs(tmp.path()).unwrap();
        assert!(
            real_entry.exists(),
            "unrelated cache entry must be preserved"
        );
        assert!(other.exists(), "unrelated directory must be preserved");
    }

    /// Non-UTF-8 filenames in the cache root must be skipped
    /// silently — they cannot be a `.tmp-{key}-{pid}` directory
    /// this module created (all our names are ASCII), and bailing
    /// on every stray non-UTF-8 entry would fail the whole cleanup
    /// pass.
    ///
    /// Unix-only because the byte-level name construction uses
    /// `OsStr::from_bytes`, which is Unix-only. Other platforms
    /// cannot produce a non-UTF-8 filesystem name from this test
    /// code.
    #[test]
    #[cfg(unix)]
    fn clean_orphaned_tmp_dirs_skips_non_utf8_names() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        let tmp = TempDir::new().unwrap();
        // Name that looks like a tempdir prefix but has a non-UTF-8
        // byte after. `into_string()` in the scan returns Err(_)
        // and the continue branch skips it.
        let mut bytes: Vec<u8> = b".tmp-".to_vec();
        bytes.push(0xFF);
        bytes.extend_from_slice(b"-123");
        let bad_name = OsStr::from_bytes(&bytes);
        let bad_path = tmp.path().join(bad_name);
        std::fs::create_dir(&bad_path).unwrap();

        clean_orphaned_tmp_dirs(tmp.path()).unwrap();
        assert!(
            bad_path.exists(),
            "non-UTF-8 entry must be left alone — the scan cannot \
             confirm it matches our format, so safe-default is skip",
        );
    }

    /// A nonexistent cache root returns `Ok(())` without error —
    /// called from the `store()` prologue, which may execute
    /// before any cache operation has created the directory.
    #[test]
    fn clean_orphaned_tmp_dirs_handles_missing_cache_root() {
        let tmp = TempDir::new().unwrap();
        let never_created = tmp.path().join("never-created");
        // `is_dir()` short-circuits to Ok(()) without read_dir.
        clean_orphaned_tmp_dirs(&never_created).unwrap();
    }

    /// Multi-entry mix: a DEAD-pid orphan and a LIVE-pid tempdir
    /// side by side — only the dead one is removed. Pins the
    /// per-entry classification logic against a regression that
    /// bailed on the first entry's liveness-probe error or
    /// short-circuited after the first successful remove.
    #[test]
    fn clean_orphaned_tmp_dirs_mixed_entries() {
        let tmp = TempDir::new().unwrap();
        let dead_pid = libc::pid_t::MAX;
        let live_pid = unsafe { libc::getpid() };
        let dead = tmp.path().join(format!("{TMP_DIR_PREFIX}a-{dead_pid}"));
        let live = tmp.path().join(format!("{TMP_DIR_PREFIX}b-{live_pid}"));
        let unrelated = tmp.path().join("c-regular-entry");
        std::fs::create_dir_all(&dead).unwrap();
        std::fs::create_dir_all(&live).unwrap();
        std::fs::create_dir_all(&unrelated).unwrap();

        clean_orphaned_tmp_dirs(tmp.path()).unwrap();
        assert!(!dead.exists(), "dead orphan must be removed");
        assert!(live.exists(), "live-pid entry must survive");
        assert!(unrelated.exists(), "unrelated entry must survive");
    }

    /// `pid == 0` suffix: `nix::sys::signal::kill(Pid::from_raw(0),
    /// None)` broadcasts a signal probe to the process group of the
    /// CURRENT process (POSIX semantics: signal pid 0 = self's
    /// pgrp). Under a running test process, that check almost
    /// always returns `Ok(())`, which the scan classifies as
    /// "alive" and LEAVES the entry in place. Pins the
    /// safe-default behavior: a `.tmp-key-0` orphan is not touched,
    /// even though "pid 0" is not a real process — because the
    /// liveness probe's answer is not specific enough to justify
    /// removal.
    #[test]
    fn clean_orphaned_tmp_dirs_preserves_pid_zero_suffix() {
        let tmp = TempDir::new().unwrap();
        let entry = tmp.path().join(format!("{TMP_DIR_PREFIX}somekey-0"));
        std::fs::create_dir_all(&entry).unwrap();
        clean_orphaned_tmp_dirs(tmp.path()).unwrap();
        assert!(
            entry.exists(),
            "pid=0 suffix must be preserved — kill(0, None) reports \
             the current process group as alive, safe-default is \
             skip rather than remove",
        );
    }

    /// "Negative pid suffix" unreachability pin. The parser uses
    /// `rsplit_once('-')` to extract the suffix AFTER the last `-`,
    /// which by construction never contains a `-` — so
    /// `parse::<i32>()` on the suffix can only produce a
    /// non-negative integer (or fail to parse). A filename like
    /// `.tmp-key--12345` rsplits into `(".tmp-key-", "12345")`:
    /// the suffix `"12345"` parses to pid 12345 (POSITIVE).
    ///
    /// This test documents the invariant and pins the observable
    /// behavior: under `.tmp-key--12345`, the pid parses as a
    /// real positive integer (12345), `kill(12345, None)` likely
    /// returns `Err(ESRCH)` on a fresh pid space, and the entry
    /// is REMOVED (not preserved). The test verifies the REMOVAL
    /// path under this input so a future refactor that changed
    /// `rsplit_once('-')` to `splitn(3, '-')` or a regex — which
    /// COULD emit a `-12345` suffix and open the negative-pid
    /// door — would change the observable behavior and trip this
    /// test's "entry must be gone" assertion.
    ///
    /// Note: the test's observable outcome depends on pid 12345
    /// NOT being alive on the host. If a coincidental live
    /// process happens to hold pid 12345, the entry would be
    /// preserved instead; accept the ≈1-in-N-pids risk
    /// (empirically negligible in CI / dev environments) rather
    /// than contort the test to force a guaranteed-dead pid (the
    /// existing `dead_pid = libc::pid_t::MAX` technique produces
    /// a suffix too large to demonstrate the `--` splitting
    /// behavior).
    #[test]
    fn clean_orphaned_tmp_dirs_double_dash_parses_as_positive_pid() {
        let tmp = TempDir::new().unwrap();
        // Name with a double-dash so the rsplit-once path produces
        // the suffix "12345" (no leading dash — rsplit_once
        // guarantees no delimiter in the suffix). A future regex
        // that emitted "-12345" would behave differently here.
        let entry = tmp.path().join(format!("{TMP_DIR_PREFIX}somekey--12345"));
        std::fs::create_dir_all(&entry).unwrap();
        clean_orphaned_tmp_dirs(tmp.path()).unwrap();

        // Whether the entry is removed depends on whether pid
        // 12345 is alive at test time. The invariant being pinned
        // is the parse direction (positive, not negative), which
        // is a prerequisite for either the remove or preserve
        // branch — a refactor to a negative-suffix parser would
        // land in the `kill(-12345, None)` broadcast probe
        // instead, which returns `Ok(())` and preserves
        // unconditionally. Testing both "parses as positive" AND
        // "either removed or preserved based on liveness" together
        // requires nothing stronger than a liveness check here.
        //
        // The TEST IS PRIMARILY A DOC — the comment above explains
        // the negative-pid unreachability. The assertion below
        // guards against the most likely concrete regression: a
        // regex that emits a `-N` suffix and thereby lands in the
        // broadcast-probe branch. Under that regression, `kill(-N,
        // None)` returns Ok and the entry is ALWAYS preserved;
        // this assertion is satisfied only if the current parse
        // direction (positive pid, real liveness probe) holds.
        //
        // Use `kill(12345, None)` here to decide what we expect:
        // if the pid is live, the entry is preserved; if dead,
        // removed. Either result confirms positive-pid parse.
        let pid_alive = matches!(
            nix::sys::signal::kill(nix::unistd::Pid::from_raw(12345), None),
            Ok(()),
        );
        if pid_alive {
            assert!(
                entry.exists(),
                "pid 12345 was alive at probe time → entry must be \
                 preserved; got: entry removed (regression?)",
            );
        } else {
            assert!(
                !entry.exists(),
                "pid 12345 was dead at probe time → entry must be \
                 removed (proves positive-pid parse). A regression to \
                 negative-pid parse would preserve unconditionally; \
                 entry still exists.",
            );
        }
    }

    /// Regular file entry (not a directory) whose name MATCHES the
    /// `.tmp-{key}-{pid}` pattern with a dead pid. `fs::remove_dir_all`
    /// on a regular file returns `ENOTDIR` / `NotADirectory`; the
    /// scan catches the error in its match arm, logs + continues,
    /// and the file stays in place. Pins that the scan does NOT
    /// fall through to `fs::remove_file` on type mismatch —
    /// quietly removing a file with a tempdir-shaped name could
    /// destroy state belonging to an unrelated tool that happened
    /// to pick a colliding name.
    #[test]
    fn clean_orphaned_tmp_dirs_leaves_regular_file_entry() {
        let tmp = TempDir::new().unwrap();
        let dead_pid = libc::pid_t::MAX;
        let file_entry = tmp
            .path()
            .join(format!("{TMP_DIR_PREFIX}fileshaped-{dead_pid}"));
        std::fs::write(&file_entry, b"not a directory").unwrap();
        clean_orphaned_tmp_dirs(tmp.path()).unwrap();
        assert!(
            file_entry.exists(),
            "regular file with tempdir-shaped name + dead pid must \
             NOT be removed — `remove_dir_all` errors on a file, \
             and the scan's error-tolerance contract leaves it",
        );
    }

    /// Symlink entry whose NAME matches the tempdir pattern but
    /// whose TARGET is an unrelated path outside the cache. The
    /// scan must not follow the symlink — following would risk
    /// `remove_dir_all` deleting the target's contents (the very
    /// bug that a production cache cleaner must never commit).
    ///
    /// Rust's `std::fs::remove_dir_all` on modern platforms uses
    /// `openat` + symlink-aware checks to refuse to follow
    /// symlinks; this test pins that guarantee against a regression
    /// that reached for `fs::remove_dir` (which follows) or hand-
    /// rolled a recursive walk that followed links.
    #[test]
    #[cfg(unix)]
    fn clean_orphaned_tmp_dirs_leaves_symlink_entry() {
        let tmp = TempDir::new().unwrap();

        // Create the real target directory OUTSIDE the cache root
        // — the test asserts the target's contents survive even
        // though the symlink shares the tempdir-like name + dead
        // pid.
        let target_root = TempDir::new().unwrap();
        let target_file = target_root.path().join("sentinel.txt");
        std::fs::write(&target_file, b"must-not-be-deleted").unwrap();

        let dead_pid = libc::pid_t::MAX;
        let symlink = tmp
            .path()
            .join(format!("{TMP_DIR_PREFIX}symkey-{dead_pid}"));
        std::os::unix::fs::symlink(target_root.path(), &symlink).unwrap();

        clean_orphaned_tmp_dirs(tmp.path()).unwrap();

        // Either the symlink itself was removed (modern
        // `remove_dir_all` removes the link without following) OR
        // the symlink stayed (older `remove_dir_all` that errored
        // on symlinks). The LOAD-BEARING invariant is that the
        // TARGET's contents survive — the test's safety guarantee
        // is "data outside the cache root is untouched", not
        // "the symlink entry itself must survive".
        assert!(
            target_file.exists(),
            "symlink target's contents must survive the scan — \
             following symlinks would delete unrelated state \
             outside the cache root, a critical security / data- \
             safety regression",
        );
        assert_eq!(
            std::fs::read(&target_file).unwrap(),
            b"must-not-be-deleted",
            "target file content must be unchanged",
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
        // set, calling lookup() and checking that when Some(entry)
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
        // Metadata records absence of vmlinux; vmlinux_stripped is
        // meaningless without a vmlinux but must still be false (the
        // strip pipeline never ran).
        assert!(!entry.metadata.has_vmlinux);
        assert!(!entry.metadata.vmlinux_stripped);
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
        assert!(
            entry.metadata.vmlinux_stripped,
            "strip-succeeds path must set vmlinux_stripped = true"
        );
    }

    #[test]
    fn cache_dir_store_falls_back_when_strip_fails() {
        // Unparseable vmlinux: strip errors, store() falls back to
        // copying the raw bytes. has_vmlinux stays true (so consumers
        // still see the sidecar) but vmlinux_stripped is false (so
        // consumers can tell the raw-fallback path ran).
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
            "raw-fallback path must set vmlinux_stripped = false so \
             `ktstr cache list --json` surfaces the strip failure"
        );
    }

    // -- should_warn_unstripped (pure decision logic driving
    //    `CacheDir::lookup`'s per-lookup "unstripped vmlinux" warning).

    /// Helper for the three `should_warn_unstripped` tests below:
    /// construct a synthetic [`CacheEntry`] with explicit
    /// `has_vmlinux` / `vmlinux_stripped` bits and the rest of the
    /// metadata filled in from [`KernelMetadata::new`]. The entry-dir
    /// path is never touched (the decision logic only reads the
    /// metadata bools), so a synthetic PathBuf is enough.
    fn make_warn_test_entry(has_vmlinux: bool, vmlinux_stripped: bool) -> CacheEntry {
        let mut meta = KernelMetadata::new(
            KernelSource::Tarball,
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

    /// An entry with a vmlinux that came from the raw-fallback path
    /// (strip failed at store time) MUST trigger the warning. The
    /// per-lookup eprintln is the operator's persistent signal to
    /// rebuild the cache.
    #[test]
    fn should_warn_unstripped_fires_when_vmlinux_present_and_unstripped() {
        let entry = make_warn_test_entry(true, false);
        assert!(
            should_warn_unstripped(&entry),
            "has_vmlinux=true + vmlinux_stripped=false must warn"
        );
    }

    /// An entry with a successfully-stripped vmlinux MUST NOT warn.
    /// This is the common case; warning here would be noise that
    /// operators learn to ignore, defeating the signal on the
    /// genuine failure case above.
    #[test]
    fn should_warn_unstripped_silent_when_vmlinux_stripped() {
        let entry = make_warn_test_entry(true, true);
        assert!(
            !should_warn_unstripped(&entry),
            "has_vmlinux=true + vmlinux_stripped=true must not warn"
        );
    }

    /// An entry with no vmlinux at all MUST NOT warn. The
    /// `vmlinux_stripped` bit is meaningless in that shape (always
    /// `false` by construction in [`CacheDir::store`]'s no-vmlinux
    /// branch) and warning would fire on every cache hit that simply
    /// did not cache a vmlinux — pure noise.
    #[test]
    fn should_warn_unstripped_silent_when_no_vmlinux() {
        let entry = make_warn_test_entry(false, false);
        assert!(
            !should_warn_unstripped(&entry),
            "has_vmlinux=false must not warn (no vmlinux to worry about)"
        );
    }

    #[test]
    fn cache_dir_store_preserves_original_vmlinux() {
        // strip_vmlinux_debug reads the source path; check the
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
        // Pre-kconfig-tracking cache entries (ktstr_kconfig_hash == None)
        // must surface as `Untracked`, not `Stale`. `find_kernel`'s
        // stale-filter at lib.rs treats `Untracked` as "keep" so legacy
        // entries remain usable — checked here so a regression that
        // conflates "no recorded hash" with "different hash" surfaces
        // at unit-test time.
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        // test_metadata seeds ktstr_kconfig_hash = Some("def456"); strip
        // it here to hit the None branch in kconfig_status.
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
        // `KconfigStatus::Stale { cached, current }` names the two
        // hashes for diagnostics: `cached` is what the entry recorded
        // at build time, `current` is what the caller is comparing
        // against. A swap would invert the "was / is" story in every
        // diagnostic message consuming these fields (e.g. `kernel list`
        // tags, future error-formatting code). Pin the mapping so a
        // refactor that swaps the two construction args breaks this
        // test before it ships a misleading diagnostic.
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
            vmlinux_stripped: true,
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
            vmlinux_stripped: false,
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
            vmlinux_stripped: true,
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

    /// Malformed `metadata.json` — present on disk but not valid
    /// [`KernelMetadata`] — must short-circuit to `None`. The
    /// `read_metadata(..).ok()?` guard in
    /// [`prefer_source_tree_for_dwarf`] converts the parse failure
    /// into `None` without bailing, so callers fall back to the
    /// cache directory for symbol-only lookup rather than having
    /// the DWARF path blow up on a corrupted entry.
    ///
    /// A regression that replaced the `.ok()?` with `.unwrap()`,
    /// `.expect(..)`, or an `anyhow::Result` propagation would
    /// break this test — preserving the "silent fallback" contract
    /// documented on the function.
    #[test]
    fn prefer_source_tree_metadata_parse_failure_returns_none() {
        let tmp = TempDir::new().unwrap();
        let cache_entry = tmp.path().join("cache");
        fs::create_dir_all(&cache_entry).unwrap();
        // Valid JSON shape but NOT a `KernelMetadata` — missing
        // every required field. serde_json::from_str errors with
        // "missing field", which `read_metadata` maps to
        // `Err(String)`, which `prefer_source_tree_for_dwarf`'s
        // `.ok()?` turns into `None`.
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

        // Completely invalid JSON (not parseable at the token level)
        // must also short-circuit. Covers serde's two distinct
        // error classes — tokenizer failure vs shape mismatch —
        // both of which map to `Err(String)` inside `read_metadata`.
        let other_entry = tmp.path().join("other");
        fs::create_dir_all(&other_entry).unwrap();
        fs::write(other_entry.join("metadata.json"), b"not json at all {{{").unwrap();
        assert_eq!(
            prefer_source_tree_for_dwarf(&other_entry),
            None,
            "unparseable metadata.json must short-circuit to None, not bail",
        );
    }

    /// Local-source cache entry whose `source_tree_path` is
    /// explicitly `None` short-circuits at the `let src_path =
    /// source_tree_path?;` line — no filesystem probe runs for the
    /// missing path. Pins the "tree location not recorded" branch
    /// documented on [`prefer_source_tree_for_dwarf`].
    ///
    /// Distinct from `prefer_source_tree_local_without_vmlinux_in_tree`:
    /// that test has `source_tree_path = Some(...)` but the
    /// filesystem lacks `vmlinux`, so the function reaches the
    /// `src_path.join("vmlinux").is_file()` check before returning
    /// None. This test short-circuits one step earlier, before any
    /// filesystem inspection — a regression that replaced the `?`
    /// with a `.unwrap_or_else(|| default_path)` or a fallback
    /// would break it.
    #[test]
    fn prefer_source_tree_local_with_none_source_tree_path_returns_none() {
        let tmp = TempDir::new().unwrap();
        let cache_entry = tmp.path().join("cache");
        fs::create_dir_all(&cache_entry).unwrap();

        let meta = KernelMetadata {
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
            has_vmlinux: true,
            vmlinux_stripped: true,
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
             to None at the `let src_path = source_tree_path?;` line \
             — no filesystem probe must run",
        );
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
                sh.sh_type,
                SHT_NOBITS,
                "fixture {name} must start non-SHT_NOBITS so the strip is observable; got sh_type={} ({})",
                sh.sh_type,
                sh_type_name(sh.sh_type),
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
                sh_type,
                SHT_NOBITS,
                "section {name} should be SHT_NOBITS after strip, got sh_type={sh_type} ({})",
                sh_type_name(sh_type),
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
        let processed = neutralize_relocs(&raw).unwrap();

        let stripped = strip_debug_prefix(&processed).unwrap();
        let elf = goblin::elf::Elf::parse(&stripped).unwrap();
        let names: Vec<&str> = elf
            .section_headers
            .iter()
            .filter_map(|s| elf.shdr_strtab.get_at(s.sh_name))
            .collect();

        // .debug_* sections deleted. The fallback also removes the
        // `.comment` section, but this fixture does not emit one, so
        // that branch of the delete set is not exercised here.
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

    /// `strip_debug_prefix`'s delete filter matches on six predicates:
    /// `name.starts_with(b".debug_")`, `name == b".comment"`,
    /// `name.starts_with(b".rela.")`, `name.starts_with(b".rel.")`,
    /// `name.starts_with(b".relr.")`, and `name.starts_with(b".crel.")`.
    /// The `.debug_*` branch is exercised by
    /// `strip_debug_prefix_removes_debug_and_preserves_rest` against
    /// the shared fixture; the four reloc-name prefix arms are
    /// exercised by `strip_debug_prefix_removes_reloc_prefix_sections`.
    /// This test covers the `.comment` branch against a focused
    /// fixture that specifically emits one — the shared fixture
    /// deliberately does not, to keep the keep-list assertions
    /// scoped.
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

        // `neutralize_relocs` is a no-op on this fixture (no
        // SHF_ALLOC relocation sections) — run it anyway so the test
        // exercises the exact input pipeline `strip_vmlinux_debug` uses.
        let processed = neutralize_relocs(&data).unwrap();
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

    /// `strip_debug_prefix` deletes reloc-named sections via the
    /// `.rela.`, `.rel.`, `.relr.`, and `.crel.` prefix arms so the
    /// fallback output doesn't carry the zero-size ghost headers that
    /// `neutralize_relocs` left behind. Exercise each prefix on a
    /// focused fixture — a real kernel vmlinux might carry only a
    /// subset, so the synthetic shape pins every arm.
    #[test]
    fn strip_debug_prefix_removes_reloc_prefix_sections() {
        use object::elf::{SHT_REL, SHT_RELA, SHT_RELR};

        // Base ELF with .text + anchor symbol so `.symtab`/`.strtab`
        // are present.
        let mut obj = build_base_elf_with_text_symbol(object::Architecture::X86_64);
        // One section per reloc-prefix arm. `.crel.*` uses SHT_CREL
        // below. Each carries a nonzero payload so `neutralize_relocs`
        // has observable work to do before the fallback runs.
        let rela_id = obj.add_section(
            Vec::new(),
            b".rela.text".to_vec(),
            object::SectionKind::Elf(SHT_RELA),
        );
        obj.append_section_data(rela_id, &[0xA5; 24], 1);
        let rel_id = obj.add_section(
            Vec::new(),
            b".rel.data".to_vec(),
            object::SectionKind::Elf(SHT_REL),
        );
        obj.append_section_data(rel_id, &[0xC7; 16], 1);
        let relr_id = obj.add_section(
            Vec::new(),
            b".relr.dyn".to_vec(),
            object::SectionKind::Elf(SHT_RELR),
        );
        obj.append_section_data(relr_id, &[0xD3; 16], 1);
        let crel_id = obj.add_section(
            Vec::new(),
            b".crel.text".to_vec(),
            object::SectionKind::Elf(object::elf::SHT_CREL),
        );
        obj.append_section_data(crel_id, &[0xE4; 8], 1);
        let data = obj.write().unwrap();

        // Positive control: every reloc-named section must exist
        // pre-strip; a silent rename by `object::write` would false-pass
        // the post-strip absence assertions below.
        let source_elf = goblin::elf::Elf::parse(&data).unwrap();
        let source_names: Vec<&str> = source_elf
            .section_headers
            .iter()
            .filter_map(|s| source_elf.shdr_strtab.get_at(s.sh_name))
            .collect();
        for name in [
            ".rela.text",
            ".rel.data",
            ".relr.dyn",
            ".crel.text",
            ".text",
        ] {
            assert!(
                source_names.contains(&name),
                "fixture missing expected section {name}; got {source_names:?}"
            );
        }

        let processed = neutralize_relocs(&data).unwrap();
        let stripped = strip_debug_prefix(&processed).unwrap();
        let elf = goblin::elf::Elf::parse(&stripped).unwrap();
        let names: Vec<&str> = elf
            .section_headers
            .iter()
            .filter_map(|s| elf.shdr_strtab.get_at(s.sh_name))
            .collect();

        // All four reloc-prefix arms deleted.
        for name in [".rela.text", ".rel.data", ".relr.dyn", ".crel.text"] {
            assert!(
                !names.contains(&name),
                "fallback must delete {name} (prefix arm), got sections {names:?}"
            );
        }
        // Non-reloc section survives — guards against an overly broad
        // filter that would drop e.g. `.text` on an unrelated name prefix.
        assert!(
            names.contains(&".text"),
            "fallback must preserve .text, got sections {names:?}"
        );
    }

    /// `neutralize_relocs` rewrites two section-header fields on every
    /// section whose `sh_type` is `SHT_REL`, `SHT_RELA`, `SHT_RELR`, or
    /// `SHT_CREL`, regardless of the `SHF_ALLOC` flag: `sh_type`
    /// becomes `SHT_PROGBITS` and `sh_size` becomes 0. Pin the
    /// observable invariants against a focused fixture:
    ///
    /// 1a. SHF_ALLOC + SHT_RELA section has sh_type rewritten to
    ///     SHT_PROGBITS and sh_size zeroed post-call.
    /// 1b. SHF_ALLOC + SHT_REL section has sh_type rewritten to
    ///     SHT_PROGBITS and sh_size zeroed post-call.
    /// 1c. Non-ALLOC SHT_RELA section has sh_type rewritten to
    ///     SHT_PROGBITS and sh_size zeroed post-call (the SHF_ALLOC
    ///     gate was dropped — aarch64 kernels emit non-alloc rela
    ///     sections whose byte ranges trip
    ///     `object::build::elf::Builder::read`).
    /// 1d. SHT_RELR section has sh_type rewritten to SHT_PROGBITS and
    ///     sh_size zeroed post-call (defense-in-depth for arm64
    ///     kernels with `CONFIG_PIE` + `CONFIG_RELR` that emit
    ///     `.relr.dyn`).
    /// 2. Non-RELA section (e.g. `.text`) has sh_type and sh_size
    ///    preserved (guards against an accidentally-broader filter).
    ///
    /// Also pins content preservation: `neutralize_relocs` only
    /// mutates the section HEADER's `sh_type` and `sh_size`, not the
    /// section's data bytes. Raw bytes at the original sh_offset must
    /// remain bit-identical post-call.
    #[test]
    fn neutralize_relocs_zeros_sh_size_of_every_reloc_section() {
        use object::elf::{SHF_ALLOC, SHT_REL, SHT_RELA, SHT_RELR};

        // Base ELF with .text + anchor symbol (so object::write
        // emits .symtab/.strtab). Reloc sections are added below.
        let mut obj = build_base_elf_with_text_symbol(object::Architecture::X86_64);
        // .rela.kaslr — SHT_RELA + SHF_ALLOC. Shape matches what
        // CONFIG_RELOCATABLE + CONFIG_RANDOMIZE_BASE kernels emit.
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
        // .rela.debug_info — SHT_RELA WITHOUT SHF_ALLOC. After the
        // SHF_ALLOC gate was dropped, this must also be zeroed — a
        // regression that re-added the gate would preserve sh_size
        // here and fail the new invariant 1c.
        let rdbg_id = obj.add_section(
            Vec::new(),
            b".rela.debug_info".to_vec(),
            object::SectionKind::Elf(SHT_RELA),
        );
        obj.append_section_data(rdbg_id, &[0xB6; 16], 1);
        // .relr.dyn — SHT_RELR + SHF_ALLOC. Defense-in-depth for
        // arm64 kernels that emit packed relative relocations.
        let relr_id = obj.add_section(
            Vec::new(),
            b".relr.dyn".to_vec(),
            object::SectionKind::Elf(SHT_RELR),
        );
        obj.append_section_data(relr_id, &[0xD3; 24], 1);
        obj.section_mut(relr_id).flags = object::SectionFlags::Elf {
            sh_flags: u64::from(SHF_ALLOC),
        };

        let data = obj.write().unwrap();

        // Positive-control the fixture: the five sections we assert on
        // must actually exist in the produced ELF with the expected
        // sh_type/sh_flags/sh_size. If `object::write` renamed or
        // reshaped one, the post-call assertions would false-pass.
        let pre_elf = goblin::elf::Elf::parse(&data).unwrap();
        let mut pre_kaslr = None;
        let mut pre_rel = None;
        let mut pre_rdbg = None;
        let mut pre_relr = None;
        let mut pre_text = None;
        for sh in pre_elf.section_headers.iter() {
            let name = pre_elf.shdr_strtab.get_at(sh.sh_name).unwrap_or("");
            match name {
                ".rela.kaslr" => pre_kaslr = Some(sh.clone()),
                ".rel.foo" => pre_rel = Some(sh.clone()),
                ".rela.debug_info" => pre_rdbg = Some(sh.clone()),
                ".relr.dyn" => pre_relr = Some(sh.clone()),
                ".text" => pre_text = Some(sh.clone()),
                _ => {}
            }
        }
        let pre_kaslr = pre_kaslr.expect("fixture must carry .rela.kaslr");
        let pre_rel = pre_rel.expect("fixture must carry .rel.foo");
        let pre_rdbg = pre_rdbg.expect("fixture must carry .rela.debug_info");
        let pre_relr = pre_relr.expect("fixture must carry .relr.dyn");
        let pre_text = pre_text.expect("fixture must carry .text");
        assert_eq!(
            pre_kaslr.sh_type,
            SHT_RELA,
            ".rela.kaslr sh_type must be SHT_RELA; got sh_type={} ({})",
            pre_kaslr.sh_type,
            sh_type_name(pre_kaslr.sh_type),
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
        assert_eq!(
            pre_rel.sh_type,
            SHT_REL,
            ".rel.foo sh_type must be SHT_REL; got sh_type={} ({})",
            pre_rel.sh_type,
            sh_type_name(pre_rel.sh_type),
        );
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
            pre_rdbg.sh_type,
            SHT_RELA,
            ".rela.debug_info sh_type must be SHT_RELA; got sh_type={} ({})",
            pre_rdbg.sh_type,
            sh_type_name(pre_rdbg.sh_type),
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
            pre_relr.sh_type,
            SHT_RELR,
            ".relr.dyn sh_type must be SHT_RELR (19); got sh_type={} ({})",
            pre_relr.sh_type,
            sh_type_name(pre_relr.sh_type),
        );
        assert_eq!(
            pre_relr.sh_size, 24,
            ".relr.dyn sh_size must match 24-byte payload"
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

        let processed = neutralize_relocs(&data).unwrap();
        assert_eq!(
            processed.len(),
            data.len(),
            "neutralize_relocs must not resize the ELF; only sh_size header fields are rewritten"
        );

        let post_elf = goblin::elf::Elf::parse(&processed).unwrap();
        let mut post_kaslr = None;
        let mut post_rel = None;
        let mut post_rdbg = None;
        let mut post_relr = None;
        let mut post_text = None;
        for sh in post_elf.section_headers.iter() {
            let name = post_elf.shdr_strtab.get_at(sh.sh_name).unwrap_or("");
            match name {
                ".rela.kaslr" => post_kaslr = Some(sh.clone()),
                ".rel.foo" => post_rel = Some(sh.clone()),
                ".rela.debug_info" => post_rdbg = Some(sh.clone()),
                ".relr.dyn" => post_relr = Some(sh.clone()),
                ".text" => post_text = Some(sh.clone()),
                _ => {}
            }
        }
        let post_kaslr = post_kaslr.expect(".rela.kaslr must survive");
        let post_rel = post_rel.expect(".rel.foo must survive");
        let post_rdbg = post_rdbg.expect(".rela.debug_info must survive");
        let post_relr = post_relr.expect(".relr.dyn must survive");
        let post_text = post_text.expect(".text must survive");

        // Invariant 1a: SHF_ALLOC + SHT_RELA section has sh_size zeroed.
        assert_eq!(
            post_kaslr.sh_size, 0,
            ".rela.kaslr sh_size must be zeroed; got {}",
            post_kaslr.sh_size
        );
        // Invariant 1b: SHF_ALLOC + SHT_REL section has sh_size zeroed
        // (the SHT_REL arm of the filter).
        assert_eq!(
            post_rel.sh_size, 0,
            ".rel.foo sh_size must be zeroed; got {}",
            post_rel.sh_size
        );
        // Invariant 1c: Non-ALLOC SHT_RELA section ALSO zeroed (the
        // SHF_ALLOC gate was dropped so aarch64 non-alloc rela
        // sections get neutralized).
        assert_eq!(
            post_rdbg.sh_size, 0,
            ".rela.debug_info sh_size must be zeroed (SHF_ALLOC gate dropped); got {}",
            post_rdbg.sh_size
        );
        // Invariant 1d: SHT_RELR section zeroed (defense-in-depth
        // for arm64 CONFIG_RELR kernels).
        assert_eq!(
            post_relr.sh_size, 0,
            ".relr.dyn sh_size must be zeroed (SHT_RELR match arm); got {}",
            post_relr.sh_size
        );
        // Invariant 2: Non-RELA section preserved.
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

        // sh_offset and sh_flags are preserved; sh_type is rewritten
        // to SHT_PROGBITS so the Builder reads the section via the
        // opaque-data arm instead of the rel/rela parse arms that
        // break on zero-length slices with align != 1.
        assert_eq!(
            post_kaslr.sh_offset, pre_kaslr.sh_offset,
            "sh_offset must be preserved"
        );
        assert_eq!(
            post_kaslr.sh_type,
            object::elf::SHT_PROGBITS,
            "sh_type must be rewritten to SHT_PROGBITS; got sh_type={} ({})",
            post_kaslr.sh_type,
            sh_type_name(post_kaslr.sh_type),
        );
        assert_eq!(
            post_kaslr.sh_flags, pre_kaslr.sh_flags,
            "sh_flags must be preserved"
        );
        // The sibling reloc sections should also be re-typed to
        // SHT_PROGBITS (the fn applies sh_type rewrite to every
        // matching section, not just the first).
        assert_eq!(
            post_rel.sh_type,
            object::elf::SHT_PROGBITS,
            ".rel.foo sh_type must be SHT_PROGBITS"
        );
        assert_eq!(
            post_rdbg.sh_type,
            object::elf::SHT_PROGBITS,
            ".rela.debug_info sh_type must be SHT_PROGBITS"
        );
        assert_eq!(
            post_relr.sh_type,
            object::elf::SHT_PROGBITS,
            ".relr.dyn sh_type must be SHT_PROGBITS"
        );
    }

    /// For ELFs that carry no relocation sections at all,
    /// `neutralize_relocs` returns an unchanged copy —
    /// documented as the "no-op" branch in the fn docstring.
    #[test]
    fn neutralize_relocs_noop_when_no_reloc_sections() {
        // Base ELF carries only .text + anchor symbol — no reloc
        // sections at all, so the filter matches nothing.
        let data = build_base_elf_with_text_symbol(object::Architecture::X86_64)
            .write()
            .unwrap();

        let processed = neutralize_relocs(&data).unwrap();
        assert_eq!(
            processed, data,
            "neutralize_relocs must be a byte-identity no-op when no reloc sections are present"
        );
    }

    /// `neutralize_relocs` must be byte-identity idempotent:
    /// `f(f(x)) == f(x)`. The production filter inside
    /// [`neutralize_relocs`] keys on `sh_type` — which IS rewritten
    /// (to `SHT_PROGBITS`) on matching sections. Idempotence still
    /// holds because after the first pass the neutralized sections
    /// no longer match the `is_reloc` predicate (sh_type is now
    /// `SHT_PROGBITS`, not one of `SHT_REL`/`SHT_RELA`/`SHT_RELR`/
    /// `SHT_CREL`), so the second pass walks every section without
    /// touching any header field and the output is byte-identical to
    /// the first-pass output.
    ///
    /// Guards against a future mutation that rewrites sh_type to a
    /// still-matched value (e.g. flipping `SHT_REL` to `SHT_RELA` —
    /// both match `is_reloc`, which would make the second pass
    /// re-neutralize to yet another sh_type value and break
    /// idempotence).
    ///
    /// Uses the same multi-section fixture as
    /// `neutralize_relocs_zeros_sh_size_of_every_reloc_section`
    /// so every reloc-type arm of the filter (SHT_RELA with and
    /// without SHF_ALLOC, SHT_REL) and the non-RELA negative control
    /// re-walk on the second pass.
    #[test]
    fn neutralize_relocs_is_idempotent() {
        use object::elf::{SHF_ALLOC, SHT_REL, SHT_RELA};

        // Base .text + anchor symbol; the reloc sections added below
        // intentionally mirror the sibling zeros-every-reloc test's
        // fixture so the filter re-walks on the second pass.
        let mut obj = build_base_elf_with_text_symbol(object::Architecture::X86_64);
        // .rela.kaslr — SHT_RELA + SHF_ALLOC.
        let kaslr_id = obj.add_section(
            Vec::new(),
            b".rela.kaslr".to_vec(),
            object::SectionKind::Elf(SHT_RELA),
        );
        obj.append_section_data(kaslr_id, &[0xA5; 32], 1);
        obj.section_mut(kaslr_id).flags = object::SectionFlags::Elf {
            sh_flags: u64::from(SHF_ALLOC),
        };
        // .rel.foo — SHT_REL + SHF_ALLOC. Exercises the SHT_REL arm
        // of the filter so a regression that special-cased only
        // SHT_RELA on re-entry would surface here.
        let rel_id = obj.add_section(
            Vec::new(),
            b".rel.foo".to_vec(),
            object::SectionKind::Elf(SHT_REL),
        );
        obj.append_section_data(rel_id, &[0xC7; 24], 1);
        obj.section_mut(rel_id).flags = object::SectionFlags::Elf {
            sh_flags: u64::from(SHF_ALLOC),
        };
        // .rela.debug_info — SHT_RELA without SHF_ALLOC. After the
        // SHF_ALLOC gate was dropped, this gets neutralized too —
        // but must re-neutralize to byte-identical bytes on the
        // second pass.
        let rdbg_id = obj.add_section(
            Vec::new(),
            b".rela.debug_info".to_vec(),
            object::SectionKind::Elf(SHT_RELA),
        );
        obj.append_section_data(rdbg_id, &[0xB6; 16], 1);
        // flags left as SectionFlags::None — no SHF_ALLOC.

        let data = obj.write().unwrap();

        let first_pass = neutralize_relocs(&data).unwrap();
        let second_pass = neutralize_relocs(&first_pass).unwrap();

        // Non-vacuous guard: the first call must actually modify bytes
        // on this fixture (which carries reloc sections); a degenerate
        // no-op implementation of `neutralize_relocs` would
        // trivially satisfy idempotence and must not pass.
        assert_ne!(
            first_pass, data,
            "first call must modify bytes on a fixture with reloc sections; \
             if this fails, neutralize_relocs is a no-op"
        );

        // Primary idempotence assertion: byte equality between passes.
        assert_eq!(
            second_pass, first_pass,
            "neutralize_relocs must be idempotent: a second pass over its own output produces byte-identical bytes"
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

        // All reloc sections stay zeroed on the second pass,
        // regardless of SHF_ALLOC.
        assert_eq!(
            post_kaslr.sh_size, 0,
            ".rela.kaslr sh_size must remain zero after the second pass"
        );
        assert_eq!(
            post_rel.sh_size, 0,
            ".rel.foo sh_size must remain zero after the second pass"
        );
        assert_eq!(
            post_rdbg.sh_size, 0,
            ".rela.debug_info sh_size must remain zero after the second pass (SHF_ALLOC gate dropped)"
        );

        // SHF_ALLOC flag must still be set on the ALLOC sections —
        // the function touches sh_type and sh_size, never sh_flags.
        // The non-ALLOC `.rela.debug_info` likewise retains its
        // (cleared) SHF_ALLOC bit.
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
        assert_eq!(
            post_rdbg.sh_flags & u64::from(SHF_ALLOC),
            0,
            ".rela.debug_info must retain its (cleared) SHF_ALLOC flag across both passes; got sh_flags={:#x}",
            post_rdbg.sh_flags
        );
    }

    /// `neutralize_relocs` fails loudly when fed bytes that do
    /// not parse as an ELF — the goblin parse returns Err and the
    /// function wraps it in an `anyhow::anyhow!("parse vmlinux ELF
    /// for preprocess: {e}")`. Pin only the stable "parse vmlinux ELF
    /// for preprocess" wrapper in `neutralize_relocs`; the
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
    fn neutralize_relocs_rejects_invalid_elf() {
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
            let err = neutralize_relocs(input).unwrap_err();
            let rendered = format!("{err:#}");
            assert!(
                rendered.contains("parse vmlinux ELF for preprocess"),
                "[{label}] expected error context to name the ELF parse step; got: {rendered}"
            );
        }
    }

    /// ELF32 counterpart of
    /// [`neutralize_relocs_zeros_sh_size_of_every_reloc_section`].
    ///
    /// `neutralize_relocs` dispatches on `elf.is_64` at the
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
    /// `is_64 == false`, driving `neutralize_relocs` through the
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
    fn neutralize_relocs_zeros_sh_size_in_elf32_fixture() {
        use object::elf::{SHF_ALLOC, SHT_REL, SHT_RELA};

        // Base shape — .text + test_text_symbol with ELF32-sized
        // anchor — is shared with the ELF64 fixtures via the
        // helper. Passing `Architecture::I386` flips the ELF class
        // (is_64 false) and downgrades the symbol size to 4 bytes.
        let mut obj = build_base_elf_with_text_symbol(object::Architecture::I386);
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
        // .rel.foo — SHT_REL + SHF_ALLOC. Exercises the `SHT_REL` arm
        // of the `is_reloc` match on the ELF32 code path. A regression
        // that dropped SHT_REL from the filter on the 32-bit path
        // would leave this section's sh_size unchanged and trip the
        // post-call assertion below.
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
            pre_kaslr.sh_type,
            SHT_RELA,
            ".rela.kaslr sh_type must be SHT_RELA; got sh_type={} ({})",
            pre_kaslr.sh_type,
            sh_type_name(pre_kaslr.sh_type),
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
        assert_eq!(
            pre_rel.sh_type,
            SHT_REL,
            ".rel.foo sh_type must be SHT_REL; got sh_type={} ({})",
            pre_rel.sh_type,
            sh_type_name(pre_rel.sh_type),
        );
        assert!(
            pre_rel.sh_flags & u64::from(SHF_ALLOC) != 0,
            ".rel.foo must carry SHF_ALLOC; got sh_flags={:#x}",
            pre_rel.sh_flags
        );
        assert_eq!(
            pre_rel.sh_size, 12,
            ".rel.foo sh_size must match 12-byte payload pre-call"
        );

        let processed = neutralize_relocs(&data).unwrap();
        assert_eq!(
            processed.len(),
            data.len(),
            "neutralize_relocs must not resize the ELF32 buffer"
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

    /// ELF32 counterpart of
    /// [`neutralize_relocs_noop_when_no_reloc_sections`].
    ///
    /// When the input carries no reloc sections, the ELF32 code path
    /// in `neutralize_relocs` must return a byte-identity copy
    /// of the input — same invariant as ELF64, but exercised through
    /// the `(20, 4)` offset/width branch. A regression that filled
    /// zeros even on the "no match" path, or mis-read the section
    /// header count / size on 32-bit inputs, would break byte-identity
    /// here without tripping the ELF64 sibling test.
    #[test]
    fn neutralize_relocs_noop_when_no_reloc_sections_elf32() {
        use object::write;

        let mut obj = write::Object::new(
            object::BinaryFormat::Elf,
            object::Architecture::I386,
            object::Endianness::Little,
        );
        // .text + symbol mirror the sibling fixture so object::write
        // emits a valid ELF32 with the same structural sections but
        // zero reloc entries.
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
        let data = obj.write().unwrap();

        // Positive-control: the fixture must parse as ELF32
        // (is_64 == false) so the no-match path through the
        // `(20, 4)` branch is what gets exercised. A future object
        // change that remapped I386 to ELF64 would turn this into a
        // duplicate of the ELF64 sibling without visible failure.
        let pre_elf = goblin::elf::Elf::parse(&data).unwrap();
        assert!(
            !pre_elf.is_64,
            "fixture must produce ELF32 (is_64 == false) to exercise the (20, 4) branch",
        );

        let processed = neutralize_relocs(&data).unwrap();
        assert_eq!(
            processed, data,
            "neutralize_relocs must be byte-identity on ELF32 when no reloc sections are present",
        );
    }

    /// ELF32 counterpart of
    /// [`neutralize_relocs_is_idempotent`].
    ///
    /// Idempotence (`f(f(x)) == f(x)`) must hold through the ELF32
    /// `(20, 4)` branch of `neutralize_relocs`. The ELF64
    /// sibling covers the `(32, 8)` branch; pinning both prevents a
    /// future offset-width mismatch where e.g. the second pass on
    /// ELF32 reads sh_size through an ELF64 offset and silently
    /// tripped idempotence on 32-bit inputs.
    ///
    /// Uses the same SHT_RELA+ALLOC / SHT_REL+ALLOC / SHT_RELA-no-
    /// ALLOC section mix as the ELF32 zeros fixture so the SHT_REL
    /// and SHT_RELA arms of the `is_reloc` match re-walk on the
    /// second pass. A no-match section is present to rule out a
    /// degenerate "zero every sh_size" implementation.
    #[test]
    fn neutralize_relocs_is_idempotent_elf32() {
        use object::elf::{SHF_ALLOC, SHT_REL, SHT_RELA};
        use object::write;

        let mut obj = write::Object::new(
            object::BinaryFormat::Elf,
            object::Architecture::I386,
            object::Endianness::Little,
        );
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
        // .rela.kaslr — SHT_RELA + SHF_ALLOC.
        let kaslr_id = obj.add_section(
            Vec::new(),
            b".rela.kaslr".to_vec(),
            object::SectionKind::Elf(SHT_RELA),
        );
        obj.append_section_data(kaslr_id, &[0xA5; 16], 1);
        obj.section_mut(kaslr_id).flags = object::SectionFlags::Elf {
            sh_flags: u64::from(SHF_ALLOC),
        };
        // .rel.foo — SHT_REL + SHF_ALLOC. Second filter arm.
        let rel_id = obj.add_section(
            Vec::new(),
            b".rel.foo".to_vec(),
            object::SectionKind::Elf(SHT_REL),
        );
        obj.append_section_data(rel_id, &[0xC7; 12], 1);
        obj.section_mut(rel_id).flags = object::SectionFlags::Elf {
            sh_flags: u64::from(SHF_ALLOC),
        };
        // .rela.debug_info — SHT_RELA without SHF_ALLOC. The
        // SHF_ALLOC gate was dropped, so this also gets neutralized
        // on both passes — a regression that re-added the gate would
        // leave sh_size preserved here but still satisfy idempotence,
        // so the post-second-pass assertions below pin the neutralized
        // value directly.
        let rdbg_id = obj.add_section(
            Vec::new(),
            b".rela.debug_info".to_vec(),
            object::SectionKind::Elf(SHT_RELA),
        );
        obj.append_section_data(rdbg_id, &[0xB6; 8], 1);

        let data = obj.write().unwrap();

        // Positive-control ELF32: any post-parse assertion depends on
        // this; a silent promotion to ELF64 would make the idempotence
        // check run through the (32, 8) branch instead.
        assert!(
            !goblin::elf::Elf::parse(&data).unwrap().is_64,
            "fixture must be ELF32 to exercise the (20, 4) idempotence path",
        );

        let first_pass = neutralize_relocs(&data).unwrap();
        let second_pass = neutralize_relocs(&first_pass).unwrap();

        // Non-vacuous guard: first pass must actually rewrite bytes on
        // this fixture. Without this the test could false-pass on a
        // degenerate no-op implementation that trivially satisfies
        // idempotence.
        assert_ne!(
            first_pass, data,
            "first pass must rewrite sh_size on ELF32 reloc sections",
        );
        assert_eq!(
            second_pass, first_pass,
            "neutralize_relocs must be byte-identity idempotent on ELF32",
        );

        // Pin the post-second-pass sh_size values directly so a
        // regression that re-added the SHF_ALLOC gate (leaving
        // `.rela.debug_info` un-zeroed) surfaces even though
        // idempotence alone would still hold.
        let post_elf = goblin::elf::Elf::parse(&second_pass).unwrap();
        for name in [".rela.kaslr", ".rel.foo", ".rela.debug_info"] {
            let sh = post_elf
                .section_headers
                .iter()
                .find(|sh| post_elf.shdr_strtab.get_at(sh.sh_name) == Some(name))
                .unwrap_or_else(|| panic!("{name} must survive second pass"));
            assert_eq!(
                sh.sh_size, 0,
                "ELF32 {name} sh_size must be zeroed after both passes (SHF_ALLOC gate dropped)"
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

    /// Build an ELF on disk matching `create_strip_test_fixture`'s
    /// shape (keep-list sections, code, debug, data sidecars, and a
    /// symtab anchor) plus one extra section provided by the caller.
    /// Returns the path.
    ///
    /// The helper is generic over the per-test extra section so each
    /// of the four end-to-end pipeline tests can focus on one failure
    /// mode (non-alloc SHT_RELA with invalid entries, non-alloc
    /// SHT_RELA with sh_size past EOF, SHT_RELR with sh_size past
    /// EOF) while sharing the rest of the fixture shape.
    ///
    /// The `mutate_header` closure receives a goblin-parsed view of
    /// the produced ELF plus a mutable byte buffer and can rewrite
    /// the section header of the extra section in-place. Tests use it
    /// to push `sh_size` or `sh_offset` past the file end — a direct
    /// rewrite is safer than trying to coax `object::write` into
    /// emitting malformed headers.
    fn build_reloc_fixture(
        dir: &Path,
        extra_section_name: &[u8],
        extra_section_sh_type: u32,
        extra_section_data: &[u8],
        mutate_header: impl FnOnce(&mut [u8]),
    ) -> PathBuf {
        use object::write;

        let mut obj = write::Object::new(
            object::BinaryFormat::Elf,
            object::Architecture::X86_64,
            object::Endianness::Little,
        );
        // .text anchors the symtab.
        let text_id = obj.add_section(Vec::new(), b".text".to_vec(), object::SectionKind::Text);
        obj.append_section_data(text_id, &[0xCC; 64], 16);
        let _ = obj.add_symbol(write::Symbol {
            name: b"pipeline_anchor".to_vec(),
            value: 0x10,
            size: 8,
            kind: object::SymbolKind::Data,
            scope: object::SymbolScope::Compilation,
            weak: false,
            section: write::SymbolSection::Section(text_id),
            flags: object::SymbolFlags::None,
        });
        // .BTF — kept by probe BTF keep-list.
        let btf_id = obj.add_section(Vec::new(), b".BTF".to_vec(), object::SectionKind::Other);
        obj.append_section_data(btf_id, &[0x42; 128], 1);
        // .rodata — kept by monitor CONFIG_IKCONFIG keep-list.
        let rodata_id = obj.add_section(
            Vec::new(),
            b".rodata".to_vec(),
            object::SectionKind::ReadOnlyData,
        );
        obj.append_section_data(rodata_id, &[0xAA; 256], 1);
        // Extra caller-provided section (a reloc section in the tests
        // below). Flags left as SectionFlags::None so it is non-alloc
        // — exercising the SHF_ALLOC-gate-drop path.
        let extra_id = obj.add_section(
            Vec::new(),
            extra_section_name.to_vec(),
            object::SectionKind::Elf(extra_section_sh_type),
        );
        obj.append_section_data(extra_id, extra_section_data, 1);

        let mut bytes = obj.write().unwrap();
        mutate_header(&mut bytes);
        let path = dir.join("vmlinux");
        fs::write(&path, &bytes).unwrap();
        path
    }

    /// Assert a successful [`strip_vmlinux_debug`] run on the fixture
    /// preserves the keep-list sections and deletes the extra
    /// (reloc-name) section via `strip_keep_list`'s name-based policy.
    ///
    /// This is the shared oracle for the four end-to-end pipeline
    /// tests: every variant that `strip_vmlinux_debug` accepts must
    /// yield the same output shape — keep-list sections present,
    /// reloc-name section absent.
    fn assert_stripped_preserves_keep_list_and_deletes(stripped: &Path, reloc_name: &str) {
        let data = fs::read(stripped).unwrap();
        let elf = goblin::elf::Elf::parse(&data).unwrap();
        let names: Vec<&str> = elf
            .section_headers
            .iter()
            .filter_map(|s| elf.shdr_strtab.get_at(s.sh_name))
            .collect();
        for name in [".symtab", ".strtab", ".BTF", ".rodata"] {
            assert!(
                names.contains(&name),
                "keep-list section {name} must survive strip_vmlinux_debug; got {names:?}"
            );
        }
        assert!(
            !names.contains(&reloc_name),
            "reloc section {reloc_name} must be deleted by strip_vmlinux_debug; got {names:?}"
        );
    }

    /// Pipeline pin #1: strip_vmlinux_debug handles a non-ALLOC
    /// `SHT_RELA` section with VALID byte range but entries whose
    /// `r_info` symbol indices are garbage (`0xA5A5...`).
    ///
    /// Before the SHF_ALLOC gate was dropped from
    /// [`neutralize_relocs`], non-ALLOC reloc sections were skipped,
    /// and `object::build::elf::Builder::read` then called
    /// `section.rela()` → `data_as_array` → `read_relocations_impl`
    /// on the raw bytes. With `sh_link == 0` (no linked symbol
    /// table) the impl uses `dynamic_symbols.len() == 0` for the
    /// bounds check; any non-null symbol index fails with
    /// `"Invalid symbol index N in relocation section at index M"`
    /// and `strip_vmlinux_debug` bubbled the error up. This test
    /// FAILS on that pre-fix codepath and PASSES after neutralize
    /// rewrites `sh_type` to `SHT_PROGBITS` on every `SHT_REL`/
    /// `SHT_RELA` section regardless of `SHF_ALLOC`.
    #[test]
    fn strip_vmlinux_debug_handles_nonalloc_rela_with_invalid_entries() {
        let src = TempDir::new().unwrap();
        let vmlinux = build_reloc_fixture(
            src.path(),
            b".rela.invalid",
            object::elf::SHT_RELA,
            // 24 bytes = one Elf64_Rela entry. 0xA5 bytes give
            // r_info = 0xA5A5A5A5A5A5A5A5 — a non-null, out-of-range
            // symbol index that `read_relocations_impl`'s bounds
            // check rejects when sh_link=0 directs the parse to the
            // empty dynamic symbol table.
            &[0xA5; 24],
            |_| {},
        );
        let stripped = strip_vmlinux_debug(&vmlinux).unwrap();
        assert_stripped_preserves_keep_list_and_deletes(stripped.path(), ".rela.invalid");
    }

    /// Pipeline pin #2: strip_vmlinux_debug handles a non-ALLOC
    /// `SHT_RELA` section whose `sh_size` is not a multiple of the
    /// `Elf64_Rela` entry size (24 bytes) — a shape that passes
    /// goblin's section-bounds check but fails object-crate's
    /// `data_as_array` divisibility check with `"Invalid ELF
    /// relocation section offset or size"`.
    ///
    /// This is the realistic arm64 kernel 7.0 failure mode: the
    /// section's byte range fits inside the file (so goblin accepts
    /// it) but doesn't represent a well-formed stream of `Elf64_Rela`
    /// entries from `object::build::elf::Builder::read`'s
    /// perspective.
    ///
    /// Before the fix, `Builder::read` failed at
    /// `slice_from_all_bytes` (non-exact multiple of entry size ⇒
    /// tail bytes remaining ⇒ Err). After the fix,
    /// [`neutralize_relocs`] rewrites `sh_type` to `SHT_PROGBITS` on
    /// every reloc section before `Builder::read` sees it; the sh_type
    /// mismatch short-circuits `section.rel()`/`section.rela()` at
    /// the type-check line (object-0.37.3/src/read/elf/section.rs:829,
    /// 849) and `data_as_array` is never called.
    #[test]
    fn strip_vmlinux_debug_handles_nonalloc_rela_with_non_entsize_sh_size() {
        let src = TempDir::new().unwrap();
        let vmlinux = build_reloc_fixture(
            src.path(),
            b".rela.odd",
            object::elf::SHT_RELA,
            // 24 bytes = one valid Elf64_Rela. We'll rewrite sh_size
            // to 17 below — fits inside the file's byte range (so
            // goblin accepts it) but 17 % 24 != 0 so object-crate's
            // `slice_from_all_bytes::<Rela64>` rejects the size.
            &[0x11; 24],
            |bytes| {
                let elf = goblin::elf::Elf::parse(bytes).unwrap();
                let shoff = elf.header.e_shoff as usize;
                let shentsize = elf.header.e_shentsize as usize;
                let idx = elf
                    .section_headers
                    .iter()
                    .position(|sh| elf.shdr_strtab.get_at(sh.sh_name) == Some(".rela.odd"))
                    .expect("fixture must carry .rela.odd");
                drop(elf);
                let sh_size_off = shoff + idx * shentsize + 32;
                // sh_size = 17 bytes, not divisible by 24
                // (sizeof(Elf64_Rela)). In-bounds (section payload is
                // 24 bytes) so goblin accepts, but the Builder's
                // `slice_from_all_bytes` check leaves a 17-byte tail
                // that rejects — matching the arm64 kernel 7.0
                // failure mode.
                let bad_size: u64 = 17;
                bytes[sh_size_off..sh_size_off + 8].copy_from_slice(&bad_size.to_le_bytes());
            },
        );
        let stripped = strip_vmlinux_debug(&vmlinux).unwrap();
        assert_stripped_preserves_keep_list_and_deletes(stripped.path(), ".rela.odd");
    }

    /// Pipeline pin #3: strip_vmlinux_debug handles `SHT_RELR`
    /// sections — arm64 kernels with `CONFIG_PIE` + `CONFIG_RELR`
    /// emit `.relr.dyn` with packed relative-relocation entries.
    ///
    /// This test locks in the SHT_RELR match arm in
    /// [`neutralize_relocs`] by checking BOTH invariants:
    ///
    /// 1. **Neutralize reaches SHT_RELR**: after the first pass,
    ///    the `.relr.dyn` section's `sh_type` is `SHT_PROGBITS`
    ///    (not its original `SHT_RELR`) and `sh_size` is 0. A
    ///    regression that drops SHT_RELR from the match arm leaves
    ///    the section with `sh_type = SHT_RELR` (19) — this
    ///    assertion fires.
    ///
    /// 2. **End-to-end strip succeeds**: `strip_vmlinux_debug` runs
    ///    cleanly and the output has `.relr.dyn` removed by
    ///    keep-list policy.
    ///
    /// Even on a well-formed `.relr.dyn` payload (Builder::read
    /// handles SHT_RELR opaquely via `section.data()` with no
    /// alignment check, so a "happy-path" SHT_RELR might pass
    /// `strip_vmlinux_debug` even without neutralization), the
    /// invariant-1 check locks in the neutralize reach to guard
    /// against a future regression that silently stops rewriting
    /// SHT_RELR sections.
    #[test]
    fn strip_vmlinux_debug_handles_relr_section() {
        let src = TempDir::new().unwrap();
        let vmlinux = build_reloc_fixture(
            src.path(),
            b".relr.dyn",
            object::elf::SHT_RELR,
            // 16 bytes = two packed RELR entries (each u64).
            &[0x77; 16],
            |_| {},
        );

        // Invariant 1: `neutralize_relocs` must rewrite the .relr.dyn
        // section's sh_type to SHT_PROGBITS and zero its sh_size.
        // Checking the function output directly locks in the
        // SHT_RELR match arm — a regression that drops SHT_RELR
        // from the arm would leave sh_type == SHT_RELR here.
        let raw = fs::read(&vmlinux).unwrap();
        let neutralized = neutralize_relocs(&raw).unwrap();
        let neutralized_elf = goblin::elf::Elf::parse(&neutralized).unwrap();
        let relr_sh = neutralized_elf
            .section_headers
            .iter()
            .find(|sh| neutralized_elf.shdr_strtab.get_at(sh.sh_name) == Some(".relr.dyn"))
            .expect(".relr.dyn must survive neutralize");
        assert_eq!(
            relr_sh.sh_type,
            object::elf::SHT_PROGBITS,
            ".relr.dyn sh_type must be rewritten to SHT_PROGBITS (SHT_RELR arm of the match); got sh_type={}",
            relr_sh.sh_type,
        );
        assert_eq!(
            relr_sh.sh_size, 0,
            ".relr.dyn sh_size must be zeroed post-neutralize",
        );

        // Invariant 2: end-to-end strip succeeds and removes the
        // reloc section.
        let stripped = strip_vmlinux_debug(&vmlinux).unwrap();
        assert_stripped_preserves_keep_list_and_deletes(stripped.path(), ".relr.dyn");
    }

    /// Pipeline pin #4: after strip_vmlinux_debug succeeds on an ELF
    /// carrying BOTH a non-alloc `SHT_RELA` and a `SHT_RELR` section,
    /// the output has every keep-list section (`.symtab`, `.strtab`,
    /// `.BTF`, `.rodata`) present and every reloc-named section
    /// deleted.
    ///
    /// Guards against a regression where the fix skipped one or the
    /// other reloc type (e.g. a future refactor that splits
    /// [`neutralize_relocs`]'s match arm and drops SHT_RELR). The
    /// pipeline pins above each cover one reloc type in isolation;
    /// this combined fixture ensures the fix holds when both types
    /// appear in the same kernel image.
    #[test]
    fn strip_vmlinux_debug_deletes_reloc_sections_and_preserves_keep_list() {
        use object::write;

        let src = TempDir::new().unwrap();
        let mut obj = write::Object::new(
            object::BinaryFormat::Elf,
            object::Architecture::X86_64,
            object::Endianness::Little,
        );
        // .text anchors the symtab.
        let text_id = obj.add_section(Vec::new(), b".text".to_vec(), object::SectionKind::Text);
        obj.append_section_data(text_id, &[0xCC; 64], 16);
        let _ = obj.add_symbol(write::Symbol {
            name: b"pipeline_anchor".to_vec(),
            value: 0x10,
            size: 8,
            kind: object::SymbolKind::Data,
            scope: object::SymbolScope::Compilation,
            weak: false,
            section: write::SymbolSection::Section(text_id),
            flags: object::SymbolFlags::None,
        });
        let btf_id = obj.add_section(Vec::new(), b".BTF".to_vec(), object::SectionKind::Other);
        obj.append_section_data(btf_id, &[0x42; 128], 1);
        let rodata_id = obj.add_section(
            Vec::new(),
            b".rodata".to_vec(),
            object::SectionKind::ReadOnlyData,
        );
        obj.append_section_data(rodata_id, &[0xAA; 256], 1);
        // Two reloc sections: .rela.dbg (non-alloc SHT_RELA with
        // garbage entries) and .relr.dyn (SHT_RELR). Both must be
        // deleted from the output and neither must break the strip.
        let rela_id = obj.add_section(
            Vec::new(),
            b".rela.dbg".to_vec(),
            object::SectionKind::Elf(object::elf::SHT_RELA),
        );
        obj.append_section_data(rela_id, &[0xA5; 24], 1);
        let relr_id = obj.add_section(
            Vec::new(),
            b".relr.dyn".to_vec(),
            object::SectionKind::Elf(object::elf::SHT_RELR),
        );
        obj.append_section_data(relr_id, &[0xD3; 24], 1);

        let bytes = obj.write().unwrap();
        let vmlinux = src.path().join("vmlinux");
        fs::write(&vmlinux, &bytes).unwrap();

        // Positive control: fixture must carry all sections the
        // post-strip assertion inspects. A silent rename by
        // object::write would false-pass the absence checks.
        let source_elf = goblin::elf::Elf::parse(&bytes).unwrap();
        let source_names: Vec<&str> = source_elf
            .section_headers
            .iter()
            .filter_map(|s| source_elf.shdr_strtab.get_at(s.sh_name))
            .collect();
        for name in [
            ".text",
            ".BTF",
            ".rodata",
            ".rela.dbg",
            ".relr.dyn",
            ".symtab",
            ".strtab",
        ] {
            assert!(
                source_names.contains(&name),
                "fixture missing expected section {name}; got {source_names:?}"
            );
        }

        let stripped = strip_vmlinux_debug(&vmlinux).unwrap();
        let data = fs::read(stripped.path()).unwrap();
        let elf = goblin::elf::Elf::parse(&data).unwrap();
        let names: Vec<&str> = elf
            .section_headers
            .iter()
            .filter_map(|s| elf.shdr_strtab.get_at(s.sh_name))
            .collect();

        // Keep-list sections survive.
        for name in [".symtab", ".strtab", ".BTF", ".rodata"] {
            assert!(
                names.contains(&name),
                "keep-list section {name} must survive strip; got {names:?}"
            );
        }
        // Both reloc sections deleted.
        for name in [".rela.dbg", ".relr.dyn"] {
            assert!(
                !names.contains(&name),
                "reloc section {name} must be deleted by strip; got {names:?}"
            );
        }
    }

    #[test]
    fn strip_vmlinux_debug_preserves_monitor_symbols() {
        let Some(path) = crate::monitor::find_test_vmlinux() else {
            skip!("no vmlinux found; {}", crate::KTSTR_KERNEL_HINT);
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
            skip!("no vmlinux found; {}", crate::KTSTR_KERNEL_HINT);
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
            skip!("no vmlinux found; {}", crate::KTSTR_KERNEL_HINT);
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
            skip!("no vmlinux found; {}", crate::KTSTR_KERNEL_HINT);
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

    // -- KconfigStatus Display impl --
    //
    // Pins the three Display strings that flow through `kernel list
    // --json` as the `kconfig_status` field. CI scripts consume these
    // exact strings, so any rewording is a downstream-visible
    // contract change.

    #[test]
    fn kconfig_status_display_matches_renders_lowercase_word() {
        assert_eq!(KconfigStatus::Matches.to_string(), "matches");
    }

    #[test]
    fn kconfig_status_display_stale_renders_lowercase_word_without_hashes() {
        let s = KconfigStatus::Stale {
            cached: "deadbeef".to_string(),
            current: "cafebabe".to_string(),
        }
        .to_string();
        assert_eq!(
            s, "stale",
            "Display elides the cached/current hashes; callers that need them must match on the variant directly"
        );
    }

    #[test]
    fn kconfig_status_display_untracked_renders_lowercase_word() {
        assert_eq!(KconfigStatus::Untracked.to_string(), "untracked");
    }

    // ------------------------------------------------------------
    // Cache-entry coordination lock tests
    // ------------------------------------------------------------

    /// `acquire_shared_lock` on a fresh cache root creates the
    /// lockfile at `{root}/.locks/{key}.lock` (and the parent
    /// `.locks/` subdirectory) — guards against drift to the old
    /// sibling layout.
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

    /// Two concurrent `acquire_shared_lock` calls on the same key
    /// both succeed — LOCK_SH coexists. Uses separate threads so
    /// each gets its own open-file-description (flock is per-OFD).
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
                // Hold briefly so all threads concurrently hold the
                // lock. Without this sleep, threads could serialize
                // through a narrow no-contention window and pass
                // even if the lock mistakenly rejected coexistence.
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

    /// `try_acquire_exclusive_lock` fails with an error naming the
    /// lockfile when a concurrent reader holds LOCK_SH. A
    /// spawned thread takes LOCK_SH and sleeps; the main thread
    /// attempts `try_acquire_exclusive_lock` non-blocking and
    /// asserts the error path fires.
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
            // Block until the main thread's non-blocking attempt
            // has had its chance to fail. Without this gate, the
            // reader could drop its lock before the main thread's
            // try_acquire_exclusive_lock ran, producing a
            // false-pass.
            release_rx.recv().unwrap();
        });
        ready_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("reader thread did not signal ready in time");
        // Now the reader is holding LOCK_SH. A non-blocking LOCK_EX
        // must bail.
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
        // Release the reader so the test cleans up.
        release_tx.send(()).unwrap();
        reader.join().expect("reader thread panicked");
    }

    /// `acquire_exclusive_lock_blocking` times out with the
    /// documented wording when a concurrent reader holds LOCK_SH
    /// longer than the timeout allows. Uses a 200ms timeout + a
    /// reader that holds for >500ms to reliably trip the bail.
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
        release_tx.send(()).unwrap();
        reader.join().expect("reader thread panicked");
    }

    /// `store()` acquires its own exclusive lock and completes
    /// successfully when no readers contend. Regression pin for
    /// the internal `acquire_exclusive_lock_blocking` call.
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
        // Lockfile should exist in the .locks/ subdirectory
        // (acquired during store).
        assert!(
            tmp.path()
                .join("cache")
                .join(".locks")
                .join("internal-lock.lock")
                .exists(),
            "lockfile materialized during store must persist after \
             store returns (it's fine; the flock is released on fd \
             drop but the file stays as a reusable sentinel)",
        );
    }

    /// `store()` blocks while a reader holds LOCK_SH, then
    /// completes after the reader releases. Drives the path by
    /// spawning a reader that holds its lock while attempting
    /// store in a thread; probes that store() does NOT complete
    /// within 200ms, then releases the reader and asserts store()
    /// completes within 10s.
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

        // Reader is holding LOCK_SH. A store attempt must block.
        // Spawn the store in a thread and check it hasn't
        // completed within a short window.
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
        // Short probe: store must NOT complete while reader holds lock.
        let early = store_done_rx.recv_timeout(std::time::Duration::from_millis(200));
        assert!(
            early.is_err(),
            "store() must block while reader holds LOCK_SH; got completion signal early",
        );
        // Release the reader — store should now unblock and finish.
        release_tx.send(()).unwrap();
        let finish = store_done_rx.recv_timeout(std::time::Duration::from_secs(10));
        assert!(
            finish.is_ok(),
            "store() must complete after reader releases; got timeout",
        );
        reader.join().expect("reader thread panicked");
        store_thread.join().expect("store thread panicked");
    }

    /// `lock_path` returns `{cache_root}/.locks/{key}.lock` — pins
    /// the exact on-disk shape against a refactor that relocates
    /// the lockfile. Pure path construction, no filesystem access.
    #[test]
    fn lock_path_returns_expected_shape() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        let path = cache.lock_path("my-key-42");
        assert_eq!(path, tmp.path().join(".locks").join("my-key-42.lock"));
    }

    /// `.locks/` subdirectory PERSISTS after the lock guard drops.
    /// Kernel_clean and any other walker that relies on list()
    /// filtering dotfiles assumes `.locks/` outlives individual
    /// acquires. A regression that rm'd the directory on guard
    /// drop would cause next-acquire to re-`mkdir` on a different
    /// inode and invalidate any /proc/locks peer-holder lookup
    /// (the peer's inode would be stale).
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
        // Guard dropped. .locks/ must still exist.
        assert!(
            locks_dir.is_dir(),
            ".locks/ must persist after guard drop — next acquire \
             keys /proc/locks on the existing inode",
        );
    }

    /// `CacheDir::list` skips `.locks/` — pins the dotfile-filter
    /// contract in `CacheDir::list`. kernel_clean iterates what
    /// `list()` returns, so this is the same guard: `.locks/` is
    /// NEVER visible to the cleanup path. A future refactor that
    /// removed the `starts_with('.')` filter would regress
    /// through this test.
    #[test]
    fn list_skips_locks_dotfile_subdirectory() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        // Materialize .locks/ via acquire, then list().
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

    /// Empty cache root: acquire creates `.locks/` lazily.
    /// Distinct from `acquire_shared_lock_creates_lockfile_at_expected_path`
    /// above because THAT test asserts the lockfile path; this
    /// one pins the LAZY-create behavior — the cache root can be
    /// totally empty (no kernel entries) and first acquire still
    /// works.
    #[test]
    fn acquire_on_empty_root_creates_locks_dir_lazily() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("pristine");
        std::fs::create_dir(&root).unwrap();
        let cache = CacheDir::with_root(root.clone());
        // Pre-acquire: no .locks/ yet.
        assert!(!root.join(".locks").exists());
        let _guard = cache
            .acquire_shared_lock("lazy-test")
            .expect("first acquire on empty root must succeed");
        assert!(
            root.join(".locks").is_dir(),
            "first acquire must materialize .locks/ lazily",
        );
    }

    /// `clean_all` MUST preserve the `.locks/` subdirectory. The
    /// `list()` filter skips dotfile children (tested elsewhere);
    /// `clean_all` removes what `list()` returns, so dotfiles — and
    /// specifically `.locks/` — survive. Without this guarantee,
    /// cleaning would delete a live SH flock's lockfile inode,
    /// leaving the next acquirer's `/proc/locks` lookup blind to
    /// the peer that still holds the (now-orphaned) fd.
    ///
    /// Repro sequence: populate an entry, acquire SH, clean_all,
    /// assert `.locks/` still exists AND the lockfile still exists
    /// inside it (the held fd keeps the inode alive even if the
    /// directory entry were removed — we're checking the directory
    /// entry specifically).
    #[test]
    fn cache_dir_clean_all_preserves_locks_subdir() {
        let tmp = TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let cache = CacheDir::with_root(cache_root.clone());
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());

        // Populate a cache entry so clean_all has something to
        // remove (its job is to tell the dotfile filter apart from
        // real entries).
        cache
            .store(
                "entry-a",
                &CacheArtifacts::new(&image),
                &test_metadata("6.14.0"),
            )
            .expect("store must succeed");
        // Acquire a shared lock so .locks/{key}.lock materializes.
        let _guard = cache
            .acquire_shared_lock("entry-a")
            .expect("SH acquire must succeed");

        let locks_dir = cache_root.join(".locks");
        let lockfile = locks_dir.join("entry-a.lock");
        assert!(locks_dir.is_dir(), "precondition: .locks/ must exist");
        assert!(lockfile.exists(), "precondition: lockfile must exist");

        // Clean every entry. .locks/ is a dotfile-prefixed child
        // and must NOT be treated as a cache entry.
        let removed = cache.clean_all().expect("clean_all must succeed");
        assert_eq!(removed, 1, "clean_all must remove exactly 1 entry");

        // Post-clean: .locks/ subdirectory survives so the held SH
        // flock's inode is still the one /proc/locks points at.
        assert!(
            locks_dir.is_dir(),
            ".locks/ subdirectory must survive clean_all — the live \
             SH flock's inode would otherwise orphan",
        );
        assert!(
            lockfile.exists(),
            "lockfile must still exist under .locks/ after clean_all",
        );

        // And the entry itself is gone.
        assert!(
            !cache_root.join("entry-a").exists(),
            "cache entry must be removed by clean_all",
        );
    }

    /// `acquire_shared_lock` MUST reject cache keys containing path
    /// traversal components (`..`, `/`). Without the rejection, a
    /// key of `"../../etc/passwd"` would join against the cache
    /// root and materialize a lockfile OUTSIDE `.locks/`, which is
    /// both a security concern (attacker-controlled write through
    /// a library entry point) and a correctness failure (the lock
    /// file's inode won't match anything in subsequent enumeration).
    ///
    /// Pins the `validate_cache_key` rejection from the two path-
    /// traversal entry points — the `/` separator check and the
    /// `..` component check — with a single test input that
    /// triggers both. The error text must be actionable; asserting
    /// against the `"path"` substring in the message catches both
    /// the separator and traversal rejection arms.
    #[test]
    fn cache_dir_acquire_rejects_path_traversal_key() {
        let tmp = TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let cache = CacheDir::with_root(cache_root.clone());

        // Attacker-shaped key: contains both `/` separators and
        // `..` traversal, hitting both rejection arms in
        // `validate_cache_key`.
        let err = cache
            .acquire_shared_lock("../../etc/passwd")
            .expect_err("path-traversal key must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("path"),
            "error must mention path rejection: {msg}",
        );

        // Critically: no lockfile must have been created anywhere
        // outside `.locks/`. Walk two levels above the cache root
        // to verify nothing landed in `tmp.path()` or a traversal
        // destination. The cache root itself may or may not exist
        // (acquire creates `.locks/` lazily, but the validator
        // rejects BEFORE that materialization).
        let etc_passwd_lock = tmp.path().join("etc").join("passwd.lock");
        assert!(
            !etc_passwd_lock.exists(),
            "path traversal must NOT create a lockfile outside .locks/",
        );
        // And verify .locks/ wasn't touched either — the validator
        // rejects before any FS state is mutated.
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

    use crate::test_support::test_helpers::{EnvVarGuard, lock_env};

    // -- validate_home_for_cache direct unit tests --
    //
    // These tests pin the helper directly. The helper reads
    // `HOME` from the process environment, so each test holds
    // [`lock_env`] across the env mutation and uses
    // [`EnvVarGuard`] to scope the change. The integration-level
    // pins on the full `KTSTR_CACHE_DIR → XDG_CACHE_HOME → HOME`
    // cascade live in model.rs and cache.rs as
    // `resolve_cache_root_*` tests; this set covers the helper's
    // contract surface so a regression in the validation logic
    // surfaces against this dedicated entry point as well as the
    // integration paths.

    /// Unset `HOME` — `env::var()` returns `Err(NotPresent)` and
    /// the validator surfaces "HOME is unset" as the matching
    /// arm. Distinguished from the empty-string case below so an
    /// operator hitting either shape sees the actual misconfiguration
    /// in the diagnostic.
    #[test]
    fn validate_home_for_cache_rejects_unset() {
        let _env_lock = lock_env();
        let _home = EnvVarGuard::remove("HOME");
        let err = super::validate_home_for_cache().expect_err("unset HOME must be rejected");
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

    /// Empty `HOME` — explicitly assigned to the empty string.
    /// `env::var()` returns `Ok("")` and the validator surfaces
    /// "HOME is set to the empty string" so an operator can
    /// identify a Dockerfile `ENV HOME=` or shell-rc `export HOME=`
    /// typo as the cause rather than confusing it with the
    /// container-init-dropped-HOME case.
    #[test]
    fn validate_home_for_cache_rejects_empty() {
        let _env_lock = lock_env();
        let _home = EnvVarGuard::set("HOME", "");
        let err = super::validate_home_for_cache().expect_err("empty HOME must be rejected");
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

    /// Literal `/` — the container-init / no-home shape. Pins the
    /// dedicated arm (separate from the more general
    /// `is_empty()` check) so the operator-facing diagnostic stays
    /// specific to this case rather than collapsing into a generic
    /// "unset" message.
    #[test]
    fn validate_home_for_cache_rejects_root_slash() {
        let _env_lock = lock_env();
        let _home = EnvVarGuard::set("HOME", "/");
        let err = super::validate_home_for_cache().expect_err("HOME=/ must be rejected");
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

    /// Relative path — would resolve against CWD at every call,
    /// silently relocating the cache as the operator changes
    /// directories. Pins the absolute-path requirement.
    #[test]
    fn validate_home_for_cache_rejects_relative_path() {
        let _env_lock = lock_env();
        for rel in ["relative", "./relative", "home/user", "."] {
            let _home = EnvVarGuard::set("HOME", rel);
            let err = super::validate_home_for_cache()
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

    /// Acceptable shapes — absolute paths starting with `/` and
    /// longer than just `/`. Pins the happy path so a regression
    /// that tightened one of the rejection arms (e.g. a length
    /// check that accidentally rejected `/a`) surfaces here.
    /// Also pins that the returned PathBuf carries the HOME bytes
    /// verbatim — no canonicalization, no .cache/ktstr suffix.
    #[test]
    fn validate_home_for_cache_accepts_absolute_paths() {
        let _env_lock = lock_env();
        for ok in [
            "/home/user",
            "/var/empty",
            "/root",
            "/a", // shortest non-`/` absolute path
            "/home/user with spaces",
            "/home/user/.local/share",
        ] {
            let _home = EnvVarGuard::set("HOME", ok);
            let got = super::validate_home_for_cache()
                .unwrap_or_else(|e| panic!("absolute path {ok:?} must be accepted; got: {e:#}"));
            assert_eq!(
                got,
                std::path::PathBuf::from(ok),
                "returned PathBuf must equal the HOME value verbatim — \
                 helper does not append the cache suffix or canonicalize",
            );
        }
    }

    /// Edge: a path that starts with `/` but contains junk later
    /// (e.g. `//`, `/./`, `/.`). The helper does NOT canonicalize —
    /// these accept and surface the OS-level diagnostic at use
    /// time per the body comments above the helper. Pins this
    /// "intentionally not caught" boundary so a future change that
    /// adds canonicalization (which would BREAK this test) is
    /// forced to update the doc comments at the same time.
    #[test]
    fn validate_home_for_cache_does_not_canonicalize_dots_and_doubles() {
        let _env_lock = lock_env();
        for not_normalized in ["//", "/./", "/.", "/foo//bar", "/./home"] {
            let _home = EnvVarGuard::set("HOME", not_normalized);
            super::validate_home_for_cache().unwrap_or_else(|e| {
                panic!(
                    "non-normalized but absolute path {not_normalized:?} must \
                     pass the helper (downstream OS surfaces the diagnostic); \
                     got: {e:#}",
                )
            });
        }
    }
}
