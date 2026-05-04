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

/// Trailing literal of every "image_missing" reason string. Pinned
/// here so [`format_image_missing_reason`] (the producer) and
/// [`classify_corrupt_reason`] (the consumer) reference the same
/// constant — the exact-suffix match in the classifier's
/// `image_missing` arm cannot drift if a future edit changes the
/// trailing wording without updating both sites.
pub(crate) const IMAGE_MISSING_SUFFIX: &str = " missing from entry directory";

/// Leading literal of every "image_missing" reason string. Pinned
/// alongside [`IMAGE_MISSING_SUFFIX`] so both the producer and the
/// classifier key on the same constants.
pub(crate) const IMAGE_MISSING_PREFIX: &str = "image file ";

/// Format the canonical "image_missing" reason string emitted by
/// [`crate::cache::CacheDir::list`] when an entry's
/// `metadata.json` is parseable but the boot image it names is
/// absent from the entry directory.
///
/// Centralised here so the producer site (`cache_dir.rs`'s
/// `list` corrupt-entry arm) and the classifier
/// [`classify_corrupt_reason`] cannot drift out of lockstep —
/// the result begins with [`IMAGE_MISSING_PREFIX`] and ends with
/// [`IMAGE_MISSING_SUFFIX`], so the classifier's exact prefix +
/// exact suffix predicate matches by construction without
/// admitting unrelated reasons that merely happen to contain
/// either literal.
pub(crate) fn format_image_missing_reason(image_name: &str) -> String {
    format!("{IMAGE_MISSING_PREFIX}{image_name}{IMAGE_MISSING_SUFFIX}")
}

/// Shared prefix → `error_kind` classifier.
///
/// Each `ListedEntry::Corrupt` carries a free-form `reason` string
/// produced by [`super::housekeeping::read_metadata`]. This helper
/// flattens those strings into a small, stable enum-of-strings the
/// CLI surfaces in `cargo ktstr kernel list --json` as the
/// `error_kind` field.
///
/// Reason-prefix → kind mapping (matched in this order):
///
/// | Reason (prefix or exact)                     | `error_kind`     |
/// |----------------------------------------------|------------------|
/// | `"metadata.json missing"` (exact)            | `"missing"`      |
/// | `"metadata.json unreadable: …"`              | `"unreadable"`   |
/// | `"metadata.json schema drift: …"`            | `"schema_drift"` |
/// | `"metadata.json malformed: …"`               | `"malformed"`    |
/// | `"metadata.json truncated: …"`               | `"truncated"`    |
/// | `"metadata.json parse error: …"`             | `"parse_error"`  |
/// | `"image file <name> missing from entry directory"` | `"image_missing"`|
/// | (anything else)                              | `"unknown"`      |
///
/// The `image_missing` arm matches on the exact prefix
/// [`IMAGE_MISSING_PREFIX`] AND the exact suffix
/// [`IMAGE_MISSING_SUFFIX`] — both literals are produced verbatim
/// by [`format_image_missing_reason`]. A loose `contains("missing")`
/// would also match unrelated future reasons that happen to mention
/// "missing" inside an `image file …` prefix (e.g. an "image file X
/// missing checksum field" reason), so the dispatcher pins both ends
/// of the canonical form.
///
/// The producer in [`super::housekeeping::read_metadata`] is the
/// authoritative source of these prefixes. If a new failure mode is
/// added there, both this dispatcher and the
/// `classify_corrupt_reason_covers_every_documented_prefix` test
/// must be updated in lockstep so the JSON contract stays stable.
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
    } else if reason.starts_with(IMAGE_MISSING_PREFIX)
        && reason.ends_with(IMAGE_MISSING_SUFFIX)
        && reason.len() > IMAGE_MISSING_PREFIX.len() + IMAGE_MISSING_SUFFIX.len()
    {
        "image_missing"
    } else {
        "unknown"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::shared_test_helpers::{create_fake_image, test_metadata};
    use crate::cache::{CacheArtifacts, CacheDir};
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

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
            extra_kconfig_hash: None,
            has_vmlinux: false,
            vmlinux_stripped: false,
            source_vmlinux_size: None,
            source_vmlinux_mtime_secs: None,
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
            extra_kconfig_hash: None,
            has_vmlinux: true,
            vmlinux_stripped: true,
            source_vmlinux_size: None,
            source_vmlinux_mtime_secs: None,
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
    /// rather than being omitted.
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
        assert!(
            git_json.contains(r#""ref":null"#),
            "Git.git_ref=None must serialize as explicit null under the `ref` key, got {git_json}"
        );
    }

    /// Older `metadata.json` files written before `Option` fields
    /// were emitted as explicit `null` simply omit the keys.
    #[test]
    fn kernel_source_absent_option_keys_deserialize_as_none() {
        let git_bare: KernelSource = serde_json::from_str(r#"{"type":"git"}"#)
            .expect("Git with absent Option keys must deserialize");
        assert!(matches!(
            git_bare,
            KernelSource::Git {
                git_hash: None,
                git_ref: None,
            }
        ));

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

        let git_ref_only: KernelSource = serde_json::from_str(r#"{"type":"git","ref":"main"}"#)
            .expect("Git with only ref must deserialize");
        assert!(matches!(
            git_ref_only,
            KernelSource::Git {
                git_hash: None,
                git_ref: Some(ref r),
            } if r == "main"
        ));

        let local_bare: KernelSource = serde_json::from_str(r#"{"type":"local"}"#)
            .expect("Local with absent Option keys must deserialize");
        assert!(matches!(
            local_bare,
            KernelSource::Local {
                source_tree_path: None,
                git_hash: None,
            }
        ));

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

    /// Table-drive every prefix → `error_kind` classifier mapping.
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

    /// The `image_missing` arm requires BOTH the canonical prefix
    /// AND the canonical trailing literal — strings that share only
    /// one half (or that would have matched the legacy loose
    /// `starts_with("image file ") && contains("missing")` predicate)
    /// must drop into `unknown`. Locks the tightening described on
    /// [`classify_corrupt_reason`].
    #[test]
    fn classify_corrupt_reason_image_missing_requires_exact_suffix() {
        // Loose predicate would match (prefix + the substring
        // "missing" in the wrong position). The tightened predicate
        // rejects it because the suffix isn't the canonical
        // ` missing from entry directory`.
        assert_eq!(
            classify_corrupt_reason("image file bzImage missing checksum field"),
            "unknown",
            "non-canonical 'image file … missing X' reason must NOT \
             classify as `image_missing` — only the exact \
             format_image_missing_reason() form is accepted",
        );
        // Suffix matches but prefix doesn't — must not classify.
        assert_eq!(
            classify_corrupt_reason("foo bzImage missing from entry directory"),
            "unknown",
            "suffix-only match without the canonical prefix must classify as unknown",
        );
        // Empty image name produces a degenerate prefix+suffix abut —
        // the length guard rejects it so a future bug that emits
        // `format_image_missing_reason("")` cannot silently classify.
        assert_eq!(
            classify_corrupt_reason("image file  missing from entry directory"),
            "unknown",
            "prefix+suffix with no image_name in between must classify as unknown",
        );
    }

    /// Producer→consumer round trip: every reason produced by
    /// [`format_image_missing_reason`] must classify as
    /// `image_missing`, regardless of the image_name value
    /// (alphanumerics, dots, dashes, multi-word names).
    #[test]
    fn classify_corrupt_reason_accepts_every_format_image_missing_output() {
        for image_name in [
            "bzImage",
            "Image",
            "vmlinuz-6.14.2",
            "kernel.bin",
            "image_with_underscores",
            "name-with-many-dashes",
        ] {
            let reason = format_image_missing_reason(image_name);
            assert_eq!(
                classify_corrupt_reason(&reason),
                "image_missing",
                "every produced reason (image_name={image_name:?}) must \
                 classify as image_missing — got reason {reason:?}",
            );
        }
    }

    /// `ListedEntry::error_kind()` returns `None` on a Valid entry
    /// and the classifier result on a Corrupt entry.
    #[test]
    fn listed_entry_error_kind_dispatches_on_variant() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");
        cache
            .store("valid-ek", &CacheArtifacts::new(&image), &meta)
            .unwrap();

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
}
