//! Kernel image cache for ktstr.
//!
//! Manages a local cache of built kernel images under an XDG-compliant
//! directory. Each cached kernel is a directory containing the boot
//! image, optionally a stripped vmlinux ELF (BTF + symbol table)
//! and `.config` (for CONFIG_HZ resolution), plus a `metadata.json`
//! descriptor.
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
//!     vmlinux           # stripped ELF (BTF + symbol table, optional)
//!     .config           # kernel config (CONFIG_HZ, optional)
//!     metadata.json     # KernelMetadata descriptor
//!   local-deadbee-x86_64-kc{kconfig_hash}/
//!     bzImage
//!     vmlinux
//!     .config
//!     metadata.json
//! ```
//!
//! # Atomic writes
//!
//! [`CacheDir::store`] writes to a temporary directory inside the cache
//! root, then atomically renames to the final path. Partial failures
//! never leave corrupt entries.

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Kernel source type recorded in cache metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceType {
    /// Downloaded tarball from kernel.org (version / prefix / EOL
    /// probe paths).
    Tarball,
    /// Shallow clone of a git URL at a caller-specified ref.
    Git,
    /// Build of a local on-disk kernel source tree.
    Local,
}

impl fmt::Display for SourceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SourceType::Tarball => f.write_str("tarball"),
            SourceType::Git => f.write_str("git"),
            SourceType::Local => f.write_str("local"),
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
    /// How the kernel source was acquired.
    pub source: SourceType,
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
    /// Git commit hash of the kernel source (short form).
    #[serde(default)]
    pub git_hash: Option<String>,
    /// Git ref used for checkout (branch, tag, or ref spec).
    #[serde(default)]
    pub git_ref: Option<String>,
    /// Path to the source tree on disk (local builds only).
    #[serde(default)]
    pub source_tree_path: Option<PathBuf>,
    /// Filename of the cached vmlinux ELF (BTF + symbol table).
    /// DWARF debug sections are stripped before caching.
    /// `None` when vmlinux was not available at cache time.
    #[serde(default)]
    pub vmlinux_name: Option<String>,
}

impl KernelMetadata {
    /// Create a new KernelMetadata with required fields.
    ///
    /// Optional fields default to `None`. Use struct update syntax
    /// within the crate or setter methods to populate them.
    pub fn new(source: SourceType, arch: String, image_name: String, built_at: String) -> Self {
        KernelMetadata {
            version: None,
            source,
            arch,
            image_name,
            config_hash: None,
            built_at,
            ktstr_kconfig_hash: None,
            git_hash: None,
            git_ref: None,
            source_tree_path: None,
            vmlinux_name: None,
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

    /// Set the git commit hash (short form).
    pub fn with_git_hash(mut self, hash: Option<String>) -> Self {
        self.git_hash = hash;
        self
    }

    /// Set the git ref used for checkout.
    pub fn with_git_ref(mut self, git_ref: Option<String>) -> Self {
        self.git_ref = git_ref;
        self
    }

    /// Set the source tree path (local builds only).
    pub fn with_source_tree_path(mut self, path: Option<std::path::PathBuf>) -> Self {
        self.source_tree_path = path;
        self
    }
}

// Re-export KernelId from kernel_path (canonical definition, std-only).
pub use crate::kernel_path::KernelId;

/// A cached kernel entry returned by [`CacheDir::lookup`],
/// [`CacheDir::store`], and [`CacheDir::list`].
#[derive(Debug)]
#[non_exhaustive]
pub struct CacheEntry {
    /// Cache key (directory name).
    pub key: String,
    /// Path to the cache entry directory.
    pub path: PathBuf,
    /// Deserialized metadata, if the metadata file is valid.
    pub metadata: Option<KernelMetadata>,
}

impl CacheEntry {
    /// Check if this entry was built with a different kconfig than `current_hash`.
    ///
    /// Returns `false` when metadata is missing or the entry has no
    /// recorded kconfig hash (pre-kconfig-tracking entries).
    pub fn has_stale_kconfig(&self, current_hash: &str) -> bool {
        self.metadata
            .as_ref()
            .and_then(|m| m.ktstr_kconfig_hash.as_deref())
            .is_some_and(|h| h != current_hash)
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
    /// Open or create a cache directory.
    ///
    /// Resolution order:
    /// 1. `KTSTR_CACHE_DIR` environment variable
    /// 2. `$XDG_CACHE_HOME/ktstr/kernels/`
    /// 3. `$HOME/.cache/ktstr/kernels/`
    ///
    /// Creates the directory tree if it does not exist.
    pub fn new() -> anyhow::Result<Self> {
        let root = resolve_cache_root()?;
        fs::create_dir_all(&root)?;
        Ok(CacheDir { root })
    }

    /// Open a cache directory at a specific path.
    ///
    /// Creates the directory if it does not exist. Used by tests and
    /// callers that need an explicit cache location.
    pub fn with_root(root: PathBuf) -> anyhow::Result<Self> {
        fs::create_dir_all(&root)?;
        Ok(CacheDir { root })
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
        let metadata = read_metadata(&entry_dir);
        // Entry must have a kernel image file.
        let image_exists = metadata
            .as_ref()
            .map(|m| entry_dir.join(&m.image_name).exists())
            .unwrap_or(false);
        if !image_exists {
            return None;
        }
        Some(CacheEntry {
            key: cache_key.to_string(),
            path: entry_dir,
            metadata,
        })
    }

    /// List all cached kernel entries, sorted by build time (newest first).
    ///
    /// Entries with missing or corrupt metadata are included with
    /// `metadata: None`. The caller decides how to handle them.
    pub fn list(&self) -> anyhow::Result<Vec<CacheEntry>> {
        let mut entries = Vec::new();
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
            let metadata = read_metadata(&path);
            entries.push(CacheEntry {
                key: name,
                path,
                metadata,
            });
        }
        // Sort by built_at descending (newest first). Entries without
        // metadata sort last.
        entries.sort_by(|a, b| {
            let a_time = a.metadata.as_ref().map(|m| m.built_at.as_str());
            let b_time = b.metadata.as_ref().map(|m| m.built_at.as_str());
            b_time.cmp(&a_time)
        });
        Ok(entries)
    }

    /// Store a kernel image in the cache.
    ///
    /// `cache_key`: directory name for the entry (e.g.
    /// `6.14.2-tarball-x86_64-kc{kconfig_hash}`).
    ///
    /// `image_path`: path to the kernel boot image to cache.
    ///
    /// `vmlinux_path`: optional path to the vmlinux ELF (stripped of
    /// DWARF debug sections). When present, vmlinux is copied alongside
    /// the boot image for BTF and symbol table access.
    ///
    /// `config_path`: optional path to the kernel `.config`. When
    /// present, cached alongside the image so `guest_kernel_hz` can
    /// resolve CONFIG_HZ without IKCONFIG (which lives in `.rodata`
    /// and is stripped from the cached vmlinux).
    ///
    /// `metadata`: descriptor to serialize as `metadata.json`.
    ///
    /// Files are copied (not moved) so the caller retains the
    /// originals. Writes atomically via a temporary directory that is
    /// renamed into place on success.
    pub fn store(
        &self,
        cache_key: &str,
        image_path: &Path,
        vmlinux_path: Option<&Path>,
        config_path: Option<&Path>,
        metadata: &KernelMetadata,
    ) -> anyhow::Result<CacheEntry> {
        validate_cache_key(cache_key)?;
        validate_filename(&metadata.image_name)?;
        let final_dir = self.root.join(cache_key);
        let tmp_dir = self
            .root
            .join(format!(".tmp-{}-{}", cache_key, std::process::id()));

        // Clean up any stale temp dir from a prior crash.
        if tmp_dir.exists() {
            fs::remove_dir_all(&tmp_dir)?;
        }
        fs::create_dir_all(&tmp_dir)?;

        // TmpGuard ensures the temp dir is cleaned up on any error
        // path, including serde serialization failures.
        let guard = TmpDirGuard(&tmp_dir);

        // Copy boot image.
        let image_dest = tmp_dir.join(&metadata.image_name);
        fs::copy(image_path, &image_dest)
            .map_err(|e| anyhow::anyhow!("copy kernel image to cache: {e}"))?;

        // Copy vmlinux (BTF + symbol table for monitor and probe).
        let has_vmlinux = if let Some(vmlinux) = vmlinux_path {
            fs::copy(vmlinux, tmp_dir.join("vmlinux"))
                .map_err(|e| anyhow::anyhow!("copy vmlinux to cache: {e}"))?;
            true
        } else {
            false
        };

        // Copy .config (CONFIG_HZ resolution for stripped vmlinux).
        if let Some(cfg) = config_path {
            fs::copy(cfg, tmp_dir.join(".config"))
                .map_err(|e| anyhow::anyhow!("copy .config to cache: {e}"))?;
        }

        // Write metadata (record vmlinux_name if vmlinux was stored).
        let mut meta = metadata.clone();
        meta.vmlinux_name = if has_vmlinux {
            Some("vmlinux".to_string())
        } else {
            None
        };
        let meta_json = serde_json::to_string_pretty(&meta)?;
        fs::write(tmp_dir.join("metadata.json"), meta_json)
            .map_err(|e| anyhow::anyhow!("write cache metadata: {e}"))?;

        // Atomic rename. Try rename first; if the target exists
        // (ENOTEMPTY / EEXIST), remove it and retry once. This avoids
        // the TOCTOU race of check-then-remove-then-rename.
        match fs::rename(&tmp_dir, &final_dir) {
            Ok(()) => {}
            Err(e)
                if e.raw_os_error() == Some(libc::ENOTEMPTY)
                    || e.raw_os_error() == Some(libc::EEXIST) =>
            {
                fs::remove_dir_all(&final_dir)?;
                fs::rename(&tmp_dir, &final_dir)
                    .map_err(|e2| anyhow::anyhow!("atomic rename cache entry (retry): {e2}"))?;
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
            metadata: Some(meta),
        })
    }

    /// Remove cached entries, optionally keeping the N most recent.
    ///
    /// When `keep` is `Some(n)`, retains the `n` most recent entries
    /// (by `built_at` timestamp). When `keep` is `None`, removes all
    /// entries.
    ///
    /// Returns the number of entries removed.
    pub fn clean(&self, keep: Option<usize>) -> anyhow::Result<usize> {
        let entries = self.list()?;
        let skip = keep.unwrap_or(0);
        let to_remove = entries.into_iter().skip(skip).collect::<Vec<_>>();
        let count = to_remove.len();
        for entry in &to_remove {
            fs::remove_dir_all(&entry.path)?;
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

/// Sections whose bytes are preserved in the cached vmlinux.
/// Everything not in this list or [`VMLINUX_ZERO_DATA_SECTIONS`] is
/// deleted (non-code) or has its bytes dropped via SHT_NOBITS (code).
/// Keep-list is safer than a remove-list: new debug or data sections
/// added by future compiler/kernel versions are stripped automatically
/// without updating this list.
const VMLINUX_KEEP_SECTIONS: &[&[u8]] = &[
    b"",           // null section (index 0)
    b".BTF",       // BPF Type Format — probe field resolution
    b".symtab",    // symbol table — monitor address resolution
    b".strtab",    // symbol string table
    b".shstrtab",  // section header string table (structural)
    b".rodata",    // IKCONFIG gzip blob read by read_hz_from_ikconfig
    b".bss",       // already SHT_NOBITS — holds swapper_pg_dir, scx_root
    b".init.data", // holds init-time data some kernels use for page tables
];

/// Data sections whose headers are kept (so symbols with `st_shndx`
/// pointing at them survive `Builder::delete_orphans`) but whose
/// bytes are dropped via SHT_NOBITS + zero-length data. Monitor code
/// reads symbol addresses (`st_value`) from these, never the backing
/// bytes — guest memory is the source of truth.
const VMLINUX_ZERO_DATA_SECTIONS: &[&[u8]] = &[
    b".data",         // holds init_top_pgt, map_idr, prog_idr, scx_watchdog_timeout
    b".data..percpu", // holds runqueues (per-CPU runqueue template)
];

/// Strip vmlinux for caching: drop code and debug sections, keep
/// symbol table, BTF, and the data sections monitor symbols
/// reference.
///
/// Uses a keep-list ([`VMLINUX_KEEP_SECTIONS`]): all sections not
/// in the list are deleted. This removes DWARF debug info, `.text`,
/// and other sections unused by the monitor. The cached vmlinux is
/// read for symbol addresses (`.symtab`) and BTF type info (`.BTF`);
/// the data sections themselves are kept because the object-crate
/// ELF builder drops symbols whose referenced section has been
/// deleted.
///
/// DWARF from the build tree (not the cache) is used by blazesym
/// for probe source locations — stripping the cached copy does not
/// affect that path.
///
/// If the keep-list strip fails (e.g. `Builder::read` encounters
/// an unsupported ELF feature), falls back to removing only
/// `.debug_*` sections, which preserves all other sections
/// including those the symbol table references.
///
/// Writes the stripped vmlinux to a temporary file and returns its
/// path. The caller must keep the returned `TempDir` alive until
/// `cache.store()` has copied the file.
pub fn strip_vmlinux_debug(vmlinux_path: &Path) -> anyhow::Result<(tempfile::TempDir, PathBuf)> {
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
    Ok((tmp_dir, stripped_path))
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
/// Sections in [`VMLINUX_KEEP_SECTIONS`] keep their bytes (symbol
/// tables, BTF, `.shstrtab`, `.rodata` for IKCONFIG, `.bss` already
/// SHT_NOBITS).
///
/// Sections in [`VMLINUX_ZERO_DATA_SECTIONS`] (`.data`,
/// `.data..percpu`) have their headers preserved but bytes dropped
/// via `SHT_NOBITS` + zero-length data. Monitor code reads symbol
/// addresses (`st_value`) from these, never the backing bytes.
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
        if VMLINUX_KEEP_SECTIONS.contains(&name) {
            continue;
        }
        if VMLINUX_ZERO_DATA_SECTIONS.contains(&name) {
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
            source: SourceType::Tarball,
            arch: "x86_64".to_string(),
            image_name: "bzImage".to_string(),
            config_hash: Some("abc123".to_string()),
            built_at: "2026-04-12T10:00:00Z".to_string(),
            ktstr_kconfig_hash: Some("def456".to_string()),
            git_hash: None,
            git_ref: None,
            source_tree_path: None,
            vmlinux_name: None,
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
        assert_eq!(parsed.source, SourceType::Tarball);
        assert_eq!(parsed.arch, "x86_64");
        assert_eq!(parsed.image_name, "bzImage");
        assert_eq!(parsed.config_hash.as_deref(), Some("abc123"));
        assert_eq!(parsed.built_at, "2026-04-12T10:00:00Z");
        assert_eq!(parsed.ktstr_kconfig_hash.as_deref(), Some("def456"));
        assert!(parsed.git_hash.is_none());
        assert!(parsed.git_ref.is_none());
        assert!(parsed.source_tree_path.is_none());
    }

    #[test]
    fn cache_metadata_serde_with_optional_fields() {
        let meta = KernelMetadata {
            version: Some("6.15-rc3".to_string()),
            source: SourceType::Git,
            arch: "aarch64".to_string(),
            image_name: "Image".to_string(),
            config_hash: None,
            built_at: "2026-04-12T12:00:00Z".to_string(),
            ktstr_kconfig_hash: None,
            git_hash: Some("a1b2c3d".to_string()),
            git_ref: Some("v6.15-rc3".to_string()),
            source_tree_path: None,
            vmlinux_name: None,
        };
        let json = serde_json::to_string(&meta).unwrap();
        let parsed: KernelMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.source, SourceType::Git);
        assert_eq!(parsed.git_hash.as_deref(), Some("a1b2c3d"));
        assert_eq!(parsed.git_ref.as_deref(), Some("v6.15-rc3"));
    }

    #[test]
    fn cache_metadata_serde_local_with_source_tree() {
        let meta = KernelMetadata {
            version: Some("6.14.0".to_string()),
            source: SourceType::Local,
            arch: "x86_64".to_string(),
            image_name: "bzImage".to_string(),
            config_hash: Some("fff000".to_string()),
            built_at: "2026-04-12T14:00:00Z".to_string(),
            ktstr_kconfig_hash: Some("aaa111".to_string()),
            git_hash: Some("deadbeef".to_string()),
            git_ref: None,
            source_tree_path: Some(PathBuf::from("/tmp/linux")),
            vmlinux_name: None,
        };
        let json = serde_json::to_string(&meta).unwrap();
        let parsed: KernelMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.source, SourceType::Local);
        assert_eq!(parsed.source_tree_path, Some(PathBuf::from("/tmp/linux")));
    }

    #[test]
    fn cache_metadata_deserialize_missing_optional_fields() {
        let json = r#"{
            "version": "6.14.2",
            "source": "tarball",
            "arch": "x86_64",
            "image_name": "bzImage",
            "config_hash": null,
            "built_at": "2026-04-12T10:00:00Z",
            "ktstr_kconfig_hash": null,
            "git_hash": null,
            "git_ref": null,
            "source_tree_path": null
        }"#;
        let parsed: KernelMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.version.as_deref(), Some("6.14.2"));
        assert!(parsed.config_hash.is_none());
    }

    #[test]
    fn cache_metadata_deserialize_null_version() {
        let json = r#"{
            "version": null,
            "source": "local",
            "arch": "x86_64",
            "image_name": "bzImage",
            "config_hash": null,
            "built_at": "2026-04-12T10:00:00Z",
            "ktstr_kconfig_hash": null,
            "git_hash": null,
            "git_ref": null,
            "source_tree_path": null
        }"#;
        let parsed: KernelMetadata = serde_json::from_str(json).unwrap();
        assert!(parsed.version.is_none());
        assert_eq!(parsed.source, SourceType::Local);
    }

    #[test]
    fn cache_metadata_deserialize_absent_optional_keys() {
        // Optional field keys entirely absent from JSON (not null —
        // absent). #[serde(default)] on Option<T> fields makes this
        // work for forward compatibility when new fields are added.
        let json = r#"{
            "source": "tarball",
            "arch": "x86_64",
            "image_name": "bzImage",
            "built_at": "2026-04-12T10:00:00Z"
        }"#;
        let parsed: KernelMetadata = serde_json::from_str(json).unwrap();
        assert!(parsed.version.is_none());
        assert!(parsed.config_hash.is_none());
        assert!(parsed.ktstr_kconfig_hash.is_none());
        assert!(parsed.git_hash.is_none());
        assert!(parsed.git_ref.is_none());
        assert!(parsed.source_tree_path.is_none());
        assert_eq!(parsed.source, SourceType::Tarball);
        assert_eq!(parsed.arch, "x86_64");
    }

    #[test]
    fn cache_metadata_deserialize_legacy_ktstr_git_hash() {
        // Pre-existing metadata.json files may carry a ktstr_git_hash
        // key from an older build. serde_json ignores unknown fields
        // by default (no deny_unknown_fields), so legacy entries
        // remain readable.
        let json = r#"{
            "source": "tarball",
            "arch": "x86_64",
            "image_name": "bzImage",
            "built_at": "2026-04-12T10:00:00Z",
            "ktstr_git_hash": "deadbeefcafef00d"
        }"#;
        let parsed: KernelMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.source, SourceType::Tarball);
        assert_eq!(parsed.image_name, "bzImage");
    }

    #[test]
    fn cache_metadata_source_type_serde() {
        // Verify lowercase serialization.
        let tarball = serde_json::to_string(&SourceType::Tarball).unwrap();
        assert_eq!(tarball, "\"tarball\"");
        let git = serde_json::to_string(&SourceType::Git).unwrap();
        assert_eq!(git, "\"git\"");
        let local = serde_json::to_string(&SourceType::Local).unwrap();
        assert_eq!(local, "\"local\"");
    }

    // -- CacheDir --

    #[test]
    fn cache_dir_with_root_creates_dir() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("kernels");
        assert!(!root.exists());
        let cache = CacheDir::with_root(root.clone()).unwrap();
        assert!(root.exists());
        assert_eq!(cache.root(), root);
    }

    #[test]
    fn cache_dir_list_empty() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf()).unwrap();
        let entries = cache.list().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn cache_dir_store_and_lookup() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();

        // Create a fake kernel image.
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");

        // Store.
        let entry = cache
            .store("6.14.2-tarball-x86_64", &image, None, None, &meta)
            .unwrap();
        assert_eq!(entry.key, "6.14.2-tarball-x86_64");
        assert!(entry.path.join("bzImage").exists());
        assert!(entry.path.join("metadata.json").exists());

        // Lookup.
        let found = cache.lookup("6.14.2-tarball-x86_64");
        assert!(found.is_some());
        let found = found.unwrap();
        assert_eq!(found.key, "6.14.2-tarball-x86_64");
        assert!(found.metadata.is_some());
        let found_meta = found.metadata.unwrap();
        assert_eq!(found_meta.version.as_deref(), Some("6.14.2"));
    }

    #[test]
    fn cache_dir_lookup_missing() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf()).unwrap();
        assert!(cache.lookup("nonexistent").is_none());
    }

    #[test]
    fn cache_dir_lookup_corrupt_metadata() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf()).unwrap();

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
        let cache = CacheDir::with_root(tmp.path().to_path_buf()).unwrap();

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
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());

        let meta1 = KernelMetadata {
            built_at: "2026-04-12T10:00:00Z".to_string(),
            ..test_metadata("6.14.2")
        };
        cache
            .store("6.14.2-tarball-x86_64", &image, None, None, &meta1)
            .unwrap();

        let meta2 = KernelMetadata {
            built_at: "2026-04-12T11:00:00Z".to_string(),
            ..test_metadata("6.14.2")
        };
        cache
            .store("6.14.2-tarball-x86_64", &image, None, None, &meta2)
            .unwrap();

        let found = cache.lookup("6.14.2-tarball-x86_64").unwrap();
        let found_meta = found.metadata.unwrap();
        assert_eq!(found_meta.built_at, "2026-04-12T11:00:00Z");
    }

    #[test]
    fn cache_dir_list_sorted_newest_first() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();
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
        cache.store("old", &image, None, None, &meta_old).unwrap();
        cache.store("new", &image, None, None, &meta_new).unwrap();
        cache.store("mid", &image, None, None, &meta_mid).unwrap();

        let entries = cache.list().unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].key, "new");
        assert_eq!(entries[1].key, "mid");
        assert_eq!(entries[2].key, "old");
    }

    #[test]
    fn cache_dir_list_includes_corrupt_entries() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf()).unwrap();

        // Create a valid entry.
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");
        cache.store("valid", &image, None, None, &meta).unwrap();

        // Create a corrupt entry (no metadata).
        let bad_dir = tmp.path().join("corrupt");
        fs::create_dir_all(&bad_dir).unwrap();

        let entries = cache.list().unwrap();
        assert_eq!(entries.len(), 2);
        // Valid entry has metadata.
        let valid = entries.iter().find(|e| e.key == "valid").unwrap();
        assert!(valid.metadata.is_some());
        // Corrupt entry has no metadata.
        let corrupt = entries.iter().find(|e| e.key == "corrupt").unwrap();
        assert!(corrupt.metadata.is_none());
    }

    #[test]
    fn cache_dir_list_skips_tmp_dirs() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf()).unwrap();

        // Create a .tmp- directory (in-progress store).
        let tmp_dir = tmp.path().join(".tmp-in-progress-12345");
        fs::create_dir_all(&tmp_dir).unwrap();

        let entries = cache.list().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn cache_dir_list_skips_regular_files() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf()).unwrap();

        // Create a regular file in the cache root.
        fs::write(tmp.path().join("stray-file.txt"), b"stray").unwrap();

        let entries = cache.list().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn cache_dir_clean_all() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());

        cache
            .store("a", &image, None, None, &test_metadata("6.14.0"))
            .unwrap();
        cache
            .store("b", &image, None, None, &test_metadata("6.14.1"))
            .unwrap();
        cache
            .store("c", &image, None, None, &test_metadata("6.14.2"))
            .unwrap();

        let removed = cache.clean(None).unwrap();
        assert_eq!(removed, 3);
        assert!(cache.list().unwrap().is_empty());
    }

    #[test]
    fn cache_dir_clean_keep_n() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();
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

        cache.store("old", &image, None, None, &meta_old).unwrap();
        cache.store("new", &image, None, None, &meta_new).unwrap();
        cache.store("mid", &image, None, None, &meta_mid).unwrap();

        let removed = cache.clean(Some(1)).unwrap();
        assert_eq!(removed, 2);

        let remaining = cache.list().unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].key, "new");
    }

    #[test]
    fn cache_dir_clean_keep_more_than_exist() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());

        cache
            .store("only", &image, None, None, &test_metadata("6.14.2"))
            .unwrap();

        let removed = cache.clean(Some(5)).unwrap();
        assert_eq!(removed, 0);
        assert_eq!(cache.list().unwrap().len(), 1);
    }

    #[test]
    fn cache_dir_clean_empty_cache() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf()).unwrap();
        let removed = cache.clean(None).unwrap();
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
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let mut meta = test_metadata("6.14.2");
        meta.image_name = "../escape".to_string();

        let err = cache
            .store("valid-key", &image, None, None, &meta)
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
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");

        let err = cache
            .store(".tmp-sneaky", &image, None, None, &meta)
            .unwrap_err();
        assert!(
            err.to_string().contains(".tmp-"),
            "expected .tmp- rejection, got: {err}"
        );
    }

    #[test]
    fn cache_dir_lookup_tmp_prefix_returns_none() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf()).unwrap();
        assert!(cache.lookup(".tmp-sneaky").is_none());
    }

    // -- cache key validation via store/lookup --

    #[test]
    fn cache_dir_store_empty_key_rejected() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");

        let err = cache.store("", &image, None, None, &meta).unwrap_err();
        assert!(
            err.to_string().contains("empty"),
            "expected empty-key error, got: {err}"
        );
    }

    #[test]
    fn cache_dir_lookup_empty_key_returns_none() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf()).unwrap();
        assert!(cache.lookup("").is_none());
    }

    #[test]
    fn cache_dir_store_path_traversal_rejected() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");

        let err = cache
            .store("../escape", &image, None, None, &meta)
            .unwrap_err();
        assert!(
            err.to_string().contains("path"),
            "expected path-traversal error, got: {err}"
        );
    }

    #[test]
    fn cache_dir_lookup_path_traversal_returns_none() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf()).unwrap();
        assert!(cache.lookup("../escape").is_none());
        assert!(cache.lookup("foo/../bar").is_none());
    }

    #[test]
    fn cache_dir_store_slash_in_key_rejected() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");

        let err = cache.store("a/b", &image, None, None, &meta).unwrap_err();
        assert!(
            err.to_string().contains("path separator"),
            "expected path-separator error, got: {err}"
        );
    }

    #[test]
    fn cache_dir_store_whitespace_only_key_rejected() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");

        let err = cache.store("   ", &image, None, None, &meta).unwrap_err();
        assert!(
            err.to_string().contains("empty"),
            "expected empty/whitespace error, got: {err}"
        );
    }

    // -- clean with mixed valid + corrupt entries --

    #[test]
    fn cache_dir_clean_keep_n_with_mixed_entries() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();
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
        cache.store("new", &image, None, None, &meta_new).unwrap();
        cache.store("old", &image, None, None, &meta_old).unwrap();

        // One corrupt entry (no metadata).
        let corrupt_dir = tmp.path().join("cache").join("corrupt");
        fs::create_dir_all(&corrupt_dir).unwrap();

        // list() returns 3 entries. Corrupt entries (no built_at) sort
        // last. keep=1 should keep the newest valid entry and remove
        // the old valid + corrupt entries.
        let removed = cache.clean(Some(1)).unwrap();
        assert_eq!(removed, 2);

        let remaining = cache.list().unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].key, "new");
    }

    // -- atomic write safety --

    #[test]
    fn cache_dir_store_cleans_stale_tmp() {
        let tmp = TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let cache = CacheDir::with_root(cache_root.clone()).unwrap();

        // Create a stale .tmp- directory simulating a prior crash.
        let stale_tmp = cache_root.join(format!(".tmp-mykey-{}", std::process::id()));
        fs::create_dir_all(&stale_tmp).unwrap();
        fs::write(stale_tmp.join("junk"), b"leftover").unwrap();

        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");

        // Store should succeed despite stale tmp dir.
        let entry = cache.store("mykey", &image, None, None, &meta).unwrap();
        assert!(entry.path.join("bzImage").exists());
        // Stale tmp dir should be gone.
        assert!(!stale_tmp.exists());
    }

    #[test]
    fn cache_dir_store_with_vmlinux() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let vmlinux = src_dir.path().join("vmlinux");
        fs::write(&vmlinux, b"fake vmlinux ELF").unwrap();
        let meta = test_metadata("6.14.2");

        let entry = cache
            .store("with-vmlinux", &image, Some(&vmlinux), None, &meta)
            .unwrap();
        assert!(entry.path.join("bzImage").exists());
        assert!(entry.path.join("vmlinux").exists());
        assert!(entry.path.join("metadata.json").exists());
        // Metadata records vmlinux_name.
        let entry_meta = entry.metadata.unwrap();
        assert_eq!(entry_meta.vmlinux_name.as_deref(), Some("vmlinux"));
        // Original files still exist (copy, not move).
        assert!(image.exists());
        assert!(vmlinux.exists());
    }

    #[test]
    fn cache_dir_store_without_vmlinux() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");

        let entry = cache
            .store("no-vmlinux", &image, None, None, &meta)
            .unwrap();
        assert!(entry.path.join("bzImage").exists());
        assert!(!entry.path.join("vmlinux").exists());
        assert!(entry.path.join("metadata.json").exists());
        // Metadata has no vmlinux_name.
        let entry_meta = entry.metadata.unwrap();
        assert!(entry_meta.vmlinux_name.is_none());
    }

    #[test]
    fn cache_dir_store_with_config() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let config = src_dir.path().join(".config");
        let config_content = b"CONFIG_HZ=1000\nCONFIG_SCHED_CLASS_EXT=y\n";
        fs::write(&config, config_content).unwrap();
        let meta = test_metadata("6.14.2");

        let entry = cache
            .store("with-config", &image, None, Some(&config), &meta)
            .unwrap();
        assert!(entry.path.join("bzImage").exists());
        assert!(entry.path.join(".config").exists());
        assert!(entry.path.join("metadata.json").exists());
        // Cached .config contents match original.
        let cached = fs::read(entry.path.join(".config")).unwrap();
        assert_eq!(cached, config_content);
        // Original .config still exists (copy, not move).
        assert!(config.exists());
    }

    #[test]
    fn cache_dir_store_without_config() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");

        let entry = cache.store("no-config", &image, None, None, &meta).unwrap();
        assert!(entry.path.join("bzImage").exists());
        assert!(!entry.path.join(".config").exists());
    }

    #[test]
    fn cache_dir_store_preserves_original_image() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");

        cache.store("key", &image, None, None, &meta).unwrap();

        // Original image must still exist (copy, not move).
        assert!(image.exists());
    }

    // -- strip_vmlinux_debug --

    /// Build a minimal ELF with .BTF, .text, .debug_*, and symtab.
    fn create_elf_with_debug(dir: &Path) -> PathBuf {
        use object::write;
        let mut obj = write::Object::new(
            object::BinaryFormat::Elf,
            object::Architecture::X86_64,
            object::Endianness::Little,
        );
        // .text — loadable code (not in keep-list, stripped by keep-list path).
        let text_id = obj.add_section(Vec::new(), b".text".to_vec(), object::SectionKind::Text);
        obj.append_section_data(text_id, &[0xCC; 64], 1);
        // Symbol so .symtab and .strtab are generated.
        let sym_id = obj.add_symbol(write::Symbol {
            name: b"test_symbol".to_vec(),
            value: 0x1000,
            size: 8,
            kind: object::SymbolKind::Data,
            scope: object::SymbolScope::Compilation,
            weak: false,
            section: write::SymbolSection::Section(text_id),
            flags: object::SymbolFlags::None,
        });
        let _ = sym_id;
        // .BTF — kept by both keep-list and fallback.
        let btf_id = obj.add_section(Vec::new(), b".BTF".to_vec(), object::SectionKind::Metadata);
        obj.append_section_data(btf_id, &[0xEB; 256], 1);
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

        let data = obj.write().unwrap();
        let path = dir.join("vmlinux");
        fs::write(&path, &data).unwrap();
        path
    }

    #[test]
    fn strip_vmlinux_debug_removes_debug_keeps_btf_symtab() {
        let src = TempDir::new().unwrap();
        let vmlinux = create_elf_with_debug(src.path());
        let original_size = fs::metadata(&vmlinux).unwrap().len();

        let (_dir, stripped_path) = strip_vmlinux_debug(&vmlinux).unwrap();
        let stripped_size = fs::metadata(&stripped_path).unwrap().len();

        assert!(
            stripped_size < original_size,
            "stripped ({stripped_size}) should be smaller than original ({original_size})"
        );

        let data = fs::read(&stripped_path).unwrap();
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
        // .BTF preserved (in keep-list).
        assert!(section_names.contains(&".BTF"), "should preserve .BTF");
        // .symtab preserved (in keep-list).
        assert!(
            section_names.contains(&".symtab"),
            "should preserve .symtab"
        );
        assert!(
            section_names.contains(&".strtab"),
            "should preserve .strtab"
        );
    }

    #[test]
    fn strip_vmlinux_debug_symtab_readable() {
        let src = TempDir::new().unwrap();
        let vmlinux = create_elf_with_debug(src.path());

        let (_dir, stripped_path) = strip_vmlinux_debug(&vmlinux).unwrap();
        let data = fs::read(&stripped_path).unwrap();
        let elf = goblin::elf::Elf::parse(&data).unwrap();

        // Smoke check: stripping a synthetic ELF produces a readable
        // symbol table whose strtab still contains our test symbol
        // name. End-to-end symbol preservation on real vmlinuxes is
        // covered by the *_preserves_monitor_symbols tests below.
        let found = elf
            .syms
            .iter()
            .any(|s| elf.strtab.get_at(s.st_name) == Some("test_symbol"));
        assert!(found, "stripped ELF should contain test_symbol in symtab");
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
        let path = match crate::monitor::find_test_vmlinux() {
            Some(p) => p,
            None => return,
        };
        // find_test_vmlinux may return /sys/kernel/btf/vmlinux (raw BTF,
        // not an ELF), which strip_vmlinux_debug cannot parse.
        if path.starts_with("/sys/") {
            skip!("vmlinux is raw BTF (not ELF), cannot strip debug");
        }
        let (_dir, stripped_path) = strip_vmlinux_debug(&path).unwrap();
        let syms = crate::monitor::symbols::KernelSymbols::from_vmlinux(&stripped_path).unwrap();
        assert_ne!(
            syms.runqueues, 0,
            "runqueues symbol missing from stripped vmlinux"
        );
        assert_ne!(
            syms.per_cpu_offset, 0,
            "__per_cpu_offset symbol missing from stripped vmlinux"
        );
        assert!(
            syms.init_top_pgt.is_some(),
            "init_top_pgt/swapper_pg_dir symbol missing from stripped vmlinux"
        );
        // For every optional symbol KernelSymbols tracks: presence must
        // survive the strip. A symbol that is absent from the source
        // vmlinux stays absent (kernel-config-dependent); a symbol that
        // is present must still be present.
        let source_syms = crate::monitor::symbols::KernelSymbols::from_vmlinux(&path).unwrap();
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
    }

    #[test]
    fn strip_vmlinux_debug_preserves_bpf_idr_symbols() {
        let path = match crate::monitor::find_test_vmlinux() {
            Some(p) => p,
            None => return,
        };
        if path.starts_with("/sys/") {
            skip!("vmlinux is raw BTF (not ELF), cannot strip debug");
        }
        let (_dir, stripped_path) = strip_vmlinux_debug(&path).unwrap();
        let data = fs::read(&stripped_path).unwrap();
        let elf = goblin::elf::Elf::parse(&data).unwrap();
        let has = |name: &str| {
            elf.syms
                .iter()
                .any(|s| s.st_value != 0 && elf.strtab.get_at(s.st_name) == Some(name))
        };
        assert!(
            has("map_idr"),
            "map_idr symbol missing from stripped vmlinux"
        );
        assert!(
            has("prog_idr"),
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
        let path = match crate::monitor::find_test_vmlinux() {
            Some(p) => p,
            None => return,
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
        let source_has_schedule = source_elf
            .syms
            .iter()
            .any(|s| s.st_value != 0 && source_elf.strtab.get_at(s.st_name) == Some("schedule"));
        if !source_has_schedule {
            skip!(
                "source vmlinux has no `schedule` symbol \
                 (already stripped by older ktstr) -- rebuild the kernel \
                 cache to exercise this test"
            );
        }

        let (_dir, stripped_path) = strip_vmlinux_debug(&path).unwrap();
        let data = fs::read(&stripped_path).unwrap();
        let elf = goblin::elf::Elf::parse(&data).unwrap();
        let has_func = |name: &str| {
            elf.syms
                .iter()
                .any(|s| s.st_value != 0 && elf.strtab.get_at(s.st_name) == Some(name))
        };
        assert!(
            has_func("schedule"),
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
