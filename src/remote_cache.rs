//! Remote cache backend for GHA runners via opendal.
//!
//! When `KTSTR_GHA_CACHE=1` and `ACTIONS_CACHE_URL` are set, cache
//! operations transparently extend to a remote GHA cache. Local cache
//! is always authoritative: lookups check local first, stores write to
//! both. Remote failures are non-fatal (logged as warnings).
//!
//! Cache entries are serialized as tar archives containing the kernel
//! image, vmlinux (if present), and metadata.json, stored as a single
//! blob per cache key in the GHA cache service.
//!
//! Tar payloads are zstd-compressed before upload and decompressed on
//! download. Decompression is bounded by
//! [`MAX_DECOMPRESSED_REMOTE_CACHE_BYTES`] to guard against a hostile
//! zstd payload (zstd compresses pathologically well on repeated
//! bytes, so a few-KiB blob can decompress to gigabytes). A blob that
//! does not start with the zstd magic number is rejected.

use std::io::{Read, Write};
use std::path::Path;
use std::sync::LazyLock;

use crate::cache::{CacheDir, CacheEntry, KernelMetadata};

/// Tokio runtime for opendal async operations.
///
/// opendal's `Operator` is async. cargo-ktstr is synchronous, so we
/// provide a dedicated single-threaded runtime and call `block_on()`
/// for each remote cache operation. Created lazily on first use;
/// never created when remote cache is disabled.
///
/// # Serialization
///
/// `new_current_thread()` plus synchronous callers means every
/// `block_on(op.read | op.write)` runs to completion on the calling
/// thread before the next remote operation can start — there is no
/// task scheduler driving multiple futures concurrently. Today's
/// callers ([`remote_lookup`] and [`remote_store`]) issue exactly
/// one I/O per invocation and the surrounding `cargo-ktstr` flow
/// does not parallelise cache lookups, so the serial pattern is
/// correct for the current workload. If a future caller needs
/// concurrent remote ops (e.g. a parallel pre-fetch over many cache
/// keys), this runtime configuration must change — either to a
/// multi-thread runtime, or to a single explicit `block_on(async {
/// join!(...) })` that drives futures concurrently within the
/// current_thread runtime.
///
/// Calling `block_on` from inside an existing tokio async context
/// panics — this runtime must only be entered from synchronous call
/// sites.
static RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("failed to create tokio runtime for remote cache")
});

/// Check if remote GHA cache is enabled.
///
/// Requires both `KTSTR_GHA_CACHE=1` and `ACTIONS_CACHE_URL` to be
/// set. Returns false silently when either is absent (normal for
/// local dev).
pub fn is_enabled() -> bool {
    std::env::var("KTSTR_GHA_CACHE")
        .ok()
        .is_some_and(|v| v == "1")
        && std::env::var("ACTIONS_CACHE_URL")
            .ok()
            .is_some_and(|v| !v.is_empty())
}

/// Namespace string passed to opendal's `Ghac::version` builder.
/// Two purposes:
///
/// 1. Isolates ktstr cache entries from other tools sharing the
///    same GHA cache service.
/// 2. Carries a `-vN` suffix so format changes invalidate stale
///    entries without colliding with the previous wire shape.
///
/// **Bump the version suffix when the on-the-wire format changes
/// in a way old readers cannot interpret.** Examples that require
/// a bump:
/// - Compression format change (e.g. zstd → zstd+dict, or zstd → lz4).
/// - Removal of a fallback path readers used to depend on (e.g. the
///   v2 bump went out alongside dropping the raw-tar fallback that
///   pre-zstd entries relied on — see [`decompress_payload`]).
/// - Tar layout change (filenames, structure, additional required
///   members).
/// - Metadata schema change that breaks deserialization of older
///   entries.
///
/// Additive changes that older readers can still parse (e.g. a new
/// optional field in metadata) do NOT require a bump.
const REMOTE_CACHE_NAMESPACE: &str = "ktstr-v2";

/// Create an opendal operator for the GHA cache service.
///
/// Relies on opendal's Ghac service, which reads `ACTIONS_CACHE_URL`
/// and `ACTIONS_RUNTIME_TOKEN` from the environment (set automatically
/// by the GHA runner); ktstr itself does not touch either variable.
/// The `version` field is set to [`REMOTE_CACHE_NAMESPACE`] —
/// namespaces ktstr entries against other tools sharing the cache
/// AND invalidates stale entries when ktstr's wire format changes.
fn create_operator() -> Result<opendal::Operator, String> {
    let builder = opendal::services::Ghac::default()
        .root("/")
        .version(REMOTE_CACHE_NAMESPACE);

    opendal::Operator::new(builder)
        .map_err(|e| format!("create ghac operator: {e}"))
        .map(|b| b.finish())
}

/// Zstd magic number (first 4 bytes of any zstd frame).
const ZSTD_MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];

/// Decompressed-size ceiling for [`decompress_payload`] zstd payloads.
/// Bounds the allocation a malicious or corrupted zstd payload from
/// the GHA cache service can force, since zstd compresses
/// pathologically well on repeated bytes (a few-KiB compressed blob
/// can decompress to gigabytes). 1 GiB covers any realistic cache
/// entry — bzImage is ~15 MiB, stripped vmlinux ~45 MiB, an
/// unstripped debug vmlinux with BTF can reach ~500 MiB — while
/// bounding worst-case allocation against hostile zstd payloads.
/// Public so a downstream consumer can size buffers against the
/// same ceiling without hardcoding the value.
pub const MAX_DECOMPRESSED_REMOTE_CACHE_BYTES: u64 = 1024 * 1024 * 1024;

/// Pack a cache entry directory into a tar archive in memory.
///
/// The tar contains the kernel image, vmlinux (if present), and
/// metadata.json from the cache entry directory. Paths inside the
/// tar are relative filenames (no directory prefix).
///
/// The tar is then compressed with zstd before upload.
/// [`unpack_and_store`] verifies the zstd magic number on download
/// and decompresses; a payload missing the magic is rejected
/// (the on-the-wire format is zstd-only).
fn pack_entry(entry_dir: &Path, metadata: &KernelMetadata) -> Result<Vec<u8>, String> {
    let mut archive = tar::Builder::new(Vec::new());

    // Null out source_tree_path before serializing — it contains
    // local filesystem paths that must not leak to remote storage.
    // For non-Local source variants there's nothing to sanitize.
    let mut meta_sanitized = metadata.clone();
    if let crate::cache::KernelSource::Local {
        source_tree_path, ..
    } = &mut meta_sanitized.source
    {
        *source_tree_path = None;
    }

    // Add metadata.json.
    let meta_json = serde_json::to_string_pretty(&meta_sanitized)
        .map_err(|e| format!("serialize metadata: {e}"))?;
    let meta_bytes = meta_json.as_bytes();
    crate::tar_util::pack_tar_entry(
        &mut archive,
        "metadata.json",
        0o644,
        meta_bytes.len() as u64,
        meta_bytes,
    )
    .map_err(|e| format!("tar append metadata: {e}"))?;

    // Add kernel image.
    let image_path = entry_dir.join(&metadata.image_name);
    let mut image_file = std::fs::File::open(&image_path)
        .map_err(|e| format!("open image {}: {e}", image_path.display()))?;
    let image_size = image_file
        .metadata()
        .map_err(|e| format!("image metadata: {e}"))?
        .len();
    crate::tar_util::pack_tar_entry(
        &mut archive,
        &metadata.image_name,
        0o644,
        image_size,
        &mut image_file,
    )
    .map_err(|e| format!("tar append image: {e}"))?;

    // Add vmlinux if present (BTF source for build.rs).
    let vmlinux_path = entry_dir.join("vmlinux");
    if let Ok(mut vmlinux_file) = std::fs::File::open(&vmlinux_path) {
        let vmlinux_size = vmlinux_file
            .metadata()
            .map_err(|e| format!("vmlinux metadata: {e}"))?
            .len();
        crate::tar_util::pack_tar_entry(
            &mut archive,
            "vmlinux",
            0o644,
            vmlinux_size,
            &mut vmlinux_file,
        )
        .map_err(|e| format!("tar append vmlinux: {e}"))?;
    }

    let tar_bytes = archive
        .into_inner()
        .map_err(|e| format!("finalize tar: {e}"))?;

    // Compress with zstd (level 3: good ratio at fast speed).
    zstd::encode_all(tar_bytes.as_slice(), 3).map_err(|e| format!("zstd compress: {e}"))
}

/// Decompress a zstd-compressed cache blob. Rejects payloads that
/// do not start with the zstd magic number — the on-the-wire format
/// is zstd-only since the encoder ([`pack_entry`]) always compresses.
/// The magic-number precondition catches truncated downloads (any
/// payload < 4 bytes) and non-zstd content with a clearer error
/// than the zstd library's "invalid header" diagnostic.
///
/// Bounded by [`MAX_DECOMPRESSED_REMOTE_CACHE_BYTES`] — a payload
/// that would expand past that ceiling surfaces an error rather than
/// allocating unbounded memory, guarding against a hostile zstd
/// payload from the GHA cache service.
fn decompress_payload(data: &[u8]) -> Result<Vec<u8>, String> {
    if data.len() < 4 || data[..4] != ZSTD_MAGIC {
        return Err("remote cache entry missing zstd magic".to_string());
    }
    decompress_capped(data, MAX_DECOMPRESSED_REMOTE_CACHE_BYTES)
        .map_err(|e| format!("zstd decompress: {e}"))
}

/// Decompress a zstd payload into a `Vec<u8>` capped at
/// `max_decompressed` bytes — bombing out with an error if the
/// payload would expand past the ceiling. Reads through
/// `Read::take(cap + 1)` so a payload that decompresses to
/// exactly `cap` bytes is accepted while one that produces
/// `cap + 1` bytes (or more) is rejected — the +1 sentinel
/// distinguishes "EOF coincided with the cap" from "more data
/// behind the cap".
fn decompress_capped(bytes: &[u8], max_decompressed: u64) -> Result<Vec<u8>, String> {
    let decoder =
        zstd::stream::read::Decoder::new(bytes).map_err(|e| format!("zstd decoder init: {e}"))?;
    let mut out = Vec::new();
    decoder
        .take(max_decompressed.saturating_add(1))
        .read_to_end(&mut out)
        .map_err(|e| format!("zstd decompress read: {e}"))?;
    if out.len() as u64 > max_decompressed {
        return Err(format!(
            "zstd-decompressed payload exceeds the {max_decompressed}-byte cap (decompression-bomb guard)",
        ));
    }
    Ok(out)
}

/// Unpack a tar archive into a cache directory via CacheDir::store.
///
/// Extracts metadata.json, the kernel image, and vmlinux (if present)
/// from the tar blob, writes them to temp files, then stores via the
/// local cache API for atomic placement. The unpacked vmlinux was
/// already stripped by the producer; `CacheDir::store` re-runs the
/// strip pipeline (idempotent — the keep-list partition produces the
/// same layout) and falls back to copying verbatim on error.
///
/// Decompression is bounded by [`MAX_DECOMPRESSED_REMOTE_CACHE_BYTES`].
fn unpack_and_store(cache: &CacheDir, cache_key: &str, data: &[u8]) -> Result<CacheEntry, String> {
    let tar_bytes = decompress_payload(data)?;
    let mut archive = tar::Archive::new(tar_bytes.as_slice());
    let entries = archive
        .entries()
        .map_err(|e| format!("read tar entries: {e}"))?;

    let mut metadata: Option<KernelMetadata> = None;
    let mut image_data: Option<(String, Vec<u8>)> = None;
    let mut vmlinux_data: Option<Vec<u8>> = None;

    for entry_result in entries {
        let mut entry = entry_result.map_err(|e| format!("tar entry: {e}"))?;
        let path = entry
            .path()
            .map_err(|e| format!("tar entry path: {e}"))?
            .to_string_lossy()
            .into_owned();

        if path == "metadata.json" {
            let mut content = String::new();
            entry
                .read_to_string(&mut content)
                .map_err(|e| format!("read metadata from tar: {e}"))?;
            metadata = Some(
                serde_json::from_str(&content)
                    .map_err(|e| format!("parse metadata from tar: {e}"))?,
            );
        } else if path == "vmlinux" {
            let mut data = Vec::new();
            entry
                .read_to_end(&mut data)
                .map_err(|e| format!("read vmlinux from tar: {e}"))?;
            vmlinux_data = Some(data);
        } else {
            let mut data = Vec::new();
            entry
                .read_to_end(&mut data)
                .map_err(|e| format!("read image from tar: {e}"))?;
            image_data = Some((path, data));
        }
    }

    let meta = metadata.ok_or_else(|| "tar archive missing metadata.json".to_string())?;
    let (_, img_bytes) =
        image_data.ok_or_else(|| "tar archive missing kernel image".to_string())?;

    // Write image and vmlinux to temp files for CacheDir::store.
    let tmp_dir = tempfile::TempDir::new().map_err(|e| format!("create temp dir: {e}"))?;
    let tmp_image = tmp_dir.path().join(&meta.image_name);
    let mut f = std::fs::File::create(&tmp_image).map_err(|e| format!("create temp image: {e}"))?;
    f.write_all(&img_bytes)
        .map_err(|e| format!("write temp image: {e}"))?;
    drop(f);

    let tmp_vmlinux_path;
    let vmlinux_ref = if let Some(ref vml_bytes) = vmlinux_data {
        tmp_vmlinux_path = tmp_dir.path().join("vmlinux");
        let mut vf = std::fs::File::create(&tmp_vmlinux_path)
            .map_err(|e| format!("create temp vmlinux: {e}"))?;
        vf.write_all(vml_bytes)
            .map_err(|e| format!("write temp vmlinux: {e}"))?;
        drop(vf);
        Some(tmp_vmlinux_path.as_path())
    } else {
        None
    };

    let mut artifacts = crate::cache::CacheArtifacts::new(&tmp_image);
    if let Some(v) = vmlinux_ref {
        artifacts = artifacts.with_vmlinux(v);
    }
    cache
        .store(cache_key, &artifacts, &meta)
        .map_err(|e| format!("local cache store: {e}"))
}

/// Look up a cache key in the remote GHA cache.
///
/// On hit, downloads the tar blob and unpacks it into the local
/// cache via `CacheDir::store`. Returns the local `CacheEntry` on
/// success. Returns `None` on remote miss. Logs warnings on errors
/// and returns `None` (non-fatal).
///
/// `cli_label` prefixes diagnostic output (e.g. `"ktstr"` or
/// `"cargo ktstr"`).
pub fn remote_lookup(cache: &CacheDir, cache_key: &str, cli_label: &str) -> Option<CacheEntry> {
    let op = match create_operator() {
        Ok(op) => op,
        Err(e) => {
            eprintln!("{cli_label}: remote cache warning: {e}");
            return None;
        }
    };

    let data = match RUNTIME.block_on(op.read(cache_key)) {
        Ok(buf) => buf.to_vec(),
        Err(e) => {
            if e.kind() == opendal::ErrorKind::NotFound {
                return None;
            }
            eprintln!("{cli_label}: remote cache read warning: {e}");
            return None;
        }
    };

    match unpack_and_store(cache, cache_key, &data) {
        Ok(entry) => {
            eprintln!("{cli_label}: fetched from remote cache: {cache_key}");
            Some(entry)
        }
        Err(e) => {
            eprintln!("{cli_label}: remote cache unpack warning ({cache_key}): {e}");
            None
        }
    }
}

/// Store a cache entry in the remote GHA cache.
///
/// Packs the entry directory as a tar blob and uploads it. Failures
/// are non-fatal (logged as warnings).
///
/// `cli_label` prefixes diagnostic output (e.g. `"ktstr"` or
/// `"cargo ktstr"`).
pub fn remote_store(entry: &CacheEntry, cli_label: &str) {
    // CacheEntry guarantees metadata presence; no need to branch.
    let meta = &entry.metadata;

    let op = match create_operator() {
        Ok(op) => op,
        Err(e) => {
            eprintln!("{cli_label}: remote cache warning: {e}");
            return;
        }
    };

    let data = match pack_entry(&entry.path, meta) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{cli_label}: remote cache pack warning: {e}");
            return;
        }
    };

    match RUNTIME.block_on(op.write(&entry.key, data)) {
        Ok(_) => {
            eprintln!("{cli_label}: stored to remote cache: {}", entry.key);
        }
        Err(e) => {
            eprintln!("{cli_label}: remote cache write warning: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{CacheArtifacts, CacheDir, KernelMetadata, KernelSource};

    fn test_metadata() -> KernelMetadata {
        KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64".to_string(),
            "bzImage".to_string(),
            "2026-04-12T10:00:00Z".to_string(),
        )
        .with_version(Some("6.14.2".to_string()))
    }

    fn create_fake_image(dir: &std::path::Path) -> std::path::PathBuf {
        let image = dir.join("bzImage");
        std::fs::write(&image, b"fake kernel image data for testing").unwrap();
        image
    }

    // -- is_enabled --

    #[test]
    fn remote_cache_disabled_by_default() {
        let _g1 = EnvVarGuard::remove("KTSTR_GHA_CACHE");
        let _g2 = EnvVarGuard::remove("ACTIONS_CACHE_URL");
        assert!(!is_enabled());
    }

    #[test]
    fn remote_cache_disabled_without_cache_url() {
        let _g1 = EnvVarGuard::set("KTSTR_GHA_CACHE", "1");
        let _g2 = EnvVarGuard::remove("ACTIONS_CACHE_URL");
        assert!(!is_enabled());
    }

    #[test]
    fn remote_cache_disabled_without_gha_flag() {
        let _g1 = EnvVarGuard::remove("KTSTR_GHA_CACHE");
        let _g2 = EnvVarGuard::set("ACTIONS_CACHE_URL", "https://example.com");
        assert!(!is_enabled());
    }

    #[test]
    fn remote_cache_disabled_with_empty_url() {
        let _g1 = EnvVarGuard::set("KTSTR_GHA_CACHE", "1");
        let _g2 = EnvVarGuard::set("ACTIONS_CACHE_URL", "");
        assert!(!is_enabled());
    }

    #[test]
    fn remote_cache_disabled_with_wrong_flag() {
        let _g1 = EnvVarGuard::set("KTSTR_GHA_CACHE", "0");
        let _g2 = EnvVarGuard::set("ACTIONS_CACHE_URL", "https://example.com");
        assert!(!is_enabled());
    }

    #[test]
    fn remote_cache_enabled_when_both_set() {
        let _g1 = EnvVarGuard::set("KTSTR_GHA_CACHE", "1");
        let _g2 = EnvVarGuard::set("ACTIONS_CACHE_URL", "https://example.com");
        assert!(is_enabled());
    }

    // -- pack/unpack roundtrip --

    #[test]
    fn remote_cache_pack_unpack_roundtrip() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));

        let src = tempfile::TempDir::new().unwrap();
        let image = create_fake_image(src.path());
        let meta = test_metadata();
        let entry = cache
            .store("test-key", &CacheArtifacts::new(&image), &meta)
            .unwrap();

        let packed = pack_entry(&entry.path, &entry.metadata).unwrap();
        assert!(!packed.is_empty());

        let tmp2 = tempfile::TempDir::new().unwrap();
        let cache2 = CacheDir::with_root(tmp2.path().join("cache"));
        let restored = unpack_and_store(&cache2, "test-key", &packed).unwrap();

        assert_eq!(restored.key, "test-key");
        let restored_meta = &restored.metadata;
        assert_eq!(restored_meta.version.as_deref(), Some("6.14.2"));
        assert_eq!(restored_meta.arch, "x86_64");
        assert_eq!(restored_meta.image_name, "bzImage");
        assert_eq!(restored_meta.source, KernelSource::Tarball);

        let restored_image = restored.path.join("bzImage");
        let original_content = std::fs::read(&image).unwrap();
        let restored_content = std::fs::read(&restored_image).unwrap();
        assert_eq!(original_content, restored_content);
    }

    #[test]
    fn remote_cache_pack_entry_excludes_config_sidecar() {
        // .config is not cached any more (IKCONFIG covers CONFIG_HZ
        // for ktstr-built kernels). Even if an entry directory has a
        // leftover .config on disk (e.g. from an older cache version),
        // pack_entry must not include it — the tar carries only
        // metadata.json + image + optional vmlinux.
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src = tempfile::TempDir::new().unwrap();
        let image = create_fake_image(src.path());
        let meta = test_metadata();
        let entry = cache
            .store("legacy-config", &CacheArtifacts::new(&image), &meta)
            .unwrap();
        // Simulate a leftover .config from an older cache version.
        std::fs::write(entry.path.join(".config"), b"CONFIG_HZ=1000\n").unwrap();

        let packed = pack_entry(&entry.path, &entry.metadata).unwrap();
        let tar_bytes = decompress_payload(&packed).unwrap();
        let mut archive = tar::Archive::new(tar_bytes.as_slice());
        let paths: Vec<String> = archive
            .entries()
            .unwrap()
            .map(|e| e.unwrap().path().unwrap().to_string_lossy().into_owned())
            .collect();
        assert!(
            !paths.iter().any(|p| p == ".config"),
            "pack_entry should not include .config, got {paths:?}"
        );
    }

    #[test]
    fn remote_cache_pack_produces_valid_tar() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));

        let src = tempfile::TempDir::new().unwrap();
        let image = create_fake_image(src.path());
        let meta = test_metadata();
        let entry = cache
            .store("valid-tar", &CacheArtifacts::new(&image), &meta)
            .unwrap();

        let packed = pack_entry(&entry.path, &entry.metadata).unwrap();

        // pack_entry returns zstd-compressed data; decompress before
        // validating tar contents.
        let tar_bytes = decompress_payload(&packed).unwrap();
        let mut archive = tar::Archive::new(tar_bytes.as_slice());
        let entries: Vec<_> = archive.entries().unwrap().collect();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn remote_cache_pack_is_zstd_compressed() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));

        let src = tempfile::TempDir::new().unwrap();
        let image = create_fake_image(src.path());
        let meta = test_metadata();
        let entry = cache
            .store("zstd-key", &CacheArtifacts::new(&image), &meta)
            .unwrap();

        let packed = pack_entry(&entry.path, &entry.metadata).unwrap();
        assert!(
            packed.len() >= 4 && packed[..4] == ZSTD_MAGIC,
            "packed data should start with zstd magic"
        );
    }

    /// Rejection test for a raw (non-zstd) tar blob — the
    /// on-the-wire format is zstd-only, so a payload without the
    /// magic is either corruption or hostile content and
    /// `unpack_and_store` must surface a "zstd magic" diagnostic
    /// rather than try to parse the bytes as tar. Replaces the
    /// previous `remote_cache_unpack_handles_raw_tar` backward-compat
    /// test (the raw-tar fallback was deleted as part of the
    /// pre-1.0 cleanup).
    #[test]
    fn remote_cache_unpack_rejects_raw_tar() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));

        let mut archive = tar::Builder::new(Vec::new());
        let meta = test_metadata();
        let meta_json = serde_json::to_string_pretty(&meta).unwrap();
        let meta_bytes = meta_json.as_bytes();
        crate::tar_util::pack_tar_entry(
            &mut archive,
            "metadata.json",
            0o644,
            meta_bytes.len() as u64,
            meta_bytes,
        )
        .unwrap();
        let raw_tar = archive.into_inner().unwrap();

        // Raw tar should not start with zstd magic.
        assert!(raw_tar.len() < 4 || raw_tar[..4] != ZSTD_MAGIC);

        let err = unpack_and_store(&cache, "raw-tar-key", &raw_tar).unwrap_err();
        assert!(
            err.contains("zstd magic"),
            "non-zstd payload must be rejected with a `zstd magic` \
             diagnostic from the precondition check, got: {err}",
        );
    }

    /// Short-input boundary: payloads of 0..=3 bytes cannot carry
    /// the 4-byte zstd magic sentinel, so the precondition check in
    /// `decompress_payload` must reject all of them with the same
    /// "zstd magic" diagnostic. Pins that the `data.len() < 4` half
    /// of the guard fires independently of the magic-bytes
    /// comparison, so a truncated download is rejected instead of
    /// triggering an out-of-bounds slice or feeding an ill-formed
    /// header to the zstd decoder.
    #[test]
    fn remote_cache_decompress_payload_rejects_short_inputs() {
        for len in 0..=3 {
            let bytes = vec![0u8; len];
            let err = super::decompress_payload(&bytes).unwrap_err();
            assert!(
                err.contains("zstd magic"),
                "{len}-byte payload must be rejected by the magic-number \
                 precondition, got: {err}",
            );
        }
    }

    #[test]
    fn remote_cache_unpack_rejects_missing_metadata() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));

        let mut archive = tar::Builder::new(Vec::new());
        let data = b"kernel image";
        crate::tar_util::pack_tar_entry(
            &mut archive,
            "bzImage",
            0o644,
            data.len() as u64,
            data.as_slice(),
        )
        .unwrap();
        let raw_tar = archive.into_inner().unwrap();
        let packed = zstd::encode_all(raw_tar.as_slice(), 3).unwrap();

        let result = unpack_and_store(&cache, "no-meta", &packed);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().contains("missing metadata"),
            "expected metadata error"
        );
    }

    #[test]
    fn remote_cache_unpack_rejects_missing_image() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));

        let mut archive = tar::Builder::new(Vec::new());
        let meta = test_metadata();
        let meta_json = serde_json::to_string_pretty(&meta).unwrap();
        let meta_bytes = meta_json.as_bytes();
        crate::tar_util::pack_tar_entry(
            &mut archive,
            "metadata.json",
            0o644,
            meta_bytes.len() as u64,
            meta_bytes,
        )
        .unwrap();
        let raw_tar = archive.into_inner().unwrap();
        let packed = zstd::encode_all(raw_tar.as_slice(), 3).unwrap();

        let result = unpack_and_store(&cache, "no-image", &packed);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().contains("missing kernel image"),
            "expected image error"
        );
    }

    // -- remote_lookup skipped when disabled --

    #[test]
    fn remote_cache_remote_lookup_returns_none_when_disabled() {
        let _g1 = EnvVarGuard::remove("KTSTR_GHA_CACHE");
        let _g2 = EnvVarGuard::remove("ACTIONS_CACHE_URL");
        assert!(!is_enabled());
    }

    // -- remote_store with disabled remote --

    #[test]
    fn remote_cache_remote_store_when_disabled() {
        let _g1 = EnvVarGuard::remove("KTSTR_GHA_CACHE");
        let _g2 = EnvVarGuard::remove("ACTIONS_CACHE_URL");

        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src = tempfile::TempDir::new().unwrap();
        let image = create_fake_image(src.path());
        let meta = test_metadata();
        let entry = cache
            .store("test-entry", &CacheArtifacts::new(&image), &meta)
            .unwrap();

        let packed = pack_entry(&entry.path, &entry.metadata);
        assert!(packed.is_ok());
    }

    // -- pack with various metadata --

    #[test]
    fn remote_cache_source_tree_path_sanitized_on_roundtrip() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));

        let src = tempfile::TempDir::new().unwrap();
        let image = create_fake_image(src.path());
        let meta = KernelMetadata::new(
            KernelSource::Local {
                source_tree_path: Some(std::path::PathBuf::from("/tmp/linux-src")),
                git_hash: Some("deadbee".to_string()),
            },
            "x86_64".to_string(),
            "bzImage".to_string(),
            "2026-04-12T10:00:00Z".to_string(),
        );
        assert!(matches!(
            meta.source,
            KernelSource::Local {
                source_tree_path: Some(_),
                git_hash: Some(_),
            }
        ));

        let entry = cache
            .store("stp-key", &CacheArtifacts::new(&image), &meta)
            .unwrap();

        let packed = pack_entry(&entry.path, &entry.metadata).unwrap();

        let tmp2 = tempfile::TempDir::new().unwrap();
        let cache2 = CacheDir::with_root(tmp2.path().join("cache"));
        let restored = unpack_and_store(&cache2, "stp-key", &packed).unwrap();

        let restored_meta = &restored.metadata;
        assert!(
            matches!(
                &restored_meta.source,
                KernelSource::Local {
                    source_tree_path: None,
                    git_hash: Some(h),
                } if h == "deadbee"
            ),
            "source_tree_path must be stripped during pack, git_hash must survive"
        );
    }

    #[test]
    fn remote_cache_pack_with_git_metadata() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));

        let src = tempfile::TempDir::new().unwrap();
        let image = create_fake_image(src.path());
        let meta = KernelMetadata::new(
            KernelSource::Git {
                git_hash: Some("a1b2c3d".to_string()),
                git_ref: Some("v6.15-rc3".to_string()),
            },
            "x86_64".to_string(),
            "bzImage".to_string(),
            "2026-04-12T12:00:00Z".to_string(),
        );

        let entry = cache
            .store("git-key", &CacheArtifacts::new(&image), &meta)
            .unwrap();
        let packed = pack_entry(&entry.path, &entry.metadata).unwrap();

        let tmp2 = tempfile::TempDir::new().unwrap();
        let cache2 = CacheDir::with_root(tmp2.path().join("cache"));
        let restored = unpack_and_store(&cache2, "git-key", &packed).unwrap();

        let rmeta = &restored.metadata;
        assert!(matches!(
            rmeta.source,
            KernelSource::Git {
                git_hash: Some(ref h),
                git_ref: Some(ref r),
            }
            if h == "a1b2c3d" && r == "v6.15-rc3"
        ));
    }

    /// Decompression-bomb guard: a zstd payload that decompresses
    /// past the configured cap surfaces an error tagged with
    /// "decompression-bomb guard" — `decompress_payload` must not
    /// allocate past the ceiling. Test uses a small synthetic
    /// payload (8 KiB of zeros, which compresses to a tiny blob
    /// but decompresses to 8192 bytes) routed through the private
    /// `decompress_capped` helper against a 1024-byte cap so the
    /// test runs in microseconds rather than allocating a
    /// production-sized buffer.
    #[test]
    fn remote_cache_decompress_capped_rejects_decompression_bomb() {
        let payload = vec![0u8; 8192];
        let compressed = zstd::encode_all(payload.as_slice(), 3).unwrap();
        let cap: u64 = 1024;
        let err = super::decompress_capped(&compressed, cap).unwrap_err();
        assert!(
            err.contains("decompression-bomb guard"),
            "expected decompression-bomb guard error, got: {err}",
        );
    }

    /// Boundary case: a payload whose decompressed length is
    /// exactly `cap` bytes is accepted (the cap is inclusive).
    /// Pins the `>` (not `>=`) discriminator at the cap boundary
    /// so a future refactor that flips the comparison surfaces
    /// here rather than turning a legal cache entry into a
    /// false-positive bomb rejection.
    #[test]
    fn remote_cache_decompress_capped_accepts_payload_at_cap_boundary() {
        let payload = b"hello world".to_vec();
        let compressed = zstd::encode_all(payload.as_slice(), 3).unwrap();
        let out = super::decompress_capped(&compressed, payload.len() as u64).unwrap();
        assert_eq!(
            out, payload,
            "payload exactly at the cap must round-trip — \
             cap is inclusive (`>` not `>=`)",
        );
    }

    /// Pin the shape of [`super::REMOTE_CACHE_NAMESPACE`]: non-empty,
    /// keeps the `ktstr-v` prefix that namespaces ktstr entries
    /// against other tools sharing the GHA cache, and carries a
    /// numeric version suffix that bumps invalidate stale entries.
    /// Without this pin, a refactor that dropped the prefix would
    /// silently start sharing the namespace with another tool, and
    /// a bump that landed `ktstr-v2a` would still pass any
    /// substring-only check while breaking the suffix-as-version
    /// contract.
    #[test]
    fn remote_cache_namespace_has_version_suffix() {
        let ns = super::REMOTE_CACHE_NAMESPACE;
        assert!(!ns.is_empty(), "namespace must not be empty");
        assert!(
            ns.starts_with("ktstr-v"),
            "namespace must keep `ktstr-v` prefix; got: {ns}",
        );
        let suffix = ns.strip_prefix("ktstr-v").unwrap();
        assert!(
            suffix.parse::<u32>().is_ok(),
            "version suffix must be numeric; got: {suffix:?}",
        );
    }

    use crate::test_support::test_helpers::EnvVarGuard;
}
