//! Remote cache backend for GHA runners via opendal.
//!
//! When `KTSTR_GHA_CACHE=1` and `ACTIONS_CACHE_URL` are set, cache
//! operations transparently extend to a remote GHA cache. Local cache
//! is always authoritative: lookups check local first, stores write to
//! both. Remote failures are non-fatal (logged as warnings).
//!
//! Cache entries are serialized as tar archives containing the kernel
//! image, vmlinux (if present), .config (if present), and
//! metadata.json, stored as a single blob per cache key in the GHA
//! cache service.

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

/// Create an opendal operator for the GHA cache service.
///
/// Uses `ACTIONS_CACHE_URL` and `ACTIONS_RUNTIME_TOKEN` from the
/// environment (set automatically by the GHA runner). The `version`
/// field namespaces cache entries to avoid collisions with other
/// tools using the same GHA cache.
fn create_operator() -> Result<opendal::Operator, String> {
    let builder = opendal::services::Ghac::default()
        .root("/")
        .version("ktstr");

    opendal::Operator::new(builder)
        .map_err(|e| format!("create ghac operator: {e}"))
        .map(|b| b.finish())
}

/// Zstd magic number (first 4 bytes of any zstd frame).
const ZSTD_MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];

/// Pack a cache entry directory into a tar archive in memory.
///
/// The tar contains the kernel image, vmlinux (if present), .config
/// (if present), and metadata.json from the cache entry directory.
/// Paths inside the tar are relative filenames (no directory prefix).
///
/// The tar is then compressed with zstd before upload.
/// [`unpack_and_store`] detects the zstd magic number on download
/// and decompresses transparently, falling back to raw tar for
/// entries stored before compression was added.
fn pack_entry(entry_dir: &Path, metadata: &KernelMetadata) -> Result<Vec<u8>, String> {
    let mut archive = tar::Builder::new(Vec::new());

    // Null out source_tree_path before serializing — it contains
    // local filesystem paths that must not leak to remote storage.
    let mut meta_sanitized = metadata.clone();
    meta_sanitized.source_tree_path = None;

    // Add metadata.json.
    let meta_json = serde_json::to_string_pretty(&meta_sanitized)
        .map_err(|e| format!("serialize metadata: {e}"))?;
    let meta_bytes = meta_json.as_bytes();
    let mut header = tar::Header::new_gnu();
    header.set_size(meta_bytes.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();
    archive
        .append_data(&mut header, "metadata.json", meta_bytes)
        .map_err(|e| format!("tar append metadata: {e}"))?;

    // Add kernel image.
    let image_path = entry_dir.join(&metadata.image_name);
    let mut image_file = std::fs::File::open(&image_path)
        .map_err(|e| format!("open image {}: {e}", image_path.display()))?;
    let image_size = image_file
        .metadata()
        .map_err(|e| format!("image metadata: {e}"))?
        .len();
    let mut header = tar::Header::new_gnu();
    header.set_size(image_size);
    header.set_mode(0o644);
    header.set_cksum();
    archive
        .append_data(&mut header, &metadata.image_name, &mut image_file)
        .map_err(|e| format!("tar append image: {e}"))?;

    // Add vmlinux if present (BTF source for build.rs).
    let vmlinux_path = entry_dir.join("vmlinux");
    if let Ok(mut vmlinux_file) = std::fs::File::open(&vmlinux_path) {
        let vmlinux_size = vmlinux_file
            .metadata()
            .map_err(|e| format!("vmlinux metadata: {e}"))?
            .len();
        let mut header = tar::Header::new_gnu();
        header.set_size(vmlinux_size);
        header.set_mode(0o644);
        header.set_cksum();
        archive
            .append_data(&mut header, "vmlinux", &mut vmlinux_file)
            .map_err(|e| format!("tar append vmlinux: {e}"))?;
    }

    // Add .config if present (CONFIG_HZ resolution).
    let config_path = entry_dir.join(".config");
    if let Ok(mut config_file) = std::fs::File::open(&config_path) {
        let config_size = config_file
            .metadata()
            .map_err(|e| format!(".config metadata: {e}"))?
            .len();
        let mut header = tar::Header::new_gnu();
        header.set_size(config_size);
        header.set_mode(0o644);
        header.set_cksum();
        archive
            .append_data(&mut header, ".config", &mut config_file)
            .map_err(|e| format!("tar append .config: {e}"))?;
    }

    let tar_bytes = archive
        .into_inner()
        .map_err(|e| format!("finalize tar: {e}"))?;

    // Compress with zstd (level 3: good ratio at fast speed).
    zstd::encode_all(tar_bytes.as_slice(), 3).map_err(|e| format!("zstd compress: {e}"))
}

/// Decompress data if it starts with the zstd magic number,
/// otherwise return as-is (backward compatibility with
/// uncompressed tar entries written before zstd was added).
fn maybe_decompress(data: &[u8]) -> Result<Vec<u8>, String> {
    if data.len() >= 4 && data[..4] == ZSTD_MAGIC {
        zstd::decode_all(data).map_err(|e| format!("zstd decompress: {e}"))
    } else {
        Ok(data.to_vec())
    }
}

/// Unpack a tar archive into a cache directory via CacheDir::store.
///
/// Extracts metadata.json, the kernel image, vmlinux (if present),
/// and .config (if present) from the tar blob, writes them to temp
/// files, then stores via the local cache API for atomic placement.
fn unpack_and_store(cache: &CacheDir, cache_key: &str, data: &[u8]) -> Result<CacheEntry, String> {
    let tar_bytes = maybe_decompress(data)?;
    let mut archive = tar::Archive::new(tar_bytes.as_slice());
    let entries = archive
        .entries()
        .map_err(|e| format!("read tar entries: {e}"))?;

    let mut metadata: Option<KernelMetadata> = None;
    let mut image_data: Option<(String, Vec<u8>)> = None;
    let mut vmlinux_data: Option<Vec<u8>> = None;
    let mut config_data: Option<Vec<u8>> = None;

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
        } else if path == ".config" {
            let mut data = Vec::new();
            entry
                .read_to_end(&mut data)
                .map_err(|e| format!("read .config from tar: {e}"))?;
            config_data = Some(data);
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

    // Write image, vmlinux, and .config to temp files for CacheDir::store.
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

    let tmp_config_path;
    let config_ref = if let Some(ref cfg_bytes) = config_data {
        tmp_config_path = tmp_dir.path().join(".config");
        let mut cf = std::fs::File::create(&tmp_config_path)
            .map_err(|e| format!("create temp .config: {e}"))?;
        cf.write_all(cfg_bytes)
            .map_err(|e| format!("write temp .config: {e}"))?;
        drop(cf);
        Some(tmp_config_path.as_path())
    } else {
        None
    };

    cache
        .store(cache_key, &tmp_image, vmlinux_ref, config_ref, &meta)
        .map_err(|e| format!("local cache store: {e}"))
}

/// Look up a cache key in the remote GHA cache.
///
/// On hit, downloads the tar blob and unpacks it into the local
/// cache via `CacheDir::store`. Returns the local `CacheEntry` on
/// success. Returns `None` on remote miss. Logs warnings on errors
/// and returns `None` (non-fatal).
pub fn remote_lookup(cache: &CacheDir, cache_key: &str) -> Option<CacheEntry> {
    let op = match create_operator() {
        Ok(op) => op,
        Err(e) => {
            eprintln!("cargo-ktstr: remote cache warning: {e}");
            return None;
        }
    };

    let data = match RUNTIME.block_on(op.read(cache_key)) {
        Ok(buf) => buf.to_vec(),
        Err(e) => {
            if e.kind() == opendal::ErrorKind::NotFound {
                return None;
            }
            eprintln!("cargo-ktstr: remote cache read warning: {e}");
            return None;
        }
    };

    match unpack_and_store(cache, cache_key, &data) {
        Ok(entry) => {
            eprintln!("cargo-ktstr: fetched from remote cache: {cache_key}");
            Some(entry)
        }
        Err(e) => {
            eprintln!("cargo-ktstr: remote cache unpack warning: {e}");
            None
        }
    }
}

/// Store a cache entry in the remote GHA cache.
///
/// Packs the entry directory as a tar blob and uploads it. Failures
/// are non-fatal (logged as warnings).
pub fn remote_store(entry: &CacheEntry) {
    let meta = match &entry.metadata {
        Some(m) => m,
        None => {
            eprintln!("cargo-ktstr: remote cache store skipped: no metadata");
            return;
        }
    };

    let op = match create_operator() {
        Ok(op) => op,
        Err(e) => {
            eprintln!("cargo-ktstr: remote cache warning: {e}");
            return;
        }
    };

    let data = match pack_entry(&entry.path, meta) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("cargo-ktstr: remote cache pack warning: {e}");
            return;
        }
    };

    match RUNTIME.block_on(op.write(&entry.key, data)) {
        Ok(_) => {
            eprintln!("cargo-ktstr: stored to remote cache: {}", entry.key);
        }
        Err(e) => {
            eprintln!("cargo-ktstr: remote cache write warning: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{CacheDir, KernelMetadata, SourceType};

    fn test_metadata() -> KernelMetadata {
        KernelMetadata::new(
            SourceType::Tarball,
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
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();

        let src = tempfile::TempDir::new().unwrap();
        let image = create_fake_image(src.path());
        let meta = test_metadata();
        let entry = cache.store("test-key", &image, None, None, &meta).unwrap();

        let packed = pack_entry(&entry.path, entry.metadata.as_ref().unwrap()).unwrap();
        assert!(!packed.is_empty());

        let tmp2 = tempfile::TempDir::new().unwrap();
        let cache2 = CacheDir::with_root(tmp2.path().join("cache")).unwrap();
        let restored = unpack_and_store(&cache2, "test-key", &packed).unwrap();

        assert_eq!(restored.key, "test-key");
        let restored_meta = restored.metadata.unwrap();
        assert_eq!(restored_meta.version.as_deref(), Some("6.14.2"));
        assert_eq!(restored_meta.arch, "x86_64");
        assert_eq!(restored_meta.image_name, "bzImage");
        assert_eq!(restored_meta.source, SourceType::Tarball);

        let restored_image = restored.path.join("bzImage");
        let original_content = std::fs::read(&image).unwrap();
        let restored_content = std::fs::read(&restored_image).unwrap();
        assert_eq!(original_content, restored_content);
    }

    #[test]
    fn remote_cache_pack_unpack_roundtrip_with_config() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();

        let src = tempfile::TempDir::new().unwrap();
        let image = create_fake_image(src.path());
        let config = src.path().join(".config");
        let config_content = b"CONFIG_HZ=1000\nCONFIG_SCHED_CLASS_EXT=y\n";
        std::fs::write(&config, config_content).unwrap();
        let meta = test_metadata();
        let entry = cache
            .store("config-key", &image, None, Some(&config), &meta)
            .unwrap();

        let packed = pack_entry(&entry.path, entry.metadata.as_ref().unwrap()).unwrap();

        let tmp2 = tempfile::TempDir::new().unwrap();
        let cache2 = CacheDir::with_root(tmp2.path().join("cache")).unwrap();
        let restored = unpack_and_store(&cache2, "config-key", &packed).unwrap();

        assert_eq!(restored.key, "config-key");
        // .config must exist in restored entry with original contents.
        let restored_config = restored.path.join(".config");
        assert!(
            restored_config.exists(),
            ".config missing from restored cache entry"
        );
        let restored_content = std::fs::read(&restored_config).unwrap();
        assert_eq!(restored_content, config_content);
    }

    #[test]
    fn remote_cache_pack_produces_valid_tar() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();

        let src = tempfile::TempDir::new().unwrap();
        let image = create_fake_image(src.path());
        let meta = test_metadata();
        let entry = cache.store("valid-tar", &image, None, None, &meta).unwrap();

        let packed = pack_entry(&entry.path, entry.metadata.as_ref().unwrap()).unwrap();

        // pack_entry returns zstd-compressed data; decompress before
        // validating tar contents.
        let tar_bytes = maybe_decompress(&packed).unwrap();
        let mut archive = tar::Archive::new(tar_bytes.as_slice());
        let entries: Vec<_> = archive.entries().unwrap().collect();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn remote_cache_pack_is_zstd_compressed() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();

        let src = tempfile::TempDir::new().unwrap();
        let image = create_fake_image(src.path());
        let meta = test_metadata();
        let entry = cache.store("zstd-key", &image, None, None, &meta).unwrap();

        let packed = pack_entry(&entry.path, entry.metadata.as_ref().unwrap()).unwrap();
        assert!(
            packed.len() >= 4 && packed[..4] == ZSTD_MAGIC,
            "packed data should start with zstd magic"
        );
    }

    #[test]
    fn remote_cache_unpack_handles_raw_tar() {
        // Verify backward compatibility: unpack_and_store accepts
        // uncompressed tar data (entries written before zstd was added).
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();

        let mut archive = tar::Builder::new(Vec::new());
        let meta = test_metadata();
        let meta_json = serde_json::to_string_pretty(&meta).unwrap();
        let meta_bytes = meta_json.as_bytes();
        let mut header = tar::Header::new_gnu();
        header.set_size(meta_bytes.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        archive
            .append_data(&mut header, "metadata.json", meta_bytes)
            .unwrap();

        let img_data = b"fake kernel image";
        let mut header = tar::Header::new_gnu();
        header.set_size(img_data.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        archive
            .append_data(&mut header, "bzImage", img_data.as_slice())
            .unwrap();
        let raw_tar = archive.into_inner().unwrap();

        // Raw tar should not start with zstd magic.
        assert!(raw_tar.len() < 4 || raw_tar[..4] != ZSTD_MAGIC);

        let restored = unpack_and_store(&cache, "raw-tar-key", &raw_tar).unwrap();
        assert_eq!(restored.key, "raw-tar-key");
        let rmeta = restored.metadata.unwrap();
        assert_eq!(rmeta.version.as_deref(), Some("6.14.2"));
    }

    #[test]
    fn remote_cache_unpack_rejects_missing_metadata() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();

        let mut archive = tar::Builder::new(Vec::new());
        let data = b"kernel image";
        let mut header = tar::Header::new_gnu();
        header.set_size(data.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        archive
            .append_data(&mut header, "bzImage", data.as_slice())
            .unwrap();
        let packed = archive.into_inner().unwrap();

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
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();

        let mut archive = tar::Builder::new(Vec::new());
        let meta = test_metadata();
        let meta_json = serde_json::to_string_pretty(&meta).unwrap();
        let meta_bytes = meta_json.as_bytes();
        let mut header = tar::Header::new_gnu();
        header.set_size(meta_bytes.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        archive
            .append_data(&mut header, "metadata.json", meta_bytes)
            .unwrap();
        let packed = archive.into_inner().unwrap();

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
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();
        let src = tempfile::TempDir::new().unwrap();
        let image = create_fake_image(src.path());
        let meta = test_metadata();
        let entry = cache
            .store("test-entry", &image, None, None, &meta)
            .unwrap();

        let packed = pack_entry(&entry.path, entry.metadata.as_ref().unwrap());
        assert!(packed.is_ok());
    }

    // -- pack with various metadata --

    #[test]
    fn remote_cache_source_tree_path_sanitized_on_roundtrip() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();

        let src = tempfile::TempDir::new().unwrap();
        let image = create_fake_image(src.path());
        let meta =
            test_metadata().with_source_tree_path(Some(std::path::PathBuf::from("/tmp/linux-src")));
        assert!(meta.source_tree_path.is_some());

        let entry = cache.store("stp-key", &image, None, None, &meta).unwrap();

        let packed = pack_entry(&entry.path, entry.metadata.as_ref().unwrap()).unwrap();

        let tmp2 = tempfile::TempDir::new().unwrap();
        let cache2 = CacheDir::with_root(tmp2.path().join("cache")).unwrap();
        let restored = unpack_and_store(&cache2, "stp-key", &packed).unwrap();

        let restored_meta = restored.metadata.unwrap();
        assert!(
            restored_meta.source_tree_path.is_none(),
            "source_tree_path must be stripped during pack"
        );
    }

    #[test]
    fn remote_cache_pack_with_git_metadata() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();

        let src = tempfile::TempDir::new().unwrap();
        let image = create_fake_image(src.path());
        let meta = KernelMetadata::new(
            SourceType::Git,
            "x86_64".to_string(),
            "bzImage".to_string(),
            "2026-04-12T12:00:00Z".to_string(),
        )
        .with_git_hash(Some("a1b2c3d".to_string()))
        .with_git_ref(Some("v6.15-rc3".to_string()));

        let entry = cache.store("git-key", &image, None, None, &meta).unwrap();
        let packed = pack_entry(&entry.path, entry.metadata.as_ref().unwrap()).unwrap();

        let tmp2 = tempfile::TempDir::new().unwrap();
        let cache2 = CacheDir::with_root(tmp2.path().join("cache")).unwrap();
        let restored = unpack_and_store(&cache2, "git-key", &packed).unwrap();

        let rmeta = restored.metadata.unwrap();
        assert_eq!(rmeta.source, SourceType::Git);
        assert_eq!(rmeta.git_hash.as_deref(), Some("a1b2c3d"));
        assert_eq!(rmeta.git_ref.as_deref(), Some("v6.15-rc3"));
    }

    // -- EnvVarGuard (same pattern as cache.rs tests) --

    struct EnvVarGuard {
        key: String,
        original: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &str, value: &str) -> Self {
            let original = std::env::var(key).ok();
            // SAFETY: nextest runs each test in its own process.
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
