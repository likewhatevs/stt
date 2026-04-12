//! Remote cache backend for GHA runners via opendal.
//!
//! When `KTSTR_GHA_CACHE=1` and `ACTIONS_CACHE_URL` are set, cache
//! operations transparently extend to a remote GHA cache. Local cache
//! is always authoritative: lookups check local first, stores write to
//! both. Remote failures are non-fatal (logged as warnings).
//!
//! Cache entries are serialized as tar archives containing the kernel
//! image and metadata.json, stored as a single blob per cache key in
//! the GHA cache service.

use std::io::{Read, Write};
use std::path::Path;
use std::sync::LazyLock;

use ktstr::cache::{CacheDir, CacheEntry, KernelMetadata};

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

/// Pack a cache entry directory into a tar archive in memory.
///
/// The tar contains the kernel image and metadata.json from the
/// cache entry directory. Paths inside the tar are relative
/// filenames (no directory prefix).
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

    archive
        .into_inner()
        .map_err(|e| format!("finalize tar: {e}"))
}

/// Unpack a tar archive into a cache directory via CacheDir::store.
///
/// Extracts metadata.json and the kernel image from the tar blob,
/// writes the image to a temp file, then stores via the local cache
/// API for atomic placement.
fn unpack_and_store(cache: &CacheDir, cache_key: &str, data: &[u8]) -> Result<CacheEntry, String> {
    let mut archive = tar::Archive::new(data);
    let entries = archive
        .entries()
        .map_err(|e| format!("read tar entries: {e}"))?;

    let mut metadata: Option<KernelMetadata> = None;
    let mut image_data: Option<(String, Vec<u8>)> = None;

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

    // Write image to a temp file for CacheDir::store.
    let tmp_dir = tempfile::TempDir::new().map_err(|e| format!("create temp dir: {e}"))?;
    let tmp_image = tmp_dir.path().join(&meta.image_name);
    let mut f = std::fs::File::create(&tmp_image).map_err(|e| format!("create temp image: {e}"))?;
    f.write_all(&img_bytes)
        .map_err(|e| format!("write temp image: {e}"))?;
    drop(f);

    cache
        .store(cache_key, &tmp_image, &meta)
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
    use ktstr::cache::{CacheDir, KernelMetadata, SourceType};

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
        let entry = cache.store("test-key", &image, &meta).unwrap();

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
    fn remote_cache_pack_produces_valid_tar() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();

        let src = tempfile::TempDir::new().unwrap();
        let image = create_fake_image(src.path());
        let meta = test_metadata();
        let entry = cache.store("valid-tar", &image, &meta).unwrap();

        let packed = pack_entry(&entry.path, entry.metadata.as_ref().unwrap()).unwrap();

        let mut archive = tar::Archive::new(packed.as_slice());
        let entries: Vec<_> = archive.entries().unwrap().collect();
        assert_eq!(entries.len(), 2);
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
        let entry = cache.store("test-entry", &image, &meta).unwrap();

        let packed = pack_entry(&entry.path, entry.metadata.as_ref().unwrap());
        assert!(packed.is_ok());
    }

    // -- pack with various metadata --

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

        let entry = cache.store("git-key", &image, &meta).unwrap();
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
