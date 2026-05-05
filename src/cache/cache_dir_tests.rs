//! Tests for [`super`] (the `cache_dir` module).
//!
//! Source-shared into `cache_dir.rs` via `#[path]` so the test
//! body becomes the `cache_dir::tests` submodule and `use super::*`
//! preserves access to the parent's private items
//! (`should_emit_unstripped_warn`, `store_exclusive_lock_timeout`,
//! `STORE_EXCLUSIVE_LOCK_*`, `lookup_silent`, etc.). Split out of
//! `cache_dir.rs` purely to keep that file under the per-file
//! line cap; the tests have not changed.

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
            !name.starts_with(".tmp-"),
            "unexpected leftover .tmp- directory under cache_root: {name}"
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
    // Per-version observation counters strengthen the prior
    // assertion that "lookup_observed > 0": without splitting
    // the count, a reader that ONLY ever sees content_a (e.g.
    // because it raced through every write window before any
    // B publish landed) would still let the test pass. The
    // split counts let the test surface whether the race
    // window was actually exercised across BOTH writer
    // versions.
    let observed_a = Arc::new(AtomicUsize::new(0));
    let observed_b = Arc::new(AtomicUsize::new(0));

    let reader_count = 4;
    let mut readers = Vec::with_capacity(reader_count);
    for _ in 0..reader_count {
        let cache = Arc::clone(&cache);
        let stop = Arc::clone(&stop);
        let lookups_observed = Arc::clone(&lookups_observed);
        let violations = Arc::clone(&atomicity_violations);
        let observed_a = Arc::clone(&observed_a);
        let observed_b = Arc::clone(&observed_b);
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
                if bytes == expected_a {
                    observed_a.fetch_add(1, Ordering::Relaxed);
                } else if bytes == expected_b {
                    observed_b.fetch_add(1, Ordering::Relaxed);
                } else {
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

    // Soft observation: under realistic scheduling, readers
    // SHOULD observe BOTH content_a and content_b. Logged not
    // asserted — under adversarial scheduling, all reader
    // wakeups could land between writer transitions and miss
    // one version entirely. The hard atomicity assertion
    // above guards correctness; this soft check surfaces
    // whether the race window was actually exercised across
    // both versions. Print to stderr so an operator running
    // with --no-capture sees the coverage signal without
    // bricking the test run.
    let saw_a = observed_a.load(Ordering::Relaxed);
    let saw_b = observed_b.load(Ordering::Relaxed);
    if saw_a == 0 || saw_b == 0 {
        eprintln!(
            "cache_dir_store_atomic_under_concurrent_readers: \
             one writer version was never observed by readers \
             (saw_a={saw_a}, saw_b={saw_b}). Atomicity invariant \
             still holds; coverage of the race window is \
             probabilistic under scheduling pressure.",
        );
    }

    let final_entry = cache.lookup("atomic-key").expect("entry must exist");
    let final_bytes = fs::read(final_entry.image_path()).unwrap();
    assert!(
        final_bytes == content_a || final_bytes == content_b,
        "final image must match one of the writer's versions",
    );
    for dirent in fs::read_dir(&cache_root).unwrap() {
        let name = dirent.unwrap().file_name().to_string_lossy().into_owned();
        assert!(
            !name.starts_with(".tmp-"),
            "unexpected leftover .tmp- directory under cache_root: {name}",
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

// -- should_emit_unstripped_warn dedup gate --
//
// The gate combines `should_warn_unstripped` (does this entry
// need warning at all?) with the once-per-key dedup set
// (have we already warned for this key in this process?).
// Tests use a fresh per-test `Mutex<HashSet<String>>` so the
// process-wide `warned_keys()` static is not polluted across
// unit tests.

/// Helper: build a CacheEntry with `has_vmlinux=true,
/// vmlinux_stripped=false` (the "stale entry needs warn" shape)
/// under a caller-chosen cache_key, so a single test can drive
/// the dedup gate against multiple keys.
fn make_stale_entry_with_key(key: &str) -> CacheEntry {
    let mut meta = KernelMetadata::new(
        super::super::metadata::KernelSource::Tarball,
        "x86_64".to_string(),
        "bzImage".to_string(),
        "2026-04-24T12:00:00Z".to_string(),
    );
    meta.set_has_vmlinux(true);
    meta.set_vmlinux_stripped(false);
    CacheEntry {
        key: key.to_string(),
        path: PathBuf::from("/nonexistent/entry"),
        metadata: meta,
    }
}

/// First call against a fresh dedup set with a stale entry must
/// return `true` — the warn has not fired yet for this key, so
/// the caller should emit it now.
#[test]
fn should_emit_unstripped_warn_first_call_returns_true() {
    let set = Mutex::new(HashSet::new());
    let entry = make_stale_entry_with_key("first-call-key");
    assert!(
        should_emit_unstripped_warn(&entry, &set),
        "first call against an empty set must return true so the \
         caller emits the warn",
    );
    // After the call the key must be recorded.
    let recorded = set.lock().unwrap().contains("first-call-key");
    assert!(
        recorded,
        "first call must insert the key into the dedup set so \
         subsequent calls suppress",
    );
}

/// Second call with the SAME key against the same set must
/// return `false` — the dedup gate suppresses.
#[test]
fn should_emit_unstripped_warn_repeat_call_same_key_returns_false() {
    let set = Mutex::new(HashSet::new());
    let entry = make_stale_entry_with_key("dedup-key");
    let first = should_emit_unstripped_warn(&entry, &set);
    let second = should_emit_unstripped_warn(&entry, &set);
    assert!(first, "first call must return true (warn fires)");
    assert!(
        !second,
        "second call for the same key must return false (dedup \
         suppression — the warn already fired in this process)",
    );
}

/// A different key against the SAME set must still return
/// `true` — dedup is per-key, not global. Pins that two stale
/// entries with distinct keys each get their own warn.
#[test]
fn should_emit_unstripped_warn_distinct_keys_each_warn_once() {
    let set = Mutex::new(HashSet::new());
    let entry_a = make_stale_entry_with_key("key-a");
    let entry_b = make_stale_entry_with_key("key-b");
    assert!(
        should_emit_unstripped_warn(&entry_a, &set),
        "key-a's first call must return true",
    );
    assert!(
        should_emit_unstripped_warn(&entry_b, &set),
        "key-b is distinct from key-a, so its first call must \
         also return true (per-key dedup, not global)",
    );
    // Now repeat each — both should return false.
    assert!(
        !should_emit_unstripped_warn(&entry_a, &set),
        "key-a's second call must dedup",
    );
    assert!(
        !should_emit_unstripped_warn(&entry_b, &set),
        "key-b's second call must dedup",
    );
}

/// Even if the dedup set is empty, an entry that does NOT need
/// warning (e.g. has_vmlinux=false, or vmlinux_stripped=true)
/// must return `false` and must NOT be inserted into the set —
/// the dedup set should only track keys for which the warn
/// actually fires, not every key the gate looks at.
#[test]
fn should_emit_unstripped_warn_no_warn_needed_skips_dedup_insert() {
    let set = Mutex::new(HashSet::new());
    // has_vmlinux=false → no warn needed.
    let no_vmlinux = make_warn_test_entry(false, false);
    assert!(
        !should_emit_unstripped_warn(&no_vmlinux, &set),
        "an entry that doesn't need warning must return false",
    );
    assert!(
        set.lock().unwrap().is_empty(),
        "no-warn-needed path must NOT pollute the dedup set; \
         the gate must short-circuit before the insert",
    );

    // has_vmlinux=true + vmlinux_stripped=true → no warn needed.
    let stripped = make_warn_test_entry(true, true);
    assert!(
        !should_emit_unstripped_warn(&stripped, &set),
        "an entry whose vmlinux WAS stripped must return false",
    );
    assert!(
        set.lock().unwrap().is_empty(),
        "stripped-vmlinux entry must also leave the dedup set \
         empty — only stale entries get recorded",
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
    let guard_b = cache_b.try_acquire_exclusive_lock("shared-name").expect(
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

/// End-to-end: race peers carrying a MIX of recheck-equivalent
/// and recheck-different content under the same cache key. The
/// content-defining axis is `config_hash` — peers split into two
/// groups: half publish hash "A", half publish hash "B". Each
/// group's peers are recheck-equivalent within the group, so
/// only one head writer per group pays the publish cost — but
/// the cross-group peers must NOT collapse into one another's
/// state. The final on-disk entry must match exactly one of the
/// two distinct content states (whichever group's late writer
/// won the rename), and every per-peer return must be one of the
/// two on-disk states (never torn, never a third).
///
/// Each peer's `built_at` is the secondary observable: the
/// recheck path returns the EXISTING entry verbatim, so a peer
/// that short-circuits inherits the head writer's built_at
/// rather than its own. Peer i carries
/// `built_at = 2026-04-12T10:00:{i:02}Z`, so group A's
/// timestamps are `{:00, :02, :04, :06}` and group B's are
/// `{:01, :03, :05, :07}` — DISJOINT sets. A group-A peer's
/// returned built_at must therefore live inside group A's input
/// set: cross-group bleed (group A peer returning a group-B
/// timestamp) would prove the recheck collapsed across the
/// content-divergence axis, which is the bug this test guards.
///
/// This pins THREE properties:
///   1. recheck-bypass on cross-group divergence — a recheck
///      miss MUST proceed to a real publish, not silently fold
///      into the prior group's state.
///   2. atomic publish across overlapping writers — the cache
///      root never carries a half-written state mid-race.
///   3. recheck-collapse on within-group equivalence — the
///      built_at axis pins that late peers borrow the head
///      writer's timestamp from THEIR group's input set, never
///      from the other group.
///
/// Note on non-determinism: scheduling decides whether multiple
///  peers in a group successfully short-circuit on the same
///  head writer (collapse fires) or whether cross-group
///  overwrites force every group-X peer to publish its own
///  timestamp afresh. The test asserts the deterministic
///  invariants (cross-group separation, valid input-set
///  membership) as hard checks, and the probabilistic
///  collapse-fired observation as a softer informational
///  check that can fail under adversarial scheduling without
///  bricking the test — pinning the WEAKER invariant is the
///  correct trade per CLAUDE.md (probabilistic flakes mask
///  real regressions).
///
/// Sibling of `store_in_lock_recheck_serialises_concurrent_peers`
/// (which exercises only recheck-equivalent peers). Together the
/// two tests cover both branches of `cache_content_matches`.
#[test]
fn store_in_lock_recheck_mixed_content_peers_publish_one_per_group() {
    use std::collections::BTreeSet;
    use std::sync::Arc;
    use std::sync::Barrier;
    use std::thread;
    let tmp = TempDir::new().unwrap();
    let cache = Arc::new(CacheDir::with_root(tmp.path().join("cache")));
    let src_dir = TempDir::new().unwrap();
    let image = src_dir.path().join("bzImage");
    std::fs::write(&image, b"shared image bytes").unwrap();

    const PEER_COUNT: usize = 8;
    // Disjoint per-group input timestamp sets — group A holds
    // every even-index `:NN`, group B every odd-index. The hard
    // cross-group separation invariant rides on this disjointness.
    let group_a_inputs: BTreeSet<String> = (0..PEER_COUNT)
        .filter(|i| i % 2 == 0)
        .map(|i| format!("2026-04-12T10:00:{i:02}Z"))
        .collect();
    let group_b_inputs: BTreeSet<String> = (0..PEER_COUNT)
        .filter(|i| i % 2 == 1)
        .map(|i| format!("2026-04-12T10:00:{i:02}Z"))
        .collect();
    assert!(
        group_a_inputs.is_disjoint(&group_b_inputs),
        "test setup invariant: per-group input timestamps must \
         be disjoint so the cross-group bleed assertion below is \
         well-defined",
    );

    let barrier = Arc::new(Barrier::new(PEER_COUNT));
    let mut handles = Vec::with_capacity(PEER_COUNT);
    for i in 0..PEER_COUNT {
        let cache = Arc::clone(&cache);
        let barrier = Arc::clone(&barrier);
        let image = image.clone();
        handles.push(thread::spawn(move || {
            let mut meta = test_metadata("6.14.2");
            // Half the peers publish config_hash="hash-a",
            // half publish config_hash="hash-b" — distinct
            // content-defining axis means the two groups MUST
            // not recheck-collapse into one another. Peers
            // within a group share the same hash and so are
            // recheck-equivalent (only one head writer per
            // group pays the publish cost).
            let (label, hash) = if i % 2 == 0 {
                ("a", "hash-a")
            } else {
                ("b", "hash-b")
            };
            meta.built_at = format!("2026-04-12T10:00:{i:02}Z");
            meta.config_hash = Some(hash.to_string());
            barrier.wait();
            let entry = cache
                .store("mixed-key", &CacheArtifacts::new(&image), &meta)
                .expect("every peer's store must succeed");
            (
                label,
                entry.metadata.config_hash.clone(),
                entry.metadata.built_at.clone(),
            )
        }));
    }
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // Every peer's returned entry must carry one of the two
    // valid content states — never a torn third state, never
    // None. The exact set of returned config_hash values
    // depends on the interleaving (a single state if one group
    // finished entirely before the other started, two states
    // otherwise) but every observed value MUST belong to
    // {Some("hash-a"), Some("hash-b")}.
    for (label, observed_hash, observed_built_at) in &results {
        let observed_hash_str = observed_hash.as_deref();
        assert!(
            matches!(observed_hash_str, Some("hash-a") | Some("hash-b")),
            "peer (group={label}) observed an invalid \
             config_hash {observed_hash:?} — recheck must never \
             produce a third state, only return one of the two \
             published group hashes",
        );

        // Label↔hash correspondence (HARD invariant, deterministic).
        // Each peer's LABEL was computed locally from its loop
        // index BEFORE the store() call, and the local `hash`
        // it published was derived directly from that label.
        // After the round-trip the returned config_hash MUST
        // therefore still match the label's expected hash. A
        // regression that loosened `cache_content_matches` so a
        // group-B peer collapses onto group-A's published entry
        // would let this assertion catch the cross-hash
        // collapse (peer label "b" returning Some("hash-a")) —
        // the cross-group bleed check below pins built_at, this
        // pins the hash itself.
        let expected_hash = match *label {
            "a" => "hash-a",
            "b" => "hash-b",
            _ => unreachable!(),
        };
        assert_eq!(
            observed_hash_str,
            Some(expected_hash),
            "peer (group={label}) returned config_hash \
             {observed_hash:?} — expected {expected_hash}; \
             cross-group recheck collapse detected (a recheck \
             hit MUST require matching content-defining hashes)",
        );

        // Cross-group bleed check (HARD invariant, deterministic).
        // The returned config_hash and built_at must agree on
        // their group. A group-A peer that short-circuits
        // borrows the head writer's published state — and the
        // head writer that produced an entry with hash-X drew
        // its built_at from group-X's input set. So the
        // returned (hash, built_at) pair must be group-coherent.
        let observed_hash_bytes = observed_hash_str.unwrap_or("");
        let in_a = group_a_inputs.contains(observed_built_at);
        let in_b = group_b_inputs.contains(observed_built_at);
        assert!(
            in_a || in_b,
            "peer (group={label}) returned built_at \
             {observed_built_at:?} that is NOT one of the \
             precomputed input timestamps — recheck must \
             never synthesize a fresh timestamp",
        );
        match observed_hash_bytes {
            "hash-a" => assert!(
                in_a && !in_b,
                "config_hash=hash-a entry returned built_at \
                 {observed_built_at:?} which lives in group B's \
                 input set — recheck-bypass on cross-group \
                 divergence broke and a group-A return is \
                 carrying a group-B timestamp",
            ),
            "hash-b" => assert!(
                in_b && !in_a,
                "config_hash=hash-b entry returned built_at \
                 {observed_built_at:?} which lives in group A's \
                 input set — recheck-bypass on cross-group \
                 divergence broke and a group-B return is \
                 carrying a group-A timestamp",
            ),
            _ => unreachable!(),
        }
    }

    // Soft observation: at least one group SHOULD show
    // recheck collapse (multiple peers sharing the same
    // built_at). Logged not asserted — under adversarial
    // scheduling, cross-group overwrites can prevent any
    // within-group peer from successfully short-circuiting,
    // and asserting collapse would be a flake. The hard
    // invariants above are sufficient to prove the recheck
    // semantics; this is informational.
    let group_a_built_ats: BTreeSet<&String> = results
        .iter()
        .filter(|(label, _, _)| *label == "a")
        .map(|(_, _, built_at)| built_at)
        .collect();
    let group_b_built_ats: BTreeSet<&String> = results
        .iter()
        .filter(|(label, _, _)| *label == "b")
        .map(|(_, _, built_at)| built_at)
        .collect();
    let group_a_size = results.iter().filter(|(l, _, _)| *l == "a").count();
    let group_b_size = results.iter().filter(|(l, _, _)| *l == "b").count();
    let collapse_fired_a = group_a_built_ats.len() < group_a_size;
    let collapse_fired_b = group_b_built_ats.len() < group_b_size;
    if !(collapse_fired_a || collapse_fired_b) {
        // Not a panic — would be a flake under adversarial
        // scheduling. Surface via the test's stderr so an
        // operator running with --no-capture sees that
        // collapse did not fire on this run.
        eprintln!(
            "store_in_lock_recheck_mixed_content_peers: \
             collapse did not fire on this run (group_a \
             distinct={}, size={}; group_b distinct={}, \
             size={}). Hard invariants still hold; collapse \
             firing is probabilistic under cross-group churn.",
            group_a_built_ats.len(),
            group_a_size,
            group_b_built_ats.len(),
            group_b_size,
        );
    }

    // The final on-disk entry must match exactly one of the two
    // valid states (whichever cross-group writer won the last
    // publish). A torn or absent on-disk state means the
    // atomic-publish guarantee broke under cross-group churn.
    let final_entry = cache.lookup("mixed-key").expect("entry must exist");
    let final_hash = final_entry.metadata.config_hash.as_deref();
    assert!(
        matches!(final_hash, Some("hash-a") | Some("hash-b")),
        "final on-disk config_hash {final_hash:?} must be one \
         of the two published group hashes — anything else \
         means publish was not atomic across overlapping writers",
    );
    // Final built_at must also obey the cross-group separation
    // rule: it must come from the input set of whichever
    // group's hash won the final rename.
    let final_built_at = &final_entry.metadata.built_at;
    let expected_set = match final_hash {
        Some("hash-a") => &group_a_inputs,
        Some("hash-b") => &group_b_inputs,
        _ => unreachable!(),
    };
    assert!(
        expected_set.contains(final_built_at),
        "final on-disk built_at {final_built_at:?} must come \
         from the input set of the winning group ({final_hash:?}) — \
         a foreign timestamp would prove recheck wrote across \
         the content-divergence axis",
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
    let timestamps: std::collections::BTreeSet<_> = entries
        .iter()
        .map(|e| e.metadata.built_at.clone())
        .collect();
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

// -- store: vmlinux strip-fallback warn capture --
//
// `cache_dir_store_falls_back_when_strip_fails` (above) pins
// the on-disk observable: an unrecognised vmlinux ELF lands
// verbatim under the cache entry with `vmlinux_stripped =
// false`. This test pins the OPERATOR-VISIBLE observable: the
// strip-failure path emits a `tracing::warn!` so an operator
// running with a default subscriber sees the diagnostic and
// knows the on-disk payload is larger than expected. A
// regression that silenced the warn (e.g. demoted to debug or
// dropped) would leave the operator wondering why their cache
// entry is suddenly 300 MB instead of 30 MB — pinning the
// log fragment locks the diagnostic in place.

#[tracing_test::traced_test]
#[test]
fn store_emits_warn_when_vmlinux_strip_fails() {
    let tmp = TempDir::new().unwrap();
    let cache = CacheDir::with_root(tmp.path().join("cache"));
    let src_dir = TempDir::new().unwrap();
    let image = create_fake_image(src_dir.path());
    // Non-ELF bytes — strip_vmlinux_debug returns Err, the
    // store() path falls back to copying the raw vmlinux.
    let vmlinux = src_dir.path().join("vmlinux");
    std::fs::write(&vmlinux, b"not an ELF file").unwrap();
    let meta = test_metadata("6.14.2");

    cache
        .store(
            "warn-on-strip-fail",
            &CacheArtifacts::new(&image).with_vmlinux(&vmlinux),
            &meta,
        )
        .expect("strip fallback must still produce a successful store");

    // Substring match to remain resilient to wording polish
    // that doesn't change the operator-relevant signal. The
    // `vmlinux strip failed` literal is the load-bearing
    // fragment — without it, the operator can't search for
    // strip-failure incidents in their tracing output.
    assert!(
        logs_contain("vmlinux strip failed"),
        "the strip-fallback path MUST emit a tracing::warn! \
         with the 'vmlinux strip failed' literal so an operator \
         can see the strip pipeline degraded — without the \
         warn, an unstripped 300 MB vmlinux lands silently and \
         the operator can't correlate cache-bloat reports with \
         strip failures",
    );
    assert!(
        logs_contain("caching unstripped"),
        "the warn body MUST tell the operator the caller fell \
         back to caching the raw bytes (not that the cache \
         refused) — so the operator understands the cache \
         entry is usable but oversized",
    );
}

// -- store: error paths --
//
// `store()` runs a copy/strip/write/rename pipeline; every
// step can return an error. The success path is exercised by
// every test above, but the error paths each have distinct
// diagnostic messages and cleanup obligations that need
// direct coverage.

/// `fs::copy` of the kernel image fails when the source path
/// does not exist. The error must surface with a "copy kernel
/// image to cache" prefix so an operator can attribute the
/// failure to the image-copy step (vs the vmlinux-copy step
/// or the metadata-write step).
#[test]
fn store_image_copy_failure_surfaces_diagnostic() {
    let tmp = TempDir::new().unwrap();
    let cache = CacheDir::with_root(tmp.path().join("cache"));
    let src_dir = TempDir::new().unwrap();
    let nonexistent = src_dir.path().join("never-created-bzImage");
    // Pre-condition: the source path must NOT exist — fs::copy
    // surfaces ENOENT and store() wraps it with the
    // "copy kernel image to cache" prefix.
    assert!(!nonexistent.exists());
    let meta = test_metadata("6.14.2");

    let err = cache
        .store("img-copy-fail", &CacheArtifacts::new(&nonexistent), &meta)
        .expect_err("missing source image must fail the store");
    let msg = format!("{err:#}");
    assert!(
        msg.starts_with("copy kernel image to cache:"),
        "diagnostic must START with the exact `copy kernel image \
         to cache:` prefix so an operator can attribute the \
         failure to the image-copy step (vs the stripped-vmlinux \
         `copy stripped vmlinux to cache:` arm or the \
         fallback-vmlinux `copy vmlinux to cache:` arm); got: {msg}",
    );

    // Cleanup obligation: the TmpDirGuard must remove the
    // staging directory on this error path — no `.tmp-*`
    // entries can survive a failed store.
    for dirent in std::fs::read_dir(tmp.path().join("cache")).unwrap() {
        let name = dirent.unwrap().file_name().to_string_lossy().into_owned();
        assert!(
            !name.starts_with(".tmp-"),
            "TmpDirGuard must remove the staging directory on \
             the image-copy error path; found leftover: {name}",
        );
    }
}

/// `fs::copy` of the unstripped vmlinux on the strip-fallback
/// path fails when the source vmlinux does not exist. Pins the
/// EXACT error-context prefix `"copy vmlinux to cache:"` so a
/// future refactor that reroutes the fallback (e.g. moves to a
/// separate helper) but drops the context prefix is caught
/// immediately. The test exercises the strip-fallback fs::copy
/// arm specifically — strip_vmlinux_debug errors first on the
/// missing read, store() falls through to the raw-bytes
/// fallback copy, which then ALSO errors with ENOENT against
/// the same missing source. The final error surfaced to the
/// caller MUST carry the exact "copy vmlinux to cache:" prefix
/// that distinguishes this arm from the "copy stripped vmlinux
/// to cache:" success-path arm and the "copy kernel image to
/// cache:" image-copy arm.
#[test]
fn store_vmlinux_copy_failure_uses_exact_error_prefix() {
    let tmp = TempDir::new().unwrap();
    let cache = CacheDir::with_root(tmp.path().join("cache"));
    let src_dir = TempDir::new().unwrap();
    let image = create_fake_image(src_dir.path());
    let vmlinux = src_dir.path().join("never-created-vmlinux");
    assert!(!vmlinux.exists());
    let meta = test_metadata("6.14.2");

    let err = cache
        .store(
            "vml-fallback-copy-fail",
            &CacheArtifacts::new(&image).with_vmlinux(&vmlinux),
            &meta,
        )
        .expect_err("missing vmlinux must fail the store on the fallback path");
    let msg = format!("{err:#}");
    assert!(
        msg.starts_with("copy vmlinux to cache:"),
        "the fallback fs::copy arm wraps with the exact prefix \
         `copy vmlinux to cache:` — this distinguishes it from \
         the success-path stripped-copy `copy stripped vmlinux \
         to cache:` and the kernel-image arm `copy kernel image \
         to cache:`; a regression that drops the context \
         wrapping or rewords the prefix would lose the \
         arm-attribution diagnostic. Got: {msg}",
    );

    // Cleanup obligation across the vmlinux-fallback error path.
    for dirent in std::fs::read_dir(tmp.path().join("cache")).unwrap() {
        let name = dirent.unwrap().file_name().to_string_lossy().into_owned();
        assert!(
            !name.starts_with(".tmp-"),
            "TmpDirGuard must remove the staging directory on \
             the vmlinux-fallback-copy error path; found \
             leftover: {name}",
        );
    }
}

// -- TmpDirGuard cleanup-on-error --
//
// The guard is a `Drop` impl — when `store()` returns Err,
// the guard's drop must remove the `.tmp-{key}-{pid}` dir.
// The error-path tests above verify the observable
// (no leftover .tmp- entries after a failed store); this test
// pins the contract more directly by injecting a stale tmp dir
// (simulating a prior crashed store) under the SAME pid+key,
// proving that store()'s pre-stage `fs::remove_dir_all(tmp_dir)`
// and TmpDirGuard cooperate correctly.
#[test]
fn tmp_dir_guard_removes_staging_dir_after_failed_store() {
    let tmp = TempDir::new().unwrap();
    let cache_root = tmp.path().join("cache");
    let cache = CacheDir::with_root(cache_root.clone());
    let src_dir = TempDir::new().unwrap();
    // Missing source forces fs::copy to fail mid-store.
    let nonexistent = src_dir.path().join("never-created");
    let meta = test_metadata("6.14.2");

    let _ = cache
        .store("guard-test", &CacheArtifacts::new(&nonexistent), &meta)
        .expect_err("missing source must fail");

    // After the failed store returns, the TmpDirGuard's drop
    // must have removed the staging dir. Walk the cache root
    // and verify no .tmp- entries survive.
    let mut leftover_tmp_count = 0;
    if cache_root.exists() {
        for dirent in std::fs::read_dir(&cache_root).unwrap() {
            let name = dirent.unwrap().file_name().to_string_lossy().into_owned();
            if name.starts_with(".tmp-") {
                leftover_tmp_count += 1;
            }
        }
    }
    assert_eq!(
        leftover_tmp_count, 0,
        "TmpDirGuard's Drop impl must clean up .tmp- staging \
         directories after a failed store — found {leftover_tmp_count} leftover(s)",
    );

    // The final cache entry must NOT exist either — a failed
    // store must not leave a partial publish.
    assert!(
        !cache_root.join("guard-test").exists(),
        "a failed store must not publish a partial entry",
    );
}

// -- store: pre-existing tmp_dir as a regular FILE --
//
// store()'s pre-stage cleanup does:
//   if tmp_dir.exists() {
//       fs::remove_dir_all(&tmp_dir)?;
//   }
// The remove_dir_all docs (rust stdlib) state it FAILS if the
// path is not a directory. So if a prior crash (or hostile
// operator action) leaves a regular FILE at the
// `.tmp-{key}-{pid}` path, the cleanup step bails — store()
// surfaces the error rather than silently overwriting the
// file or proceeding into an inconsistent state. Pin this
// edge case so a future refactor that, e.g., switches to
// `remove_file` first or silently swallows the error doesn't
// regress the safety invariant.
#[test]
fn store_fails_when_tmp_dir_path_is_regular_file() {
    let tmp = TempDir::new().unwrap();
    let cache_root = tmp.path().join("cache");
    let cache = CacheDir::with_root(cache_root.clone());
    let src_dir = TempDir::new().unwrap();
    let image = create_fake_image(src_dir.path());
    let meta = test_metadata("6.14.2");

    // Pre-create the EXACT path store() will try to mkdir
    // (`.tmp-{cache_key}-{pid}`) as a regular FILE. The path
    // must match store()'s computed tmp_dir name exactly,
    // including the current process pid.
    std::fs::create_dir_all(&cache_root).unwrap();
    let pid = std::process::id();
    let blocking_file = cache_root.join(format!(".tmp-blocked-key-{pid}"));
    std::fs::write(&blocking_file, b"i am a regular file, not a directory").unwrap();

    let err = cache
        .store("blocked-key", &CacheArtifacts::new(&image), &meta)
        .expect_err(
            "pre-existing regular FILE at the tmp_dir path must \
             fail the store — fs::remove_dir_all rejects non-directories",
        );
    // The exact error wording is platform-dependent (Linux:
    // ENOTDIR "Not a directory"); pin only the broadly-true
    // observable: the error is surfaced, not silently swallowed.
    let msg = format!("{err:#}");
    assert!(
        !msg.is_empty(),
        "store error must carry a non-empty diagnostic; got: {msg}",
    );

    // The blocking file MUST remain intact — store() must not
    // delete (and must not silently overwrite) operator state
    // that doesn't fit the cache's expected shape. This is the
    // critical safety invariant: a future regression that
    // swapped remove_dir_all for `remove_file` to "fix" the
    // failure would silently delete the operator's file.
    assert!(
        blocking_file.exists(),
        "the pre-existing regular file at the tmp_dir path MUST \
         remain in place after the failed store — silently \
         overwriting it would erase operator state without \
         warning",
    );
    assert_eq!(
        std::fs::read(&blocking_file).unwrap(),
        b"i am a regular file, not a directory",
        "the blocking file's CONTENTS must also be unchanged — \
         not just the inode",
    );

    // The final cache entry MUST NOT exist — a failed store
    // must not leave a partial publish.
    assert!(
        !cache_root.join("blocked-key").exists(),
        "a failed store must not publish a partial entry under \
         the cache_key",
    );
}

// -- kconfig_status with empty-string hash --
//
// The hash is treated as an opaque string by `kconfig_status`;
// `Some("")` is a valid (degenerate) hash. The classifier MUST
// dispatch on string equality, not on emptiness, so two
// entries with `Some("")` MATCH each other rather than both
// being stale or untracked. Pins the equality semantic — a
// regression that special-cased empty would mis-classify
// (degenerate, but legitimate) test fixtures.

/// Both cached and current are `Some("")`: status MUST be
/// `Matches` (string equality, not "treat empty as untracked").
#[test]
fn kconfig_status_empty_strings_classify_as_matches() {
    let tmp = TempDir::new().unwrap();
    let cache = CacheDir::with_root(tmp.path().join("cache"));
    let src_dir = TempDir::new().unwrap();
    let image = create_fake_image(src_dir.path());
    let meta = test_metadata("6.14.2").with_ktstr_kconfig_hash(Some("".to_string()));
    let entry = cache
        .store("empty-vs-empty", &CacheArtifacts::new(&image), &meta)
        .unwrap();
    assert_eq!(
        entry.kconfig_status(""),
        KconfigStatus::Matches,
        "Some(\"\") cached + \"\" current must classify as \
         Matches — the predicate is string equality on the \
         inner string, not a separate emptiness check",
    );
}

/// Cached `Some("")` vs current non-empty: status MUST be
/// `Stale { cached: "", current: "..." }`. Empty cached is
/// distinct from `None` (Untracked) — the variant carries
/// data even when the inner string is empty.
#[test]
fn kconfig_status_empty_cached_vs_nonempty_current_is_stale() {
    let tmp = TempDir::new().unwrap();
    let cache = CacheDir::with_root(tmp.path().join("cache"));
    let src_dir = TempDir::new().unwrap();
    let image = create_fake_image(src_dir.path());
    let meta = test_metadata("6.14.2").with_ktstr_kconfig_hash(Some("".to_string()));
    let entry = cache
        .store("empty-vs-nonempty", &CacheArtifacts::new(&image), &meta)
        .unwrap();
    match entry.kconfig_status("real_hash") {
        KconfigStatus::Stale { cached, current } => {
            assert_eq!(
                cached, "",
                "Stale.cached must carry the empty string \
                 verbatim — empty Some(\"\") is NOT collapsed \
                 to None at compare time",
            );
            assert_eq!(
                current, "real_hash",
                "Stale.current must carry the caller's hash",
            );
        }
        other => panic!("expected Stale, got {other:?}"),
    }
}

/// Cached `None` + current `""` → Untracked. Single-assertion
/// focused variant of
/// [`kconfig_status_none_cached_returns_untracked_regardless_of_current`]
/// pinned under the canonical name used in team review notes
/// so a `grep` against that exact name finds the test.
#[test]
fn kconfig_status_none_cached_vs_empty_current_is_untracked() {
    let tmp = TempDir::new().unwrap();
    let cache = CacheDir::with_root(tmp.path().join("cache"));
    let src_dir = TempDir::new().unwrap();
    let image = create_fake_image(src_dir.path());
    let meta = KernelMetadata {
        ktstr_kconfig_hash: None,
        ..test_metadata("6.14.2")
    };
    let entry = cache
        .store("none-vs-empty", &CacheArtifacts::new(&image), &meta)
        .unwrap();
    assert_eq!(
        entry.kconfig_status(""),
        KconfigStatus::Untracked,
        "None cached + \"\" current MUST classify as Untracked",
    );
}

/// Cached `None` (no recorded hash) vs current `""` (empty hash
/// passed in by the caller): status MUST be `Untracked` — the
/// classifier dispatches on `Option::None`, never on the
/// caller's hash content. A regression that special-cased
/// `current == ""` to short-circuit to Matches would mistake
/// a pre-tracking-format entry for a clean cache hit. Pins
/// the `None` short-circuit in `CacheEntry::kconfig_status` —
/// the inner match arm `None => KconfigStatus::Untracked`
/// MUST fire before any string-equality check on `current`.
#[test]
fn kconfig_status_none_cached_returns_untracked_regardless_of_current() {
    let tmp = TempDir::new().unwrap();
    let cache = CacheDir::with_root(tmp.path().join("cache"));
    let src_dir = TempDir::new().unwrap();
    let image = create_fake_image(src_dir.path());
    // Explicitly None — pre-tracking-format entry.
    let meta = KernelMetadata {
        ktstr_kconfig_hash: None,
        ..test_metadata("6.14.2")
    };
    let entry = cache
        .store("none-cached", &CacheArtifacts::new(&image), &meta)
        .unwrap();
    // Empty current must NOT collapse to Matches.
    assert_eq!(
        entry.kconfig_status(""),
        KconfigStatus::Untracked,
        "None cached + \"\" current must classify as Untracked — \
         the caller's hash content is ignored when the cached \
         entry has no recorded hash; a regression that \
         special-cased current==\"\" to short-circuit to Matches \
         would mistake pre-tracking-format entries for clean hits",
    );
    // Non-empty current ALSO untracked — the dispatch is on
    // Option::None, not on the caller's input.
    assert_eq!(
        entry.kconfig_status("any-hash"),
        KconfigStatus::Untracked,
        "None cached + non-empty current must also classify as \
         Untracked — the predicate is variant-driven, not \
         input-driven",
    );
}

// -- lookup vs list semantics --
//
// A corrupt entry (metadata.json missing/malformed) returns
// None from lookup() and Corrupt from list(). The two methods
// serve different purposes: lookup is a direct hit-or-miss
// probe, list enumerates everything the operator should be
// aware of. Pins the divergent semantics.

/// A corrupt entry whose metadata.json is malformed:
/// lookup() returns None, list() returns Corrupt for the
/// same key. Pins the semantic that lookup hides corruption
/// (the caller treats it as a miss and rebuilds), while list
/// surfaces it (the operator sees and decides).
#[test]
fn lookup_vs_list_diverge_on_corrupt_entry() {
    let tmp = TempDir::new().unwrap();
    let cache_root = tmp.path().join("cache");
    let cache = CacheDir::with_root(cache_root.clone());
    let entry_dir = cache_root.join("corrupt-entry");
    std::fs::create_dir_all(&entry_dir).unwrap();
    std::fs::write(entry_dir.join("metadata.json"), b"not valid json {[").unwrap();

    // lookup: corruption is hidden as a miss.
    assert!(
        cache.lookup("corrupt-entry").is_none(),
        "lookup() MUST return None on a corrupt entry — the \
         caller treats it as a miss and proceeds to rebuild; \
         surfacing the corruption here would force every \
         caller to handle a third state besides hit/miss",
    );

    // list: the same entry surfaces as Corrupt with a reason.
    let entries = cache.list().unwrap();
    let listed = entries.iter().find(|e| e.key() == "corrupt-entry").expect(
        "list MUST surface the corrupt entry — the \
                 operator needs to see and decide what to do \
                 about it (clean? investigate?)",
    );
    // Pin the EXACT Corrupt classification rather than the
    // weaker `as_valid().is_none()` check — `as_valid()` only
    // distinguishes Valid from non-Valid, so a future variant
    // expansion (e.g. adding `Pending` or `Locked`) would
    // silently let this test pass without proving the entry
    // landed under the Corrupt arm specifically.
    assert!(
        matches!(listed, ListedEntry::Corrupt { .. }),
        "list MUST classify the entry as ListedEntry::Corrupt \
         — the lookup miss is the same on-disk state as a list \
         Corrupt entry, and the variant must be the Corrupt arm \
         specifically (not just non-Valid). Got: {listed:?}",
    );
}

// -- cache_content_matches all-None --
//
// The shared `test_metadata` fixture sets
// `config_hash = Some("abc123")` and
// `ktstr_kconfig_hash = Some("def456")` but
// `extra_kconfig_hash = None`. The match-all test
// (`cache_content_matches_when_only_built_at_differs`) therefore
// already exercises `extra_kconfig_hash = None`; the all-None
// case for `config_hash` and `ktstr_kconfig_hash` (a
// freshly-constructed KernelMetadata via ::new with no setters
// chained) is uncovered. This test fills that gap by
// constructing both sides via ::new so every Option hash field
// is None, then asserting the predicate accepts `None == None`
// rather than treating None as "unknown, can't compare".

/// Two metadata values with EVERY `Option<String>` hash field
/// set to None match each other. Pins that None == None for
/// the recheck — without this, two cache misses that both
/// landed without a config_hash recorded would each redundantly
/// republish.
#[test]
fn cache_content_matches_when_all_hashes_are_none() {
    // Build metadata via ::new (no chainable setters) so every
    // Option<String> hash field is None.
    let cached = KernelMetadata::new(
        super::super::metadata::KernelSource::Tarball,
        "x86_64".to_string(),
        "bzImage".to_string(),
        "2026-04-12T10:00:00Z".to_string(),
    );
    let caller = KernelMetadata::new(
        super::super::metadata::KernelSource::Tarball,
        "x86_64".to_string(),
        "bzImage".to_string(),
        // Distinct built_at — must NOT change the recheck's
        // verdict because built_at is excluded from the
        // predicate. The hashes are all None on both sides,
        // and None == None.
        "2026-04-13T10:00:00Z".to_string(),
    );
    assert!(
        cache_content_matches(&cached, &caller, false),
        "two metadata values with every hash field set to None \
         must classify as content-equal — None == None for the \
         predicate; without this, a cache that doesn't track \
         hashes would recheck-miss on every concurrent peer \
         and redundantly republish",
    );
}

/// All-None hashes WITH vmlinux on both sides matches: pins the
/// vmlinux-presence axis of the predicate when every hash is
/// None. The previous all-None test exercises only the
/// caller_has_vmlinux=false branch; this test covers the
/// caller_has_vmlinux=true branch with cached.has_vmlinux=true,
/// so a regression that special-cased "all hashes None implies
/// no vmlinux" would surface.
#[test]
fn cache_content_matches_all_none_with_vmlinux_on_both_sides() {
    let mut cached = KernelMetadata::new(
        super::super::metadata::KernelSource::Tarball,
        "x86_64".to_string(),
        "bzImage".to_string(),
        "2026-04-12T10:00:00Z".to_string(),
    );
    cached.set_has_vmlinux(true);
    let caller = KernelMetadata::new(
        super::super::metadata::KernelSource::Tarball,
        "x86_64".to_string(),
        "bzImage".to_string(),
        "2026-04-13T10:00:00Z".to_string(),
    );
    // caller_has_vmlinux=true (caller is publishing a vmlinux
    // sidecar); cached has has_vmlinux=true; every hash is
    // None on both sides. All four predicate axes match →
    // content-equal.
    assert!(
        cache_content_matches(&cached, &caller, true),
        "all-None hashes + matched vmlinux presence (true=true) \
         must classify as content-equal — the vmlinux-axis is \
         the only non-hash axis of the predicate, and the \
         test_metadata-based test only covers vmlinux=false. \
         Without this case, a regression that special-cased \
         'all-None implies no vmlinux' would silently pass \
         the existing tests.",
    );
}

/// Asymmetric None: cached has Some, caller has None (or vice
/// versa). The predicate must reject both cases — None is
/// distinct from Some(s) regardless of s. Pins that the
/// recheck does NOT silently fall through "unknown" to a hit.
#[test]
fn cache_content_matches_asymmetric_none_misses() {
    // cached has config_hash=Some, caller has None.
    let mut cached = test_metadata("6.14.2");
    cached.config_hash = Some("hash-cached".to_string());
    let mut caller = test_metadata("6.14.2");
    caller.config_hash = None;
    assert!(
        !cache_content_matches(&cached, &caller, false),
        "cached=Some, caller=None must classify as \
         content-different — None != Some(s); a regression \
         that treated None as 'matches everything' would \
         break the recheck for any caller that lost its hash",
    );

    // Symmetric: cached None, caller Some.
    let mut cached = test_metadata("6.14.2");
    cached.config_hash = None;
    let mut caller = test_metadata("6.14.2");
    caller.config_hash = Some("hash-caller".to_string());
    assert!(
        !cache_content_matches(&cached, &caller, false),
        "cached=None, caller=Some must also classify as \
         content-different — the asymmetric direction must \
         also fail to recheck",
    );
}

// -- lookup_silent contract (no warn dedup pollution) --
//
// store()'s in-lock recheck calls
// `lookup_silent` rather than `lookup` precisely so a recheck
// hit on a strip-fallback entry does NOT consume the once-per-
// process dedup slot. The test that pins this contract is
// critical — without it, a regression that swapped
// `lookup_silent` for `lookup` inside `store()` would silently
// burn the dedup slot, making the user-facing `lookup` call
// (which fires AFTER `store()` returns) silent. The result:
// operators stop seeing the strip-fallback warn after a
// store-hit pattern.

/// Direct unit test on the contract: `lookup_silent` must not
/// emit the unstripped-vmlinux warn even when the entry is in
/// the warn-eligible state. Drives the contract through the
/// public `lookup` (which DOES warn) and a private inspection
/// of the warned-keys static is unavailable, so the test
/// instead pins the OBSERVABLE: a `lookup_silent` followed by
/// a `lookup` for the same key BOTH proceed without dedup
/// suppressing the second call's warn. If `lookup_silent` had
/// burned the dedup slot, the subsequent `lookup` would be
/// silent and the test would fail.
#[tracing_test::traced_test]
#[test]
fn lookup_silent_does_not_consume_warn_dedup_slot() {
    let tmp = TempDir::new().unwrap();
    let cache = CacheDir::with_root(tmp.path().join("cache"));
    let src_dir = TempDir::new().unwrap();
    let image = create_fake_image(src_dir.path());
    // Use a unique cache_key for this test so the
    // process-wide warned_keys static (shared across all
    // tests) does not let a prior test's dedup state mask
    // the assertion. The key must not collide with any other
    // test's stale-vmlinux key; "lookup-silent-contract" is
    // distinct.
    let key = "lookup-silent-contract";
    // Store a strip-failing vmlinux so the entry has
    // has_vmlinux=true + vmlinux_stripped=false (warn-eligible).
    let vmlinux = src_dir.path().join("vmlinux");
    std::fs::write(&vmlinux, b"not an ELF file").unwrap();
    let meta = test_metadata("6.14.2");
    cache
        .store(
            key,
            &CacheArtifacts::new(&image).with_vmlinux(&vmlinux),
            &meta,
        )
        .unwrap();

    // Step 1: silent lookup — must NOT emit the warn.
    let _silent = cache.lookup_silent(key);
    // Step 2: public lookup — MUST emit the warn (the dedup
    // slot for this key is still empty because step 1 used
    // lookup_silent).
    let _public = cache.lookup(key);
    assert!(
        logs_contain("using unstripped vmlinux"),
        "lookup() after lookup_silent() MUST emit the \
         unstripped-vmlinux warn — if lookup_silent had \
         consumed the once-per-key dedup slot, this assertion \
         would fail and the operator would never see the \
         warn for entries that store() saw first via the \
         in-lock recheck",
    );
}

// -- remove_entries reports the input size as count --
//
// `clean_keep`/`clean_all` collect the input iterator into a
// Vec, capture `count = len()`, then loop `fs::remove_dir_all`.
// The returned count is the input size, not the number of
// entries actually removed. This test pins the success-path
// contract that count reflects the LIST of candidates passed
// to the loop, so an operator-facing "removed N entries"
// diagnostic at least carries the correct input cardinality.
//
// Partial-failure semantics (a remove_dir_all mid-loop error
// returning Err while count already encoded the full list)
// are not directly testable here: the test process runs as
// root, and CAP_DAC_OVERRIDE bypasses every chmod-based
// permission denial used to engineer such a failure. Any
// alternative trigger (immutable attributes, read-only
// mounts, exotic filesystem error codes) is non-portable and
// would flake across CI environments.
#[test]
fn clean_all_count_matches_listed_entry_count() {
    let tmp = TempDir::new().unwrap();
    let cache = CacheDir::with_root(tmp.path().join("cache"));
    let src_dir = TempDir::new().unwrap();
    let image = create_fake_image(src_dir.path());

    // Mix: 2 valid entries + 1 corrupt-shaped entry. The list()
    // call returns ALL three (Valid + Corrupt). clean_all
    // walks the list and removes them all, returning a count
    // that equals the list cardinality.
    let cache_root = tmp.path().join("cache");
    cache
        .store(
            "valid-1",
            &CacheArtifacts::new(&image),
            &test_metadata("6.13.0"),
        )
        .unwrap();
    cache
        .store(
            "valid-2",
            &CacheArtifacts::new(&image),
            &test_metadata("6.14.2"),
        )
        .unwrap();
    // Add a corrupt-shaped entry by creating an empty
    // directory with no metadata.json. list() classifies it
    // as Corrupt; clean_all removes it alongside the valid
    // entries.
    let corrupt = cache_root.join("corrupt-1");
    std::fs::create_dir_all(&corrupt).unwrap();

    let removed = cache
        .clean_all()
        .expect("clean_all must succeed on a clean fs");
    assert_eq!(
        removed, 3,
        "clean_all MUST report a count equal to the listed \
         entry count (Valid + Corrupt) — operator-facing \
         reporting that mismatched the actual cleanup would \
         undermine trust in the diagnostic",
    );

    // Verify on disk: every entry the list returned has been
    // removed. The .locks/ subdir (if present) is preserved.
    let surviving: Vec<_> = std::fs::read_dir(&cache_root)
        .unwrap()
        .filter_map(|d| d.ok())
        .map(|d| d.file_name().to_string_lossy().into_owned())
        .filter(|n| !n.starts_with('.'))
        .collect();
    assert!(
        surviving.is_empty(),
        "every non-dotfile entry must be gone after clean_all; \
         surviving: {surviving:?}",
    );
}
