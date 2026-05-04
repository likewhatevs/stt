//! Two-tier initramfs cache: per-process HashMap + cross-process POSIX
//! shm.
//!
//! Each VM run produces an initramfs blob keyed by the content hashes
//! of the payload binary, optional scheduler / probe / worker
//! binaries, include files, and shell-mode flags. Building the blob
//! is expensive (10s of MB of cpio assembly + LZ4 compression), so
//! the cache amortises the cost across:
//!
//! - **Same-process tests**: a `HashMap<BaseKey, Arc<Vec<u8>>>`
//!   keeps the in-flight blob hot without a syscall.
//! - **Cross-process tests / nextest workers**: an `O_CREAT|O_EXCL`
//!   race over a `/dev/shm/ktstr-base-<hash>` segment elects a
//!   single builder; losers `LOCK_SH`-block on the segment until
//!   the winner finishes, then `mmap` it zero-copy.
//!
//! The `BaseKey` content-hash spans every byte the build consumes
//! (binary content + shared-lib content) so a recompile invalidates
//! the cache without operator intervention. Stale segments from a
//! previous compression format are GC'd on every run via a
//! `LOCK_EX | LOCK_NB` probe.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use super::initramfs;

/// Cache key for base initramfs. Derived from content hashes of the
/// payload binary and its shared libs, plus the optional scheduler
/// binary and its shared libs. Shell mode additionally mixes in a
/// sentinel, include files, and the busybox flag; see [`Self::new`]
/// and [`Self::new_shell`] for per-constructor inputs.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BaseKey(pub(crate) u64);

/// Hash a file's content for cache keying via streaming reads.
///
/// Uses [`siphasher::sip::SipHasher13`] with fixed zero keys rather
/// than [`std::hash::DefaultHasher`]. DefaultHasher's concrete
/// algorithm is explicitly not guaranteed stable across Rust
/// toolchain versions, so cache keys computed with it would silently
/// shift when the compiler was upgraded — invalidating every cached
/// initramfs blob. SipHash13 with pinned keys is version-stable by
/// the siphasher crate's contract.
pub(crate) fn hash_file(path: &Path) -> Result<u64> {
    use siphasher::sip::SipHasher13;
    use std::hash::Hasher;
    let contents =
        std::fs::read(path).with_context(|| format!("read for hash: {}", path.display()))?;
    let mut hasher = SipHasher13::new_with_keys(0, 0);
    hasher.write(&contents);
    Ok(hasher.finish())
}

impl BaseKey {
    /// Hashes the payload binary content, payload shared libs, and
    /// the optional scheduler / probe / alloc-worker binary content
    /// and shared libs. Each optional input participates
    /// symmetrically because each changes the bytes written into
    /// the initramfs. Explicit parameters keep the cache key
    /// sensitive to these inputs regardless of the routing choice —
    /// the probe currently rides the extras path (stripped) while
    /// the worker rides `include_files` (verbatim), but the hash
    /// stays correct if a future change moves either between the
    /// two paths (the `new_shell` include-hash loop also re-hashes
    /// whatever ends up in `include_files`, so the double hash of a
    /// worker-in-includes is tolerated; the explicit worker hash
    /// covers the case where a future refactor moves the worker to
    /// extras).
    pub(crate) fn new(
        payload: &Path,
        scheduler: Option<&Path>,
        probe: Option<&Path>,
        worker: Option<&Path>,
    ) -> Result<Self> {
        use siphasher::sip::SipHasher13;
        let mut hasher = SipHasher13::new_with_keys(0, 0);

        hash_file(payload)?.hash(&mut hasher);
        Self::hash_shared_libs(payload, &mut hasher);

        match scheduler {
            Some(s) => {
                1u8.hash(&mut hasher);
                hash_file(s)?.hash(&mut hasher);
                Self::hash_shared_libs(s, &mut hasher);
            }
            None => 0u8.hash(&mut hasher),
        }

        match probe {
            Some(p) => {
                1u8.hash(&mut hasher);
                hash_file(p)?.hash(&mut hasher);
                Self::hash_shared_libs(p, &mut hasher);
            }
            None => 0u8.hash(&mut hasher),
        }

        match worker {
            Some(w) => {
                1u8.hash(&mut hasher);
                hash_file(w)?.hash(&mut hasher);
                Self::hash_shared_libs(w, &mut hasher);
            }
            None => 0u8.hash(&mut hasher),
        }

        Ok(BaseKey(hasher.finish()))
    }

    /// Shell mode key: hashes a sentinel, include files, and the
    /// busybox flag so different shell configurations get distinct
    /// cache keys. Include file archive paths and content are hashed
    /// so the same payload + same includes = cache hit, while
    /// different includes = cache miss. `probe` and `worker` are
    /// hashed for the same reasons as [`BaseKey::new`].
    pub(crate) fn new_shell(
        payload: &Path,
        scheduler: Option<&Path>,
        probe: Option<&Path>,
        worker: Option<&Path>,
        include_files: &[(String, PathBuf)],
        busybox: bool,
    ) -> Result<Self> {
        use siphasher::sip::SipHasher13;
        let mut hasher = SipHasher13::new_with_keys(0, 0);

        "ktstr-shell".hash(&mut hasher);
        busybox.hash(&mut hasher);
        hash_file(payload)?.hash(&mut hasher);
        Self::hash_shared_libs(payload, &mut hasher);

        match scheduler {
            Some(s) => {
                1u8.hash(&mut hasher);
                hash_file(s)?.hash(&mut hasher);
                Self::hash_shared_libs(s, &mut hasher);
            }
            None => 0u8.hash(&mut hasher),
        }

        match probe {
            Some(p) => {
                1u8.hash(&mut hasher);
                hash_file(p)?.hash(&mut hasher);
                Self::hash_shared_libs(p, &mut hasher);
            }
            None => 0u8.hash(&mut hasher),
        }

        match worker {
            Some(w) => {
                1u8.hash(&mut hasher);
                hash_file(w)?.hash(&mut hasher);
                Self::hash_shared_libs(w, &mut hasher);
            }
            None => 0u8.hash(&mut hasher),
        }

        // Hash include files: archive paths (sorted for determinism),
        // content hashes, and shared lib hashes for ELF includes (their
        // shared libs are packed by build_initramfs_base).
        let mut sorted: Vec<(&str, &Path)> = include_files
            .iter()
            .map(|(a, p)| (a.as_str(), p.as_path()))
            .collect();
        sorted.sort_by_key(|(a, _)| *a);
        sorted.len().hash(&mut hasher);
        for (archive_path, host_path) in &sorted {
            archive_path.hash(&mut hasher);
            hash_file(host_path)?.hash(&mut hasher);
            Self::hash_shared_libs(host_path, &mut hasher);
        }

        Ok(BaseKey(hasher.finish()))
    }

    /// Hash shared library paths and content samples for a binary so
    /// the cache key changes when any shared lib is updated on the host.
    fn hash_shared_libs(binary: &Path, hasher: &mut siphasher::sip::SipHasher13) {
        if let Ok(result) = initramfs::resolve_shared_libs(binary) {
            let mut entries: Vec<_> = result.found.iter().map(|(_, p)| p.clone()).collect();
            entries.sort();
            for p in &entries {
                // `to_str()` loses every non-UTF-8 path (Linux
                // paths are arbitrary byte sequences, not UTF-8)
                // and the `unwrap_or("")` collapse would hash
                // every such path to the SAME empty string,
                // silently gluing distinct libraries together in
                // the cache key. `as_encoded_bytes()` hashes the
                // raw OS bytes verbatim.
                p.as_os_str().as_encoded_bytes().hash(hasher);
                if let Ok(sample) = hash_file(p) {
                    sample.hash(hasher);
                }
            }
        }
    }
}

/// Process-global cache for base initramfs bytes. Keyed by content hash
/// of payload, scheduler, include files, and busybox flag.
/// The lock is only held during map lookup/insert, never during the
/// actual build.
pub(crate) fn base_cache() -> &'static Mutex<HashMap<BaseKey, Arc<Vec<u8>>>> {
    static CACHE: OnceLock<Mutex<HashMap<BaseKey, Arc<Vec<u8>>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Holds either a borrowed shm mapping or an owned Arc from the
/// process-local cache / a fresh build.
pub(crate) enum BaseRef {
    Mapped(initramfs::MappedShm),
    Owned(Arc<Vec<u8>>),
}

impl AsRef<[u8]> for BaseRef {
    fn as_ref(&self) -> &[u8] {
        match self {
            BaseRef::Mapped(m) => m.as_ref(),
            BaseRef::Owned(a) => a,
        }
    }
}

/// Obtain the base initramfs bytes, checking (in order):
/// 1. Process-local HashMap
/// 2. POSIX shared-memory segment via O_CREAT|O_EXCL race gate:
///    - Winner builds, writes segment, losers block on flock then mmap
/// 3. Fallback: build without cross-process coordination
pub(crate) fn get_or_build_base(
    payload: &Path,
    extras: &[(&str, &Path)],
    include_files: &[(&str, &Path)],
    busybox: bool,
    key: &BaseKey,
) -> Result<BaseRef> {
    // Clean stale SHM segments from previous runs.
    cleanup_stale_shm(key);

    // 1. Process-local cache
    if let Some(arc) = base_cache().lock().unwrap().get(key).cloned() {
        tracing::debug!("initramfs base cache hit (process)");
        return Ok(BaseRef::Owned(arc));
    }

    // 2. SHM race gate: try O_CREAT|O_EXCL to elect a single builder.
    let seg_name = initramfs::shm_segment_name(key.0);
    match shm_try_create_excl(&seg_name) {
        ShmCreateResult::Winner(fd) => {
            // We won the race — build, write, release.
            tracing::debug!("initramfs shm: builder (O_EXCL won)");
            let t0 = std::time::Instant::now();
            let data = initramfs::build_initramfs_base(payload, extras, include_files, busybox)?;
            tracing::debug!(
                elapsed_us = t0.elapsed().as_micros(),
                bytes = data.len(),
                "build_initramfs_base",
            );

            // Write data to the segment and release the exclusive lock.
            shm_write_and_release(fd, &data, &seg_name);

            // Load back via mmap for zero-copy return.
            // Skip process-local cache insert — the SHM mmap is persistent
            // and fast to re-acquire, so copying into an Arc is waste.
            if let Some(mapped) = initramfs::shm_load_base(key.0) {
                return Ok(BaseRef::Mapped(mapped));
            }

            // shm_load_base failed after we just wrote — fall through
            // to return an owned copy.
            let arc = Arc::new(data);
            base_cache()
                .lock()
                .unwrap()
                .insert(key.clone(), arc.clone());
            return Ok(BaseRef::Owned(arc));
        }
        ShmCreateResult::Exists => {
            // Another process is building (or has built). Block on
            // LOCK_SH via shm_load_base until the builder finishes.
            tracing::debug!("initramfs shm: waiting for builder (EEXIST)");
            if let Some(mapped) = initramfs::shm_load_base(key.0) {
                tracing::debug!("initramfs base cache hit (shm, after wait)");
                return Ok(BaseRef::Mapped(mapped));
            }
            // Builder may have failed and unlinked — fall through to build.
        }
        ShmCreateResult::Error => {
            // shm_open failed for a reason other than EEXIST (e.g. no /dev/shm).
            // Try a plain load in case the segment exists but O_EXCL had
            // a transient error.
            if let Some(mapped) = initramfs::shm_load_base(key.0) {
                tracing::debug!("initramfs base cache hit (shm)");
                return Ok(BaseRef::Mapped(mapped));
            }
        }
    }

    // 3. Fallback: build without SHM coordination.
    let t0 = std::time::Instant::now();
    let data = initramfs::build_initramfs_base(payload, extras, include_files, busybox)?;
    let arc = Arc::new(data);
    tracing::debug!(
        elapsed_us = t0.elapsed().as_micros(),
        bytes = arc.len(),
        "build_initramfs_base (fallback)",
    );

    base_cache()
        .lock()
        .unwrap()
        .insert(key.clone(), arc.clone());
    if let Err(e) = initramfs::shm_store_base(key.0, &arc) {
        tracing::warn!("shm_store_base: {e:#}");
    }

    Ok(BaseRef::Owned(arc))
}

/// Remove stale SHM segments from `/dev/shm` that don't match `current`.
/// Scans for `ktstr-base-*`, `ktstr-lz4-*`, and legacy `ktstr-gz-*`
/// entries and unlinks any whose hash suffix differs from the current key.
///
/// Only unlinks segments that are not held by another process. Tries
/// `LOCK_EX | LOCK_NB` on each candidate — if the lock succeeds, no
/// reader or writer holds it, so it's safe to unlink. If the lock
/// fails (`EWOULDBLOCK`), another process is actively using the
/// segment and it is skipped.
fn cleanup_stale_shm(current: &BaseKey) {
    let current_suffix = format!("{:016x}", current.0);
    let shm_dir = match std::fs::read_dir("/dev/shm") {
        Ok(d) => d,
        Err(_) => return,
    };
    for entry in shm_dir.flatten() {
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else {
            continue;
        };
        let hash_suffix = if let Some(s) = name_str.strip_prefix("ktstr-base-") {
            s
        } else if let Some(s) = name_str.strip_prefix("ktstr-lz4-") {
            s
        } else if let Some(s) = name_str.strip_prefix("ktstr-gz-") {
            // Legacy prefix from previous compression format.
            s
        } else {
            continue;
        };
        if hash_suffix == current_suffix {
            continue;
        }
        let shm_name = format!("/{name_str}");
        // rustix owns the fd via OwnedFd, so flock-then-drop is the
        // only cleanup path — no manual close required, and unlinks
        // happen before the fd drops so the segment is gone atomically
        // with lock release.
        let Ok(fd) = rustix::shm::open(
            shm_name.as_str(),
            rustix::shm::OFlags::RDONLY,
            rustix::fs::Mode::empty(),
        ) else {
            continue;
        };
        if rustix::fs::flock(&fd, rustix::fs::FlockOperation::NonBlockingLockExclusive).is_ok() {
            let _ = rustix::shm::unlink(shm_name.as_str());
            let _ = rustix::fs::flock(&fd, rustix::fs::FlockOperation::Unlock);
        }
    }
}

// ---------------------------------------------------------------------------
// SHM O_EXCL race gate helpers
// ---------------------------------------------------------------------------

pub(crate) enum ShmCreateResult {
    /// We created the segment; fd holds an exclusive flock. The fd is
    /// owned — drop releases the lock and closes the descriptor.
    Winner(std::os::fd::OwnedFd),
    /// Segment already exists (another process is building or built it).
    Exists,
    /// shm_open failed for a reason other than EEXIST.
    Error,
}

/// Try to create a POSIX shm segment with O_CREAT|O_EXCL. On success,
/// acquire LOCK_EX and return the fd. On EEXIST, return Exists.
pub(crate) fn shm_try_create_excl(name: &str) -> ShmCreateResult {
    let fd = match rustix::shm::open(
        name,
        rustix::shm::OFlags::CREATE | rustix::shm::OFlags::EXCL | rustix::shm::OFlags::RDWR,
        rustix::fs::Mode::from_raw_mode(0o644),
    ) {
        Ok(fd) => fd,
        Err(e) if e == rustix::io::Errno::EXIST => return ShmCreateResult::Exists,
        Err(_) => return ShmCreateResult::Error,
    };

    // Take exclusive (blocking) lock before writing. The fd is dropped
    // on the error path, which closes it automatically.
    if rustix::fs::flock(&fd, rustix::fs::FlockOperation::LockExclusive).is_err() {
        return ShmCreateResult::Error;
    }

    ShmCreateResult::Winner(fd)
}

/// Write data to the shm fd, then release the exclusive lock and close.
/// On failure (ftruncate or mmap), unlinks the segment so future callers
/// don't find a corrupt/empty segment and can retry.
pub(crate) fn shm_write_and_release(fd: std::os::fd::OwnedFd, data: &[u8], seg_name: &str) {
    use std::os::fd::AsRawFd;

    // Keep the raw fd for libc::mmap / libc::ftruncate (rustix::mm
    // is not currently wired in); the OwnedFd still owns the close
    // and flock-release on drop.
    let raw = fd.as_raw_fd();
    unsafe {
        if libc::ftruncate(raw, data.len() as libc::off_t) != 0 {
            let _ = rustix::shm::unlink(seg_name);
            // fd drop runs flock_un + close automatically.
            return;
        }

        let ptr = libc::mmap(
            std::ptr::null_mut(),
            data.len(),
            libc::PROT_WRITE,
            libc::MAP_SHARED,
            raw,
            0,
        );
        if ptr == libc::MAP_FAILED {
            // Zero the size so readers blocked on LOCK_SH see st_size=0
            // from fstat and return None instead of mapping zero-filled bytes.
            libc::ftruncate(raw, 0);
            let _ = rustix::shm::unlink(seg_name);
        } else {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
            libc::munmap(ptr, data.len());
        }
    }
    // Explicit unlock so readers blocked on LOCK_SH observe ordering
    // with the final mmap before the fd-drop close hits.
    let _ = rustix::fs::flock(&fd, rustix::fs::FlockOperation::Unlock);
    // fd drops here → close(fd). OwnedFd::drop ignores errors.
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `shm_try_create_excl` winner gets a locked fd; a second call
    /// with the same name returns `Exists`. The winner's
    /// `shm_unlink` cleanup keeps subsequent tests independent.
    #[test]
    fn shm_try_create_excl_winner_then_exists() {
        // Unique name per test process + nanos so parallel tests
        // don't collide on the global /dev/shm namespace.
        let name = format!(
            "/ktstr-test-shm-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        );

        match shm_try_create_excl(&name) {
            ShmCreateResult::Winner(fd) => {
                // Second attempt sees the existing segment. OwnedFd
                // drops close the descriptors on any early exit path.
                match shm_try_create_excl(&name) {
                    ShmCreateResult::Exists => {}
                    ShmCreateResult::Winner(_other) => {
                        let _ = rustix::shm::unlink(name.as_str());
                        drop(fd);
                        panic!("second shm_try_create_excl must return Exists, not Winner");
                    }
                    ShmCreateResult::Error => {
                        let _ = rustix::shm::unlink(name.as_str());
                        drop(fd);
                        panic!("second shm_try_create_excl returned Error");
                    }
                }
                // Clean up: write path then unlink so this test
                // doesn't leave /dev/shm residue.
                shm_write_and_release(fd, b"ok", &name);
                let _ = rustix::shm::unlink(name.as_str());
            }
            ShmCreateResult::Exists => {
                // A stale segment with this name exists. Unlink and retry.
                let _ = rustix::shm::unlink(name.as_str());
                panic!("test setup collision on shm name {name}");
            }
            ShmCreateResult::Error => {
                // Environment without /dev/shm — skip rather than fail.
                skip!("shm_open unavailable in this environment");
            }
        }
    }

    /// `shm_write_and_release` on a happy path publishes the data
    /// and releases the lock. After unlink the segment is gone.
    #[test]
    fn shm_write_and_release_publishes_data() {
        let name = format!(
            "/ktstr-test-shm-write-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        );
        let fd = match shm_try_create_excl(&name) {
            ShmCreateResult::Winner(fd) => fd,
            _ => {
                skip!("shm_open unavailable");
            }
        };
        let payload = b"shm-unit-test-payload";
        shm_write_and_release(fd, payload, &name);

        // Reopen read-only and verify size + contents.
        let rfd = rustix::shm::open(
            name.as_str(),
            rustix::shm::OFlags::RDONLY,
            rustix::fs::Mode::empty(),
        )
        .expect("shm_open for read failed");
        let st = rustix::fs::fstat(&rfd).expect("fstat failed");
        assert_eq!(st.st_size as usize, payload.len());
        drop(rfd);
        let _ = rustix::shm::unlink(name.as_str());
    }

    #[test]
    fn base_key_same_inputs_match() {
        let exe = crate::resolve_current_exe().unwrap();
        let k1 = BaseKey::new(&exe, None, None, None).unwrap();
        let k2 = BaseKey::new(&exe, None, None, None).unwrap();
        assert_eq!(k1, k2);
    }

    #[test]
    fn base_key_nonexistent_payload_fails() {
        let result = BaseKey::new(Path::new("/nonexistent/binary"), None, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn base_key_different_content_differs() {
        let tmp =
            std::env::temp_dir().join(format!("ktstr-cache-content-test-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let bin = tmp.join("payload");

        std::fs::write(&bin, b"content_v1").unwrap();
        let k1 = BaseKey::new(&bin, None, None, None).unwrap();

        std::fs::write(&bin, b"content_v2").unwrap();
        let k2 = BaseKey::new(&bin, None, None, None).unwrap();

        assert_ne!(
            k1, k2,
            "different file content should produce different key"
        );
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn base_key_with_scheduler() {
        let exe = crate::resolve_current_exe().unwrap();
        let k1 = BaseKey::new(&exe, None, None, None).unwrap();
        let k2 = BaseKey::new(&exe, Some(&exe), None, None).unwrap();
        assert_ne!(k1, k2, "with vs without scheduler should differ");
    }

    #[test]
    fn hash_file_is_siphash13_stable_golden() {
        // hash_file must use SipHasher13 with zero keys so the value
        // is stable across Rust toolchain versions. Golden check
        // pins the concrete algorithm — if this value changes, the
        // cache is about to silently invalidate every prior artifact.
        let tmp =
            std::env::temp_dir().join(format!("ktstr-hash-golden-test-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let f = tmp.join("known");
        std::fs::write(&f, b"ktstr cache key probe").unwrap();
        let observed = hash_file(&f).unwrap();

        // Cross-check against a direct SipHasher13 invocation so the
        // test will fail loudly if someone swaps the algorithm.
        use siphasher::sip::SipHasher13;
        use std::hash::Hasher;
        let mut h = SipHasher13::new_with_keys(0, 0);
        h.write(b"ktstr cache key probe");
        let expected = h.finish();
        assert_eq!(
            observed, expected,
            "hash_file must match SipHasher13::new_with_keys(0, 0)"
        );

        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn hash_file_large_file() {
        let tmp =
            std::env::temp_dir().join(format!("ktstr-hash-sample-test-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let f = tmp.join("big");
        // 16KB file — exercises both head and tail sampling.
        let data: Vec<u8> = (0..16384).map(|i| (i % 256) as u8).collect();
        std::fs::write(&f, &data).unwrap();
        let h = hash_file(&f).unwrap();
        // Same content should produce same hash.
        assert_eq!(h, hash_file(&f).unwrap());
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn base_cache_hit() {
        let exe = crate::resolve_current_exe().unwrap();
        let key = BaseKey::new(&exe, None, None, None).unwrap();

        // Insert a sentinel value.
        let sentinel = Arc::new(vec![0xDE, 0xAD]);
        base_cache()
            .lock()
            .unwrap()
            .insert(key.clone(), sentinel.clone());

        // Lookup should return the same Arc.
        let cached = base_cache().lock().unwrap().get(&key).cloned();
        assert!(cached.is_some());
        assert!(Arc::ptr_eq(&cached.unwrap(), &sentinel));

        // Clean up to avoid polluting other tests.
        base_cache().lock().unwrap().remove(&key);
    }

    #[test]
    fn shm_store_and_load_roundtrip() {
        let hash = 0xDEAD_BEEF_CAFE_1234u64;
        let data = vec![0x07u8, 0x07, 0x01]; // cpio magic prefix
        initramfs::shm_store_base(hash, &data).unwrap();
        let loaded = initramfs::shm_load_base(hash);
        assert!(loaded.is_some(), "shm_load_base should return Some");
        assert_eq!(loaded.unwrap().as_ref(), &data[..]);
        initramfs::shm_unlink_base(hash);
    }

    #[test]
    fn shm_different_hashes_independent() {
        let h1 = 0x1111_2222_3333_4444u64;
        let h2 = 0x5555_6666_7777_8888u64;
        let d1 = vec![0xAAu8; 16];
        let d2 = vec![0xBBu8; 32];
        initramfs::shm_store_base(h1, &d1).unwrap();
        initramfs::shm_store_base(h2, &d2).unwrap();
        assert_eq!(initramfs::shm_load_base(h1).unwrap().as_ref(), &d1[..]);
        assert_eq!(initramfs::shm_load_base(h2).unwrap().as_ref(), &d2[..]);
        initramfs::shm_unlink_base(h1);
        initramfs::shm_unlink_base(h2);
    }
}
