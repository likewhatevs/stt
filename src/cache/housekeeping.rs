//! Atomic-rename install primitives, cache-key validators, and
//! orphan-tempdir sweep for the kernel image cache.
//!
//! Exports:
//! - [`atomic_swap_dirs`] — `renameat2(RENAME_EXCHANGE)` wrapper
//!   used by [`super::cache_dir::CacheDir::store`] to publish a
//!   freshly-built cache entry over an existing one without ever
//!   leaving readers observing a partial state.
//! - [`TmpDirGuard`] — RAII drop guard that unlinks an
//!   in-progress staging directory on any error path; pairs with
//!   [`super::TMP_DIR_PREFIX`] to keep the cache root self-cleaning.
//! - [`read_metadata`] — metadata.json deserializer; the producer
//!   side of the prefix → kind contract documented on
//!   [`super::metadata::classify_corrupt_reason`].
//! - [`clean_orphaned_tmp_dirs`] — cross-PID GC sweep that removes
//!   `.tmp-{key}-{pid}` directories when `{pid}` is no longer a
//!   live process. Run opportunistically by `store()` to keep the
//!   cache root from accumulating dead writes after a writer crash.
//! - [`validate_cache_key`] / [`validate_filename`] — input
//!   sanitisers that reject path traversal, separators, NUL,
//!   leading dot, the `TMP_DIR_PREFIX` reservation, and any name
//!   that could escape the cache root. Both run before any I/O so
//!   bad input fails fast rather than half-writing a malformed
//!   entry.
//!
//! Sibling modules:
//! - [`super::metadata`] — pure types and the
//!   [`super::metadata::classify_corrupt_reason`] dispatcher whose
//!   prefix list `read_metadata` is the producer for.
//! - [`super::cache_dir`] — orchestrates `store`/`lookup`/
//!   `list`/`clean`, calling into every helper here.
//! - [`super::resolve`] — supplies the cache root path that
//!   `clean_orphaned_tmp_dirs` walks.
//!
//! No public API in this module is `pub` — every helper is
//! `pub(crate)` and only reachable through `super::cache_dir`.

use std::fs;
use std::path::Path;

use super::TMP_DIR_PREFIX;
use super::metadata::KernelMetadata;

/// Rejects empty keys, whitespace-only keys, keys starting with
/// `.tmp-` (reserved for in-progress stores), and keys containing
/// path separators (`/`, `\`), parent-directory traversal (`..`),
/// or null bytes. Returns `Ok(())` on valid keys.
pub(crate) fn validate_cache_key(key: &str) -> anyhow::Result<()> {
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
pub(crate) fn validate_filename(name: &str) -> anyhow::Result<()> {
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
pub(crate) struct TmpDirGuard<'a>(pub(crate) &'a Path);

impl Drop for TmpDirGuard<'_> {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(self.0);
    }
}

/// Atomically swap two filesystem paths via renameat2(RENAME_EXCHANGE).
pub(crate) fn atomic_swap_dirs(src: &Path, dst: &Path) -> anyhow::Result<()> {
    rustix::fs::renameat_with(
        rustix::fs::CWD,
        src,
        rustix::fs::CWD,
        dst,
        rustix::fs::RenameFlags::EXCHANGE,
    )
    .map_err(|e| {
        anyhow::anyhow!(
            "renameat2(RENAME_EXCHANGE) {} <-> {}: {e}",
            src.display(),
            dst.display(),
        )
    })
}

/// Read and deserialize metadata.json from a cache entry directory.
///
/// On failure returns a human-readable reason with a distinct prefix
/// per failure mode (missing / unreadable / schema-drift / malformed
/// / truncated / parse error). Prefix consumers key on
/// [`super::metadata::classify_corrupt_reason`].
///
/// **Producer↔classifier contract.** The reason strings emitted
/// below are the authoritative source of truth for the JSON
/// `error_kind` field that `cargo ktstr kernel list --json`
/// surfaces. Each `Err(format!("metadata.json …: {e}"))` arm in
/// this function corresponds to exactly one row in the prefix→kind
/// table documented on
/// [`super::metadata::classify_corrupt_reason`]. If you add a new
/// failure mode here, both that classifier dispatcher and the
/// `classify_corrupt_reason_covers_every_documented_prefix` test
/// (in `metadata.rs`) MUST be updated in lockstep — silently
/// adding an unrecognised prefix here drops the new failure into
/// the catch-all `"unknown"` bucket and breaks scripted consumers
/// dispatching on `error_kind`.
pub(crate) fn read_metadata(dir: &Path) -> Result<KernelMetadata, String> {
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

/// Scan `cache_root` for `.tmp-{key}-{pid}` directories whose `{pid}`
/// is no longer a live process and remove them.
///
/// Cross-PID orphan sweep. `kill(pid, None)` returning `Err(ESRCH)`
/// is the only outcome that justifies removal; alive / EPERM
/// preserve.
pub(crate) fn clean_orphaned_tmp_dirs(cache_root: &Path) -> anyhow::Result<()> {
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
            Err(_) => continue,
        };
        if !name.starts_with(TMP_DIR_PREFIX) {
            continue;
        }
        let pid_str = match name.rsplit_once('-') {
            Some((_, suffix)) if !suffix.is_empty() => suffix,
            _ => continue,
        };
        let pid: i32 = match pid_str.parse() {
            Ok(p) => p,
            Err(_) => continue,
        };
        if pid <= 0 {
            continue;
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // -- clean_orphaned_tmp_dirs unit tests --
    //
    // Parser/dispatcher coverage: the scan must remove directories
    // under `.tmp-{key}-{pid}` whose `{pid}` is verifiably dead,
    // must LEAVE malformed entries and non-`.tmp-` entries alone,
    // and must tolerate a nonexistent cache root.

    /// A `.tmp-{key}-{pid}` directory whose pid refers to a dead
    /// process is removed.
    #[test]
    fn clean_orphaned_tmp_dirs_removes_dead_pid_tempdir() {
        let tmp = TempDir::new().unwrap();
        let dead_pid = libc::pid_t::MAX;
        let orphan = tmp
            .path()
            .join(format!("{TMP_DIR_PREFIX}somekey-{dead_pid}"));
        std::fs::create_dir_all(&orphan).unwrap();
        std::fs::write(orphan.join("inner.txt"), b"data").unwrap();

        clean_orphaned_tmp_dirs(tmp.path()).unwrap();
        assert!(
            !orphan.exists(),
            "dead-pid tempdir must be removed by clean_orphaned_tmp_dirs",
        );
    }

    /// A `.tmp-{key}-{pid}` directory whose pid is LIVE (the test
    /// process itself) must be preserved.
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
    /// or empty after the trailing `-`) must be left alone.
    #[test]
    fn clean_orphaned_tmp_dirs_leaves_malformed_suffix_alone() {
        let tmp = TempDir::new().unwrap();
        let nonnum = tmp.path().join(format!("{TMP_DIR_PREFIX}somekey-notapid"));
        std::fs::create_dir_all(&nonnum).unwrap();
        let empty_suf = tmp.path().join(format!("{TMP_DIR_PREFIX}somekey-"));
        std::fs::create_dir_all(&empty_suf).unwrap();
        let no_dash = tmp.path().join(format!("{TMP_DIR_PREFIX}nokeyhere"));
        std::fs::create_dir_all(&no_dash).unwrap();

        clean_orphaned_tmp_dirs(tmp.path()).unwrap();
        assert!(nonnum.exists(), "non-numeric pid suffix must be left alone");
        assert!(empty_suf.exists(), "empty pid suffix must be left alone");
        assert!(no_dash.exists(), "no-pid-suffix entry must be left alone");
    }

    /// Directories that do not begin with [`TMP_DIR_PREFIX`] must
    /// never be touched.
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

    /// Non-UTF-8 filenames in the cache root must be skipped silently.
    #[test]
    #[cfg(unix)]
    fn clean_orphaned_tmp_dirs_skips_non_utf8_names() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        let tmp = TempDir::new().unwrap();
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

    /// A nonexistent cache root returns `Ok(())` without error.
    #[test]
    fn clean_orphaned_tmp_dirs_handles_missing_cache_root() {
        let tmp = TempDir::new().unwrap();
        let never_created = tmp.path().join("never-created");
        clean_orphaned_tmp_dirs(&never_created).unwrap();
    }

    /// Multi-entry mix: a DEAD-pid orphan and a LIVE-pid tempdir
    /// side by side — only the dead one is removed.
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

    /// `pid == 0` suffix: the scan rejects non-positive pids before
    /// the liveness probe runs.
    #[test]
    fn clean_orphaned_tmp_dirs_preserves_pid_zero_suffix() {
        let tmp = TempDir::new().unwrap();
        let entry = tmp.path().join(format!("{TMP_DIR_PREFIX}somekey-0"));
        std::fs::create_dir_all(&entry).unwrap();
        clean_orphaned_tmp_dirs(tmp.path()).unwrap();
        assert!(
            entry.exists(),
            "pid=0 suffix must be preserved — `pid <= 0` filter \
             skips the entry before kill(0, None)'s pgrp-broadcast \
             ambiguity can reach the liveness probe",
        );
    }

    /// Documents that `rsplit_once('-')` parses double-dash suffix
    /// as a positive pid, never negative.
    #[test]
    fn clean_orphaned_tmp_dirs_double_dash_parses_as_positive_pid() {
        let tmp = TempDir::new().unwrap();
        let entry = tmp.path().join(format!("{TMP_DIR_PREFIX}somekey--12345"));
        std::fs::create_dir_all(&entry).unwrap();
        clean_orphaned_tmp_dirs(tmp.path()).unwrap();

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
    /// `.tmp-{key}-{pid}` pattern with a dead pid stays in place.
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
    /// whose TARGET is an unrelated path outside the cache.
    #[test]
    #[cfg(unix)]
    fn clean_orphaned_tmp_dirs_leaves_symlink_entry() {
        let tmp = TempDir::new().unwrap();
        let target_root = TempDir::new().unwrap();
        let target_file = target_root.path().join("sentinel.txt");
        std::fs::write(&target_file, b"must-not-be-deleted").unwrap();

        let dead_pid = libc::pid_t::MAX;
        let symlink = tmp
            .path()
            .join(format!("{TMP_DIR_PREFIX}symkey-{dead_pid}"));
        std::os::unix::fs::symlink(target_root.path(), &symlink).unwrap();

        clean_orphaned_tmp_dirs(tmp.path()).unwrap();

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
}
