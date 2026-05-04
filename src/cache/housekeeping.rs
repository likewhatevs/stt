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

/// Rejects empty keys, whitespace-only keys, keys starting with `.`
/// (reserved for ktstr bookkeeping — `.locks/`, `.tmp-*`), and keys
/// containing path separators (`/`, `\`), parent-directory traversal
/// (`..`), or null bytes. Returns `Ok(())` on valid keys.
///
/// The leading-dot rejection mirrors `CacheDir::list`'s dotfile
/// filter: every name starting with `.` is treated as ktstr
/// bookkeeping and skipped at list-time, so admitting a dotfile key
/// at store-time would create a silent divergence (the entry is
/// stored on disk but invisible to `list`). Reject up front to make
/// the divergence impossible by construction. The `.tmp-` arm is
/// retained as a more-specific error message because the
/// `TMP_DIR_PREFIX` reservation is the externally-documented contract
/// and operator-facing diagnostics name it explicitly.
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
    if key.starts_with('.') {
        anyhow::bail!(
            "cache key must not start with `.` (reserved for ktstr \
             bookkeeping; `list` skips every dotfile child): {key:?}",
        );
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
        // silent: clean_orphaned_tmp_dirs sweeps any leftover on the next store()
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
        // pid_t::MAX (i32::MAX = 2147483647) is well beyond Linux's
        // PID_MAX_LIMIT (4194304 on 64-bit). No real PID can match,
        // so kill(MAX, 0) returns ESRCH deterministically. Same
        // sentinel is reused at the other dead-pid sites in this
        // module.
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
        // pid_t::MAX sentinel — see comment in
        // `clean_orphaned_tmp_dirs_removes_dead_pid_tempdir` above.
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
             skips before the liveness probe so non-positive parses \
             cannot reach kill()",
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
        // pid_t::MAX sentinel — see comment in
        // `clean_orphaned_tmp_dirs_removes_dead_pid_tempdir` above.
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

        // pid_t::MAX sentinel — see comment in
        // `clean_orphaned_tmp_dirs_removes_dead_pid_tempdir` above.
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

    /// Any leading-dot key (not just `.tmp-`) is rejected because
    /// `CacheDir::list`'s dotfile filter would skip it — admitting it
    /// at store-time would produce an entry that exists on disk but
    /// is invisible to `list`. The error message names the
    /// bookkeeping reservation so an operator who hits the rejection
    /// understands why their key was refused.
    #[test]
    fn cache_validate_key_rejects_other_leading_dots() {
        for bad in [".locks", ".bookkeeping", ".my-key"] {
            let err = validate_cache_key(bad).unwrap_err();
            assert!(
                err.to_string().contains("must not start with `.`"),
                "expected leading-dot rejection for {bad:?}, got: {err}",
            );
        }
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

    // -- atomic_swap_dirs direct unit tests --
    //
    // The swap is the publish step of `CacheDir::store` when the
    // destination cache_key already exists; it must atomically
    // swap two existing directory inodes via
    // renameat2(RENAME_EXCHANGE) so a concurrent reader never sees
    // a partial state. Direct coverage exercises the kernel
    // syscall's semantics (both sides materialised, neither lost,
    // contents preserved by reference rather than copy) without
    // the `store()` orchestration on top.

    /// Happy path: two existing directories swap their on-disk
    /// contents in a single atomic operation. Verifies both the
    /// content-swap observable AND that the underlying directory
    /// inodes are preserved across the swap (renameat2 swaps
    /// dentries, not contents — a regression to a copy-based
    /// fallback would observably change the inode numbers).
    #[test]
    #[cfg(unix)]
    fn atomic_swap_dirs_exchanges_two_existing_directories() {
        use std::os::unix::fs::MetadataExt;
        let tmp = TempDir::new().unwrap();
        let a = tmp.path().join("alpha");
        let b = tmp.path().join("bravo");
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        std::fs::write(a.join("payload"), b"alpha-bytes").unwrap();
        std::fs::write(b.join("payload"), b"bravo-bytes").unwrap();
        let a_ino_before = std::fs::metadata(&a).unwrap().ino();
        let b_ino_before = std::fs::metadata(&b).unwrap().ino();

        atomic_swap_dirs(&a, &b).expect("happy-path swap must succeed");

        assert_eq!(
            std::fs::read(a.join("payload")).unwrap(),
            b"bravo-bytes",
            "after RENAME_EXCHANGE, the path `a` must reference the \
             contents that were under `b` before the swap",
        );
        assert_eq!(
            std::fs::read(b.join("payload")).unwrap(),
            b"alpha-bytes",
            "after RENAME_EXCHANGE, the path `b` must reference the \
             contents that were under `a` before the swap",
        );
        let a_ino_after = std::fs::metadata(&a).unwrap().ino();
        let b_ino_after = std::fs::metadata(&b).unwrap().ino();
        assert_eq!(
            a_ino_after, b_ino_before,
            "inode at path `a` must equal the pre-swap inode at `b` — \
             a copy-based fallback would assign a fresh inode here",
        );
        assert_eq!(
            b_ino_after, a_ino_before,
            "inode at path `b` must equal the pre-swap inode at `a` — \
             a copy-based fallback would assign a fresh inode here",
        );
    }

    /// `RENAME_EXCHANGE` requires BOTH endpoints to exist. A
    /// missing source must surface as an error rather than silently
    /// creating one or losing data — the diagnostic must name both
    /// paths so the operator can pinpoint the missing side.
    #[test]
    fn atomic_swap_dirs_missing_source_surfaces_error() {
        let tmp = TempDir::new().unwrap();
        let nonexistent = tmp.path().join("never-created");
        let dst = tmp.path().join("dst");
        std::fs::create_dir_all(&dst).unwrap();
        let err = atomic_swap_dirs(&nonexistent, &dst)
            .expect_err("missing source must produce an Err");
        let msg = format!("{err:#}");
        assert!(
            msg.contains(&nonexistent.display().to_string()),
            "diagnostic must name the missing source path: {msg}",
        );
        assert!(
            msg.contains(&dst.display().to_string()),
            "diagnostic must also name the destination path: {msg}",
        );
        assert!(
            dst.exists(),
            "destination must remain in place when the swap fails",
        );
    }

    /// Symmetric: a missing destination must produce an actionable
    /// error rather than a silent rename.
    #[test]
    fn atomic_swap_dirs_missing_destination_surfaces_error() {
        let tmp = TempDir::new().unwrap();
        let src = tmp.path().join("src");
        std::fs::create_dir_all(&src).unwrap();
        let nonexistent = tmp.path().join("never-created");
        let err = atomic_swap_dirs(&src, &nonexistent)
            .expect_err("missing destination must produce an Err");
        let msg = format!("{err:#}");
        assert!(
            msg.contains(&src.display().to_string())
                && msg.contains(&nonexistent.display().to_string()),
            "diagnostic must name BOTH endpoints so the operator \
             can attribute the failure: {msg}",
        );
        assert!(
            src.exists(),
            "source must remain in place when the swap fails",
        );
    }

    /// Swap preserves arbitrary subtree shape — multiple files,
    /// nested subdirs — by inode reference rather than recursive
    /// copy. A regression that fell back to copy-then-rename would
    /// be observable through changes to inode identity (file
    /// metadata.ino()) but the simpler observable check is that
    /// the swap is fast and doesn't traverse contents: we rely on
    /// content equality post-swap as the proxy assertion.
    #[test]
    fn atomic_swap_dirs_preserves_subtree_shape() {
        let tmp = TempDir::new().unwrap();
        let a = tmp.path().join("alpha");
        let b = tmp.path().join("bravo");
        std::fs::create_dir_all(a.join("nested/deep")).unwrap();
        std::fs::create_dir_all(b.join("other")).unwrap();
        std::fs::write(a.join("nested/deep/leaf"), b"alpha-leaf").unwrap();
        std::fs::write(a.join("top"), b"alpha-top").unwrap();
        std::fs::write(b.join("other/file"), b"bravo-file").unwrap();

        atomic_swap_dirs(&a, &b).expect("subtree swap must succeed");

        assert_eq!(
            std::fs::read(a.join("other/file")).unwrap(),
            b"bravo-file",
            "post-swap, `a` must contain the original `b` subtree",
        );
        assert_eq!(
            std::fs::read(b.join("nested/deep/leaf")).unwrap(),
            b"alpha-leaf",
            "post-swap, `b` must contain the original `a` subtree",
        );
        assert_eq!(
            std::fs::read(b.join("top")).unwrap(),
            b"alpha-top",
            "all files in the swapped subtree must remain reachable",
        );
    }

    // -- read_metadata direct unit tests --
    //
    // `read_metadata` is the producer half of the prefix→kind
    // contract documented on `metadata::classify_corrupt_reason`.
    // The per-failure-mode prefixes are surfaced as `error_kind`
    // strings via `kernel list --json`, so each prefix is part of
    // the JSON contract and needs direct coverage that doesn't
    // require driving a full `CacheDir::list` cycle.

    /// Happy path: a valid metadata.json deserializes into a
    /// `KernelMetadata` whose required fields round-trip.
    #[test]
    fn read_metadata_happy_path_parses_valid_json() {
        use super::super::metadata::KernelSource;
        let tmp = TempDir::new().unwrap();
        let entry_dir = tmp.path().join("entry");
        std::fs::create_dir_all(&entry_dir).unwrap();
        let meta = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64".to_string(),
            "bzImage".to_string(),
            "2026-04-12T10:00:00Z".to_string(),
        );
        let json = serde_json::to_string(&meta).unwrap();
        std::fs::write(entry_dir.join("metadata.json"), &json).unwrap();

        let parsed = read_metadata(&entry_dir).expect("valid metadata must parse");
        assert_eq!(parsed.image_name, "bzImage");
        assert_eq!(parsed.arch, "x86_64");
        assert_eq!(parsed.built_at, "2026-04-12T10:00:00Z");
    }

    /// Missing metadata.json → exact reason string `"metadata.json
    /// missing"`. The string is the input the classifier dispatches
    /// on for the `"missing"` error_kind, so the EXACT spelling is
    /// part of the JSON contract.
    #[test]
    fn read_metadata_missing_returns_exact_missing_reason() {
        let tmp = TempDir::new().unwrap();
        let entry_dir = tmp.path().join("entry");
        std::fs::create_dir_all(&entry_dir).unwrap();

        let reason = read_metadata(&entry_dir)
            .expect_err("absent metadata.json must produce an Err");
        assert_eq!(
            reason, "metadata.json missing",
            "exact missing reason is the classifier dispatch key for `missing`",
        );
    }

    /// metadata.json shaped as a directory rather than a file →
    /// `"metadata.json unreadable: …"` prefix. `read_to_string` on
    /// a directory returns `EISDIR`, surfaced through the
    /// `Err(_) => "unreadable"` arm of the producer.
    #[test]
    fn read_metadata_unreadable_returns_unreadable_prefix() {
        let tmp = TempDir::new().unwrap();
        let entry_dir = tmp.path().join("entry");
        std::fs::create_dir_all(&entry_dir).unwrap();
        // Materialise metadata.json as a DIRECTORY — read_to_string
        // surfaces EISDIR which is neither NotFound nor a successful
        // read. Drives the `Err(e) => unreadable` arm.
        std::fs::create_dir_all(entry_dir.join("metadata.json")).unwrap();

        let reason = read_metadata(&entry_dir)
            .expect_err("metadata.json shaped as a directory must produce an Err");
        assert!(
            reason.starts_with("metadata.json unreadable: "),
            "EISDIR-on-read must surface under the `unreadable` prefix \
             so the classifier dispatches to error_kind=unreadable; \
             got: {reason}",
        );
    }

    /// Malformed JSON (`Category::Syntax`) → `"metadata.json
    /// malformed: "` prefix. The exact prefix is documented on
    /// `metadata::classify_corrupt_reason` as the dispatch key for
    /// the `"malformed"` error_kind.
    #[test]
    fn read_metadata_malformed_json_returns_malformed_prefix() {
        let tmp = TempDir::new().unwrap();
        let entry_dir = tmp.path().join("entry");
        std::fs::create_dir_all(&entry_dir).unwrap();
        std::fs::write(entry_dir.join("metadata.json"), b"not valid json {[").unwrap();

        let reason = read_metadata(&entry_dir).expect_err("malformed JSON must produce an Err");
        assert!(
            reason.starts_with("metadata.json malformed: "),
            "syntax-error JSON must surface under the `malformed` prefix; \
             got: {reason}",
        );
    }

    /// Truncated JSON (`Category::Eof`) → `"metadata.json
    /// truncated: "` prefix.
    #[test]
    fn read_metadata_truncated_json_returns_truncated_prefix() {
        let tmp = TempDir::new().unwrap();
        let entry_dir = tmp.path().join("entry");
        std::fs::create_dir_all(&entry_dir).unwrap();
        // Truncated mid-value — Category::Eof.
        std::fs::write(entry_dir.join("metadata.json"), br#"{"source":"#).unwrap();

        let reason = read_metadata(&entry_dir).expect_err("truncated JSON must produce an Err");
        assert!(
            reason.starts_with("metadata.json truncated: "),
            "EOF-mid-parse must surface under the `truncated` prefix; \
             got: {reason}",
        );
    }

    /// Missing required field (`Category::Data`) → `"metadata.json
    /// schema drift: "` prefix.
    #[test]
    fn read_metadata_schema_drift_returns_schema_drift_prefix() {
        let tmp = TempDir::new().unwrap();
        let entry_dir = tmp.path().join("entry");
        std::fs::create_dir_all(&entry_dir).unwrap();
        // Valid JSON, but missing every required `KernelMetadata`
        // field — Category::Data.
        std::fs::write(entry_dir.join("metadata.json"), br#"{"version": "6.14"}"#).unwrap();

        let reason = read_metadata(&entry_dir)
            .expect_err("incomplete JSON must produce an Err");
        assert!(
            reason.starts_with("metadata.json schema drift: "),
            "missing required field must surface under the `schema drift` \
             prefix; got: {reason}",
        );
    }

    /// Producer-classifier round-trip: every direct producer call
    /// surfaces a prefix that the classifier dispatches into a
    /// non-`unknown` `error_kind`. Locks the documented contract
    /// at the producer side without dragging in the consumer-side
    /// table-driven test.
    #[test]
    fn read_metadata_every_failure_mode_is_classifier_recognised() {
        use super::super::metadata::classify_corrupt_reason;
        let tmp = TempDir::new().unwrap();

        // missing
        let entry = tmp.path().join("absent");
        std::fs::create_dir_all(&entry).unwrap();
        let reason = read_metadata(&entry).unwrap_err();
        assert_eq!(classify_corrupt_reason(&reason), "missing");

        // unreadable
        let entry = tmp.path().join("isdir");
        std::fs::create_dir_all(entry.join("metadata.json")).unwrap();
        let reason = read_metadata(&entry).unwrap_err();
        assert_eq!(classify_corrupt_reason(&reason), "unreadable");

        // malformed
        let entry = tmp.path().join("malformed");
        std::fs::create_dir_all(&entry).unwrap();
        std::fs::write(entry.join("metadata.json"), b"not valid json {[").unwrap();
        let reason = read_metadata(&entry).unwrap_err();
        assert_eq!(classify_corrupt_reason(&reason), "malformed");

        // truncated
        let entry = tmp.path().join("truncated");
        std::fs::create_dir_all(&entry).unwrap();
        std::fs::write(entry.join("metadata.json"), br#"{"source":"#).unwrap();
        let reason = read_metadata(&entry).unwrap_err();
        assert_eq!(classify_corrupt_reason(&reason), "truncated");

        // schema_drift
        let entry = tmp.path().join("schema-drift");
        std::fs::create_dir_all(&entry).unwrap();
        std::fs::write(entry.join("metadata.json"), br#"{"version":"6.14"}"#).unwrap();
        let reason = read_metadata(&entry).unwrap_err();
        assert_eq!(classify_corrupt_reason(&reason), "schema_drift");
    }
}
