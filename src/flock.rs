//! Advisory flock(2) primitives shared across every ktstr lock file.
//!
//! ktstr uses advisory `flock(2)` in three places:
//!
//!  - LLC reservation locks at `/tmp/ktstr-llc-{N}.lock` and per-CPU
//!    locks at `/tmp/ktstr-cpu-{C}.lock` (see
//!    `crate::vmm::host_topology::acquire_resource_locks` and
//!    friends).
//!  - Per-cache-entry coordination locks at
//!    `{cache_root}/.locks/{cache_key}.lock` (see
//!    `crate::cache::CacheDir::acquire_shared_lock` and friends).
//!  - Observational enumeration from `ktstr locks --json` — a
//!    read-only scan that does NOT acquire flocks; reads
//!    /proc/locks through [`read_holders`] to attribute holders
//!    without contending with active acquirers.
//!
//! All three share:
//!  - Non-blocking `LOCK_NB` attempt (the cache-entry path wraps this
//!    in a poll loop for timed-wait semantics).
//!  - `O_CLOEXEC` on every open so the kernel's "release flock when
//!    the last fd referring to the OFD closes" invariant matches what
//!    `OwnedFd::drop` does — a leaked fd across `exec(2)` would keep
//!    the lock alive in the child and fool the next acquirer's
//!    `/proc/locks` scan into naming the wrong pid.
//!  - /proc/locks parsing keyed on the mount-point-derived
//!    `{major:02x}:{minor:02x}:{inode}` triple, resolved via
//!    `/proc/self/mountinfo` (not `stat().st_dev` — see below).
//!  - `HolderInfo` with `pid` + truncated `/proc/{pid}/cmdline` for
//!    actionable error messages.
//!
//! # Why mountinfo (and fdinfo), not `stat().st_dev`
//!
//! `/proc/locks` emits `i_sb->s_dev` for each held flock — the
//! filesystem's superblock device id. For most filesystems that
//! matches `stat().st_dev`, but on btrfs, overlayfs, and bind-mounts
//! the kernel installs a custom `getattr` implementation that returns
//! an anonymous device id (`anon_dev`) distinct from `s_dev`. That
//! divergence means the stat-derived needle would never match the
//! /proc/locks line — a naive `read_holders()` would silently return
//! empty on every btrfs-backed `/tmp`, every overlay-rootfs
//! container, and every bind-mounted /tmp, which is a silent
//! correctness failure for `--llc-cap` contention diagnostics and
//! the `ktstr locks` observational command.
//!
//! Two needle producers, one format:
//!
//!  - [`needle_from_path`] — path-only callers. Resolves `path` to
//!    the mount-point covering it via `/proc/self/mountinfo`
//!    (longest-prefix match on the mount_point field), then reads
//!    the `{major:minor}` field of that mount entry. Combines with
//!    `stat().st_ino` for the full triple. The mountinfo
//!    `{major:minor}` is the kernel's `i_sb->s_dev` verbatim, so
//!    the resulting needle matches /proc/locks by construction.
//!  - [`needle_from_fd`] — callers that hold a freshly-flocked
//!    [`OwnedFd`]. Reads the fd's `/proc/self/fdinfo/{fd}` and
//!    extracts the `lock:` line's `{major:02x}:{minor:02x}:{inode}`
//!    triple verbatim from the kernel's own formatting. No
//!    mountinfo parse, no per-host stat() round-trip; works even
//!    when the path has been unlinked while the fd is still open.
//!
//! Both producers feed [`read_holders_for_needle`], which scans
//! `/proc/locks` exactly once and byte-compares. All in-tree
//! callers today are path-only (they call [`read_holders`], which
//! is the path-only adapter); `needle_from_fd` is exposed for
//! future telemetry and cross-check tests.
//!
//! # Remote-filesystem rejection
//!
//! [`try_flock`] refuses to operate on NFS / CIFS / SMB2 / CEPH /
//! AFS / FUSE (see [`reject_remote_fs`]). `flock(2)` on those
//! filesystems is either advisory-only under some server
//! configurations (NFSv3 without NLM coordination) or silently
//! returns success without serializing peers (FUSE when the
//! userspace server doesn't implement the flock op). ktstr's
//! resource-budget contract is not robust to that silent
//! degradation, so the safe call is to reject at lockfile-open
//! time with an actionable message.

use anyhow::Result;
use serde::Serialize;
use std::os::fd::OwnedFd;
use std::path::{Path, PathBuf};

/// Cmdline truncation limit. Matches the 100-char cap shared with the
/// rest of the crate's user-facing diagnostic output.
pub(crate) const CMDLINE_MAX_CHARS: usize = 100;

/// Diagnostic text for lock-holder error messages when /proc/locks
/// lists no PID against the lockfile inode. Centralized so every
/// caller renders the empty-holders case with the same string.
/// Non-empty so log-scrapers can key on it without accidentally
/// matching a blank field.
pub(crate) const NO_HOLDERS_RECORDED: &str = "<none recorded>";

/// Requested sharing mode for [`try_flock`]. Translated to the
/// corresponding non-blocking [`rustix::fs::FlockOperation`]
/// internally; callers never see the libc-specific constants.
///
/// Shared between LLC + per-CPU flocks (`vmm::host_topology`) and
/// cache-entry flocks (`cache`). A single type prevents the
/// three-enum drift the convergence review flagged (earlier revisions
/// had `FlockMode` + `FlockKind` + `LlcLockMode` with identical
/// shape). `LlcLockMode` remains distinct as the scheduler-intent
/// layer (perf-mode vs. no-perf-mode request), not a flock operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlockMode {
    /// Exclusive (`LOCK_EX`) — sole access to the lock file.
    Exclusive,
    /// Shared (`LOCK_SH`) — multiple holders can coexist.
    Shared,
}

/// Identity of a process holding an advisory flock. Used by error
/// messages in both LLC-coordination and cache-entry paths, plus the
/// `ktstr locks` observational subcommand.
///
/// Cmdline is read from `/proc/{pid}/cmdline`, NUL-separated by the
/// kernel, lossy-UTF-8 decoded, `\0 → space`, and truncated to
/// [`CMDLINE_MAX_CHARS`] chars with a `…` marker so a log line remains
/// single-line. A missing / racing / permission-denied
/// `/proc/{pid}/cmdline` produces `"<cmdline unavailable>"` so the pid
/// still surfaces with diagnostic value.
///
/// `#[non_exhaustive]` so future fields (`start_time`, `fd_count`,
/// etc.) don't break external match arms or struct literals. Derives
/// `Serialize` (with `snake_case` field renaming for JSON schema
/// stability) for the `ktstr locks --json` surface; no `Deserialize`
/// because this type is produced-only.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub struct HolderInfo {
    /// PID of the flock holder as reported by `/proc/locks`.
    pub pid: u32,
    /// Truncated `/proc/{pid}/cmdline` of the holder process.
    pub cmdline: String,
}

/// Filesystem-magic constants for [`reject_remote_fs`]. Values from
/// `<linux/magic.h>`. Kept as a deny-list (reject known-bad) rather
/// than an allow-list — exotic local filesystems (zfs, erofs, …)
/// are safer to accept with unreliable flock than to reject.
mod fs_magic {
    /// `nfs_super_magic` — NFSv2/3/4 mounts. `flock(2)` on NFS is
    /// advisory-only under some server configurations; rejecting
    /// at lockfile open prevents silent false-success.
    pub(super) const NFS: i64 = 0x6969;
    /// `cifs_magic_number` — CIFS / SMB1.
    pub(super) const CIFS: i64 = 0xFF53_4D42;
    /// `smb2_magic_number` — SMB2+. Distinct from the CIFS constant.
    pub(super) const SMB2: i64 = 0xFE53_4D42;
    /// `ceph_super_magic` — CephFS.
    pub(super) const CEPH: i64 = 0x00c3_6400;
    /// `AFS_FS_MAGIC` — the in-tree kAFS client's superblock magic
    /// (`fs/afs/super.c:460`, linux/magic.h defines the constant as
    /// `0x6B414653`). Distinct from the legacy `AFS_SUPER_MAGIC =
    /// 0x5346414F` that lingers in `<linux/magic.h>` but is not
    /// emitted by any in-tree AFS driver today — we only reject
    /// what the running kernel actually reports.
    pub(super) const AFS: i64 = 0x6B41_4653;
    /// `FUSE_SUPER_MAGIC` — any FUSE mount (linux/magic.h line 39:
    /// `#define FUSE_SUPER_MAGIC 0x65735546`). FUSE flock
    /// reliability depends on whether the userspace server
    /// implements the flock op; the safe default is to reject.
    pub(super) const FUSE: i64 = 0x6573_5546;
}

/// Refuse to operate on filesystems where `flock(2)` is unreliable.
/// Called before every [`try_flock`] open so a misconfigured
/// lockfile path (NFS-mounted `/tmp`, bind-mounted over FUSE) surfaces
/// actionably instead of silently returning an unserialized `OwnedFd`.
///
/// Returns `Ok(())` on accepted filesystems and on statfs failure
/// (non-existent path is an allowed pre-create state — the open
/// below will create it on the parent filesystem's type). Only
/// filesystems whose magic appears in [`fs_magic`]'s deny-list
/// produce an error.
fn reject_remote_fs(path: &Path) -> Result<()> {
    // statfs on the path's PARENT when the path itself does not yet
    // exist — we want to classify the filesystem the lockfile will
    // live on, not error on "path doesn't exist" which is normal for
    // a first-time acquire.
    let target: &Path = if path.exists() {
        path
    } else {
        path.parent().unwrap_or(Path::new("/"))
    };
    let sfs = match rustix::fs::statfs(target) {
        Ok(s) => s,
        // Statfs failure (missing parent, unreadable) is not itself
        // a rejection — defer to the open call to produce a
        // canonical "No such file or directory" error with the right
        // context.
        Err(_) => return Ok(()),
    };
    classify_fs_magic(sfs.f_type as i64).map_err(|rejection| {
        anyhow::anyhow!(
            "{}: filesystem {rejection} Move the lockfile path to a \
             local filesystem (tmpfs, ext4, xfs, btrfs, f2fs, bcachefs).",
            path.display()
        )
    })
}

/// Pure classifier over the [`fs_magic`] deny-list. Returns `Ok(())`
/// when `magic` is an accepted (or unknown-but-not-denied)
/// filesystem, and `Err` with an operator-facing "{name} is not
/// supported for ktstr lockfiles ({reason})." string when the
/// filesystem is on the deny-list.
///
/// Separated from [`reject_remote_fs`] so tests can feed synthetic
/// magic values without a real mount. The caller decorates the
/// error with the lockfile path and the "Move to tmpfs, …" hint;
/// this function produces only the fs-specific middle clause.
pub(crate) fn classify_fs_magic(magic: i64) -> Result<()> {
    let (name, reason) = match magic {
        fs_magic::NFS => (
            "NFS",
            "NFSv3 is advisory-only without an NLM peer; NFSv4 byte-range \
             locking does not cover flock(2)",
        ),
        fs_magic::CIFS | fs_magic::SMB2 => (
            "CIFS/SMB",
            "SMB does not emit /proc/locks entries; ktstr cannot enumerate \
             peer holders",
        ),
        fs_magic::CEPH => (
            "CephFS",
            "Ceph MDS does not participate in flock serialization between \
             ktstr peers on distinct nodes",
        ),
        fs_magic::AFS => ("AFS", "AFS does not support flock(2)"),
        fs_magic::FUSE => (
            "FUSE",
            "flock reliability depends on the userspace server's op \
             implementation",
        ),
        _ => return Ok(()),
    };
    anyhow::bail!("{name} is not supported for ktstr lockfiles ({reason}).")
}

/// Ensure the lockfile exists on disk without acquiring a lock.
/// Used by the DISCOVER phase of `acquire_llc_plan` (see
/// [`crate::vmm::host_topology::discover_llc_snapshots`]): the
/// snapshot pass needs every per-LLC lockfile's inode to exist so a
/// subsequent `/proc/locks` match has a target, but DISCOVER itself
/// must not contend with peer acquires.
///
/// Opens with the same `O_CREAT | O_RDWR | O_CLOEXEC | 0o666` mode
/// as [`try_flock`] so the resulting inode and fd mode match what a
/// first-time acquirer would create. Immediately closes the fd —
/// `OwnedFd::drop` releases the open-file description and (since
/// no flock was ever taken on this fd) cannot release a lock held
/// by a peer fd.
///
/// Runs [`reject_remote_fs`] first so the caller never materializes
/// a lockfile on NFS / CIFS / etc.
pub(crate) fn materialize<P: AsRef<Path>>(path: P) -> Result<()> {
    use rustix::fs::{Mode, OFlags, open};

    let path = path.as_ref();
    reject_remote_fs(path)?;
    let fd = open(
        path,
        OFlags::CREATE | OFlags::RDWR | OFlags::CLOEXEC,
        Mode::from_raw_mode(0o666),
    )
    .map_err(|e| anyhow::anyhow!("materialize lockfile {}: {e}", path.display()))?;
    drop(fd);
    Ok(())
}

/// Open a lock file and attempt `flock` with `LOCK_NB`.
///
/// Creates the file with mode 0o666 if absent. Returns
/// `Ok(Some(fd))` on successful acquire, `Ok(None)` on
/// `EWOULDBLOCK` (peer already holds an incompatible lock), and
/// propagates other errors. The returned fd owns the open-file
/// description; dropping it closes the fd AND releases the kernel
/// flock (the kernel releases `flock(2)` only when the last fd
/// referring to its OFD closes — `OwnedFd::drop` is what makes that
/// work).
///
/// `O_CLOEXEC` is mandatory: a leaked fd across `exec(2)` (cargo
/// subcommand, build-pipeline subprocess, initramfs compressor) would
/// keep the lock alive in the child process after the parent's
/// `OwnedFd::drop` runs, producing phantom holders the next acquirer
/// would blame on the wrong pid.
///
/// Calls [`reject_remote_fs`] before the open to fail-fast on NFS /
/// CIFS / SMB2 / CEPH / AFS / FUSE — see the module-level rationale.
///
/// Accepts any `AsRef<Path>` so `&str`, `&Path`, `&PathBuf`, and
/// `String` callers all work without string-ifying round trips. LLC
/// lockfile paths are built as `String` via `format!` and cache
/// lockfile paths are built as `PathBuf` via `Path::join` — both
/// pass straight through.
pub fn try_flock<P: AsRef<Path>>(path: P, mode: FlockMode) -> Result<Option<OwnedFd>> {
    use rustix::fs::{FlockOperation, Mode, OFlags, flock, open};

    let path = path.as_ref();
    reject_remote_fs(path)?;
    let fd = open(
        path,
        OFlags::CREATE | OFlags::RDWR | OFlags::CLOEXEC,
        Mode::from_raw_mode(0o666),
    )
    .map_err(|e| anyhow::anyhow!("open {}: {e}", path.display()))?;
    let op = match mode {
        FlockMode::Exclusive => FlockOperation::NonBlockingLockExclusive,
        FlockMode::Shared => FlockOperation::NonBlockingLockShared,
    };
    match flock(&fd, op) {
        Ok(()) => Ok(Some(fd)),
        Err(e) if e == rustix::io::Errno::WOULDBLOCK => Ok(None),
        Err(e) => anyhow::bail!("flock {}: {e}", path.display()),
    }
}

/// Parse `/proc/locks` and return [`HolderInfo`] entries for every
/// process holding an advisory `FLOCK` matching `needle`.
///
/// `needle` must be the `{major:02x}:{minor:02x}:{inode}` triple in
/// /proc/locks' own formatting — the two producers are:
///
///  - [`needle_from_path`]: resolves `(major, minor)` via
///    `/proc/self/mountinfo` and `inode` via `stat().st_ino`. Used
///    by path-only callers ([`read_holders`], the `ktstr locks`
///    observational scan, and the EWOULDBLOCK-branch peer-holder
///    lookup in `cache.rs`). `acquire_llc_plan`'s DISCOVER phase
///    uses [`needle_from_path_with_mountinfo`] instead so the
///    mountinfo read amortizes across every LLC in one invocation.
///  - [`needle_from_fd`]: reads `/proc/self/fdinfo/{fd}` and
///    extracts the `lock:` line's triple verbatim. Used by callers
///    that hold a freshly-flocked fd and want to key the needle
///    against the kernel's own formatting without a mountinfo parse.
///    No in-tree caller today — exposed for future telemetry and
///    for completeness: the two producers must agree on all hosts,
///    and a future test can cross-check them.
///
/// Best-effort: returns `Ok(vec![])` when no /proc/locks entry
/// matches the needle, and propagates only the hard `/proc/locks`
/// read failure.
///
/// For each matching PID, reads `/proc/{pid}/cmdline`, decodes as
/// lossy UTF-8, replaces `\0` with ` `, and truncates to
/// [`CMDLINE_MAX_CHARS`] with a `…` suffix on overflow. A cmdline
/// read failure is non-fatal — the entry carries
/// `"<cmdline unavailable>"` so the pid still surfaces.
pub(crate) fn read_holders_for_needle(needle: &str) -> Result<Vec<HolderInfo>> {
    use anyhow::Context;
    use std::fs;

    let contents = fs::read_to_string("/proc/locks")
        .with_context(|| "read /proc/locks for lockfile holder lookup")?;
    Ok(read_holders_from_contents(&contents, needle))
}

/// Content-based seam behind [`read_holders_for_needle`]. Takes
/// already-read `/proc/locks` `contents` plus the match `needle` and
/// returns the [`HolderInfo`] vector. Skips the `/proc/locks` read so
/// a caller with N needles (e.g. `acquire_llc_plan`'s DISCOVER phase,
/// which visits every host LLC's lockfile) can read `/proc/locks`
/// ONCE and call this N times instead of re-reading the same file
/// per iteration — the per-LLC scan was O(N) file reads against a
/// kernel-synthesized text source that is already consistent across
/// the whole batch.
///
/// Thin shell over [`parse_flock_pids_for_needle`]: the latter filters
/// `/proc/locks` lines to the matching FLOCK PIDs; this function adds
/// the per-PID cmdline lookup via [`holder_info_for_pid`] that the
/// [`read_holders_for_needle`] caller expects. Extracted so batched
/// callers and the per-needle wrapper both key against the same seam
/// rather than duplicating the `.into_iter().map()` plumbing.
pub(crate) fn read_holders_from_contents(contents: &str, needle: &str) -> Vec<HolderInfo> {
    let pids = parse_flock_pids_for_needle(contents, needle);
    pids.into_iter().map(holder_info_for_pid).collect()
}

/// Pure parser seam behind [`read_holders_for_needle`]. Takes
/// already-read `/proc/locks` `contents` and the match `needle`, walks
/// every line, and returns the PIDs of processes holding a FLOCK
/// whose `{major:02x}:{minor:02x}:{inode}` triple byte-equals the
/// needle. POSIX-byte-range locks (`POSIX`) and open-file-description
/// locks (`OFDLCK`) are skipped — ktstr coordinates exclusively
/// through `flock(2)`, and misclassifying a POSIX range-lock as a
/// ktstr holder would confuse the holder-enumeration diagnostic.
///
/// Exposed as `pub(crate)` so tests can feed synthetic `/proc/locks`
/// fixtures (POSIX + OFDLCK + FLOCK interleavings, malformed lines,
/// empty input) without touching the real filesystem. The production
/// wrapper above reads `/proc/locks` and calls this seam; everything
/// below is pure text processing.
pub(crate) fn parse_flock_pids_for_needle(contents: &str, needle: &str) -> Vec<u32> {
    let mut pids: Vec<u32> = Vec::new();
    for line in contents.lines() {
        // Expected format (after the id colon):
        //   "1: FLOCK ADVISORY WRITE 12345 08:02:1234 0 EOF"
        // POSIX / OFDLCK lines have the same pid + dev_inode slot
        // shape but a different lock_type keyword in the second
        // field — filter them out here.
        let mut fields = line.split_whitespace();
        // Skip the "N:" id.
        let _id = fields.next();
        let lock_type = fields.next();
        if lock_type != Some("FLOCK") {
            continue;
        }
        // advisory/mandatory
        let _adv = fields.next();
        // READ/WRITE
        let _mode = fields.next();
        let pid = match fields.next().and_then(|s| s.parse::<u32>().ok()) {
            Some(p) => p,
            None => continue,
        };
        let dev_inode = match fields.next() {
            Some(s) => s,
            None => continue,
        };
        if dev_inode == needle && !pids.contains(&pid) {
            pids.push(pid);
        }
    }
    pids
}

/// Path-only adapter over [`read_holders_for_needle`]. Computes the
/// needle via [`needle_from_path`] and forwards. This is the stable
/// entry point for callers that only have a lockfile path — cache
/// EWOULDBLOCK diagnostics and `ktstr locks`.
///
/// `acquire_llc_plan`'s DISCOVER phase does NOT call this adapter —
/// it threads a pre-read `/proc/self/mountinfo` through
/// [`read_holders_with_mountinfo`] so the whole per-LLC walk reads
/// mountinfo exactly once per plan invocation. See
/// [`needle_from_path_with_mountinfo`] for the seam.
///
/// Propagates stat failures on the path (context: "stat lockfile …
/// for holder lookup") and mountinfo failures ("resolve kernel
/// major:minor …").
pub(crate) fn read_holders(path: &Path) -> Result<Vec<HolderInfo>> {
    let needle = needle_from_path(path)?;
    read_holders_for_needle(&needle)
}

/// Variant of [`read_holders`] that accepts pre-read
/// `/proc/self/mountinfo` contents. Used by callers that walk a
/// batch of lockfiles in one invocation (e.g.
/// `acquire_llc_plan`'s DISCOVER phase, which visits every LLC's
/// lockfile) and want to amortize the mountinfo read across the
/// whole batch instead of re-reading per lockfile.
///
/// Semantically identical to [`read_holders`] — the same needle
/// format, the same /proc/locks scan, the same HolderInfo shape —
/// just with the mountinfo text supplied by the caller rather than
/// read inside this function.
pub(crate) fn read_holders_with_mountinfo(path: &Path, mountinfo: &str) -> Result<Vec<HolderInfo>> {
    let needle = needle_from_path_with_mountinfo(path, mountinfo)?;
    read_holders_for_needle(&needle)
}

/// Read `/proc/self/mountinfo` once. Callers that need to derive
/// needles for multiple lockfiles in a single pass (e.g.
/// `acquire_llc_plan`'s DISCOVER phase, which visits every host
/// LLC's lockfile on every DISCOVER attempt) read mountinfo via
/// this helper once per batch and hand the resulting `String` to
/// [`read_holders_with_mountinfo`] / [`needle_from_path_with_mountinfo`].
///
/// One-shot callers ([`needle_from_path`], [`read_holders`]) also
/// route through this helper so every /proc/self/mountinfo read in
/// the crate shares the same error context and any future retry /
/// instrumentation has a single place to land.
pub(crate) fn read_mountinfo() -> Result<String> {
    use anyhow::Context;
    std::fs::read_to_string("/proc/self/mountinfo").context("read /proc/self/mountinfo")
}

/// Build a /proc/locks match needle for `path` using
/// `/proc/self/mountinfo` (for `i_sb->s_dev`) and `stat().st_ino`
/// (for the inode). Format: `{major:02x}:{minor:02x}:{inode}` —
/// kernel's own /proc/locks formatting, so a byte-equality check
/// suffices downstream.
///
/// Refuses to derive a needle from `stat().st_dev`: on btrfs,
/// overlayfs, and bind-mounts that dev diverges from the
/// superblock dev that /proc/locks emits, and a stat-derived
/// needle would silently never match. See module-level rationale.
pub(crate) fn needle_from_path(path: &Path) -> Result<String> {
    let mountinfo = read_mountinfo()?;
    needle_from_path_with_mountinfo(path, &mountinfo)
}

/// Variant of [`needle_from_path`] that accepts pre-read
/// `/proc/self/mountinfo` contents. Both functions produce
/// byte-identical needles for the same `path` — this one just
/// skips the mountinfo read.
///
/// Used by [`read_holders_with_mountinfo`] so a caller walking N
/// lockfiles pays for exactly one mountinfo read instead of N.
pub(crate) fn needle_from_path_with_mountinfo(path: &Path, mountinfo: &str) -> Result<String> {
    use anyhow::Context;
    use std::fs;
    use std::os::unix::fs::MetadataExt;

    let meta = fs::metadata(path)
        .with_context(|| format!("stat lockfile {} for holder lookup", path.display()))?;
    let inode = meta.ino();
    let (major, minor) =
        mount_major_minor_for_path_with_contents(path, mountinfo).with_context(|| {
            format!(
                "resolve kernel major:minor for {} via /proc/self/mountinfo",
                path.display()
            )
        })?;
    Ok(format!("{major:02x}:{minor:02x}:{inode}"))
}

/// Build a /proc/locks match needle from `/proc/self/fdinfo/{fd}`.
/// Reads the fd's fdinfo, parses the `lock:` line (one per lock
/// held on this OFD), and extracts the first `FLOCK` line's
/// `{major:02x}:{minor:02x}:{inode}` triple verbatim.
///
/// Intended for callers that hold a freshly-flocked [`OwnedFd`] and
/// want the needle without a mountinfo parse. The resulting needle
/// must byte-equal the one [`needle_from_path`] produces for the
/// same lockfile — both derive from the same kernel state, by
/// different paths.
///
/// Returns `None` when fdinfo has no `lock:` line (fd opened but
/// not flocked; observational callers that just want the inode
/// triple cannot use this path — they must go through
/// [`needle_from_path`]).
///
/// No in-tree caller today; exposed for future use and cross-check
/// testing. See module-level rationale.
#[allow(dead_code)]
pub(crate) fn needle_from_fd<F: std::os::fd::AsRawFd>(fd: &F) -> Result<Option<String>> {
    use anyhow::Context;
    use std::fs;

    let raw = fd.as_raw_fd();
    let path = format!("/proc/self/fdinfo/{raw}");
    let contents = fs::read_to_string(&path)
        .with_context(|| format!("read {path} for fd-needle derivation"))?;

    // fdinfo format for flock-holding fds:
    //   lock:\t1: FLOCK ADVISORY WRITE 12345 08:02:1234 0 EOF
    // The "1:" id is arbitrary (there can be multiple lock: lines).
    // We take the first FLOCK line's dev:inode triple.
    for line in contents.lines() {
        let rest = match line.strip_prefix("lock:") {
            Some(s) => s.trim_start(),
            None => continue,
        };
        let mut fields = rest.split_whitespace();
        let _id = fields.next(); // "N:"
        let lock_type = fields.next();
        if lock_type != Some("FLOCK") {
            continue;
        }
        let _adv = fields.next(); // ADVISORY/MANDATORY
        let _mode = fields.next(); // READ/WRITE
        let _pid = fields.next(); // our own pid
        if let Some(triple) = fields.next() {
            return Ok(Some(triple.to_string()));
        }
    }
    Ok(None)
}

/// Resolve `path` to its containing mount point's kernel major:minor
/// from pre-read `/proc/self/mountinfo` text.
///
/// Format per `Documentation/filesystems/proc.rst` §3.5:
/// ```
/// {mount_id} {parent_id} {major:minor} {root} {mount_point} {options} ...
/// ```
/// We canonicalize `path` (fall back to lexical absolute form when
/// canonicalize fails — the lockfile may not yet exist), enumerate
/// every mountinfo line, find the longest-prefix match on the
/// `mount_point` field, and return that entry's `{major:minor}`
/// decoded as `(u32, u32)`.
///
/// Longest-prefix is load-bearing: a bind mount of `/tmp/ktstr-cache`
/// onto `/tmp/ktstr-cache` stacked over tmpfs `/tmp` must match the
/// bind's mountinfo entry (more specific), not tmpfs's (less
/// specific). The lockfile lives on the bind-backing filesystem;
/// /proc/locks emits the bind's s_dev.
///
/// Production callers obtain `contents` from [`read_mountinfo`] —
/// either once per batch (acquire_llc_plan DISCOVER) or per-call
/// (one-shot needle derivation via [`needle_from_path`]).
fn mount_major_minor_for_path_with_contents(path: &Path, contents: &str) -> Result<(u32, u32)> {
    use std::fs;

    // Canonicalize the query path. When the lockfile doesn't yet
    // exist (first-call create path), canonicalize fails; fall back
    // to the caller's path verbatim, which is already absolute for
    // every ktstr call site (`/tmp/ktstr-llc-{N}.lock`,
    // `{cache_root}/.locks/{key}.lock`, …).
    let canon: PathBuf = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());

    mount_major_minor_for_path_from_contents(contents, &canon)
}

/// Pure mountinfo-parser seam behind
/// [`mount_major_minor_for_path_with_contents`].
/// Takes the already-read mountinfo `contents` and the
/// already-canonicalized `path`, walks the lines, longest-prefix-
/// matches `mount_point` against `path`, and returns the matching
/// entry's `(major, minor)` decoded from the `{major:minor}` field.
///
/// Exposed as `pub(crate)` so tests can feed synthetic mountinfo
/// text (bind mounts stacked over tmpfs, btrfs subvolume mounts,
/// mount points with whitespace) without having to reproduce those
/// states in the host filesystem. The production wrapper above
/// canonicalizes before calling this seam; everything below is pure
/// text processing.
pub(crate) fn mount_major_minor_for_path_from_contents(
    contents: &str,
    path: &Path,
) -> Result<(u32, u32)> {
    let mut best: Option<(usize, u32, u32)> = None;
    for line in contents.lines() {
        // Split on whitespace once, walk fields by index. Field 2 is
        // `major:minor`, field 4 is the mount point. A single pass
        // collects both without re-splitting. Whitespace inside a
        // mount_point is safe to whitespace-split on: the kernel
        // octal-escapes space/tab/newline in the mount_point field
        // (fs/proc_namespace.c: seq_path_root(..., " \t\n\\") →
        // fs/seq_file.c: mangle_path()), so a literal space never
        // appears inline — it arrives as the 4-byte sequence `\040`,
        // which splitting preserves. [`unescape_mountinfo_field`]
        // restores the original bytes below before the prefix match.
        let mut fields = line.split_whitespace();
        let _mount_id = fields.next();
        let _parent_id = fields.next();
        let major_minor = match fields.next() {
            Some(s) => s,
            None => continue,
        };
        let _root = fields.next();
        let mount_point_raw = match fields.next() {
            Some(s) => s,
            None => continue,
        };
        // Optional fields, then `-`, then fs_type — we don't consume
        // them; `fields` is discarded after this line.

        // Kernel escapes space (`\040`), tab (`\011`), newline
        // (`\012`), and backslash (`\134`) in the mount_point field
        // via fs/seq_file.c: mangle_path(). `path` arrives from the
        // caller with literal bytes (a tempdir named "my dir" has a
        // real space, not `\040`), so we must octal-unescape the
        // mountinfo field before the prefix check or a path with any
        // of those bytes would silently miss its covering mount —
        // producing "no mountinfo entry covers {path}" on otherwise
        // valid hosts that happened to place `/tmp` or a cache root
        // under a mount point containing whitespace.
        let mount_point = unescape_mountinfo_field(mount_point_raw);

        // Prefix match: `mount_point` must be a prefix of `path`
        // on a path-component boundary. A pure string prefix check
        // would accept `/tmp/foo` against `/tmp/foobar`, so anchor
        // the comparison on components.
        if !path_starts_with(path, Path::new(mount_point.as_ref())) {
            continue;
        }
        let (major, minor) = match parse_major_minor(major_minor) {
            Some(mm) => mm,
            None => continue,
        };
        let len = mount_point.len();
        if best.is_none_or(|(best_len, _, _)| len > best_len) {
            best = Some((len, major, minor));
        }
    }
    match best {
        Some((_, major, minor)) => Ok((major, minor)),
        None => anyhow::bail!(
            "no mountinfo entry covers {} — is /proc mounted?",
            path.display()
        ),
    }
}

/// Decode the kernel's `\NNN` octal escape sequences in a mountinfo
/// text field back to the original bytes. The kernel's mountinfo
/// writer (`fs/proc_namespace.c:show_mountinfo`) passes the escape
/// set `" \t\n\\"` to `fs/seq_file.c:seq_path_root`, which then calls
/// `mangle_path()` with 3-digit-octal for each matched character:
///
/// - space (0x20) → `\040`
/// - tab   (0x09) → `\011`
/// - LF    (0x0A) → `\012`
/// - `\`   (0x5C) → `\134`
///
/// Bytes outside that set are copied verbatim. This decoder handles
/// the general form `\NNN` (3 octal digits), not just the 4
/// characters above — the kernel's escape logic is parameterized,
/// and a future kernel could extend the escape set without changing
/// the wire format; a generic decoder matches whatever the kernel
/// emits. Non-`\NNN` backslashes (none in practice, but defensive)
/// are kept as literal bytes, so malformed input cannot produce a
/// shorter string that silently matches a different mount point
/// than the caller intended.
///
/// Returns `Cow::Borrowed(raw)` when no `\` appears — avoids an
/// allocation for the overwhelmingly common "no escape needed" case
/// (`/tmp`, `/home`, `/var`, …).
fn unescape_mountinfo_field(raw: &str) -> std::borrow::Cow<'_, str> {
    if !raw.contains('\\') {
        return std::borrow::Cow::Borrowed(raw);
    }
    // `\` was found — switch to the owned path. Byte-level walk so
    // `\NNN` with non-ASCII octal-decoded bytes (e.g. `\200`)
    // produces the exact kernel-emitted byte sequence; pushing via
    // `push_str` would require UTF-8 validation the kernel does not
    // itself apply to path components.
    let bytes = raw.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\'
            && i + 3 < bytes.len()
            && is_octal_digit(bytes[i + 1])
            && is_octal_digit(bytes[i + 2])
            && is_octal_digit(bytes[i + 3])
        {
            let b =
                ((bytes[i + 1] - b'0') << 6) | ((bytes[i + 2] - b'0') << 3) | (bytes[i + 3] - b'0');
            out.push(b);
            i += 4;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    // The kernel's mangle_path never produces invalid UTF-8 for the
    // escaped set (`" \t\n\\"` are all ASCII). Lossy decode matches
    // the same contract [`holder_info_for_pid`] applies to argv
    // bytes: a malformed mountinfo line produces U+FFFD
    // substitutions rather than aborting the whole parse.
    std::borrow::Cow::Owned(String::from_utf8_lossy(&out).into_owned())
}

/// True when `b` is one of `b'0'..=b'7'` — the valid digits of a
/// `\NNN` octal escape. Inlined so the hot mountinfo parse loop
/// stays branch-light; the intent is obvious enough that a named
/// helper documents itself.
#[inline]
fn is_octal_digit(b: u8) -> bool {
    (b'0'..=b'7').contains(&b)
}

/// True when `path` begins with `prefix` on a path-component
/// boundary. Distinct from byte-level `String::starts_with`:
/// `/tmp/foo` does NOT start with `/tmp/foobar`. `Path::starts_with`
/// already handles this correctly; wrap it for readability at the
/// mountinfo call site.
fn path_starts_with(path: &Path, prefix: &Path) -> bool {
    path.starts_with(prefix)
}

/// Parse a mountinfo `major:minor` field (e.g. `"259:3"`) into a
/// `(u32, u32)` tuple. Decimal — the kernel emits these in base 10,
/// unlike /proc/locks which uses hex for the same pair.
fn parse_major_minor(s: &str) -> Option<(u32, u32)> {
    let (maj, min) = s.split_once(':')?;
    Some((maj.parse().ok()?, min.parse().ok()?))
}

/// Read and shape `/proc/{pid}/cmdline` for a [`HolderInfo`].
/// `\0` → ` `, lossy UTF-8, truncated to [`CMDLINE_MAX_CHARS`] with
/// `…` suffix on overflow. Missing / racing / permission-denied on
/// `/proc/{pid}/cmdline` produces `"<cmdline unavailable>"` — the
/// pid still carries diagnostic value even without the command.
fn holder_info_for_pid(pid: u32) -> HolderInfo {
    let raw = match std::fs::read(format!("/proc/{pid}/cmdline")) {
        Ok(bytes) => bytes,
        Err(_) => {
            return HolderInfo {
                pid,
                cmdline: "<cmdline unavailable>".to_string(),
            };
        }
    };
    // Kernel writes argv joined with \0 and terminated by \0. Lossy
    // decode handles non-UTF-8 argv bytes (rare — most binaries use
    // UTF-8 args, but the kernel does not enforce it).
    let text: String = String::from_utf8_lossy(&raw)
        .chars()
        .map(|c| if c == '\0' { ' ' } else { c })
        .collect::<String>()
        .trim_end()
        .to_string();
    let truncated = if text.chars().count() > CMDLINE_MAX_CHARS {
        let head: String = text.chars().take(CMDLINE_MAX_CHARS).collect();
        format!("{head}…")
    } else if text.is_empty() {
        "<cmdline unavailable>".to_string()
    } else {
        text
    };
    HolderInfo {
        pid,
        cmdline: truncated,
    }
}

/// Format a [`HolderInfo`] slice for inclusion in user-facing error
/// strings. Empty slice yields [`NO_HOLDERS_RECORDED`] so the
/// diagnostic is unambiguous — a stale lockfile whose holder has
/// exited presents as empty, and the error should say so rather than
/// print a misleading blank. Non-empty renders one
/// `pid={pid} cmd={cmdline}` line per holder, newline-separated and
/// indented two spaces, so a multi-holder error stays readable when
/// embedded in a wrapping anyhow chain; the prior comma-joined form
/// ran every holder into a single wide line that terminals wrapped
/// arbitrarily mid-cmdline.
pub fn format_holder_list(holders: &[HolderInfo]) -> String {
    if holders.is_empty() {
        NO_HOLDERS_RECORDED.to_string()
    } else {
        holders
            .iter()
            .map(|h| format!("  pid={} cmd={}", h.pid, h.cmdline))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // fs_magic constant pins — regression guards
    // ---------------------------------------------------------------
    //
    // Both constants were originally typed with one-digit errors
    // that silently missed real FUSE / AFS mounts. These tests pin
    // the corrected values against the kernel's own definitions in
    // `linux/magic.h` so a future "clean-up" that reverts to the
    // legacy/typo variant fails the build.

    /// `FUSE_SUPER_MAGIC` per `linux/magic.h` line 39. A prior
    /// typo used `0x65737546` (wrong digit at position 4) so real
    /// FUSE mounts never matched the deny-list. Regression guard.
    #[test]
    fn fuse_magic_matches_linux_magic_h() {
        assert_eq!(fs_magic::FUSE, 0x65735546);
    }

    /// `AFS_FS_MAGIC` per `linux/magic.h` line 56 — the in-tree
    /// kAFS client's superblock magic (`fs/afs/super.c:460`). A
    /// prior revision used the legacy `AFS_SUPER_MAGIC =
    /// 0x5346414F` which no in-tree driver emits today, so real
    /// AFS mounts never matched. Regression guard.
    #[test]
    fn afs_magic_matches_in_tree_kafs() {
        assert_eq!(fs_magic::AFS, 0x6B414653);
    }

    // ---------------------------------------------------------------
    // classify_fs_magic — deny-list coverage
    // ---------------------------------------------------------------

    /// Every deny-listed magic produces an error naming the
    /// filesystem. The user-facing error string must include the
    /// fs name so operators grepping "NFS is not supported" find
    /// the right diagnostic.
    #[test]
    fn classify_fs_magic_rejects_nfs() {
        let err = classify_fs_magic(fs_magic::NFS).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("NFS"), "err={msg}");
        // Reason substring pins the NFSv3 advisory-only rationale
        // so a future refactor that drops the actionable "why" text
        // regresses through this test, not just the name check.
        assert!(
            msg.contains("NFSv3"),
            "err must name NFSv3 in reason: {msg}"
        );
        assert!(
            msg.contains("is not supported"),
            "err must contain the canonical rejection phrase: {msg}",
        );
    }

    #[test]
    fn classify_fs_magic_rejects_cifs_and_smb2() {
        let err_cifs = classify_fs_magic(fs_magic::CIFS).unwrap_err();
        let err_smb2 = classify_fs_magic(fs_magic::SMB2).unwrap_err();
        // Both classify as CIFS/SMB — same arm. Reason pins the
        // "/proc/locks entries" rationale from classify_fs_magic.
        let cifs_msg = format!("{err_cifs:#}");
        let smb2_msg = format!("{err_smb2:#}");
        assert!(cifs_msg.contains("CIFS/SMB"));
        assert!(smb2_msg.contains("CIFS/SMB"));
        assert!(
            cifs_msg.contains("/proc/locks"),
            "err must cite /proc/locks: {cifs_msg}",
        );
        assert!(
            smb2_msg.contains("/proc/locks"),
            "err must cite /proc/locks: {smb2_msg}",
        );
        assert!(
            cifs_msg.contains("is not supported"),
            "err must contain the canonical rejection phrase: {cifs_msg}",
        );
        assert!(
            smb2_msg.contains("is not supported"),
            "err must contain the canonical rejection phrase: {smb2_msg}",
        );
    }

    #[test]
    fn classify_fs_magic_rejects_ceph() {
        let err = classify_fs_magic(fs_magic::CEPH).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("CephFS"));
        // Reason pins the MDS-doesn't-serialize rationale.
        assert!(msg.contains("MDS"), "err must name Ceph MDS: {msg}");
        assert!(
            msg.contains("is not supported"),
            "err must contain the canonical rejection phrase: {msg}",
        );
    }

    #[test]
    fn classify_fs_magic_rejects_afs() {
        let err = classify_fs_magic(fs_magic::AFS).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("AFS"));
        // Reason pins the "AFS does not support flock(2)" substring.
        assert!(msg.contains("flock(2)"), "err must cite flock(2): {msg}");
        assert!(
            msg.contains("is not supported"),
            "err must contain the canonical rejection phrase: {msg}",
        );
    }

    #[test]
    fn classify_fs_magic_rejects_fuse() {
        let err = classify_fs_magic(fs_magic::FUSE).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("FUSE"));
        // Reason pins the userspace-server rationale.
        assert!(
            msg.contains("userspace server"),
            "err must name userspace server: {msg}",
        );
        assert!(
            msg.contains("is not supported"),
            "err must contain the canonical rejection phrase: {msg}",
        );
    }

    /// Accepted local filesystems pass through `Ok`. Values from
    /// `linux/magic.h`: TMPFS 0x01021994, EXT4 0xEF53 (also
    /// EXT2/3), XFS 0x58465342, BTRFS 0x9123683E, F2FS 0xF2F52010,
    /// BCACHEFS 0xCA451A4E. Deny-list semantics: unknown local
    /// filesystems (zfs, erofs, …) also pass.
    #[test]
    fn classify_fs_magic_accepts_local_filesystems() {
        classify_fs_magic(0x01021994).expect("tmpfs accepted");
        classify_fs_magic(0xEF53).expect("ext4 accepted");
        classify_fs_magic(0x58465342).expect("xfs accepted");
        classify_fs_magic(0x9123683E).expect("btrfs accepted");
        classify_fs_magic(0xF2F52010).expect("f2fs accepted");
        classify_fs_magic(0xCA451A4E).expect("bcachefs accepted");
        // Unknown magic — deny-list semantics mean "not on the
        // reject list" → Ok.
        classify_fs_magic(0xDEAD_BEEF).expect("unknown magic accepted");
    }

    // ---------------------------------------------------------------
    // mountinfo parsing — mount_major_minor_for_path_from_contents
    // ---------------------------------------------------------------

    /// Single-mount synthetic mountinfo: path on tmpfs-mounted
    /// `/tmp` resolves to the tmpfs entry's (major, minor).
    /// Values from `man 5 proc` mountinfo format example.
    #[test]
    fn mountinfo_single_mount_hits_right_major_minor() {
        let mountinfo = "\
22 28 0:21 / /tmp rw,nosuid,nodev shared:5 - tmpfs tmpfs rw,size=8g
";
        let (major, minor) =
            mount_major_minor_for_path_from_contents(mountinfo, Path::new("/tmp/ktstr-llc-0.lock"))
                .expect("tmp mount covers the lockfile path");
        assert_eq!((major, minor), (0, 21));
    }

    /// Longest-prefix wins: a bind mount at `/tmp/ktstr-cache`
    /// stacked over tmpfs `/tmp` must resolve to the BIND's
    /// major:minor, not tmpfs's. `/proc/locks` emits the bind's
    /// `s_dev`, so the lookup must match the more-specific mount.
    #[test]
    fn mountinfo_longest_prefix_wins_for_bind_over_tmpfs() {
        let mountinfo = "\
22 28 0:21 / /tmp rw,nosuid,nodev shared:5 - tmpfs tmpfs rw,size=8g
35 22 0:99 / /tmp/ktstr-cache rw,nosuid - tmpfs tmpfs rw,size=1g
";
        let (major, minor) = mount_major_minor_for_path_from_contents(
            mountinfo,
            Path::new("/tmp/ktstr-cache/entry.lock"),
        )
        .expect("bind mount wins longest-prefix match");
        assert_eq!((major, minor), (0, 99), "bind's major:minor expected");
    }

    /// A path not covered by any mount errors out with an
    /// actionable message. The production wrapper always pre-
    /// populates `/proc/self/mountinfo`, which always contains at
    /// minimum the root `/` entry, so this failure is effectively
    /// "path has no leading slash" territory.
    #[test]
    fn mountinfo_uncovered_path_errors() {
        let mountinfo = "\
22 28 0:21 / /tmp rw - tmpfs tmpfs rw
";
        let err = mount_major_minor_for_path_from_contents(
            mountinfo,
            Path::new("/var/log/unrelated.lock"),
        )
        .expect_err("no mountinfo entry covers /var/log/...");
        let msg = format!("{err:#}");
        assert!(msg.contains("no mountinfo entry covers"), "msg={msg}");
    }

    /// Component-boundary prefix check: a path at `/tmp/foo` must
    /// NOT match a mount at `/tmp/foobar`. Byte-level string
    /// prefix would incorrectly accept this; `Path::starts_with`
    /// anchors on components and rejects it. This is the correctness
    /// test for `path_starts_with`.
    ///
    /// Covers both directions on the same mountinfo fixture:
    ///   - `/tmp/foo/entry.lock`      → (0, 21) — the /tmp mount
    ///     (NOT /tmp/foobar, despite string-prefix overlap).
    ///   - `/tmp/foobar/entry.lock`   → (0, 99) — the /tmp/foobar
    ///     mount wins longest-prefix because it IS a component-
    ///     boundary prefix of the query path.
    #[test]
    fn mountinfo_respects_component_boundary() {
        let mountinfo = "\
22 28 0:21 / /tmp rw - tmpfs tmpfs rw
35 22 0:99 / /tmp/foobar rw - tmpfs tmpfs rw
";
        // /tmp/foo/ is NOT under /tmp/foobar/ on a component
        // boundary — only /tmp matches.
        let (major, minor) =
            mount_major_minor_for_path_from_contents(mountinfo, Path::new("/tmp/foo/entry.lock"))
                .expect("path under /tmp (not /tmp/foobar) resolves to the tmp mount");
        assert_eq!(
            (major, minor),
            (0, 21),
            "/tmp/foo must NOT match the /tmp/foobar mount",
        );

        // Reverse: /tmp/foobar/ IS under /tmp/foobar on a component
        // boundary — the more-specific mount wins longest-prefix.
        let (major, minor) = mount_major_minor_for_path_from_contents(
            mountinfo,
            Path::new("/tmp/foobar/entry.lock"),
        )
        .expect("path under /tmp/foobar resolves to the /tmp/foobar mount");
        assert_eq!(
            (major, minor),
            (0, 99),
            "/tmp/foobar/ must match the /tmp/foobar mount, not the /tmp one",
        );
    }

    /// Malformed major:minor on one line doesn't prevent a later
    /// valid line from matching. Graceful degradation — a corrupt
    /// mountinfo line (unlikely but possible on exotic hosts)
    /// must not kill the whole lookup.
    #[test]
    fn mountinfo_skips_malformed_major_minor() {
        let mountinfo = "\
22 28 BAD:NUMBER / /tmp rw - tmpfs tmpfs rw
35 28 0:42 / /tmp rw - tmpfs tmpfs rw
";
        let (major, minor) =
            mount_major_minor_for_path_from_contents(mountinfo, Path::new("/tmp/entry.lock"))
                .expect("second (valid) line still matches after malformed first");
        assert_eq!((major, minor), (0, 42));
    }

    /// Short line (missing fields) is skipped without error.
    /// Real mountinfo always has ≥5 fields before the `-`, but
    /// defensive coding protects against proc-fs corruption.
    #[test]
    fn mountinfo_skips_truncated_lines() {
        let mountinfo = "\
22 28 0:21
35 28 0:42 / /tmp rw - tmpfs tmpfs rw
";
        let (major, minor) =
            mount_major_minor_for_path_from_contents(mountinfo, Path::new("/tmp/entry.lock"))
                .expect("truncated line skipped; second line matches");
        assert_eq!((major, minor), (0, 42));
    }

    /// Mount point containing a literal space. The kernel emits it
    /// as `\040` in the mountinfo `mount_point` field via
    /// `fs/seq_file.c:mangle_path`; without unescaping, a path like
    /// `/mnt/my dir/cache.lock` would byte-split into `/mnt/my` and
    /// `dir/cache.lock`, the parser would see the mount_point as
    /// `/mnt/my\040dir`, and `path_starts_with` would compare
    /// against the escaped form — a silent miss on any host whose
    /// cache root or `/tmp` happens to sit under a whitespace mount.
    /// Pins the fix: the parser unescapes before comparing.
    #[test]
    fn mountinfo_unescapes_space_in_mount_point() {
        let mountinfo = "\
22 28 0:77 / /mnt/my\\040dir rw,nosuid - tmpfs tmpfs rw
";
        let (major, minor) = mount_major_minor_for_path_from_contents(
            mountinfo,
            Path::new("/mnt/my dir/cache.lock"),
        )
        .expect(
            "mount point with `\\040`-escaped space must unescape to real \
             space and match the query path's literal space",
        );
        assert_eq!((major, minor), (0, 77));
    }

    /// Mount point containing a literal tab (`\011`) — same fix
    /// surface as `\040`, different escape byte. Tabs in mount
    /// points are vanishingly rare but the kernel escapes them
    /// alongside spaces; testing the class broadly pins the
    /// general-octal contract rather than just the space-specific
    /// one.
    #[test]
    fn mountinfo_unescapes_tab_in_mount_point() {
        let mountinfo = "\
22 28 0:78 / /mnt/tab\\011dir rw,nosuid - tmpfs tmpfs rw
";
        let (major, minor) = mount_major_minor_for_path_from_contents(
            mountinfo,
            Path::new("/mnt/tab\tdir/cache.lock"),
        )
        .expect("mount point with `\\011` must unescape to real tab");
        assert_eq!((major, minor), (0, 78));
    }

    /// Mount point containing a literal backslash — escaped as
    /// `\134` per the kernel's `mangle_path(..., " \\t\\n\\\\")`
    /// escape set. A caller's path bytes include the literal
    /// backslash; the parser must match after unescaping.
    #[test]
    fn mountinfo_unescapes_backslash_in_mount_point() {
        // Rust source: `\\134` → the four bytes `\`, `1`, `3`, `4`
        // (the on-wire octal escape). The query path holds a
        // literal backslash (Rust source: `\\` → `\`).
        let mountinfo = "\
22 28 0:79 / /mnt/bs\\134dir rw,nosuid - tmpfs tmpfs rw
";
        let (major, minor) = mount_major_minor_for_path_from_contents(
            mountinfo,
            Path::new("/mnt/bs\\dir/cache.lock"),
        )
        .expect("mount point with `\\134` must unescape to real backslash");
        assert_eq!((major, minor), (0, 79));
    }

    /// `unescape_mountinfo_field` returns a borrowed `Cow` when
    /// the input contains no `\` — the common case on every Linux
    /// host. Pins the zero-allocation contract so a future refactor
    /// that always allocates regresses through this test.
    #[test]
    fn unescape_mountinfo_field_borrows_when_no_escapes() {
        let raw = "/tmp";
        let decoded = unescape_mountinfo_field(raw);
        match decoded {
            std::borrow::Cow::Borrowed(b) => assert_eq!(b, raw),
            std::borrow::Cow::Owned(_) => {
                panic!("unescape must return Cow::Borrowed when input has no `\\`")
            }
        }
    }

    /// `unescape_mountinfo_field` decodes multi-escape inputs —
    /// `/a b\tc` encodes as `/a\040b\011c`, and all three bytes
    /// must decode simultaneously in one pass. Pins the loop
    /// correctness against a future refactor that only handles
    /// a single escape per field.
    #[test]
    fn unescape_mountinfo_field_handles_multiple_escapes() {
        let raw = "/a\\040b\\011c";
        let decoded = unescape_mountinfo_field(raw);
        assert_eq!(decoded.as_ref(), "/a b\tc");
    }

    /// Non-`\NNN` backslash — defensive: the kernel never emits
    /// this form, but if corrupt /proc/self/mountinfo has a bare
    /// `\` or a partial escape (e.g. `\4` with < 3 following
    /// digits), we must keep the byte literal rather than advance
    /// past it. Silently consuming would produce a shorter
    /// mount_point that could match a different mount than the
    /// caller intended.
    #[test]
    fn unescape_mountinfo_field_preserves_non_octal_backslash() {
        // `\9` — 9 is not an octal digit.
        let raw = "/bad\\9suffix";
        let decoded = unescape_mountinfo_field(raw);
        assert_eq!(decoded.as_ref(), "/bad\\9suffix");

        // Trailing `\` with < 3 bytes after — defensive.
        let raw = "/trunc\\04";
        let decoded = unescape_mountinfo_field(raw);
        assert_eq!(decoded.as_ref(), "/trunc\\04");
    }

    /// `is_octal_digit` accepts exactly the 8 valid digits of a
    /// `\NNN` escape and rejects everything else. Pins the
    /// boundary so a future refactor that uses
    /// `char::is_ascii_digit` (which accepts 8 and 9) regresses
    /// through this test — those two bytes would silently admit
    /// corrupt input as valid octal and produce the wrong
    /// decoded byte.
    #[test]
    fn is_octal_digit_rejects_8_and_9() {
        for b in b'0'..=b'7' {
            assert!(is_octal_digit(b), "byte 0x{b:02x} must be octal");
        }
        assert!(!is_octal_digit(b'8'), "byte 0x38 must NOT be octal");
        assert!(!is_octal_digit(b'9'), "byte 0x39 must NOT be octal");
        assert!(!is_octal_digit(b'a'), "non-digit must NOT be octal");
        assert!(!is_octal_digit(b'/'), "byte before '0' must NOT be octal");
    }

    // ---------------------------------------------------------------
    // path_starts_with + parse_major_minor — helper primitives
    // ---------------------------------------------------------------

    /// Component-boundary semantics are the correctness property of
    /// `path_starts_with`. Pins the wrapper's contract against a
    /// future refactor that inlines the call but forgets the
    /// `Path::starts_with` semantics.
    #[test]
    fn path_starts_with_respects_component_boundary() {
        assert!(
            path_starts_with(Path::new("/tmp/foo"), Path::new("/tmp")),
            "/tmp/foo must start with /tmp",
        );
        assert!(
            path_starts_with(Path::new("/tmp/foo/bar"), Path::new("/tmp/foo")),
            "/tmp/foo/bar must start with /tmp/foo (deeper component path)",
        );
        assert!(
            !path_starts_with(Path::new("/tmp/foobar"), Path::new("/tmp/foo")),
            "/tmp/foobar must NOT start with /tmp/foo (component boundary)",
        );
        assert!(
            path_starts_with(Path::new("/tmp"), Path::new("/tmp")),
            "/tmp must start with itself (identity)",
        );
        assert!(
            !path_starts_with(Path::new("/"), Path::new("/tmp")),
            "/ is a parent of /tmp, not a child — must NOT match",
        );
    }

    /// `parse_major_minor` happy path — kernel's decimal
    /// `{major}:{minor}` (NOT the hex form /proc/locks emits).
    #[test]
    fn parse_major_minor_happy_path() {
        assert_eq!(parse_major_minor("0:21"), Some((0, 21)));
        assert_eq!(parse_major_minor("259:3"), Some((259, 3)));
    }

    /// Missing colon — invalid format.
    #[test]
    fn parse_major_minor_missing_colon() {
        assert_eq!(parse_major_minor("notvalid"), None);
        assert_eq!(parse_major_minor(""), None);
    }

    /// Non-numeric major or minor.
    #[test]
    fn parse_major_minor_non_numeric() {
        assert_eq!(parse_major_minor("abc:21"), None);
        assert_eq!(parse_major_minor("0:xyz"), None);
        assert_eq!(parse_major_minor(":"), None);
    }

    /// Negative integers. `parse_major_minor` uses `parse::<u32>()`,
    /// which rejects the leading `-`. Pins the unsigned contract —
    /// the kernel never emits negative major:minor, but a corrupt
    /// /proc/self/mountinfo or hand-crafted synthetic must not be
    /// accepted silently (would otherwise miscompare /proc/locks).
    #[test]
    fn parse_major_minor_negative_numbers() {
        assert_eq!(parse_major_minor("-1:0"), None);
        assert_eq!(parse_major_minor("0:-1"), None);
    }

    // ---------------------------------------------------------------
    // format_holder_list — rendering contract
    // ---------------------------------------------------------------

    /// Empty slice yields the sentinel `NO_HOLDERS_RECORDED` so
    /// log-scrapers have a stable key.
    #[test]
    fn format_holder_list_empty_yields_sentinel() {
        assert_eq!(format_holder_list(&[]), NO_HOLDERS_RECORDED);
    }

    /// Single holder renders with the `  pid={pid} cmd={cmdline}`
    /// shape. Two-space indent is load-bearing — a future revert
    /// to comma-join would break terminal rendering on
    /// multi-holder lockfiles.
    #[test]
    fn format_holder_list_single_holder() {
        let holders = [HolderInfo {
            pid: 12345,
            cmdline: "cargo build".to_string(),
        }];
        assert_eq!(format_holder_list(&holders), "  pid=12345 cmd=cargo build");
    }

    /// Multiple holders newline-separated (not comma-joined). The
    /// previous shape was `", "` — this test pins the newline.
    #[test]
    fn format_holder_list_multiple_newline_separated() {
        let holders = [
            HolderInfo {
                pid: 1,
                cmdline: "a".to_string(),
            },
            HolderInfo {
                pid: 2,
                cmdline: "b".to_string(),
            },
        ];
        let out = format_holder_list(&holders);
        assert!(out.contains("\n"), "must contain newline: {out}");
        assert!(!out.contains(", "), "must NOT contain comma-space: {out}");
        assert_eq!(out, "  pid=1 cmd=a\n  pid=2 cmd=b");
    }

    // ---------------------------------------------------------------
    // read_holders_for_needle + needle_from_fd — light /proc-touching
    // ---------------------------------------------------------------

    /// `parse_flock_pids_for_needle` skips `POSIX` and `OFDLCK`
    /// lines and matches only `FLOCK` lines whose dev:inode triple
    /// byte-equals the needle.
    ///
    /// Feeds a synthetic `/proc/locks` fixture containing one POSIX,
    /// one OFDLCK, and one FLOCK line — all with the same dev:inode
    /// triple — and asserts only the FLOCK PID is returned. This
    /// pins the lock_type filter at the second-field check: without
    /// it, POSIX-byte-range locks would be misclassified as ktstr
    /// flock holders and the holder-enumeration diagnostic would
    /// name the wrong peers.
    #[test]
    fn parse_flock_pids_for_needle_skips_posix_and_ofdlck() {
        let needle = "08:02:1234";
        let contents = "\
1: POSIX  ADVISORY  WRITE 11111 08:02:1234 0 EOF
2: OFDLCK ADVISORY  READ  22222 08:02:1234 0 EOF
3: FLOCK  ADVISORY  WRITE 33333 08:02:1234 0 EOF
4: FLOCK  ADVISORY  READ  44444 08:02:5678 0 EOF
";
        let pids = parse_flock_pids_for_needle(contents, needle);
        assert_eq!(
            pids,
            vec![33333],
            "only the FLOCK line at the matching triple must contribute a PID; \
             POSIX/OFDLCK must be filtered",
        );
    }

    /// `parse_flock_pids_for_needle` deduplicates PIDs when a single
    /// process holds multiple FLOCK entries on the same lockfile
    /// (e.g. the kernel emits one `lock:` line per OFD, and a
    /// process that dup'd its fd has multiple OFDs on the same
    /// inode). One PID per holder, regardless of how many entries.
    #[test]
    fn parse_flock_pids_for_needle_deduplicates_pids() {
        let needle = "08:02:1234";
        let contents = "\
1: FLOCK  ADVISORY  WRITE 55555 08:02:1234 0 EOF
2: FLOCK  ADVISORY  READ  55555 08:02:1234 0 EOF
3: FLOCK  ADVISORY  WRITE 66666 08:02:1234 0 EOF
";
        let pids = parse_flock_pids_for_needle(contents, needle);
        assert_eq!(pids, vec![55555, 66666], "PIDs must dedupe");
    }

    /// `parse_flock_pids_for_needle` with empty contents returns an
    /// empty Vec — degenerate case.
    #[test]
    fn parse_flock_pids_for_needle_empty_contents_returns_empty() {
        let pids = parse_flock_pids_for_needle("", "08:02:1234");
        assert!(pids.is_empty());
    }

    /// `parse_flock_pids_for_needle` skips malformed lines (missing
    /// fields, non-numeric PIDs) without failing the whole parse.
    /// Pins the graceful-degradation contract for corrupt
    /// `/proc/locks` (unlikely but possible).
    #[test]
    fn parse_flock_pids_for_needle_skips_malformed_lines() {
        let needle = "08:02:1234";
        let contents = "\
1: FLOCK
2: FLOCK ADVISORY WRITE notanumber 08:02:1234 0 EOF
3: FLOCK ADVISORY WRITE 77777 08:02:1234 0 EOF
";
        let pids = parse_flock_pids_for_needle(contents, needle);
        assert_eq!(
            pids,
            vec![77777],
            "only the well-formed matching line contributes",
        );
    }

    /// `read_holders_from_contents` preserves the HolderInfo shape
    /// for matching FLOCK lines. This is the batched-read seam used
    /// by callers that have N needles and want to read `/proc/locks`
    /// exactly once across the batch — the function must return one
    /// [`HolderInfo`] per matching PID in the same order the parser
    /// produces them. `holder_info_for_pid` reads our own cmdline so
    /// we can assert the PID half deterministically on any host.
    #[test]
    fn read_holders_from_contents_returns_holder_info_per_matching_pid() {
        let our_pid = std::process::id();
        let needle = "08:02:1234";
        let contents = format!(
            "1: FLOCK  ADVISORY  WRITE {our_pid} 08:02:1234 0 EOF\n\
             2: POSIX  ADVISORY  WRITE 11111 08:02:1234 0 EOF\n",
        );
        let holders = read_holders_from_contents(&contents, needle);
        assert_eq!(
            holders.len(),
            1,
            "only the FLOCK line at the matching triple produces a holder; \
             POSIX must be filtered: {holders:?}",
        );
        assert_eq!(holders[0].pid, our_pid);
        // cmdline comes from our own /proc/self/cmdline — must be non-empty
        // and distinct from the unavailable sentinel.
        assert_ne!(holders[0].cmdline, "<cmdline unavailable>");
    }

    /// `read_holders_from_contents` with contents empty (no `/proc/locks`
    /// lines at all) returns an empty Vec. Degenerate case — ensures
    /// the batched seam never errors on a clean pool.
    #[test]
    fn read_holders_from_contents_empty_returns_empty() {
        let holders = read_holders_from_contents("", "08:02:1234");
        assert!(holders.is_empty());
    }

    /// `read_holders_from_contents` is deterministic across the same
    /// contents — feeding the same contents+needle twice produces
    /// identical output (no hidden iteration-order dependency). Pins
    /// the batched-call-site invariant: callers that loop `N` needles
    /// over one `contents` must see the same result as `N` per-call
    /// reads of the same snapshot.
    #[test]
    fn read_holders_from_contents_deterministic_for_same_input() {
        let contents = format!(
            "1: FLOCK  ADVISORY  WRITE {pid} 08:02:1234 0 EOF\n",
            pid = std::process::id(),
        );
        let a = read_holders_from_contents(&contents, "08:02:1234");
        let b = read_holders_from_contents(&contents, "08:02:1234");
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].pid, b[0].pid);
        assert_eq!(a[0].cmdline, b[0].cmdline);
    }

    /// `read_holders_for_needle` with an impossible needle returns
    /// an empty Vec. Exercises the /proc/locks read path on any
    /// Linux host without requiring specific lockfile state. The
    /// needle format is `{major:02x}:{minor:02x}:{inode}`; pick
    /// values guaranteed-not-to-exist (major=ff, minor=ff, inode
    /// larger than any real inode at test time).
    #[test]
    fn read_holders_for_needle_no_match_returns_empty() {
        // u64 max inode, max 8-bit major:minor pair. No real
        // /proc/locks entry will match this.
        let needle = "ff:ff:18446744073709551615";
        let holders = read_holders_for_needle(needle)
            .expect("/proc/locks read must succeed on any Linux host");
        assert!(
            holders.is_empty(),
            "impossible needle must not match any holder: {holders:?}"
        );
    }

    /// `needle_from_fd` on an fd opened without flock returns
    /// Ok(None). Verifies the "no `lock:` line" branch via a
    /// bare open — no flock ever taken.
    ///
    /// Uses `tempfile::TempDir` so cleanup runs via RAII on panic
    /// — replaces the earlier manual `/tmp/ktstr-flock-test-...`
    /// that leaked a file if the test panicked.
    #[test]
    fn needle_from_fd_without_flock_returns_none() {
        use std::fs::OpenOptions;
        use tempfile::TempDir;

        let tmp = TempDir::new().expect("tempdir");
        let path = tmp.path().join("unflocked.lock");
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&path)
            .expect("create temp file");
        let needle = needle_from_fd(&file).expect("fdinfo readable");
        assert!(
            needle.is_none(),
            "unflocked fd must produce no needle: {needle:?}"
        );
    }

    /// Positive cross-check: EX-flock a tempfile, derive the
    /// needle from the held fd, then scan /proc/locks with that
    /// needle and verify OUR pid appears as a holder.
    ///
    /// This pins the "both producers must agree on all hosts"
    /// invariant promised in the module doc at the top of this
    /// file: `needle_from_fd` (fdinfo-based) and the implicit
    /// round-trip through `/proc/locks` must meet on the same
    /// triple. A divergence between the fdinfo `lock:` line and
    /// the /proc/locks FLOCK line would break holder enumeration
    /// for any held fd; this test catches it.
    #[test]
    fn needle_from_fd_and_read_holders_round_trip() {
        use tempfile::TempDir;

        let tmp = TempDir::new().expect("tempdir");
        let path = tmp.path().join("flocked.lock");
        let fd = try_flock(&path, FlockMode::Exclusive)
            .expect("try_flock must succeed on fresh tempfile")
            .expect("EX must acquire on clean pool");

        // 1. Derive needle via fdinfo. Must be Some (we hold a flock).
        let needle = needle_from_fd(&fd)
            .expect("fdinfo readable")
            .expect("fdinfo must have a `lock:` line for flocked fd");
        // Format: `{major:02x}:{minor:02x}:{inode}` — exactly 2 colons.
        assert_eq!(
            needle.chars().filter(|&c| c == ':').count(),
            2,
            "needle must be 2-colon format (major:minor:inode), got {needle}",
        );

        // 2. Scan /proc/locks with that needle — our pid must be
        // a holder of the just-taken flock.
        let holders = read_holders_for_needle(&needle).expect("/proc/locks readable");
        let our_pid = std::process::id();
        assert!(
            holders.iter().any(|h| h.pid == our_pid),
            "our pid {our_pid} must appear in holders {holders:?} for \
             needle {needle}",
        );

        drop(fd);
    }

    /// Equivalence between the cached-mountinfo and one-shot
    /// needle-derivation paths.
    ///
    /// `acquire_llc_plan`'s DISCOVER phase reads
    /// `/proc/self/mountinfo` once at the plan level and threads it
    /// through [`needle_from_path_with_mountinfo`] for every LLC
    /// lockfile in the host. The one-shot path
    /// [`needle_from_path`] reads mountinfo inline for each call.
    /// Both must produce byte-identical needles for the same path —
    /// if they diverge, `/proc/locks` byte-equality would fail and
    /// the cached DISCOVER walk would misreport holders.
    ///
    /// Pins equivalence on a real tempfile: the cached path reads
    /// mountinfo once via [`read_mountinfo`] and hands that text to
    /// the contents-seam; the uncached path walks its own internal
    /// read. For the same path, the needles must be equal.
    #[test]
    fn needle_cached_mountinfo_equals_uncached() {
        use tempfile::TempDir;

        let tmp = TempDir::new().expect("tempdir");
        let path = tmp.path().join("cache-equivalence.lock");

        // Materialize a lockfile inode so both paths stat the same
        // underlying file. We use `materialize` because it's the
        // same entry point DISCOVER uses in production, so a
        // divergence between materialize+stat and bare-open would
        // also regress through this test.
        materialize(&path).expect("materialize lockfile");

        // Uncached: inline mountinfo read.
        let uncached = needle_from_path(&path).expect("uncached needle");

        // Cached: read mountinfo once, pass the text through.
        let mountinfo = read_mountinfo().expect("read mountinfo");
        let cached = needle_from_path_with_mountinfo(&path, &mountinfo).expect("cached needle");

        assert_eq!(
            cached, uncached,
            "cached and uncached paths must produce byte-identical needles \
             for the same lockfile. Divergence means DISCOVER's /proc/locks \
             lookup would miss holders the one-shot path would see. \
             uncached={uncached} cached={cached}",
        );
    }

    /// Holder-list equivalence under a live flock.
    ///
    /// Complements `needle_cached_mountinfo_equals_uncached` at the
    /// higher level: beyond "both needles are equal strings," the
    /// full `/proc/locks` scan must surface the same `HolderInfo`
    /// set via both the cached batch API
    /// ([`read_holders_with_mountinfo`]) and the one-shot API
    /// ([`read_holders`]) for a lockfile we actually hold. A
    /// regression where the cached path e.g. canonicalizes
    /// differently (altering the mount-point prefix match) would
    /// surface here: the needles would still be valid triples but
    /// point at different (major, minor) for the same path, and
    /// exactly one of the two scans would find our pid.
    #[test]
    fn read_holders_cached_mountinfo_equals_uncached() {
        use tempfile::TempDir;

        let tmp = TempDir::new().expect("tempdir");
        let path = tmp.path().join("cache-holder-equivalence.lock");

        let fd = try_flock(&path, FlockMode::Exclusive)
            .expect("try_flock must succeed on fresh tempfile")
            .expect("EX must acquire on clean pool");

        // Uncached: inline mountinfo read per call.
        let uncached = read_holders(&path).expect("uncached holders");

        // Cached: read mountinfo once, pass through.
        let mountinfo = read_mountinfo().expect("read mountinfo");
        let cached = read_holders_with_mountinfo(&path, &mountinfo).expect("cached holders");

        // /proc/locks race-safety: holder sets can drift between two
        // scans on a loaded host (peer exits, a separate test flock
        // created/released). Pin the invariant we actually care
        // about: OUR pid appears in BOTH sets.
        let our_pid = std::process::id();
        assert!(
            uncached.iter().any(|h| h.pid == our_pid),
            "our pid {our_pid} must appear in uncached holders {uncached:?}",
        );
        assert!(
            cached.iter().any(|h| h.pid == our_pid),
            "our pid {our_pid} must appear in cached holders {cached:?}",
        );

        drop(fd);
    }

    /// Contents-seam parity: identical synthetic mountinfo text
    /// must produce identical `(major, minor)` tuples via the
    /// cached-wrapper API and the raw parser seam. Catches a
    /// regression where the wrapper's canonicalize-or-fallback
    /// step differs from what the parser expects.
    ///
    /// Uses a tmpfs-covered path so canonicalize succeeds; the
    /// mountinfo fixture covers `/tmp` so both calls hit the same
    /// mount entry.
    #[test]
    fn mount_major_minor_wrapper_matches_parser_seam() {
        let mountinfo = "\
22 28 0:21 / /tmp rw,nosuid,nodev shared:5 - tmpfs tmpfs rw,size=8g
";
        // A tmpfs-covered path the wrapper can canonicalize. `/tmp`
        // itself works on every Linux host nextest runs on.
        let path = Path::new("/tmp");
        let (wrapper_major, wrapper_minor) =
            mount_major_minor_for_path_with_contents(path, mountinfo)
                .expect("wrapper must resolve /tmp under synthetic mountinfo");
        let (parser_major, parser_minor) =
            mount_major_minor_for_path_from_contents(mountinfo, path)
                .expect("parser seam must resolve /tmp");
        assert_eq!(
            (wrapper_major, wrapper_minor),
            (parser_major, parser_minor),
            "wrapper + parser must produce the same (major, minor) for the \
             same (path, mountinfo). Divergence means the cached DISCOVER \
             path is reading different mount state than the uncached \
             one-shot path would.",
        );
        assert_eq!((wrapper_major, wrapper_minor), (0, 21));
    }

    /// `HolderInfo` serializes with `pid` and `cmdline` as
    /// snake_case keys — stable JSON contract for `ktstr locks
    /// --json` downstream consumers. A future refactor that
    /// rename_all = "camelCase" (or drops the derive) would
    /// silently break shell-script consumers that `jq .[].pid`;
    /// this test pins the key names so that regression fails the
    /// build.
    #[test]
    fn holder_info_json_keys_are_snake_case() {
        let holder = HolderInfo {
            pid: 123,
            cmdline: "bash".to_string(),
        };
        let val = serde_json::to_value(&holder).expect("serialize");
        // Pin both keys exist + have expected types.
        assert_eq!(val["pid"], serde_json::json!(123));
        assert_eq!(val["cmdline"], serde_json::json!("bash"));
        // Negative: no camelCase variants slipped in.
        assert!(
            val.get("cmdLine").is_none(),
            "camelCase cmdLine must not appear: {val}",
        );
    }

    /// `try_flock` sets `O_CLOEXEC` on the returned fd. Earlier
    /// revisions missed this flag, which leaked flock-held fds
    /// through `execve` into child processes — the child inherited
    /// the lock, broke assumptions about RAII scope, and
    /// manifested as phantom holders in `/proc/locks` long after
    /// the parent had dropped its guard.
    ///
    /// Verifies the bit directly via `fcntl(F_GETFD)` rather than
    /// asserting via a side-effect (forking an exec'd child is
    /// noisier and harder to match). Failure mode: if the bit is
    /// cleared by a future refactor that re-opens the fd without
    /// re-applying O_CLOEXEC, this test fails the build.
    #[test]
    fn try_flock_sets_cloexec_on_returned_fd() {
        use std::os::fd::AsRawFd;
        use tempfile::TempDir;

        let tmp = TempDir::new().expect("tempdir");
        let path = tmp.path().join("cloexec.lock");
        let fd = try_flock(&path, FlockMode::Exclusive)
            .expect("try_flock must succeed on fresh tempfile")
            .expect("EX must acquire on clean pool");

        // SAFETY: fd is a valid OwnedFd — fcntl F_GETFD is a pure
        // accessor, no concurrent modification, no ownership move.
        let flags = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GETFD) };
        assert!(
            flags >= 0,
            "fcntl F_GETFD must succeed on our fd; got errno={}",
            std::io::Error::last_os_error(),
        );
        assert_eq!(
            flags & libc::FD_CLOEXEC,
            libc::FD_CLOEXEC,
            "FD_CLOEXEC must be set on try_flock-returned fd; \
             flags=0x{flags:x}. Without it, exec'd children \
             inherit the flock and produce phantom holders.",
        );

        drop(fd);
    }
}
