//! Disk-template cache and per-test fan-out.
//!
//! v0 status: this module ships the cache and clone primitives — the
//! `(Filesystem, capacity)` keyed lookup, atomic-rename publish,
//! per-key flock coordination, statfs-based btrfs/xfs gate at the
//! cache root, FICLONE per-test fan-out, and host `mkfs.btrfs`
//! locator. [`ensure_template`] returns an actionable error on cache
//! miss until the host-side template-VM driver lands (see the
//! follow-up task referenced from [`build_template_via_vm`]). The
//! guest-side `mkfs.btrfs /dev/vda` runner already exists in
//! [`crate::vmm::rust_init`] and is gated on
//! `KTSTR_MODE=disk_template`.
//!
//! Once the driver lands, the design is: the framework caches a
//! guest-formatted backing image on the host and per-test
//! reflink-clones it via the `FICLONE` ioctl. The host never execs
//! `mkfs.btrfs` against a real backing file — the kernel inside a
//! one-off template VM is the on-disk-format authority (see the
//! project CLAUDE.md "disk template lifecycle" section).
//!
//! # Lifecycle
//!
//! 1. **Cache lookup.** [`ensure_template`] is called by
//!    [`crate::vmm::init_virtio_blk`] (or callers that pre-warm the
//!    cache). The lookup keys off
//!    `(Filesystem::cache_tag, capacity_bytes)`. Hit → return the
//!    template path.
//! 2. **Lockfile.** Miss → acquire an exclusive flock under
//!    `<cache>/disk_templates/.locks/<key>.lock`. If a peer process is
//!    already populating the cache, this blocks until they finish (or
//!    the timeout fires). After acquire, re-check the cache for
//!    publish-while-waiting.
//! 3. **Template VM boot.** v0: [`build_template_via_vm`] is a stub
//!    that bails with an actionable error after probing
//!    [`locate_host_mkfs_btrfs`]. Once landed, the driver will build
//!    a sparse image of the requested capacity under a tempdir on
//!    the cache filesystem, boot a one-shot guest that runs the
//!    host-supplied `mkfs.btrfs` against `/dev/vda`, wait for clean
//!    shutdown, and discard the guest. The guest dispatch is gated
//!    on `KTSTR_MODE=disk_template` (see
//!    [`crate::vmm::rust_init`]).
//! 4. **Atomic install.** The formatted image is moved into
//!    `<cache>/disk_templates/<key>/template.img` via tempdir +
//!    `rename(2)` ([`store_atomic`]). Partial failures leave no
//!    entry behind.
//! 5. **Per-test fan-out.** [`clone_to_per_test`] FICLONE-clones the
//!    template into a tempfile on the same cache filesystem.
//!    `FICLONE` is O(metadata) — independent of capacity — and copy-
//!    on-write at the extent level so per-test writes do not touch
//!    the template.
//!
//! # Filesystem requirements
//!
//! `FICLONE` is implemented only on btrfs and xfs (kernel
//! `fs/remap_range.c:do_clone_file_range`; the VFS gates on the
//! `remap_file_range` superblock op which neither tmpfs nor ext4
//! provide). [`verify_cache_dir_supports_reflink`] checks the cache
//! filesystem's `statfs.f_type` and bails fast on non-supporting
//! filesystems with an actionable error.
//!
//! # Why not the `reflink` crate
//!
//! The `reflink` crate (v0.1.3) hardcodes
//! `IOCTL_FICLONE = 0x40049409` with a TODO questioning cross-arch
//! validity. The Linux generic ioctl encoding makes this number the
//! same on x86_64 and aarch64 (both use `<asm-generic/ioctl.h>`),
//! but `reflink::reflink` also opens the destination via
//! `OpenOptions::create_new`, which obscures the tempfile pattern
//! the cache fan-out wants (caller already controls dest creation
//! to apply mode bits and chown atomically). A direct `libc::ioctl`
//! call lets the cache module own dest semantics and produce
//! errno-precise diagnostics.

use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};

use crate::flock::{FlockMode, acquire_flock_with_timeout};
use crate::vmm::disk_config::Filesystem;

/// Cache subdirectory suffix passed to
/// [`crate::cache::resolve_cache_root_with_suffix`]. Distinct from
/// `"kernels"` (kernel image cache) and `"models"` (LLM cache) so
/// the three flavors share a parent root via `KTSTR_CACHE_DIR` /
/// `XDG_CACHE_HOME` without colliding on filesystem paths.
const CACHE_SUFFIX: &str = "disk_templates";

/// Filename used for the template image inside each cache entry.
const TEMPLATE_FILENAME: &str = "template.img";

/// Lockfile subdirectory name (per-key serialization).
const LOCK_DIR_NAME: &str = ".locks";

/// `FICLONE` ioctl command number per Linux uapi
/// `include/uapi/linux/fs.h:312` — `_IOW(0x94, 9, int)`.
///
/// Generic ioctl encoding (shared by x86_64 and aarch64):
/// `(_IOC_WRITE << 30) | (sizeof(int) << 16) | (0x94 << 8) | 9`
/// = `0x40000000 | 0x40000 | 0x9400 | 9` = `0x40049409`.
///
/// This is the same value `reflink-0.1.3/src/sys/unix.rs:10`
/// hardcodes (with a now-stale "is this equal on all archs?" TODO).
/// Pinned here as a `const` so a future arch port can audit it
/// against the host's `<asm/ioctl.h>` instead of grepping a
/// transitive dep.
const FICLONE_IOCTL: libc::c_ulong = 0x4004_9409;

/// Maximum wall-clock duration to wait for a peer process holding
/// the cache lockfile while it builds the template.
///
/// 10 minutes accommodates the worst-case template build (cold
/// kernel cache + first-run `mkfs.btrfs` on slow storage) without
/// hanging interactive runs forever. Operators who hit the timeout
/// see a holder list parsed from `/proc/locks` so they can kill a
/// stuck peer or wait by hand.
const TEMPLATE_LOCK_TIMEOUT: Duration = Duration::from_secs(600);

/// btrfs `statfs.f_type` magic per `linux/magic.h`. `libc::BTRFS_SUPER_MAGIC`
/// covers GNU but is gated on Linux; pinning the constant defends
/// against a future libc minor release that drops/renames it.
const BTRFS_SUPER_MAGIC: i64 = 0x9123_683e;
/// xfs `statfs.f_type` magic per `linux/magic.h`. Same reasoning as
/// `BTRFS_SUPER_MAGIC`.
const XFS_SUPER_MAGIC: i64 = 0x5846_5342;

/// Run `statfs(2)` against an existing path and return the populated
/// `libc::statfs` buffer. Used by [`verify_cache_dir_supports_reflink`]
/// and [`store_atomic`] (the latter compares two `f_type`s and the
/// `f_fsid` pair to detect cross-filesystem renames before they fail
/// with a less-obvious `EXDEV`).
fn statfs_path(path: &Path) -> Result<libc::statfs> {
    let cstr = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())
        .with_context(|| format!("path contains nul bytes: {path:?}"))?;
    // SAFETY: cstr is a NUL-terminated C string, statfs writes into
    // a stack-allocated zero-initialized buffer of the correct
    // layout. The kernel returns 0 on success and -1 with errno set
    // on failure.
    let mut buf: libc::statfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statfs(cstr.as_ptr(), &mut buf) };
    if rc != 0 {
        let err = io::Error::last_os_error();
        return Err(anyhow!("statfs({path:?}) failed: {err}"));
    }
    Ok(buf)
}

/// Resolve the cache root directory for disk templates.
///
/// Reuses the global `KTSTR_CACHE_DIR` / `XDG_CACHE_HOME` / `$HOME`
/// cascade documented at
/// [`crate::cache::resolve_cache_root_with_suffix`]. Does not
/// create the directory; callers materialize on demand via
/// [`std::fs::create_dir_all`].
pub(crate) fn cache_root() -> Result<PathBuf> {
    crate::cache::resolve_cache_root_with_suffix(CACHE_SUFFIX)
}

/// Verify that `dir` lives on a filesystem that supports `FICLONE`.
///
/// Returns `Ok(())` for btrfs and xfs. Other filesystems (tmpfs,
/// ext4, fuse, …) bail with an actionable error naming the
/// filesystem magic and pointing the operator at
/// `KTSTR_CACHE_DIR` / `XDG_CACHE_HOME` for an override.
///
/// Walks up the path tree until a real component exists — the cache
/// root is created lazily, and `statfs` on a path that does not
/// exist yet returns `ENOENT`. Walking up reaches the parent
/// `XDG_CACHE_HOME` (or `$HOME/.cache`) and probes that filesystem
/// instead, which is the correct answer because filesystem boundaries
/// only show up at mount points and the cache root inherits its
/// parent's filesystem unless an operator mounted something custom
/// on top.
pub(crate) fn verify_cache_dir_supports_reflink(dir: &Path) -> Result<()> {
    let mut probe: PathBuf = dir.to_path_buf();
    loop {
        if probe.exists() {
            break;
        }
        match probe.parent() {
            Some(p) => probe = p.to_path_buf(),
            None => bail!(
                "no existing ancestor of {dir:?} found while probing \
                 cache filesystem; cannot verify FICLONE support",
            ),
        }
    }
    let buf = statfs_path(&probe).with_context(|| {
        format!(
            "cannot verify FICLONE support for cache directory {dir:?} \
             (probed ancestor {probe:?})"
        )
    })?;
    let fs_type = buf.f_type as i64;
    if fs_type == BTRFS_SUPER_MAGIC || fs_type == XFS_SUPER_MAGIC {
        return Ok(());
    }
    bail!(
        "ktstr disk-template cache requires a btrfs or xfs filesystem \
         for FICLONE-based per-test fan-out; cache directory {dir:?} \
         lives on a filesystem whose statfs.f_type=0x{fs_type:x} (not \
         btrfs=0x{btrfs:x}, not xfs=0x{xfs:x}). Set KTSTR_CACHE_DIR \
         to a directory on a btrfs/xfs mount, or use \
         Filesystem::Raw which does not need a reflink-capable cache.",
        btrfs = BTRFS_SUPER_MAGIC,
        xfs = XFS_SUPER_MAGIC,
    );
}

/// Cache key for one template flavor (filesystem variant +
/// capacity).
///
/// Renders as `"{tag}-{capacity_mib}m"`, e.g. `"btrfs-256m"`. The
/// rendering is stable across rebuilds — capacity is forced into MiB
/// (rather than raw bytes) so every entry has the same magnitude
/// regardless of compiler-side rounding, and the `m` suffix
/// disambiguates from any future GiB/sector-count keying. New
/// `Filesystem` variants must pick a new `cache_tag` (see the
/// `cache_tag` doc).
pub(crate) fn template_cache_key(fs: Filesystem, capacity_bytes: u64) -> String {
    let mib = capacity_bytes / (1024 * 1024);
    format!("{tag}-{mib}m", tag = fs.cache_tag())
}

/// Path to the template image for the given key, relative to the
/// cache root. Does not check existence — use [`lookup`] for that.
pub(crate) fn template_path_for_key(key: &str) -> Result<PathBuf> {
    let root = cache_root()?;
    Ok(root.join(key).join(TEMPLATE_FILENAME))
}

/// Path to the per-key lockfile, relative to the cache root.
fn lock_path_for_key(key: &str) -> Result<PathBuf> {
    let root = cache_root()?;
    Ok(root.join(LOCK_DIR_NAME).join(format!("{key}.lock")))
}

/// Look up a cached template by key.
///
/// Returns `Some(path)` if the template image exists and is
/// readable, `None` otherwise (cache miss, partial install, or
/// removed by hand). Callers materialize a miss via
/// [`ensure_template`].
pub(crate) fn lookup(key: &str) -> Result<Option<PathBuf>> {
    let path = template_path_for_key(key)?;
    match std::fs::metadata(&path) {
        Ok(meta) if meta.is_file() => Ok(Some(path)),
        Ok(_) => Ok(None),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("stat cached template {path:?}")),
    }
}

/// Atomically install the file at `src_path` as the template for
/// `key`.
///
/// Stages under `<cache>/<key>.tmp.<pid>/template.img` and then
/// `rename(2)`'s the staging directory into place. Concurrent
/// installs serialize via the per-key lockfile (callers acquire
/// the lock before staging — see [`ensure_template`]); this
/// function trusts the caller already holds the lock.
///
/// The atomic-rename pattern matches [`crate::cache::CacheDir::store`]:
/// on partial failure the staging directory is removed by the OS on
/// next boot if not by the caller, but the live cache always sees
/// either no entry or a complete entry — never a half-written one.
pub(crate) fn store_atomic(key: &str, src_path: &Path) -> Result<PathBuf> {
    let root = cache_root()?;
    std::fs::create_dir_all(&root)
        .with_context(|| format!("create disk-template cache root {root:?}"))?;
    let final_dir = root.join(key);
    if final_dir.exists() {
        // A peer published the entry between our lookup and store
        // calls. Discard the new one — both should be byte-identical
        // (same capacity, same fs, same mkfs.btrfs version on the
        // host).
        return Ok(final_dir.join(TEMPLATE_FILENAME));
    }
    // Pre-flight cross-filesystem check. `rename(2)` returns EXDEV
    // when src and dest live on different filesystems; the caller
    // should have staged src_path on the cache filesystem, but a
    // bug in caller logic (e.g. staging via `tempfile::tempfile()`
    // which honors `TMPDIR`) would surface as a less-obvious EXDEV
    // from `fs::rename` below. statfs both paths up front and bail
    // with a precise diagnostic naming the f_type magics. f_fsid is
    // also compared because two distinct btrfs subvolumes share an
    // f_type but differ in f_fsid, and rename(2) treats them as
    // different filesystems on most kernels.
    let src_buf = statfs_path(src_path)
        .with_context(|| format!("statfs source {src_path:?} for cross-fs check"))?;
    let dest_buf = statfs_path(&root)
        .with_context(|| format!("statfs cache root {root:?} for cross-fs check"))?;
    if src_buf.f_type != dest_buf.f_type
        || fsid_bytes(&src_buf) != fsid_bytes(&dest_buf)
    {
        bail!(
            "disk-template store_atomic: source {src_path:?} \
             (f_type=0x{src_type:x}) and cache root {root:?} \
             (f_type=0x{dest_type:x}) live on different filesystems. \
             rename(2) would return EXDEV. Stage the template image \
             on the cache filesystem before calling store_atomic.",
            src_type = src_buf.f_type as i64,
            dest_type = dest_buf.f_type as i64,
        );
    }
    let staging = root.join(format!("{key}.tmp.{pid}", pid = std::process::id()));
    if staging.exists() {
        std::fs::remove_dir_all(&staging)
            .with_context(|| format!("remove stale staging directory {staging:?}"))?;
    }
    std::fs::create_dir_all(&staging)
        .with_context(|| format!("create staging directory {staging:?}"))?;
    let staging_image = staging.join(TEMPLATE_FILENAME);
    // Move src_path into the staging dir. `fs::rename` is atomic on
    // the same filesystem; the cross-fs gate above guarantees that.
    std::fs::rename(src_path, &staging_image)
        .with_context(|| format!("rename {src_path:?} -> {staging_image:?}"))?;
    // Final atomic publish.
    std::fs::rename(&staging, &final_dir).with_context(|| {
        format!(
            "publish staging {staging:?} -> {final_dir:?} (cache key {key})",
        )
    })?;
    Ok(final_dir.join(TEMPLATE_FILENAME))
}

/// Extract `f_fsid` as a fixed-size byte tuple for equality
/// comparisons between two `statfs` results. `libc::fsid_t` has
/// platform-dependent struct layout (the public ABI is "an array of
/// two ints" but the layout varies between glibc and musl), so we
/// pull the bytes through a fixed-width read instead of relying on
/// `PartialEq` which `fsid_t` does not implement.
fn fsid_bytes(buf: &libc::statfs) -> [u8; std::mem::size_of::<libc::fsid_t>()] {
    let mut out = [0u8; std::mem::size_of::<libc::fsid_t>()];
    // SAFETY: we read exactly size_of::<fsid_t>() bytes out of an
    // initialized statfs struct. Both source and destination cover
    // the same byte range, no aliasing, no out-of-bounds.
    unsafe {
        std::ptr::copy_nonoverlapping(
            &buf.f_fsid as *const libc::fsid_t as *const u8,
            out.as_mut_ptr(),
            std::mem::size_of::<libc::fsid_t>(),
        );
    }
    out
}

/// Acquire an exclusive flock on the per-key cache lockfile.
///
/// Held by [`ensure_template`] for the duration of a template build
/// to serialize concurrent test starts that all want the same
/// template. The lockfile lives under the cache root's `.locks/`
/// subdirectory so the cache enumeration code skips it.
///
/// Returns the flock fd; dropping releases the lock. Bails on
/// timeout with a holder list (PIDs, comms) so operators can
/// triage a stuck peer.
pub(crate) fn acquire_template_lock(key: &str) -> Result<std::os::fd::OwnedFd> {
    let lock_path = lock_path_for_key(key)?;
    acquire_flock_with_timeout(
        &lock_path,
        FlockMode::Exclusive,
        TEMPLATE_LOCK_TIMEOUT,
        &format!("disk-template cache entry {key}"),
        Some(
            "A peer ktstr process is currently building this template. \
             Wait for it to finish, kill the peer with the listed PID, \
             or remove the lockfile if you are sure it is stale.",
        ),
    )
}

/// FICLONE-clone `src_path` into `dest_path`.
///
/// Both paths must reside on the same filesystem AND that filesystem
/// must implement `remap_file_range` (btrfs or xfs).
/// [`verify_cache_dir_supports_reflink`] gates on this for the cache
/// root; per-test fan-out callers must arrange for `dest_path` to
/// live under the cache root or another filesystem-validated path.
///
/// Returns the open `File` for `dest_path` ready for the device to
/// use. Caller is responsible for `unlink`-ing `dest_path` after
/// use. Failures with `EOPNOTSUPP` / `EXDEV` / `EINVAL` indicate a
/// reflink-incapable filesystem or cross-fs attempt and bail with a
/// hint at the operator's KTSTR_CACHE_DIR.
pub(crate) fn clone_to_per_test(src_path: &Path, dest_path: &Path) -> Result<File> {
    let src = OpenOptions::new()
        .read(true)
        .open(src_path)
        .with_context(|| format!("open template source {src_path:?}"))?;
    // Open dest with O_CREAT | O_EXCL — if a peer materialized the
    // same path between our caller's tempfile generation and this
    // call, we want a hard error, not a silent overwrite. The
    // tempfile name space (mkstemp-style) plus per-process pid
    // suffix makes a real collision astronomically unlikely.
    let dest = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(dest_path)
        .with_context(|| format!("open dest path {dest_path:?} for FICLONE"))?;
    // SAFETY: ioctl FICLONE takes (dest_fd, FICLONE, src_fd) per
    // Linux uapi. Both fds are valid for the duration of the call;
    // ioctl does not retain them. The kernel returns 0 on success
    // and -1 with errno set on failure.
    let rc = unsafe { libc::ioctl(dest.as_raw_fd(), FICLONE_IOCTL, src.as_raw_fd()) };
    if rc != 0 {
        let err = io::Error::last_os_error();
        // Best-effort cleanup of the half-written dest file.
        let _ = std::fs::remove_file(dest_path);
        return Err(anyhow!(
            "FICLONE {src_path:?} -> {dest_path:?} failed: {err}. \
             This usually means the destination filesystem does not \
             support reflinks (btrfs/xfs only) or the source and \
             destination live on different filesystems. Set \
             KTSTR_CACHE_DIR to a directory on a btrfs/xfs mount.",
        ));
    }
    Ok(dest)
}

/// Locate `mkfs.btrfs` on the host `PATH` so it can be packed into
/// the template-VM initramfs.
///
/// Walks `PATH` (split on `:`) and returns the first directory that
/// contains an executable `mkfs.btrfs`. Bails with an actionable
/// error when the binary is absent — this is the operator's signal
/// to install `btrfs-progs` (or equivalent distro package) before
/// using `Filesystem::Btrfs`.
///
/// The host binary is NOT exec'd at template-build time — it is
/// embedded into the template-VM initramfs and exec'd by guest init
/// inside the VM. The kernel inside the VM is the on-disk-format
/// authority; the host binary just provides the `mkfs.btrfs`
/// userspace driver to drive the kernel into formatting.
pub(crate) fn locate_host_mkfs_btrfs() -> Result<PathBuf> {
    locate_host_binary("mkfs.btrfs", "btrfs-progs")
}

/// Locate a binary by name on the host `PATH`. Used for
/// `mkfs.btrfs` today; future filesystem variants ([`Filesystem`]
/// extensions) reuse the same machinery for their respective mkfs
/// binaries.
fn locate_host_binary(name: &str, package_hint: &str) -> Result<PathBuf> {
    let path_var = std::env::var_os("PATH")
        .ok_or_else(|| anyhow!("PATH environment variable is unset; cannot locate {name}"))?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(name);
        // `metadata` follows symlinks (`stat`, not `lstat`), so a
        // PATH entry like `/usr/sbin/mkfs.btrfs -> /usr/bin/btrfs`
        // resolves to the target's regular-file metadata. After
        // confirming we have a non-empty regular file, canonicalize
        // the candidate so the embedded copy in the template-VM
        // initramfs points at the real binary instead of a symlink
        // that the guest cannot follow once unpacked.
        let Ok(meta) = std::fs::metadata(&candidate) else {
            continue;
        };
        if !meta.is_file() {
            continue;
        }
        // Reject zero-byte stand-ins (a directory entry that exists
        // but contains no executable code). A bare `touch` or a
        // failed-install leftover lands here. Without this gate the
        // template-VM initramfs would pack a 0-byte binary that
        // exec(2) returns ENOEXEC on, with no clear hint at the
        // host-side root cause.
        if meta.len() == 0 {
            continue;
        }
        // Don't filter by mode bits — different distros have
        // different group/world execute settings on
        // /usr/sbin/mkfs.* binaries; the `exec` syscall checks
        // permissions correctly when the guest runs the binary, so
        // the host-side resolver only verifies "regular non-empty
        // file exists at this path."
        let canonical = std::fs::canonicalize(&candidate)
            .with_context(|| format!("canonicalize {candidate:?}"))?;
        return Ok(canonical);
    }
    bail!(
        "{name} not found on PATH. \
         Install the {package_hint} package (or your distro's \
         equivalent) so the disk-template VM can format Btrfs disks. \
         PATH={path:?}",
        path = path_var,
    )
}

/// Ensure a template exists for `(fs, capacity_bytes)` and return
/// the cached image path.
///
/// Cache hits return immediately (no lock acquisition, no boot).
/// Misses acquire the per-key flock, re-check, then build the
/// template via [`build_template_via_vm`] and atomically install it.
///
/// Callers (typically [`crate::vmm::init_virtio_blk`]) then pass
/// the returned path to [`clone_to_per_test`] for the per-test
/// reflink clone.
pub(crate) fn ensure_template(fs: Filesystem, capacity_bytes: u64) -> Result<PathBuf> {
    let key = template_cache_key(fs, capacity_bytes);
    if let Some(hit) = lookup(&key)? {
        return Ok(hit);
    }
    let root = cache_root()?;
    // First-pass walk-up check: catches the common case (operator
    // pointed KTSTR_CACHE_DIR at a non-reflink fs) before we
    // create_dir_all on a doomed path.
    verify_cache_dir_supports_reflink(&root)?;
    std::fs::create_dir_all(&root)
        .with_context(|| format!("create disk-template cache root {root:?}"))?;
    // Re-verify against the now-existing cache root. Closes the
    // case where the walk-up landed on an ancestor that lives on a
    // different mount than the eventual cache directory (e.g. the
    // operator created a fresh sub-mount under HOME between probe
    // and now, or `~/.cache` is itself a separate mountpoint that
    // is not reflink-capable while `$HOME` is).
    verify_cache_dir_supports_reflink(&root)?;
    let _lock = acquire_template_lock(&key)?;
    // Re-check after acquire — a peer may have published while we
    // waited.
    if let Some(hit) = lookup(&key)? {
        return Ok(hit);
    }
    let staged = build_template_via_vm(fs, capacity_bytes, &root)
        .with_context(|| format!("build disk template for {key}"))?;
    let final_path = store_atomic(&key, &staged)
        .with_context(|| format!("install disk template {key}"))?;
    Ok(final_path)
}

/// Build a fresh template image by booting a one-shot template VM.
///
/// Stub today — the host-side template-VM driver lives in a future
/// patch that wires this function up to [`crate::vmm::KtstrVmBuilder`]
/// with `template_mode=true`. Callers see an actionable error
/// instead of a silent failure.
///
/// The driver, once landed, will:
/// 1. Materialize a sparse `template.img.in-flight` of size
///    `capacity_bytes` under `cache_root` (so it shares the cache's
///    btrfs/xfs filesystem and survives the rename(2) into place).
/// 2. Locate `mkfs.btrfs` via [`locate_host_mkfs_btrfs`].
/// 3. Construct a `KtstrVmBuilder` with `KTSTR_MODE=disk_template`,
///    the sparse image as the disk backing, and `mkfs.btrfs` packed
///    as an extra binary at `bin/mkfs.btrfs`.
/// 4. Run the VM until it reboots; the guest dispatch in
///    [`crate::vmm::rust_init`] runs `mkfs.btrfs /dev/vda` and
///    reboots cleanly.
/// 5. Return the populated `template.img.in-flight` path for
///    [`store_atomic`] to publish.
fn build_template_via_vm(
    fs: Filesystem,
    capacity_bytes: u64,
    _cache_root: &Path,
) -> Result<PathBuf> {
    // Probe that the host has the mkfs binary BEFORE bailing on the
    // VM driver — operators get the more actionable error first.
    let _mkfs = locate_host_mkfs_btrfs()?;
    bail!(
        "disk-template VM builder is not yet wired to KtstrVmBuilder \
         (fs={fs:?}, capacity_bytes={capacity_bytes}). The host-side \
         driver is staged for a follow-up patch in this task series. \
         Until it lands, ktstr supports Filesystem::Raw on this code \
         path; pre-populate the disk-template cache by hand if you \
         need Filesystem::Btrfs.",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_renders_capacity_in_mib() {
        let key = template_cache_key(Filesystem::Btrfs, 256 * 1024 * 1024);
        assert_eq!(key, "btrfs-256m");
        let key = template_cache_key(Filesystem::Raw, 1024 * 1024 * 1024);
        assert_eq!(key, "raw-1024m");
    }

    #[test]
    fn cache_key_truncates_sub_mib_capacity_to_zero() {
        // Capacity less than 1 MiB rounds down to 0m. This is
        // intentional — DiskConfig's capacity is u32 megabytes (see
        // capacity_mb), so the only way to hit this is constructing
        // capacity_bytes by hand below 2^20. Pinning the rendering
        // for that corner so a future bug that rounds up silently
        // is caught.
        let key = template_cache_key(Filesystem::Btrfs, 1024);
        assert_eq!(key, "btrfs-0m");
    }

    #[test]
    fn template_path_includes_filename_constant() {
        let path = template_path_for_key("btrfs-256m").expect("resolve template path");
        assert!(path.ends_with(format!("btrfs-256m/{TEMPLATE_FILENAME}")));
    }

    #[test]
    fn lookup_missing_returns_none() {
        // Use a tempdir as cache root so we don't pollute the
        // operator's real cache. The cache_root() helper reads
        // KTSTR_CACHE_DIR; setting it for the lifetime of the test
        // via EnvVarGuard isolates per-test state.
        let tmp = tempfile::tempdir().expect("create tempdir");
        let _guard = crate::test_support::test_helpers::EnvVarGuard::set(
            "KTSTR_CACHE_DIR",
            tmp.path(),
        );
        let result = lookup("missing-key").expect("lookup must not error on miss");
        assert!(result.is_none());
    }

    #[test]
    fn store_atomic_publishes_then_lookup_finds() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let _guard = crate::test_support::test_helpers::EnvVarGuard::set(
            "KTSTR_CACHE_DIR",
            tmp.path(),
        );
        // Stage a fake template under the cache root so the rename
        // is on the same filesystem.
        let cache_root_path = cache_root().unwrap();
        std::fs::create_dir_all(&cache_root_path).unwrap();
        let staged = cache_root_path.join("staged.img");
        std::fs::write(&staged, b"FAKE_TEMPLATE_BODY").unwrap();
        let key = "test-key";
        let installed = store_atomic(key, &staged).expect("store_atomic publishes");
        assert!(installed.ends_with(format!("{key}/{TEMPLATE_FILENAME}")));
        // Now lookup must find it.
        let found = lookup(key).expect("lookup ok").expect("lookup must hit");
        assert_eq!(found, installed);
        // And content survived the rename.
        let body = std::fs::read(&found).unwrap();
        assert_eq!(body, b"FAKE_TEMPLATE_BODY");
    }

    #[test]
    fn store_atomic_idempotent_on_existing_entry() {
        // If a peer published between lookup() and store_atomic(),
        // the second store_atomic returns the existing path rather
        // than raising — by design (both writes produce
        // byte-identical templates for the same key).
        let tmp = tempfile::tempdir().expect("create tempdir");
        let _guard = crate::test_support::test_helpers::EnvVarGuard::set(
            "KTSTR_CACHE_DIR",
            tmp.path(),
        );
        let cache_root_path = cache_root().unwrap();
        std::fs::create_dir_all(&cache_root_path).unwrap();
        let staged1 = cache_root_path.join("staged1.img");
        std::fs::write(&staged1, b"FIRST").unwrap();
        let key = "idem-key";
        let installed1 = store_atomic(key, &staged1).unwrap();
        // Second call with a different staging file must return the
        // already-installed path without overwriting it.
        let staged2 = cache_root_path.join("staged2.img");
        std::fs::write(&staged2, b"SECOND").unwrap();
        let installed2 = store_atomic(key, &staged2).unwrap();
        assert_eq!(installed1, installed2);
        // Content must remain "FIRST" — store_atomic on an existing
        // entry is a no-op publish.
        let body = std::fs::read(&installed2).unwrap();
        assert_eq!(body, b"FIRST");
    }

    #[test]
    fn locate_host_binary_actionable_error_when_missing() {
        // Override PATH to a single empty dir so the host binary is
        // guaranteed to be missing.
        let tmp = tempfile::tempdir().expect("create tempdir");
        let _guard = crate::test_support::test_helpers::EnvVarGuard::set(
            "PATH",
            tmp.path(),
        );
        let err = locate_host_binary("nonexistent-binary-9242", "imagined-package")
            .expect_err("must error when binary absent");
        let msg = err.to_string();
        assert!(
            msg.contains("nonexistent-binary-9242"),
            "error names the binary: {msg}",
        );
        assert!(
            msg.contains("imagined-package"),
            "error names the package hint: {msg}",
        );
    }

    #[test]
    fn build_template_via_vm_bails_with_actionable_message() {
        // Until the VM driver is wired up, the stub bails with a
        // message that names the deferred work and the workaround.
        // Pin the diagnostic wording so a future driver landing
        // updates this test in lock-step (and a regression to
        // silent-failure surfaces here).
        let tmp = tempfile::tempdir().expect("create tempdir");
        let _guard = crate::test_support::test_helpers::EnvVarGuard::set(
            "KTSTR_CACHE_DIR",
            tmp.path(),
        );
        // The mkfs probe runs first; on hosts without btrfs-progs
        // the error talks about that. Skip the assertion on those
        // hosts to avoid CI-environment skew.
        if locate_host_mkfs_btrfs().is_err() {
            return;
        }
        let err = build_template_via_vm(Filesystem::Btrfs, 256 * 1024 * 1024, tmp.path())
            .expect_err("driver is not wired yet");
        let msg = err.to_string();
        assert!(msg.contains("KtstrVmBuilder") || msg.contains("Filesystem::Raw"));
    }

    #[test]
    fn verify_cache_dir_walks_up_to_existing_ancestor() {
        // A non-existent cache root must still produce a usable
        // statfs result by walking up. Use a path under /tmp (which
        // exists) and confirm the helper does not bail on ENOENT.
        let nonexistent =
            std::path::PathBuf::from("/tmp/ktstr-disk-template-test-does-not-exist-9242/sub/dir");
        // The result depends on the host's /tmp filesystem; this
        // test only pins that the helper does not panic and either
        // returns Ok (btrfs/xfs /tmp) or a fs-magic-named error
        // (anything else).
        match verify_cache_dir_supports_reflink(&nonexistent) {
            Ok(()) => { /* host /tmp is btrfs/xfs */ }
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("statfs.f_type") || msg.contains("FICLONE"),
                    "unexpected error wording: {msg}",
                );
            }
        }
    }
}
