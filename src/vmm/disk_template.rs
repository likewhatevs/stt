//! Disk-template cache and per-test fan-out.
//!
//! This module ships the cache and clone primitives — the
//! `(Filesystem, capacity)` keyed lookup, atomic-rename publish,
//! per-key flock coordination, statfs-based btrfs/xfs gate at the
//! cache root, FICLONE per-test fan-out, host mkfs locator (see
//! [`Filesystem::mkfs_binary_name`] and [`locate_host_mkfs`]),
//! AND the host-side template-VM driver in
//! [`build_template_via_vm`] that boots a one-shot guest to run
//! the variant's `mkfs.<fstype>` against a sparse staging image
//! (`mkfs.btrfs /dev/vda` for `Filesystem::Btrfs`). The
//! guest-side dispatch lives in [`crate::vmm::rust_init`] and is
//! gated on `KTSTR_MODE=disk_template`.
//!
//! Design: the framework caches a guest-formatted backing image on
//! the host and per-test reflink-clones it via the `FICLONE` ioctl.
//! The host never execs `mkfs.btrfs` against a real backing file —
//! the kernel inside a one-off template VM is the on-disk-format
//! authority (see the project CLAUDE.md "disk template lifecycle"
//! section).
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
//! 3. **Template VM boot.** [`build_template_via_vm`] materialises
//!    a sparse `template.img.in-flight.<cache_key>.<pid>` of the
//!    requested capacity under the cache root (so `rename(2)` into
//!    place is same-filesystem; the `<cache_key>` qualifier
//!    disambiguates cross-key concurrent builds in the same pid —
//!    see [`staging_image_path`]), packs the host's mkfs binary
//!    (resolved via [`locate_host_mkfs`]) into the template-VM
//!    initramfs at `bin/<mkfs_name>`, and boots a one-shot guest
//!    with `KTSTR_MODE=disk_template` on the kernel cmdline. The
//!    disk attaches via
//!    [`crate::vmm::KtstrVmBuilder::template_staging_image`], which
//!    bypasses both the per-test `Raw` tempfile branch AND the
//!    `Btrfs` ensure_template branch in
//!    [`crate::vmm::KtstrVm::init_virtio_blk`] — the template-build
//!    VM cannot recursively re-enter the cache it is itself
//!    populating. Guest dispatch
//!    ([`crate::vmm::rust_init::run_disk_template_mode`]) execs
//!    `/bin/<mkfs_binary_name>` against `/dev/vda` (currently
//!    `mkfs.btrfs` for `Filesystem::Btrfs` per
//!    [`Filesystem::mkfs_binary_name`]) and reboots cleanly; on
//!    non-zero exit / timeout the staging image is unlinked and
//!    the build bails with the trailing guest stderr.
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
//! `fs/remap_range.c:vfs_clone_file_range`; the VFS gates on the
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

use crate::flock::{FlockMode, acquire_flock_with_timeout, try_flock};
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
/// # Budget breakdown
///
/// 600s = 10 minutes. The template build inside the lock holder
/// covers, in order:
///
/// - **Kernel boot** (~2-30s on a cold page cache, sub-second when
///   the kernel image is already mapped from a prior test).
///   First-run on a host without the kernel image cached can stall
///   on disk reads of the kernel + initramfs.
/// - **`mkfs.<fstype>` execution against `/dev/vda`** (~1-30s for a
///   256 MiB-1 GiB device on tmpfs/btrfs/xfs; 1-3 minutes on slow
///   spinning storage when the cache directory points at HDD-backed
///   storage). `mkfs.btrfs` does extent-tree initialisation plus
///   metadata block allocation — bound by storage IOPS, not CPU.
/// - **VM teardown** (sub-second).
///
/// The 10-minute ceiling absorbs the worst plausible host: a cold
/// HDD-backed `KTSTR_CACHE_DIR` running its first ever `mkfs.btrfs`
/// against a multi-GiB capacity. Below 10 minutes, a CI runner with
/// a cold cache and contentious IO would surface flaky-template
/// timeouts. Above 10 minutes, an interactive run against a
/// genuinely-stuck peer would hang the developer's terminal beyond
/// their patience threshold.
///
/// Operators who hit the timeout see a holder list parsed from
/// `/proc/locks` so they can kill a stuck peer (`kill <pid>`) or
/// wait by hand. The lockfile path is also surfaced so manual
/// cleanup is always available.
const TEMPLATE_LOCK_TIMEOUT: Duration = Duration::from_secs(600);

/// btrfs `statfs.f_type` magic per `linux/magic.h`. `libc::BTRFS_SUPER_MAGIC`
/// covers GNU but is gated on Linux; pinning the constant defends
/// against a future libc minor release that drops/renames it.
///
/// Stored as `u64` so the comparison expression has matching unsigned
/// types. `statfs.f_type` is `__fsword_t` — `i64` on 64-bit Linux
/// (LP64), and ktstr only targets 64-bit Linux (`x86_64-unknown-linux-*`
/// and `aarch64-unknown-linux-*`). The call-site `as u64` cast
/// preserves the bit pattern of an `i64` source, so the comparison
/// against `0x9123_683e` matches the on-disk magic correctly on every
/// supported target. Bit 31 of `BTRFS_SUPER_MAGIC` IS set
/// (`0x9123_683e` = `1001 0001 ...`); on a hypothetical 32-bit Linux
/// port where `__fsword_t` is `i32`, the `as u64` cast would
/// sign-extend the negative value into the high 32 bits and break
/// the comparison. ktstr does not support 32-bit targets, so the
/// sign-extension trap is unreachable in practice. `XFS_SUPER_MAGIC`
/// (`0x5846_5342`) has bit 31 clear and would survive that
/// hypothetical 32-bit port; the asymmetry is documented here so a
/// future 32-bit attempt fails loudly on btrfs rather than silently
/// rejecting valid btrfs cache directories.
const BTRFS_SUPER_MAGIC: u64 = 0x9123_683e;
/// xfs `statfs.f_type` magic per `linux/magic.h`. Same reasoning as
/// `BTRFS_SUPER_MAGIC`.
const XFS_SUPER_MAGIC: u64 = 0x5846_5342;

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
///
/// When the walk-up lands on an ancestor that is not `dir` itself —
/// because no leaf component of `dir` exists yet — the bail
/// diagnostic names both `dir` and the probed ancestor so the
/// operator can tell the f_type they see came from an ancestor, not
/// from `dir`. This matters when `dir` would, once created, mount on
/// a different filesystem than the ancestor (e.g. `KTSTR_CACHE_DIR`
/// points at a not-yet-mounted btrfs subvolume): the diagnostic does
/// not silently mislead about which filesystem was probed.
///
/// Symlink behaviour: `Path::exists` follows symlinks, so a
/// dangling symlink probes as missing and the walk-up moves to the
/// symlink's parent (the directory containing the symlink), not the
/// symlink target's parent. Operators who set `KTSTR_CACHE_DIR` to a
/// dangling symlink see the diagnostic name the symlink container's
/// filesystem rather than the (nonexistent) target's. Resolving the
/// symlink target before probing is intentionally NOT done — the
/// missing target is a configuration error, not a filesystem-type
/// question.
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
    let fs_type = buf.f_type as u64;
    if fs_type == BTRFS_SUPER_MAGIC || fs_type == XFS_SUPER_MAGIC {
        return Ok(());
    }
    // Surface the probed ancestor in the diagnostic when it differs
    // from `dir`: the f_type we read came from `probe`, not from
    // `dir`, and an operator who reads only "dir lives on f_type X"
    // can be misled when X is the root filesystem's magic and the
    // intended cache mount simply does not exist yet.
    let probe_note = if probe == dir {
        String::new()
    } else {
        format!(
            " (no part of {dir:?} exists yet; the f_type was read from \
             ancestor {probe:?} — once {dir:?} is created on that same \
             filesystem the cache will inherit f_type=0x{fs_type:x}, \
             so create the intermediate mount first if you intended a \
             different filesystem)"
        )
    };
    bail!(
        "ktstr disk-template cache requires a btrfs or xfs filesystem \
         for FICLONE-based per-test fan-out; cache directory {dir:?} \
         lives on a filesystem whose statfs.f_type=0x{fs_type:x} (not \
         btrfs=0x{btrfs:x}, not xfs=0x{xfs:x}).{probe_note} Set \
         KTSTR_CACHE_DIR to a directory on a btrfs/xfs mount, or use \
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
/// on partial failure the staging directory is removed by the
/// caller (best-effort), and the live cache always sees either no
/// entry or a complete entry — never a half-written one.
///
/// # Failure cleanup
///
/// Two failure points after the staging directory is created can
/// strand intermediate state:
///
/// - The first `fs::rename(src_path, &staging_image)` failure
///   leaves an empty staging directory (`src_path` is untouched —
///   `rename(2)` does not modify the source on failure). The
///   staging dir is removed best-effort before propagating.
/// - The second `fs::rename(&staging, &final_dir)` failure leaves
///   the populated staging dir on disk (the first rename moved
///   `src_path` into `staging_image` and is irreversible). The
///   staging dir AND its contained image are removed best-effort
///   before propagating; without this cleanup the staging tree
///   would accumulate across retries inside the cache root, where
///   neither the per-key flock nor a future `ensure_template` peer
///   would garbage-collect it.
///
/// Cleanup errors are best-effort because the original error is the
/// dominant signal; a `remove_dir_all` failure on top of an already-
/// failing publish adds no actionable diagnostic for the caller.
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
    if src_buf.f_type != dest_buf.f_type || fsid_bytes(&src_buf) != fsid_bytes(&dest_buf) {
        bail!(
            "disk-template store_atomic: source {src_path:?} \
             (f_type=0x{src_type:x}) and cache root {root:?} \
             (f_type=0x{dest_type:x}) live on different filesystems. \
             rename(2) would return EXDEV. Stage the template image \
             on the cache filesystem before calling store_atomic.",
            src_type = src_buf.f_type as u64,
            dest_type = dest_buf.f_type as u64,
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
    // On failure src_path is unchanged (rename(2) is atomic), but
    // the empty staging directory is left behind — clean it up
    // before propagating so the cache root does not accumulate
    // empty `.tmp.<pid>` directories across retries.
    if let Err(e) = std::fs::rename(src_path, &staging_image) {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(e).with_context(|| format!("rename {src_path:?} -> {staging_image:?}"));
    }
    // Final atomic publish. On failure the staging directory now
    // contains `staging_image` (the first rename moved src_path into
    // it and is not reversible). Without the cleanup arm below the
    // populated staging dir would persist across retries — the
    // per-key flock prevents a peer from racing on the same key,
    // but the in-flight staging tree is not garbage-collected by
    // any other code path.
    if let Err(e) = std::fs::rename(&staging, &final_dir) {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(e).with_context(|| {
            format!("publish staging {staging:?} -> {final_dir:?} (cache key {key})",)
        });
    }
    Ok(final_dir.join(TEMPLATE_FILENAME))
}

/// Extract `f_fsid` as a fixed-size byte tuple for equality
/// comparisons between two `statfs` results. `libc::fsid_t` is
/// `__val: [c_int; 2]` across glibc, musl, and uClibc, but `__val`
/// is a private field — direct field access does not compile. The
/// bytewise read via `ptr::copy_nonoverlapping` is layout-opaque
/// and does not depend on which libc backend the build links
/// against. `fsid_t` also does not implement `PartialEq`, so the
/// fixed-width byte read also serves as the equality primitive
/// [`store_atomic`]'s cross-fs gate uses.
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
///
/// # Stale per-test debris and `EEXIST` diagnostics
///
/// `dest_path` is opened with `O_CREAT | O_EXCL` (via
/// [`OpenOptions::create_new`]), so the open returns `EEXIST` when
/// a regular file already sits at that path. Operators reading an
/// `EEXIST` here should NOT look at [`acquire_template_lock`] —
/// the per-key flock guards the cache *template* (read-only after
/// publish), not the per-test fan-out *dest*. The `EEXIST` surfaces
/// at the dest open, NOT at lock acquisition.
///
/// The realistic source of an `EEXIST` here is leftover staging
/// debris from a previous run that crashed before unlinking its
/// per-test fan-out file. The caller's tempfile name embeds a pid
/// (mkstemp-style); a prior ktstr peer that crashed mid-test (SIGKILL,
/// host reboot, OOM kill, panic before the per-test cleanup ran)
/// can leave its dest file in place. If the operating system later
/// reuses the same pid for a new ktstr process and that process
/// happens to generate a tempfile name colliding with the leaked
/// file's name, the `O_EXCL` open trips on the leftover. PID reuse
/// alone does not collide — the mkstemp randomization disambiguates
/// most cases — but the check is `O_EXCL` precisely to surface the
/// rare collision as a hard error rather than a silent overwrite.
///
/// **Triage checklist for an `EEXIST`-shaped failure here**:
///
/// 1. List the cache directory for orphan per-test files matching
///    the dest tempfile pattern. They are unlinked by ktstr after
///    each test; survivors indicate a crashed predecessor.
/// 2. Verify no live ktstr peer holds the file open
///    (`fuser`/`lsof`-equivalent against the path); a live owner
///    means the collision is real and the tempfile generator is the
///    bug, not the leftover.
/// 3. If no live owner, remove the leftover by hand and retry. The
///    cache template (under [`acquire_template_lock`]) is unaffected
///    by per-test fan-out failures — only the per-test dest file
///    needs cleanup.
///
/// The flock itself is irrelevant to this failure mode: a stale
/// flock on the per-key lockfile would cause [`ensure_template`] to
/// time out at [`acquire_template_lock`] long before any per-test
/// fan-out runs, surfacing as a holder-list bail with the lockfile
/// path — a visibly different diagnostic than the `EEXIST` here.
pub(crate) fn clone_to_per_test(src_path: &Path, dest_path: &Path) -> Result<File> {
    let src = OpenOptions::new()
        .read(true)
        .open(src_path)
        .with_context(|| format!("open template source {src_path:?}"))?;
    // O_CREAT | O_EXCL — surface stale leftover debris as EEXIST
    // instead of silently overwriting. See "Stale per-test debris
    // and EEXIST diagnostics" on this fn's doc comment.
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

/// Locate the host mkfs binary for `fs` so it can be packed into
/// the template-VM initramfs.
///
/// Resolves the userspace formatter name via
/// [`Filesystem::mkfs_binary_name`] and walks `PATH` (split on `:`)
/// for the first directory containing an executable of that name.
/// Returns `Ok(None)` when the variant requires no formatter
/// (`Filesystem::Raw`). Bails with an actionable error when a
/// formatter-requiring variant's binary is absent — the operator's
/// signal to install the corresponding distro package (e.g.
/// `btrfs-progs` for `Btrfs`) before using that filesystem.
///
/// The host binary is NOT exec'd at template-build time — it is
/// embedded into the template-VM initramfs and exec'd by guest init
/// inside the VM. The kernel inside the VM is the on-disk-format
/// authority; the host binary just provides the `mkfs.<fstype>`
/// userspace driver to drive the kernel into formatting.
pub(crate) fn locate_host_mkfs(fs: Filesystem) -> Result<Option<PathBuf>> {
    let Some(name) = fs.mkfs_binary_name() else {
        return Ok(None);
    };
    locate_host_binary(name, mkfs_package_hint(fs)).map(Some)
}

/// Distro package hint for the formatter binary returned by
/// [`Filesystem::mkfs_binary_name`]. Surfaced in
/// [`locate_host_binary`]'s "binary not found" diagnostic so an
/// operator hitting the missing-formatter case sees a concrete
/// install target.
///
/// The match is exhaustive on `Filesystem` so a future variant
/// that ships a `mkfs_binary_name` Some(_) without picking a
/// package hint here surfaces as a non-exhaustive-match build
/// error. The `Raw` arm is unreachable in practice — callers gate
/// on `mkfs_binary_name().is_some()` first — but the arm is
/// retained so the match stays exhaustive at the type level.
fn mkfs_package_hint(fs: Filesystem) -> &'static str {
    match fs {
        Filesystem::Btrfs => "btrfs-progs",
        Filesystem::Raw => "<none — Raw needs no formatter>",
    }
}

/// Locate a binary by name on the host `PATH`. Used by
/// [`locate_host_mkfs`] today; future filesystem variants
/// ([`Filesystem`] extensions) reuse the same machinery via
/// [`Filesystem::mkfs_binary_name`] for their respective mkfs
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
         equivalent) so the disk-template VM can format the requested \
         filesystem. PATH={path:?}",
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
    let staged = build_template_via_vm(fs, capacity_bytes, &root, &key)
        .with_context(|| format!("build disk template for {key}"))?;
    // store_atomic moves `staged` into the cache via rename. On
    // failure (cross-fs detection, staging-dir creation, the rename
    // itself) `staged` is stranded: the per-key flock prevents a
    // peer from observing a partial cache entry, but the in-flight
    // file persists in the cache root until the next build. Unlink
    // before propagating so retries find a clean root. Best-effort
    // because the store_atomic error is the dominant signal — a
    // remove_file failure here adds no actionable diagnostic.
    let final_path = match store_atomic(&key, &staged) {
        Ok(p) => p,
        Err(e) => {
            let _ = std::fs::remove_file(&staged);
            return Err(e).with_context(|| format!("install disk template {key}"));
        }
    };
    Ok(final_path)
}

/// Compose the staging-image path for a `(cache_key, pid)` pair.
///
/// The filename includes BOTH the cache key and the pid because the
/// per-key flock only serialises peers within a single key — the
/// same process holds different per-key flocks concurrently across
/// distinct `(fs, capacity)` pairs (cross-key concurrency is
/// permitted). Without the key in the filename, two simultaneous
/// in-flight builds for `btrfs-256m` and `btrfs-1024m` from the
/// same pid would collide on `template.img.in-flight.<pid>` — the
/// second open would truncate the first's image while it boots,
/// corrupting the template the first build is formatting. Including
/// the key makes the filename unique per `(key, pid)`.
///
/// Pulled out as a free fn so the uniqueness invariant has a
/// dedicated test (`staging_image_path_is_unique_per_key_and_pid`).
fn staging_image_path(cache_root: &Path, cache_key: &str, pid: u32) -> PathBuf {
    cache_root.join(format!("template.img.in-flight.{cache_key}.{pid}"))
}

/// Materialise an empty sparse image at `staging_path` of exactly
/// `capacity_bytes`.
///
/// Removes any same-path leftover from a prior crashed run (the
/// per-key flock guarantees no live peer holds it; same-pid debris
/// is the only realistic source). On `set_len` failure (the
/// specific errno depends on the cache filesystem — common
/// examples include ENOSPC and EFBIG) the empty file is
/// unlinked best-effort before propagating; without that cleanup
/// a 0-byte staging image would accumulate in the cache root
/// across retries, mirroring the leak-cleanup behaviour at the
/// VM-boot/run failure sites farther down. The file descriptor is
/// dropped before the unlink as defense-in-depth: local
/// filesystems (btrfs/ext4/xfs) propagate truncate synchronously
/// but FUSE/NFS backings can delay until close.
///
/// Pulled out as a free fn so the cleanup arm has a dedicated
/// test (`create_and_size_staging_image_cleans_up_on_set_len_failure`)
/// that does not require booting a VM. Production callsites in
/// [`build_template_via_vm`] reach this helper via the standard
/// resource-bootstrap path.
fn create_and_size_staging_image(staging_path: &Path, capacity_bytes: u64) -> Result<()> {
    if staging_path.exists() {
        std::fs::remove_file(staging_path).with_context(|| {
            format!("remove leftover staging image {staging_path:?} before rebuild")
        })?;
    }
    let staging_file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(staging_path)
        .with_context(|| format!("create staging image {staging_path:?}"))?;
    if let Err(e) = staging_file.set_len(capacity_bytes) {
        drop(staging_file);
        let _ = std::fs::remove_file(staging_path);
        return Err(e).with_context(|| {
            format!(
                "set staging image length to {capacity_bytes} bytes \
                 ({staging_path:?})"
            )
        });
    }
    // Drop the host-side fd before returning; the VM opens its own
    // RW fd via `template_staging_image`, and host writes through a
    // stale fd would race the guest's mkfs.
    drop(staging_file);
    Ok(())
}

/// Build a fresh template image by booting a one-shot template VM.
///
/// Steps:
/// 1. Materialise a sparse `template.img.in-flight.<key>.<pid>`
///    of `capacity_bytes` under `cache_root` so the file shares
///    the cache filesystem ([`store_atomic`]'s rename requires
///    same-fs source/dest). The `<key>` qualifier disambiguates
///    cross-key concurrent builds in the same process; the per-key
///    flock already serialises within a single key.
/// 2. Locate `mkfs.<fstype>` on the host PATH and pack it into the
///    template-VM initramfs at `bin/mkfs.<fstype>`. The kernel
///    inside the VM is the on-disk-format authority — the host's
///    `mkfs` binary just provides the userspace driver that runs
///    against `/dev/vda` inside the guest.
/// 3. Boot a [`crate::vmm::KtstrVm`] with the sparse image attached
///    via [`crate::vmm::KtstrVmBuilder::template_staging_image`],
///    which short-circuits the per-test backing-file branches in
///    [`crate::vmm::KtstrVm::init_virtio_blk`] so the template-build
///    VM cannot recursively re-enter [`ensure_template`] for its
///    own `(fs, capacity_bytes)` key. Cmdline carries
///    `KTSTR_MODE=disk_template`; the guest dispatch at
///    [`crate::vmm::rust_init::run_disk_template_mode`] execs the
///    embedded `bin/<mkfs_binary_name>` against `/dev/vda`
///    (currently `mkfs.btrfs` for `Filesystem::Btrfs` per
///    [`Filesystem::mkfs_binary_name`]) and reboots.
/// 4. After clean exit (`VmResult::success` and `exit_code == 0`),
///    return the staging path for [`store_atomic`] to rename into
///    the cache. Non-zero exit, timeout, or run failure unlinks
///    the staging file and bails.
///
/// Filesystem variants whose [`Filesystem::mkfs_binary_name`]
/// returns `None` (currently `Filesystem::Raw`) are unreachable on
/// this path: [`ensure_template`] only invokes this driver from the
/// gated formatting arm in [`crate::vmm::KtstrVm::init_virtio_blk`].
/// Such an argument means a caller bypassed that gate; bail with an
/// actionable error rather than build an unformatted template
/// (which would be a no-op).
fn build_template_via_vm(
    fs: Filesystem,
    capacity_bytes: u64,
    cache_root: &Path,
    cache_key: &str,
) -> Result<PathBuf> {
    // Resolve the mkfs binary for `fs` via the typed accessor so
    // the exhaustive match forces a future `Filesystem` variant to
    // declare its formatter at compile time. `locate_host_mkfs`
    // returns `Ok(None)` when the variant has no formatter
    // (currently only `Filesystem::Raw`); that case is unreachable
    // on this path because [`ensure_template`] gates on
    // [`Filesystem::mkfs_binary_name`] before calling here. A
    // `None` result means a caller bypassed the gate in
    // [`crate::vmm::KtstrVm::init_virtio_blk`]; reject with an
    // actionable diagnostic.
    let mkfs = locate_host_mkfs(fs)?.ok_or_else(|| {
        anyhow!(
            "build_template_via_vm called with Filesystem::{fs:?} — \
             this filesystem variant has no userspace formatter \
             (mkfs_binary_name() returned None) so there is no \
             template image to build. ensure_template should only \
             invoke this path for filesystem variants that require \
             pre-formatting; this call indicates a bypass of the gate \
             in init_virtio_blk."
        )
    })?;
    // mkfs_name comes from the same accessor; reuse it for the
    // in-archive path so the host-PATH lookup name and the in-
    // archive path stay in lockstep without a parallel match arm.
    // Safe to unwrap because `locate_host_mkfs` only returns Some
    // when `mkfs_binary_name()` was Some.
    let mkfs_name = fs
        .mkfs_binary_name()
        .expect("locate_host_mkfs returned Some only when mkfs_binary_name is Some");

    // Resolve a kernel image so the template-build VM can boot.
    // Reuses the same KTSTR_KERNEL / cache / sysroot cascade the
    // test framework uses, so an operator who set KTSTR_KERNEL for
    // tests gets the same kernel for the template build.
    let kernel = crate::find_kernel()
        .context("locate kernel image for template-build VM")?
        .ok_or_else(|| {
            anyhow!(
                "no kernel image found for template-build VM. {}",
                crate::KTSTR_KERNEL_HINT,
            )
        })?;

    // Stage the sparse image under the cache root so the eventual
    // rename(2) into place is on the same filesystem.
    std::fs::create_dir_all(cache_root)
        .with_context(|| format!("create cache root {cache_root:?} for staging image"))?;
    let staging_path = staging_image_path(cache_root, cache_key, std::process::id());
    create_and_size_staging_image(&staging_path, capacity_bytes)?;

    // Build the template VM. The `template_staging_image` setter
    // makes init_virtio_blk open `staging_path` directly, bypassing
    // BOTH the Raw tempfile and the Btrfs ensure_template branches —
    // this is what breaks the recursion that would otherwise occur
    // (a Btrfs disk inside a build-time VM would re-call
    // ensure_template on the same key while we already hold the
    // per-key flock above).
    //
    // The mkfs binary rides through `include_files` packed at
    // `bin/<name>` so the guest's KTSTR_MODE=disk_template
    // dispatch can spawn `/bin/<name>` against `/dev/vda`. The
    // disk attached here uses Filesystem::Raw — the guest sees an
    // unformatted device exactly as expected for the staging
    // image (the whole point of this VM is to format it).
    //
    // `mkfs_name` was non-None at the function-entry gate above
    // (the only `None` variant — Raw — bails before reaching this
    // point), so reusing it for the archive path is sound. Driving
    // the archive path off the same accessor keeps the host-PATH
    // lookup name and the in-archive path in lockstep without a
    // parallel match arm to drift.
    let mkfs_archive_path = format!("bin/{mkfs_name}");
    let disk = crate::vmm::disk_config::DiskConfig::default()
        .capacity_mb((capacity_bytes / (1024 * 1024)) as u32)
        .filesystem(Filesystem::Raw);
    // VM-level timeout for the template build. 120s = 2 minutes,
    // chosen as the inner bound that lets the outer
    // [`TEMPLATE_LOCK_TIMEOUT`] (10 minutes) catch stuck peers
    // without firing on the legitimate worst-case build:
    //
    // - Kernel boot inside the VM: ~1-15s once the kernel image is
    //   already cached on the host (first-run cold-page-cache boot
    //   can stretch toward 30s on slow storage but is dominated by
    //   host-side disk reads, NOT this in-VM timeout).
    // - `mkfs.<fstype>` against `/dev/vda` inside the guest:
    //   ~1-60s for 256 MiB-2 GiB capacities on a backing image that
    //   itself lives on tmpfs/btrfs/xfs. The host backing-file IO
    //   cost (sparse-file zero-fill on first write) is included in
    //   this budget.
    // - VM shutdown: sub-second.
    //
    // 120s sits above the expected worst-case build cost
    // (kernel boot + mkfs + shutdown summed at the upper end of
    // the per-stage ranges above), which lets `mkfs` finish even
    // when KVM contention or a briefly-loaded host slows the
    // guest. If a build genuinely hangs (e.g. mkfs deadlocked,
    // kernel oops), the 120s VM timeout fires inside `vm.run()`,
    // the caller unlinks the staging image, and `ensure_template`
    // propagates the failure up — no peer holds the per-key flock
    // past this point.
    let build_result = crate::vmm::KtstrVm::builder()
        .kernel(kernel)
        .topology(1, 1, 1, 1)
        .memory_mb(256)
        .timeout(std::time::Duration::from_secs(120))
        .cmdline("KTSTR_MODE=disk_template")
        .disk(disk)
        .template_staging_image(staging_path.clone())
        .include_files(vec![(mkfs_archive_path, mkfs)])
        .busybox(true)
        .build();
    // .build() can fail for host-resource reasons (KVM ioctl
    // ENOMEM, sysfs unreadable, hugepage planning) AFTER the
    // staging image is already on disk. Without the cleanup arm
    // below, those failures leak the staging file across retries
    // — same pattern as the .run() error handler farther down,
    // but earlier in the lifecycle.
    let vm = match build_result {
        Ok(vm) => vm,
        Err(e) => {
            let _ = std::fs::remove_file(&staging_path);
            return Err(e).with_context(|| {
                format!("build template-VM for {fs:?} capacity_bytes={capacity_bytes}")
            });
        }
    };
    let result = vm.run().with_context(|| {
        format!("run template-build VM for {fs:?} capacity_bytes={capacity_bytes}")
    });
    let result = match result {
        Ok(r) => r,
        Err(e) => {
            // Best-effort cleanup of the staging image. The
            // template-build error itself is the dominant signal
            // and any remove_file error here is a tertiary
            // problem the caller cannot meaningfully act on.
            let _ = std::fs::remove_file(&staging_path);
            return Err(e);
        }
    };
    if result.timed_out || result.exit_code != 0 || !result.success {
        let _ = std::fs::remove_file(&staging_path);
        bail!(
            "template-build VM did not complete cleanly \
             (timed_out={}, exit_code={}, success={}). \
             Tail of guest stderr: {}",
            result.timed_out,
            result.exit_code,
            result.success,
            tail_lines(&result.stderr, 20),
        );
    }
    Ok(staging_path)
}

/// Sweep stale staging debris out of the disk-template cache root.
///
/// Two debris shapes accumulate when a template-build peer dies
/// before its [`store_atomic`] rename completes:
///
/// 1. **`template.img.in-flight.<cache_key>.<pid>`** — sparse
///    staging images created by [`create_and_size_staging_image`]
///    when [`build_template_via_vm`] runs. Normally unlinked at
///    the failure-cleanup arms inside that function, AND moved
///    into a `.tmp.<pid>/` directory by `store_atomic` on success.
///    A SIGKILL between size-up and store_atomic leaks the file.
/// 2. **`<cache_key>.tmp.<pid>/`** — staging directories created
///    by [`store_atomic`] for the rename-into-place dance.
///    Normally renamed onto the final `<cache_key>/` directory at
///    the end of `store_atomic`. A SIGKILL during the
///    src→staging_image rename or the staging→final_dir rename
///    leaves the populated tmpdir on disk.
///
/// Both shapes embed the originating peer's pid in the filename.
/// The sweep parses that pid and probes liveness via
/// `kill(pid, None)` (rust-side: [`nix::sys::signal::kill`] with
/// `Signal::None`). The kernel returns:
/// - `Ok(())` — pid is live AND in-policy for our uid (the signal
///   COULD have been delivered). Debris is owned by a peer that
///   may still publish; leave alone.
/// - `Err(ESRCH)` — pid does not exist. Debris is safe to remove.
/// - `Err(EPERM)` — pid is live but owned by a different uid.
///   Not ours to clean up; leave alone.
/// - any other errno — treat as live and skip; false negatives
///   (debris left on disk) are recoverable, false positives
///   (deleting live state) are not.
///
/// Mirrors [`crate::cache::clean_orphaned_tmp_dirs`] in
/// `src/cache.rs` — the disk-template cache and the kernel-image
/// cache use the same pid-in-suffix + ESRCH-probe contract for
/// cross-process cleanup. The two are independent because their
/// debris namespaces don't overlap (kernel cache uses `.tmp-`
/// prefix, disk-template cache uses `.tmp.` infix on the
/// directories and a `template.img.in-flight.` prefix on the
/// staging images).
///
/// Returns the count of debris entries removed. Errors during
/// individual `remove_dir_all` / `remove_file` calls are logged
/// at `warn` and the sweep continues — operator visibility into
/// "this entry could not be cleaned" beats abandoning the rest of
/// the sweep on the first failure.
///
/// Refuses to descend into the `.locks/` subdirectory (the only
/// non-debris namespace inside the cache root); the prefix filter
/// excludes it via the `template.img.in-flight.` and `*.tmp.*`
/// pattern match. Published cache entries (`<cache_key>/`) are
/// left untouched — they have no pid suffix and don't match either
/// debris shape.
pub fn clean_orphaned_tmp_dirs(cache_root: &Path) -> Result<usize> {
    if !cache_root.is_dir() {
        // Cache root not yet materialised — nothing to sweep.
        // Mirrors the early-return at the head of
        // [`crate::cache::clean_orphaned_tmp_dirs`].
        return Ok(0);
    }
    let read_dir = match std::fs::read_dir(cache_root) {
        Ok(rd) => rd,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(0),
        Err(e) => {
            return Err(anyhow!("read cache root {cache_root:?}: {e}"));
        }
    };
    let mut removed: usize = 0;
    for dir_entry in read_dir {
        let dir_entry = match dir_entry {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(
                    err = %format!("{e:#}"),
                    "skip unreadable disk-template cache root entry",
                );
                continue;
            }
        };
        let name = match dir_entry.file_name().into_string() {
            Ok(n) => n,
            // Non-UTF-8 filename — neither of our patterns can
            // match (both contain only ASCII), so it's foreign
            // debris (not ours to touch).
            Err(_) => continue,
        };
        // Identify which debris shape this entry is, and extract
        // the pid suffix. Patterns:
        //
        //   - `template.img.in-flight.<cache_key>.<pid>` — staging
        //     image (see [`staging_image_path`]).
        //   - `<cache_key>.tmp.<pid>` — staging directory (see
        //     [`store_atomic`]).
        //
        // Anything else (notably the `.locks/` subdirectory and
        // the published `<cache_key>/` entries) is skipped.
        let pid_str = if let Some(rest) = name.strip_prefix("template.img.in-flight.") {
            // The trailing `.<pid>` is what we need; key may
            // itself contain `-` / `.` so we split at the LAST
            // `.` token.
            match rest.rsplit_once('.') {
                Some((_, suffix)) if !suffix.is_empty() => suffix,
                _ => continue,
            }
        } else if name.contains(".tmp.") {
            // `<cache_key>.tmp.<pid>` — the pid is everything
            // after the LAST `.tmp.`.
            match name.rsplit_once(".tmp.") {
                Some((_, suffix)) if !suffix.is_empty() => suffix,
                _ => continue,
            }
        } else {
            continue;
        };
        let pid: i32 = match pid_str.parse() {
            Ok(p) => p,
            Err(_) => continue,
        };
        // Reject non-positive pids defensively — `kill(0, ...)`
        // probes the caller's own process group, `kill(-N, ...)`
        // probes process group N. Same hardening as
        // [`crate::cache::clean_orphaned_tmp_dirs`].
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
        // The two debris shapes need different removers. Probe
        // via `metadata` rather than re-checking the prefix —
        // the prefix already classified the path; metadata picks
        // the right `remove_*` arm.
        let result = match dir_entry.file_type() {
            Ok(ft) if ft.is_dir() => std::fs::remove_dir_all(&path),
            Ok(_) => std::fs::remove_file(&path),
            Err(e) => {
                tracing::warn!(
                    err = %format!("{e:#}"),
                    path = %path.display(),
                    "skip disk-template cache entry; \
                     file_type() failed",
                );
                continue;
            }
        };
        match result {
            Ok(()) => {
                tracing::info!(
                    path = %path.display(),
                    orphan_pid = pid,
                    "cleaned orphaned disk-template debris from \
                     prior crashed process",
                );
                removed += 1;
            }
            Err(e) => {
                tracing::warn!(
                    err = %format!("{e:#}"),
                    path = %path.display(),
                    "failed to remove orphaned disk-template debris; \
                     leaving in place",
                );
            }
        }
    }
    Ok(removed)
}

/// Remove every published disk-template cache entry, returning the
/// count of entries actually removed.
///
/// Mirrors [`crate::cache::CacheDir::clean_all`] in `src/cache.rs`.
/// Operators run this after a host `mkfs.<fstype>` upgrade that
/// invalidates the on-disk format the cached templates were
/// formatted with: today the cache key is `(fs_tag, capacity_mib)`
/// only — it does NOT capture the mkfs binary version — so a
/// post-upgrade run would silently reuse stale templates whose
/// internal format the new kernel may reject. Until the cache key
/// learns the mkfs version (tracked separately in #566), `clean_all`
/// is the operator-driven escape hatch.
///
/// # Concurrency
///
/// Each entry's per-key lockfile is acquired non-blocking in
/// `LOCK_EX` mode via [`crate::flock::try_flock`]. An entry whose
/// lock is held by a live peer (an active test run mid-FICLONE,
/// or a concurrent template build that finished its rename but is
/// still inside the lock holder's critical section) is skipped
/// rather than removed — the holder is using the entry; deleting
/// it would yank the template out from under a live `clone_to_per_test`.
///
/// The flock is held across the `remove_dir_all` so a peer that
/// blocks on the lock while we're removing observes a clean
/// "entry gone, rebuild from scratch" sequence: their post-lock
/// `lookup()` returns `None` and `ensure_template` proceeds to
/// rebuild. Without holding the lock during removal, a peer that
/// raced through `acquire_template_lock` → `lookup` between our
/// lock-release and our `remove_dir_all` would see the template
/// path, `clone_to_per_test` would race against the rmtree, and
/// either side could win unpredictably.
///
/// The lockfile inode itself is NOT removed — other peers may
/// have it open, and dropping the file while peers wait on it
/// would orphan their fds. Lockfile inodes are sized at a few
/// bytes each and accumulate at the rate of distinct `(fs,
/// capacity)` keys; leaving them is bounded growth, not a leak.
///
/// # Sweeps debris first
///
/// Calls [`clean_orphaned_tmp_dirs`] before walking published
/// entries so a rebuilding peer that hits the freshly-empty cache
/// doesn't trip on stale staging debris from a crashed predecessor
/// during its first `store_atomic`.
///
/// # What gets skipped
///
/// - The `.locks/` subdirectory (lockfile namespace).
/// - Any cache entry whose lockfile is currently held by a live
///   peer (logged at `info` so the operator sees what was kept).
/// - Any cache entry whose `template.img` is missing (corrupt /
///   half-installed) — those are removed regardless of lock state
///   because they can't serve a `clone_to_per_test` and waste
///   inode space.
/// - Non-UTF-8 entry names (foreign — not produced by ktstr).
/// - Files at the cache root (only directories are cache entries;
///   `clean_orphaned_tmp_dirs` already swept the staging-image
///   files before we got here).
pub fn clean_all() -> Result<usize> {
    let root = cache_root()?;
    if !root.is_dir() {
        return Ok(0);
    }
    // Sweep staging debris first so a peer that re-acquires a
    // freshly-emptied cache doesn't trip on leftover .tmp.<pid>
    // / template.img.in-flight.* files from a crashed predecessor.
    // The result is logged but not fed into the return count —
    // `clean_all` reports published-entry removals only, matching
    // the [`crate::cache::CacheDir::clean_all`] contract.
    let _debris = clean_orphaned_tmp_dirs(&root)?;
    // Ensure the lockfile parent directory exists. `try_flock` opens
    // the lockfile with `O_CREAT`, but the open fails with `ENOENT`
    // when the parent `.locks/` subdirectory is absent — which is
    // the steady state on a freshly-published cache that was never
    // touched by `acquire_template_lock` (e.g. a cache populated by
    // an earlier ktstr run that crashed between `store_atomic` and
    // first lock acquire). Without this, `clean_all` would silently
    // skip every published entry on such a cache: the `try_flock`
    // call returns `Err(ENOENT)`, the loop's tracing-warn branch
    // logs and `continue`s, and the operator-driven `clean_all`
    // becomes a no-op. `create_dir_all` is idempotent — it's a
    // no-op when `.locks/` already exists, so this also covers the
    // mixed-cache case (some keys lock-touched, others not).
    let lock_dir = root.join(LOCK_DIR_NAME);
    std::fs::create_dir_all(&lock_dir).with_context(|| {
        format!(
            "create disk-template lock subdirectory {} for clean_all",
            lock_dir.display(),
        )
    })?;
    let read_dir = match std::fs::read_dir(&root) {
        Ok(rd) => rd,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(0),
        Err(e) => {
            return Err(anyhow!("read cache root {root:?}: {e}"));
        }
    };
    let mut removed: usize = 0;
    for dir_entry in read_dir {
        let dir_entry = match dir_entry {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(
                    err = %format!("{e:#}"),
                    "skip unreadable disk-template cache root entry \
                     during clean_all",
                );
                continue;
            }
        };
        let file_type = match dir_entry.file_type() {
            Ok(ft) => ft,
            Err(e) => {
                tracing::warn!(
                    err = %format!("{e:#}"),
                    path = %dir_entry.path().display(),
                    "skip disk-template entry; file_type() failed",
                );
                continue;
            }
        };
        // Only published cache entries are directories at the
        // cache root. Files are staging images already swept by
        // clean_orphaned_tmp_dirs above; skip them here so we
        // don't double-account.
        if !file_type.is_dir() {
            continue;
        }
        let name = match dir_entry.file_name().into_string() {
            Ok(n) => n,
            Err(_) => continue,
        };
        // Skip the lockfile subdirectory — it's not a cache
        // entry. Its pathname is fixed by [`LOCK_DIR_NAME`].
        if name == LOCK_DIR_NAME {
            continue;
        }
        // Skip staging directories left by store_atomic (handled
        // by clean_orphaned_tmp_dirs above; defense-in-depth in
        // case the sweep returned early on a syscall error).
        if name.contains(".tmp.") {
            continue;
        }
        let entry_path = dir_entry.path();
        // Probe via try_flock that no live peer is currently
        // using this entry. The lockfile is acquired non-blocking
        // in LOCK_EX mode: success means there are zero readers
        // AND zero writers on this key. Failure (Ok(None)) means
        // a peer holds the lock — skip this entry.
        let lock_path = match lock_path_for_key(&name) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(
                    err = %format!("{e:#}"),
                    cache_key = %name,
                    "skip disk-template entry; lock_path resolution \
                     failed",
                );
                continue;
            }
        };
        let lock_fd = match try_flock(&lock_path, FlockMode::Exclusive) {
            Ok(Some(fd)) => fd,
            Ok(None) => {
                // A live peer holds the lock. Skip — its work
                // would race a `remove_dir_all` mid-clone.
                tracing::info!(
                    cache_key = %name,
                    lockfile = %lock_path.display(),
                    "skip disk-template entry during clean_all — \
                     locked by live peer",
                );
                continue;
            }
            Err(e) => {
                tracing::warn!(
                    err = %format!("{e:#}"),
                    cache_key = %name,
                    "skip disk-template entry; try_flock failed",
                );
                continue;
            }
        };
        // Lock acquired. Perform the removal while holding the
        // lock so any peer that subsequently blocks on this
        // lockfile observes "no entry, rebuild from scratch" via
        // their re-check after acquire (see [`ensure_template`]).
        match std::fs::remove_dir_all(&entry_path) {
            Ok(()) => {
                tracing::info!(
                    cache_key = %name,
                    path = %entry_path.display(),
                    "removed disk-template cache entry during clean_all",
                );
                removed += 1;
            }
            Err(e) => {
                tracing::warn!(
                    err = %format!("{e:#}"),
                    cache_key = %name,
                    path = %entry_path.display(),
                    "failed to remove disk-template cache entry \
                     during clean_all; leaving in place",
                );
            }
        }
        // OwnedFd `lock_fd` drops here, releasing the per-key
        // flock. The lockfile inode at `lock_path` stays — see
        // the doc comment "the lockfile inode itself is NOT
        // removed".
        drop(lock_fd);
    }
    Ok(removed)
}

/// Extract the last `n` lines of `text` for an error context.
/// Used by [`build_template_via_vm`] to surface the trailing guest
/// stderr — typically the `mkfs` failure message — without
/// dumping the whole transcript into the bail message.
fn tail_lines(text: &str, n: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let start = lines.len().saturating_sub(n);
    lines[start..].join("\n")
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
        // Isolate from operator state: KTSTR_CACHE_DIR / XDG_CACHE_HOME
        // / $HOME bleed into template_path_for_key via cache_root().
        let tmp = tempfile::tempdir().expect("create tempdir");
        let _guard =
            crate::test_support::test_helpers::EnvVarGuard::set("KTSTR_CACHE_DIR", tmp.path());
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
        let _guard =
            crate::test_support::test_helpers::EnvVarGuard::set("KTSTR_CACHE_DIR", tmp.path());
        let result = lookup("missing-key").expect("lookup must not error on miss");
        assert!(result.is_none());
    }

    #[test]
    fn store_atomic_publishes_then_lookup_finds() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let _guard =
            crate::test_support::test_helpers::EnvVarGuard::set("KTSTR_CACHE_DIR", tmp.path());
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
        let _guard =
            crate::test_support::test_helpers::EnvVarGuard::set("KTSTR_CACHE_DIR", tmp.path());
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
        let _guard = crate::test_support::test_helpers::EnvVarGuard::set("PATH", tmp.path());
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
    fn build_template_via_vm_rejects_raw_filesystem() {
        // [`build_template_via_vm`] is only supposed to be invoked
        // from filesystem variants that require pre-formatting. A
        // `Filesystem::Raw` argument means a caller bypassed the
        // gate in [`crate::vmm::KtstrVm::init_virtio_blk`] and would
        // produce a no-op template (Raw disks have no on-disk
        // format). Pin the rejection so that bypass surfaces as a
        // bail with a hint at the offending caller rather than as a
        // silent empty template.
        let tmp = tempfile::tempdir().expect("create tempdir");
        let _guard =
            crate::test_support::test_helpers::EnvVarGuard::set("KTSTR_CACHE_DIR", tmp.path());
        let err = build_template_via_vm(Filesystem::Raw, 256 * 1024 * 1024, tmp.path(), "raw-256m")
            .expect_err("Raw must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("Filesystem::Raw"),
            "error must name the rejected variant: {msg}",
        );
        assert!(
            msg.contains("init_virtio_blk"),
            "error must name the gate location for the operator: {msg}",
        );
    }

    #[test]
    fn verify_cache_dir_walks_up_to_existing_ancestor() {
        // A non-existent cache root must still produce a usable
        // statfs result by walking up. Anchor the missing path under
        // a per-test tempdir so parallel runs do not collide on a
        // shared system path; the tempdir itself exists and walking
        // up from `<tempdir>/nonexistent/sub/dir` reaches it on the
        // first ancestor probe.
        let tmp = tempfile::tempdir().expect("create tempdir");
        let nonexistent = tmp.path().join("nonexistent/sub/dir");
        // The result depends on the tempdir's filesystem; this test
        // only pins that the helper does not panic and either
        // returns Ok (btrfs/xfs tempdir) or a fs-magic-named error
        // (anything else).
        match verify_cache_dir_supports_reflink(&nonexistent) {
            Ok(()) => { /* tempdir lives on btrfs/xfs */ }
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("statfs.f_type") || msg.contains("FICLONE"),
                    "unexpected error wording: {msg}",
                );
            }
        }
    }

    /// When the walk-up lands on an ancestor (`probe != dir`), the
    /// bail diagnostic appends a `probe_note` that names the probed
    /// ancestor explicitly so the operator can tell the f_type came
    /// from an ancestor rather than `dir` itself. Pins the
    /// conditional interpolation: a regression that drops
    /// `{probe_note}` from the bail string would silently strip the
    /// "(no part of {dir:?} exists yet; ... ancestor {probe:?} ...)"
    /// guidance, leaving operators with the misleading
    /// "cache directory X lives on f_type Y" wording even when Y
    /// came from a probed ancestor.
    ///
    /// Skipped when the tempdir lives on btrfs/xfs — the helper
    /// returns Ok and there is no diagnostic to inspect. Most
    /// CI runners use tmpfs or ext4 for `TMPDIR`, so the
    /// assertion fires there.
    #[test]
    fn verify_cache_dir_probe_note_fires_when_probe_differs_from_dir() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let nonexistent = tmp.path().join("nonexistent/sub/dir");
        match verify_cache_dir_supports_reflink(&nonexistent) {
            Ok(()) => {
                // tempdir lives on btrfs/xfs — no diagnostic emitted,
                // skip the probe_note assertion.
            }
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("ancestor") && msg.contains("no part of"),
                    "walk-up diagnostic must surface the probed \
                     ancestor when probe != dir; got: {msg}",
                );
            }
        }
    }

    /// When `dir` itself exists (`probe == dir`), the bail diagnostic
    /// MUST NOT include the probe_note text — that text is
    /// conditional on the walk-up landing on an ancestor. Pins the
    /// `probe == dir` branch of the conditional interpolation: a
    /// regression that always emits the probe_note (e.g. drops the
    /// `if probe == dir` guard) would leak the misleading "no part
    /// of dir exists yet" wording on every non-btrfs/xfs probe.
    ///
    /// Skipped when the tempdir lives on btrfs/xfs — the helper
    /// returns Ok and there is no diagnostic to inspect.
    #[test]
    fn verify_cache_dir_probe_note_absent_when_probe_equals_dir() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        match verify_cache_dir_supports_reflink(tmp.path()) {
            Ok(()) => {
                // tempdir lives on btrfs/xfs — no diagnostic emitted.
            }
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    !msg.contains("ancestor") && !msg.contains("no part of"),
                    "probe == dir branch must NOT emit the probe_note \
                     text; got: {msg}",
                );
                // Sanity: the rest of the diagnostic still names the
                // f_type so the operator gets actionable guidance.
                assert!(
                    msg.contains("statfs.f_type") || msg.contains("FICLONE"),
                    "diagnostic must still name the f_type; got: {msg}",
                );
            }
        }
    }

    /// `Path::exists` follows symlinks, so a dangling symlink
    /// probes as missing and the walk-up moves to the symlink
    /// container's parent rather than the (nonexistent) target's
    /// parent. Pin the documented behaviour at
    /// `verify_cache_dir_supports_reflink`'s "Symlink behaviour"
    /// paragraph: the diagnostic must reference the tempdir's
    /// f_type (the container, which exists) rather than failing on
    /// the broken symlink.
    ///
    /// A regression that switches `Path::exists` to
    /// `Path::try_exists` would surface here: try_exists returns
    /// `Err` on a broken symlink, breaking the walk-up loop
    /// invariant.
    ///
    /// Linux-only: requires `std::os::unix::fs::symlink`. Skipped
    /// when the tempdir lives on btrfs/xfs (helper returns Ok by
    /// walking up to a reflink-capable filesystem, which is the
    /// correct outcome).
    #[cfg(target_os = "linux")]
    #[test]
    fn verify_cache_dir_walks_through_dangling_symlink() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let symlink_path = tmp.path().join("dangling");
        // Target does not exist; dangling symlink lands in the
        // tempdir.
        std::os::unix::fs::symlink("/nonexistent-symlink-target-9242", &symlink_path)
            .expect("create dangling symlink");
        // Probing a path under the dangling symlink: walk-up
        // ascends to symlink_path → tmp.path() (the symlink's
        // container). The symlink target's parent is never
        // consulted.
        let probe_path = symlink_path.join("sub");
        match verify_cache_dir_supports_reflink(&probe_path) {
            Ok(()) => {
                // tempdir lives on btrfs/xfs — helper returned Ok
                // by walking up to a reflink-capable filesystem,
                // which is the correct outcome.
            }
            Err(e) => {
                let msg = e.to_string();
                // The diagnostic must reference the f_type of the
                // walked-up ancestor (tempdir's filesystem) rather
                // than failing on the dangling symlink. The error
                // wording always names the f_type magic, regardless
                // of whether the probed ancestor is the original
                // dir or an ancestor.
                assert!(
                    msg.contains("statfs.f_type") || msg.contains("FICLONE"),
                    "symlink walk-up must produce an f_type-named \
                     diagnostic, not a symlink-resolution error; got: {msg}",
                );
            }
        }
    }

    /// Cross-key concurrency invariant: two distinct cache keys held
    /// by the same pid produce distinct staging-image paths. Without
    /// the cache_key qualifier in the filename, the same process
    /// concurrently building `btrfs-256m` and `btrfs-1024m` would
    /// collide on `template.img.in-flight.<pid>` — the second open
    /// would truncate the first's image while it boots, corrupting
    /// the template the first build is formatting. Pin the
    /// uniqueness contract here so a regression that drops the
    /// cache_key from [`staging_image_path`] surfaces immediately
    /// rather than as a flaky cross-key test.
    #[test]
    fn staging_image_path_is_unique_per_key_and_pid() {
        let cache_root = std::path::Path::new("/tmp/ktstr-fake-cache-root");
        let pid = 12_345u32;
        let p_256 = staging_image_path(cache_root, "btrfs-256m", pid);
        let p_1024 = staging_image_path(cache_root, "btrfs-1024m", pid);
        // Same pid, different keys → different paths.
        assert_ne!(
            p_256, p_1024,
            "cache_key qualifier missing from staging-image path: \
             distinct keys collided",
        );
        // Both paths embed the cache_key and the pid verbatim.
        assert!(
            p_256
                .to_string_lossy()
                .contains("template.img.in-flight.btrfs-256m.12345"),
            "256m staging path missing key/pid token: {p_256:?}",
        );
        assert!(
            p_1024
                .to_string_lossy()
                .contains("template.img.in-flight.btrfs-1024m.12345"),
            "1024m staging path missing key/pid token: {p_1024:?}",
        );
        // Same key, different pids → different paths (per-pid debris
        // never collides with a live peer's staging file).
        let p_256_other_pid = staging_image_path(cache_root, "btrfs-256m", 67_890);
        assert_ne!(p_256, p_256_other_pid);

        // Idempotence: same input → same output. Defends against a
        // future regression that introduces nondeterminism (e.g.
        // reads `process::id()` internally instead of taking pid as
        // an argument, or appends a randomised suffix). The function
        // must be a pure mapping from `(cache_root, key, pid)` to
        // `PathBuf` so the per-key flock and the staging-image path
        // can coordinate without surprise.
        assert_eq!(
            p_256,
            staging_image_path(cache_root, "btrfs-256m", pid),
            "staging_image_path must be a pure function of its inputs",
        );
    }

    /// Cleanup contract for the [`create_and_size_staging_image`]
    /// helper: when `set_len` fails (ENOSPC, EFBIG, EINVAL, etc.)
    /// the just-created empty file must be unlinked before
    /// propagating the error, so the cache root does not accumulate
    /// 0-byte staging images across retries.
    ///
    /// Drives the failure via `set_len(u64::MAX)`:
    /// [`std::fs::File::set_len`] internally `try_into::<i64>()`-s
    /// its `u64` argument and returns an `io::Error` of kind
    /// `InvalidInput` ("out of range integral type conversion
    /// attempted") for any value above `i64::MAX`, BEFORE issuing
    /// the `ftruncate(2)` syscall. That gives a deterministic,
    /// process-local, signal-free failure path — no `RLIMIT_FSIZE`
    /// manipulation, no SIGXFSZ disposition juggling, no parallel-
    /// test cross-talk. The cleanup arm semantics are identical
    /// regardless of whether the failure originates in the std
    /// pre-syscall guard or in the kernel itself, so this exercises
    /// the same drop-fd-then-unlink path that ENOSPC / EFBIG / EINVAL
    /// in production hit.
    ///
    /// Without the cleanup, the just-created 0-byte file would
    /// persist (the open succeeded; only the size enlargement
    /// failed). The post-condition asserts ENOENT at the staging
    /// path after the helper returns Err.
    #[test]
    fn create_and_size_staging_image_cleans_up_on_set_len_failure() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let staging_path = tmp.path().join("template.img.in-flight.btrfs-256m.0");

        // u64::MAX > i64::MAX → File::set_len returns InvalidInput
        // before any ftruncate syscall is issued. Sentinel choice
        // pins to this Rust-side guard rather than to a kernel
        // errno that varies across filesystems.
        let err = create_and_size_staging_image(&staging_path, u64::MAX)
            .expect_err("set_len(u64::MAX) must fail at the i64 cast");
        let msg = err.to_string();
        assert!(
            msg.contains("set staging image length"),
            "error must surface the set_len-failed context: {msg}",
        );

        // The cleanup arm must have unlinked the 0-byte file.
        // Verify by stat'ing the path: ENOENT is the success
        // criterion. Distinguishes the cleanup-fired success case
        // from the cleanup-skipped regression where the empty file
        // still sits on disk waiting to leak across retries.
        match std::fs::metadata(&staging_path) {
            Err(e) if e.kind() == io::ErrorKind::NotFound => { /* ok */ }
            Ok(m) => panic!(
                "staging image not cleaned up after set_len failure: \
                 still exists at {staging_path:?} ({} bytes)",
                m.len(),
            ),
            Err(e) => panic!("unexpected stat error: {e}"),
        }
    }

    /// Determinism contract for [`fsid_bytes`]: two `statfs` calls
    /// against the same path must produce byte-identical
    /// `fsid_bytes` outputs. The bytewise `f_fsid` read in
    /// [`fsid_bytes`] sidesteps the private `__val` field on
    /// `libc::fsid_t`; this test pins the same-input → same-output
    /// property through the actual host libc. A regression that,
    /// for instance, mis-sizes the read or includes uninitialised
    /// padding would surface here as flaky byte mismatches across
    /// the pair of statfs calls.
    ///
    /// Uses a tempdir so the test does not depend on operator
    /// state — `tempfile::tempdir()` resolves under `TMPDIR` /
    /// `$XDG_RUNTIME_DIR` / `/tmp`, all real filesystems with a
    /// stable `f_fsid` for the duration of the test.
    #[test]
    fn fsid_bytes_is_deterministic_for_same_path() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let buf1 = statfs_path(tmp.path()).expect("first statfs");
        let buf2 = statfs_path(tmp.path()).expect("second statfs");
        assert_eq!(
            fsid_bytes(&buf1),
            fsid_bytes(&buf2),
            "fsid_bytes must be deterministic across repeated statfs \
             calls against the same path; a mismatch would indicate \
             the bytewise f_fsid read produces different output for \
             the same input on this host",
        );
    }

    /// Cross-filesystem distinguishability for [`fsid_bytes`]: two
    /// paths that live on distinct filesystems must produce
    /// different `fsid_bytes` outputs. This is the property
    /// [`store_atomic`] relies on at the cross-fs gate (`f_fsid`
    /// inequality across two distinct btrfs subvolumes is the
    /// reason `f_fsid` is compared in addition to `f_type`).
    ///
    /// Probes `tempfile::tempdir()` (typically `TMPDIR` /
    /// `/tmp`, often tmpfs) against `/proc` (procfs). On almost
    /// every Linux host these land on different filesystems —
    /// procfs has a synthetic `f_fsid` distinct from any real
    /// backing filesystem. The test guards against the rare host
    /// where the two probed paths happen to share an `f_fsid`
    /// (e.g. a minimal container without `/proc` mounted, where
    /// `/proc` is fabricated by the rootfs): if both probes
    /// resolve to byte-identical `f_fsid` AND byte-identical
    /// `f_type`, the cross-fs distinguishability is genuinely
    /// not exercised on this host and we skip the inequality
    /// assertion rather than fail noisily.
    ///
    /// The skip-arm is the standard hermetic-test guard
    /// (see `verify_cache_dir_walks_through_dangling_symlink` for
    /// the same pattern in this module): we cannot demand a
    /// specific filesystem layout in CI, but when the layout
    /// holds we pin the property.
    #[test]
    fn fsid_bytes_distinguishes_different_filesystems() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let tmp_buf = statfs_path(tmp.path()).expect("statfs tempdir");
        let proc_buf = match statfs_path(std::path::Path::new("/proc")) {
            Ok(b) => b,
            Err(_) => {
                // /proc not mounted (rare — minimal containers,
                // chroots without procfs). The cross-fs property
                // cannot be exercised here; skip rather than fail.
                return;
            }
        };
        let tmp_fsid = fsid_bytes(&tmp_buf);
        let proc_fsid = fsid_bytes(&proc_buf);
        // f_type alone discriminates tmpfs from procfs; if it
        // matches AND f_fsid matches, the two probes resolved to
        // the same filesystem and the cross-fs property is not
        // exercised on this host.
        if tmp_buf.f_type == proc_buf.f_type && tmp_fsid == proc_fsid {
            return;
        }
        assert_ne!(
            tmp_fsid, proc_fsid,
            "fsid_bytes must differ across distinct filesystems \
             (tempdir f_type=0x{:x}, /proc f_type=0x{:x}); a match \
             would indicate the bytewise f_fsid read is producing a \
             constant byte pattern instead of the real fsid_t — \
             e.g. reading from a wrong offset within libc::statfs",
            tmp_buf.f_type, proc_buf.f_type,
        );
    }

    // -- clean_orphaned_tmp_dirs / clean_all coverage ------------

    /// `clean_orphaned_tmp_dirs` returns `Ok(0)` and does not
    /// error when the cache root does not exist. Mirrors the
    /// early-return contract that lets `clean_all` invoke this on
    /// a never-materialised root without bailing.
    #[test]
    fn clean_orphaned_tmp_dirs_handles_missing_root() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let nonexistent = tmp.path().join("never-created");
        let count = clean_orphaned_tmp_dirs(&nonexistent).expect("missing root must not error");
        assert_eq!(count, 0, "missing root sweeps zero entries");
    }

    /// `clean_orphaned_tmp_dirs` removes a stale staging image
    /// (`template.img.in-flight.<key>.<pid>`) when the embedded
    /// pid is dead. Uses pid=1 with a sentinel suffix that
    /// distinguishes the "dead" path from a real pid: pid=1 is
    /// reserved for init and exists; instead we use the highest
    /// possible pid value (`i32::MAX`) which is guaranteed not
    /// to be allocated on Linux — `kernel/pid.c` caps at
    /// `PID_MAX_LIMIT = 4194304` (2^22), well below i32::MAX.
    #[test]
    fn clean_orphaned_tmp_dirs_removes_dead_pid_staging_image() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let cache_root = tmp.path();
        // i32::MAX > PID_MAX_LIMIT (2^22); guaranteed-dead.
        let dead_pid = i32::MAX;
        let leaked = cache_root.join(format!("template.img.in-flight.btrfs-256m.{dead_pid}",));
        std::fs::write(&leaked, b"FAKE_STAGING_IMG").unwrap();
        let count = clean_orphaned_tmp_dirs(cache_root).expect("sweep must succeed");
        assert_eq!(count, 1, "exactly one debris entry removed");
        assert!(!leaked.exists(), "dead-pid staging image must be unlinked",);
    }

    /// `clean_orphaned_tmp_dirs` removes a stale staging directory
    /// (`<key>.tmp.<pid>`) when the embedded pid is dead. Mirrors
    /// the previous test for the second debris shape.
    #[test]
    fn clean_orphaned_tmp_dirs_removes_dead_pid_staging_directory() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let cache_root = tmp.path();
        let dead_pid = i32::MAX;
        let leaked = cache_root.join(format!("btrfs-256m.tmp.{dead_pid}"));
        std::fs::create_dir_all(&leaked).unwrap();
        std::fs::write(leaked.join("template.img"), b"PARTIAL").unwrap();
        let count = clean_orphaned_tmp_dirs(cache_root).expect("sweep must succeed");
        assert_eq!(count, 1, "exactly one debris entry removed");
        assert!(
            !leaked.exists(),
            "dead-pid staging directory must be removed",
        );
    }

    /// `clean_orphaned_tmp_dirs` PRESERVES debris owned by a live
    /// peer pid. The current process's own pid is the obvious
    /// "live" sentinel: as long as this test is running,
    /// `kill(getpid(), None)` returns `Ok(())`, NOT `Err(ESRCH)`.
    /// Without this skip, a multi-process ktstr operator running
    /// `cargo ktstr disk-template clean` while a sibling test is
    /// in flight would yank the sibling's staging file mid-build.
    #[test]
    fn clean_orphaned_tmp_dirs_preserves_live_pid_debris() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let cache_root = tmp.path();
        let live_pid = std::process::id();
        let live_image = cache_root.join(format!("template.img.in-flight.btrfs-256m.{live_pid}",));
        std::fs::write(&live_image, b"LIVE_PEER_DEBRIS").unwrap();
        let count = clean_orphaned_tmp_dirs(cache_root).expect("sweep must succeed");
        assert_eq!(
            count, 0,
            "no entries removed when only live-pid debris exists",
        );
        assert!(
            live_image.exists(),
            "live-pid debris must be preserved across sweep",
        );
    }

    /// `clean_orphaned_tmp_dirs` does NOT touch published cache
    /// entries (`<cache_key>/`) — those have no pid suffix and
    /// don't match either debris pattern. Pin the
    /// non-removal contract for published entries; a regression
    /// that broadened the prefix filter would silently delete
    /// healthy templates.
    #[test]
    fn clean_orphaned_tmp_dirs_preserves_published_entries() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let cache_root = tmp.path();
        // Published entry: directory whose name matches a cache
        // key (no `.tmp.` infix, no `template.img.in-flight.`
        // prefix) containing a `template.img`.
        let published = cache_root.join("btrfs-256m");
        std::fs::create_dir_all(&published).unwrap();
        std::fs::write(published.join(TEMPLATE_FILENAME), b"GOOD").unwrap();
        let count = clean_orphaned_tmp_dirs(cache_root).expect("sweep must succeed");
        assert_eq!(
            count, 0,
            "published cache entries must not be swept by debris GC",
        );
        assert!(published.is_dir(), "published entry must survive");
        assert!(
            published.join(TEMPLATE_FILENAME).is_file(),
            "published template.img must survive",
        );
    }

    /// `clean_orphaned_tmp_dirs` skips the `.locks/` subdirectory
    /// — it's not debris, it's the lockfile namespace. Pin the
    /// skip so a regression that broadened the prefix filter
    /// (e.g. adding `.locks` to a generic dotfile bucket) does
    /// not shatter the lockfile inodes that live peers may have
    /// open.
    #[test]
    fn clean_orphaned_tmp_dirs_preserves_lock_subdirectory() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let cache_root = tmp.path();
        let locks = cache_root.join(LOCK_DIR_NAME);
        std::fs::create_dir_all(&locks).unwrap();
        std::fs::write(locks.join("btrfs-256m.lock"), b"").unwrap();
        let count = clean_orphaned_tmp_dirs(cache_root).expect("sweep must succeed");
        assert_eq!(count, 0, ".locks/ must be invisible to the debris sweep",);
        assert!(locks.is_dir(), ".locks/ subdirectory must survive");
        assert!(
            locks.join("btrfs-256m.lock").is_file(),
            "individual lockfiles must survive",
        );
    }

    /// `clean_all` removes a published entry and reports the
    /// count. Stages a fake template via `store_atomic`, then
    /// calls `clean_all` and asserts the entry is gone and the
    /// returned count is 1.
    #[test]
    fn clean_all_removes_published_entry() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let _guard =
            crate::test_support::test_helpers::EnvVarGuard::set("KTSTR_CACHE_DIR", tmp.path());
        let cache_root_path = cache_root().unwrap();
        std::fs::create_dir_all(&cache_root_path).unwrap();
        let staged = cache_root_path.join("staged.img");
        std::fs::write(&staged, b"FAKE_TEMPLATE").unwrap();
        let installed = store_atomic("btrfs-256m", &staged).expect("store_atomic publishes");
        assert!(installed.is_file());
        let count = clean_all().expect("clean_all must succeed");
        assert_eq!(count, 1, "exactly one published entry removed");
        // The published entry directory is gone.
        assert!(
            lookup("btrfs-256m").expect("lookup ok").is_none(),
            "published entry must be gone after clean_all",
        );
        // But the lockfile inode survives.
        let lock_path = lock_path_for_key("btrfs-256m").unwrap();
        if lock_path.exists() {
            // Lock dir/file may or may not exist depending on
            // whether store_atomic touched it (this code path
            // doesn't); but if it does exist, it must NOT have
            // been removed by clean_all.
            assert!(lock_path.is_file(), "lockfile inode must survive clean_all",);
        }
    }

    /// `clean_all` reports 0 for an empty cache root. Pin the
    /// "no entries" return value so a regression that double-
    /// counts (e.g. counts the `.locks/` subdirectory) trips here.
    #[test]
    fn clean_all_reports_zero_on_empty_cache() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let _guard =
            crate::test_support::test_helpers::EnvVarGuard::set("KTSTR_CACHE_DIR", tmp.path());
        let count = clean_all().expect("clean_all must succeed on empty");
        assert_eq!(count, 0);
    }

    /// `clean_all` returns 0 (not Err) on a never-materialised
    /// cache root. Lets operator-driven runs against a fresh host
    /// (where the cache directory has not been created yet)
    /// succeed silently rather than bail.
    #[test]
    fn clean_all_handles_missing_cache_root() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        // KTSTR_CACHE_DIR points at a path that does NOT exist
        // (no create_dir_all, no store_atomic call). cache_root()
        // resolves the path string but the directory is absent.
        let nonexistent = tmp.path().join("never-created");
        let _guard =
            crate::test_support::test_helpers::EnvVarGuard::set("KTSTR_CACHE_DIR", &nonexistent);
        let count = clean_all().expect("missing cache root must not error");
        assert_eq!(count, 0);
    }

    /// `clean_all` SKIPS an entry whose lockfile is currently
    /// held by a live peer — even when run inside the same
    /// process. Acquire the lock via `acquire_template_lock`
    /// before calling `clean_all` and assert the entry survives.
    /// This covers the most operationally important contract:
    /// a `cargo ktstr disk-template clean` invoked while another
    /// ktstr process holds the lock for an in-flight test must
    /// NOT remove that entry.
    ///
    /// We hold the lock from the SAME process to avoid spawning
    /// a child; flock is per-open-file-description, so an
    /// independent open in the same process produces a distinct
    /// fd that is observed as a separate holder by `try_flock`
    /// on a third open from `clean_all`.
    #[test]
    fn clean_all_skips_entry_locked_by_live_peer() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let _guard =
            crate::test_support::test_helpers::EnvVarGuard::set("KTSTR_CACHE_DIR", tmp.path());
        // Stage a published entry so there's something to skip.
        let cache_root_path = cache_root().unwrap();
        std::fs::create_dir_all(&cache_root_path).unwrap();
        let staged = cache_root_path.join("staged.img");
        std::fs::write(&staged, b"FAKE_TEMPLATE").unwrap();
        let installed = store_atomic("btrfs-256m", &staged).expect("store_atomic publishes");
        assert!(installed.is_file());
        // Hold the per-key flock from this process. `clean_all`'s
        // `try_flock(LOCK_EX|LOCK_NB)` against the same file
        // returns `Ok(None)` because EX is exclusive — even our
        // own process's prior fd blocks the second acquire (flock
        // semantics: fd-scoped, not process-scoped).
        let _hold = acquire_template_lock("btrfs-256m").expect("acquire template lock");
        let count = clean_all().expect("clean_all must succeed");
        assert_eq!(count, 0, "locked entry must not be removed by clean_all",);
        // And the entry directory must still be on disk.
        assert!(
            lookup("btrfs-256m").expect("lookup ok").is_some(),
            "locked entry must survive clean_all",
        );
    }

    /// `clean_all` invokes `clean_orphaned_tmp_dirs` before
    /// walking published entries. Stage a dead-pid staging image
    /// alongside a published entry, run `clean_all`, and assert
    /// BOTH are removed. The published entry counts toward the
    /// returned value; the debris does not (per the doc
    /// "`clean_all` reports published-entry removals only").
    #[test]
    fn clean_all_sweeps_debris_alongside_published_entries() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let _guard =
            crate::test_support::test_helpers::EnvVarGuard::set("KTSTR_CACHE_DIR", tmp.path());
        let cache_root_path = cache_root().unwrap();
        std::fs::create_dir_all(&cache_root_path).unwrap();
        // Published entry.
        let staged = cache_root_path.join("staged.img");
        std::fs::write(&staged, b"FAKE_TEMPLATE").unwrap();
        store_atomic("btrfs-256m", &staged).unwrap();
        // Dead-pid staging image debris.
        let dead_pid = i32::MAX;
        let debris =
            cache_root_path.join(format!("template.img.in-flight.btrfs-1024m.{dead_pid}",));
        std::fs::write(&debris, b"DEBRIS").unwrap();
        // Sanity: both exist before clean_all.
        assert!(debris.is_file());
        assert!(lookup("btrfs-256m").unwrap().is_some());
        let count = clean_all().expect("clean_all must succeed");
        // The returned count covers published entries only (1).
        // The debris removal is documented in clean_all's body
        // but not folded into the count.
        assert_eq!(count, 1, "one published entry removed");
        // Both should be gone on disk regardless of count
        // accounting.
        assert!(
            !debris.exists(),
            "debris must be removed by the embedded sweep",
        );
        assert!(
            lookup("btrfs-256m").unwrap().is_none(),
            "published entry must be removed by clean_all",
        );
    }
}
