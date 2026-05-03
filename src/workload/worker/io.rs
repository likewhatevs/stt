//! Real-disk IO helpers for the `IoSyncWrite` / `IoRandRead` /
//! `IoConvoy` worker variants. Holds the [`IoBacking`] /
//! [`PhaseIoTempfile`] / [`DirectIoBuf`] RAII wrappers, the
//! `ensure_io_*` lazy-init helpers, and the per-worker xorshift PRNG
//! used to pick random offsets within a stripe. Extracted from
//! `worker/mod.rs` so the production file stays under the per-file
//! line budget.

use std::sync::atomic::{AtomicBool, Ordering};

/// Block size for the IO workloads. The block layer requires
/// `O_DIRECT` IO to be logical-block-aligned (512 bytes for
/// virtio-blk); page-sized blocks (4 KiB on x86_64 / aarch64)
/// are a convenience, not a kernel requirement, but matching
/// the page size keeps the BIO submission fast-path simple. The
/// BIO path rejects misaligned `O_DIRECT` IO with -EINVAL.
pub(super) const IO_BLOCK_SIZE: usize = 4096;

/// Sector size enforced by the virtio-blk device. Every offset the
/// workloads pass to pread/pwrite must be a multiple of this.
pub(super) const IO_SECTOR_SIZE: u64 = 512;

/// Number of stripes the per-worker striping divides the device
/// into. Matches the upper bound on plausible worker counts for
/// the smoke-test fan-out so each worker gets its own
/// non-overlapping write region.
pub(super) const IO_NUM_STRIPES: u64 = 64;

/// Linux ioctl number for BLKGETSIZE64 (returns device size in
/// bytes via `*u64`). Magic encoding: `_IOR(0x12, 114, size_t)`
/// per `<linux/fs.h>` — direction=READ (2), type=0x12, nr=114,
/// size=8 (size_t is 8 bytes on x86_64 / aarch64, the only ktstr
/// targets). The libc crate does not export this constant; it's
/// the same value GLIBC's `<sys/mount.h>` exposes when included.
pub(super) const BLKGETSIZE64: libc::c_ulong = 0x80081272;

/// Tempfile capacity for the host-side fallback when /dev/vda is
/// absent. 16 MiB is enough room for `IO_NUM_STRIPES` stripes of
/// `256 KiB` each, large enough that the random-offset PRNG hits
/// many sectors per second without wrapping immediately.
pub(super) const IO_TEMPFILE_CAPACITY: u64 = 16 * 1024 * 1024;

/// RAII handle to the IO backing for a worker — either `/dev/vda`
/// (block-device path; `tempfile_path: None`) or a per-worker
/// host-side tempfile (`tempfile_path: Some(path)`). Drop closes
/// the file (via `File`'s own Drop) and unlinks `tempfile_path` if
/// set; block-device paths are never deleted.
///
/// Pulling the unlink into Drop closes the panic-leak window the
/// previous tuple shape left open: a panic between
/// `open_io_backing` returning the tempfile path and the manual
/// `remove_file` in the worker_main cleanup tail leaked the file.
/// File close is already RAII via `std::fs::File`; the value
/// added here is the unlink.
pub(super) struct IoBacking {
    pub(super) file: std::fs::File,
    pub(super) capacity_bytes: u64,
    pub(super) tempfile_path: Option<String>,
}

impl Drop for IoBacking {
    fn drop(&mut self) {
        // Drop has nothing to assert against — swallow remove_file
        // errors. The file's own Drop closes the fd; the unlink is
        // the only host-visible cleanup we can still miss.
        if let Some(path) = self.tempfile_path.take() {
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// RAII handle to the simulated-IO tempfile [`Phase::Io`] uses.
/// Always tempfile-backed (no `/dev/vda` path), so the design is
/// simpler than [`IoBacking`]: file + path, both unconditional. The
/// path is unlinked on Drop alongside the file's own Drop closing
/// the fd. Same panic-safety rationale as [`IoBacking`] — pulling
/// the unlink into Drop closes the leak window the previous tuple
/// shape left open between iteration and the manual cleanup tail.
pub(super) struct PhaseIoTempfile {
    pub(super) file: std::fs::File,
    pub(super) path: String,
}

impl Drop for PhaseIoTempfile {
    fn drop(&mut self) {
        // Drop has nothing to assert against — swallow remove_file
        // errors. File close is RAII via `std::fs::File::drop`.
        let _ = std::fs::remove_file(&self.path);
    }
}

/// RAII handle to a logical-block-aligned scratch buffer used by
/// the `O_DIRECT` IO workloads (IoRandRead, IoConvoy). Owns a
/// non-null pointer + the layout it was allocated with, and frees
/// the allocation on Drop. Zero-initialised at construction
/// (one-shot); subsequent iterations see stale data from prior
/// `pread`/`pwrite`. The zero-init defends only against a
/// read-before-fill on the very first iteration — it is not a
/// per-iteration scrub.
///
/// Stack buffers cannot satisfy `O_DIRECT`'s 512-byte alignment
/// requirement (the BIO path rejects misaligned `O_DIRECT` IO with
/// EINVAL) on every Rust-stack target, so the heap allocation is
/// load-bearing for the workload's pathology shape.
pub(super) struct DirectIoBuf {
    ptr: std::ptr::NonNull<u8>,
    layout: std::alloc::Layout,
}

impl DirectIoBuf {
    /// Allocate a logical-block-aligned 4 KiB buffer (`IO_BLOCK_SIZE`
    /// bytes, `IO_BLOCK_SIZE`-byte alignment). Returns `None` on
    /// allocator failure so the caller can yield-and-continue
    /// rather than abort.
    pub(super) fn alloc() -> Option<Self> {
        // 4 KiB / 4 KiB align is well-defined (size is a multiple
        // of align, both powers of two). `from_size_align` returns
        // Err only if align is not a power of two or the rounded
        // size overflows isize::MAX — neither holds here.
        let layout = std::alloc::Layout::from_size_align(IO_BLOCK_SIZE, IO_BLOCK_SIZE)
            .expect("logical-block-aligned 4 KiB layout is valid");
        // SAFETY: layout has non-zero size (4 KiB > 0). alloc_zeroed
        // returns null on failure (returned to caller as None) or a
        // valid pointer to `layout.size()` bytes initialized to
        // zero. Zero-init defends against a future code path that
        // reads the buffer before it has been filled.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr = std::ptr::NonNull::new(ptr)?;
        Some(Self { ptr, layout })
    }

    /// Raw pointer to the buffer head. Used as the `pread`/`pwrite`
    /// `buf` argument. Returns `*mut u8` because the `pwrite` call
    /// site needs `*mut c_void` cast and `pread` needs the same
    /// — matches `NonNull::as_ptr` convention.
    pub(super) fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

impl Drop for DirectIoBuf {
    fn drop(&mut self) {
        // SAFETY: the pointer was obtained from `alloc_zeroed` with
        // `self.layout`; same layout passes to `dealloc`. Drop runs
        // exactly once per allocation (NonNull is not Copy and the
        // field is private, so no aliasing).
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

/// Open `/dev/vda` (or a host-side tempfile fallback) with the
/// requested flags, query its capacity in bytes, and return an
/// [`IoBacking`] that owns the file + (when the fallback fired)
/// the tempfile path. The tempfile is unlinked when the returned
/// value is dropped; block-device paths are never deleted.
pub(super) fn open_io_backing(extra_flags: libc::c_int, tid: libc::pid_t) -> Option<IoBacking> {
    use std::os::unix::io::FromRawFd;

    let dev_vda = std::path::Path::new("/dev/vda");
    if dev_vda.exists() {
        // SAFETY: nul-terminated string literal, valid for the
        // duration of the open call.
        let cstr = c"/dev/vda";
        let fd = unsafe { libc::open(cstr.as_ptr(), libc::O_RDWR | extra_flags) };
        if fd < 0 {
            return None;
        }
        let mut size_bytes: u64 = 0;
        // SAFETY: BLKGETSIZE64 writes a u64 through the pointer; we
        // own the storage and pass a valid mutable pointer. The
        // ioctl is documented for any block-device fd in
        // `<linux/fs.h>`.
        let rc = unsafe { libc::ioctl(fd, BLKGETSIZE64, &mut size_bytes as *mut u64) };
        if rc != 0 {
            unsafe { libc::close(fd) };
            return None;
        }
        // SAFETY: `fd` is owned and valid; from_raw_fd takes
        // ownership and the resulting File closes the fd on drop.
        let file = unsafe { std::fs::File::from_raw_fd(fd) };
        return Some(IoBacking {
            file,
            capacity_bytes: size_bytes,
            tempfile_path: None,
        });
    }

    // Host-side fallback: per-worker tempfile sized to
    // `IO_TEMPFILE_CAPACITY`. Opened via OpenOptions so the file is
    // created+truncated in one call; flags are then applied via
    // fcntl since OpenOptions doesn't expose O_SYNC / O_DIRECT
    // directly.
    let path = std::env::temp_dir()
        .join(format!("ktstr_iodev_{tid}"))
        .to_string_lossy()
        .to_string();
    // One-shot per-worker warn that the fallback path is in use.
    // The `tracing` crate has no `warn_once!` macro, so the
    // codebase's idiom (also used in `VirtioBlk::process_requests`
    // for `mem_unset_warned`) is an `AtomicBool::swap(true)` guard
    // around `tracing::warn!`. Each forked worker process gets its
    // own copy of this static at fork time, so the warn fires
    // exactly once per worker even though the function is called
    // on every workload that uses real disk IO.
    static FALLBACK_WARNED: AtomicBool = AtomicBool::new(false);
    if !FALLBACK_WARNED.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            path = %path,
            "virtio-blk /dev/vda absent; using tempfile fallback at {path}. \
             IO workload pathology may not reproduce."
        );
    }
    use std::os::unix::fs::OpenOptionsExt;
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .custom_flags(extra_flags)
        .open(&path)
        .ok()?;
    file.set_len(IO_TEMPFILE_CAPACITY).ok()?;
    Some(IoBacking {
        file,
        capacity_bytes: IO_TEMPFILE_CAPACITY,
        tempfile_path: Some(path),
    })
}

/// Lazy-init `io_disk` if it is not yet open. Returns `true` on
/// success (caller proceeds with IO); `false` when the open failed
/// and the caller should yield + continue this iteration. Collapses
/// the previously per-arm open-or-yield-and-warn block (3×
/// duplicated across IoSyncWrite, IoRandRead, IoConvoy) into a
/// single helper. The one-shot warn fires across all callers.
pub(super) fn ensure_io_disk(
    io_disk: &mut Option<IoBacking>,
    extra_flags: libc::c_int,
    tid: libc::pid_t,
) -> bool {
    if io_disk.is_some() {
        return true;
    }
    if let Some(d) = open_io_backing(extra_flags, tid) {
        *io_disk = Some(d);
        true
    } else {
        // One-shot per-worker error log shared across all IO
        // variants — a fallback failure in one variant is the same
        // root cause as a failure in another (both routes through
        // `open_io_backing`), and the previous per-arm static
        // multiplied the log lines without adding signal.
        static OPEN_FAILED_WARNED: AtomicBool = AtomicBool::new(false);
        if !OPEN_FAILED_WARNED.swap(true, Ordering::Relaxed) {
            tracing::error!("IO backing open failed; worker yielding without IO.");
        }
        false
    }
}

/// Lazy-init `io_buf` if it is not yet allocated. Returns `true`
/// on success; `false` on allocator failure so the caller can
/// yield + continue this iteration. Used only by IoRandRead and
/// IoConvoy — IoSyncWrite uses a stack buffer because it does not
/// open with `O_DIRECT` and so does not need the heap-aligned
/// scratch.
pub(super) fn ensure_io_buf(io_buf: &mut Option<DirectIoBuf>) -> bool {
    if io_buf.is_some() {
        return true;
    }
    match DirectIoBuf::alloc() {
        Some(b) => {
            *io_buf = Some(b);
            true
        }
        None => false,
    }
}

/// xorshift64 PRNG step. Returns the next state. One self-citing
/// invariant: the input `state` must be non-zero (xorshift's
/// fixed-point); callers seed with a tid-derived non-zero value
/// in `worker_main`.
#[inline]
pub(super) fn xorshift64(state: u64) -> u64 {
    let mut x = state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

/// Pick a sector-aligned random offset in `[0, capacity - block_size)`.
pub(super) fn rand_io_offset(rng_state: &mut u64, capacity_bytes: u64) -> u64 {
    *rng_state = xorshift64(*rng_state);
    let max_offset = capacity_bytes.saturating_sub(IO_BLOCK_SIZE as u64);
    if max_offset == 0 {
        return 0;
    }
    // Round down to sector boundary. `IO_SECTOR_SIZE` is a power of
    // 2 so the mask is a single `&` (no division).
    let raw = *rng_state % max_offset;
    raw & !(IO_SECTOR_SIZE - 1)
}

/// Compute the per-worker stripe base offset for sequential writes.
/// `tid % IO_NUM_STRIPES` selects the stripe index; `stripe_size`
/// is `capacity / IO_NUM_STRIPES`. Result is sector-aligned because
/// `capacity` is a sector-aligned device size and the divisor is a
/// power of 2.
pub(super) fn stripe_base(tid: libc::pid_t, capacity_bytes: u64) -> u64 {
    let stripe_size = (capacity_bytes / IO_NUM_STRIPES) & !(IO_SECTOR_SIZE - 1);
    let stripe_idx = (tid as u64) % IO_NUM_STRIPES;
    stripe_idx * stripe_size
}
