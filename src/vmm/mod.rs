pub mod console;
pub mod host_topology;
pub mod initramfs;
pub(crate) mod rust_init;
pub mod shm_ring;
pub mod topology;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;
#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(target_arch = "x86_64")]
pub use x86_64::acpi;
#[cfg(target_arch = "x86_64")]
pub use x86_64::boot;
#[cfg(target_arch = "x86_64")]
pub use x86_64::kvm;
#[cfg(target_arch = "x86_64")]
pub use x86_64::kvm_stats;
#[cfg(target_arch = "x86_64")]
pub use x86_64::mptable;

#[cfg(target_arch = "aarch64")]
pub use aarch64::boot;
#[cfg(target_arch = "aarch64")]
pub use aarch64::kvm;
#[cfg(target_arch = "aarch64")]
pub use aarch64::topology as arch_topology;

pub use topology::Topology;

use anyhow::{Context, Result};
use kvm_ioctls::VcpuExit;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::os::unix::thread::JoinHandleExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use vm_memory::{Bytes, GuestAddress, GuestMemory, GuestMemoryMmap};

use crate::monitor;

// ---------------------------------------------------------------------------
// Shared hugepage memory allocation
// ---------------------------------------------------------------------------

/// Allocate guest memory backed by 2MB hugepages at the given base address.
///
/// Uses MmapRegionBuilder with MAP_HUGETLB to request hugepage-backed
/// anonymous memory. Falls back to regular pages if hugepages fail.
pub(crate) fn allocate_hugepage_memory(size: usize, base: GuestAddress) -> Result<GuestMemoryMmap> {
    use vm_memory::mmap::{GuestRegionMmap, MmapRegionBuilder};

    let needed_pages = size / (2 << 20);
    let free_pages = host_topology::hugepages_free();
    if free_pages < needed_pages as u64 {
        eprintln!(
            "performance_mode: WARNING: not enough hugepages \
             (needed {} MB = {} pages, available {} pages). \
             Using regular pages.",
            size >> 20,
            needed_pages,
            free_pages,
        );
        return GuestMemoryMmap::<()>::from_ranges(&[(base, size)])
            .context("allocate guest memory (hugepage fallback)");
    }

    let flags = libc::MAP_ANONYMOUS | libc::MAP_PRIVATE | libc::MAP_HUGETLB | libc::MAP_HUGE_2MB;

    let region = MmapRegionBuilder::new(size)
        .with_mmap_prot(libc::PROT_READ | libc::PROT_WRITE)
        .with_mmap_flags(flags)
        .with_hugetlbfs(true)
        .build();

    match region {
        Ok(r) => {
            // Pre-fault hugepages to detect allocation failures now rather
            // than as cryptic guest-side "uncompression error" page faults.
            let ret = unsafe {
                libc::madvise(
                    r.as_ptr() as *mut libc::c_void,
                    r.size(),
                    libc::MADV_POPULATE_WRITE,
                )
            };
            if ret != 0 {
                let err = std::io::Error::last_os_error();
                eprintln!(
                    "performance_mode: WARNING: hugepage pre-fault failed ({err}), \
                     not enough hugepages (needed: {} MB, available: {} pages). \
                     Using regular pages.",
                    size >> 20,
                    free_pages,
                );
                return GuestMemoryMmap::<()>::from_ranges(&[(base, size)])
                    .context("allocate guest memory (hugepage fallback)");
            }
            eprintln!(
                "performance_mode: allocated {} MB with 2MB hugepages",
                size >> 20
            );
            let guest_region = GuestRegionMmap::new(r, base)
                .ok_or_else(|| anyhow::anyhow!("hugepage region overflow"))?;
            GuestMemoryMmap::from_regions(vec![guest_region])
                .context("create guest memory from hugepage region")
        }
        Err(e) => {
            eprintln!(
                "performance_mode: WARNING: hugepage allocation failed ({e}), \
                 not enough hugepages (needed: {} MB, available: {} pages). \
                 Using regular pages.",
                size >> 20,
                free_pages,
            );
            GuestMemoryMmap::<()>::from_ranges(&[(base, size)])
                .context("allocate guest memory (hugepage fallback)")
        }
    }
}

// ---------------------------------------------------------------------------
// PiMutex — priority-inheritance mutex via pthread_mutex + PTHREAD_PRIO_INHERIT
// ---------------------------------------------------------------------------

/// Mutex that uses the kernel's priority-inheritance protocol to avoid
/// priority inversion between RT and non-RT threads.
///
/// When a SCHED_FIFO thread blocks on a PiMutex held by a SCHED_OTHER
/// thread, the kernel temporarily boosts the holder to the waiter's
/// priority, ensuring the critical section completes without unbounded
/// delay.
///
/// Uses `pthread_mutexattr_setprotocol(PTHREAD_PRIO_INHERIT)` which maps
/// to `FUTEX_LOCK_PI` in the kernel.
pub(crate) struct PiMutex<T> {
    inner: std::cell::UnsafeCell<T>,
    mutex: std::cell::UnsafeCell<libc::pthread_mutex_t>,
}

// SAFETY: PiMutex provides mutual exclusion via pthread_mutex_lock/unlock.
// The UnsafeCell<T> is only accessed while the mutex is held.
unsafe impl<T: Send> Send for PiMutex<T> {}
unsafe impl<T: Send> Sync for PiMutex<T> {}

impl<T> PiMutex<T> {
    /// Create a new PI mutex wrapping `value`.
    pub(crate) fn new(value: T) -> Self {
        unsafe {
            let mut attr: libc::pthread_mutexattr_t = std::mem::zeroed();
            libc::pthread_mutexattr_init(&mut attr);
            libc::pthread_mutexattr_setprotocol(&mut attr, libc::PTHREAD_PRIO_INHERIT);
            let mut mutex: libc::pthread_mutex_t = std::mem::zeroed();
            libc::pthread_mutex_init(&mut mutex, &attr);
            libc::pthread_mutexattr_destroy(&mut attr);
            PiMutex {
                inner: std::cell::UnsafeCell::new(value),
                mutex: std::cell::UnsafeCell::new(mutex),
            }
        }
    }

    /// Lock the mutex and return a guard providing `&mut T`.
    pub(crate) fn lock(&self) -> PiMutexGuard<'_, T> {
        unsafe {
            let rc = libc::pthread_mutex_lock(self.mutex.get());
            debug_assert_eq!(rc, 0, "pthread_mutex_lock failed: {rc}");
        }
        PiMutexGuard { mutex: self }
    }
}

impl<T> Drop for PiMutex<T> {
    fn drop(&mut self) {
        unsafe {
            libc::pthread_mutex_destroy(self.mutex.get());
        }
    }
}

pub(crate) struct PiMutexGuard<'a, T> {
    mutex: &'a PiMutex<T>,
}

impl<T> std::ops::Deref for PiMutexGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { &*self.mutex.inner.get() }
    }
}

impl<T> std::ops::DerefMut for PiMutexGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.mutex.inner.get() }
    }
}

impl<T> Drop for PiMutexGuard<'_, T> {
    fn drop(&mut self) {
        unsafe {
            let rc = libc::pthread_mutex_unlock(self.mutex.mutex.get());
            debug_assert_eq!(rc, 0, "pthread_mutex_unlock failed: {rc}");
        }
    }
}

// ---------------------------------------------------------------------------
// Initramfs cache — two-tier: POSIX shm (cross-process) + in-process HashMap
// ---------------------------------------------------------------------------

/// Cache key for base initramfs (payload + scheduler, no args).
/// Derived from a content hash of the binary files so identical inputs
/// produce the same key regardless of path or mtime.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BaseKey(u64);

/// Hash a file's content for cache keying via streaming reads.
pub(crate) fn hash_file(path: &Path) -> Result<u64> {
    use std::io::Read;
    let mut f =
        std::fs::File::open(path).with_context(|| format!("open for hash: {}", path.display()))?;
    let mut hasher = std::hash::DefaultHasher::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = f
            .read(&mut buf)
            .with_context(|| format!("read for hash: {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.write(&buf[..n]);
    }
    Ok(hasher.finish())
}

impl BaseKey {
    pub(crate) fn new(payload: &Path, scheduler: Option<&Path>) -> Result<Self> {
        let mut hasher = std::hash::DefaultHasher::new();

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

        Ok(BaseKey(hasher.finish()))
    }

    /// Hash shared library paths and content samples for a binary so
    /// the cache key changes when any shared lib is updated on the host.
    fn hash_shared_libs(binary: &Path, hasher: &mut std::hash::DefaultHasher) {
        if let Ok(libs) = initramfs::resolve_shared_libs(binary) {
            let mut entries: Vec<_> = libs.iter().map(|(_, p)| p.clone()).collect();
            entries.sort();
            for p in &entries {
                p.to_str().unwrap_or("").hash(hasher);
                if let Ok(sample) = hash_file(p) {
                    sample.hash(hasher);
                }
            }
        }
    }
}

/// Process-global cache for base initramfs bytes. Keyed by content hash
/// of payload + scheduler binaries. The lock is only held during map
/// lookup/insert, never during the actual build.
fn base_cache() -> &'static Mutex<HashMap<BaseKey, Arc<Vec<u8>>>> {
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
            let data = initramfs::create_initramfs_base(payload, extras)?;
            tracing::debug!(
                elapsed_us = t0.elapsed().as_micros(),
                bytes = data.len(),
                "create_initramfs_base",
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
    let data = initramfs::create_initramfs_base(payload, extras)?;
    let arc = Arc::new(data);
    tracing::debug!(
        elapsed_us = t0.elapsed().as_micros(),
        bytes = arc.len(),
        "create_initramfs_base (fallback)",
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
/// Scans for `stt-base-*` and `stt-gz-*` entries and unlinks any whose
/// hash suffix differs from the current key.
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
        let hash_suffix = if let Some(s) = name_str.strip_prefix("stt-base-") {
            s
        } else if let Some(s) = name_str.strip_prefix("stt-gz-") {
            s
        } else {
            continue;
        };
        if hash_suffix == current_suffix {
            continue;
        }
        let shm_name = format!("/{name_str}");
        if let Ok(cname) = std::ffi::CString::new(shm_name) {
            unsafe {
                libc::shm_unlink(cname.as_ptr());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SHM O_EXCL race gate helpers
// ---------------------------------------------------------------------------

enum ShmCreateResult {
    /// We created the segment; fd holds an exclusive flock.
    Winner(std::os::unix::io::RawFd),
    /// Segment already exists (another process is building or built it).
    Exists,
    /// shm_open failed for a reason other than EEXIST.
    Error,
}

/// Try to create a POSIX shm segment with O_CREAT|O_EXCL. On success,
/// acquire LOCK_EX and return the fd. On EEXIST, return Exists.
fn shm_try_create_excl(name: &str) -> ShmCreateResult {
    let Ok(cname) = std::ffi::CString::new(name) else {
        return ShmCreateResult::Error;
    };
    unsafe {
        let fd = libc::shm_open(
            cname.as_ptr(),
            libc::O_CREAT | libc::O_EXCL | libc::O_RDWR,
            0o644,
        );
        if fd < 0 {
            let err = *libc::__errno_location();
            return if err == libc::EEXIST {
                ShmCreateResult::Exists
            } else {
                ShmCreateResult::Error
            };
        }

        // Take exclusive lock before writing.
        if libc::flock(fd, libc::LOCK_EX) != 0 {
            libc::close(fd);
            return ShmCreateResult::Error;
        }

        ShmCreateResult::Winner(fd)
    }
}

/// Write data to the shm fd, then release the exclusive lock and close.
/// On failure (ftruncate or mmap), unlinks the segment so future callers
/// don't find a corrupt/empty segment and can retry.
fn shm_write_and_release(fd: std::os::unix::io::RawFd, data: &[u8], seg_name: &str) {
    unsafe {
        if libc::ftruncate(fd, data.len() as libc::off_t) != 0 {
            if let Ok(cname) = std::ffi::CString::new(seg_name) {
                libc::shm_unlink(cname.as_ptr());
            }
            libc::flock(fd, libc::LOCK_UN);
            libc::close(fd);
            return;
        }

        let ptr = libc::mmap(
            std::ptr::null_mut(),
            data.len(),
            libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd,
            0,
        );
        if ptr == libc::MAP_FAILED {
            // Zero the size so readers blocked on LOCK_SH see st_size=0
            // from fstat and return None instead of mapping zero-filled bytes.
            libc::ftruncate(fd, 0);
            if let Ok(cname) = std::ffi::CString::new(seg_name) {
                libc::shm_unlink(cname.as_ptr());
            }
        } else {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
            libc::munmap(ptr, data.len());
        }

        libc::flock(fd, libc::LOCK_UN);
        libc::close(fd);
    }
}

// ---------------------------------------------------------------------------
// Initramfs memory floor
// ---------------------------------------------------------------------------

/// Minimum guest memory (in MB) needed to extract an initramfs.
///
/// During kernel boot the compressed cpio sits in the ramdisk region
/// while the kernel decompresses it into tmpfs. Both coexist until
/// free_initrd_mem() releases the compressed copy. The uncompressed
/// content dominates: `2 * cpio_size` bounds the peak, plus 128 MB
/// for kernel image, page tables, slab, and process execution.
/// `shm_bytes` is the SHM region carved from the top of guest memory
/// (E820 gap) — it reduces usable RAM.
pub fn initramfs_min_memory_mb(initramfs_bytes: u64, shm_bytes: u64) -> u32 {
    let initramfs_mb = ((initramfs_bytes + (1 << 20) - 1) >> 20) as u32;
    let shm_mb = ((shm_bytes + (1 << 20) - 1) >> 20) as u32;
    2 * initramfs_mb + 128 + shm_mb
}

/// Estimate minimum guest memory from binary file sizes.
///
/// The uncompressed cpio is approximately the sum of the payload,
/// scheduler, and shared library sizes plus cpio metadata (~5%).
/// `shm_bytes` accounts for the SHM region carved from guest memory.
pub fn estimate_min_memory_mb(
    payload: &std::path::Path,
    scheduler: Option<&std::path::Path>,
    shm_bytes: u64,
) -> u32 {
    // Debug binaries are stripped before packing into the initramfs.
    // Estimate stripped size as 1/2 of debug size. Actual ratios range
    // from 4% to 30%, but polars-heavy binaries retain more data
    // sections, so /2 avoids under-allocation at the cost of ~500MB
    // over-estimate in the common case.
    let payload_size = std::fs::metadata(payload).map(|m| m.len()).unwrap_or(0) / 2;
    let sched_size = scheduler
        .and_then(|p| std::fs::metadata(p).ok())
        .map(|m| m.len() / 2)
        .unwrap_or(0);
    let lib_size: u64 = [Some(payload), scheduler]
        .into_iter()
        .flatten()
        .filter_map(|p| initramfs::resolve_shared_libs(p).ok())
        .flat_map(|libs| libs.into_iter().map(|(_, p)| p))
        .filter_map(|p| std::fs::metadata(&p).ok())
        .map(|m| m.len())
        .sum();
    // 5% overhead for cpio headers and alignment.
    let uncompressed = ((payload_size + sched_size + lib_size) as f64 * 1.05) as u64;
    initramfs_min_memory_mb(uncompressed, shm_bytes)
}

// ---------------------------------------------------------------------------
// ImmediateExitHandle — cross-thread access to kvm_run.immediate_exit
// ---------------------------------------------------------------------------

/// Handle for setting the `immediate_exit` field in a vCPU's mmap'd `kvm_run`
/// struct from outside the vCPU thread.
///
/// The `kvm_run` page is `MAP_SHARED` between kernel and userspace; the
/// `immediate_exit` field is a single byte read by KVM atomically before
/// entering `KVM_RUN`. Setting it to 1 causes the next `KVM_RUN` to return
/// immediately with `EINTR`.
struct ImmediateExitHandle {
    ptr: *mut u8,
}

// SAFETY: The `kvm_run` page is mmap'd MAP_SHARED and designed for cross-thread
// access. The `immediate_exit` field is a single byte with no torn-read risk.
// The pointer remains valid for the lifetime of the VcpuFd that owns the mmap.
unsafe impl Send for ImmediateExitHandle {}
unsafe impl Sync for ImmediateExitHandle {}

impl ImmediateExitHandle {
    /// Extract the `immediate_exit` pointer from a VcpuFd before the fd is
    /// moved into a thread. Must be called while the caller has `&mut VcpuFd`.
    fn from_vcpu(vcpu: &mut kvm_ioctls::VcpuFd) -> Self {
        let kvm_run = vcpu.get_kvm_run();
        let ptr: *mut u8 = &mut kvm_run.immediate_exit;
        Self { ptr }
    }

    /// Set `immediate_exit` to the given value.
    fn set(&self, val: u8) {
        // SAFETY: ptr points into a MAP_SHARED mmap that outlives this handle.
        // Single-byte write is atomic on all architectures KVM supports.
        unsafe {
            std::ptr::write_volatile(self.ptr, val);
        }
    }
}

// ---------------------------------------------------------------------------
// Signal handling — Firecracker/libkrun pattern: SIGRTMIN + immediate_exit
// ---------------------------------------------------------------------------

/// Signal used to kick vCPU threads out of KVM_RUN.
/// All three Rust reference VMMs (Firecracker, Cloud Hypervisor, libkrun)
/// use SIGRTMIN. SIGUSR1/SIGUSR2 conflict with application-level signals.
fn vcpu_signal() -> libc::c_int {
    libc::SIGRTMIN()
}

/// Signal handler — Firecracker pattern.
/// The handler itself is a no-op; its sole purpose is to cause KVM_RUN
/// to return with EINTR. The fence ensures that a write to
/// `kvm_run.immediate_exit` from another thread (via ImmediateExitHandle)
/// is visible when KVM_RUN returns.
extern "C" fn vcpu_signal_handler(_: libc::c_int, _: *mut libc::siginfo_t, _: *mut libc::c_void) {
    std::sync::atomic::fence(Ordering::Acquire);
}

/// Register the vCPU signal handler and unblock the signal in this thread.
/// Must be called from each vCPU thread before entering the run loop.
/// Follows Firecracker's register_kick_signal_handler + QEMU's
/// kvm_init_cpu_signals: register SA_SIGINFO handler, then unblock via
/// pthread_sigmask so the signal is deliverable inside KVM_RUN.
fn register_vcpu_signal_handler() {
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = vcpu_signal_handler as *const () as usize;
        sa.sa_flags = libc::SA_SIGINFO;
        libc::sigemptyset(&mut sa.sa_mask);
        libc::sigaction(vcpu_signal(), &sa, std::ptr::null_mut());

        // Unblock the signal in this thread so pthread_kill can deliver it.
        let mut set: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut set);
        libc::sigaddset(&mut set, vcpu_signal());
        libc::pthread_sigmask(libc::SIG_UNBLOCK, &set, std::ptr::null_mut());
    }
}

// ---------------------------------------------------------------------------
// vCPU affinity
// ---------------------------------------------------------------------------

/// Pin the calling thread to a single host CPU via sched_setaffinity(0, ...).
/// Logs success or warning; does not fail the VM.
fn pin_current_thread(cpu: usize, label: &str) {
    let mut cpuset = nix::sched::CpuSet::new();
    if let Err(e) = cpuset.set(cpu) {
        eprintln!("performance_mode: WARNING: cpuset.set({cpu}) for {label}: {e}");
        return;
    }
    match nix::sched::sched_setaffinity(nix::unistd::Pid::from_raw(0), &cpuset) {
        Ok(()) => eprintln!("performance_mode: pinned {label} to host CPU {cpu}"),
        Err(e) => eprintln!("performance_mode: WARNING: pin {label} to CPU {cpu}: {e}"),
    }
}

/// Set the calling thread to SCHED_FIFO at the given priority.
/// Logs success or warning; does not fail the VM.
fn set_rt_priority(priority: i32, label: &str) {
    let param = libc::sched_param {
        sched_priority: priority,
    };
    let rc = unsafe { libc::sched_setscheduler(0, libc::SCHED_FIFO, &param) };
    if rc == 0 {
        eprintln!("performance_mode: {label} set to SCHED_FIFO priority {priority}");
    } else {
        let err = std::io::Error::last_os_error();
        eprintln!("performance_mode: WARNING: SCHED_FIFO for {label}: {err} (need CAP_SYS_NICE)");
    }
}

// ---------------------------------------------------------------------------
// VmResult
// ---------------------------------------------------------------------------

/// Result of a VM execution.
#[derive(Debug)]
pub struct VmResult {
    pub success: bool,
    pub exit_code: i32,
    pub duration: Duration,
    pub timed_out: bool,
    pub output: String,
    pub stderr: String,
    pub monitor: Option<monitor::MonitorReport>,
    /// Data drained from the SHM ring buffer after VM exit.
    pub shm_data: Option<shm_ring::ShmDrainResult>,
    /// Stimulus events extracted from SHM ring entries.
    pub stimulus_events: Vec<shm_ring::StimulusEvent>,
    /// BPF verifier stats collected from host-side memory reads.
    pub verifier_stats: Vec<monitor::bpf_prog::ProgVerifierStats>,
    /// KVM per-vCPU cumulative stats (requires Linux >= 5.15, x86_64 only).
    pub kvm_stats: Option<KvmStatsTotals>,
}

/// Per-vCPU KVM stats read after VM exit. Each map holds cumulative
/// counter values from the VM's lifetime.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct KvmStatsTotals {
    /// Per-vCPU stat maps. Index is vCPU id.
    pub per_vcpu: Vec<HashMap<String, u64>>,
}

/// Trust-relevant stats for scheduler testing.
pub const KVM_INTERESTING_STATS: &[&str] = &[
    "exits",
    "halt_exits",
    "halt_successful_poll",
    "halt_attempted_poll",
    "halt_wait_ns",
    "preemption_reported",
    "signal_exits",
    "hypercalls",
];

impl KvmStatsTotals {
    /// Sum a stat across all vCPUs.
    pub fn sum(&self, name: &str) -> u64 {
        self.per_vcpu.iter().filter_map(|m| m.get(name)).sum()
    }

    /// Average a stat across all vCPUs (returns 0 if no vCPUs).
    pub fn avg(&self, name: &str) -> u64 {
        if self.per_vcpu.is_empty() {
            return 0;
        }
        self.sum(name) / self.per_vcpu.len() as u64
    }
}

/// State returned by [`SttVm::run_vm`] after the BSP exits.
/// Passed to [`SttVm::collect_results`] to produce [`VmResult`].
struct VmRunState {
    exit_code: i32,
    timed_out: bool,
    ap_threads: Vec<VcpuThread>,
    monitor_handle: Option<JoinHandle<(Vec<monitor::MonitorSample>, shm_ring::ShmDrainResult)>>,
    bpf_write_handle: Option<JoinHandle<()>>,
    com1: Arc<PiMutex<console::Serial>>,
    com2: Arc<PiMutex<console::Serial>>,
    kill: Arc<AtomicBool>,
    vm: kvm::SttKvm,
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Start of the guest physical address space used for RAM.
/// x86_64: PA 0 (sub-1MB legacy regions share the same PA space).
/// aarch64: device MMIO below DRAM_START, RAM above.
#[cfg(target_arch = "x86_64")]
const DRAM_BASE: u64 = 0;

#[cfg(target_arch = "aarch64")]
const DRAM_BASE: u64 = kvm::DRAM_START;

/// Address where initramfs is loaded in guest memory.
#[cfg(target_arch = "x86_64")]
const INITRD_ADDR: u64 = 0x800_0000; // 128 MB

/// Compute initramfs load address at the high end of DRAM, just below
/// the FDT. Matches Firecracker/Cloud Hypervisor placement pattern —
/// avoids conflicts with early kernel allocations near the kernel image.
#[cfg(target_arch = "aarch64")]
fn aarch64_initrd_addr(memory_mb: u32, shm_size: u64, initrd_max_size: u64) -> u64 {
    let fdt_addr = aarch64::fdt::fdt_address(memory_mb, shm_size);
    // Place initrd just below FDT, page-aligned.
    (fdt_addr - initrd_max_size) & !0xFFF
}

// ---------------------------------------------------------------------------
// VcpuThread — Cloud Hypervisor pattern with Firecracker's immediate_exit
// ---------------------------------------------------------------------------

/// Per-vCPU thread handle with signal-based kick and ACK flag.
struct VcpuThread {
    handle: JoinHandle<kvm_ioctls::VcpuFd>,
    /// Set by the thread after it exits the KVM_RUN loop.
    exited: Arc<AtomicBool>,
    /// Handle to set `kvm_run.immediate_exit` from outside the vCPU thread.
    /// `None` when KVM_CAP_IMMEDIATE_EXIT is not available.
    immediate_exit: Option<ImmediateExitHandle>,
}

impl VcpuThread {
    /// Kick a vCPU out of KVM_RUN. If immediate_exit is available, sets the
    /// flag before sending the signal (Firecracker pattern). Otherwise falls
    /// back to signal-only (the signal handler causes EINTR).
    fn kick(&self) {
        if let Some(ref ie) = self.immediate_exit {
            ie.set(1);
            std::sync::atomic::fence(Ordering::Release);
        }
        self.signal();
    }

    /// Send the kick signal to interrupt a blocked KVM_RUN.
    fn signal(&self) {
        unsafe {
            libc::pthread_kill(self.handle.as_pthread_t() as libc::pthread_t, vcpu_signal());
        }
    }

    /// Wait for the thread to exit, retrying the kick periodically.
    /// Cloud Hypervisor pattern: poll exited flag, re-kick every 10ms.
    fn wait_for_exit(&self, timeout: Duration) {
        let start = Instant::now();
        let mut last_kick = Instant::now();
        while !self.exited.load(Ordering::Acquire) {
            if start.elapsed() > timeout {
                break;
            }
            if last_kick.elapsed() > Duration::from_millis(10) {
                self.kick();
                last_kick = Instant::now();
            }
            std::thread::yield_now();
        }
    }
}

// ---------------------------------------------------------------------------
// SttVm — builder + run
// ---------------------------------------------------------------------------

/// Builder for creating and running VMs with custom topologies.
pub struct SttVm {
    kernel: PathBuf,
    init_binary: Option<PathBuf>,
    scheduler_binary: Option<PathBuf>,
    run_args: Vec<String>,
    sched_args: Vec<String>,
    topology: Topology,
    memory_mb: u32,
    cmdline_extra: String,
    timeout: Duration,
    /// Size of the SHM ring buffer region at the top of guest memory. 0 = disabled.
    shm_size: u64,
    /// Thresholds for reactive SysRq-D dump. When set and the monitor
    /// detects a sustained violation, it writes the dump flag to guest SHM.
    monitor_thresholds: Option<crate::monitor::MonitorThresholds>,
    /// Override for `scx_watchdog_timeout` in the guest kernel (seconds).
    /// Converted to jiffies via CONFIG_HZ at monitor start time.
    watchdog_timeout_s: Option<u64>,
    /// Host-side BPF map write parameters. When set, a thread polls for
    /// BPF map discoverability, waits for scenario start via SHM ring,
    /// then writes a u32 value at the specified offset.
    bpf_map_write: Option<BpfMapWriteParams>,
    /// Performance mode: vCPU pinning to host LLCs, hugepage-backed guest
    /// memory, KVM_HINTS_REALTIME CPUID hint, PAUSE and HLT VM exit
    /// disabling via KVM_CAP_X86_DISABLE_EXITS, and oversubscription
    /// validation. KVM_CAP_HALT_POLL is skipped (guest haltpoll cpuidle
    /// disables host halt polling via MSR_KVM_POLL_CONTROL).
    performance_mode: bool,
    /// Pinning plan computed during build() when performance_mode is enabled.
    /// Stored so topology is read once and the plan is reused at VM start.
    pinning_plan: Option<host_topology::PinningPlan>,
    /// NUMA nodes to mbind guest memory to (derived from pinning plan).
    mbind_nodes: Vec<usize>,
    /// CPU flock fds for non-perf VMs. Held for the VM's lifetime to
    /// prevent other VMs from double-booking the same CPUs.
    #[allow(dead_code)]
    cpu_locks: Vec<std::os::fd::OwnedFd>,
    /// Shell commands to run in the guest to enable a kernel-built scheduler.
    sched_enable_cmds: Vec<String>,
    /// Shell commands to run in the guest to disable a kernel-built scheduler.
    sched_disable_cmds: Vec<String>,
}

/// Parameters for a host-side BPF map write during VM execution.
#[derive(Clone)]
struct BpfMapWriteParams {
    map_name_suffix: String,
    offset: usize,
    value: u32,
}

impl SttVm {
    pub fn builder() -> SttVmBuilder {
        SttVmBuilder::default()
    }

    /// Boot the VM, run until shutdown/timeout, return captured output.
    pub fn run(&self) -> Result<VmResult> {
        let start = Instant::now();

        let initramfs_handle = self.spawn_initramfs_resolve();
        let (vm, kernel_result) = self.create_vm_and_load_kernel()?;

        #[cfg(target_arch = "x86_64")]
        {
            self.setup_memory(&vm, &kernel_result, initramfs_handle)?;
            self.setup_vcpus(&vm, kernel_result.entry)?;
        }
        #[cfg(target_arch = "aarch64")]
        {
            self.setup_memory_aarch64(&vm, &kernel_result, initramfs_handle)?;
            self.setup_vcpus_aarch64(&vm, kernel_result.entry)?;
        }

        // Open persistent stats fds before vCPUs move to threads.
        // Stats fds hold kernel references independent of VcpuFd ownership.
        // Read once after VM exit to capture cumulative totals.
        #[cfg(target_arch = "x86_64")]
        let stats_ctx = kvm_stats::open_stats_context(&vm.vcpus);
        #[cfg(target_arch = "x86_64")]
        if stats_ctx.is_none() {
            tracing::debug!("KVM_GET_STATS_FD not supported, skipping stats collection");
        }

        tracing::debug!(elapsed_us = start.elapsed().as_micros(), "total_setup");

        let run = self.run_vm(start, vm)?;

        let mut result = self.collect_results(start, run)?;

        // Read cumulative KVM stats after VM exit.
        #[cfg(target_arch = "x86_64")]
        if let Some(ctx) = stats_ctx {
            result.kvm_stats = Some(ctx.read_stats());
        }

        Ok(result)
    }

    /// Create the KVM VM and load the kernel.
    fn create_vm_and_load_kernel(&self) -> Result<(kvm::SttKvm, boot::KernelLoadResult)> {
        let t0 = Instant::now();
        let use_hugepages = self.performance_mode
            && host_topology::hugepages_free() >= host_topology::hugepages_needed(self.memory_mb);
        let vm = if use_hugepages {
            kvm::SttKvm::new_with_hugepages(self.topology, self.memory_mb, self.performance_mode)
                .context("create VM with hugepages")?
        } else {
            kvm::SttKvm::new(self.topology, self.memory_mb, self.performance_mode)
                .context("create VM")?
        };
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "kvm_create");

        // mbind guest memory to NUMA node(s) of pinned vCPUs.
        // With hugepages, pages may already be faulted from the
        // allocating CPU's node. mbind is best-effort NUMA placement
        // for pages not yet faulted; already-resident pages are not moved.
        if self.performance_mode
            && !self.mbind_nodes.is_empty()
            && let Ok(host_addr) = vm.guest_mem.get_host_address(GuestAddress(DRAM_BASE))
        {
            let mem_size = (self.memory_mb as u64) << 20;
            host_topology::mbind_to_nodes(host_addr, mem_size as usize, &self.mbind_nodes);
        }

        let t0 = Instant::now();
        let kernel_result =
            boot::load_kernel(&vm.guest_mem, &self.kernel).context("load kernel")?;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "load_kernel");

        Ok((vm, kernel_result))
    }

    /// Spawn initramfs resolution on a background thread.
    /// Returns the handle to join later (after KVM creation completes).
    fn spawn_initramfs_resolve(&self) -> Option<JoinHandle<Result<(BaseRef, BaseKey)>>> {
        let bin = self.init_binary.as_ref()?;
        let payload = bin.clone();
        let scheduler = self.scheduler_binary.clone();
        std::thread::Builder::new()
            .name("initramfs-resolve".into())
            .spawn(move || -> Result<(BaseRef, BaseKey)> {
                let extras: Vec<(&str, &std::path::Path)> = scheduler
                    .as_deref()
                    .map(|p| vec![("scheduler", p)])
                    .unwrap_or_default();
                let key = BaseKey::new(&payload, scheduler.as_deref())?;
                let base = get_or_build_base(&payload, &extras, &key)?;
                Ok((base, key))
            })
            .ok()
    }

    /// Join the initramfs thread and load the result into guest memory.
    ///
    /// Compresses base+suffix with gzip. Attempts COW overlay from a
    /// compressed SHM segment to share physical pages across VMs.
    /// Falls back to write_slice if COW fails.
    #[cfg(target_arch = "x86_64")]
    fn join_and_load_initramfs(
        &self,
        vm: &kvm::SttKvm,
        handle: JoinHandle<Result<(BaseRef, BaseKey)>>,
        load_addr: u64,
    ) -> Result<(Option<u64>, Option<u32>)> {
        let t0 = Instant::now();
        let (base, key) = handle
            .join()
            .map_err(|_| anyhow::anyhow!("initramfs-resolve thread panicked"))??;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "initramfs_join");
        let base_bytes: &[u8] = base.as_ref();

        let t0 = Instant::now();
        let enable_refs: Vec<&str> = self.sched_enable_cmds.iter().map(|s| s.as_str()).collect();
        let disable_refs: Vec<&str> = self.sched_disable_cmds.iter().map(|s| s.as_str()).collect();
        let suffix = initramfs::build_suffix_full(
            base_bytes.len(),
            &self.run_args,
            &self.sched_args,
            &enable_refs,
            &disable_refs,
        )?;
        let uncompressed_size = base_bytes.len() + suffix.len();
        tracing::debug!(
            elapsed_us = t0.elapsed().as_micros(),
            base_bytes = base_bytes.len(),
            suffix_bytes = suffix.len(),
            "build_suffix",
        );

        // Enforce minimum memory for initramfs extraction.
        let min_mb = initramfs_min_memory_mb(uncompressed_size as u64, self.shm_size);
        if self.memory_mb < min_mb {
            anyhow::bail!(
                "VM memory {}MB insufficient for initramfs \
                 (uncompressed={}MB): need {}MB",
                self.memory_mb,
                uncompressed_size >> 20,
                min_mb,
            );
        }

        // Compress base and suffix as separate gzip streams. The kernel
        // initramfs decompressor handles concatenated gzip natively.
        // Keeping them separate lets us COW-map the base from SHM.
        let t0 = Instant::now();
        let gz_base = self.get_or_compress_base(base_bytes, &key)?;
        let gz_suffix = {
            use flate2::write::GzEncoder;
            use std::io::Write;
            let mut enc = GzEncoder::new(Vec::new(), flate2::Compression::fast());
            enc.write_all(&suffix).context("gzip suffix")?;
            enc.finish().context("finish gzip suffix")?
        };
        let total_compressed = gz_base.len() + gz_suffix.len();
        tracing::debug!(
            elapsed_us = t0.elapsed().as_micros(),
            uncompressed = uncompressed_size,
            gz_base = gz_base.len(),
            gz_suffix = gz_suffix.len(),
            ratio = format!("{:.1}x", uncompressed_size as f64 / total_compressed as f64),
            "gzip_initramfs",
        );

        // Try COW overlay: mmap compressed base from SHM fd directly
        // into guest memory, sharing physical pages across VMs.
        let t0 = Instant::now();
        let cow_ok = self.try_cow_overlay(vm, &key, gz_base.len(), load_addr);
        if cow_ok {
            // Base is COW-mapped. Write suffix after it.
            vm.guest_mem
                .write_slice(&gz_suffix, GuestAddress(load_addr + gz_base.len() as u64))
                .context("write gz suffix after COW base")?;
            tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "cow_initramfs");
        } else {
            // Fallback: write both parts via write_slice.
            initramfs::load_initramfs_parts(&vm.guest_mem, &[&gz_base, &gz_suffix], load_addr)?;
            tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "copy_initramfs");
        }

        Ok((Some(load_addr), Some(total_compressed as u32)))
    }

    /// Get or build the compressed base. Checks gz SHM first, then
    /// compresses and stores.
    #[cfg(target_arch = "x86_64")]
    fn get_or_compress_base(&self, base_bytes: &[u8], key: &BaseKey) -> Result<Vec<u8>> {
        // Try loading compressed base from gz SHM.
        if let Some((fd, len)) = initramfs::shm_open_gz(key.0) {
            let mut buf = vec![0u8; len];
            unsafe {
                let ptr = libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ,
                    libc::MAP_SHARED,
                    fd,
                    0,
                );
                if ptr != libc::MAP_FAILED {
                    std::ptr::copy_nonoverlapping(ptr as *const u8, buf.as_mut_ptr(), len);
                    libc::munmap(ptr, len);
                    initramfs::shm_close_fd(fd);
                    tracing::debug!(bytes = len, "gz_base cache hit (shm)");
                    return Ok(buf);
                }
            }
            initramfs::shm_close_fd(fd);
        }

        // Compress and store.
        use flate2::write::GzEncoder;
        use std::io::Write;
        let mut enc = GzEncoder::new(Vec::new(), flate2::Compression::fast());
        enc.write_all(base_bytes).context("gzip base")?;
        let gz = enc.finish().context("finish gzip base")?;

        if let Err(e) = initramfs::shm_store_gz(key.0, &gz) {
            tracing::warn!("shm_store_gz: {e:#}");
        }
        Ok(gz)
    }

    /// Try to COW-overlay the compressed base from gz SHM into guest
    /// memory. Returns true on success.
    #[cfg(target_arch = "x86_64")]
    fn try_cow_overlay(
        &self,
        vm: &kvm::SttKvm,
        key: &BaseKey,
        expected_len: usize,
        load_addr: u64,
    ) -> bool {
        let Some((fd, len)) = initramfs::shm_open_gz(key.0) else {
            return false;
        };
        if len != expected_len {
            initramfs::shm_close_fd(fd);
            return false;
        }
        let Ok(host_addr) = vm.guest_mem.get_host_address(GuestAddress(load_addr)) else {
            initramfs::shm_close_fd(fd);
            return false;
        };
        let ok = unsafe { initramfs::cow_overlay(host_addr, len, fd) };
        initramfs::shm_close_fd(fd);
        ok
    }

    /// Initialize the SHM ring buffer header at `shm_base` in guest memory.
    fn init_shm_region(&self, guest_mem: &GuestMemoryMmap, shm_base: u64) -> Result<()> {
        let shm_size = self.shm_size as usize;
        let header = shm_ring::ShmRingHeader {
            magic: shm_ring::SHM_RING_MAGIC,
            version: shm_ring::SHM_RING_VERSION,
            capacity: (shm_size - shm_ring::HEADER_SIZE) as u32,
            _pad: 0,
            write_ptr: 0,
            read_ptr: 0,
            drops: 0,
        };
        guest_mem
            .write_slice(
                zerocopy::IntoBytes::as_bytes(&header),
                GuestAddress(shm_base),
            )
            .context("write SHM header")
    }

    /// Write cmdline, boot params, SHM header, and topology tables to guest memory.
    #[cfg(target_arch = "x86_64")]
    fn setup_memory(
        &self,
        vm: &kvm::SttKvm,
        kernel_result: &boot::KernelLoadResult,
        initramfs_handle: Option<JoinHandle<Result<(BaseRef, BaseKey)>>>,
    ) -> Result<()> {
        let (initrd_addr, initrd_size) = match initramfs_handle {
            Some(handle) => self.join_and_load_initramfs(vm, handle, INITRD_ADDR)?,
            None => (None, None),
        };

        let mut cmdline = concat!(
            "console=ttyS0 nomodules mitigations=off ",
            "no_timer_check clocksource=kvm-clock ",
            "random.trust_cpu=on swiotlb=noforce ",
            "i8042.noaux i8042.nomux i8042.nopnp i8042.dumbkbd ",
            "pci=off reboot=k panic=-1 iomem=relaxed nokaslr lockdown=none ",
            "sysctl.kernel.unprivileged_bpf_disabled=0 ",
            "sysctl.kernel.sched_schedstats=1",
        )
        .to_string();
        let verbose = std::env::var("STT_VERBOSE")
            .map(|v| v == "1")
            .unwrap_or(false)
            || std::env::var("RUST_BACKTRACE").is_ok_and(|v| v == "1" || v == "full");
        if verbose {
            cmdline.push_str(" earlyprintk=serial loglevel=7");
        } else {
            cmdline.push_str(" loglevel=0");
        }
        if self.init_binary.is_some() {
            cmdline.push_str(" rdinit=/init");
        }
        if self.shm_size > 0 {
            let mem_size = (self.memory_mb as u64) << 20;
            let shm_base = mem_size - self.shm_size;
            cmdline.push_str(&format!(
                " STT_SHM_BASE={:#x} STT_SHM_SIZE={:#x}",
                shm_base, self.shm_size
            ));
        }
        if !self.cmdline_extra.is_empty() {
            cmdline.push(' ');
            cmdline.push_str(&self.cmdline_extra);
        }

        let t0 = Instant::now();
        boot::write_cmdline(&vm.guest_mem, &cmdline)?;
        boot::write_boot_params(
            &vm.guest_mem,
            &cmdline,
            self.memory_mb,
            initrd_addr,
            initrd_size,
            kernel_result.setup_header.as_ref(),
            self.shm_size,
        )?;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "cmdline_boot_params");

        // Initialize SHM ring buffer.
        let t0 = Instant::now();
        if self.shm_size > 0 {
            let mem_size = (self.memory_mb as u64) << 20;
            let shm_base = mem_size - self.shm_size;
            self.init_shm_region(&vm.guest_mem, shm_base)?;
        }
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "shm_ring_init");

        // Write topology tables (MP table + ACPI MADT).
        let t0 = Instant::now();
        mptable::setup_mptable(&vm.guest_mem, &self.topology)?;
        let _acpi_layout =
            acpi::setup_acpi(&vm.guest_mem, &self.topology, self.memory_mb, self.shm_size)?;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "mptable_acpi");

        Ok(())
    }

    /// Configure BSP and AP vCPUs.
    #[cfg(target_arch = "x86_64")]
    fn setup_vcpus(&self, vm: &kvm::SttKvm, kernel_entry: u64) -> Result<()> {
        let t0 = Instant::now();
        boot::setup_sregs(&vm.guest_mem, &vm.vcpus[0], vm.split_irqchip)?;
        boot::setup_regs(&vm.vcpus[0], kernel_entry)?;
        boot::setup_fpu(&vm.vcpus[0])?;
        boot::setup_msrs(&vm.vcpus[0], None)?;
        boot::setup_lapic(&vm.vcpus[0], true)?;
        vm.vcpus[0]
            .set_mp_state(kvm_bindings::kvm_mp_state {
                mp_state: kvm_bindings::KVM_MP_STATE_RUNNABLE,
            })
            .context("set BSP mp_state")?;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "bsp_setup");

        let t0 = Instant::now();
        for vcpu in &vm.vcpus[1..] {
            boot::setup_fpu(vcpu)?;
            boot::setup_lapic(vcpu, false)?;
            vcpu.set_mp_state(kvm_bindings::kvm_mp_state {
                mp_state: kvm_bindings::KVM_MP_STATE_UNINITIALIZED,
            })
            .context("set AP mp_state")?;
        }
        tracing::debug!(
            elapsed_us = t0.elapsed().as_micros(),
            ap_count = vm.vcpus.len().saturating_sub(1),
            "ap_setup"
        );

        Ok(())
    }

    /// Spawn threads and run the BSP. Returns all state needed for
    /// `collect_results`.
    #[allow(clippy::type_complexity)]
    fn run_vm(&self, start: Instant, mut vm: kvm::SttKvm) -> Result<VmRunState> {
        let com1 = Arc::new(PiMutex::new(console::Serial::new(console::COM1_BASE)));
        let com2 = Arc::new(PiMutex::new(console::Serial::new(console::COM2_BASE)));

        // Register serial EventFds with KVM's irqfd for interrupt-driven TX.
        #[cfg(target_arch = "x86_64")]
        if !vm.split_irqchip {
            vm.vm_fd
                .register_irqfd(com1.lock().irq_evt(), console::COM1_IRQ)
                .context("register COM1 irqfd")?;
            vm.vm_fd
                .register_irqfd(com2.lock().irq_evt(), console::COM2_IRQ)
                .context("register COM2 irqfd")?;
        }
        #[cfg(target_arch = "aarch64")]
        {
            vm.vm_fd
                .register_irqfd(com1.lock().irq_evt(), kvm::SERIAL_IRQ)
                .context("register serial irqfd")?;
            vm.vm_fd
                .register_irqfd(com2.lock().irq_evt(), kvm::SERIAL2_IRQ)
                .context("register serial2 irqfd")?;
        }

        let kill = Arc::new(AtomicBool::new(false));

        let has_immediate_exit = vm.has_immediate_exit;
        let mut vcpus = std::mem::take(&mut vm.vcpus);
        let mut bsp = vcpus.remove(0);

        // Build per-vCPU pin targets from the stored pinning plan.
        // Index i holds the host CPU for vCPU i. BSP is index 0.
        let pin_targets: Vec<Option<usize>> = if let Some(ref plan) = self.pinning_plan {
            let total = self.topology.total_cpus() as usize;
            let mut targets = vec![None; total];
            for &(vcpu_id, host_cpu) in &plan.assignments {
                if (vcpu_id as usize) < total {
                    targets[vcpu_id as usize] = Some(host_cpu);
                }
            }
            targets
        } else {
            Vec::new()
        };

        // AP pin targets: indices 1..N.
        let ap_pins: Vec<Option<usize>> = if pin_targets.len() > 1 {
            pin_targets[1..].to_vec()
        } else {
            vec![None; vcpus.len()]
        };

        let ap_threads =
            self.spawn_ap_threads(vcpus, has_immediate_exit, &com1, &com2, &kill, &ap_pins)?;

        // Pin BSP (runs on current thread, pid=0 means calling thread).
        if let Some(Some(host_cpu)) = pin_targets.first() {
            pin_current_thread(*host_cpu, "BSP (vCPU 0)");
        }
        if self.performance_mode {
            set_rt_priority(1, "BSP (vCPU 0)");
        }

        // Collect vCPU pthread_t handles for monitor stall detection.
        // BSP runs on the current thread; APs have spawned threads.
        let vcpu_pthreads = {
            let mut pts = Vec::with_capacity(1 + ap_threads.len());
            pts.push(unsafe { libc::pthread_self() } as libc::pthread_t);
            for vt in &ap_threads {
                pts.push(vt.handle.as_pthread_t() as libc::pthread_t);
            }
            pts
        };

        let monitor_handle = self.start_monitor(&vm, &kill, start, vcpu_pthreads)?;

        // BPF map write thread: sleeps, discovers a BPF map, writes a value.
        let bpf_write_handle = self.start_bpf_map_write(&vm, &kill)?;

        // Run BSP on this thread.
        register_vcpu_signal_handler();
        let timeout = self.timeout;

        // Watchdog thread.
        let bsp_ie = if has_immediate_exit {
            Some(ImmediateExitHandle::from_vcpu(&mut bsp))
        } else {
            None
        };
        let bsp_tid = unsafe { libc::pthread_self() };
        let bsp_done = Arc::new(AtomicBool::new(false));
        let bsp_done_for_wd = bsp_done.clone();
        let kill_for_watchdog = kill.clone();
        let rt_watchdog = self.performance_mode;
        let wd_service_cpu = self.pinning_plan.as_ref().and_then(|p| p.service_cpu);
        let watchdog = std::thread::Builder::new()
            .name("vmm-watchdog".into())
            .spawn(move || {
                if let Some(cpu) = wd_service_cpu {
                    pin_current_thread(cpu, "watchdog");
                }
                if rt_watchdog {
                    set_rt_priority(2, "watchdog");
                }
                let deadline = Instant::now() + timeout;
                eprintln!("watchdog: started, timeout={timeout:?}");
                loop {
                    if bsp_done_for_wd.load(Ordering::Acquire) {
                        eprintln!("watchdog: BSP done, returning");
                        return;
                    }
                    if kill_for_watchdog.load(Ordering::Acquire) || Instant::now() >= deadline {
                        // Either an AP set kill or timeout expired.
                        // Re-check bsp_done: if the BSP already exited its
                        // run loop, the VcpuFd (and kvm_run mmap backing
                        // bsp_ie) may be dropped. Writing to ie after drop
                        // is a use-after-free.
                        if bsp_done_for_wd.load(Ordering::Acquire) {
                            eprintln!("watchdog: BSP already done, returning");
                            return;
                        }
                        let reason = if Instant::now() >= deadline {
                            "timeout expired"
                        } else {
                            "kill set by AP"
                        };
                        eprintln!("watchdog: {reason}, kicking BSP");
                        if let Some(ref ie) = bsp_ie {
                            ie.set(1);
                            std::sync::atomic::fence(Ordering::Release);
                        }
                        unsafe {
                            libc::pthread_kill(bsp_tid, vcpu_signal());
                        }
                        eprintln!("watchdog: BSP kicked");
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
            })
            .context("spawn watchdog thread")?;

        // BSP run loop.
        eprintln!("BSP: entering run loop");
        let (exit_code, timed_out) =
            self.run_bsp_loop(&mut bsp, &com1, &com2, &kill, has_immediate_exit, start);
        bsp_done.store(true, Ordering::Release);
        eprintln!("BSP: exited run loop, code={exit_code} timed_out={timed_out}");

        // Join the watchdog before dropping `bsp`. The watchdog holds an
        // ImmediateExitHandle pointing into bsp's kvm_run mmap. If bsp is
        // dropped first, the watchdog may write to unmapped memory.
        let _ = watchdog.join();

        Ok(VmRunState {
            exit_code,
            timed_out,
            ap_threads,
            monitor_handle,
            bpf_write_handle,
            com1,
            com2,
            kill,
            vm,
        })
    }

    /// Spawn AP vCPU threads. Each thread optionally pins itself to a
    /// host CPU from `pin_targets` (indexed by AP order, 0-based).
    fn spawn_ap_threads(
        &self,
        vcpus: Vec<kvm_ioctls::VcpuFd>,
        has_immediate_exit: bool,
        com1: &Arc<PiMutex<console::Serial>>,
        com2: &Arc<PiMutex<console::Serial>>,
        kill: &Arc<AtomicBool>,
        pin_targets: &[Option<usize>],
    ) -> Result<Vec<VcpuThread>> {
        let mut ap_threads: Vec<VcpuThread> = Vec::new();
        for (i, mut vcpu) in vcpus.into_iter().enumerate() {
            let ie_handle = if has_immediate_exit {
                Some(ImmediateExitHandle::from_vcpu(&mut vcpu))
            } else {
                None
            };
            let kill_clone = kill.clone();
            let com1_clone = com1.clone();
            let com2_clone = com2.clone();
            let exited = Arc::new(AtomicBool::new(false));
            let exited_clone = exited.clone();
            let pin_cpu = pin_targets.get(i).copied().flatten();

            let rt = self.performance_mode;
            let handle = std::thread::Builder::new()
                .name(format!("vcpu-{}", i + 1))
                .spawn(move || {
                    register_vcpu_signal_handler();
                    // Pin inside the thread using pid=0 (calling thread).
                    if let Some(cpu) = pin_cpu {
                        pin_current_thread(cpu, &format!("vCPU {}", i + 1));
                    }
                    if rt {
                        set_rt_priority(1, &format!("vCPU {}", i + 1));
                    }
                    vcpu_run_loop_unified(&mut vcpu, &com1_clone, &com2_clone, &kill_clone);
                    exited_clone.store(true, Ordering::Release);
                    vcpu
                })
                .with_context(|| format!("spawn vCPU {} thread", i + 1))?;

            ap_threads.push(VcpuThread {
                handle,
                exited,
                immediate_exit: ie_handle,
            });
        }
        Ok(ap_threads)
    }

    /// Start the monitor thread if vmlinux is available.
    #[allow(clippy::type_complexity)]
    fn start_monitor(
        &self,
        vm: &kvm::SttKvm,
        kill: &Arc<AtomicBool>,
        start: Instant,
        vcpu_pthreads: Vec<libc::pthread_t>,
    ) -> Result<Option<JoinHandle<(Vec<monitor::MonitorSample>, shm_ring::ShmDrainResult)>>> {
        let Some(vmlinux) = find_vmlinux(&self.kernel) else {
            return Ok(None);
        };
        let offsets = monitor::btf_offsets::KernelOffsets::from_vmlinux(&vmlinux);
        let symbols = monitor::symbols::KernelSymbols::from_vmlinux(&vmlinux);

        let (Ok(offsets), Ok(symbols)) = (offsets, symbols) else {
            return Ok(None);
        };

        let host_base = vm
            .guest_mem
            .get_host_address(GuestAddress(DRAM_BASE))
            .unwrap();
        let mem_size = (self.memory_mb as u64) << 20;
        let mem = monitor::reader::GuestMem::new(host_base, mem_size);
        let num_cpus = self.topology.total_cpus();
        let kill_clone = kill.clone();
        let dump_trigger =
            self.monitor_thresholds
                .filter(|_| self.shm_size > 0)
                .map(|thresholds| {
                    let shm_base_pa = mem_size - self.shm_size;
                    monitor::reader::DumpTrigger {
                        shm_base_pa,
                        thresholds,
                    }
                });

        let hz = monitor::guest_kernel_hz(Some(&self.kernel));
        let watchdog_jiffies = self.watchdog_timeout_s.map(|s| s * hz);
        let preemption_threshold_ns = monitor::vcpu_preemption_threshold_ns(Some(&self.kernel));
        let rt_monitor = self.performance_mode;
        let service_cpu = self.pinning_plan.as_ref().and_then(|p| p.service_cpu);
        let shm_base_pa = if self.shm_size > 0 {
            Some(DRAM_BASE + mem_size - self.shm_size)
        } else {
            None
        };

        let vmlinux_clone = vmlinux.clone();

        let handle = std::thread::Builder::new()
            .name("vmm-monitor".into())
            .spawn(move || {
                if let Some(cpu) = service_cpu {
                    pin_current_thread(cpu, "monitor");
                }
                if rt_monitor {
                    set_rt_priority(2, "monitor");
                }
                std::thread::sleep(Duration::from_millis(500));

                let page_offset = monitor::symbols::resolve_page_offset(&mem, &symbols);

                // __per_cpu_offset is a kernel data symbol: use text mapping.
                let pco_pa = monitor::symbols::text_kva_to_pa(symbols.per_cpu_offset);
                let offsets_arr = unsafe {
                    monitor::symbols::read_per_cpu_offsets(mem.base_ptr(), pco_pa, num_cpus)
                };
                // Per-CPU addresses (runqueues + offset) are in the
                // direct mapping: use PAGE_OFFSET.
                let rq_pas =
                    monitor::symbols::compute_rq_pas(symbols.runqueues, &offsets_arr, page_offset);

                let watchdog_override = watchdog_jiffies.and_then(|jiffies| {
                    symbols.scx_watchdog_timeout.map(|kva| {
                        let pa = monitor::symbols::text_kva_to_pa(kva);
                        monitor::reader::WatchdogOverride { pa, jiffies }
                    })
                });
                if watchdog_jiffies.is_some() && watchdog_override.is_none() {
                    tracing::warn!("scx_watchdog_timeout symbol not found in vmlinux",);
                }

                let event_pcpu_pas = symbols
                    .scx_root
                    .zip(offsets.event_offsets.as_ref())
                    .and_then(|(scx_root_kva, ev)| {
                        // scx_root is a kernel data symbol: use text mapping.
                        let scx_root_pa = monitor::symbols::text_kva_to_pa(scx_root_kva);
                        monitor::reader::resolve_event_pcpu_pas(
                            &mem,
                            scx_root_pa,
                            ev,
                            &offsets_arr,
                            page_offset,
                        )
                    });

                let vcpu_timing = monitor::reader::VcpuTiming {
                    pthreads: vcpu_pthreads,
                };

                // Wait for the guest to signal slot 1 (scheduler loaded)
                // before discovering struct_ops programs. Without this,
                // discovery races with scheduler BPF program registration.
                if let Some(base) = shm_base_pa {
                    let slot_pa = base + shm_ring::SIGNAL_SLOT_BASE as u64 + 1;
                    let deadline = start + Duration::from_secs(30);
                    while std::time::Instant::now() < deadline {
                        if kill_clone.load(std::sync::atomic::Ordering::Relaxed) {
                            break;
                        }
                        if mem.read_u8(slot_pa, 0) != 0 {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(100));
                    }
                }

                // Discover struct_ops programs for per-cycle stats.
                let prog_stats_ctx =
                    monitor::btf_offsets::BpfProgOffsets::from_vmlinux(&vmlinux_clone)
                        .ok()
                        .and_then(|prog_offsets| {
                            let prog_idr_kva = symbols.prog_idr?;
                            let cached = monitor::bpf_prog::discover_struct_ops_stats(
                                &mem,
                                monitor::symbols::text_kva_to_pa(symbols.init_top_pgt.unwrap_or(0)),
                                page_offset,
                                prog_idr_kva,
                                &prog_offsets,
                                monitor::symbols::resolve_pgtable_l5(&mem, &symbols),
                            );
                            if cached.is_empty() {
                                return None;
                            }
                            Some(monitor::reader::ProgStatsCtx {
                                cached,
                                per_cpu_offsets: offsets_arr.clone(),
                                page_offset,
                                offsets: prog_offsets,
                            })
                        });

                let mon_cfg = monitor::reader::MonitorConfig {
                    event_pcpu_pas: event_pcpu_pas.as_deref(),
                    dump_trigger: dump_trigger.as_ref(),
                    watchdog_override: watchdog_override.as_ref(),
                    vcpu_timing: Some(&vcpu_timing),
                    preemption_threshold_ns,
                    shm_base_pa,
                    prog_stats_ctx: prog_stats_ctx.as_ref(),
                    page_offset,
                };
                monitor::reader::monitor_loop(
                    &mem,
                    &rq_pas,
                    &offsets,
                    Duration::from_millis(100),
                    &kill_clone,
                    start,
                    &mon_cfg,
                )
            })
            .context("spawn monitor thread")?;

        Ok(Some(handle))
    }

    /// Spawn a thread that writes to a BPF map in guest memory.
    ///
    /// Event-driven sequence:
    /// 1. Poll `BpfMapAccessorOwned::new` until kernel page tables are up
    /// 2. Poll `find_map` until the scheduler's BPF maps are discoverable
    /// 3. Write the crash value and signal guest via SHM slot 0
    fn start_bpf_map_write(
        &self,
        vm: &kvm::SttKvm,
        kill: &Arc<AtomicBool>,
    ) -> Result<Option<JoinHandle<()>>> {
        let Some(ref params) = self.bpf_map_write else {
            return Ok(None);
        };
        let Some(vmlinux) = find_vmlinux(&self.kernel) else {
            eprintln!("bpf_map_write: vmlinux not found, skipping");
            return Ok(None);
        };

        let host_base = vm
            .guest_mem
            .get_host_address(GuestAddress(DRAM_BASE))
            .unwrap();
        let mem_size = (self.memory_mb as u64) << 20;
        let mem = monitor::reader::GuestMem::new(host_base, mem_size);
        let kill_clone = kill.clone();
        let params = params.clone();
        let shm_size = self.shm_size;

        let handle = std::thread::Builder::new()
            .name("bpf-map-write".into())
            .spawn(move || {
                if kill_clone.load(Ordering::Acquire) {
                    return;
                }

                // Phase 1: wait for BPF map accessor (kernel booted, page tables up).
                let phase1_deadline =
                    std::time::Instant::now() + std::time::Duration::from_secs(30);
                let accessor = loop {
                    match monitor::bpf_map::BpfMapAccessorOwned::new(&mem, &vmlinux) {
                        Ok(a) => break a,
                        Err(e) => {
                            if kill_clone.load(Ordering::Acquire) {
                                return;
                            }
                            if std::time::Instant::now() >= phase1_deadline {
                                eprintln!("bpf_map_write: accessor init timed out: {e:#}");
                                return;
                            }
                            std::thread::sleep(std::time::Duration::from_millis(200));
                        }
                    }
                };

                // Phase 2: poll find_map until the scheduler's BPF maps are discoverable.
                let retry_deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
                let mut attempt = 0u32;
                let map_info = loop {
                    attempt += 1;
                    if let Some(info) = accessor.find_map(&params.map_name_suffix) {
                        break info;
                    }
                    if kill_clone.load(Ordering::Acquire) {
                        eprintln!("bpf_map_write: VM exited during map search");
                        return;
                    }
                    if std::time::Instant::now() >= retry_deadline {
                        eprintln!(
                            "bpf_map_write: map *{} not found after {} attempts",
                            params.map_name_suffix, attempt,
                        );
                        return;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(200));
                };
                eprintln!(
                    "bpf_map_write: map '{}' found after {} attempts",
                    map_info.name, attempt,
                );

                // Phase 3: write the crash trigger and signal the guest.
                // The guest blocks on wait_for(0, timeout) before starting
                // the scenario. Signaling immediately after the BPF write
                // ensures the crash is active when the scenario runs.

                // Log all maps for diagnostic visibility.
                let all_maps = accessor.maps();
                eprintln!(
                    "bpf_map_write: maps() found {} map(s): [{}]",
                    all_maps.len(),
                    all_maps
                        .iter()
                        .map(|m| format!("{}(type={})", m.name, m.map_type))
                        .collect::<Vec<_>>()
                        .join(", "),
                );

                // Read before write for round-trip verification.
                let before = accessor.read_value_u32(&map_info, params.offset);
                let ok = accessor.write_value_u32(&map_info, params.offset, params.value);
                let after = accessor.read_value_u32(&map_info, params.offset);

                eprintln!(
                    "bpf_map_write: map '{}' write={} (value={} offset={} before={:?} after={:?})",
                    map_info.name, ok, params.value, params.offset, before, after,
                );

                // Signal the guest that the BPF map write is done.
                if ok && shm_size > 0 {
                    let shm_base = mem.size() - shm_size;
                    shm_ring::signal_guest(&mem, shm_base, 0);
                    eprintln!("bpf_map_write: signaled slot 0");
                }
            })
            .context("spawn bpf-map-write thread")?;

        Ok(Some(handle))
    }

    /// Unified BSP KVM_RUN loop. Returns (exit_code, timed_out).
    ///
    /// Handles arch-specific I/O dispatch (port I/O on x86_64, MMIO on
    /// aarch64) and HLT semantics (x86_64 BSP: check kill + continue;
    /// aarch64 BSP: shutdown).
    fn run_bsp_loop(
        &self,
        bsp: &mut kvm_ioctls::VcpuFd,
        com1: &Arc<PiMutex<console::Serial>>,
        com2: &Arc<PiMutex<console::Serial>>,
        kill: &Arc<AtomicBool>,
        has_immediate_exit: bool,
        start: Instant,
    ) -> (i32, bool) {
        let timeout = self.timeout;
        let mut exit_code: i32 = -1;

        loop {
            if start.elapsed() > timeout {
                return (exit_code, true);
            }
            if kill.load(Ordering::Acquire) {
                break;
            }

            match bsp.run() {
                Ok(mut exit) => {
                    // HLT is role-dependent: aarch64 BSP = shutdown,
                    // x86_64 BSP = check kill + continue.
                    if matches!(exit, VcpuExit::Hlt) {
                        #[cfg(target_arch = "aarch64")]
                        {
                            exit_code = 0;
                            break;
                        }
                        #[cfg(target_arch = "x86_64")]
                        {
                            if kill.load(Ordering::Acquire) {
                                break;
                            }
                            continue;
                        }
                    }
                    match classify_exit(com1, com2, &mut exit) {
                        Some(ExitAction::Continue) | None => {}
                        Some(ExitAction::Shutdown) => {
                            exit_code = 0;
                            break;
                        }
                        Some(ExitAction::Fatal(reason)) => {
                            if let Some(r) = reason {
                                tracing::error!(r, "BSP VM entry failed");
                            } else {
                                tracing::error!("BSP internal error");
                            }
                            break;
                        }
                    }
                }
                Err(e) => {
                    if e.errno() == libc::EAGAIN || e.errno() == libc::EINTR {
                        if has_immediate_exit {
                            bsp.set_kvm_immediate_exit(0);
                        }
                        continue;
                    }
                    tracing::error!(%e, "BSP run failed");
                    break;
                }
            }
        }

        (exit_code, false)
    }

    /// Shutdown threads and collect output.
    fn collect_results(&self, start: Instant, run: VmRunState) -> Result<VmResult> {
        let mut exit_code = run.exit_code;
        let timed_out = run.timed_out;
        run.kill.store(true, Ordering::Release);

        // Kick APs still in KVM_RUN, then join. Skip APs that already
        // exited — their VcpuFd (and kvm_run mmap) may be dropped, so
        // writing to ImmediateExitHandle would hit unmapped memory.
        for vt in &run.ap_threads {
            if !vt.exited.load(Ordering::Acquire) {
                vt.kick();
            }
        }
        for vt in run.ap_threads {
            vt.wait_for_exit(Duration::from_secs(5));
            let _ = vt.handle.join();
        }

        let (monitor_report, mid_flight_drain) =
            match run.monitor_handle.and_then(|h| h.join().ok()) {
                Some((samples, drain)) => {
                    let preemption_threshold_ns =
                        monitor::vcpu_preemption_threshold_ns(Some(&self.kernel));
                    let summary = monitor::MonitorSummary::from_samples_with_threshold(
                        &samples,
                        preemption_threshold_ns,
                    );
                    let report = monitor::MonitorReport {
                        samples,
                        summary,
                        preemption_threshold_ns,
                    };
                    (Some(report), drain)
                }
                None => (None, shm_ring::ShmDrainResult::default()),
            };

        if let Some(h) = run.bpf_write_handle {
            let _ = h.join();
        }

        // Merge mid-flight drain (from monitor thread) with post-mortem
        // drain (snapshot after VM exit). Mid-flight entries come first
        // since they were drained during execution.
        let (shm_data, stimulus_events) = if self.shm_size > 0 {
            let mem_size = (self.memory_mb as u64) << 20;
            let shm_base = DRAM_BASE + mem_size - self.shm_size;
            let shm_size = self.shm_size as usize;
            let mut shm_buf = vec![0u8; shm_size];
            run.vm
                .guest_mem
                .read_slice(&mut shm_buf, GuestAddress(shm_base))
                .context("read SHM region")?;
            let post_mortem = shm_ring::shm_drain(&shm_buf, 0);

            let mut all_entries = mid_flight_drain.entries;
            all_entries.extend(post_mortem.entries);
            let drops = mid_flight_drain.drops.max(post_mortem.drops);

            let events: Vec<shm_ring::StimulusEvent> = all_entries
                .iter()
                .filter(|e| e.msg_type == shm_ring::MSG_TYPE_STIMULUS && e.crc_ok)
                .filter_map(|e| shm_ring::StimulusEvent::from_payload(&e.payload))
                .collect();
            (
                Some(shm_ring::ShmDrainResult {
                    entries: all_entries,
                    drops,
                }),
                events,
            )
        } else {
            (None, Vec::new())
        };

        let app_output = run.com2.lock().output();
        let console_output = run.com1.lock().output();

        // Extract exit code: SHM (primary), COM2 sentinel (fallback).
        let shm_exit = shm_data.as_ref().and_then(|d| {
            d.entries
                .iter()
                .rev()
                .find(|e| e.msg_type == shm_ring::MSG_TYPE_EXIT && e.crc_ok && e.payload.len() == 4)
                .map(|e| i32::from_ne_bytes(e.payload[..4].try_into().unwrap()))
        });
        if let Some(code) = shm_exit {
            exit_code = code;
        } else if let Some(line) = app_output
            .lines()
            .rev()
            .find(|l| l.starts_with("STT_EXIT="))
            && let Ok(code) = line.trim_start_matches("STT_EXIT=").trim().parse::<i32>()
        {
            exit_code = code;
        }

        // Collect BPF verifier stats from host-side memory reads.
        let verifier_stats = self.collect_verifier_stats(&run.vm);

        Ok(VmResult {
            success: !timed_out && exit_code == 0,
            exit_code,
            duration: start.elapsed(),
            timed_out,
            output: app_output,
            stderr: console_output,
            monitor: monitor_report,
            shm_data,
            stimulus_events,
            verifier_stats,
            kvm_stats: None,
        })
    }

    /// Read BPF verifier stats from guest memory after VM exit.
    ///
    /// Enumerates struct_ops programs in the kernel's `prog_idr` and
    /// reads `bpf_prog_aux->verified_insns` for each.
    fn collect_verifier_stats(
        &self,
        vm: &kvm::SttKvm,
    ) -> Vec<monitor::bpf_prog::ProgVerifierStats> {
        let vmlinux = match find_vmlinux(&self.kernel) {
            Some(v) => v,
            None => return Vec::new(),
        };
        let host_base = match vm.guest_mem.get_host_address(GuestAddress(DRAM_BASE)) {
            Ok(ptr) => ptr,
            Err(_) => return Vec::new(),
        };
        let mem_size = (self.memory_mb as u64) << 20;
        let mem = monitor::reader::GuestMem::new(host_base, mem_size);
        let kernel = match monitor::guest::GuestKernel::new(&mem, &vmlinux) {
            Ok(k) => k,
            Err(_) => return Vec::new(),
        };
        let accessor =
            match monitor::bpf_prog::BpfProgAccessor::from_guest_kernel(&kernel, &vmlinux) {
                Ok(a) => a,
                Err(_) => return Vec::new(),
            };
        accessor.struct_ops_progs()
    }
}

// ---------------------------------------------------------------------------
// aarch64 run path — MMIO-based serial, FDT instead of ACPI
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
impl SttVm {
    fn setup_memory_aarch64(
        &self,
        vm: &kvm::SttKvm,
        _kernel_result: &boot::KernelLoadResult,
        initramfs_handle: Option<JoinHandle<Result<(BaseRef, BaseKey)>>>,
    ) -> Result<()> {
        // Build initramfs data, then place it at the high end of DRAM
        // (just below FDT) to avoid conflicts with early kernel allocations.
        let (initrd_addr, initrd_size) = match initramfs_handle {
            Some(handle) => {
                let (base, _key) = handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("initramfs-resolve thread panicked"))??;
                let base_bytes: &[u8] = base.as_ref();
                let enable_refs: Vec<&str> =
                    self.sched_enable_cmds.iter().map(|s| s.as_str()).collect();
                let disable_refs: Vec<&str> =
                    self.sched_disable_cmds.iter().map(|s| s.as_str()).collect();
                let suffix = initramfs::build_suffix_full(
                    base_bytes.len(),
                    &self.run_args,
                    &self.sched_args,
                    &enable_refs,
                    &disable_refs,
                )?;
                // Gzip-compress the cpio data. The kernel's initramfs
                // handler tries decompression first, so this avoids the
                // "invalid magic" fallback path and its free_initrd_mem
                // edge cases on some kernel builds.
                let compressed = {
                    use flate2::write::GzEncoder;
                    use std::io::Write;
                    let mut enc = GzEncoder::new(Vec::new(), flate2::Compression::fast());
                    enc.write_all(base_bytes).context("gzip initramfs base")?;
                    enc.write_all(&suffix).context("gzip initramfs suffix")?;
                    enc.finish().context("finish gzip initramfs")?
                };
                let total_size = compressed.len() as u64;
                let load_addr = aarch64_initrd_addr(self.memory_mb, self.shm_size, total_size);
                initramfs::load_initramfs_parts(&vm.guest_mem, &[&compressed], load_addr)?;
                (Some(load_addr), Some(total_size as u32))
            }
            None => (None, None),
        };

        let mut cmdline = concat!(
            "console=ttyS0 ",
            "nomodules mitigations=off ",
            "random.trust_cpu=on swiotlb=noforce ",
            "pci=off reboot=k panic=-1 nokaslr lockdown=none ",
            "sysctl.kernel.unprivileged_bpf_disabled=0 ",
            "sysctl.kernel.sched_schedstats=1 ",
            "kfence.sample_interval=0",
        )
        .to_string();
        let verbose = std::env::var("STT_VERBOSE")
            .map(|v| v == "1")
            .unwrap_or(false)
            || std::env::var("RUST_BACKTRACE").is_ok_and(|v| v == "1" || v == "full");
        if verbose {
            cmdline.push_str(" earlycon=uart,mmio,0x09000000 loglevel=7");
        } else {
            cmdline.push_str(" loglevel=0");
        }
        if self.init_binary.is_some() {
            cmdline.push_str(" rdinit=/init");
        }
        if self.shm_size > 0 {
            let mem_size = (self.memory_mb as u64) << 20;
            let shm_base = kvm::DRAM_START + mem_size - self.shm_size;
            cmdline.push_str(&format!(
                " STT_SHM_BASE={:#x} STT_SHM_SIZE={:#x}",
                shm_base, self.shm_size
            ));
        }
        if !self.cmdline_extra.is_empty() {
            cmdline.push(' ');
            cmdline.push_str(&self.cmdline_extra);
        }

        let t0 = Instant::now();
        // Validate length only — aarch64 kernel reads cmdline from FDT
        // /chosen/bootargs, not from a fixed memory address.
        boot::validate_cmdline(&cmdline)?;

        // Generate and load the FDT.
        let fdt_addr = aarch64::fdt::fdt_address(self.memory_mb, self.shm_size);
        let mpidrs =
            aarch64::topology::read_mpidrs(&vm.vcpus).context("read vCPU MPIDRs for FDT")?;
        let dtb = aarch64::fdt::create_fdt(
            &mpidrs,
            self.memory_mb,
            &cmdline,
            initrd_addr,
            initrd_size,
            self.shm_size,
        )
        .context("create FDT")?;
        vm.guest_mem
            .write_slice(&dtb, GuestAddress(fdt_addr))
            .context("write FDT to guest memory")?;
        tracing::debug!(
            elapsed_us = t0.elapsed().as_micros(),
            fdt_addr,
            fdt_len = dtb.len(),
            "cmdline_fdt",
        );

        // Initialize SHM ring buffer.
        let t0 = Instant::now();
        if self.shm_size > 0 {
            let mem_size = (self.memory_mb as u64) << 20;
            let shm_base = kvm::DRAM_START + mem_size - self.shm_size;
            self.init_shm_region(&vm.guest_mem, shm_base)?;
        }
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "shm_ring_init");

        Ok(())
    }

    fn setup_vcpus_aarch64(&self, vm: &kvm::SttKvm, kernel_entry: u64) -> Result<()> {
        let t0 = Instant::now();
        let fdt_addr = aarch64::fdt::fdt_address(self.memory_mb, self.shm_size);
        boot::setup_regs(&vm.vcpus[0], kernel_entry, fdt_addr)?;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "bsp_setup");
        // APs start powered off via PSCI — no register setup needed.
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// aarch64 MMIO dispatch — serial over MMIO
// ---------------------------------------------------------------------------

/// Dispatch an MMIO write to serial devices.
/// Returns `true` if the caller should exit (shutdown detected).
#[cfg(target_arch = "aarch64")]
fn dispatch_mmio_write(
    com1: &PiMutex<console::Serial>,
    com2: &PiMutex<console::Serial>,
    addr: u64,
    data: &[u8],
) -> bool {
    if let Some(offset) = mmio_serial_offset(addr, kvm::SERIAL_MMIO_BASE) {
        if let Some(&byte) = data.first() {
            com1.lock().inner_write(offset, byte);
        }
    } else if let Some(offset) = mmio_serial_offset(addr, kvm::SERIAL2_MMIO_BASE)
        && let Some(&byte) = data.first()
    {
        com2.lock().inner_write(offset, byte);
    }
    false
}

/// Dispatch an MMIO read from serial devices.
#[cfg(target_arch = "aarch64")]
fn dispatch_mmio_read(
    com1: &PiMutex<console::Serial>,
    com2: &PiMutex<console::Serial>,
    addr: u64,
    data: &mut [u8],
) {
    if let Some(offset) = mmio_serial_offset(addr, kvm::SERIAL_MMIO_BASE) {
        if let Some(first) = data.first_mut() {
            *first = com1.lock().inner_read(offset);
        }
    } else if let Some(offset) = mmio_serial_offset(addr, kvm::SERIAL2_MMIO_BASE) {
        if let Some(first) = data.first_mut() {
            *first = com2.lock().inner_read(offset);
        }
    } else {
        // Unknown MMIO: return 0xFF for reads (missing device).
        for b in data.iter_mut() {
            *b = 0xff;
        }
    }
}

/// Compute register offset for an MMIO address within a serial region.
#[cfg(target_arch = "aarch64")]
fn mmio_serial_offset(addr: u64, base: u64) -> Option<u8> {
    let size = kvm::SERIAL_MMIO_SIZE;
    if addr >= base && addr < base + size {
        Some((addr - base) as u8)
    } else {
        None
    }
}

/// Unified per-vCPU KVM_RUN loop for AP threads.
///
/// HLT on APs: check kill + continue on both arches (KVM delivers
/// interrupts to wake the vCPU). Shutdown sets the kill flag so all
/// other vCPUs exit.
fn vcpu_run_loop_unified(
    vcpu: &mut kvm_ioctls::VcpuFd,
    com1: &Arc<PiMutex<console::Serial>>,
    com2: &Arc<PiMutex<console::Serial>>,
    kill: &Arc<AtomicBool>,
) {
    loop {
        if kill.load(Ordering::Acquire) {
            break;
        }

        match vcpu.run() {
            Ok(mut exit) => {
                if matches!(exit, VcpuExit::Hlt) {
                    if kill.load(Ordering::Acquire) {
                        break;
                    }
                    continue;
                }
                match classify_exit(com1, com2, &mut exit) {
                    Some(ExitAction::Continue) | None => {}
                    Some(ExitAction::Shutdown) => {
                        kill.store(true, Ordering::Release);
                        break;
                    }
                    Some(ExitAction::Fatal(_)) => break,
                }
            }
            Err(e) => {
                if e.errno() == libc::EINTR || e.errno() == libc::EAGAIN {
                    vcpu.set_kvm_immediate_exit(0);
                    if kill.load(Ordering::Acquire) {
                        break;
                    }
                    continue;
                }
                if kill.load(Ordering::Acquire) {
                    break;
                }
            }
        }

        if kill.load(Ordering::Acquire) {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// I/O dispatch — shared between BSP and AP run loops
// ---------------------------------------------------------------------------

const KVM_SYSTEM_EVENT_SHUTDOWN: u32 = 1;
const KVM_SYSTEM_EVENT_RESET: u32 = 2;

/// Classified vCPU exit action from `classify_exit`.
enum ExitAction {
    /// Continue running (I/O handled, etc.).
    Continue,
    /// Clean shutdown (system reset, VcpuExit::Shutdown, etc.).
    Shutdown,
    /// Fatal error. `Some(reason)` for FailEntry, `None` for InternalError.
    Fatal(Option<u64>),
}

/// Classify a VcpuExit into an ExitAction, dispatching arch-specific I/O.
///
/// Returns `None` for HLT (role-dependent) and unknown exits (continue).
/// Takes the exit by mutable reference so IoIn/MmioRead data buffers
/// can be written back.
fn classify_exit(
    com1: &PiMutex<console::Serial>,
    com2: &PiMutex<console::Serial>,
    exit: &mut VcpuExit,
) -> Option<ExitAction> {
    match exit {
        #[cfg(target_arch = "x86_64")]
        VcpuExit::IoOut(port, data) => {
            if dispatch_io_out(com1, com2, *port, data) {
                Some(ExitAction::Shutdown)
            } else {
                Some(ExitAction::Continue)
            }
        }
        #[cfg(target_arch = "x86_64")]
        VcpuExit::IoIn(port, data) => {
            dispatch_io_in(com1, com2, *port, data);
            Some(ExitAction::Continue)
        }
        #[cfg(target_arch = "aarch64")]
        VcpuExit::MmioWrite(addr, data) => {
            if dispatch_mmio_write(com1, com2, *addr, data) {
                Some(ExitAction::Shutdown)
            } else {
                Some(ExitAction::Continue)
            }
        }
        #[cfg(target_arch = "aarch64")]
        VcpuExit::MmioRead(addr, data) => {
            dispatch_mmio_read(com1, com2, *addr, data);
            Some(ExitAction::Continue)
        }
        VcpuExit::Hlt => None,
        VcpuExit::Shutdown => Some(ExitAction::Shutdown),
        VcpuExit::SystemEvent(event_type, _) => {
            if *event_type == KVM_SYSTEM_EVENT_SHUTDOWN || *event_type == KVM_SYSTEM_EVENT_RESET {
                Some(ExitAction::Shutdown)
            } else {
                Some(ExitAction::Continue)
            }
        }
        VcpuExit::FailEntry(reason, _cpu) => Some(ExitAction::Fatal(Some(*reason))),
        VcpuExit::InternalError => Some(ExitAction::Fatal(None)),
        #[cfg(target_arch = "x86_64")]
        VcpuExit::MmioRead(_addr, data) => {
            for b in data.iter_mut() {
                *b = 0xff;
            }
            Some(ExitAction::Continue)
        }
        #[cfg(target_arch = "x86_64")]
        VcpuExit::MmioWrite(_addr, _data) => Some(ExitAction::Continue),
        _ => None,
    }
}

/// I8042 ports and commands — minimal emulation for x86 guest reboot.
/// The kernel's default reboot method (`reboot=k`) writes CMD_RESET_CPU
/// (0xFE) to the i8042 command port (0x64).
#[cfg(target_arch = "x86_64")]
const I8042_DATA_PORT: u16 = 0x60;
#[cfg(target_arch = "x86_64")]
const I8042_CMD_PORT: u16 = 0x64;
#[cfg(target_arch = "x86_64")]
const I8042_CMD_RESET_CPU: u8 = 0xFE;

/// Dispatch an I/O out to serial ports or system devices.
/// Returns `true` if the caller should exit (system reset detected).
#[cfg(target_arch = "x86_64")]
fn dispatch_io_out(
    com1: &PiMutex<console::Serial>,
    com2: &PiMutex<console::Serial>,
    port: u16,
    data: &[u8],
) -> bool {
    // I8042 reset: kernel writes 0xFE to port 0x64 during reboot.
    if port == I8042_CMD_PORT && data.first() == Some(&I8042_CMD_RESET_CPU) {
        return true;
    }
    // Only lock the matching serial port based on port range.
    if (console::COM1_BASE..console::COM1_BASE + 8).contains(&port) {
        com1.lock().handle_out(port, data);
    } else if (console::COM2_BASE..console::COM2_BASE + 8).contains(&port) {
        com2.lock().handle_out(port, data);
    }
    false
}

/// Dispatch an I/O in from serial ports or system devices.
/// Handles i8042 reads to satisfy the kernel's keyboard probe.
#[cfg(target_arch = "x86_64")]
fn dispatch_io_in(
    com1: &PiMutex<console::Serial>,
    com2: &PiMutex<console::Serial>,
    port: u16,
    data: &mut [u8],
) {
    match port {
        // I8042 status: return 0 (no data, buffer empty).
        I8042_CMD_PORT => {
            if let Some(b) = data.first_mut() {
                *b = 0;
            }
        }
        // I8042 data: return 0 (no keypress).
        I8042_DATA_PORT => {
            if let Some(b) = data.first_mut() {
                *b = 0;
            }
        }
        // Only lock the matching serial port based on port range.
        p if (console::COM1_BASE..console::COM1_BASE + 8).contains(&p) => {
            com1.lock().handle_in(port, data);
        }
        p if (console::COM2_BASE..console::COM2_BASE + 8).contains(&p) => {
            com2.lock().handle_in(port, data);
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// vCPU run loop — Firecracker/CH hybrid pattern
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// vmlinux discovery
// ---------------------------------------------------------------------------

/// Find the vmlinux ELF next to a kernel image path.
///
/// On x86_64, checks the bzImage's parent directory and, if the path
/// looks like `<root>/arch/x86/boot/bzImage`, checks `<root>/vmlinux`.
#[cfg(target_arch = "x86_64")]
pub(crate) fn find_vmlinux(kernel_path: &Path) -> Option<PathBuf> {
    let dir = kernel_path.parent()?;
    let candidate = dir.join("vmlinux");
    if candidate.exists() {
        return Some(candidate);
    }
    // kernel_path is typically <root>/arch/x86/boot/bzImage
    if let Ok(root) = dir.join("../../..").canonicalize() {
        let candidate = root.join("vmlinux");
        if candidate.exists() {
            return Some(candidate);
        }
    }
    // CI/distro kernel: try /usr/lib/debug/boot/vmlinux-<version>
    if let Some(name) = kernel_path.file_name().and_then(|n| n.to_str()) {
        let version = name.strip_prefix("vmlinuz-").unwrap_or(name);
        let debug = PathBuf::from(format!("/usr/lib/debug/boot/vmlinux-{version}"));
        if debug.exists() {
            return Some(debug);
        }
    }
    None
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn find_vmlinux(kernel_path: &Path) -> Option<PathBuf> {
    let dir = kernel_path.parent()?;
    let candidate = dir.join("vmlinux");
    if candidate.exists() {
        return Some(candidate);
    }
    // kernel_path is typically <root>/arch/arm64/boot/Image
    if let Ok(root) = dir.join("../../..").canonicalize() {
        let candidate = root.join("vmlinux");
        if candidate.exists() {
            return Some(candidate);
        }
    }
    // Distro kernel: extract version from vmlinuz-<version> and check
    // /boot/vmlinux-<version> or /lib/modules/<version>/build/vmlinux.
    if let Some(name) = kernel_path.file_name().and_then(|n| n.to_str()) {
        let version = name.strip_prefix("vmlinuz-").unwrap_or(name);
        let boot = PathBuf::from(format!("/boot/vmlinux-{version}"));
        if boot.exists() {
            return Some(boot);
        }
        let modules = PathBuf::from(format!("/lib/modules/{version}/build/vmlinux"));
        if modules.exists() {
            return Some(modules);
        }
    }
    // kernel_path may be /lib/modules/<version>/vmlinuz — extract version
    // from the parent directory name.
    if let Some(parent_name) = dir.file_name().and_then(|n| n.to_str()) {
        let build = dir.join("build/vmlinux");
        if build.exists() {
            return Some(build);
        }
        let boot = PathBuf::from(format!("/boot/vmlinux-{parent_name}"));
        if boot.exists() {
            return Some(boot);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

pub struct SttVmBuilder {
    kernel: Option<PathBuf>,
    init_binary: Option<PathBuf>,
    scheduler_binary: Option<PathBuf>,
    run_args: Vec<String>,
    sched_args: Vec<String>,
    topology: Topology,
    memory_mb: u32,
    cmdline_extra: String,
    timeout: Duration,
    shm_size: u64,
    monitor_thresholds: Option<crate::monitor::MonitorThresholds>,
    watchdog_timeout_s: Option<u64>,
    bpf_map_write: Option<BpfMapWriteParams>,
    performance_mode: bool,
    sched_enable_cmds: Vec<String>,
    sched_disable_cmds: Vec<String>,
}

impl Default for SttVmBuilder {
    fn default() -> Self {
        SttVmBuilder {
            kernel: None,
            init_binary: None,
            scheduler_binary: None,
            run_args: Vec::new(),
            sched_args: Vec::new(),
            topology: Topology {
                sockets: 1,
                cores_per_socket: 1,
                threads_per_core: 1,
            },
            memory_mb: 256,
            cmdline_extra: String::new(),
            timeout: Duration::from_secs(60),
            shm_size: 0,
            monitor_thresholds: None,
            watchdog_timeout_s: Some(4),
            bpf_map_write: None,
            performance_mode: false,
            sched_enable_cmds: Vec::new(),
            sched_disable_cmds: Vec::new(),
        }
    }
}

impl SttVmBuilder {
    pub fn kernel(mut self, path: impl Into<PathBuf>) -> Self {
        self.kernel = Some(path.into());
        self
    }

    pub fn init_binary(mut self, path: impl Into<PathBuf>) -> Self {
        self.init_binary = Some(path.into());
        self
    }

    pub fn scheduler_binary(mut self, path: impl Into<PathBuf>) -> Self {
        self.scheduler_binary = Some(path.into());
        self
    }

    pub fn run_args(mut self, args: &[String]) -> Self {
        self.run_args = args.to_vec();
        self
    }

    #[allow(dead_code)]
    pub fn sched_args(mut self, args: &[String]) -> Self {
        self.sched_args = args.to_vec();
        self
    }

    #[allow(dead_code)]
    pub fn kernel_dir(mut self, path: impl Into<PathBuf>) -> Self {
        let dir: PathBuf = path.into();
        #[cfg(target_arch = "x86_64")]
        {
            self.kernel = Some(dir.join("arch/x86/boot/bzImage"));
        }
        #[cfg(target_arch = "aarch64")]
        {
            self.kernel = Some(dir.join("arch/arm64/boot/Image"));
        }
        self
    }

    pub fn topology(mut self, sockets: u32, cores: u32, threads: u32) -> Self {
        self.topology = Topology {
            sockets,
            cores_per_socket: cores,
            threads_per_core: threads,
        };
        self
    }

    pub fn memory_mb(mut self, mb: u32) -> Self {
        self.memory_mb = mb;
        self
    }

    #[allow(dead_code)]
    pub fn cmdline(mut self, extra: &str) -> Self {
        self.cmdline_extra = extra.to_string();
        self
    }

    pub fn timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }

    #[allow(dead_code)]
    pub fn shm_size(mut self, bytes: u64) -> Self {
        self.shm_size = bytes;
        self
    }

    #[allow(dead_code)]
    pub fn monitor_thresholds(mut self, thresholds: crate::monitor::MonitorThresholds) -> Self {
        self.monitor_thresholds = Some(thresholds);
        self
    }

    #[allow(dead_code)]
    pub fn watchdog_timeout_s(mut self, seconds: u64) -> Self {
        self.watchdog_timeout_s = Some(seconds);
        self
    }

    #[allow(dead_code)]
    pub fn bpf_map_write(mut self, map_name_suffix: &str, offset: usize, value: u32) -> Self {
        self.bpf_map_write = Some(BpfMapWriteParams {
            map_name_suffix: map_name_suffix.to_string(),
            offset,
            value,
        });
        self
    }

    /// Enable performance mode: vCPU pinning to host LLCs,
    /// hugepage-backed guest memory, KVM_HINTS_REALTIME CPUID
    /// hint (disables PV spinlocks, PV TLB flush, PV sched_yield;
    /// enables haltpoll cpuidle), and PAUSE + HLT VM exit disabling
    /// via KVM_CAP_X86_DISABLE_EXITS. HLT disable falls back to
    /// PAUSE-only when mitigate_smt_rsb is active on the host.
    /// KVM_CAP_HALT_POLL is skipped (guest haltpoll cpuidle disables
    /// host halt polling via MSR_KVM_POLL_CONTROL). Validated at
    /// build time -- oversubscription returns `ResourceContention`,
    /// insufficient hugepages is a warning.
    #[allow(dead_code)]
    pub fn performance_mode(mut self, enabled: bool) -> Self {
        self.performance_mode = enabled;
        self
    }

    pub fn sched_enable_cmds(mut self, cmds: &[&str]) -> Self {
        self.sched_enable_cmds = cmds.iter().map(|s| s.to_string()).collect();
        self
    }

    pub fn sched_disable_cmds(mut self, cmds: &[&str]) -> Self {
        self.sched_disable_cmds = cmds.iter().map(|s| s.to_string()).collect();
        self
    }

    pub fn build(mut self) -> Result<SttVm> {
        let (pinning_plan, mbind_nodes, cpu_locks) = if self.performance_mode {
            let (plan, host_topo) = self.validate_performance_mode()?;
            let pinned_cpus: Vec<usize> = plan.assignments.iter().map(|a| a.1).collect();
            let nodes = host_topo.numa_nodes_for_cpus(&pinned_cpus);
            // Perf VMs already hold CPU locks via PinningPlan.locks.
            (Some(plan), nodes, Vec::new())
        } else {
            let total_cpus = self.topology.total_cpus() as usize;
            let host_topo = host_topology::HostTopology::from_sysfs().ok();
            let host_cpus = host_topo
                .as_ref()
                .map(|h| h.total_cpus())
                .unwrap_or(total_cpus);
            let deadline = crate::test_support::resource_deadline();
            let locks = host_topology::acquire_cpu_locks(
                total_cpus,
                host_cpus,
                deadline,
                host_topo.as_ref(),
            )?;
            (None, Vec::new(), locks)
        };

        let kernel = self.kernel.context("kernel path required")?;
        anyhow::ensure!(kernel.exists(), "kernel not found: {}", kernel.display());
        let t = &self.topology;
        anyhow::ensure!(t.sockets > 0, "sockets must be > 0");
        anyhow::ensure!(t.cores_per_socket > 0, "cores_per_socket must be > 0");
        anyhow::ensure!(t.threads_per_core > 0, "threads_per_core must be > 0");
        if let Some(ref bin) = self.init_binary
            && !bin.starts_with("/proc/")
        {
            anyhow::ensure!(bin.exists(), "init binary not found: {}", bin.display());
        }
        if let Some(ref bin) = self.scheduler_binary {
            anyhow::ensure!(
                bin.exists(),
                "scheduler binary not found: {}",
                bin.display()
            );
        }

        Ok(SttVm {
            kernel,
            init_binary: self.init_binary,
            scheduler_binary: self.scheduler_binary,
            run_args: self.run_args,
            sched_args: self.sched_args,
            topology: self.topology,
            memory_mb: self.memory_mb,
            cmdline_extra: self.cmdline_extra,
            timeout: self.timeout,
            shm_size: self.shm_size,
            monitor_thresholds: self.monitor_thresholds,
            watchdog_timeout_s: self.watchdog_timeout_s,
            bpf_map_write: self.bpf_map_write,
            performance_mode: self.performance_mode,
            pinning_plan,
            mbind_nodes,
            cpu_locks,
            sched_enable_cmds: self.sched_enable_cmds,
            sched_disable_cmds: self.sched_disable_cmds,
        })
    }

    /// Validate host resources for performance_mode and compute the
    /// pinning plan. Returns both the plan and the host topology (needed
    /// for NUMA node discovery). Returns `ResourceContention` when the
    /// host lacks CPUs or LLC slots. Warnings are printed for degraded
    /// conditions (hugepages, host load).
    fn validate_performance_mode(
        &mut self,
    ) -> Result<(host_topology::PinningPlan, host_topology::HostTopology)> {
        let host_topo = host_topology::HostTopology::from_sysfs()
            .context("performance_mode: read host topology")?;

        let t = &self.topology;
        let total_vcpus = t.total_cpus();

        // Validate LLC exclusivity: each virtual socket should map to
        // its own physical LLC group. Sum actual per-group CPU counts
        // to handle asymmetric LLCs.
        let llcs_needed = t.sockets as usize;
        let reserved: usize = host_topo
            .llc_groups
            .iter()
            .take(llcs_needed)
            .map(|g| g.cpus.len())
            .sum();
        let total_reserved = reserved + 1; // +1 for service CPU
        if total_reserved > host_topo.total_cpus() {
            return Err(anyhow::Error::new(host_topology::ResourceContention {
                reason: format!(
                    "performance_mode: need {} CPUs ({} across {} LLCs + 1 service) \
                     but only {} host CPUs available",
                    total_reserved,
                    reserved,
                    llcs_needed,
                    host_topo.total_cpus(),
                ),
            }));
        }

        let plan = acquire_slot_with_locks(
            &host_topo,
            t.sockets,
            t.cores_per_socket,
            t.threads_per_core,
        )?;

        // WARN: hugepages.
        let free = host_topology::hugepages_free();
        let needed = host_topology::hugepages_needed(self.memory_mb);
        if free == 0 {
            eprintln!(
                "performance_mode: WARNING: no 2MB hugepages available, \
                 guest memory will use regular pages",
            );
        } else if free < needed {
            eprintln!(
                "performance_mode: WARNING: need {} 2MB hugepages, \
                 only {} free — falling back to regular pages",
                needed, free,
            );
        }

        // WARN: host load.
        if let Some((running, total)) = host_topology::host_load_estimate() {
            let threshold = (total_vcpus as f64 * 0.5) as usize;
            if running > threshold {
                eprintln!(
                    "performance_mode: WARNING: {} processes running on {} CPUs \
                     (threshold {} for {} vCPUs) — results may be noisy",
                    running, total, threshold, total_vcpus,
                );
            }
        }

        Ok((plan, host_topo))
    }
}

/// Try each LLC slot, compute a pinning plan, and acquire resource
/// locks. Polls with 500ms sleep until a slot is acquired or the
/// deadline expires. Returns a `ResourceContention` error when the
/// deadline expires.
fn acquire_slot_with_locks(
    host_topo: &host_topology::HostTopology,
    sockets: u32,
    cores_per_socket: u32,
    threads_per_core: u32,
) -> Result<host_topology::PinningPlan> {
    let deadline = crate::test_support::resource_deadline();
    let num_llcs = host_topo.llc_groups.len();
    let sockets_needed = sockets as usize;
    let max_slots = num_llcs
        .checked_div(sockets_needed)
        .unwrap_or(num_llcs)
        .max(1);
    let llc_mode = host_topology::LlcLockMode::Exclusive;
    let start = std::time::Instant::now();

    loop {
        for slot in 0..max_slots {
            let offset = slot * sockets_needed;
            let llc_indices: Vec<usize> = (offset..offset + sockets_needed).collect();

            let candidate = host_topo
                .compute_pinning(sockets, cores_per_socket, threads_per_core, true, offset)
                .context("performance_mode: topology mapping")?;

            match host_topology::acquire_resource_locks(
                &candidate,
                &llc_indices,
                llc_mode,
                std::time::Duration::ZERO,
            )? {
                host_topology::LockOutcome::Acquired { locks, .. } => {
                    let mut plan = candidate;
                    plan.locks = locks;
                    eprintln!(
                        "performance_mode: reserved LLC slot {} (offset {}, max {})",
                        slot, offset, max_slots,
                    );
                    return Ok(plan);
                }
                host_topology::LockOutcome::Unavailable(_) => continue,
            }
        }

        if start.elapsed() >= deadline {
            return Err(anyhow::Error::new(host_topology::ResourceContention {
                reason: format!(
                    "all {} LLC slots busy — waited {}s",
                    max_slots,
                    deadline.as_secs(),
                ),
            }));
        }
        let jitter = (std::process::id() as u64).wrapping_mul(7) % 100;
        std::thread::sleep(std::time::Duration::from_millis(400 + jitter));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialize boot tests that create full VMs. Running multiple VMs
    /// simultaneously causes signal delivery contention (SIGRTMIN for
    /// vCPU kick) and serial output loss.
    ///
    /// The in-process Mutex handles `cargo test` (threads). The file
    /// lock handles `cargo nextest` (separate processes).
    static BOOT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// RAII guard that holds both the in-process Mutex and a file-based
    /// flock for cross-process serialization (nextest).
    struct BootLockGuard {
        _mutex: std::sync::MutexGuard<'static, ()>,
        _file: std::fs::File,
    }

    impl Drop for BootLockGuard {
        fn drop(&mut self) {
            unsafe {
                libc::flock(
                    std::os::unix::io::AsRawFd::as_raw_fd(&self._file),
                    libc::LOCK_UN,
                );
            }
        }
    }

    fn acquire_boot_lock() -> BootLockGuard {
        let mutex = BOOT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let file = std::fs::File::create("/tmp/stt-vm-boot.lock").expect("create boot lock file");
        unsafe {
            libc::flock(std::os::unix::io::AsRawFd::as_raw_fd(&file), libc::LOCK_EX);
        }
        BootLockGuard {
            _mutex: mutex,
            _file: file,
        }
    }

    #[test]
    fn builder_default() {
        let b = SttVmBuilder::default();
        assert_eq!(b.memory_mb, 256);
        assert_eq!(b.topology.total_cpus(), 1);
    }

    #[test]
    fn builder_topology() {
        let b = SttVmBuilder::default().topology(2, 4, 2);
        assert_eq!(b.topology.total_cpus(), 16);
        assert_eq!(b.topology.sockets, 2);
    }

    #[test]
    fn builder_requires_kernel() {
        let result = SttVmBuilder::default().build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_rejects_missing_kernel() {
        let result = SttVmBuilder::default()
            .kernel("/nonexistent/vmlinuz")
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_chain() {
        let b = SttVmBuilder::default()
            .topology(2, 2, 2)
            .memory_mb(4096)
            .cmdline("root=/dev/sda")
            .timeout(Duration::from_secs(300));
        assert_eq!(b.memory_mb, 4096);
        assert_eq!(b.topology.total_cpus(), 8);
        assert_eq!(b.cmdline_extra, "root=/dev/sda");
        assert_eq!(b.timeout, Duration::from_secs(300));
    }

    #[test]
    fn builder_with_init_binary() {
        let exe = crate::resolve_current_exe().unwrap();
        let b = SttVmBuilder::default().init_binary(&exe);
        assert_eq!(b.init_binary.as_deref(), Some(exe.as_path()));
    }

    #[test]
    fn builder_rejects_missing_init_binary() {
        let result = SttVmBuilder::default()
            .kernel("/nonexistent/vmlinuz")
            .init_binary("/nonexistent/binary")
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_rejects_missing_scheduler_binary() {
        let exe = crate::resolve_current_exe().unwrap();
        let result = SttVmBuilder::default()
            .kernel(&exe)
            .scheduler_binary("/nonexistent/scheduler")
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_run_args() {
        let b = SttVmBuilder::default().run_args(&["run".into(), "--json".into()]);
        assert_eq!(b.run_args, vec!["run", "--json"]);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn builder_kernel_dir_resolves_bzimage() {
        let b = SttVmBuilder::default().kernel_dir("/some/linux");
        assert_eq!(
            b.kernel.as_deref(),
            Some(std::path::Path::new("/some/linux/arch/x86/boot/bzImage"))
        );
    }

    #[test]
    fn vm_result_fields_carry_values() {
        let r = VmResult {
            success: true,
            exit_code: 0,
            duration: Duration::from_secs(5),
            timed_out: false,
            output: "hello world".into(),
            stderr: "boot log".into(),
            monitor: None,
            shm_data: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
        };
        assert!(r.success);
        assert_eq!(r.exit_code, 0);
        assert!(!r.timed_out);
        assert_eq!(r.duration, Duration::from_secs(5));
        assert_eq!(r.output, "hello world");
        assert_eq!(r.stderr, "boot log");
        assert!(r.monitor.is_none());
        assert!(r.shm_data.is_none());
        assert!(r.stimulus_events.is_empty());
        // Verify a failed result carries different values.
        let r2 = VmResult {
            success: false,
            exit_code: 1,
            duration: Duration::from_millis(500),
            timed_out: true,
            output: String::new(),
            stderr: String::new(),
            monitor: None,
            shm_data: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
        };
        assert!(!r2.success);
        assert_eq!(r2.exit_code, 1);
        assert!(r2.timed_out);
        assert_eq!(r2.duration, Duration::from_millis(500));
    }

    #[test]
    fn vcpu_exit_flag_transitions() {
        // AtomicBool used as vcpu exit flag must transition false->true
        // and the store must be visible to a subsequent load.
        let exited = Arc::new(AtomicBool::new(false));
        assert!(
            !exited.load(Ordering::Acquire),
            "initial state must be false"
        );
        // Simulate vcpu exit: another thread sets the flag.
        let exited_clone = Arc::clone(&exited);
        let handle = std::thread::spawn(move || {
            exited_clone.store(true, Ordering::Release);
        });
        handle.join().unwrap();
        assert!(
            exited.load(Ordering::Acquire),
            "flag must be true after cross-thread store"
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn ap_mp_state_set_correctly() {
        let topo = Topology {
            sockets: 2,
            cores_per_socket: 2,
            threads_per_core: 1,
        };
        let vm = kvm::SttKvm::new(topo, 128, false).unwrap();
        for vcpu in &vm.vcpus[1..] {
            let state = vcpu.get_mp_state().unwrap();
            assert_eq!(
                state.mp_state,
                kvm_bindings::KVM_MP_STATE_UNINITIALIZED,
                "AP should default to UNINITIALIZED"
            );
        }
    }

    #[test]
    fn vcpu_signal_is_sigrtmin() {
        let sig = vcpu_signal();
        assert!(sig >= libc::SIGRTMIN(), "signal should be >= SIGRTMIN");
        assert!(sig <= libc::SIGRTMAX(), "signal should be <= SIGRTMAX");
    }

    /// Boot a real kernel and verify it produces console output.
    /// No initramfs — the kernel boots to panic, which is enough to
    /// confirm KVM, kernel loading, and serial console all work.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn boot_kernel_produces_output() {
        let _lock = acquire_boot_lock();
        let Some(kernel) = crate::find_kernel() else {
            return;
        };

        let vm = SttVm::builder()
            .kernel(&kernel)
            .topology(1, 1, 1)
            .memory_mb(256)
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap();
        let result = vm.run().unwrap();
        assert!(
            result.stderr.contains("Linux") || result.stderr.contains("Booting"),
            "kernel console should contain boot messages"
        );
    }

    /// Boot with SMP topology and verify kernel detects multiple CPUs.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn boot_kernel_smp_topology() {
        let _lock = acquire_boot_lock();
        let Some(kernel) = crate::find_kernel() else {
            return;
        };

        let vm = SttVm::builder()
            .kernel(&kernel)
            .topology(2, 2, 1) // 4 CPUs
            .memory_mb(256)
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap();
        let result = vm.run().unwrap();
        assert!(!result.stderr.is_empty(), "no console output from SMP boot");
    }

    /// Benchmark: measure VM boot time to kernel panic (no init = fastest path).
    /// The kernel boots, finds no initramfs, panics. The panic timestamp
    /// IS the boot time. With `panic=-1`, the kernel calls
    /// `emergency_restart()` which triggers an I8042 reset (port 0x64,
    /// 0xFE via `reboot=k`), returning to userspace.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn bench_boot_time() {
        let _lock = acquire_boot_lock();
        let Some(kernel) = crate::find_kernel() else {
            return;
        };

        for (label, sockets, cores, threads, mem) in
            [("1cpu", 1, 1, 1, 256), ("4cpu", 2, 2, 1, 512)]
        {
            let start = Instant::now();
            let vm = SttVm::builder()
                .kernel(&kernel)
                .topology(sockets, cores, threads)
                .memory_mb(mem)
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap();
            let setup = start.elapsed();
            let result = vm.run().unwrap();
            // Extract kernel timestamp from last line (e.g. "[    0.189300] Kernel panic")
            let boot_ms = result
                .stderr
                .lines()
                .rev()
                .find(|l| l.contains("Kernel panic") || l.contains("end Kernel panic"))
                .and_then(|l| {
                    l.trim()
                        .strip_prefix('[')
                        .and_then(|s| s.split(']').next())
                        .and_then(|s| s.trim().parse::<f64>().ok())
                })
                .map(|s| (s * 1000.0) as u64)
                .unwrap_or(0);
            eprintln!(
                "BENCH {label}: setup={:.0}ms kernel_boot={boot_ms}ms wall={:.0}ms timed_out={}",
                setup.as_millis(),
                result.duration.as_millis(),
                result.timed_out,
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn kvm_has_immediate_exit_cap() {
        let topo = Topology {
            sockets: 1,
            cores_per_socket: 1,
            threads_per_core: 1,
        };
        let vm = kvm::SttKvm::new(topo, 64, false).unwrap();
        // KVM_CAP_IMMEDIATE_EXIT has been available since Linux 4.12.
        assert!(
            vm.has_immediate_exit,
            "KVM_CAP_IMMEDIATE_EXIT should be available on modern kernels"
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn immediate_exit_handle_set_clear() {
        let topo = Topology {
            sockets: 1,
            cores_per_socket: 1,
            threads_per_core: 1,
        };
        let mut vm = kvm::SttKvm::new(topo, 64, false).unwrap();
        let handle = ImmediateExitHandle::from_vcpu(&mut vm.vcpus[0]);

        // Initial state should be 0.
        assert_eq!(
            vm.vcpus[0].get_kvm_run().immediate_exit,
            0,
            "immediate_exit should start at 0"
        );

        // Set via handle, verify via VcpuFd.
        handle.set(1);
        assert_eq!(
            vm.vcpus[0].get_kvm_run().immediate_exit,
            1,
            "handle.set(1) should be visible via get_kvm_run()"
        );

        // Clear via VcpuFd, verify.
        vm.vcpus[0].set_kvm_immediate_exit(0);
        assert_eq!(
            vm.vcpus[0].get_kvm_run().immediate_exit,
            0,
            "set_kvm_immediate_exit(0) should clear the flag"
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn immediate_exit_handle_cross_vcpu() {
        let topo = Topology {
            sockets: 1,
            cores_per_socket: 2,
            threads_per_core: 1,
        };
        let mut vm = kvm::SttKvm::new(topo, 64, false).unwrap();
        let h0 = ImmediateExitHandle::from_vcpu(&mut vm.vcpus[0]);
        let h1 = ImmediateExitHandle::from_vcpu(&mut vm.vcpus[1]);

        // Setting one vCPU's handle should not affect the other.
        h0.set(1);
        assert_eq!(vm.vcpus[0].get_kvm_run().immediate_exit, 1);
        assert_eq!(
            vm.vcpus[1].get_kvm_run().immediate_exit,
            0,
            "setting vcpu0 handle should not affect vcpu1"
        );

        h1.set(1);
        assert_eq!(vm.vcpus[1].get_kvm_run().immediate_exit, 1);

        // Clear both.
        h0.set(0);
        h1.set(0);
        assert_eq!(vm.vcpus[0].get_kvm_run().immediate_exit, 0);
        assert_eq!(vm.vcpus[1].get_kvm_run().immediate_exit, 0);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn vcpu_thread_kick_sets_immediate_exit() {
        let topo = Topology {
            sockets: 1,
            cores_per_socket: 1,
            threads_per_core: 1,
        };
        let mut vm = kvm::SttKvm::new(topo, 64, false).unwrap();
        let ie = ImmediateExitHandle::from_vcpu(&mut vm.vcpus[0]);

        ie.set(1);
        std::sync::atomic::fence(Ordering::Release);
        assert_eq!(
            vm.vcpus[0].get_kvm_run().immediate_exit,
            1,
            "kick pattern should set immediate_exit=1"
        );

        vm.vcpus[0].set_kvm_immediate_exit(0);
        assert_eq!(vm.vcpus[0].get_kvm_run().immediate_exit, 0);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn find_vmlinux_from_bzimage_path() {
        // Create a temp dir simulating <root>/arch/x86/boot/bzImage with vmlinux at <root>.
        let tmp = std::env::temp_dir().join("stt-find-vmlinux-test");
        let boot_dir = tmp.join("arch/x86/boot");
        std::fs::create_dir_all(&boot_dir).unwrap();
        let vmlinux = tmp.join("vmlinux");
        std::fs::write(&vmlinux, b"ELF").unwrap();
        let bzimage = boot_dir.join("bzImage");
        std::fs::write(&bzimage, b"kernel").unwrap();

        let found = find_vmlinux(&bzimage);
        assert_eq!(found, Some(vmlinux));

        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn find_vmlinux_sibling() {
        // vmlinux in the same directory as the kernel image.
        let tmp = std::env::temp_dir().join("stt-find-vmlinux-sibling");
        std::fs::create_dir_all(&tmp).unwrap();
        let vmlinux = tmp.join("vmlinux");
        std::fs::write(&vmlinux, b"ELF").unwrap();
        let kernel = tmp.join("bzImage");
        std::fs::write(&kernel, b"kernel").unwrap();

        let found = find_vmlinux(&kernel);
        assert_eq!(found, Some(vmlinux));

        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn find_vmlinux_bare_filename() {
        // A bare filename — parent is "" so no vmlinux sibling found.
        assert_eq!(find_vmlinux(Path::new("vmlinuz")), None);
    }

    #[test]
    fn find_vmlinux_root_parent() {
        // /vmlinuz has parent "/" — no vmlinux there (or if there is, fine).
        // The function should not panic.
        let result = find_vmlinux(Path::new("/vmlinuz"));
        // /vmlinux almost certainly doesn't exist; if it does, that's still valid.
        if !Path::new("/vmlinux").exists() {
            assert_eq!(result, None);
        }
    }

    #[test]
    fn builder_rejects_zero_sockets() {
        let exe = crate::resolve_current_exe().unwrap();
        let result = SttVmBuilder::default()
            .kernel(&exe)
            .topology(0, 2, 2)
            .build();
        assert!(result.is_err(), "sockets=0 should fail validation");
    }

    #[test]
    fn builder_rejects_zero_cores() {
        let exe = crate::resolve_current_exe().unwrap();
        let result = SttVmBuilder::default()
            .kernel(&exe)
            .topology(2, 0, 2)
            .build();
        assert!(result.is_err(), "cores=0 should fail validation");
    }

    #[test]
    fn builder_rejects_zero_threads() {
        let exe = crate::resolve_current_exe().unwrap();
        let result = SttVmBuilder::default()
            .kernel(&exe)
            .topology(2, 2, 0)
            .build();
        assert!(result.is_err(), "threads=0 should fail validation");
    }

    #[test]
    fn vm_result_without_monitor_has_no_samples() {
        let r = VmResult {
            success: true,
            exit_code: 0,
            duration: Duration::from_secs(1),
            timed_out: false,
            output: "test output".into(),
            stderr: String::new(),
            monitor: None,
            shm_data: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
        };
        assert!(r.monitor.is_none());
        // Output and exit_code must still be accessible.
        assert_eq!(r.output, "test output");
        assert_eq!(r.exit_code, 0);
    }

    #[test]
    fn vm_result_with_monitor_carries_summary() {
        use crate::monitor;
        let summary = monitor::MonitorSummary {
            prog_stats_deltas: None,
            total_samples: 5,
            max_imbalance_ratio: 3.5,
            max_local_dsq_depth: 10,
            stall_detected: true,
            event_deltas: None,
            schedstat_deltas: None,
        };
        let report = monitor::MonitorReport {
            samples: vec![],
            summary: summary.clone(),
            ..Default::default()
        };
        let r = VmResult {
            success: false,
            exit_code: 1,
            duration: Duration::from_millis(500),
            timed_out: true,
            output: String::new(),
            stderr: "kernel panic".into(),
            monitor: Some(report),
            shm_data: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
        };
        let mon = r.monitor.as_ref().unwrap();
        assert_eq!(mon.summary.total_samples, 5);
        assert!((mon.summary.max_imbalance_ratio - 3.5).abs() < f64::EPSILON);
        assert_eq!(mon.summary.max_local_dsq_depth, 10);
        assert!(mon.summary.stall_detected);
        assert!(r.timed_out);
        assert_eq!(r.exit_code, 1);
        assert_eq!(r.stderr, "kernel panic");
    }

    #[test]
    fn find_vmlinux_missing_returns_none() {
        let tmp = std::env::temp_dir().join("stt-find-vmlinux-none");
        std::fs::create_dir_all(&tmp).unwrap();
        let kernel = tmp.join("bzImage");
        std::fs::write(&kernel, b"kernel").unwrap();

        assert_eq!(find_vmlinux(&kernel), None);

        std::fs::remove_dir_all(&tmp).unwrap();
    }

    /// Boot a kernel with vmlinux available and verify the monitor produces samples.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn boot_kernel_with_monitor() {
        let _lock = acquire_boot_lock();
        let Some(kernel) = crate::find_kernel() else {
            return;
        };
        // Monitor needs vmlinux — skip if not present.
        let Some(_vmlinux) = find_vmlinux(&kernel) else {
            return;
        };

        let vm = SttVm::builder()
            .kernel(&kernel)
            .topology(1, 2, 1)
            .memory_mb(256)
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap();
        let result = vm.run().unwrap();
        if let Some(ref report) = result.monitor {
            assert!(
                report.summary.total_samples > 0,
                "monitor should have collected at least one sample"
            );
        }
        // If monitor is None, the kernel/BTF wasn't compatible — not a failure.
    }

    // -- initramfs cache tests --

    #[test]
    fn base_key_same_inputs_match() {
        let exe = crate::resolve_current_exe().unwrap();
        let k1 = BaseKey::new(&exe, None).unwrap();
        let k2 = BaseKey::new(&exe, None).unwrap();
        assert_eq!(k1, k2);
    }

    #[test]
    fn base_key_nonexistent_payload_fails() {
        let result = BaseKey::new(Path::new("/nonexistent/binary"), None);
        assert!(result.is_err());
    }

    #[test]
    fn base_key_different_content_differs() {
        let tmp =
            std::env::temp_dir().join(format!("stt-cache-content-test-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let bin = tmp.join("payload");

        std::fs::write(&bin, b"content_v1").unwrap();
        let k1 = BaseKey::new(&bin, None).unwrap();

        std::fs::write(&bin, b"content_v2").unwrap();
        let k2 = BaseKey::new(&bin, None).unwrap();

        assert_ne!(
            k1, k2,
            "different file content should produce different key"
        );
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn base_key_with_scheduler() {
        let exe = crate::resolve_current_exe().unwrap();
        let k1 = BaseKey::new(&exe, None).unwrap();
        let k2 = BaseKey::new(&exe, Some(&exe)).unwrap();
        assert_ne!(k1, k2, "with vs without scheduler should differ");
    }

    #[test]
    fn hash_file_large_file() {
        let tmp = std::env::temp_dir().join(format!("stt-hash-sample-test-{}", std::process::id()));
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
        let key = BaseKey::new(&exe, None).unwrap();

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

    // -- dispatch_io_out / dispatch_io_in tests --

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dispatch_io_out_i8042_reset_is_shutdown_signal() {
        // The BSP relies on I8042 reset (port 0x64, 0xFE) for shutdown
        // detection instead of VcpuExit::Hlt. Verify that dispatch_io_out
        // returns true for the reset command.
        let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
        let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
        assert!(
            dispatch_io_out(&com1, &com2, I8042_CMD_PORT, &[I8042_CMD_RESET_CPU]),
            "I8042 reset (0xFE to port 0x64) must signal shutdown"
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dispatch_io_out_i8042_non_reset() {
        let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
        let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
        assert!(!dispatch_io_out(&com1, &com2, I8042_CMD_PORT, &[0x00]));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dispatch_io_out_serial_com1() {
        let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
        let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
        // Write 'A' to COM1 THR — should not trigger reset.
        assert!(!dispatch_io_out(&com1, &com2, console::COM1_BASE, b"A"));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dispatch_io_out_serial_com2() {
        let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
        let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
        assert!(!dispatch_io_out(&com1, &com2, console::COM2_BASE, b"B"));
        let output = com2.lock().output();
        assert!(output.contains('B'));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dispatch_io_out_unknown_port() {
        let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
        let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
        assert!(!dispatch_io_out(&com1, &com2, 0x1234, &[0xFF]));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dispatch_io_in_i8042_status() {
        let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
        let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
        let mut data = [0xFFu8; 1];
        dispatch_io_in(&com1, &com2, I8042_CMD_PORT, &mut data);
        assert_eq!(data[0], 0);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dispatch_io_in_i8042_data() {
        let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
        let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
        let mut data = [0xFFu8; 1];
        dispatch_io_in(&com1, &com2, I8042_DATA_PORT, &mut data);
        assert_eq!(data[0], 0);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dispatch_io_in_unknown_port() {
        let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
        let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
        let mut data = [0xFFu8; 1];
        dispatch_io_in(&com1, &com2, 0x1234, &mut data);
        assert_eq!(data[0], 0xFF, "unknown port should not modify data");
    }

    // -- PiMutex tests --

    #[test]
    fn pi_mutex_lock_unlock() {
        let m = PiMutex::new(42u32);
        {
            let mut guard = m.lock();
            assert_eq!(*guard, 42);
            *guard = 99;
        }
        assert_eq!(*m.lock(), 99);
    }

    #[test]
    fn pi_mutex_cross_thread() {
        let m = Arc::new(PiMutex::new(0u32));
        let m2 = m.clone();
        let handle = std::thread::spawn(move || {
            *m2.lock() += 1;
        });
        handle.join().unwrap();
        assert_eq!(*m.lock(), 1);
    }

    // -- builder watchdog_timeout_s --

    #[test]
    fn builder_watchdog_timeout_default() {
        let b = SttVmBuilder::default();
        assert_eq!(b.watchdog_timeout_s, Some(4));
    }

    #[test]
    fn builder_watchdog_timeout_override() {
        let b = SttVmBuilder::default().watchdog_timeout_s(5);
        assert_eq!(b.watchdog_timeout_s, Some(5));
    }

    #[test]
    fn builder_monitor_thresholds_sets() {
        let t = crate::monitor::MonitorThresholds {
            max_imbalance_ratio: 2.0,
            ..Default::default()
        };
        let b = SttVmBuilder::default().monitor_thresholds(t);
        assert!(b.monitor_thresholds.is_some());
    }

    #[test]
    fn builder_shm_size() {
        let b = SttVmBuilder::default().shm_size(65536);
        assert_eq!(b.shm_size, 65536);
    }

    #[test]
    fn builder_sched_args() {
        let b = SttVmBuilder::default().sched_args(&["--enable-borrow".into()]);
        assert_eq!(b.sched_args, vec!["--enable-borrow"]);
    }

    // -- performance_mode builder tests --

    #[test]
    fn builder_performance_mode_default_false() {
        let b = SttVmBuilder::default();
        assert!(!b.performance_mode);
    }

    #[test]
    fn builder_performance_mode_set() {
        let b = SttVmBuilder::default().performance_mode(true);
        assert!(b.performance_mode);
    }

    #[test]
    fn builder_performance_mode_false_no_validation() {
        // performance_mode=false should not trigger validation, even with
        // a topology that exceeds host capacity.
        let exe = crate::resolve_current_exe().unwrap();
        let result = SttVmBuilder::default()
            .kernel(&exe)
            .topology(1, 1, 1)
            .performance_mode(false)
            .build();
        assert!(
            result.is_ok(),
            "performance_mode=false should not validate host topology",
        );
    }

    #[test]
    fn builder_performance_mode_oversubscribed_fails() {
        let exe = crate::resolve_current_exe().unwrap();
        let host_topo = host_topology::HostTopology::from_sysfs().unwrap();
        let too_many = host_topo.total_cpus() as u32 + 1;
        let result = SttVmBuilder::default()
            .kernel(&exe)
            .topology(1, too_many, 1)
            .performance_mode(true)
            .build();
        match result {
            Ok(_) => panic!("oversubscribed topology should fail"),
            Err(e) => {
                let msg = format!("{e}");
                assert!(
                    msg.contains("performance_mode"),
                    "error should mention performance_mode: {msg}",
                );
            }
        }
    }

    #[test]
    fn builder_performance_mode_too_many_sockets_fails() {
        let exe = crate::resolve_current_exe().unwrap();
        let host_topo = host_topology::HostTopology::from_sysfs().unwrap();
        let too_many_sockets = host_topo.llc_groups.len() as u32 + 1;
        // Need total vCPUs + 1 service CPU to fit without oversubscription.
        if (too_many_sockets as usize + 1) <= host_topo.total_cpus() {
            let result = SttVmBuilder::default()
                .kernel(&exe)
                .topology(too_many_sockets, 1, 1)
                .performance_mode(true)
                .build();
            assert!(result.is_err(), "more sockets than LLCs should fail",);
        }
    }

    #[test]
    fn builder_performance_mode_valid_succeeds() {
        let exe = crate::resolve_current_exe().unwrap();
        let host_topo = host_topology::HostTopology::from_sysfs().unwrap();
        // Need 2 vCPUs + 1 service CPU = 3 host CPUs minimum,
        // plus the LLC must have >= 2 cores.
        if host_topo.total_cpus() < 3 {
            return;
        }
        let result = SttVmBuilder::default()
            .kernel(&exe)
            .topology(1, 2, 1)
            .performance_mode(true)
            .build();
        match result {
            Ok(_) => {}
            Err(e)
                if e.downcast_ref::<host_topology::ResourceContention>()
                    .is_some() =>
            {
                // Host lacks resources (e.g. not enough CPUs across
                // LLCs) — skip, not fail.
            }
            Err(e) => panic!("valid topology with performance_mode should build: {e:#}",),
        }
    }

    #[test]
    fn builder_performance_mode_preserves_in_vm() {
        let exe = crate::resolve_current_exe().unwrap();
        let host_topo = host_topology::HostTopology::from_sysfs().unwrap();
        // Need 2 vCPUs + 1 service CPU = 3 host CPUs minimum.
        if host_topo.total_cpus() < 3 {
            return;
        }
        let vm = match SttVmBuilder::default()
            .kernel(&exe)
            .topology(1, 2, 1)
            .performance_mode(true)
            .build()
        {
            Ok(vm) => vm,
            Err(e)
                if e.downcast_ref::<host_topology::ResourceContention>()
                    .is_some() =>
            {
                return;
            }
            Err(e) => panic!("{e:#}"),
        };
        assert!(vm.performance_mode);
    }

    #[test]
    fn builder_performance_mode_false_preserves_in_vm() {
        let exe = crate::resolve_current_exe().unwrap();
        let vm = SttVmBuilder::default()
            .kernel(&exe)
            .topology(1, 1, 1)
            .performance_mode(false)
            .build()
            .unwrap();
        assert!(!vm.performance_mode);
    }

    #[test]
    fn builder_performance_mode_mbind_nodes_populated() {
        let exe = crate::resolve_current_exe().unwrap();
        let host_topo = host_topology::HostTopology::from_sysfs().unwrap();
        if host_topo.total_cpus() < 3 {
            return;
        }
        let vm = SttVmBuilder::default()
            .kernel(&exe)
            .topology(1, 2, 1)
            .performance_mode(true)
            .build();
        if let Ok(vm) = vm {
            assert!(
                !vm.mbind_nodes.is_empty(),
                "mbind_nodes should be populated for performance_mode",
            );
        }
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

    #[test]
    fn pi_mutex_concurrent_increment() {
        let m = Arc::new(PiMutex::new(0u64));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let m = m.clone();
                std::thread::spawn(move || {
                    for _ in 0..1000 {
                        *m.lock() += 1;
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(*m.lock(), 8000);
    }

    #[test]
    fn pi_mutex_protocol_is_inherit() {
        // Verify PTHREAD_PRIO_INHERIT is supported on this system.
        unsafe {
            let mut attr: libc::pthread_mutexattr_t = std::mem::zeroed();
            assert_eq!(libc::pthread_mutexattr_init(&mut attr), 0);
            assert_eq!(
                libc::pthread_mutexattr_setprotocol(&mut attr, libc::PTHREAD_PRIO_INHERIT),
                0,
            );
            let mut protocol: libc::c_int = 0;
            assert_eq!(libc::pthread_mutexattr_getprotocol(&attr, &mut protocol), 0);
            assert_eq!(protocol, libc::PTHREAD_PRIO_INHERIT);
            libc::pthread_mutexattr_destroy(&mut attr);
        }
    }

    // -- RT scheduling tests --

    #[test]
    fn set_rt_priority_applies_when_capable() {
        // Verify set_rt_priority sets SCHED_FIFO when the process has
        // CAP_SYS_NICE. Skip (pass) if not capable.
        let param = libc::sched_param { sched_priority: 1 };
        let rc = unsafe { libc::sched_setscheduler(0, libc::SCHED_FIFO, &param) };
        if rc != 0 {
            // No CAP_SYS_NICE — skip test.
            eprintln!("skipping set_rt_priority test: no CAP_SYS_NICE");
            return;
        }
        // Verify it took effect.
        let policy = unsafe { libc::sched_getscheduler(0) };
        assert_eq!(policy, libc::SCHED_FIFO);
        let mut out_param: libc::sched_param = unsafe { std::mem::zeroed() };
        unsafe { libc::sched_getparam(0, &mut out_param) };
        assert_eq!(out_param.sched_priority, 1);
        // Restore SCHED_OTHER to avoid affecting other tests.
        let restore = libc::sched_param { sched_priority: 0 };
        unsafe { libc::sched_setscheduler(0, libc::SCHED_OTHER, &restore) };
    }

    #[test]
    fn set_rt_priority_warns_without_cap() {
        // Verify set_rt_priority does not panic when called without
        // CAP_SYS_NICE — it should print a warning and continue.
        // This test always passes; it exercises the warning path.
        set_rt_priority(1, "test-thread");
        // If we get here, set_rt_priority didn't panic.
    }

    // -- aarch64 boot tests --

    /// Find an aarch64 kernel suitable for boot tests.
    /// Accepts both raw Image and gzip-compressed vmlinuz — load_kernel
    /// decompresses transparently.
    #[cfg(target_arch = "aarch64")]
    fn find_aarch64_image() -> Option<std::path::PathBuf> {
        crate::find_kernel()
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn boot_kernel_produces_output_aarch64() {
        let _lock = acquire_boot_lock();
        let Some(kernel) = find_aarch64_image() else {
            eprintln!("skipping: no aarch64 Image found (only compressed vmlinuz available)");
            return;
        };

        let vm = SttVm::builder()
            .kernel(&kernel)
            .topology(1, 1, 1)
            .memory_mb(256)
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap();
        let result = vm.run().unwrap();
        assert!(
            result.stderr.contains("Linux") || result.stderr.contains("Booting"),
            "kernel console should contain boot messages, got: {}",
            &result.stderr[..result.stderr.len().min(200)],
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn boot_kernel_smp_topology_aarch64() {
        let _lock = acquire_boot_lock();
        let Some(kernel) = find_aarch64_image() else {
            eprintln!("skipping: no aarch64 Image found");
            return;
        };

        let vm = SttVm::builder()
            .kernel(&kernel)
            .topology(2, 2, 1) // 4 CPUs
            .memory_mb(256)
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap();
        let result = vm.run().unwrap();
        assert!(!result.stderr.is_empty(), "no console output from SMP boot");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn aarch64_kvm_has_immediate_exit() {
        let topo = Topology {
            sockets: 1,
            cores_per_socket: 1,
            threads_per_core: 1,
        };
        let vm = kvm::SttKvm::new(topo, 64, false).unwrap();
        assert!(
            vm.has_immediate_exit,
            "KVM_CAP_IMMEDIATE_EXIT should be available on modern kernels"
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn builder_kernel_dir_resolves_image() {
        let b = SttVmBuilder::default().kernel_dir("/some/linux");
        assert_eq!(
            b.kernel.as_deref(),
            Some(std::path::Path::new("/some/linux/arch/arm64/boot/Image"))
        );
    }
}
