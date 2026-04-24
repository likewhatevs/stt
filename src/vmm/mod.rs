//! Virtual machine monitor for booting Linux kernels in KVM to host
//! scheduler test scenarios.
//!
//! The entry point is [`KtstrVm::builder()`], which returns a
//! [`KtstrVmBuilder`] for configuring the kernel, init binary,
//! virtual topology, memory, host-side performance options, and
//! monitor thresholds. Calling `.build()?.run()?` on the result
//! boots the guest and returns a [`VmResult`] containing exit state,
//! captured console, monitor samples, and drained SHM ring data.
//!
//! See the [VMM architecture
//! page](https://likewhatevs.github.io/ktstr/guide/architecture/vmm.html)
//! for the boot flow and the [Performance Mode
//! page](https://likewhatevs.github.io/ktstr/guide/concepts/performance-mode.html)
//! for the isolation options the builder exposes.

pub mod cgroup_sandbox;
pub mod console;
mod exit_dispatch;
pub mod host_topology;
pub mod initramfs;
mod memory_budget;
pub(crate) mod numa_mem;
mod pi_mutex;
pub(crate) mod rust_init;
pub mod shm_ring;
mod terminal;
pub mod topology;
mod vcpu_panic;
pub(crate) mod virtio_console;
mod vmlinux;

pub(crate) use exit_dispatch::{ExitAction, classify_exit, vcpu_run_loop_unified};
pub(crate) use memory_budget::{MemoryBudget, initramfs_min_memory_mb, read_kernel_init_size};
pub(crate) use pi_mutex::PiMutex;
pub(crate) use terminal::TerminalRawGuard;
pub(crate) use vmlinux::find_vmlinux;

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

/// Create a KVM VM with EINTR retry (up to 5 attempts, exponential backoff).
///
/// KVM_CREATE_VM can return EINTR when a signal arrives mid-ioctl.
/// Retrying with backoff matches the Firecracker pattern.
pub(crate) fn create_vm_with_retry(kvm: &kvm_ioctls::Kvm) -> Result<kvm_ioctls::VmFd> {
    let mut attempts = 0;
    loop {
        match kvm.create_vm() {
            Ok(fd) => break Ok(fd),
            Err(e) if e.errno() == libc::EINTR && attempts < 5 => {
                attempts += 1;
                std::thread::sleep(std::time::Duration::from_micros(1 << attempts));
            }
            Err(e) => break Err(e).context("create VM"),
        }
    }
}

// ---------------------------------------------------------------------------
// Initramfs cache — two-tier: POSIX shm (cross-process) + in-process HashMap
// ---------------------------------------------------------------------------

/// Cache key for base initramfs. Derived from content hashes of the
/// payload binary and its shared libs, plus the optional scheduler
/// binary and its shared libs. Shell mode additionally mixes in a
/// sentinel, include files, and the busybox flag; see [`Self::new`]
/// and [`Self::new_shell`] for per-constructor inputs.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BaseKey(u64);

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
    use std::io::Read;
    let mut f =
        std::fs::File::open(path).with_context(|| format!("open for hash: {}", path.display()))?;
    let mut hasher = SipHasher13::new_with_keys(0, 0);
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
    /// Hashes the payload binary content, payload shared libs, and
    /// the optional scheduler / probe / alloc-worker binary content
    /// and shared libs. Each optional input participates
    /// symmetrically because each changes the bytes written into
    /// the initramfs. Even though probe + alloc-worker are
    /// currently routed through `include_files` (so their content
    /// also appears in the `new_shell` include-hash loop), an
    /// explicit parameter keeps the cache key sensitive to those
    /// inputs regardless of the routing choice — if a future change
    /// moves them back to extras, the key stays correct without a
    /// separate update.
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

enum ShmCreateResult {
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
fn shm_try_create_excl(name: &str) -> ShmCreateResult {
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
fn shm_write_and_release(fd: std::os::fd::OwnedFd, data: &[u8], seg_name: &str) {
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

/// Set the calling thread's CPU mask to the supplied set. Distinct
/// from [`pin_current_thread`]: that one locks a thread to a single
/// CPU (the perf-mode contract), this one constrains a thread to a
/// pool without picking a specific CPU. The kernel picks a runnable
/// CPU from the mask.
///
/// Used by the no-perf + `--llc-cap` path at
/// [`KtstrVmBuilder::build`]: every vCPU thread gets the reserved
/// LLC's CPUs as its mask so the vCPU runs inside the resource
/// budget without fighting the kernel scheduler for a hard pin it
/// doesn't actually need.
///
/// Logs success or warning; does not fail the VM.
fn set_thread_cpumask(cpus: &[usize], label: &str) {
    let mut cpuset = nix::sched::CpuSet::new();
    for &cpu in cpus {
        if let Err(e) = cpuset.set(cpu) {
            eprintln!("no_perf_mode: WARNING: cpuset.set({cpu}) for {label}: {e}");
            return;
        }
    }
    match nix::sched::sched_setaffinity(nix::unistd::Pid::from_raw(0), &cpuset) {
        Ok(()) => eprintln!("no_perf_mode: mask {label} to host CPUs {cpus:?}"),
        Err(e) => eprintln!("no_perf_mode: WARNING: mask {label} to {cpus:?}: {e}"),
    }
}

/// Set the calling thread to SCHED_FIFO at the given priority.
/// Logs success or warning via tracing; does not fail the VM.
///
/// Uses `tracing::info!` / `tracing::warn!` rather than `eprintln!`
/// so the warn-without-CAP_SYS_NICE branch is observable by tests
/// that install a tracing subscriber (e.g. `tracing-test`).
/// Previously `eprintln!` made the warning invisible to any test
/// that didn't fork + redirect fd 2.
fn set_rt_priority(priority: i32, label: &str) {
    let param = libc::sched_param {
        sched_priority: priority,
    };
    let rc = unsafe { libc::sched_setscheduler(0, libc::SCHED_FIFO, &param) };
    if rc == 0 {
        tracing::info!(
            label = label,
            priority = priority,
            "performance_mode: {label} set to SCHED_FIFO priority {priority}",
        );
    } else {
        let err = std::io::Error::last_os_error();
        tracing::warn!(
            label = label,
            priority = priority,
            err = %err,
            "performance_mode: WARNING: SCHED_FIFO for {label}: {err} (need CAP_SYS_NICE)",
        );
    }
}

// ---------------------------------------------------------------------------
// VmResult
// ---------------------------------------------------------------------------

/// Result of a VM execution.
#[derive(Debug)]
pub struct VmResult {
    /// Overall success flag: `true` when the test reported a pass AND
    /// the VM exited cleanly without crash, timeout, or watchdog.
    pub success: bool,
    /// Guest exit code as surfaced through the SHM ring
    /// (`MSG_TYPE_EXIT`) or COM2 sentinel.
    pub exit_code: i32,
    /// Wall-clock duration of the VM run.
    pub duration: Duration,
    /// True when the host hit its watchdog before the guest exited.
    pub timed_out: bool,
    /// Captured guest stdout (and any non-dmesg serial console content).
    pub output: String,
    /// Captured guest stderr (separated from `output` when the guest
    /// reported them distinctly).
    pub stderr: String,
    /// Host-side monitor report: sampled per-CPU state, stall
    /// verdicts, and SCX event deltas. `None` when the monitor did
    /// not run (host-only tests, early VM failure).
    pub monitor: Option<monitor::MonitorReport>,
    /// Data drained from the SHM ring buffer after VM exit.
    pub shm_data: Option<shm_ring::ShmDrainResult>,
    /// Stimulus events extracted from SHM ring entries.
    #[allow(dead_code)]
    pub stimulus_events: Vec<shm_ring::StimulusEvent>,
    /// BPF verifier stats collected from host-side memory reads.
    pub verifier_stats: Vec<monitor::bpf_prog::ProgVerifierStats>,
    /// KVM per-vCPU cumulative stats (requires Linux >= 5.15, x86_64 only).
    pub kvm_stats: Option<KvmStatsTotals>,
    /// Crash message from SHM (MSG_TYPE_CRASH). Reliable delivery via
    /// memcpy unlike serial which truncates large backtraces.
    pub crash_message: Option<String>,
}

impl VmResult {
    /// Minimal "nothing happened" fixture for tests that exercise
    /// code consuming a [`VmResult`] without actually booting a VM
    /// (the sidecar-write tests in `src/test_support/sidecar.rs`
    /// are the primary users). Every field carries the empty /
    /// default / `None` value that `run_vm` would produce for a
    /// VM that launched, exited cleanly with exit code 0, and
    /// produced no telemetry. Tests that need a specific field
    /// override it with a struct-update expression:
    ///
    /// ```ignore
    /// let result = VmResult { success: false, ..VmResult::test_fixture() };
    /// ```
    ///
    /// Gated on `#[cfg(test)]` so the symbol does not appear in
    /// release builds — production `VmResult` values flow from
    /// `run_vm` and never from this fixture. See
    /// `sidecar_vm_result_is_test_fixture_boilerplate` in
    /// `test_support/sidecar.rs` for the motivating deduplication
    /// (7 identical literal constructions collapsed to a single
    /// call).
    #[cfg(test)]
    pub fn test_fixture() -> Self {
        Self {
            success: true,
            exit_code: 0,
            duration: Duration::from_secs(1),
            timed_out: false,
            output: String::new(),
            stderr: String::new(),
            monitor: None,
            shm_data: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
        }
    }
}

/// Per-vCPU KVM stats read after VM exit. Each map holds cumulative
/// counter values from the VM's lifetime.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct KvmStatsTotals {
    /// Per-vCPU stat maps. Index is vCPU id.
    pub per_vcpu: Vec<HashMap<String, u64>>,
}

/// KVM stat names surfaced in sidecar output for scheduler testing.
///
/// Covers VM exit rate, halt-polling behavior, preemption notifications,
/// signal-driven exits, and hypercall counts; all fields scheduler
/// authors typically correlate with scx decisions.
#[allow(dead_code)]
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

/// State returned by [`KtstrVm::run_vm`] after the BSP exits.
/// Passed to [`KtstrVm::collect_results`] to produce [`VmResult`].
struct VmRunState {
    exit_code: i32,
    timed_out: bool,
    ap_threads: Vec<VcpuThread>,
    monitor_handle: Option<JoinHandle<monitor::reader::MonitorLoopResult>>,
    bpf_write_handle: Option<JoinHandle<()>>,
    com1: Arc<PiMutex<console::Serial>>,
    com2: Arc<PiMutex<console::Serial>>,
    kill: Arc<AtomicBool>,
    vm: kvm::KtstrKvm,
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
// KtstrVm — builder + run
// ---------------------------------------------------------------------------

/// Builder for creating and running VMs with custom topologies.
pub struct KtstrVm {
    kernel: PathBuf,
    init_binary: Option<PathBuf>,
    scheduler_binary: Option<PathBuf>,
    run_args: Vec<String>,
    sched_args: Vec<String>,
    topology: Topology,
    /// Guest memory in MB. `None` = deferred: computed from actual
    /// initramfs size after the initramfs build completes.
    memory_mb: Option<u32>,
    /// Minimum memory in MB for deferred allocation. When non-zero,
    /// the deferred path uses `max(computed, memory_min_mb)` so topology
    /// configs that need more memory than the initramfs floor are honored.
    memory_min_mb: u32,
    cmdline_extra: String,
    timeout: Duration,
    /// Size of the SHM ring buffer region at the top of guest memory. 0 = disabled.
    shm_size: u64,
    /// Thresholds for reactive SysRq-D dump. When set and the monitor
    /// detects a sustained violation, it writes the dump flag to guest SHM.
    monitor_thresholds: Option<crate::monitor::MonitorThresholds>,
    /// Override for `scx_sched.watchdog_timeout` in the guest kernel.
    /// Converted to jiffies via CONFIG_HZ at monitor start time and
    /// written at each monitor iteration after the scheduler attaches.
    watchdog_timeout: Option<Duration>,
    /// Host-side BPF map writes. Empty slice disables the thread.
    /// When non-empty, a thread polls for BPF map discoverability,
    /// waits for scenario start via SHM ring, then writes each
    /// `u32` value at its specified map/offset. All writes complete
    /// before the guest is signaled via SHM slot 0, so the guest
    /// sees a single unblock regardless of how many writes ran.
    bpf_map_writes: Vec<BpfMapWriteParams>,
    /// Performance mode: vCPU pinning to host LLCs, hugepage-backed
    /// guest memory, NUMA mbind, and RT scheduling on both
    /// architectures. On x86_64, additionally: KVM_HINTS_REALTIME
    /// CPUID hint, PAUSE and HLT VM exit disabling via
    /// KVM_CAP_X86_DISABLE_EXITS, and KVM_CAP_HALT_POLL skipped
    /// (guest haltpoll cpuidle disables host halt polling via
    /// MSR_KVM_POLL_CONTROL). Oversubscription validation at build
    /// time on both architectures.
    performance_mode: bool,
    /// Pinning plan computed during build() when performance_mode is enabled.
    /// Stored so topology is read once and the plan is reused at VM start.
    pinning_plan: Option<host_topology::PinningPlan>,
    /// Per-guest-NUMA-node host NUMA nodes for mbind. Indexed by guest
    /// node ID. Each entry is the set of host NUMA nodes that the guest
    /// node's vCPUs are pinned to. Empty when performance_mode is off.
    mbind_node_map: Vec<Vec<usize>>,
    /// CPU flock fds for non-perf VMs. Held for the VM's lifetime to
    /// prevent other VMs from double-booking the same CPUs.
    #[allow(dead_code)]
    cpu_locks: Vec<std::os::fd::OwnedFd>,
    /// No-perf-mode resource plan, populated when the builder resolves
    /// an `LlcCap` via `KTSTR_LLC_CAP` (set by `ktstr shell --llc-cap N`
    /// or `cargo ktstr shell --llc-cap N`). Holds the flattened CPU
    /// list + RAII flock fds returned by
    /// [`host_topology::acquire_llc_plan`]. `run_vm` reads the CPU
    /// list to `sched_setaffinity` every vCPU thread to the reserved
    /// LLCs' host-CPU set, and `Drop` releases the LLC flocks with
    /// the VM.
    ///
    /// `None` for the pre-flag no-perf path (which keeps using
    /// `cpu_locks` above) and for perf-mode (which uses
    /// `pinning_plan`). The two paths are orthogonal — perf-mode
    /// hard-pins single CPUs, --llc-cap soft-masks a pool.
    #[allow(dead_code)]
    no_perf_plan: Option<host_topology::LlcPlan>,
    /// Shell commands to run in the guest to enable a kernel-built scheduler.
    sched_enable_cmds: Vec<String>,
    /// Shell commands to run in the guest to disable a kernel-built scheduler.
    sched_disable_cmds: Vec<String>,
    /// Files to include in the guest initramfs at their archive paths.
    /// Each entry is (archive_path, host_path).
    include_files: Vec<(String, PathBuf)>,
    /// Embed busybox in the initramfs for shell mode.
    busybox: bool,
    /// Forward COM1 (kernel console) to stderr in real-time during
    /// interactive shell mode. Useful for watching virtio probe and
    /// kernel messages alongside the shell session.
    dmesg: bool,
    /// Command to execute non-interactively in shell mode (--exec).
    /// Passed to the guest via /exec_cmd in the initramfs.
    exec_cmd: Option<String>,
    /// Optional host path to `ktstr-jemalloc-probe`. When `Some`, the
    /// probe is packed into the guest initramfs as an extra binary at
    /// `bin/ktstr-jemalloc-probe`. Consumed by `spawn_initramfs_resolve`.
    jemalloc_probe_binary: Option<PathBuf>,
    /// Optional host path to `ktstr-jemalloc-alloc-worker`. When
    /// `Some`, the worker is packed alongside the probe as an
    /// extra. The cross-process closed-loop test in
    /// `tests/jemalloc_probe_tests.rs` spawns it as a background
    /// payload and probes its pid.
    jemalloc_alloc_worker_binary: Option<PathBuf>,
}

/// Parameters for a host-side BPF map write during VM execution.
#[derive(Clone)]
struct BpfMapWriteParams {
    map_name_suffix: String,
    offset: usize,
    value: u32,
}

impl KtstrVm {
    pub fn builder() -> KtstrVmBuilder {
        KtstrVmBuilder::default()
    }

    /// Borrow this VM's per-invocation initramfs-suffix inputs into an
    /// [`initramfs::SuffixParams`]. Centralizes the `run_args` /
    /// `sched_args` / sched-enable / sched-disable / `exec_cmd`
    /// bundling so both x86_64 and aarch64 paths construct the suffix
    /// from the same source of truth.
    fn suffix_params(&self) -> initramfs::SuffixParams<'_> {
        initramfs::SuffixParams {
            args: &self.run_args,
            sched_args: &self.sched_args,
            sched_enable: &self.sched_enable_cmds,
            sched_disable: &self.sched_disable_cmds,
            exec_cmd: self.exec_cmd.as_deref(),
        }
    }

    /// Boot the VM, run until shutdown/timeout, return captured output.
    pub fn run(&self) -> Result<VmResult> {
        let start = Instant::now();

        let initramfs_handle = self.spawn_initramfs_resolve();
        let (mut vm, kernel_result) = self.create_vm_and_load_kernel()?;

        #[cfg(target_arch = "x86_64")]
        let _kernel_result = {
            let kr = self.setup_memory(&mut vm, kernel_result, initramfs_handle)?;
            self.setup_vcpus(&vm, kr.entry)?;
            kr
        };
        #[cfg(target_arch = "aarch64")]
        let _kernel_result = {
            let kr = self.setup_memory_aarch64(&mut vm, kernel_result, initramfs_handle)?;
            self.setup_vcpus_aarch64(&vm, kr.entry)?;
            kr
        };

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

        // mut needed on x86_64 for kvm_stats assignment below.
        #[allow(unused_mut)]
        let mut result = self.collect_results(start, run)?;

        // Read cumulative KVM stats after VM exit.
        #[cfg(target_arch = "x86_64")]
        if let Some(ctx) = stats_ctx {
            result.kvm_stats = Some(ctx.read_stats());
        }

        Ok(result)
    }

    /// Boot the VM with bidirectional stdin/stdout forwarding via virtio-console.
    ///
    /// Sets the host terminal to raw mode, spawns threads for stdin->hvc0
    /// and hvc0->stdout forwarding, and runs until the guest shuts down.
    /// Terminal state is restored on all exit paths including panic and
    /// process-killing signals (SIGINT, SIGTERM, SIGQUIT).
    ///
    /// Builder settings ignored in interactive mode: `monitor_thresholds`,
    /// `watchdog_timeout`, `bpf_map_write`, `performance_mode` pinning,
    /// and KVM stats collection. These are test-specific features that
    /// do not apply to interactive shell sessions.
    pub fn run_interactive(&self) -> Result<()> {
        let start = Instant::now();

        let initramfs_handle = self.spawn_initramfs_resolve();
        let (mut vm, kernel_result) = self.create_vm_and_load_kernel()?;

        #[cfg(target_arch = "x86_64")]
        {
            let kr = self.setup_memory(&mut vm, kernel_result, initramfs_handle)?;
            self.setup_vcpus(&vm, kr.entry)?;
        }
        #[cfg(target_arch = "aarch64")]
        {
            let kr = self.setup_memory_aarch64(&mut vm, kernel_result, initramfs_handle)?;
            self.setup_vcpus_aarch64(&vm, kr.entry)?;
        }

        let com1 = Arc::new(PiMutex::new(console::Serial::new(console::COM1_BASE)));
        let com2 = Arc::new(PiMutex::new(console::Serial::new(console::COM2_BASE)));

        // Virtio-console for shell I/O via /dev/hvc0.
        let mut vc = virtio_console::VirtioConsole::new();
        vc.set_mem(vm.guest_mem.clone());
        let virtio_con = Arc::new(PiMutex::new(vc));

        #[cfg(target_arch = "x86_64")]
        if !vm.split_irqchip {
            vm.vm_fd
                .register_irqfd(com1.lock().irq_evt(), console::COM1_IRQ)
                .context("register COM1 irqfd")?;
            vm.vm_fd
                .register_irqfd(com2.lock().irq_evt(), console::COM2_IRQ)
                .context("register COM2 irqfd")?;
            vm.vm_fd
                .register_irqfd(virtio_con.lock().irq_evt(), kvm::VIRTIO_CONSOLE_IRQ)
                .context("register virtio-console irqfd")?;
        }
        #[cfg(target_arch = "aarch64")]
        {
            vm.vm_fd
                .register_irqfd(com1.lock().irq_evt(), kvm::SERIAL_IRQ)
                .context("register serial irqfd")?;
            vm.vm_fd
                .register_irqfd(com2.lock().irq_evt(), kvm::SERIAL2_IRQ)
                .context("register serial2 irqfd")?;
            vm.vm_fd
                .register_irqfd(virtio_con.lock().irq_evt(), kvm::VIRTIO_CONSOLE_IRQ)
                .context("register virtio-console irqfd")?;
        }

        // Non-interactive exec mode (--exec) does not need a TTY.
        let exec_mode = self.exec_cmd.is_some();

        // Pre-flight: verify stdin is a tty, enter raw mode, and create
        // the wakeup pipe before spawning threads. Failing after thread
        // spawn would abandon AP threads.
        if !exec_mode {
            use std::os::unix::io::AsRawFd;
            let stdin_fd = std::io::stdin().as_raw_fd();
            let borrowed = unsafe { std::os::unix::io::BorrowedFd::borrow_raw(stdin_fd) };
            anyhow::ensure!(
                nix::unistd::isatty(borrowed).unwrap_or(false),
                "stdin must be a terminal for interactive shell mode",
            );
        }

        // Set host terminal to raw mode. TerminalRawGuard restores on drop
        // and installs signal handlers for SIGINT, SIGTERM, SIGQUIT,
        // SIGABRT, and SIGFPE so every terminating signal routes through
        // the terminal-restore path before the process exits (see
        // `src/terminal.rs`). Skip for exec mode — no interactive
        // terminal needed.
        let _raw_guard = if exec_mode {
            None
        } else {
            Some(TerminalRawGuard::enter().context("failed to set terminal to raw mode")?)
        };

        // Wakeup pipe: write end signals the stdin reader to exit when
        // the kill flag is set, avoiding a blocking read that prevents join.
        let (wakeup_r, wakeup_w) = nix::unistd::pipe().context("create stdin wakeup pipe")?;

        let kill = Arc::new(AtomicBool::new(false));
        let has_immediate_exit = vm.has_immediate_exit;
        let mut vcpus = std::mem::take(&mut vm.vcpus);
        let mut bsp = vcpus.remove(0);

        let ap_pins = vec![None; vcpus.len()];
        // Shell/interactive path mirrors run_vm: no-perf + --llc-cap
        // applies the LlcPlan's CPU list as a sched_setaffinity mask
        // on every vCPU thread. Perf-mode's pin_targets doesn't
        // apply here — interactive shell runs under no-perf by
        // convention, and `pin_targets` is empty in this branch.
        let no_perf_mask: Option<&[usize]> = self.no_perf_plan.as_ref().map(|p| p.cpus.as_slice());
        let ap_threads = self.spawn_ap_threads(
            vcpus,
            has_immediate_exit,
            &com1,
            &com2,
            Some(&virtio_con),
            &kill,
            &ap_pins,
            no_perf_mask,
        )?;

        // BSP kick handles for the stdin escape sequence. The stdin thread
        // needs to force the BSP out of KVM_RUN when Ctrl+A X is pressed.
        let bsp_ie_for_stdin = if has_immediate_exit {
            Some(ImmediateExitHandle::from_vcpu(&mut bsp))
        } else {
            None
        };
        let bsp_tid = unsafe { libc::pthread_self() };

        // Stdin reader thread: host stdin -> virtio-console RX queue.
        // The guest reads stdin from /dev/hvc0 (virtio-console), never
        // from COM2. pending_rx buffers input until the guest activates
        // the RX queue. Uses poll() on both stdin and the wakeup pipe
        // so the thread can be cleanly joined on shutdown.
        //
        // Escape sequence: Ctrl+A X (0x01 followed by 'x' or 'X') triggers
        // host-side VM teardown without guest cooperation.
        let vc_for_stdin = virtio_con.clone();
        let kill_for_stdin = kill.clone();
        let stdin_thread = std::thread::Builder::new()
            .name("interactive-stdin".into())
            .spawn(move || {
                use std::io::Read;
                use std::os::unix::io::{AsFd, AsRawFd};

                // wakeup_r is an OwnedFd moved into this closure; closed on exit.
                let wakeup_fd = wakeup_r;
                let stdin_fd = std::io::stdin().as_raw_fd();
                let mut buf = [0u8; 4096];
                let mut saw_ctrl_a = false;

                loop {
                    if kill_for_stdin.load(Ordering::Acquire) {
                        break;
                    }

                    // Poll stdin and wakeup fd with 100ms timeout.
                    let stdin_borrowed =
                        unsafe { std::os::unix::io::BorrowedFd::borrow_raw(stdin_fd) };
                    let wakeup_borrowed = wakeup_fd.as_fd();
                    let mut fds = [
                        nix::poll::PollFd::new(stdin_borrowed, nix::poll::PollFlags::POLLIN),
                        nix::poll::PollFd::new(wakeup_borrowed, nix::poll::PollFlags::POLLIN),
                    ];
                    match nix::poll::poll(&mut fds, 100u16) {
                        Ok(0) => continue, // timeout
                        Err(nix::errno::Errno::EINTR) => continue,
                        Err(_) => break,
                        Ok(_) => {}
                    }

                    // Wakeup fd readable means shutdown requested.
                    if fds[1]
                        .revents()
                        .is_some_and(|r| r.intersects(nix::poll::PollFlags::POLLIN))
                    {
                        break;
                    }

                    // Stdin readable.
                    if fds[0]
                        .revents()
                        .is_some_and(|r| r.intersects(nix::poll::PollFlags::POLLIN))
                    {
                        let mut stdin = std::io::stdin().lock();
                        match stdin.read(&mut buf) {
                            Ok(0) => break,
                            Ok(n) => {
                                // Scan for Ctrl+A X escape sequence. Filter
                                // escape bytes from the forwarded input so
                                // neither the 0x01 nor 'x'/'X' reaches the
                                // guest.
                                let mut forward_start = 0usize;
                                for i in 0..n {
                                    if saw_ctrl_a {
                                        saw_ctrl_a = false;
                                        if buf[i] == b'x' || buf[i] == b'X' {
                                            // Trigger host-side teardown. Bytes
                                            // before the Ctrl+A were already
                                            // flushed when saw_ctrl_a was set.
                                            eprintln!("\r\nTerminated.");
                                            kill_for_stdin.store(true, Ordering::Release);
                                            if let Some(ref ie) = bsp_ie_for_stdin {
                                                ie.set(1);
                                                std::sync::atomic::fence(Ordering::Release);
                                            }
                                            unsafe {
                                                libc::pthread_kill(bsp_tid, vcpu_signal());
                                            }
                                            return;
                                        }
                                        // Not 'x'/'X' after Ctrl+A: the 0x01
                                        // was a real keystroke. Forward it now.
                                        vc_for_stdin.lock().queue_input(&[0x01]);
                                        // Current byte is processed normally
                                        // below (may itself be 0x01).
                                    }
                                    if buf[i] == 0x01 {
                                        // Flush bytes before the Ctrl+A.
                                        if forward_start < i {
                                            vc_for_stdin.lock().queue_input(&buf[forward_start..i]);
                                        }
                                        saw_ctrl_a = true;
                                        forward_start = i + 1;
                                        continue;
                                    }
                                }
                                // Forward remaining bytes.
                                if forward_start < n {
                                    vc_for_stdin.lock().queue_input(&buf[forward_start..n]);
                                }
                            }
                            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                            Err(_) => break,
                        }
                    }
                }
            })
            .context("spawn stdin reader thread")?;

        // Stdout writer thread: virtio-console TX -> host stdout.
        // Polls tx_evt for zero-latency wakeup when guest writes data.
        // On write errors (including BrokenPipe), sets kill flag and exits
        // to stop the VM rather than polling a dead pipe until timeout.
        let vc_for_stdout = virtio_con.clone();
        let kill_for_stdout = kill.clone();
        let stdout_thread: JoinHandle<bool> = std::thread::Builder::new()
            .name("interactive-stdout".into())
            .spawn(move || {
                use std::io::Write;

                let mut wrote_any = false;

                // Cache the raw fd for poll. The eventfd lives as long as
                // VirtioConsole which is behind Arc<PiMutex> — valid for
                // the thread's lifetime.
                let tx_evt_raw_fd = {
                    let guard = vc_for_stdout.lock();
                    std::os::unix::io::AsRawFd::as_raw_fd(guard.tx_evt())
                };
                let mut stdout = std::io::stdout().lock();
                loop {
                    if kill_for_stdout.load(Ordering::Acquire) {
                        break;
                    }
                    let borrowed =
                        unsafe { std::os::unix::io::BorrowedFd::borrow_raw(tx_evt_raw_fd) };
                    let mut fds = [nix::poll::PollFd::new(
                        borrowed,
                        nix::poll::PollFlags::POLLIN,
                    )];
                    match nix::poll::poll(&mut fds, 50u16) {
                        Ok(0) => continue,
                        Err(nix::errno::Errno::EINTR) => continue,
                        Err(_) => break,
                        Ok(_) => {
                            // Consume eventfd counter.
                            let _ = vc_for_stdout.lock().tx_evt().read();
                        }
                    }
                    // Re-check kill after poll. During shutdown the
                    // dying guest may enqueue a stray byte into the
                    // virtio TX queue (from kernel hvc_close flushing
                    // n_outbuf via tty_wait_until_sent → hvc_push →
                    // put_chars). That byte passes from_utf8 (valid
                    // single-byte UTF-8) but is unprintable, producing
                    // a garbled character on the terminal.
                    if kill_for_stdout.load(Ordering::Acquire) {
                        break;
                    }
                    let data = vc_for_stdout.lock().drain_output();
                    if !data.is_empty() {
                        // Write only valid UTF-8 prefix. Trailing
                        // incomplete sequences (from guest shutdown
                        // mid-write) are dropped to prevent garbled
                        // output.
                        let valid_len = match std::str::from_utf8(&data) {
                            Ok(_) => data.len(),
                            Err(e) => e.valid_up_to(),
                        };
                        if valid_len > 0 {
                            if stdout.write_all(&data[..valid_len]).is_err()
                                || stdout.flush().is_err()
                            {
                                kill_for_stdout.store(true, Ordering::Release);
                                break;
                            }
                            wrote_any = true;
                        }
                    }
                }
                // Final drain: the guest may have flushed output just
                // before shutdown that hasn't been polled yet.
                let data = vc_for_stdout.lock().drain_output();
                if !data.is_empty() {
                    let valid_len = match std::str::from_utf8(&data) {
                        Ok(_) => data.len(),
                        Err(e) => e.valid_up_to(),
                    };
                    if valid_len > 0 {
                        let _ = stdout.write_all(&data[..valid_len]);
                        let _ = stdout.flush();
                        wrote_any = true;
                    }
                }
                wrote_any
            })
            .context("spawn stdout writer thread")?;

        // Optional dmesg thread: COM1 -> stderr in real-time.
        // Only spawned when --dmesg is active. Gives the user kernel
        // messages (including virtio probe results) alongside the shell.
        let dmesg_thread = if self.dmesg {
            let com1_for_dmesg = com1.clone();
            let kill_for_dmesg = kill.clone();
            Some(
                std::thread::Builder::new()
                    .name("interactive-dmesg".into())
                    .spawn(move || {
                        use std::io::Write;
                        // Lock stderr per-write, not for the whole loop.
                        // Holding the lock blocks Ctrl+A X's eprintln.
                        loop {
                            if kill_for_dmesg.load(Ordering::Acquire) {
                                break;
                            }
                            std::thread::sleep(std::time::Duration::from_millis(50));
                            let data = com1_for_dmesg.lock().drain_output();
                            if !data.is_empty() {
                                let mut stderr = std::io::stderr().lock();
                                let _ = stderr.write_all(&data);
                                let _ = stderr.flush();
                            }
                        }
                        // Final drain.
                        let data = com1_for_dmesg.lock().drain_output();
                        if !data.is_empty() {
                            let mut stderr = std::io::stderr().lock();
                            let _ = stderr.write_all(&data);
                            let _ = stderr.flush();
                        }
                    })
                    .context("spawn dmesg thread")?,
            )
        } else {
            None
        };

        // BSP run loop (same shutdown detection as run()).
        // Interactive sessions are user-controlled; the builder's timeout
        // (default 60s) must not kill the shell. Use 24 hours as a
        // practical upper bound.
        //
        // Apply the no-perf + --llc-cap mask to the BSP thread so
        // interactive `ktstr shell --no-perf-mode --llc-cap N` runs
        // inside the reserved LLCs just like run_vm's BSP. No pin
        // here — perf-mode doesn't apply to interactive shell:
        // `--llc-cap` requires `--no-perf-mode` on Shell (clap
        // `requires` attribute on the llc_cap field).
        if let Some(mask) = self.no_perf_plan.as_ref().map(|p| p.cpus.as_slice()) {
            set_thread_cpumask(mask, "BSP (shell)");
        }
        register_vcpu_signal_handler();
        let interactive_timeout = Duration::from_secs(24 * 60 * 60);
        self.run_bsp_loop(
            &mut bsp,
            &com1,
            &com2,
            Some(&virtio_con),
            &kill,
            has_immediate_exit,
            start,
            interactive_timeout,
        );

        // Shutdown.
        kill.store(true, Ordering::Release);

        // Wake the stdin reader so it exits poll() and can be joined.
        let _ = nix::unistd::write(&wakeup_w, &[0u8]);
        drop(wakeup_w);

        for vt in &ap_threads {
            if !vt.exited.load(Ordering::Acquire) {
                vt.kick();
            }
        }
        for vt in ap_threads {
            vt.wait_for_exit(Duration::from_secs(5));
            let _ = vt.handle.join();
        }

        let stdout_wrote = stdout_thread.join().unwrap_or(false);
        let _ = stdin_thread.join();
        if let Some(dt) = dmesg_thread {
            let _ = dt.join();
        }

        // _raw_guard drops here, restoring terminal and signal handlers.
        drop(_raw_guard);

        // Exec mode fallback: if virtio-console produced no output
        // (kernel lacks CONFIG_VIRTIO_CONSOLE, guest fell back to
        // COM2), print COM2 output to stdout so the caller sees it.
        // Filter out the KTSTR_EXEC_EXIT sentinel which the guest
        // writes to stderr (also COM2 in the fallback case).
        if exec_mode && !stdout_wrote {
            let app_output = com2.lock().output();
            if !app_output.is_empty() {
                use std::io::Write;
                let mut stdout = std::io::stdout().lock();
                for line in app_output.lines() {
                    if !line.starts_with(crate::test_support::SENTINEL_EXEC_EXIT_PREFIX) {
                        let _ = writeln!(stdout, "{line}");
                    }
                }
                let _ = stdout.flush();
            }
        }

        // Print kernel console output (COM1) to stderr if non-empty.
        // Skip when --dmesg was active (already streamed to stderr).
        if !self.dmesg {
            let console_output = com1.lock().output();
            if !console_output.is_empty() {
                eprintln!("{console_output}");
            }
        }

        if !exec_mode {
            eprintln!("Connection to VM closed.");
        }
        Ok(())
    }

    /// Create the KVM VM and optionally load the kernel.
    ///
    /// When `memory_mb` is `Some`, allocates guest memory and loads the
    /// kernel immediately (existing path). When `None` (deferred), creates
    /// the VM without memory — allocation and kernel loading happen later
    /// in `setup_memory` after the actual initramfs size is known.
    fn create_vm_and_load_kernel(&self) -> Result<(kvm::KtstrKvm, Option<boot::KernelLoadResult>)> {
        let t0 = Instant::now();
        let use_hugepages = self.performance_mode
            && self.memory_mb.is_some_and(|mb| {
                host_topology::hugepages_free() >= host_topology::hugepages_needed(mb)
            });

        let vm = match self.memory_mb {
            Some(mb) => {
                if use_hugepages {
                    kvm::KtstrKvm::new_with_hugepages(self.topology, mb, self.performance_mode)
                        .context("create VM with hugepages")?
                } else {
                    kvm::KtstrKvm::new(self.topology, mb, self.performance_mode)
                        .context("create VM")?
                }
            }
            None => {
                kvm::KtstrKvm::new_deferred(self.topology, use_hugepages, self.performance_mode)
                    .context("create VM (deferred memory)")?
            }
        };
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "kvm_create");

        // When memory is already allocated (non-deferred path), do mbind
        // and load kernel now. Deferred path does this in setup_memory.
        let kernel_result = if self.memory_mb.is_some() {
            if self.performance_mode && !self.mbind_node_map.is_empty() {
                let layout = vm.numa_layout.as_ref().expect(
                    "numa_layout is Some on the non-deferred allocation path: \
                     allocate_and_register_memory ran during `vm_new` because \
                     memory_mb was provided up front, and that call sets \
                     numa_layout to Some(...) in src/vmm/{x86_64,aarch64}/kvm.rs",
                );
                layout.mbind_regions(&vm.guest_mem, &self.mbind_node_map);
            }

            let t0 = Instant::now();
            let kr = boot::load_kernel(&vm.guest_mem, &self.kernel).context("load kernel")?;
            tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "load_kernel");
            Some(kr)
        } else {
            None
        };

        Ok((vm, kernel_result))
    }

    /// Spawn initramfs resolution on a background thread.
    /// Returns the handle to join later (after KVM creation completes).
    fn spawn_initramfs_resolve(&self) -> Option<JoinHandle<Result<(BaseRef, BaseKey)>>> {
        let bin = self.init_binary.as_ref()?;
        let payload = bin.clone();
        let scheduler = self.scheduler_binary.clone();
        let probe = self.jemalloc_probe_binary.clone();
        let worker = self.jemalloc_alloc_worker_binary.clone();
        let include_files = self.include_files.clone();
        let busybox = self.busybox;
        std::thread::Builder::new()
            .name("initramfs-resolve".into())
            .spawn(move || -> Result<(BaseRef, BaseKey)> {
                // Extras are stripped by `build_initramfs_base`
                // before write. The scheduler can lose its DWARF
                // without functional impact, but the probe and
                // worker MUST retain DWARF — the probe resolves
                // `tsd_s.thread_allocated` field offsets against
                // the running target's `/proc/<pid>/exe`, and a
                // stripped target has no DWARF to walk. Route
                // probe + worker through `include_files` instead,
                // which copies files verbatim (no `strip_debug`).
                let mut extras: Vec<(&str, &std::path::Path)> = Vec::new();
                if let Some(s) = scheduler.as_deref() {
                    extras.push(("scheduler", s));
                }
                // Shell-mode cache keying treats ANY include_files
                // as shell-mode. `jemalloc_probe_binary` and `jemalloc_alloc_worker_binary`
                // are real include_files at the cache-key layer —
                // hash them accordingly so a binary-change
                // invalidates the cache. The scheduler stays in
                // the non-shell path.
                let has_jemalloc_extras = probe.as_deref().is_some() || worker.as_deref().is_some();
                let shell_mode = busybox || !include_files.is_empty() || has_jemalloc_extras;

                // Merge include_files with probe + worker so both
                // the cache key and the actual archive build see
                // the same input set.
                let mut merged_includes: Vec<(String, PathBuf)> = include_files.clone();
                if let Some(p) = probe.as_deref() {
                    merged_includes.push(("bin/ktstr-jemalloc-probe".to_string(), p.to_path_buf()));
                }
                if let Some(w) = worker.as_deref() {
                    merged_includes.push((
                        "bin/ktstr-jemalloc-alloc-worker".to_string(),
                        w.to_path_buf(),
                    ));
                }

                let key = if shell_mode {
                    BaseKey::new_shell(
                        &payload,
                        scheduler.as_deref(),
                        probe.as_deref(),
                        worker.as_deref(),
                        &merged_includes,
                        busybox,
                    )?
                } else {
                    BaseKey::new(
                        &payload,
                        scheduler.as_deref(),
                        probe.as_deref(),
                        worker.as_deref(),
                    )?
                };

                let include_refs: Vec<(&str, &std::path::Path)> = merged_includes
                    .iter()
                    .map(|(a, p)| (a.as_str(), p.as_path()))
                    .collect();
                let base = get_or_build_base(&payload, &extras, &include_refs, busybox, &key)?;
                Ok((base, key))
            })
            .ok()
    }

    /// Compress base+suffix as separate LZ4 legacy streams, load into
    /// guest memory via COW overlay (falling back to write_slice), and
    /// verify the write. Returns `total_compressed_size`.
    ///
    /// On a successful COW overlay, the returned `CowOverlayGuard` is
    /// pushed onto `vm.cow_overlay_guards` IMMEDIATELY — before any
    /// subsequent fallible operation (suffix write, read-back verify)
    /// runs. This is deliberate: if a later `?` unwinds this function
    /// after the MAP_FIXED overlay is in place, a locally-held guard
    /// would drop first, releasing `LOCK_SH` while the COW VMAs are
    /// still live. A concurrent writer could then take `LOCK_EX` and
    /// truncate the segment → SIGBUS on the mapped pages. Pushing the
    /// guard onto `vm` transfers ownership to the VM, where Drop
    /// order is structurally enforced (guard drops AFTER
    /// `_reservation` munmaps the COW VMAs).
    #[cfg(target_arch = "x86_64")]
    fn compress_and_load_initrd(
        &self,
        vm: &mut kvm::KtstrKvm,
        base_bytes: &[u8],
        suffix: &[u8],
        key: &BaseKey,
        load_addr: u64,
    ) -> Result<u32> {
        let uncompressed_size = base_bytes.len() + suffix.len();

        // Compress base and suffix as separate LZ4 legacy streams. The
        // kernel initramfs decompressor handles concatenated LZ4 natively
        // (re-encountering the magic mid-stream resets the decoder).
        // Keeping them separate lets us COW-map the base from SHM.
        let t0 = Instant::now();
        let lz4_base = self.get_or_compress_base(base_bytes, key)?;
        let lz4_suffix = initramfs::lz4_legacy_compress(suffix);
        let total_compressed = lz4_base.len() + lz4_suffix.len();
        tracing::debug!(
            elapsed_us = t0.elapsed().as_micros(),
            uncompressed = uncompressed_size,
            lz4_base = lz4_base.len(),
            lz4_suffix = lz4_suffix.len(),
            ratio = format!("{:.1}x", uncompressed_size as f64 / total_compressed as f64),
            "lz4_initramfs",
        );

        tracing::debug!(
            base_magic = format!(
                "{:02x}{:02x}{:02x}{:02x}",
                lz4_base[0], lz4_base[1], lz4_base[2], lz4_base[3]
            ),
            suffix_magic = format!(
                "{:02x}{:02x}{:02x}{:02x}",
                lz4_suffix[0], lz4_suffix[1], lz4_suffix[2], lz4_suffix[3]
            ),
            base_len = lz4_base.len(),
            suffix_len = lz4_suffix.len(),
            total = total_compressed,
            load_addr = format!("{:#x}", load_addr),
            suffix_addr = format!("{:#x}", load_addr + lz4_base.len() as u64),
            "initrd_load_debug",
        );

        // Try COW overlay: mmap compressed base from SHM fd directly
        // into guest memory, sharing physical pages across VMs.
        let t0 = Instant::now();
        let cow_guard = self.try_cow_overlay(&vm.guest_mem, key, lz4_base.len(), load_addr);
        // IMPORTANT: stash the guard on the VM IMMEDIATELY — before
        // any fallible operation below. If a `?` unwinds this function
        // with a locally-held guard still on the stack, the guard
        // drops first, releasing LOCK_SH while the COW VMAs are still
        // live. Owned by `vm`, the guard drops with the VM's
        // declared-order Drop, which is strictly after
        // `_reservation` (and thus the COW VMAs). See
        // `try_cow_overlay_rejects_cross_region_span` and the C4
        // comment on `cow_overlay_guards` in kvm.rs.
        let cow_active = cow_guard.is_some();
        if let Some(guard) = cow_guard {
            vm.cow_overlay_guards.push(guard);
        }
        if cow_active {
            vm.guest_mem
                .write_slice(&lz4_suffix, GuestAddress(load_addr + lz4_base.len() as u64))
                .context("write lz4 suffix after COW base")?;
            tracing::debug!(
                elapsed_us = t0.elapsed().as_micros(),
                cow = true,
                "initrd_write"
            );
        } else {
            initramfs::load_initramfs_parts(&vm.guest_mem, &[&lz4_base, &lz4_suffix], load_addr)?;
            tracing::debug!(
                elapsed_us = t0.elapsed().as_micros(),
                cow = false,
                "initrd_write"
            );
        }

        // Read back first 8 bytes from guest memory to check write.
        let mut check_buf = [0u8; 8];
        vm.guest_mem
            .read_slice(&mut check_buf, GuestAddress(load_addr))
            .context("read-back initrd check")?;
        tracing::debug!(
            first_8 = format!(
                "{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
                check_buf[0],
                check_buf[1],
                check_buf[2],
                check_buf[3],
                check_buf[4],
                check_buf[5],
                check_buf[6],
                check_buf[7]
            ),
            expected_magic = "02214c18",
            "initrd_verify",
        );

        Ok(total_compressed as u32)
    }

    /// Join the initramfs thread and load the result into guest memory.
    /// Memory must already be allocated (non-deferred path). Validates
    /// that allocated memory is sufficient for the initramfs.
    #[cfg(target_arch = "x86_64")]
    fn join_and_load_initramfs(
        &self,
        vm: &mut kvm::KtstrKvm,
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
        let suffix = initramfs::build_suffix(base_bytes.len(), &self.suffix_params())?;
        let uncompressed_size = base_bytes.len() + suffix.len();
        tracing::debug!(
            elapsed_us = t0.elapsed().as_micros(),
            base_bytes = base_bytes.len(),
            suffix_bytes = suffix.len(),
            "build_suffix",
        );

        // Enforce minimum memory for initramfs extraction.
        // This path is only reached when memory_mb was set explicitly.
        let memory_mb = self.memory_mb.expect(
            "join_and_load_initramfs called in deferred mode; \
             use join_compute_memory_and_load instead",
        );
        // Compress first to get actual compressed size for validation.
        let lz4_base = self.get_or_compress_base(base_bytes, &key)?;
        let lz4_suffix = initramfs::lz4_legacy_compress(&suffix);
        let compressed_size = lz4_base.len() + lz4_suffix.len();
        let kernel_init_size = read_kernel_init_size(&self.kernel).unwrap_or(0) as u64;
        let budget = MemoryBudget {
            uncompressed_initramfs_bytes: uncompressed_size as u64,
            compressed_initrd_bytes: compressed_size as u64,
            kernel_init_size,
            shm_bytes: self.shm_size,
        };
        let min_mb = initramfs_min_memory_mb(&budget);
        if memory_mb < min_mb {
            anyhow::bail!(
                "VM memory {}MB insufficient for initramfs \
                 (uncompressed={}MB, compressed={}MB, \
                 init_size={}MB): need {}MB",
                memory_mb,
                uncompressed_size >> 20,
                compressed_size >> 20,
                kernel_init_size >> 20,
                min_mb,
            );
        }

        let size = self.compress_and_load_initrd(vm, base_bytes, &suffix, &key, load_addr)?;
        Ok((Some(load_addr), Some(size)))
    }

    /// Deferred memory path: join initramfs, compute memory from actual
    /// size, allocate guest memory, then load initramfs.
    ///
    /// Returns `(initrd_addr, initrd_size, memory_mb)`.
    #[cfg(target_arch = "x86_64")]
    fn join_compute_memory_and_load(
        &self,
        vm: &mut kvm::KtstrKvm,
        handle: JoinHandle<Result<(BaseRef, BaseKey)>>,
        load_addr: u64,
    ) -> Result<(Option<u64>, Option<u32>, u32)> {
        let t0 = Instant::now();
        let (base, key) = handle
            .join()
            .map_err(|_| anyhow::anyhow!("initramfs-resolve thread panicked"))??;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "initramfs_join");
        let base_bytes: &[u8] = base.as_ref();

        let t0 = Instant::now();
        let suffix = initramfs::build_suffix(base_bytes.len(), &self.suffix_params())?;
        let uncompressed_size = base_bytes.len() + suffix.len();
        tracing::debug!(
            elapsed_us = t0.elapsed().as_micros(),
            base_bytes = base_bytes.len(),
            suffix_bytes = suffix.len(),
            "build_suffix",
        );

        // Compress before computing memory so the formula uses actual
        // compressed size instead of guessing.
        let t0_compress = Instant::now();
        let lz4_base = self.get_or_compress_base(base_bytes, &key)?;
        let lz4_suffix = initramfs::lz4_legacy_compress(&suffix);
        let compressed_size = lz4_base.len() + lz4_suffix.len();
        tracing::debug!(
            elapsed_us = t0_compress.elapsed().as_micros(),
            uncompressed = uncompressed_size,
            compressed = compressed_size,
            ratio = format!("{:.1}x", uncompressed_size as f64 / compressed_size as f64),
            "deferred_lz4_compress",
        );

        // Compute memory from actual sizes, honoring the
        // topology-requested minimum when non-zero.
        let kernel_init_size = read_kernel_init_size(&self.kernel).unwrap_or(0) as u64;
        let budget = MemoryBudget {
            uncompressed_initramfs_bytes: uncompressed_size as u64,
            compressed_initrd_bytes: compressed_size as u64,
            kernel_init_size,
            shm_bytes: self.shm_size,
        };
        let memory_mb = initramfs_min_memory_mb(&budget).max(self.memory_min_mb);
        tracing::debug!(
            uncompressed_mb = uncompressed_size >> 20,
            compressed_mb = compressed_size >> 20,
            init_size_mb = kernel_init_size >> 20,
            memory_min_mb = self.memory_min_mb,
            memory_mb,
            "deferred_memory_computed",
        );

        // Allocate and register guest memory.
        vm.allocate_and_register_memory(memory_mb)
            .with_context(|| format!("allocate deferred memory ({memory_mb}MB)"))?;

        // Load pre-compressed data into guest memory. The base is already
        // in the LZ4 SHM cache from get_or_compress_base above, so
        // compress_and_load_initrd will hit the cache.
        let size = self.compress_and_load_initrd(vm, base_bytes, &suffix, &key, load_addr)?;
        Ok((Some(load_addr), Some(size), memory_mb))
    }

    fn effective_memory_mb(&self, guest_mem: &GuestMemoryMmap) -> u32 {
        use vm_memory::GuestMemoryRegion;
        match self.memory_mb {
            Some(mb) => mb,
            None => {
                let total_bytes: u64 = guest_mem.iter().map(|r| r.len()).sum();
                (total_bytes >> 20) as u32
            }
        }
    }

    /// Get or build the compressed base. Checks LZ4 SHM first, then
    /// compresses and stores.
    #[cfg(target_arch = "x86_64")]
    fn get_or_compress_base(&self, base_bytes: &[u8], key: &BaseKey) -> Result<Vec<u8>> {
        // Try loading compressed base from LZ4 SHM.
        if let Some((fd, len)) = initramfs::shm_open_lz4(key.0) {
            use std::os::fd::AsRawFd;
            let mut buf = vec![0u8; len];
            unsafe {
                let ptr = libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ,
                    libc::MAP_SHARED,
                    fd.as_raw_fd(),
                    0,
                );
                if ptr != libc::MAP_FAILED {
                    std::ptr::copy_nonoverlapping(ptr as *const u8, buf.as_mut_ptr(), len);
                    libc::munmap(ptr, len);
                    initramfs::shm_close_fd(fd);

                    // Validate LZ4 legacy magic. Stale segments from a
                    // previous compression format (zstd) must be discarded.
                    if buf.len() >= 4 && buf[..4] == initramfs::LZ4_LEGACY_MAGIC {
                        tracing::debug!(bytes = len, "lz4_base cache hit (shm)");
                        return Ok(buf);
                    }
                    tracing::warn!(
                        bytes = len,
                        magic = format!("{:02x}{:02x}{:02x}{:02x}", buf[0], buf[1], buf[2], buf[3]),
                        "stale compressed shm segment (wrong magic), recompressing"
                    );
                } else {
                    initramfs::shm_close_fd(fd);
                }
            }
        }

        // Compress with LZ4 legacy format.
        let lz4 = initramfs::lz4_legacy_compress(base_bytes);

        if let Err(e) = initramfs::shm_store_lz4(key.0, &lz4) {
            tracing::warn!("shm_store_lz4: {e:#}");
        }
        Ok(lz4)
    }

    /// Try to COW-overlay the compressed base from LZ4 SHM into guest
    /// memory. Returns `Some(CowOverlayGuard)` on success — the guard
    /// owns the SHM fd and holds `LOCK_SH` for the mapping's lifetime,
    /// and MUST be kept alive as long as the COW overlay is in use
    /// (typically the VM lifetime). Validates the segment starts with
    /// LZ4 legacy magic to reject stale data from a previous
    /// compression format.
    #[cfg(target_arch = "x86_64")]
    fn try_cow_overlay(
        &self,
        guest_mem: &GuestMemoryMmap,
        key: &BaseKey,
        expected_len: usize,
        load_addr: u64,
    ) -> Option<initramfs::CowOverlayGuard> {
        let (fd, len) = initramfs::shm_open_lz4(key.0)?;
        if len != expected_len {
            initramfs::shm_close_fd(fd);
            return None;
        }
        // Validate LZ4 legacy magic before COW-mapping.
        use std::os::fd::AsRawFd;
        let mut magic = [0u8; 4];
        unsafe {
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            );
            if ptr == libc::MAP_FAILED {
                initramfs::shm_close_fd(fd);
                return None;
            }
            std::ptr::copy_nonoverlapping(ptr as *const u8, magic.as_mut_ptr(), 4);
            libc::munmap(ptr, len);
        }
        if magic != initramfs::LZ4_LEGACY_MAGIC {
            tracing::warn!(
                magic = format!(
                    "{:02x}{:02x}{:02x}{:02x}",
                    magic[0], magic[1], magic[2], magic[3]
                ),
                "stale compressed shm segment in COW path, skipping"
            );
            initramfs::shm_close_fd(fd);
            return None;
        }
        // Refuse zero-length: mmap(len=0) is EINVAL and serves no
        // purpose; the suffix-write fallback handles empty bases
        // trivially. Also refuse load_addr + len overflow before
        // bounds-checking, since GuestAddress arithmetic wraps
        // silently on u64 overflow.
        if len == 0 || load_addr.checked_add(len as u64).is_none() {
            tracing::debug!(
                load_addr = format!("{:#x}", load_addr),
                len,
                "cow_overlay: invalid range (zero-length or overflow), falling back"
            );
            initramfs::shm_close_fd(fd);
            return None;
        }
        // Bounds-check [load_addr, load_addr + len) against guest
        // memory BEFORE the MAP_FIXED mmap. `get_host_address` only
        // validates the start address — without a length check,
        // MAP_FIXED would silently overwrite whatever host VA happens
        // to follow the region (other guest regions, reserved VA, or
        // unrelated mappings). `get_slice` fails if the range extends
        // past the region's end or spans a region boundary, which is
        // exactly the guarantee MAP_FIXED needs.
        if guest_mem.get_slice(GuestAddress(load_addr), len).is_err() {
            tracing::debug!(
                load_addr = format!("{:#x}", load_addr),
                len,
                "cow_overlay: range exceeds guest memory region, falling back"
            );
            initramfs::shm_close_fd(fd);
            return None;
        }
        let Ok(host_addr) = guest_mem.get_host_address(GuestAddress(load_addr)) else {
            initramfs::shm_close_fd(fd);
            return None;
        };
        // cow_overlay takes ownership of `fd` on both Some and None
        // paths: on success the guard carries it; on failure
        // cow_overlay itself closes it. Do NOT call shm_close_fd here.
        unsafe { initramfs::cow_overlay(host_addr, len, fd) }
    }

    /// Initialize the SHM ring buffer header at `shm_base` in guest memory.
    fn init_shm_region(&self, guest_mem: &GuestMemoryMmap, shm_base: u64) -> Result<()> {
        let header = shm_ring::ShmRingHeader::new(self.shm_size as usize);
        guest_mem
            .write_slice(
                zerocopy::IntoBytes::as_bytes(&header),
                GuestAddress(shm_base),
            )
            .context("write SHM header")
    }

    /// Write cmdline, boot params, SHM header, and topology tables to guest memory.
    ///
    /// When `kernel_result` is `None` (deferred memory mode), this method
    /// first joins the initramfs thread to learn the actual size, allocates
    /// guest memory from that size, does mbind, and loads the kernel — all
    /// before proceeding with the normal initramfs load and boot param setup.
    #[cfg(target_arch = "x86_64")]
    fn setup_memory(
        &self,
        vm: &mut kvm::KtstrKvm,
        kernel_result: Option<boot::KernelLoadResult>,
        initramfs_handle: Option<JoinHandle<Result<(BaseRef, BaseKey)>>>,
    ) -> Result<boot::KernelLoadResult> {
        // Deferred memory path: join initramfs first to learn its size,
        // then allocate memory, load kernel, and load initramfs — all in
        // one shot with no estimation.
        let (kernel_result, initrd_addr, initrd_size) = if let Some(kr) = kernel_result {
            // Non-deferred: memory already allocated, kernel already loaded.
            // compress_and_load_initrd transfers the CowOverlayGuard
            // directly onto vm.cow_overlay_guards before any fallible
            // operation, so a mid-function `?` cannot drop the guard
            // before the COW VMAs are torn down.
            let (initrd_addr, initrd_size) = match initramfs_handle {
                Some(handle) => self.join_and_load_initramfs(vm, handle, INITRD_ADDR)?,
                None => (None, None),
            };
            (kr, initrd_addr, initrd_size)
        } else {
            // Deferred memory path: join initramfs first to learn its size,
            // then allocate memory, load kernel, and load initramfs — all in
            // one shot with no estimation.
            let (initrd_addr, initrd_size, _memory_mb) = match initramfs_handle {
                Some(handle) => self.join_compute_memory_and_load(vm, handle, INITRD_ADDR)?,
                None => {
                    // No initramfs — allocate minimum memory.
                    let memory_mb = 256u32;
                    vm.allocate_and_register_memory(memory_mb)
                        .context("allocate deferred memory (no initramfs)")?;
                    (None, None, memory_mb)
                }
            };

            if self.performance_mode && !self.mbind_node_map.is_empty() {
                let layout = vm.numa_layout.as_ref().expect(
                    "numa_layout is Some after the deferred allocate_and_register_memory \
                     call above: that call sets numa_layout to Some(...) in \
                     src/vmm/{x86_64,aarch64}/kvm.rs before this branch can reach here",
                );
                layout.mbind_regions(&vm.guest_mem, &self.mbind_node_map);
            }

            // Load kernel into the freshly allocated memory.
            let t0 = Instant::now();
            let kr = boot::load_kernel(&vm.guest_mem, &self.kernel).context("load kernel")?;
            tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "load_kernel");

            (kr, initrd_addr, initrd_size)
        };

        // Resolve effective memory_mb for boot params / ACPI / SHM.
        let memory_mb = self.effective_memory_mb(&vm.guest_mem);

        // Kernel cmdline rationale (per flag):
        //   console=ttyS0        — serial console for host-visible output.
        //   nomodules            — no out-of-tree modules are shipped; skip modprobe paths.
        //   mitigations=off      — skip Spectre/Meltdown mitigations for VM perf.
        //   no_timer_check       — suppress APIC timer-calibration failure under KVM.
        //   clocksource=kvm-clock — stable paravirt clock; avoid TSC drift under KVM.
        //   random.trust_cpu=on  — seed RNG from RDRAND so userspace doesn't block on entropy.
        //   swiotlb=noforce      — skip the IOMMU bounce buffer — no passthrough devices.
        //   i8042.*=noaux/nomux/nopnp/dumbkbd — skip legacy PS/2 probing; no keyboard/mouse in VM.
        //   pci=off              — no PCI devices emulated; shave boot time by skipping the scan.
        //   reboot=k             — use keyboard-controller reset method.
        //   panic=-1             — reboot immediately on panic; host detects via exit.
        //   iomem=relaxed        — allow guest /dev/mem mmap of the SHM region (see shm_ring.rs).
        //   nokaslr              — deterministic kernel addresses for symbol/offset resolution.
        //   lockdown=none        — permit /dev/mem and unrestricted BPF needed by the test runtime.
        //   sysctl.kernel.unprivileged_bpf_disabled=0 — allow BPF load from the test runtime.
        //   sysctl.kernel.sched_schedstats=1          — enable /proc/schedstat for workload reports.
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
        let verbose = std::env::var("KTSTR_VERBOSE")
            .map(|v| v == "1")
            .unwrap_or(false)
            || std::env::var("RUST_BACKTRACE").is_ok_and(|v| v == "1" || v == "full");
        if verbose {
            cmdline.push_str(" earlyprintk=serial loglevel=7");
        } else {
            cmdline.push_str(" loglevel=0");
        }
        if self.init_binary.is_some() {
            cmdline.push_str(" rdinit=/init initramfs_options=size=90%");
        }
        // Virtio-console MMIO device on the kernel cmdline. The kernel's
        // virtio_mmio_cmdline_devices driver parses this to register the
        // MMIO transport at the given base address and IRQ.
        cmdline.push_str(&format!(
            " virtio_mmio.device={:#x}@{:#x}:{}",
            virtio_console::VIRTIO_MMIO_SIZE,
            kvm::VIRTIO_CONSOLE_MMIO_BASE,
            kvm::VIRTIO_CONSOLE_IRQ,
        ));
        if self.shm_size > 0 {
            let mem_size = (memory_mb as u64) << 20;
            let shm_base = mem_size - self.shm_size;
            cmdline.push_str(&format!(
                " KTSTR_SHM_BASE={:#x} KTSTR_SHM_SIZE={:#x}",
                shm_base, self.shm_size
            ));
        }
        if self.topology.has_memory_only_nodes() {
            cmdline.push_str(" numa_balancing=enable");
        } else {
            cmdline.push_str(" numa_balancing=0");
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
            memory_mb,
            initrd_addr,
            initrd_size,
            kernel_result.setup_header.as_ref(),
            self.shm_size,
        )?;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "cmdline_boot_params");

        // Initialize SHM ring buffer.
        let t0 = Instant::now();
        if self.shm_size > 0 {
            let mem_size = (memory_mb as u64) << 20;
            let shm_base = mem_size - self.shm_size;
            self.init_shm_region(&vm.guest_mem, shm_base)?;
        }
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "shm_ring_init");

        let t0 = Instant::now();
        mptable::setup_mptable(&vm.guest_mem, &self.topology)?;
        let _acpi_layout = acpi::setup_acpi(
            &vm.guest_mem,
            &self.topology,
            vm.numa_layout.as_ref().expect(
                "numa_layout is Some by the time setup_acpi runs: \
                 memory allocation (whether deferred or not) ran earlier \
                 in this function and set numa_layout via \
                 allocate_and_register_memory in src/vmm/x86_64/kvm.rs",
            ),
        )?;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "mptable_acpi");

        Ok(kernel_result)
    }

    /// Configure BSP and AP vCPUs.
    #[cfg(target_arch = "x86_64")]
    fn setup_vcpus(&self, vm: &kvm::KtstrKvm, kernel_entry: u64) -> Result<()> {
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
    fn run_vm(&self, start: Instant, mut vm: kvm::KtstrKvm) -> Result<VmRunState> {
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

        // No-perf + --llc-cap: flat CPU list from the LLC plan gets
        // sched_setaffinity'd on every vCPU thread as a mask (not a
        // hard pin). Mutually exclusive with perf-mode's pin_targets.
        let no_perf_mask: Option<&[usize]> = self.no_perf_plan.as_ref().map(|p| p.cpus.as_slice());

        let ap_threads = self.spawn_ap_threads(
            vcpus,
            has_immediate_exit,
            &com1,
            &com2,
            None,
            &kill,
            &ap_pins,
            no_perf_mask,
        )?;

        // Pin / mask BSP (runs on current thread, pid=0 means calling thread).
        if let Some(Some(host_cpu)) = pin_targets.first() {
            pin_current_thread(*host_cpu, "BSP (vCPU 0)");
        } else if let Some(mask) = no_perf_mask {
            set_thread_cpumask(mask, "BSP (vCPU 0)");
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

        // Build GuestMem for the watchdog's graceful shutdown handshake.
        let wd_shm = if self.shm_size > 0 {
            let mem = match vm.numa_layout.as_ref() {
                Some(layout) => monitor::reader::GuestMem::from_layout(layout, &vm.guest_mem),
                None => {
                    let host_base = vm
                        .guest_mem
                        .get_host_address(GuestAddress(DRAM_BASE))
                        .unwrap();
                    let mem_size = (self.effective_memory_mb(&vm.guest_mem) as u64) << 20;
                    // SAFETY: host_base came from GuestMemoryMmap's
                    // get_host_address, mapping is owned by vm.guest_mem
                    // which outlives this GuestMem (both captured by
                    // the surrounding closure and used only while the
                    // VM runs).
                    unsafe { monitor::reader::GuestMem::new(host_base, mem_size) }
                }
            };
            let shm_base = mem.size() - self.shm_size;
            Some((mem, shm_base))
        } else {
            None
        };

        let watchdog = std::thread::Builder::new()
            .name("vmm-watchdog".into())
            .spawn(move || {
                if let Some(cpu) = wd_service_cpu {
                    pin_current_thread(cpu, "watchdog");
                }
                if rt_watchdog {
                    set_rt_priority(2, "watchdog");
                }
                let hard_deadline = Instant::now() + timeout;
                // Soft phase needs enough headroom for the guest to
                // flush serial and reboot. Skip when timeout < 5s.
                let soft_deadline = if timeout > Duration::from_secs(5) {
                    Some(hard_deadline - Duration::from_secs(3))
                } else {
                    None
                };
                let mut soft_fired = false;
                eprintln!("watchdog: started, timeout={timeout:?}");
                loop {
                    if bsp_done_for_wd.load(Ordering::Acquire) {
                        eprintln!("watchdog: BSP done, returning");
                        return;
                    }
                    if kill_for_watchdog.load(Ordering::Acquire) || Instant::now() >= hard_deadline
                    {
                        // Either an AP set kill or hard timeout expired.
                        // Re-check bsp_done: if the BSP already exited its
                        // run loop, the VcpuFd (and kvm_run mmap backing
                        // bsp_ie) may be dropped. Writing to ie after drop
                        // is a use-after-free.
                        if bsp_done_for_wd.load(Ordering::Acquire) {
                            eprintln!("watchdog: BSP already done, returning");
                            return;
                        }
                        let reason = if Instant::now() >= hard_deadline {
                            "hard timeout expired"
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
                    // Soft deadline: request graceful shutdown via SHM.
                    // The BSP keeps running so the guest can flush serial
                    // and reboot normally.
                    if !soft_fired && soft_deadline.is_some_and(|d| Instant::now() >= d) {
                        soft_fired = true;
                        if let Some((ref mem, shm_base)) = wd_shm {
                            eprintln!("watchdog: soft deadline, requesting graceful shutdown");
                            shm_ring::signal_guest_value(
                                mem,
                                shm_base,
                                0,
                                shm_ring::SIGNAL_SHUTDOWN_REQ,
                            );
                        }
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
            })
            .context("spawn watchdog thread")?;

        // BSP run loop. Wrapped in the same `with_vcpu_panic_ctx`
        // scope the APs use (symmetric panic-hook signaling) —
        // `kill` plus `bsp_done` are the pair analogous to a
        // vCPU thread's `kill` + `exited` so a BSP panic flips the
        // watchdog-observed flags before the panic=abort teardown.
        // `vcpu_panic::install_once` was already called in
        // `spawn_ap_threads` above, which runs even for a zero-AP VM,
        // so the hook is live by the time BSP enters its loop.
        eprintln!("BSP: entering run loop");
        let (exit_code, timed_out) = vcpu_panic::with_vcpu_panic_ctx(
            vcpu_panic::VcpuPanicCtx {
                kill: kill.clone(),
                exited: bsp_done.clone(),
            },
            || {
                self.run_bsp_loop(
                    &mut bsp,
                    &com1,
                    &com2,
                    None,
                    &kill,
                    has_immediate_exit,
                    start,
                    timeout,
                )
            },
        );
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
    /// host CPU from `pin_targets` (indexed by AP order, 0-based), OR
    /// applies a CPU mask from `no_perf_mask` when the no-perf +
    /// `--llc-cap` path is active. The two are mutually exclusive —
    /// perf-mode produces `pin_targets` via the PinningPlan;
    /// `--llc-cap` no-perf produces `no_perf_mask` via the LlcPlan.
    #[allow(clippy::too_many_arguments)]
    fn spawn_ap_threads(
        &self,
        vcpus: Vec<kvm_ioctls::VcpuFd>,
        has_immediate_exit: bool,
        com1: &Arc<PiMutex<console::Serial>>,
        com2: &Arc<PiMutex<console::Serial>>,
        virtio_con: Option<&Arc<PiMutex<virtio_console::VirtioConsole>>>,
        kill: &Arc<AtomicBool>,
        pin_targets: &[Option<usize>],
        no_perf_mask: Option<&[usize]>,
    ) -> Result<Vec<VcpuThread>> {
        // Register the process-wide panic hook that flips `kill` +
        // `exited` on a panicking vCPU thread before the
        // panic=abort-induced process teardown. Idempotent via
        // `Once`; safe to call on every VM spawn.
        vcpu_panic::install_once();
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
            let vc_clone = virtio_con.cloned();
            let exited = Arc::new(AtomicBool::new(false));
            let exited_clone = exited.clone();
            let pin_cpu = pin_targets.get(i).copied().flatten();
            let mask_for_thread: Option<Vec<usize>> = no_perf_mask.map(|m| m.to_vec());

            let rt = self.performance_mode;
            let panic_ctx = vcpu_panic::VcpuPanicCtx {
                kill: kill.clone(),
                exited: exited.clone(),
            };
            let handle = std::thread::Builder::new()
                .name(format!("vcpu-{}", i + 1))
                .spawn(move || {
                    register_vcpu_signal_handler();
                    if let Some(cpu) = pin_cpu {
                        pin_current_thread(cpu, &format!("vCPU {}", i + 1));
                    } else if let Some(mask) = mask_for_thread.as_deref() {
                        set_thread_cpumask(mask, &format!("vCPU {}", i + 1));
                    }
                    if rt {
                        set_rt_priority(1, &format!("vCPU {}", i + 1));
                    }
                    vcpu_panic::with_vcpu_panic_ctx(panic_ctx, || {
                        vcpu_run_loop_unified(
                            &mut vcpu,
                            &com1_clone,
                            &com2_clone,
                            vc_clone.as_ref(),
                            &kill_clone,
                        );
                    });
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
    fn start_monitor(
        &self,
        vm: &kvm::KtstrKvm,
        kill: &Arc<AtomicBool>,
        start: Instant,
        vcpu_pthreads: Vec<libc::pthread_t>,
    ) -> Result<Option<JoinHandle<monitor::reader::MonitorLoopResult>>> {
        let Some(vmlinux) = find_vmlinux(&self.kernel) else {
            return Ok(None);
        };
        let offsets = monitor::btf_offsets::KernelOffsets::from_vmlinux(&vmlinux);
        let symbols = monitor::symbols::KernelSymbols::from_vmlinux(&vmlinux);

        let (Ok(offsets), Ok(symbols)) = (offsets, symbols) else {
            return Ok(None);
        };

        let mem = match vm.numa_layout.as_ref() {
            Some(layout) => monitor::reader::GuestMem::from_layout(layout, &vm.guest_mem),
            None => {
                let host_base = vm
                    .guest_mem
                    .get_host_address(GuestAddress(DRAM_BASE))
                    .unwrap();
                let mem_size = (self.effective_memory_mb(&vm.guest_mem) as u64) << 20;
                // SAFETY: host_base is from GuestMemoryMmap's mapping,
                // which outlives this GuestMem (owned by `vm` until
                // return).
                unsafe { monitor::reader::GuestMem::new(host_base, mem_size) }
            }
        };
        let mem_size = mem.size();
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
        let watchdog_jiffies = self.watchdog_timeout.map(|d| d.as_secs() * hz);
        let preemption_threshold_ns = monitor::vcpu_preemption_threshold_ns(Some(&self.kernel));
        let rt_monitor = self.performance_mode;
        let service_cpu = self.pinning_plan.as_ref().and_then(|p| p.service_cpu);
        let shm_base_pa = if self.shm_size > 0 {
            Some(mem_size - self.shm_size)
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
                let offsets_arr = monitor::symbols::read_per_cpu_offsets(&mem, pco_pa, num_cpus);
                // Per-CPU addresses (runqueues + offset) are in the
                // direct mapping: use PAGE_OFFSET.
                let rq_pas =
                    monitor::symbols::compute_rq_pas(symbols.runqueues, &offsets_arr, page_offset);

                let watchdog_override = watchdog_jiffies.and_then(|jiffies| {
                    // 7.1+ path: deref scx_root -> scx_sched.watchdog_timeout.
                    if let Some((scx_root_kva, wd_offs)) = symbols
                        .scx_root
                        .zip(offsets.watchdog_offsets.as_ref())
                    {
                        let scx_root_pa = monitor::symbols::text_kva_to_pa(scx_root_kva);
                        return Some(monitor::reader::WatchdogOverride::ScxSched {
                            scx_root_pa,
                            watchdog_offset: wd_offs.scx_sched_watchdog_timeout_off,
                            jiffies,
                            page_offset,
                        });
                    }
                    // Pre-7.1 fallback: direct write to scx_watchdog_timeout static global.
                    if let Some(wdt_kva) = symbols.scx_watchdog_timeout {
                        let watchdog_timeout_pa = monitor::symbols::text_kva_to_pa(wdt_kva);
                        return Some(monitor::reader::WatchdogOverride::StaticGlobal {
                            watchdog_timeout_pa,
                            jiffies,
                        });
                    }
                    None
                });
                if watchdog_jiffies.is_some() && watchdog_override.is_none() {
                    tracing::warn!(
                        "no watchdog override path available — neither scx_sched.watchdog_timeout BTF field nor scx_watchdog_timeout symbol found"
                    );
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
        vm: &kvm::KtstrKvm,
        kill: &Arc<AtomicBool>,
    ) -> Result<Option<JoinHandle<()>>> {
        if self.bpf_map_writes.is_empty() {
            return Ok(None);
        }
        let Some(vmlinux) = find_vmlinux(&self.kernel) else {
            eprintln!("bpf_map_write: vmlinux not found, skipping");
            return Ok(None);
        };

        let mem = match vm.numa_layout.as_ref() {
            Some(layout) => monitor::reader::GuestMem::from_layout(layout, &vm.guest_mem),
            None => {
                let host_base = vm
                    .guest_mem
                    .get_host_address(GuestAddress(DRAM_BASE))
                    .unwrap();
                let mem_size = (self.effective_memory_mb(&vm.guest_mem) as u64) << 20;
                // SAFETY: host_base is from GuestMemoryMmap's mapping,
                // which outlives this GuestMem (owned by `vm` until
                // return).
                unsafe { monitor::reader::GuestMem::new(host_base, mem_size) }
            }
        };
        let kill_clone = kill.clone();
        let writes = self.bpf_map_writes.clone();
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
                let owned = loop {
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
                let accessor = owned.as_accessor();

                // Phase 2: resolve every queued map before signaling the
                // guest. All-or-nothing: if any map fails to resolve
                // within the deadline, the thread aborts without
                // signaling slot 0. The guest then proceeds under its
                // own timeout rather than observing a partial setup.
                // Running writes serially against partially-resolved
                // maps would let a late-discovery failure leave the
                // guest blocked waiting for slot 0 with no way to
                // recover.
                let retry_deadline =
                    std::time::Instant::now() + std::time::Duration::from_secs(30);
                let mut resolved: Vec<(BpfMapWriteParams, monitor::bpf_map::BpfMapInfo)> =
                    Vec::with_capacity(writes.len());
                for params in writes.iter() {
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
                    resolved.push((params.clone(), map_info));
                }

                // Phase 3: wait for probes ready, run every queued
                // write, signal guest once all writes complete.
                //
                // The guest signals slot 1 with SIGNAL_PROBES_READY after
                // the probe pipeline attaches and the scenario is starting.
                // Without this gate, the crash fires during scheduler load
                // before probes capture any events.
                if shm_size > 0 {
                    let shm_base = mem.size() - shm_size;
                    let ready_deadline =
                        std::time::Instant::now() + std::time::Duration::from_secs(30);
                    loop {
                        if kill_clone.load(Ordering::Acquire) {
                            return;
                        }
                        if std::time::Instant::now() >= ready_deadline {
                            eprintln!("bpf_map_write: timed out waiting for probes ready");
                            return;
                        }
                        let val = mem.read_u8(shm_base, shm_ring::SIGNAL_SLOT_BASE + 1);
                        if val >= shm_ring::SIGNAL_PROBES_READY {
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    }
                    eprintln!("bpf_map_write: guest probes ready, applying queued writes");
                }

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

                let mut all_ok = true;
                for (params, map_info) in &resolved {
                    let before = accessor.read_value_u32(map_info, params.offset);
                    let ok = accessor.write_value_u32(map_info, params.offset, params.value);
                    let after = accessor.read_value_u32(map_info, params.offset);
                    eprintln!(
                        "bpf_map_write: map '{}' write={} (value={} offset={} before={:?} after={:?})",
                        map_info.name, ok, params.value, params.offset, before, after,
                    );
                    all_ok &= ok;
                }

                // Signal the guest once every queued write has been
                // applied. Partial success (one failing write) still
                // suppresses the signal so the guest proceeds under
                // its own timeout rather than observing half-applied
                // state.
                if all_ok && shm_size > 0 {
                    let shm_base = mem.size() - shm_size;
                    shm_ring::signal_guest(&mem, shm_base, 0);
                    eprintln!(
                        "bpf_map_write: signaled slot 0 after {} write(s)",
                        resolved.len(),
                    );
                }
            })
            .context("spawn bpf-map-write thread")?;

        Ok(Some(handle))
    }

    /// Unified BSP KVM_RUN loop. Returns (exit_code, timed_out).
    ///
    /// Handles arch-specific I/O dispatch (port I/O on x86_64, MMIO on
    /// aarch64). HLT/WFI checks the kill flag and continues (both arches).
    /// Shutdown is via PSCI SystemEvent (aarch64) or VcpuExit::Shutdown (x86_64).
    #[allow(clippy::too_many_arguments)]
    fn run_bsp_loop(
        &self,
        bsp: &mut kvm_ioctls::VcpuFd,
        com1: &Arc<PiMutex<console::Serial>>,
        com2: &Arc<PiMutex<console::Serial>>,
        virtio_con: Option<&Arc<PiMutex<virtio_console::VirtioConsole>>>,
        kill: &Arc<AtomicBool>,
        has_immediate_exit: bool,
        start: Instant,
        timeout: Duration,
    ) -> (i32, bool) {
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
                    // HLT/WFI = kernel idle. Check kill flag, then continue.
                    // arm64 shutdown is PSCI reset (SystemEvent), not HLT.
                    if matches!(exit, VcpuExit::Hlt) {
                        if kill.load(Ordering::Acquire) {
                            break;
                        }
                        continue;
                    }
                    match classify_exit(com1, com2, virtio_con.map(|a| a.as_ref()), &mut exit) {
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
                Some(monitor::reader::MonitorLoopResult {
                    samples,
                    drain,
                    watchdog_observation,
                }) => {
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
                        watchdog_observation,
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
            let mem_size = (self.effective_memory_mb(&run.vm.guest_mem) as u64) << 20;
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
            .find(|l| l.starts_with(crate::test_support::SENTINEL_EXIT_PREFIX))
            && let Ok(code) = line
                .trim_start_matches(crate::test_support::SENTINEL_EXIT_PREFIX)
                .trim()
                .parse::<i32>()
        {
            exit_code = code;
        }

        // Extract crash message from SHM (reliable, full backtrace).
        let crash_message = shm_data.as_ref().and_then(|d| {
            d.entries
                .iter()
                .find(|e| e.msg_type == shm_ring::MSG_TYPE_CRASH && e.crc_ok)
                .and_then(|e| String::from_utf8(e.payload.clone()).ok())
        });

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
            crash_message,
        })
    }

    /// Read BPF verifier stats from guest memory after VM exit.
    ///
    /// Enumerates struct_ops programs in the kernel's `prog_idr` and
    /// reads `bpf_prog_aux->verified_insns` for each.
    fn collect_verifier_stats(
        &self,
        vm: &kvm::KtstrKvm,
    ) -> Vec<monitor::bpf_prog::ProgVerifierStats> {
        let vmlinux = match find_vmlinux(&self.kernel) {
            Some(v) => v,
            None => return Vec::new(),
        };
        let mem = match vm.numa_layout.as_ref() {
            Some(layout) => monitor::reader::GuestMem::from_layout(layout, &vm.guest_mem),
            None => {
                let host_base = match vm.guest_mem.get_host_address(GuestAddress(DRAM_BASE)) {
                    Ok(ptr) => ptr,
                    Err(_) => return Vec::new(),
                };
                let mem_size = (self.effective_memory_mb(&vm.guest_mem) as u64) << 20;
                // SAFETY: host_base is from GuestMemoryMmap's mapping,
                // which outlives this GuestMem (borrowed via `vm` for
                // the body of this function).
                unsafe { monitor::reader::GuestMem::new(host_base, mem_size) }
            }
        };
        let kernel = match monitor::guest::GuestKernel::new(&mem, &vmlinux) {
            Ok(k) => k,
            Err(_) => return Vec::new(),
        };
        let offsets = match monitor::btf_offsets::BpfProgOffsets::from_vmlinux(&vmlinux) {
            Ok(o) => o,
            Err(_) => return Vec::new(),
        };
        let accessor =
            match monitor::bpf_prog::BpfProgAccessor::from_guest_kernel(&kernel, &offsets) {
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
impl KtstrVm {
    /// Allocate and register guest memory regions for aarch64, including
    /// NUMA-aware placement.
    fn setup_memory_aarch64(
        &self,
        vm: &mut kvm::KtstrKvm,
        kernel_result: Option<boot::KernelLoadResult>,
        initramfs_handle: Option<JoinHandle<Result<(BaseRef, BaseKey)>>>,
    ) -> Result<boot::KernelLoadResult> {
        // Deferred memory path for aarch64.
        let kernel_result = if let Some(kr) = kernel_result {
            kr
        } else {
            // Join initramfs to learn actual size, then allocate memory.
            if let Some(handle) = initramfs_handle {
                let (base, _key) = handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("initramfs-resolve thread panicked"))??;
                let base_bytes: &[u8] = base.as_ref();
                let suffix = initramfs::build_suffix(base_bytes.len(), &self.suffix_params())?;
                let uncompressed_size = base_bytes.len() + suffix.len();

                // Compress before computing memory so the formula uses
                // actual compressed size.
                let initrd_data = initramfs::lz4_compress_combined(base_bytes, &suffix);
                let total_size = initrd_data.len() as u64;

                let kernel_init_size = read_kernel_init_size(&self.kernel).unwrap_or(0);
                let budget = MemoryBudget {
                    uncompressed_initramfs_bytes: uncompressed_size as u64,
                    compressed_initrd_bytes: total_size,
                    kernel_init_size,
                    shm_bytes: self.shm_size,
                };
                let memory_mb = initramfs_min_memory_mb(&budget).max(self.memory_min_mb);

                vm.allocate_and_register_memory(memory_mb)
                    .with_context(|| {
                        format!("allocate deferred memory ({memory_mb}MB, aarch64)")
                    })?;

                // Load kernel.
                let kr = boot::load_kernel(&vm.guest_mem, &self.kernel)
                    .context("load kernel (aarch64)")?;
                let load_addr = aarch64_initrd_addr(memory_mb, self.shm_size, total_size);
                initramfs::load_initramfs_parts(&vm.guest_mem, &[&initrd_data], load_addr)?;

                // Fall through to cmdline/FDT setup below with the initrd info.
                // We need to set up a scope that merges into the non-deferred path.
                // For simplicity, we re-enter the shared path with kernel_result set.
                return self.finish_aarch64_setup(vm, kr, Some(load_addr), Some(total_size as u32));
            } else {
                let memory_mb = 256u32;
                vm.allocate_and_register_memory(memory_mb)
                    .context("allocate deferred memory (no initramfs, aarch64)")?;
                let kr = boot::load_kernel(&vm.guest_mem, &self.kernel)
                    .context("load kernel (aarch64)")?;
                return self.finish_aarch64_setup(vm, kr, None, None);
            }
        };

        // Non-deferred path: memory already allocated, kernel already loaded.
        let (initrd_addr, initrd_size) = match initramfs_handle {
            Some(handle) => {
                let memory_mb = self.memory_mb.unwrap();
                let (base, _key) = handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("initramfs-resolve thread panicked"))??;
                let base_bytes: &[u8] = base.as_ref();
                let suffix = initramfs::build_suffix(base_bytes.len(), &self.suffix_params())?;
                let initrd_data = initramfs::lz4_compress_combined(base_bytes, &suffix);
                let total_size = initrd_data.len() as u64;
                let load_addr = aarch64_initrd_addr(memory_mb, self.shm_size, total_size);
                initramfs::load_initramfs_parts(&vm.guest_mem, &[&initrd_data], load_addr)?;
                (Some(load_addr), Some(total_size as u32))
            }
            None => (None, None),
        };

        self.finish_aarch64_setup(vm, kernel_result, initrd_addr, initrd_size)
    }

    #[cfg(target_arch = "aarch64")]
    fn finish_aarch64_setup(
        &self,
        vm: &kvm::KtstrKvm,
        kernel_result: boot::KernelLoadResult,
        initrd_addr: Option<u64>,
        initrd_size: Option<u32>,
    ) -> Result<boot::KernelLoadResult> {
        let memory_mb = self.effective_memory_mb(&vm.guest_mem);

        // Kernel cmdline rationale (per flag) — aarch64 subset of the
        // x86_64 block above. Flags present on both arches carry the
        // same justification; see the x86_64 comment for details.
        // aarch64-specific:
        //   kfence.sample_interval=0 — disable KFENCE sampling; no real
        //                              driver faults to catch in the
        //                              test VM, and KFENCE adds boot-time
        //                              page-allocation pressure.
        let mut cmdline = concat!(
            "console=ttyS0 ",
            "nomodules mitigations=off ",
            "random.trust_cpu=on swiotlb=noforce ",
            "panic=-1 iomem=relaxed nokaslr lockdown=none ",
            "sysctl.kernel.unprivileged_bpf_disabled=0 ",
            "sysctl.kernel.sched_schedstats=1 ",
            "kfence.sample_interval=0",
        )
        .to_string();
        // earlycon is always enabled so the kernel has a console from
        // the earliest boot stage. Without it, stdout-path auto-detection
        // is the only path to early output — and that can fail silently
        // if the FDT node isn't matched by OF_EARLYCON_DECLARE.
        cmdline.push_str(" earlycon=uart,mmio,0x09000000");
        let verbose = std::env::var("KTSTR_VERBOSE")
            .map(|v| v == "1")
            .unwrap_or(false)
            || std::env::var("RUST_BACKTRACE").is_ok_and(|v| v == "1" || v == "full");
        if verbose {
            cmdline.push_str(" loglevel=7");
        } else {
            cmdline.push_str(" loglevel=0");
        }
        if self.init_binary.is_some() {
            cmdline.push_str(" rdinit=/init initramfs_options=size=90%");
        }
        if self.shm_size > 0 {
            let mem_size = (memory_mb as u64) << 20;
            let shm_base = kvm::DRAM_START + mem_size - self.shm_size;
            cmdline.push_str(&format!(
                " KTSTR_SHM_BASE={:#x} KTSTR_SHM_SIZE={:#x}",
                shm_base, self.shm_size
            ));
        }
        if self.topology.has_memory_only_nodes() {
            cmdline.push_str(" numa_balancing=enable");
        } else {
            cmdline.push_str(" numa_balancing=0");
        }
        if !self.cmdline_extra.is_empty() {
            cmdline.push(' ');
            cmdline.push_str(&self.cmdline_extra);
        }

        let t0 = Instant::now();
        boot::validate_cmdline(&cmdline)?;

        let fdt_addr = aarch64::fdt::fdt_address(memory_mb, self.shm_size);
        let mpidrs =
            aarch64::topology::read_mpidrs(&vm.vcpus).context("read vCPU MPIDRs for FDT")?;
        let hw_cache_level = aarch64::topology::host_cache_levels();
        let guest_l1_unified = aarch64::topology::host_l1_is_unified();
        let dtb = aarch64::fdt::create_fdt(
            &self.topology,
            &mpidrs,
            memory_mb,
            &cmdline,
            initrd_addr,
            initrd_size,
            self.shm_size,
            hw_cache_level,
            guest_l1_unified,
            vm.numa_layout.as_ref().expect(
                "numa_layout is Some by the time FDT creation runs: \
                 memory allocation (whether deferred or not) ran earlier \
                 in this function and set numa_layout via \
                 allocate_and_register_memory in src/vmm/aarch64/kvm.rs",
            ),
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
            let mem_size = (memory_mb as u64) << 20;
            let shm_base = kvm::DRAM_START + mem_size - self.shm_size;
            self.init_shm_region(&vm.guest_mem, shm_base)?;
        }
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "shm_ring_init");

        Ok(kernel_result)
    }

    #[cfg(target_arch = "aarch64")]
    fn setup_vcpus_aarch64(&self, vm: &kvm::KtstrKvm, kernel_entry: u64) -> Result<()> {
        let t0 = Instant::now();
        let memory_mb = self.effective_memory_mb(&vm.guest_mem);
        let fdt_addr = aarch64::fdt::fdt_address(memory_mb, self.shm_size);
        boot::setup_regs(&vm.vcpus[0], kernel_entry, fdt_addr)?;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "bsp_setup");
        // APs start powered off via PSCI — no register setup needed.
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builder for [`KtstrVm`].
///
/// Obtain via [`KtstrVm::builder()`], configure with the chained
/// setters below, then call [`build`](Self::build) to validate the
/// configuration and materialise a `KtstrVm`. Required inputs are a
/// `kernel` source directory or image, an `init_binary`, and either
/// a `run_args` payload (for test runs) or an `exec_cmd` / shell
/// configuration (for `ktstr shell`). Everything else is optional.
pub struct KtstrVmBuilder {
    kernel: Option<PathBuf>,
    init_binary: Option<PathBuf>,
    scheduler_binary: Option<PathBuf>,
    run_args: Vec<String>,
    sched_args: Vec<String>,
    topology: Topology,
    memory_mb: Option<u32>,
    memory_min_mb: u32,
    cmdline_extra: String,
    timeout: Duration,
    shm_size: u64,
    monitor_thresholds: Option<crate::monitor::MonitorThresholds>,
    watchdog_timeout: Option<Duration>,
    bpf_map_writes: Vec<BpfMapWriteParams>,
    performance_mode: bool,
    no_perf_mode: bool,
    sched_enable_cmds: Vec<String>,
    sched_disable_cmds: Vec<String>,
    include_files: Vec<(String, PathBuf)>,
    busybox: bool,
    dmesg: bool,
    exec_cmd: Option<String>,
    /// Optional host path to the `ktstr-jemalloc-probe` binary.
    /// When `Some`, the probe is packed into the guest initramfs at
    /// `bin/ktstr-jemalloc-probe` and becomes spawnable by bare name
    /// inside the guest — used by the closed-loop probe tests in
    /// `tests/jemalloc_probe_tests.rs`.
    jemalloc_probe_binary: Option<PathBuf>,
    /// Optional host path to `ktstr-jemalloc-alloc-worker`. When
    /// `Some`, packed into the initramfs at `bin/ktstr-jemalloc-
    /// alloc-worker`. Used together with `jemalloc_probe_binary` for the
    /// cross-process closed-loop test.
    jemalloc_alloc_worker_binary: Option<PathBuf>,
}

impl Default for KtstrVmBuilder {
    fn default() -> Self {
        KtstrVmBuilder {
            kernel: None,
            init_binary: None,
            scheduler_binary: None,
            run_args: Vec::new(),
            sched_args: Vec::new(),
            topology: Topology {
                llcs: 1,
                cores_per_llc: 1,
                threads_per_core: 1,
                numa_nodes: 1,
                nodes: None,
                distances: None,
            },
            memory_mb: Some(256),
            memory_min_mb: 0,
            cmdline_extra: String::new(),
            timeout: Duration::from_secs(60),
            shm_size: 0,
            monitor_thresholds: None,
            watchdog_timeout: Some(Duration::from_secs(4)),
            bpf_map_writes: Vec::new(),
            performance_mode: false,
            no_perf_mode: false,
            sched_enable_cmds: Vec::new(),
            sched_disable_cmds: Vec::new(),
            include_files: Vec::new(),
            busybox: false,
            dmesg: false,
            exec_cmd: None,
            jemalloc_probe_binary: None,
            jemalloc_alloc_worker_binary: None,
        }
    }
}

impl KtstrVmBuilder {
    /// Path to the guest kernel: either a source directory (the VMM
    /// extracts `arch/*/boot/{bzImage,Image}`) or a prebuilt image.
    pub fn kernel(mut self, path: impl Into<PathBuf>) -> Self {
        self.kernel = Some(path.into());
        self
    }

    /// Path to the userspace init binary run as PID 1 inside the
    /// guest (typically the current test binary).
    pub fn init_binary(mut self, path: impl Into<PathBuf>) -> Self {
        self.init_binary = Some(path.into());
        self
    }

    /// Path to an optional scheduler binary loaded alongside the
    /// init binary; the init spawns it before dispatching the test.
    pub fn scheduler_binary(mut self, path: impl Into<PathBuf>) -> Self {
        self.scheduler_binary = Some(path.into());
        self
    }

    /// CLI argv passed to the init binary inside the guest (typically
    /// the per-test dispatch string like `--ktstr-test-fn NAME`).
    pub fn run_args(mut self, args: &[String]) -> Self {
        self.run_args = args.to_vec();
        self
    }

    /// Extra CLI arguments appended to the scheduler binary invocation.
    #[allow(dead_code)]
    pub fn sched_args(mut self, args: &[String]) -> Self {
        self.sched_args = args.to_vec();
        self
    }

    /// Resolve the kernel image from a source-tree root (sets
    /// `kernel` to `arch/<arch>/boot/<image>`).
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

    /// Set a uniform virtual CPU topology (big-to-little:
    /// `numa_nodes, llcs, cores_per_llc, threads_per_core`).
    ///
    /// Produces a topology with uniform LLC/memory distribution and
    /// default 10/20 NUMA distances. For per-node configuration
    /// (asymmetric memory, CXL nodes, custom distances), use
    /// [`with_topology`](Self::with_topology).
    pub fn topology(mut self, numa_nodes: u32, llcs: u32, cores: u32, threads: u32) -> Self {
        self.topology = Topology::new(numa_nodes, llcs, cores, threads);
        self
    }

    /// Set a pre-constructed topology with full per-node configuration.
    ///
    /// Accepts a [`Topology`] built via [`Topology::with_nodes`] and
    /// optionally [`Topology::with_distances`], preserving per-node
    /// memory sizes, CXL memory-only nodes, and custom distance matrices.
    pub fn with_topology(mut self, topo: Topology) -> Self {
        self.topology = topo;
        self
    }

    /// Pin guest memory to an explicit MB value and clear the
    /// deferred-sizing hint. Use `memory_deferred` when the payload
    /// size should drive the allocation.
    pub fn memory_mb(mut self, mb: u32) -> Self {
        self.memory_mb = Some(mb);
        self.memory_min_mb = 0;
        self
    }

    /// Defer memory allocation until after the initramfs is built.
    ///
    /// Memory will be computed from the actual initramfs size. Use this
    /// when no explicit `--memory` override is provided.
    pub fn memory_deferred(mut self) -> Self {
        self.memory_mb = None;
        self.memory_min_mb = 0;
        self
    }

    /// Defer memory allocation with a minimum floor. The deferred path
    /// computes memory from actual initramfs size, then takes the max
    /// of that and `min_mb`. Use when the topology needs more memory
    /// than the initramfs alone requires (e.g. NUMA tests with 4096 MB).
    pub fn memory_deferred_min(mut self, min_mb: u32) -> Self {
        self.memory_mb = None;
        self.memory_min_mb = min_mb;
        self
    }

    /// Append extra tokens to the guest kernel command line. Useful
    /// for one-off debug knobs (e.g. enabling extra subsystem
    /// verbosity) that shouldn't live in `ktstr.kconfig`.
    #[allow(dead_code)]
    pub fn cmdline(mut self, extra: &str) -> Self {
        self.cmdline_extra = extra.to_string();
        self
    }

    /// Host-side watchdog timeout. The VM is killed if it has not
    /// exited on its own within this duration; the `VmResult`
    /// returned will have `timed_out = true`.
    pub fn timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }

    /// Size the guest-to-host SHM ring in bytes. `0` lets the builder
    /// derive a sensible default from the guest payload.
    #[allow(dead_code)]
    pub fn shm_size(mut self, bytes: u64) -> Self {
        self.shm_size = bytes;
        self
    }

    /// Override the `MonitorThresholds` used for stall detection and
    /// verdict rendering. Defaults to `MonitorThresholds::DEFAULT`.
    #[allow(dead_code)]
    pub fn monitor_thresholds(mut self, thresholds: crate::monitor::MonitorThresholds) -> Self {
        self.monitor_thresholds = Some(thresholds);
        self
    }

    /// Override the guest scx watchdog timeout. Applied via
    /// `scx_sched.watchdog_timeout` (7.1+) or the static
    /// `scx_watchdog_timeout` symbol (pre-7.1); silently no-ops on
    /// kernels where neither path is available.
    #[allow(dead_code)]
    pub fn watchdog_timeout(mut self, timeout: Duration) -> Self {
        self.watchdog_timeout = Some(timeout);
        self
    }

    /// Schedule a host-side write into a named BPF map after the
    /// scheduler is loaded. `map_name_suffix` is matched against
    /// `bpf_map.name` (kernel truncates to 15 chars); `offset` is
    /// the byte offset within the array-map value region; `value`
    /// is a `u32` written in native byte order.
    ///
    /// Repeated calls queue additional writes; all queued writes run
    /// sequentially on the same `BpfMapAccessor` after the scheduler
    /// attaches, with a single guest-side unblock once every write
    /// completes. Order of calls is preserved.
    #[allow(dead_code)]
    pub fn bpf_map_write(mut self, map_name_suffix: &str, offset: usize, value: u32) -> Self {
        self.bpf_map_writes.push(BpfMapWriteParams {
            map_name_suffix: map_name_suffix.to_string(),
            offset,
            value,
        });
        self
    }

    /// Enable performance mode: vCPU pinning to host LLCs,
    /// hugepage-backed guest memory, NUMA mbind, and RT scheduling
    /// on both architectures. On x86_64, additionally:
    /// KVM_HINTS_REALTIME CPUID hint (disables PV spinlocks, PV TLB
    /// flush, PV sched_yield; enables haltpoll cpuidle), PAUSE + HLT
    /// VM exit disabling via KVM_CAP_X86_DISABLE_EXITS (HLT falls
    /// back to PAUSE-only when mitigate_smt_rsb is active), and
    /// KVM_CAP_HALT_POLL skipped (guest haltpoll cpuidle disables
    /// host halt polling via MSR_KVM_POLL_CONTROL). On aarch64, KVM
    /// exit suppression and CPUID hints are not available. Validated
    /// at build time -- oversubscription returns `ResourceContention`,
    /// insufficient hugepages is a warning.
    #[allow(dead_code)]
    pub fn performance_mode(mut self, enabled: bool) -> Self {
        self.performance_mode = enabled;
        self
    }

    /// Skip flock topology reservation and force `performance_mode=false`
    /// (disables pinning, RT scheduling, hugepages, NUMA mbind, KVM exit
    /// suppression). For shared runners or unprivileged containers.
    pub fn no_perf_mode(mut self, enabled: bool) -> Self {
        self.no_perf_mode = enabled;
        self
    }

    /// Shell commands run inside the guest before the scenario to
    /// switch on a kernel-builtin scheduler (mirrors
    /// `SchedulerSpec::KernelBuiltin::enable`).
    pub fn sched_enable_cmds(mut self, cmds: &[&str]) -> Self {
        self.sched_enable_cmds = cmds.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Shell commands run inside the guest after the scenario to
    /// revert a kernel-builtin scheduler change (mirrors
    /// `SchedulerSpec::KernelBuiltin::disable`).
    pub fn sched_disable_cmds(mut self, cmds: &[&str]) -> Self {
        self.sched_disable_cmds = cmds.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Add files to include in the guest initramfs.
    /// Each entry is `(archive_path, host_path)`.
    pub fn include_files(mut self, files: Vec<(String, PathBuf)>) -> Self {
        self.include_files = files;
        self
    }

    /// Host path to `ktstr-jemalloc-probe`. When set, the probe is
    /// packed into the guest initramfs as an extra binary under
    /// `bin/` and resolves by bare name on the guest `PATH`. Tests
    /// that target the jemalloc TLS probe from a guest-side
    /// `ctx.payload(&PROBE)` invocation must set this to the host
    /// path obtained via `env!("CARGO_BIN_EXE_ktstr-jemalloc-probe")`.
    ///
    /// The probe attaches to a separately-spawned
    /// `ktstr-jemalloc-alloc-worker` via `--pid <worker_pid>`; the
    /// worker ships with DWARF, which is what the probe resolves
    /// offsets against, so the init binary does NOT need to retain
    /// DWARF. An earlier
    /// design attempted to preserve DWARF on the init binary so the
    /// probe could resolve offsets against the running init; that
    /// inflated the initramfs past practical VM memory budgets (the
    /// unstripped test binary is ~1 GB) and was abandoned in favor
    /// of routing DWARF through the probe and worker binaries.
    pub fn jemalloc_probe_binary(mut self, path: impl Into<PathBuf>) -> Self {
        self.jemalloc_probe_binary = Some(path.into());
        self
    }

    /// Host path to `ktstr-jemalloc-alloc-worker`. When set, the
    /// worker is packed alongside the probe in the guest initramfs
    /// as `/bin/ktstr-jemalloc-alloc-worker`. Used by the
    /// cross-process closed-loop test — spawned as a background
    /// payload that allocates a known number of bytes on the
    /// huge-size path (the jemalloc code path that unconditionally
    /// updates `thread_allocated` regardless of tcache state), then
    /// probed externally. The worker is much smaller than the full
    /// ktstr test binary (a single `fn main` linked against
    /// tikv-jemallocator) so shipping it keeps the initramfs well
    /// inside VM memory budgets — the init-DWARF approach that
    /// inflated the archive past those budgets was abandoned in
    /// favor of per-binary DWARF on the probe and worker.
    pub fn jemalloc_alloc_worker_binary(mut self, path: impl Into<PathBuf>) -> Self {
        self.jemalloc_alloc_worker_binary = Some(path.into());
        self
    }

    /// Embed busybox in the initramfs for shell mode.
    #[allow(dead_code)]
    pub fn busybox(mut self, enabled: bool) -> Self {
        self.busybox = enabled;
        self
    }

    /// Stream the guest kernel console (COM1/dmesg) to stderr in
    /// real time. Also bumps `loglevel=7` for verbose kernel output.
    pub fn dmesg(mut self, enabled: bool) -> Self {
        self.dmesg = enabled;
        self
    }

    /// Run a single command inside the guest instead of an
    /// interactive shell; the VM exits when the command completes.
    /// Requires `busybox(true)` and is typically paired with
    /// `KtstrVm::new_shell`.
    #[allow(dead_code)]
    pub fn exec_cmd(mut self, cmd: String) -> Self {
        self.exec_cmd = Some(cmd);
        self
    }

    /// Validate the builder configuration and materialise a [`KtstrVm`].
    ///
    /// Returns `Err` for missing required inputs (kernel, init binary),
    /// invalid topology, or host resources insufficient to satisfy
    /// `performance_mode` requirements (the last surfaces as
    /// `ResourceContention`, which callers typically treat as a
    /// skip rather than a failure).
    pub fn build(mut self) -> Result<KtstrVm> {
        if self.no_perf_mode {
            self.performance_mode = false;
        }

        let (pinning_plan, mbind_node_map, cpu_locks, no_perf_plan) = if self.no_perf_mode {
            // No-perf-mode VMs have unrestricted vCPU affinity — the
            // host kernel places their threads on any online CPU,
            // including ones a perf-mode peer has flocked and bound
            // its RT-FIFO vCPUs to. Injecting that thread competition
            // destroys perf-mode's measurement contract. The coordination
            // mechanism is an LLC-level flock set (same as
            // `kernel_build_pipeline`) so perf-mode's required
            // `LOCK_EX` blocks on any of them and fails over cleanly.
            //
            // Two paths converge here depending on `KTSTR_LLC_CAP`
            // (set by `ktstr shell --llc-cap N` or
            // `cargo ktstr shell --llc-cap N`):
            //
            //  - Cap set: `acquire_llc_plan` picks a
            //    consolidation-aware, NUMA-aware subset of N LLCs,
            //    returns its `cpus` list + RAII flock fds. The
            //    `cpus` are threaded into `no_perf_plan` so `run_vm`
            //    can `sched_setaffinity` every vCPU thread to that
            //    pool. `cpu_locks` stays empty — the plan owns the
            //    flocks.
            //  - No cap: same `acquire_llc_plan` path with
            //    `llc_cap = None`; the planner's target degenerates
            //    to "select every LLC" — the pre-flag default.
            //    The returned
            //    LlcPlan's flock set matches the pre-flag "every
            //    LLC shared" reservation. `no_perf_plan` still
            //    carries the plan so run_vm can apply the host-
            //    wide mask; on a normal host with full LLC set,
            //    sched_setaffinity with every online CPU is a
            //    no-op, matching the pre-flag behaviour.
            //
            // `from_sysfs` returning `Err` (non-Linux, sysfs absent)
            // degenerates the lock set to empty AND forces the no-cap
            // branch; no coordination is possible, but the VM still
            // runs. `KTSTR_BYPASS_LLC_LOCKS=1` bypasses both paths.
            //
            // The CLI binaries reject `--llc-cap` + bypass at parse
            // time (see `ktstr::cli::LLC_CAP_HELP` and the Shell/
            // kernel-build dispatch checks in bin/ktstr.rs and
            // bin/cargo-ktstr.rs), but library consumers building
            // a `KtstrVmBuilder` directly with both env vars set
            // would silently lose the cap under a bare `if bypass
            // { return None-plan }`. Mirror the CLI check here so
            // the enforcement contract holds for every entry point,
            // not just the ones that go through the binaries.
            let bypass = std::env::var("KTSTR_BYPASS_LLC_LOCKS")
                .ok()
                .is_some_and(|v| !v.is_empty());
            let llc_cap = host_topology::LlcCap::resolve(None)?;
            if bypass {
                if llc_cap.is_some() {
                    anyhow::bail!(
                        "no-perf-mode: KTSTR_LLC_CAP conflicts with \
                         KTSTR_BYPASS_LLC_LOCKS=1; unset one of them. \
                         KTSTR_LLC_CAP is a resource contract; bypass \
                         disables the contract entirely."
                    );
                }
                (None, Vec::new(), Vec::new(), None)
            } else if let Ok(host_topo) = host_topology::HostTopology::from_sysfs() {
                let test_topo = crate::topology::TestTopology::from_system()?;
                let plan = host_topology::acquire_llc_plan(&host_topo, &test_topo, llc_cap)?;
                host_topology::warn_if_cross_node_spill(&plan, &host_topo);
                (None, Vec::new(), Vec::new(), Some(plan))
            } else {
                if llc_cap.is_some() {
                    anyhow::bail!(
                        "--llc-cap set but host LLC topology unreadable from \
                         sysfs — cannot enforce the resource budget. Run on a \
                         host with /sys/devices/system/cpu populated, or drop \
                         --llc-cap to run without enforcement."
                    );
                }
                tracing::warn!(
                    "no-perf-mode: could not read host LLC topology from sysfs; \
                     skipping all-LLC LOCK_SH acquisition. Concurrent perf-mode \
                     runs on this host will NOT be serialized against this VM"
                );
                (None, Vec::new(), Vec::new(), None)
            }
        } else if self.performance_mode {
            let (plan, host_topo) = self.validate_performance_mode()?;
            let node_map = build_per_node_map(&plan, &host_topo, &self.topology);
            (Some(plan), node_map, Vec::new(), None)
        } else {
            let total_cpus = self.topology.total_cpus() as usize;
            let host_topo = host_topology::HostTopology::from_sysfs().ok();
            let host_cpus = host_topo
                .as_ref()
                .map(|h| h.total_cpus())
                .unwrap_or(total_cpus);
            let locks =
                host_topology::acquire_cpu_locks(total_cpus, host_cpus, host_topo.as_ref())?;
            (None, Vec::new(), locks, None)
        };

        let kernel = self.kernel.context("kernel path required")?;
        anyhow::ensure!(kernel.exists(), "kernel not found: {}", kernel.display());
        let t = &self.topology;
        anyhow::ensure!(t.llcs > 0, "llcs must be > 0");
        anyhow::ensure!(t.cores_per_llc > 0, "cores_per_llc must be > 0");
        anyhow::ensure!(t.threads_per_core > 0, "threads_per_core must be > 0");
        anyhow::ensure!(t.numa_nodes > 0, "numa_nodes must be > 0");
        // `memory_mb == Some(0)` would forward a literal `-m 0` to the
        // VMM backend (KVM rejects it at ioctl time with an opaque
        // error). Catch it here with a clear message so the caller
        // learns they set 0 explicitly rather than seeing a generic
        // kvm failure later. `None` falls back to the default (256 MB).
        if matches!(self.memory_mb, Some(0)) {
            anyhow::bail!(
                "memory_mb must be > 0 (a VM with zero memory cannot boot); \
                 omit `.memory_mb(...)` to use the builder default"
            );
        }
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

        Ok(KtstrVm {
            kernel,
            init_binary: self.init_binary,
            scheduler_binary: self.scheduler_binary,
            run_args: self.run_args,
            sched_args: self.sched_args,
            topology: self.topology,
            memory_mb: self.memory_mb,
            memory_min_mb: self.memory_min_mb,
            cmdline_extra: self.cmdline_extra,
            timeout: self.timeout,
            shm_size: self.shm_size,
            monitor_thresholds: self.monitor_thresholds,
            watchdog_timeout: self.watchdog_timeout,
            bpf_map_writes: self.bpf_map_writes,
            performance_mode: self.performance_mode,
            pinning_plan,
            mbind_node_map,
            cpu_locks,
            no_perf_plan,
            sched_enable_cmds: self.sched_enable_cmds,
            sched_disable_cmds: self.sched_disable_cmds,
            include_files: self.include_files,
            busybox: self.busybox,
            dmesg: self.dmesg,
            exec_cmd: self.exec_cmd,
            jemalloc_probe_binary: self.jemalloc_probe_binary,
            jemalloc_alloc_worker_binary: self.jemalloc_alloc_worker_binary,
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

        // Validate LLC exclusivity: each virtual LLC should map to
        // its own physical LLC group. Sum actual per-group CPU counts
        // to handle asymmetric LLCs.
        let llcs_needed = t.llcs as usize;
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
                     but only {} host CPUs available\n  \
                     hint: pass --no-perf-mode or set KTSTR_NO_PERF_MODE=1 to run without CPU reservation",
                    total_reserved,
                    reserved,
                    llcs_needed,
                    host_topo.total_cpus(),
                ),
            }));
        }

        let plan = acquire_slot_with_locks(&host_topo, t)?;

        // WARN: hugepages (only when memory is known upfront).
        if let Some(mb) = self.memory_mb {
            let free = host_topology::hugepages_free();
            let needed = host_topology::hugepages_needed(mb);
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
/// Build per-guest-NUMA-node host NUMA node mapping from a pinning plan.
fn build_per_node_map(
    plan: &host_topology::PinningPlan,
    host_topo: &host_topology::HostTopology,
    topo: &crate::vmm::topology::Topology,
) -> Vec<Vec<usize>> {
    let n = topo.numa_nodes as usize;
    let mut map: Vec<std::collections::BTreeSet<usize>> =
        vec![std::collections::BTreeSet::new(); n];
    let cpus_per_llc = topo.cores_per_llc * topo.threads_per_core;
    for &(vcpu_id, host_cpu) in &plan.assignments {
        let llc_id = vcpu_id / cpus_per_llc;
        let guest_node = topo.numa_node_of(llc_id) as usize;
        let host_node = host_topo.cpu_to_node.get(&host_cpu).copied().unwrap_or(0);
        if guest_node < n {
            map[guest_node].insert(host_node);
        }
    }
    map.into_iter().map(|s| s.into_iter().collect()).collect()
}

/// locks (non-blocking). Single pass through all available slots.
/// Returns `ResourceContention` when all slots are busy; callers
/// rely on nextest retry backoff for contention resolution.
fn acquire_slot_with_locks(
    host_topo: &host_topology::HostTopology,
    topo: &topology::Topology,
) -> Result<host_topology::PinningPlan> {
    let num_llcs = host_topo.llc_groups.len();
    let llcs_needed = topo.llcs as usize;
    let max_slots = num_llcs.checked_div(llcs_needed).unwrap_or(num_llcs).max(1);
    let llc_mode = host_topology::LlcLockMode::Exclusive;

    for slot in 0..max_slots {
        let offset = slot * llcs_needed;

        let candidate = host_topo
            .compute_pinning(topo, true, offset)
            .context("performance_mode: topology mapping")?;

        match host_topology::acquire_resource_locks(&candidate, &candidate.llc_indices, llc_mode)? {
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

    Err(anyhow::Error::new(host_topology::ResourceContention {
        reason: format!(
            "all {max_slots} LLC slots busy\n  \
             hint: pass --no-perf-mode or set KTSTR_NO_PERF_MODE=1 to run without CPU reservation"
        ),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_default() {
        let b = KtstrVmBuilder::default();
        assert_eq!(b.memory_mb, Some(256));
        assert_eq!(b.topology.total_cpus(), 1);
    }

    /// Explicit `memory_mb(0)` must be rejected at build time rather
    /// than surfacing as an opaque KVM ioctl failure later. The
    /// builder default (None→256) passes.
    #[test]
    fn builder_rejects_explicit_zero_memory() {
        // Point at a real file so the kernel-existence check
        // (which runs before the memory_mb guard) does not short-
        // circuit. /bin/true exists on every host the tests care
        // about; its contents don't matter for this check.
        let kernel = std::path::PathBuf::from("/bin/true");
        let result = KtstrVmBuilder::default()
            .kernel(&kernel)
            .memory_mb(0)
            .no_perf_mode(true)
            .build();
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("build() must reject memory_mb(0)"),
        };
        let msg = format!("{err:#}");
        assert!(
            msg.contains("memory_mb") && msg.contains("> 0"),
            "error must name the field and constraint: {msg}"
        );
    }

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
    fn builder_topology() {
        let b = KtstrVmBuilder::default().topology(1, 2, 4, 2);
        assert_eq!(b.topology.total_cpus(), 16);
        assert_eq!(b.topology.llcs, 2);
    }

    #[test]
    fn builder_requires_kernel() {
        let result = KtstrVmBuilder::default().build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_rejects_missing_kernel() {
        let result = KtstrVmBuilder::default()
            .kernel("/nonexistent/vmlinuz")
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_chain() {
        let b = KtstrVmBuilder::default()
            .topology(1, 2, 2, 2)
            .memory_mb(4096)
            .cmdline("root=/dev/sda")
            .timeout(Duration::from_secs(300));
        assert_eq!(b.memory_mb, Some(4096));
        assert_eq!(b.topology.total_cpus(), 8);
        assert_eq!(b.cmdline_extra, "root=/dev/sda");
        assert_eq!(b.timeout, Duration::from_secs(300));
    }

    #[test]
    fn builder_with_init_binary() {
        let exe = crate::resolve_current_exe().unwrap();
        let b = KtstrVmBuilder::default().init_binary(&exe);
        assert_eq!(b.init_binary.as_deref(), Some(exe.as_path()));
    }

    #[test]
    fn builder_rejects_missing_init_binary() {
        let result = KtstrVmBuilder::default()
            .kernel("/nonexistent/vmlinuz")
            .init_binary("/nonexistent/binary")
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_rejects_missing_scheduler_binary() {
        let exe = crate::resolve_current_exe().unwrap();
        let result = KtstrVmBuilder::default()
            .kernel(&exe)
            .scheduler_binary("/nonexistent/scheduler")
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_run_args() {
        let b = KtstrVmBuilder::default().run_args(&["run".into(), "--json".into()]);
        assert_eq!(b.run_args, vec!["run", "--json"]);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn builder_kernel_dir_resolves_bzimage() {
        let b = KtstrVmBuilder::default().kernel_dir("/some/linux");
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
            crash_message: None,
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
        // Second construction covers the opposite polarity of
        // every boolean/numeric field so no field is silently
        // dropped by a future refactor that only exercises the
        // success path.
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
            crash_message: None,
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
            llcs: 2,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let vm = kvm::KtstrKvm::new(topo, 128, false).unwrap();
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
        let kernel = crate::test_support::require_kernel();

        let vm = skip_on_contention!(
            KtstrVm::builder()
                .kernel(&kernel)
                .topology(1, 1, 1, 1)
                .memory_mb(256)
                .timeout(Duration::from_secs(10))
                .cmdline("loglevel=7")
                .build()
        );
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
        let kernel = crate::test_support::require_kernel();

        let vm = skip_on_contention!(
            KtstrVm::builder()
                .kernel(&kernel)
                .topology(1, 2, 2, 1) // 4 CPUs
                .memory_mb(256)
                .timeout(Duration::from_secs(10))
                .cmdline("loglevel=7")
                .build()
        );
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
        let kernel = crate::test_support::require_kernel();

        for (label, llcs, cores, threads, mem) in [("1cpu", 1, 1, 1, 256), ("4cpu", 2, 2, 1, 512)] {
            let start = Instant::now();
            let vm = match KtstrVm::builder()
                .kernel(&kernel)
                .topology(1, llcs, cores, threads)
                .memory_mb(mem)
                .timeout(Duration::from_secs(10))
                .build()
            {
                Ok(vm) => vm,
                Err(e)
                    if e.downcast_ref::<host_topology::ResourceContention>()
                        .is_some() =>
                {
                    crate::report::test_skip(format_args!("{label}: resource contention: {e}"));
                    continue;
                }
                Err(e) => panic!("{e:#}"),
            };
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
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let vm = kvm::KtstrKvm::new(topo, 64, false).unwrap();
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
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let mut vm = kvm::KtstrKvm::new(topo, 64, false).unwrap();
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
            llcs: 1,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let mut vm = kvm::KtstrKvm::new(topo, 64, false).unwrap();
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
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let mut vm = kvm::KtstrKvm::new(topo, 64, false).unwrap();
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
    #[should_panic(expected = "invalid Topology")]
    fn builder_rejects_zero_llcs() {
        KtstrVmBuilder::default().topology(1, 0, 2, 2);
    }

    #[test]
    #[should_panic(expected = "invalid Topology")]
    fn builder_rejects_zero_cores() {
        KtstrVmBuilder::default().topology(1, 2, 0, 2);
    }

    #[test]
    #[should_panic(expected = "invalid Topology")]
    fn builder_rejects_zero_threads() {
        KtstrVmBuilder::default().topology(1, 2, 2, 0);
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
            crash_message: None,
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
            ..Default::default()
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
            crash_message: None,
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

    /// Boot a kernel with vmlinux available and verify the monitor
    /// produces samples with meaningful runqueue data and degrades
    /// gracefully for scx_root-gated paths.
    ///
    /// No scheduler is loaded. Event counters (gated on scx_root)
    /// must be None. Watchdog observation may be Some on kernels
    /// with a static watchdog_timeout symbol (pre-7.1); if present,
    /// the write/read roundtrip must match.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn boot_kernel_with_monitor() {
        let kernel = crate::test_support::require_kernel();
        let _vmlinux = crate::test_support::require_vmlinux(&kernel);

        let vm = skip_on_contention!(
            KtstrVm::builder()
                .kernel(&kernel)
                .topology(1, 1, 2, 1)
                .memory_mb(256)
                .timeout(Duration::from_secs(10))
                .build()
        );
        let result = vm.run().unwrap();
        let Some(ref report) = result.monitor else {
            return;
        };
        assert!(
            report.summary.total_samples > 0,
            "monitor should have collected at least one sample"
        );
        let last = report.samples.last().unwrap();
        assert_eq!(
            last.cpus.len(),
            2,
            "topology requested 2 CPUs but monitor saw {}",
            last.cpus.len()
        );
        for (i, cpu) in last.cpus.iter().enumerate() {
            assert!(
                cpu.rq_clock > 1_000_000,
                "cpu {i}: rq_clock must be > 1ms (ns), got {}",
                cpu.rq_clock
            );
            assert!(
                cpu.rq_clock < 300_000_000_000,
                "cpu {i}: rq_clock must be < 300s (ns), got {}",
                cpu.rq_clock
            );
        }
        if let Some(ref obs) = report.watchdog_observation {
            assert_eq!(
                obs.expected_jiffies, obs.observed_jiffies,
                "watchdog write/read roundtrip mismatch: expected={} observed={}",
                obs.expected_jiffies, obs.observed_jiffies
            );
        }
        for (i, cpu) in last.cpus.iter().enumerate() {
            assert!(
                cpu.event_counters.is_none(),
                "cpu {i}: event_counters must be None when no scheduler is loaded"
            );
        }
    }

    /// Regression guard for the `scx_sched.watchdog_timeout` host-write
    /// mechanism. Boots a VM with scx-ktstr loaded plus a distinctive
    /// 7-second watchdog override, then asserts the monitor loop
    /// observed the expected jiffies value in guest memory.
    ///
    /// Skips gracefully when: no host kernel available, no vmlinux for
    /// BTF, `scx_root` symbol or `scx_sched.watchdog_timeout` BTF field
    /// missing, or the scheduler failed to attach. Real failure
    /// requires the override path to silently stop writing — which is
    /// exactly what we want to catch.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn watchdog_timeout_override_lands_in_guest_memory() {
        let kernel = crate::test_support::require_kernel();
        let vmlinux = crate::test_support::require_vmlinux(&kernel);

        // Version-dependent skips, in order of check cost. scx_root
        // is a 6.16+ symbol; its absence means either the kernel
        // predates the 6.16 scx_sched refactor (sched_ext still
        // present via the older scx_ops API, e.g. 6.14) or sched_ext
        // was not compiled in. Either way this test has nothing to
        // verify — skip. watchdog_offsets depends on BTF field layout
        // that only exists on 7.1+ kernels where
        // `scx_sched.watchdog_timeout` is a struct field.
        let syms = crate::test_support::require_kernel_symbols(&vmlinux);
        if syms.scx_root.is_none() {
            skip!("scx_root not present (needs Linux 6.16+ with sched_ext enabled)");
        }
        let offsets = crate::test_support::require_kernel_offsets(&vmlinux);
        if offsets.watchdog_offsets.is_none() {
            skip!(
                "scx_sched.watchdog_timeout field not in BTF \
                 (needs Linux 7.1+; pre-7.1 exposes watchdog timeout as a file-scope \
                 scx_watchdog_timeout symbol handled separately)"
            );
        }

        const TIMEOUT_SECS: u64 = 7;
        let hz = crate::monitor::guest_kernel_hz(Some(&kernel));
        let expected_jiffies = TIMEOUT_SECS * hz;

        let sched_bin = crate::test_support::require_binary("scx-ktstr");

        let vm = skip_on_contention!(
            KtstrVm::builder()
                .kernel(&kernel)
                .topology(1, 1, 1, 1)
                .memory_mb(256)
                .timeout(Duration::from_secs(10))
                .scheduler_binary(&sched_bin)
                .watchdog_timeout(Duration::from_secs(TIMEOUT_SECS))
                .build()
        );
        let result = vm.run().unwrap();
        let report = result.monitor.as_ref().expect(
            "ktstr: monitor report missing — require_kernel_offsets, scx_root, and \
             watchdog_offsets all resolved at setup, so monitor initialization must \
             have succeeded. A None report here is a bug in monitor startup",
        );
        let Some(obs) = &report.watchdog_observation else {
            // scx_root remained null for the whole run — the scheduler
            // never attached. Not a watchdog regression — skip.
            skip!(
                "watchdog observation missing — the scheduler did not attach \
                 (scx_root remained null throughout the run)"
            );
        };
        assert_eq!(
            obs.expected_jiffies, expected_jiffies,
            "expected_jiffies recorded by monitor ({}) does not match {} * HZ {} = {}",
            obs.expected_jiffies, TIMEOUT_SECS, hz, expected_jiffies,
        );
        assert_eq!(
            obs.observed_jiffies, obs.expected_jiffies,
            "host wrote {} jiffies to scx_sched.watchdog_timeout but guest memory holds {} — host-write mechanism broken",
            obs.expected_jiffies, obs.observed_jiffies,
        );
    }

    /// Prove the kernel uses the host-written watchdog timeout.
    ///
    /// Sets a 300-second watchdog and runs the scheduler for 15s.
    /// If the host write is effective, the kernel's watchdog timer
    /// uses 300s and no stall exit occurs. If the write were
    /// ineffective (kernel ignoring the value), the default timeout
    /// would apply and could spuriously fire on a slow guest.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn watchdog_override_prevents_stall_exit() {
        let kernel = crate::test_support::require_kernel();
        let _vmlinux = crate::test_support::require_vmlinux(&kernel);

        let sched_bin = crate::test_support::require_binary("scx-ktstr");

        let vm = skip_on_contention!(
            KtstrVm::builder()
                .kernel(&kernel)
                .topology(1, 1, 2, 1)
                .memory_mb(256)
                .timeout(Duration::from_secs(15))
                .scheduler_binary(&sched_bin)
                .watchdog_timeout(Duration::from_secs(300))
                .build()
        );
        let result = vm.run().unwrap();
        // Prior versions asserted `result.success` here. That's the
        // conjunction `!timed_out && exit_code == 0`, which depends
        // on init writing MSG_TYPE_EXIT to SHM before the AP-triggered
        // reboot propagates through the watchdog-kicks-BSP path. When
        // init is slightly slow (cold host cache, contended CPU),
        // exit_code lands at -1 (BSP run-loop default) and the
        // assertion fires even though the thing under test — scx
        // stall-exit behavior — is unaffected. Assert the actual
        // invariants instead: no guest crash, no scheduler
        // stall-exit markers in guest output. These are what would
        // change if the 300s watchdog override had failed.
        assert!(
            result.crash_message.is_none(),
            "no crash expected with 300s watchdog: {:?}",
            result.crash_message
        );
        // SCHEDULER_DIED / SCHEDULER_NOT_ATTACHED sentinels are
        // written by start_scheduler in rust_init on attach failure
        // or scheduler exit. "sched_ext: disabled" is the kernel's
        // own disable message when scx tears down a scheduler (e.g.
        // on watchdog stall). Any of these appearing proves the
        // watchdog either fired or the scheduler exited for another
        // reason — either way the test's "no stall exit" invariant
        // is broken.
        let output = &result.output;
        let stderr = &result.stderr;
        assert!(
            !output.contains(crate::test_support::SENTINEL_SCHEDULER_DIED)
                && !stderr.contains(crate::test_support::SENTINEL_SCHEDULER_DIED),
            "scheduler no longer running after 15s — either the watchdog fired or the \
             scheduler exited for another reason. output: {output:?}, stderr: {stderr:?}",
        );
        assert!(
            !output.contains(crate::test_support::SENTINEL_SCHEDULER_NOT_ATTACHED)
                && !stderr.contains(crate::test_support::SENTINEL_SCHEDULER_NOT_ATTACHED),
            "scheduler did not attach — no watchdog override to evaluate. \
             output: {output:?}, stderr: {stderr:?}",
        );
        assert!(
            !output.contains("sched_ext: disabled") && !stderr.contains("sched_ext: disabled"),
            "kernel disabled sched_ext during run — a watchdog stall or ops \
             error fired. output: {output:?}, stderr: {stderr:?}",
        );
        if let Some(ref report) = result.monitor
            && let Some(ref obs) = report.watchdog_observation
        {
            let hz = crate::monitor::guest_kernel_hz(Some(&kernel));
            let expected_jiffies = 300 * hz;
            assert_eq!(
                obs.expected_jiffies, expected_jiffies,
                "watchdog override should be 300s * HZ={hz}"
            );
            assert_eq!(
                obs.observed_jiffies, obs.expected_jiffies,
                "write/read roundtrip mismatch"
            );
        }
    }

    /// Validate that the core monitoring path reads meaningful
    /// runqueue data when a scheduler is loaded.
    ///
    /// Boots a VM with scx-ktstr, then asserts per-CPU snapshots
    /// contain plausible values. When schedstat data is present
    /// (CONFIG_SCHEDSTATS enabled), asserts sched_count is in a
    /// plausible range.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn monitor_reads_runqueue_data_with_scheduler() {
        let kernel = crate::test_support::require_kernel();
        let vmlinux = crate::test_support::require_vmlinux(&kernel);

        // Monitor-reads-runqueue asserts on cpu.rq_clock and cpu.schedstat,
        // which resolve through the non-optional rq offsets inside
        // KernelOffsets. Gating these at setup turns a silently-skipped
        // test (on BTF parse failure) into a loud infrastructure error.
        let _offsets = crate::test_support::require_kernel_offsets(&vmlinux);

        let sched_bin = crate::test_support::require_binary("scx-ktstr");

        let vm = skip_on_contention!(
            KtstrVm::builder()
                .kernel(&kernel)
                .topology(1, 1, 2, 1)
                .memory_mb(256)
                .timeout(Duration::from_secs(15))
                .scheduler_binary(&sched_bin)
                .build()
        );
        let result = vm.run().unwrap();
        let report = result.monitor.as_ref().expect(
            "ktstr: monitor report missing — require_kernel_offsets resolved at \
             setup, so monitor initialization must have succeeded. A None report \
             here is a bug in monitor startup",
        );

        assert!(
            report.summary.total_samples >= 2,
            "need at least 2 monitor samples, got {}",
            report.summary.total_samples
        );

        // Scan samples in reverse chronological order looking for the
        // first sample where EVERY CPU reports a rq_clock past the
        // early-boot noise floor (1 ms in ns). `.last()` alone flaked
        // on slow hosts where the final sample was still captured
        // mid-boot: rq_clock near zero on at least one CPU, and
        // `schedstat` / rq field reads would surface the pre-stabilization
        // state. Reverse-searching for a sample that meets the invariant
        // ANYWHERE in the run is the correct assertion — the monitor
        // captured many samples and any one of them showing a populated
        // runqueue proves the kernel path works.
        let populated = report
            .samples
            .iter()
            .rev()
            .find(|s| s.cpus.iter().all(|c| c.rq_clock > 1_000_000))
            .expect(
                "no monitor sample showed populated runqueue data — every sample \
                 was captured mid-boot with at least one CPU at rq_clock <= 1ms, \
                 or the monitor is reading the wrong rq offsets",
            );
        for (i, cpu) in populated.cpus.iter().enumerate() {
            assert!(
                cpu.rq_clock > 1_000_000,
                "cpu {i}: rq_clock must be > 1ms (ns), got {}",
                cpu.rq_clock
            );
            assert!(
                cpu.rq_clock < 300_000_000_000,
                "cpu {i}: rq_clock must be < 300s (ns), got {}",
                cpu.rq_clock
            );
        }

        for (i, cpu) in populated.cpus.iter().enumerate() {
            if let Some(ref ss) = cpu.schedstat {
                assert!(
                    ss.sched_count < 100_000_000,
                    "cpu {i}: sched_count {} exceeds plausible range — offset may be wrong",
                    ss.sched_count
                );
            }
        }
    }

    /// Validate that scx event counters populate on scx_sched kernels
    /// (Linux 6.16+). `event_offsets` resolves via either the 6.18+
    /// `scx_sched.pcpu → scx_sched_pcpu.event_stats` path or the
    /// 6.16–6.17 `scx_sched.event_stats_cpu` fallback; see
    /// `resolve_event_offsets` in `crate::monitor::btf_offsets` for
    /// the resolver that tries both.
    ///
    /// Gates on scx_root symbol presence and event_offsets BTF
    /// resolution. On pre-6.16 kernels (no scx_sched struct) or when
    /// neither BTF path resolves, event_offsets is None and this test
    /// skips.
    ///
    /// Event-counter physical-address resolution happens once at
    /// monitor start. If the scheduler has not attached by then
    /// (scx_root is still null), the monitor skips event counters
    /// for the entire run. The test skips in that case rather than
    /// asserting, matching the watchdog test's approach to
    /// scheduler-attach timing.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn event_counters_populated_with_scheduler() {
        let kernel = crate::test_support::require_kernel();
        let vmlinux = crate::test_support::require_vmlinux(&kernel);

        let syms = crate::test_support::require_kernel_symbols(&vmlinux);
        if syms.scx_root.is_none() {
            skip!("scx_root not present (needs Linux 6.16+ with sched_ext enabled)");
        }
        let offsets = crate::test_support::require_kernel_offsets(&vmlinux);
        if offsets.event_offsets.is_none() {
            skip!(
                "scx event-counter BTF fields not found \
                 (need either scx_sched.pcpu→scx_sched_pcpu.event_stats [Linux 6.18+] \
                 or scx_sched.event_stats_cpu [Linux 6.16–6.17])"
            );
        }

        let sched_bin = crate::test_support::require_binary("scx-ktstr");

        let vm = skip_on_contention!(
            KtstrVm::builder()
                .kernel(&kernel)
                .topology(1, 1, 2, 1)
                .memory_mb(256)
                .timeout(Duration::from_secs(15))
                .scheduler_binary(&sched_bin)
                .build()
        );
        let result = vm.run().unwrap();
        let report = result.monitor.as_ref().expect(
            "ktstr: monitor report missing — require_kernel_offsets, scx_root, and \
             event_offsets all resolved at setup, so monitor initialization must \
             have succeeded. A None report here is a bug in monitor startup",
        );

        assert!(
            report.summary.total_samples > 0,
            "monitor should have collected at least one sample"
        );

        let last = report.samples.last().unwrap();
        let has_event_data = last.cpus.iter().any(|c| c.event_counters.is_some());
        if !has_event_data {
            skip!(
                "event counters remained None despite resolved offsets — \
                 the scheduler may not have attached before the monitor \
                 resolved event-counter physical addresses"
            );
        }

        let any_nonzero = last.cpus.iter().any(|c| {
            c.event_counters.as_ref().is_some_and(|ev| {
                ev.select_cpu_fallback != 0
                    || ev.dispatch_local_dsq_offline != 0
                    || ev.dispatch_keep_last != 0
                    || ev.enq_skip_exiting != 0
                    || ev.enq_skip_migration_disabled != 0
            })
        });
        assert!(
            any_nonzero,
            "event counters present but all zero — offset resolution may \
             have produced addresses that read uninitialized memory"
        );
        for (i, cpu) in last.cpus.iter().enumerate() {
            if let Some(ref ev) = cpu.event_counters {
                assert!(
                    ev.select_cpu_fallback >= 0 && ev.select_cpu_fallback < 1_000_000_000,
                    "cpu {i}: select_cpu_fallback {} outside plausible range",
                    ev.select_cpu_fallback
                );
                assert!(
                    ev.dispatch_local_dsq_offline >= 0
                        && ev.dispatch_local_dsq_offline < 1_000_000_000,
                    "cpu {i}: dispatch_local_dsq_offline {} outside plausible range",
                    ev.dispatch_local_dsq_offline
                );
                assert!(
                    ev.dispatch_keep_last >= 0 && ev.dispatch_keep_last < 1_000_000_000,
                    "cpu {i}: dispatch_keep_last {} outside plausible range",
                    ev.dispatch_keep_last
                );
                assert!(
                    ev.enq_skip_exiting >= 0 && ev.enq_skip_exiting < 1_000_000_000,
                    "cpu {i}: enq_skip_exiting {} outside plausible range",
                    ev.enq_skip_exiting
                );
                assert!(
                    ev.enq_skip_migration_disabled >= 0
                        && ev.enq_skip_migration_disabled < 1_000_000_000,
                    "cpu {i}: enq_skip_migration_disabled {} outside plausible range",
                    ev.enq_skip_migration_disabled
                );
            }
        }
    }

    /// Validate that sched_domain data is populated when BTF offsets
    /// resolve. Domains are kernel-built at boot and do not require a
    /// scheduler.
    ///
    /// Gates on sched_domain_offsets BTF availability. Uses a 2-CPU
    /// topology so the domain tree spans multiple CPUs.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn sched_domain_data_populated() {
        let kernel = crate::test_support::require_kernel();
        let vmlinux = crate::test_support::require_vmlinux(&kernel);

        let offsets = crate::test_support::require_kernel_offsets(&vmlinux);
        if offsets.sched_domain_offsets.is_none() {
            skip!(
                "sched_domain BTF fields not found (likely CONFIG_SMP=n; \
                 struct sched_domain is absent or incomplete in BTF on UP kernels, \
                 and on pre-6.17 kernels the rq.sd field is also compiled out)"
            );
        }

        let vm = skip_on_contention!(
            KtstrVm::builder()
                .kernel(&kernel)
                .topology(1, 1, 2, 1)
                .memory_mb(256)
                .timeout(Duration::from_secs(10))
                .build()
        );
        let result = vm.run().unwrap();
        let report = result.monitor.as_ref().expect(
            "ktstr: monitor report missing — require_kernel_offsets and \
             sched_domain_offsets resolved at setup, so monitor initialization \
             must have succeeded. A None report here is a bug in monitor startup",
        );

        assert!(
            report.summary.total_samples > 0,
            "monitor should have collected at least one sample"
        );

        // Scan samples in reverse chronological order for the first
        // one where at least one CPU reports a non-empty sched_domains
        // list. `.last()` alone flaked on slow hosts where the final
        // sample was captured before the kernel finished building the
        // domain tree — sched_domains is populated via kernel threads
        // at boot, and the per-CPU `rq.sd` pointer lags the first rq
        // samples. Reverse-searching guards against that boot race:
        // if ANY sample in the run carries populated domains, the
        // kernel path works and the assertion passes.
        let populated = report
            .samples
            .iter()
            .rev()
            .find(|s| {
                s.cpus.iter().any(|c| {
                    c.sched_domains
                        .as_ref()
                        .is_some_and(|doms| !doms.is_empty())
                })
            })
            .unwrap_or_else(|| {
                panic!(
                    "no sample had any CPU with non-empty sched_domains across \
                     {} collected samples — monitor samples may be racing boot-time \
                     kernel thread that builds the domain tree, or `rq.sd` offsets \
                     are wrong",
                    report.samples.len(),
                );
            });

        for cpu in &populated.cpus {
            if let Some(ref doms) = cpu.sched_domains {
                if doms.is_empty() {
                    continue;
                }
                for w in doms.windows(2) {
                    assert!(
                        w[1].level > w[0].level,
                        "domain levels must be strictly increasing: {} -> {}",
                        w[0].level,
                        w[1].level
                    );
                }
                assert!(
                    doms[0].span_weight >= 2,
                    "lowest domain span_weight must be >= 2 for a 2-CPU topology, got {}",
                    doms[0].span_weight
                );
                for dom in doms {
                    assert!(
                        dom.span_weight > 0,
                        "domain level {} span_weight must be > 0",
                        dom.level
                    );
                }
            }
        }
    }

    // -- initramfs cache tests --

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

    // -- builder watchdog_timeout --

    #[test]
    fn builder_watchdog_timeout_default() {
        let b = KtstrVmBuilder::default();
        assert_eq!(b.watchdog_timeout, Some(Duration::from_secs(4)));
    }

    #[test]
    fn builder_watchdog_timeout_override() {
        let b = KtstrVmBuilder::default().watchdog_timeout(Duration::from_secs(5));
        assert_eq!(b.watchdog_timeout, Some(Duration::from_secs(5)));
    }

    #[test]
    fn builder_monitor_thresholds_sets() {
        let t = crate::monitor::MonitorThresholds {
            max_imbalance_ratio: 2.0,
            ..Default::default()
        };
        let b = KtstrVmBuilder::default().monitor_thresholds(t);
        assert!(b.monitor_thresholds.is_some());
    }

    #[test]
    fn builder_shm_size() {
        let b = KtstrVmBuilder::default().shm_size(65536);
        assert_eq!(b.shm_size, 65536);
    }

    #[test]
    fn builder_sched_args() {
        let b = KtstrVmBuilder::default().sched_args(&["--enable-borrow".into()]);
        assert_eq!(b.sched_args, vec!["--enable-borrow"]);
    }

    // -- performance_mode builder tests --

    #[test]
    fn builder_performance_mode_default_false() {
        let b = KtstrVmBuilder::default();
        assert!(!b.performance_mode);
    }

    #[test]
    fn builder_performance_mode_set() {
        let b = KtstrVmBuilder::default().performance_mode(true);
        assert!(b.performance_mode);
    }

    #[test]
    fn builder_performance_mode_false_no_validation() {
        // performance_mode=false should not trigger validation, even with
        // a topology that exceeds host capacity.
        let exe = crate::resolve_current_exe().unwrap();
        let result = KtstrVmBuilder::default()
            .kernel(&exe)
            .topology(1, 1, 1, 1)
            .performance_mode(false)
            .build();
        match result {
            Ok(_) => {}
            Err(e)
                if e.downcast_ref::<host_topology::ResourceContention>()
                    .is_some() =>
            {
                skip!("resource contention: {e}");
            }
            Err(e) => panic!("performance_mode=false should not validate host topology: {e:#}",),
        }
    }

    #[test]
    fn builder_performance_mode_oversubscribed_fails() {
        let exe = crate::resolve_current_exe().unwrap();
        let host_topo = host_topology::HostTopology::from_sysfs().unwrap();
        let too_many = host_topo.total_cpus() as u32 + 1;
        let result = KtstrVmBuilder::default()
            .kernel(&exe)
            .topology(1, 1, too_many, 1)
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
    fn builder_performance_mode_too_many_llcs_fails() {
        let exe = crate::resolve_current_exe().unwrap();
        let host_topo = host_topology::HostTopology::from_sysfs().unwrap();
        let too_many_llcs = host_topo.llc_groups.len() as u32 + 1;
        // Need total vCPUs + 1 service CPU to fit without oversubscription.
        if (too_many_llcs as usize + 1) <= host_topo.total_cpus() {
            let result = KtstrVmBuilder::default()
                .kernel(&exe)
                .topology(1, too_many_llcs, 1, 1)
                .performance_mode(true)
                .build();
            assert!(
                result.is_err(),
                "more virtual LLCs than host LLCs should fail",
            );
        }
    }

    #[test]
    fn builder_performance_mode_valid_succeeds() {
        let exe = crate::resolve_current_exe().unwrap();
        let host_topo = host_topology::HostTopology::from_sysfs().unwrap();
        if host_topo.total_cpus() < 3 {
            skip!("need >= 3 host CPUs for performance_mode test");
        }
        let result = KtstrVmBuilder::default()
            .kernel(&exe)
            .topology(1, 1, 2, 1)
            .performance_mode(true)
            .build();
        match result {
            Ok(_) => {}
            Err(e)
                if e.downcast_ref::<host_topology::ResourceContention>()
                    .is_some() =>
            {
                skip!("resource contention: {e}");
            }
            Err(e) => panic!("valid topology with performance_mode should build: {e:#}",),
        }
    }

    #[test]
    fn builder_performance_mode_preserves_in_vm() {
        let exe = crate::resolve_current_exe().unwrap();
        let host_topo = host_topology::HostTopology::from_sysfs().unwrap();
        if host_topo.total_cpus() < 3 {
            skip!("need >= 3 host CPUs for performance_mode test");
        }
        let vm = skip_on_contention!(
            KtstrVmBuilder::default()
                .kernel(&exe)
                .topology(1, 1, 2, 1)
                .performance_mode(true)
                .build()
        );
        assert!(vm.performance_mode);
    }

    #[test]
    fn builder_performance_mode_false_preserves_in_vm() {
        let exe = crate::resolve_current_exe().unwrap();
        let vm = skip_on_contention!(
            KtstrVmBuilder::default()
                .kernel(&exe)
                .topology(1, 1, 1, 1)
                .performance_mode(false)
                .build()
        );
        assert!(!vm.performance_mode);
    }

    #[test]
    fn builder_performance_mode_mbind_nodes_populated() {
        let exe = crate::resolve_current_exe().unwrap();
        let host_topo = host_topology::HostTopology::from_sysfs().unwrap();
        if host_topo.total_cpus() < 3 {
            skip!("need >= 3 host CPUs for performance_mode test");
        }
        let vm = KtstrVmBuilder::default()
            .kernel(&exe)
            .topology(1, 1, 2, 1)
            .performance_mode(true)
            .build();
        if let Ok(vm) = vm {
            assert!(
                !vm.mbind_node_map.is_empty(),
                "mbind_node_map should be populated for performance_mode",
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

    // -- RT scheduling tests --

    #[test]
    fn set_rt_priority_applies_when_capable() {
        // Probe CAP_SYS_NICE via a direct sched_setscheduler call
        // first: RT policies require the capability, and CI
        // containers frequently drop it. If the probe fails, skip
        // rather than fail — the permission check is the feature
        // under test.
        let param = libc::sched_param { sched_priority: 1 };
        let rc = unsafe { libc::sched_setscheduler(0, libc::SCHED_FIFO, &param) };
        if rc != 0 {
            skip!("no CAP_SYS_NICE capability available");
        }
        let policy = unsafe { libc::sched_getscheduler(0) };
        assert_eq!(policy, libc::SCHED_FIFO);
        let mut out_param: libc::sched_param = unsafe { std::mem::zeroed() };
        unsafe { libc::sched_getparam(0, &mut out_param) };
        assert_eq!(out_param.sched_priority, 1);
        // Restore SCHED_OTHER so later tests in the same nextest
        // process don't inherit this thread's RT policy.
        let restore = libc::sched_param { sched_priority: 0 };
        unsafe { libc::sched_setscheduler(0, libc::SCHED_OTHER, &restore) };
    }

    /// `set_rt_priority` emits a `tracing::warn!` with the
    /// "need CAP_SYS_NICE" substring when `sched_setscheduler`
    /// returns an error — the warn-and-proceed invariant that keeps
    /// vCPU threads running in unprivileged containers with the
    /// default scheduling policy instead of failing the VM.
    ///
    /// Captures tracing output via `tracing_test::traced_test` so the
    /// assertion observes the actual warn event (not just "the call
    /// did not panic"). Runs ONLY when the test process lacks
    /// CAP_SYS_NICE — if the capability is present, the success
    /// branch fires instead and the warn is never emitted, leaving
    /// nothing to assert; in that case we restore SCHED_OTHER on
    /// the probe thread and skip.
    #[test]
    #[tracing_test::traced_test]
    fn set_rt_priority_warns_without_cap() {
        // Probe CAP_SYS_NICE: if we CAN set SCHED_FIFO, the test
        // can't exercise the warn path. Restore SCHED_OTHER and
        // skip — we can't observe the warn event without actually
        // failing the syscall.
        let probe = libc::sched_param { sched_priority: 1 };
        let rc = unsafe { libc::sched_setscheduler(0, libc::SCHED_FIFO, &probe) };
        if rc == 0 {
            // Restore SCHED_OTHER so later tests don't inherit RT.
            let restore = libc::sched_param { sched_priority: 0 };
            unsafe { libc::sched_setscheduler(0, libc::SCHED_OTHER, &restore) };
            skip!("CAP_SYS_NICE present — cannot exercise warn path");
        }
        // Now we know the syscall will fail. Call set_rt_priority
        // and assert the warn event fires with the expected
        // substring. `logs_contain` is injected into the test by
        // the `#[traced_test]` macro and scans the per-test tracing
        // buffer.
        set_rt_priority(1, "test-thread");
        assert!(
            logs_contain("need CAP_SYS_NICE"),
            "warn event must include the 'need CAP_SYS_NICE' hint \
             so operators reading stderr know what permission to \
             grant",
        );
        assert!(
            logs_contain("SCHED_FIFO"),
            "warn event must name the policy whose attachment failed",
        );
        assert!(
            logs_contain("test-thread"),
            "warn event must name the label so operators can attribute \
             the warning to a specific vCPU / monitor / watchdog thread",
        );
    }

    // -- aarch64 boot tests --

    /// Find an aarch64 kernel suitable for boot tests.
    /// Accepts both raw Image and gzip-compressed vmlinuz — load_kernel
    /// decompresses transparently.
    #[cfg(target_arch = "aarch64")]
    fn find_aarch64_image() -> Option<std::path::PathBuf> {
        crate::find_kernel().unwrap()
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn boot_kernel_produces_output_aarch64() {
        let Some(kernel) = find_aarch64_image() else {
            skip!("no aarch64 kernel image found");
        };

        let vm = skip_on_contention!(
            KtstrVm::builder()
                .kernel(&kernel)
                .topology(1, 1, 1, 1)
                .memory_mb(256)
                .timeout(Duration::from_secs(10))
                .cmdline("loglevel=7")
                .build()
        );
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
        let Some(kernel) = find_aarch64_image() else {
            skip!("no aarch64 kernel image found");
        };

        let vm = skip_on_contention!(
            KtstrVm::builder()
                .kernel(&kernel)
                .topology(1, 2, 2, 1) // 4 CPUs
                .memory_mb(256)
                .timeout(Duration::from_secs(10))
                .cmdline("loglevel=7")
                .build()
        );
        let result = vm.run().unwrap();
        assert!(!result.stderr.is_empty(), "no console output from SMP boot");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn aarch64_kvm_has_immediate_exit() {
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let vm = kvm::KtstrKvm::new(topo, 64, false).unwrap();
        assert!(
            vm.has_immediate_exit,
            "KVM_CAP_IMMEDIATE_EXIT should be available on modern kernels"
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn builder_kernel_dir_resolves_image() {
        let b = KtstrVmBuilder::default().kernel_dir("/some/linux");
        assert_eq!(
            b.kernel.as_deref(),
            Some(std::path::Path::new("/some/linux/arch/arm64/boot/Image"))
        );
    }
}
