pub mod acpi;
pub mod boot;
pub mod console;
pub mod initramfs;
pub mod kvm;
pub mod mptable;
pub mod shm_ring;
pub mod topology;

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
use vm_memory::{Bytes, GuestAddress, GuestMemory};

use crate::monitor;

// ---------------------------------------------------------------------------
// Initramfs cache — two-tier: POSIX shm (cross-process) + in-process HashMap
// ---------------------------------------------------------------------------

/// Cache key for base initramfs (payload + scheduler, no args).
/// Derived from a content hash of the binary files so identical inputs
/// produce the same key regardless of path or mtime.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BaseKey(u64);

/// Hash a file for cache keying. Samples length + first 4KB + last 4KB
/// rather than reading the entire file, so the cost is constant regardless
/// of binary size.
pub(crate) fn hash_file_sample(path: &Path) -> Result<u64> {
    use std::io::{Seek, SeekFrom};
    const SAMPLE: usize = 4096;

    let mut f =
        std::fs::File::open(path).with_context(|| format!("open for hash: {}", path.display()))?;
    let len = f
        .metadata()
        .with_context(|| format!("stat for hash: {}", path.display()))?
        .len();

    let mut hasher = std::hash::DefaultHasher::new();
    len.hash(&mut hasher);

    let mut buf = [0u8; SAMPLE];
    let n = std::io::Read::read(&mut f, &mut buf)
        .with_context(|| format!("read head: {}", path.display()))?;
    hasher.write(&buf[..n]);

    if len > SAMPLE as u64 {
        f.seek(SeekFrom::End(-(SAMPLE as i64)))
            .with_context(|| format!("seek tail: {}", path.display()))?;
        let n = std::io::Read::read(&mut f, &mut buf)
            .with_context(|| format!("read tail: {}", path.display()))?;
        hasher.write(&buf[..n]);
    }

    Ok(hasher.finish())
}

impl BaseKey {
    pub(crate) fn new(payload: &Path, scheduler: Option<&Path>) -> Result<Self> {
        let mut hasher = std::hash::DefaultHasher::new();

        hash_file_sample(payload)?.hash(&mut hasher);

        match scheduler {
            Some(s) => {
                1u8.hash(&mut hasher);
                hash_file_sample(s)?.hash(&mut hasher);
            }
            None => 0u8.hash(&mut hasher),
        }

        Ok(BaseKey(hasher.finish()))
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
            // and fast to re-acquire, so copying ~183MB into an Arc is waste.
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
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Address where initramfs is loaded in guest memory.
const INITRD_ADDR: u64 = 0x800_0000; // 128 MB

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
    /// Override for `scx_watchdog_timeout` in the guest kernel (jiffies).
    /// Written to guest memory via the monitor thread after the kernel boots.
    watchdog_timeout_jiffies: Option<u64>,
}

impl SttVm {
    pub fn builder() -> SttVmBuilder {
        SttVmBuilder::default()
    }

    /// Boot the VM, run until halt/shutdown/timeout, return captured output.
    pub fn run(&self) -> Result<VmResult> {
        let start = Instant::now();

        // Spawn initramfs resolution in parallel with KVM creation.
        let initramfs_handle = self.spawn_initramfs_resolve();
        let (vm, kernel_result) = self.create_vm_and_load_kernel()?;
        self.setup_memory(&vm, &kernel_result, initramfs_handle)?;
        self.setup_vcpus(&vm, kernel_result.entry)?;
        tracing::debug!(elapsed_us = start.elapsed().as_micros(), "total_setup");

        let (exit_code, timed_out, ap_threads, monitor_handle, com1, com2, kill, vm) =
            self.run_vm(start, vm)?;

        self.collect_results(
            start,
            exit_code,
            timed_out,
            ap_threads,
            monitor_handle,
            com1,
            com2,
            kill,
            vm,
        )
    }

    /// Create the KVM VM and load the kernel. Returns the VM and kernel
    /// load result. Spawns the initramfs-resolve thread to run in parallel.
    fn create_vm_and_load_kernel(&self) -> Result<(kvm::SttKvm, boot::KernelLoadResult)> {
        let t0 = Instant::now();
        let vm = kvm::SttKvm::new(self.topology, self.memory_mb).context("create VM")?;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "kvm_create");

        let t0 = Instant::now();
        let kernel_result =
            boot::load_kernel(&vm.guest_mem, &self.kernel).context("load kernel")?;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "load_kernel");

        Ok((vm, kernel_result))
    }

    /// Spawn initramfs resolution on a background thread.
    /// Returns the handle to join later (after KVM creation completes).
    fn spawn_initramfs_resolve(&self) -> Option<JoinHandle<Result<BaseRef>>> {
        let bin = self.init_binary.as_ref()?;
        let payload = bin.clone();
        let scheduler = self.scheduler_binary.clone();
        std::thread::Builder::new()
            .name("initramfs-resolve".into())
            .spawn(move || -> Result<BaseRef> {
                let extras: Vec<(&str, &std::path::Path)> = scheduler
                    .as_deref()
                    .map(|p| vec![("scheduler", p)])
                    .unwrap_or_default();
                let key = BaseKey::new(&payload, scheduler.as_deref())?;
                get_or_build_base(&payload, &extras, &key)
            })
            .ok()
    }

    /// Join the initramfs thread and load the result into guest memory.
    fn join_and_load_initramfs(
        &self,
        vm: &kvm::SttKvm,
        handle: JoinHandle<Result<BaseRef>>,
    ) -> Result<(Option<u64>, Option<u32>)> {
        let t0 = Instant::now();
        let base = handle
            .join()
            .map_err(|_| anyhow::anyhow!("initramfs-resolve thread panicked"))??;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "initramfs_join");
        let base_bytes: &[u8] = base.as_ref();

        let t0 = Instant::now();
        let suffix = initramfs::build_suffix(base_bytes.len(), &self.run_args, &self.sched_args)?;
        tracing::debug!(
            elapsed_us = t0.elapsed().as_micros(),
            base_bytes = base_bytes.len(),
            suffix_bytes = suffix.len(),
            "build_suffix",
        );

        let t0 = Instant::now();
        let (addr, size) =
            initramfs::load_initramfs_parts(&vm.guest_mem, &[base_bytes, &suffix], INITRD_ADDR)?;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "load_initramfs");
        Ok((Some(addr), Some(size)))
    }

    /// Write cmdline, boot params, SHM header, and topology tables to guest memory.
    fn setup_memory(
        &self,
        vm: &kvm::SttKvm,
        kernel_result: &boot::KernelLoadResult,
        initramfs_handle: Option<JoinHandle<Result<BaseRef>>>,
    ) -> Result<()> {
        let (initrd_addr, initrd_size) = match initramfs_handle {
            Some(handle) => self.join_and_load_initramfs(vm, handle)?,
            None => (None, None),
        };

        let mut cmdline = concat!(
            "console=ttyS0 earlyprintk=serial nomodules mitigations=off ",
            "no_timer_check clocksource=kvm-clock ",
            "random.trust_cpu=on swiotlb=noforce ",
            "i8042.noaux i8042.nomux i8042.nopnp i8042.dumbkbd ",
            "pci=off reboot=k panic=-1 iomem=relaxed ",
            "8250.nr_uarts=2",
        )
        .to_string();
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
            vm.guest_mem
                .write_slice(
                    zerocopy::IntoBytes::as_bytes(&header),
                    GuestAddress(shm_base),
                )
                .context("write SHM header")?;
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
    fn run_vm(
        &self,
        start: Instant,
        mut vm: kvm::SttKvm,
    ) -> Result<(
        i32,
        bool,
        Vec<VcpuThread>,
        Option<JoinHandle<Vec<monitor::MonitorSample>>>,
        Arc<std::sync::Mutex<console::Serial>>,
        Arc<std::sync::Mutex<console::Serial>>,
        Arc<AtomicBool>,
        kvm::SttKvm,
    )> {
        let com1 = Arc::new(std::sync::Mutex::new(console::Serial::new(
            console::COM1_BASE,
        )));
        let com2 = Arc::new(std::sync::Mutex::new(console::Serial::new(
            console::COM2_BASE,
        )));

        // Register serial EventFds with KVM's irqfd so vm-superio trigger()
        // calls inject the corresponding IRQ into the guest. Without this the
        // 8250 driver never receives THRE interrupts and write() hangs.
        if !vm.split_irqchip {
            vm.vm_fd
                .register_irqfd(com1.lock().unwrap().irq_evt(), console::COM1_IRQ)
                .context("register COM1 irqfd")?;
            vm.vm_fd
                .register_irqfd(com2.lock().unwrap().irq_evt(), console::COM2_IRQ)
                .context("register COM2 irqfd")?;
        }

        let kill = Arc::new(AtomicBool::new(false));

        let has_immediate_exit = vm.has_immediate_exit;
        let mut vcpus = std::mem::take(&mut vm.vcpus);
        let mut bsp = vcpus.remove(0);

        let ap_threads = self.spawn_ap_threads(vcpus, has_immediate_exit, &com1, &com2, &kill)?;

        let monitor_handle = self.start_monitor(&vm, &kill, start)?;

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
        let watchdog = std::thread::Builder::new()
            .name("vmm-watchdog".into())
            .spawn(move || {
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
            self.run_bsp(&mut bsp, &com1, &com2, &kill, has_immediate_exit, start);
        bsp_done.store(true, Ordering::Release);
        eprintln!("BSP: exited run loop, code={exit_code} timed_out={timed_out}");

        // Join the watchdog before dropping `bsp`. The watchdog holds an
        // ImmediateExitHandle pointing into bsp's kvm_run mmap. If bsp is
        // dropped first, the watchdog may write to unmapped memory.
        let _ = watchdog.join();

        Ok((
            exit_code,
            timed_out,
            ap_threads,
            monitor_handle,
            com1,
            com2,
            kill,
            vm,
        ))
    }

    /// Spawn AP vCPU threads.
    fn spawn_ap_threads(
        &self,
        vcpus: Vec<kvm_ioctls::VcpuFd>,
        has_immediate_exit: bool,
        com1: &Arc<std::sync::Mutex<console::Serial>>,
        com2: &Arc<std::sync::Mutex<console::Serial>>,
        kill: &Arc<AtomicBool>,
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

            let handle = std::thread::Builder::new()
                .name(format!("vcpu-{}", i + 1))
                .spawn(move || {
                    register_vcpu_signal_handler();
                    vcpu_run_loop(&mut vcpu, &com1_clone, &com2_clone, &kill_clone);
                    exited_clone.store(true, Ordering::Release);
                    // Return VcpuFd instead of dropping it here. The main
                    // thread drops it after join, ensuring the kvm_run mmap
                    // stays valid throughout the kick/wait_for_exit sequence.
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
        vm: &kvm::SttKvm,
        kill: &Arc<AtomicBool>,
        start: Instant,
    ) -> Result<Option<JoinHandle<Vec<monitor::MonitorSample>>>> {
        let Some(vmlinux) = find_vmlinux(&self.kernel) else {
            return Ok(None);
        };
        let offsets = monitor::btf_offsets::KernelOffsets::from_vmlinux(&vmlinux);
        let symbols = monitor::symbols::KernelSymbols::from_vmlinux(&vmlinux);

        let (Ok(offsets), Ok(symbols)) = (offsets, symbols) else {
            return Ok(None);
        };

        let host_base = vm.guest_mem.get_host_address(GuestAddress(0)).unwrap() as *const u8;
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

        let watchdog_jiffies = self.watchdog_timeout_jiffies;

        let handle = std::thread::Builder::new()
            .name("vmm-monitor".into())
            .spawn(move || {
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

                monitor::reader::monitor_loop(
                    &mem,
                    &rq_pas,
                    &offsets,
                    event_pcpu_pas.as_deref(),
                    Duration::from_millis(100),
                    &kill_clone,
                    start,
                    dump_trigger.as_ref(),
                    watchdog_override.as_ref(),
                )
            })
            .context("spawn monitor thread")?;

        Ok(Some(handle))
    }

    /// BSP KVM_RUN loop. Returns (exit_code, timed_out).
    fn run_bsp(
        &self,
        bsp: &mut kvm_ioctls::VcpuFd,
        com1: &Arc<std::sync::Mutex<console::Serial>>,
        com2: &Arc<std::sync::Mutex<console::Serial>>,
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
                Ok(VcpuExit::IoOut(port, data)) => {
                    if dispatch_io_out(com1, com2, port, data) {
                        exit_code = 0;
                        break;
                    }
                }
                Ok(VcpuExit::IoIn(port, data)) => {
                    dispatch_io_in(com1, com2, port, data);
                }
                Ok(VcpuExit::Hlt) => {
                    exit_code = 0;
                    break;
                }
                Ok(VcpuExit::Shutdown) => {
                    exit_code = 0;
                    break;
                }
                Ok(VcpuExit::MmioRead(_addr, data)) => {
                    for byte in data.iter_mut() {
                        *byte = 0xff;
                    }
                }
                Ok(VcpuExit::MmioWrite(_addr, _data)) => {}
                Ok(VcpuExit::FailEntry(reason, _cpu)) => {
                    tracing::error!(reason, "BSP VM entry failed");
                    break;
                }
                Ok(VcpuExit::InternalError) => {
                    tracing::error!("BSP internal error");
                    break;
                }
                Ok(VcpuExit::SystemEvent(event_type, _)) => {
                    if event_type == KVM_SYSTEM_EVENT_SHUTDOWN
                        || event_type == KVM_SYSTEM_EVENT_RESET
                    {
                        exit_code = 0;
                        break;
                    }
                }
                Ok(_) => continue,
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
    #[allow(clippy::too_many_arguments)]
    fn collect_results(
        &self,
        start: Instant,
        mut exit_code: i32,
        timed_out: bool,
        ap_threads: Vec<VcpuThread>,
        monitor_handle: Option<JoinHandle<Vec<monitor::MonitorSample>>>,
        com1: Arc<std::sync::Mutex<console::Serial>>,
        com2: Arc<std::sync::Mutex<console::Serial>>,
        kill: Arc<AtomicBool>,
        vm: kvm::SttKvm,
    ) -> Result<VmResult> {
        kill.store(true, Ordering::Release);

        // Kick APs still in KVM_RUN, then join. Skip APs that already
        // exited — their VcpuFd (and kvm_run mmap) may be dropped, so
        // writing to ImmediateExitHandle would hit unmapped memory.
        for vt in &ap_threads {
            if !vt.exited.load(Ordering::Acquire) {
                vt.kick();
            }
        }
        for vt in ap_threads {
            vt.wait_for_exit(Duration::from_secs(5));
            // join() returns the VcpuFd — it drops HERE, after all kicks
            // are done. This ensures the kvm_run mmap (used by
            // ImmediateExitHandle) stays valid throughout wait_for_exit.
            let _ = vt.handle.join();
        }

        let monitor_report = monitor_handle.and_then(|h| h.join().ok()).map(|samples| {
            let summary = monitor::MonitorSummary::from_samples(&samples);
            monitor::MonitorReport { samples, summary }
        });

        let (shm_data, stimulus_events) = if self.shm_size > 0 {
            let mem_size = (self.memory_mb as u64) << 20;
            let shm_base = mem_size - self.shm_size;
            let shm_size = self.shm_size as usize;
            let mut shm_buf = vec![0u8; shm_size];
            vm.guest_mem
                .read_slice(&mut shm_buf, GuestAddress(shm_base))
                .context("read SHM region")?;
            let drain = shm_ring::shm_drain(&shm_buf, 0);
            let events: Vec<shm_ring::StimulusEvent> = drain
                .entries
                .iter()
                .filter(|e| e.msg_type == shm_ring::MSG_TYPE_STIMULUS && e.crc_ok)
                .filter_map(|e| shm_ring::StimulusEvent::from_payload(&e.payload))
                .collect();
            (Some(drain), events)
        } else {
            (None, Vec::new())
        };

        let app_output = com2.lock().unwrap().output();
        let console_output = com1.lock().unwrap().output();

        if let Some(line) = app_output
            .lines()
            .rev()
            .find(|l| l.starts_with("STT_EXIT="))
            && let Ok(code) = line.trim_start_matches("STT_EXIT=").trim().parse::<i32>()
        {
            exit_code = code;
        }

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
        })
    }
}

// ---------------------------------------------------------------------------
// I/O dispatch — shared between BSP and AP run loops
// ---------------------------------------------------------------------------

const KVM_SYSTEM_EVENT_SHUTDOWN: u32 = 1;
const KVM_SYSTEM_EVENT_RESET: u32 = 2;

/// I8042 ports and commands — minimal emulation for guest reboot.
/// Reference: firecracker and cloud-hypervisor both emulate i8042 for
/// x86 guest shutdown. The kernel's default reboot method (`reboot=k`)
/// writes CMD_RESET_CPU (0xFE) to the i8042 command port (0x64).
const I8042_DATA_PORT: u16 = 0x60;
const I8042_CMD_PORT: u16 = 0x64;
const I8042_CMD_RESET_CPU: u8 = 0xFE;

/// Dispatch an I/O out to serial ports or system devices.
/// Returns `true` if the caller should exit (system reset detected).
fn dispatch_io_out(
    com1: &std::sync::Mutex<console::Serial>,
    com2: &std::sync::Mutex<console::Serial>,
    port: u16,
    data: &[u8],
) -> bool {
    // I8042 reset: kernel writes 0xFE to port 0x64 during reboot.
    if port == I8042_CMD_PORT && data.first() == Some(&I8042_CMD_RESET_CPU) {
        return true;
    }
    // Only lock the matching serial port based on port range.
    if (console::COM1_BASE..console::COM1_BASE + 8).contains(&port) {
        com1.lock().unwrap().handle_out(port, data);
    } else if (console::COM2_BASE..console::COM2_BASE + 8).contains(&port) {
        com2.lock().unwrap().handle_out(port, data);
    }
    false
}

/// Dispatch an I/O in from serial ports or system devices.
/// Handles i8042 reads to satisfy the kernel's keyboard probe.
fn dispatch_io_in(
    com1: &std::sync::Mutex<console::Serial>,
    com2: &std::sync::Mutex<console::Serial>,
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
            com1.lock().unwrap().handle_in(port, data);
        }
        p if (console::COM2_BASE..console::COM2_BASE + 8).contains(&p) => {
            com2.lock().unwrap().handle_in(port, data);
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// vCPU run loop — Firecracker/CH hybrid pattern
// ---------------------------------------------------------------------------

/// Per-vCPU KVM_RUN loop for AP threads.
/// Checks kill flag before and after every KVM_RUN.
/// On EINTR: clears immediate_exit (QEMU kvm_eat_signals pattern) and
/// re-checks kill flag before re-entering KVM_RUN.
/// HLT for APs is normal — KVM wakes them on SIPI/interrupt delivery.
fn vcpu_run_loop(
    vcpu: &mut kvm_ioctls::VcpuFd,
    com1: &Arc<std::sync::Mutex<console::Serial>>,
    com2: &Arc<std::sync::Mutex<console::Serial>>,
    kill: &Arc<AtomicBool>,
) {
    loop {
        if kill.load(Ordering::Acquire) {
            break;
        }

        match vcpu.run() {
            Ok(VcpuExit::IoOut(port, data)) => {
                if dispatch_io_out(com1, com2, port, data) {
                    kill.store(true, Ordering::Release);
                    break;
                }
            }
            Ok(VcpuExit::IoIn(port, data)) => {
                dispatch_io_in(com1, com2, port, data);
            }
            Ok(VcpuExit::Hlt) => {
                // AP halted — KVM wakes it on interrupt delivery (SIPI/timer).
                // Check kill between HLT exits for clean shutdown.
                if kill.load(Ordering::Acquire) {
                    break;
                }
            }
            Ok(VcpuExit::Shutdown) => break,
            Ok(VcpuExit::SystemEvent(event_type, _)) => {
                if event_type == KVM_SYSTEM_EVENT_SHUTDOWN || event_type == KVM_SYSTEM_EVENT_RESET {
                    break;
                }
            }
            Ok(VcpuExit::MmioRead(_addr, data)) => {
                for b in data.iter_mut() {
                    *b = 0xff;
                }
            }
            Ok(VcpuExit::MmioWrite(_addr, _data)) => {}
            Ok(VcpuExit::FailEntry(_, _)) | Ok(VcpuExit::InternalError) => break,
            Ok(_) => {}
            Err(e) => {
                if e.errno() == libc::EINTR || e.errno() == libc::EAGAIN {
                    // QEMU kvm_eat_signals pattern: clear immediate_exit so
                    // the next KVM_RUN doesn't exit immediately again.
                    vcpu.set_kvm_immediate_exit(0);
                    if kill.load(Ordering::Acquire) {
                        break;
                    }
                    continue;
                }
                // Before SIPI delivery, APs may get errors — check kill.
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
// vmlinux discovery
// ---------------------------------------------------------------------------

/// Find the vmlinux ELF next to a kernel bzImage path.
///
/// Checks the bzImage's parent directory and, if the path looks like
/// `<root>/arch/x86/boot/bzImage`, checks `<root>/vmlinux` as well.
fn find_vmlinux(kernel_path: &Path) -> Option<PathBuf> {
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
    watchdog_timeout_jiffies: Option<u64>,
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
            watchdog_timeout_jiffies: None,
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
        self.kernel = Some(dir.join("arch/x86/boot/bzImage"));
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
    pub fn watchdog_timeout_jiffies(mut self, jiffies: u64) -> Self {
        self.watchdog_timeout_jiffies = Some(jiffies);
        self
    }

    pub fn build(self) -> Result<SttVm> {
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
            watchdog_timeout_jiffies: self.watchdog_timeout_jiffies,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialize boot tests that create full VMs. Running multiple VMs
    /// simultaneously causes signal delivery contention (SIGRTMIN for
    /// vCPU kick) and serial output loss.
    static BOOT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
    fn ap_mp_state_set_correctly() {
        let topo = Topology {
            sockets: 2,
            cores_per_socket: 2,
            threads_per_core: 1,
        };
        let vm = kvm::SttKvm::new(topo, 128).unwrap();
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
    fn boot_kernel_produces_output() {
        let _lock = BOOT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
    fn boot_kernel_smp_topology() {
        let _lock = BOOT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
    /// IS the boot time. With `panic=-1`, the kernel halts immediately
    /// on panic, causing KVM_EXIT_HLT which returns to userspace.
    #[test]
    fn bench_boot_time() {
        let _lock = BOOT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
    fn kvm_has_immediate_exit_cap() {
        let topo = Topology {
            sockets: 1,
            cores_per_socket: 1,
            threads_per_core: 1,
        };
        let vm = kvm::SttKvm::new(topo, 64).unwrap();
        // KVM_CAP_IMMEDIATE_EXIT has been available since Linux 4.12.
        assert!(
            vm.has_immediate_exit,
            "KVM_CAP_IMMEDIATE_EXIT should be available on modern kernels"
        );
    }

    #[test]
    fn immediate_exit_handle_set_clear() {
        let topo = Topology {
            sockets: 1,
            cores_per_socket: 1,
            threads_per_core: 1,
        };
        let mut vm = kvm::SttKvm::new(topo, 64).unwrap();
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
    fn immediate_exit_handle_cross_vcpu() {
        let topo = Topology {
            sockets: 1,
            cores_per_socket: 2,
            threads_per_core: 1,
        };
        let mut vm = kvm::SttKvm::new(topo, 64).unwrap();
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
    fn vcpu_thread_kick_sets_immediate_exit() {
        let topo = Topology {
            sockets: 1,
            cores_per_socket: 1,
            threads_per_core: 1,
        };
        let mut vm = kvm::SttKvm::new(topo, 64).unwrap();
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
            total_samples: 5,
            max_imbalance_ratio: 3.5,
            max_local_dsq_depth: 10,
            stall_detected: true,
            event_deltas: None,
        };
        let report = monitor::MonitorReport {
            samples: vec![],
            summary: summary.clone(),
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
    fn boot_kernel_with_monitor() {
        let _lock = BOOT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
        let tmp = std::env::temp_dir().join("stt-cache-content-test");
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
    fn hash_file_sample_large_file() {
        let tmp = std::env::temp_dir().join("stt-hash-sample-test");
        std::fs::create_dir_all(&tmp).unwrap();
        let f = tmp.join("big");
        // 16KB file — exercises both head and tail sampling.
        let data: Vec<u8> = (0..16384).map(|i| (i % 256) as u8).collect();
        std::fs::write(&f, &data).unwrap();
        let h = hash_file_sample(&f).unwrap();
        // Same content should produce same hash.
        assert_eq!(h, hash_file_sample(&f).unwrap());
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
    fn dispatch_io_out_i8042_reset() {
        let com1 = std::sync::Mutex::new(console::Serial::new(console::COM1_BASE));
        let com2 = std::sync::Mutex::new(console::Serial::new(console::COM2_BASE));
        assert!(dispatch_io_out(
            &com1,
            &com2,
            I8042_CMD_PORT,
            &[I8042_CMD_RESET_CPU]
        ));
    }

    #[test]
    fn dispatch_io_out_i8042_non_reset() {
        let com1 = std::sync::Mutex::new(console::Serial::new(console::COM1_BASE));
        let com2 = std::sync::Mutex::new(console::Serial::new(console::COM2_BASE));
        assert!(!dispatch_io_out(&com1, &com2, I8042_CMD_PORT, &[0x00]));
    }

    #[test]
    fn dispatch_io_out_serial_com1() {
        let com1 = std::sync::Mutex::new(console::Serial::new(console::COM1_BASE));
        let com2 = std::sync::Mutex::new(console::Serial::new(console::COM2_BASE));
        // Write 'A' to COM1 THR — should not trigger reset.
        assert!(!dispatch_io_out(&com1, &com2, console::COM1_BASE, b"A"));
    }

    #[test]
    fn dispatch_io_out_serial_com2() {
        let com1 = std::sync::Mutex::new(console::Serial::new(console::COM1_BASE));
        let com2 = std::sync::Mutex::new(console::Serial::new(console::COM2_BASE));
        assert!(!dispatch_io_out(&com1, &com2, console::COM2_BASE, b"B"));
        let output = com2.lock().unwrap().output();
        assert!(output.contains('B'));
    }

    #[test]
    fn dispatch_io_out_unknown_port() {
        let com1 = std::sync::Mutex::new(console::Serial::new(console::COM1_BASE));
        let com2 = std::sync::Mutex::new(console::Serial::new(console::COM2_BASE));
        assert!(!dispatch_io_out(&com1, &com2, 0x1234, &[0xFF]));
    }

    #[test]
    fn dispatch_io_in_i8042_status() {
        let com1 = std::sync::Mutex::new(console::Serial::new(console::COM1_BASE));
        let com2 = std::sync::Mutex::new(console::Serial::new(console::COM2_BASE));
        let mut data = [0xFFu8; 1];
        dispatch_io_in(&com1, &com2, I8042_CMD_PORT, &mut data);
        assert_eq!(data[0], 0);
    }

    #[test]
    fn dispatch_io_in_i8042_data() {
        let com1 = std::sync::Mutex::new(console::Serial::new(console::COM1_BASE));
        let com2 = std::sync::Mutex::new(console::Serial::new(console::COM2_BASE));
        let mut data = [0xFFu8; 1];
        dispatch_io_in(&com1, &com2, I8042_DATA_PORT, &mut data);
        assert_eq!(data[0], 0);
    }

    #[test]
    fn dispatch_io_in_unknown_port() {
        let com1 = std::sync::Mutex::new(console::Serial::new(console::COM1_BASE));
        let com2 = std::sync::Mutex::new(console::Serial::new(console::COM2_BASE));
        let mut data = [0xFFu8; 1];
        dispatch_io_in(&com1, &com2, 0x1234, &mut data);
        assert_eq!(data[0], 0xFF, "unknown port should not modify data");
    }

    // -- builder watchdog_timeout_jiffies --

    #[test]
    fn builder_watchdog_timeout() {
        let b = SttVmBuilder::default().watchdog_timeout_jiffies(5000);
        assert_eq!(b.watchdog_timeout_jiffies, Some(5000));
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
