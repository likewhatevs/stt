//! vCPU thread infrastructure: signal-based kicks, immediate_exit
//! handles, affinity / RT scheduling helpers, and the freeze
//! coordinator's per-AP state.
//!
//! Each vCPU runs on its own host thread inside `KVM_RUN`. Kicking a
//! vCPU out of guest mode requires (a) writing the
//! `kvm_run.immediate_exit` byte from outside the thread (the
//! Firecracker pattern) and (b) sending the dedicated `SIGRTMIN`
//! signal so the in-progress ioctl returns `EINTR`. This module owns
//! the cross-thread handle ([`ImmediateExitHandle`]), the signal
//! handler registration, and the `VcpuThread` struct used by the run
//! orchestrator.
//!
//! Affinity helpers ([`pin_current_thread`], [`set_thread_cpumask`])
//! and RT priority ([`set_rt_priority`]) live here too — they're
//! shared between the BSP / AP run loops and the host-side
//! `LlmExtract` pipeline (which broadens its own mask after a
//! perf-mode VM run).

use std::os::unix::thread::JoinHandleExt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::exit_dispatch;
use crate::monitor;

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
///
/// Clone+Copy so multiple threads (vCPU loop, watchdog, freeze coordinator)
/// can each carry a handle pointing at the same MAP_SHARED `kvm_run` page.
/// All writes go through `set` (single-byte `write_volatile`), so a value
/// copy of `Self` is exactly equivalent to a borrowed reference for the
/// access pattern KVM cares about.
#[derive(Clone, Copy)]
pub(crate) struct ImmediateExitHandle {
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
    pub(crate) fn from_vcpu(vcpu: &mut kvm_ioctls::VcpuFd) -> Self {
        let kvm_run = vcpu.get_kvm_run();
        let ptr: *mut u8 = &mut kvm_run.immediate_exit;
        Self { ptr }
    }

    /// Set `immediate_exit` to the given value.
    pub(crate) fn set(&self, val: u8) {
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

/// Convert a host-side `Duration` to guest jiffies, using the
/// guest kernel's CONFIG_HZ.
///
/// Computed as `(d.as_millis() * hz) / 1000` rather than
/// `d.as_secs() * hz` so sub-second durations don't truncate to 0
/// — a 500 ms watchdog with HZ=1000 should land at 500 jiffies, not
/// at 0 (the bug that masked the early-trigger path before this
/// helper existed). Truncation is to the jiffies tick boundary
/// (1000/HZ ms), which is the kernel's own arithmetic precision.
///
/// Two call sites today: the freeze coordinator's
/// `half_threshold_jiffies` (compares against scanned per-task
/// runnable-age in jiffies) and the `watchdog_override` setup
/// (writes a jiffies count into `scx_sched.watchdog_timeout` in
/// guest memory). Both pre-existed scattered as inline expressions;
/// centralising the conversion keeps the precision rule in one
/// place and eliminates drift opportunities.
pub(crate) fn duration_to_jiffies(d: Duration, hz: u64) -> u64 {
    // saturating_mul guards against the theoretical overflow of
    // pathologically-large `Duration` * pathologically-large `hz`.
    // Real ktstr inputs (watchdog_timeout in seconds, HZ in 100..1000)
    // never approach the u64 boundary, but a `Duration::MAX` /
    // `u64::MAX` HZ pair would otherwise wrap and silently produce a
    // garbage jiffies value. Saturating to u64::MAX (then `/ 1000`)
    // at least keeps the threshold check semantics "this jiffies
    // count is unreachable" rather than "this jiffies count is small,
    // so the trigger fires immediately".
    (d.as_millis() as u64).saturating_mul(hz) / 1000
}

/// Signal used to kick vCPU threads out of KVM_RUN.
/// All three Rust reference VMMs (Firecracker, Cloud Hypervisor, libkrun)
/// use SIGRTMIN. SIGUSR1/SIGUSR2 conflict with application-level signals.
pub(crate) fn vcpu_signal() -> libc::c_int {
    libc::SIGRTMIN()
}

/// Resolve the byte offset of `ktstr_err_exit_detected` within the
/// probe BPF program's `.bss` section by walking the program's BTF
/// Datasec. Returns `None` when any step fails (program BTF not yet
/// loaded, struct btf untranslatable, blob short-read, BTF parse
/// reject, no matching VarSecinfo).
///
/// `btf_kva` is the kernel KVA of the probe map's `struct btf`;
/// `base` is the host's parsed vmlinux BTF used as the split-BTF
/// base when the program BTF is split. Lives next to
/// [`vcpu_signal`] because the freeze coordinator is the sole
/// consumer.
pub(crate) fn load_probe_bss_offset(
    kernel: &crate::monitor::guest::GuestKernel<'_>,
    btf_kva: u64,
    base: &btf_rs::Btf,
    offsets: &crate::monitor::btf_offsets::BpfMapOffsets,
) -> Option<u32> {
    let mem = kernel.mem();
    let cr3_pa = kernel.cr3_pa();
    let page_offset = kernel.page_offset();
    let l5 = kernel.l5();
    let btf_pa = crate::monitor::idr::translate_any_kva(mem, cr3_pa, page_offset, btf_kva, l5)?;
    let data_kva = mem.read_u64(btf_pa, offsets.btf_data);
    let data_size = mem.read_u32(btf_pa, offsets.btf_data_size) as usize;
    let base_kva = mem.read_u64(btf_pa, offsets.btf_base_btf);
    if data_kva == 0 || data_size == 0 {
        return None;
    }
    if data_size > crate::monitor::dump::MAX_BTF_BLOB {
        return None;
    }
    // The chunked vmalloc reader handles per-page translate + bulk
    // copy and honours all-or-nothing on short reads — the previous
    // hand-rolled loop here duplicated `GuestKernel::read_kva_bytes_chunked`
    // for no benefit.
    let blob = kernel.read_kva_bytes_chunked(data_kva, data_size)?;
    let prog_btf = if base_kva != 0 {
        btf_rs::Btf::from_split_bytes(&blob, base).ok()?
    } else {
        btf_rs::Btf::from_bytes(&blob).ok()?
    };
    crate::monitor::btf_offsets::resolve_var_offset_in_section(
        &prog_btf,
        ".bss",
        "ktstr_err_exit_detected",
    )
}

/// Signal handler — Firecracker pattern.
/// The handler itself is a no-op; its sole purpose is to cause KVM_RUN
/// to return with EINTR. The fence ensures that a write to
/// `kvm_run.immediate_exit` from another thread (via ImmediateExitHandle)
/// is visible when KVM_RUN returns. This Acquire fence pairs with the
/// proximal `Ordering::Release` fence in [`super::freeze_coord`]'s
/// freeze coordinator — the `std::sync::atomic::fence(Ordering::Release)`
/// that runs between pass 1 (writing `kvm_run.immediate_exit` for every
/// vCPU via `ImmediateExitHandle::set(1)`) and pass 2 (issuing
/// `pthread_kill(tid, SIGRTMIN)` for every vCPU). The Release fence
/// publishes every immediate_exit byte before any signal is delivered;
/// the Acquire fence here, executed when the signal handler runs in the
/// receiving vCPU thread, observes those writes. Without the pair, a
/// vCPU could process its signal, re-enter KVM_RUN, and miss the
/// immediate_exit byte that was supposed to short-circuit guest entry.
extern "C" fn vcpu_signal_handler(_: libc::c_int, _: *mut libc::siginfo_t, _: *mut libc::c_void) {
    std::sync::atomic::fence(Ordering::Acquire);
}

/// Register the vCPU signal handler and unblock the signal in this thread.
/// Must be called from each vCPU thread before entering the run loop.
/// Follows Firecracker's register_kick_signal_handler + QEMU's
/// kvm_init_cpu_signals: register SA_SIGINFO handler, then unblock via
/// pthread_sigmask so the signal is deliverable inside KVM_RUN.
pub(crate) fn register_vcpu_signal_handler() {
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
pub(crate) fn pin_current_thread(cpu: usize, label: &str) {
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
/// Used by the no-perf + `--cpu-cap` path at
/// [`KtstrVmBuilder::build`]: every vCPU thread gets the reserved
/// LLC's CPUs as its mask so the vCPU runs inside the resource
/// budget without fighting the kernel scheduler for a hard pin it
/// doesn't actually need.
///
/// Logs success or warning; does not fail the VM.
///
/// Best-effort partial-mask semantics: a single bad CPU (out of
/// `CpuSet`'s static bitmap range) does NOT abort the whole call.
/// The bad entry is logged and skipped, and the resulting mask
/// reflects every CPU that fit. This is preferable to the
/// alternative — silently inheriting whatever overly-narrow mask
/// the thread already had (often a single-CPU perf-mode pin) and
/// quietly losing the broadening the caller asked for. The only
/// case that early-returns is "every requested CPU was rejected,"
/// which would otherwise call `sched_setaffinity` with an empty
/// mask and block the thread forever.
///
/// `pub(crate)` so non-vmm consumers (the host-side LlmExtract
/// pipeline in `test_support::eval`) can use the same primitive
/// to broaden the calling thread's mask before running inference,
/// which would otherwise inherit a perf-mode single-CPU pin from
/// the just-finished VM run.
pub(crate) fn set_thread_cpumask(cpus: &[usize], label: &str) {
    // Build the cpuset by adding every CPU we can. A bad CPU
    // (out-of-range for `CpuSet`'s static bitmap, currently 1024 on
    // x86_64) skips that single entry and continues the loop rather
    // than aborting the whole call. The early-return form gave us
    // the worst of both worlds: the thread inherited whatever
    // overly-narrow mask was already in place (e.g. a single-CPU
    // perf-mode pin) and the caller silently lost the broadening
    // it asked for. A partial mask — every CPU that fit, minus the
    // bad one — preserves most of the intent and remains observable
    // via the per-skip warning + the post-loop summary.
    let mut cpuset = nix::sched::CpuSet::new();
    let mut applied: Vec<usize> = Vec::with_capacity(cpus.len());
    let mut skipped: Vec<usize> = Vec::new();
    for &cpu in cpus {
        match cpuset.set(cpu) {
            Ok(()) => applied.push(cpu),
            Err(e) => {
                eprintln!("no_perf_mode: WARNING: cpuset.set({cpu}) for {label}: {e}; skipping");
                skipped.push(cpu);
            }
        }
    }
    if !skipped.is_empty() {
        eprintln!(
            "no_perf_mode: {label}: skipped {} of {} requested CPUs ({skipped:?}); proceeding with {applied:?}",
            skipped.len(),
            cpus.len(),
        );
    }
    // If every requested CPU failed to bind we have nothing to apply
    // — calling sched_setaffinity with an empty mask would block the
    // thread forever. Bail rather than mask to zero.
    if applied.is_empty() {
        eprintln!(
            "no_perf_mode: WARNING: {label}: no valid CPUs to mask (every requested entry failed)"
        );
        return;
    }
    match nix::sched::sched_setaffinity(nix::unistd::Pid::from_raw(0), &cpuset) {
        Ok(()) => eprintln!("no_perf_mode: mask {label} to host CPUs {applied:?}"),
        Err(e) => eprintln!("no_perf_mode: WARNING: mask {label} to {applied:?}: {e}"),
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
pub(crate) fn set_rt_priority(priority: i32, label: &str) {
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

/// Wait for every vCPU thread's TID to publish into its slot, then
/// open per-vCPU `perf_event_open` counters bound to those TIDs. The
/// returned [`monitor::perf_counters::PerfCountersCapture`] is shared
/// (via `Arc`) by the monitor sampling loop and the freeze
/// coordinator so the per-tick timeline and the freeze-instant
/// snapshot read through the same fds — opening twice would burn
/// twice the perf slots and produce two slightly-different time
/// bases.
///
/// `vcpu_tid_slots[i]` is the AP-thread-published TID for vCPU `i`
/// (0 = BSP, written synchronously before this function runs). AP
/// slots may still be `0` if an AP hasn't reached its
/// `tid_slot.store` yet; poll up to 1s. Any slot still `0` at the
/// deadline is treated as "no perf data for that vCPU"; the whole
/// capture returns `None` so the timeline + freeze paths consume
/// `Option::as_ref()` and emit `None` per-CPU.
///
/// Failure paths (perf_event_paranoid too high, missing
/// CAP_PERFMON, hardware lacks the requested counter) log a warning
/// via `tracing::warn!` and return `None`. The dump pipeline still
/// runs without per-vCPU perf data.
pub(crate) fn open_vcpu_perf_capture(
    vcpu_tid_slots: &[Arc<AtomicI32>],
) -> Option<monitor::perf_counters::PerfCountersCapture> {
    let perf_deadline = Instant::now() + Duration::from_secs(1);
    let mut tids: Vec<libc::pid_t> = Vec::with_capacity(vcpu_tid_slots.len());
    for slot in vcpu_tid_slots {
        let mut v = slot.load(Ordering::Acquire);
        while v == 0 && Instant::now() < perf_deadline {
            std::thread::sleep(Duration::from_millis(10));
            v = slot.load(Ordering::Acquire);
        }
        tids.push(v);
    }
    if !tids.iter().all(|&t| t > 0) {
        let missing: Vec<usize> = tids
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| (t == 0).then_some(i))
            .collect();
        tracing::warn!(
            ?missing,
            "vCPU TID slots never published; per-vCPU perf capture disabled"
        );
        return None;
    }
    match monitor::perf_counters::PerfCountersCapture::open(&tids) {
        Ok(cap) => Some(cap),
        Err(e) => {
            tracing::warn!(
                err = %e,
                "perf_event_open failed; per-vCPU IPC/cache-miss capture disabled"
            );
            None
        }
    }
}

// ---------------------------------------------------------------------------
// VcpuThread — Cloud Hypervisor pattern with Firecracker's immediate_exit
// ---------------------------------------------------------------------------

/// Per-vCPU thread handle with signal-based kick and ACK flag.
pub(crate) struct VcpuThread {
    pub(crate) handle: JoinHandle<kvm_ioctls::VcpuFd>,
    /// Set by the thread after it exits the KVM_RUN loop.
    pub(crate) exited: Arc<AtomicBool>,
    /// Handle to set `kvm_run.immediate_exit` from outside the vCPU thread.
    /// `None` when KVM_CAP_IMMEDIATE_EXIT is not available.
    pub(crate) immediate_exit: Option<ImmediateExitHandle>,
}

/// Per-AP freeze-rendezvous state held outside `VcpuThread`. Cloned
/// out of `spawn_ap_threads` and into the freeze coordinator at run
/// startup; not needed for teardown (kick/join), so it lives apart
/// from `VcpuThread` to keep that struct minimal.
pub(crate) struct ApFreezeHandles {
    /// Per-AP `parked` ack flags. Set by the AP thread when it has
    /// completed the freeze drain dance and is parked, awaiting
    /// clearance to resume. The freeze coordinator polls each entry
    /// with Acquire ordering before reading guest memory; the
    /// thread's prior Release store synchronizes-with that load,
    /// providing the happens-before edge that makes host-side
    /// guest-memory reads consistent on weakly-ordered
    /// architectures.
    pub(crate) parked: Vec<Arc<AtomicBool>>,
    /// Per-AP register-snapshot slots captured at freeze time
    /// (RIP/RSP/CR3 on x86_64, PC/SP/TTBR1+TTBR0 on aarch64). Written
    /// by the AP thread on its own thread (KVM_GET_REGS is fd-bound
    /// and not safe cross-thread) just before the `parked` Release
    /// store; read by the freeze coordinator after the rendezvous
    /// Acquire. `None` until the first freeze fires; reset to
    /// `None` on thaw is NOT done — a successive freeze overwrites
    /// with the new capture.
    pub(crate) regs: Vec<Arc<std::sync::Mutex<Option<exit_dispatch::VcpuRegSnapshot>>>>,
}

impl VcpuThread {
    /// Kick a vCPU out of KVM_RUN. If immediate_exit is available, sets the
    /// flag before sending the signal (Firecracker pattern). Otherwise falls
    /// back to signal-only (the signal handler causes EINTR).
    pub(crate) fn kick(&self) {
        if let Some(ref ie) = self.immediate_exit {
            ie.set(1);
            std::sync::atomic::fence(Ordering::Release);
        }
        self.signal();
    }

    /// Send the kick signal to interrupt a blocked KVM_RUN.
    pub(crate) fn signal(&self) {
        unsafe {
            libc::pthread_kill(self.handle.as_pthread_t() as libc::pthread_t, vcpu_signal());
        }
    }

    /// Wait for the thread to exit, retrying the kick periodically.
    /// Cloud Hypervisor pattern: poll exited flag, re-kick every 10ms.
    pub(crate) fn wait_for_exit(&self, timeout: Duration) {
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

/// Parameters for a host-side BPF map write during VM execution.
#[derive(Clone)]
pub(crate) struct BpfMapWriteParams {
    pub(crate) map_name_suffix: String,
    pub(crate) offset: usize,
    pub(crate) value: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vmm::kvm;
    use crate::vmm::topology::Topology;

    #[test]
    fn vcpu_signal_is_sigrtmin() {
        let sig = vcpu_signal();
        assert!(sig >= libc::SIGRTMIN(), "signal should be >= SIGRTMIN");
        assert!(sig <= libc::SIGRTMAX(), "signal should be <= SIGRTMAX");
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

    /// Pin the millisecond-precision Duration→jiffies conversion.
    /// Sub-second inputs must NOT truncate to 0 (the bug that masked
    /// the freeze-coord early trigger before this helper existed),
    /// whole-second inputs must scale by HZ, and HZ != 1000 must
    /// scale correctly down to the jiffies tick boundary.
    #[test]
    fn duration_to_jiffies_basic() {
        // 500 ms at HZ=1000 → 500 jiffies (the bug case: as_secs()
        // would yield 0 here).
        assert_eq!(duration_to_jiffies(Duration::from_millis(500), 1000), 500);
        // 1500 ms at HZ=1000 → 1500 jiffies (the fractional-second
        // input path must not truncate the integer-seconds component
        // either).
        assert_eq!(duration_to_jiffies(Duration::from_millis(1500), 1000), 1500);
        // 4 s at HZ=250 → 1000 jiffies (lower HZ tick rate; the
        // ms→jiffies arithmetic should land on the same answer as
        // the as_secs()*hz form for whole seconds).
        assert_eq!(duration_to_jiffies(Duration::from_secs(4), 250), 1000);
        // Zero duration → zero jiffies (no UB, no spurious tick).
        assert_eq!(duration_to_jiffies(Duration::from_millis(0), 1000), 0);
        // Degenerate HZ=0 → zero jiffies. Guards against an
        // unresolvable guest-side CONFIG_HZ where
        // `monitor::guest_kernel_hz` falls back to 0; the resulting
        // `half_threshold_jiffies` of 0 means "early-trigger threshold
        // never fires," which is the right degradation — better than
        // a divide-by-zero or an unbounded sentinel that would fire
        // on every iteration.
        assert_eq!(duration_to_jiffies(Duration::from_secs(1), 0), 0);
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
}
