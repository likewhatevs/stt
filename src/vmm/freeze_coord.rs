//! Run-loop orchestration for `KtstrVm`: spawning AP vCPU threads,
//! the freeze coordinator, the BPF map writer, the BSP loop, and
//! result collection. This is the kernel-boundary heart of the VMM
//! runtime — every method here runs after [`super::setup`] hands the
//! configured [`KtstrKvm`](super::kvm::KtstrKvm) over and before the
//! VM exits.
//!
//! Reopens [`impl KtstrVm`](super::KtstrVm) so the canonical struct
//! definition stays in [`super`].

use anyhow::{Context, Result};
use kvm_ioctls::VcpuExit;
use std::os::unix::thread::JoinHandleExt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use vm_memory::{Bytes, GuestAddress, GuestMemory};

use crate::monitor;

use super::exit_dispatch::{self, ExitAction, classify_exit, vcpu_run_loop_unified};
use super::pi_mutex::PiMutex;
use super::result::{VmResult, VmRunState};
use super::vcpu::{
    ApFreezeHandles, BpfMapWriteParams, ImmediateExitHandle, VcpuThread, duration_to_jiffies,
    load_probe_bss_offset, open_vcpu_perf_capture, pin_current_thread,
    register_vcpu_signal_handler, set_rt_priority, set_thread_cpumask, vcpu_signal,
};
use super::vmlinux::find_vmlinux;
use super::{KtstrVm, console, shm_ring, vcpu_panic, virtio_blk, virtio_console, virtio_net};

#[cfg(target_arch = "aarch64")]
use super::aarch64::kvm;
#[cfg(target_arch = "x86_64")]
use super::x86_64::kvm;

// `DRAM_BASE` is defined in `super` and used here for guest-memory
// host-address resolution. The const is arch-gated; the import
// follows the same gating implicitly via where it is consumed.
use super::DRAM_BASE;

/// Maximum wall-clock duration the freeze coordinator will wait for
/// every vCPU to acknowledge parked state before logging a timeout
/// and giving up on the dump. Well above the worst-case drain-dance
/// and single-iteration park latency on healthy guests; a real
/// timeout indicates a vCPU stuck in KVM_RUN that the
/// `immediate_exit` kick failed to interrupt.
const FREEZE_RENDEZVOUS_TIMEOUT: Duration = Duration::from_secs(30);

impl KtstrVm {
    /// Spawn threads and run the BSP. Returns all state needed for
    /// `collect_results`.
    ///
    /// # Failure-dump freeze
    ///
    /// When the BPF probe latches a sched_ext error-class exit
    /// (SCX_EXIT_ERROR / _BPF / _STALL), a host-side coordinator
    /// thread freezes every vCPU long enough to read BPF map state
    /// for post-mortem analysis. The freeze is transparent to test
    /// authors — the test still observes the same failure verdict
    /// and exit path — but adds up to ~10 ms of thaw latency to the
    /// failure path (the parked-vCPU poll cadence). Healthy runs
    /// never enter the freeze path; the latch only fires on an
    /// error-class scheduler exit.
    pub(super) fn run_vm(&self, run_start: Instant, mut vm: kvm::KtstrKvm) -> Result<VmRunState> {
        let com1 = Arc::new(PiMutex::new(console::Serial::new(console::COM1_BASE)));
        let com2 = Arc::new(PiMutex::new(console::Serial::new(console::COM2_BASE)));

        // Register serial EventFds with KVM's irqfd for interrupt-driven TX.
        // Split-irqchip mode lacks IOAPIC routing (LAPIC-only kernel
        // emulation; PIC/IOAPIC live in userspace and the framework
        // does not implement the userspace IOAPIC dispatch). Without
        // an IRQ delivery path the guest's serial driver hangs on the
        // first TX/RX wake — the kernel uart driver has no polling
        // fallback. Reject loudly so test setups exceeding the
        // 8-bit xAPIC limit (max APIC ID > 254) are caught here
        // instead of producing a silent guest hang.
        #[cfg(target_arch = "x86_64")]
        if vm.split_irqchip {
            anyhow::bail!(
                "serial COM1/COM2 require irqfd; split-irqchip mode \
                 has no IOAPIC and the kernel uart driver has no \
                 polling fallback — reduce topology so all APIC IDs \
                 are at or below 254 (MAX_XAPIC_ID)",
            );
        }
        #[cfg(target_arch = "x86_64")]
        {
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

        // Optional virtio-blk: `None` when no disks are attached,
        // `Some` when the builder has at least one `DiskConfig`.
        // Constructed BEFORE we tear down vm.vcpus so the helper
        // can still read `vm.guest_mem` and the irqchip state.
        let virtio_blk = self.init_virtio_blk(&vm)?;

        // Optional virtio-net: `None` when the builder has no
        // `NetConfig` attached, `Some` when configured. Same
        // construction-before-vcpu-takedown rule as virtio-blk.
        let virtio_net = self.init_virtio_net(&vm)?;

        let kill = Arc::new(AtomicBool::new(false));
        // Failure-dump freeze rendezvous: broadcast `freeze` flag plus a
        // per-vCPU `parked` ACK, parallel to the existing `kill` +
        // `exited` shutdown rendezvous. The freeze coordinator
        // (spawned below alongside the watchdog) polls the BPF probe's
        // `ktstr_err_exit_detected` .bss flag via `BpfMapAccessor`;
        // when the flag flips it sets `freeze`, kicks every vCPU,
        // awaits N-of-N parked confirmations, runs the dump (placeholder
        // in this batch), and then clears `freeze` to thaw.
        let freeze = Arc::new(AtomicBool::new(false));
        let bsp_parked = Arc::new(AtomicBool::new(false));
        let bsp_regs: Arc<std::sync::Mutex<Option<exit_dispatch::VcpuRegSnapshot>>> =
            Arc::new(std::sync::Mutex::new(None));

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

        // No-perf + --cpu-cap: flat CPU list from the LLC plan gets
        // sched_setaffinity'd on every vCPU thread as a mask (not a
        // hard pin). Mutually exclusive with perf-mode's pin_targets.
        let no_perf_mask: Option<&[usize]> = self.no_perf_plan.as_ref().map(|p| p.cpus.as_slice());

        // Per-AP TID slots — each AP thread stamps gettid() into its
        // slot at startup so the monitor can open per-vCPU
        // perf_event_open counters bound to the right thread.
        // Index = AP index (0-based among APs); the BSP TID is stamped
        // into a separate slot below since it runs on the current
        // thread.
        let ap_tid_slots: Vec<Arc<AtomicI32>> = (0..vcpus.len())
            .map(|_| Arc::new(AtomicI32::new(0)))
            .collect();

        let (ap_threads, ap_freeze_handles) = self.spawn_ap_threads(
            vcpus,
            has_immediate_exit,
            &com1,
            &com2,
            None,
            virtio_blk.as_ref(),
            virtio_net.as_ref(),
            &kill,
            &freeze,
            &ap_pins,
            no_perf_mask,
            &ap_tid_slots,
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

        // Build the per-vCPU TID vec the monitor needs for
        // `perf_event_open(2)`. Index 0 is the BSP — running on this
        // thread, so SYS_gettid here returns the current thread's
        // TID. Indexes 1..n are AP slots stamped by each AP thread at
        // startup. Slots may still be 0 here if an AP hasn't reached
        // its tid_slot.store; the monitor polls them with a deadline
        // before opening counters and skips per-vCPU perf for any
        // slot still 0 at the deadline.
        let bsp_tid_slot = Arc::new(AtomicI32::new(unsafe {
            libc::syscall(libc::SYS_gettid) as i32
        }));
        let vcpu_tid_slots: Vec<Arc<AtomicI32>> = std::iter::once(bsp_tid_slot)
            .chain(ap_tid_slots.iter().cloned())
            .collect();

        // Open per-vCPU `perf_event_open` counters once at run-vm
        // scope so both the monitor thread (per-tick timeline) and
        // the freeze coordinator (freeze-instant snapshot) can read
        // through a shared `Arc`. Polling vCPU TIDs here (rather than
        // inside the monitor closure) lets the freeze coord see a
        // consistent capture immediately when the latch fires —
        // before the monitor has even taken its first sample. AP
        // threads stamp their TID into the slots before they enter
        // KVM_RUN; BSP slot is stamped synchronously above.
        // `Arc<Option<...>>` lets a host that lacks
        // `perf_event_open` permission still run the rest of the
        // dump pipeline; the inner Option is None and every
        // consumer's `as_ref()` chain produces None for that field.
        let perf_capture = Arc::new(open_vcpu_perf_capture(&vcpu_tid_slots));

        let monitor_handle =
            self.start_monitor(&vm, &kill, run_start, vcpu_pthreads, perf_capture.clone())?;

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
                    use vm_memory::GuestMemoryRegion;
                    let host_base = vm
                        .guest_mem
                        .get_host_address(GuestAddress(DRAM_BASE))
                        .context("resolve guest DRAM base host address (watchdog)")?;
                    // Size of the first contiguous region only.
                    // host_base addresses that single mapping; using the
                    // sum of all region lengths would extend past the
                    // mapping into host heap when multiple regions exist.
                    // `guest_mem` is constructed with at least one region
                    // by every `KtstrKvm` constructor; surfacing this as
                    // an error rather than `expect()` keeps the
                    // VM-builder path panic-free even if a future refactor
                    // introduces a degenerate zero-region path.
                    let mem_size = vm
                        .guest_mem
                        .iter()
                        .next()
                        .context("guest_mem must have at least one region (watchdog)")?
                        .len();
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

        // Freeze coordinator thread: triggers a failure-dump freeze when
        // the BPF probe's `ktstr_err_exit_detected` .bss latch fires
        // (sched_ext error-class exit observed by tp_btf inside
        // probe.bpf.c). The flag lives in the probe BPF program's
        // .bss map — the coordinator polls it via host-side guest
        // physical memory access, NOT via SHM TLV. Discovery is
        // lazy: each iteration tries `BpfMapAccessor::find_map(".bss")`
        // until the probe is loaded into map_idr, caches the
        // value-region PA, then polls `mem.read_u32(pa, 0)` until
        // non-zero.
        //
        // Sequencing combines Cloud Hypervisor's pause/snapshot
        // pattern (drain dance + N-of-N rendezvous on parked acks)
        // with Firecracker's SIGRTMIN+immediate_exit kick:
        //   1. observe `ktstr_err_exit_detected != 0` via .bss read
        //   2. set `freeze=true`
        //   3. set every vCPU's immediate_exit=1 (two-pass kick: all
        //      flags first, then signal all)
        //   4. signal every vCPU thread (pthread_kill SIGRTMIN)
        //   5. wait for N-of-N parked acks (Acquire-load on each
        //      `parked` flag — synchronizes-with the vCPU's Release
        //      store after the drain dance, providing the happens-
        //      before edge that makes guest-memory reads correct on
        //      weakly-ordered architectures)
        //   6. call dump_state to read BPF map state, vCPU regs,
        //      and per-CPU prog/cputime captures into a
        //      FailureDumpReport, then emit the report as JSON
        //      via tracing::error and the optional file sink
        //   7. clear freeze=false; each parked vCPU polling on
        //      park_timeout(10ms) observes the clear within 10 ms
        //      and resumes — no explicit unpark needed
        //
        // DMA quiescence: virtio-blk's independent worker thread
        // is paused before the vCPU SIGRTMIN kick (see
        // `blk.lock().pause()` in freeze_and_capture below); the
        // rendezvous waits for the worker's paused ack alongside
        // the vCPU parked acks. virtio-net (v0) and virtio-console
        // run synchronously on the vCPU thread, so they freeze
        // automatically once the vCPU rendezvous completes. A
        // future device with its own worker thread would need to
        // be added to the pause sequence.
        let freeze_coord_freeze = freeze.clone();
        let freeze_coord_kill = kill.clone();
        // Optional virtio-blk handle for the failure-dump
        // worker-pause rendezvous. None when no disk is attached.
        // Cloned into the closure so the dump path can call
        // `dev.lock().pause()` BEFORE kicking the vCPUs and
        // `dev.lock().resume()` after the dump completes — without
        // this, the worker thread would continue mutating the
        // backing file (and the avail/used rings) while the host
        // reads guest memory for the dump. Only virtio-blk has an
        // independent worker thread; virtio-net (v0) and
        // virtio-console run synchronously on the vCPU thread and
        // are automatically frozen when the vCPU rendezvous
        // completes (their `mmio_write` handlers must have already
        // returned for the vCPU to reach the parked state).
        let freeze_coord_virtio_blk = virtio_blk.clone();
        let freeze_coord_bsp_parked = bsp_parked.clone();
        let freeze_coord_bsp_regs = bsp_regs.clone();
        let freeze_coord_bsp_done = bsp_done.clone();
        // Shared per-vCPU perf-counter capture. The Arc lets the
        // monitor sampling loop (per-tick timeline) and the freeze
        // coordinator (freeze-instant snapshot) read through the same
        // fds. Inner `Option` is `None` when `perf_event_open` was
        // unavailable on the host; both consumers gracefully degrade
        // to "no perf data" without aborting the run.
        let freeze_coord_perf_capture = perf_capture.clone();
        let freeze_coord_vmlinux = find_vmlinux(&self.kernel);
        // Optional file sink for the failure-dump JSON. Cloned out
        // of the builder field so the closure owns a copy and the
        // freeze coord can write the file without touching the env
        // or the parent `KtstrVm`.
        let freeze_coord_dump_path = self.failure_dump_path.clone();
        // Dual-snapshot mode: when true, the freeze coordinator
        // additionally polls per-CPU `rq->scx.runnable_list` for any
        // task whose `jiffies - p->scx.runnable_at` crosses
        // `watchdog_timeout/2`, takes a snapshot at that point, and
        // wraps both early + late snapshots into a
        // [`monitor::dump::DualFailureDumpReport`]. Set by
        // `attempt_auto_repro` for the repro VM only.
        let freeze_coord_dual_snapshot = self.dual_snapshot;
        // Half of the configured watchdog timeout, in nanoseconds.
        // Used by the dual-snapshot scanner to compare against each
        // task's runnable-age in jiffies (converted via the guest's
        // CONFIG_HZ at scan time). The fallback default
        // (`Duration::from_secs(4)` per the builder default) means
        // a coord that never received an explicit
        // `watchdog_timeout()` call still has a coherent half-way
        // mark — 2 s of stall before the early snapshot fires.
        let freeze_coord_watchdog_half = self
            .watchdog_timeout
            .unwrap_or(Duration::from_secs(4))
            .checked_div(2)
            .unwrap_or(Duration::ZERO);
        // Guest CONFIG_HZ resolved from the kernel image. Used to
        // convert the watchdog_half Duration into a jiffies-domain
        // threshold the runnable_at scan can compare against.
        let freeze_coord_hz = monitor::guest_kernel_hz(Some(&self.kernel));
        // GuestMem for the coordinator's .bss-poll path. Built from
        // the same guest_mem the monitor uses; lifetime tied to the
        // VM run.
        let freeze_coord_mem = match vm.numa_layout.as_ref() {
            Some(layout) => Some(monitor::reader::GuestMem::from_layout(
                layout,
                &vm.guest_mem,
            )),
            None => {
                use vm_memory::GuestMemoryRegion;
                if let Ok(host_base) = vm.guest_mem.get_host_address(GuestAddress(DRAM_BASE))
                    && let Some(r) = vm.guest_mem.iter().next()
                {
                    let mem_size = r.len();
                    // SAFETY: host_base came from GuestMemoryMmap's
                    // get_host_address; mapping outlives this GuestMem
                    // (vm.guest_mem outlives the coordinator thread —
                    // collect_results joins the coordinator before vm
                    // is dropped).
                    Some(unsafe { monitor::reader::GuestMem::new(host_base, mem_size) })
                } else {
                    None
                }
            }
        };
        // Extract a fresh ImmediateExitHandle for the freeze coord —
        // the watchdog grabs another one below for its own kick path.
        // Both views address the same kvm_run.immediate_exit byte
        // (single-byte volatile writes), distinct from the BSP's own
        // owned handle inside its run loop.
        let freeze_coord_bsp_ie_handle = if has_immediate_exit {
            Some(ImmediateExitHandle::from_vcpu(&mut bsp))
        } else {
            None
        };
        let freeze_coord_bsp_tid = unsafe { libc::pthread_self() };
        // Snapshot the AP-side freeze handles. `parked` flags and
        // register-snapshot slots come from `ap_freeze_handles` —
        // populated alongside the threads inside `spawn_ap_threads`,
        // kept out of `VcpuThread` so that struct stays minimal
        // (only `handle` + `exited` + `immediate_exit` are needed
        // for teardown). The freeze coordinator owns these Vecs
        // for the rest of run_vm. `pthread_t`s and immediate-exit
        // handles still come from `ap_threads` because those are
        // teardown-relevant too.
        let ApFreezeHandles {
            parked: freeze_coord_ap_parked,
            regs: freeze_coord_ap_regs,
        } = ap_freeze_handles;
        let freeze_coord_ap_pthreads: Vec<libc::pthread_t> = ap_threads
            .iter()
            .map(|vt| vt.handle.as_pthread_t() as libc::pthread_t)
            .collect();
        // ImmediateExitHandle is Copy+Send+Sync, so the coordinator
        // captures a Vec of them by move. The kvm_run mmap is shared
        // between the spawned vCPU thread (which owns its handle
        // inside VcpuThread) and the coordinator's copy — single-byte
        // volatile writes through `set` from either side address the
        // same MAP_SHARED page.
        let freeze_coord_ap_ies: Vec<Option<ImmediateExitHandle>> =
            ap_threads.iter().map(|vt| vt.immediate_exit).collect();
        // Total vCPU count (BSP + APs). Forwarded into dump_state so
        // PERCPU_ARRAY map rendering knows how many per-CPU slots to
        // read — `bpf_array.pptrs[k]` is a `void __percpu *` whose
        // per-CPU expansion needs `__per_cpu_offset[0..nr_cpu_ids]`.
        let freeze_coord_num_cpus = (ap_threads.len() + 1) as u32;
        let freeze_coord_handle = std::thread::Builder::new()
            .name("vmm-freeze-coord".into())
            .spawn(move || {
                // Per-CPU runnable_at scanner context. Holds every
                // input the scanner needs, all resolved once and
                // cached for the rest of the run. Only built when
                // dual_snapshot is enabled AND every prerequisite
                // resolves (vmlinux ELF parses, BTF resolves the
                // four runnable_scan offsets, jiffies_64 symbol is
                // present, the GuestKernel handshake completes so
                // we have a cr3_pa / page_offset / l5 view).
                struct RunnableScanCtx {
                    rq_pas: Vec<u64>,
                    rq_kvas: Vec<u64>,
                    offsets: crate::monitor::btf_offsets::RunnableScanOffsets,
                    rq_scx_offset: usize,
                    jiffies_64_pa: u64,
                    cr3_pa: u64,
                    page_offset: u64,
                    l5: bool,
                }
                // Lazy-construct BpfMapAccessorOwned. The constructor
                // parses vmlinux ELF (goblin) and BTF (~MB-scale
                // work) and reads guest-memory bootstrap symbols
                // (`page_offset_base`, `pgtable_l5_enabled`,
                // `init_top_pgt`); the latter aren't readable until
                // the guest kernel has populated them, so a
                // construction attempt at coord-start can fail with
                // a still-booting guest. The fix is the same lazy-
                // discovery pattern that `cached_bss_pa` uses below:
                // try each iteration until success, then cache —
                // gated on `owned_accessor.is_none()` so the heavy
                // parse runs at most once per coordinator (only the
                // failed attempts re-pay it, and only until the
                // first success). A single one-shot construct at
                // coord-start would have left the accessor None
                // permanently if the guest hadn't booted yet,
                // disabling freeze detection AND the dump for the
                // entire run.
                let mut owned_accessor: Option<crate::monitor::bpf_map::GuestMemMapAccessorOwned> =
                    None;
                // Lazy-construct GuestMemProgAccessorOwned for the
                // failure-dump prog_runtime_stats capture. Same
                // boot-race rationale as `owned_accessor`: the
                // GuestKernel handshake depends on guest-memory
                // bootstrap symbols populated during boot, so an
                // attempt at coord-start can fail. Retry each
                // iteration until success; gated on
                // `owned_prog_accessor.is_none()` so the BTF parse
                // pays once. Constructed independently from
                // `owned_accessor` because the prog-side lookups
                // (`prog_idr`) and offsets (`BpfProgOffsets`) are
                // disjoint from the map side, so a kernel that
                // exposes maps but lacks `prog_idr` (theoretical)
                // still gets map rendering.
                let mut owned_prog_accessor:
                    Option<crate::monitor::bpf_prog::GuestMemProgAccessorOwned> = None;
                // Per-CPU offset array used by `runtime_stats` to
                // locate each CPU's `bpf_prog_stats` slot. Resolved
                // once after `owned_prog_accessor` lands by reading
                // `__per_cpu_offset` from guest memory; cached so
                // every dump iteration reuses it. None until either
                // the prog accessor isn't ready yet or the
                // `__per_cpu_offset` symbol couldn't be located in
                // the kernel's symbol table.
                let mut prog_per_cpu_offsets: Option<Vec<u64>> = None;
                // BTF + arena offsets resolved once at coordinator
                // start. Used by `dump_state` after the rendezvous
                // succeeds to render every BPF map's contents. None
                // values disable rendering for the relevant code path
                // (no BTF → no BTF-driven rendering at all; no arena
                // offsets → arena maps fall back to an explanatory
                // error string in the report).
                //
                // Arena offsets derive from the same parsed `Btf`
                // handle (`from_btf`, not `from_vmlinux`) so the
                // ELF-to-BTF parse runs exactly once per coordinator
                // — a second `from_vmlinux` would re-read and
                // re-parse the same file.
                let dump_btf = freeze_coord_vmlinux
                    .as_ref()
                    .and_then(|v| crate::monitor::btf_offsets::load_btf_from_path(v).ok());
                let dump_arena_offsets = dump_btf
                    .as_ref()
                    .and_then(|btf| crate::monitor::arena::BpfArenaOffsets::from_btf(btf).ok());
                // Per-CPU CPU-time / softirq / IRQ / iowait offsets
                // and the matching `.data..percpu` symbol KVAs.
                // Resolved once at coordinator start, mirroring
                // `dump_arena_offsets`. Both Option-typed: a stripped
                // vmlinux without any of `kernel_cpustat` / `kstat` /
                // `tick_cpu_sched` symbols still resolves the BTF
                // offsets fine, but the dump path checks both sides
                // before constructing a `CpuTimeCapture` so the
                // capture site only fires when the data is actually
                // readable.
                let dump_cpu_time_offsets = dump_btf
                    .as_ref()
                    .and_then(|btf| crate::monitor::btf_offsets::CpuTimeOffsets::from_btf(btf).ok());
                let dump_cpu_time_symbols = freeze_coord_vmlinux
                    .as_ref()
                    .and_then(|v| crate::monitor::symbols::KernelSymbols::from_vmlinux(v).ok());
                // Lazy-discovered cached PA of `ktstr_err_exit_detected`
                // within the probe BPF program's .bss map. None until
                // the probe loads into map_idr (rust_init phase 2b);
                // discovery retries each iteration until success.
                let mut cached_bss_pa: Option<u64> = None;
                // Cache the BTF-resolved offset of the field within
                // the .bss section. The Datasec walk parses the
                // probe's BTF (a few-KB blob copy + parse) every
                // call — caching keeps that work to once-per-coord-
                // lifetime instead of once-per-discovery-iteration.
                // Resolution can fail two ways:
                //   - guest still booting → retry (offset stays None)
                //   - BTF parse / Datasec walk broken → fall back
                //     to offset 0 once, log a warn, and stop retrying
                //     (warn_logged is the latch).
                let mut cached_bss_offset: Option<u32> = None;
                let mut bss_offset_warn_logged = false;
                // Dual-snapshot state machine. Only used when
                // `freeze_coord_dual_snapshot` is true; the
                // single-snapshot path drives the same transitions
                // but skips the early branch entirely.
                //
                // - Idle      → no dump captured yet.
                // - TookEarly → early snapshot captured (dual-snapshot
                //               mode only); waiting for the err_exit
                //               latch to fire.
                // - Done      → late snapshot captured and emission
                //               complete; coord just idles until
                //               kill / bsp_done.
                #[derive(Debug, Clone, Copy, PartialEq, Eq)]
                enum FreezeState {
                    Idle,
                    TookEarly,
                    Done,
                }
                let mut freeze_state = FreezeState::Idle;
                // Cached early snapshot from a midway-trigger freeze.
                // Held until the late freeze fires; then both early
                // and late are wrapped into a DualFailureDumpReport
                // and emitted as one file. Discarded silently when
                // the run ends without a late freeze (the run passed
                // and the early sample is not useful as a standalone
                // artifact).
                let mut early_snapshot: Option<crate::monitor::dump::FailureDumpReport> = None;
                // Per-snapshot scanner metadata, captured at the
                // early-trigger site and threaded into the
                // DualFailureDumpReport wrapper alongside the
                // snapshots themselves. Both fields stay 0 when no
                // early snapshot fires, mirroring the report's
                // `skip_serializing_if = is_zero` behaviour.
                let mut early_max_age_jiffies: u64 = 0;
                let mut early_threshold_jiffies: u64 = 0;
                // Lazy-resolved runnable_at scanner context. Built
                // from the same vmlinux ELF the .bss-poll path uses.
                // None disables the early-trigger path on this
                // iteration but does not block the late path. Becomes
                // Some once every prerequisite resolves; cached for
                // the rest of the run.
                let mut scan_ctx: Option<RunnableScanCtx> = None;
                // Retry counter and one-shot warn latch for the
                // scan_ctx resolve. The resolve runs once per 100 ms
                // poll iteration until it succeeds; without a
                // diagnostic an operator who built ktstr against a
                // kernel lacking sched_ext_entity (or stripped of
                // jiffies_64) gets a silent dual-snapshot disable.
                // Wait `SCAN_CTX_WARN_AFTER_ITERS` iterations
                // (~3 s at 100 ms cadence) before warning so legit
                // boot-time delays (owned_accessor not yet ready,
                // GuestKernel handshake mid-flight) don't trigger
                // false alarms. The latch ensures the warn fires at
                // most once per VM run.
                let mut scan_ctx_retries: u32 = 0;
                let mut scan_ctx_warned: bool = false;
                const SCAN_CTX_WARN_AFTER_ITERS: u32 = 30;
                // Half of the configured watchdog timeout, expressed
                // in guest jiffies. Computed once from
                // freeze_coord_watchdog_half + freeze_coord_hz so each
                // poll's comparison is a cheap u64 compare against
                // the scan's max age.
                //
                // ms-precision conversion lives in [`duration_to_jiffies`];
                // see its doc for why the seconds-based form is wrong.
                let half_threshold_jiffies =
                    duration_to_jiffies(freeze_coord_watchdog_half, freeze_coord_hz);
                while !freeze_coord_kill.load(Ordering::Acquire) {
                    if freeze_coord_bsp_done.load(Ordering::Acquire) {
                        return;
                    }
                    // Lazy retry: the accessor's GuestKernel walk
                    // depends on guest-memory bootstrap symbols
                    // populated by the guest kernel during boot, so
                    // an attempt at coord-start can fail. Retry each
                    // iteration until success; gated on
                    // `owned_accessor.is_none()` so the heavy
                    // ELF/BTF parse runs at most once after the
                    // first successful attempt.
                    if owned_accessor.is_none()
                        && let (Some(mem), Some(vmlinux)) =
                            (freeze_coord_mem.as_ref(), freeze_coord_vmlinux.as_ref())
                    {
                        owned_accessor =
                            crate::monitor::bpf_map::GuestMemMapAccessorOwned::new(mem, vmlinux).ok();
                    }
                    // Lazy retry for the prog-side accessor. Same
                    // pattern as `owned_accessor` above: the
                    // GuestMemProgAccessorOwned needs the GuestKernel
                    // handshake (boot-time symbols) AND the BTF
                    // parse to succeed, so coord-start may be too
                    // early. Retry each iteration until success.
                    if owned_prog_accessor.is_none()
                        && let (Some(mem), Some(vmlinux)) =
                            (freeze_coord_mem.as_ref(), freeze_coord_vmlinux.as_ref())
                    {
                        owned_prog_accessor =
                            crate::monitor::bpf_prog::GuestMemProgAccessorOwned::new(mem, vmlinux).ok();
                    }
                    // Resolve the per-CPU offset array once the prog
                    // accessor lands. Reads `__per_cpu_offset` from
                    // the kernel's static symbol table and uses it
                    // to read each CPU's offset slot. Cached for the
                    // rest of the run — the array is fixed at boot
                    // (per-CPU areas are allocated at kernel init,
                    // see `setup_per_cpu_areas`) and the freeze
                    // coordinator never sees a CPU hot-plug event,
                    // so a single read is enough.
                    if prog_per_cpu_offsets.is_none()
                        && let (Some(mem), Some(vmlinux)) =
                            (freeze_coord_mem.as_ref(), freeze_coord_vmlinux.as_ref())
                        && let Ok(syms) =
                            crate::monitor::symbols::KernelSymbols::from_vmlinux(vmlinux)
                    {
                        let pco_pa = crate::monitor::symbols::text_kva_to_pa(
                            syms.per_cpu_offset,
                        );
                        let offsets = crate::monitor::symbols::read_per_cpu_offsets(
                            mem,
                            pco_pa,
                            freeze_coord_num_cpus,
                        );
                        // Defer caching until the read returns at
                        // least one non-zero offset — a guest still
                        // populating per-CPU areas yields all-zero
                        // reads, and caching that would alias every
                        // CPU's stats to CPU 0. Once any CPU's
                        // offset is non-zero the array is stable
                        // for the run.
                        if offsets.iter().any(|&o| o != 0) {
                            prog_per_cpu_offsets = Some(offsets);
                        }
                    }
                    // Try to discover the probe .bss map and cache the
                    // PA of ktstr_err_exit_detected. Match by suffix
                    // "probe_bp.bss" rather than ".bss" so we don't
                    // race a scheduler-under-test's own .bss map when
                    // multiple BPF programs are loaded — libbpf names
                    // BPF program .bss maps as "<obj_short_name>.bss",
                    // and the probe object's name is "probe_bp" (per
                    // build.rs probe-skel generation, see the
                    // generated probe_skel.rs match arm
                    // `"probe_bp.bss" => bss = Some(map)`).
                    //
                    // Resolve the byte offset of
                    // `ktstr_err_exit_detected` within the probe's
                    // `.bss` section via BTF Datasec rather than
                    // hardcoding 0. The probe BPF program ships its
                    // own split BTF; its Datasec for `.bss` carries
                    // a VarSecinfo per writable global with the
                    // exact byte offset the BPF JIT places it at. A
                    // hardcoded 0 worked while the field was the
                    // sole writable global in `probe.bpf.c`, but a
                    // future addition that reorders globals (or that
                    // adds another writable global before this one)
                    // would silently shift the offset and break the
                    // freeze trigger. The BTF lookup keeps the
                    // detection robust across declaration changes.
                    //
                    // Falls back to offset 0 when the program BTF
                    // can't be loaded yet (guest still booting) or
                    // the Datasec walk fails — same recovery
                    // behaviour as the previous always-zero path.
                    if cached_bss_pa.is_none()
                        && let Some(ref owned) = owned_accessor
                        && let Some(ref mem) = freeze_coord_mem
                    {
                        let accessor = owned.as_accessor();
                        // Single map_idr walk per discovery attempt.
                        // value_kva is Some for ARRAY maps (the .bss
                        // map is a single-key ARRAY whose flex array
                        // holds the section's bytes); translate it
                        // (vmalloc-backed) to PA via the existing
                        // GuestMem page-walk and cache the result so
                        // subsequent polls are pure DRAM reads.
                        if let Some(map) = accessor.find_map("probe_bp.bss")
                            && let Some(value_kva) = map.value_kva
                        {
                            let cr3_pa = owned.guest_kernel().cr3_pa();
                            let page_offset = owned.guest_kernel().page_offset();
                            let l5 = owned.guest_kernel().l5();
                            // BTF-driven offset: load the probe's
                            // program BTF and walk its `.bss`
                            // Datasec for the named global. The
                            // result is cached in `cached_bss_offset`
                            // — only the first successful resolution
                            // pays the BTF parse cost. A None here
                            // before the cache is populated means
                            // either the program BTF isn't loaded
                            // yet (still-booting guest, retry
                            // silently) or the BTF walk is broken
                            // (warn once, fall back to offset 0).
                            if cached_bss_offset.is_none()
                                && map.btf_kva != 0
                                && let Some(ref base) = dump_btf
                            {
                                match load_probe_bss_offset(
                                    owned.guest_kernel(),
                                    map.btf_kva,
                                    base,
                                    accessor.offsets(),
                                ) {
                                    Some(off) => {
                                        cached_bss_offset = Some(off);
                                    }
                                    None => {
                                        // map.btf_kva is non-zero
                                        // and dump_btf is loaded,
                                        // so the probe IS loaded
                                        // — a None now means the
                                        // BTF parse / Datasec
                                        // walk failed. Fall back
                                        // to 0 and stop retrying.
                                        if !bss_offset_warn_logged {
                                            tracing::warn!(
                                                "freeze-coord: BTF Datasec resolution \
                                                     failed, falling back to offset 0"
                                            );
                                            bss_offset_warn_logged = true;
                                        }
                                        cached_bss_offset = Some(0);
                                    }
                                }
                                // else: probe not loaded yet
                                // (map.btf_kva == 0 or dump_btf
                                // missing). Leave cached_bss_offset
                                // None so the next iteration retries
                                // without the warn fallback.
                            }
                            let bss_field_offset = cached_bss_offset.unwrap_or(0);
                            if let Some(translated) = crate::monitor::idr::translate_any_kva(
                                mem,
                                cr3_pa,
                                page_offset,
                                value_kva,
                                l5,
                            ) {
                                cached_bss_pa =
                                    Some(translated.wrapping_add(bss_field_offset as u64));
                            }
                        }
                    }
                    // Lazy-resolve the per-CPU runnable_at scan
                    // context once `owned_accessor` lands and the
                    // bootstrap symbols are readable. Skipped entirely
                    // when dual_snapshot is off; failed prerequisites
                    // (missing jiffies_64 symbol, BTF without
                    // sched_ext_entity, etc.) leave `scan_ctx` None
                    // and the early-trigger path stays dormant for
                    // the rest of the run — the late path still works.
                    //
                    // Each failed prerequisite emits a per-iteration
                    // `tracing::debug!` line under the
                    // `RUST_LOG=ktstr=debug` filter — the
                    // DualFailureDumpReport's absent-early Display
                    // message points operators here. Per-iteration
                    // (not single-shot) is the right cadence for
                    // debug output: an operator who asked for verbose
                    // logging wants to see the full retry pattern,
                    // not just one snapshot. The aggregate "something
                    // is wrong" signal stays at the warn level (see
                    // `scan_ctx_warned` below) so default-visible
                    // output still surfaces a single line per run.
                    if freeze_coord_dual_snapshot && scan_ctx.is_none() {
                        let try_resolve = || -> Option<RunnableScanCtx> {
                            let owned = match owned_accessor.as_ref() {
                                Some(o) => o,
                                None => {
                                    tracing::debug!(
                                        "freeze-coord: scan resolve: \
                                         owned_accessor not ready (guest still booting)"
                                    );
                                    return None;
                                }
                            };
                            let vmlinux = match freeze_coord_vmlinux.as_ref() {
                                Some(v) => v,
                                None => {
                                    tracing::debug!(
                                        "freeze-coord: scan resolve: \
                                         vmlinux path absent (no kernel image to parse)"
                                    );
                                    return None;
                                }
                            };
                            let btf = match dump_btf.as_ref() {
                                Some(b) => b,
                                None => {
                                    tracing::debug!(
                                        "freeze-coord: scan resolve: \
                                         dump_btf not loaded (vmlinux BTF parse failed)"
                                    );
                                    return None;
                                }
                            };
                            let syms = match crate::monitor::symbols::KernelSymbols::from_vmlinux(
                                vmlinux,
                            ) {
                                Ok(s) => s,
                                Err(e) => {
                                    tracing::debug!(
                                        "freeze-coord: scan resolve: \
                                         KernelSymbols::from_vmlinux failed: {e}"
                                    );
                                    return None;
                                }
                            };
                            let jiffies_64_kva = match syms.jiffies_64 {
                                Some(k) => k,
                                None => {
                                    tracing::debug!(
                                        "freeze-coord: scan resolve: \
                                         jiffies_64 symbol absent from vmlinux"
                                    );
                                    return None;
                                }
                            };
                            let scan_offsets =
                                match crate::monitor::btf_offsets::RunnableScanOffsets::from_btf(
                                    btf,
                                ) {
                                    Ok(o) => o,
                                    Err(e) => {
                                        tracing::debug!(
                                            "freeze-coord: scan resolve: \
                                             RunnableScanOffsets::from_btf failed: {e} \
                                             (BTF likely lacks sched_ext_entity)"
                                        );
                                        return None;
                                    }
                                };
                            let rq_offsets =
                                match crate::monitor::btf_offsets::KernelOffsets::from_vmlinux(
                                    vmlinux,
                                ) {
                                    Ok(o) => o,
                                    Err(e) => {
                                        tracing::debug!(
                                            "freeze-coord: scan resolve: \
                                             KernelOffsets::from_vmlinux failed: {e}"
                                        );
                                        return None;
                                    }
                                };
                            let mem = match freeze_coord_mem.as_ref() {
                                Some(m) => m,
                                None => {
                                    tracing::debug!(
                                        "freeze-coord: scan resolve: \
                                         GuestMem not ready"
                                    );
                                    return None;
                                }
                            };
                            let kernel = owned.guest_kernel();
                            let cr3_pa = kernel.cr3_pa();
                            let page_offset = kernel.page_offset();
                            let l5 = kernel.l5();
                            // Translate jiffies_64's KVA to a PA.
                            // Lives in the kernel text/data mapping
                            // (text_kva_to_pa) — same as scx_root
                            // et al.
                            let jiffies_64_pa =
                                crate::monitor::symbols::text_kva_to_pa(jiffies_64_kva);
                            // Per-CPU rq PAs come from the existing
                            // `compute_rq_pas` helper. It needs the
                            // per_cpu_offsets array, which we read
                            // from guest memory via the symbol's KVA.
                            let per_cpu_offset_pa =
                                crate::monitor::symbols::text_kva_to_pa(syms.per_cpu_offset);
                            let per_cpu_offsets =
                                crate::monitor::symbols::read_per_cpu_offsets(
                                    mem,
                                    per_cpu_offset_pa,
                                    freeze_coord_num_cpus,
                                );
                            // runqueues is a percpu symbol — its
                            // st_value is a section-relative offset,
                            // not a KVA. compute_rq_pas adds
                            // per_cpu_offset[cpu] before translating.
                            let rq_pas = crate::monitor::symbols::compute_rq_pas(
                                syms.runqueues,
                                &per_cpu_offsets,
                                page_offset,
                            );
                            // Recover each per-CPU rq KVA so the
                            // runnable_list head's KVA can serve as
                            // the loop terminator. Mirrors the kva
                            // used to build the PA: pa = kva - po,
                            // so kva = pa + po.
                            let rq_kvas: Vec<u64> = rq_pas
                                .iter()
                                .map(|&pa| pa.wrapping_add(page_offset))
                                .collect();
                            Some(RunnableScanCtx {
                                rq_pas,
                                rq_kvas,
                                offsets: scan_offsets,
                                rq_scx_offset: rq_offsets.rq_scx,
                                jiffies_64_pa,
                                cr3_pa,
                                page_offset,
                                l5,
                            })
                        };
                        if let Some(ctx) = try_resolve() {
                            scan_ctx = Some(ctx);
                        }
                    }
                    // Single-shot warn when the resolve has been
                    // failing long enough that "still booting" is no
                    // longer a plausible explanation. Without this
                    // an operator running ktstr against a kernel that
                    // lacks `sched_ext_entity` BTF (sched_ext disabled)
                    // or `jiffies_64` (stripped vmlinux) gets the
                    // dual-snapshot path silently disabled; the late
                    // dump still works, but the early snapshot would
                    // never fire and the missing wrapper could be
                    // mistaken for "stall fired before half-way
                    // threshold". Counting iterations under the
                    // dual-snapshot gate ensures the message only
                    // surfaces in runs where the path was requested.
                    if freeze_coord_dual_snapshot && scan_ctx.is_none() {
                        scan_ctx_retries += 1;
                        if !scan_ctx_warned && scan_ctx_retries >= SCAN_CTX_WARN_AFTER_ITERS {
                            tracing::warn!(
                                "freeze-coord: runnable_at scan prerequisites unavailable \
                                 (most commonly: guest still booting; or BTF lacks \
                                 sched_ext_entity, jiffies_64 symbol missing) — \
                                 early-trigger path delayed — will continue retrying"
                            );
                            scan_ctx_warned = true;
                        }
                    }
                    // Poll the cached PA for the err_exit latch flip.
                    let err_triggered =
                        if let (Some(pa), Some(mem)) = (cached_bss_pa, freeze_coord_mem.as_ref()) {
                            mem.read_u32(pa, 0) != 0
                        } else {
                            false
                        };
                    // Once the late snapshot has been emitted, the
                    // coordinator's only remaining job is to keep
                    // the freeze=false invariant clear and wait for
                    // teardown. Idle in 100 ms steps (matching the
                    // pre-trigger cadence) so kill / bsp_done is
                    // observed promptly.
                    if freeze_state == FreezeState::Done {
                        std::thread::sleep(Duration::from_millis(100));
                        continue;
                    }
                    // Closures capture by reference. Building the
                    // full freeze-rendezvous-dump cycle once and
                    // calling it for either the early or late
                    // snapshot keeps the drain-dance contract
                    // (immediate_exit pass 1 → release fence →
                    // signal pass 2 → N-of-N rendezvous) defined in
                    // exactly one place. Returns
                    // `Some(FailureDumpReport)` when the rendezvous
                    // succeeded; None on timeout (the surrounding
                    // logic still thaws). The thaw is the caller's
                    // responsibility so the same closure works for
                    // a state-resetting late freeze (thaw to allow
                    // teardown to run) and a transient early freeze
                    // (thaw to let the test continue).
                    let freeze_and_capture =
                        || -> Option<crate::monitor::dump::FailureDumpReport> {
                            tracing::info!(
                                "freeze-coord: freezing vCPUs for snapshot"
                            );
                            freeze_coord_freeze.store(true, Ordering::Release);
                            // Pass 0: signal every device worker to
                            // pause. virtio-blk has an independent
                            // worker thread that must be parked
                            // before we read guest memory — otherwise
                            // it can race-mutate the avail/used rings
                            // and the backing file mid-dump,
                            // producing a torn view of in-flight
                            // requests. Other devices (virtio-net,
                            // virtio-console) run on the vCPU thread
                            // and freeze automatically at the vCPU
                            // rendezvous below.
                            //
                            // The worker may be in `pread`/`pwrite`
                            // when this lands; the eventfd write
                            // returns immediately (counter mode +
                            // EFD_NONBLOCK) and the syscall completes
                            // before the worker reaches the next
                            // `epoll_wait` and observes PAUSE_TOKEN.
                            // The rendezvous loop below polls each
                            // worker's `paused` flag with the same
                            // FREEZE_RENDEZVOUS_TIMEOUT budget that
                            // bounds the vCPU wait — workers ack
                            // within ~1 ms in healthy state and the
                            // 30 s ceiling absorbs sick-system stalls.
                            if let Some(ref blk) = freeze_coord_virtio_blk {
                                blk.lock().pause();
                            }
                            // Pass 1: set every immediate_exit=1.
                            // Each ImmediateExitHandle::set is a
                            // single-byte write_volatile into the
                            // corresponding kvm_run mmap (MAP_SHARED,
                            // lifetime tied to the running VcpuFd
                            // that owns it).
                            for ie in freeze_coord_ap_ies.iter().flatten() {
                                ie.set(1);
                            }
                            if let Some(ref ie) = freeze_coord_bsp_ie_handle {
                                ie.set(1);
                            }
                            // Release fence between pass 1 and pass 2
                            // so all immediate_exit writes are
                            // observable before any vCPU thread
                            // receives the kick signal — without
                            // this, a thread could process its signal,
                            // enter KVM_RUN, and miss the
                            // immediate_exit byte that is supposed to
                            // short-circuit guest entry.
                            std::sync::atomic::fence(Ordering::Release);
                            // Pass 2: signal every vCPU.
                            for &tid in &freeze_coord_ap_pthreads {
                                unsafe {
                                    libc::pthread_kill(tid, vcpu_signal());
                                }
                            }
                            unsafe {
                                libc::pthread_kill(freeze_coord_bsp_tid, vcpu_signal());
                            }
                            // Wait for N-of-N parked acks. The
                            // Acquire load synchronizes-with the
                            // vCPU's Release store after the drain
                            // dance — this rendezvous IS the memory
                            // barrier that makes the future host-side
                            // guest-memory reads correct.
                            let deadline = Instant::now() + FREEZE_RENDEZVOUS_TIMEOUT;
                            let mut all_parked = false;
                            loop {
                                if freeze_coord_kill.load(Ordering::Acquire)
                                    || freeze_coord_bsp_done.load(Ordering::Acquire)
                                {
                                    break;
                                }
                                let aps_parked = freeze_coord_ap_parked
                                    .iter()
                                    .all(|p| p.load(Ordering::Acquire));
                                let bsp_p = freeze_coord_bsp_parked.load(Ordering::Acquire);
                                // Worker pause ack. None-or-paused:
                                // when no virtio-blk is attached the
                                // condition is vacuously true. The
                                // Acquire load synchronizes-with the
                                // worker's `paused.store(true,
                                // Release)` so the host's subsequent
                                // guest-memory reads happen-after
                                // every queue mutation the worker
                                // performed pre-pause.
                                let blk_parked = freeze_coord_virtio_blk
                                    .as_ref()
                                    .is_none_or(|d| d.lock().is_paused());
                                if aps_parked && bsp_p && blk_parked {
                                    all_parked = true;
                                    break;
                                }
                                if Instant::now() > deadline {
                                    let ap_states: Vec<bool> = freeze_coord_ap_parked
                                        .iter()
                                        .map(|p| p.load(Ordering::Acquire))
                                        .collect();
                                    tracing::error!(
                                        ?ap_states,
                                        bsp_parked = bsp_p,
                                        blk_parked,
                                        "freeze-coord: timed out waiting for vCPUs / worker to park. \
                                         If blk_parked=false, the worker is most likely stuck in a \
                                         slow pread/pwrite against the backing file — verify the \
                                         vCPU-blocking-budget assumption (tmpfs / warm page cache \
                                         per CLAUDE.md 'vCPU thread blocking budget' invariant). \
                                         The worker observes PAUSE_TOKEN only between blocking \
                                         syscalls, so a long pread/pwrite delays the park-ack \
                                         until the syscall returns."
                                    );
                                    break;
                                }
                                std::thread::sleep(Duration::from_micros(100));
                            }
                            // Collect per-vCPU register snapshots.
                            // Reads happens-after the rendezvous
                            // Acquire on each vCPU's `parked` flag,
                            // which synchronizes-with the vCPU
                            // thread's Release store after its
                            // capture_vcpu_regs / regs_slot write —
                            // so these Mutex reads see the captured
                            // values even on weakly-ordered
                            // architectures. Index 0 = BSP, 1..N =
                            // APs.
                            let collect_vcpu_regs = ||
                                -> Vec<Option<exit_dispatch::VcpuRegSnapshot>> {
                                let mut regs:
                                    Vec<Option<exit_dispatch::VcpuRegSnapshot>> =
                                    Vec::with_capacity(1 + freeze_coord_ap_regs.len());
                                regs.push(
                                    *freeze_coord_bsp_regs
                                        .lock()
                                        .unwrap_or_else(|e| e.into_inner()),
                                );
                                for ap in &freeze_coord_ap_regs {
                                    regs.push(
                                        *ap.lock().unwrap_or_else(|e| e.into_inner()),
                                    );
                                }
                                regs
                            };
                            if !all_parked {
                                // Rendezvous timed out — at least
                                // one vCPU never set its parked
                                // flag, so we cannot safely read
                                // guest memory.
                                tracing::debug!(
                                    "freeze-coord: dump skipped: rendezvous timed out"
                                );
                                return None;
                            }
                            if let Some(ref owned) = owned_accessor
                                && let Some(ref btf) = dump_btf
                            {
                                // Build the prog-runtime capture
                                // when both prerequisites are ready.
                                // Each is independent of the other
                                // (the accessor needs prog_idr +
                                // BTF; the offsets need __per_cpu_offset),
                                // so a partial setup yields no
                                // capture rather than a half-correct
                                // one — `dump_state` then writes an
                                // empty `prog_runtime_stats` vec
                                // alongside the full map render.
                                let prog_acc_borrow =
                                    owned_prog_accessor.as_ref().map(|o| o.as_accessor());
                                let prog_capture = match (
                                    prog_acc_borrow.as_ref(),
                                    prog_per_cpu_offsets.as_deref(),
                                ) {
                                    (Some(acc), Some(offsets)) => {
                                        Some(crate::monitor::dump::ProgRuntimeCapture {
                                            accessor: acc,
                                            per_cpu_offsets: offsets,
                                        })
                                    }
                                    _ => None,
                                };
                                let map_accessor = owned.as_accessor();
                                // Per-CPU CPU-time / softirq / IRQ
                                // capture context. All four prereqs
                                // must be present to fire: BTF
                                // offsets (resolved at coord start),
                                // KernelSymbols carrying the
                                // `kernel_cpustat`/`kstat`
                                // per-CPU symbol KVAs (also at coord
                                // start), the per-CPU offset array
                                // (lazy-resolved alongside the prog
                                // accessor), and the freeze-coord
                                // GuestMem. Either of `kernel_cpustat`
                                // or `kstat` symbol absent makes the
                                // capture useless — both backing
                                // structs are needed for the dump's
                                // narrative (`tick_cpu_sched` is
                                // optional and feeds only the
                                // iowait_sleeptime field). The
                                // `tick_cpu_sched_kva` is forwarded
                                // to dump.rs as Option so the per-CPU
                                // walker can skip iowait_sleeptime
                                // independently per CPU.
                                let cpu_time_capture = match (
                                    freeze_coord_mem.as_ref(),
                                    dump_cpu_time_offsets.as_ref(),
                                    dump_cpu_time_symbols.as_ref(),
                                    prog_per_cpu_offsets.as_deref(),
                                ) {
                                    (Some(mem), Some(offsets), Some(syms), Some(pcpu)) => {
                                        match (syms.kernel_cpustat, syms.kstat) {
                                            (Some(kcpustat_kva), Some(kstat_kva)) => {
                                                let page_offset =
                                                    owned.guest_kernel().page_offset();
                                                Some(crate::monitor::dump::CpuTimeCapture {
                                                    mem,
                                                    offsets,
                                                    kernel_cpustat_kva: kcpustat_kva,
                                                    kstat_kva,
                                                    tick_cpu_sched_kva: syms.tick_cpu_sched,
                                                    per_cpu_offsets: pcpu,
                                                    page_offset,
                                                })
                                            }
                                            _ => None,
                                        }
                                    }
                                    _ => None,
                                };
                                let mut report = crate::monitor::dump::dump_state(
                                    crate::monitor::dump::DumpContext {
                                        accessor: &map_accessor,
                                        btf,
                                        num_cpus: freeze_coord_num_cpus,
                                        arena_offsets: dump_arena_offsets.as_ref(),
                                        prog_capture: prog_capture.as_ref(),
                                        cpu_time_capture: cpu_time_capture.as_ref(),
                                        // Per-task enrichment is library-ready
                                        // but has no walker producer until the
                                        // rq->scx walker lands. The
                                        // `task_enrichments_unavailable` field
                                        // records this state so the dump
                                        // consumer sees "no task walker
                                        // available" rather than a silent empty
                                        // vec.
                                        task_enrichment_capture: None,
                                        // Per-sample SCX_EV_* event counter
                                        // timeline. Today's freeze coordinator
                                        // does not share the monitor sampler's
                                        // accumulated samples vec — that
                                        // would require an Arc<Mutex<...>>
                                        // hand-off plumbed through
                                        // `start_monitor` / `monitor_loop`.
                                        // Leaving None preserves current
                                        // behavior (event_counter_timeline
                                        // stays empty in the failure dump
                                        // JSON); the timeline is still
                                        // recorded on `VmResult.monitor.samples`
                                        // for the post-run sidecar consumer.
                                        // A future task wiring the share
                                        // populates this with
                                        // `Some(EventCounterCapture { samples })`.
                                        event_counter_capture: None,
                                        // Per-CPU rq->scx + DSQ walker.
                                        // Library-ready; the freeze
                                        // coordinator does not currently
                                        // resolve ScxWalkerOffsets nor
                                        // build the rq_kvas/rq_pas arrays
                                        // outside the dual_snapshot
                                        // scan_ctx path. The dump emits
                                        // empty rq_scx_states /
                                        // dsq_states with
                                        // `scx_walker_unavailable: Some(
                                        // "no scx walker capture")`.
                                        scx_walker_capture: None,
                                        // Per-vCPU PMU capture is shared
                                        // with the monitor sampler via the
                                        // `freeze_coord_perf_capture` Arc;
                                        // dump_state reads it once at the
                                        // freeze instant into
                                        // `vcpu_perf_at_freeze`. None when
                                        // perf was unavailable on this host
                                        // (paranoid > 2 / no CAP_PERFMON /
                                        // hardware lacks counters).
                                        perf_capture: (*freeze_coord_perf_capture).as_ref(),
                                    },
                                );
                                report.vcpu_regs = collect_vcpu_regs();
                                Some(report)
                            } else {
                                // Partial dump: vcpu_regs only.
                                let report = crate::monitor::dump::FailureDumpReport {
                                    schema: crate::monitor::dump::SCHEMA_SINGLE.to_string(),
                                    maps: Vec::new(),
                                    vcpu_regs: collect_vcpu_regs(),
                                    sdt_allocations: Vec::new(),
                                    prog_runtime_stats: Vec::new(),
                                    prog_runtime_stats_unavailable: Some(
                                        "dump prerequisites unavailable".to_string(),
                                    ),
                                    per_cpu_time: Vec::new(),
                                    task_enrichments: Vec::new(),
                                    task_enrichments_unavailable: Some(
                                        "dump prerequisites unavailable".to_string(),
                                    ),
                                    event_counter_timeline: Vec::new(),
                                    rq_scx_states: Vec::new(),
                                    dsq_states: Vec::new(),
                                    scx_sched_state: None,
                                    scx_walker_unavailable: Some(
                                        "dump prerequisites unavailable".to_string(),
                                    ),
                                    vcpu_perf_at_freeze: Vec::new(),
                                    per_node_numa: Vec::new(),
                                    per_node_numa_unavailable: Some(
                                        "dump prerequisites unavailable".to_string(),
                                    ),
                                };
                                tracing::warn!(
                                    owned_accessor = owned_accessor.is_some(),
                                    dump_btf = dump_btf.is_some(),
                                    "freeze-coord: dump prerequisites unavailable; \
                                     emitting partial report with vcpu_regs only"
                                );
                                Some(report)
                            }
                        };
                    // Helper: emit the JSON of any Serialize value
                    // (FailureDumpReport for the single-snapshot
                    // path, DualFailureDumpReport for dual-snapshot)
                    // via tracing::error and the optional file sink.
                    // Wrapped in a closure so the file-sink contract
                    // (mkdir parent, write, log warn on failure)
                    // lives in one place across both report shapes.
                    let emit_json =
                        |json: &str, map_count: usize, vcpu_regs_count: usize| {
                            tracing::error!(
                                target: "ktstr::failure_dump",
                                map_count,
                                vcpu_regs_count,
                                "freeze-coord: failure dump\n{json}"
                            );
                            if let Some(ref path) = freeze_coord_dump_path {
                                if let Some(parent) = path.parent() {
                                    let _ = std::fs::create_dir_all(parent);
                                }
                                if let Err(e) = std::fs::write(path, json) {
                                    tracing::warn!(
                                        path = %path.display(),
                                        error = %e,
                                        "freeze-coord: failure-dump file write failed"
                                    );
                                }
                            }
                        };
                    // Early-snapshot trigger: dual_snapshot mode and
                    // we have a working scan context. Mirror the
                    // kernel's `check_rq_for_timeouts` logic — any
                    // task whose `jiffies - p->scx.runnable_at`
                    // exceeds the half-way mark trips the trigger.
                    // Half-way comes from the configured
                    // watchdog_timeout (already plumbed through
                    // `KtstrTestEntry.watchdog_timeout`), so the
                    // early snapshot lands well before the kernel
                    // would emit SCX_EXIT_ERROR_STALL — gives the
                    // operator pre-stall BPF state to diff against
                    // the late snapshot.
                    if freeze_state == FreezeState::Idle
                        && freeze_coord_dual_snapshot
                        && half_threshold_jiffies > 0
                        && let Some(ref ctx) = scan_ctx
                        && let Some(ref mem) = freeze_coord_mem
                    {
                        let jiffies = mem.read_u64(ctx.jiffies_64_pa, 0);
                        let max_age = crate::monitor::runnable_scan::max_runnable_age(
                            mem,
                            &ctx.rq_pas,
                            &ctx.rq_kvas,
                            &ctx.offsets,
                            ctx.rq_scx_offset,
                            jiffies,
                            ctx.cr3_pa,
                            ctx.page_offset,
                            ctx.l5,
                        );
                        if max_age >= half_threshold_jiffies {
                            tracing::info!(
                                max_age,
                                half_threshold_jiffies,
                                "freeze-coord: dual-snapshot early threshold tripped"
                            );
                            // Persist the trigger metric and the
                            // half-way threshold ONLY when the freeze
                            // capture succeeds. The
                            // `DualFailureDumpReport` doc says "Zero
                            // when `early` is `None`", which a
                            // consumer relies on to detect the
                            // capture-failed case from JSON alone:
                            // a `late`-only wrapper with non-zero
                            // metric values would be ambiguous (did
                            // the early capture fail, or did the
                            // trigger never fire?). Co-gating both
                            // sides on `Some(report)` keeps the
                            // invariant.
                            if let Some(report) = freeze_and_capture() {
                                early_max_age_jiffies = max_age;
                                early_threshold_jiffies = half_threshold_jiffies;
                                early_snapshot = Some(report);
                            }
                            // Thaw whether or not we got a report:
                            // the freeze flag was set unconditionally,
                            // and a stuck rendezvous already logged.
                            // Resume the worker BEFORE clearing the
                            // freeze flag: the worker's
                            // `paused.load(Acquire)` poll is the only
                            // path out of its `park_timeout(10ms)`
                            // loop, so a freeze flag clear without a
                            // resume() leaves the worker parked
                            // indefinitely. The freeze flag governs
                            // the vCPU rendezvous, which is
                            // orthogonal — vCPUs poll `freeze`, the
                            // worker polls `paused`. Order:
                            // resume()→freeze.store(false) means
                            // both wake paths land cleanly.
                            if let Some(ref blk) = freeze_coord_virtio_blk {
                                blk.lock().resume();
                            }
                            freeze_coord_freeze.store(false, Ordering::Release);
                            freeze_state = FreezeState::TookEarly;
                        }
                    }
                    // Late-snapshot trigger: err_exit_detected has
                    // flipped. The state-machine guard ensures we
                    // only fire once per VM run — TookEarly → late
                    // is allowed (capturing both halves of the
                    // dual-snapshot wrapper); Done is terminal.
                    if err_triggered
                        && (freeze_state == FreezeState::Idle
                            || freeze_state == FreezeState::TookEarly)
                    {
                        tracing::info!(
                            "freeze-coord: ktstr_err_exit_detected latched, freezing vCPUs"
                        );
                        let late_report = freeze_and_capture();
                        // Thaw before emission so a slow JSON
                        // serialise doesn't keep vCPUs parked any
                        // longer than the dump strictly needs.
                        // Resume worker first — see early-snapshot
                        // path doc for the freeze-vs-paused
                        // ordering rationale.
                        if let Some(ref blk) = freeze_coord_virtio_blk {
                            blk.lock().resume();
                        }
                        freeze_coord_freeze.store(false, Ordering::Release);
                        if let Some(late) = late_report {
                            if freeze_coord_dual_snapshot {
                                let dual = crate::monitor::dump::DualFailureDumpReport {
                                    schema: crate::monitor::dump::SCHEMA_DUAL.to_string(),
                                    early: early_snapshot.take(),
                                    late,
                                    early_max_age_jiffies,
                                    early_threshold_jiffies,
                                };
                                match serde_json::to_string_pretty(&dual) {
                                    Ok(json) => {
                                        let map_count = dual.late.maps.len();
                                        let vcpu_regs_count = dual.late.vcpu_regs.len();
                                        emit_json(&json, map_count, vcpu_regs_count);
                                    }
                                    Err(e) => {
                                        tracing::error!(
                                            error = %e,
                                            "freeze-coord: dual failure dump (JSON serialization failed)"
                                        );
                                    }
                                }
                            } else {
                                match serde_json::to_string_pretty(&late) {
                                    Ok(json) => {
                                        let map_count = late.maps.len();
                                        let vcpu_regs_count = late.vcpu_regs.len();
                                        emit_json(&json, map_count, vcpu_regs_count);
                                    }
                                    Err(e) => {
                                        tracing::error!(
                                            error = %e,
                                            map_count = late.maps.len(),
                                            vcpu_regs_count = late.vcpu_regs.len(),
                                            "freeze-coord: failure dump (JSON serialization failed)"
                                        );
                                    }
                                }
                            }
                        }
                        freeze_state = FreezeState::Done;
                        continue;
                    }
                    // No trigger this iteration. Sleep and retry.
                    std::thread::sleep(Duration::from_millis(100));
                }
            })
            .context("spawn freeze coordinator thread")?;

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
                    virtio_blk.as_ref(),
                    virtio_net.as_ref(),
                    &kill,
                    &freeze,
                    &bsp_parked,
                    &bsp_regs,
                    has_immediate_exit,
                    run_start,
                    timeout,
                )
            },
        );
        bsp_done.store(true, Ordering::Release);
        // Sample cleanup start at the earliest moment after BSP exit so
        // every host-side teardown step lands inside the window, in
        // execution order: watchdog join (immediately below), AP joins,
        // monitor join, BPF writer join, SHM drain, exit-code and
        // crash-message extraction, and verifier-stat read (the rest
        // run inside `collect_results`). `collect_results` reads
        // `Instant::now()` at the end and the difference becomes
        // `VmResult::cleanup_duration`.
        let cleanup_start = Instant::now();
        eprintln!("BSP: exited run loop, code={exit_code} timed_out={timed_out}");

        // Join the watchdog before dropping `bsp`. The watchdog holds an
        // ImmediateExitHandle pointing into bsp's kvm_run mmap. If bsp is
        // dropped first, the watchdog may write to unmapped memory.
        let _ = watchdog.join();

        // Make sure freeze is cleared before vCPU teardown so the
        // freeze coordinator sees `kill || bsp_done` and exits its
        // loop, and APs don't park-loop after we kick them. The
        // coordinator joins below.
        freeze.store(false, Ordering::Release);
        // Wake the coordinator if it's sleeping; it will observe
        // bsp_done and return.
        freeze_coord_handle.thread().unpark();

        // Capture the virtio-blk counter Arc before the device's
        // outer `Arc<PiMutex<VirtioBlk>>` falls out of scope. The
        // device's `counters()` accessor clones the inner
        // `Arc<VirtioBlkCounters>`; this transfers a reader-side
        // handle onto `VmRunState` so `collect_results` can attach
        // it to `VmResult` without holding the device alive past
        // its current ownership.
        let virtio_blk_counters = virtio_blk.as_ref().map(|d| d.lock().counters());
        let virtio_net_counters = virtio_net.as_ref().map(|d| d.lock().counters());

        Ok(VmRunState {
            exit_code,
            timed_out,
            ap_threads,
            monitor_handle,
            bpf_write_handle,
            freeze_coordinator: Some(freeze_coord_handle),
            com1,
            com2,
            kill,
            freeze,
            vm,
            cleanup_start,
            virtio_blk_counters,
            virtio_net_counters,
        })
    }

    /// Spawn AP vCPU threads. Each thread optionally pins itself to a
    /// host CPU from `pin_targets` (indexed by AP order, 0-based), OR
    /// applies a CPU mask from `no_perf_mask` when the no-perf +
    /// `--cpu-cap` path is active. The two are mutually exclusive —
    /// perf-mode produces `pin_targets` via the PinningPlan;
    /// `--cpu-cap` no-perf produces `no_perf_mask` via the LlcPlan.
    ///
    /// Returns `(threads, freeze_handles)`. The freeze handles
    /// (per-AP `parked` flags + register-snapshot slots) are the
    /// freeze coordinator's view of each AP; they live separately
    /// from `VcpuThread` so the thread struct stays minimal —
    /// `VcpuThread` carries only what teardown (kick + join) needs.
    /// Callers that don't run a freeze coordinator (e.g. interactive
    /// shell) discard `freeze_handles`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn spawn_ap_threads(
        &self,
        vcpus: Vec<kvm_ioctls::VcpuFd>,
        has_immediate_exit: bool,
        com1: &Arc<PiMutex<console::Serial>>,
        com2: &Arc<PiMutex<console::Serial>>,
        virtio_con: Option<&Arc<PiMutex<virtio_console::VirtioConsole>>>,
        virtio_blk: Option<&Arc<PiMutex<virtio_blk::VirtioBlk>>>,
        virtio_net: Option<&Arc<PiMutex<virtio_net::VirtioNet>>>,
        kill: &Arc<AtomicBool>,
        freeze: &Arc<AtomicBool>,
        pin_targets: &[Option<usize>],
        no_perf_mask: Option<&[usize]>,
        ap_tid_slots: &[Arc<AtomicI32>],
    ) -> Result<(Vec<VcpuThread>, ApFreezeHandles)> {
        // Register the process-wide panic hook that flips `kill` +
        // `exited` on a panicking vCPU thread before the
        // panic=abort-induced process teardown. Idempotent via
        // `Once`; safe to call on every VM spawn.
        vcpu_panic::install_once();
        let n = vcpus.len();
        debug_assert_eq!(ap_tid_slots.len(), n);
        let mut ap_threads: Vec<VcpuThread> = Vec::with_capacity(n);
        let mut freeze_parked: Vec<Arc<AtomicBool>> = Vec::with_capacity(n);
        let mut freeze_regs: Vec<Arc<std::sync::Mutex<Option<exit_dispatch::VcpuRegSnapshot>>>> =
            Vec::with_capacity(n);
        for (i, mut vcpu) in vcpus.into_iter().enumerate() {
            let ie_handle = if has_immediate_exit {
                Some(ImmediateExitHandle::from_vcpu(&mut vcpu))
            } else {
                None
            };
            let kill_clone = kill.clone();
            let freeze_clone = freeze.clone();
            let com1_clone = com1.clone();
            let com2_clone = com2.clone();
            let vc_clone = virtio_con.cloned();
            let vblk_clone = virtio_blk.cloned();
            let vnet_clone = virtio_net.cloned();
            let exited = Arc::new(AtomicBool::new(false));
            let exited_clone = exited.clone();
            let parked = Arc::new(AtomicBool::new(false));
            let parked_clone = parked.clone();
            let regs = Arc::new(std::sync::Mutex::new(None));
            let regs_clone = regs.clone();
            let has_immediate_exit_clone = has_immediate_exit;
            let pin_cpu = pin_targets.get(i).copied().flatten();
            let mask_for_thread: Option<Vec<usize>> = no_perf_mask.map(|m| m.to_vec());

            let rt = self.performance_mode;
            let panic_ctx = vcpu_panic::VcpuPanicCtx {
                kill: kill.clone(),
                exited: exited.clone(),
            };
            let tid_slot_clone = ap_tid_slots[i].clone();
            let handle = std::thread::Builder::new()
                .name(format!("vcpu-{}", i + 1))
                .spawn(move || {
                    register_vcpu_signal_handler();
                    // Stamp this thread's Linux TID into the per-AP
                    // slot so the monitor can open `perf_event_open`
                    // counters bound to the vCPU thread. Done
                    // BEFORE pinning / RT / KVM_RUN so the value is
                    // visible to any reader the moment the thread is
                    // schedulable. SAFETY: SYS_gettid is the
                    // standard syscall returning this thread's
                    // pid_t; no inputs.
                    let tid = unsafe { libc::syscall(libc::SYS_gettid) } as i32;
                    tid_slot_clone.store(tid, Ordering::Release);
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
                            vblk_clone.as_ref(),
                            vnet_clone.as_ref(),
                            &kill_clone,
                            &freeze_clone,
                            &parked_clone,
                            &regs_clone,
                            has_immediate_exit_clone,
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
            freeze_parked.push(parked);
            freeze_regs.push(regs);
        }
        Ok((
            ap_threads,
            ApFreezeHandles {
                parked: freeze_parked,
                regs: freeze_regs,
            },
        ))
    }

    /// Start the monitor thread if vmlinux is available.
    pub(super) fn start_monitor(
        &self,
        vm: &kvm::KtstrKvm,
        kill: &Arc<AtomicBool>,
        run_start: Instant,
        vcpu_pthreads: Vec<libc::pthread_t>,
        perf_capture: Arc<Option<monitor::perf_counters::PerfCountersCapture>>,
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
                use vm_memory::GuestMemoryRegion;
                let host_base = vm
                    .guest_mem
                    .get_host_address(GuestAddress(DRAM_BASE))
                    .context("resolve guest DRAM base host address (monitor)")?;
                // Size of the first contiguous region only.
                // host_base addresses that single mapping; using the
                // sum of all region lengths would extend past the
                // mapping into host heap when multiple regions exist.
                let mem_size = vm
                    .guest_mem
                    .iter()
                    .next()
                    .context("guest_mem must have at least one region (monitor)")?
                    .len();
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
        // ms-precision conversion lives in [`duration_to_jiffies`];
        // see its doc for why the seconds-based form is wrong.
        let watchdog_jiffies = self.watchdog_timeout.map(|d| duration_to_jiffies(d, hz));
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
                    let deadline = run_start + Duration::from_secs(30);
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
                // `cr3_pa` and `l5` are shared with `discover_struct_ops_stats`
                // and `ProgStatsCtx` so per-CPU `bpf_prog_stats` reads can
                // page-walk vmalloc-backed percpu.
                let cr3_pa =
                    monitor::symbols::text_kva_to_pa(symbols.init_top_pgt.unwrap_or(0));
                let l5 = monitor::symbols::resolve_pgtable_l5(&mem, &symbols);
                let prog_stats_ctx =
                    monitor::btf_offsets::BpfProgOffsets::from_vmlinux(&vmlinux_clone)
                        .ok()
                        .and_then(|prog_offsets| {
                            let prog_idr_kva = symbols.prog_idr?;
                            // The fused walker
                            // (`walk_struct_ops_runtime_stats`) re-walks
                            // `prog_idr` each sample, which is cheap on
                            // ktstr workloads (idr_next is in the
                            // dozens) and removes the staleness window
                            // the prior cached-discovery design opened.
                            // No upfront discovery — the walker
                            // returns an empty Vec when no struct_ops
                            // programs are loaded yet, and the monitor
                            // sample emits an empty `prog_stats` for
                            // those cycles.
                            Some(monitor::reader::ProgStatsCtx {
                                per_cpu_offsets: offsets_arr.clone(),
                                cr3_pa,
                                page_offset,
                                l5,
                                prog_idr_kva,
                                offsets: prog_offsets,
                            })
                        });

                let mon_cfg = monitor::reader::MonitorConfig {
                    event_pcpu_pas: event_pcpu_pas.as_deref(),
                    dump_trigger: dump_trigger.as_ref(),
                    watchdog_override: watchdog_override.as_ref(),
                    vcpu_timing: Some(&vcpu_timing),
                    // `perf_capture` is `Arc<Option<PerfCountersCapture>>`;
                    // outer deref through `Arc::as_ref` yields
                    // `&Option<PerfCountersCapture>`, inner
                    // `Option::as_ref` yields the
                    // `Option<&PerfCountersCapture>` MonitorConfig wants.
                    perf_capture: (*perf_capture).as_ref(),
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
                    run_start,
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
    pub(super) fn start_bpf_map_write(
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
                use vm_memory::GuestMemoryRegion;
                let host_base = vm
                    .guest_mem
                    .get_host_address(GuestAddress(DRAM_BASE))
                    .context("resolve guest DRAM base host address (bpf-map-write)")?;
                // Size of the first contiguous region only.
                // host_base addresses that single mapping; using the
                // sum of all region lengths would extend past the
                // mapping into host heap when multiple regions exist.
                let mem_size = vm
                    .guest_mem
                    .iter()
                    .next()
                    .context("guest_mem must have at least one region (bpf-map-write)")?
                    .len();
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
                use crate::monitor::bpf_map::BpfMapAccessor;
                if kill_clone.load(Ordering::Acquire) {
                    return;
                }

                // Phase 1: wait for BPF map accessor (kernel booted, page tables up).
                let phase1_deadline =
                    std::time::Instant::now() + std::time::Duration::from_secs(30);
                let owned = loop {
                    match monitor::bpf_map::GuestMemMapAccessorOwned::new(&mem, &vmlinux) {
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
    ///
    /// `freeze` and `bsp_parked` plumb the BSP into the failure-dump
    /// rendezvous: when the freeze coordinator latches `freeze=true`
    /// and kicks the BSP out of KVM_RUN, the loop performs the
    /// drain dance (set_immediate_exit(1)→run→set_immediate_exit(0)),
    /// stores `bsp_parked=true` (Release), then polls `freeze` on
    /// `park_timeout(10ms)` until the coordinator clears it. Same
    /// pattern as [`exit_dispatch::vcpu_run_loop_unified`] for APs.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_bsp_loop(
        &self,
        bsp: &mut kvm_ioctls::VcpuFd,
        com1: &Arc<PiMutex<console::Serial>>,
        com2: &Arc<PiMutex<console::Serial>>,
        virtio_con: Option<&Arc<PiMutex<virtio_console::VirtioConsole>>>,
        virtio_blk: Option<&Arc<PiMutex<virtio_blk::VirtioBlk>>>,
        virtio_net: Option<&Arc<PiMutex<virtio_net::VirtioNet>>>,
        kill: &Arc<AtomicBool>,
        freeze: &Arc<AtomicBool>,
        bsp_parked: &Arc<AtomicBool>,
        bsp_regs: &Arc<std::sync::Mutex<Option<exit_dispatch::VcpuRegSnapshot>>>,
        has_immediate_exit: bool,
        run_start: Instant,
        timeout: Duration,
    ) -> (i32, bool) {
        let mut exit_code: i32 = -1;

        loop {
            if run_start.elapsed() > timeout {
                return (exit_code, true);
            }
            if kill.load(Ordering::Acquire) {
                break;
            }
            // Honour a pending freeze before re-entering KVM_RUN.
            // Same drain-dance + park pattern as the AP run loop —
            // delegated to the shared `exit_dispatch::handle_freeze`
            // so the two paths cannot drift.
            if freeze.load(Ordering::Acquire) {
                exit_dispatch::handle_freeze(
                    bsp,
                    has_immediate_exit,
                    kill,
                    freeze,
                    bsp_parked,
                    bsp_regs,
                );
                if kill.load(Ordering::Acquire) {
                    break;
                }
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
                    match classify_exit(
                        com1,
                        com2,
                        virtio_con.map(|a| a.as_ref()),
                        virtio_blk.map(|a| a.as_ref()),
                        virtio_net.map(|a| a.as_ref()),
                        &mut exit,
                    ) {
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
    pub(super) fn collect_results(&self, start: Instant, run: VmRunState) -> Result<VmResult> {
        let mut exit_code = run.exit_code;
        let timed_out = run.timed_out;
        run.kill.store(true, Ordering::Release);
        // Clear freeze before kicking APs so any vCPU still in the
        // park loop observes `freeze=false` next iteration and exits
        // toward kill. Without this, an AP parked at the moment the
        // BSP exited would stay parked through the kill check, since
        // park_loop holds park_timeout(10ms) ignoring kill until
        // freeze clears.
        run.freeze.store(false, Ordering::Release);

        // Kick APs out of KVM_RUN. Skip APs that already exited —
        // their VcpuFd (and kvm_run mmap) may be dropped, so writing
        // to ImmediateExitHandle would hit unmapped memory. Unpark
        // each so a parked AP observes the cleared freeze flag
        // promptly without waiting for the 10ms park_timeout.
        for vt in &run.ap_threads {
            if !vt.exited.load(Ordering::Acquire) {
                vt.kick();
            }
            vt.handle.thread().unpark();
        }
        // Join the freeze coordinator BEFORE the AP join loop. The
        // coordinator owns ImmediateExitHandles into AP kvm_run mmaps;
        // joining APs first would drop the VcpuFds (and unmap the
        // pages) while the coordinator is still alive and could write
        // to a stale immediate_exit pointer (use-after-free). Setting
        // run.kill=true and run.freeze=false above guarantees the
        // coordinator's outer loop and inner park-rendezvous exit on
        // the next iteration, so this join doesn't deadlock.
        if let Some(h) = run.freeze_coordinator {
            let _ = h.join();
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
        let (shm_data, stimulus_events) = if (self.shm_size as usize) >= shm_ring::HEADER_SIZE {
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

        // Sample cleanup elapsed AFTER every blocking step that runs on
        // the post-BSP-exit critical path so the duration captures the
        // full host-side teardown cost, not a partial window. The full
        // ordered set is: watchdog join (in `run_vm`, before
        // `cleanup_start` is stored on `VmRunState`), AP joins, monitor
        // join, BPF writer join, SHM drain, exit-code and crash-message
        // extraction, verifier-stat read. Captured before constructing
        // the result so the `Instant::now()` here is the latest possible
        // read.
        let cleanup_duration = Some(run.cleanup_start.elapsed());

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
            cleanup_duration,
            virtio_blk_counters: run.virtio_blk_counters,
            virtio_net_counters: run.virtio_net_counters,
        })
    }

    /// Read BPF verifier stats from guest memory after VM exit.
    ///
    /// Enumerates struct_ops programs in the kernel's `prog_idr` and
    /// reads `bpf_prog_aux->verified_insns` for each.
    pub(super) fn collect_verifier_stats(
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
                use vm_memory::GuestMemoryRegion;
                let host_base = match vm.guest_mem.get_host_address(GuestAddress(DRAM_BASE)) {
                    Ok(ptr) => ptr,
                    Err(_) => return Vec::new(),
                };
                // Size of the first contiguous region only.
                // host_base addresses that single mapping; using the
                // sum of all region lengths would extend past the
                // mapping into host heap when multiple regions exist.
                let mem_size = match vm.guest_mem.iter().next() {
                    Some(r) => r.len(),
                    None => return Vec::new(),
                };
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
            match monitor::bpf_prog::GuestMemProgAccessor::from_guest_kernel(&kernel, &offsets) {
                Ok(a) => a,
                Err(_) => return Vec::new(),
            };
        // Trait method — `BpfProgAccessor::struct_ops_progs` is in
        // scope at the call site via the `use monitor::bpf_prog::*`
        // glob (see top of file); calling it on the concrete type
        // dispatches statically.
        use monitor::bpf_prog::BpfProgAccessor;
        accessor.struct_ops_progs()
    }
}
