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
use std::os::fd::AsRawFd;
use std::os::unix::thread::JoinHandleExt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use vm_memory::{GuestAddress, GuestMemory};
use vmm_sys_util::epoll::{ControlOperation, Epoll, EpollEvent, EventSet};
use vmm_sys_util::eventfd::{EFD_NONBLOCK, EventFd};
use vmm_sys_util::timerfd::TimerFd;

use crate::monitor;

use super::exit_dispatch::{self, ExitAction, classify_exit, vcpu_run_loop_unified};
use super::host_comms::BulkDrainResult;
use super::pi_mutex::PiMutex;
use super::result::{VmResult, VmRunState};
use super::vcpu::{
    ApFreezeHandles, BpfMapWriteParams, ImmediateExitHandle, VcpuThread, WatchpointArm,
    duration_to_jiffies, load_probe_bss_offset, open_vcpu_perf_capture, pin_current_thread,
    register_vcpu_signal_handler, self_arm_watchpoint, set_rt_priority, set_thread_cpumask,
    vcpu_signal,
};
use super::vmlinux::{cached_vmlinux_bytes, find_vmlinux};
use super::{
    KtstrVm, console, host_comms, vcpu_panic, virtio_blk, virtio_console, virtio_net, wire,
};

#[cfg(target_arch = "aarch64")]
use super::aarch64::kvm;
#[cfg(target_arch = "x86_64")]
use super::x86_64::kvm;

// `DRAM_BASE` is defined in `super` and used here for guest-memory
// host-address resolution. The const is arch-gated; the import
// follows the same gating implicitly via where it is consumed.
use super::DRAM_BASE;

mod dispatch;
mod lazy_init;
mod snapshot;
mod state;
mod watchpoint;

#[cfg(test)]
mod bss_tests;

use self::dispatch::{BulkDispatchSinks, dispatch_bulk_message};
#[allow(unused_imports)]
use self::lazy_init::{
    try_init_owned_accessor_with_hint, try_init_owned_prog_accessor_with_hint,
    try_init_prog_per_cpu_offsets,
};
#[allow(unused_imports)]
use self::snapshot::{
    VmlinuxSymbolCache, arm_user_watchpoint, decode_snapshot_request, frame_snapshot_reply,
    poll_eventfd_until_ready_or_timeout, snapshot_tagged_path,
};
use self::state::{
    BspExitReason, FREEZE_RENDEZVOUS_TIMEOUT, FreezeState, SnapshotRequest,
    compute_periodic_boundaries_ns, periodic_tag,
};
use self::watchpoint::{WatchpointPublishResult, republish_watchpoint_on_rebind};

/// Three-way result of polling the BPF probe's `.bss` latch via the
/// cached guest-physical-address path used by [`bss_read_state`].
///
/// `read_u32` returns `0` for two semantically distinct reasons: the
/// probe has not latched yet (genuine "no fire") AND the cached PA no
/// longer resolves to a live DRAM region (out-of-bounds, hole between
/// regions). Conflating the two masks a stale-cache regression as
/// "still waiting for the trigger" and lets the freeze coordinator
/// drift past a real fire when the probe has been torn down or its
/// vmalloc page recycled. Each consumer decides how to react —
/// production gates the err_triggered flag on `Triggered` only and
/// surfaces `OutOfBounds` as a diagnostic so an operator can correlate
/// late-run BSS misses with map-idr churn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BssReadState {
    /// Cache is unset (probe not yet discovered) or `mem` is None
    /// (no NUMA layout published yet — pre-boot window). The read
    /// path short-circuits without touching guest memory.
    NotResolved,
    /// Cache is set and the PA is in-bounds, but the latched u32 is
    /// still `0`. The probe has not flipped its sticky 0→1 latch yet.
    NotTriggered,
    /// Cache is set and the PA is in-bounds; the read returned a
    /// non-zero value. The probe has latched its
    /// `ktstr_err_exit_detected` flag.
    Triggered,
    /// Cache is set but the PA falls outside every live DRAM region.
    /// Distinct from `NotTriggered` so callers can warn on a stale
    /// cache without conflating it with "no fire yet". A bare
    /// `read_u32` on the same PA returns `0` per
    /// `monitor::reader::GuestMem::read_scalar`'s OOB-zero path,
    /// which would hide the regression.
    OutOfBounds,
}

/// Resolve the BPF `.bss` latch read into the three-way
/// [`BssReadState`].
///
/// Pure function so the freeze coordinator's poll loop can be tested
/// in isolation — drives the same `mem.read_u32` and `region_avail`
/// calls the production loop performs at the `bss_state` binding,
/// but without booting a VM. The `OutOfBounds` branch uses
/// `region_avail(pa) >= 4` to confirm the cached PA still resolves
/// to a 4-byte-readable mapping; without that check, an OOB PA would
/// silently report `NotTriggered` because
/// [`monitor::reader::GuestMem::read_u32`] returns zeroes for
/// out-of-bounds PAs.
pub(super) fn bss_read_state(
    mem: Option<&monitor::reader::GuestMem>,
    cached_pa: Option<u64>,
) -> BssReadState {
    match (mem, cached_pa) {
        (Some(m), Some(pa)) => {
            if m.region_avail(pa) < 4 {
                BssReadState::OutOfBounds
            } else if m.read_u32(pa, 0) != 0 {
                BssReadState::Triggered
            } else {
                BssReadState::NotTriggered
            }
        }
        _ => BssReadState::NotResolved,
    }
}

/// Combine the watchpoint hit latch and the bss-latch state into the
/// run-loop's "fire this iteration" verdict. The hardware watchpoint
/// is the primary path (synchronous KVM_EXIT_DEBUG delivery); the
/// bss-latch read is the fallback for kernels where the watchpoint
/// could not be armed (no `scx_root` symbol, BTF stripped of
/// `scx_sched`, KVM_SET_GUEST_DEBUG ioctl rejected). Either signal
/// alone is sufficient to start the late-trigger freeze.
///
/// Only [`BssReadState::Triggered`] counts as a fire on the bss
/// path — `OutOfBounds`, `NotResolved`, and `NotTriggered` all
/// resolve to "no observable fire this iteration" so a stale
/// cached PA after probe unload cannot
/// synthesise a phantom fire from arbitrary DRAM bytes.
pub(super) fn compute_err_triggered(watchpoint_hit: bool, bss_state: BssReadState) -> bool {
    watchpoint_hit || matches!(bss_state, BssReadState::Triggered)
}

/// Predicate the post-rendezvous re-read uses to detect a
/// watchpoint-only trigger: the hardware watchpoint fired but the
/// bss latch did NOT — the rendezvous either gate-suppressed the
/// dump (non-error exit_kind value, see the SCX_EXIT_ERROR threshold)
/// or timed out before a sticky bss flip could land. The caller
/// resets `watchpoint.hit` and keeps watching instead of marking
/// Done so a subsequent genuine error-class write retriggers cleanly.
///
/// A bss flip observed during the rendezvous window (the post-read
/// returns `BssReadState::Triggered`) routes via the bss-or-mixed
/// arm — that path marks Done because the kernel-side latch is
/// sticky and retrying would just hit the same timeout.
pub(super) fn compute_watchpoint_only_trigger(
    watchpoint_hit: bool,
    bss_state: BssReadState,
) -> bool {
    watchpoint_hit && !matches!(bss_state, BssReadState::Triggered)
}

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
    pub(super) fn run_vm(
        &self,
        run_start: Instant,
        mut vm: kvm::KtstrKvm,
        default_cpu_mask: Option<&[usize]>,
        effective_pinning_plan: Option<&super::host_topology::PinningPlan>,
    ) -> Result<VmRunState> {
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

        // Serialises on-demand captures against themselves: the
        // coordinator sets this Acquire-bool while a TLV-driven
        // snapshot dispatch runs and clears it on completion, so a
        // user-watchpoint hit firing during a CAPTURE-class request
        // does not open a second concurrent capture window. The TX
        // handler is single-threaded on the freeze coord, so the
        // gate's primary defence is against the user-watchpoint
        // dispatcher (which runs in the same iteration body after
        // pending TLV requests drain). Independent of `freeze_state`,
        // which governs only the error-class trigger machine —
        // on-demand captures must service even when
        // `freeze_state == Done` so post-failure `Op::Snapshot` calls
        // still work.
        let on_demand_in_flight = Arc::new(AtomicBool::new(false));

        // Host-side snapshot bridge. Owned by the freeze coordinator
        // and exposed back through `VmRunState` so test code can
        // drain captured reports after the VM exits. The bridge's
        // capture callback returns `None` — the coordinator never
        // calls `bridge.capture()`; instead it runs
        // `freeze_and_capture(false)` directly and stores the
        // resulting report via `bridge.store(name, report)` so the
        // host owns the entire capture pipeline.
        let snapshot_bridge = {
            let cb: crate::scenario::snapshot::CaptureCallback = Arc::new(|_| None);
            crate::scenario::snapshot::SnapshotBridge::new(cb)
        };

        // Probes-ready broadcast EventFd. Shared between the monitor
        // thread's slot-1 wait and the bpf-map-write thread's
        // accessor-init / map-discovery / probes-ready waits — all
        // of which previously slept on independent 100-200 ms timers
        // while polling guest kernel state via .bss latch reads.
        // Replacing the bare sleeps with `poll(POLLIN)` against this
        // eventfd lets ANY waiter that detects its own readiness
        // condition write 1, immediately waking every other waiter
        // for an early re-check. `EFD_NONBLOCK` keeps the writer's
        // `write()` from stalling if the counter is already
        // saturated; readers use `poll`, never `read`, so the level
        // stays high once any writer has fired and the wake fans out
        // to every cloned fd. `try_clone()` uses `dup(2)`, so all
        // clones share the same kernel counter — the broadcast
        // works across as many waiters as we hand out clones to.
        let probes_ready_evt = EventFd::new(EFD_NONBLOCK).context("create probes-ready EventFd")?;
        let probes_ready_evt_for_monitor = probes_ready_evt
            .try_clone()
            .context("clone probes-ready EventFd for monitor")?;
        let probes_ready_evt_for_bpf = probes_ready_evt
            .try_clone()
            .context("clone probes-ready EventFd for bpf-map-write")?;
        // The original is unused once both consumers hold their own
        // dup'd fds. Drop it eagerly so its file descriptor is freed
        // immediately rather than at the end of the run; the clones
        // share the same kernel counter via dup(2) and remain
        // independent.
        drop(probes_ready_evt);

        // Shared parked_evt: every vCPU thread + the virtio-blk
        // worker writes 1 to this counter-mode EventFd immediately
        // after its respective `parked.store(true, Release)` /
        // `paused.store(true, Release)`. The freeze coordinator's
        // rendezvous loop polls this fd alongside kill_evt and
        // bsp_done_evt instead of spin-sleeping on a 100µs cadence.
        // EFD_NONBLOCK so a writer never stalls; counter mode (no
        // EFD_SEMAPHORE) so a single drain consumes any number of
        // coalesced parked signals — the coordinator drains once
        // and re-checks every parked flag.
        //
        // Allocated BEFORE init_virtio_blk so we can plumb the fd
        // into the device's `set_parked_evt` setter immediately
        // after construction, before the worker spawns and observes
        // its first pause.
        let parked_evt = Arc::new(EventFd::new(EFD_NONBLOCK).context("create parked EventFd")?);
        // Shared thaw_evt: written by the freeze coordinator after
        // `freeze.store(false, Release)` so every parked vCPU
        // observes the thaw within microseconds rather than waiting
        // up to 10ms on `park_timeout`. Same EFD_NONBLOCK + counter
        // semantics as parked_evt.
        let thaw_evt = Arc::new(EventFd::new(EFD_NONBLOCK).context("create thaw EventFd")?);

        // Optional virtio-blk: `None` when no disks are attached,
        // `Some` when the builder has at least one `DiskConfig`.
        // Constructed BEFORE we tear down vm.vcpus so the helper
        // can still read `vm.guest_mem` and the irqchip state.
        let virtio_blk = self.init_virtio_blk(&vm)?;
        // Plumb the shared parked_evt into the device so its worker
        // wakes the freeze coordinator's rendezvous on park.
        if let Some(ref blk) = virtio_blk {
            blk.lock().set_parked_evt(parked_evt.clone());
        }

        // Optional virtio-net: `None` when the builder has no
        // `NetConfig` attached, `Some` when configured. Same
        // construction-before-vcpu-takedown rule as virtio-blk.
        let virtio_net = self.init_virtio_net(&vm)?;

        // Virtio-console for host→guest wake delivery. The setup_memory
        // path always emits the device's MMIO node on the kernel
        // cmdline (x86_64) / FDT (aarch64), so the kernel's
        // `virtio_mmio` driver probes for the device unconditionally.
        // The guest's `hvc0_poll_loop` blocks on `/dev/hvc0` and wakes
        // within microseconds when the host pushes a byte. The
        // coordinator and watchdog use this as the host→guest signal
        // channel: the monitor pushes `SIGNAL_VC_DUMP` for SysRq-D
        // dump requests (the dispatch is wake-byte-only — no SHM
        // control byte), the watchdog pushes `SIGNAL_VC_SHUTDOWN` for
        // graceful shutdown, and the bpf-map-write thread pushes
        // `SIGNAL_BPF_WRITE_DONE` to release `wait_for_map_write`.
        let mut vc = virtio_console::VirtioConsole::new();
        vc.set_mem((*vm.guest_mem).clone());
        let virtio_con = Arc::new(PiMutex::new(vc));
        // x86_64: split_irqchip bailed above (line ~137); reaching
        // here implies a unified kernel irqchip, so irqfd registration
        // is safe. aarch64: GICv3 is always kernel-side.
        vm.vm_fd
            .register_irqfd(virtio_con.lock().irq_evt(), kvm::VIRTIO_CONSOLE_IRQ)
            .context("register virtio-console irqfd")?;

        let kill = Arc::new(AtomicBool::new(false));
        // Watchdog-set timeout flag. Distinct from `kill` because
        // `kill` flips on every shutdown path (BSP shutdown, AP
        // panic, watchdog hard timeout) and the consumer
        // (`VmResult::timed_out`) only wants to know when the
        // watchdog fired its hard-deadline branch. The watchdog
        // thread sets this to `true` ONLY on the
        // `Instant::now() >= effective_deadline` arm; the BSP
        // reads it post-loop and the resulting `(exit_code,
        // timed_out)` tuple flows through `run_bsp_loop` →
        // `VmRunState::timed_out` → `VmResult::timed_out`.
        let timed_out_flag = Arc::new(AtomicBool::new(false));
        // Wake fd paired with the `kill` AtomicBool. Setters that
        // flip `kill` (run_vm post-BSP-exit, vCPU shutdown classifier,
        // panic hook) ALSO write to this EventFd so any consumer
        // sleeping on `epoll_wait` returns within microseconds of
        // the flip rather than waiting up to one full poll
        // interval. Production consumers: the monitor loop and the
        // watchdog thread, both spawned below. `EFD_NONBLOCK` keeps
        // the writer's `write()` from stalling if the counter is
        // already saturated; the AtomicBool remains the source of
        // truth — the EventFd is purely a wake signal.
        let kill_evt = Arc::new(EventFd::new(EFD_NONBLOCK).context("create kill EventFd")?);
        // Boot-complete eventfd. Fired by the freeze coordinator
        // when the guest publishes a CRC-valid
        // [`crate::vmm::wire::MSG_TYPE_SYS_RDY`] TLV frame on the
        // virtio-console bulk port. The monitor thread's pre-sample
        // `epoll_wait` registers this fd alongside `kill_evt` and
        // a 5 s timeout — the SYS_RDY frame is the explicit
        // boot-complete signal from the guest's userspace init,
        // sent after `mount_filesystems()` so by the time the
        // monitor wakes the kernel-side prerequisites
        // (`__per_cpu_offset[]` populated by `setup_per_cpu_areas`,
        // `page_offset_base` populated by KASLR randomization) are
        // already met. Replaces an earlier port-0-TX trigger that
        // depended on incidental console traffic. `EFD_NONBLOCK`
        // because the only writer is the coordinator's TLV dispatch
        // and the only reader is the monitor's `epoll_wait`; a
        // stuck or saturated counter is harmless because the wake
        // semantics are level-triggered. Surfaced as a `warn`
        // rather than a hard failure so a kernel without eventfd
        // support (extremely unlikely for KVM-capable hosts) still
        // boots — the monitor will fall through its 5 s timeout
        // without a guest signal.
        let sys_rdy_evt: Option<Arc<EventFd>> = match EventFd::new(EFD_NONBLOCK) {
            Ok(evt) => Some(Arc::new(evt)),
            Err(e) => {
                tracing::warn!(
                    err = %e,
                    "failed to create sys_rdy EventFd; \
                     monitor will not gate on guest-boot signal"
                );
                None
            }
        };
        // Failure-dump freeze rendezvous: broadcast `freeze` flag plus a
        // per-vCPU `parked` ACK, parallel to the existing `kill` +
        // `exited` shutdown rendezvous. The freeze coordinator
        // (spawned below alongside the watchdog) polls the BPF probe's
        // `ktstr_err_exit_detected` .bss flag via `BpfMapAccessor`;
        // when the flag flips it sets `freeze`, kicks every vCPU,
        // awaits N-of-N parked confirmations, runs the dump (placeholder
        // in this batch), and then clears `freeze` to thaw.
        let freeze = Arc::new(AtomicBool::new(false));
        // Scheduler-stats client. Constructed only when the run
        // has a scheduler attached — without a scheduler there is
        // nothing to query, and spawning a drainer thread plus
        // plumbing a client onto `VmResult` would force every test
        // that does `stats_client.unwrap().stats(...)` to wait for
        // its full timeout before discovering "no scheduler". When
        // `scheduler_binary` is `None`, the field on
        // [`VmResult::stats_client`] stays `None` and callers can
        // branch on `.is_none()` to skip the stats path entirely.
        let stats_client = if self.scheduler_binary.is_some() {
            Some(
                crate::vmm::sched_stats::SchedStatsClient::new(
                    virtio_con.clone(),
                    Some(freeze.clone()),
                    // Run-wide kill flag plumbed as the cancel
                    // signal: when the BSP / watchdog flips
                    // `kill`, blocked `request_raw` calls wake and
                    // return `Cancelled` instead of hanging forever.
                    // The host watchdog is the only "timeout" in
                    // the stats path.
                    Some(kill.clone()),
                    // Paired wake fd: the drainer's epoll watches
                    // `kill_evt` so the cancel edge propagates to
                    // a blocked cvar wait within microseconds.
                    Some(kill_evt.clone()),
                )
                .context("construct scheduler-stats client")?,
            )
        } else {
            None
        };
        // Hardware data-write watchpoint state shared between the
        // freeze coordinator (publishes the resolved
        // `*scx_root->exit_kind` KVA into `request_kva`) and every
        // vCPU thread (self-arms when `request_kva` changes; sets
        // `hit` on `KVM_EXIT_DEBUG`). See [`WatchpointArm`] for the
        // full protocol; this Arc is the only carrier and outlives
        // every consumer (the coordinator joins before the vCPU
        // teardown drops the kvm_run mmaps).
        let watchpoint =
            Arc::new(WatchpointArm::new().context("create WatchpointArm.hit_evt EventFd")?);
        let bsp_parked = Arc::new(AtomicBool::new(false));
        let bsp_regs: Arc<std::sync::Mutex<Option<exit_dispatch::VcpuRegSnapshot>>> =
            Arc::new(std::sync::Mutex::new(None));

        let has_immediate_exit = vm.has_immediate_exit;
        let mut vcpus = std::mem::take(&mut vm.vcpus);
        let mut bsp = vcpus.remove(0);

        // Build per-vCPU pin targets from the stored pinning plan.
        // Index i holds the host CPU for vCPU i. BSP is index 0.
        let pin_targets: Vec<Option<usize>> = if let Some(plan) = effective_pinning_plan {
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
        let no_perf_mask: Option<&[usize]> = self
            .no_perf_plan
            .as_ref()
            .map(|p| p.cpus.as_slice())
            .or(default_cpu_mask);

        // Per-AP TID slots — each AP thread stamps gettid() into its
        // `AtomicI32` and fires the paired `Latch` at startup so the
        // monitor can open per-vCPU `perf_event_open` counters bound
        // to the right thread. Index = AP index (0-based among APs);
        // the BSP TID is stamped into a separate slot below since it
        // runs on the current thread. The latch lets the
        // perf-capture path block in `Latch::wait_timeout` instead
        // of sleep-polling the atomic — see
        // [`open_vcpu_perf_capture`].
        let ap_tid_slots: Vec<(Arc<AtomicI32>, Arc<crate::sync::Latch>)> = (0..vcpus.len())
            .map(|_| {
                (
                    Arc::new(AtomicI32::new(0)),
                    Arc::new(crate::sync::Latch::new()),
                )
            })
            .collect();

        let (ap_threads, ap_freeze_handles) = self.spawn_ap_threads(
            vcpus,
            has_immediate_exit,
            &com1,
            &com2,
            Some(&virtio_con),
            virtio_blk.as_ref(),
            virtio_net.as_ref(),
            &kill,
            &kill_evt,
            &freeze,
            &watchpoint,
            &ap_pins,
            no_perf_mask,
            &ap_tid_slots,
            Some(&parked_evt),
            Some(&thaw_evt),
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
        // BSP latch is pre-set so `open_vcpu_perf_capture` returns
        // immediately for index 0 — the BSP TID is stamped
        // synchronously above on this very thread.
        let bsp_latch = Arc::new(crate::sync::Latch::new());
        bsp_latch.set();
        let vcpu_tid_slots: Vec<(Arc<AtomicI32>, Arc<crate::sync::Latch>)> =
            std::iter::once((bsp_tid_slot, bsp_latch))
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

        // aarch64 TCR_EL1 cache. Populated by the BSP loop on first
        // successful read post-MMU-bringup. `None` on x86_64 (the
        // register does not exist there). Threads that build a
        // `GuestKernel` for page-table walks (monitor, BPF map
        // writer, freeze coordinator's scan_tick path,
        // collect_verifier_stats) load this atomic.
        #[cfg(target_arch = "aarch64")]
        let tcr_el1_cache: Option<Arc<std::sync::atomic::AtomicU64>> =
            Some(Arc::new(std::sync::atomic::AtomicU64::new(0)));
        #[cfg(target_arch = "x86_64")]
        let tcr_el1_cache: Option<Arc<std::sync::atomic::AtomicU64>> = None;

        // CR3 (x86_64) / TTBR1_EL1 (aarch64) cache. Populated lazily
        // by the BSP loop after the kernel has established its
        // initial page tables. Used by host-side `GuestKernel`
        // constructions to walk the page tables for `phys_base`
        // resolution — see [`crate::monitor::symbols::resolve_phys_base`].
        // `0` is the bootstrap value; readers tolerate it (the walk
        // fails and `phys_base` falls back to `0`, which is correct
        // on non-KASLR boots).
        let cr3_cache: Arc<std::sync::atomic::AtomicU64> =
            Arc::new(std::sync::atomic::AtomicU64::new(0));
        // Scheduler-attach watchdog reset. Shared `AtomicU64`
        // written once by the host monitor when it observes
        // `*scx_root` transition from null to non-null in guest
        // memory (a scheduler attached); read each tick by the
        // watchdog so the hard deadline resets to attach moment +
        // `self.workload_duration` instead of being counted from
        // VM boot. `0` (the default) is the "no reset requested"
        // sentinel — the watchdog ignores it and keeps using the
        // original `timeout`-derived deadline. The reset CAN extend
        // past the original deadline (no min clamp) so boot-time
        // delays do not eat into the workload budget. Defined ahead of
        // `start_monitor` so the monitor closure can capture a
        // clone; the watchdog clone is taken below at the
        // watchdog setup site.
        let watchdog_reset_ns: Arc<std::sync::atomic::AtomicU64> =
            Arc::new(std::sync::atomic::AtomicU64::new(0));
        let kern_phys_base: Arc<std::sync::atomic::AtomicU64> =
            Arc::new(std::sync::atomic::AtomicU64::new(0));
        let kern_phys_base_evt = Arc::new(EventFd::new(0).expect("eventfd for kern_phys_base"));
        let accessor_ready_evt = Arc::new(EventFd::new(0).expect("eventfd for accessor_ready"));

        let monitor_handle = self.start_monitor(
            &vm,
            &kill,
            &kill_evt,
            run_start,
            vcpu_pthreads,
            perf_capture.clone(),
            probes_ready_evt_for_monitor,
            Some(virtio_con.clone()),
            sys_rdy_evt.clone(),
            tcr_el1_cache.clone(),
            cr3_cache.clone(),
            watchdog_reset_ns.clone(),
            kern_phys_base.clone(),
            kern_phys_base_evt.clone(),
        )?;
        let watchdog_reset_for_coord = watchdog_reset_ns.clone();
        let watchdog_pause_ns: Arc<std::sync::atomic::AtomicU64> =
            Arc::new(std::sync::atomic::AtomicU64::new(0));
        let watchdog_pause_for_coord = watchdog_pause_ns.clone();
        let workload_duration_for_coord = self.workload_duration;
        // First-ScenarioStart timestamp (nanos since `run_start`),
        // biased by `+1` so `0` means "no ScenarioStart frame
        // observed yet". The dispatch.rs ScenarioStart arm
        // CAS-stamps this on the first frame so the periodic-
        // snapshot loop in the coord run-loop can anchor the
        // 10%–90% workload-duration window at the moment the guest
        // workload actually starts (not at boot or at `run_start`).
        // `Arc<AtomicU64>` gives the coord thread shared ownership
        // with the dispatch sinks; both run in the same thread so
        // Relaxed ordering suffices, but the AtomicU64 keeps the
        // type story uniform with `watchdog_reset_for_coord` /
        // `watchdog_pause_for_coord`.
        let scenario_start_ns: Arc<std::sync::atomic::AtomicU64> =
            Arc::new(std::sync::atomic::AtomicU64::new(0));
        let scenario_start_ns_for_coord = scenario_start_ns.clone();
        // Cumulative wall-clock pause time observed between
        // matched `MSG_TYPE_SCENARIO_PAUSE` / `MSG_TYPE_SCENARIO_RESUME`
        // pairs (nanoseconds). Periodic-snapshot boundaries are
        // anchored to workload time, NOT wall-clock time — when the
        // guest workload pauses, the host's `run_start.elapsed()`
        // ticks during the pause but the workload's logical clock
        // does not. The dispatch.rs `ScenarioResume` arm bumps this
        // atomic by `pause_duration` so the run-loop can subtract it
        // from `run_start.elapsed()` to get effective workload-time
        // for the boundary-crossing check. The matching
        // `watchdog_pause_ns` atomic continues to track only the
        // current pause's start (used by the watchdog deadline-
        // extension path); cumulative pause is a periodic-only
        // concern.
        let scenario_pause_cumulative_ns: Arc<std::sync::atomic::AtomicU64> =
            Arc::new(std::sync::atomic::AtomicU64::new(0));
        let scenario_pause_cumulative_for_coord = scenario_pause_cumulative_ns.clone();
        // Periodic-snapshot count plumbed through KtstrVm for the
        // coord run-loop's periodic-capture cadence. `0` (the
        // default) skips the loop entirely — no boundary
        // computation, no per-iteration check.
        let freeze_coord_num_snapshots = self.num_snapshots;
        // Live periodic-fire count published by the run-loop after
        // each successful capture / placeholder store. Threaded
        // out to `VmResult::periodic_fired` so test code can
        // assert coverage. Written by the coordinator thread,
        // read by run_vm AFTER the coordinator joins, so Relaxed
        // ordering paired with the join's happens-before suffices.
        let periodic_fired_slot: Arc<std::sync::atomic::AtomicU32> =
            Arc::new(std::sync::atomic::AtomicU32::new(0));
        let periodic_fired_for_coord = periodic_fired_slot.clone();

        // BPF map write thread: sleeps, discovers a BPF map, writes a value.
        let bpf_write_handle = self.start_bpf_map_write(
            &vm,
            &kill,
            probes_ready_evt_for_bpf,
            tcr_el1_cache.clone(),
            cr3_cache.clone(),
            virtio_con.clone(),
            kern_phys_base.clone(),
        )?;

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
        // BSP-IE-handle liveness gate. The freeze coordinator's
        // captured `ImmediateExitHandle` for the BSP addresses the
        // BSP `VcpuFd`'s kvm_run mmap; that mapping disappears the
        // moment `bsp` (a local in run_vm) falls out of scope. The
        // primary defense against UAF is the `freeze_coord_handle`
        // join inside run_vm BEFORE bsp drops, but this flag is a
        // cheap secondary check the closure consults before any
        // `bsp_ie_handle.set(1)` call so a future restructure that
        // moves the join doesn't silently reintroduce the UAF. Set
        // to `false` by run_vm right before bsp drops; gate every
        // BSP-side immediate_exit write on `bsp_alive.load(Acquire)`.
        let bsp_alive = Arc::new(AtomicBool::new(true));
        let bsp_alive_for_coord = bsp_alive.clone();
        // Wake fd paired with `bsp_done`. Setters (run_vm post-loop,
        // BSP panic hook) flip the AtomicBool AND write `1` to this
        // EventFd so the freeze coordinator's epoll wait returns
        // immediately. Mirrors the `kill` / `kill_evt` pair above.
        // EFD_NONBLOCK so a doubled write (panic hook AND post-loop
        // store) cannot stall — either edge is sufficient.
        let bsp_done_evt = Arc::new(EventFd::new(EFD_NONBLOCK).context("create bsp_done EventFd")?);
        let kill_for_watchdog = kill.clone();
        let timed_out_for_watchdog = timed_out_flag.clone();
        // Wake fds the watchdog blocks on via epoll, paired with the
        // `kill_for_watchdog` and `bsp_done_for_wd` AtomicBools above.
        // The watchdog wakes within microseconds of either flip
        // instead of polling on a 100 ms thread::sleep cadence.
        let kill_evt_for_watchdog = kill_evt.clone();
        let bsp_done_evt_for_wd = bsp_done_evt.clone();
        let rt_watchdog = self.performance_mode;
        let wd_service_cpu = effective_pinning_plan.and_then(|p| p.service_cpu);
        // Clone the virtio-console Arc into the watchdog so the
        // soft-deadline path can push `SIGNAL_VC_SHUTDOWN` to
        // `/dev/hvc0` for graceful shutdown. The guest's
        // `hvc0_poll_loop` blocks on the device read and recognises
        // the byte directly — no SHM signal slot involved.
        let wd_virtio_con = virtio_con.clone();
        // Watchdog-side clones for the scheduler-attach reset
        // signal. The shared `AtomicU64` and the policy decision
        // ("reset is meaningful only when a distinct workload
        // duration was set") are bound here for the watchdog's
        // `move` closure; the matching monitor clone was taken
        // above at `start_monitor` invocation. Skipped at decode
        // time when `workload_duration_for_wd` is `None` — see
        // the watchdog's per-tick reset block.
        let watchdog_reset_for_wd = watchdog_reset_ns.clone();
        let workload_duration_for_wd = self.workload_duration;

        // Freeze coordinator thread: triggers a failure-dump freeze when
        // the BPF probe's `ktstr_err_exit_detected` .bss latch fires
        // (sched_ext error-class exit observed by tp_btf inside
        // probe.bpf.c). The flag lives in the probe BPF program's
        // .bss map — the coordinator polls it via host-side guest
        // physical memory access, NOT via SHM TLV. Discovery is
        // lazy: each iteration tries
        // `BpfMapAccessor::find_map("probe_bp.bss")` (suffix-matched
        // to avoid colliding with a scheduler-under-test's own .bss
        // map) until the probe is loaded into map_idr, then caches
        // the field PA — the .bss value-region PA plus the
        // BTF-resolved byte offset of `ktstr_err_exit_detected`
        // within the section (see `cached_bss_offset`). Subsequent
        // polls run through [`bss_read_state`], which returns a
        // typed Triggered / NotTriggered / OutOfBounds /
        // NotResolved result so a stale PA after a probe unload
        // surfaces as an explicit diagnostic rather than
        // masquerading as "no fire".
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
        // Lock-free `paused` flag handle. The freeze coordinator
        // polls the worker's parked-state in two paths (the
        // rendezvous timeout-diagnostic snapshot and the post-thaw
        // barrier predicate). Both previously read via
        // `d.lock().is_paused()`, which contends with every
        // concurrent device operation that holds the device mutex
        // — `mmio_read`/`mmio_write` from the vCPU thread and any
        // other freeze-coord call site holding the lock. The
        // underlying field is already `Arc<AtomicBool>`, so
        // exposing a clone here lets the rendezvous read it
        // lock-free. The Acquire/Release ordering on the worker's
        // `paused` writes provides the same happens-before edges
        // with the worker's parked-state stores that
        // `is_paused()` does.
        let freeze_coord_virtio_blk_paused: Option<Arc<AtomicBool>> =
            virtio_blk.as_ref().map(|d| d.lock().paused_handle());
        // Clone the virtio-console Arc into the coordinator so it
        // can drain port-1 bulk TLV bytes as the guest writes them
        // (event-driven via the tx_evt eventfd registered into the
        // coord's epoll set below). Bytes are accumulated into
        // `coord_bulk_buf` and parsed at the end of the run; an
        // early SCHED_EXIT TLV flips `kill` so the watchdog and
        // BSP loop exit promptly.
        let freeze_coord_virtio_con = virtio_con.clone();
        // Clone the virtio-console tx_evt so the coord epoll wakes
        // immediately whenever the guest publishes a TX descriptor
        // chain on port 0 or port 1. The tx_evt is shared between
        // those two ports — a spurious wake on port-0 traffic is
        // harmless: the coord just calls `drain_bulk()` and finds
        // an empty buffer. Port 2 TX (scheduler stats) is owned
        // entirely by [`crate::vmm::sched_stats::SchedStatsClient`]
        // and never reaches this coordinator's epoll set.
        let freeze_coord_tx_evt = virtio_con
            .lock()
            .tx_evt()
            .try_clone()
            .context("clone virtio-console tx_evt for coordinator")?;
        let freeze_coord_bsp_parked = bsp_parked.clone();
        let freeze_coord_bsp_regs = bsp_regs.clone();
        let freeze_coord_bsp_done = bsp_done.clone();
        // Watchpoint-arming state shared with every vCPU thread (BSP
        // + APs). The coordinator publishes the resolved
        // `*scx_root->exit_kind` KVA into `request_kva` and polls
        // `hit` instead of the prior BPF .bss latch read. See
        // [`WatchpointArm`] for the full protocol; the Arc outlives
        // every vCPU thread because `collect_results` joins the
        // coordinator BEFORE the AP thread joins drop the VcpuFds.
        let freeze_coord_watchpoint = watchpoint.clone();
        // Shared per-vCPU perf-counter capture. The Arc lets the
        // monitor sampling loop (per-tick timeline) and the freeze
        // coordinator (freeze-instant snapshot) read through the same
        // fds. Inner `Option` is `None` when `perf_event_open` was
        // unavailable on the host; both consumers gracefully degrade
        // to "no perf data" without aborting the run.
        let freeze_coord_perf_capture = perf_capture.clone();
        let freeze_coord_vmlinux = find_vmlinux(&self.kernel);
        // Read vmlinux bytes once at run_vm scope. Shared via Arc
        // with the coordinator closure (for accessor init, dump_btf,
        // dump_cpu_time_symbols) and VmRunState (for
        // collect_verifier_stats). Eliminates the 14-28s cold-cache
        // re-read that caused cleanup hangs.
        let vmlinux_data_shared: Option<Arc<Vec<u8>>> = freeze_coord_vmlinux
            .as_ref()
            .and_then(|p| super::vmlinux::cached_vmlinux_bytes(p));
        // Cached `name -> KVA` map for `Op::WatchSnapshot` arming.
        // Build once here at run_vm scope so every TLV-driven
        // WATCH request is an O(1) HashMap lookup instead of a
        // 50MB+ vmlinux read + ELF parse. None when vmlinux can't
        // be found or the parse failed — `arm_user_watchpoint`
        // will report a clean diagnostic on lookup. Hoisted out of
        // the closure so the spawn-time parse cost is paid once
        // even when the run ends without any WATCH requests.
        let vmlinux_data_for_result = vmlinux_data_shared.clone();
        let prog_accessor_slot: Arc<
            std::sync::Mutex<Option<crate::monitor::bpf_prog::GuestMemProgAccessorOwned>>,
        > = Arc::new(std::sync::Mutex::new(None));
        let prog_accessor_slot_for_coord = prog_accessor_slot.clone();
        let freeze_coord_symbol_cache: Option<Arc<VmlinuxSymbolCache>> = freeze_coord_vmlinux
            .as_deref()
            .and_then(|p| match VmlinuxSymbolCache::from_path(p) {
                Ok(c) => Some(Arc::new(c)),
                Err(e) => {
                    tracing::warn!(
                        path = %p.display(),
                        error = %e,
                        "freeze-coord: vmlinux symbol cache build failed; \
                         Op::WatchSnapshot WATCH requests will return errors"
                    );
                    None
                }
            });
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
        // GuestMem owns its host pointer for the duration of the run.
        // Wrapped in `Arc` so the worker thread that lazy-builds the
        // `GuestMemMapAccessorOwned` and the coordinator's own
        // accessor-borrow paths share the same backing mapping.
        // `Arc<GuestMem>` is `Send` because `GuestMem` is `Send + Sync`
        // (see `unsafe impl Send for GuestMem` in `monitor::reader`).
        let freeze_coord_mem: Option<Arc<monitor::reader::GuestMem>> = match vm.numa_layout.as_ref()
        {
            Some(layout) => Some(Arc::new(monitor::reader::GuestMem::from_layout(
                layout,
                &vm.guest_mem,
            ))),
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
                    Some(Arc::new(unsafe {
                        monitor::reader::GuestMem::new(host_base, mem_size)
                    }))
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
        // Per-AP `alive` flags paired with the IE handles above. The
        // coordinator's pass-1 kick (in `freeze_and_capture`) and
        // `arm_user_watchpoint` gate each `ie.set` on a fresh
        // Acquire load of the corresponding entry, mirroring the
        // BSP-side `bsp_alive` TOCTOU-tightened gate. Without this,
        // an AP panic-unwind under `panic = "unwind"` (test profile)
        // can drop `vcpu` mid-cycle and the coordinator's
        // `Vec<ImmediateExitHandle>` would issue a `write_volatile`
        // through a freed `kvm_run` mapping. The Vec lives the
        // entire coordinator lifetime; index alignment with
        // `freeze_coord_ap_ies` and `freeze_coord_ap_pthreads` is
        // load-bearing — every AP-loop site uses `iter().enumerate()`
        // (or `zip`) so a future change that drops or reorders any
        // one Vec is loud about the regression.
        let freeze_coord_ap_alive: Vec<Arc<AtomicBool>> =
            ap_threads.iter().map(|vt| vt.alive.clone()).collect();
        // Total vCPU count (BSP + APs). Forwarded into dump_state so
        // PERCPU_ARRAY map rendering knows how many per-CPU slots to
        // read — `bpf_array.pptrs[k]` is a `void __percpu *` whose
        // per-CPU expansion needs `__per_cpu_offset[0..nr_cpu_ids]`.
        let freeze_coord_num_cpus = (ap_threads.len() + 1) as u32;
        // NUMA node count from the configured topology. Forwarded
        // into the scx walker (per-node global DSQ pass) and the
        // per-node NUMA event walker. Defaults to 1 on UMA topologies.
        let freeze_coord_num_nodes = self.topology.num_numa_nodes();
        // Lazy BPF cast-analysis handle produced at builder time.
        // The handle is `Arc<LazyCastMap>` and holds only the
        // scheduler binary path plus a `OnceLock` slot; the
        // analyzer runs only when `.get_full()` is first called
        // at dump time on the freeze-coordinator host thread (NOT
        // a vCPU thread — the freeze rendezvous has already
        // paused vCPUs by the time `dump_state` runs). The clone
        // shares the `OnceLock`, so a periodic-capture dump and
        // the final freeze in the same VM both resolve to the
        // same analyzed `Arc<CastAnalysisOutput>` after the first
        // `.get_full()`.
        let freeze_coord_cast_map: Arc<crate::vmm::cast_analysis_load::LazyCastMap> =
            self.cast_map.clone();
        let freeze_coord_on_demand_in_flight = on_demand_in_flight.clone();
        let freeze_coord_snapshot_bridge = snapshot_bridge.clone();
        // Stats-client clone for the periodic-capture path. The
        // periodic-fire branch issues a `stats(&[])` request BEFORE
        // calling `freeze_and_capture(false)` so the JSON it returns
        // reflects the running scheduler — once the freeze rendezvous
        // begins the scheduler's userspace thread is paused and the
        // request would either time out or wedge until thaw. The
        // resulting `serde_json::Value` is bundled with the
        // FailureDumpReport via `SnapshotBridge::store_with_stats` so
        // a later `Sample` view exposes both axes from the same
        // boundary. `None` when no scheduler is configured (the
        // outer `stats_client` builder above returns `None` when
        // `scheduler_binary.is_none()`); in that case periodic
        // captures store `None` in the parallel stats slot and the
        // temporal-stats projection surfaces a per-sample missing-
        // stats failure that the test author can opt to ignore.
        let freeze_coord_stats_client = stats_client.clone();
        // Wake-fd handles for the coord epoll loop. `kill_evt` and
        // `bsp_done_evt` are written by every thread that flips the
        // matching AtomicBool (run_vm post-BSP-exit, vCPU shutdown
        // classifier, BSP panic hook, AP panic hook); the epoll wait
        // fires immediately on either edge instead of polling on a
        // 500 µs sleep cadence. The watchpoint hit_evt clone lets
        // the coord wake on a hardware-watchpoint fire (vCPU thread
        // calls `WatchpointArm::latch_hit`, which writes the
        // eventfd alongside the AtomicBool flip). All three live
        // for the lifetime of the run — `run_vm` joins the coord
        // BEFORE the eventfds drop.
        let freeze_coord_kill_evt = kill_evt.clone();
        // aarch64 TCR_EL1 cache populated by the BSP loop. Threaded
        // into `GuestKernel::new` constructions inside the
        // freeze-coord scan_tick closure (BPF map accessor and
        // prog accessor) so vmalloc-backed kernel reads succeed
        // post-MMU-bringup. None on x86_64.
        let freeze_coord_tcr_el1 = tcr_el1_cache.clone();
        // CR3 (x86_64) / TTBR1_EL1 (aarch64) cache populated by
        // the BSP loop. Threaded into `GuestKernel::new` so the
        // boot-time `phys_base` resolution can walk the live
        // kernel page tables.
        let freeze_coord_cr3 = cr3_cache.clone();
        let freeze_coord_bsp_done_evt = bsp_done_evt.clone();
        // Clone the WatchpointArm.hit_evt for the epoll set. EventFd
        // clones share the underlying counter via dup(2), so the
        // vCPU's `latch_hit` write delivers an edge to every clone.
        let freeze_coord_hit_evt = watchpoint
            .hit_evt
            .try_clone()
            .context("clone WatchpointArm.hit_evt for coordinator")?;
        // Shared parked_evt for the rendezvous wait. Every vCPU
        // thread + the virtio-blk worker writes to this fd
        // immediately after their respective parked/paused Release
        // store; the rendezvous loop polls on this fd alongside
        // kill_evt and bsp_done_evt instead of spin-sleeping.
        let freeze_coord_parked_evt = parked_evt.clone();
        // Shared thaw_evt: the coordinator writes 1 here AFTER the
        // `freeze.store(false, Release)` so every parked vCPU's
        // poll wakes within microseconds rather than waiting on the
        // legacy 10ms park_timeout cadence.
        let freeze_coord_thaw_evt = thaw_evt.clone();
        // Shared bulk-message buffer: the TOKEN_TX handler in the
        // coordinator parses port-1 TLV bytes via `HostAssembler`
        // and drains the per-frame `BulkMessage` values. Without
        // this buffer those messages would be discarded after the
        // SCHED_EXIT scan, leaving `collect_results` blind to every
        // EXIT / TEST / PAYLOAD_METRICS / RAW_PAYLOAD_OUTPUT /
        // PROFRAW frame the guest already published mid-run. The
        // post-exit `drain_bulk()` only catches what arrived AFTER
        // the coordinator stopped draining — not the bulk of a
        // typical run. The Mutex serialises the coord's pushes
        // against `collect_results`'s drain; both occur strictly
        // after the closure spawns and strictly before the
        // coordinator joins, so contention is rare.
        let freeze_coord_bulk_messages: Arc<std::sync::Mutex<Vec<crate::vmm::wire::ShmEntry>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let freeze_coord_bulk_messages_for_closure = freeze_coord_bulk_messages.clone();

        // Captured sys_rdy eventfd for the coordinator's TLV
        // dispatch loop. The TOKEN_TX handler promotes a CRC-valid
        // `MSG_TYPE_SYS_RDY` frame into a single
        // [`EventFd::write`] on this fd, releasing the monitor
        // thread's pre-sample `epoll_wait`. The `Option<Arc<...>>`
        // is replaced with `None` after the first promotion via
        // [`Option::take`] so subsequent SYS_RDY frames (a hostile
        // guest could in principle resend) skip the eventfd write
        // and do not pump the counter. `None` initially when the
        // sys_rdy machinery was not constructed (`EventFd::new`
        // failed at boot — already logged); in that case the
        // monitor will fall through its 5 s boot-wait timeout
        // without a guest signal. `move` semantics on the closure
        // mean the moved-in `Option` is dropped at coordinator
        // shutdown, releasing the host-side reference.
        let mut freeze_coord_sys_rdy_evt = sys_rdy_evt.clone();

        // One-time probe of the host's hardware-watchpoint slot
        // count via `KVM_CHECK_EXTENSION(KVM_CAP_GUEST_DEBUG_HW_WPS)`.
        // The slot 0 reservation for `*scx_root->exit_kind` plus
        // [`crate::scenario::snapshot::MAX_WATCH_SNAPSHOTS`] user
        // slots means the framework needs at least 4 hardware
        // watchpoint slots to arm every requested
        // [`crate::scenario::ops::Op::WatchSnapshot`]. KVM returns
        // the count via `check_extension_int`; `<= 0` means the
        // capability is unavailable. Log only — do not block VM
        // creation: a kernel without the capability still runs
        // tests, just without the watch-driven snapshots, and a
        // probe failure surfacing here is more actionable than a
        // silent `KVM_SET_GUEST_DEBUG` rejection later.
        let hw_wps = vm.vm_fd.check_extension_int(kvm_ioctls::Cap::DebugHwWps);
        if hw_wps <= 0 {
            tracing::warn!(
                "KVM_CAP_GUEST_DEBUG_HW_WPS unavailable on this host \
                 (returned {hw_wps}); Op::WatchSnapshot triggers may \
                 not arm — falling back to BPF .bss poll for the \
                 error-class freeze trigger"
            );
        } else {
            tracing::info!(
                "KVM host advertises {hw_wps} hardware watchpoint \
                 slots via KVM_CAP_GUEST_DEBUG_HW_WPS"
            );
            if hw_wps < 4 {
                tracing::warn!(
                    "KVM host advertises only {hw_wps} hardware \
                     watchpoint slots; the framework reserves slot 0 \
                     for the *scx_root->exit_kind error-class trigger \
                     plus up to {} user slots for Op::WatchSnapshot \
                     — some watch_snapshot arms may fail",
                    crate::scenario::snapshot::MAX_WATCH_SNAPSHOTS,
                );
            }
        }

        let kern_phys_base_for_result = kern_phys_base.clone();
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
                    /// KVA of the kernel's global `scx_tasks` LIST_HEAD
                    /// (`kernel/sched/ext.c:47`). The walker reads
                    /// `scx_tasks.next` via the runtime kernel image
                    /// base ([`Self::start_kernel_map`]) and
                    /// container_of's each list entry back to its
                    /// `task_struct`.
                    scx_tasks_kva: u64,
                    /// Per-CPU `struct rq` PAs (one per logical CPU).
                    /// Built by `compute_rq_pas(runqueues_kva,
                    /// __per_cpu_offset[*], page_offset)`. Each entry
                    /// addresses the rq whose `scx.runnable_list`
                    /// the per-rq walker walks; vec index = CPU index.
                    /// Empty when the per-CPU offset array can't be
                    /// resolved (per-rq walk silently falls back to
                    /// the global walk).
                    rq_pas: Vec<u64>,
                    offsets: crate::monitor::btf_offsets::RunnableScanOffsets,
                    jiffies_64_pa: u64,
                    /// PA of `scx_watchdog_timestamp`
                    /// (`kernel/sched/ext.c:94`). The kernel's
                    /// `scx_tick` (`kernel/sched/ext.c:3409`) compares
                    /// `jiffies - scx_watchdog_timestamp` against the
                    /// scheduler's `watchdog_timeout` and fires
                    /// `SCX_EXIT_ERROR_STALL` when the workqueue
                    /// stopped running. Reading the same value here
                    /// gives the dual-snapshot path the global stall
                    /// signal regardless of whether any individual
                    /// task is stuck on a per-rq runnable_list. None
                    /// when the symbol is absent (kernel without
                    /// sched_ext or stripped vmlinux); per-rq /
                    /// global walks still cover the per-task case.
                    watchdog_timestamp_pa: Option<u64>,
                    /// Paging context (cr3_pa / page_offset / l5 /
                    /// tcr_el1) threaded into the runnable_scan helpers.
                    walk: crate::monitor::reader::WalkContext,
                    /// Runtime kernel image base
                    /// (`__START_KERNEL_map` on x86_64,
                    /// `KIMAGE_VADDR` on aarch64). Threaded into the
                    /// runnable_scan helpers so `scx_tasks` and other
                    /// kernel-text-mapped symbols translate via the
                    /// VA-bits-aware base resolved from `TCR_EL1` —
                    /// matches the [`super::super::monitor::guest::GuestKernel`]
                    /// the surrounding accessors share.
                    start_kernel_map: u64,
                    /// Runtime KASLR offset (`phys_base` on x86_64;
                    /// `0` on aarch64 / non-KASLR boots). Required by
                    /// `text_kva_to_pa_with_base` so KASLR kernels
                    /// resolve `scx_tasks` / `jiffies_64` /
                    /// `scx_watchdog_timestamp` correctly.
                    phys_base: u64,
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
                // Cached vmlinux bytes shared across every retry of
                // `try_init_owned_accessor` and
                // `try_init_owned_prog_accessor`. The previous code
                // re-ran `std::fs::read(vmlinux)` inside both helpers
                // on every scan tick — at 50-340 MB per call on cold
                // disk cache the pair could exceed the 12 s post-
                // BSP-done kill timer before the coord ever reached
                // its epoll wait. Reading once at coord scope cuts
                // the per-iteration cost to a few-millisecond
                // `goblin::elf::Elf::parse` against the cached bytes.
                //
                // The borrow lifetime constraint blocks caching the
                // parsed `Elf<'static>` (it borrows from the Vec); the
                // helpers re-parse the cached bytes per call instead.
                // Parsing is microseconds — only the file read was
                // slow.
                let _tvmr = std::time::Instant::now();
                let vmlinux_data: Option<Arc<Vec<u8>>> = vmlinux_data_shared.clone();
                // Worker-populated accessor pair. Built off the freeze
                // coordinator thread so the slow ELF + BTF parse +
                // symbol HashMap (~4 s on debug vmlinux) does not
                // block the coordinator from servicing TOKEN_TX
                // events on its epoll loop. The worker writes both
                // accessors atomically via `OnceLock::set` once the
                // GuestKernel handshake succeeds and both BTF parses
                // land. Subsequent reads from the coordinator are
                // nanosecond-scale `OnceLock::get` calls.
                //
                // `Arc<OnceLock<(...)>>` shape: `Arc` so the worker
                // and coordinator share ownership; `OnceLock` so the
                // publish is one-shot and lock-free on read; the
                // tuple shape so both accessors land atomically — a
                // failure-dump path that builds a `ScxWalkerCapture`
                // must also have the matching `prog_runtime_stats`
                // accessor, so partial pairs would skew the dump.
                //
                // `GuestMemMapAccessorOwned` and
                // `GuestMemProgAccessorOwned` are `Send` because they
                // own `GuestKernel`, which holds `Arc<GuestMem>`
                // (was `&'a GuestMem`). The Arc shape lets the worker
                // own the kernel handle independently of the
                // coordinator's stack.
                let accessors_oncelock: Arc<std::sync::OnceLock<(
                    crate::monitor::bpf_map::GuestMemMapAccessorOwned,
                    Option<crate::monitor::bpf_prog::GuestMemProgAccessorOwned>,
                )>> = Arc::new(std::sync::OnceLock::new());
                // Borrowed views into the OnceLock pair. Reset to
                // `Some(...)` on the first scan tick after the worker
                // publishes; remain `None` while the worker is still
                // retrying. Each loop body that reads these gates on
                // `Some` exactly as the prior `Option<...Owned>`-typed
                // shape did, so no call-site logic changes. Borrowing
                // is lifetime-clean: the OnceLock is moved into the
                // worker's clone via `Arc`, the coordinator retains
                // its own clone, and `&` borrows from a shared `Arc`
                // are valid for the entire freeze_coord closure scope.
                let mut owned_accessor:
                    Option<&crate::monitor::bpf_map::GuestMemMapAccessorOwned> = None;
                let mut owned_prog_accessor:
                    Option<&crate::monitor::bpf_prog::GuestMemProgAccessorOwned> = None;
                let mut coord_kaslr_offset: u64 = 0;
                // Spawn the accessor-init worker before entering the
                // coordinator's epoll loop. The worker:
                //   1. Loops `try_init_owned_accessor` +
                //      `try_init_owned_prog_accessor` against the
                //      shared `Arc<GuestMem>` until both succeed.
                //   2. On success: stores `phys_base + 1` (biased) in
                //      `kern_phys_base` via `compare_exchange(0, ..)`
                //      so the monitor thread observes the value
                //      regardless of whether the guest's port-2
                //      publish landed first.
                //   3. Publishes the pair via `OnceLock::set`.
                //   4. Exits — the OnceLock is read-only thereafter.
                //
                // The worker honors `freeze_coord_kill` between
                // retries and bails immediately on shutdown so a
                // still-booting VM that's killed mid-init does not
                // delay coord teardown. The 60s budget is the same
                // order as `start_bpf_map_write`'s phase-1 deadline;
                // a boot that hasn't published the bootstrap symbols
                // by then is genuinely stuck and the dump path is
                // unavailable for the rest of the run regardless.
                let accessor_init_handle: Option<std::thread::JoinHandle<()>> = match (
                    freeze_coord_mem.as_ref(),
                    freeze_coord_vmlinux.as_ref(),
                    vmlinux_data.as_deref(),
                ) {
                    (Some(mem), Some(vmlinux), Some(data)) => {
                        let mem_for_worker = mem.clone();
                        let vmlinux_for_worker = vmlinux.clone();
                        let data_for_worker = data.clone();
                        let tcr_for_worker = freeze_coord_tcr_el1.clone();
                        let cr3_for_worker = freeze_coord_cr3.clone();
                        let kern_phys_base_for_worker = kern_phys_base.clone();
                        let kern_phys_base_evt_for_worker = kern_phys_base_evt.clone();
                        let accessor_ready_evt_for_worker = accessor_ready_evt.clone();
                        let kill_for_worker = freeze_coord_kill.clone();
                        let kill_evt_for_worker = freeze_coord_kill_evt.clone();
                        let oncelock_for_worker = accessors_oncelock.clone();
                        std::thread::Builder::new()
                            .name("vmm-accessor-init".into())
                            .spawn(move || {
                                let deadline = Instant::now()
                                    + Duration::from_secs(60);
                                let _init_t0 = Instant::now();
                                // poll() on kill_evt so the worker
                                // wakes instantly on shutdown instead
                                // of sleeping through a 100ms window.
                                let kill_fd = {
                                    use std::os::unix::io::AsRawFd;
                                    kill_evt_for_worker.as_raw_fd()
                                };
                                let elf = match goblin::elf::Elf::parse(&data_for_worker) {
                                    Ok(e) => e,
                                    Err(e) => {
                                        tracing::warn!(
                                            error = %e,
                                            "accessor-init: vmlinux ELF parse failed"
                                        );
                                        return;
                                    }
                                };
                                loop {
                                    if kill_for_worker.load(Ordering::Acquire) {
                                        return;
                                    }
                                    if Instant::now() >= deadline {
                                        tracing::warn!(
                                            "freeze-coord accessor-init worker: \
                                             60s deadline exceeded; coordinator \
                                             will run without owned-accessor \
                                             pair (freeze dump path unavailable)"
                                        );
                                        return;
                                    }
                                    // Use the guest-reported phys_base
                                    // (includes kaslr_offset) as a hint
                                    // so GuestKernel gets the correct
                                    // value instead of 0 from the
                                    // failing page-table walk.
                                    let biased = kern_phys_base_for_worker
                                        .load(Ordering::Acquire);
                                    let pb_hint = if biased != 0 {
                                        biased.wrapping_sub(1)
                                    } else {
                                        let pb_evt_fd = {
                                            use std::os::unix::io::AsRawFd;
                                            kern_phys_base_evt_for_worker.as_raw_fd()
                                        };
                                        let mut pfds = [
                                            libc::pollfd { fd: kill_fd, events: libc::POLLIN, revents: 0 },
                                            libc::pollfd { fd: pb_evt_fd, events: libc::POLLIN, revents: 0 },
                                        ];
                                        unsafe { libc::poll(pfds.as_mut_ptr(), 2, 200) };
                                        continue;
                                    };
                                    let tcr_val = tcr_for_worker
                                        .as_ref()
                                        .map(|c| c.load(Ordering::Acquire))
                                        .unwrap_or(0);
                                    let cr3_val = cr3_for_worker.load(Ordering::Acquire);
                                    let map_res = crate::monitor::bpf_map::GuestMemMapAccessorOwned
                                        ::from_elf_with_hint(
                                            mem_for_worker.clone(),
                                            &elf,
                                            &data_for_worker,
                                            &vmlinux_for_worker,
                                            tcr_val,
                                            cr3_val,
                                            pb_hint,
                                        );
                                    if kill_for_worker.load(Ordering::Acquire) {
                                        return;
                                    }
                                    if let Ok(map) = map_res {
                                        let po = map.guest_kernel()
                                            .walk_context().page_offset;
                                        if po & (1u64 << 63) == 0
                                            || po & 0xFFF != 0
                                        {
                                            let mut pfd = libc::pollfd {
                                                fd: kill_fd,
                                                events: libc::POLLIN,
                                                revents: 0,
                                            };
                                            unsafe {
                                                libc::poll(&mut pfd, 1, 100);
                                            }
                                            continue;
                                        }
                                        let phys_base =
                                            map.guest_kernel().phys_base();
                                        let _ = kern_phys_base_for_worker
                                            .compare_exchange(
                                                0,
                                                phys_base.wrapping_add(1),
                                                Ordering::Release,
                                                Ordering::Relaxed,
                                            );
                                        let prog_res = {
                                            let shared_syms = map.guest_kernel().symbols_arc();
                                            let kernel = crate::monitor::guest::GuestKernel
                                                ::from_elf_with_symbols(
                                                    mem_for_worker.clone(),
                                                    shared_syms,
                                                    &elf,
                                                    tcr_val,
                                                    cr3_val,
                                                    pb_hint,
                                                );
                                            kernel.and_then(|k| {
                                                crate::monitor::bpf_prog::GuestMemProgAccessorOwned
                                                    ::finish(k, &elf, &data_for_worker, &vmlinux_for_worker)
                                            })
                                        };
                                        let _ = oncelock_for_worker
                                            .set((map, prog_res.ok()));
                                        let _ = accessor_ready_evt_for_worker.write(1);
                                        return;
                                    }
                                    // Wait on kill_evt with 200ms
                                    // timeout. Wakes instantly on
                                    // kill; retries on timeout.
                                    let mut pfd = libc::pollfd {
                                        fd: kill_fd,
                                        events: libc::POLLIN,
                                        revents: 0,
                                    };
                                    unsafe {
                                        libc::poll(&mut pfd, 1, 200);
                                    }
                                }
                            })
                            .ok()
                    }
                    _ => None,
                };
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
                let dump_btf = vmlinux_data.as_deref().zip(freeze_coord_vmlinux.as_ref())
                    .and_then(|(data, path)| crate::monitor::btf_offsets::load_btf_from_bytes(data, path).ok());
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
                let dump_cpu_time_symbols = vmlinux_data.as_deref()
                    .and_then(|data| crate::monitor::symbols::KernelSymbols::from_vmlinux_bytes(data).ok());
                // SCX walker BTF sub-group offsets. Resolved once at
                // coord start; per-sub-group resolution failures land
                // inside the composite as None so the walker's
                // `missing_groups()` can report which passes are blind
                // (a kernel built without CONFIG_NUMA loses
                // `scx_sched_pnode`, etc.).
                let dump_scx_walker_offsets = dump_btf
                    .as_ref()
                    .and_then(|btf| {
                        crate::monitor::btf_offsets::ScxWalkerOffsets::from_btf(btf).ok()
                    });
                // Per-task enrichment BTF offsets. All-or-nothing —
                // any missing sub-group leaves the composite Err and
                // the enrichment capture is skipped. The walker
                // never runs partially: every Tier-1 field must be
                // resolvable, otherwise the dump path falls back to
                // `REASON_NO_TASK_WALKER`.
                let dump_task_enrichment_offsets = dump_btf
                    .as_ref()
                    .and_then(|btf| {
                        crate::monitor::btf_offsets::TaskEnrichmentOffsets::from_btf(btf).ok()
                    });
                // Per-node NUMA event BTF offsets. Required for the
                // per-node `vm_numa_event[]` walker. Resolved once at
                // coord start; absent on stripped vmlinux or kernels
                // built without `CONFIG_NUMA + CONFIG_VM_EVENT_COUNTERS`.
                let dump_numa_offsets = dump_btf
                    .as_ref()
                    .and_then(|btf| {
                        crate::monitor::btf_offsets::NumaStatsOffsets::from_btf(btf).ok()
                    });
                // Hoisted scan_ctx prerequisites. These are pure
                // functions of the host inputs (vmlinux ELF and the
                // already-loaded BTF), so they succeed or fail
                // deterministically at coord-start — no boot-race
                // window to retry through. Computing once here avoids
                // re-parsing the BTF on every scan_ctx try_resolve
                // iteration. The previous per-iteration retry pattern
                // was harmless functionally (idempotent) but burned
                // ~MB-scale ELF reparse work every SCAN_INTERVAL until
                // owned_accessor caught up. These two values plus
                // `dump_cpu_time_symbols.scx_tasks` and `runqueues`
                // feed RunnableScanCtx construction below — the
                // global walker reads `scx_tasks` directly via
                // `text_kva_to_pa_with_base` (or
                // `GuestKernel::text_kva_to_pa`), the per-rq walker uses
                // `runqueues` + `__per_cpu_offset` to address each
                // CPU's `rq`.
                let scan_offsets = dump_btf.as_ref().and_then(|btf| {
                    crate::monitor::btf_offsets::RunnableScanOffsets::from_btf(btf).ok()
                });
                // jiffies_64 lives on the KernelSymbols instance
                // computed above for the dump capture. Reusing it
                // pays a single from_vmlinux cost per coordinator.
                let scan_jiffies_64_kva =
                    dump_cpu_time_symbols.as_ref().and_then(|s| s.jiffies_64);
                // Lazy-discovered cached PA of `ktstr_err_exit_detected`
                // within the probe BPF program's .bss map. None until
                // the probe loads into map_idr (rust_init phase 2b);
                // discovery retries each iteration until success.
                //
                // Invalidated each scan tick when the source `.bss`
                // map disappears from `map_idr` or rebinds to a
                // different `value_kva` — see the rediscovery guard
                // below. Without that, a probe BPF program that
                // unloads mid-run leaves the freed vmalloc page's
                // PA cached here; the kernel can re-allocate that
                // page for unrelated guest memory, and the next
                // `read_u32(pa, 0)` returns whatever bytes that
                // page now holds (any non-zero value latches a
                // phantom `err_triggered` and synthesizes a bogus
                // failure dump).
                let mut cached_bss_pa: Option<u64> = None;
                // Companion to `cached_bss_pa`: the `value_kva` of
                // the `.bss` map that produced it. Used as a stale-
                // probe canary — if the next scan tick finds the
                // same-named map with a different `value_kva` (the
                // bpf_array slab moved across an unload+reload) the
                // PA is invalidated and re-resolved. Stays in sync
                // with `cached_bss_pa`: both Some or both None.
                let mut cached_bss_value_kva: Option<u64> = None;
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
                // One-shot latch for the cached_bss_pa-points-OOB
                // diagnostic. The OOB read state can occur if the
                // cached PA was resolved against a probe `.bss` map
                // that has since been freed (probe unload mid-run,
                // vmalloc page recycled). The first observation
                // surfaces a warn so an operator inspecting the run
                // knows the .bss path has gone silent; subsequent
                // observations stay debug-level so the logs do not
                // fill up across the remaining run lifetime.
                let mut bss_oob_warn_logged = false;
                // Cached `*scx_root` value (the vmalloc/slab KVA of
                // the live `struct scx_sched`). Tracked across scan
                // ticks so we can detect a sched_ext detach + reattach
                // cycle: when the kernel tears down the scheduler the
                // pointer goes 0 (the slab page is freed); when a new
                // scheduler attaches it points at a fresh slab. Each
                // change re-publishes `request_kva` AND
                // `kind_host_ptr` so vCPU threads re-arm on the new
                // KVA and post-fire `read_volatile` reads land on the
                // current slab — the previous one-shot publish gate
                // pinned `kind_host_ptr` at the original slab page
                // forever, and a stale deref after rebind would touch
                // freed (or repurposed) host memory.
                //
                // Resolution sequence per scan tick:
                //   1. read scx_root_kva from KernelSymbols (resolved
                //      once at coord-start via vmlinux);
                //   2. translate scx_root_kva → root_pa via
                //      `GuestKernel::text_kva_to_pa` (it lives in the
                //      kernel text mapping, not vmalloc);
                //   3. read u64 at root_pa to get sched_kva (the
                //      vmalloc-allocated `struct scx_sched`);
                //   4. compare against `last_sched_kva` — bail on no
                //      change (fast path on every scan tick post-
                //      attach);
                //   5. on change to non-zero: publish
                //      `sched_kva + exit_kind_offset` into
                //      `request_kva` (and the matching host pointer
                //      into `kind_host_ptr`); each vCPU thread polls
                //      that slot before its next KVM_RUN and re-arms;
                //   6. on change to zero (detach): publish 0 / null so
                //      vCPUs disarm via `KVM_SET_GUEST_DEBUG` without
                //      this slot's enable bits and stop tripping on
                //      the now-freed slab address.
                //
                // `*scx_root` only becomes non-NULL once a sched_ext
                // scheduler attaches; before that we silently retry
                // — the BPF .bss fallback (still wired up below)
                // covers the gap.
                let mut last_sched_kva: u64 = 0;
                let mut cached_exit_kind_pa: Option<u64> = None;
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
                // Latest skip reason from try_resolve. Captures the
                // specific prerequisite that prevented resolution on
                // the most recent attempt (most useful when scan_ctx
                // is still None at the late-trigger point) so the
                // late-trigger emission can stamp it into
                // `DualFailureDumpReport::early_skipped_reason`. Set
                // back to None on a successful resolve so a once-
                // failed-then-recovered run does not carry stale
                // breadcrumbs forward.
                let mut scan_ctx_skip_reason: Option<&'static str> = None;
                // Retry counter and one-shot warn latch for the
                // scan_ctx resolve. The resolve runs once per
                // SCAN_INTERVAL (250 ms) poll iteration until it
                // succeeds; without a diagnostic an operator who
                // built ktstr against a kernel lacking
                // sched_ext_entity (or stripped of jiffies_64)
                // gets a silent dual-snapshot disable.
                // Wait `SCAN_CTX_WARN_AFTER_ITERS` iterations
                // (~3 s at 250 ms cadence) before warning so legit
                // boot-time delays (owned_accessor not yet ready,
                // GuestKernel handshake mid-flight) don't trigger
                // false alarms. The latch ensures the warn fires at
                // most once per VM run.
                let mut scan_ctx_retries: u32 = 0;
                let mut scan_ctx_warned: bool = false;
                const SCAN_CTX_WARN_AFTER_ITERS: u32 = 12;
                // The accessor-init worker spawned above owns the
                // retry/warn discipline for its two `try_init_*`
                // helpers; the coordinator no longer tracks
                // `accessor_retries` / `accessor_warned` /
                // `accessor_last_err` fields here. The constant below
                // is reused by the `prog_per_cpu_offsets` /
                // `scan_ctx` retry blocks further down.
                const LAZY_ACCESSOR_WARN_AFTER_ITERS: u32 = 10;
                // Sibling state for `try_init_prog_per_cpu_offsets`.
                // Two distinct failure modes warrant different
                // diagnostics: a missing `__per_cpu_offset` symbol
                // (`per_cpu_offset_kva == 0`) is a PERMANENT failure
                // that warns immediately on the first observation —
                // the symbol won't materialise mid-run, so retrying
                // silently masks a stripped vmlinux. Conversely, a
                // present symbol whose live array still has zero
                // slots (`offsets.contains(&0)`) is a TRANSIENT
                // boot-progress condition that resolves once the
                // guest's `setup_per_cpu_areas` populates each
                // CPU's slot; warn after `LAZY_ACCESSOR_WARN_AFTER_ITERS`
                // retries so a guest that genuinely fails to bring
                // up its per-CPU areas surfaces a diagnostic
                // instead of permanently-disabled
                // prog_runtime_stats. Each warn latches via its
                // own `_warned` bool to fire at most once per VM run.
                let mut per_cpu_offsets_retries: u32 = 0;
                let mut per_cpu_offsets_warned: bool = false;
                let mut per_cpu_offsets_kva_warned: bool = false;
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
                // Trajectory tracking for the early-trigger diagnostic.
                // Records the max `max_age` observed across the run
                // and how many scan iterations have run. Surfaced in a
                // warn when err_triggered fires while
                // freeze_state == Idle (i.e. the early path never
                // captured) so an operator can distinguish three
                // failure modes from a single log line:
                //
                //   - early_scan_iters == 0   → scan_ctx never resolved
                //                                (scan_ctx_warn already
                //                                fires earlier; this
                //                                cross-checks).
                //   - peak_max_age == 0       → scan ran but never
                //                                observed a live task
                //                                (likely empty
                //                                runnable_list, wrong
                //                                offsets, or the scan
                //                                was reading unmapped
                //                                memory).
                //   - peak_max_age > 0 but    → scan was working but
                //     < half_threshold          the kernel watchdog
                //                                fired before any task
                //                                aged past the
                //                                half-way mark (very
                //                                short stalls or an
                //                                err-class exit that
                //                                isn't a stall, e.g.
                //                                scx_bpf_error()).
                //
                // The Display fallback at dump/display.rs:65 already
                // points operators at RUST_LOG=ktstr=debug for scan
                // resolution; this trajectory snapshot is the more
                // actionable signal because it's emitted at the
                // moment of failure with structured fields rather
                // than as a per-iteration debug stream.
                let mut early_peak_max_age_jiffies: u64 = 0;
                let mut early_scan_iters: u64 = 0;
                // Cadence policy. The loop blocks in `epoll_wait`
                // until one of the registered fds fires (kill,
                // bsp_done, virtio-console TX, watchpoint hit,
                // scanner tick) OR `POLL_TIMEOUT_MS` elapses. The
                // previous
                // implementation drove this by `thread::sleep(500
                // µs)` and a `poll_iter % 200 == 0` decimator. The
                // event-driven design wakes the coordinator within
                // microseconds of any trigger source — including
                // the watchpoint hit and the kill / bsp_done flips —
                // and only does heavy work (boot-race accessor
                // construction, BPF .bss-PA lookup, runnable_at
                // scan) when the periodic scanner timerfd fires.
                const POLL_TIMEOUT_MS: i32 = 500;
                // 250 ms gives enough resolution for typical
                // half-watchdog thresholds (e.g. 4000 jiffies on
                // a 1 kHz HZ kernel = 4 s, so a 250 ms scan
                // cadence catches the half-way crossing within
                // 6.25% of the threshold) while halving the
                // freeze coord's scan-tick CPU draw vs the
                // legacy 100 ms cadence. The early-trigger path
                // walks both the global `scx_tasks` list and
                // every per-CPU `rq->scx.runnable_list` per
                // tick; on a many-vCPU host the larger interval
                // matters.
                const SCAN_INTERVAL: Duration = Duration::from_millis(250);
                // Per-fd epoll tokens. Match-on tokens dispatches
                // events without re-reading fd numbers.
                const TOKEN_KILL: u64 = 0;
                const TOKEN_BSP_DONE: u64 = 1;
                const TOKEN_WATCHPOINT: u64 = 3;
                const TOKEN_SCANNER: u64 = 4;
                /// virtio-console tx_evt — wakes whenever the guest
                /// publishes a TX descriptor chain on port 0 or port 1.
                /// The coordinator drains port-1 bulk TLV bytes and
                /// promotes a SCHED_EXIT entry into the run-wide
                /// `kill` flag, and intercepts
                /// [`crate::vmm::wire::MSG_TYPE_SNAPSHOT_REQUEST`]
                /// frames so the matching dispatch (CAPTURE / WATCH)
                /// runs in the same iteration body and the reply
                /// is pushed back to the guest via
                /// [`crate::vmm::virtio_console::VirtioConsole::queue_input_port1`].
                /// Port-0 (console) TX wakes are harmless: the coord
                /// drain returns an empty buffer and the byte stays
                /// in the host stdout thread's `drain_output` slot.
                /// Port 2 TX (scheduler stats) does not reach this
                /// epoll set — the
                /// [`crate::vmm::sched_stats::SchedStatsClient`]
                /// owns its own drainer thread and stats_tx_evt
                /// epoll, leaving this coordinator unaffected by
                /// stats traffic.
                const TOKEN_TX: u64 = 5;
                const TOKEN_ACCESSOR_READY: u64 = 6;
                let epoll = match Epoll::new() {
                    Ok(e) => e,
                    Err(e) => {
                        tracing::error!(
                            error = %e,
                            "freeze-coord: epoll_create1 failed; aborting coordinator"
                        );
                        return;
                    }
                };
                use std::os::unix::io::AsRawFd;
                let mut scanner_tfd = match TimerFd::new() {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::error!(
                            error = %e,
                            "freeze-coord: timerfd_create failed; aborting coordinator"
                        );
                        return;
                    }
                };
                if let Err(e) = scanner_tfd.reset(SCAN_INTERVAL, Some(SCAN_INTERVAL)) {
                    tracing::error!(
                        error = %e,
                        "freeze-coord: timerfd_settime failed; aborting coordinator"
                    );
                    return;
                }
                // Register every fd. Failure to register any one of
                // these would cause the coordinator to silently miss
                // a wake source, so abort instead of degrading.
                for (fd, token, name) in [
                    (freeze_coord_kill_evt.as_raw_fd(), TOKEN_KILL, "kill_evt"),
                    (
                        freeze_coord_bsp_done_evt.as_raw_fd(),
                        TOKEN_BSP_DONE,
                        "bsp_done_evt",
                    ),
                    (
                        freeze_coord_hit_evt.as_raw_fd(),
                        TOKEN_WATCHPOINT,
                        "watchpoint_hit_evt",
                    ),
                    (scanner_tfd.as_raw_fd(), TOKEN_SCANNER, "scanner_tfd"),
                    (freeze_coord_tx_evt.as_raw_fd(), TOKEN_TX, "virtio_console_tx_evt"),
                    (accessor_ready_evt.as_raw_fd(), TOKEN_ACCESSOR_READY, "accessor_ready_evt"),
                ] {
                    if let Err(e) = epoll.ctl(
                        ControlOperation::Add,
                        fd,
                        EpollEvent::new(EventSet::IN, token),
                    ) {
                        tracing::error!(
                            error = %e,
                            fd_name = name,
                            "freeze-coord: epoll_ctl ADD failed; aborting coordinator"
                        );
                        return;
                    }
                }
                let mut events_buf = [EpollEvent::default(); 6];
                // Accumulator for partially-received TLV bulk frames.
                // The kernel's virtio_console TX path issues
                // descriptor chains as the guest writes; a single
                // logical TLV frame can span multiple wakes if the
                // guest's `write_all` was split across pages or
                // descriptor sizes. The streaming
                // [`crate::vmm::bulk::HostAssembler`] retains partial
                // bytes across `feed` calls so a frame split across
                // multiple TX wakes is recovered without loss.
                //
                // SCHED_EXIT promotion: every drained message is
                // inspected for [`wire::MSG_TYPE_SCHED_EXIT`]; when
                // observed, the run-wide `kill` flag flips so the
                // BSP run loop and the watchdog exit promptly
                // instead of waiting for the watchdog deadline.
                let mut bulk_assembler = crate::vmm::bulk::HostAssembler::new();
                // Per-iteration accumulator for guest-side
                // [`crate::vmm::wire::MSG_TYPE_SNAPSHOT_REQUEST`]
                // frames the TOKEN_TX handler decoded. Drained later
                // in the iteration body where `freeze_and_capture` /
                // `thaw_and_barrier` / `arm_user_watchpoint` are in
                // scope; the dispatch frames a
                // `MSG_TYPE_SNAPSHOT_REPLY` TLV and pushes it back
                // through `queue_input_port1`. CRC-failed frames are
                // never appended — a torn frame would otherwise let
                // a hostile guest force a spurious capture, mirroring
                // the SCHED_EXIT promotion gate.
                let mut snapshot_requests_pending: Vec<SnapshotRequest> = Vec::new();
                // CAPTURE requests received before `owned_accessor`
                // adoption are queued here instead of being serviced
                // immediately. Servicing pre-adoption produces a
                // partial-dump report (0 maps, vcpu_regs only — see
                // the `// Partial dump:` branch in
                // `freeze_and_capture`) which is useless to the test
                // author who asked for `Op::snapshot("...")`.
                //
                // The queue is drained at the accessor-adoption site
                // by appending its contents back onto
                // `snapshot_requests_pending`, so the same iteration's
                // CAPTURE drain dispatches them through the normal
                // `freeze_and_capture(false)` flow with the accessor
                // present.
                //
                // If the accessor never adopts (worker permanently
                // failed past its 60 s deadline), the queue is
                // dropped at coord exit and the guest's blocking
                // reader on `/dev/vport0p1` times out at the per-Op
                // 30 s deadline — same observable behaviour as a
                // late-boot rendezvous timeout. WATCH requests are
                // NOT deferred: WATCH only needs the symbol cache,
                // which is independent of `owned_accessor`.
                let mut capture_requests_deferred: Vec<SnapshotRequest> = Vec::new();
                // Periodic-capture state. `periodic_boundaries_ns`
                // is the precomputed list of `Instant` deadlines
                // (encoded as nanos-since-`run_start`) at which the
                // run-loop fires `freeze_and_capture(false)`. Lazily
                // built on the first iteration AFTER BOTH:
                //   1. `KtstrVm::num_snapshots > 0` (periodic capture
                //      is requested), AND
                //   2. `workload_duration_for_coord` is `Some(d)`
                //      (the workload has a duration to slice), AND
                //   3. `scenario_start_ns_for_coord` reads non-zero
                //      (the first ScenarioStart frame has been
                //      observed and stamped by the dispatch arm).
                //
                // Boundaries divide the 10%–90% workload window into
                // `N + 1` equal intervals, producing `N` interior
                // boundaries — `N == 1` lands a single sample at
                // `start + 0.5 d` (midpoint); `N == 3` lands at
                // 0.3 d, 0.5 d, 0.7 d. The 10% pre-boundary buffer
                // and 10% post-boundary buffer give the workload
                // ramp-up / ramp-down room without periodic samples
                // landing on transient state.
                //
                // `next_periodic_idx` tracks how many boundaries
                // have already fired. When the gate
                // (`freeze_coord_on_demand_in_flight`) is held by a
                // concurrent on-demand or watchpoint capture, the
                // periodic boundary is deferred (NOT skipped) until
                // a subsequent iteration finds the gate clear — the
                // 10% buffer is the slack budget for this wait.
                let mut periodic_boundaries_ns: Option<Vec<u64>> = None;
                let mut next_periodic_idx: u32 = 0;
                // Consecutive parked-vCPU rendezvous failures during
                // periodic capture. Reset to 0 on every successful
                // `freeze_and_capture(..)`. After 2 consecutive
                // timeouts the run-loop abandons the remaining
                // periodic boundaries and logs once — repeated
                // 30 s rendezvous waits on a wedged guest would
                // otherwise eat the entire wall-clock budget without
                // producing useful captures, and a single abandoned
                // boundary keeps periodic noise off a guest the
                // operator already knows is degraded.
                let mut periodic_consecutive_timeouts: u32 = 0;
                let mut periodic_abandoned: bool = false;
                const PERIODIC_TIMEOUT_ABANDON_THRESHOLD: u32 = 2;
                // First iteration always runs scan-tick work so
                // boot-race lazy resolution attempts fire
                // immediately rather than waiting up to 100 ms for
                // the timerfd's first edge. Subsequent iterations
                // gate scan-tick on the SCANNER token (or on a
                // POLL_TIMEOUT-driven wake) — the watchpoint event
                // itself never sets scan_tick, which is correct:
                // that trigger is a fast path that should not block
                // the next wake on heavy bss-PA / scan_ctx work.
                let mut scan_tick: bool;
                let mut first_iter = true;
                let mut bsp_done_final_pass = false;
                'coord: while !freeze_coord_kill.load(Ordering::Acquire)
                    || (freeze_coord_bsp_done.load(Ordering::Acquire)
                        && !bsp_done_final_pass)
                {
                    if freeze_coord_bsp_done.load(Ordering::Acquire) {
                        if bsp_done_final_pass {
                            if capture_requests_deferred.is_empty()
                                || owned_accessor.is_some()
                            {
                                break 'coord;
                            }
                            eprintln!(
                                "freeze-coord: staying alive for deferred captures: deferred={} accessor={}",
                                capture_requests_deferred.len(),
                                owned_accessor.is_some()
                            );
                        }
                        bsp_done_final_pass = true;
                    }
                    // Unified event dispatch: epoll.wait on EVERY
                    // iteration. iter1 uses timeout=0 (non-blocking)
                    // so scan_tick fires immediately; subsequent
                    // iterations block up to POLL_TIMEOUT_MS. This
                    // ensures TOKEN_TX (KERN_ADDRS, SYS_RDY) is
                    // dispatched event-driven on every iteration —
                    // no manual drain calls needed.
                    let poll_ms = if first_iter { 0 } else { POLL_TIMEOUT_MS };
                    if first_iter {
                        scan_tick = true;
                        first_iter = false;
                    } else {
                        scan_tick = false;
                    }
                    if bsp_done_final_pass {
                        scan_tick = true;
                    }
                    {
                        let event_count = match epoll.wait(poll_ms, &mut events_buf) {
                            Ok(n) => n,
                            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                            Err(e) => {
                                tracing::error!(
                                    error = %e,
                                    "freeze-coord: epoll_wait failed; exiting coordinator"
                                );
                                break 'coord;
                            }
                        };
                        // Drain every fd that fired. Tokens map
                        // 1:1 to source fds; KILL / BSP_DONE both
                        // exit the loop, the others either set
                        // scan_tick (SCANNER) or surface state via
                        // the existing latch reads later in the
                        // body (WATCHPOINT).
                        for ev in &events_buf[..event_count] {
                            match ev.data() {
                                TOKEN_KILL | TOKEN_BSP_DONE => {
                                    let _ = freeze_coord_kill_evt.read();
                                    let _ = freeze_coord_bsp_done_evt.read();
                                }
                                TOKEN_SCANNER => {
                                    // Drain the timerfd's expiry
                                    // counter — re-arming is
                                    // automatic for periodic
                                    // timers, but the counter
                                    // accumulates and would re-
                                    // wake on the next epoll_wait
                                    // if not drained.
                                    let _ = scanner_tfd.wait();
                                    scan_tick = true;
                                }
                                TOKEN_WATCHPOINT => {
                                    // Drain the eventfd counter so
                                    // a subsequent epoll_wait
                                    // doesn't immediately re-fire
                                    // on the same edge. The
                                    // watchpoint.hit AtomicBool is
                                    // the source of truth — its
                                    // state survives the eventfd
                                    // drain and the late-trigger
                                    // detection later in the loop
                                    // re-loads it with Acquire.
                                    let _ = freeze_coord_hit_evt.read();
                                }
                                TOKEN_ACCESSOR_READY => {
                                    let _ = accessor_ready_evt.read();
                                    scan_tick = true;
                                }
                                TOKEN_TX => {
                                    // Drain the tx_evt counter so
                                    // a subsequent epoll_wait
                                    // doesn't immediately re-fire
                                    // on the same edge. The drain
                                    // below uses the device's TX
                                    // buffer (port1_tx_buf) as the
                                    // source of truth — bytes the
                                    // device accumulated since the
                                    // last wake are returned by
                                    // `drain_bulk` and threaded
                                    // through `bulk_assembler`. A
                                    // counter overflow under
                                    // EFD_NONBLOCK is benign
                                    // because the buffer state is
                                    // authoritative.
                                    //
                                    // Critical-section discipline:
                                    // `tx_evt.read()` is a syscall
                                    // and `bulk_assembler.feed()`
                                    // does TLV parsing (memcpy +
                                    // CRC + per-frame cap check).
                                    // Both are kept STRICTLY
                                    // outside the device mutex so
                                    // the vCPU thread emitting
                                    // bytes via virtio-console TX
                                    // never blocks behind the
                                    // coord. The explicit
                                    // `let bytes = { ... };`
                                    // block bounds the lock to the
                                    // single `drain_bulk` call —
                                    // a future refactor that
                                    // moves work into the block
                                    // is loud about the regression.
                                    let _ = freeze_coord_tx_evt.read();
                                    let bytes = {
                                        let mut g =
                                            freeze_coord_virtio_con.lock();
                                        g.drain_bulk()
                                    };
                                    let drained = bulk_assembler.feed(&bytes);
                                    // Per-frame typed dispatch.
                                    // Exhaustive `match
                                    // MsgType::from_wire(...)` so a
                                    // future MsgType variant addition
                                    // is a compile error here — the
                                    // arms call out exactly which
                                    // frames have coordinator-side
                                    // side effects (SchedExit / SysRdy
                                    // / SnapshotRequest), and every
                                    // other variant falls through to a
                                    // single "test-verdict-bearing"
                                    // arm whose only action is to
                                    // accumulate the entry into the
                                    // shared bucket. Reference VMMs
                                    // (libkrun, cloud-hypervisor, qemu)
                                    // all dispatch port-1 TX through a
                                    // single typed-tag matcher; the
                                    // prior if-ladder of `msg.msg_type
                                    // == MSG_TYPE_*` checks let a new
                                    // variant slip past the host
                                    // without an explicit decision.
                                    //
                                    // Every CRC-bearing arm gates on
                                    // `msg.crc_ok` so a torn frame
                                    // cannot promote into kill_evt /
                                    // sys_rdy_evt or trigger a
                                    // capture — same hostile-guest
                                    // discipline as the prior code.
                                    let mut bucket: Vec<crate::vmm::wire::ShmEntry> =
                                        Vec::new();
                                    let mut sinks = BulkDispatchSinks {
                                        kill: &freeze_coord_kill,
                                        kill_evt: &freeze_coord_kill_evt,
                                        sys_rdy_evt: &mut freeze_coord_sys_rdy_evt,
                                        snapshot_requests_pending:
                                            &mut snapshot_requests_pending,
                                        kern_phys_base: &kern_phys_base,
                                        kern_phys_base_evt: &kern_phys_base_evt,
                                        watchdog_reset: workload_duration_for_coord.map(|d| {
                                            (watchdog_reset_for_coord.as_ref(), d, run_start)
                                        }),
                                        watchdog_pause_ns: watchdog_pause_for_coord.as_ref(),
                                        scenario_start_ns: scenario_start_ns_for_coord.as_ref(),
                                        scenario_pause_cumulative_ns:
                                            scenario_pause_cumulative_for_coord.as_ref(),
                                        run_start,
                                    };
                                    for msg in &drained.messages {
                                        if let Some(entry) =
                                            dispatch_bulk_message(msg, &mut sinks)
                                        {
                                            bucket.push(entry);
                                        }
                                    }
                                    // Append the verdict-bearing entries
                                    // to the shared bucket so
                                    // `collect_results` can merge them
                                    // into the final `BulkDrainResult`.
                                    // Coordinator-internal control
                                    // frames are filtered inside
                                    // `dispatch_bulk_message` (the
                                    // SysRdy / SnapshotRequest arms
                                    // return None) — keying on
                                    // [`crate::vmm::wire::MsgType::is_coordinator_internal`]
                                    // keeps the filter set in lockstep
                                    // with `collect_results`'s post-run
                                    // drain. Without this stash, every
                                    // TLV frame the guest published
                                    // mid-run is silently dropped —
                                    // only late-arriving bytes that
                                    // landed in `port1_tx_buf` after
                                    // the coord stopped polling reach
                                    // the verdict.
                                    if !bucket.is_empty() {
                                        let mut buf = freeze_coord_bulk_messages_for_closure
                                            .lock()
                                            .unwrap_or_else(|e| e.into_inner());
                                        buf.extend(bucket);
                                    }
                                }
                                _ => {}
                            }
                        }
                        // No break on kill/bsp_done here — the
                        // iteration body must run so the
                        // late-trigger err_triggered check and
                        // freeze_and_capture can fire. The while
                        // condition + inner bsp_done check handle
                        // loop exit after the body completes.
                    }
                    // Adopt the worker-published accessor pair as
                    // soon as it lands. Both halves of the pair land
                    // atomically via `OnceLock::set` so a single
                    // `get()` returns either both or neither — no
                    // partial-Some shape to handle. The worker logs
                    // its own warn/eprintln on a permanent failure
                    // (60 s deadline exceeded), so the coordinator
                    // doesn't track separate retry counters here:
                    // a None pair after the worker has exited just
                    // means the dump path is unavailable for this
                    // run, which the existing call-site `is_some()`
                    // gates already handle gracefully.
                    if scan_tick && owned_accessor.is_none()
                        && let Some((map, prog)) = accessors_oncelock.get()
                    {
                        owned_accessor = Some(map);
                        if let Some(prog) = prog.as_ref() {
                            owned_prog_accessor = Some(prog);
                        }
                        {
                            let kernel = map.guest_kernel();
                            let pb = kernel.phys_base();
                            if pb != 0 {
                                if let Some(pb_kva) = dump_cpu_time_symbols
                                    .as_ref()
                                    .and_then(|s| s.phys_base_kva)
                                {
                                    let pb_pa = kernel.text_kva_to_pa(pb_kva);
                                    if let Some(ref mem) = freeze_coord_mem {
                                        let real_pb = mem.read_u64(pb_pa, 0);
                                        coord_kaslr_offset = pb.wrapping_sub(real_pb);
                                    }
                                }
                            }
                        }
                        // Drain CAPTURE requests deferred during the
                        // pre-adoption window. Append onto
                        // `snapshot_requests_pending` so the existing
                        // CAPTURE drain (further down this iteration
                        // body) dispatches them through the normal
                        // `freeze_and_capture(false)` flow with the
                        // accessor present — no flow duplication.
                        if !capture_requests_deferred.is_empty() {
                            let n = capture_requests_deferred.len();
                            tracing::info!(
                                deferred_count = n,
                                "freeze-coord: draining deferred CAPTURE \
                                 requests after owned_accessor adoption"
                            );
                            snapshot_requests_pending.append(
                                &mut capture_requests_deferred,
                            );
                        }
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
                    //
                    // The `__per_cpu_offset` KVA is sourced from the
                    // already-cached `dump_cpu_time_symbols` —
                    // re-parsing vmlinux every scan tick (~100 ms)
                    // while waiting for the per-CPU areas to come up
                    // would re-read 50 MB+ of ELF and rebuild the
                    // symbol table on every iteration. The KVA is
                    // fixed at kernel link time so a single resolution
                    // suffices for the rest of the run; if
                    // `dump_cpu_time_symbols` is None (vmlinux
                    // unparseable at coord start) or its
                    // `per_cpu_offset` is 0 (symbol stripped),
                    // `try_init_prog_per_cpu_offsets` returns None
                    // and the cache stays unset — same behaviour as
                    // the prior in-helper parse path.
                    if scan_tick
                        && prog_per_cpu_offsets.is_none()
                        && let Some(mem) = freeze_coord_mem.as_deref()
                    {
                        let per_cpu_offset_kva = dump_cpu_time_symbols
                            .as_ref()
                            .map(|s| s.per_cpu_offset)
                            .unwrap_or(0);
                        if per_cpu_offset_kva == 0 {
                            // Permanent failure: the symbol is absent
                            // from `dump_cpu_time_symbols` (vmlinux
                            // unparseable at coord start, or
                            // `__per_cpu_offset` stripped from the
                            // image). Warn immediately on the first
                            // observation — no amount of retrying
                            // will materialise a missing symbol — and
                            // latch via `per_cpu_offsets_kva_warned`
                            // so the warn fires at most once per VM
                            // run. The `prog_per_cpu_offsets` cache
                            // stays None and downstream
                            // prog_runtime_stats capture is
                            // permanently degraded for this run.
                            if !per_cpu_offsets_kva_warned {
                                tracing::warn!(
                                    "freeze-coord: __per_cpu_offset symbol absent from \
                                     dump_cpu_time_symbols (vmlinux unparseable at coord \
                                     start, or symbol stripped) — prog_runtime_stats \
                                     capture is permanently degraded for this run; \
                                     will not retry"
                                );
                                per_cpu_offsets_kva_warned = true;
                            }
                        } else {
                            let phys_base = owned_accessor
                                .as_ref()
                                .map(|a| a.guest_kernel().phys_base())
                                .unwrap_or(0);
                            prog_per_cpu_offsets = try_init_prog_per_cpu_offsets(
                                mem,
                                per_cpu_offset_kva,
                                freeze_coord_tcr_el1.as_ref(),
                                phys_base,
                                freeze_coord_num_cpus,
                            );
                            if prog_per_cpu_offsets.is_none() {
                                // Transient boot-progress condition:
                                // the symbol is present (kva != 0)
                                // but at least one CPU's offset slot
                                // is still zero. The guest's
                                // `setup_per_cpu_areas` populates
                                // every slot before SMP bringup, so
                                // a non-zero retry count after
                                // `LAZY_ACCESSOR_WARN_AFTER_ITERS`
                                // iterations indicates the guest
                                // genuinely failed to bring up its
                                // per-CPU areas (or
                                // `freeze_coord_num_cpus` exceeds
                                // the configured `nr_cpu_ids` so
                                // slots beyond the live count
                                // legitimately read 0).
                                per_cpu_offsets_retries += 1;
                                if !per_cpu_offsets_warned
                                    && per_cpu_offsets_retries
                                        >= LAZY_ACCESSOR_WARN_AFTER_ITERS
                                {
                                    tracing::warn!(
                                        retries = per_cpu_offsets_retries,
                                        num_cpus = freeze_coord_num_cpus,
                                        "freeze-coord: __per_cpu_offset array still has \
                                         zero slots after retries — most commonly a \
                                         still-booting guest (per-CPU areas not yet \
                                         allocated); a permanent failure (num_cpus \
                                         exceeds nr_cpu_ids, partial SMP bringup) \
                                         leaves prog_runtime_stats degraded. Will \
                                         continue retrying."
                                    );
                                    per_cpu_offsets_warned = true;
                                }
                            }
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
                    //
                    // Invalidation pass first: a previously-cached
                    // PA is only as valid as the underlying map. If
                    // the probe BPF program unloads (test teardown,
                    // userspace explicit unload, parent process
                    // panicking before Drop) the kernel frees the
                    // bpf_array vmalloc page; any subsequent
                    // `read_u32(cached_bss_pa, 0)` reads whatever
                    // the page allocator hands out next — typically
                    // non-zero for slab pages reused by an unrelated
                    // subsystem. The result latches a phantom
                    // `err_triggered` and synthesizes a bogus
                    // failure dump on a healthy run. Re-walk
                    // `map_idr` and require the same-named map's
                    // `value_kva` to match the one we resolved
                    // against; on mismatch (map gone OR rebound to
                    // a fresh slab) clear the PA + companion
                    // value_kva cache so the discovery block below
                    // re-resolves from scratch. The walk uses a
                    // fresh `as_accessor()` instance — its
                    // `maps_cache` re-fills from a current map_idr
                    // traversal, so a stale entry from a prior dump
                    // cannot keep an unloaded map visible.
                    if scan_tick
                        && cached_bss_pa.is_some()
                        && let Some(owned) = owned_accessor
                    {
                        let accessor = owned.as_accessor();
                        let still_valid = match accessor.find_map("probe_bp.bss") {
                            Some(m) => m.value_kva == cached_bss_value_kva,
                            None => false,
                        };
                        if !still_valid {
                            tracing::warn!(
                                stale_value_kva = cached_bss_value_kva
                                    .map(|k| format!("{k:#x}"))
                                    .unwrap_or_else(|| "None".to_string()),
                                "freeze-coord: probe_bp.bss map gone or \
                                 rebound — invalidating cached_bss_pa to \
                                 prevent reads of a freed vmalloc page \
                                 (probe unload mid-run)"
                            );
                            cached_bss_pa = None;
                            cached_bss_value_kva = None;
                            // bss_field_offset is BTF-derived from
                            // probe.bpf.c globals; the layout
                            // cannot change across an unload+reload
                            // of the same probe object so the
                            // offset cache stays valid. Re-resolving
                            // it would re-pay the BTF parse for no
                            // semantic gain.
                        }
                    }
                    if scan_tick
                        && cached_bss_pa.is_none()
                        && let Some(owned) = owned_accessor
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
                            // Bind kernel once and reuse — pre-fix
                            // owned.guest_kernel() ran three times here
                            // and once again at the BTF Datasec walk
                            // below. The accessor is cheap but the
                            // repetition was noisy at the freeze hot
                            // path's read site.
                            let kernel = owned.guest_kernel();
                            let walk = kernel.walk_context();
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
                                    kernel,
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
                            // Bound the BTF-derived offset against
                            // the map's declared `value_size`. The
                            // probe's BTF Datasec walk parses
                            // guest-supplied bytes — a corrupted
                            // (or hostile) BTF can return a u32
                            // offset that extends past the ARRAY's
                            // flex-array storage, so the
                            // `wrapping_add(bss_field_offset)`
                            // below would wrap into an unrelated
                            // guest page. Reading the resulting PA
                            // latches a phantom `err_triggered`
                            // and synthesizes a bogus failure
                            // dump. Reject any offset whose 4-byte
                            // read would walk past the map's
                            // value bytes; treat the failure
                            // exactly like a broken BTF walk —
                            // warn once via the existing latch and
                            // fall back to offset 0 for this and
                            // every subsequent iteration so the
                            // detection survives in degraded form
                            // instead of going silent. Saturating
                            // subtract guards `value_size < 4`
                            // (the map could not legitimately
                            // hold a u32 in that case, so
                            // `bss_field_offset > 0` rejects every
                            // non-zero offset, matching the
                            // "value_size too small" intent
                            // without a separate branch).
                            let max_offset = map.value_size.saturating_sub(4);
                            let bss_field_offset = if bss_field_offset > max_offset {
                                if !bss_offset_warn_logged {
                                    tracing::warn!(
                                        bss_field_offset,
                                        value_size = map.value_size,
                                        "freeze-coord: BTF-resolved bss field \
                                         offset exceeds value_size - 4 — \
                                         refusing to cache PA that would \
                                         read past the .bss flex array; \
                                         falling back to offset 0"
                                    );
                                    bss_offset_warn_logged = true;
                                }
                                cached_bss_offset = Some(0);
                                0
                            } else {
                                bss_field_offset
                            };
                            if let Some(translated) = crate::monitor::idr::translate_any_kva(
                                mem,
                                walk.cr3_pa,
                                walk.page_offset,
                                value_kva,
                                walk.l5,
                                walk.tcr_el1,
                            ) {
                                cached_bss_pa =
                                    Some(translated.wrapping_add(bss_field_offset as u64));
                                cached_bss_value_kva = Some(value_kva);
                            }
                        }
                    }
                    // Resolve the watchpoint target KVA
                    // (`*scx_root + exit_kind_offset`) and (re-)
                    // publish it whenever `*scx_root` changes. Runs
                    // every scan tick — the `last_sched_kva == new`
                    // fast path keeps the steady-state cost a single
                    // u64 read.
                    //
                    // Resolution requires:
                    //   - dump_cpu_time_symbols (KernelSymbols) for
                    //     `scx_root` symbol KVA — present whenever
                    //     vmlinux parsed at coord-start;
                    //   - dump_scx_walker_offsets.sched.exit_kind for
                    //     the field offset within `struct scx_sched`
                    //     — present whenever BTF carries the type;
                    //   - owned_accessor's GuestKernel for cr3_pa /
                    //     page_offset / l5 — needed for the same
                    //     direct-mapping translation `cached_bss_pa`
                    //     uses.
                    //
                    // The BPF .bss fallback below continues to update
                    // `cached_bss_pa`; both signals can fire and the
                    // late-trigger arm (a few iterations down the
                    // loop) treats either as ground truth. The
                    // watchpoint's advantages are synchronous
                    // delivery (no 100 ms polling window) AND
                    // independence from the probe BPF program loading
                    // correctly.
                    if scan_tick
                        && owned_accessor.is_some()
                        && let Some(ref syms) = dump_cpu_time_symbols
                        && let Some(scx_root_kva) = syms.scx_root
                        && let Some(ref scx_offsets) = dump_scx_walker_offsets
                        && let Some(ref sched_offs) = scx_offsets.sched
                        && let Some(ref mem) = freeze_coord_mem
                    {
                        // scx_root is a kernel-text-mapped pointer.
                        // The owned_accessor's GuestKernel carries the
                        // VA-bits-aware kernel image base resolved from
                        // TCR_EL1 (mirrors `read_scx_sched_state` in
                        // `monitor/scx_walker.rs`).
                        let kernel_for_root = owned_accessor
                            .as_ref()
                            .expect("owned_accessor.is_some() gate above")
                            .guest_kernel();
                        let root_pa = kernel_for_root.text_kva_to_pa(scx_root_kva);
                        let sched_kva = mem.read_u64(root_pa, 0);
                        // Drive the watchpoint state machine via the
                        // pure helper so unit tests can exercise the
                        // full `(last_sched_kva, sched_kva)` transition
                        // matrix (Unchanged / Detached / RebindDisarmed
                        // / Published / PublishDeferred) without
                        // booting a VM. The helper performs all
                        // ordered atomic stores per the contract on
                        // [`super::vcpu::WatchpointArm`]; the caller
                        // owns `last_sched_kva` and the result-driven
                        // logging.
                        match republish_watchpoint_on_rebind(
                            sched_kva,
                            last_sched_kva,
                            sched_offs.exit_kind as u32,
                            &freeze_coord_watchpoint,
                            kernel_for_root,
                            mem,
                        ) {
                            WatchpointPublishResult::Unchanged => {}
                            WatchpointPublishResult::Detached => {
                                tracing::info!(
                                    "freeze-coord: scx_root cleared (scheduler \
                                     detached); watchpoint disarmed pending next \
                                     attach"
                                );
                                last_sched_kva = 0;
                            }
                            WatchpointPublishResult::RebindDisarmed {
                                previous,
                                next,
                            } => {
                                tracing::info!(
                                    last_sched_kva = format_args!("{:#x}", previous),
                                    new_sched_kva = format_args!("{:#x}", next),
                                    "freeze-coord: scx_root rebind detected \
                                     (A → B); watchpoint disarmed this tick, \
                                     B will be republished next tick after \
                                     vCPUs clear DR0"
                                );
                                last_sched_kva = 0;
                            }
                            WatchpointPublishResult::Published {
                                exit_kind_kva,
                                kind_pa,
                            } => {
                                last_sched_kva = sched_kva;
                                cached_exit_kind_pa = Some(kind_pa);
                                tracing::info!(
                                    exit_kind_kva =
                                        format_args!("{:#x}", exit_kind_kva),
                                    sched_kva = format_args!("{:#x}", sched_kva),
                                    kind_pa = format_args!("{:#x}", kind_pa),
                                    "freeze-coord: watchpoint target \
                                     published; vCPU threads will self-arm \
                                     KVM_SET_GUEST_DEBUG on next iteration"
                                );
                            }
                            WatchpointPublishResult::PublishDeferred {
                                exit_kind_kva,
                            } => {
                                tracing::debug!(
                                    exit_kind_kva =
                                        format_args!("{:#x}", exit_kind_kva),
                                    "freeze-coord: exit_kind translate or \
                                     host-ptr lookup failed; deferring \
                                     watchpoint publish"
                                );
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
                    if scan_tick && freeze_coord_dual_snapshot && scan_ctx.is_none() {
                        // try_resolve consumes the hoisted prereqs
                        // (scan_offsets, scan_jiffies_64_kva,
                        // dump_cpu_time_symbols).
                        // The only field that can flip Some after
                        // coord-start is owned_accessor (boot-race);
                        // every other input is a deterministic function
                        // of the host inputs and was already attempted
                        // at coord-start. A None among them means the
                        // dependency is permanently absent — the
                        // diagnostic warn already names which leg
                        // failed. The closure returns the reason
                        // string alongside None so the late-trigger
                        // skip-reason path can quote it directly into
                        // DualFailureDumpReport::early_skipped_reason.
                        let try_resolve = || -> Result<RunnableScanCtx, &'static str> {
                            let owned = owned_accessor
                                .as_ref()
                                .ok_or("owned_accessor not ready (guest still booting)")?;
                            let scan_offsets = scan_offsets
                                .ok_or("RunnableScanOffsets unavailable (BTF lacks sched_ext_entity)")?;
                            let jiffies_64_kva = scan_jiffies_64_kva
                                .ok_or("jiffies_64 symbol absent from vmlinux")?;
                            let syms = dump_cpu_time_symbols
                                .as_ref()
                                .ok_or("KernelSymbols unavailable (vmlinux parse failed)")?;
                            // The global `scx_tasks` LIST_HEAD is the
                            // walker's only memory anchor. Absent on a
                            // stripped vmlinux or a kernel without
                            // sched_ext — fail the resolve so the
                            // late-trigger skip-reason path quotes the
                            // missing symbol.
                            let scx_tasks_kva = syms.scx_tasks.ok_or(
                                "scx_tasks symbol absent from vmlinux \
                                 (kernel without sched_ext or stripped vmlinux)",
                            )?;
                            let mem = freeze_coord_mem
                                .as_ref()
                                .ok_or("GuestMem unavailable")?;
                            let kernel = owned.guest_kernel();
                            let walk = kernel.walk_context();
                            // Translate jiffies_64's KVA to a PA.
                            // Lives in the kernel text/data mapping —
                            // same as scx_root et al. Use the
                            // GuestKernel-resident base so VA_BITS=47
                            // hosts translate correctly.
                            let jiffies_64_pa = kernel.text_kva_to_pa(jiffies_64_kva);
                            // Compute per-CPU rq PAs for the per-rq
                            // runnable_list walker. The KernelOffsets
                            // schema guarantees `runqueues != 0` (see
                            // `monitor/symbols.rs` — its absence is a
                            // construction-time error), so the only
                            // failure path here is reading
                            // `__per_cpu_offset` early during boot:
                            // the per-CPU offset table reads as zero
                            // for not-yet-online CPUs. A zero offset
                            // does NOT yield a zero PA — `compute_rq_pas`
                            // wraps via `wrapping_sub` into the
                            // upper-half KVA region (see
                            // `compute_rq_pas` doc comment in
                            // `monitor/symbols.rs`), so the resulting
                            // PA is bogus, not zero, and there is no
                            // downstream `rq_pa == 0` short-circuit
                            // to suppress it. Caching such a vec is
                            // permanent for the run and would have
                            // every subsequent walk read garbage for
                            // the not-yet-online slots. Mirror the
                            // `prog_per_cpu_offsets` gate above:
                            // defer scan_ctx construction until every
                            // offset slot is non-zero. A retry is
                            // cheap; a cached miss is permanent.
                            let pco_pa = kernel.text_kva_to_pa(syms.per_cpu_offset);
                            let pco_offsets = crate::monitor::symbols::read_per_cpu_offsets(
                                mem,
                                pco_pa,
                                freeze_coord_num_cpus,
                            );
                            if pco_offsets.contains(&0) {
                                return Err(
                                    "not all per_cpu_offsets resolved \
                                     (some CPUs still booting)",
                                );
                            }
                            // The coordinator's scan_ctx path uses
                            // the accessor's own phys_base (from
                            // page-table walk). TODO: derive
                            // kaslr_offset for the coordinator too.
                            let rq_pas = crate::monitor::symbols::compute_rq_pas(
                                syms.runqueues,
                                &pco_offsets,
                                walk.page_offset,
                                syms.per_cpu_start,
                                0,
                            );
                            // scx_watchdog_timestamp is a `.data`
                            // file-scope static — same text-mapping
                            // translation as scx_watchdog_timeout
                            // (which lives a few lines below the
                            // timestamp in kernel/sched/ext.c).
                            // Optional because the symbol is absent
                            // on kernels without sched_ext or
                            // stripped vmlinux; max_runnable_age
                            // skips the contribution when None.
                            let watchdog_timestamp_pa =
                                syms.scx_watchdog_timestamp.map(|kva| kernel.text_kva_to_pa(kva));
                            Ok(RunnableScanCtx {
                                scx_tasks_kva,
                                rq_pas,
                                offsets: scan_offsets,
                                jiffies_64_pa,
                                watchdog_timestamp_pa,
                                walk,
                                start_kernel_map: kernel.start_kernel_map(),
                                phys_base: kernel.phys_base(),
                            })
                        };
                        match try_resolve() {
                            Ok(ctx) => {
                                scan_ctx = Some(ctx);
                                scan_ctx_skip_reason = None;
                            }
                            Err(reason) => {
                                scan_ctx_skip_reason = Some(reason);
                            }
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
                    if scan_tick && freeze_coord_dual_snapshot && scan_ctx.is_none() {
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
                    // Poll for the late-trigger condition. The
                    // hardware watchpoint on `*scx_root->exit_kind`
                    // is the primary path: every vCPU thread sets
                    // `freeze_coord_watchpoint.hit` (Release) on
                    // `KVM_EXIT_DEBUG`, which the Acquire load here
                    // observes synchronously — no 100 ms polling
                    // window. The BPF .bss `cached_bss_pa` read
                    // (gated through [`bss_read_state`] for
                    // PA-validity vs not-fired distinction) is
                    // checked alongside the watchpoint every
                    // iteration: it remains a useful redundancy on
                    // kernels where the watchpoint armed (the
                    // typed three-way result also catches a stale
                    // cached PA that bare `read_u32` would mask as
                    // "no fire") AND a fallback for kernels where
                    // the watchpoint never armed (no `scx_root`
                    // symbol, BTF stripped of `scx_sched`, or
                    // `KVM_SET_GUEST_DEBUG` rejected by the host).
                    //
                    // Once `freeze_state == Done` the late-trigger
                    // dispatch has already taken its terminal
                    // transition — re-evaluating
                    // `compute_err_triggered(...)` is wasted work
                    // for the rest of the run (sticky bss latch
                    // keeps reporting Triggered, sticky watchpoint
                    // hit keeps reporting true). Skip the read
                    // entirely once the state machine has closed.
                    let (watchpoint_hit, bss_state) =
                        if freeze_state == FreezeState::Done {
                            (false, BssReadState::NotResolved)
                        } else {
                            let wp =
                                freeze_coord_watchpoint.hit.load(Ordering::Acquire);
                            let st = bss_read_state(
                                freeze_coord_mem.as_deref(),
                                cached_bss_pa,
                            );
                            // OnlyTriggered counts as "fire";
                            // OutOfBounds and NotResolved /
                            // NotTriggered all mean "no
                            // observable fire this iteration".
                            // Surfacing OOB once with a warn lets
                            // an operator notice when the .bss
                            // path has gone stale without
                            // changing the trigger arithmetic.
                            if matches!(st, BssReadState::OutOfBounds)
                                && !bss_oob_warn_logged
                            {
                                tracing::warn!(
                                    cached_bss_pa =
                                        cached_bss_pa
                                            .map(|p| format!("{p:#x}"))
                                            .unwrap_or_else(|| "None".to_string()),
                                    "freeze-coord: cached BPF .bss PA no \
                                     longer resolves to a 4-byte-readable \
                                     DRAM region — probe map likely freed \
                                     mid-run; .bss late-trigger fallback is \
                                     now silent for the rest of the run \
                                     (watchpoint path, if armed, remains \
                                     active)"
                                );
                                bss_oob_warn_logged = true;
                            }
                            (wp, st)
                        };
                    let mut err_triggered =
                        compute_err_triggered(watchpoint_hit, bss_state);
                    if !err_triggered
                        && scan_tick
                        && freeze_state != FreezeState::Done
                        && let Some(ek_pa) = cached_exit_kind_pa
                        && let Some(ref mem) = freeze_coord_mem
                    {
                        let kind = mem.read_u32(ek_pa, 0);
                        const SCX_EXIT_ERROR: u32 = 1024;
                        if kind >= SCX_EXIT_ERROR {
                            err_triggered = true;
                        }
                    }
                    if !err_triggered
                        && bsp_done_final_pass
                        && freeze_state != FreezeState::Done
                    {
                        if let (Some(owned), Some(mem)) =
                            (owned_accessor.as_ref(), freeze_coord_mem.as_deref())
                        {
                            let kernel = owned.guest_kernel();
                            let walk = kernel.walk_context();
                            let mut ek_kva = freeze_coord_watchpoint
                                .request_kva
                                .load(Ordering::Acquire);
                            if ek_kva == 0 {
                                if let Some(syms) = dump_cpu_time_symbols.as_ref()
                                    && let Some(root_kva) = syms.scx_root
                                    && let Some(ref offs) = dump_scx_walker_offsets
                                    && let Some(ref so) = offs.sched
                                {
                                    let root_pa = kernel.text_kva_to_pa(root_kva);
                                    let sched_kva = mem.read_u64(root_pa, 0);
                                    if sched_kva != 0 {
                                        ek_kva = sched_kva + so.exit_kind as u64;
                                    }
                                }
                            }
                            if ek_kva != 0 {
                                if let Some(pa) = crate::monitor::idr::translate_any_kva(
                                    mem,
                                    walk.cr3_pa,
                                    walk.page_offset,
                                    ek_kva,
                                    walk.l5,
                                    walk.tcr_el1,
                                ) {
                                    let kind = mem.read_u32(pa, 0);
                                    const SCX_EXIT_ERROR: u32 = 1024;
                                    if kind >= SCX_EXIT_ERROR {
                                        err_triggered = true;
                                    }
                                }
                            }
                        }
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
                    // `gate_on_exit_kind` filters out spurious watchpoint
                    // fires on a non-error `exit_kind` value. The
                    // hardware watchpoint catches every write to
                    // `*scx_root->exit_kind` regardless of value —
                    // including transient writes during init/teardown
                    // that the kernel sets to `SCX_EXIT_NONE` (0) or
                    // `SCX_EXIT_DONE` (1). Without the gate, every
                    // clean scheduler shutdown would synthesize a
                    // bogus failure dump. The gate runs AFTER the
                    // rendezvous succeeds (vCPUs parked → guest
                    // memory consistent) and BEFORE building the
                    // dump: read the 4-byte `exit_kind` value at the
                    // already-resolved KVA, compare against the
                    // error-class boundary `SCX_EXIT_ERROR = 1024`
                    // (per `kernel/sched/ext_internal.h::scx_exit_kind`).
                    // Gate failures return None — the late-trigger
                    // call site treats this as "spurious watchpoint
                    // fire, reset hit and keep watching" rather than
                    // the normal "rendezvous timed out, give up"
                    // semantics. The early (runnable_at) trigger and
                    // BPF-bss late trigger pass `false`: those paths
                    // are already gated on their own conditions
                    // (half-way age threshold; tp_btf handler latch
                    // on error-class kinds), so an extra exit_kind
                    // read would be redundant overhead.
                    let freeze_and_capture =
                        |gate_on_exit_kind: bool|
                            -> Option<(crate::monitor::dump::FailureDumpReport, Instant)> {
                            let skip_freeze =
                                freeze_coord_bsp_done.load(Ordering::Acquire);
                            if skip_freeze {
                                tracing::info!(
                                    gate_on_exit_kind,
                                    "freeze-coord: BSP exited, capturing \
                                     quiesced guest memory without freeze"
                                );
                            } else {
                                tracing::info!(
                                    gate_on_exit_kind,
                                    "freeze-coord: freezing vCPUs for snapshot"
                                );
                            }
                            // Capture wall-clock start for the
                            // post-dump timing summary one-liner.
                            // Returned alongside the report so the
                            // call site can reuse it across the
                            // post-thaw JSON emit (covers freeze
                            // rendezvous → dump_state → numa-stats →
                            // serialise → file-write window with one
                            // anchor).
                            let capture_start = Instant::now();
                            // Soft deadline for the whole capture path
                            // (rendezvous + dump_state + numa stats).
                            // Set to half the configured watchdog so a
                            // slow dump can't keep vCPUs parked past
                            // the kernel's own SCX_EXIT_ERROR_STALL
                            // emission line. `freeze_coord_watchdog_half`
                            // already encodes the divide-by-2 (see
                            // its definition above) and falls back to
                            // 2s when the builder didn't set
                            // watchdog_timeout. Using it here couples
                            // the dump bailout to the same horizon
                            // the per-CPU runnable_at scanner uses
                            // for the dual-snapshot half-way trigger.
                            let capture_deadline = if freeze_coord_watchdog_half
                                > Duration::ZERO
                            {
                                Some(capture_start + freeze_coord_watchdog_half)
                            } else {
                                None
                            };
                            // 'capture labeled block: every exit
                            // from the freeze→park→dump phases
                            // (rendezvous timeout, gate-suppressed,
                            // full dump, partial dump) `break
                            // 'capture <result>` so all paths
                            // converge on the labeled block's value
                            // — which is the closure's return.
                            // The caller is responsible for invoking
                            // `thaw_and_barrier` AFTER it has done
                            // any while-frozen work it needs (the
                            // late-trigger backstop reads guest
                            // memory while quiesced, so the thaw
                            // cannot be unconditional inside the
                            // closure).
                            'capture: {
                            // Cycle-entry snapshot of BSP liveness
                            // used for non-UAF-sensitive bookkeeping:
                            // parked_evt pre-seed gating
                            // (`bsp_parked` lookup), `expected_parks`
                            // accounting (+1 for BSP), pass-2
                            // `pthread_kill`, and the rendezvous-wait
                            // diagnostics. None of those callsites
                            // dereference the BSP's `kvm_run` mmap, so
                            // a stale `true` is benign:
                            // `pthread_kill` against an exited tid
                            // returns ESRCH, an over-counted
                            // `expected_parks` heals on the next
                            // SIGRTMIN/park-ack overshoot path, and
                            // pre-seed reads only the AtomicBool
                            // `bsp_parked` flag.
                            //
                            // The TOCTOU-sensitive
                            // `ImmediateExitHandle::set(1)` against
                            // the BSP's `kvm_run` mmap is gated by
                            // its own fresh Acquire load further
                            // below — see the "Re-load `bsp_alive`
                            // immediately before the BSP `ie.set()`"
                            // comment for the full rationale (a stale
                            // snapshot there would write through a
                            // pointer into freed `kvm_run` pages
                            // after the BSP drops its `VcpuFd`).
                            //
                            // The primary line of defense remains
                            // `freeze_coord_handle.join()` in run_vm
                            // BEFORE the BSP `VcpuFd` falls out of
                            // scope; the in-closure loads are
                            // defense-in-depth.
                            let bsp_alive_at_start =
                                bsp_alive_for_coord.load(Ordering::Acquire);
                            // Drain `parked_evt` BEFORE flipping
                            // `freeze=true` and BEFORE issuing
                            // pass-0 (worker pause), pass-1
                            // (immediate_exit), or pass-2 (SIGRTMIN).
                            // From this point forward every
                            // increment to the parked_evt counter
                            // is unambiguously a park-ack for THIS
                            // cycle. Draining AFTER the kicks is a
                            // race: a fast vCPU or worker may park
                            // and bump the counter between the kick
                            // and the drain — that ack is then
                            // absorbed by the drain instead of
                            // counted toward `parked_count`, and
                            // the rendezvous waits 30 s for an ack
                            // that already fired.
                            //
                            // EAGAIN under EFD_NONBLOCK (counter
                            // already 0 from the prior cycle's
                            // post-thaw barrier drain) is benign.
                            //
                            // The Acquire ordering synchronizes-with
                            // the parker's Release store after its
                            // drain dance — this rendezvous IS the
                            // memory barrier that makes the future
                            // host-side guest-memory reads correct.
                            // The eventfd write ordering is
                            // load-bearing: the AtomicBool Release
                            // happens-before the eventfd write, so
                            // every counter increment we observe in
                            // the loop below implies every
                            // guest-side queue mutation the parker
                            // performed pre-park is visible to the
                            // dump.
                            //
                            // Also drain `thaw_evt` here. The
                            // coordinator writes thaw_evt ONCE per
                            // thaw, and every parked vCPU polls the
                            // SAME fd in `handle_freeze` without
                            // draining (the multi-reader fan-out
                            // wake design pinned at
                            // `vmm/exit_dispatch.rs::handle_freeze`).
                            // Without a per-cycle drain by the
                            // coordinator the counter is monotonic
                            // — every successive freeze cycle sees
                            // a level-high thaw_evt left over from
                            // the previous cycle, which makes
                            // `handle_freeze`'s poll return
                            // immediately on every iteration and
                            // burns CPU spinning on
                            // `freeze.load(Acquire)` until the
                            // coordinator clears `freeze`. Draining
                            // before pass-1 / pass-2 means the next
                            // poll inside `handle_freeze` blocks on
                            // an empty counter and wakes only when
                            // (a) the coordinator's post-rendezvous
                            // `thaw_evt.write(1)` lands or (b) the
                            // 100 ms poll backstop fires.
                            use std::os::fd::AsRawFd;
                            let _ = freeze_coord_parked_evt.read();
                            let _ = freeze_coord_thaw_evt.read();
                            // Snapshot virtio-blk worker liveness
                            // BEFORE pause(). When the device exists
                            // but the worker thread is not yet
                            // spawned (pre-DRIVER_OK) or has been
                            // joined (post-stop / failed-respawn),
                            // pause() short-circuits with the
                            // "no-live-worker" fast path and writes
                            // no parked_evt ack — counting +1 in
                            // that case makes the rendezvous wait
                            // 30 s for a worker that does not
                            // exist. The pre-pause `paused` flag
                            // is the cheapest available proxy: the
                            // worker spawn flips it to false on
                            // entry to the run loop and the
                            // post-thaw barrier guarantees a live
                            // worker has cleared it before this
                            // cycle starts. `paused == true` at
                            // cycle entry therefore means "no live
                            // worker" (the construction sentinel
                            // or post-stop re-armed sentinel from
                            // `resume()`). Gate the +1 below on
                            // `worker_was_running` instead of bare
                            // `is_some()`.
                            let worker_was_running = freeze_coord_virtio_blk_paused
                                .as_ref()
                                .is_some_and(|p| !p.load(Ordering::Acquire));
                            // Pre-seed the parked_evt counter for any
                            // parker whose flag is STILL `true` at
                            // cycle entry. The post-thaw barrier at
                            // the end of every prior cycle SHOULD
                            // have observed every parker clear its
                            // flag before returning, but the barrier
                            // can hit its FREEZE_RENDEZVOUS_TIMEOUT
                            // and break early (logged as
                            // "post-thaw barrier timed out — a parker
                            // did not clear within
                            // FREEZE_RENDEZVOUS_TIMEOUT" further
                            // below). When that happens the parker is
                            // still inside `handle_freeze`'s park
                            // loop with `parked=true`. The next cycle
                            // sets `freeze=true` and SIGRTMINs every
                            // vCPU; the kicked vCPU's poll wakes
                            // (EINTR), re-checks `freeze` (now true),
                            // and stays in the SAME `handle_freeze`
                            // invocation — `parked.store(true)` plus
                            // `parked_evt.write(1)` only run on
                            // ENTRY to `handle_freeze`
                            // (exit_dispatch.rs:1051 / :1067), NOT
                            // per `freeze=true` flip while parked.
                            // Without pre-seeding, the rendezvous
                            // countdown latch never receives an ack
                            // from that parker and waits the full
                            // 30 s for an event that already happened
                            // a cycle ago.
                            //
                            // Pre-seeding +1 to `parked_evt` per
                            // still-parked parker compensates: the
                            // rendezvous loop drains the counter
                            // and credits each as a park-ack for
                            // THIS cycle. This is equivalent to the
                            // historical force-clear of `parked`
                            // flags but targeted — only fires for
                            // the timed-out subset, leaving a
                            // healthy mid-thaw parker (which still
                            // has its `parked=true` from the prior
                            // cycle and is about to clear it within
                            // a few ms) untouched. The worst case
                            // is a healthy parker that races the
                            // pre-seed: its own
                            // `parked_evt.write(1)` on the next
                            // entry to `handle_freeze` adds another
                            // count, which is harmless — the
                            // rendezvous loop only checks
                            // `parked_count >= expected_parks` and
                            // overshoot is fine.
                            //
                            // The bsp/ap loads here are Acquire to
                            // synchronise-with the prior cycle's
                            // post-thaw barrier reads — the post-
                            // thaw barrier already loaded these
                            // with Acquire, but a healthy parker may
                            // have flipped its flag back to false
                            // between the barrier's last load and
                            // this point. The seed only fires when
                            // we still observe `true`, so a healthy
                            // late-clear is a no-op.
                            let mut still_parked: u32 = 0;
                            for ap in freeze_coord_ap_parked.iter() {
                                if ap.load(Ordering::Acquire) {
                                    still_parked = still_parked.saturating_add(1);
                                }
                            }
                            if bsp_alive_at_start
                                && freeze_coord_bsp_parked.load(Ordering::Acquire)
                            {
                                still_parked = still_parked.saturating_add(1);
                            }
                            // virtio-blk worker: only pre-seed when
                            // the worker was running (otherwise
                            // pause() short-circuits and we won't
                            // count +1 anyway). If the worker is
                            // running AND `paused == true`, the
                            // worker is mid-park from the prior
                            // cycle and won't re-write its ack on
                            // the next pause()-driven epoll wake,
                            // mirroring the vCPU case. Pause-fd
                            // writes happen-before `paused.store
                            // (true)`, which happens-before the
                            // worker's `parked_evt.write(1)` —
                            // matching the vCPU sequence in
                            // `handle_freeze`.
                            if worker_was_running
                                && freeze_coord_virtio_blk_paused
                                    .as_ref()
                                    .is_some_and(|p| p.load(Ordering::Acquire))
                            {
                                still_parked = still_parked.saturating_add(1);
                            }
                            if still_parked > 0 {
                                tracing::warn!(
                                    still_parked,
                                    "freeze-coord: detected stale parked=true \
                                     parker(s) at cycle entry — prior post-thaw \
                                     barrier likely timed out. Pre-seeding \
                                     parked_evt to credit them as acks for this \
                                     cycle so the rendezvous does not wait 30s \
                                     for events that already fired."
                                );
                                if let Err(e) =
                                    freeze_coord_parked_evt.write(still_parked as u64)
                                {
                                    tracing::warn!(
                                        err = %e,
                                        still_parked,
                                        "freeze-coord: parked_evt pre-seed write \
                                         failed; rendezvous may wait full \
                                         FREEZE_RENDEZVOUS_TIMEOUT for stale \
                                         parker(s)"
                                    );
                                }
                            }
                            freeze_coord_freeze.store(true, Ordering::Release);
                            // No force-clear of `parked` flags here.
                            // The post-thaw barrier at the END of
                            // every prior freeze_and_capture cycle
                            // (see `// Post-thaw barrier` below)
                            // is the primary guarantee that every
                            // vCPU has run its trailing
                            // `parked.store(false)` before this
                            // cycle starts. Force-clearing
                            // mid-cycle would erase the legitimate
                            // `parked=true` of a vCPU still in cycle
                            // N's park loop and deadlock the
                            // rendezvous (vCPU never re-stores
                            // parked=true; coord waits 30 s). The
                            // pre-seed above handles the residual
                            // case where the post-thaw barrier
                            // itself timed out.
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
                            //
                            // Primary defense for the AP path: the
                            // AP threads are joined in
                            // `collect_results` AFTER the coord
                            // joins, so in the normal lifecycle the
                            // coord cannot outlive an AP's `VcpuFd`.
                            // The exception is panic-unwind under
                            // `panic = "unwind"` (test profile),
                            // where the AP's panic hook fires
                            // synchronously on the panicking thread
                            // and the subsequent stack drop unmaps
                            // the AP's `kvm_run` page mid-cycle —
                            // before any join. Without a per-AP
                            // gate the unguarded `ie.set(1)` above
                            // would `write_volatile` through a
                            // pointer into freed memory.
                            //
                            // Secondary defense: each AP carries an
                            // `Arc<AtomicBool>` (`VcpuThread::alive`)
                            // that the AP's panic hook flips to
                            // `false` BEFORE unwinding starts.
                            // The Acquire load below
                            // synchronizes-with that Release store
                            // (panic hook runs synchronously on the
                            // panicking AP thread before unwind),
                            // so a `true` reading observed here
                            // happens-before any subsequent unwind
                            // drop of `vcpu`. Mirrors the
                            // BSP-side `bsp_alive` TOCTOU-tightened
                            // gate: load fresh at the actual
                            // `ie.set` site, not at cycle entry.
                            // `iter().enumerate()` walks index
                            // alongside the handle so the
                            // `freeze_coord_ap_alive[i]` lookup
                            // stays index-aligned.
                            //
                            // The BSP IE write is gated on
                            // `bsp_alive` because run_vm drops the
                            // BSP before collect_results runs; see
                            // the gate's doc above.
                            for (i, ie) in freeze_coord_ap_ies.iter().enumerate() {
                                if let Some(ie) = ie
                                    && freeze_coord_ap_alive[i]
                                        .load(Ordering::Acquire)
                                {
                                    ie.set(1);
                                }
                            }
                            // Re-load `bsp_alive` immediately before the
                            // BSP `ie.set()` instead of reusing the
                            // cycle-entry snapshot (`bsp_alive_at_start`).
                            // The snapshot is captured at the top of
                            // 'capture and is many milliseconds stale by
                            // the time pass-1 runs (worker pause()+ack,
                            // parked_evt pre-seed, the freeze=true
                            // Release store, and the virtio-blk
                            // pause()-rendezvous all happen in between).
                            // The BSP run-loop can transition
                            // `bsp_alive=false` and drop its `VcpuFd` at
                            // any point in that window. Without a
                            // fresh load, `ImmediateExitHandle::set(1)`
                            // would issue a `write_volatile` through a
                            // pointer into a `kvm_run` mmap whose
                            // backing pages were unmapped when the BSP
                            // `VcpuFd` was dropped (the kernel's
                            // `kvm_vcpu_release` path tears down the
                            // `kvm_run` MAP_SHARED region; subsequent
                            // userspace writes against the stale
                            // pointer are use-after-free into freed
                            // pages). The Acquire load pairs with the
                            // BSP run-loop's Release store of `false`
                            // on its way out: a `bsp_alive_now == true`
                            // observed here happens-before any
                            // `false` the BSP could subsequently
                            // store, which means the BSP `VcpuFd` is
                            // still alive AT the moment of `ie.set()`
                            // and cannot be dropped until the next
                            // load reads false. Pass-2's pthread_kill
                            // and the rendezvous-wait below issue
                            // their own fresh Acquire loads for the
                            // same TOCTOU reason.
                            let bsp_alive_for_ie =
                                bsp_alive_for_coord.load(Ordering::Acquire);
                            if bsp_alive_for_ie
                                && let Some(ref ie) = freeze_coord_bsp_ie_handle
                            {
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
                            // Pass 2: signal every vCPU. AP signals
                            // are always safe; the BSP signal is
                            // gated on `bsp_alive_at_start` — the
                            // cycle-entry snapshot — rather than a
                            // fresh load. `pthread_kill` against an
                            // exited tid returns ESRCH and is
                            // harmless either way: a stale `true`
                            // here just adds one ESRCH-suppressing
                            // log line; a stale `false` is fine
                            // because the BSP transitioned dead
                            // between entry and now, so it neither
                            // needs nor can receive the kick. Unlike
                            // `ImmediateExitHandle::set(1)` above,
                            // `pthread_kill` does not dereference
                            // any per-`VcpuFd`-owned mmap, so there
                            // is no use-after-free hazard requiring
                            // a re-load.
                            for &tid in &freeze_coord_ap_pthreads {
                                unsafe {
                                    libc::pthread_kill(tid, vcpu_signal());
                                }
                            }
                            if bsp_alive_at_start {
                                unsafe {
                                    libc::pthread_kill(freeze_coord_bsp_tid, vcpu_signal());
                                }
                            }
                            // Wait for N-of-N parked acks via a
                            // countdown latch over `parked_evt`. The
                            // counter-mode eventfd accumulates one
                            // write per parker (every vCPU + the
                            // virtio-blk worker writes 1 AFTER its
                            // own Release store on parked/paused).
                            // Each `read()` drains the accumulated
                            // count atomically and resets it; the
                            // closure tallies these drains until the
                            // total reaches `expected`. Replaces the
                            // per-iteration O(N) AtomicBool scan with
                            // an O(1) counter add — the AtomicBool
                            // flags remain the synchronizes-with
                            // anchor for the diagnostic timeout-log
                            // path below, but they are no longer the
                            // hot-path readiness check.
                            //
                            // The pre-pass drain above (before the
                            // `freeze=true` flip and the kicks)
                            // ensures every increment we observe
                            // from here on is a park-ack for THIS
                            // cycle, not an ack from cycle N-1
                            // that arrived after the post-thaw
                            // barrier's drain.
                            //
                            // The +1 for virtio-blk is gated on
                            // `worker_was_running` — when the
                            // worker thread is not alive, pause()
                            // is a no-op and writes no parked_evt
                            // ack, so counting +1 would make the
                            // rendezvous wait 30 s for an ack that
                            // never comes.
                            let mut expected_parks: u64 =
                                freeze_coord_ap_parked.len() as u64
                                    + if bsp_alive_at_start { 1 } else { 0 }
                                    + if worker_was_running { 1 } else { 0 };
                            let deadline = Instant::now() + FREEZE_RENDEZVOUS_TIMEOUT;
                            // Sub-deadline for the virtio-blk worker
                            // ack. `device.rs::stop_worker_and_reclaim_state`
                            // (and any sibling shutdown path) writes
                            // `paused.store(false, Release)` BEFORE
                            // signalling stop_fd and joining the
                            // worker — see lines around device.rs
                            // 3561 (`self.paused.store(false,
                            // Ordering::Release)` + `signal_worker_stop`).
                            // Between that store and the worker
                            // exiting (with no further `paused=true`
                            // store on the shutdown path), the
                            // freeze-coord pre-pause snapshot here
                            // observes `paused == false` and counts
                            // `worker_was_running = true → +1` —
                            // but no live thread will write
                            // `parked_evt` for this cycle. Without a
                            // sub-deadline the rendezvous waits the
                            // full 30 s for an ack the worker
                            // physically cannot send.
                            //
                            // 1 s budget covers a healthy worker's
                            // `pread`/`pwrite` drain on warm page
                            // cache (the same envelope
                            // `DROP_JOIN_TIMEOUT` (1 s) commits to
                            // for the worker join in
                            // `device.rs`). If the worker hasn't
                            // parked within 1 s, it's likely
                            // mid-shutdown (signal_worker_stop
                            // pre-clears paused=false). Dropping
                            // the +1 avoids a 30 s timeout. A
                            // slow-but-alive worker mid-drain
                            // could still mutate ring state
                            // concurrently; this is accepted
                            // because tmpfs backing bounds drain
                            // time below the sub-timeout.
                            const WORKER_PARK_SUB_TIMEOUT: Duration =
                                Duration::from_secs(1);
                            let worker_sub_deadline =
                                Instant::now() + WORKER_PARK_SUB_TIMEOUT;
                            let mut worker_dropped: bool = false;
                            let mut parked_count: u64 = 0;
                            let mut all_parked = false;
                            loop {
                                if freeze_coord_bsp_done.load(Ordering::Acquire) {
                                    break;
                                }
                                if parked_count >= expected_parks {
                                    all_parked = true;
                                    break;
                                }
                                // Worker sub-timeout. Only fires
                                // when the worker was counted in
                                // `expected_parks` (i.e.
                                // `worker_was_running` was true at
                                // pre-pause snapshot) and we have
                                // not yet decremented for it. The
                                // condition `parked_count <
                                // expected_parks` plus the wall-
                                // clock check (`now >=
                                // worker_sub_deadline`) localises
                                // the bookkeeping change to the
                                // path where the worker really did
                                // not ack. The Acquire load on
                                // `paused` synchronises-with any
                                // worker `Release` it might still
                                // perform on a slow path; if the
                                // worker DID park we observe
                                // `paused == true` and DO NOT
                                // decrement (the matching ack will
                                // arrive imminently or has already
                                // arrived in `parked_count`).
                                if !worker_dropped
                                    && worker_was_running
                                    && Instant::now() >= worker_sub_deadline
                                    && freeze_coord_virtio_blk_paused
                                        .as_ref()
                                        .is_some_and(|p| !p.load(Ordering::Acquire))
                                {
                                    // Final paused re-check before
                                    // decrementing. The condition
                                    // above sampled paused==false,
                                    // but a slow-but-alive worker
                                    // could have transitioned
                                    // paused=true between that
                                    // sample and here. Re-loading
                                    // with Acquire pairs with the
                                    // worker's Release store on
                                    // pause(). If the worker DID
                                    // park, skip the drop and let
                                    // the next loop iteration
                                    // observe the matching
                                    // parked_evt ack — never
                                    // double-count by both dropping
                                    // expected_parks and absorbing
                                    // the eventfd write.
                                    if freeze_coord_virtio_blk_paused
                                        .as_ref()
                                        .is_some_and(|p| p.load(Ordering::Acquire))
                                    {
                                        continue;
                                    }
                                    tracing::warn!(
                                        worker_park_sub_timeout_ms =
                                            WORKER_PARK_SUB_TIMEOUT.as_millis() as u64,
                                        parked_count,
                                        expected_parks,
                                        "freeze-coord: virtio-blk worker did \
                                         not ack park within sub-timeout AND \
                                         `paused` is still false — most \
                                         likely the worker is mid-shutdown \
                                         (signal_worker_stop already cleared \
                                         paused=false on its way out), so no \
                                         live thread will write parked_evt \
                                         for this cycle. Dropping the +1 \
                                         from expected_parks so the \
                                         rendezvous proceeds without waiting \
                                         the full FREEZE_RENDEZVOUS_TIMEOUT \
                                         for an ack that physically cannot \
                                         arrive."
                                    );
                                    expected_parks =
                                        expected_parks.saturating_sub(1);
                                    worker_dropped = true;
                                    // Re-check the `all_parked`
                                    // predicate immediately so a
                                    // concurrent vCPU ack that just
                                    // pushed parked_count to the
                                    // (now lower) expected value is
                                    // recognised in this iteration
                                    // rather than after another
                                    // poll cycle.
                                    if parked_count >= expected_parks {
                                        all_parked = true;
                                        break;
                                    }
                                }
                                let now = Instant::now();
                                if now > deadline {
                                    // Diagnostic snapshot of every
                                    // parker's flag, computed once on
                                    // timeout for the error log. Hot
                                    // path no longer reads these
                                    // bools per iteration.
                                    let ap_states: Vec<bool> = freeze_coord_ap_parked
                                        .iter()
                                        .map(|p| p.load(Ordering::Acquire))
                                        .collect();
                                    let bsp_p = freeze_coord_bsp_parked.load(Ordering::Acquire);
                                    // Lock-free read via the
                                    // pre-acquired `paused_handle()`
                                    // Arc — avoids taking the device
                                    // mutex on the timeout-diagnostic
                                    // path. Acquire ordering pairs
                                    // with the worker's Release on
                                    // `paused.store(true)` so the
                                    // diagnostic sees a coherent
                                    // worker state.
                                    let blk_parked = freeze_coord_virtio_blk_paused
                                        .as_ref()
                                        .is_none_or(|p| p.load(Ordering::Acquire));
                                    tracing::error!(
                                        ?ap_states,
                                        bsp_parked = bsp_p,
                                        blk_parked,
                                        parked_count,
                                        expected_parks,
                                        "freeze-coord: timed out waiting for vCPUs / worker to park. \
                                         If blk_parked=false, the worker is most likely stuck in a \
                                         slow pread/pwrite against the backing file — verify the \
                                         backing is fast (tmpfs / warm page cache); the vCPU \
                                         thread's blocking budget is bounded by the freeze \
                                         rendezvous timeout, so a backing slow enough to push \
                                         per-request IO past that bound prevents the rendezvous \
                                         from completing. The worker observes PAUSE_TOKEN only \
                                         between blocking syscalls, so a long pread/pwrite delays \
                                         the park-ack until the syscall returns."
                                    );
                                    break;
                                }
                                let remaining_ms = (deadline - now)
                                    .as_millis()
                                    .min(i32::MAX as u128) as i32;
                                let mut pfds = [
                                    libc::pollfd {
                                        fd: freeze_coord_parked_evt.as_raw_fd(),
                                        events: libc::POLLIN,
                                        revents: 0,
                                    },
                                    libc::pollfd {
                                        fd: freeze_coord_kill_evt.as_raw_fd(),
                                        events: libc::POLLIN,
                                        revents: 0,
                                    },
                                    libc::pollfd {
                                        fd: freeze_coord_bsp_done_evt.as_raw_fd(),
                                        events: libc::POLLIN,
                                        revents: 0,
                                    },
                                ];
                                // SAFETY: pfds is a 3-element pollfd
                                // array; nfds matches. Every poll
                                // outcome (ready, timeout, EINTR,
                                // error) loops back to the
                                // countdown predicate at the top.
                                // EINTR from SIGRTMIN is harmless:
                                // the wait simply restarts.
                                unsafe {
                                    libc::poll(
                                        pfds.as_mut_ptr(),
                                        pfds.len() as libc::nfds_t,
                                        remaining_ms,
                                    );
                                }
                                // Drain parked_evt counter once per
                                // wake. Counter mode: a single read
                                // returns the accumulated count and
                                // resets to 0; multiple coalesced
                                // parker writes are absorbed in one
                                // drain. EAGAIN (counter already 0)
                                // is benign — the poll wake may have
                                // come from kill_evt or
                                // bsp_done_evt (those are NOT
                                // drained here; the outer epoll
                                // loop owns them). Saturating add
                                // is defensive — counter mode
                                // eventfd values cap at 2^64 - 2
                                // and physically cannot overflow
                                // a u64 in any realistic VM run.
                                if let Ok(n) = freeze_coord_parked_evt.read() {
                                    parked_count = parked_count.saturating_add(n);
                                }
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
                                if skip_freeze {
                                    tracing::info!(
                                        "freeze-coord: skip_freeze — vCPUs \
                                         exited, proceeding with dump on \
                                         quiesced memory"
                                    );
                                } else {
                                    tracing::debug!(
                                        "freeze-coord: dump skipped: \
                                         rendezvous timed out"
                                    );
                                    break 'capture None;
                                }
                            }
                            // Exit-kind gate. The hardware watchpoint
                            // catches every write to
                            // `*scx_root->exit_kind`, including
                            // transient writes during init/teardown
                            // that the kernel sets to
                            // `SCX_EXIT_NONE` (0) or `SCX_EXIT_DONE`
                            // (1). Without this gate every clean
                            // scheduler shutdown produces a bogus
                            // failure dump. Read the live `exit_kind`
                            // value through the same direct-mapping +
                            // page-walk translation
                            // `read_scx_sched_state` uses; gate on
                            // `kind >= SCX_EXIT_ERROR (= 1024)` per
                            // `kernel/sched/ext_internal.h::scx_exit_kind`.
                            //
                            // Required prerequisites all flow from
                            // the same set the watchpoint resolution
                            // earlier in the loop validated when it
                            // published the `request_kva`:
                            //   - request_kva non-zero (resolved
                            //     `*scx_root + exit_kind_offset`)
                            //   - owned_accessor for cr3_pa /
                            //     page_offset / l5
                            //   - freeze_coord_mem for read_u32
                            // Any prereq absence means the watchpoint
                            // could not have armed (publish path
                            // requires the same handles), so a
                            // gated call without prerequisites is a
                            // logic bug — log + dump anyway rather
                            // than silently swallow the trigger.
                            if gate_on_exit_kind {
                                let exit_kind_kva = freeze_coord_watchpoint
                                    .request_kva
                                    .load(Ordering::Acquire);
                                let gate_decision = match (
                                    exit_kind_kva,
                                    owned_accessor.as_ref(),
                                    freeze_coord_mem.as_deref(),
                                ) {
                                    (0, _, _) | (_, None, _) | (_, _, None) => {
                                        tracing::warn!(
                                            exit_kind_kva = format_args!(
                                                "{:#x}",
                                                exit_kind_kva
                                            ),
                                            owned_accessor_present =
                                                owned_accessor.is_some(),
                                            mem_present = freeze_coord_mem.is_some(),
                                            "freeze-coord: exit_kind gate \
                                             prerequisites missing — proceeding \
                                             with dump (watchpoint should not \
                                             have armed without these)"
                                        );
                                        // Treat missing prereqs as
                                        // "do not gate" so a bona fide
                                        // dump still emits.
                                        true
                                    }
                                    (kva, Some(owned), Some(mem)) => {
                                        let kernel = owned.guest_kernel();
                                        let walk = kernel.walk_context();
                                        match crate::monitor::idr::translate_any_kva(
                                            mem,
                                            walk.cr3_pa,
                                            walk.page_offset,
                                            kva,
                                            walk.l5,
                                            walk.tcr_el1,
                                        ) {
                                            Some(pa) => {
                                                let kind = mem.read_u32(pa, 0);
                                                // SCX_EXIT_ERROR = 1024 — the
                                                // first error-class value in
                                                // `enum scx_exit_kind`. All
                                                // values below are clean
                                                // (NONE/DONE) or normal
                                                // unregister classes
                                                // (UNREG/SYSRQ/PARENT) that
                                                // do not warrant a failure
                                                // dump.
                                                const SCX_EXIT_ERROR: u32 = 1024;
                                                if kind < SCX_EXIT_ERROR {
                                                    tracing::info!(
                                                        kind,
                                                        exit_kind_kva =
                                                            format_args!("{:#x}", kva),
                                                        "freeze-coord: \
                                                         exit_kind gate \
                                                         suppressed dump \
                                                         (kind < 1024 = clean \
                                                         shutdown / non-error \
                                                         transition)"
                                                    );
                                                    false
                                                } else {
                                                    tracing::debug!(
                                                        kind,
                                                        "freeze-coord: \
                                                         exit_kind gate passed \
                                                         (kind >= 1024)"
                                                    );
                                                    true
                                                }
                                            }
                                            None => {
                                                // KVA was published but no
                                                // longer translates: most
                                                // likely the slab page that
                                                // held `*scx_root` was
                                                // freed during teardown.
                                                // Suppress the dump — there
                                                // is no scheduler state to
                                                // capture anyway.
                                                tracing::info!(
                                                    exit_kind_kva =
                                                        format_args!("{:#x}", kva),
                                                    "freeze-coord: exit_kind \
                                                     gate translate failed \
                                                     (scheduler likely torn \
                                                     down) — suppressing dump"
                                                );
                                                false
                                            }
                                        }
                                    }
                                };
                                if !gate_decision {
                                    break 'capture None;
                                }
                            }
                            if let Some(owned) = owned_accessor
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
                                // Bind kernel once for the whole dump
                                // block. Pre-fix this called
                                // owned.guest_kernel() three times
                                // (scx_walker_capture, task_enrichment_capture,
                                // cpu_time_capture). The accessor is a
                                // trivial &-return but the repetition
                                // obscured ownership — every consumer
                                // wants the same kernel handle.
                                let dump_kernel = owned.guest_kernel();
                                // Pre-collect register snapshots: needed
                                // for both the report's vcpu_regs field
                                // AND the per-task enrichment running_pc
                                // mapping (walking rq->scx.curr to the
                                // corresponding vCPU's IP). Capturing
                                // here before the dump means the same
                                // snapshot drives every consumer.
                                let vcpu_regs = collect_vcpu_regs();
                                // SCX walker owned data — backs the
                                // borrow-only `ScxWalkerCapture`. The
                                // capture runs while every vCPU is
                                // paused at the freeze rendezvous, so
                                // each phase emits a tracing::debug
                                // duration line so operators can
                                // budget against the watchdog timeout.
                                let scx_build_t0 = std::time::Instant::now();
                                let scx_owned = crate::vmm::capture_scx::build(
                                    owned,
                                    dump_scx_walker_offsets.as_ref(),
                                    dump_cpu_time_symbols.as_ref(),
                                    prog_per_cpu_offsets.as_deref(),
                                );
                                tracing::debug!(
                                    elapsed_us = scx_build_t0.elapsed().as_micros() as u64,
                                    populated = scx_owned.is_some(),
                                    "freeze-coord: capture_scx::build"
                                );
                                let scx_walker_capture = scx_owned.as_ref().and_then(|so| {
                                    let offsets = dump_scx_walker_offsets.as_ref()?;
                                    Some(crate::monitor::dump::ScxWalkerCapture {
                                        kernel: dump_kernel,
                                        offsets,
                                        scx_root_kva: so.scx_root_kva,
                                        rq_kvas: &so.rq_kvas,
                                        rq_pas: &so.rq_pas,
                                        per_cpu_offsets: prog_per_cpu_offsets
                                            .as_deref()
                                            .unwrap_or(&[]),
                                        nr_nodes: freeze_coord_num_nodes,
                                    })
                                });
                                // Task-enrichment owned data — backs the
                                // borrow-only `TaskEnrichmentCapture`.
                                let task_build_t0 = std::time::Instant::now();
                                let task_owned = crate::vmm::capture_tasks::build(
                                    owned,
                                    scx_owned.as_ref(),
                                    dump_scx_walker_offsets.as_ref(),
                                    dump_task_enrichment_offsets.as_ref(),
                                    &vcpu_regs,
                                );
                                tracing::debug!(
                                    elapsed_us = task_build_t0.elapsed().as_micros() as u64,
                                    populated = task_owned.is_some(),
                                    tasks = task_owned.as_ref().map(|t| t.tasks.len()).unwrap_or(0),
                                    "freeze-coord: capture_tasks::build"
                                );
                                let task_enrichment_capture = task_owned.as_ref().and_then(|to| {
                                    let te_offsets = dump_task_enrichment_offsets.as_ref()?;
                                    Some(crate::monitor::dump::TaskEnrichmentCapture {
                                        kernel: dump_kernel,
                                        offsets: te_offsets,
                                        sched_classes: &to.sched_classes,
                                        lock_slowpaths: &to.lock_slowpaths,
                                        tasks: &to.tasks,
                                    })
                                });
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
                                // to dump/mod.rs as Option so the per-CPU
                                // walker can skip iowait_sleeptime
                                // independently per CPU.
                                let cpu_time_capture = match (
                                    freeze_coord_mem.as_deref(),
                                    dump_cpu_time_offsets.as_ref(),
                                    dump_cpu_time_symbols.as_ref(),
                                    prog_per_cpu_offsets.as_deref(),
                                ) {
                                    (Some(mem), Some(offsets), Some(syms), Some(pcpu)) => {
                                        match (syms.kernel_cpustat, syms.kstat) {
                                            (Some(kcpustat_kva), Some(kstat_kva)) => {
                                                let page_offset = dump_kernel.page_offset();
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
                                // Force the lazy cast-analysis on this
                                // dump's host coordinator thread (NOT a
                                // vCPU thread — vCPUs are paused at the
                                // freeze rendezvous). Bind the resulting
                                // `Option<Arc<CastAnalysisOutput>>`
                                // BEFORE the `DumpContext` literal so
                                // the inner `Arc` outlives
                                // `dump_state`'s borrow. First dump does
                                // the work; subsequent dumps in the same
                                // VM hit the `OnceLock` and return
                                // immediately.
                                //
                                // The full output carries both the cast
                                // map AND the cross-BTF Fwd resolution
                                // index (every parsed embedded BPF
                                // object's BTF + a name-keyed lookup).
                                // Both halves are threaded into
                                // `DumpContext` so the renderer's chase
                                // paths can resolve `BTF_KIND_FWD`
                                // pointees that live in a sibling
                                // object's BTF — the typical multi-
                                // `.bpf.objs` shape where one object
                                // declares `struct foo;` (forward) and
                                // another defines the body.
                                let cast_analysis = freeze_coord_cast_map.get_full();
                                let cast_map_ref = cast_analysis
                                    .as_ref()
                                    .and_then(|out| out.cast_maps.first().map(|m| m.as_ref()));
                                let cross_btf_fwd_index_owned = cast_analysis.as_ref().map(|out| {
                                    crate::monitor::dump::CrossBtfFwdIndex {
                                        btfs: &out.btfs,
                                        fwd_index: &out.fwd_index,
                                    }
                                });
                                let dump_state_t0 = std::time::Instant::now();
                                let mut report = crate::monitor::dump::dump_state(
                                    crate::monitor::dump::DumpContext {
                                        accessor: &map_accessor,
                                        btf,
                                        num_cpus: freeze_coord_num_cpus,
                                        arena_offsets: dump_arena_offsets.as_ref(),
                                        prog_capture: prog_capture.as_ref(),
                                        cpu_time_capture: cpu_time_capture.as_ref(),
                                        task_enrichment_capture: task_enrichment_capture
                                            .as_ref(),
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
                                        scx_walker_capture: scx_walker_capture
                                            .as_ref(),
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
                                        deadline: capture_deadline,
                                        // The bound `cast_map_ref` is
                                        // `Option<&CastMap>` derived from
                                        // the full output's inner `Arc`.
                                        // The full output keeps the
                                        // `CastMap` alive for the
                                        // duration of this `dump_state`
                                        // call.
                                        cast_map: cast_map_ref,
                                        // Cross-BTF Fwd resolution
                                        // context — see
                                        // [`DumpContext::cross_btf_fwd_index`].
                                        cross_btf_fwd_index: cross_btf_fwd_index_owned,
                                        alloc_size_types: cast_analysis
                                            .as_ref()
                                            .map(|o| o.alloc_size_types.as_slice())
                                            .unwrap_or(&[]),
                                    },
                                );
                                tracing::debug!(
                                    elapsed_us = dump_state_t0.elapsed().as_micros() as u64,
                                    maps = report.maps.len(),
                                    "freeze-coord: dump_state"
                                );
                                report.vcpu_regs = vcpu_regs;
                                // Per-node NUMA stats — overwrite the
                                // empty default `dump_state` writes when
                                // the producer lands a non-empty Vec.
                                let numa_build_t0 = std::time::Instant::now();
                                let numa_stats = crate::vmm::capture_numa::build(
                                    owned,
                                    dump_numa_offsets.as_ref(),
                                    dump_cpu_time_symbols.as_ref(),
                                    freeze_coord_num_nodes,
                                );
                                tracing::debug!(
                                    elapsed_us = numa_build_t0.elapsed().as_micros() as u64,
                                    nodes = numa_stats.as_ref().map(|s| s.len()).unwrap_or(0),
                                    "freeze-coord: capture_numa::build"
                                );
                                if let Some(stats) = numa_stats
                                    && !stats.is_empty()
                                {
                                    report.per_node_numa = stats;
                                    report.per_node_numa_unavailable = None;
                                }
                                Some((report, capture_start))
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
                                    dump_truncated_at_us: None,
                                    probe_counters: None,
                                    scx_static_ranges: Default::default(),
                                    is_placeholder: false,
                                    sdt_alloc_unavailable: Some(
                                        "dump prerequisites unavailable".to_string(),
                                    ),
                                };
                                tracing::warn!(
                                    owned_accessor = owned_accessor.is_some(),
                                    dump_btf = dump_btf.is_some(),
                                    "freeze-coord: dump prerequisites unavailable; \
                                     emitting partial report with vcpu_regs only"
                                );
                                Some((report, capture_start))
                            }
                        } // end 'capture labeled block (the closure
                          // returns this block's value; the caller
                          // is responsible for invoking
                          // `thaw_and_barrier` AFTER any
                          // while-frozen work it needs to perform
                          // — the late-trigger backstop reads
                          // guest memory while quiesced, so the
                          // thaw cannot be unconditional inside
                          // the closure).
                        };
                    // Unified thaw + post-thaw barrier. Called by
                    // every site after `freeze_and_capture` returns
                    // (and after any while-frozen work the site
                    // needs). Replaces the per-site thaw block that
                    // previously diverged on which ordering rules
                    // fired. Resumes the virtio-blk worker FIRST so
                    // its `paused.load(Acquire)` poll exits before
                    // the freeze flag clears (worker polls `paused`,
                    // vCPUs poll `freeze`; resume-then-freeze=false
                    // means both wake paths land cleanly), then
                    // clears `freeze` and writes `thaw_evt` so every
                    // parked vCPU's poll wakes within microseconds.
                    //
                    // Post-thaw barrier — wait for every parker to
                    // clear its flag (vCPUs run their trailing
                    // `parked.store(false)` in handle_freeze AFTER
                    // observing freeze=false; the worker clears
                    // `paused` on resume()). Cycle N+1's
                    // rendezvous loop assumes all parked flags are
                    // false at entry; without this barrier a
                    // still-mid-thaw vCPU's `parked=true` would
                    // either be cleared by a force-clear and
                    // deadlock the cycle (legitimate parked=true
                    // for cycle N+1 never re-stored), OR be
                    // race-observed as a false positive (vCPU never
                    // parked for cycle N+1).
                    //
                    // No dedicated unparked_evt fd exists
                    // (handle_freeze does not write any eventfd on
                    // its trailing `parked.store(false)`); the
                    // barrier polls the AtomicBools at a 10 ms
                    // cadence — the same backstop handle_freeze
                    // uses for its `freeze.load(Acquire)` re-check
                    // when the thaw_evt poll's level fans across
                    // multiple parkers. EINTR / partial wakes are
                    // harmless; the predicate re-evaluates each
                    // iteration.
                    //
                    // Finally drain `parked_evt` so cycle N+1's
                    // countdown latch starts at 0.
                    let thaw_and_barrier = || {
                        // Always unfreeze + thaw even on teardown
                        // so vCPUs don't stay parked.
                        if let Some(ref blk) = freeze_coord_virtio_blk {
                            blk.lock().resume();
                        }
                        freeze_coord_freeze.store(false, Ordering::Release);
                        let _ = freeze_coord_thaw_evt.write(1);
                        if freeze_coord_bsp_done.load(Ordering::Acquire) {
                            return;
                        }

                        let post_thaw_deadline =
                            Instant::now() + FREEZE_RENDEZVOUS_TIMEOUT;
                        loop {
                            if freeze_coord_kill.load(Ordering::Acquire)
                                || freeze_coord_bsp_done.load(Ordering::Acquire)
                            {
                                break;
                            }
                            let aps_unparked = freeze_coord_ap_parked
                                .iter()
                                .all(|p| !p.load(Ordering::Acquire));
                            let bsp_unparked = !freeze_coord_bsp_parked
                                .load(Ordering::Acquire);
                            // Lock-free read via the pre-acquired
                            // `paused_handle()` Arc — avoids
                            // taking the device mutex inside the
                            // post-thaw barrier hot loop. Acquire
                            // ordering pairs with the worker's
                            // Release on `paused.store(false)`
                            // (resume path) so the predicate sees
                            // a coherent worker state.
                            let blk_unpaused = freeze_coord_virtio_blk_paused
                                .as_ref()
                                .is_none_or(|p| !p.load(Ordering::Acquire));
                            if aps_unparked && bsp_unparked && blk_unpaused {
                                break;
                            }
                            let now = Instant::now();
                            if now > post_thaw_deadline {
                                let ap_states: Vec<bool> =
                                    freeze_coord_ap_parked
                                        .iter()
                                        .map(|p| p.load(Ordering::Acquire))
                                        .collect();
                                tracing::warn!(
                                    ?ap_states,
                                    bsp_parked = !bsp_unparked,
                                    blk_paused = !blk_unpaused,
                                    "freeze-coord: post-thaw barrier timed out — \
                                     a parker did not clear within \
                                     FREEZE_RENDEZVOUS_TIMEOUT; subsequent freeze \
                                     cycles may see stale parked=true and timeout \
                                     the rendezvous"
                                );
                                break;
                            }
                            let remaining_ms = (post_thaw_deadline - now)
                                .as_millis()
                                .min(i32::MAX as u128) as i32;
                            let mut pfds = [
                                libc::pollfd {
                                    fd: freeze_coord_kill_evt.as_raw_fd(),
                                    events: libc::POLLIN,
                                    revents: 0,
                                },
                                libc::pollfd {
                                    fd: freeze_coord_bsp_done_evt.as_raw_fd(),
                                    events: libc::POLLIN,
                                    revents: 0,
                                },
                            ];
                            // SAFETY: pfds is a 2-element pollfd
                            // array; nfds matches. Bounded 10 ms
                            // wait is the cadence at which the
                            // AtomicBool predicate re-runs.
                            let wait_ms = 10.min(remaining_ms);
                            unsafe {
                                libc::poll(
                                    pfds.as_mut_ptr(),
                                    pfds.len() as libc::nfds_t,
                                    wait_ms,
                                );
                            }
                        }
                        // Drain parked_evt so cycle N+1's countdown
                        // latch starts at 0. EAGAIN (counter already
                        // 0) is benign.
                        let _ = freeze_coord_parked_evt.read();
                    };
                    // Helper: extend the watchdog deadline by the
                    // wall-clock duration of a single
                    // `freeze_and_capture(..)` cycle. Captures eat
                    // host wall-clock that would otherwise count
                    // against the workload's `workload_duration`
                    // budget; without this push, a 5 s test that
                    // fires a 2 s freeze gets only 3 s of guest
                    // execution before the watchdog kicks. Reads the
                    // current encoded reset target (or falls back to
                    // `workload_duration` counted from now) and
                    // writes back the sum + freeze_duration so the
                    // watchdog observes the extended deadline on its
                    // next tick. The watchdog only consults this
                    // atomic when `workload_duration` is set; runs
                    // without a workload budget remain on the
                    // boot-relative `hard_deadline` and this push is
                    // a no-op.
                    //
                    // Shared with the TLV CAPTURE handler, the
                    // user-watchpoint dispatcher, and the periodic-
                    // capture drain so the same arithmetic and
                    // ordering discipline apply at every fire site.
                    let extend_watchdog_for_freeze = |freeze_start: Instant| {
                        if let Some(d) = workload_duration_for_coord {
                            let freeze_duration = freeze_start.elapsed();
                            let prior = watchdog_reset_for_coord.load(Ordering::Acquire);
                            let prior_ns = if prior == 0 {
                                run_start
                                    .elapsed()
                                    .as_nanos()
                                    .saturating_add(d.as_nanos())
                            } else {
                                prior as u128
                            };
                            let new_target_ns =
                                prior_ns.saturating_add(freeze_duration.as_nanos());
                            let encoded = u64::try_from(new_target_ns).unwrap_or(u64::MAX).max(1);
                            watchdog_reset_for_coord.store(encoded, Ordering::Release);
                        }
                    };
                    // Helper: persist the JSON to the optional file
                    // sink, then log a single info-level summary line
                    // referencing the file path + byte count +
                    // capture timing. The JSON is NOT inlined into
                    // the trace log — a 50-map dump runs hundreds of
                    // KB and floods every downstream sink (file
                    // logger, journald, stderr) with a payload that
                    // is already on disk at the dump path.
                    #[allow(clippy::too_many_arguments)]
                    let emit_json = |json: &str,
                                     map_count: usize,
                                     vcpu_regs_count: usize,
                                     tasks_enriched: usize,
                                     elapsed_ms: u64,
                                     truncated_at_us: Option<u64>| {
                        let path_str: Option<String> =
                            freeze_coord_dump_path.as_ref().and_then(|p| {
                                if let Some(parent) = p.parent() {
                                    let _ = std::fs::create_dir_all(parent);
                                }
                                match std::fs::write(p, json) {
                                    Ok(()) => Some(p.display().to_string()),
                                    Err(e) => {
                                        tracing::warn!(
                                            path = %p.display(),
                                            error = %e,
                                            "freeze-coord: failure-dump file write failed"
                                        );
                                        None
                                    }
                                }
                            });
                        let json_bytes = json.len();
                        let path_part = path_str
                            .as_deref()
                            .map(|p| format!(" -> {p}"))
                            .unwrap_or_else(|| " (no file sink)".to_string());
                        let trunc_part = truncated_at_us
                            .map(|us| format!(" (truncated at {us}us)"))
                            .unwrap_or_default();
                        tracing::info!(
                            target: "ktstr::failure_dump",
                            map_count,
                            vcpu_regs_count,
                            tasks_enriched,
                            json_bytes,
                            elapsed_ms,
                            truncated_at_us,
                            path = path_str.as_deref(),
                            "freeze-coord: dump complete{trunc_part}, {map_count} maps, {tasks_enriched} tasks enriched, {elapsed_ms}ms freeze, {json_bytes} bytes{path_part}"
                        );
                    };
                    // On-demand snapshot handler. Drains every
                    // [`crate::vmm::wire::MSG_TYPE_SNAPSHOT_REQUEST`]
                    // frame the TOKEN_TX handler accumulated this
                    // iteration, regardless of `freeze_state`. The
                    // `on_demand_in_flight` AcqRel-bool serialises
                    // CAPTURE/WATCH against the user-watchpoint
                    // dispatcher below — a snapshot capture in
                    // progress here makes the watchpoint loop re-arm
                    // its `hit` flag for the next iteration instead
                    // of opening a second concurrent capture window.
                    //
                    // CAPTURE runs `freeze_and_capture(false)` and
                    // stores the report on the bridge under the
                    // tag, then frames a `MSG_TYPE_SNAPSHOT_REPLY`
                    // TLV (header + 72-byte payload) and pushes it
                    // through `queue_input_port1` so the guest's
                    // blocking reader on `/dev/vport0p1` wakes
                    // within microseconds and observes
                    // `reply.request_id == request.request_id`.
                    // WATCH resolves the symbol via the cached
                    // vmlinux ELF symbol table, allocates a free
                    // user watchpoint slot, publishes the resolved
                    // KVA + tag into `WatchpointArm`, kicks every
                    // vCPU so `self_arm_watchpoint` picks up the
                    // new arm before the next `KVM_RUN`, and
                    // replies OK over the same TLV channel. A
                    // future guest write to the resolved KVA fires
                    // the corresponding `KVM_EXIT_DEBUG` and the
                    // user-watchpoint dispatcher (further down the
                    // iteration) drives the matching capture.
                    let pending = std::mem::take(&mut snapshot_requests_pending);
                    for SnapshotRequest {
                        request_id,
                        kind,
                        tag,
                    } in pending
                    {
                        if kind == crate::vmm::wire::SNAPSHOT_KIND_CAPTURE
                            && owned_accessor.is_none()
                        {
                            tracing::info!(
                                request_id,
                                %tag,
                                "freeze-coord: TLV CAPTURE deferred \
                                 (owned_accessor not yet adopted)"
                            );
                            capture_requests_deferred.push(SnapshotRequest {
                                request_id,
                                kind,
                                tag,
                            });
                            continue;
                        }
                        if freeze_coord_on_demand_in_flight
                            .swap(true, Ordering::AcqRel)
                        {
                            // A user-watchpoint capture is already
                            // in flight (or a prior iteration
                            // somehow left the gate set). Reply
                            // ERR rather than let the guest block
                            // its full 30 s deadline; the test
                            // can retry once the in-flight
                            // capture completes.
                            let reply = frame_snapshot_reply(
                                request_id,
                                crate::vmm::wire::SNAPSHOT_STATUS_ERR,
                                "another snapshot capture is in flight; retry",
                            );
                            freeze_coord_virtio_con
                                .lock()
                                .queue_input_port1(&reply);
                            tracing::warn!(
                                request_id,
                                %tag,
                                kind,
                                "freeze-coord: snapshot request rejected (in-flight gate held)"
                            );
                            continue;
                        }
                        match kind {
                            crate::vmm::wire::SNAPSHOT_KIND_CAPTURE => {
                                tracing::info!(
                                    request_id,
                                    %tag,
                                    "freeze-coord: TLV CAPTURE request"
                                );
                                // CAPTURE has no while-frozen work,
                                // so thaw immediately after the
                                // dump returns. Then extend the
                                // watchdog deadline by the freeze
                                // duration via the shared closure
                                // (TLV CAPTURE / user watchpoint /
                                // periodic-capture all use the same
                                // arithmetic — see
                                // `extend_watchdog_for_freeze` for
                                // the full rationale).
                                let freeze_start = Instant::now();
                                let on_demand = freeze_and_capture(false);
                                thaw_and_barrier();
                                extend_watchdog_for_freeze(freeze_start);
                                let mut reply_status =
                                    crate::vmm::wire::SNAPSHOT_STATUS_OK;
                                let mut reply_reason = String::new();
                                if let Some((report, capture_start)) = on_demand {
                                    let map_count = report.maps.len();
                                    let vcpu_regs_count =
                                        report.vcpu_regs.len();
                                    let tasks_enriched =
                                        report.task_enrichments.len();
                                    // File mirror first via `&report`
                                    // (no clone). Bridge `store`
                                    // consumes the report, so any
                                    // additional reader needs to run
                                    // BEFORE the move. `to_string`
                                    // (compact) replaces
                                    // `to_string_pretty` to halve
                                    // serialization cost — the JSON
                                    // is consumed by tests and tools,
                                    // not by humans, and `jq` /
                                    // `serde_json::from_str` parse
                                    // both forms identically. Avoids
                                    // the prior `report.clone()` deep
                                    // copy of hundreds-of-KB-scale
                                    // dump data.
                                    if let Some(ref base_path) =
                                        freeze_coord_dump_path
                                    {
                                        let tagged = snapshot_tagged_path(
                                            base_path, &tag,
                                        );
                                        if let Some(parent) = tagged.parent() {
                                            let _ = std::fs::create_dir_all(parent);
                                        }
                                        match serde_json::to_string(&report) {
                                            Ok(json) => {
                                                if let Err(e) =
                                                    std::fs::write(&tagged, &json)
                                                {
                                                    tracing::warn!(
                                                        path = %tagged.display(),
                                                        error = %e,
                                                        "freeze-coord: on-demand dump file write failed"
                                                    );
                                                }
                                            }
                                            Err(e) => tracing::error!(
                                                error = %e,
                                                map_count,
                                                vcpu_regs_count,
                                                "freeze-coord: on-demand dump (JSON serialization failed)"
                                            ),
                                        }
                                    }
                                    let elapsed_ms = capture_start
                                        .elapsed()
                                        .as_millis() as u64;
                                    tracing::info!(
                                        target: "ktstr::failure_dump",
                                        kind = "on_demand_capture",
                                        request_id,
                                        %tag,
                                        map_count,
                                        vcpu_regs_count,
                                        tasks_enriched,
                                        elapsed_ms,
                                        "freeze-coord: snapshot captured and stored on bridge"
                                    );
                                    // Persist on the bridge LAST —
                                    // store moves the report. Test
                                    // code drains the bridge after
                                    // VM exit and walks the reports
                                    // via the public `Snapshot`
                                    // accessor.
                                    freeze_coord_snapshot_bridge.store(&tag, report);
                                } else {
                                    reply_status =
                                        crate::vmm::wire::SNAPSHOT_STATUS_ERR;
                                    reply_reason =
                                        "freeze rendezvous timed out (vCPU stuck \
                                         in KVM_RUN past FREEZE_RENDEZVOUS_TIMEOUT)"
                                            .to_string();
                                    tracing::warn!(
                                        request_id,
                                        %tag,
                                        "freeze-coord: on-demand capture failed (rendezvous timeout)"
                                    );
                                }
                                let reply = frame_snapshot_reply(
                                    request_id,
                                    reply_status,
                                    &reply_reason,
                                );
                                freeze_coord_virtio_con
                                    .lock()
                                    .queue_input_port1(&reply);
                            }
                            crate::vmm::wire::SNAPSHOT_KIND_WATCH => {
                                if coord_kaslr_offset == 0
                                    && owned_accessor.is_none()
                                {
                                    tracing::info!(
                                        request_id,
                                        %tag,
                                        "freeze-coord: TLV WATCH deferred \
                                         (kaslr_offset not yet resolved)"
                                    );
                                    freeze_coord_on_demand_in_flight
                                        .store(false, Ordering::Release);
                                    capture_requests_deferred.push(SnapshotRequest {
                                        request_id,
                                        kind,
                                        tag,
                                    });
                                    continue;
                                }
                                tracing::info!(
                                    request_id,
                                    %tag,
                                    "freeze-coord: TLV WATCH request"
                                );
                                // Reply path branches on whether the
                                // cached vmlinux symbol map is available.
                                // The fall-through (no `continue`) lets
                                // the user-watchpoint loop and the
                                // late-trigger handler later in this
                                // iteration still run, so a WATCH that
                                // cannot resolve does not stall an
                                // already-pending err_triggered dump
                                // for a full poll interval.
                                let (status, reason) = match freeze_coord_symbol_cache.as_ref() {
                                    None => (
                                        crate::vmm::wire::SNAPSHOT_STATUS_ERR,
                                        "vmlinux symbol cache unavailable \
                                         (vmlinux not found or parse failed at \
                                         coord init)"
                                            .to_string(),
                                    ),
                                    Some(symbol_cache) => {
                                        // Pass the bsp_alive Arc by
                                        // reference so each BSP-touching
                                        // site inside `arm_user_watchpoint`
                                        // (the BSP `ie.set` and the BSP
                                        // `pthread_kill`) issues its own
                                        // fresh Acquire load immediately
                                        // before the syscall. A bool
                                        // snapshot taken here would be
                                        // stale by the time the kick
                                        // pass reaches the BSP — long
                                        // enough for the BSP run-loop to
                                        // publish `false` (Release) and
                                        // drop its `VcpuFd`, leaving a
                                        // `true`-snapshot writing through
                                        // freed kvm_run mmap pages.
                                        // `run_vm` flips bsp_alive to
                                        // false only AFTER joining the
                                        // coordinator (see `bsp_alive`
                                        // in run_vm), so a `true`
                                        // reading inside the helper is
                                        // load-bearing for the BSP
                                        // kvm_run mmap's liveness.
                                        match arm_user_watchpoint(
                                            &freeze_coord_watchpoint,
                                            symbol_cache,
                                            &tag,
                                            coord_kaslr_offset,
                                            &freeze_coord_ap_pthreads,
                                            &freeze_coord_ap_ies,
                                            &freeze_coord_ap_alive,
                                            freeze_coord_bsp_tid,
                                            freeze_coord_bsp_ie_handle.as_ref(),
                                            &bsp_alive_for_coord,
                                        ) {
                                            Ok(slot_idx) => {
                                                tracing::info!(
                                                    request_id,
                                                    %tag,
                                                    slot_idx,
                                                    "freeze-coord: hardware watchpoint armed"
                                                );
                                                (
                                                    crate::vmm::wire::SNAPSHOT_STATUS_OK,
                                                    String::new(),
                                                )
                                            }
                                            Err(reason) => {
                                                tracing::warn!(
                                                    request_id,
                                                    %tag,
                                                    %reason,
                                                    "freeze-coord: WATCH register failed"
                                                );
                                                (
                                                    crate::vmm::wire::SNAPSHOT_STATUS_ERR,
                                                    reason,
                                                )
                                            }
                                        }
                                    }
                                };
                                let reply = frame_snapshot_reply(
                                    request_id,
                                    status,
                                    &reason,
                                );
                                freeze_coord_virtio_con
                                    .lock()
                                    .queue_input_port1(&reply);
                            }
                            unknown => {
                                tracing::warn!(
                                    request_id,
                                    %tag,
                                    kind = unknown,
                                    "freeze-coord: TLV snapshot request with unknown kind"
                                );
                                let reply = frame_snapshot_reply(
                                    request_id,
                                    crate::vmm::wire::SNAPSHOT_STATUS_ERR,
                                    &format!("unknown snapshot kind {unknown}"),
                                );
                                freeze_coord_virtio_con
                                    .lock()
                                    .queue_input_port1(&reply);
                            }
                        }
                        freeze_coord_on_demand_in_flight
                            .store(false, Ordering::Release);
                    }
                    // Periodic-capture cadence runs BEFORE the
                    // user-watchpoint dispatch below so periodic
                    // boundaries get priority over Op::Snapshot /
                    // Op::WatchSnapshot fires when both contend for
                    // the same `freeze_coord_on_demand_in_flight`
                    // gate. Iteration ordering within the body:
                    // TLV CAPTURE runs first (request-reply,
                    // self-throttling); periodic runs second with
                    // priority over user-watchpoint hits.
                    // Lazily compute the boundary list once
                    // `num_snapshots > 0`, the workload duration is
                    // known, and the first ScenarioStart has been
                    // stamped — then on every iteration check
                    // whether `now` has crossed the next un-fired
                    // boundary, and fire a host-side
                    // `freeze_and_capture(false)` for each crossed
                    // boundary. Reuses the same gate
                    // (`freeze_coord_on_demand_in_flight`) the
                    // TLV CAPTURE / user-watchpoint paths use —
                    // when the gate is held the boundary is
                    // deferred to the next iteration rather than
                    // skipped, so a burst of on-demand captures
                    // cannot cause a missed periodic sample. The
                    // 10% / 10% pre/post buffers in the boundary
                    // formula are the budget that absorbs this
                    // deferral lag.
                    if freeze_coord_num_snapshots > 0 && !periodic_abandoned {
                        if periodic_boundaries_ns.is_none()
                            && let Some(workload_d) = workload_duration_for_coord
                        {
                            let scenario_anchor =
                                scenario_start_ns_for_coord.load(Ordering::Relaxed);
                            if scenario_anchor != 0 {
                                let boundaries = compute_periodic_boundaries_ns(
                                    scenario_anchor,
                                    workload_d,
                                    freeze_coord_num_snapshots,
                                );
                                tracing::info!(
                                    target: "ktstr::failure_dump",
                                    num_snapshots = freeze_coord_num_snapshots,
                                    scenario_anchor_ns = scenario_anchor,
                                    workload_duration_ns = workload_d.as_nanos() as u64,
                                    "freeze-coord: periodic snapshot boundaries computed"
                                );
                                periodic_boundaries_ns = Some(boundaries);
                            }
                        }
                        if let Some(ref boundaries) = periodic_boundaries_ns {
                            // Drain every crossed boundary in this
                            // iteration. `now_ns` is recomputed at
                            // the top of every inner-loop iteration
                            // so a mid-drain ScenarioPause /
                            // ScenarioResume pair (a single periodic
                            // capture can run for several seconds
                            // through the parked-vCPU rendezvous)
                            // shifts un-fired boundaries forward as
                            // soon as the cumulative pause atomic
                            // updates.
                            loop {
                                if (next_periodic_idx as usize) >= boundaries.len() {
                                    break;
                                }
                                let raw_now_ns =
                                    u64::try_from(run_start.elapsed().as_nanos())
                                        .unwrap_or(u64::MAX);
                                let cumulative_pause = scenario_pause_cumulative_for_coord
                                    .load(Ordering::Acquire);
                                let in_flight_pause_at = watchdog_pause_for_coord
                                    .load(Ordering::Acquire);
                                let in_flight_pause = if in_flight_pause_at > 0 {
                                    raw_now_ns.saturating_sub(in_flight_pause_at)
                                } else {
                                    0
                                };
                                let now_ns = raw_now_ns
                                    .saturating_sub(cumulative_pause)
                                    .saturating_sub(in_flight_pause);
                                if boundaries[next_periodic_idx as usize] > now_ns {
                                    break;
                                }
                                if freeze_coord_kill.load(Ordering::Acquire) {
                                    break;
                                }
                                if freeze_coord_on_demand_in_flight
                                    .swap(true, Ordering::AcqRel)
                                {
                                    // Gate held — defer (do NOT
                                    // skip): leave next_periodic_idx
                                    // as-is so the next iteration
                                    // retries this same boundary
                                    // once the gate clears.
                                    tracing::info!(
                                        target: "ktstr::failure_dump",
                                        idx = next_periodic_idx,
                                        tag = %periodic_tag(next_periodic_idx),
                                        "freeze-coord: periodic snapshot deferred \
                                         (in-flight gate held by another capture); \
                                         retrying next iteration"
                                    );
                                    break;
                                }
                                let tag = periodic_tag(next_periodic_idx);
                                tracing::info!(
                                    target: "ktstr::failure_dump",
                                    idx = next_periodic_idx,
                                    %tag,
                                    "freeze-coord: periodic snapshot boundary crossed"
                                );
                                // Request scx_stats BEFORE the freeze
                                // rendezvous so the scheduler's
                                // userspace thread is still alive to
                                // service the request. Failure modes
                                // (no scheduler, relay error, non-
                                // zero envelope errno) all collapse
                                // to `None` — the parallel stats
                                // slot stays absent and the
                                // temporal-stats projection surfaces
                                // a per-sample missing-stats failure
                                // the test author can opt to ignore.
                                let stats_value: Option<serde_json::Value> =
                                    if let Some(ref client) = freeze_coord_stats_client {
                                        match client.stats(&[]) {
                                            Ok(v) => Some(v),
                                            Err(e) => {
                                                tracing::debug!(
                                                    target: "ktstr::failure_dump",
                                                    %tag,
                                                    error = %e,
                                                    "freeze-coord: periodic stats request \
                                                     failed; bundling None into Sample"
                                                );
                                                None
                                            }
                                        }
                                    } else {
                                        None
                                    };
                                // Sample timestamp anchor = the moment
                                // the stats request COMPLETED (or
                                // failed). Captured AFTER the stats
                                // client returns so the value
                                // reflects when the running
                                // scheduler's stats were observed,
                                // NOT when we entered the
                                // periodic-fire branch. Stats and
                                // BPF freeze can be ~50 ms apart;
                                // the stats-completion timestamp is
                                // the authoritative anchor for the
                                // sample because the JSON content
                                // was observed at this instant. The
                                // BPF state captured by the freeze
                                // that follows is observed up to
                                // FREEZE_RENDEZVOUS_TIMEOUT later.
                                //
                                // Pause-adjusted: subtract cumulative
                                // ScenarioPause/Resume pause time and
                                // any in-flight pause currently
                                // running, mirroring the boundary
                                // check above. Without this, a
                                // scenario that pauses (e.g. for a
                                // multi-second on-demand capture)
                                // would advance the elapsed_ms
                                // anchor through the pause window
                                // and the temporal patterns would
                                // see false-positive rate drops as
                                // the workload appears to "skip" a
                                // window of progress.
                                let anchor_raw_now_ns =
                                    u64::try_from(run_start.elapsed().as_nanos())
                                        .unwrap_or(u64::MAX);
                                let anchor_cumulative_pause =
                                    scenario_pause_cumulative_for_coord
                                        .load(Ordering::Acquire);
                                let anchor_in_flight_pause_at =
                                    watchdog_pause_for_coord.load(Ordering::Acquire);
                                let anchor_in_flight_pause =
                                    if anchor_in_flight_pause_at > 0 {
                                        anchor_raw_now_ns
                                            .saturating_sub(anchor_in_flight_pause_at)
                                    } else {
                                        0
                                    };
                                let sample_elapsed_ms_anchor = anchor_raw_now_ns
                                    .saturating_sub(anchor_cumulative_pause)
                                    .saturating_sub(anchor_in_flight_pause)
                                    / 1_000_000;
                                let freeze_start = Instant::now();
                                let on_demand = freeze_and_capture(false);
                                thaw_and_barrier();
                                extend_watchdog_for_freeze(freeze_start);
                                if let Some((report, capture_start)) = on_demand {
                                    let map_count = report.maps.len();
                                    let vcpu_regs_count = report.vcpu_regs.len();
                                    let tasks_enriched = report.task_enrichments.len();
                                    if let Some(ref base_path) = freeze_coord_dump_path {
                                        let tagged =
                                            snapshot_tagged_path(base_path, &tag);
                                        if let Some(parent) = tagged.parent() {
                                            let _ = std::fs::create_dir_all(parent);
                                        }
                                        if let Ok(json) =
                                            serde_json::to_string(&report)
                                            && let Err(e) =
                                                std::fs::write(&tagged, &json)
                                        {
                                            tracing::warn!(
                                                path = %tagged.display(),
                                                error = %e,
                                                "freeze-coord: periodic dump file write failed"
                                            );
                                        }
                                    }
                                    let elapsed_ms =
                                        capture_start.elapsed().as_millis() as u64;
                                    tracing::info!(
                                        target: "ktstr::failure_dump",
                                        kind = "periodic",
                                        idx = next_periodic_idx,
                                        %tag,
                                        map_count,
                                        vcpu_regs_count,
                                        tasks_enriched,
                                        elapsed_ms,
                                        stats_present = stats_value.is_some(),
                                        sample_elapsed_ms = sample_elapsed_ms_anchor,
                                        "freeze-coord: periodic snapshot captured"
                                    );
                                    freeze_coord_snapshot_bridge.store_with_stats(
                                        &tag,
                                        report,
                                        stats_value,
                                        Some(sample_elapsed_ms_anchor),
                                    );
                                    // Successful capture resets the
                                    // consecutive-timeout counter so
                                    // a transient rendezvous miss
                                    // does not arm the abandon
                                    // threshold for unrelated future
                                    // boundaries.
                                    periodic_consecutive_timeouts = 0;
                                } else {
                                    tracing::warn!(
                                        idx = next_periodic_idx,
                                        %tag,
                                        "freeze-coord: periodic capture failed \
                                         (freeze_and_capture returned None — most \
                                         commonly a parked-vCPU rendezvous \
                                         timeout); storing placeholder report"
                                    );
                                    let placeholder =
                                        crate::monitor::dump::FailureDumpReport::placeholder(
                                            "freeze rendezvous timed out",
                                        );
                                    // Even when the freeze fails the
                                    // pre-freeze stats response (when
                                    // available) plus the workload-
                                    // relative timestamp ARE valid —
                                    // they sample the running
                                    // scheduler and the wall-clock
                                    // instant at which we attempted
                                    // the boundary. Bundle them into
                                    // the placeholder so a Sample
                                    // view at least carries the
                                    // stats axis and timing for this
                                    // boundary; the BPF axis falls
                                    // through to the placeholder
                                    // report and any temporal
                                    // pattern projecting BPF data
                                    // surfaces it as the upstream
                                    // missing-data error variant.
                                    freeze_coord_snapshot_bridge.store_with_stats(
                                        &tag,
                                        placeholder,
                                        stats_value,
                                        Some(sample_elapsed_ms_anchor),
                                    );
                                    periodic_consecutive_timeouts =
                                        periodic_consecutive_timeouts
                                            .saturating_add(1);
                                }
                                freeze_coord_on_demand_in_flight
                                    .store(false, Ordering::Release);
                                next_periodic_idx =
                                    next_periodic_idx.saturating_add(1);
                                // Publish the live fire count so
                                // run_vm can read it after the
                                // coordinator joins and forward
                                // onto VmResult::periodic_fired.
                                periodic_fired_for_coord.store(
                                    next_periodic_idx,
                                    Ordering::Relaxed,
                                );
                                // After PERIODIC_TIMEOUT_ABANDON_THRESHOLD
                                // consecutive rendezvous timeouts the
                                // remaining boundaries are unlikely
                                // to produce useful captures — every
                                // fire costs up to
                                // FREEZE_RENDEZVOUS_TIMEOUT (30 s)
                                // of wall-clock waiting on a wedged
                                // guest. Set the abandon flag and
                                // break the inner drain; the outer
                                // periodic guard short-circuits on
                                // the next iteration.
                                if periodic_consecutive_timeouts
                                    >= PERIODIC_TIMEOUT_ABANDON_THRESHOLD
                                    && !periodic_abandoned
                                {
                                    let remaining = boundaries
                                        .len()
                                        .saturating_sub(next_periodic_idx as usize);
                                    tracing::warn!(
                                        target: "ktstr::failure_dump",
                                        consecutive_timeouts =
                                            periodic_consecutive_timeouts,
                                        threshold =
                                            PERIODIC_TIMEOUT_ABANDON_THRESHOLD,
                                        remaining_boundaries = remaining,
                                        "freeze-coord: periodic capture abandoned \
                                         after {} consecutive rendezvous timeouts \
                                         ({} boundaries skipped)",
                                        periodic_consecutive_timeouts,
                                        remaining,
                                    );
                                    periodic_abandoned = true;
                                    break;
                                }
                            }
                        }
                    }
                    // After every TLV-driven snapshot dispatch path
                    // runs, also service any user-watchpoint hits on
                    // slots 1..=3.
                    // The vCPU's KVM_EXIT_DEBUG handler latches the
                    // matching slot's `hit` flag and writes hit_evt;
                    // the coordinator's epoll fires WATCHPOINT, the
                    // hit_evt drain at the top of the loop already
                    // ran. Walk every slot and dispatch a capture
                    // for each hit.
                    for slot_idx in 0..3 {
                        if !freeze_coord_watchpoint.user[slot_idx]
                            .hit
                            .swap(false, Ordering::AcqRel)
                        {
                            continue;
                        }
                        let tag = freeze_coord_watchpoint.user[slot_idx]
                            .tag
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .clone();
                        if freeze_coord_on_demand_in_flight
                            .swap(true, Ordering::AcqRel)
                        {
                            // A capture is already in flight (e.g.
                            // a CAPTURE-class TLV request still
                            // holds the gate). Re-arm the slot's
                            // hit flag so a subsequent iteration
                            // services it, and write a fresh
                            // `hit_evt` edge so the outer
                            // `epoll.wait` wakes promptly — the
                            // hit_evt drain at the top of this
                            // iteration consumed the original wake,
                            // and without a new edge the re-armed
                            // hit could sit for the full
                            // POLL_TIMEOUT_MS before re-inspection.
                            // `continue` (rather than `break`) so
                            // OTHER slots in the same iteration
                            // still get checked — each slot's
                            // `hit` is independent (per-slot
                            // hardware watchpoint dispatch), so a
                            // gate-blocked slot N must not strand
                            // an unrelated fire on slot N+1
                            // waiting for the next iteration's
                            // wake. The outer loop's next iteration
                            // re-evaluates the gate and either
                            // services the re-armed slot or hits
                            // the same in-flight branch and
                            // re-arms again — bounded by the
                            // single-threaded freeze coordinator's
                            // serial dispatch of CAPTURE/WATCH,
                            // which always clears the gate before
                            // returning here.
                            freeze_coord_watchpoint.user[slot_idx]
                                .hit
                                .store(true, Ordering::Release);
                            let _ = freeze_coord_watchpoint.hit_evt.write(1);
                            continue;
                        }
                        tracing::info!(
                            slot_idx,
                            %tag,
                            "freeze-coord: user watchpoint fire; capturing"
                        );
                        // User watchpoint has no while-frozen work,
                        // so thaw immediately.
                        let on_demand = freeze_and_capture(false);
                        thaw_and_barrier();
                        if let Some((report, capture_start)) = on_demand {
                            let map_count = report.maps.len();
                            // File mirror via `&report` (no clone),
                            // then move the report into the bridge.
                            // See the CAPTURE-class TLV handler
                            // above for the full rationale on the
                            // serialize-then-store ordering and the
                            // `to_string` vs `to_string_pretty`
                            // tradeoff.
                            if let Some(ref base_path) = freeze_coord_dump_path
                            {
                                let tagged =
                                    snapshot_tagged_path(base_path, &tag);
                                if let Some(parent) = tagged.parent() {
                                    let _ = std::fs::create_dir_all(parent);
                                }
                                if let Ok(json) =
                                    serde_json::to_string(&report)
                                    && let Err(e) = std::fs::write(&tagged, &json)
                                {
                                    tracing::warn!(
                                        path = %tagged.display(),
                                        error = %e,
                                        "freeze-coord: user-watchpoint dump file write failed"
                                    );
                                }
                            }
                            let elapsed_ms =
                                capture_start.elapsed().as_millis() as u64;
                            tracing::info!(
                                target: "ktstr::failure_dump",
                                kind = "user_watchpoint",
                                slot_idx,
                                %tag,
                                map_count,
                                elapsed_ms,
                                "freeze-coord: user-watchpoint snapshot captured"
                            );
                            freeze_coord_snapshot_bridge.store(&tag, report);
                        } else {
                            // Rendezvous timeout (or any other path
                            // through `freeze_and_capture` that
                            // returns None). Without an entry on the
                            // bridge here, the user's
                            // `Op::WatchSnapshot` fire is silently
                            // lost: the in-loop `hit.swap(false)`
                            // above already cleared the latch, so the
                            // teardown final-drain placeholder loop
                            // (search for `final-drain placeholder`
                            // in this file) skips this slot too.
                            // Publish a degraded placeholder under
                            // the same tag so a test that registered
                            // `Op::WatchSnapshot` sees an entry on
                            // the bridge with an `_unavailable`
                            // reason instead of a missing snapshot
                            // that's indistinguishable from "the
                            // watched KVA was never written." Mirrors
                            // the teardown placeholder's shape with
                            // a different reason string so an
                            // operator can tell the two paths apart.
                            tracing::warn!(
                                slot_idx,
                                %tag,
                                "freeze-coord: user-watchpoint capture failed \
                                 (freeze_and_capture returned None — most \
                                 commonly a parked-vCPU rendezvous timeout); \
                                 storing placeholder report"
                            );
                            let placeholder =
                                crate::monitor::dump::FailureDumpReport::placeholder(
                                    "freeze rendezvous timed out",
                                );
                            freeze_coord_snapshot_bridge.store(&tag, placeholder);
                        }
                        // Release the slot for future arm requests.
                        // `arm_user_watchpoint` finds a free slot by
                        // `request_kva.load(Acquire) == 0`; without
                        // clearing here every fire permanently consumes
                        // its slot, exhausting the cap of three after
                        // three captures and rejecting subsequent
                        // `Op::WatchSnapshot` arms with "no free slot".
                        // Clear `request_kva` and `tag` together so
                        // `arm_user_watchpoint`'s tag publish ordering
                        // (tag first, then `request_kva` Release) sees
                        // a clean slot; vCPU `self_arm_watchpoint` calls
                        // observe the zeroed `request_kva` next iteration
                        // and re-issue `KVM_SET_GUEST_DEBUG` without
                        // this slot's DR/WCR enable so the now-stale
                        // KVA stops trapping. `Release` pairs with the
                        // `Acquire` in `arm_user_watchpoint`'s free-slot
                        // search and the per-vCPU `self_arm_watchpoint`
                        // load.
                        {
                            let mut tag_guard = freeze_coord_watchpoint
                                .user[slot_idx]
                                .tag
                                .lock()
                                .unwrap_or_else(|e| e.into_inner());
                            tag_guard.clear();
                        }
                        freeze_coord_watchpoint.user[slot_idx]
                            .request_kva
                            .store(0, Ordering::Release);
                        freeze_coord_on_demand_in_flight
                            .store(false, Ordering::Release);
                    }
                    // Once the late snapshot has been emitted, the
                    // coordinator's only remaining job is to keep
                    // the freeze=false invariant clear, service
                    // any pending TLV snapshot requests, and wait
                    // for teardown. Skip the error-trigger paths
                    // below; the next `epoll.wait` at the top of
                    // the loop blocks until kill / bsp_done /
                    // virtio-console TX / watchpoint / scanner
                    // tick — no separate sleep cadence needed.
                    // Goes AFTER the snapshot-request dispatch so
                    // on-demand captures still service post-Done.
                    if freeze_state == FreezeState::Done {
                        continue;
                    }
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
                    if scan_tick
                        && freeze_state == FreezeState::Idle
                        && freeze_coord_dual_snapshot
                        && half_threshold_jiffies > 0
                        && let Some(ref ctx) = scan_ctx
                        && let Some(ref mem) = freeze_coord_mem
                    {
                        let jiffies = mem.read_u64(ctx.jiffies_64_pa, 0);
                        let max_age = crate::monitor::runnable_scan::max_runnable_age(
                            mem,
                            ctx.scx_tasks_kva,
                            &ctx.rq_pas,
                            &ctx.offsets,
                            jiffies,
                            ctx.walk,
                            ctx.watchdog_timestamp_pa,
                            ctx.start_kernel_map,
                            ctx.phys_base,
                        );
                        // Track scan trajectory for the diagnostic
                        // logged when err_triggered fires before the
                        // early path captures. peak survives across
                        // iterations even when each individual
                        // max_age dips back to 0 (a task on the list
                        // gets dispatched between two polls), so an
                        // operator viewing the post-hoc warn sees the
                        // closest the run came to tripping the
                        // threshold.
                        early_scan_iters = early_scan_iters.wrapping_add(1);
                        if max_age > early_peak_max_age_jiffies {
                            early_peak_max_age_jiffies = max_age;
                        }
                        if max_age >= half_threshold_jiffies
                            && !freeze_coord_bsp_done.load(Ordering::Acquire)
                        {
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
                            // Early-trigger only persists the report;
                            // the timing summary line is emitted at
                            // the late-trigger emit_json site (which
                            // is where JSON serialisation happens).
                            // Discarding the early `_capture_start`
                            // avoids a separate timing log for the
                            // early path that would not include
                            // json_bytes.
                            // Early trigger uses runnable_at age as
                            // its precondition; exit_kind has not
                            // necessarily been written yet, so pass
                            // `false` to skip the gate. Early
                            // snapshot has no while-frozen work, so
                            // thaw immediately after the dump
                            // returns (whether or not it produced a
                            // report — a stuck rendezvous already
                            // logged inside the closure).
                            if let Some((report, _capture_start)) =
                                freeze_and_capture(false)
                            {
                                early_max_age_jiffies = max_age;
                                early_threshold_jiffies = half_threshold_jiffies;
                                early_snapshot = Some(report);
                            }
                            thaw_and_barrier();
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
                        // When dual-snapshot mode is on but the early
                        // path never captured, surface why so the
                        // operator can act without re-running with
                        // RUST_LOG=ktstr=debug. The three diagnoses
                        // (no scan_ctx, scan ran but always-zero,
                        // scan ran but never crossed threshold) map
                        // to distinct fixes: the first points at
                        // missing kernel symbols / BTF, the second
                        // points at offset/translation bugs in the
                        // scan, the third points at err-class exits
                        // that aren't watchdog stalls (where there
                        // is no half-way state to capture). The warn
                        // fires only when state is genuinely Idle —
                        // a successful TookEarly path has already
                        // logged at info level above.
                        if freeze_coord_dual_snapshot
                            && freeze_state == FreezeState::Idle
                        {
                            tracing::warn!(
                                early_scan_iters,
                                early_peak_max_age_jiffies,
                                half_threshold_jiffies,
                                scan_ctx_resolved = scan_ctx.is_some(),
                                "freeze-coord: dual-snapshot late firing without \
                                 early — runnable_at scan never crossed half-way \
                                 threshold (peak_max_age vs half_threshold tells \
                                 you which case: 0 peak with 0 iters = scan_ctx \
                                 unresolved; 0 peak with non-zero iters = scan ran \
                                 but found no aged tasks; non-zero peak under \
                                 threshold = err-class exit fired before stall \
                                 progressed past half-way)"
                            );
                        }
                        // Gate the dump on `*scx_root->exit_kind`
                        // when the watchpoint was the trigger. The
                        // hardware watchpoint catches every write,
                        // including transient init/teardown writes
                        // setting kind to NONE/DONE; gating on
                        // `kind >= 1024` (SCX_EXIT_ERROR boundary)
                        // suppresses those false positives. The BPF
                        // bss path is its own gate (the tp_btf
                        // handler only latches on error-class kinds),
                        // so when bss alone fired the gate is
                        // redundant and we let the dump run
                        // unconditionally — `bss_state == Triggered`
                        // already proves kind >= 1024.
                        let watchpoint_only_trigger =
                            compute_watchpoint_only_trigger(
                                watchpoint_hit, bss_state,
                            );
                        let late_capture =
                            freeze_and_capture(watchpoint_only_trigger);
                        // Late-trigger backstop: while guest memory
                        // is still quiesced (vCPUs parked, virtio-blk
                        // worker paused, freeze flag still set), do a
                        // final runnable_at scan and — if it crosses
                        // the threshold and the early snapshot never
                        // captured — clone the just-captured late
                        // report into the early slot. The early and
                        // late slots end up as identical snapshots in
                        // that case, but the wrapper's
                        // `early_max_age_jiffies` /
                        // `early_threshold_jiffies` fields tell the
                        // consumer the trigger condition was met at
                        // freeze time, and the wrapper Display
                        // surfaces "early=present" rather than
                        // "early=absent" so an operator inspecting a
                        // stall dump sees the runnable_at evidence
                        // even when the host coordinator's poll
                        // cadence missed the half-way crossing.
                        //
                        // The backstop runs unconditionally on a
                        // quiesced guest — same memory the dump just
                        // captured — so a positive max_age here is
                        // ground truth for "tasks were stuck on
                        // runnable_list at the error-exit instant",
                        // not a transient observation that could have
                        // dipped before the next poll. Functionally
                        // independent of (and complementary to) the
                        // per-poll early trigger above: the per-poll
                        // path captures the half-way moment; the
                        // backstop captures the late-instant ground
                        // truth.
                        let mut backstop_max_age: u64 = 0;
                        if freeze_coord_dual_snapshot
                            && early_snapshot.is_none()
                            && half_threshold_jiffies > 0
                            && let Some((ref late, _)) = late_capture
                            && let Some(ref ctx) = scan_ctx
                            && let Some(ref mem) = freeze_coord_mem
                        {
                            let jiffies = mem.read_u64(ctx.jiffies_64_pa, 0);
                            backstop_max_age =
                                crate::monitor::runnable_scan::max_runnable_age(
                                    mem,
                                    ctx.scx_tasks_kva,
                                    &ctx.rq_pas,
                                    &ctx.offsets,
                                    jiffies,
                                    ctx.walk,
                                    ctx.watchdog_timestamp_pa,
                                    ctx.start_kernel_map,
                                    ctx.phys_base,
                                );
                            if backstop_max_age >= half_threshold_jiffies {
                                tracing::info!(
                                    backstop_max_age,
                                    half_threshold_jiffies,
                                    "freeze-coord: late-trigger backstop \
                                     promoting late capture to early slot \
                                     (per-poll early path missed the \
                                     half-way crossing — runnable_at scan \
                                     of frozen guest memory shows the \
                                     stall was real)"
                                );
                                early_snapshot = Some(late.clone());
                                early_max_age_jiffies = backstop_max_age;
                                early_threshold_jiffies = half_threshold_jiffies;
                            }
                        }
                        // Compute the structured early-skip reason
                        // BEFORE thaw, while the relevant state
                        // (peak, threshold, scan_ctx, skip_reason) is
                        // current. The reason is consumed when
                        // building the DualFailureDumpReport below; a
                        // None means "early was captured" or
                        // "single-snapshot mode" — the dual wrapper
                        // serializes None via skip_serializing_if so
                        // a populated `early` keeps the JSON tight.
                        let early_skipped_reason: Option<String> =
                            if !freeze_coord_dual_snapshot
                                || early_snapshot.is_some()
                            {
                                None
                            } else if let Some(reason) = scan_ctx_skip_reason {
                                Some(format!(
                                    "scan prerequisites unavailable: {reason}"
                                ))
                            } else if early_peak_max_age_jiffies == 0
                                && backstop_max_age == 0
                            {
                                Some(
                                    "scx_tick stall — no per-task \
                                     runnable_at data".to_string(),
                                )
                            } else {
                                Some(format!(
                                    "max_age never crossed threshold \
                                     (peak={early_peak_max_age_jiffies}j, \
                                     threshold={half_threshold_jiffies}j)"
                                ))
                            };
                        // Thaw before emission so a slow JSON
                        // serialise doesn't keep vCPUs parked any
                        // longer than the dump strictly needs. The
                        // backstop above (dual-snapshot only) ran
                        // while still frozen, so the backstop's
                        // runnable_at scan saw the same quiesced
                        // memory the dump captured — thawing here
                        // is safe because every site that depends
                        // on quiesced state has completed.
                        thaw_and_barrier();
                        // Re-read both trigger flags AFTER
                        // freeze_and_capture returned. The capture
                        // path can sit in rendezvous up to the
                        // configured watchdog (~30 s) while vCPUs
                        // ack SIGRTMIN; during that window the BPF
                        // tp_btf handler running on a not-yet-parked
                        // vCPU can latch ktstr_err_exit_detected in
                        // .bss (sticky kernel-side), and another
                        // vCPU's hardware watchpoint can fire on a
                        // fresh exit_kind write. The
                        // suppression-vs-Done decision below must
                        // use post-rendezvous truth: a
                        // mid-rendezvous bss flip means the kernel
                        // latch will keep reporting Triggered, and
                        // taking the watchpoint-only suppression
                        // path (reset hit, keep watching) would
                        // re-fire the late-trigger every iteration
                        // forever (re-rendezvous, re-suppress, ...
                        // — the original bug). The pre-rendezvous
                        // value at the freeze_and_capture call site
                        // above is still correct for the
                        // gate_on_exit_kind argument: that gate
                        // filters spurious init/teardown writes,
                        // which is independent of whether the bss
                        // latch flipped during the rendezvous.
                        // Acquire ordering matches the iteration-
                        // top reads — paired with the vCPU-thread
                        // Release on `hit`. The bss read goes
                        // through the same bss_read_state helper as
                        // the iteration-top read; parked vCPUs at
                        // the time of the post-thaw read are
                        // already running again, but the kernel-
                        // side bss latch is monotonic-rising
                        // (probe.bpf.c only stores 1, never clears),
                        // so any flip observed at this point will
                        // remain observable on subsequent reads.
                        let watchpoint_hit_post =
                            freeze_coord_watchpoint.hit.load(Ordering::Acquire);
                        let bss_state_post = bss_read_state(
                            freeze_coord_mem.as_deref(),
                            cached_bss_pa,
                        );
                        let watchpoint_only_trigger_post =
                            compute_watchpoint_only_trigger(
                                watchpoint_hit_post,
                                bss_state_post,
                            );
                        // Branch on three outcomes:
                        //   Some(...)             → dump, mark Done
                        //   None + watchpoint-only → gate-suppressed
                        //                            (or rendezvous
                        //                            timeout); reset
                        //                            `watchpoint.hit`
                        //                            and DO NOT mark
                        //                            Done so the
                        //                            coordinator keeps
                        //                            watching for an
                        //                            error-class
                        //                            exit_kind
                        //   None + bss-or-mixed    → rendezvous timed
                        //                            out under a
                        //                            sticky bss
                        //                            latch; mark Done
                        //                            because the
                        //                            kernel-side
                        //                            latch isn't
                        //                            going to retract
                        match late_capture {
                            Some((late, capture_start)) => {
                                // capture_start anchors the freeze→emit
                                // timing summary; emit_json reads
                                // Instant::now() - capture_start at log
                                // time so it covers serialise + write.
                                let map_count = late.maps.len();
                                let vcpu_regs_count = late.vcpu_regs.len();
                                let tasks_enriched = late.task_enrichments.len();
                                let truncated_at_us = late.dump_truncated_at_us;
                                // `to_string` (compact) replaces
                                // `to_string_pretty` to halve
                                // serialization cost on the hot
                                // failure-dump path. JSON consumers
                                // (sidecar tooling, repro probe) all
                                // parse via serde_json which
                                // tolerates either form identically.
                                let json_result = if freeze_coord_dual_snapshot {
                                    let dual = crate::monitor::dump::DualFailureDumpReport {
                                        schema: crate::monitor::dump::SCHEMA_DUAL
                                            .to_string(),
                                        early: early_snapshot.take(),
                                        late,
                                        early_max_age_jiffies,
                                        early_threshold_jiffies,
                                        early_skipped_reason,
                                    };
                                    serde_json::to_string(&dual)
                                } else {
                                    serde_json::to_string(&late)
                                };
                                match json_result {
                                    Ok(json) => emit_json(
                                        &json,
                                        map_count,
                                        vcpu_regs_count,
                                        tasks_enriched,
                                        capture_start.elapsed().as_millis() as u64,
                                        truncated_at_us,
                                    ),
                                    Err(e) => tracing::error!(
                                        error = %e,
                                        map_count,
                                        vcpu_regs_count,
                                        "freeze-coord: failure dump (JSON serialization failed)"
                                    ),
                                }
                                freeze_state = FreezeState::Done;
                                // Error-class exit dump complete: tear
                                // the run down immediately rather than
                                // looping back to epoll_wait under EEVDF
                                // fallback for the remainder of the
                                // host-watchdog window. The dump is
                                // already serialized and emitted above,
                                // the probe ringbuf has drained by the
                                // time sched_ext_exit fired, and serial
                                // output is flushed — no useful work
                                // remains in the post-exit window. Set
                                // the run-level kill AtomicBool and kick
                                // the eventfd so the BSP run loop
                                // (kill.load) and this coord loop
                                // (freeze_coord_kill.load at line 2026)
                                // both observe the edge on the next
                                // wake.
                                tracing::info!(
                                    "freeze-coord: kill triggered after \
                                     error-exit dump capture"
                                );
                                freeze_coord_kill
                                    .store(true, Ordering::Release);
                                let _ = freeze_coord_kill_evt.write(1);
                            }
                            None if watchpoint_only_trigger_post => {
                                freeze_coord_watchpoint
                                    .hit
                                    .store(false, Ordering::Release);
                                freeze_state = FreezeState::Done;
                            }
                            None => {
                                // bss-triggered with rendezvous
                                // timeout (or a bss flip that
                                // happened DURING the rendezvous —
                                // the post-rendezvous re-read above
                                // catches that case and routes here
                                // instead of the watchpoint-only
                                // arm). The bss latch is sticky on
                                // the kernel side; retrying would
                                // just hit the same timeout. Mark
                                // Done and let the run end normally.
                                freeze_state = FreezeState::Done;
                            }
                        }
                        continue;
                    }
                    // End of body. Loop back to the `epoll.wait`
                    // at the top, which blocks until any registered
                    // fd fires (kill, bsp_done, virtio-console TX,
                    // watchpoint hit, scanner tick) or
                    // POLL_TIMEOUT_MS elapses.
                    // The watchpoint hit and bss-pa edges are
                    // delivered as eventfd writes from the vCPU
                    // thread, so the trigger latency is bounded by
                    // epoll_wait's microsecond-scale wakeup, NOT by
                    // any host-side polling cadence. Heavy work
                    // (boot-race accessor construction, scan_ctx
                    // resolve, runnable_at scan) remains gated on
                    // `scan_tick`, which only fires on the SCANNER
                    // timerfd edge (every 100 ms).
                }
                // Final drain of any pending user-watchpoint hits.
                // The hot-path for-loop at the end of each
                // coordinator iteration handles slot[i].hit fires
                // synchronously, but two race windows can leave a
                // hit `true` past loop exit:
                //
                //   1. The "already in flight" branch in the
                //      hot-path for-loop re-arms the slot's `hit`
                //      and `break`s when `on_demand_in_flight` is
                //      true on entry. If kill / bsp_done flips
                //      before the next iteration runs, the
                //      re-armed hit is never serviced.
                //   2. A vCPU's `latch_user_hit` Release that
                //      raced the loop exit (kill flipped between
                //      the for-loop terminating and the next
                //      `epoll.wait`).
                //
                // Without this drain the snapshot the test author
                // requested is silently dropped — `Snapshot::watch`
                // produces no entry, which a passing test
                // misinterprets as "the watched address was never
                // written" instead of "the VMM exited before the
                // capture pipeline serviced the fire". Store a
                // "watch-fired-but-coord-exited" placeholder under
                // the slot's tag so the test's lookup gets a
                // distinguishable result. Same minimal-report
                // shape the in-loop "dump prerequisites
                // unavailable" partial path uses, with a
                // dedicated reason string so consumers can tell
                // the two cases apart.
                for slot_idx in 0..freeze_coord_watchpoint.user.len() {
                    if !freeze_coord_watchpoint.user[slot_idx]
                        .hit
                        .swap(false, Ordering::AcqRel)
                    {
                        continue;
                    }
                    let tag = freeze_coord_watchpoint.user[slot_idx]
                        .tag
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .clone();
                    // Skip the placeholder entirely when the bridge
                    // already has a real report under this tag. The
                    // in-loop dispatch publishes via
                    // `snapshot_bridge.store(&tag, report)`; a vCPU
                    // re-arm of `hit=true` after that successful
                    // publish (e.g. a second guest write to the
                    // watched KVA in the same tag, or a vCPU
                    // dispatch racing the in-loop hit.swap) leaves
                    // the slot's hit flag set at coord exit. Without
                    // this guard the final drain stomps the
                    // already-published real report with a hollow
                    // "coord exited before capture" placeholder,
                    // which a test misinterprets as "the watchpoint
                    // mostly didn't fire" rather than "the watch
                    // fired AND was captured." The has() lookup
                    // takes the bridge mutex briefly; teardown is
                    // single-threaded with no concurrent store
                    // (every vCPU thread joins AFTER this drain
                    // returns), so the check is race-free.
                    if freeze_coord_snapshot_bridge.has(&tag) {
                        tracing::debug!(
                            slot_idx,
                            %tag,
                            "freeze-coord: user-watchpoint fire pending at coord \
                             exit, but the bridge already has a real report under \
                             this tag — skipping placeholder to preserve the \
                             captured report"
                        );
                        continue;
                    }
                    tracing::warn!(
                        slot_idx,
                        %tag,
                        "freeze-coord: user-watchpoint fire pending at coord exit; \
                         storing placeholder report (no capture possible during \
                         teardown — vCPU rendezvous would race teardown joins)"
                    );
                    let placeholder = crate::monitor::dump::FailureDumpReport::placeholder(
                        "coord exited before capture",
                    );
                    freeze_coord_snapshot_bridge.store(&tag, placeholder);
                }
                // Post-drain advisory: vCPU threads (BSP + APs) are
                // still alive at this point — they only join inside
                // `collect_results` after the coord thread closure
                // returns (see `run_vm` join sequencing: coord first
                // via `freeze_coord_handle.join()`, AP threads later
                // via `wait_for_exit` + `handle.join` inside
                // `collect_results`). Any vCPU that calls
                // `latch_user_hit` between the drain loop above and
                // its eventual join will set `hit = true` AND
                // increment `hit_evt`, but the coordinator's epoll
                // is already gone — nothing services that hit. The
                // count of slots whose `request_kva != 0` here is
                // the upper bound on hits that could still be lost
                // (each such slot is currently armed in
                // KVM_SET_GUEST_DEBUG on every vCPU and capable of
                // firing on the next guest write to its KVA). This
                // warn surfaces the observability gap so an operator
                // who finds a missing snapshot in
                // `Snapshot::watch_results` can tell "VMM lost the
                // hit during teardown" from "guest never wrote to
                // the watched KVA". Acquire load is overkill (the
                // armed slot publication uses Release / vCPU
                // self-arm uses Acquire) but cheap.
                let still_armed = freeze_coord_watchpoint
                    .user
                    .iter()
                    .filter(|slot| slot.request_kva.load(Ordering::Acquire) != 0)
                    .count();
                if still_armed > 0 {
                    tracing::warn!(
                        still_armed,
                        "freeze-coord: post-drain teardown advisory — {still_armed} \
                         user-watchpoint slot(s) remain armed on every vCPU at \
                         coord exit. Hits latched by a vCPU between this drain \
                         and the eventual vCPU join in collect_results are NOT \
                         serviced (the coord epoll is already gone). Tests \
                         observing a missing snapshot in Snapshot::watch_results \
                         should treat this warn as evidence that the watched \
                         address WAS written to, just past the host-side \
                         capture window."
                    );
                }
                // Flush any partial-frame bytes the bulk_assembler
                // is still buffering back into the device's
                // `port1_tx_buf`. The assembler retains tail bytes
                // when a TLV frame straddles two TX wakes — without
                // this push-back the residual is dropped on the
                // floor when the assembler is dropped at closure
                // exit, and `collect_results`'s end-of-run
                // `drain_bulk` + `parse_tlv_stream` path never sees
                // them. Pushing them back means
                // `collect_results`'s drain returns the residual
                // alongside any bytes the device accumulated after
                // the last coordinator drain, and `parse_tlv_stream`
                // completes the frame.
                let coord_exit_t = std::time::Instant::now();
                eprintln!("CLEANUP: coord loop exited");
                // Periodic-capture teardown summary. When
                // `num_snapshots > 0`, log the fired/total ratio so
                // an operator reading the test's tracing output can
                // tell at a glance whether the periodic-sampling
                // path delivered. Three distinct shapes surface:
                //   * `0/N` with no scenario_start_ns stamp — the
                //     guest never published a CRC-valid
                //     `MSG_TYPE_SCENARIO_START`, so boundaries were
                //     never computed. Most commonly a guest that
                //     crashed mid-boot or a workload that never
                //     reached the host-comms phase.
                //   * `0/N` with scenario_start_ns stamped — the
                //     boundaries were computed but no boundary was
                //     reached (very-short run, kill before first
                //     boundary). The doc on
                //     `KtstrTestEntry::num_snapshots` warns the
                //     test author to assert `>= some_lower_bound`
                //     rather than `== num_snapshots` exactly for
                //     this case.
                //   * `K/N` with `K < N` — the run terminated mid-
                //     sequence. Same best-effort contract; tests
                //     should assert `>= K` not `== N`.
                if freeze_coord_num_snapshots > 0 {
                    let scenario_anchor =
                        scenario_start_ns_for_coord.load(Ordering::Acquire);
                    if scenario_anchor == 0 {
                        tracing::warn!(
                            target: "ktstr::failure_dump",
                            num_snapshots = freeze_coord_num_snapshots,
                            fired = next_periodic_idx,
                            "freeze-coord: 0/{} periodic snapshots fired — \
                             scenario_start_ns never stamped (no CRC-valid \
                             MSG_TYPE_SCENARIO_START observed). The guest most \
                             likely crashed mid-boot or never reached the \
                             host-comms phase; periodic sampling has no anchor",
                            freeze_coord_num_snapshots,
                        );
                    } else {
                        tracing::info!(
                            target: "ktstr::failure_dump",
                            num_snapshots = freeze_coord_num_snapshots,
                            fired = next_periodic_idx,
                            scenario_anchor_ns = scenario_anchor,
                            "freeze-coord: {}/{} periodic snapshots fired",
                            next_periodic_idx,
                            freeze_coord_num_snapshots,
                        );
                    }
                }
                let residual = bulk_assembler.take_residual();
                if !residual.is_empty() {
                    freeze_coord_virtio_con.lock().push_back_bulk(&residual);
                }
                // Drop the borrowed views into the OnceLock-owned
                // accessor pair before joining the init worker. The
                // worker itself never touches these references — it
                // only writes to the OnceLock — but explicitly
                // dropping makes the Arc reference-count transitions
                // visible at one site instead of leaving the borrows
                // implicitly alive across the join.
                let _ = owned_accessor;
                let _ = owned_prog_accessor;
                // Join the accessor-init worker before the closure
                // returns. The worker holds an `Arc<GuestMem>` whose
                // host pointer addresses `vm.guest_mem`; that mapping
                // is dropped right after run_vm joins the freeze
                // coordinator thread (`freeze_coord_handle.join()`
                // in run_vm), so any worker still running past this
                // join would dereference freed memory through stale
                // `Arc<GuestMem>` on its next `try_init_*` retry.
                // The kill flag was flipped by the run-loop tear-down
                // path; the worker honors it between retries and
                // exits within ~100 ms (its sleep interval). On the
                // happy path the worker has long since published the
                // OnceLock and exited, so the join is a no-op.
                if let Some(handle) = accessor_init_handle {
                    let jt = std::time::Instant::now();
                    let _ = handle.join();
                    eprintln!("CLEANUP: accessor-init worker joined {:?}", jt.elapsed());
                }
                // Extract the prog accessor for collect_verifier_stats
                // and stash it in the shared slot so run_vm can pass
                // it to VmRunState.
                {
                    let slot = &prog_accessor_slot_for_coord;
                    let extracted = Arc::try_unwrap(accessors_oncelock)
                        .ok()
                        .and_then(|lock| lock.into_inner())
                        .and_then(|(_map, prog)| prog);
                    *slot.lock().unwrap_or_else(|e| e.into_inner()) = extracted;
                }
                eprintln!("CLEANUP: coord closure done {:?}", coord_exit_t.elapsed());
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
                // Cached scheduler-attach reset deadline. Decoded
                // lazily from `watchdog_reset_for_wd` after the
                // host monitor stores a non-zero value (the
                // moment `*scx_root` flips from null to non-null
                // in guest memory). `None` means the workload's
                // clock has not started yet, so the original
                // `hard_deadline` (counted from VM boot) still
                // applies. Once `Some(reset)`, the effective
                // deadline becomes the reset value
                // (`reset_deadline.unwrap_or(hard_deadline)` — no
                // min clamp), so boot-time delays do not eat
                // into the workload budget. Cached so the per-tick
                // check is a single compare against
                // `effective_deadline` rather than re-decoding
                // the encoded `Duration::from_nanos` form.
                // Computed only when `workload_duration_for_wd`
                // is set; absent, the load is skipped entirely
                // (no workload duration → nothing to reset to).
                let mut reset_deadline: Option<Instant> = None;
                eprintln!("watchdog: started, timeout={timeout:?}");

                // Wake plumbing. `tick_tfd` is a periodic 100 ms
                // timerfd that drives the deadline-progress checks
                // (matches the legacy `thread::sleep(100ms)` cadence
                // exactly). `kill_evt_for_watchdog` and
                // `bsp_done_evt_for_wd` are fast-wake fds bumped by
                // the kill / bsp_done setters so the deadline-arm
                // path runs within microseconds of the flip rather
                // than at the next 100 ms tick. Construction failure
                // for any of these means the watchdog cannot
                // observe wake signals; surface as `tracing::error`
                // and return so the symptom is visible — the
                // deadline-armed BSP still gets kicked by the
                // freeze coordinator's own paths if those fire.
                let mut tick_tfd = match TimerFd::new() {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::error!(err = %e, "watchdog: timerfd_create failed");
                        return;
                    }
                };
                let tick = Duration::from_millis(100);
                if let Err(e) = tick_tfd.reset(tick, Some(tick)) {
                    tracing::error!(err = %e, "watchdog: timerfd_settime failed");
                    return;
                }
                let epoll = match Epoll::new() {
                    Ok(e) => e,
                    Err(e) => {
                        tracing::error!(err = %e, "watchdog: epoll_create1 failed");
                        return;
                    }
                };
                let tick_fd = tick_tfd.as_raw_fd();
                let kill_fd = kill_evt_for_watchdog.as_raw_fd();
                let bsp_done_fd = bsp_done_evt_for_wd.as_raw_fd();
                if let Err(e) = epoll.ctl(
                    ControlOperation::Add,
                    tick_fd,
                    EpollEvent::new(EventSet::IN, tick_fd as u64),
                ) {
                    tracing::error!(err = %e, "watchdog: epoll_ctl add timerfd failed");
                    return;
                }
                if let Err(e) = epoll.ctl(
                    ControlOperation::Add,
                    kill_fd,
                    EpollEvent::new(EventSet::IN, kill_fd as u64),
                ) {
                    tracing::error!(err = %e, "watchdog: epoll_ctl add kill_evt failed");
                    return;
                }
                if let Err(e) = epoll.ctl(
                    ControlOperation::Add,
                    bsp_done_fd,
                    EpollEvent::new(EventSet::IN, bsp_done_fd as u64),
                ) {
                    tracing::error!(err = %e, "watchdog: epoll_ctl add bsp_done_evt failed");
                    return;
                }
                let mut epoll_buf = [EpollEvent::default(); 3];

                loop {
                    if bsp_done_for_wd.load(Ordering::Acquire) {
                        eprintln!("watchdog: BSP done, returning");
                        return;
                    }
                    // Decode a pending scheduler-attach reset
                    // when not already cached. Skip when the
                    // workload duration was not configured (no
                    // distinct workload budget; `hard_deadline`
                    // is the only deadline that matters).
                    if workload_duration_for_wd.is_some() {
                        let stored_ns = watchdog_reset_for_wd.load(Ordering::Acquire);
                        if stored_ns != 0 {
                            let candidate = run_start
                                .checked_add(Duration::from_nanos(stored_ns))
                                .unwrap_or(hard_deadline);
                            if reset_deadline.is_none()
                                || reset_deadline.is_some_and(|prev| candidate > prev)
                            {
                                reset_deadline = Some(candidate);
                                eprintln!(
                                    "watchdog: scheduler attach observed, hard \
                                     deadline reset to {:?} from VM start",
                                    candidate.saturating_duration_since(run_start),
                                );
                            }
                        }
                    }
                    let effective_deadline =
                        reset_deadline.map_or(hard_deadline, |r| r.max(hard_deadline));
                    if kill_for_watchdog.load(Ordering::Acquire)
                        || Instant::now() >= effective_deadline
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
                        let hard_timeout_fired = Instant::now() >= effective_deadline;
                        let reason = if hard_timeout_fired {
                            "hard timeout expired"
                        } else {
                            "kill set by AP"
                        };
                        eprintln!("watchdog: {reason}, kicking BSP");
                        // Actionable diagnostics. Without this dump the
                        // operator-visible failure is just `timed_out =
                        // true` with no clue why. Print the deadline
                        // values (`effective_deadline` is what actually
                        // fired; `hard_deadline` is the original boot-
                        // anchored deadline before any
                        // scheduler-attach reset) plus the
                        // `timeout`/`workload_duration` knobs the
                        // operator can tune, the cause path
                        // (hard-timeout-expired vs kill-set-by-AP),
                        // and whether the deadline was reset. Both
                        // deadlines are rendered as offsets from
                        // `run_start` so the numbers line up with the
                        // wall-clock the operator sees in the test
                        // output. The `kill_set_by_AP` branch also
                        // ages `effective_deadline` against now so
                        // the operator can see how much budget was
                        // unused when the kill arrived.
                        let now = Instant::now();
                        let effective_offset =
                            effective_deadline.saturating_duration_since(run_start);
                        let hard_offset = hard_deadline.saturating_duration_since(run_start);
                        let elapsed = now.saturating_duration_since(run_start);
                        let was_reset = reset_deadline.is_some();
                        eprintln!("watchdog: deadline expired at {elapsed:?} from VM start");
                        eprintln!(
                            "  cause={reason}, hard_timeout_fired={hard_timeout_fired}, \
                             kill_set_by_AP={}",
                            !hard_timeout_fired
                        );
                        eprintln!(
                            "  effective_deadline={effective_offset:?} from VM start \
                             (reset_by_scheduler_attach={was_reset})"
                        );
                        eprintln!("  hard_deadline={hard_offset:?} from VM start (timeout knob)");
                        eprintln!(
                            "  timeout={timeout:?}, workload_duration={:?}",
                            workload_duration_for_wd
                        );
                        eprintln!(
                            "  hint: if the test body needs more wall time, increase \
                             duration (the `duration` field on `KtstrTestEntry` / \
                             `#[ktstr_test(duration_ms = ...)]`); the VM timeout is \
                             derived as max(watchdog_timeout, duration) so raising \
                             duration also extends the host watchdog deadline"
                        );
                        // Set `timed_out` ONLY for the hard-deadline
                        // branch. The "kill set by AP" path is not a
                        // watchdog timeout — propagating it as
                        // `timed_out=true` would mislabel a panic-
                        // driven kill as a deadline expiry.
                        if hard_timeout_fired {
                            timed_out_for_watchdog.store(true, Ordering::Release);
                        }
                        // Propagate kill so handle_freeze's poll loop
                        // exits and the monitor + bpf-write threads stop.
                        kill_for_watchdog.store(true, Ordering::Release);
                        let _ = kill_evt_for_watchdog.write(1);
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
                    // Soft deadline: request graceful shutdown by
                    // pushing `SIGNAL_VC_SHUTDOWN` into virtio-console
                    // RX. The guest's `hvc0_poll_loop` blocks on
                    // `/dev/hvc0` and recognises the byte directly —
                    // no SHM signal slot needed. The BSP keeps running
                    // so the guest can flush serial and reboot
                    // normally.
                    //
                    // Recompute the soft window from the effective
                    // deadline so a scheduler-attach reset shifts
                    // the soft deadline alongside the hard
                    // deadline. The reset can extend past
                    // hard_deadline (no min clamp), so the
                    // recomputed `effective_deadline - 3s` shifts
                    // forward whenever the reset extends; the
                    // guest still gets its 3s flush window
                    // relative to the deadline that actually
                    // fires. Skip when the original
                    // `soft_deadline` was `None` (timeout < 5s;
                    // no soft phase configured) — the reset path
                    // inherits that decision rather than
                    // synthesising a soft phase out of nothing.
                    let effective_soft = soft_deadline
                        .and_then(|_| effective_deadline.checked_sub(Duration::from_secs(3)));
                    if !soft_fired && effective_soft.is_some_and(|d| Instant::now() >= d) {
                        soft_fired = true;
                        eprintln!("watchdog: soft deadline, requesting graceful shutdown");
                        super::host_comms::request_shutdown(&wd_virtio_con);
                    }
                    // Block until the next tick or a kill_evt /
                    // bsp_done_evt write. -1 timeout: deadlines
                    // (hard + soft) are checked at the top of each
                    // iteration after the wake; the 100 ms timerfd
                    // guarantees the loop wakes at least that often
                    // even when no eventfd writes arrive, which
                    // preserves the legacy cadence exactly.
                    match epoll.wait(-1, &mut epoll_buf) {
                        Ok(n) => {
                            for ev in &epoll_buf[..n] {
                                if ev.fd() == tick_fd {
                                    // Drain the timerfd counter so
                                    // the next epoll_wait blocks
                                    // again instead of returning
                                    // immediately on the residual
                                    // ready bit.
                                    let _ = tick_tfd.wait();
                                }
                                // kill_fd / bsp_done_fd: implicitly
                                // drained because the loop body
                                // re-loads the AtomicBool source of
                                // truth on every iteration. The
                                // EventFd counter accumulates but
                                // is harmless — we only care about
                                // the edge.
                            }
                        }
                        Err(e) => {
                            if e.raw_os_error() != Some(libc::EINTR) {
                                tracing::warn!(err = %e, "watchdog: epoll_wait failed");
                                // Fall through to the next iteration
                                // so the deadline check still runs;
                                // a persistent failure is eventually
                                // caught by the hard deadline.
                            }
                        }
                    }
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
                kill_evt: Some(kill_evt.clone()),
                exited_evt: Some(bsp_done_evt.clone()),
                // Hand the BSP's `bsp_alive` flag to the panic hook so a
                // panic-unwind path flips it to `false` BEFORE the
                // stack drop unmaps `bsp`'s `kvm_run` page. The
                // normal-exit path's post-join store at line 5344
                // covers `panic = "abort"` and the no-panic path; the
                // panic hook covers `panic = "unwind"` (test profile)
                // where the post-join store is unreachable. Mirrors
                // the AP-side `alive: Some(alive.clone())` plumbing in
                // spawn_ap_threads — every cross-thread holder of a
                // BSP `ImmediateExitHandle` (the freeze coordinator,
                // the watchdog) gates `ie.set` on this flag's
                // Acquire load, and a panic-released Release store
                // happens-before the unwind drop of `bsp`.
                alive: Some(bsp_alive.clone()),
            },
            || {
                self.run_bsp_loop(
                    &mut bsp,
                    &com1,
                    &com2,
                    Some(&virtio_con),
                    virtio_blk.as_ref(),
                    virtio_net.as_ref(),
                    &kill,
                    &freeze,
                    &watchpoint,
                    &bsp_parked,
                    &bsp_regs,
                    has_immediate_exit,
                    run_start,
                    timeout,
                    Some(&parked_evt),
                    Some(&thaw_evt),
                    Some(&kill_evt),
                    tcr_el1_cache.as_ref(),
                    &cr3_cache,
                    &timed_out_flag,
                )
            },
        );
        bsp_done.store(true, Ordering::Release);
        // Wake the freeze coordinator's epoll loop. Failure
        // (counter overflow / EAGAIN under EFD_NONBLOCK) is benign
        // — the panic-hook path may have already pushed an edge,
        // and the AtomicBool above is still authoritative for
        // `freeze_coord_bsp_done.load(Acquire)` if the eventfd
        // fails to deliver.
        let _ = bsp_done_evt.write(1);
        // Stop the monitor (wakes via kill_evt epoll) and bpf-write
        // thread (observes kill on next 200ms poll cycle).
        // Previously kill was deferred to collect_results, leaving
        // the monitor sampling at 100ms cadence through the entire
        // run_vm cleanup window (watchdog join + coord join).
        kill.store(true, Ordering::Release);
        let _ = kill_evt.write(1);
        // Sample cleanup start at the earliest moment after BSP exit so
        // every host-side teardown step lands inside the window, in
        // execution order: watchdog join (immediately below), AP joins,
        // monitor join, BPF writer join, bulk drain, exit-code and
        // crash-message extraction, and verifier-stat read (the rest
        // run inside `collect_results`). `collect_results` reads
        // `Instant::now()` at the end and the difference becomes
        // `VmResult::cleanup_duration`.
        let cleanup_start = Instant::now();
        // `code` here is the run-loop sentinel (0 only on a BSP-
        // observed `ExitAction::Shutdown`, -1 otherwise — see
        // [`BspExitReason`] and the preceding `BSP: loop exit
        // reason=...` line). The caller-visible exit code is
        // derived from bulk-port `MSG_TYPE_EXIT` or the COM2 `KTSTR_EXIT:`
        // sentinel inside [`KtstrVm::collect_results`], not from
        // this value.
        eprintln!(
            "BSP: exited run loop, code={exit_code} timed_out={timed_out} \
             (run-loop sentinel — final exit code comes from bulk port / COM2 in collect_results)"
        );

        // Join the watchdog before dropping `bsp`. The watchdog holds an
        // ImmediateExitHandle pointing into bsp's kvm_run mmap. If bsp is
        // dropped first, the watchdog may write to unmapped memory.
        let _ = watchdog.join();
        eprintln!("CLEANUP: watchdog joined");

        // Join the freeze coordinator BEFORE `bsp` falls out of scope at
        // the end of this function. The coordinator's captured BSP
        // `ImmediateExitHandle` addresses bsp's kvm_run mmap; reachable
        // from multiple paths inside `freeze_and_capture` (TLV-driven
        // CAPTURE, user watchpoint, late-trigger, even after `bsp_done`
        // flips). Without this join, any of those paths can write
        // through a freed kvm_run mapping after bsp drops — a
        // use-after-free with hostile-input semantics.
        //
        // `bsp_done.store(true)` + `bsp_done_evt.write(1)` above
        // (lines around `BSP: exited run loop`) wake the coordinator's
        // epoll loop and break it out of the outer loop on the next
        // iteration, so this join does not deadlock; the watchdog's
        // own kill/bsp_done writes are also covered.
        //
        // Flip `bsp_alive` to `false` AFTER the join completes — at
        // that point the coordinator thread is gone and the gate is
        // belt-and-braces for any future restructuring that could
        // share the BSP IE handle outside this lifecycle.
        let _ = freeze_coord_handle.join();
        eprintln!("CLEANUP: freeze_coord joined");
        bsp_alive.store(false, Ordering::Release);

        // Make sure freeze is cleared before vCPU teardown so the APs
        // don't park-loop after we kick them. The freeze coordinator
        // has already joined above so it cannot re-set freeze=true.

        // Capture the virtio-blk counter Arc before the device's
        // outer `Arc<PiMutex<VirtioBlk>>` falls out of scope. The
        // device's `counters()` accessor clones the inner
        // `Arc<VirtioBlkCounters>`; this transfers a reader-side
        // handle onto `VmRunState` so `collect_results` can attach
        // it to `VmResult` without holding the device alive past
        // its current ownership.
        let virtio_blk_counters = virtio_blk.as_ref().map(|d| d.lock().counters());
        let virtio_net_counters = virtio_net.as_ref().map(|d| d.lock().counters());

        // Best-effort final TCR_EL1 read from the post-exit BSP.
        // The BSP loop's lazy CAS already populates `tcr_el1_cache`
        // via `read_tcr_el1`; this final read covers the (rare)
        // case where the loop exited before the kernel programmed
        // the MMU (early-boot crash). On x86_64 `read_tcr_el1`
        // returns None and the cache stays None.
        if let Some(ref cache) = tcr_el1_cache
            && cache.load(Ordering::Acquire) == 0
            && let Some(val) = exit_dispatch::read_tcr_el1(&mut bsp)
            && val != 0
        {
            cache.store(val, Ordering::Release);
        }
        // Best-effort final CR3 / TTBR1_EL1 read from the post-exit
        // BSP. Mirrors the TCR_EL1 catch-up above: the BSP loop's
        // lazy CAS populates `cr3_cache` once the kernel installs
        // its post-randomization page tables; this catch-up store
        // covers the (rare) case where the loop exited before
        // `__startup_64` / `__cpu_setup` ran. Failure-dump consumers
        // that read `cr3_cache` post-exit (e.g. for late
        // `phys_base` resolution against a frozen VM) get the live
        // CR3 instead of the bootstrap zero.
        if cr3_cache.load(Ordering::Acquire) == 0
            && let Some(val) = exit_dispatch::read_cr3(&mut bsp)
            && val != 0
        {
            cr3_cache.store(val, Ordering::Release);
        }

        Ok(VmRunState {
            exit_code,
            timed_out,
            ap_threads,
            monitor_handle,
            bpf_write_handle,
            // Coordinator is already joined above (before `bsp` drops)
            // to prevent UAF on the BSP `ImmediateExitHandle`.
            // `collect_results`'s `if let Some(h) = ...` join is a
            // no-op for the `None` arm.
            freeze_coordinator: None,
            com1,
            com2,
            kill,
            kill_evt,
            freeze,
            vm,
            cleanup_start,
            virtio_blk_counters,
            virtio_net_counters,
            // Snapshot bridge owning every report stored by the
            // freeze coordinator's TLV-driven snapshot handler
            // over the run's lifetime. Forwarded to
            // `VmResult::snapshot_bridge` by `collect_results`.
            snapshot_bridge,
            tcr_el1: tcr_el1_cache,
            cr3: cr3_cache,
            vmlinux_data: vmlinux_data_for_result,
            prog_accessor: prog_accessor_slot
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take(),
            kern_phys_base: kern_phys_base_for_result.load(Ordering::Acquire),
            // Virtio-console handle threaded into `collect_results`
            // for the post-exit `drain_bulk()` call. Carries any
            // port-1 TLV bytes the guest wrote that the freeze
            // coordinator's tx_evt-driven mid-run drain did not
            // already consume; the merge into `guest_messages` keeps
            // existing readers (eval.rs, sidecar) working without
            // any per-message-type code change.
            virtio_con,
            // Mid-run TLV entries the freeze coordinator already
            // consumed. `collect_results` merges these with the
            // post-exit bulk drain and the COM2 panic-message
            // extraction so every frame the guest published reaches
            // the verdict.
            bulk_messages: freeze_coord_bulk_messages,
            // Scheduler-stats client constructed at the top of
            // `run_vm`. Its drainer thread has been alive since the
            // guest started forwarding stats responses; the client
            // is threaded onto `VmResult` for test-code access.
            stats_client,
            // Periodic-capture count published by the coordinator
            // run-loop after every successful fire / placeholder
            // store. Read AFTER `freeze_coord_handle.join()` ran so
            // the AtomicU32's value is the final advance count;
            // `collect_results` forwards onto
            // `VmResult::periodic_fired`.
            periodic_fired: periodic_fired_slot.load(Ordering::Relaxed),
            // Configured periodic-target plumbed onto KtstrVm via
            // `KtstrVmBuilder::num_snapshots`. Forwarded to
            // `VmResult::periodic_target` so test code can compute
            // coverage as `fired / target`.
            periodic_target: self.num_snapshots,
            // Watchpoint Arc forwarded so `collect_results` can
            // invalidate `kind_host_ptr` and `request_kva` after
            // every vCPU thread joins but before `vm` drops.
            watchpoint,
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
        kill_evt: &Arc<EventFd>,
        freeze: &Arc<AtomicBool>,
        watchpoint: &Arc<WatchpointArm>,
        pin_targets: &[Option<usize>],
        no_perf_mask: Option<&[usize]>,
        ap_tid_slots: &[(Arc<AtomicI32>, Arc<crate::sync::Latch>)],
        parked_evt: Option<&Arc<EventFd>>,
        thaw_evt: Option<&Arc<EventFd>>,
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
            let kill_evt_clone = kill_evt.clone();
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
            // Per-AP `alive` flag mirroring the BSP `bsp_alive` gate.
            // Initialised to `true`; the AP panic hook (via
            // `VcpuPanicCtx::alive`) flips it to `false` BEFORE
            // unwinding drops `vcpu` and its `kvm_run` mmap, so the
            // freeze coordinator's pass-1 kick loop and the
            // `arm_user_watchpoint` kick gate every `ie.set` on a
            // fresh Acquire load and skip indices whose mmap is
            // about to disappear. Under `panic = "abort"` (release)
            // unwinding never runs and the flag stays `true` for
            // the life of the run; the gate is then a no-op,
            // matching the BSP belt-and-braces semantic.
            let alive = Arc::new(AtomicBool::new(true));
            let has_immediate_exit_clone = has_immediate_exit;
            let pin_cpu = pin_targets.get(i).copied().flatten();
            let mask_for_thread: Option<Vec<usize>> = no_perf_mask.map(|m| m.to_vec());
            // Per-AP shared watchpoint state. Cloned once per AP;
            // the AP polls `wp_clone.request_kva` before each
            // KVM_RUN (via the per-iteration hook in
            // `vcpu_run_loop_unified`) and self-arms via
            // [`self_arm_watchpoint`] when the freeze coordinator
            // publishes the resolved `*scx_root->exit_kind` KVA.
            // The same clone is what the `VcpuExit::Debug` arm in
            // [`exit_dispatch::classify_exit`] uses to set
            // `wp_clone.hit` so the late-trigger poll observes the
            // watchpoint fire.
            let wp_clone = watchpoint.clone();

            let rt = self.performance_mode;
            // Per-AP exit eventfd for `VcpuThread::wait_for_exit` so
            // teardown blocks in `epoll_wait` instead of sleep-polling
            // `exited`. Bumped from inside the closure right after
            // `exited.store(true)` and from the panic hook (via
            // `panic_ctx.exited_evt`) so the parent observes both
            // normal-exit and panic-classified shutdowns through the
            // same fd. EFD_NONBLOCK so a Drop-time write cannot
            // stall.
            let exit_evt =
                Arc::new(EventFd::new(EFD_NONBLOCK).context("create AP vCPU exit eventfd")?);
            let exit_evt_thread = Arc::clone(&exit_evt);
            let panic_ctx = vcpu_panic::VcpuPanicCtx {
                kill: kill.clone(),
                exited: exited.clone(),
                kill_evt: Some(kill_evt.clone()),
                exited_evt: Some(Arc::clone(&exit_evt)),
                // Hand the AP's `alive` flag to the panic hook so a
                // panic-unwind path flips it to `false` BEFORE the
                // stack drop unmaps `vcpu`'s `kvm_run` page. The
                // freeze coordinator's pass-1 kick gates each
                // `ie.set` on this flag's Acquire load.
                alive: Some(alive.clone()),
            };
            let (tid_slot_clone, tid_latch_clone) = {
                let (s, l) = &ap_tid_slots[i];
                (Arc::clone(s), Arc::clone(l))
            };
            // Clone the shared parked_evt + thaw_evt for this AP.
            // None when the caller (interactive shell) doesn't run a
            // freeze coordinator; in that case `vcpu_run_loop_unified`
            // never observes a freeze and the eventfd is unused.
            let parked_evt_clone: Option<Arc<EventFd>> = parked_evt.cloned();
            let thaw_evt_clone: Option<Arc<EventFd>> = thaw_evt.cloned();
            let handle = std::thread::Builder::new()
                .name(format!("vcpu-{}", i + 1))
                .spawn(move || {
                    register_vcpu_signal_handler();
                    // Stamp this thread's Linux TID into the per-AP
                    // slot so the monitor can open `perf_event_open`
                    // counters bound to the vCPU thread. Done
                    // BEFORE pinning / RT / KVM_RUN so the value is
                    // visible to any reader the moment the thread is
                    // schedulable. The companion `Latch::set` lets
                    // `open_vcpu_perf_capture` block in
                    // `Latch::wait_timeout` instead of sleep-polling
                    // the atomic. SAFETY: SYS_gettid is the standard
                    // syscall returning this thread's pid_t; no
                    // inputs.
                    let tid = unsafe { libc::syscall(libc::SYS_gettid) } as i32;
                    tid_slot_clone.store(tid, Ordering::Release);
                    tid_latch_clone.set();
                    if let Some(cpu) = pin_cpu {
                        pin_current_thread(cpu, &format!("vCPU {}", i + 1));
                    } else if let Some(mask) = mask_for_thread.as_deref() {
                        set_thread_cpumask(mask, &format!("vCPU {}", i + 1));
                    }
                    if rt {
                        set_rt_priority(1, &format!("vCPU {}", i + 1));
                    }
                    // The watchpoint Arc travels into the run loop
                    // via the `vcpu_run_loop_unified` parameter; the
                    // loop self-arms before each `vcpu.run()` and
                    // sets `watchpoint.hit` on `KVM_EXIT_DEBUG`. The
                    // per-AP `armed_kva` slot that tracks the
                    // currently-programmed `debugreg[0]` lives
                    // inside the loop now, so a single pre-loop
                    // attempt would have been a redundant ioctl
                    // with no effect — the coordinator typically
                    // publishes the resolved KVA AFTER the AP has
                    // entered the loop (once a sched_ext scheduler
                    // attaches and `*scx_root != 0`).
                    vcpu_panic::with_vcpu_panic_ctx(panic_ctx, || {
                        vcpu_run_loop_unified(
                            &mut vcpu,
                            &com1_clone,
                            &com2_clone,
                            vc_clone.as_ref(),
                            vblk_clone.as_ref(),
                            vnet_clone.as_ref(),
                            &kill_clone,
                            &kill_evt_clone,
                            &freeze_clone,
                            &parked_clone,
                            &regs_clone,
                            &wp_clone,
                            has_immediate_exit_clone,
                            parked_evt_clone.as_ref(),
                            thaw_evt_clone.as_ref(),
                        );
                    });
                    // wp_clone is held for the AP's entire lifetime
                    // so the strong count never drops to zero before
                    // the freeze coordinator joins.
                    drop(wp_clone);
                    exited_clone.store(true, Ordering::Release);
                    // Wake any thread blocked in `wait_for_exit` on
                    // this AP's exit_evt. Failure (counter overflow)
                    // is harmless — a previous edge already unblocks
                    // the waiter; only the edge from 0 to non-zero
                    // matters.
                    let _ = exit_evt_thread.write(1);
                    vcpu
                })
                .with_context(|| format!("spawn vCPU {} thread", i + 1))?;

            ap_threads.push(VcpuThread {
                handle,
                exited,
                immediate_exit: ie_handle,
                exit_evt,
                alive,
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
    ///
    /// `probes_ready_evt` is the broadcast EventFd shared with the
    /// bpf-map-write thread (see [`run_vm`]); the slot-1 wait below
    /// `poll`s it instead of bare-sleeping, and writes 1 to it on
    /// detection so any other waiter blocked in `poll` wakes
    /// immediately and re-checks its own readiness condition.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn start_monitor(
        &self,
        vm: &kvm::KtstrKvm,
        kill: &Arc<AtomicBool>,
        kill_evt: &Arc<EventFd>,
        run_start: Instant,
        vcpu_pthreads: Vec<libc::pthread_t>,
        perf_capture: Arc<Option<monitor::perf_counters::PerfCountersCapture>>,
        _probes_ready_evt: EventFd,
        virtio_con: Option<Arc<PiMutex<virtio_console::VirtioConsole>>>,
        sys_rdy_evt: Option<Arc<EventFd>>,
        tcr_el1: Option<Arc<std::sync::atomic::AtomicU64>>,
        cr3: Arc<std::sync::atomic::AtomicU64>,
        watchdog_reset_ns: Arc<std::sync::atomic::AtomicU64>,
        kern_phys_base_shared: Arc<std::sync::atomic::AtomicU64>,
        kern_phys_base_evt: Arc<EventFd>,
    ) -> Result<Option<JoinHandle<monitor::reader::MonitorLoopResult>>> {
        let Some(vmlinux) = find_vmlinux(&self.kernel) else {
            return Ok(None);
        };
        // Read the vmlinux bytes once and feed both the BTF loader
        // and the ELF symbol parser. The previous structure called
        // `load_btf_from_path` and `KernelSymbols::from_vmlinux` back
        // to back, each running its own `std::fs::read` — on a debug
        // vmlinux that is two ~1 GB reads through the page cache for
        // a single byte slice's worth of work.
        let vmlinux_data_arc = match super::vmlinux::cached_vmlinux_bytes(&vmlinux) {
            Some(d) => d,
            None => return Ok(None),
        };
        let vmlinux_data = &*vmlinux_data_arc;
        let elf = match goblin::elf::Elf::parse(vmlinux_data) {
            Ok(e) => e,
            Err(_) => return Ok(None),
        };
        // Single BTF parse for both `KernelOffsets` and
        // `BpfProgOffsets`. The previous structure parsed BTF twice
        // (KernelOffsets up here, BpfProgOffsets inside the spawned
        // monitor thread closure), each call hitting
        // `load_btf_from_path` and `Btf::from_bytes`. On debug-built
        // vmlinux the parse is hundreds of ms; doing it twice
        // pushed the monitor thread past the no-scheduler boot
        // window so early samples saw the rq's pre-AP-online state.
        // One parse, two `from_btf` consumers, both share the
        // resolved offsets. On a BTF sidecar cache hit the supplied
        // `elf` is unused; on a miss `load_btf_from_elf` reuses it
        // instead of running its own `Elf::parse`.
        let btf = match monitor::btf_offsets::load_btf_from_elf(&elf, &vmlinux_data, &vmlinux) {
            Ok(b) => b,
            Err(_) => return Ok(None),
        };
        let offsets = monitor::btf_offsets::KernelOffsets::from_btf(&btf);
        let prog_offsets = monitor::btf_offsets::BpfProgOffsets::from_btf(&btf).ok();
        let symbols = monitor::symbols::KernelSymbols::from_elf(&elf);

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
        let num_cpus = self.topology.total_cpus();
        let kill_clone = kill.clone();
        let kill_evt_clone = kill_evt.clone();
        // Clone the boot-complete eventfd handle for the monitor
        // closure. Captured by `move` into the spawned thread so
        // the `epoll_wait` dispatch can register the fd alongside
        // `kill_evt` and the timerfd. `Option::None` short-circuits
        // the pre-sample wait so the test path (no virtio-console)
        // and any `EventFd::new` failure both fall through to the
        // sample loop directly.
        let monitor_sys_rdy_evt = sys_rdy_evt.clone();
        let dump_trigger = self
            .monitor_thresholds
            .map(|thresholds| monitor::reader::DumpTrigger {
                thresholds,
                virtio_con: virtio_con.clone(),
            });

        let hz = monitor::guest_kernel_hz(Some(&self.kernel));
        // ms-precision conversion lives in [`duration_to_jiffies`];
        // see its doc for why the seconds-based form is wrong.
        let watchdog_jiffies = self.watchdog_timeout.map(|d| duration_to_jiffies(d, hz));
        let preemption_threshold_ns = monitor::vcpu_preemption_threshold_ns(Some(&self.kernel));
        let rt_monitor = self.performance_mode;
        let service_cpu = self.pinning_plan.as_ref().and_then(|p| p.service_cpu);
        // Workload duration captured for the scheduler-attach
        // watchdog reset. `Some(d)` enables the reset; the
        // monitor closure constructs a
        // [`monitor::reader::WatchdogReset`] payload pairing this
        // duration with the resolved `scx_root_pa` and the shared
        // `watchdog_reset_ns` atomic once `symbols.scx_root`
        // resolves below. `None` (the builder default) leaves
        // [`monitor::reader::MonitorConfig::watchdog_reset`] as
        // `None`, and the loop's reset detection short-circuits.
        let workload_duration = self.workload_duration;

        let handle = std::thread::Builder::new()
            .name("vmm-monitor".into())
            .spawn(move || {
                if let Some(cpu) = service_cpu {
                    pin_current_thread(cpu, "monitor");
                }
                if rt_monitor {
                    set_rt_priority(2, "monitor");
                }
                // Pre-resolution boot-complete wait, hoisted ABOVE
                // the `phys_base` / `pco_pa` / scx_root_pa /
                // watchdog_pa / `page_offset_base_pa` resolution
                // that follows. Previously this thread either
                // resolved `phys_base` immediately (with `cr3=0` →
                // `phys_base=0` → every KASLR text/data PA wrong)
                // or polled CR3 with a short busy-wait that fires
                // too early — CR3 is set in `__startup_64`, but
                // `setup_per_cpu_areas` (which populates
                // `__per_cpu_offset[]`) and KASLR randomization of
                // `page_offset_base` finish much later in
                // `start_kernel`. Resolving `pco_pa` /
                // `page_offset_base_pa` between those two events
                // produces baked-in stale PAs that the
                // per-iteration refresh inside `monitor_loop` cannot
                // recover from.
                //
                // The `MSG_TYPE_SYS_RDY` TLV frame is emitted by
                // `ktstr_guest_init` after `mount_filesystems()`
                // — strictly AFTER `__startup_64` (CR3 latch),
                // `__cpu_setup` (TCR_EL1 latch), `setup_per_cpu_areas`
                // (`__per_cpu_offset[]` populated), KASLR
                // randomization (`page_offset_base` populated), and
                // userspace init startup. By blocking here on the
                // sys_rdy eventfd, the resolution that follows runs
                // against a guest in steady state: every read in
                // `resolve_phys_base`, `resolve_page_offset_with_tcr`,
                // and the text-mapped PA recomputes lands on
                // populated guest memory.
                //
                // Three exit conditions:
                //   1. sys_rdy fires: proceed to phys_base resolve.
                //   2. kill fires: VM died before booting; return
                //      empty MonitorLoopResult immediately.
                //   3. 5 s timeout: best-effort fall through. The
                //      downstream `data_valid` gate inside
                //      `monitor_loop` still guards every walk, so
                //      reads of pre-boot zeros are tolerated and
                //      the monitor produces an empty sample set
                //      rather than chasing pointers through wrong
                //      PAs.
                //
                // `MonitorConfig::sys_rdy` is set to `None` below
                // because the wait has already happened here —
                // re-running the wait inside `monitor_loop` would
                // be a no-op (sys_rdy is edge-triggered, the eventfd
                // counter has been read by this wait and the
                // `Option::take` in the freeze-coord TOKEN_TX
                // handler also fires only once).
                if let Some(sys_rdy) = monitor_sys_rdy_evt.as_deref() {
                    use std::os::unix::io::AsRawFd;
                    use vmm_sys_util::epoll::{
                        ControlOperation, Epoll, EpollEvent, EventSet,
                    };
                    // Upfront kill check: BSP can exit before the
                    // monitor thread is scheduled (fast 1-CPU tests
                    // that fall through `test_main` in milliseconds).
                    // In that case `run_vm` has already stored
                    // kill + written kill_evt; entering the
                    // boot epoll would still wake immediately on
                    // kill_fd, but skipping the syscall trip
                    // entirely is cheaper and avoids the small
                    // window where epoll_create / epoll_ctl could
                    // race with VM teardown.
                    if kill_clone.load(std::sync::atomic::Ordering::Acquire) {
                        return monitor::reader::MonitorLoopResult {
                            samples: Vec::new(),
                            drain: crate::vmm::host_comms::BulkDrainResult {
                                entries: Vec::new(),
                            },
                            watchdog_observation: None,
                            page_offset: 0,
                            preemption_threshold_ns,
                        };
                    }
                    let kill_fd = kill_evt_clone.as_raw_fd();
                    let boot_fd = sys_rdy.as_raw_fd();
                    if let Ok(boot_epoll) = Epoll::new() {
                        let _ = boot_epoll.ctl(
                            ControlOperation::Add,
                            boot_fd,
                            EpollEvent::new(EventSet::IN, boot_fd as u64),
                        );
                        let _ = boot_epoll.ctl(
                            ControlOperation::Add,
                            kill_fd,
                            EpollEvent::new(EventSet::IN, kill_fd as u64),
                        );
                        let mut boot_buf = [EpollEvent::default(); 2];
                        // 5 s ceiling: a healthy guest emits SYS_RDY
                        // within ~3 s of boot; longer is a stuck
                        // guest. Tests that exit without sending
                        // SYS_RDY (e.g. early-init crash) must wait
                        // here only until either the eventfd fires
                        // or `run_vm` propagates the kill
                        // flag — the timeout is the fallback for
                        // the case where neither wake arrives, and
                        // tighter is better because the host VM
                        // teardown waits on this thread joining.
                        let _ = boot_epoll.wait(5_000, &mut boot_buf);
                    }
                    if kill_clone.load(std::sync::atomic::Ordering::Acquire) {
                        return monitor::reader::MonitorLoopResult {
                            samples: Vec::new(),
                            drain: crate::vmm::host_comms::BulkDrainResult {
                                entries: Vec::new(),
                            },
                            watchdog_observation: None,
                            page_offset: 0,
                            preemption_threshold_ns,
                        };
                    }
                }

                // Resolve the kernel image base. On x86_64 this is
                // the compile-time constant; on aarch64 it depends
                // on `VA_BITS_MIN` derived from `TCR_EL1.T1SZ` and
                // `TCR_EL1.TG1` (granule). After the sys_rdy wait
                // the BSP has executed many run-loop iterations
                // and the lazy CAS for `tcr_el1_cache` has fired
                // (kernel programs TCR_EL1 in `__cpu_setup` long
                // before userspace init runs).
                let tcr_el1_value = tcr_el1
                    .as_ref()
                    .map(|c| c.load(std::sync::atomic::Ordering::Acquire))
                    .unwrap_or(0);
                let start_kernel_map_for_thread =
                    monitor::symbols::start_kernel_map_for_tcr(tcr_el1_value)
                        .unwrap_or(monitor::symbols::START_KERNEL_MAP);

                // Resolve `phys_base` via a page-table walk through
                // the live BSP CR3. After the sys_rdy wait above the
                // BSP has populated `cr3_cache` via its lazy CAS —
                // `__startup_64` / `__cpu_setup` runs strictly before
                // userspace init emits SYS_RDY, and the BSP run loop
                // has executed thousands of iterations by then so
                // every iteration's CAS attempt has had a chance to
                // observe a non-zero CR3.
                //
                // Race window: in cold-cache or coverage-instrumented
                // builds the BSP can lag the kernel's userspace-init
                // emission of SYS_RDY by tens of ms — the freeze
                // coordinator promotes the SYS_RDY frame to the
                // monitor's eventfd from its TLV dispatch, which
                // runs on the freeze coord thread independent of
                // BSP run-loop progress. If the monitor reads
                // cr3_cache before the BSP has executed its first
                // run-loop iteration's lazy CAS, cr3_value is 0
                // and `phys_base` falls back to 0 for the entire
                // run. On KASLR builds that turns every text-mapped
                // PA derivation (pco_pa, scx_root_pa,
                // page_offset_base_pa) into an out-of-DRAM address;
                // every subsequent monitor read returns 0 and
                // `data_valid` never latches. Brief retry-until-
                // non-zero closes the window: by the time SYS_RDY
                // fires the BSP has been in the run loop for
                // hundreds of ms in steady state, so the vast
                // majority of paths return on the first load. The
                // 500-iteration cap (500 ms total) handles the
                // pathological case where the BSP genuinely never
                // populated the cache (early-boot crash, kill
                // before first iteration); in that case `phys_base`
                // falls back to 0 and the downstream `data_valid`
                // gate keeps every walk safe.
                // Wait for guest-reported phys_base. The guest
                // reads it from /proc/iomem and writes
                // `phys_base + 1` (biased +1 so the AtomicU64's
                // initial 0 means "not yet received") to
                // `/dev/vport0p2`; the host's virtio-console MMIO
                // handler captures the value inline on the vCPU
                // thread, independently of the TLV bulk port and the
                // freeze coordinator's iter1 vmlinux load. The 3 s
                // budget gives the kernel virtio_console driver time
                // to complete its multiport handshake under cold-
                // cache boots; on the fast path the value is already
                // visible by the time SYS_RDY fires. No KASLR/PTI
                // page-table walking is needed.
                let phys_base = {
                    let mut pb = 0u64;
                    let pb_fd = {
                        use std::os::unix::io::AsRawFd;
                        kern_phys_base_evt.as_raw_fd()
                    };
                    let kill_fd = {
                        use std::os::unix::io::AsRawFd;
                        kill_evt_clone.as_raw_fd()
                    };
                    for _ in 0..30 {
                        if kill_clone.load(std::sync::atomic::Ordering::Acquire) {
                            break;
                        }
                        let biased = kern_phys_base_shared.load(
                            std::sync::atomic::Ordering::Acquire,
                        );
                        if biased != 0 {
                            pb = biased.wrapping_sub(1);
                            break;
                        }
                        let cr3_val = cr3.load(std::sync::atomic::Ordering::Acquire);
                        if cr3_val != 0 {
                            let l5 = monitor::symbols::resolve_pgtable_l5(
                                &mem, &symbols, start_kernel_map_for_thread, 0,
                            );
                            if let Some(v) = monitor::symbols::resolve_phys_base(
                                &mem, &symbols, cr3_val, l5, tcr_el1_value,
                            ) {
                                if v != 0 {
                                    pb = v;
                                    break;
                                }
                            }
                        }
                        let mut pfds = [
                            libc::pollfd { fd: pb_fd, events: libc::POLLIN, revents: 0 },
                            libc::pollfd { fd: kill_fd, events: libc::POLLIN, revents: 0 },
                        ];
                        unsafe { libc::poll(pfds.as_mut_ptr(), 2, 100) };
                    }
                    pb
                };
                if phys_base != 0 {
                    let _ = kern_phys_base_shared.compare_exchange(
                        0,
                        phys_base.wrapping_add(1),
                        std::sync::atomic::Ordering::Release,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    let _ = kern_phys_base_evt.write(1);
                }

                // Derive kaslr_offset by reading the kernel's real
                // phys_base variable from guest memory. The guest
                // reported `output - LOAD_PHYSICAL_ADDR` which equals
                // `real_phys_base + kaslr_offset`. Reading the kernel's
                // own phys_base and subtracting recovers kaslr_offset.
                let kaslr_offset = if phys_base != 0
                    && let Some(pb_kva) = symbols.phys_base_kva
                {
                    let pb_pa = crate::monitor::symbols::text_kva_to_pa_with_base(
                        pb_kva,
                        start_kernel_map_for_thread,
                        phys_base,
                    );
                    let real_phys_base = mem.read_u64(pb_pa, 0);
                    if real_phys_base != 0 || phys_base == 0 {
                        phys_base.wrapping_sub(real_phys_base)
                    } else {
                        0
                    }
                } else {
                    0
                };

                // Kill check between sys_rdy wait and the long-tail
                // setup work below (page-table walks, watchdog override
                // resolve, post-wait re-resolve, BTF prog_offsets
                // consumption, monitor_loop entry). On debug builds
                // with cold caches the resolution path can spend
                // multiple seconds in `resolve_phys_base` /
                // `resolve_pgtable_l5` / `text_kva_to_pa_with_base`,
                // and `run_vm`'s `kill_evt.write(1)` cannot
                // interrupt code that is not blocked on epoll. Sample
                // the kill flag at every major boundary so a VM that
                // exits during setup tears the monitor down within
                // microseconds rather than having `monitor_handle.join`
                // block until the setup runs to completion.
                if kill_clone.load(std::sync::atomic::Ordering::Acquire) {
                    return monitor::reader::MonitorLoopResult {
                        samples: Vec::new(),
                        drain: crate::vmm::host_comms::BulkDrainResult {
                            entries: Vec::new(),
                        },
                        watchdog_observation: None,
                        page_offset: 0,
                        preemption_threshold_ns,
                    };
                }

                let page_offset = monitor::symbols::resolve_page_offset_with_tcr(
                    &mem,
                    &symbols,
                    start_kernel_map_for_thread,
                    tcr_el1_value,
                    phys_base,
                );

                // `__per_cpu_offset[]` lives in the kernel image
                // mapping (text PA). `setup_per_cpu_areas` in
                // `start_kernel` populates every slot before SMP
                // bringup IN THE GUEST — but the host monitor thread
                // spawns before the guest BSP enters KVM_RUN, so a
                // pre-loop one-shot read sees BSS zeros. Pass the
                // PAs that drive the recompute through `RqRefresh`
                // so the loop body re-reads each sample; see
                // [`monitor::reader::RqRefresh`].
                let pco_pa = monitor::symbols::text_kva_to_pa_with_base(
                    symbols.per_cpu_offset,
                    start_kernel_map_for_thread,
                    phys_base,
                );

                let watchdog_override = watchdog_jiffies.and_then(|jiffies| {
                    // 7.1+ path: deref scx_root -> scx_sched.watchdog_timeout.
                    if let Some((scx_root_kva, wd_offs)) = symbols
                        .scx_root
                        .zip(offsets.watchdog_offsets.as_ref())
                    {
                        let scx_root_pa = monitor::symbols::text_kva_to_pa_with_base(
                            scx_root_kva,
                            start_kernel_map_for_thread,
                            phys_base,
                        );
                        let resolve_pa = |kva| {
                            monitor::symbols::text_kva_to_pa_with_base(
                                kva,
                                start_kernel_map_for_thread,
                                phys_base,
                            )
                        };
                        let interval_pa = symbols.scx_watchdog_interval.map(&resolve_pa);
                        let timestamp_pa = symbols.scx_watchdog_timestamp.map(&resolve_pa);
                        let jiffies_64_pa = symbols.jiffies_64.map(&resolve_pa);
                        return Some(monitor::reader::WatchdogOverride::ScxSched {
                            scx_root_pa,
                            watchdog_offset: wd_offs.scx_sched_watchdog_timeout_off,
                            jiffies,
                            interval_pa,
                            timestamp_pa,
                            jiffies_64_pa,
                        });
                    }
                    if let Some(wdt_kva) = symbols.scx_watchdog_timeout {
                        let resolve_pa = |kva| {
                            monitor::symbols::text_kva_to_pa_with_base(
                                kva,
                                start_kernel_map_for_thread,
                                phys_base,
                            )
                        };
                        let watchdog_timeout_pa = resolve_pa(wdt_kva);
                        let interval_pa = symbols.scx_watchdog_interval.map(&resolve_pa);
                        let timestamp_pa = symbols.scx_watchdog_timestamp.map(&resolve_pa);
                        let jiffies_64_pa = symbols.jiffies_64.map(&resolve_pa);
                        return Some(monitor::reader::WatchdogOverride::StaticGlobal {
                            watchdog_timeout_pa,
                            jiffies,
                            interval_pa,
                            timestamp_pa,
                            jiffies_64_pa,
                        });
                    }
                    None
                });
                if watchdog_jiffies.is_some() && watchdog_override.is_none() {
                    tracing::warn!(
                        "no watchdog override path available — neither scx_sched.watchdog_timeout BTF field nor scx_watchdog_timeout symbol found"
                    );
                }

                // Kill check after watchdog override resolve. The
                // BTF / symbol-table lookups above can themselves
                // touch hundreds of kilobytes of vmlinux ELF, so a
                // VM that exits while we are still here would
                // otherwise have to wait for the entire setup tail
                // to drain before `monitor_handle.join` returns.
                if kill_clone.load(std::sync::atomic::Ordering::Acquire) {
                    return monitor::reader::MonitorLoopResult {
                        samples: Vec::new(),
                        drain: crate::vmm::host_comms::BulkDrainResult {
                            entries: Vec::new(),
                        },
                        watchdog_observation: None,
                        page_offset: 0,
                        preemption_threshold_ns,
                    };
                }

                // `event_pcpu_pas` derives from
                // `*scx_root -> scx_sched.pcpu` (or
                // `event_stats_cpu` on pre-6.18 kernels) plus
                // `__per_cpu_offset[]`. Both inputs change with VM
                // lifetime: `*scx_root` is null until a scheduler
                // attaches, and the percpu base table is BSS zero
                // until `setup_per_cpu_areas` runs. Stash the
                // text-mapped PA of `scx_root` plus the BTF offsets
                // and let the monitor loop refresh per-iteration.
                let event_refresh =
                    symbols
                        .scx_root
                        .zip(offsets.event_offsets.as_ref())
                        .map(|(scx_root_kva, ev)| {
                            let scx_root_pa = monitor::symbols::text_kva_to_pa_with_base(
                                scx_root_kva,
                                start_kernel_map_for_thread,
                                phys_base,
                            );
                            monitor::reader::EventRefresh {
                                scx_root_pa,
                                event_offsets: ev.clone(),
                            }
                        });
                // Scheduler-attach watchdog-reset PA, derived
                // independently of `event_refresh` so the reset
                // works on kernels without resolvable
                // `event_offsets` (e.g. older kernels lacking the
                // BTF struct, or stripped vmlinux). Always derives
                // from `symbols.scx_root` directly — the same
                // text-mapped global the kernel itself uses to
                // publish the active `scx_sched`. `None` when the
                // symbol could not be resolved (no scx support in
                // the kernel image, or `KernelSymbols::from_elf`
                // failed to find it); the loop's
                // `cfg.watchdog_reset` short-circuits in that
                // case.
                let scx_root_pa_for_reset = symbols.scx_root.map(|kva| {
                    monitor::symbols::text_kva_to_pa_with_base(
                        kva,
                        start_kernel_map_for_thread,
                        phys_base,
                    )
                });
                // `page_offset_base` is x86_64-only (a KASLR direct-map
                // base randomized by `CONFIG_RANDOMIZE_MEMORY`).
                // `KernelSymbols::from_vmlinux` returns `None` on
                // aarch64 and on kernels built without the symbol —
                // the per-iteration refresh tolerates that and
                // leaves `page_offset` at the pre-loop default.
                let page_offset_base_pa = symbols.page_offset_base_kva.map(|kva| {
                    monitor::symbols::text_kva_to_pa_with_base(
                        kva, start_kernel_map_for_thread, phys_base,
                    )
                });
                let rq_refresh = monitor::reader::RqRefresh {
                    pco_pa,
                    runqueues_kva: symbols.runqueues,
                    per_cpu_start: symbols.per_cpu_start,
                    kaslr_offset,
                    num_cpus,
                    page_offset_base_pa,
                    event: event_refresh,
                };

                let vcpu_timing = monitor::reader::VcpuTiming {
                    pthreads: vcpu_pthreads,
                };

                // The legacy SHM signal slot 1 (`SIGNAL_PROBES_READY`)
                // gate before struct_ops discovery has been removed
                // along with the SHM signal-slot infrastructure. The
                // discovery walker tolerates an empty IDR (returns an
                // empty `Vec` when no struct_ops programs are loaded
                // yet) and re-runs every monitor sample, so a race
                // with scheduler BPF program registration recovers on
                // the next cycle.

                // Discover struct_ops programs for per-cycle stats.
                // `cr3_pa` and `l5` are shared with `discover_struct_ops_stats`
                // and `ProgStatsCtx` so per-CPU `bpf_prog_stats` reads can
                // page-walk vmalloc-backed percpu.
                //
                // Re-derive the kernel image base at this point: we
                // just blocked on the guest's slot-1 signal, so the
                // BSP loop has had time to populate the TCR_EL1
                // cache even if it was still 0 at thread start.
                // This is the value that flows into ProgStatsCtx
                // and the GuestKernel constructions below, so a
                // late re-read here gets aarch64 VA_BITS=47 hosts
                // out of the early-boot fallback window.
                let start_kernel_map_post_wait = monitor::symbols::start_kernel_map_for_tcr(
                    tcr_el1
                        .as_ref()
                        .map(|c| c.load(std::sync::atomic::Ordering::Acquire))
                        .unwrap_or(0),
                )
                .unwrap_or(start_kernel_map_for_thread);
                // Use the live BSP CR3 directly (it's already a PA;
                // no `phys_base`-dependent translation needed). When
                // the retry above timed out without observing a
                // non-zero CR3, fall back to the text-symbol
                // translation of `init_top_pgt` — historical
                // behaviour, correct on non-KASLR boots. The earlier
                // post-sys_rdy retry has already waited up to 500 ms
                // for `cr3_cache` to land (warning if it didn't), so
                // a second `cr3.load` here would observe the same
                // value and the post-wait `resolve_phys_base` /
                // `cr3_pa` derivations are folded back onto the
                // pre-wait `cr3_value` / `phys_base` directly.
                let cr3_latest = cr3.load(std::sync::atomic::Ordering::Acquire);
                let cr3_pa = if cr3_latest != 0 {
                    cr3_latest & !0x1FFFu64
                } else {
                    monitor::symbols::text_kva_to_pa_with_base(
                        symbols.init_top_pgt.unwrap_or(0),
                        start_kernel_map_post_wait,
                        phys_base,
                    )
                };
                let l5 = monitor::symbols::resolve_pgtable_l5(
                    &mem,
                    &symbols,
                    start_kernel_map_post_wait,
                    phys_base,
                );
                // Kill check after the post-wait re-resolve.
                // `resolve_phys_base` and `resolve_pgtable_l5` are the
                // most expensive operations in the closure on cold
                // caches — each performs a multi-level page-table
                // walk through guest memory. Return promptly if the
                // VM has already torn down.
                if kill_clone.load(std::sync::atomic::Ordering::Acquire) {
                    return monitor::reader::MonitorLoopResult {
                        samples: Vec::new(),
                        drain: crate::vmm::host_comms::BulkDrainResult {
                            entries: Vec::new(),
                        },
                        watchdog_observation: None,
                        page_offset: 0,
                        preemption_threshold_ns,
                    };
                }
                // aarch64 TCR_EL1 (granule + T1SZ) for the
                // page-table walker. Threaded through ProgStatsCtx
                // so vmalloc-backed percpu `bpf_prog_stats`
                // translations succeed once the BSP populates the
                // cache. Always 0 on x86_64.
                let tcr_el1_val = tcr_el1
                    .as_ref()
                    .map(|c| c.load(std::sync::atomic::Ordering::Acquire))
                    .unwrap_or(0);
                // `prog_offsets` was resolved up front from the
                // single shared `Btf` parse — see the BTF load at
                // the top of `start_monitor`. A previous version
                // re-parsed BTF here via
                // `BpfProgOffsets::from_vmlinux`, doubling the
                // setup cost on every VM run. Dropping that second
                // parse trims hundreds of ms off monitor-thread
                // startup on debug-built vmlinux, so monitor_loop
                // entry — and the first sample push — lands
                // earlier in the VM lifetime. On short-lived
                // no-scheduler boots where the VM exits within a
                // second, the saved time is the difference between
                // sampling rq_clock pre-tick (zero) and post-tick
                // (real values).
                let prog_stats_ctx = prog_offsets.and_then(|prog_offsets| {
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
                    //
                    // `per_cpu_offsets` left empty here: when
                    // `rq_refresh` is set on the
                    // [`monitor::reader::MonitorConfig`], the
                    // monitor loop refreshes `__per_cpu_offset[]`
                    // per iteration and threads the live array
                    // through to `walk_struct_ops_runtime_stats`,
                    // ignoring this seed.
                    Some(monitor::reader::ProgStatsCtx {
                        per_cpu_offsets: Vec::new(),
                        walk: monitor::reader::WalkContext {
                            cr3_pa,
                            page_offset,
                            l5,
                            tcr_el1: tcr_el1_val,
                        },
                        prog_idr_kva,
                        offsets: prog_offsets,
                        start_kernel_map: start_kernel_map_post_wait,
                        phys_base,
                    })
                });

                // Kill check between prog_stats_ctx construction and
                // monitor_loop entry. `monitor_loop` itself honours
                // `kill_evt` via its own epoll registration (see
                // `monitor/reader.rs`), so the check here is the
                // last guard that prevents an idle thread closure
                // from racing into the loop after the VM has been
                // told to shut down.
                if kill_clone.load(std::sync::atomic::Ordering::Acquire) {
                    return monitor::reader::MonitorLoopResult {
                        samples: Vec::new(),
                        drain: crate::vmm::host_comms::BulkDrainResult {
                            entries: Vec::new(),
                        },
                        watchdog_observation: None,
                        page_offset: 0,
                        preemption_threshold_ns,
                    };
                }

                // Construct the scheduler-attach reset payload
                // when both ingredients are present: a workload
                // duration on the VM (the test set `duration` →
                // `KtstrVm::workload_duration` is `Some`) AND a
                // resolvable `scx_root` symbol (the kernel image
                // ships scx and the symbol parser found it). Both
                // missing means there is nothing to reset to / no
                // detection point — leave the field `None` so
                // the monitor's per-iteration check
                // short-circuits.
                let watchdog_reset_cfg = workload_duration.zip(scx_root_pa_for_reset).map(
                    |(workload_duration, scx_root_pa)| monitor::reader::WatchdogReset {
                        scx_root_pa,
                        workload_duration,
                        reset_ns: watchdog_reset_ns.as_ref(),
                    },
                );
                let mon_cfg = monitor::reader::MonitorConfig {
                    // `event_pcpu_pas` left `None` here: the loop
                    // recomputes it each iteration via
                    // `rq_refresh.event` so newly attached
                    // schedulers surface event counters from the
                    // first post-attach sample without a restart.
                    event_pcpu_pas: None,
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
                    prog_stats_ctx: prog_stats_ctx.as_ref(),
                    page_offset,
                    start_kernel_map: start_kernel_map_post_wait,
                    phys_base,
                    rq_refresh: Some(&rq_refresh),
                    // `sys_rdy: None` — the boot-complete wait has
                    // already happened above, BEFORE
                    // `phys_base` resolution and the text-mapped PA
                    // recomputes. Re-running the wait here would be
                    // redundant: the freeze coordinator's TOKEN_TX
                    // handler fires the eventfd exactly once
                    // (`Option::take` makes the write fire-once), and
                    // the per-iteration `page_offset` /
                    // `__per_cpu_offset[]` refresh + `data_valid`
                    // gate inside `monitor_loop` already covers the
                    // pre-boot-zero defense in depth.
                    sys_rdy: None,
                    watchdog_reset: watchdog_reset_cfg,
                };
                // `rq_pas` empty: the loop sources every per-CPU
                // PA from `rq_refresh` per iteration so the static
                // slice would be both stale and redundant.
                monitor::reader::monitor_loop(
                    &mem,
                    &[],
                    &offsets,
                    Duration::from_millis(100),
                    &kill_clone,
                    &kill_evt_clone,
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
    /// 3. Write each queued value, then push `SIGNAL_BPF_WRITE_DONE`
    ///    through virtio-console RX so the guest's `hvc0_poll_loop`
    ///    sets the `bpf_map_write_done` latch; the scenario's
    ///    `wait_for_map_write` gate (`Ctx::wait_for_map_write=true`)
    ///    blocks on that latch until this thread fires.
    ///
    /// `probes_ready_evt` is the broadcast EventFd shared with the
    /// monitor thread (see [`run_vm`]); each phase below `poll`s it
    /// instead of bare-sleeping, and writes 1 to it on detection so
    /// the monitor (and any future waiter) wakes immediately to
    /// re-check its own readiness condition.
    ///
    /// `virtio_con` is the shared virtio-console device used to push
    /// the host→guest wake byte after the writes land. Replaces the
    /// legacy SHM signal slot 0 notification.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn start_bpf_map_write(
        &self,
        vm: &kvm::KtstrKvm,
        kill: &Arc<AtomicBool>,
        probes_ready_evt: EventFd,
        tcr_el1: Option<Arc<std::sync::atomic::AtomicU64>>,
        cr3: Arc<std::sync::atomic::AtomicU64>,
        virtio_con: Arc<PiMutex<virtio_console::VirtioConsole>>,
        kern_phys_base: Arc<std::sync::atomic::AtomicU64>,
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

        let handle = std::thread::Builder::new()
            .name("bpf-map-write".into())
            .spawn(move || {
                use crate::monitor::bpf_map::BpfMapAccessor;
                if kill_clone.load(Ordering::Acquire) {
                    return;
                }

                // Phase 1: wait for BPF map accessor (kernel booted, page tables up).
                //
                // Sleeping is replaced by `poll(POLLIN)` against the
                // shared `probes_ready_evt`: ANY waiter that detects
                // its own readiness condition writes 1 to the eventfd
                // and the level stays high (we never `read` here), so
                // this loop wakes immediately on a sibling detection
                // and re-tries the accessor construction. The 200 ms
                // timeout preserves the prior cadence as an upper
                // bound for kill / deadline observation when no other
                // detector has fired yet. On successful construction
                // we write 1 ourselves, fanning the wake out to the
                // monitor and the later phases.
                let vmlinux_data_arc = match super::vmlinux::cached_vmlinux_bytes(&vmlinux) {
                    Some(d) => d,
                    None => {
                        eprintln!("bpf_map_write: read vmlinux failed");
                        return;
                    }
                };
                let vmlinux_data = &*vmlinux_data_arc;
                let vmlinux_elf = match goblin::elf::Elf::parse(vmlinux_data) {
                    Ok(e) => e,
                    Err(e) => {
                        eprintln!("bpf_map_write: parse vmlinux ELF failed: {e:#}");
                        return;
                    }
                };
                let mem = Arc::new(mem);
                let phase1_deadline =
                    std::time::Instant::now() + std::time::Duration::from_secs(30);
                let owned = loop {
                    let biased = kern_phys_base.load(std::sync::atomic::Ordering::Acquire);
                    if biased == 0 {
                        if kill_clone.load(Ordering::Acquire) {
                            return;
                        }
                        poll_eventfd_until_ready_or_timeout(&probes_ready_evt, 200);
                        continue;
                    }
                    let pb_hint = biased.wrapping_sub(1);
                    let tcr_val = tcr_el1
                        .as_ref()
                        .map(|c| c.load(std::sync::atomic::Ordering::Acquire))
                        .unwrap_or(0);
                    let cr3_val = cr3.load(std::sync::atomic::Ordering::Acquire);
                    match monitor::bpf_map::GuestMemMapAccessorOwned::from_elf_with_hint(Arc::clone(&mem), &vmlinux_elf, &vmlinux_data, &vmlinux, tcr_val, cr3_val, pb_hint) {
                        Ok(a) => {
                            let _ = probes_ready_evt.write(1);
                            break a;
                        }
                        Err(e) => {
                            if kill_clone.load(Ordering::Acquire) {
                                return;
                            }
                            if std::time::Instant::now() >= phase1_deadline {
                                eprintln!("bpf_map_write: accessor init timed out: {e:#}");
                                return;
                            }
                            poll_eventfd_until_ready_or_timeout(&probes_ready_evt, 200);
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
                //
                // Same `poll(POLLIN)` pattern as phase 1: wake on a
                // sibling detection, fall back to the 200 ms cadence
                // for kill / deadline coverage; write 1 on each
                // successful map resolution to fan the wake out to
                // the monitor and the still-pending phases.
                let retry_deadline =
                    std::time::Instant::now() + std::time::Duration::from_secs(30);
                let mut resolved: Vec<(BpfMapWriteParams, monitor::bpf_map::BpfMapInfo)> =
                    Vec::with_capacity(writes.len());
                for params in writes.iter() {
                    let mut attempt = 0u32;
                    let map_info = loop {
                        attempt += 1;
                        if let Some(info) = accessor.find_map(&params.map_name_suffix) {
                            let _ = probes_ready_evt.write(1);
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
                        poll_eventfd_until_ready_or_timeout(&probes_ready_evt, 200);
                    };
                    eprintln!(
                        "bpf_map_write: map '{}' found after {} attempts",
                        map_info.name(), attempt,
                    );
                    resolved.push((params.clone(), map_info));
                }

                // Phase 3: run every queued write.
                //
                // The legacy SHM signal slot 1 (`SIGNAL_PROBES_READY`)
                // gate that waited for the guest's probe pipeline to
                // attach has been removed along with the SHM
                // signal-slot infrastructure. The writes now race
                // against probe attachment; replacing the rendezvous
                // with a virtio-console signal is a follow-up.

                // Log all maps for diagnostic visibility.
                let all_maps = accessor.maps();
                eprintln!(
                    "bpf_map_write: maps() found {} map(s): [{}]",
                    all_maps.len(),
                    all_maps
                        .iter()
                        .map(|m| format!("{}(type={})", m.name(), m.map_type))
                        .collect::<Vec<_>>()
                        .join(", "),
                );

                for (params, map_info) in &resolved {
                    let before = accessor.read_value_u32(map_info, params.offset);
                    let ok = accessor.write_value_u32(map_info, params.offset, params.value);
                    let after = accessor.read_value_u32(map_info, params.offset);
                    eprintln!(
                        "bpf_map_write: map '{}' write={} (value={} offset={} before={:?} after={:?})",
                        map_info.name(), ok, params.value, params.offset, before, after,
                    );
                }

                // Notify the guest that every queued write landed by
                // pushing `SIGNAL_BPF_WRITE_DONE` into virtio-console
                // RX. The guest's `hvc0_poll_loop` blocks on
                // `/dev/hvc0`, recognises the byte, and sets the
                // `bpf_map_write_done` latch. A scenario blocked on
                // [`crate::scenario::Ctx::wait_for_map_write`] resumes
                // when the latch fires. Replaces the legacy SHM signal
                // slot 0 notification.
                super::host_comms::request_bpf_map_write_done(&virtio_con);
                let _ = (&kill_clone, &probes_ready_evt, &mem);
            })
            .context("spawn bpf-map-write thread")?;

        Ok(Some(handle))
    }

    /// Unified BSP KVM_RUN loop. Returns `(exit_code, timed_out)`.
    ///
    /// `exit_code` semantics:
    ///   - `0` only when the BSP itself observed
    ///     [`ExitAction::Shutdown`] from `classify_exit` (i8042 reset
    ///     on x86_64, PSCI SystemEvent on aarch64, or
    ///     `VcpuExit::Shutdown`).
    ///   - `-1` is a sentinel meaning "BSP exited the loop without
    ///     observing Shutdown itself." This does NOT necessarily
    ///     indicate a failure — a peer vCPU that observed Shutdown
    ///     first sets the shared `kill` flag, and the BSP then exits
    ///     via the `kill.load(Acquire)` check at the top of the loop.
    ///     [`super::KtstrVm::collect_results`] overrides the run-loop
    ///     `exit_code` with the bulk-port `MSG_TYPE_EXIT` payload (or the
    ///     COM2 `KTSTR_EXIT:` sentinel) before constructing
    ///     [`super::result::VmResult`], so the value caller-visible
    ///     code reads is the guest's reported exit code, not this
    ///     local sentinel. [`BspExitReason`] is logged at break time
    ///     so an operator reading stderr can distinguish
    ///     "AP saw Shutdown first" from "BSP itself saw Fatal" or
    ///     "BSP run() returned a permanent error" without correlating
    ///     to other diagnostics.
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
    ///
    /// `watchpoint` carries the failure-dump trigger contract: each
    /// iteration polls `watchpoint.request_kva` and self-arms a
    /// hardware data-write watchpoint on `*scx_root->exit_kind` once
    /// the freeze coordinator has resolved its KVA. When the kernel
    /// later writes the field, KVM exits via `VcpuExit::Debug`; this
    /// loop sets `watchpoint.hit` so the freeze coordinator's
    /// late-trigger poll fires immediately. The arm is one-shot per
    /// KVA value (the per-vCPU `armed_kva` slot suppresses re-arms
    /// after the ioctl lands).
    ///
    /// `tcr_el1_cache` (aarch64 only) is populated lazily on first
    /// successful sysreg read after the guest kernel programs the
    /// MMU; subsequent iterations short-circuit on a non-zero
    /// cached value. Threads that build a `GuestKernel` for
    /// page-table walks load this atomic to feed the
    /// granule-agnostic walker.
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
        watchpoint: &Arc<WatchpointArm>,
        bsp_parked: &Arc<AtomicBool>,
        bsp_regs: &Arc<std::sync::Mutex<Option<exit_dispatch::VcpuRegSnapshot>>>,
        has_immediate_exit: bool,
        _run_start: Instant,
        _timeout: Duration,
        parked_evt: Option<&Arc<EventFd>>,
        thaw_evt: Option<&Arc<EventFd>>,
        kill_evt: Option<&Arc<EventFd>>,
        tcr_el1_cache: Option<&Arc<std::sync::atomic::AtomicU64>>,
        cr3_cache: &Arc<std::sync::atomic::AtomicU64>,
        timed_out_flag: &Arc<AtomicBool>,
    ) -> (i32, bool) {
        let mut exit_code: i32 = -1;
        // Track which path drove the BSP out of the loop so the
        // post-loop log line is actionable. Without this, an operator
        // sees `code=-1 timed_out=false` and cannot distinguish
        // "external kill propagated from a peer vCPU's Shutdown" from
        // "BSP itself saw Fatal" — every non-Shutdown exit produces
        // the same `code=-1` sentinel.
        let mut exit_reason = BspExitReason::ExternalKill;
        // Per-BSP `armed_slots` mirrors the AP-side slots — see
        // [`super::vcpu::self_arm_watchpoint`]. Index 0 = slot 0
        // (exit_kind watchpoint); 1..=3 = user watchpoint slots
        // (`Op::WatchSnapshot` arms). All `0` until the coordinator
        // publishes resolved KVAs. `arm_failures` counts consecutive
        // non-EINTR ioctl failures; transient EINTR (signal race
        // with the SIGRTMIN kick path) does NOT increment so a
        // kicked-mid-arm vCPU keeps retrying instead of giving up
        // after the first racey iteration.
        let mut armed_slots: [u64; 4] = [0; 4];
        let mut arm_failures: u8 = 0;
        // aarch64 watchpoint single-step bookkeeping — mirrors the
        // AP-side state in
        // [`super::exit_dispatch::vcpu_run_loop_unified`]. The
        // aarch64 hardware watchpoint trap is taken BEFORE the
        // offending store retires (ARM ARM D2.10.5), so re-entering
        // KVM_RUN replays the same instruction unless we disable
        // the fired slot's WCR.E and assert
        // KVM_GUESTDBG_SINGLESTEP for one KVM_RUN; the next
        // KVM_EXIT_DEBUG carries EC=ESR_ELx_EC_SOFTSTP_LOW (0x32),
        // at which point the dispatch helper clears the flag and
        // `self_arm_watchpoint` restores WCR.E=1. Inert on x86_64
        // (the trap is taken AFTER the store, so re-entry advances
        // normally); the locals still pass through to keep the
        // per-arch helper signatures shared.
        let mut single_step_pending: bool = false;
        let mut single_step_slot: usize = 0;
        let mut armed_single_step: bool = false;

        loop {
            if kill.load(Ordering::Acquire) {
                break;
            }
            // Lazy TCR_EL1 cache populate (aarch64). On x86_64
            // `read_tcr_el1` returns None and the early-exit keeps
            // the atomic untouched. The kernel writes TCR_EL1 in
            // its boot-time MMU bring-up; before that the read
            // returns 0. Skip on subsequent iterations once the
            // atomic carries a non-zero value (CAS prevents races
            // with peer reads from other threads constructing a
            // `GuestKernel`).
            if let Some(cache) = tcr_el1_cache
                && cache.load(Ordering::Acquire) == 0
                && let Some(val) = exit_dispatch::read_tcr_el1(bsp)
                && val != 0
            {
                let _ = cache.compare_exchange(0, val, Ordering::Release, Ordering::Relaxed);
            }
            // CR3 / TTBR1_EL1 cache refresh. KVM_GET_SREGS at BSP
            // entry returns the boot-time CR3 (`PML4_START`, set by
            // `setup_sregs`); the kernel later overwrites this in
            // `__startup_64` after KASLR randomization. We need the
            // POST-randomization value for `phys_base` resolution
            // via page-table walk, so this MUST be a refresh
            // (overwrite each iteration), NOT a one-shot latch:
            // a "skip if non-zero" gate would freeze the cache at
            // the boot CR3 because get_sregs returns it on iter 1
            // before the guest has run `mov cr3, ...`. Accepting
            // every non-zero read also handles process context
            // switches (CR3 swaps to the new task's pgd) — the
            // kernel-half upper PML4 entries are shared across
            // every task's pgd, so any task's CR3 produces a valid
            // walk for kernel symbols. The lazy-CAS pattern still
            // gates on a non-zero `read_cr3` return so a transient
            // EINTR (None) does not zero the cache. Use a Release
            // store (not CAS) so concurrent readers see the latest
            // non-zero value.
            if let Some(val) = exit_dispatch::read_cr3(bsp)
                && val != 0
            {
                cr3_cache.store(val, Ordering::Release);
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
                    parked_evt.map(|a| a.as_ref()),
                    thaw_evt.map(|a| a.as_ref()),
                    kill_evt.map(|a| a.as_ref()),
                );
                if kill.load(Ordering::Acquire) {
                    break;
                }
            }
            // Self-arm the failure-dump watchpoint when the
            // coordinator has resolved a new KVA. Cheap (atomic load
            // and compare) when no new arm is pending. Also drives
            // the aarch64 watchpoint single-step transition: when
            // `single_step_pending` is set by the prior watchpoint
            // exit, this call reissues KVM_SET_GUEST_DEBUG with the
            // fired slot's WCR.E cleared and KVM_GUESTDBG_SINGLESTEP
            // asserted; when the SOFTSTP_LOW exit clears the flag,
            // the next call restores WCR.E=1 and drops the
            // singlestep bit.
            self_arm_watchpoint(
                bsp,
                watchpoint,
                &mut armed_slots,
                &mut arm_failures,
                single_step_pending,
                single_step_slot,
                &mut armed_single_step,
            );

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
                    // KVM_EXIT_DEBUG fires when the armed hardware
                    // data-write watchpoint trips on a guest write
                    // to `*scx_root->exit_kind`. The kernel writes
                    // the field on BOTH error transitions
                    // (`scx_error -> SCX_EXIT_ERROR/_BPF/_STALL >=
                    // 1024`) AND clean shutdown
                    // (`scx_unregister -> SCX_EXIT_DONE = 1`). Only
                    // the error transitions should trigger the
                    // failure-dump freeze; firing on every clean
                    // test exit is a regression. Read the post-store
                    // value from the host pointer the coordinator
                    // published and gate `hit` on the error
                    // threshold. The watchpoint is left armed
                    // regardless — see the AP-side
                    // `vcpu_run_loop_unified` for the same
                    // rationale.
                    if let VcpuExit::Debug(debug_arch) = &exit {
                        exit_dispatch::dispatch_watchpoint_hit(
                            watchpoint,
                            debug_arch,
                            &armed_slots,
                            &mut single_step_pending,
                            &mut single_step_slot,
                        );
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
                            exit_reason = BspExitReason::Shutdown;
                            break;
                        }
                        Some(ExitAction::Fatal(reason)) => {
                            if let Some(r) = reason {
                                tracing::error!(r, "BSP VM entry failed");
                            } else {
                                tracing::error!("BSP internal error");
                            }
                            // Propagate kill to peers and the freeze
                            // coordinator. Unlike the Shutdown arm
                            // (which exits with code=0 and lets
                            // run_vm drive the kill
                            // propagation), Fatal indicates an
                            // unrecoverable hardware/KVM failure and
                            // peers must shut down promptly rather
                            // than spinning until FREEZE_RENDEZVOUS_
                            // TIMEOUT. Mirrors the AP Fatal arm's
                            // kill-propagation in
                            // [`super::exit_dispatch::vcpu_run_loop_unified`].
                            kill.store(true, Ordering::Release);
                            if let Some(kev) = kill_evt {
                                let _ = kev.write(1);
                            }
                            exit_reason = BspExitReason::Fatal;
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
                    exit_reason = BspExitReason::RunError;
                    break;
                }
            }
        }

        eprintln!("BSP: loop exit reason={exit_reason:?}");
        // The watchdog sets `timed_out_flag` only on its hard-
        // deadline branch (NOT on "kill set by AP"). Reading it
        // here propagates the watchdog's hard-timeout verdict
        // through the BSP return tuple → `VmRunState::timed_out`
        // → `VmResult::timed_out` so callers can distinguish a
        // watchdog-driven kill from a clean shutdown or a
        // panic-driven kill.
        let timed_out = timed_out_flag.load(Ordering::Acquire);
        (exit_code, timed_out)
    }

    /// Shutdown threads and collect output.
    pub(super) fn collect_results(&self, start: Instant, run: VmRunState) -> Result<VmResult> {
        // Whole-cleanup timer for the perf-repro tracing pipeline.
        // `cleanup_duration` below already records the post-BSP-exit
        // window via `run.cleanup_start.elapsed()`; this captures the
        // collect_results function span itself so a regression isolates
        // to either the run.cleanup_start window (set by run_vm before
        // it called us) or the function body.
        let collect_results_start = Instant::now();
        let mut exit_code = run.exit_code;
        let timed_out = run.timed_out;
        // Belt-and-braces: kill + kill_evt are already set by run_vm
        // immediately after BSP exits. Re-assert here in case a
        // future code path reaches collect_results without the
        // early-kill having fired. The two consumers that observe
        // kill_evt via epoll are the monitor sampler (reader.rs
        // monitor_loop) and the bpf-map-write thread (start_bpf_map_write).
        // The freeze coordinator is NOT alive here — run_vm joins it
        // before returning VmRunState. kill_evt is level-triggered
        // (EFD_NONBLOCK eventfd); the AtomicBool kill flag is the
        // source of truth that breaks each thread's outer loop.
        run.kill.store(true, Ordering::Release);
        let _ = run.kill_evt.write(1);
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
        // The freeze coordinator was joined inside `run_vm` BEFORE
        // bsp dropped (preventing UAF on the BSP ImmediateExitHandle),
        // so `run.freeze_coordinator` is always `None` here. The
        // `Option`-typed field is preserved for backward compatibility
        // with paths that may construct VmRunState differently in
        // the future; the conditional join below is a no-op for the
        // `None` arm.
        if let Some(h) = run.freeze_coordinator {
            let _ = h.join();
        }
        {
            let mut remaining = run.ap_threads.len();
            if remaining > 0
                && let Ok(epoll) = Epoll::new()
            {
                for (i, vt) in run.ap_threads.iter().enumerate() {
                    if vt.exited.load(Ordering::Acquire) {
                        remaining -= 1;
                        continue;
                    }
                    let _ = epoll.ctl(
                        ControlOperation::Add,
                        vt.exit_evt.as_raw_fd(),
                        EpollEvent::new(EventSet::IN, i as u64),
                    );
                }
                if remaining > 0 {
                    let mut events = vec![EpollEvent::default(); remaining];
                    let deadline = Instant::now() + Duration::from_secs(2);
                    while remaining > 0 {
                        let left = deadline.saturating_duration_since(Instant::now());
                        if left.is_zero() {
                            break;
                        }
                        let ms = left.as_millis().min(i32::MAX as u128) as i32;
                        match epoll.wait(ms, &mut events) {
                            Ok(0) => break,
                            Ok(n) => remaining = remaining.saturating_sub(n),
                            Err(_) => break,
                        }
                        for vt in &run.ap_threads {
                            if !vt.exited.load(Ordering::Acquire) {
                                vt.kick();
                            }
                        }
                    }
                }
            }
            for vt in run.ap_threads {
                let _ = vt.handle.join();
            }
        }
        eprintln!("CLEANUP: all AP threads joined");

        // Invalidate the watchpoint slots BEFORE `run.vm` drops at
        // the end of this function. `kind_host_ptr` addresses a host
        // u32 inside `vm.guest_mem`'s mmap-backed mapping; once
        // `vm.guest_mem` drops, that mapping unmaps and dereffing
        // `kind_host_ptr` would touch unmapped memory.
        // `request_kva` is the paired guest-side KVA whose
        // translation goes through the same mapping. By this point
        // every vCPU thread has joined (the loop above blocked on
        // each `wait_for_exit` + `handle.join`) and the freeze
        // coordinator joined back in `run_vm` before `bsp` dropped,
        // so no live thread reads either field. The defense-in-depth
        // store here zeroes the slots so a stray future Arc clone
        // (or a follow-up that adds a new reader after teardown)
        // sees a sentinel `null_mut` / `0` that
        // [`super::exit_dispatch::latch_slot0_with_gate`] already
        // gates on, instead of dangling host memory. `Release`
        // ordering pairs with the `Acquire` reads inside the latch
        // path so any future reader sees a coherent view of the
        // invalidation.
        run.watchpoint
            .kind_host_ptr
            .store(std::ptr::null_mut(), Ordering::Release);
        run.watchpoint.request_kva.store(0, Ordering::Release);
        // Mirror the slot-0 invalidation across every user
        // watchpoint slot (1..=3, `Op::WatchSnapshot` arms). A
        // future reader that walks `watchpoint.user[..]` sees the
        // same `request_kva == 0` sentinel as slot 0 — the
        // resolved KVA is no longer reachable from any slot. `hit`
        // is also cleared so a stray Acquire load after teardown
        // observes "no fire pending" instead of a stale latch from
        // an earlier run that no longer has a captured report.
        // `Release` pairs with the `Acquire` reads in
        // `arm_user_watchpoint` and the latch path.
        for slot in &run.watchpoint.user {
            slot.request_kva.store(0, Ordering::Release);
            slot.hit.store(false, Ordering::Release);
        }

        let (monitor_report, mid_flight_drain) =
            match run.monitor_handle.and_then(|h| h.join().ok()) {
                Some(monitor::reader::MonitorLoopResult {
                    samples,
                    drain,
                    watchdog_observation,
                    page_offset,
                    preemption_threshold_ns,
                }) => {
                    // `preemption_threshold_ns` was resolved once
                    // inside `start_monitor` (and threaded through
                    // `monitor_loop`'s 0-fallback) so the cleanup
                    // path does NOT re-read the vmlinux to recompute
                    // CONFIG_HZ. The previous structure called
                    // `monitor::vcpu_preemption_threshold_ns(Some(
                    // &self.kernel))` here, which re-read the
                    // vmlinux ELF every cleanup just to derive the
                    // same value the monitor thread already had in
                    // hand.
                    let summary = monitor::MonitorSummary::from_samples_with_threshold(
                        &samples,
                        preemption_threshold_ns,
                    );
                    let report = monitor::MonitorReport {
                        samples,
                        summary,
                        preemption_threshold_ns,
                        watchdog_observation,
                        page_offset,
                    };
                    (Some(report), drain)
                }
                None => (None, BulkDrainResult::default()),
            };
        eprintln!("CLEANUP: monitor joined");
        let cleanup_t = std::time::Instant::now();

        if let Some(h) = run.bpf_write_handle {
            let _ = h.join();
        }
        eprintln!("CLEANUP: bpf_write joined {:?}", cleanup_t.elapsed());

        // Drain the virtio-console port-1 TX accumulator: the guest
        // wrote bulk TLV-framed messages (STIMULUS, EXIT, SCHED_EXIT,
        // PAYLOAD_METRICS, RAW_PAYLOAD_OUTPUT, etc.) to
        // `/dev/vport0p1`; the host side accumulated them into
        // `port1_tx_buf` and we parse them here through
        // `parse_tlv_stream`. Port-1 uses backpressure rather than
        // drops — every byte the guest emitted is delivered, in
        // order.
        //
        // `final_drain` (rather than `drain_bulk`) walks the avail
        // ring once before draining so chains the guest published
        // without a host-observed QUEUE_NOTIFY (the
        // `force_reboot()` race in `rust_init`'s `send_exit`-then-
        // reboot tail) are picked up instead of being lost. See
        // [`crate::vmm::virtio_console::VirtioConsole::final_drain`].
        let bulk_bytes = run.virtio_con.lock().final_drain();
        let mut bulk_drain = host_comms::parse_tlv_stream(&bulk_bytes);
        // Strip coordinator-internal control frames the freeze coord
        // mid-run filter (the TOKEN_TX dispatch in this same file)
        // already drops: SNAPSHOT_REQUEST has its matching reply
        // delivered over port-1 RX; SYS_RDY's only semantic is the
        // eventfd promotion in the coord's TOKEN_TX handler.
        // Without this filter, a late-arriving control frame that
        // the coord had not yet consumed when its outer loop
        // exited would land in `guest_messages` and surface as a
        // phantom verdict entry.
        //
        // Both filters key on
        // [`crate::vmm::wire::MsgType::is_coordinator_internal`] —
        // a single source of truth so adding a new internal control
        // frame is a one-line update at the classifier site.
        bulk_drain.entries.retain(|e| {
            // Keep when the msg_type is NOT a recognised
            // coordinator-internal control frame. Unknown
            // msg_types (None) are preserved verbatim so an
            // operator-side analyser can surface them rather
            // than silently dropping them here.
            match crate::vmm::wire::MsgType::from_wire(e.msg_type) {
                Some(t) => !t.is_coordinator_internal(),
                None => true,
            }
        });
        // Prepend the entries the freeze coordinator already parsed
        // mid-run. The coord's TOKEN_TX handler streams port-1
        // bytes through `HostAssembler` so a SCHED_EXIT can flip
        // the run-wide kill flag without waiting for VM exit;
        // those parsed frames stash here on every drain so
        // `collect_results` can recover them after the coord has
        // joined. Without this merge every guest-side EXIT / TEST
        // / PAYLOAD_METRICS / RAW_PAYLOAD_OUTPUT / PROFRAW frame
        // consumed mid-run would be silently lost — `drain_bulk()`
        // above only catches what arrived AFTER the coord stopped
        // polling, which on a typical run is empty. Mid-run
        // entries come first so the merged stream stays in
        // chronological order.
        let mut mid_run_bulk = match run.bulk_messages.lock() {
            Ok(mut g) => std::mem::take(&mut *g),
            Err(p) => std::mem::take(&mut *p.into_inner()),
        };
        mid_run_bulk.extend(bulk_drain.entries);
        bulk_drain.entries = mid_run_bulk;

        // Merge mid-flight drain (from monitor thread, port-1 byte
        // stream) with the post-exit `drain_bulk()`. Mid-flight
        // entries come first since they were drained during
        // execution.
        let (guest_messages, stimulus_events) =
            if !mid_flight_drain.entries.is_empty() || !bulk_drain.entries.is_empty() {
                let mut all_entries = mid_flight_drain.entries;
                all_entries.extend(bulk_drain.entries);
                let events: Vec<wire::StimulusEvent> = all_entries
                    .iter()
                    .filter(|e| e.msg_type == wire::MSG_TYPE_STIMULUS && e.crc_ok)
                    .filter_map(|e| wire::StimulusEvent::from_payload(&e.payload))
                    .collect();
                (
                    Some(BulkDrainResult {
                        entries: all_entries,
                    }),
                    events,
                )
            } else {
                (None, Vec::new())
            };

        let com2_bytes = run.com2.lock().output();
        let console_output = run.com1.lock().output();

        // Concatenate every CRC-valid `MSG_TYPE_STDOUT` /
        // `MSG_TYPE_STDERR` chunk from the bulk-port drain into a
        // single string and prepend the COM2 capture so panic-hook
        // bytes (the lone remaining COM2 writer) still surface in
        // `result.output`. The bulk-port chunks dominate steady-state
        // test output; COM2 is reserved for fault diagnostics that
        // cannot block on virtio backpressure.
        let mut app_output = String::new();
        if let Some(ref drain) = guest_messages {
            for e in &drain.entries {
                if !e.crc_ok {
                    continue;
                }
                match wire::MsgType::from_wire(e.msg_type) {
                    Some(wire::MsgType::Stdout) | Some(wire::MsgType::Stderr) => {
                        app_output.push_str(&String::from_utf8_lossy(&e.payload));
                    }
                    _ => {}
                }
            }
        }
        if !com2_bytes.is_empty() {
            app_output.push_str(&com2_bytes);
        }

        // Extract exit code: bulk port (primary), COM2 sentinel (fallback).
        let bulk_exit = guest_messages.as_ref().and_then(|d| {
            d.entries
                .iter()
                .rev()
                .find(|e| e.msg_type == wire::MSG_TYPE_EXIT && e.crc_ok && e.payload.len() == 4)
                .map(|e| i32::from_ne_bytes(e.payload[..4].try_into().unwrap()))
        });
        // Pre-bincode-migration: a COM2 `KTSTR_EXIT=N` sentinel line
        // served as the fallback when no binary `MSG_TYPE_EXIT`
        // frame arrived. The fallback is gone — bulk-port
        // backpressure guarantees delivery, and the guest no longer
        // emits the sentinel. A `None` here keeps `exit_code` at
        // whatever the BSP run-loop's local stored, matching the
        // pre-fallback path.
        if let Some(code) = bulk_exit {
            exit_code = code;
        }

        // Extract crash message from COM2 output. The guest panic
        // hook in `rust_init.rs` writes `PANIC: <info>\n<bt>\n` to
        // `/dev/ttyS1`; the host-side parser
        // [`crate::test_support::extract_panic_message`] strips the
        // prefix and returns the trimmed remainder.
        let crash_message =
            crate::test_support::extract_panic_message(&app_output).map(|s| s.to_string());

        // Collect BPF verifier stats from host-side memory reads.
        // Skip when no scheduler is active — struct_ops programs
        // only exist when a sched_ext scheduler attached (either via
        // a userspace binary or kernel-built enable commands).
        let has_scheduler = self.scheduler_binary.is_some() || !self.sched_enable_cmds.is_empty();
        let vs_t = std::time::Instant::now();
        let mut vs_path: &'static str = "skipped_no_scheduler";
        let verifier_stats = if has_scheduler {
            if let Some(ref prog) = run.prog_accessor {
                use crate::monitor::bpf_prog::BpfProgAccessor;
                vs_path = "prebuilt_accessor";
                eprintln!("CLEANUP: verifier_stats using pre-built accessor");
                let a = prog.as_accessor();
                a.struct_ops_progs()
            } else {
                vs_path = "fallback_full_parse";
                eprintln!("CLEANUP: verifier_stats fallback (full parse)");
                self.collect_verifier_stats(
                    &run.vm,
                    run.tcr_el1.as_ref(),
                    &run.cr3,
                    run.vmlinux_data.as_ref().map(|d| d.as_slice()),
                    run.kern_phys_base,
                )
            }
        } else {
            Vec::new()
        };
        tracing::info!(
            elapsed_ms = vs_t.elapsed().as_millis() as u64,
            path = vs_path,
            n_progs = verifier_stats.len(),
            "auto_repro: collect_verifier_stats",
        );
        eprintln!("CLEANUP: verifier_stats done {:?}", vs_t.elapsed());

        // Sample cleanup elapsed AFTER every blocking step that runs on
        // the post-BSP-exit critical path so the duration captures the
        // full host-side teardown cost, not a partial window. The full
        // ordered set is: watchdog join (in `run_vm`, before
        // `cleanup_start` is stored on `VmRunState`), AP joins, monitor
        // join, BPF writer join, bulk drain, exit-code and crash-message
        // extraction, verifier-stat read. Captured before constructing
        // the result so the `Instant::now()` here is the latest possible
        // read.
        let cleanup_duration = Some(run.cleanup_start.elapsed());
        tracing::info!(
            elapsed_ms = collect_results_start.elapsed().as_millis() as u64,
            cleanup_window_ms = cleanup_duration.map(|d| d.as_millis() as u64).unwrap_or(0),
            "auto_repro: collect_results",
        );
        eprintln!("CLEANUP: collect_results done {:?}", cleanup_t.elapsed());

        // Forward the scheduler-stats client. `run.stats_client` is
        // `Some(_)` when the run has a scheduler attached and
        // `None` otherwise; the field on `VmResult.stats_client`
        // mirrors this exactly. The drainer thread (when present)
        // continues to run until the last `Arc<ClientShared>` clone
        // drops; `Drop` on the field then writes the kill eventfd
        // and the drainer thread exits.
        let stats_client = run.stats_client;

        Ok(VmResult {
            success: !timed_out && exit_code == 0,
            exit_code,
            duration: start.elapsed(),
            timed_out,
            output: app_output,
            stderr: console_output,
            monitor: monitor_report,
            guest_messages,
            stimulus_events,
            verifier_stats,
            kvm_stats: None,
            crash_message,
            cleanup_duration,
            virtio_blk_counters: run.virtio_blk_counters,
            virtio_net_counters: run.virtio_net_counters,
            snapshot_bridge: run.snapshot_bridge,
            stats_client,
            periodic_fired: run.periodic_fired,
            periodic_target: run.periodic_target,
        })
    }

    /// Read BPF verifier stats from guest memory after VM exit.
    ///
    /// Enumerates struct_ops programs in the kernel's `prog_idr` and
    /// reads `bpf_prog_aux->verified_insns` for each.
    pub(super) fn collect_verifier_stats(
        &self,
        vm: &kvm::KtstrKvm,
        tcr_el1: Option<&Arc<std::sync::atomic::AtomicU64>>,
        cr3: &Arc<std::sync::atomic::AtomicU64>,
        cached_vmlinux_data: Option<&[u8]>,
        kern_phys_base_biased: u64,
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
        // TCR_EL1 (aarch64) drives the granule-agnostic page-table
        // walker. The BSP populates this Arc<AtomicU64> on first
        // successful read post-MMU-bringup; by collect_verifier_stats
        // time it is either set (kernel booted) or 0 (kernel never
        // brought MMU up, e.g. early boot crash). The walker treats
        // 0 as "no TCR available — translation unsupported", which
        // matches the boot-crash case where verifier stats are
        // unavailable anyway.
        let tcr_val = tcr_el1
            .map(|c| c.load(std::sync::atomic::Ordering::Acquire))
            .unwrap_or(0);
        let cr3_val = cr3.load(std::sync::atomic::Ordering::Acquire);
        // Fallback when the caller did not pre-load the vmlinux ELF.
        // Routes through `cached_vmlinux_bytes` so a test process that
        // boots N VMs against the same kernel pays the 50-340 MB read
        // exactly once. Without the cache this path was the dominant
        // cost on `collect_results` cleanup for nextest runs of
        // `#[ktstr_test]` cases that share a kernel.
        let owned_data;
        let vmlinux_data: &[u8] = match cached_vmlinux_data {
            Some(d) => d,
            None => match cached_vmlinux_bytes(&vmlinux) {
                Some(arc) => {
                    owned_data = arc;
                    owned_data.as_slice()
                }
                None => return Vec::new(),
            },
        };
        // Parse the vmlinux ELF once and share the result between
        // `GuestKernel` (kernel symbols + paging state) and
        // `BpfProgOffsets` (BTF section extraction on cache miss).
        // The previous structure parsed the ELF up to three times per
        // call: once inside `GuestKernel::from_vmlinux_bytes`, once
        // again via the nested `KernelSymbols::from_vmlinux_bytes`,
        // and once more via `load_btf_from_bytes` on a sidecar miss.
        // `goblin::elf::Elf::parse` is hundreds of ms on a debug
        // vmlinux, so this single parse is the cheap shared base.
        let elf = match goblin::elf::Elf::parse(vmlinux_data) {
            Ok(e) => e,
            Err(_) => return Vec::new(),
        };
        let pb_hint = if kern_phys_base_biased != 0 {
            kern_phys_base_biased.wrapping_sub(1)
        } else {
            0
        };
        let kernel = match monitor::guest::GuestKernel::from_elf_with_hint(
            Arc::new(mem),
            &elf,
            tcr_val,
            cr3_val,
            pb_hint,
        ) {
            Ok(k) => k,
            Err(_) => return Vec::new(),
        };
        // BTF sidecar cache hits skip ELF traversal entirely; on a
        // miss `load_btf_from_elf` reuses the parse above instead of
        // re-running `goblin::elf::Elf::parse(&vmlinux_data)`.
        let offsets =
            match monitor::btf_offsets::BpfProgOffsets::from_elf(&elf, vmlinux_data, &vmlinux) {
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

#[cfg(test)]
mod crc_defense_tests;
#[cfg(test)]
mod rendezvous_tests;
#[cfg(test)]
mod snapshot_tlv_tests;
#[cfg(test)]
mod tx_dispatch_tests;
