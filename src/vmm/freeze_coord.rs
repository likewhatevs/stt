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
use kvm_ioctls::{IoEventAddress, NoDatamatch, VcpuExit};
use std::os::fd::AsRawFd;
use std::os::unix::thread::JoinHandleExt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use vm_memory::{Bytes, GuestAddress, GuestMemory};
use vmm_sys_util::epoll::{ControlOperation, Epoll, EpollEvent, EventSet};
use vmm_sys_util::eventfd::{EFD_NONBLOCK, EventFd};
use vmm_sys_util::timerfd::TimerFd;

use crate::monitor;

use super::exit_dispatch::{self, ExitAction, classify_exit, vcpu_run_loop_unified};
use super::pi_mutex::PiMutex;
use super::result::{VmResult, VmRunState};
use super::vcpu::{
    ApFreezeHandles, BpfMapWriteParams, ImmediateExitHandle, VcpuThread, WatchpointArm,
    duration_to_jiffies, load_probe_bss_offset, open_vcpu_perf_capture, pin_current_thread,
    register_vcpu_signal_handler, self_arm_watchpoint, set_rt_priority, set_thread_cpumask,
    vcpu_signal,
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

/// Why [`KtstrVm::run_bsp_loop`] exited. Logged at break time so an
/// operator reading stderr (`BSP: loop exit reason=...`) can
/// diagnose a `code=-1` exit without correlating to peer-vCPU
/// stderr or `tracing` output.
///
/// Mapping to the BSP loop's exit_code:
///   - [`Shutdown`](Self::Shutdown) → exit_code = 0 (the only path
///     that overwrites the local `-1` sentinel).
///   - Every other variant → exit_code = -1, but
///     [`super::KtstrVm::collect_results`] re-derives the final
///     [`super::result::VmResult::exit_code`] from the SHM
///     `MSG_TYPE_EXIT` payload (or COM2 `KTSTR_EXIT:` sentinel) when
///     either is present, so a `-1` from the BSP run-loop is not
///     authoritative for caller-visible test outcome.
#[derive(Debug, Clone, Copy)]
enum BspExitReason {
    /// `kill.load(Acquire)` returned `true` at the top of the loop —
    /// some peer (an AP that observed [`ExitAction::Shutdown`] or
    /// [`ExitAction::Fatal`], the panic hook, the monitor thread on
    /// `MSG_TYPE_SCHED_EXIT`, or `collect_results`) flipped the flag.
    /// In particular, on a clean test exit where the kernel's i8042
    /// reset OUT is dispatched to a non-BSP vCPU, the AP path sets
    /// `kill` and the BSP exits via this branch. The default value
    /// for the local — every break path that does not explicitly
    /// reassign falls into this case.
    ExternalKill,
    /// BSP itself observed [`ExitAction::Shutdown`] from
    /// `classify_exit` (i8042 reset on x86_64, PSCI SystemEvent /
    /// `VcpuExit::Shutdown` on aarch64). The only path that sets
    /// exit_code to 0.
    Shutdown,
    /// BSP itself observed [`ExitAction::Fatal`] from `classify_exit`
    /// (`VcpuExit::FailEntry` or `VcpuExit::InternalError`). Kill
    /// flag is propagated to peers before break.
    Fatal,
    /// `bsp.run()` returned a non-EINTR/EAGAIN errno. Indicates a
    /// permanent KVM_RUN failure on the BSP vCPU fd.
    RunError,
    /// The wall-clock timeout (`run_start.elapsed() > timeout`) ran
    /// out before any other exit condition fired. Returned with
    /// `timed_out=true`. Logged from the early-return branch at the
    /// top of the loop.
    Timeout,
}

/// Decoded contents of the SHM snapshot request slot. Read by the
/// freeze coordinator's doorbell handler each time the doorbell
/// eventfd fires; identifies which dispatch path (CAPTURE / WATCH /
/// unknown) the guest requested and carries the correlated
/// `request_id` the host stamps back into `snapshot_reply_id`.
struct SnapshotRequest {
    request_id: u32,
    kind: u32,
    tag: String,
}

/// Read the snapshot request slot from the guest's SHM region.
/// Returns `Some(SnapshotRequest)` when a populated request is
/// present (`request_id != 0`); `None` when the SHM region is not
/// configured, the request_id is zero, or the kind is NONE — that
/// last case being the "host-side trigger fired the doorbell
/// without a guest request" path.
fn read_snapshot_request(
    mem: Option<&monitor::reader::GuestMem>,
    shm_base: Option<u64>,
) -> Option<SnapshotRequest> {
    let mem = mem?;
    let shm_base = shm_base?;
    let request_id = mem.read_u32(shm_base, shm_ring::SNAPSHOT_REQUEST_ID_OFFSET);
    let kind = mem.read_u32(shm_base, shm_ring::SNAPSHOT_KIND_OFFSET);
    if request_id == 0 || kind == shm_ring::SHM_SNAPSHOT_KIND_NONE {
        return None;
    }
    let mut tag_bytes = [0u8; shm_ring::SHM_SNAPSHOT_TAG_MAX];
    for (i, b) in tag_bytes.iter_mut().enumerate() {
        *b = mem.read_u8(shm_base, shm_ring::SNAPSHOT_TAG_OFFSET + i);
    }
    let len = tag_bytes
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(shm_ring::SHM_SNAPSHOT_TAG_MAX);
    let tag = String::from_utf8_lossy(&tag_bytes[..len]).to_string();
    Some(SnapshotRequest {
        request_id,
        kind,
        tag,
    })
}

/// Stamp a reply into the SHM snapshot reply slot. Writes the reason
/// buffer first (so the guest's Acquire load on `snapshot_reply_id`
/// observes the reason atomically), then the status, then the
/// reply_id last — mirroring the guest's publish ordering on the
/// request side.
fn write_snapshot_reply(
    mem: Option<&monitor::reader::GuestMem>,
    shm_base: Option<u64>,
    request_id: u32,
    status: u32,
    reason: &str,
) {
    let Some(mem) = mem else {
        return;
    };
    let Some(shm_base) = shm_base else {
        return;
    };
    // Reason buffer: NUL-terminated UTF-8, truncated to the buffer
    // size. Pad the trailing bytes with zeros so a stale reason
    // from a prior request does not bleed through.
    let reason_bytes = reason.as_bytes();
    let len = reason_bytes.len().min(shm_ring::SHM_SNAPSHOT_TAG_MAX);
    if len > 0 {
        mem.write_bytes(
            shm_base + shm_ring::SNAPSHOT_REASON_OFFSET as u64,
            &reason_bytes[..len],
        );
    }
    // Trailing zero pad (always at least one byte after the truncated
    // text so `from_utf8_lossy` stops cleanly on the guest side).
    let mut zero = [0u8; shm_ring::SHM_SNAPSHOT_TAG_MAX];
    zero[..shm_ring::SHM_SNAPSHOT_TAG_MAX - len].copy_from_slice(
        &[0u8; shm_ring::SHM_SNAPSHOT_TAG_MAX][..shm_ring::SHM_SNAPSHOT_TAG_MAX - len],
    );
    if len < shm_ring::SHM_SNAPSHOT_TAG_MAX {
        mem.write_bytes(
            shm_base + (shm_ring::SNAPSHOT_REASON_OFFSET + len) as u64,
            &zero[..shm_ring::SHM_SNAPSHOT_TAG_MAX - len],
        );
    }
    // Status, then reply_id last (publish order).
    mem.write_u32(shm_base, shm_ring::SNAPSHOT_STATUS_OFFSET, status);
    std::sync::atomic::fence(Ordering::Release);
    mem.write_u32(shm_base, shm_ring::SNAPSHOT_REPLY_ID_OFFSET, request_id);
}

/// Resolve a kernel symbol by name from the vmlinux ELF and arm a
/// user-watchpoint slot (DR1..=DR3) on it. Returns the slot index
/// (0..=2 mapping to DR1..=DR3) on success, or a host-side
/// diagnostic on failure.
///
/// The vCPU thread's `self_arm_watchpoint` notices the change on
/// the next loop iteration (Acquire load on the slot's
/// `request_kva`) and reprograms `KVM_SET_GUEST_DEBUG` with the
/// new DR layout.
fn arm_user_watchpoint(
    watchpoint: &Arc<super::vcpu::WatchpointArm>,
    kernel_path: &std::path::Path,
    symbol: &str,
) -> std::result::Result<usize, String> {
    // Check cap and find a free slot.
    let mut free_slot: Option<usize> = None;
    for (i, slot) in watchpoint.user.iter().enumerate() {
        if slot.request_kva.load(Ordering::Acquire) == 0 {
            free_slot = Some(i);
            break;
        }
    }
    let Some(idx) = free_slot else {
        return Err(format!(
            "no free DR slot — DR1..=DR3 all occupied by prior \
             Op::WatchSnapshot registrations (cap = {})",
            watchpoint.user.len()
        ));
    };
    // Resolve the symbol via vmlinux ELF parse.
    let data = std::fs::read(kernel_path)
        .map_err(|e| format!("read vmlinux at {}: {e}", kernel_path.display()))?;
    let elf = goblin::elf::Elf::parse(&data).map_err(|e| format!("parse vmlinux ELF: {e}"))?;
    let kva = elf
        .syms
        .iter()
        .find(|s| s.st_value != 0 && elf.strtab.get_at(s.st_name) == Some(symbol))
        .map(|s| s.st_value)
        .ok_or_else(|| format!("symbol '{symbol}' not found in vmlinux symtab"))?;
    if kva & 0x3 != 0 {
        return Err(format!(
            "symbol '{symbol}' KVA {kva:#x} is not 4-byte aligned \
             (Intel SDM Vol. 3B Chapter 17 requires DR_LEN_4 \
             alignment for hardware watchpoints)"
        ));
    }
    // Publish tag first, then KVA last (the vCPU's Acquire load on
    // request_kva synchronises-with this Release; the tag must be
    // visible by the time the vCPU latches a hit on this slot).
    {
        let mut tag_guard = watchpoint.user[idx]
            .tag
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *tag_guard = symbol.to_string();
    }
    watchpoint.user[idx]
        .request_kva
        .store(kva, Ordering::Release);
    // Pre-emptively kick every vCPU out of KVM_RUN so it reaches
    // self_arm_watchpoint promptly. The watchpoint's hit_evt is the
    // cleanest available wake fd; a write here causes any vCPU
    // currently inside the run loop to (best-effort) re-check the
    // slot on its next iteration. There is no per-slot kick; the
    // existing kick mechanism (immediate_exit + SIGRTMIN) is
    // reserved for the freeze rendezvous.
    Ok(idx)
}

/// Build a tagged sibling path for an on-demand snapshot dump. Given
/// `failure_dump.json` and counter `0`, returns
/// `failure_dump.on_demand_0.json`; with no extension, returns
/// `failure_dump.on_demand_0`. The error-class trigger keeps the
/// unmodified base path so the two emission paths never alias.
fn on_demand_tagged_path(base: &std::path::Path, counter: u32) -> std::path::PathBuf {
    let mut tagged = base.to_path_buf();
    let raw_stem = base.file_stem().and_then(|s| s.to_str()).unwrap_or("dump");
    let stem = raw_stem.strip_suffix(".failure-dump").unwrap_or(raw_stem);
    let ext = base.extension().and_then(|e| e.to_str());
    let new_name = match ext {
        Some(ext) => format!("{stem}.on_demand_{counter}.{ext}"),
        None => format!("{stem}.on_demand_{counter}"),
    };
    tagged.set_file_name(new_name);
    tagged
}

/// Build a name-tagged sibling path for a CAPTURE-class on-demand
/// snapshot. Given `{base}/{stem}.failure-dump.json` and tag
/// `mid_run`, returns `{base}/{stem}.snapshot.mid_run.json`. Used by
/// the freeze coordinator's CAPTURE handler so the test's
/// post-scenario reader can find the file by snapshot tag without
/// guessing the on-demand counter.
///
/// The tag is sanitised: any byte that is not `[A-Za-z0-9._-]` is
/// replaced with `_` to keep the resulting filename safe across
/// filesystems regardless of what UTF-8 the guest passed.
fn snapshot_tagged_path(base: &std::path::Path, tag: &str) -> std::path::PathBuf {
    let mut tagged = base.to_path_buf();
    let raw_stem = base.file_stem().and_then(|s| s.to_str()).unwrap_or("dump");
    let stem = raw_stem.strip_suffix(".failure-dump").unwrap_or(raw_stem);
    let ext = base.extension().and_then(|e| e.to_str());
    let safe_tag: String = tag
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let new_name = match ext {
        Some(ext) => format!("{stem}.snapshot.{safe_tag}.{ext}"),
        None => format!("{stem}.snapshot.{safe_tag}"),
    };
    tagged.set_file_name(new_name);
    tagged
}

/// Wait up to `timeout_ms` for `evt` to become readable, returning
/// when the eventfd's counter is non-zero OR the timeout elapses.
/// Does not consume the counter — `poll(POLLIN)` is level-triggered,
/// so a single `evt.write(1)` from any cloned writer fans out to
/// every reader: each reader's poll returns immediately (level held
/// high) and re-checks its own readiness condition. This is the
/// broadcast wake primitive for the probes-ready eventfd shared
/// across the monitor and bpf-map-write threads — the first thread
/// to detect its readiness writes 1, and every other waiter
/// observes the level transition without racing on a consuming
/// `read()`.
///
/// Treats every poll() return path (timeout, ready, EINTR, error)
/// as "wake-up time" — the caller re-checks its own deadline and
/// SHM-byte / kernel-state condition each iteration regardless.
/// EINTR from a signal during the wait is therefore harmless.
fn poll_eventfd_until_ready_or_timeout(evt: &EventFd, timeout_ms: i32) {
    use std::os::fd::AsRawFd;
    let mut pfd = libc::pollfd {
        fd: evt.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: pfd is a valid &mut pointing to a single pollfd; nfds
    // is 1 matching the slice length; timeout_ms is forwarded
    // directly to the kernel which interprets it per poll(2). The
    // return value is intentionally discarded — every outcome
    // (ready, timeout, EINTR, error) drives the caller back into
    // its own condition check loop, which re-evaluates kill /
    // deadline / SHM-byte each iteration.
    unsafe {
        libc::poll(&mut pfd, 1, timeout_ms);
    }
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

        // Snapshot doorbell ioeventfd. The guest fires an on-demand
        // capture by writing 4 bytes at `kvm::DOORBELL_MMIO_GPA`; KVM
        // dispatches the write in-kernel and signals this eventfd
        // without a userspace exit on the vCPU thread. The freeze
        // coordinator polls a clone of this fd in its main loop and
        // runs `freeze_and_capture(false)` on each pending event,
        // independent of the error-class freeze state machine.
        // `NoDatamatch` so the dispatch fires for any value — the
        // guest's request_id flows through the SHM ring (host-side
        // tag channel), not via KVM's datamatch comparator.
        // `EFD_NONBLOCK` so the coordinator's per-iteration `read()`
        // returns `WouldBlock` instead of stalling when no doorbell
        // is pending.
        let doorbell_evt =
            EventFd::new(EFD_NONBLOCK).context("create snapshot doorbell EventFd")?;
        vm.vm_fd
            .register_ioevent(
                &doorbell_evt,
                &IoEventAddress::Mmio(kvm::DOORBELL_MMIO_GPA),
                NoDatamatch,
            )
            .context("register snapshot doorbell ioeventfd")?;
        // Clone the fd so the coordinator owns its own handle and the
        // host-side trigger path (e.g. `Op::Snapshot` running on the
        // host with no in-guest write) can take additional clones.
        // EventFd::try_clone uses dup(2), so all clones share the
        // same kernel counter — a write on any fd wakes a read on
        // any clone.
        let doorbell_evt_for_coord = doorbell_evt
            .try_clone()
            .context("clone snapshot doorbell EventFd for coordinator")?;
        // Serialises on-demand captures against themselves: the
        // coordinator sets this Acquire-bool while a doorbell capture
        // runs and clears it on completion, so a flood of doorbell
        // writes is collapsed into one capture per thaw.
        // Independent of `freeze_state`, which governs only the
        // error-class trigger machine — on-demand captures must
        // service even when `freeze_state == Done` so post-failure
        // `Op::Snapshot` calls still work.
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
        // while polling either guest SHM or guest kernel state.
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
        // Without a corresponding host-side dispatch the probe times
        // out and the guest falls back to the legacy 200 ms SHM poll;
        // wiring the device here lets the guest's `shm_poll_loop`
        // block on `/dev/hvc0` and wake within microseconds when the
        // host pushes a byte. Coordinator and watchdog use this as the
        // "ping" channel paired with the SHM control-byte writes
        // (`DUMP_REQ_OFFSET`, `STALL_REQ_OFFSET`, signal slot 0).
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
        // Wake fd paired with the `kill` AtomicBool. Setters that
        // flip `kill` (collect_results, vCPU shutdown classifier,
        // panic hook) ALSO write to this EventFd so any consumer
        // sleeping on `epoll_wait` returns within microseconds of
        // the flip rather than waiting up to one full poll
        // interval. Production consumers: the monitor loop and the
        // watchdog thread, both spawned below. `EFD_NONBLOCK` keeps
        // the writer's `write()` from stalling if the counter is
        // already saturated; the AtomicBool remains the source of
        // truth — the EventFd is purely a wake signal.
        let kill_evt = Arc::new(EventFd::new(EFD_NONBLOCK).context("create kill EventFd")?);
        // Failure-dump freeze rendezvous: broadcast `freeze` flag plus a
        // per-vCPU `parked` ACK, parallel to the existing `kill` +
        // `exited` shutdown rendezvous. The freeze coordinator
        // (spawned below alongside the watchdog) polls the BPF probe's
        // `ktstr_err_exit_detected` .bss flag via `BpfMapAccessor`;
        // when the flag flips it sets `freeze`, kicks every vCPU,
        // awaits N-of-N parked confirmations, runs the dump (placeholder
        // in this batch), and then clears `freeze` to thaw.
        let freeze = Arc::new(AtomicBool::new(false));
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

        let monitor_handle = self.start_monitor(
            &vm,
            &kill,
            &kill_evt,
            run_start,
            vcpu_pthreads,
            perf_capture.clone(),
            probes_ready_evt_for_monitor,
            Some(virtio_con.clone()),
        )?;

        // BPF map write thread: sleeps, discovers a BPF map, writes a value.
        let bpf_write_handle = self.start_bpf_map_write(&vm, &kill, probes_ready_evt_for_bpf)?;

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
        // Wake fd paired with `bsp_done`. Setters (run_vm post-loop,
        // BSP panic hook) flip the AtomicBool AND write `1` to this
        // EventFd so the freeze coordinator's epoll wait returns
        // immediately. Mirrors the `kill` / `kill_evt` pair above.
        // EFD_NONBLOCK so a doubled write (panic hook AND post-loop
        // store) cannot stall — either edge is sufficient.
        let bsp_done_evt = Arc::new(EventFd::new(EFD_NONBLOCK).context("create bsp_done EventFd")?);
        let kill_for_watchdog = kill.clone();
        // Wake fds the watchdog blocks on via epoll, paired with the
        // `kill_for_watchdog` and `bsp_done_for_wd` AtomicBools above.
        // The watchdog wakes within microseconds of either flip
        // instead of polling on a 100 ms thread::sleep cadence.
        let kill_evt_for_watchdog = kill_evt.clone();
        let bsp_done_evt_for_wd = bsp_done_evt.clone();
        let rt_watchdog = self.performance_mode;
        let wd_service_cpu = self.pinning_plan.as_ref().and_then(|p| p.service_cpu);
        // Clone the virtio-console Arc into the watchdog so the
        // soft-deadline path can push a wake byte to `/dev/hvc0`
        // alongside the SHM `SIGNAL_SHUTDOWN_REQ` write. The guest's
        // `shm_poll_loop` blocks on the device read; without the byte
        // it observes the SHM signal only at the legacy 200 ms poll
        // cadence.
        let wd_virtio_con = virtio_con.clone();

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
        // COM2 mirror for the post-thaw SCHED_OUTPUT_END marker poll.
        // After the late-snapshot dump emission completes the
        // coordinator polls this serial's captured output for the
        // closing delimiter the guest's `dump_sched_output` writes
        // before signalling MSG_TYPE_SCHED_EXIT — the grace window
        // protects the delimiter from being truncated when host-side
        // teardown observes SCHED_EXIT before the guest's flush has
        // hit COM2. The scheduler's start_sched_exit_monitor calls
        // dump_sched_output (which writes SCHED_OUTPUT_START / log
        // body / SCHED_OUTPUT_END through write_com2) immediately
        // before write_msg(MSG_TYPE_SCHED_EXIT); without the grace
        // window a fast host monitor + slow guest serial flush race
        // surfaces as a partial COM2 capture. The verifier's
        // parse_sched_output_partial helper is the safety net for
        // cases where the marker still has not arrived after the
        // grace window expires.
        let freeze_coord_com2 = com2.clone();
        // Install the COM2 captured-output notifier and capture
        // its eventfd handle for the coordinator's epoll set. Each
        // guest write to COM2's DATA register bumps this counter
        // (see `Serial::install_data_evt` in vmm/console.rs), so a
        // wait on the eventfd wakes within microseconds of the
        // guest emitting any new byte — replacing the prior
        // `thread::sleep(50ms)` poll cadence the post-thaw grace
        // window used to detect SCHED_OUTPUT_END.
        //
        // Failure to install isn't fatal: the grace window falls
        // back to a single bounded wait on `kill_evt` /
        // `bsp_done_evt` and re-checks the buffer once at the end.
        // Edge case is marginal — a working install gates the wake
        // on every guest byte; a failed install means we wait for
        // the SCHED_OUTPUT_END_GRACE deadline regardless. Either
        // way the verifier's parse_sched_output_partial recovers
        // a truncated tail.
        let freeze_coord_com2_data_evt = match com2.lock().install_data_evt() {
            Ok(evt) => Some(evt),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "freeze-coord: install COM2 data_evt failed; \
                     grace window will wait for the deadline without \
                     event-driven wakes"
                );
                None
            }
        };
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
        // SHM base offset within `freeze_coord_mem`. The on-demand
        // doorbell handler reads the snapshot request slot (request
        // id, kind, tag) from this base + the snapshot offsets
        // declared in `shm_ring`; failure dispatch (CAPTURE vs WATCH)
        // and the reply write target the same slot. `0` when no
        // SHM region is configured — the doorbell handler then
        // skips request-driven dispatch and falls back to the
        // legacy "any doorbell == capture-now without tag"
        // semantics tagged with a placeholder name.
        let freeze_coord_shm_base = if self.shm_size > 0 {
            freeze_coord_mem.as_ref().map(|m| m.size() - self.shm_size)
        } else {
            None
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
        // NUMA node count from the configured topology. Forwarded
        // into the scx walker (per-node global DSQ pass) and the
        // per-node NUMA event walker. Defaults to 1 on UMA topologies.
        let freeze_coord_num_nodes = self.topology.num_numa_nodes();
        // On-demand snapshot doorbell — moved into the closure. The
        // coordinator polls `read()` non-blocking each iteration and
        // calls `freeze_and_capture(false)` on every pending event.
        let freeze_coord_doorbell = doorbell_evt_for_coord;
        let freeze_coord_on_demand_in_flight = on_demand_in_flight.clone();
        let freeze_coord_snapshot_bridge = snapshot_bridge.clone();
        // Wake-fd handles for the coord epoll loop. `kill_evt` and
        // `bsp_done_evt` are written by every thread that flips the
        // matching AtomicBool (vCPU shutdown classifier, BSP panic
        // hook, AP panic hook, collect_results); the epoll wait
        // fires immediately on either edge instead of polling on a
        // 500 µs sleep cadence. The watchpoint hit_evt clone lets
        // the coord wake on a hardware-watchpoint fire (vCPU thread
        // calls `WatchpointArm::latch_hit`, which writes the
        // eventfd alongside the AtomicBool flip). All three live
        // for the lifetime of the run — `collect_results` joins
        // the coord BEFORE the eventfds drop.
        let freeze_coord_kill_evt = kill_evt.clone();
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
                    /// `scx_tasks.next` via `text_kva_to_pa` and
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
                // text_kva_to_pa, the per-rq walker uses `runqueues`
                // + `__per_cpu_offset` to address each CPU's `rq`.
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
                // Lazy-resolved KVA of `*scx_root->exit_kind` — the
                // hardware-watchpoint target that replaces the prior
                // BPF .bss `ktstr_err_exit_detected` poll for
                // late-trigger detection. Resolution sequence
                // mirrors the cached_bss_pa pattern:
                //   1. read scx_root_kva from KernelSymbols (resolved
                //      once at coord-start via vmlinux);
                //   2. translate scx_root_kva → root_pa via
                //      `text_kva_to_pa` (it lives in the kernel text
                //      mapping, not vmalloc);
                //   3. read u64 at root_pa to get sched_kva (the
                //      vmalloc-allocated `struct scx_sched`);
                //   4. when sched_kva is non-zero AND the BTF
                //      `exit_kind` offset is known, publish
                //      `sched_kva + exit_kind_offset` into
                //      `freeze_coord_watchpoint.request_kva`. Each
                //      vCPU thread polls that slot before its next
                //      KVM_RUN and self-arms.
                //
                // `*scx_root` only becomes non-NULL once a sched_ext
                // scheduler attaches; before that we silently retry
                // — the BPF .bss fallback (still wired up below)
                // covers the gap. Once published, the request value
                // is monotonic for the run: the kernel scx_sched
                // struct lives until the scheduler detaches, which
                // happens AFTER err_exit fires (we only need the
                // address until the watchpoint trips once).
                let mut watchpoint_published_kva: Option<u64> = None;
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
                // On-demand snapshot counter. Bumped each time the
                // doorbell handler successfully enters the in-flight
                // critical section, used to namespace per-doorbell
                // dump file paths so a scenario with multiple
                // `Op::Snapshot` calls produces distinct artifacts
                // instead of overwriting the prior on-demand dump.
                // Independent of `freeze_state` per the on-demand
                // protocol — error-class and on-demand captures
                // never interleave path names.
                let mut on_demand_counter: u32 = 0;
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
                // bsp_done, doorbell, watchpoint hit, scanner tick)
                // OR `POLL_TIMEOUT_MS` elapses. The previous
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
                const TOKEN_DOORBELL: u64 = 2;
                const TOKEN_WATCHPOINT: u64 = 3;
                const TOKEN_SCANNER: u64 = 4;
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
                        freeze_coord_doorbell.as_raw_fd(),
                        TOKEN_DOORBELL,
                        "doorbell_evt",
                    ),
                    (
                        freeze_coord_hit_evt.as_raw_fd(),
                        TOKEN_WATCHPOINT,
                        "watchpoint_hit_evt",
                    ),
                    (scanner_tfd.as_raw_fd(), TOKEN_SCANNER, "scanner_tfd"),
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
                let mut events_buf = [EpollEvent::default(); 5];
                // First iteration always runs scan-tick work so
                // boot-race lazy resolution attempts fire
                // immediately rather than waiting up to 100 ms for
                // the timerfd's first edge. Subsequent iterations
                // gate scan-tick on the SCANNER token (or on a
                // POLL_TIMEOUT-driven wake) — the watchpoint and
                // doorbell events themselves never set scan_tick,
                // which is correct: those triggers are fast paths
                // that should not block the next wake on heavy
                // bss-PA / scan_ctx work.
                let mut scan_tick: bool;
                let mut first_iter = true;
                let mut doorbell_pending_from_epoll = false;
                while !freeze_coord_kill.load(Ordering::Acquire) {
                    if freeze_coord_bsp_done.load(Ordering::Acquire) {
                        return;
                    }
                    if first_iter {
                        scan_tick = true;
                        first_iter = false;
                    } else {
                        scan_tick = false;
                        let event_count = match epoll.wait(POLL_TIMEOUT_MS, &mut events_buf) {
                            Ok(n) => n,
                            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                            Err(e) => {
                                tracing::error!(
                                    error = %e,
                                    "freeze-coord: epoll_wait failed; exiting coordinator"
                                );
                                return;
                            }
                        };
                        // Drain every fd that fired. Tokens map
                        // 1:1 to source fds; KILL / BSP_DONE both
                        // exit the loop, the others either set
                        // scan_tick (SCANNER) or surface state via
                        // the existing latch reads later in the
                        // body (DOORBELL / WATCHPOINT).
                        for ev in &events_buf[..event_count] {
                            match ev.data() {
                                TOKEN_KILL => {
                                    // Drain the kill_evt counter
                                    // so a future re-enter (none in
                                    // this design) wouldn't see a
                                    // stale wake. Failure (counter
                                    // already at 0 from a racing
                                    // reader, EAGAIN) is benign —
                                    // the AtomicBool is the source
                                    // of truth and the outer
                                    // `while` re-checks it.
                                    let _ = freeze_coord_kill_evt.read();
                                    return;
                                }
                                TOKEN_BSP_DONE => {
                                    let _ = freeze_coord_bsp_done_evt.read();
                                    return;
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
                                TOKEN_DOORBELL => {
                                    // Drain via the same `read()`
                                    // the doorbell handler below
                                    // uses. A best-effort drain
                                    // here lets the handler treat
                                    // the on-demand path as a flag
                                    // rather than re-doing the
                                    // EAGAIN check in two places.
                                    doorbell_pending_from_epoll = true;
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
                                _ => {}
                            }
                        }
                        // Re-check kill/bsp_done — they may have
                        // flipped via the AtomicBool path before
                        // the eventfd was drained, or via a path
                        // that updated the bool but failed to write
                        // the eventfd (counter overflow under
                        // EFD_NONBLOCK).
                        if freeze_coord_kill.load(Ordering::Acquire) {
                            return;
                        }
                        if freeze_coord_bsp_done.load(Ordering::Acquire) {
                            return;
                        }
                    }
                    // Lazy retry: the accessor's GuestKernel walk
                    // depends on guest-memory bootstrap symbols
                    // populated by the guest kernel during boot, so
                    // an attempt at coord-start can fail. Retry each
                    // iteration until success; gated on
                    // `owned_accessor.is_none()` so the heavy
                    // ELF/BTF parse runs at most once after the
                    // first successful attempt.
                    if scan_tick
                        && owned_accessor.is_none()
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
                    if scan_tick
                        && owned_prog_accessor.is_none()
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
                    if scan_tick
                        && prog_per_cpu_offsets.is_none()
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
                        // Defer caching until every offset slot is
                        // non-zero — a guest still populating per-CPU
                        // areas yields zero entries for the
                        // not-yet-initialised CPUs, and caching that
                        // would alias every such CPU's stats to CPU 0.
                        // Mirror the scan_ctx fix downstream: a single
                        // bad cache disables the prog_runtime_stats
                        // path for every subsequent stall on every
                        // 2+ vCPU VM where any secondary was still
                        // coming up at first read. A retry is cheap;
                        // a cached miss is permanent for the run.
                        // For prog_runtime_stats this means stats for
                        // CPUs that haven't booted yet are simply
                        // missing — acceptable, those CPUs have no
                        // stats anyway.
                        if !offsets.contains(&0) {
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
                    if scan_tick
                        && cached_bss_pa.is_none()
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
                            // Bind kernel once and reuse — pre-fix
                            // owned.guest_kernel() ran three times here
                            // and once again at the BTF Datasec walk
                            // below. The accessor is cheap but the
                            // repetition was noisy at the freeze hot
                            // path's read site.
                            let kernel = owned.guest_kernel();
                            let cr3_pa = kernel.cr3_pa();
                            let page_offset = kernel.page_offset();
                            let l5 = kernel.l5();
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
                    // Lazy-resolve the watchpoint target KVA
                    // (`*scx_root + exit_kind_offset`) and publish
                    // it to every vCPU thread.
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
                    //     uses;
                    //   - `*scx_root != 0` (sched_kva) — only true
                    //     once a sched_ext scheduler has attached.
                    //
                    // Once published, the BPF .bss fallback below
                    // continues to update `cached_bss_pa`; both
                    // signals can fire and the late-trigger arm (a
                    // few iterations down the loop) treats either
                    // as ground truth. The watchpoint's advantage is
                    // synchronous delivery (no 100 ms polling
                    // window) AND independence from the probe BPF
                    // program loading correctly.
                    if scan_tick
                        && watchpoint_published_kva.is_none()
                        && owned_accessor.is_some()
                        && let Some(ref syms) = dump_cpu_time_symbols
                        && let Some(scx_root_kva) = syms.scx_root
                        && let Some(ref scx_offsets) = dump_scx_walker_offsets
                        && let Some(ref sched_offs) = scx_offsets.sched
                        && let Some(ref mem) = freeze_coord_mem
                    {
                        // scx_root is a kernel-text-mapped pointer:
                        // text_kva_to_pa is the same translation
                        // `read_scx_sched_state` performs (see
                        // `monitor/scx_walker.rs::read_scx_sched_state`).
                        let root_pa = crate::monitor::symbols::text_kva_to_pa(scx_root_kva);
                        let sched_kva = mem.read_u64(root_pa, 0);
                        if sched_kva != 0 {
                            // exit_kind field KVA = base of scx_sched
                            // (vmalloc/slab) + BTF-resolved field
                            // offset. The kernel writes a 4-byte
                            // atomic_t at this address via
                            // `atomic_set` in scx_exit; the
                            // hardware watchpoint catches every such
                            // write regardless of the SCX_EXIT_*
                            // class.
                            let exit_kind_kva =
                                sched_kva.wrapping_add(sched_offs.exit_kind as u64);
                            // Translate the field's KVA to a host
                            // pointer so the vCPU thread can
                            // `read_volatile` the post-store value at
                            // fire time and gate `watchpoint.hit` on
                            // the error-class threshold (1024). Without
                            // this, the watchpoint fires on every
                            // exit_kind transition — including the
                            // clean `KIND -> SCX_EXIT_DONE` write that
                            // `scx_unregister` issues at end of every
                            // test — and produces a bogus failure
                            // dump on every clean shutdown.
                            //
                            // The kva lives in scx_sched's slab/vmalloc
                            // page; translate via the same
                            // direct-mapping-or-page-walk path the
                            // BPF .bss poll uses, then look up the
                            // host pointer via `host_ptr_for_pa`.
                            // `field_size` is 4 (the atomic_t holding
                            // exit_kind is a u32). On any resolve
                            // failure we skip publication this
                            // iteration; the next iteration retries
                            // (the publish block is gated on
                            // `watchpoint_published_kva.is_none()`).
                            let kernel = owned_accessor
                                .as_ref()
                                .map(|o| o.guest_kernel());
                            let resolve = kernel.and_then(|k| {
                                let kind_pa = crate::monitor::idr::translate_any_kva(
                                    mem,
                                    k.cr3_pa(),
                                    k.page_offset(),
                                    exit_kind_kva,
                                    k.l5(),
                                )?;
                                let host_ptr =
                                    mem.host_ptr_for_pa(kind_pa, 4)? as *mut u32;
                                Some((kind_pa, host_ptr))
                            });
                            match resolve {
                                Some((kind_pa, kind_host_ptr)) => {
                                    // Publication ordering: store
                                    // `kind_host_ptr` BEFORE
                                    // `request_kva`. The vCPU thread
                                    // loads `request_kva` with
                                    // Acquire and only reads
                                    // `kind_host_ptr` after — the
                                    // Release ordering on
                                    // `request_kva` makes the
                                    // earlier `kind_host_ptr` store
                                    // visible. Without this
                                    // ordering a vCPU could observe
                                    // a non-zero `request_kva`, arm
                                    // the watchpoint, fire on the
                                    // very next instruction, and
                                    // read a still-null
                                    // `kind_host_ptr`.
                                    freeze_coord_watchpoint
                                        .kind_host_ptr
                                        .store(kind_host_ptr, Ordering::Release);
                                    freeze_coord_watchpoint
                                        .request_kva
                                        .store(exit_kind_kva, Ordering::Release);
                                    watchpoint_published_kva = Some(exit_kind_kva);
                                    tracing::info!(
                                        exit_kind_kva =
                                            format_args!("{:#x}", exit_kind_kva),
                                        sched_kva =
                                            format_args!("{:#x}", sched_kva),
                                        kind_pa = format_args!("{:#x}", kind_pa),
                                        "freeze-coord: watchpoint target \
                                         published; vCPU threads will self-arm \
                                         KVM_SET_GUEST_DEBUG on next iteration"
                                    );
                                }
                                None => {
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
                            let cr3_pa = kernel.cr3_pa();
                            let page_offset = kernel.page_offset();
                            let l5 = kernel.l5();
                            // Translate jiffies_64's KVA to a PA.
                            // Lives in the kernel text/data mapping
                            // (text_kva_to_pa) — same as scx_root
                            // et al.
                            let jiffies_64_pa =
                                crate::monitor::symbols::text_kva_to_pa(jiffies_64_kva);
                            // Compute per-CPU rq PAs for the per-rq
                            // runnable_list walker. The KernelOffsets
                            // schema guarantees `runqueues != 0` (see
                            // `monitor/symbols.rs` — its absence is a
                            // construction-time error), so the only
                            // failure path here is reading
                            // `__per_cpu_offset` early during boot:
                            // the per-CPU offset table reads as zero
                            // for not-yet-online CPUs and the
                            // resulting rq_pas vec contains zeroes
                            // for those slots. `max_runnable_age_per_rq`
                            // short-circuits on `rq_pa == 0`, so a
                            // partially-resolved per-CPU table simply
                            // contributes nothing for the
                            // not-yet-online CPUs without poisoning
                            // the walk for the online ones. Empty
                            // rq_pas falls back to "global walk only"
                            // through `max_runnable_age`'s wrapper.
                            let pco_pa = crate::monitor::symbols::text_kva_to_pa(
                                syms.per_cpu_offset,
                            );
                            let pco_offsets = crate::monitor::symbols::read_per_cpu_offsets(
                                mem,
                                pco_pa,
                                freeze_coord_num_cpus,
                            );
                            let rq_pas = crate::monitor::symbols::compute_rq_pas(
                                syms.runqueues,
                                &pco_offsets,
                                page_offset,
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
                                syms.scx_watchdog_timestamp.map(
                                    crate::monitor::symbols::text_kva_to_pa,
                                );
                            Ok(RunnableScanCtx {
                                scx_tasks_kva,
                                rq_pas,
                                offsets: scan_offsets,
                                jiffies_64_pa,
                                watchdog_timestamp_pa,
                                cr3_pa,
                                page_offset,
                                l5,
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
                    // remains as fallback for kernels where the
                    // watchpoint never armed (no `scx_root` symbol,
                    // BTF stripped of `scx_sched`, or the
                    // `set_guest_debug` ioctl was rejected): the
                    // probe BPF program continues to latch
                    // `ktstr_err_exit_detected` from its tp_btf
                    // hook, and reading the original location keeps
                    // detection alive in those degraded
                    // configurations.
                    let watchpoint_hit =
                        freeze_coord_watchpoint.hit.load(Ordering::Acquire);
                    let bss_triggered =
                        if let (Some(pa), Some(mem)) = (cached_bss_pa, freeze_coord_mem.as_ref()) {
                            mem.read_u32(pa, 0) != 0
                        } else {
                            false
                        };
                    let err_triggered = watchpoint_hit || bss_triggered;
                    // On-demand snapshot doorbell. Drained
                    // unconditionally — even after the late snapshot
                    // has marked `freeze_state == Done` — so a
                    // post-failure scenario can still capture
                    // diagnostic state. The coordinator stays in the
                    // poll loop until kill / bsp_done; the
                    // `on_demand_in_flight` Acquire-bool serialises
                    // doorbell captures against themselves, collapsing
                    // a flood of writes into one capture per thaw.
                    // Independent of `freeze_state` per the on-demand
                    // protocol documented on `SnapshotBridge` and
                    // captured in `CaptureCallback`'s wire-shape doc.
                    // Doorbell pending if the epoll dispatch just
                    // flagged it OR a non-blocking read drains a
                    // pending counter (the host-side doorbell
                    // trigger path can clone the EventFd and write
                    // outside the epoll_wait window — first iter
                    // and post-Done sleeps both miss epoll). The
                    // read is non-blocking; WouldBlock indicates no
                    // pending event, mapping to false. Reset the
                    // epoll-flag latch each iteration so a transient
                    // pending state doesn't re-trigger after the
                    // handler runs.
                    let doorbell_pending = if doorbell_pending_from_epoll {
                        // Drain the underlying counter so the
                        // single edge that triggered the epoll wake
                        // doesn't re-fire after the handler runs.
                        let _ = freeze_coord_doorbell.read();
                        doorbell_pending_from_epoll = false;
                        true
                    } else {
                        match freeze_coord_doorbell.read() {
                            Ok(_) => true,
                            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => false,
                            Err(e) => {
                                tracing::warn!(
                                    error = %e,
                                    "freeze-coord: doorbell EventFd read failed"
                                );
                                false
                            }
                        }
                    };
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
                            tracing::info!(
                                gate_on_exit_kind,
                                "freeze-coord: freezing vCPUs for snapshot"
                            );
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
                            freeze_coord_freeze.store(true, Ordering::Release);
                            // Force-clear stale parked flags from any
                            // prior freeze cycle BEFORE pass 2 sends
                            // SIGRTMIN. Without this, a vCPU that was
                            // mid-`handle_freeze` exit at the end of
                            // cycle N (had loaded `freeze=false` and
                            // was about to run the trailing
                            // `parked.store(false, Release)`) can
                            // leave `parked=true` visible to the
                            // coordinator at the start of cycle N+1,
                            // and the rendezvous loop's Acquire load
                            // would race-observe that stale `true`
                            // and falsely conclude the vCPU has
                            // parked for the new cycle. Clearing
                            // before pass 2 is the safe window: pass
                            // 2 is the trigger that drives the vCPU
                            // into the next `handle_freeze`, which is
                            // where the legitimate `parked=true` for
                            // cycle N+1 originates. Release ordering
                            // pairs with the vCPU's Acquire load on
                            // `parked` and the legitimate Release
                            // store inside `handle_freeze`.
                            for p in freeze_coord_ap_parked.iter() {
                                p.store(false, Ordering::Release);
                            }
                            freeze_coord_bsp_parked.store(false, Ordering::Release);
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
                            //
                            // Wake mechanism: every vCPU + the
                            // virtio-blk worker writes 1 to the
                            // shared `parked_evt` AFTER its Release
                            // store on parked/paused. The wait
                            // here uses `poll(2)` over
                            // [parked_evt, kill_evt, bsp_done_evt]
                            // so the loop wakes within microseconds
                            // of the last parker rather than
                            // spinning on a 100µs cadence. The
                            // eventfd write ordering is
                            // load-bearing: the AtomicBool Release
                            // happens-before the eventfd write,
                            // and the coord drains the eventfd
                            // ONCE per wake then re-checks every
                            // parked flag. EAGAIN under
                            // EFD_NONBLOCK on the writer side is
                            // benign (the AtomicBool is the source
                            // of truth); EAGAIN on the drain
                            // (counter == 0) just means a previous
                            // edge already fired.
                            use std::os::fd::AsRawFd;
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
                                let now = Instant::now();
                                if now > deadline {
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
                                // all-parked predicate at the top —
                                // there is no per-fd dispatch here
                                // because the AtomicBool reads
                                // ARE the dispatch. EINTR from
                                // SIGRTMIN is harmless: the wait
                                // simply restarts.
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
                                // is benign — a prior drain or a
                                // racing reader already absorbed
                                // the edge. We deliberately do NOT
                                // drain kill_evt / bsp_done_evt
                                // here: those are owned by the
                                // outer epoll loop in the
                                // coordinator's main body, and
                                // draining them would suppress the
                                // edges that wake the outer loop
                                // on shutdown.
                                let _ = freeze_coord_parked_evt.read();
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
                                    freeze_coord_mem.as_ref(),
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
                                        let cr3_pa = kernel.cr3_pa();
                                        let page_offset = kernel.page_offset();
                                        let l5 = kernel.l5();
                                        match crate::monitor::idr::translate_any_kva(
                                            mem, cr3_pa, page_offset, kva, l5,
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
                                    return None;
                                }
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
                                    freeze_coord_mem.as_ref(),
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
                                };
                                tracing::warn!(
                                    owned_accessor = owned_accessor.is_some(),
                                    dump_btf = dump_btf.is_some(),
                                    "freeze-coord: dump prerequisites unavailable; \
                                     emitting partial report with vcpu_regs only"
                                );
                                Some((report, capture_start))
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
                    // On-demand snapshot handler. Runs whenever the
                    // doorbell eventfd had a pending event drained
                    // above, regardless of `freeze_state`. Serialised
                    // against itself via `on_demand_in_flight`: a
                    // pending capture short-circuits a doorbell write
                    // that arrives mid-rendezvous, so doorbell floods
                    // collapse to one capture per thaw.
                    //
                    // Reads the snapshot request slot from SHM (kind,
                    // tag, request_id) and dispatches CAPTURE vs
                    // WATCH. CAPTURE runs `freeze_and_capture(false)`
                    // and stores the report on the bridge under the
                    // tag. WATCH resolves the symbol via the kernel
                    // ELF, allocates a free DR1..=DR3 slot, publishes
                    // the resolved KVA + tag into `WatchpointArm`,
                    // and replies OK so every vCPU's
                    // `self_arm_watchpoint` picks up the new arm
                    // before its next KVM_RUN.
                    if doorbell_pending
                        && !freeze_coord_on_demand_in_flight
                            .swap(true, Ordering::AcqRel)
                    {
                        let request = read_snapshot_request(
                            freeze_coord_mem.as_ref(),
                            freeze_coord_shm_base,
                        );
                        match request {
                            Some(SnapshotRequest {
                                request_id,
                                kind: shm_ring::SHM_SNAPSHOT_KIND_CAPTURE,
                                tag,
                            }) => {
                                tracing::info!(
                                    request_id,
                                    %tag,
                                    "freeze-coord: doorbell CAPTURE request"
                                );
                                let on_demand = freeze_and_capture(false);
                                if let Some(ref blk) = freeze_coord_virtio_blk {
                                    blk.lock().resume();
                                }
                                freeze_coord_freeze.store(false, Ordering::Release);
                                let _ = freeze_coord_thaw_evt.write(1);
                                let mut reply_status =
                                    shm_ring::SHM_SNAPSHOT_STATUS_OK;
                                let mut reply_reason = String::new();
                                if let Some((report, capture_start)) = on_demand {
                                    let map_count = report.maps.len();
                                    let vcpu_regs_count =
                                        report.vcpu_regs.len();
                                    let tasks_enriched =
                                        report.task_enrichments.len();
                                    // Persist the captured report on
                                    // the bridge under the
                                    // guest-supplied tag. The test
                                    // code drains the bridge after VM
                                    // exit and walks the reports via
                                    // the public `Snapshot` accessor.
                                    freeze_coord_snapshot_bridge
                                        .store(&tag, report.clone());
                                    // File mirror for
                                    // operator inspection AND for
                                    // the in-guest scenario to read
                                    // back via `sidecar_dir()` —
                                    // `{base}.snapshot.{tag}.json`
                                    // is deterministic from the
                                    // guest-supplied tag, no
                                    // counter race.
                                    if let Some(ref base_path) =
                                        freeze_coord_dump_path
                                    {
                                        let tagged = snapshot_tagged_path(
                                            base_path, &tag,
                                        );
                                        if let Some(parent) = tagged.parent() {
                                            let _ = std::fs::create_dir_all(parent);
                                        }
                                        match serde_json::to_string_pretty(
                                            &report,
                                        ) {
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
                                } else {
                                    reply_status =
                                        shm_ring::SHM_SNAPSHOT_STATUS_ERR;
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
                                write_snapshot_reply(
                                    freeze_coord_mem.as_ref(),
                                    freeze_coord_shm_base,
                                    request_id,
                                    reply_status,
                                    &reply_reason,
                                );
                            }
                            Some(SnapshotRequest {
                                request_id,
                                kind: shm_ring::SHM_SNAPSHOT_KIND_WATCH,
                                tag,
                            }) => {
                                tracing::info!(
                                    request_id,
                                    %tag,
                                    "freeze-coord: doorbell WATCH request"
                                );
                                let Some(ref vmlinux) = freeze_coord_vmlinux else {
                                    write_snapshot_reply(
                                        freeze_coord_mem.as_ref(),
                                        freeze_coord_shm_base,
                                        request_id,
                                        shm_ring::SHM_SNAPSHOT_STATUS_ERR,
                                        "vmlinux not found in kernel dir",
                                    );
                                    on_demand_in_flight.store(false, Ordering::Release);
                                    continue;
                                };
                                let arm_result = arm_user_watchpoint(
                                    &freeze_coord_watchpoint,
                                    vmlinux.as_path(),
                                    &tag,
                                );
                                let (status, reason) = match arm_result {
                                    Ok(slot_idx) => {
                                        tracing::info!(
                                            request_id,
                                            %tag,
                                            slot_idx,
                                            "freeze-coord: hardware watchpoint armed"
                                        );
                                        (
                                            shm_ring::SHM_SNAPSHOT_STATUS_OK,
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
                                            shm_ring::SHM_SNAPSHOT_STATUS_ERR,
                                            reason,
                                        )
                                    }
                                };
                                write_snapshot_reply(
                                    freeze_coord_mem.as_ref(),
                                    freeze_coord_shm_base,
                                    request_id,
                                    status,
                                    &reason,
                                );
                            }
                            Some(SnapshotRequest {
                                request_id,
                                kind,
                                tag,
                            }) => {
                                tracing::warn!(
                                    request_id,
                                    %tag,
                                    kind,
                                    "freeze-coord: doorbell with unknown kind"
                                );
                                write_snapshot_reply(
                                    freeze_coord_mem.as_ref(),
                                    freeze_coord_shm_base,
                                    request_id,
                                    shm_ring::SHM_SNAPSHOT_STATUS_ERR,
                                    &format!("unknown snapshot kind {kind}"),
                                );
                            }
                            None => {
                                // Doorbell fired without an SHM
                                // request slot populated — most
                                // likely a host-side trigger via
                                // `doorbell_evt.write(1)` outside
                                // any guest-driven flow. Fall back
                                // to the legacy "anonymous capture"
                                // behaviour, storing under the same
                                // synthetic tag the file mirror
                                // uses.
                                tracing::info!(
                                    "freeze-coord: doorbell fired without SHM request \
                                     (host-side trigger?); running anonymous capture"
                                );
                                let on_demand = freeze_and_capture(false);
                                if let Some(ref blk) = freeze_coord_virtio_blk {
                                    blk.lock().resume();
                                }
                                freeze_coord_freeze.store(false, Ordering::Release);
                                let _ = freeze_coord_thaw_evt.write(1);
                                if let Some((report, _)) = on_demand {
                                    let counter = on_demand_counter;
                                    on_demand_counter =
                                        on_demand_counter.wrapping_add(1);
                                    let synth_tag =
                                        format!("anonymous_{counter}");
                                    freeze_coord_snapshot_bridge
                                        .store(&synth_tag, report.clone());
                                    if let Some(ref base_path) =
                                        freeze_coord_dump_path
                                    {
                                        let tagged = on_demand_tagged_path(
                                            base_path, counter,
                                        );
                                        if let Some(parent) = tagged.parent() {
                                            let _ = std::fs::create_dir_all(parent);
                                        }
                                        if let Ok(json) =
                                            serde_json::to_string_pretty(&report)
                                            && let Err(e) =
                                                std::fs::write(&tagged, &json)
                                        {
                                            tracing::warn!(
                                                path = %tagged.display(),
                                                error = %e,
                                                "freeze-coord: anonymous on-demand dump file write failed"
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        freeze_coord_on_demand_in_flight
                            .store(false, Ordering::Release);
                    }
                    // After every doorbell-driven path runs, also
                    // service any user-watchpoint hits on DR1..=DR3.
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
                            // CAPTURE doorbell handler is running).
                            // Re-arm the slot's hit flag so the next
                            // epoll iteration handles it.
                            freeze_coord_watchpoint.user[slot_idx]
                                .hit
                                .store(true, Ordering::Release);
                            break;
                        }
                        tracing::info!(
                            slot_idx,
                            %tag,
                            "freeze-coord: user watchpoint fire; capturing"
                        );
                        let on_demand = freeze_and_capture(false);
                        if let Some(ref blk) = freeze_coord_virtio_blk {
                            blk.lock().resume();
                        }
                        freeze_coord_freeze.store(false, Ordering::Release);
                        let _ = freeze_coord_thaw_evt.write(1);
                        if let Some((report, capture_start)) = on_demand {
                            let map_count = report.maps.len();
                            freeze_coord_snapshot_bridge
                                .store(&tag, report.clone());
                            if let Some(ref base_path) = freeze_coord_dump_path
                            {
                                let tagged =
                                    snapshot_tagged_path(base_path, &tag);
                                if let Some(parent) = tagged.parent() {
                                    let _ = std::fs::create_dir_all(parent);
                                }
                                if let Ok(json) =
                                    serde_json::to_string_pretty(&report)
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
                        }
                        freeze_coord_on_demand_in_flight
                            .store(false, Ordering::Release);
                    }
                    // Once the late snapshot has been emitted, the
                    // coordinator's only remaining job is to keep
                    // the freeze=false invariant clear, drain the
                    // doorbell, and wait for teardown. Skip the
                    // error-trigger paths below; the next
                    // `epoll.wait` at the top of the loop blocks
                    // until kill / bsp_done / doorbell / scanner
                    // tick — no separate sleep cadence needed.
                    // Goes AFTER the doorbell handler so on-demand
                    // captures still service post-Done.
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
                            ctx.cr3_pa,
                            ctx.page_offset,
                            ctx.l5,
                            ctx.watchdog_timestamp_pa,
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
                            // `false` to skip the gate.
                            if let Some((report, _capture_start)) =
                                freeze_and_capture(false)
                            {
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
                            // Wake every parked vCPU. See the
                            // doorbell-handler thaw_evt write for
                            // the ordering rationale.
                            let _ = freeze_coord_thaw_evt.write(1);
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
                        // unconditionally — `bss_triggered` already
                        // proves kind >= 1024.
                        let watchpoint_only_trigger =
                            watchpoint_hit && !bss_triggered;
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
                                    ctx.cr3_pa,
                                    ctx.page_offset,
                                    ctx.l5,
                                    ctx.watchdog_timestamp_pa,
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
                        // longer than the dump strictly needs.
                        // Resume worker first — see early-snapshot
                        // path doc for the freeze-vs-paused
                        // ordering rationale.
                        if let Some(ref blk) = freeze_coord_virtio_blk {
                            blk.lock().resume();
                        }
                        freeze_coord_freeze.store(false, Ordering::Release);
                        // Wake every parked vCPU. See the
                        // doorbell-handler thaw_evt write for the
                        // ordering rationale.
                        let _ = freeze_coord_thaw_evt.write(1);
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
                                    serde_json::to_string_pretty(&dual)
                                } else {
                                    serde_json::to_string_pretty(&late)
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
                            }
                            None if watchpoint_only_trigger => {
                                // Gate-suppressed dump (or rendezvous
                                // timeout on a watchpoint-only
                                // trigger). Reset `watchpoint.hit`
                                // so the next genuine fire re-
                                // triggers cleanly. Without this
                                // reset, the stale `hit=true` would
                                // re-fire the late-trigger every
                                // iteration, re-running the
                                // rendezvous and re-suppressing
                                // forever.
                                freeze_coord_watchpoint
                                    .hit
                                    .store(false, Ordering::Release);
                                tracing::debug!(
                                    "freeze-coord: watchpoint-only trigger \
                                     produced no dump (gate suppressed or \
                                     rendezvous timed out); resetting hit \
                                     latch and continuing"
                                );
                            }
                            None => {
                                // bss-triggered with rendezvous
                                // timeout. The bss latch is sticky
                                // on the kernel side; retrying would
                                // just hit the same timeout. Mark
                                // Done and let the run end normally.
                                freeze_state = FreezeState::Done;
                            }
                        }
                        // Post-thaw COM2 grace window. After a late-
                        // trigger freeze captures the dump and thaws
                        // the vCPUs, the guest's scheduler binary
                        // typically still has scheduler-log bytes in
                        // userspace buffers — the kernel's struct_ops
                        // detach is what makes the binary exit, and
                        // its libbpf wait only unblocks after the
                        // scheduler-side error exit propagates back.
                        // Once it exits, the guest's
                        // start_sched_exit_monitor calls
                        // dump_sched_output (writing SCHED_OUTPUT_START
                        // / log body / SCHED_OUTPUT_END to COM2)
                        // immediately before write_msg(MSG_TYPE_SCHED_EXIT);
                        // the host monitor reader sets `kill` the moment
                        // it sees SCHED_EXIT in SHM. Without a grace
                        // window the host can race the guest flush
                        // and tear down before the closing delimiter
                        // hits the COM2 capture buffer, leaving the
                        // scheduler log truncated. Wait up to
                        // SCHED_OUTPUT_END_GRACE for the closing
                        // delimiter to land; bail out the moment it
                        // arrives (or kill / bsp_done flips). The
                        // verifier's parse_sched_output_partial is the
                        // safety net for runs where the delimiter
                        // never arrives within the grace window — it
                        // still extracts the partial scheduler log
                        // for the auto-repro probe pipeline.
                        //
                        // Skipped on the watchpoint-only-trigger path
                        // (FreezeState still Idle) because that path
                        // produced no dump and the freeze cycle can
                        // legitimately re-fire on the next genuine
                        // error-class write — adding a 3 s sleep there
                        // would delay the next dump trigger by exactly
                        // the grace duration for no benefit.
                        // The grace window is meaningful only when
                        // the guest's start_sched_exit_monitor will
                        // actually emit SCHED_OUTPUT_END to COM2 —
                        // i.e. when probes are NOT active. With
                        // probes active, that monitor's `else if`
                        // branch suppresses dump_sched_output (see
                        // vmm/rust_init.rs::start_sched_exit_monitor),
                        // so the marker never lands. Waiting up to
                        // 3 s for a write that will never come is
                        // 3 s of teardown latency every auto-repro
                        // run for no value. `dual_snapshot` is the
                        // reliable proxy for "probes active" in
                        // ktstr today: the auto-repro path sets
                        // both flags together, primary VMs leave
                        // both off. Tying the gate to dual_snapshot
                        // (rather than a separate suppress flag)
                        // keeps the surface lean while making the
                        // skip exact for the known caller.
                        if freeze_state == FreezeState::Done && !freeze_coord_dual_snapshot {
                            const SCHED_OUTPUT_END_GRACE: Duration =
                                Duration::from_secs(3);
                            let grace_deadline =
                                Instant::now() + SCHED_OUTPUT_END_GRACE;
                            let needle =
                                crate::verifier::SCHED_OUTPUT_END.as_bytes();
                            // Initial check: the guest may have
                            // emitted the marker before we entered
                            // the grace path (cases where the
                            // dump emission was fast enough that
                            // the post-thaw flush already landed).
                            // Skip the wait entirely in that case.
                            let mut found = freeze_coord_com2
                                .lock()
                                .output_contains(needle);
                            // Build a pollfd list once; the data
                            // eventfd may be `None` if installation
                            // failed at coord setup. Without it the
                            // grace window degrades to "wait for
                            // kill/bsp_done OR the deadline" — the
                            // re-check after the wait still covers
                            // a marker that landed during the wait.
                            use std::os::fd::AsRawFd;
                            let kill_fd = freeze_coord_kill_evt.as_raw_fd();
                            let bsp_done_fd = freeze_coord_bsp_done_evt.as_raw_fd();
                            let data_fd = freeze_coord_com2_data_evt
                                .as_ref()
                                .map(|e| e.as_raw_fd());
                            while !found {
                                if freeze_coord_kill.load(Ordering::Acquire)
                                    || freeze_coord_bsp_done
                                        .load(Ordering::Acquire)
                                {
                                    break;
                                }
                                let now = Instant::now();
                                if now >= grace_deadline {
                                    tracing::debug!(
                                        grace_ms = SCHED_OUTPUT_END_GRACE
                                            .as_millis() as u64,
                                        "freeze-coord: SCHED_OUTPUT_END \
                                         grace window expired without \
                                         marker; partial-handler will \
                                         recover the scheduler log tail"
                                    );
                                    break;
                                }
                                let remaining_ms =
                                    (grace_deadline - now).as_millis() as i32;
                                // Build pollfd list for this wait.
                                let mut pfds = [
                                    libc::pollfd {
                                        fd: kill_fd,
                                        events: libc::POLLIN,
                                        revents: 0,
                                    },
                                    libc::pollfd {
                                        fd: bsp_done_fd,
                                        events: libc::POLLIN,
                                        revents: 0,
                                    },
                                    libc::pollfd {
                                        fd: data_fd.unwrap_or(-1),
                                        events: libc::POLLIN,
                                        revents: 0,
                                    },
                                ];
                                let nfds = if data_fd.is_some() { 3 } else { 2 };
                                unsafe {
                                    libc::poll(
                                        pfds.as_mut_ptr(),
                                        nfds as libc::nfds_t,
                                        remaining_ms,
                                    );
                                }
                                // Drain the COM2 data eventfd
                                // counter so the next iteration
                                // doesn't re-fire on the same edge.
                                // The buffer-grew predicate (re-
                                // check below) is the source of
                                // truth — the eventfd's role is
                                // purely the wake signal.
                                if let Some(ref evt) = freeze_coord_com2_data_evt {
                                    let _ = evt.read();
                                }
                                found = freeze_coord_com2
                                    .lock()
                                    .output_contains(needle);
                                if found {
                                    tracing::debug!(
                                        "freeze-coord: SCHED_OUTPUT_END \
                                         observed within grace window"
                                    );
                                    break;
                                }
                            }
                        } else if freeze_state == FreezeState::Done {
                            tracing::debug!(
                                "freeze-coord: SCHED_OUTPUT_END grace window \
                                 skipped (dual_snapshot/probes active — guest \
                                 suppresses the marker)"
                            );
                        }
                        continue;
                    }
                    // End of body. Loop back to the `epoll.wait`
                    // at the top, which blocks until any registered
                    // fd fires (kill, bsp_done, doorbell, watchpoint
                    // hit, scanner tick) or POLL_TIMEOUT_MS elapses.
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
                        // Wake the guest's shm-poll thread by pushing
                        // a byte into virtio-console RX. The byte
                        // value is informational only — any byte
                        // forces the guest to re-read the SHM signal
                        // slot and the dump/stall control bytes.
                        // Pushed AFTER the SHM write so the guest
                        // observes the new signal value when it
                        // re-checks.
                        wd_virtio_con
                            .lock()
                            .queue_input(&[virtio_console::SIGNAL_VC_SHUTDOWN]);
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
        // Sample cleanup start at the earliest moment after BSP exit so
        // every host-side teardown step lands inside the window, in
        // execution order: watchdog join (immediately below), AP joins,
        // monitor join, BPF writer join, SHM drain, exit-code and
        // crash-message extraction, and verifier-stat read (the rest
        // run inside `collect_results`). `collect_results` reads
        // `Instant::now()` at the end and the difference becomes
        // `VmResult::cleanup_duration`.
        let cleanup_start = Instant::now();
        // `code` here is the run-loop sentinel (0 only on a BSP-
        // observed `ExitAction::Shutdown`, -1 otherwise — see
        // [`BspExitReason`] and the preceding `BSP: loop exit
        // reason=...` line). The caller-visible exit code is
        // derived from SHM `MSG_TYPE_EXIT` or the COM2 `KTSTR_EXIT:`
        // sentinel inside [`KtstrVm::collect_results`], not from
        // this value.
        eprintln!(
            "BSP: exited run loop, code={exit_code} timed_out={timed_out} \
             (run-loop sentinel — final exit code comes from SHM/COM2 in collect_results)"
        );

        // Join the watchdog before dropping `bsp`. The watchdog holds an
        // ImmediateExitHandle pointing into bsp's kvm_run mmap. If bsp is
        // dropped first, the watchdog may write to unmapped memory.
        let _ = watchdog.join();

        // Make sure freeze is cleared before vCPU teardown so the
        // freeze coordinator sees `kill || bsp_done` and exits its
        // loop, and APs don't park-loop after we kick them. The
        // coordinator joins below (in `collect_results`); the
        // coordinator's epoll loop wakes on `kill_evt` /
        // `bsp_done_evt` writes, not on `thread::unpark`.

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
            kill_evt,
            freeze,
            vm,
            cleanup_start,
            virtio_blk_counters,
            virtio_net_counters,
            // Original doorbell EventFd handle. The coordinator owns
            // a clone in `freeze_coord_doorbell`; both fds share the
            // same kernel counter via `dup(2)`, so a write to either
            // wakes the coordinator's read.
            doorbell_evt: Some(doorbell_evt),
            // Snapshot bridge owning every report stored by the
            // freeze coordinator's doorbell handler over the run's
            // lifetime. Forwarded to `VmResult::snapshot_bridge`
            // by `collect_results`.
            snapshot_bridge,
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
        probes_ready_evt: EventFd,
        virtio_con: Option<Arc<PiMutex<virtio_console::VirtioConsole>>>,
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
        let kill_evt_clone = kill_evt.clone();
        let dump_trigger =
            self.monitor_thresholds
                .filter(|_| self.shm_size > 0)
                .map(|thresholds| {
                    let shm_base_pa = mem_size - self.shm_size;
                    monitor::reader::DumpTrigger {
                        shm_base_pa,
                        thresholds,
                        virtio_con: virtio_con.clone(),
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
                // No boot-delay sleep needed:
                //   - `resolve_page_offset` checks validity and falls
                //     back to DEFAULT_PAGE_OFFSET when the symbol
                //     read returns zero or unmapped.
                //   - `setup_per_cpu_areas` runs in `start_kernel`
                //     before SMP is brought up, which is before any
                //     guest userspace runs, which is before the
                //     monitor thread can spawn (the monitor only
                //     spawns after run_vm enters its loop, well past
                //     start_kernel).
                //   - The downstream lazy-retry pattern (e.g. the
                //     freeze coordinator's `owned_accessor.is_none()`
                //     gate) already covers any post-boot resolve
                //     race.
                // The 500 ms delay bought nothing.

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
                //
                // Sleeping is replaced by `poll(POLLIN)` against the
                // shared `probes_ready_evt`: ANY waiter that detects
                // its own readiness condition writes 1 to the eventfd
                // and the level stays high (we never `read` here), so
                // every other waiter wakes immediately and re-checks.
                // The 100 ms timeout preserves the prior cadence as
                // an upper bound for kill / deadline observation when
                // no other detector has fired yet. On detection here
                // we write 1 ourselves, fanning out to the
                // bpf-map-write thread's pollers.
                if let Some(base) = shm_base_pa {
                    let slot_pa = base + shm_ring::SIGNAL_SLOT_BASE as u64 + 1;
                    let deadline = run_start + Duration::from_secs(30);
                    while std::time::Instant::now() < deadline {
                        if kill_clone.load(std::sync::atomic::Ordering::Relaxed) {
                            break;
                        }
                        if mem.read_u8(slot_pa, 0) != 0 {
                            let _ = probes_ready_evt.write(1);
                            break;
                        }
                        poll_eventfd_until_ready_or_timeout(&probes_ready_evt, 100);
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
    /// 3. Write the crash value and signal guest via SHM slot 0
    ///
    /// `probes_ready_evt` is the broadcast EventFd shared with the
    /// monitor thread (see [`run_vm`]); each phase below `poll`s it
    /// instead of bare-sleeping, and writes 1 to it on detection so
    /// the monitor (and any future waiter) wakes immediately to
    /// re-check its own readiness condition.
    pub(super) fn start_bpf_map_write(
        &self,
        vm: &kvm::KtstrKvm,
        kill: &Arc<AtomicBool>,
        probes_ready_evt: EventFd,
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
                let phase1_deadline =
                    std::time::Instant::now() + std::time::Duration::from_secs(30);
                let owned = loop {
                    match monitor::bpf_map::GuestMemMapAccessorOwned::new(&mem, &vmlinux) {
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
                //
                // Same `poll(POLLIN)` pattern as phases 1 and 2: wake
                // on a sibling detection, fall back to the 100 ms
                // cadence for kill / deadline coverage; write 1 on
                // detection to fan the wake out to the monitor.
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
                            let _ = probes_ready_evt.write(1);
                            break;
                        }
                        poll_eventfd_until_ready_or_timeout(&probes_ready_evt, 100);
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
    ///     `exit_code` with the SHM `MSG_TYPE_EXIT` payload (or the
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
        run_start: Instant,
        timeout: Duration,
        parked_evt: Option<&Arc<EventFd>>,
        thaw_evt: Option<&Arc<EventFd>>,
        kill_evt: Option<&Arc<EventFd>>,
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
        // [`super::vcpu::self_arm_watchpoint`]. Index 0 = DR0
        // (err_exit watchpoint); 1..=3 = DR1/DR2/DR3 (user
        // `Op::WatchSnapshot` arms). All `0` until the coordinator
        // publishes resolved KVAs. `arm_failures` counts consecutive
        // non-EINTR ioctl failures; transient EINTR (signal race
        // with the SIGRTMIN kick path) does NOT increment so a
        // kicked-mid-arm vCPU keeps retrying instead of giving up
        // after the first racey iteration.
        let mut armed_slots: [u64; 4] = [0; 4];
        let mut arm_failures: u8 = 0;

        loop {
            if run_start.elapsed() > timeout {
                eprintln!(
                    "BSP: loop exit reason={reason:?} (timed_out)",
                    reason = BspExitReason::Timeout
                );
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
            // and compare) when no new arm is pending.
            self_arm_watchpoint(bsp, watchpoint, &mut armed_slots, &mut arm_failures);

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
                    // Hardware watchpoint dispatch is x86-only. See
                    // the AP-side handler in
                    // `exit_dispatch::vcpu_run_loop_unified` for the
                    // arch-gating rationale and the aarch64 stub in
                    // `super::vcpu::self_arm_watchpoint`.
                    #[cfg(target_arch = "x86_64")]
                    if let VcpuExit::Debug(debug_arch) = &exit {
                        let dr6 = debug_arch.dr6;
                        let dr0_hit = (dr6 & (1 << 0)) != 0;
                        let dr1_hit = (dr6 & (1 << 1)) != 0;
                        let dr2_hit = (dr6 & (1 << 2)) != 0;
                        let dr3_hit = (dr6 & (1 << 3)) != 0;
                        if dr0_hit {
                            let host_ptr = watchpoint.kind_host_ptr.load(Ordering::Acquire);
                            if !host_ptr.is_null() {
                                // SAFETY: see the AP-side handler in
                                // `exit_dispatch::vcpu_run_loop_unified`.
                                let kind = unsafe { std::ptr::read_volatile(host_ptr) };
                                if kind >= super::vcpu::SCX_EXIT_ERROR_THRESHOLD {
                                    watchpoint.latch_hit();
                                } else {
                                    tracing::debug!(
                                        kind,
                                        threshold = super::vcpu::SCX_EXIT_ERROR_THRESHOLD,
                                        "BSP watchpoint fired on non-error \
                                         exit_kind transition (e.g. \
                                         SCX_EXIT_DONE on clean shutdown); \
                                         skipping freeze trigger"
                                    );
                                }
                            } else {
                                // Conservative fallback when
                                // host_ptr is null — see the AP
                                // handler for rationale.
                                watchpoint.latch_hit();
                            }
                        }
                        if dr1_hit {
                            watchpoint.latch_user_hit(0);
                        }
                        if dr2_hit {
                            watchpoint.latch_user_hit(1);
                        }
                        if dr3_hit {
                            watchpoint.latch_user_hit(2);
                        }
                        if kill.load(Ordering::Acquire) {
                            break;
                        }
                        continue;
                    }
                    #[cfg(target_arch = "aarch64")]
                    if let VcpuExit::Debug(_debug_arch) = &exit {
                        // aarch64 watchpoint arming is not implemented;
                        // a KVM_EXIT_DEBUG here would mean a stale
                        // KVM_GUESTDBG arm we did not request. Log and
                        // continue rather than silently dropping the
                        // exit.
                        tracing::warn!(
                            "BSP: unexpected KVM_EXIT_DEBUG on aarch64 \
                             (watchpoint arming not implemented); ignoring"
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
                            // collect_results drive the kill
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
        (exit_code, false)
    }

    /// Shutdown threads and collect output.
    pub(super) fn collect_results(&self, start: Instant, run: VmRunState) -> Result<VmResult> {
        let mut exit_code = run.exit_code;
        let timed_out = run.timed_out;
        run.kill.store(true, Ordering::Release);
        // Wake the freeze coordinator (and the monitor sampler) if
        // either is still blocked in epoll_wait. The freeze
        // coordinator is still alive at this point — it is joined
        // a few lines below in this same `collect_results`, AFTER
        // the kill propagation here ensures its outer loop exits
        // promptly. The monitor sampler is also alive and uses the
        // same kill_evt.
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
            snapshot_bridge: run.snapshot_bridge,
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
