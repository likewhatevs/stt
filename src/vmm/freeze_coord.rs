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
use super::pi_mutex::PiMutex;
use super::result::{VmResult, VmRunState};
use super::vcpu::{
    ApFreezeHandles, BpfMapWriteParams, ImmediateExitHandle, VcpuThread, WatchpointArm,
    duration_to_jiffies, load_probe_bss_offset, open_vcpu_perf_capture, pin_current_thread,
    register_vcpu_signal_handler, self_arm_watchpoint, set_rt_priority, set_thread_cpumask,
    vcpu_signal,
};
use super::vmlinux::find_vmlinux;
use super::host_comms::BulkDrainResult;
use super::{KtstrVm, console, host_comms, vcpu_panic, virtio_blk, virtio_console, virtio_net, wire};

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

/// Decoded contents of a guest-side `MSG_TYPE_SNAPSHOT_REQUEST` TLV
/// frame consumed from the virtio-console port-1 TX stream by the
/// coordinator's TOKEN_TX handler. The request id is echoed in the
/// matching `MSG_TYPE_SNAPSHOT_REPLY` payload so the guest's blocking
/// reader can pair the reply against its outstanding request; `kind`
/// selects the CAPTURE / WATCH dispatch path and `tag` carries the
/// snapshot name (CAPTURE) or symbol path (WATCH).
struct SnapshotRequest {
    request_id: u32,
    kind: u32,
    tag: String,
}

/// Frame a `MSG_TYPE_SNAPSHOT_REPLY` TLV — header (16 bytes) plus
/// [`crate::vmm::wire::SnapshotReplyPayload`] (72 bytes) — into a
/// single buffer the coordinator pushes through
/// [`crate::vmm::virtio_console::VirtioConsole::queue_input_port1`].
/// The reply is delivered atomically as one TLV: the buffer is
/// concatenated before the call so a partial push that splits header
/// and payload across multiple `queue_input_port1` invocations cannot
/// arise. CRC32 is computed over the payload bytes only — matches
/// the wire-format contract `parse_tlv_stream` enforces on the
/// guest's `read_bulk_port_frame`.
fn frame_snapshot_reply(request_id: u32, status: u32, reason: &str) -> Vec<u8> {
    use crate::vmm::wire::{
        FRAME_HEADER_SIZE, MSG_TYPE_SNAPSHOT_REPLY, SNAPSHOT_REASON_MAX, ShmMessage,
        SnapshotReplyPayload,
    };
    use zerocopy::IntoBytes;
    // Reason buffer: NUL-terminated UTF-8, truncated to the buffer
    // size. Trailing zeros remain from the array initializer so a
    // shorter reason terminates cleanly on the guest side.
    let reason_bytes = reason.as_bytes();
    let reason_len = reason_bytes.len().min(SNAPSHOT_REASON_MAX);
    let mut reason_buf = [0u8; SNAPSHOT_REASON_MAX];
    reason_buf[..reason_len].copy_from_slice(&reason_bytes[..reason_len]);
    let payload = SnapshotReplyPayload {
        request_id,
        status,
        reason: reason_buf,
    };
    let payload_bytes = payload.as_bytes();
    let header = ShmMessage {
        msg_type: MSG_TYPE_SNAPSHOT_REPLY,
        length: payload_bytes.len() as u32,
        crc32: crc32fast::hash(payload_bytes),
        _pad: 0,
    };
    let mut buf = Vec::with_capacity(FRAME_HEADER_SIZE + payload_bytes.len());
    buf.extend_from_slice(header.as_bytes());
    buf.extend_from_slice(payload_bytes);
    buf
}

/// Decode a guest-side `MSG_TYPE_SNAPSHOT_REQUEST` TLV payload into
/// the typed [`SnapshotRequest`]. `payload` must be exactly
/// `size_of::<SnapshotRequestPayload>()` bytes — the bulk parser
/// already enforces the per-frame cap, but a malformed guest may
/// publish a frame whose announced length doesn't match the typed
/// payload size. Returns `None` for any size or layout mismatch so
/// the TOKEN_TX handler can drop the frame without touching dispatch.
fn decode_snapshot_request(payload: &[u8]) -> Option<SnapshotRequest> {
    use crate::vmm::wire::{SNAPSHOT_KIND_NONE, SNAPSHOT_TAG_MAX, SnapshotRequestPayload};
    use zerocopy::FromBytes;
    if payload.len() != std::mem::size_of::<SnapshotRequestPayload>() {
        return None;
    }
    let req = SnapshotRequestPayload::read_from_bytes(payload).ok()?;
    if req.request_id == 0 || req.kind == SNAPSHOT_KIND_NONE {
        return None;
    }
    let len = req
        .tag
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(SNAPSHOT_TAG_MAX);
    let tag = String::from_utf8_lossy(&req.tag[..len]).to_string();
    Some(SnapshotRequest {
        request_id: req.request_id,
        kind: req.kind,
        tag,
    })
}

/// Cached `name -> KVA` map built once at coordinator init from the
/// vmlinux ELF symbol table. Lets [`arm_user_watchpoint`] look up
/// `Op::WatchSnapshot` symbols without re-reading and re-parsing
/// the 50MB+ vmlinux per request.
struct VmlinuxSymbolCache {
    symbols: std::collections::HashMap<String, u64>,
}

impl VmlinuxSymbolCache {
    /// Read and parse `path` once, extracting every symbol with a
    /// non-zero `st_value` into the cache. Errors propagate as
    /// caller-side diagnostics so arming surfaces the same reason
    /// strings the per-call parse used.
    fn from_path(path: &std::path::Path) -> std::result::Result<Self, String> {
        let data = std::fs::read(path)
            .map_err(|e| format!("read vmlinux at {}: {e}", path.display()))?;
        let elf = goblin::elf::Elf::parse(&data)
            .map_err(|e| format!("parse vmlinux ELF: {e}"))?;
        let mut symbols = std::collections::HashMap::new();
        for s in elf.syms.iter() {
            if s.st_value == 0 {
                continue;
            }
            if let Some(name) = elf.strtab.get_at(s.st_name) {
                symbols.insert(name.to_string(), s.st_value);
            }
        }
        Ok(Self { symbols })
    }

    fn lookup(&self, symbol: &str) -> Option<u64> {
        self.symbols.get(symbol).copied()
    }
}

/// Resolve a kernel symbol by name from the cached vmlinux symbol
/// table and arm a user watchpoint slot (slots 1..=3) on it.
/// Returns the slot index (0..=2 mapping to slots 1..=3) on
/// success, or a host-side diagnostic on failure.
///
/// The vCPU thread's `self_arm_watchpoint` notices the change on
/// the next loop iteration (Acquire load on the slot's
/// `request_kva`) and reprograms `KVM_SET_GUEST_DEBUG` with the
/// new DR layout.
/// Arm a user watchpoint slot on `symbol`'s resolved KVA.
///
/// On success, the slot's `request_kva` is published with `Release`,
/// `WatchpointArm::mark_armed()` flips the fast-path gate, and every
/// vCPU thread (BSP + APs) is kicked out of `KVM_RUN` so its next
/// loop iteration runs `self_arm_watchpoint` and reprograms
/// `KVM_SET_GUEST_DEBUG`.
///
/// Without the gate flip, the per-vCPU `self_arm_watchpoint` short-
/// circuits at the `any_armed.load(Relaxed) == 0` check and never
/// observes the published `request_kva`. Without the kick, vCPU
/// threads sitting in `KVM_RUN` only re-check the slot on their next
/// natural exit (HLT, IO, IRQ) — for compute-bound guests that can
/// be many seconds, missing the very write the user requested to
/// observe. Mirrors the freeze-rendezvous kick pattern (pass 1: set
/// every immediate_exit byte; pass 2: deliver SIGRTMIN to every
/// vCPU TID), differing only in that arming does NOT request a
/// freeze — vCPUs immediately re-enter `KVM_RUN` after the arm.
///
/// `bsp_alive_load` is the same Acquire-bool the freeze_and_capture
/// closure consults: a `false` reading means the BSP `VcpuFd` is
/// gone and writing through `bsp_ie_handle` would touch unmapped
/// memory. The check happens once at the start of the kick pass and
/// gates BOTH the BSP `ie.set(1)` and the BSP `pthread_kill` so the
/// pair stays symmetric.
#[allow(clippy::too_many_arguments)]
fn arm_user_watchpoint(
    watchpoint: &Arc<super::vcpu::WatchpointArm>,
    symbol_cache: &VmlinuxSymbolCache,
    symbol: &str,
    ap_pthreads: &[libc::pthread_t],
    ap_ies: &[Option<ImmediateExitHandle>],
    bsp_tid: libc::pthread_t,
    bsp_ie: Option<&ImmediateExitHandle>,
    bsp_alive_load: bool,
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
            "no free user watchpoint slot — slots 1..=3 all occupied by prior \
             Op::WatchSnapshot registrations (cap = {})",
            watchpoint.user.len()
        ));
    };
    // Resolve the symbol via the cached vmlinux symbol table.
    // The cache is built once at coord init; per-call lookups are
    // O(1) HashMap reads instead of 50MB+ file reads + ELF parses.
    let kva = symbol_cache
        .lookup(symbol)
        .ok_or_else(|| format!("symbol '{symbol}' not found in vmlinux symtab"))?;
    if kva & 0x3 != 0 {
        return Err(format!(
            "symbol '{symbol}' KVA {kva:#x} is not 4-byte aligned. \
             x86_64 DR_LEN_4 watchpoints (Intel SDM Vol. 3B Ch. 17) \
             and aarch64 DBGWVR (ARM ARM D7.3.10, requires VA[1:0] = \
             00) both require 4-byte aligned targets for the 4-byte \
             write-watch the failure-dump trigger uses"
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
    // Flip the fast-path gate so per-vCPU `self_arm_watchpoint` calls
    // stop short-circuiting on `any_armed == 0`. Idempotent — repeated
    // calls keep the gate at 1. Must happen AFTER the Release on
    // `request_kva`: the `mark_armed` store is `Relaxed`, so the
    // synchronizes-with edge that publishes the new KVA value comes
    // from `request_kva`'s Release / per-vCPU Acquire pair, not the
    // gate. Once a vCPU sees `any_armed == 1` it falls through to the
    // Acquire load on `request_kva` which carries the edge.
    watchpoint.mark_armed();
    // Two-pass kick (pass 1: every immediate_exit byte; pass 2:
    // SIGRTMIN to every vCPU TID), separated by a Release fence so
    // the immediate_exit writes are observable before any vCPU's
    // signal handler returns and re-enters KVM_RUN. Mirrors the
    // freeze rendezvous kick path so a future refactor of either
    // changes them in lock-step.
    //
    // The eventfd write that USED to live here (commented as "the
    // cleanest available wake fd") was load-bearing-shaped but
    // semantically a no-op: vCPU threads do not block on `hit_evt`,
    // so writing to it does NOT wake them out of `KVM_RUN`. The
    // actual wake mechanism is immediate_exit + SIGRTMIN — the same
    // pair the freeze rendezvous uses for parking.
    for ie in ap_ies.iter().flatten() {
        ie.set(1);
    }
    if bsp_alive_load
        && let Some(ie) = bsp_ie
    {
        ie.set(1);
    }
    std::sync::atomic::fence(Ordering::Release);
    for &tid in ap_pthreads {
        // SAFETY: pthread_kill against a tid whose thread has
        // already exited returns ESRCH. The AP threads are joined
        // by `collect_results` AFTER this coordinator joins (see
        // `run_vm`); during arm_user_watchpoint the coord is alive
        // and every AP `pthread_t` it captured at spawn is still
        // valid. ESRCH is harmless here — a kicked-but-already-gone
        // AP simply means the kick is unnecessary.
        unsafe {
            libc::pthread_kill(tid, vcpu_signal());
        }
    }
    if bsp_alive_load {
        // SAFETY: bsp_alive_load is Acquire-loaded above; while
        // true the BSP `VcpuFd` and its kvm_run mmap are live. The
        // BSP TID was captured at coord spawn from the BSP thread's
        // `pthread_self()` and remains valid until the BSP thread
        // joins, which `run_vm` only allows AFTER this coordinator
        // joins.
        unsafe {
            libc::pthread_kill(bsp_tid, vcpu_signal());
        }
    }
    Ok(idx)
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

        let monitor_handle = self.start_monitor(
            &vm,
            &kill,
            &kill_evt,
            run_start,
            vcpu_pthreads,
            perf_capture.clone(),
            probes_ready_evt_for_monitor,
            Some(virtio_con.clone()),
            tcr_el1_cache.clone(),
        )?;

        // BPF map write thread: sleeps, discovers a BPF map, writes a value.
        let bpf_write_handle = self.start_bpf_map_write(
            &vm,
            &kill,
            probes_ready_evt_for_bpf,
            tcr_el1_cache.clone(),
            virtio_con.clone(),
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
        // Wake fds the watchdog blocks on via epoll, paired with the
        // `kill_for_watchdog` and `bsp_done_for_wd` AtomicBools above.
        // The watchdog wakes within microseconds of either flip
        // instead of polling on a 100 ms thread::sleep cadence.
        let kill_evt_for_watchdog = kill_evt.clone();
        let bsp_done_evt_for_wd = bsp_done_evt.clone();
        let rt_watchdog = self.performance_mode;
        let wd_service_cpu = self.pinning_plan.as_ref().and_then(|p| p.service_cpu);
        // Clone the virtio-console Arc into the watchdog so the
        // soft-deadline path can push `SIGNAL_VC_SHUTDOWN` to
        // `/dev/hvc0` for graceful shutdown. The guest's
        // `hvc0_poll_loop` blocks on the device read and recognises
        // the byte directly — no SHM signal slot involved.
        let wd_virtio_con = virtio_con.clone();

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
        // chain on either port (port 0 console or port 1 bulk).
        // The tx_evt is a per-device counter — both ports share it,
        // but a spurious wake on port-0 traffic is harmless: the
        // coord just calls `drain_bulk()` and finds an empty buffer.
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
        // Cached `name -> KVA` map for `Op::WatchSnapshot` arming.
        // Build once here at run_vm scope so every TLV-driven
        // WATCH request is an O(1) HashMap lookup instead of a
        // 50MB+ vmlinux read + ELF parse. None when vmlinux can't
        // be found or the parse failed — `arm_user_watchpoint`
        // will report a clean diagnostic on lookup. Hoisted out of
        // the closure so the spawn-time parse cost is paid once
        // even when the run ends without any WATCH requests.
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
        // NUMA node count from the configured topology. Forwarded
        // into the scx walker (per-node global DSQ pass) and the
        // per-node NUMA event walker. Defaults to 1 on UMA topologies.
        let freeze_coord_num_nodes = self.topology.num_numa_nodes();
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
        // aarch64 TCR_EL1 cache populated by the BSP loop. Threaded
        // into `GuestKernel::new` constructions inside the
        // freeze-coord scan_tick closure (BPF map accessor and
        // prog accessor) so vmalloc-backed kernel reads succeed
        // post-MMU-bringup. None on x86_64.
        let freeze_coord_tcr_el1 = tcr_el1_cache.clone();
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
                //      `GuestKernel::text_kva_to_pa` (it lives in the
                //      kernel text mapping, not vmalloc);
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
                /// publishes a TX descriptor chain on EITHER port.
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
                const TOKEN_TX: u64 = 5;
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
                'coord: while !freeze_coord_kill.load(Ordering::Acquire) {
                    if freeze_coord_bsp_done.load(Ordering::Acquire) {
                        break 'coord;
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
                                    break 'coord;
                                }
                                TOKEN_BSP_DONE => {
                                    let _ = freeze_coord_bsp_done_evt.read();
                                    break 'coord;
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
                                    for msg in &drained.messages {
                                        // Promote a guest-side
                                        // SCHED_EXIT into the
                                        // run-wide kill flag so
                                        // the BSP loop and the
                                        // watchdog exit promptly
                                        // instead of running until
                                        // the watchdog deadline.
                                        // CRC failures DO NOT
                                        // promote — a torn frame
                                        // would otherwise let a
                                        // hostile guest force a
                                        // false early exit. Only
                                        // crc_ok messages count.
                                        if msg.msg_type
                                            == crate::vmm::wire::MSG_TYPE_SCHED_EXIT
                                            && msg.crc_ok
                                        {
                                            freeze_coord_kill.store(true, Ordering::Release);
                                            let _ = freeze_coord_kill_evt.write(1);
                                        }
                                        // Decode a guest-side
                                        // `MSG_TYPE_SNAPSHOT_REQUEST`
                                        // and stash it for dispatch
                                        // later in this iteration's
                                        // body — `freeze_and_capture`
                                        // / `thaw_and_barrier` /
                                        // `arm_user_watchpoint` are
                                        // not in scope here. Only
                                        // crc_ok frames are decoded:
                                        // a torn snapshot request
                                        // would otherwise let a
                                        // hostile guest force a
                                        // capture, mirroring the
                                        // SCHED_EXIT promotion gate
                                        // above. Malformed payloads
                                        // (size mismatch, KIND_NONE,
                                        // request_id == 0) decode to
                                        // `None` and are dropped.
                                        if msg.msg_type
                                            == crate::vmm::wire::MSG_TYPE_SNAPSHOT_REQUEST
                                            && msg.crc_ok
                                            && let Some(req) =
                                                decode_snapshot_request(&msg.payload[..])
                                        {
                                            snapshot_requests_pending.push(req);
                                        }
                                    }
                                    // Stash the parsed messages on
                                    // the shared buffer so
                                    // `collect_results` can merge
                                    // them into the final
                                    // `BulkDrainResult`. Snapshot
                                    // request frames are filtered out
                                    // — they are coordinator-internal
                                    // control traffic, not test
                                    // verdict data, and the matching
                                    // reply is delivered over port-1
                                    // RX rather than recorded in the
                                    // verdict drain. Without this
                                    // stash, every TLV frame the
                                    // guest published mid-run (EXIT,
                                    // TEST, PAYLOAD_METRICS,
                                    // RAW_PAYLOAD_OUTPUT, PROFRAW)
                                    // is silently dropped — only
                                    // late-arriving bytes that
                                    // landed in `port1_tx_buf`
                                    // after the coord stopped
                                    // polling reach the verdict.
                                    // `BulkMessage` and `ShmEntry`
                                    // share the same field shape
                                    // (msg_type / payload / crc_ok)
                                    // so the conversion is a
                                    // direct field copy.
                                    if !drained.messages.is_empty() {
                                        let mut buf = freeze_coord_bulk_messages_for_closure
                                            .lock()
                                            .unwrap_or_else(|e| e.into_inner());
                                        // `BulkMessage::payload` is
                                        // `Arc<[u8]>` (cheap clone via
                                        // refcount); `ShmEntry::payload`
                                        // is `Vec<u8>`. Convert via
                                        // `to_vec()` — the per-frame
                                        // cap (`MAX_BULK_FRAME_PAYLOAD`
                                        // in `vmm::bulk`) bounds the
                                        // allocation, so the conversion
                                        // is one-shot per drained
                                        // message and not a hot-path
                                        // concern.
                                        buf.extend(
                                            drained
                                                .messages
                                                .iter()
                                                .filter(|m| {
                                                    m.msg_type
                                                        != crate::vmm::wire::MSG_TYPE_SNAPSHOT_REQUEST
                                                })
                                                .map(|m| crate::vmm::wire::ShmEntry {
                                                    msg_type: m.msg_type,
                                                    payload: m.payload.to_vec(),
                                                    crc_ok: m.crc_ok,
                                                }),
                                        );
                                    }
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
                            break 'coord;
                        }
                        if freeze_coord_bsp_done.load(Ordering::Acquire) {
                            break 'coord;
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
                        let tcr_val = freeze_coord_tcr_el1
                            .as_ref()
                            .map(|c| c.load(Ordering::Acquire))
                            .unwrap_or(0);
                        owned_accessor = crate::monitor::bpf_map::GuestMemMapAccessorOwned::new(
                            mem, vmlinux, tcr_val,
                        )
                        .ok();
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
                        let tcr_val = freeze_coord_tcr_el1
                            .as_ref()
                            .map(|c| c.load(Ordering::Acquire))
                            .unwrap_or(0);
                        owned_prog_accessor =
                            crate::monitor::bpf_prog::GuestMemProgAccessorOwned::new(
                                mem, vmlinux, tcr_val,
                            )
                            .ok();
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
                        let tcr_val = freeze_coord_tcr_el1
                            .as_ref()
                            .map(|c| c.load(Ordering::Acquire))
                            .unwrap_or(0);
                        let start_kernel_map =
                            crate::monitor::symbols::start_kernel_map_for_tcr(tcr_val)
                                .unwrap_or(crate::monitor::symbols::START_KERNEL_MAP);
                        let pco_pa = crate::monitor::symbols::text_kva_to_pa_with_base(
                            syms.per_cpu_offset,
                            start_kernel_map,
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
                                let walk = k.walk_context();
                                let kind_pa = crate::monitor::idr::translate_any_kva(
                                    mem,
                                    walk.cr3_pa,
                                    walk.page_offset,
                                    exit_kind_kva,
                                    walk.l5,
                                    walk.tcr_el1,
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
                                    // Flip the fast-path gate so
                                    // every vCPU's
                                    // `self_arm_watchpoint` stops
                                    // short-circuiting on
                                    // `any_armed == 0` and falls
                                    // through to the per-slot
                                    // Acquire load on
                                    // `request_kva`. Idempotent.
                                    // Must follow the Release on
                                    // `request_kva` — `mark_armed`
                                    // is `Relaxed`, so the
                                    // synchronizes-with edge that
                                    // publishes the new KVA value
                                    // comes from the slot's
                                    // Release/Acquire pair, not the
                                    // gate. Without this call the
                                    // gate stays at 0 forever and
                                    // every vCPU's self-arm
                                    // returns false at the
                                    // pre-load short-circuit —
                                    // i.e. the watchpoint never
                                    // arms in `KVM_SET_GUEST_DEBUG`
                                    // and no fire ever reaches
                                    // `KVM_EXIT_DEBUG`.
                                    freeze_coord_watchpoint.mark_armed();
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
                            let pco_pa = kernel.text_kva_to_pa(syms.per_cpu_offset);
                            let pco_offsets = crate::monitor::symbols::read_per_cpu_offsets(
                                mem,
                                pco_pa,
                                freeze_coord_num_cpus,
                            );
                            let rq_pas = crate::monitor::symbols::compute_rq_pas(
                                syms.runqueues,
                                &pco_offsets,
                                walk.page_offset,
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
                            // Defense-in-depth UAF gate. The primary
                            // defense is the `freeze_coord_handle.join()`
                            // call in run_vm BEFORE the BSP `VcpuFd`
                            // falls out of scope; this Acquire load
                            // is the secondary check that no
                            // freeze_and_capture body issues a
                            // `bsp_ie_handle.set(1)` write through a
                            // stale kvm_run mmap pointer. Re-read
                            // before the BSP-side ie.set() below.
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
                            // that owns it). AP IE writes are safe
                            // because the AP threads are joined in
                            // collect_results AFTER the coord joins
                            // — the coord cannot outlive an AP's
                            // VcpuFd. The BSP IE write is gated on
                            // `bsp_alive` because run_vm drops the
                            // BSP before collect_results runs; see
                            // the gate's doc above.
                            for ie in freeze_coord_ap_ies.iter().flatten() {
                                ie.set(1);
                            }
                            if bsp_alive_at_start
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
                            // gated on the same `bsp_alive` flag the
                            // ie.set above used — pthread_kill
                            // against a tid whose thread has already
                            // exited returns ESRCH and is harmless,
                            // but the gate keeps the path symmetric
                            // and avoids spurious ESRCH log noise.
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
                                if freeze_coord_kill.load(Ordering::Acquire)
                                    || freeze_coord_bsp_done.load(Ordering::Acquire)
                                {
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
                                // Rendezvous timed out — at least
                                // one vCPU never set its parked
                                // flag, so we cannot safely read
                                // guest memory. Break out of the
                                // 'capture block so the unified
                                // thaw + post-thaw barrier still
                                // runs.
                                tracing::debug!(
                                    "freeze-coord: dump skipped: rendezvous timed out"
                                );
                                break 'capture None;
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
                        if let Some(ref blk) = freeze_coord_virtio_blk {
                            blk.lock().resume();
                        }
                        freeze_coord_freeze.store(false, Ordering::Release);
                        let _ = freeze_coord_thaw_evt.write(1);

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
                                // dump returns.
                                let on_demand = freeze_and_capture(false);
                                thaw_and_barrier();
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
                                        // BSP-alive gate is consulted
                                        // once at the kick site; same
                                        // Acquire-load the freeze
                                        // closure uses so the BSP IE
                                        // write and pthread_kill share
                                        // the snapshot — `run_vm`
                                        // flips this to false only
                                        // AFTER joining the coordinator
                                        // (see `bsp_alive` in run_vm),
                                        // so a `true` reading here is
                                        // load-bearing for the BSP
                                        // kvm_run mmap's liveness.
                                        let bsp_alive_at_arm =
                                            bsp_alive_for_coord
                                                .load(Ordering::Acquire);
                                        match arm_user_watchpoint(
                                            &freeze_coord_watchpoint,
                                            symbol_cache,
                                            &tag,
                                            &freeze_coord_ap_pthreads,
                                            &freeze_coord_ap_ies,
                                            freeze_coord_bsp_tid,
                                            freeze_coord_bsp_ie_handle.as_ref(),
                                            bsp_alive_at_arm,
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
                            // a CAPTURE-class TLV request is still
                            // running). Re-arm the slot's hit flag
                            // so the next epoll iteration handles it.
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
                        }
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
                                    ctx.walk,
                                    ctx.watchdog_timestamp_pa,
                                    ctx.start_kernel_map,
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
                    tracing::warn!(
                        slot_idx,
                        %tag,
                        "freeze-coord: user-watchpoint fire pending at coord exit; \
                         storing placeholder report (no capture possible during \
                         teardown — vCPU rendezvous would race teardown joins)"
                    );
                    let placeholder = crate::monitor::dump::FailureDumpReport {
                        schema: crate::monitor::dump::SCHEMA_SINGLE.to_string(),
                        maps: Vec::new(),
                        vcpu_regs: Vec::new(),
                        sdt_allocations: Vec::new(),
                        prog_runtime_stats: Vec::new(),
                        prog_runtime_stats_unavailable: Some(
                            "coord exited before capture".to_string(),
                        ),
                        per_cpu_time: Vec::new(),
                        task_enrichments: Vec::new(),
                        task_enrichments_unavailable: Some(
                            "coord exited before capture".to_string(),
                        ),
                        event_counter_timeline: Vec::new(),
                        rq_scx_states: Vec::new(),
                        dsq_states: Vec::new(),
                        scx_sched_state: None,
                        scx_walker_unavailable: Some(
                            "coord exited before capture".to_string(),
                        ),
                        vcpu_perf_at_freeze: Vec::new(),
                        per_node_numa: Vec::new(),
                        per_node_numa_unavailable: Some(
                            "coord exited before capture".to_string(),
                        ),
                        dump_truncated_at_us: None,
                        probe_counters: None,
                    };
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
                let residual = bulk_assembler.take_residual();
                if !residual.is_empty() {
                    freeze_coord_virtio_con.lock().push_back_bulk(&residual);
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
                    // Soft deadline: request graceful shutdown by
                    // pushing `SIGNAL_VC_SHUTDOWN` into virtio-console
                    // RX. The guest's `hvc0_poll_loop` blocks on
                    // `/dev/hvc0` and recognises the byte directly —
                    // no SHM signal slot needed. The BSP keeps running
                    // so the guest can flush serial and reboot
                    // normally.
                    if !soft_fired && soft_deadline.is_some_and(|d| Instant::now() >= d) {
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
             (run-loop sentinel — final exit code comes from bulk port / COM2 in collect_results)"
        );

        // Join the watchdog before dropping `bsp`. The watchdog holds an
        // ImmediateExitHandle pointing into bsp's kvm_run mmap. If bsp is
        // dropped first, the watchdog may write to unmapped memory.
        let _ = watchdog.join();

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
            // post-exit drain and the SHM CRASH ring drain so
            // every frame the guest published reaches the verdict.
            bulk_messages: freeze_coord_bulk_messages,
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
        _probes_ready_evt: EventFd,
        virtio_con: Option<Arc<PiMutex<virtio_console::VirtioConsole>>>,
        tcr_el1: Option<Arc<std::sync::atomic::AtomicU64>>,
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
        let num_cpus = self.topology.total_cpus();
        let kill_clone = kill.clone();
        let kill_evt_clone = kill_evt.clone();
        let dump_trigger = self.monitor_thresholds.map(|thresholds| {
            monitor::reader::DumpTrigger {
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

                // Resolve the kernel image base. On x86_64 this is
                // the compile-time constant; on aarch64 it depends
                // on `VA_BITS_MIN` derived from `TCR_EL1.T1SZ` and
                // `TCR_EL1.TG1` (granule). The `tcr_el1` cache is
                // populated lazily by the BSP loop on first
                // successful read post-MMU bringup — if it's still
                // 0 here, fall back to the const (48-bit VA), which
                // is correct on aarch64 with T1SZ=16. VA_BITS=47
                // (16 KB granule) hosts produce the wrong base in
                // that race window; the post-wait re-derive below
                // catches up once TCR_EL1 lands.
                let tcr_el1_value = tcr_el1
                    .as_ref()
                    .map(|c| c.load(std::sync::atomic::Ordering::Acquire))
                    .unwrap_or(0);
                let start_kernel_map_for_thread =
                    monitor::symbols::start_kernel_map_for_tcr(tcr_el1_value)
                        .unwrap_or(monitor::symbols::START_KERNEL_MAP);

                let page_offset = monitor::symbols::resolve_page_offset_with_tcr(
                    &mem,
                    &symbols,
                    start_kernel_map_for_thread,
                    tcr_el1_value,
                );

                // __per_cpu_offset is a kernel data symbol: use text mapping.
                let pco_pa = monitor::symbols::text_kva_to_pa_with_base(
                    symbols.per_cpu_offset,
                    start_kernel_map_for_thread,
                );
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
                        let scx_root_pa = monitor::symbols::text_kva_to_pa_with_base(
                            scx_root_kva,
                            start_kernel_map_for_thread,
                        );
                        return Some(monitor::reader::WatchdogOverride::ScxSched {
                            scx_root_pa,
                            watchdog_offset: wd_offs.scx_sched_watchdog_timeout_off,
                            jiffies,
                            page_offset,
                        });
                    }
                    // Pre-7.1 fallback: direct write to scx_watchdog_timeout static global.
                    if let Some(wdt_kva) = symbols.scx_watchdog_timeout {
                        let watchdog_timeout_pa = monitor::symbols::text_kva_to_pa_with_base(
                            wdt_kva,
                            start_kernel_map_for_thread,
                        );
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
                        let scx_root_pa = monitor::symbols::text_kva_to_pa_with_base(
                            scx_root_kva,
                            start_kernel_map_for_thread,
                        );
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
                let cr3_pa = monitor::symbols::text_kva_to_pa_with_base(
                    symbols.init_top_pgt.unwrap_or(0),
                    start_kernel_map_post_wait,
                );
                let l5 = monitor::symbols::resolve_pgtable_l5(
                    &mem,
                    &symbols,
                    start_kernel_map_post_wait,
                );
                // aarch64 TCR_EL1 (granule + T1SZ) for the
                // page-table walker. Threaded through ProgStatsCtx
                // so vmalloc-backed percpu `bpf_prog_stats`
                // translations succeed once the BSP populates the
                // cache. Always 0 on x86_64.
                let tcr_el1_val = tcr_el1
                    .as_ref()
                    .map(|c| c.load(std::sync::atomic::Ordering::Acquire))
                    .unwrap_or(0);
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
                                walk: monitor::reader::WalkContext {
                                    cr3_pa,
                                    page_offset,
                                    l5,
                                    tcr_el1: tcr_el1_val,
                                },
                                prog_idr_kva,
                                offsets: prog_offsets,
                                start_kernel_map: start_kernel_map_post_wait,
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
                    prog_stats_ctx: prog_stats_ctx.as_ref(),
                    page_offset,
                    start_kernel_map: start_kernel_map_post_wait,
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
    pub(super) fn start_bpf_map_write(
        &self,
        vm: &kvm::KtstrKvm,
        kill: &Arc<AtomicBool>,
        probes_ready_evt: EventFd,
        tcr_el1: Option<Arc<std::sync::atomic::AtomicU64>>,
        virtio_con: Arc<PiMutex<virtio_console::VirtioConsole>>,
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
                let phase1_deadline =
                    std::time::Instant::now() + std::time::Duration::from_secs(30);
                let owned = loop {
                    let tcr_val = tcr_el1
                        .as_ref()
                        .map(|c| c.load(std::sync::atomic::Ordering::Acquire))
                        .unwrap_or(0);
                    match monitor::bpf_map::GuestMemMapAccessorOwned::new(&mem, &vmlinux, tcr_val) {
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
        run_start: Instant,
        timeout: Duration,
        parked_evt: Option<&Arc<EventFd>>,
        thaw_evt: Option<&Arc<EventFd>>,
        kill_evt: Option<&Arc<EventFd>>,
        tcr_el1_cache: Option<&Arc<std::sync::atomic::AtomicU64>>,
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
        for vt in run.ap_threads {
            vt.wait_for_exit(Duration::from_secs(5));
            let _ = vt.handle.join();
        }

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
        run.watchpoint
            .request_kva
            .store(0, Ordering::Release);
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
                None => (None, BulkDrainResult::default()),
            };

        if let Some(h) = run.bpf_write_handle {
            let _ = h.join();
        }

        // Drain the virtio-console port-1 TX accumulator: the guest
        // wrote bulk TLV-framed messages (STIMULUS, EXIT, SCHED_EXIT,
        // PAYLOAD_METRICS, RAW_PAYLOAD_OUTPUT, etc.) to
        // `/dev/vport0p1`; the host side accumulated them into
        // `port1_tx_buf` and we parse them here through
        // `parse_tlv_stream`. Port-1 uses backpressure rather than
        // drops — every byte the guest emitted is delivered, in
        // order.
        let bulk_bytes = run.virtio_con.lock().drain_bulk();
        let mut bulk_drain = host_comms::parse_tlv_stream(&bulk_bytes);
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
        let (guest_messages, stimulus_events) = if !mid_flight_drain.entries.is_empty()
            || !bulk_drain.entries.is_empty()
        {
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

        let app_output = run.com2.lock().output();
        let console_output = run.com1.lock().output();

        // Extract exit code: bulk port (primary), COM2 sentinel (fallback).
        let bulk_exit = guest_messages.as_ref().and_then(|d| {
            d.entries
                .iter()
                .rev()
                .find(|e| e.msg_type == wire::MSG_TYPE_EXIT && e.crc_ok && e.payload.len() == 4)
                .map(|e| i32::from_ne_bytes(e.payload[..4].try_into().unwrap()))
        });
        if let Some(code) = bulk_exit {
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

        // Extract crash message from COM2 output. The guest panic
        // hook in `rust_init.rs` writes `PANIC: <info>\n<bt>\n` to
        // `/dev/ttyS1`; the host-side parser
        // [`crate::test_support::extract_panic_message`] strips the
        // prefix and returns the trimmed remainder.
        let crash_message =
            crate::test_support::extract_panic_message(&app_output).map(|s| s.to_string());

        // Collect BPF verifier stats from host-side memory reads.
        let verifier_stats = self.collect_verifier_stats(&run.vm, run.tcr_el1.as_ref());

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
            guest_messages,
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
        tcr_el1: Option<&Arc<std::sync::atomic::AtomicU64>>,
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
        let kernel = match monitor::guest::GuestKernel::new(&mem, &vmlinux, tcr_val) {
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

#[cfg(test)]
mod snapshot_tlv_tests {
    //! Unit coverage for the TLV-based snapshot req/reply wiring.
    //!
    //! `decode_snapshot_request` and `frame_snapshot_reply` are
    //! testable in isolation — every assertion in the freeze
    //! coordinator's TOKEN_TX dispatch flows through these two
    //! helpers, so verifying their wire-format contract pins the
    //! load-bearing behaviour without booting a VM. Chain-level
    //! integration coverage of `queue_input_port1` lives in
    //! `virtio_console`'s own test module; here we cover only the
    //! payload encode / decode boundary.
    use super::*;
    use crate::vmm::wire::{
        FRAME_HEADER_SIZE, MSG_TYPE_SNAPSHOT_REPLY, SNAPSHOT_KIND_CAPTURE,
        SNAPSHOT_KIND_NONE, SNAPSHOT_KIND_WATCH, SNAPSHOT_REASON_MAX,
        SNAPSHOT_STATUS_ERR, SNAPSHOT_STATUS_OK, SNAPSHOT_TAG_MAX,
        ShmMessage, SnapshotReplyPayload, SnapshotRequestPayload,
    };
    use zerocopy::{FromBytes, IntoBytes};

    fn make_request_bytes(request_id: u32, kind: u32, tag: &str) -> Vec<u8> {
        let tag_bytes = tag.as_bytes();
        let mut tag_buf = [0u8; SNAPSHOT_TAG_MAX];
        let n = tag_bytes.len().min(SNAPSHOT_TAG_MAX);
        tag_buf[..n].copy_from_slice(&tag_bytes[..n]);
        SnapshotRequestPayload {
            request_id,
            kind,
            tag: tag_buf,
        }
        .as_bytes()
        .to_vec()
    }

    /// Happy-path CAPTURE request decodes to the matching typed
    /// fields and trims the tag at the first NUL.
    #[test]
    fn decode_capture_request_round_trip() {
        let bytes = make_request_bytes(7, SNAPSHOT_KIND_CAPTURE, "snap_1");
        let req = decode_snapshot_request(&bytes).expect("valid request decodes");
        assert_eq!(req.request_id, 7);
        assert_eq!(req.kind, SNAPSHOT_KIND_CAPTURE);
        assert_eq!(req.tag, "snap_1");
    }

    /// WATCH request decodes the same way as CAPTURE — the kind
    /// dispatch happens at the call site, not inside the decoder.
    #[test]
    fn decode_watch_request_round_trip() {
        let bytes = make_request_bytes(99, SNAPSHOT_KIND_WATCH, "scx_root");
        let req = decode_snapshot_request(&bytes).expect("valid request decodes");
        assert_eq!(req.kind, SNAPSHOT_KIND_WATCH);
        assert_eq!(req.tag, "scx_root");
    }

    /// Wrong-sized payload (1 byte short of the typed payload) is
    /// rejected — protects against a malformed guest stamping a
    /// partial request that would otherwise zerocopy into stack
    /// garbage.
    #[test]
    fn decode_rejects_undersized_payload() {
        let mut bytes = make_request_bytes(1, SNAPSHOT_KIND_CAPTURE, "x");
        bytes.pop();
        assert!(decode_snapshot_request(&bytes).is_none());
    }

    /// Wrong-sized payload (1 byte longer than typed payload) is
    /// rejected.
    #[test]
    fn decode_rejects_oversized_payload() {
        let mut bytes = make_request_bytes(1, SNAPSHOT_KIND_CAPTURE, "x");
        bytes.push(0xAA);
        assert!(decode_snapshot_request(&bytes).is_none());
    }

    /// `request_id == 0` is rejected — the wire-format contract
    /// reserves zero so a zero-initialised reply payload from a
    /// prior protocol version cannot accidentally match.
    #[test]
    fn decode_rejects_zero_request_id() {
        let bytes = make_request_bytes(0, SNAPSHOT_KIND_CAPTURE, "x");
        assert!(decode_snapshot_request(&bytes).is_none());
    }

    /// `kind == NONE` is rejected — the sentinel value must not
    /// appear on the wire.
    #[test]
    fn decode_rejects_kind_none() {
        let bytes = make_request_bytes(1, SNAPSHOT_KIND_NONE, "x");
        assert!(decode_snapshot_request(&bytes).is_none());
    }

    /// Unknown kind values decode to `Some` — the dispatch in the
    /// freeze coord matches on `kind` and frames an ERR reply for
    /// anything outside the CAPTURE/WATCH set, so the decoder must
    /// not pre-filter on kind.
    #[test]
    fn decode_accepts_unknown_kind_for_dispatch_handling() {
        let bytes = make_request_bytes(42, 0xDEAD_BEEF, "tag");
        let req = decode_snapshot_request(&bytes).expect("decode succeeds");
        assert_eq!(req.kind, 0xDEAD_BEEF);
        assert_eq!(req.tag, "tag");
    }

    /// Tag without an internal NUL fills the whole buffer; the
    /// decoder takes the full `SNAPSHOT_TAG_MAX` bytes.
    #[test]
    fn decode_full_buffer_tag_uses_full_length() {
        let long = "a".repeat(SNAPSHOT_TAG_MAX);
        let bytes = make_request_bytes(1, SNAPSHOT_KIND_CAPTURE, &long);
        let req = decode_snapshot_request(&bytes).expect("decode succeeds");
        assert_eq!(req.tag.len(), SNAPSHOT_TAG_MAX);
        assert!(req.tag.chars().all(|c| c == 'a'));
    }

    /// Reply frame is exactly header + 72-byte payload; CRC32
    /// over payload bytes matches the wire-format contract
    /// `parse_tlv_stream` enforces on the guest side.
    #[test]
    fn frame_reply_size_and_crc() {
        let bytes = frame_snapshot_reply(123, SNAPSHOT_STATUS_OK, "");
        assert_eq!(bytes.len(), FRAME_HEADER_SIZE + std::mem::size_of::<SnapshotReplyPayload>());
        let header = ShmMessage::read_from_bytes(&bytes[..FRAME_HEADER_SIZE])
            .expect("header decodes");
        assert_eq!(header.msg_type, MSG_TYPE_SNAPSHOT_REPLY);
        assert_eq!(header.length as usize, std::mem::size_of::<SnapshotReplyPayload>());
        let payload_bytes = &bytes[FRAME_HEADER_SIZE..];
        assert_eq!(header.crc32, crc32fast::hash(payload_bytes));
    }

    /// Reply payload round-trips through bytes — the request_id
    /// echo, the status, and the reason text are preserved
    /// exactly.
    #[test]
    fn frame_reply_payload_round_trip() {
        let bytes = frame_snapshot_reply(0xCAFE_BABE, SNAPSHOT_STATUS_ERR, "rendezvous timeout");
        let payload_bytes = &bytes[FRAME_HEADER_SIZE..];
        let reply = SnapshotReplyPayload::read_from_bytes(payload_bytes)
            .expect("payload decodes");
        assert_eq!(reply.request_id, 0xCAFE_BABE);
        assert_eq!(reply.status, SNAPSHOT_STATUS_ERR);
        let len = reply
            .reason
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(SNAPSHOT_REASON_MAX);
        assert_eq!(&reply.reason[..len], b"rendezvous timeout");
    }

    /// Reasons longer than `SNAPSHOT_REASON_MAX` are truncated to
    /// the buffer; the trailing byte may be a partial UTF-8
    /// sequence but never overflows.
    #[test]
    fn frame_reply_truncates_long_reason() {
        let long = "x".repeat(SNAPSHOT_REASON_MAX + 16);
        let bytes = frame_snapshot_reply(1, SNAPSHOT_STATUS_ERR, &long);
        let payload_bytes = &bytes[FRAME_HEADER_SIZE..];
        let reply = SnapshotReplyPayload::read_from_bytes(payload_bytes)
            .expect("payload decodes");
        assert_eq!(reply.reason.len(), SNAPSHOT_REASON_MAX);
        assert!(reply.reason.iter().all(|&b| b == b'x'));
    }

    /// Empty reason yields a fully-zeroed reason buffer — the
    /// guest side renders this as the empty string.
    #[test]
    fn frame_reply_empty_reason_zero_pads() {
        let bytes = frame_snapshot_reply(1, SNAPSHOT_STATUS_OK, "");
        let payload_bytes = &bytes[FRAME_HEADER_SIZE..];
        let reply = SnapshotReplyPayload::read_from_bytes(payload_bytes)
            .expect("payload decodes");
        assert!(reply.reason.iter().all(|&b| b == 0));
    }
}
