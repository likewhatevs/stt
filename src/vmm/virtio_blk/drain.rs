//! Request-queue drain bracket for virtio-block.
//!
//! Houses the `drain_bracket_impl` free function and the
//! `DrainOutcome` it returns. Split out of `device.rs` for module
//! locality so the drain pipeline (chain validation, throttle gate,
//! handler dispatch, completion publish, IRQ raise) sits in one
//! file.
//!
//! # Public surface (within `super`)
//!
//! - [`DrainOutcome`] — return value distinguishing successful
//!   drain (`Done`) from a throttle-stall rollback
//!   (`ThrottleStalled { wait_nanos }`).
//! - [`drain_bracket_impl`] — the production drain entry point.
//!   Called by `worker_thread_main` (production) and
//!   `VirtioBlk::drain_inline` (cfg(test)) against a
//!   `BlkWorkerState` they own.
//!
//! All chain-shape validation (header presence, status descriptor,
//! SEG_MAX / SIZE_MAX bounds, direction, sub-sector data length)
//! happens BEFORE the throttle bucket is consumed — a malformed
//! request never drains the bucket. The handler dispatch
//! (`VirtioBlk::handle_*_impl`) lives in `handlers.rs`; the
//! pre-throttle terminal classifier (`classify_pre_throttle`) lives
//! on `VirtioBlk` itself in `device.rs`.

use std::sync::atomic::{AtomicU32, Ordering};

use vm_memory::{Bytes, GuestAddress, GuestMemoryMmap};
use vmm_sys_util::eventfd::EventFd;

use virtio_bindings::virtio_blk::{
    VIRTIO_BLK_S_IOERR, VIRTIO_BLK_S_UNSUPP, VIRTIO_BLK_T_FLUSH, VIRTIO_BLK_T_GET_ID,
    VIRTIO_BLK_T_IN, VIRTIO_BLK_T_OUT,
};
use virtio_bindings::virtio_config::VIRTIO_CONFIG_S_NEEDS_RESET;
use virtio_bindings::virtio_mmio::{VIRTIO_MMIO_INT_CONFIG, VIRTIO_MMIO_INT_VRING};
use virtio_queue::Error as VirtioQueueError;
use virtio_queue::{QueueOwnedT, QueueT};

use super::{
    BlkQueue, BlkWorkerState, ChainDescriptor, NUM_QUEUES, REQ_QUEUE, VIRTIO_BLK_OUTHDR_SIZE,
    VIRTIO_BLK_SECTOR_SIZE, VIRTIO_BLK_SEG_MAX, VIRTIO_BLK_SIZE_MAX, VirtioBlk, VirtioBlkOutHdr,
    publish_completion,
};

/// Outcome of a single `drain_bracket_impl` invocation.
///
/// `Done` — the inner pop loop ran to None and `enable_notification`
/// settled (no pending chains; nothing to retry). The caller should
/// rest until the next kick.
///
/// `ThrottleStalled { wait_nanos }` — a chain was popped whose IO
/// budget the throttle bucket cannot satisfy; the chain has been
/// rolled back via `set_next_avail(prev.wrapping_sub(1))` (so the
/// next drain re-pops it) and `wait_nanos` is the worst-case
/// delay before the bucket holds enough tokens to satisfy it. The
/// worker thread arms a timerfd for this duration; tests step the
/// bucket forward and re-call `process_requests`. `wait_nanos ==
/// 0` means the bucket is unlimited or already refilled to
/// sufficiency — the caller should re-drain immediately rather
/// than waiting on a timer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DrainOutcome {
    Done,
    ThrottleStalled { wait_nanos: u64 },
}

/// Drain the request queue, processing reads/writes/flushes
/// against the backing file and respecting the throttle.
///
/// The chain is walked in one pass and `add_used` is called
/// in the same loop iteration that completes the request.
/// `pop_descriptor_chain` returns a chain whose lifetime ends
/// at the bottom of the iteration (after we've collected the
/// data-segment vector), so the borrow on the queue is released
/// before `add_used` re-borrows it. This mirrors the
/// virtio-console pattern (see `process_tx` in virtio_console.rs).
///
/// Free function (not a method) so the worker thread (production)
/// and the inline test harness (cfg(test)) can both invoke it
/// against a `BlkWorkerState` they own without taking a method
/// receiver — production owns `state` on the worker thread and the
/// inline path borrows it via `self.worker.engine`.
///
/// Borrows guest memory, the irqfd, and the interrupt-status atomic
/// from the device — those live on the MMIO side (`VirtioBlk`) and
/// are passed in. `queues` is borrowed mutably so the drain can
/// pop / add_used / disable+enable_notification / needs_notification
/// in lock-pop-unlock-walk-lock-add_used order without holding any
/// queue lock during IO.
///
/// Returns `DrainOutcome::ThrottleStalled` when a chain was popped
/// but its IO budget is exhausted: the chain is rolled back via
/// `set_next_avail(prev.wrapping_sub(1))` (so the next drain re-pops
/// it) and the returned wait duration tells the caller how long
/// until the bucket will hold enough tokens to satisfy the request.
/// The worker thread arms a timerfd from this duration; when the
/// timer fires, the drain re-runs. (`go_to_previous_position` from
/// the virtio-queue crate has the same effect, but it lives on the
/// `QueueOwnedT` trait which `QueueSync` does not implement;
/// `set_next_avail` is on the base `QueueT` and works for both
/// alias targets in this module.)
///
/// On stall, no S_IOERR / no add_used / no signal — the chain stays
/// invisible to the guest until the retry. `throttled_count` is bumped
/// per stall so operators can observe the rate. `Done` indicates
/// the queue was drained to None and re-enable settled (no pending
/// chains).
pub(crate) fn drain_bracket_impl(
    state: &mut BlkWorkerState,
    queues: &mut [BlkQueue; NUM_QUEUES],
    mem: &GuestMemoryMmap,
    irq_evt: &EventFd,
    interrupt_status: &AtomicU32,
    device_status: &AtomicU32,
) -> DrainOutcome {
    // Pre-rebind / post-reset gate. After `q.reset()` clears the
    // queue (zeroing desc/avail/used GPAs and `ready`), there is
    // a window before the guest re-publishes addresses and sets
    // `QUEUE_READY = 1`. A kick or timer wakeup that lands in
    // that window must not call `disable_notification` /
    // `enable_notification` — both write to the used ring's
    // `flags` / `avail_event` fields, and a used-ring GPA of 0
    // (the post-reset state) causes a spurious write to guest
    // physical address 0. Worse, `pop_descriptor_chain` against
    // a stale ring-cursor can mis-read descriptor entries.
    //
    // `QueueT::ready()` returns `true` only after the guest has
    // written `QUEUE_READY = 1` (post-rebind). In production the
    // worker may receive kicks routed through the device's
    // `kick_fd` between `respawn_worker` and the guest's first
    // post-reset `set_ready(true)` MMIO write — this gate makes
    // those drains a no-op until the guest finishes rebinding.
    if !queues[REQ_QUEUE].ready() {
        return DrainOutcome::Done;
    }

    // Hostile-guest defense gate. A previous drain observed
    // `Error::InvalidAvailRingIndex` from `Queue::iter` (the
    // guest's avail.idx was more than `queue.size` ahead of
    // `next_avail`, violating virtio-v1.2 §2.7.13.3 avail.idx
    // semantics). The
    // structural invariant the iterator depends on is broken;
    // every subsequent `iter()` call would re-trip the same
    // error, and `enable_notification` would re-arm
    // immediately, looping the worker forever at full
    // vCPU/host-CPU cost.
    //
    // Returning `Done` without touching the queue:
    // - skips `disable_notification` (no spurious used.flags
    //   write — the guest already poisoned the queue, more
    //   side effects make the symptom worse, not better),
    // - skips `iter()` (no second `invalid_avail_idx_count`
    //   bump per kick — the counter is per-event, the flag
    //   makes it event-once),
    // - skips `enable_notification` (no Ok(true) re-loop and
    //   no irqfd write).
    //
    // The flag clears only on a full virtio reset
    // (`reset_engine_inline` / `respawn_worker` rebuilds the
    // state with `queue_poisoned: false`). Until then the
    // device will not service IO — the guest's blk-mq layer
    // observes hangs and the operator sees a non-zero
    // `invalid_avail_idx_count` in the failure dump.
    if state.queue_poisoned {
        return DrainOutcome::Done;
    }

    // The request loop calls handlers (which take `&` borrows
    // of state.backing/state.counters) plus throttle bucket
    // mutation (`&mut state.ops_bucket` / `&mut state.bytes_bucket`).
    // To keep the borrow checker happy we materialise the queue
    // handle separately (`&mut queues[REQ_QUEUE]`) and reach
    // into `&mut state` only via the disjoint fields it owns.
    // The eventfd write that signals the guest is hoisted to
    // the end so it does not alias with the queue mutation in
    // the loop.
    let mut signal_needed = false;
    // Set when the throttle path stalls; carries the
    // worst-case wait time (in nanoseconds) before the bucket
    // refills enough to satisfy the rolled-back chain. None
    // when the drain reached the natural end (all chains
    // processed, queue empty, enable_notification settled).
    let mut stall_outcome: Option<u64> = None;
    // Outer bracket: disable_notification → drain → enable_notification.
    // Canonical virtio-queue pattern — the doctest on the
    // `Queue` struct in the virtio-queue crate spells out the
    // disable/drain/enable shape this loop mirrors.
    // `Queue::enable_notification` returns Ok(true) when new
    // chain heads appeared during the disabled window — re-drain
    // to avoid stranding chains the guest has enqueued without
    // a fresh QUEUE_NOTIFY MMIO exit. Its trait-level contract
    // on `QueueT::enable_notification` documents the
    // re-iteration semantics. Without re-checking, a chain
    // enqueued after our final `pop_descriptor_chain` returns
    // None but before notifications come back on would sit
    // unprocessed until the guest's hung-task watchdog fired
    // (`kernel.hung_task_timeout_secs`, default 120 s — virtio_blk
    // has no `mq_ops->timeout`, so blk-mq won't surface the stall).
    //
    // `Queue::disable_notification` semantics depend on whether
    // EVENT_IDX is negotiated (see `Queue::set_notification`,
    // which `disable_notification` and `enable_notification`
    // both delegate to):
    //   * legacy path (event_idx_enabled=false): writes the
    //     VRING_USED_F_NO_NOTIFY flag in used.flags, telling
    //     the guest to skip QUEUE_NOTIFY MMIO writes during
    //     the drain — removes redundant vCPU exits.
    //   * EVENT_IDX path (event_idx_enabled=true):
    //     disable_notification is a no-op (queue.rs:241-244).
    //     Suppression of guest kicks relies on NOT updating
    //     avail_event during the drain — avail_event stays at
    //     whatever the prior enable_notification wrote.
    // Either way, the bracket pattern is correct; both paths
    // route through the canonical disable/enable.
    'outer: loop {
        // Best-effort disable; failure is non-fatal — the worst
        // case is the guest issues a redundant QUEUE_NOTIFY
        // mid-drain that we'd absorb on the next call anyway.
        if let Err(e) = queues[REQ_QUEUE].disable_notification(mem) {
            tracing::warn!(%e, "virtio-blk disable_notification failed");
        }
        loop {
            // Pop one chain via `iter()`/`.next()` so we OBSERVE
            // `Error::InvalidAvailRingIndex` instead of swallowing
            // it. The bare `Queue::pop_descriptor_chain` impl
            // (queue.rs:573-587) calls iter() internally, logs any
            // error, and returns None — masking the structural
            // violation as "no chain available" and letting
            // `enable_notification` re-arm immediately, looping
            // the worker forever against a hostile guest.
            //
            // `iter()` is on `QueueOwnedT`, which only the bare
            // `Queue` implements; we reach it via `q.lock()` —
            // `&mut Queue` for `Queue` (cfg(test) alias) and
            // `MutexGuard<Queue>` for `QueueSync` (cfg(not(test))).
            // Both deref to `Queue`, so `guard.iter(mem)` compiles
            // for both alias targets. Two-step extraction keeps
            // the borrow tight: take the iter inside a block and
            // capture only the outcome (Some(chain)/None/Err(_))
            // before dropping the lock guard. The
            // `DescriptorChain<M>` owns its own `mem.clone()`
            // (queue.rs:761-766), so it does not borrow from the
            // iter or the guard — we can walk it freely after the
            // guard drops, and `add_used` etc. can re-borrow the
            // queue downstream.
            let iter_outcome = {
                let q = &mut queues[REQ_QUEUE];
                // `mut` is required for the cfg(not(test)) alias
                // (`MutexGuard<QueueState>`) so `guard.iter()` can
                // call `iter(&mut self)` via `DerefMut`. In the
                // cfg(test) alias (`Queue`, returning `&mut
                // QueueState` directly), the binding is already a
                // mutable reference and the `mut` keyword is
                // redundant — hence `unused_mut` here.
                #[allow(unused_mut)]
                let mut guard = q.lock();
                match guard.iter(mem) {
                    Ok(mut iter) => Ok(iter.next()),
                    Err(e) => Err(e),
                }
            };
            let chain = match iter_outcome {
                Ok(Some(c)) => c,
                Ok(None) => break,
                Err(VirtioQueueError::InvalidAvailRingIndex) => {
                    // Hostile-guest poison. The avail.idx is more
                    // than `queue.size` ahead of the device's
                    // `next_avail` (virtio-v1.2 §2.7.13.3
                    // avail.idx-distance violation; check sits at
                    // queue.rs:707-709 in `AvailIter::new`). Mark
                    // the queue dead so future drains
                    // short-circuit, bump the per-event counter
                    // (gated by the flag — exactly one bump per
                    // poison event regardless of re-kicks), and
                    // bail without calling `enable_notification`.
                    // Re-enabling notifications would arm the
                    // next kick to re-trip the same error — a
                    // livelock. A full virtio reset is the only
                    // path back to service.
                    state.queue_poisoned = true;
                    state.counters.record_invalid_avail_idx();
                    tracing::warn!(
                        "virtio-blk avail.idx exceeds next_avail by more \
                         than queue.size (virtio-v1.2 §2.7.13.3 \
                         avail.idx-distance violation); poisoning queue \
                         until guest reset"
                    );
                    // Surface the structural failure as an
                    // observability signal:
                    //
                    // 1. NEEDS_RESET in the device_status FSM is
                    //    visible to the guest via
                    //    `mmio_read(VIRTIO_MMIO_STATUS)` and to the
                    //    host operator via sysfs / failure-dump. It
                    //    is the spec-compliant way (virtio-v1.2
                    //    §2.1.1, bit 0x40) for a device to advertise
                    //    "I need to be reset before I can service
                    //    IO." Cloud-hypervisor uses the same bit for
                    //    its hostile-guest shutdown path.
                    // 2. INT_CONFIG + irq_evt.write trigger the
                    //    guest IRQ path through
                    //    `vm_interrupt → virtio_config_changed →
                    //    __virtio_config_changed → drv->config_changed`
                    //    (drivers/virtio/virtio_mmio.c). For
                    //    virtio_blk the `config_changed` callback
                    //    (`virtblk_config_changed`) only re-reads
                    //    config-space CAPACITY — it does NOT
                    //    automatically surface NEEDS_RESET to blk-mq
                    //    or fail in-flight requests. So this
                    //    sequence is a SIGNAL to the guest, not a
                    //    recovery primitive: the guest's behavior
                    //    on a poisoned queue depends on whatever
                    //    higher-layer logic reads STATUS.
                    //
                    // The actual request-stopping defense is the
                    // `state.queue_poisoned` gate at the top of
                    // `drain_bracket_impl`: every subsequent drain
                    // short-circuits to `Done`, so no chain ever
                    // gets `add_used`-published. In-flight requests
                    // hang until the guest's hung-task watchdog
                    // fires (default 120 s, virtio_blk has no
                    // `mq_ops->timeout`) or a reset arrives. The
                    // NEEDS_RESET signal here gives operators the
                    // STATUS-read tool to detect the wedged state
                    // before the watchdog fires; it does not
                    // unwedge anything on its own.
                    //
                    // Fired exactly once at the poison transition:
                    // the queue_poisoned gate above this arm's
                    // entry returns `Done` for every subsequent
                    // kick, so re-kicks never re-enter this arm.
                    //
                    // SeqCst on the device_status fetch_or pairs
                    // with two reader sites:
                    //   1. The vCPU MMIO read of STATUS via
                    //      `load(Acquire)` in `mmio_read` — the
                    //      post-poison read reflects the bit so the
                    //      guest's STATUS query sees NEEDS_RESET.
                    //   2. `set_status`'s CAS retry loop. The
                    //      `compare_exchange` failure-side Acquire
                    //      synchronizes-with this SeqCst write so
                    //      the retry iteration's re-snapshot sees
                    //      the NEEDS_RESET bit and the monotone-bit
                    //      gate rejects the FSM advance instead of
                    //      clobbering the bit. This is the
                    //      load-bearing pairing — without it, a
                    //      vCPU set_status racing this fetch_or
                    //      would silently drop NEEDS_RESET on the
                    //      next FSM advance.
                    // The interrupt_status fetch_or uses Release
                    // ordering to mirror the existing INT_VRING
                    // write-side discipline at the V8 publish-path
                    // INTERRUPT_STATUS bit-set.
                    //
                    // INVARIANT: the worker may ONLY `fetch_or`
                    // `VIRTIO_CONFIG_S_NEEDS_RESET` into
                    // device_status — never `store`,
                    // `fetch_and`, `fetch_xor`, or `fetch_or` any
                    // OTHER bit. Termination of `set_status`'s
                    // CAS retry loop is bounded at AT MOST ONE
                    // worker-induced retry: NEEDS_RESET fetch_or
                    // is idempotent after the first call, so the
                    // worker can change `device_status` from one
                    // observable state to one other state and
                    // never again from the device side. The
                    // single-bit constraint makes that bound
                    // auditable; a future worker fetch_or'ing a
                    // different bit (e.g. a hypothetical
                    // VIRTIO_CONFIG_S_DEVICE_NEEDS_RESET-like
                    // extension) would expand the retry universe
                    // and the bound. A worker that cleared bits —
                    // store/fetch_and/fetch_xor — would let the
                    // retry loop spin indefinitely as the
                    // snapshot re-enters the previously-rejected
                    // state.
                    device_status.fetch_or(VIRTIO_CONFIG_S_NEEDS_RESET, Ordering::SeqCst);
                    interrupt_status.fetch_or(VIRTIO_MMIO_INT_CONFIG, Ordering::Release);
                    // SAFETY: EAGAIN requires counter saturation at
                    // u64::MAX-1 (~1.8e19 unobserved kicks) —
                    // implausible. EBADF means the fd closed during
                    // shutdown. The simultaneously-set INT_CONFIG
                    // bit above is the enduring guest-visible
                    // signal: `vm_interrupt`
                    // (drivers/virtio/virtio_mmio.c) reads
                    // INTERRUPT_STATUS on the next IRQ delivery and
                    // dispatches via the bit set — but on the
                    // poison path NO subsequent device IRQ fires.
                    // The queue_poisoned gate makes every later
                    // drain short-circuit to `Done` without ever
                    // calling `add_used` or triggering another
                    // signal, so a missed irqfd write here means
                    // the operator's only path to seeing the
                    // NEEDS_RESET state is `mmio_read(STATUS)` —
                    // which still works because the bit is on
                    // device_status. The guest's actual recovery
                    // path is a STATUS=0 reset, driven by the
                    // hung-task watchdog or operator action. We log
                    // any errno so a failed write surfaces in
                    // tracing rather than silently disappearing.
                    if let Err(e) = irq_evt.write(1) {
                        tracing::warn!(%e, "virtio-blk irq_evt.write failed");
                    }
                    break 'outer;
                }
                Err(e) => {
                    // Other iter() errors: `QueueNotReady` (the
                    // `ready()` gate above already filtered this;
                    // would only fire on a TOCTOU race with a
                    // vCPU-side reset MMIO write) or
                    // address-overflow on `avail_idx`. Log and
                    // bail — the kick is wasted but the device
                    // recovers on the next legitimate notify. Do
                    // NOT poison: these are not
                    // structural-invariant violations the way
                    // InvalidAvailRingIndex is, so a future
                    // legitimate kick may succeed.
                    //
                    // Re-arm notifications before bailing. The
                    // outer-loop's normal exit path calls
                    // `enable_notification` (Ok(false) arm at the
                    // bottom of the outer loop); a raw `break 'outer`
                    // here skips that re-arm and leaves used.flags
                    // with VRING_USED_F_NO_NOTIFY set (legacy path)
                    // or a stale avail_event (EVENT_IDX path) from
                    // the entry-side `disable_notification`. Without
                    // re-arm the next QUEUE_NOTIFY may not reach the
                    // device (legacy: the guest's `virtqueue_kick`
                    // skips the MMIO write when used.flags bit is
                    // set; EVENT_IDX: the guest checks avail_event
                    // for the suppression decision), and the queue
                    // hangs until the hung-task watchdog
                    // (`kernel.hung_task_timeout_secs`, default
                    // 120 s — virtio_blk has no `mq_ops->timeout`
                    // callback). Re-arming is best-effort: if it
                    // also fails we log and bail anyway.
                    if let Err(re) = queues[REQ_QUEUE].enable_notification(mem) {
                        tracing::warn!(%re, "virtio-blk enable_notification failed after iter() error");
                    }
                    tracing::warn!(%e, "virtio-blk iter() failed");
                    break 'outer;
                }
            };
            // Re-bind `q` after the iter-scoped guard drops so the
            // downstream `add_used` / `set_next_avail` /
            // `publish_completion` callers can hold a fresh mutable
            // borrow (the guard above released its lock when its
            // block expression returned).
            let q = &mut queues[REQ_QUEUE];
            let head = chain.head_index();

            // Walk the chain. Layout per virtio-v1.2 §5.2.6:
            //   - desc[0]: device-readable, 16-byte virtio_blk_outhdr
            //   - desc[1..N-1]: data segments (write-only for reads,
            //     read-only for writes; absent for flush)
            //   - desc[N-1]: device-writable, 1-byte status
            //
            // The kernel's `virtblk_add_req` always emits the status
            // descriptor last (drivers/block/virtio_blk.c). We rely
            // on that invariant: collect all descriptors, treat the
            // LAST one as the status candidate, the FIRST as the
            // header, everything in between as data segments. This
            // is simpler than the "first 1-byte write-only after
            // header" heuristic, which mis-classified chains
            // containing a 1-byte data descriptor.
            //
            // The first descriptor MUST be the header — it cannot
            // be write-only and cannot be shorter than the
            // `virtio_blk_outhdr` struct. A malformed first
            // descriptor must NOT silently fall through to a
            // later device-readable descriptor as the "header".
            // Re-use the device's scratch buffers across requests.
            // `clear()` keeps the underlying Vec capacity allocated
            // once at construction (sized by VIRTIO_BLK_SEG_MAX + 2),
            // so steady-state push/clear is amortized to zero
            // allocation. Hot-path optimization — drain_bracket_impl
            // runs on the worker thread in production (cfg(test): on
            // the test thread) and is invoked once per kick (one
            // per QUEUE_NOTIFY MMIO write in production).
            state.all_descs_scratch.clear();
            for desc in chain {
                state.all_descs_scratch.push(ChainDescriptor {
                    addr: desc.addr(),
                    len: desc.len(),
                    is_write_only: desc.is_write_only(),
                });
            }

            let chain_len = state.all_descs_scratch.len();

            let mut header_addr: Option<GuestAddress> = None;
            let mut status_addr: Option<GuestAddress> = None;
            if let Some((first, rest)) = state.all_descs_scratch.split_first() {
                if !first.is_write_only && (first.len as usize) >= VIRTIO_BLK_OUTHDR_SIZE {
                    header_addr = Some(first.addr);
                }
                if let Some((last, _middle)) = rest.split_last() {
                    // Status descriptor: device-writable, length >= 1.
                    // QEMU/firecracker/cloud-hypervisor all accept
                    // multi-byte status descriptors; the device
                    // writes the 1-byte status to the LAST byte of
                    // the descriptor (`last.addr + last.len - 1`)
                    // so the actual status-bearing position lines
                    // up with the kernel driver's `in_hdr` (the
                    // device-writable trailing buffer of the
                    // virtio_blk request, distinct from the
                    // device-readable `out_hdr` at desc[0]) per
                    // multi-byte in_hdr formats handled by
                    // `virtblk_vbr_status` at
                    // drivers/block/virtio_blk.c:329-332.
                    //
                    // `checked_add` defends against a hostile guest
                    // submitting `last.addr + last.len` near
                    // `u64::MAX`, which would wrap silently and let
                    // the device write a status byte at low GPA. On
                    // overflow `status_addr` stays None and the
                    // dispatcher drops the chain at the no-status
                    // gate.
                    if last.is_write_only && last.len >= 1 {
                        status_addr = last
                            .addr
                            .0
                            .checked_add(last.len as u64 - 1)
                            .map(GuestAddress);
                    }
                    // else: last descriptor isn't a valid status
                    // byte; status_addr stays None and the
                    // dispatcher's "no status descriptor" branch
                    // drops the chain. Data segments are not
                    // observed in the no-status path because the
                    // dispatcher returns before binding the
                    // data_segments slice.
                }
                // else: chain is exactly 1 descriptor → status
                // missing; both header (if valid) and status_addr
                // outcomes handled below.
            }

            // Validate chain shape and decode the header in one go.
            // Header missing or short → reject with S_IOERR if we
            // can identify the status descriptor; otherwise drop the
            // chain entirely (do NOT call `add_used`).
            //
            // A chain with no status descriptor MUST NOT be marked
            // used. The guest's `virtblk_done` reads the status from
            // `vbr->in_hdr.status` (drivers/block/virtio_blk.c
            // virtblk_vbr_status). That field is stale from prior
            // blk-mq tag use (initially zero from `__GFP_ZERO` at
            // allocation, stale on reuse), and `virtblk_result(0)`
            // maps to `BLK_STS_OK` — so calling `add_used` would
            // tell the guest the request SUCCEEDED when in fact the device
            // never wrote a status byte. That's a silent data
            // corruption vector for any guest read (the data buffer
            // is whatever was on the heap before the request) and a
            // silent dropped write for any guest write.
            //
            // Instead: leave the descriptor in the avail ring.
            // virtio_blk has no `mq_ops->timeout` callback (kernel
            // drivers/block/virtio_blk.c `virtio_mq_ops` has no
            // .timeout field), so blk-mq's per-request expiry path
            // (`blk_mq_rq_timed_out` in block/blk-mq.c) finds
            // `q->mq_ops->timeout == NULL`, skips the driver
            // callback, and falls through to `blk_add_timer` —
            // re-arming the same timer indefinitely. An unpublished
            // request therefore hangs the guest until either the
            // hung-task watchdog fires
            // (`kernel.hung_task_timeout_secs`, default 120s) or a
            // higher-layer (filesystem, application) retries. Hard
            // correctness requirement, not a performance trade-off.
            // Virtio-spec explicitly permits device-side stalls.
            // `io_errors` is bumped so the host operator sees the
            // malformed request.
            let Some(status_addr) = status_addr else {
                tracing::warn!(head, "virtio-blk request without status descriptor");
                state.counters.record_io_error();
                continue;
            };

            // SEG_MAX enforcement: the descriptor count includes the
            // header (1) + data segments (<= VIRTIO_BLK_SEG_MAX) +
            // status (1). Reject chains whose total count exceeds
            // `VIRTIO_BLK_SEG_MAX + 2`. Without this, the advertised
            // `seg_max` is a lie a hostile guest can ignore — it
            // could submit thousands of descriptors and force the
            // device to allocate matching scratch storage per
            // request. The check is placed AFTER status_addr
            // identification so the rejection produces a normal
            // IOERR completion (status byte write + add_used) rather
            // than dropping the chain entirely. Hoisting the check
            // earlier was the original design, but it left the
            // chain stuck in the avail ring with no path to error
            // surfacing — virtio_blk has no `mq_ops->timeout`
            // callback (drivers/block/virtio_blk.c `virtio_mq_ops`
            // has no `.timeout` field), so blk-mq alone never
            // surfaces the unpublished request; the guest only sees
            // the stall once the hung-task watchdog fires
            // (`kernel.hung_task_timeout_secs`, default 120 s).
            // Standard IOERR completion gives the guest's block
            // layer an immediate error to surface.
            if chain_len > VIRTIO_BLK_SEG_MAX as usize + 2 {
                tracing::warn!(
                    head,
                    desc_count = chain_len,
                    "virtio-blk chain exceeds seg_max + 2"
                );
                state.counters.record_io_error();
                if publish_completion(
                    mem,
                    q,
                    &state.counters,
                    head,
                    status_addr,
                    VIRTIO_BLK_S_IOERR as u8,
                    1,
                    "seg_max reject",
                ) {
                    signal_needed = true;
                }
                continue;
            }

            // When the header is missing/short but the status
            // descriptor is valid, publish IOERR via
            // `publish_completion` so the guest sees an immediate
            // error rather than hanging until the hung-task
            // watchdog fires (virtio_blk has no `mq_ops->timeout`).
            // `publish_completion` itself gates `add_used` on a
            // successful status-byte write — so a chain whose
            // status_addr is unmapped still ends up in the
            // "drop chain, request hangs the guest" branch via the
            // `false` return path (no add_used, no signal).
            // `io_errors` is bumped so the host operator sees the
            // malformed request.
            let Some(header_addr) = header_addr else {
                tracing::warn!(head, "virtio-blk request without valid header descriptor");
                state.counters.record_io_error();
                if publish_completion(
                    mem,
                    q,
                    &state.counters,
                    head,
                    status_addr,
                    VIRTIO_BLK_S_IOERR as u8,
                    1,
                    "bad header",
                ) {
                    signal_needed = true;
                }
                continue;
            };
            let hdr: VirtioBlkOutHdr = match mem.read_obj(header_addr) {
                Ok(h) => h,
                Err(_) => {
                    tracing::warn!(head, "virtio-blk header read failed");
                    state.counters.record_io_error();
                    if publish_completion(
                        mem,
                        q,
                        &state.counters,
                        head,
                        status_addr,
                        VIRTIO_BLK_S_IOERR as u8,
                        1,
                        "bad hdr read",
                    ) {
                        signal_needed = true;
                    }
                    continue;
                }
            };
            let req_type = hdr.type_;
            let sector = hdr.sector;
            // Borrow the chain's data-segment slice once. Sliced
            // directly from `all_descs_scratch[1..chain_len - 1]`
            // — header is at index 0, status is at index
            // `chain_len - 1` (we just unwrapped status_addr from
            // that descriptor), so everything in between is the
            // data payload. No separate Vec or copy.
            //
            // chain_len >= 2 here because status_addr is Some
            // (`split_last` produced a `last` element, which means
            // `rest.len() >= 1`, which means `chain_len >= 2`).
            // The slice is therefore always in-bounds.
            //
            // The borrow is immutable; `&state.all_descs_scratch[..]`
            // is disjoint from `&mut queues[..]` (the `q` borrow)
            // and `&mut state.ops_bucket` / `&mut state.bytes_bucket`,
            // so split-borrow lets all coexist.
            let data_segments: &[ChainDescriptor] = &state.all_descs_scratch[1..chain_len - 1];

            // SIZE_MAX enforcement: reject any chain that violates
            // the per-descriptor cap we advertised. A guest that
            // submits a descriptor longer than VIRTIO_BLK_SIZE_MAX
            // is either buggy or hostile; rejecting up-front
            // prevents the I/O handlers from `vec![0u8; len]`-ing
            // multi-gigabyte buffers under host control.
            if data_segments.iter().any(|d| d.len > VIRTIO_BLK_SIZE_MAX) {
                tracing::warn!(head, "virtio-blk descriptor exceeds size_max");
                state.counters.record_io_error();
                if publish_completion(
                    mem,
                    q,
                    &state.counters,
                    head,
                    status_addr,
                    VIRTIO_BLK_S_IOERR as u8,
                    1,
                    "size_max reject",
                ) {
                    signal_needed = true;
                }
                continue;
            }

            // Compute total data length (used for both throttle
            // accounting and the `add_used` length).
            let data_len: u64 = data_segments.iter().map(|d| d.len as u64).sum();

            // Zero-data T_IN/T_OUT/T_GET_ID must IOERR. virtio-v1.2
            // §5.2.6 defines IN/OUT as carrying a non-empty data
            // payload; §5.2.6.4 defines GET_ID as writing a 20-byte
            // string into a device-writable data segment — a chain
            // with only header + status has no destination buffer.
            // cloud-hypervisor explicitly rejects this for
            // IN/OUT; firecracker rejects sub-20-byte GET_ID via the
            // handler's `data_len < VIRTIO_BLK_ID_BYTES` arm. We
            // hoist the empty case here so the throttle bucket is
            // never charged for a request the handler will reject
            // anyway. T_FLUSH is exempt — flush carries no data by
            // design (kernel `virtblk_setup_cmd` sets
            // `vbr->in_hdr_len = sizeof(status)` for flushes).
            if matches!(
                req_type,
                VIRTIO_BLK_T_IN | VIRTIO_BLK_T_OUT | VIRTIO_BLK_T_GET_ID
            ) && data_segments.is_empty()
            {
                tracing::warn!(
                    head,
                    req_type,
                    "virtio-blk T_IN/T_OUT/T_GET_ID with no data segments"
                );
                state.counters.record_io_error();
                if publish_completion(
                    mem,
                    q,
                    &state.counters,
                    head,
                    status_addr,
                    VIRTIO_BLK_S_IOERR as u8,
                    1,
                    "zero-data",
                ) {
                    signal_needed = true;
                }
                continue;
            }

            // Sector-granular transfer requirement. virtio-v1.2
            // §5.2.6 defines T_IN/T_OUT in terms of sector-aligned
            // transfers; a sub-sector data length is malformed.
            // firecracker rejects this in
            // src/vmm/src/devices/virtio/block/virtio/request.rs
            // (Request::parse). A buggy or malicious guest that
            // submits e.g. 513 bytes would otherwise reach
            // handle_read_impl/handle_write_impl, which compute
            // offsets in 512-byte units but transfer arbitrary
            // byte counts — the resulting access straddles a
            // sector boundary in a way the host filesystem and
            // backing-file accounting do not expect. Reject up
            // front so the throttle bucket is never charged.
            if matches!(req_type, VIRTIO_BLK_T_IN | VIRTIO_BLK_T_OUT)
                && !data_len.is_multiple_of(VIRTIO_BLK_SECTOR_SIZE as u64)
            {
                tracing::warn!(
                    head,
                    req_type,
                    data_len,
                    "virtio-blk T_IN/T_OUT data_len not a multiple of 512"
                );
                state.counters.record_io_error();
                if publish_completion(
                    mem,
                    q,
                    &state.counters,
                    head,
                    status_addr,
                    VIRTIO_BLK_S_IOERR as u8,
                    1,
                    "sub-sector",
                ) {
                    signal_needed = true;
                }
                continue;
            }

            // Pre-throttle terminal classifications: read-only-mode
            // writes, no-op read-only-mode flushes, and unsupported
            // request types are decided BEFORE consuming throttle
            // tokens. Burning IOPS/bytes budget on a request the
            // device is going to reject anyway is a correctness
            // hazard for tight throttle limits — the guest sees
            // intermittent IOERR on legitimate retries because the
            // bucket was drained by a request that never had a
            // chance to succeed.
            //
            // read_only is checked against the host-owned
            // `self.read_only` field, NOT against re-read guest
            // memory. The header was read once into `hdr` above and
            // not consulted again — no TOCTOU.
            let backing = &state.backing;
            let counters = state.counters.as_ref();
            let cap_bytes = state.capacity_bytes;
            let read_only = state.read_only;
            let pre_throttle = VirtioBlk::classify_pre_throttle(req_type, read_only, counters);

            // Direction validation, hoisted out of
            // handle_read_impl/handle_write_impl/handle_get_id_impl
            // so it runs BEFORE the throttle bucket is consumed.
            // virtio-v1.2 §5.2.6: T_IN data segments must be
            // device-writable (is_write_only); T_OUT data segments
            // must be device-readable (!is_write_only). T_GET_ID
            // (§5.2.6.4) writes a 20-byte string into a
            // device-writable data segment, matching T_IN's
            // direction (cloud-hypervisor and firecracker both
            // reject non-write-only data segments for GET_ID). A
            // request whose data SG direction violates the spec is
            // rejected unconditionally — running it would either
            // read host data into a guest-readable-only buffer
            // (T_IN/T_GET_ID) or write guest-writable buffers to
            // the backing file (T_OUT), neither of which the
            // kernel driver expects. Pre-throttle classifications
            // skip this — RO writes and unsupported requests are
            // already terminal and never dispatch. The redundant
            // per-segment check remains in
            // handle_read_impl/handle_write_impl as
            // defense-in-depth in case a future caller bypasses
            // this gate.
            let direction_violation = pre_throttle.is_none()
                && match req_type {
                    VIRTIO_BLK_T_IN | VIRTIO_BLK_T_GET_ID => {
                        data_segments.iter().any(|d| !d.is_write_only)
                    }
                    VIRTIO_BLK_T_OUT => data_segments.iter().any(|d| d.is_write_only),
                    _ => false,
                };
            if direction_violation {
                tracing::warn!(
                    head,
                    req_type,
                    "virtio-blk T_IN/T_OUT/T_GET_ID data segment direction mismatch"
                );
                state.counters.record_io_error();
                if publish_completion(
                    mem,
                    q,
                    &state.counters,
                    head,
                    status_addr,
                    VIRTIO_BLK_S_IOERR as u8,
                    1,
                    "direction",
                ) {
                    signal_needed = true;
                }
                continue;
            }

            // Throttle: consume 1 op + data_len bytes. If either
            // bucket fails, undo the pop with
            // `set_next_avail(prev.wrapping_sub(1))`, bump
            // `throttled_count`, compute a `wait_nanos` from the
            // bucket's refill rate, and return
            // `DrainOutcome::ThrottleStalled`. The chain stays
            // invisible to the guest (no add_used, no status byte,
            // no irqfd, no `io_errors` bump) until the worker's
            // retry timer fires (`THROTTLE_TOKEN`). The bucket
            // never sleeps — `can_consume` always returns
            // promptly, so the worker stays responsive to
            // STOP_TOKEN and KICK_TOKEN. virtio-spec doesn't
            // reserve a "throttled" status code; deferring the
            // chain is preferable to surfacing transient errors
            // to the guest (which would otherwise see spurious
            // S_IOERRs that confuse the guest's filesystem or
            // application retry semantics).
            //
            // Both buckets are checked first via `can_consume` and
            // only consumed once both pass. Short-circuiting on
            // `consume()` would burn the ops token whenever the
            // bytes check failed (or vice versa), depending on
            // operand order — losing budget to a request that
            // never serviced.
            //
            // FLUSH counts against IOPS, but only when FLUSH
            // actually dispatches to the backend. RO-mode flushes
            // are pre-classified above and never reach here, so
            // they don't touch the bucket.
            if pre_throttle.is_none() {
                let ops_ok = state.ops_bucket.can_consume(1);
                let bytes_ok = state.bytes_bucket.can_consume(data_len);
                if !ops_ok || !bytes_ok {
                    // Throttle exhausted: undo the pop and stall the
                    // drain. The chain stays invisible to the guest
                    // (no add_used, no S_IOERR, no irqfd) until the
                    // worker's retry timer fires and re-drains. The
                    // `wait_nanos` value covers both buckets — pick
                    // the longer of the two waits because both must
                    // hold enough tokens before the request can run.
                    // `set_next_avail(prev - 1)` rewinds the queue's
                    // tracking cursor by one, so the next pop returns
                    // this same chain head — preserving FIFO order
                    // across the stall.  We use this instead of
                    // `go_to_previous_position` because that helper
                    // is on `QueueOwnedT`, which `QueueSync` does not
                    // implement; `set_next_avail` is on the base
                    // `QueueT` and works for both alias targets.
                    // `wrapping_sub` matches the queue's u16 wrap
                    // semantics (next_avail wraps modulo 2^16, the
                    // virtio ring counter width).
                    state.counters.record_throttled();
                    // Live gauge: only increment on the
                    // false → true transition. Re-stalls on the
                    // same head (currently_stalled already true)
                    // bump throttled_count (events) but do NOT
                    // double-bump the gauge. See the
                    // BlkWorkerState::currently_stalled doc for
                    // the transition table.
                    if !state.currently_stalled {
                        state.currently_stalled = true;
                        state.counters.record_throttle_pending_inc();
                    }
                    let prev = queues[REQ_QUEUE].next_avail();
                    queues[REQ_QUEUE].set_next_avail(prev.wrapping_sub(1));
                    let ops_wait = if !ops_ok {
                        state.ops_bucket.nanos_until_n_tokens(1)
                    } else {
                        0
                    };
                    let bytes_wait = if !bytes_ok {
                        state.bytes_bucket.nanos_until_n_tokens(data_len)
                    } else {
                        0
                    };
                    let wait_nanos = ops_wait.max(bytes_wait);
                    tracing::trace!(
                        head,
                        ops_ok,
                        bytes_ok,
                        wait_nanos,
                        "virtio-blk throttle stall; rolling back chain"
                    );
                    stall_outcome = Some(wait_nanos);
                    break;
                }
                // Both checks passed — consume now. Each bucket's
                // `consume` does its own refill+capacity check, so
                // the post-can_consume window can't see a smaller
                // bucket here (refills are monotone-non-negative).
                let ops_consumed = state.ops_bucket.consume(1);
                let bytes_consumed = state.bytes_bucket.consume(data_len);
                debug_assert!(
                    ops_consumed && bytes_consumed,
                    "throttle invariant: can_consume must imply consume",
                );
                // Live gauge: if a prior stall left the gauge
                // incremented, the chain that just satisfied the
                // throttle gate is the head-of-queue stalled
                // chain. Decrement the gauge once the tokens have
                // been consumed — from the throttle-pending
                // perspective, the chain has exited the "waiting
                // for tokens" state. Decrement BEFORE dispatch so
                // a backing-file IO error in the handler doesn't
                // leave the gauge pinned (success/IO-error
                // outcomes are accounted separately, downstream).
                if state.currently_stalled {
                    state.currently_stalled = false;
                    state.counters.record_throttle_pending_dec();
                }
            }

            // Service the request. Handlers compute the status
            // byte + used_len but do NOT write the status byte
            // themselves; this loop performs the status write +
            // add_used as a single "publish completion" step so
            // that a failed status write skips add_used.
            let (status_byte, used_len) = if let Some(out) = pre_throttle {
                out
            } else {
                // Production T_IN / T_OUT now route through the
                // vectored helpers (`handle_read_vectored_impl` /
                // `handle_write_vectored_impl`) which coalesce the
                // chain's data segments into a single
                // `preadv(2)` / `pwritev(2)` syscall against the
                // backing file. The legacy per-segment helpers
                // (`handle_read_impl` / `handle_write_impl` in
                // `handlers.rs`) remain — the cfg(test) test
                // wrappers `dev.handle_read` / `dev.handle_write`
                // continue to call them directly so the existing
                // chain-level proptest / handler-level test surface
                // is not perturbed by this change.
                //
                // `state.io_buf_scratch` is no longer used on the
                // production path; the vectored helpers write
                // directly into guest memory via `mem.get_slices`
                // host pointers, eliminating the kernel→scratch→
                // guest two-stage memcpy of the legacy path. The
                // scratch field stays on `BlkWorkerState` because
                // it is still consumed by the cfg(test) test
                // wrappers' calls into `handle_read_impl` /
                // `handle_write_impl`. `data_len` is passed
                // already-computed so the helpers don't re-derive
                // it.
                match req_type {
                    VIRTIO_BLK_T_IN => VirtioBlk::handle_read_vectored_impl(
                        backing,
                        cap_bytes,
                        counters,
                        mem,
                        sector,
                        data_segments,
                        data_len,
                    ),
                    VIRTIO_BLK_T_OUT => VirtioBlk::handle_write_vectored_impl(
                        backing,
                        cap_bytes,
                        counters,
                        mem,
                        sector,
                        data_segments,
                        data_len,
                    ),
                    VIRTIO_BLK_T_FLUSH => VirtioBlk::handle_flush_impl(backing, counters),
                    VIRTIO_BLK_T_GET_ID => {
                        VirtioBlk::handle_get_id_impl(counters, mem, data_segments)
                    }
                    // Defense-in-depth fall-through. classify_pre_throttle's
                    // catch-all `_ => Some((VIRTIO_BLK_S_UNSUPP, 1))` arm
                    // means this branch is unreachable today — but a future
                    // patch that adds a new variant to the
                    // `T_IN | T_OUT | T_FLUSH | T_GET_ID => None` arm
                    // without updating this match would otherwise panic the
                    // thread running drain_bracket_impl. Return S_UNSUPP and
                    // bump io_errors so the
                    // regression surfaces as a guest-visible error and a
                    // counter bump rather than a panic that kills the VM.
                    _ => {
                        counters.record_io_error();
                        (VIRTIO_BLK_S_UNSUPP as u8, 1)
                    }
                }
            };
            // Per-request log line. Level is `trace!`, not `debug!`,
            // because the device handles thousands of requests
            // per second under load — emitting at debug! would
            // drown out everything else in the default
            // RUST_LOG=info,ktstr=debug operator setting. Anomaly
            // events (rejected request, IOERR) log at `warn!` so
            // they always surface; throttle stalls log at `trace!`
            // (see "throttle stall; rolling back chain" above)
            // because they are deferred-not-failed and would flood
            // logs on a tight throttle. This per-request line is
            // the "happy path" record. The failure-path warns
            // above use the same field set (head, sector, etc.)
            // so log-grep correlation works.
            //
            // Map `req_type` to a human-readable string (rather
            // than the bare u32). The numeric value is preserved
            // as `req_type_raw` for cases where an unknown variant
            // slipped past `classify_pre_throttle` and the
            // operator wants the wire value.
            let req_type_name = match req_type {
                VIRTIO_BLK_T_IN => "in",
                VIRTIO_BLK_T_OUT => "out",
                VIRTIO_BLK_T_FLUSH => "flush",
                VIRTIO_BLK_T_GET_ID => "get_id",
                _ => "unsupp",
            };
            tracing::trace!(
                req_type = req_type_name,
                req_type_raw = req_type,
                sector,
                head,
                status = status_byte,
                used_len,
                "virtio-blk request done"
            );
            // Write status, then add_used ONLY if the status write
            // succeeded. `Queue::add_used` writes the descriptor
            // head/len via write_obj, then publishes used.idx with
            // Ordering::Release, so the prior status-byte
            // write_slice is ordered before the guest sees the new
            // index. The chain has already been dropped (the for
            // loop above consumed it), so this `q` re-borrow is
            // legal.
            //
            // `used_len` from the handlers measures bytes the device
            // wrote into guest memory (data + 1 status byte for
            // reads; 1 status byte for writes/flushes). When the
            // status descriptor is multi-byte we still report only
            // the bytes we wrote, not the descriptor's full length.
            if publish_completion(
                mem,
                q,
                &state.counters,
                head,
                status_addr,
                status_byte,
                used_len,
                "publish completion",
            ) {
                signal_needed = true;
            }
        }
        // Throttle stall: the inner loop's `break` (without
        // continue) ran because of `stall_outcome = Some(_)`.
        // Re-enable notifications so the guest can wake the
        // device when it adds new chains, then break the outer
        // loop. Bail unconditionally on stall to keep the path
        // simple; the worker's retry timer drives the
        // re-attempt regardless of whether the bucket happens
        // to have refilled by then.
        if stall_outcome.is_some() {
            if let Err(e) = queues[REQ_QUEUE].enable_notification(mem) {
                tracing::warn!(
                    %e,
                    "virtio-blk enable_notification failed on throttle stall"
                );
            }
            break 'outer;
        }
        // Inner drain ran to None. Re-arm notifications and
        // check whether new chains arrived during the disabled
        // window. `enable_notification` returns Ok(true) when
        // `avail_idx != next_avail` after re-enabling — those
        // chains MUST be processed before exiting or they'll
        // be stranded (V3: honour the return value).
        match queues[REQ_QUEUE].enable_notification(mem) {
            Ok(true) => continue 'outer,
            Ok(false) => break 'outer,
            Err(e) => {
                // A persistent enable failure (e.g. used-ring
                // GPA unmapped) would otherwise spin the outer
                // loop forever. Bail to avoid a livelock; on
                // the next QUEUE_NOTIFY the guest may have
                // recovered guest memory layout.
                tracing::warn!(%e, "virtio-blk enable_notification failed");
                break 'outer;
            }
        }
    }
    if signal_needed {
        // V8: always set the interrupt_status MMIO bit when
        // anything was published. The bit-set on `interrupt_status`
        // is the IRQ-handler handshake target — `vm_interrupt`
        // (drivers/virtio/virtio_mmio.c) reads and acks it on each
        // IRQ delivery. The guest does NOT poll this register; it
        // only consults it from inside the IRQ handler. If the irqfd
        // write fails, the guest never enters `vm_interrupt` and the
        // queue stalls — the bit remaining set is harmless on its
        // own; only IRQ delivery makes the guest read it.
        // Release-ordered fetch_or so the bit-set happens-after
        // the chain's add_used publish. The SeqCst fence inside
        // needs_notification then orders all prior writes
        // (including add_used and this bit-set) against the
        // used_event read that drives the IRQ decision. Result:
        // a vCPU reading INTERRUPT_STATUS via Acquire-load and
        // finding INT_VRING set is guaranteed to also observe
        // the freshly-published used.idx — no torn observation
        // where the bit appears before the ring update.
        interrupt_status.fetch_or(VIRTIO_MMIO_INT_VRING, Ordering::Release);
        // `Queue::needs_notification` consults the guest's
        // `used_event` threshold (from the avail ring) when
        // EVENT_IDX is negotiated — returns false if the guest
        // hasn't asked to be woken yet, true otherwise. In the
        // legacy path (event_idx_enabled=false) it always
        // returns Ok(true) (the trailing `Ok(true)` arm of
        // `Queue::needs_notification`), so the eventfd fires
        // every time as before.
        //
        // V6: only call `needs_notification` on the
        // signal_needed=true path. The method has side effects
        // (resets `num_added` to zero — see the doc comment on
        // `QueueT::needs_notification`) so calling it
        // speculatively would corrupt the suppression state.
        //
        // unwrap_or(true): on guest-memory errors reading the
        // `used_event` field, fail-safe to firing the IRQ. A
        // missed IRQ stalls the guest until the hung-task
        // watchdog fires (`kernel.hung_task_timeout_secs`,
        // default 120 s — virtio_blk has no `mq_ops->timeout`
        // so blk-mq alone never surfaces the stall); a
        // redundant IRQ wastes a vCPU exit.
        let q = &mut queues[REQ_QUEUE];
        if q.needs_notification(mem)
            .inspect_err(
                |e| tracing::warn!(%e, "needs_notification failed; firing IRQ as fail-safe"),
            )
            .unwrap_or(true)
        {
            // SAFETY: EAGAIN requires counter saturation at u64::MAX-1
            // (~1.8e19 unobserved kicks) — implausible. EBADF means
            // the fd closed during shutdown. The simultaneously-set
            // INT_VRING bit at the `interrupt_status.fetch_or` above
            // is the next IRQ handler's read target — but only if a
            // SUBSEQUENT request fires `irq_evt.write` successfully.
            // For the last chain in a burst with no follow-on
            // traffic, a missed write means the queue stalls until
            // hung_task_timeout (default 120s; virtio_blk has no
            // `mq_ops->timeout`). The recovery path here is the
            // next add_used + needs_notification cycle (the next
            // request's publish reaches this site again), NOT a
            // kernel timer (virtio_mmio has no periodic wake
            // mechanism). We log any errno so a failed write
            // surfaces in tracing rather than silently disappearing.
            if let Err(e) = irq_evt.write(1) {
                tracing::warn!(%e, "virtio-blk irq_evt.write failed");
            }
        }
    }
    match stall_outcome {
        Some(wait_nanos) => DrainOutcome::ThrottleStalled { wait_nanos },
        None => DrainOutcome::Done,
    }
}
