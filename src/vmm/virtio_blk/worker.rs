//! Production worker-thread loop for virtio-blk.
//!
//! Owns the epoll-driven main loop, the kick/stop/throttle dispatch
//! tokens, the stall-decision policy, and the retry-timer clamp.
//! Extracted from `mod.rs` for module locality so the worker code
//! lives next to its tests rather than scattered through the MMIO
//! and FSM code in the parent module.
//!
//! # Public surface (within `super`)
//!
//! - [`worker_thread_main`] — the production worker entry point.
//!   Spawned by `VirtioBlk::with_options` and `respawn_worker`;
//!   joined by `Drop` and `reset_engine_spawned` via the
//!   `JoinHandle<BlkWorkerState>` payload.
//! - [`StallAction`] / [`decide_stall_action`] — pure mapping from
//!   `DrainOutcome` to next-step action; tested in
//!   `super::tests` without spawning a worker.
//! - [`WorkerDispatchAction`] / [`worker_dispatch_event`] — pure
//!   mapping from `(EventSet, token)` to dispatch decision; tested
//!   in `super::tests` without constructing an `Epoll`.
//! - [`clamp_retry_nanos`], [`RETRY_TIMER_MAX_NANOS`] — retry-timer
//!   bounds; pinned by `super::tests`.
//! - Token discriminators: [`KICK_TOKEN`], [`STOP_TOKEN`],
//!   [`THROTTLE_TOKEN`] — also referenced by
//!   `super::tests::worker_dispatch_event_*` cases.
//!
//! # Threading model
//!
//! `worker_thread_main` runs on a dedicated thread spawned by
//! `VirtioBlk` (production cfg only). The vCPU's MMIO QUEUE_NOTIFY
//! handler kicks an eventfd; the worker's `epoll_wait` resumes and
//! runs `drain_bracket_impl` (defined in the parent `mod.rs`) to
//! pop chains, walk descriptors, and publish completions. See the
//! parent module's "Execution model" doc for the full
//! vCPU/worker split.
//!
//! `cfg(test)` builds use the inline engine (no worker thread —
//! `process_requests` calls `drain_bracket_impl` directly on the
//! caller thread) so the worker code in this file is gated on
//! `cfg(not(test))`. The pure helpers (`decide_stall_action`,
//! `worker_dispatch_event`, `clamp_retry_nanos`) and their tokens
//! are always-compiled so the test block in the parent module can
//! exercise them without the worker thread.

#[cfg(not(test))]
use std::os::unix::io::AsRawFd;
#[cfg(not(test))]
use std::sync::Arc;
#[cfg(not(test))]
use std::sync::OnceLock;
#[cfg(not(test))]
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

#[cfg(not(test))]
use vm_memory::GuestMemoryMmap;
use vmm_sys_util::epoll::EventSet;
#[cfg(not(test))]
use vmm_sys_util::epoll::{ControlOperation, Epoll, EpollEvent};
#[cfg(not(test))]
use vmm_sys_util::eventfd::EventFd;

#[cfg(not(test))]
use super::{BlkQueue, BlkWorkerState, NUM_QUEUES, drain_bracket_impl};
use super::DrainOutcome;

/// Cap the throttle retry timer so a pathological refill rate (e.g.
/// tens of millions of bytes per second backing a single very large
/// request) cannot stretch a stall arbitrarily far into the future.
/// 1 s gives plenty of headroom; the bucket simply re-stalls if the
/// retry is premature (cheap — re-pop, recompute deficit, re-arm).
/// Each stall-retry cycle is bounded at 1 s; a request under sustained
/// throttle may re-stall multiple times until the bucket accumulates
/// enough tokens. The cap is per-stall, not per-request — total
/// per-request latency is `n * 1 s` worst case, where `n` is the
/// number of re-stall cycles before the bucket holds enough tokens.
///
/// Requests larger than bucket capacity are handled via the
/// `TokenBucket` overconsume policy (see the type-level
/// "Overconsumption" doc): the first oversized request grants
/// immediately by driving `available` negative, and followers wait
/// proportional to the accumulated debt. The retry-timer cap still
/// applies — followers stalled behind the debt re-arm at every 1 s
/// boundary until the debt clears, with finite (not unbounded)
/// total wait.
///
/// A too-long cap is the dangerous direction: it risks tripping the
/// guest's hung-task watchdog (`kernel.hung_task_timeout_secs`,
/// default 120 s), since virtio_blk has no `mq_ops->timeout`
/// callback (drivers/block/virtio_blk.c `virtio_mq_ops` has no
/// `.timeout` field) and an unpublished request never surfaces an
/// error to the guest's block layer on its own.
pub(super) const RETRY_TIMER_MAX_NANOS: u64 = 1_000_000_000;

/// Clamp a `wait_nanos` value (from `DrainOutcome::ThrottleStalled`)
/// into the legal `timerfd_settime` range used by the worker.
///
/// `timerfd_settime(2)` treats an `it_value` of 0 as "disarm" rather
/// than "fire immediately," so a `wait_nanos == 0` outcome (the
/// bucket already refilled between the `can_consume` check and the
/// nanosecond computation) is mapped to the smallest non-zero value
/// (`1` ns) — this re-arms the timer for an immediate wake instead
/// of disarming it. The upper bound `RETRY_TIMER_MAX_NANOS` keeps a
/// pathological refill rate from pushing the retry past the guest's
/// hung-task watchdog (`kernel.hung_task_timeout_secs`, default
/// 120 s — virtio_blk has no `mq_ops->timeout`, so an unpublished
/// request never surfaces an error to the guest's block layer on
/// its own; matches the rationale on `RETRY_TIMER_MAX_NANOS` above).
/// Free function (not method) so tests can pin both boundaries
/// without constructing a worker.
pub(super) fn clamp_retry_nanos(wait_nanos: u64) -> u64 {
    wait_nanos.clamp(1, RETRY_TIMER_MAX_NANOS)
}

/// Decision the worker loop takes after a `drain_bracket_impl` call.
/// Pure mapping from `DrainOutcome` to "what side effect runs next" —
/// no IO, no fd ops, no state mutation. The worker loop owns the
/// side effects (timerfd_settime, second drain call,
/// `last_known_blocked` flag flip); this function just decides which
/// one runs.
///
/// `Continue` — drain reached `Done`. The worker should clear
/// `last_known_blocked` (so subsequent KICK_TOKENs aren't suppressed)
/// and resume the `epoll_wait` loop without arming a timer.
///
/// `ReDrain` — drain returned `ThrottleStalled { wait_nanos: 0 }`.
/// The bucket already refilled between `can_consume` and the deficit
/// computation; arming the timerfd would round-trip through epoll for
/// no reason. The worker re-calls `drain_bracket_impl` once
/// (bounded recursion — see the worker loop) and then takes the
/// resulting action. If the second drain ALSO produces `ReDrain`,
/// the worker downgrades to `Sleep { nanos: 1 }` so the loop can't
/// spin: a stall→retry→stall→retry pattern would otherwise starve
/// STOP_TOKEN/KICK_TOKEN.
///
/// `Sleep { nanos }` — drain returned `ThrottleStalled` with a
/// non-zero deficit. `nanos` is already passed through
/// `clamp_retry_nanos`: floored at 1 (so `timerfd_settime` doesn't
/// disarm the timer with `it_value = 0`) and capped at
/// `RETRY_TIMER_MAX_NANOS` (1 s, well under
/// `kernel.hung_task_timeout_secs` default of 120 s — virtio_blk has
/// no `mq_ops->timeout`, so an unpublished request only surfaces to
/// the guest's block layer when the watchdog fires or a higher layer
/// retries). The worker arms the timerfd with this value and sets
/// `last_known_blocked` so subsequent KICK_TOKENs are suppressed
/// until THROTTLE_TOKEN clears the flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StallAction {
    Continue,
    ReDrain,
    Sleep { nanos: u64 },
}

/// Map a `DrainOutcome` to the worker loop's next action. Free fn
/// (not method) so cfg(test) unit tests can drive every variant
/// without spawning a worker thread or constructing an `Epoll`.
///
/// The mapping is the single source of truth for the worker's
/// stall-decision policy:
///
/// - `Done` → `Continue` (drain emptied the queue; no retry needed).
/// - `ThrottleStalled { wait_nanos: 0 }` → `ReDrain` (refill arrived
///   between `can_consume` failure and the deficit computation;
///   a synchronous re-drain is cheaper than a timerfd round-trip).
/// - `ThrottleStalled { wait_nanos: n > 0 }` →
///   `Sleep { nanos: clamp_retry_nanos(n) }` (arm the retry timerfd).
pub(super) fn decide_stall_action(outcome: DrainOutcome) -> StallAction {
    match outcome {
        DrainOutcome::Done => StallAction::Continue,
        DrainOutcome::ThrottleStalled { wait_nanos: 0 } => StallAction::ReDrain,
        DrainOutcome::ThrottleStalled { wait_nanos } => StallAction::Sleep {
            nanos: clamp_retry_nanos(wait_nanos),
        },
    }
}

/// Worker-thread epoll dispatch tokens. Hoisted to module scope
/// so the testable `worker_dispatch_event` helper (and its unit
/// tests under `cfg(test)`) can name them without duplicating
/// the values inside `worker_thread_main`'s frame.
pub(super) const KICK_TOKEN: u64 = 1;
pub(super) const STOP_TOKEN: u64 = 2;
pub(super) const THROTTLE_TOKEN: u64 = 3;

/// Outcome of one epoll-event dispatch decision. Lifted to a
/// dedicated enum so `worker_thread_main` and its unit tests
/// share the same vocabulary for "what should the loop do next?"
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(super) enum WorkerDispatchAction {
    /// STOP_TOKEN observed — return state and exit the worker loop.
    Stop,
    /// Run one drain iteration. `throttle_token_fired` is true when
    /// THROTTLE_TOKEN was the cause: forces the drain past the
    /// `last_known_blocked` skip because the bucket refill timer
    /// just expired and the rolled-back chain may now be
    /// satisfiable.
    Drain { throttle_token_fired: bool },
    /// Unknown token — log and continue without draining.
    Skip,
}

/// Decide what one `EpollEvent` from `epoll.wait` should make the
/// worker loop do. Free fn (not method) so cfg(test) unit tests
/// can drive every (event_set, token) combination without
/// spawning a worker thread or constructing an `Epoll` instance.
///
/// EPOLLERR / EPOLLHUP semantics for the worker's three fd types
/// (kernel-grounded, verified against fs/eventfd.c::eventfd_poll
/// and fs/timerfd.c::timerfd_poll):
///
/// * eventfd EPOLLERR fires iff `count == ULLONG_MAX`. With
///   `EFD_NONBLOCK` the vCPU `kick_fd.write(1)` returns EAGAIN
///   on saturation rather than blocking, so the counter saturates
///   at `ULLONG_MAX - 1`; reaching `ULLONG_MAX` requires an
///   internal kernel write the worker doesn't issue. Treated as
///   defensive — log and fall through to the per-token handler
///   so the next eventfd `read` drains the saturated counter
///   back to 0 (eventfd's read returns the count and resets it).
///
/// * eventfd EPOLLHUP never fires. `eventfd_poll` only sets
///   `EPOLLIN` (count > 0), `EPOLLOUT` (count < ULLONG_MAX-1),
///   and `EPOLLERR` (count == ULLONG_MAX). No code path returns
///   EPOLLHUP. Observing it on our owned eventfd indicates a
///   kernel-contract change or an Epoll registration bug — log
///   and fall through.
///
/// * timerfd EPOLLERR / EPOLLHUP never fire. `timerfd_poll`
///   only sets `EPOLLIN` (ticks > 0). Same defensive log + fall
///   through.
///
/// In all anomaly cases the per-token handler's eventfd or
/// timerfd `read` is the recovery action. For the production
/// fd types the read is harmless (yields EAGAIN if no data) and
/// for an eventfd at saturation it's curative (drains the
/// counter and clears EPOLLERR). The free fn makes the dispatch
/// decision; the worker loop performs the side effects (read,
/// drain, mutate state).
pub(super) fn worker_dispatch_event(event_set: EventSet, token: u64) -> WorkerDispatchAction {
    if event_set.contains(EventSet::ERROR) {
        tracing::warn!(
            ?event_set,
            token,
            "virtio-blk worker: epoll event_set contains EPOLLERR; \
             expected only on eventfd counter saturation \
             (count == ULLONG_MAX) — fall through to per-token \
             handler so the eventfd read drains the saturated \
             counter back to 0"
        );
    }
    if event_set.contains(EventSet::HANG_UP) {
        tracing::warn!(
            ?event_set,
            token,
            "virtio-blk worker: epoll event_set contains EPOLLHUP; \
             structurally impossible for eventfd/timerfd \
             (eventfd_poll and timerfd_poll never set POLLHUP). \
             Indicates a kernel-contract change or Epoll \
             registration bug — log and fall through"
        );
    }
    match token {
        STOP_TOKEN => WorkerDispatchAction::Stop,
        KICK_TOKEN => WorkerDispatchAction::Drain {
            throttle_token_fired: false,
        },
        THROTTLE_TOKEN => WorkerDispatchAction::Drain {
            throttle_token_fired: true,
        },
        _ => {
            tracing::warn!(
                ?event_set,
                token,
                "virtio-blk worker: unknown epoll token"
            );
            WorkerDispatchAction::Skip
        }
    }
}

/// Worker thread main loop (production cfg only). Owns
/// `BlkWorkerState`, the `[QueueSync; NUM_QUEUES]` clones, and Arcs
/// for the shared atomics + mem slot for the device's lifetime. Loops
/// in epoll_wait until the device's `Drop::drop` writes to `stop_fd`,
/// at which point the loop exits and the thread terminates.
///
/// Per-iteration:
///   1. Block in `epoll_wait` until kick_fd, stop_fd, or the
///      throttle retry timerfd is readable.
///   2. If stop_fd readable → return (drop everything cleanly).
///   3. If timer_fd readable (THROTTLE_TOKEN) → consume the expiry
///      count (counter-mode timerfd) and fall through to drain so
///      the rolled-back chain can re-pop now that the bucket has
///      refilled.
///   4. If kick_fd readable (KICK_TOKEN) → drain the eventfd
///      counter (one read consumes any number of coalesced kicks
///      per the eventfd counter-mode semantics — see eventfd(2))
///      and run one drain iteration via `drain_bracket_impl`.
///
/// Reading mem from the shared `Arc<OnceLock<…>>` gives the worker
/// the `GuestMemoryMmap` set by the device's `set_mem` call via a
/// lock-free `OnceLock::get`. When `mem` is unset (kick fired before
/// set_mem — a wiring bug), the `mem_unset_warned` latch fires once
/// and the kick is silently dropped.
#[cfg(not(test))]
pub(super) fn worker_thread_main(
    mut state: BlkWorkerState,
    mut queues: [BlkQueue; NUM_QUEUES],
    mem: Arc<OnceLock<GuestMemoryMmap>>,
    irq_evt: Arc<EventFd>,
    interrupt_status: Arc<AtomicU32>,
    mem_unset_warned: Arc<AtomicBool>,
    kick_fd: EventFd,
    stop_fd: EventFd,
) -> BlkWorkerState {
    // epoll setup. KICK_TOKEN, STOP_TOKEN, THROTTLE_TOKEN are the
    // `EpollEvent::data` discriminators we'll match on after
    // `epoll_wait` returns. Using opaque 64-bit tokens (rather than
    // the raw fd numbers, which would also work) makes the dispatch
    // intent explicit at the read site.
    //
    // The function returns `state` on STOP_TOKEN (and on every
    // early-exit error path) so `VirtioBlk::reset()` can join the
    // worker, recover the underlying `BlkWorkerState`, reset its
    // throttle buckets, and respawn a fresh worker against the
    // post-`q.reset()` queue without having to reconstruct the
    // backing-file handle, scratch vectors, or counters Arc. Drop
    // discards the returned state with `let _ = handle.join()`.
    let epoll = match Epoll::new() {
        Ok(e) => e,
        Err(e) => {
            tracing::error!(%e, "virtio-blk worker: failed to create epoll instance; \
                exiting (device IO will not be serviced)");
            return state;
        }
    };
    if let Err(e) = epoll.ctl(
        ControlOperation::Add,
        kick_fd.as_raw_fd(),
        EpollEvent::new(EventSet::IN, KICK_TOKEN),
    ) {
        tracing::error!(%e, "virtio-blk worker: failed to add kick_fd to epoll; exiting");
        return state;
    }
    if let Err(e) = epoll.ctl(
        ControlOperation::Add,
        stop_fd.as_raw_fd(),
        EpollEvent::new(EventSet::IN, STOP_TOKEN),
    ) {
        tracing::error!(%e, "virtio-blk worker: failed to add stop_fd to epoll; exiting");
        return state;
    }
    // Throttle retry timerfd. CLOCK_MONOTONIC matches `Instant`'s
    // clock domain so the duration we compute from
    // `nanos_until_n_tokens` (in `Instant::now()` terms) stays
    // consistent with the timer's expiry. TFD_NONBLOCK so a stale
    // expiry read after the worker drains naturally does not stall
    // the loop. TFD_CLOEXEC inherited from libc::TFD_CLOEXEC so the
    // fd is not leaked across exec().
    //
    // SAFETY: `timerfd_create` is a normal Linux syscall whose
    // contract is "return >= 0 fd on success, -1 on failure." We
    // check the return for negative and call `from_raw_fd` only on
    // success, transferring ownership to a `File` so the kernel's
    // close-on-drop runs.
    let timer_fd_raw = unsafe {
        libc::timerfd_create(
            libc::CLOCK_MONOTONIC,
            libc::TFD_NONBLOCK | libc::TFD_CLOEXEC,
        )
    };
    if timer_fd_raw < 0 {
        tracing::error!(
            err = std::io::Error::last_os_error().to_string(),
            "virtio-blk worker: timerfd_create failed; exiting"
        );
        return state;
    }
    // SAFETY: `timer_fd_raw` is the just-created timerfd, owned by
    // this thread; wrapping in `File` transfers the close
    // responsibility on Drop.
    let timer_fd: std::fs::File = unsafe { std::os::unix::io::FromRawFd::from_raw_fd(timer_fd_raw) };
    if let Err(e) = epoll.ctl(
        ControlOperation::Add,
        timer_fd_raw,
        EpollEvent::new(EventSet::IN, THROTTLE_TOKEN),
    ) {
        tracing::error!(%e, "virtio-blk worker: failed to add timer_fd to epoll; exiting");
        return state;
    }

    // Three-element scratch — kick_fd, stop_fd, timer_fd are the
    // only fds registered, so `epoll_wait` can return at most 3
    // events per call.
    let mut events = [EpollEvent::default(); 3];
    // Worker-local "we know the throttle is blocked" flag.
    // Set when `drain_bracket_impl` returns ThrottleStalled and
    // we arm the retry timerfd; cleared when THROTTLE_TOKEN
    // fires (timer expired → tokens should be available now).
    //
    // While this flag is set, the worker skips
    // `drain_bracket_impl` calls triggered by KICK_TOKEN alone
    // — the next drain attempt is futile because the
    // head-of-queue chain will still stall on the same throttle
    // exhaustion (FIFO drain semantics rolling back the same
    // chain head). The kick eventfd counter is still drained so
    // it doesn't accumulate; the work happens on the next
    // THROTTLE_TOKEN wakeup.
    //
    // This is a perf optimization, not a correctness change.
    // Without it, every KICK_TOKEN during a stall window
    // re-runs the full pop+walk+validate+rollback pipeline on
    // the head chain, wasting CPU cycles for no progress.
    //
    // Liveness: THROTTLE_TOKEN always clears the flag, so the
    // worker is guaranteed to re-enter the drain loop within
    // `RETRY_TIMER_MAX_NANOS` (1 s) of the stall. The flag
    // never leads to a permanently-blocked worker — the
    // timerfd is the timeout authority.
    let mut last_known_blocked: bool = false;
    loop {
        let n = match epoll.wait(-1, &mut events) {
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => {
                tracing::error!(%e, "virtio-blk worker: epoll_wait failed; exiting");
                return state;
            }
        };
        let mut should_drain = false;
        // Tracks whether THROTTLE_TOKEN was among this batch of
        // events; a timer expiry forces a drain even when
        // `last_known_blocked` is set, because the timer is the
        // signal that the bucket has refilled and the head chain
        // can be retried.
        let mut throttle_token_fired = false;
        for ev in &events[..n] {
            // Dispatch via the testable free fn so cfg(test) unit
            // tests cover every (event_set, token) combination —
            // including EPOLLERR/EPOLLHUP defensive arms — without
            // spawning a worker thread.
            match worker_dispatch_event(ev.event_set(), ev.data()) {
                WorkerDispatchAction::Stop => {
                    // Stop signal — exit immediately and yield
                    // `state` back to the caller via the join
                    // handle. Drop discards the returned state
                    // with `let _ = handle.join()`; `reset()`
                    // captures it, rebuilds the throttle buckets,
                    // and re-spawns a fresh worker against the
                    // post-`q.reset()` queue. Either path leaves
                    // the queue Arcs, eventfd clones, and timerfd
                    // owned by this frame to be reclaimed at
                    // return.
                    return state;
                }
                WorkerDispatchAction::Drain {
                    throttle_token_fired: tt_fired,
                } => {
                    should_drain = true;
                    if tt_fired {
                        // Timer fired — bucket should now have
                        // refilled enough to satisfy the
                        // rolled-back chain. Counter-mode
                        // timerfd: a single `read` returns the
                        // expiry count and resets it to zero; we
                        // don't care about the count, just need to
                        // clear the readiness.
                        //
                        // Two expected Err variants are non-fatal:
                        //   * EAGAIN (WouldBlock) —
                        //     `timerfd_settime` cleared the expiry
                        //     counter between `epoll_wait` and
                        //     this `read` (e.g. a re-arm from the
                        //     immediately-prior drain raced with
                        //     readiness delivery). Harmless: the
                        //     next THROTTLE_TOKEN wakeup will read
                        //     whatever count is pending then.
                        //   * EINTR (Interrupted) — harmless: the
                        //     timerfd remains readable, and the
                        //     next epoll_wait re-delivers
                        //     THROTTLE_TOKEN.
                        // Anything else is unexpected (e.g. EBADF
                        // on a closed fd) — log it so operators
                        // can debug. In all cases the worker
                        // still falls through to the drain so the
                        // intended semantics ("re-drain because
                        // the refill timer expired") are
                        // preserved regardless of read outcome.
                        let mut buf = [0u8; 8];
                        use std::io::Read;
                        match (&timer_fd).read(&mut buf) {
                            Ok(_) => {}
                            Err(e)
                                if e.kind()
                                    == std::io::ErrorKind::WouldBlock => {}
                            Err(e)
                                if e.kind()
                                    == std::io::ErrorKind::Interrupted => {}
                            Err(e) => {
                                tracing::warn!(
                                    %e,
                                    "virtio-blk worker: unexpected timerfd read error",
                                );
                            }
                        }
                        throttle_token_fired = true;
                        // Clear the cached "blocked" flag now
                        // that the timer has fired. The actual
                        // drain outcome below will re-set it if
                        // the chain still cannot make progress
                        // (e.g. premature refill, request size
                        // larger than capacity).
                        last_known_blocked = false;
                    }
                }
                WorkerDispatchAction::Skip => {
                    // Unknown token already logged by
                    // worker_dispatch_event; nothing to do here.
                }
            }
        }
        if !should_drain {
            continue;
        }
        // Drain the kick eventfd counter (best-effort — only if a
        // kick was the trigger; THROTTLE_TOKEN drains do not bump
        // the kick counter so a leftover read here is fine and
        // simply yields EAGAIN).
        //
        // Counter-mode semantics (eventfd(2)): a single `read`
        // returns the accumulated counter value and resets it to
        // 0. So multiple coalesced kicks (vCPU wrote 1 several
        // times before we woke) all collapse to a single drain —
        // the desired property.
        //
        // This drains the kick counter even when the
        // `last_known_blocked` skip below short-circuits the
        // actual drain — otherwise the counter would accumulate
        // unbounded across the stall window and a saturating
        // `EventFd::write(1)` from `process_requests` would
        // EAGAIN until the worker eventually reads it.
        match kick_fd.read() {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // No pending kick — this iteration was driven by
                // the throttle timer. Fall through to drain.
            }
            Err(e) => {
                tracing::warn!(%e, "virtio-blk worker: kick_fd read failed");
            }
        }

        // is_blocked skip. When we know the throttle is blocked
        // AND the wakeup was a KICK_TOKEN (not a THROTTLE_TOKEN),
        // skip the drain entirely. The next THROTTLE_TOKEN will
        // trigger the retry — drain attempts in between are
        // guaranteed to re-stall on the same head and waste CPU
        // on the rollback. The kick counter has already been
        // drained above so the next vCPU-side `kick_fd.write(1)`
        // finds a clean counter.
        if last_known_blocked && !throttle_token_fired {
            continue;
        }

        // Resolve the current guest memory. If `set_mem` hasn't run
        // yet, latch the warn once and skip the drain. Lock-free
        // `OnceLock::get` returns `Option<&GuestMemoryMmap>`; the
        // borrow lives for the duration of this loop iteration so
        // we can pass it straight to `drain_bracket_impl` without
        // a clone (matching the prior path's intent — the previous
        // `Mutex<Option<…>>` field also yielded a single-iteration
        // borrow, just via a clone).
        let Some(mem_ref) = mem.get() else {
            if !mem_unset_warned.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    "virtio-blk: queue notify before set_mem; \
                     dropping requests until guest memory is wired"
                );
            }
            continue;
        };
        let outcome = drain_bracket_impl(
            &mut state,
            &mut queues,
            mem_ref,
            &irq_evt,
            &interrupt_status,
        );
        // Inline re-drain on wait_nanos == 0. When
        // `nanos_until_n_tokens` returns 0 — bucket already
        // refilled between the `can_consume` failure and the
        // per-bucket nanos computation — the original
        // timerfd-arm path would floor the wait at 1 ns
        // (`clamp_retry_nanos`), disarm-then-rearm the timerfd,
        // wake on THROTTLE_TOKEN, drain the timerfd read, and
        // re-enter the drain. That's an unnecessary epoll
        // round-trip for a state where the next drain is
        // guaranteed to succeed.
        //
        // Re-call `drain_bracket_impl` immediately, bounded at
        // depth 1, so a refill that arrived between
        // can_consume and the arm decision is observed
        // synchronously without giving up the CPU. If the
        // inline retry STILL stalls, fall through to the
        // timerfd path with `Sleep { nanos: 1 }` —
        // bounded recursion prevents a stall→retry→stall spin
        // from starving STOP_TOKEN/KICK_TOKEN. `clamp_retry_nanos`
        // would have produced the same 1 ns floor for the
        // second `wait_nanos == 0` outcome, so the downgrade
        // preserves the original `it_value`.
        let action = match decide_stall_action(outcome) {
            StallAction::ReDrain => {
                tracing::trace!(
                    "virtio-blk worker: wait_nanos==0 inline re-drain"
                );
                let outcome2 = drain_bracket_impl(
                    &mut state,
                    &mut queues,
                    mem_ref,
                    &irq_evt,
                    &interrupt_status,
                );
                match decide_stall_action(outcome2) {
                    StallAction::ReDrain => StallAction::Sleep { nanos: 1 },
                    other => other,
                }
            }
            other => other,
        };
        // Apply the decided action. The `Sleep` arm arms the retry
        // timerfd; `Continue` clears `last_known_blocked` so
        // subsequent kicks aren't suppressed (the gauge dec already
        // fired inside drain_bracket_impl when the throttle gate
        // was satisfied). `decide_stall_action` already passed the
        // raw `wait_nanos` through `clamp_retry_nanos`, so `nanos`
        // is bounded at `[1, RETRY_TIMER_MAX_NANOS]` — `it_value`
        // is never 0 (which would disarm the timer per
        // timerfd_settime(2)) and never exceeds 1 s (well under
        // `kernel.hung_task_timeout_secs` default of 120 s —
        // virtio_blk has no `mq_ops->timeout`).
        match action {
            StallAction::Sleep { nanos } => {
                // Cache the blocked state so the next KICK_TOKEN
                // skips the drain (see is_blocked skip above). The
                // flag is cleared on THROTTLE_TOKEN; if a fresh
                // THROTTLE_TOKEN re-stalls, this branch re-sets it.
                last_known_blocked = true;
                let new_value = libc::itimerspec {
                    it_interval: libc::timespec {
                        tv_sec: 0,
                        tv_nsec: 0,
                    },
                    it_value: libc::timespec {
                        tv_sec: (nanos / 1_000_000_000) as libc::time_t,
                        tv_nsec: (nanos % 1_000_000_000) as libc::c_long,
                    },
                };
                // SAFETY: `timer_fd_raw` is the live timerfd we just
                // created; `new_value` is a valid `itimerspec` with
                // it_interval=0 (one-shot), it_value=`nanos` ns. The
                // null `old_value` is allowed per timerfd_settime(2).
                let rc = unsafe {
                    libc::timerfd_settime(
                        timer_fd_raw,
                        0, // relative timer
                        &new_value as *const _,
                        std::ptr::null_mut(),
                    )
                };
                if rc < 0 {
                    tracing::warn!(
                        err = std::io::Error::last_os_error().to_string(),
                        "virtio-blk worker: timerfd_settime failed; \
                         stalled chain will not auto-retry — guest may \
                         hang on this request until kernel.hung_task_timeout_secs \
                         (default 120s) fires or higher-layer retries"
                    );
                }
            }
            StallAction::Continue => {
                last_known_blocked = false;
            }
            StallAction::ReDrain => {
                // Unreachable — the second-drain match above
                // converts any `ReDrain` outcome from the second
                // pass into `Sleep { nanos: 1 }`, so the worker
                // loop only ever applies `Sleep` or `Continue`.
                // Defensive: treat as `Sleep { nanos: 1 }` to
                // arm the timer and stay live.
                debug_assert!(false, "ReDrain leaked past bounded recursion");
                last_known_blocked = true;
            }
        }
    }
}
