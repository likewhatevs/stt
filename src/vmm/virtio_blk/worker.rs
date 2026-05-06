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

use super::DrainOutcome;
#[cfg(not(test))]
use super::{BlkQueue, BlkWorkerState, NUM_QUEUES, WorkerPlacement, drain_bracket_impl};

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
pub(crate) const RETRY_TIMER_MAX_NANOS: u64 = 1_000_000_000;

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
pub(crate) fn clamp_retry_nanos(wait_nanos: u64) -> u64 {
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
pub(crate) enum StallAction {
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
pub(crate) fn decide_stall_action(outcome: DrainOutcome) -> StallAction {
    match outcome {
        DrainOutcome::Done => StallAction::Continue,
        DrainOutcome::ThrottleStalled { wait_nanos: 0 } => StallAction::ReDrain,
        DrainOutcome::ThrottleStalled { wait_nanos } => StallAction::Sleep {
            nanos: clamp_retry_nanos(wait_nanos),
        },
    }
}

/// Final action the worker loop applies after the inline-re-drain
/// step has resolved. Strictly a subset of [`StallAction`] —
/// `ReDrain` is excluded by construction so the apply site cannot
/// observe an un-handled inline-retry leak.
///
/// Distinction from `StallAction`: the latter is the raw policy
/// output that may include `ReDrain` (the wait_nanos==0 trigger);
/// `WorkerAction` is what the worker loop ACTS on after the
/// bounded-recursion inline retry has converted any leaked
/// `ReDrain` into `Sleep { nanos: 1 }`. Splitting the type makes
/// the `match` at the apply site exhaustive without a defensive
/// arm — a regression that drops the bounded-recursion downgrade
/// would surface as a compile error in `resolve_action` rather
/// than as a runtime `debug_assert!`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WorkerAction {
    Continue,
    Sleep { nanos: u64 },
}

/// Resolve a [`StallAction`] into the apply-site [`WorkerAction`],
/// performing the bounded-recursion inline retry on `ReDrain`.
///
/// `redrain` is the closure the worker invokes when the first
/// outcome was `ReDrain`: it must call `drain_bracket_impl` once
/// and return the resulting [`DrainOutcome`]. If the second drain
/// ALSO produces `ReDrain`, this function downgrades to
/// `Sleep { nanos: 1 }` so the loop never spins (a stall→retry
/// →stall pattern would otherwise starve STOP_TOKEN/KICK_TOKEN).
/// `clamp_retry_nanos(0) == 1`, so the downgrade preserves the
/// `it_value` the timerfd_settime path would have produced for a
/// fresh `wait_nanos == 0` outcome.
///
/// Free function so tests can drive every (first, second) outcome
/// pair without spawning a worker thread or constructing an
/// `Epoll`.
pub(crate) fn resolve_action(
    first: StallAction,
    redrain: impl FnOnce() -> DrainOutcome,
) -> WorkerAction {
    match first {
        StallAction::Continue => WorkerAction::Continue,
        StallAction::Sleep { nanos } => WorkerAction::Sleep { nanos },
        StallAction::ReDrain => match decide_stall_action(redrain()) {
            StallAction::Continue => WorkerAction::Continue,
            StallAction::Sleep { nanos } => WorkerAction::Sleep { nanos },
            StallAction::ReDrain => WorkerAction::Sleep { nanos: 1 },
        },
    }
}

/// Worker-thread epoll dispatch tokens. Hoisted to module scope
/// so the testable `worker_dispatch_event` helper (and its unit
/// tests under `cfg(test)`) can name them without duplicating
/// the values inside `worker_thread_main`'s frame.
pub(crate) const KICK_TOKEN: u64 = 1;
pub(crate) const STOP_TOKEN: u64 = 2;
pub(crate) const THROTTLE_TOKEN: u64 = 3;
pub(crate) const PAUSE_TOKEN: u64 = 4;

/// Outcome of one epoll-event dispatch decision. Lifted to a
/// dedicated enum so `worker_thread_main` and its unit tests
/// share the same vocabulary for "what should the loop do next?"
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(crate) enum WorkerDispatchAction {
    /// STOP_TOKEN observed — return state and exit the worker loop.
    Stop,
    /// Run one drain iteration. `throttle_token_fired` is true when
    /// THROTTLE_TOKEN was the cause: forces the drain past the
    /// `last_known_blocked` skip because the bucket refill timer
    /// just expired and the rolled-back chain may now be
    /// satisfiable.
    Drain { throttle_token_fired: bool },
    /// PAUSE_TOKEN observed — drain the eventfd counter, set the
    /// shared `paused` flag (Release), and park in a 10 ms
    /// `park_timeout` loop until the freeze coordinator clears the
    /// flag. The worker loop's existing `last_known_blocked` and
    /// drain state survive across the pause: a stalled chain that
    /// was rolled back stays in the avail ring; the throttle
    /// timerfd may fire while parked but its expiry is observed on
    /// the next `epoll_wait` iteration after resume.
    /// Cloud-hypervisor pattern (epoll_helper.rs `pause_evt` +
    /// `paused: Arc<AtomicBool>`): the coordinator writes 1 to the
    /// pause eventfd, the worker observes it on `epoll_wait`, and
    /// the rendezvous-side load on `paused` synchronizes-with the
    /// worker's Release store so the host's subsequent guest-memory
    /// reads happen-after the worker's last queue mutation.
    Pause,
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
pub(crate) fn worker_dispatch_event(event_set: EventSet, token: u64) -> WorkerDispatchAction {
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
        PAUSE_TOKEN => WorkerDispatchAction::Pause,
        _ => {
            tracing::warn!(?event_set, token, "virtio-blk worker: unknown epoll token");
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
///   1. Block in `epoll_wait` until kick_fd, stop_fd, the
///      throttle retry timerfd, or pause_fd is readable.
///   2. If stop_fd readable → return (drop everything cleanly).
///   3. If pause_fd readable (PAUSE_TOKEN) → drain the eventfd
///      counter, store `paused=true` (Release), and park in a
///      10 ms `park_timeout` loop until the freeze coordinator
///      clears the flag via `VirtioBlk::resume`.
///   4. If timer_fd readable (THROTTLE_TOKEN) → consume the expiry
///      count (counter-mode timerfd) and fall through to drain so
///      the rolled-back chain can re-pop now that the bucket has
///      refilled.
///   5. If kick_fd readable (KICK_TOKEN) → drain the eventfd
///      counter (one read consumes any number of coalesced kicks
///      per the eventfd counter-mode semantics — see eventfd(2))
///      and run one drain iteration via `drain_bracket_impl`.
///
/// Reading mem from the shared `Arc<OnceLock<…>>` gives the worker
/// the `GuestMemoryMmap` set by the device's `set_mem` call via a
/// lock-free `OnceLock::get`. When `mem` is unset (kick fired before
/// set_mem — a wiring bug), the `mem_unset_warned` latch fires once
/// and the kick is silently dropped.
///
/// `placement` is applied via `pin_current_thread` /
/// `set_thread_cpumask` BEFORE epoll setup so the entire worker
/// lifecycle (epoll setup, drain calls, syscalls) inherits the
/// chosen affinity. Both `service_cpu` and `no_perf_cpus` `None`
/// means inherit the parent thread's affinity (the no-topology
/// default); the topology layer guarantees at most one is `Some`.
#[cfg(not(test))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn worker_thread_main(
    mut state: BlkWorkerState,
    mut queues: [BlkQueue; NUM_QUEUES],
    mem: Arc<OnceLock<GuestMemoryMmap>>,
    irq_evt: Arc<EventFd>,
    interrupt_status: Arc<AtomicU32>,
    device_status: Arc<AtomicU32>,
    mem_unset_warned: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
    placement: WorkerPlacement,
    kick_fd: EventFd,
    stop_fd: EventFd,
    pause_fd: EventFd,
    parked_evt_slot: Arc<std::sync::Mutex<Option<Arc<EventFd>>>>,
) -> BlkWorkerState {
    // Apply the configured CPU placement before any other syscall.
    // perf-mode pins to a single CPU (cache locality + isolation
    // from the workload-measured cpuset); `--cpu-cap` no-perf
    // applies an LLC mask so the worker shares the LLC with the
    // vCPUs but stays out of the workload-measured CPUs. Both
    // `None` means the worker inherits whatever affinity the
    // spawning thread had — typically the BSP's mask, which is
    // already constrained to the test's resource budget. The two
    // helpers log success / failure via eprintln; failures do NOT
    // abort the worker (`pin_current_thread` and
    // `set_thread_cpumask` both swallow errors after logging) so
    // a missing CAP_SYS_NICE on a development host degrades
    // gracefully to "shared affinity" rather than killing the
    // device.
    if let Some(cpu) = placement.service_cpu {
        crate::vmm::vcpu::pin_current_thread(cpu, "virtio-blk worker");
    } else if let Some(ref cpus) = placement.no_perf_cpus {
        crate::vmm::vcpu::set_thread_cpumask(cpus, "virtio-blk worker");
    }
    // Clear the "construction-time paused" sentinel now that the
    // worker is fully wired up and about to enter `epoll_wait`.
    // `VirtioBlk::with_options` initialises `paused = true` so
    // pre-spawn freezes pass the rendezvous vacuously instead of
    // timing out waiting for a worker that does not exist; this
    // store is the single point at which the rendezvous begins
    // observing real worker state. Release ordering pairs with the
    // freeze coordinator's `is_paused()` Acquire-load so a freeze
    // that races construction sees either `true` (sentinel — pass)
    // or `false` (worker is live — proceed to the real
    // pause-rendezvous path). There is no third "halfway" state.
    paused.store(false, Ordering::Release);
    // epoll setup. KICK_TOKEN, STOP_TOKEN, THROTTLE_TOKEN, PAUSE_TOKEN
    // are the `EpollEvent::data` discriminators we'll match on after
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
    if let Err(e) = epoll.ctl(
        ControlOperation::Add,
        pause_fd.as_raw_fd(),
        EpollEvent::new(EventSet::IN, PAUSE_TOKEN),
    ) {
        tracing::error!(%e, "virtio-blk worker: failed to add pause_fd to epoll; exiting");
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
    let timer_fd: std::fs::File =
        unsafe { std::os::unix::io::FromRawFd::from_raw_fd(timer_fd_raw) };
    if let Err(e) = epoll.ctl(
        ControlOperation::Add,
        timer_fd_raw,
        EpollEvent::new(EventSet::IN, THROTTLE_TOKEN),
    ) {
        tracing::error!(%e, "virtio-blk worker: failed to add timer_fd to epoll; exiting");
        return state;
    }

    // Four-element scratch — kick_fd, stop_fd, timer_fd, pause_fd
    // are the only fds registered, so `epoll_wait` can return at
    // most 4 events per call.
    let mut events = [EpollEvent::default(); 4];
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
                            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
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
                WorkerDispatchAction::Pause => {
                    // Drain the pause eventfd counter BEFORE
                    // entering the rendezvous. Counter-mode
                    // semantics (eventfd(2)): a single `read`
                    // returns the accumulated counter value and
                    // resets it to 0. Draining here, at the start
                    // of the PAUSE arm, makes the next
                    // pause() — issued by the coordinator AFTER
                    // resume() returns control of this rendezvous
                    // cycle — produce a fresh epoll readiness
                    // that the next iteration's `epoll_wait`
                    // observes as a new PAUSE_TOKEN.
                    //
                    // Draining AFTER park exit (the prior design)
                    // races: between park-exit and the post-park
                    // drain, the coordinator can complete cycle N
                    // (resume()) and start cycle N+1 (pause()).
                    // The pause_fd.read at the bottom of this arm
                    // would then collapse both N and N+1's
                    // counter increments into a single drained
                    // value, leaving epoll readiness cleared —
                    // and cycle N+1's PAUSE_TOKEN would never
                    // fire. The coordinator's subsequent
                    // `paused.load(Acquire)` rendezvous poll
                    // would spin until FREEZE_RENDEZVOUS_TIMEOUT.
                    //
                    // Cloud-hypervisor's epoll_helper.rs drains
                    // its pause-side fd before parking too —
                    // this matches that pattern. EAGAIN under
                    // EFD_NONBLOCK from a saturated counter is
                    // benign (counter saturation requires
                    // u64::MAX-1 unobserved writes — implausible).
                    match pause_fd.read() {
                        Ok(_) => {}
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                        Err(e) => {
                            tracing::warn!(
                                %e,
                                "virtio-blk worker: pause_fd read failed at PAUSE entry",
                            );
                        }
                    }
                    // Signal the freeze coordinator we are parked.
                    // Release ordering pairs with the coordinator's
                    // `paused.load(Acquire)` rendezvous poll: the
                    // load synchronizes-with this store, so the
                    // coordinator's subsequent guest-memory reads
                    // happen-after every queue mutation the worker
                    // performed before this point. This mirrors
                    // the vCPU rendezvous's `parked.store(Release)`
                    // pattern in [`exit_dispatch::handle_freeze`].
                    paused.store(true, Ordering::Release);
                    // Wake the freeze coordinator's rendezvous poll
                    // by writing to the shared parked_evt AFTER the
                    // Release store. The ordering is load-bearing:
                    // the coordinator's Acquire load on `paused`
                    // happens-after this Release, so its subsequent
                    // guest-memory reads observe every queue
                    // mutation the worker performed before park.
                    // EAGAIN under EFD_NONBLOCK from a saturated
                    // counter is benign — the AtomicBool is the
                    // source of truth.
                    // Recover from a poisoned mutex (a prior holder
                    // panicked). The slot itself is plain data
                    // (Option<Arc<EventFd>>); proceeding past the
                    // panic only risks reading the most-recent
                    // value, which is exactly what we need to wake
                    // the coordinator. Silently skipping on
                    // poisoning would suppress the parked_evt write
                    // and force the coordinator to wait the full
                    // FREEZE_RENDEZVOUS_TIMEOUT before observing
                    // `paused` via its periodic poll.
                    let guard = match parked_evt_slot.lock() {
                        Ok(g) => g,
                        Err(poisoned) => {
                            tracing::warn!(
                                "virtio-blk worker: parked_evt_slot lock poisoned; \
                                 recovering inner data via PoisonError::into_inner"
                            );
                            poisoned.into_inner()
                        }
                    };
                    if let Some(ref evt) = *guard
                        && let Err(e) = evt.write(1)
                    {
                        tracing::debug!(
                            err = %e,
                            "virtio-blk worker: parked_evt write failed (EAGAIN expected on counter saturation)"
                        );
                    }
                    // Park until the coordinator clears the flag.
                    // `park_timeout(10ms)` is the same poll cadence
                    // the vCPU rendezvous uses — short enough that
                    // resume is responsive, long enough that an
                    // unwoken park does not spin-burn the worker
                    // CPU. Acquire-load synchronizes-with the
                    // coordinator's `paused.store(false, Release)`
                    // in [`VirtioBlk::resume`].
                    while paused.load(Ordering::Acquire) {
                        std::thread::park_timeout(std::time::Duration::from_millis(10));
                    }
                    // No post-park pause_fd drain: the entry-side
                    // drain above already consumed cycle N's
                    // counter, and any pause() that lands AFTER
                    // resume() (cycle N+1 from the same or a
                    // different coordinator) must produce a
                    // fresh PAUSE_TOKEN on the next epoll_wait.
                    // Resume: continue the outer loop iteration.
                    // We do NOT set should_drain here; if a kick
                    // landed during the pause window, KICK_TOKEN
                    // re-fired in the same `events[..n]` batch
                    // (epoll readiness is level-triggered) and
                    // `should_drain` is already true from that arm.
                    // If no kick landed, the next epoll_wait blocks
                    // until a real event fires. Either way the
                    // pause arm is correct without forcing a drain.
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
            &device_status,
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
        // `resolve_action` performs the bounded-recursion inline
        // retry: a `ReDrain` outcome triggers exactly one extra
        // `drain_bracket_impl` call, and a second `ReDrain` is
        // downgraded to `Sleep { nanos: 1 }` so the loop never
        // spins (a stall→retry→stall pattern would otherwise
        // starve STOP_TOKEN/KICK_TOKEN). The `WorkerAction`
        // return type encodes that bound at the type level —
        // the apply-site match below is exhaustive over
        // `Continue` and `Sleep` only.
        let action = resolve_action(decide_stall_action(outcome), || {
            tracing::trace!("virtio-blk worker: wait_nanos==0 inline re-drain");
            drain_bracket_impl(
                &mut state,
                &mut queues,
                mem_ref,
                &irq_evt,
                &interrupt_status,
                &device_status,
            )
        });
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
            WorkerAction::Sleep { nanos } => {
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
            WorkerAction::Continue => {
                last_known_blocked = false;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! Tier 3 of the test co-location split: pure worker-policy tests
    //! (clamp_retry_nanos boundary, decide_stall_action mapping,
    //! worker_dispatch_event dispatch, epoll-event-set roundtrip).
    //! Each is a pure-function test of the always-compiled helpers
    //! defined in this file — no `VirtioBlk` construction, no chain
    //! plant, so they live closest to their targets and don't need
    //! anything from `super::testing`.
    use super::*;
    use vmm_sys_util::epoll::{EpollEvent, EventSet};

    /// `clamp_retry_nanos(0)` floors at 1 ns rather than 0.
    /// `timerfd_settime(2)` with `it_value = 0` disarms the
    /// timer, so a `wait_nanos == 0` outcome (the bucket already
    /// refilled between can_consume and the deficit
    /// computation) must be mapped to the smallest non-zero
    /// value to fire the retry immediately.
    #[test]
    fn clamp_retry_nanos_zero_floors_at_one() {
        assert_eq!(
            clamp_retry_nanos(0),
            1,
            "wait_nanos==0 must be floored to 1ns to avoid \
             timerfd_settime(it_value=0) disarming the timer",
        );
    }

    /// `clamp_retry_nanos(u64::MAX)` caps at
    /// `RETRY_TIMER_MAX_NANOS` so a pathological refill rate
    /// can't push the retry past the guest's hung-task watchdog
    /// (`kernel.hung_task_timeout_secs`, default 120 s — virtio_blk
    /// has no `mq_ops->timeout`).
    #[test]
    fn clamp_retry_nanos_saturates_at_cap() {
        assert_eq!(
            clamp_retry_nanos(u64::MAX),
            RETRY_TIMER_MAX_NANOS,
            "wait_nanos==u64::MAX must saturate at RETRY_TIMER_MAX_NANOS \
             (1s) — well below the guest's hung-task watchdog (120s)",
        );
        // Boundary just over the cap also clamps.
        assert_eq!(
            clamp_retry_nanos(RETRY_TIMER_MAX_NANOS + 1),
            RETRY_TIMER_MAX_NANOS,
        );
        // Boundary at the cap is unchanged.
        assert_eq!(
            clamp_retry_nanos(RETRY_TIMER_MAX_NANOS),
            RETRY_TIMER_MAX_NANOS,
        );
        // Boundary just below the cap is unchanged. Catches an
        // off-by-one regression that used `< RETRY_TIMER_MAX_NANOS`
        // instead of `<=` in the upper-bound clamp.
        assert_eq!(
            clamp_retry_nanos(RETRY_TIMER_MAX_NANOS - 1),
            RETRY_TIMER_MAX_NANOS - 1,
        );
        // Boundary at the lower floor is unchanged. Catches an
        // off-by-one regression that mapped `wait_nanos == 1` to
        // 0 (disarming the timer) or to 2 (over-correcting the
        // floor).
        assert_eq!(clamp_retry_nanos(1), 1);
        // Mid-range values pass through.
        assert_eq!(clamp_retry_nanos(500_000_000), 500_000_000);
    }

    /// Pin `RETRY_TIMER_MAX_NANOS` at 1 s. Doc comments throughout
    /// the file (module doc, `clamp_retry_nanos`, throttled_count
    /// counter doc) cite "1 s" as the cap; a regression that
    /// changed the const without updating the docs would surface as
    /// a doc-drift class-of-bug, but the const itself is the
    /// authoritative source. The docs cite this constant by name,
    /// so this assertion is the single load-bearing pin.
    #[test]
    fn retry_timer_max_nanos_constant_pin() {
        assert_eq!(RETRY_TIMER_MAX_NANOS, 1_000_000_000);
    }

    /// `decide_stall_action(Done)` produces `Continue`. The worker
    /// treats `Continue` as the cue to clear `last_known_blocked`
    /// and resume `epoll_wait` without arming the retry timerfd.
    /// Pins the happy-path mapping that the production loop depends
    /// on after every successful drain.
    #[test]
    fn decide_stall_action_done_is_continue() {
        assert_eq!(
            decide_stall_action(DrainOutcome::Done),
            StallAction::Continue,
        );
    }

    /// `decide_stall_action(ThrottleStalled { wait_nanos: 0 })`
    /// produces `ReDrain`. The wait_nanos==0 outcome is the only
    /// state that re-runs `drain_bracket_impl` synchronously —
    /// arming the timerfd would round-trip through epoll for no
    /// reason because the bucket has already refilled between
    /// `can_consume` failure and the deficit computation.
    #[test]
    fn decide_stall_action_zero_wait_is_redrain() {
        assert_eq!(
            decide_stall_action(DrainOutcome::ThrottleStalled { wait_nanos: 0 }),
            StallAction::ReDrain,
        );
    }

    /// `decide_stall_action(ThrottleStalled { wait_nanos: n > 0 })`
    /// produces `Sleep { nanos: clamp_retry_nanos(n) }`. The
    /// clamp is composed in-line so the worker can feed `nanos`
    /// directly to `timerfd_settime` without re-clamping. Pins the
    /// boundary cases:
    ///
    /// - `wait_nanos == 1` → `Sleep { nanos: 1 }` (already at floor;
    ///   the floor at 1 ns prevents timerfd disarm).
    /// - mid-range pass-through.
    /// - `wait_nanos == RETRY_TIMER_MAX_NANOS` → unchanged (cap
    ///   inclusive).
    /// - `wait_nanos == RETRY_TIMER_MAX_NANOS + 1` → capped (cap
    ///   exclusive on the over side).
    /// - `wait_nanos == u64::MAX` → capped at the same maximum, so
    ///   a pathological refill rate can't push the retry past
    ///   `kernel.hung_task_timeout_secs`.
    #[test]
    fn decide_stall_action_nonzero_wait_is_sleep_with_clamped_nanos() {
        assert_eq!(
            decide_stall_action(DrainOutcome::ThrottleStalled { wait_nanos: 1 }),
            StallAction::Sleep { nanos: 1 },
        );
        assert_eq!(
            decide_stall_action(DrainOutcome::ThrottleStalled {
                wait_nanos: 500_000_000,
            }),
            StallAction::Sleep { nanos: 500_000_000 },
        );
        assert_eq!(
            decide_stall_action(DrainOutcome::ThrottleStalled {
                wait_nanos: RETRY_TIMER_MAX_NANOS,
            }),
            StallAction::Sleep {
                nanos: RETRY_TIMER_MAX_NANOS,
            },
        );
        assert_eq!(
            decide_stall_action(DrainOutcome::ThrottleStalled {
                wait_nanos: RETRY_TIMER_MAX_NANOS + 1,
            }),
            StallAction::Sleep {
                nanos: RETRY_TIMER_MAX_NANOS,
            },
        );
        assert_eq!(
            decide_stall_action(DrainOutcome::ThrottleStalled {
                wait_nanos: u64::MAX,
            }),
            StallAction::Sleep {
                nanos: RETRY_TIMER_MAX_NANOS,
            },
        );
    }

    /// `decide_stall_action` is a pure function — calling it twice
    /// with the same input must produce the same output. Pins the
    /// "pure" property the worker loop depends on when it
    /// double-calls the decision (once on the initial drain, once
    /// on the second drain after `ReDrain`). A regression that
    /// added internal state (e.g. a bucket-refill side effect)
    /// would surface here.
    #[test]
    fn decide_stall_action_is_pure() {
        let inputs = [
            DrainOutcome::Done,
            DrainOutcome::ThrottleStalled { wait_nanos: 0 },
            DrainOutcome::ThrottleStalled { wait_nanos: 1 },
            DrainOutcome::ThrottleStalled { wait_nanos: 12345 },
            DrainOutcome::ThrottleStalled {
                wait_nanos: u64::MAX,
            },
        ];
        for input in inputs {
            assert_eq!(
                decide_stall_action(input),
                decide_stall_action(input),
                "decide_stall_action must be deterministic for {input:?}",
            );
        }
    }

    /// Pins the worker-loop's bounded-recursion contract: a second
    /// `ReDrain` produced from the inline retry must be downgraded
    /// to `Sleep { nanos: 1 }` so the loop never spins. The worker
    /// expresses this with an inline `match` against
    /// `decide_stall_action(outcome2)`; this test mirrors that
    /// match shape so a regression in the worker that drops the
    /// downgrade would break a unit test in addition to the
    /// integration paths.
    #[test]
    fn decide_stall_action_redrain_downgrades_to_sleep_one_ns() {
        // First drain returns wait_nanos==0 → ReDrain.
        let outcome1 = DrainOutcome::ThrottleStalled { wait_nanos: 0 };
        assert_eq!(decide_stall_action(outcome1), StallAction::ReDrain);
        // Second drain ALSO returns wait_nanos==0 → ReDrain. The
        // worker downgrades it to `Sleep { nanos: 1 }` to bound
        // the recursion. `clamp_retry_nanos(0) == 1`, so this is
        // exactly equivalent to `decide_stall_action` if the input
        // had been wait_nanos==1.
        let outcome2 = DrainOutcome::ThrottleStalled { wait_nanos: 0 };
        let downgraded = match decide_stall_action(outcome2) {
            StallAction::ReDrain => StallAction::Sleep { nanos: 1 },
            other => other,
        };
        assert_eq!(downgraded, StallAction::Sleep { nanos: 1 });
        // Sanity: the downgrade matches the floor that
        // clamp_retry_nanos imposes on a fresh wait_nanos==0
        // outcome — so the bounded-recursion arm produces the
        // same it_value as the legacy code (which always passed
        // through clamp_retry_nanos before arming the timerfd).
        assert_eq!(clamp_retry_nanos(0), 1);
    }

    /// `resolve_action` on `Continue` skips the inline-retry
    /// closure entirely and returns `WorkerAction::Continue`.
    /// Pins the happy-path contract: a successful drain must
    /// not invoke the redrain closure (which in production would
    /// re-call `drain_bracket_impl`).
    #[test]
    fn resolve_action_continue_skips_redrain() {
        let mut redrain_called = false;
        let action = resolve_action(StallAction::Continue, || {
            redrain_called = true;
            DrainOutcome::Done
        });
        assert_eq!(action, WorkerAction::Continue);
        assert!(
            !redrain_called,
            "Continue must NOT invoke the inline-retry closure",
        );
    }

    /// `resolve_action` on `Sleep { nanos }` skips the inline-retry
    /// closure and returns `WorkerAction::Sleep { nanos }` with
    /// the same value. The clamp was applied by `decide_stall_action`
    /// upstream — `resolve_action` is a pass-through here.
    #[test]
    fn resolve_action_sleep_skips_redrain() {
        let mut redrain_called = false;
        let action = resolve_action(StallAction::Sleep { nanos: 12345 }, || {
            redrain_called = true;
            DrainOutcome::Done
        });
        assert_eq!(action, WorkerAction::Sleep { nanos: 12345 });
        assert!(
            !redrain_called,
            "Sleep must NOT invoke the inline-retry closure",
        );
    }

    /// `resolve_action` on `ReDrain` invokes the closure exactly
    /// once and converts a `Done` second outcome into `Continue`.
    /// Pins the inline-retry success path: the wait_nanos==0
    /// trigger fires → second drain succeeds → loop continues
    /// without arming the timerfd.
    #[test]
    fn resolve_action_redrain_done_is_continue() {
        let mut call_count = 0;
        let action = resolve_action(StallAction::ReDrain, || {
            call_count += 1;
            DrainOutcome::Done
        });
        assert_eq!(action, WorkerAction::Continue);
        assert_eq!(
            call_count, 1,
            "ReDrain must invoke the inline-retry closure exactly once",
        );
    }

    /// `resolve_action` on `ReDrain` invokes the closure once and
    /// passes through a `Sleep` outcome from the second drain. Pins
    /// the path where the inline retry stalls with a non-zero
    /// deficit — the worker arms the timerfd at that deficit
    /// rather than re-recursing.
    #[test]
    fn resolve_action_redrain_sleep_passes_through() {
        let mut call_count = 0;
        let action = resolve_action(StallAction::ReDrain, || {
            call_count += 1;
            DrainOutcome::ThrottleStalled {
                wait_nanos: 500_000_000,
            }
        });
        assert_eq!(action, WorkerAction::Sleep { nanos: 500_000_000 });
        assert_eq!(
            call_count, 1,
            "ReDrain must invoke the inline-retry closure exactly once",
        );
    }

    /// `resolve_action` on `ReDrain` followed by a second `ReDrain`
    /// (wait_nanos==0 again) is downgraded to `Sleep { nanos: 1 }`.
    /// Pins the bounded-recursion contract: the loop never spins
    /// because the type system disallows a third inline retry.
    /// `clamp_retry_nanos(0) == 1`, so the downgrade preserves the
    /// `it_value` the legacy code produced.
    #[test]
    fn resolve_action_redrain_redrain_downgrades_to_sleep_one_ns() {
        let mut call_count = 0;
        let action = resolve_action(StallAction::ReDrain, || {
            call_count += 1;
            DrainOutcome::ThrottleStalled { wait_nanos: 0 }
        });
        assert_eq!(action, WorkerAction::Sleep { nanos: 1 });
        assert_eq!(
            call_count, 1,
            "ReDrain followed by ReDrain must invoke the closure \
             exactly once — the downgrade prevents a third call",
        );
        // Sanity: the 1-ns downgrade matches the floor that
        // clamp_retry_nanos imposes on a fresh wait_nanos==0
        // outcome.
        assert_eq!(clamp_retry_nanos(0), 1);
    }

    /// `worker_dispatch_event` routes STOP_TOKEN with EventSet::IN
    /// to `Stop`. The worker treats this as a clean exit signal —
    /// any other action would mean the device's `Drop::drop`
    /// stop-fd write either gets lost (no exit) or is misclassified
    /// as a drain-and-then-exit (extra drain iteration on a queue
    /// that's about to be dismantled).
    #[test]
    fn worker_dispatch_stop_token_clean() {
        assert_eq!(
            worker_dispatch_event(EventSet::IN, STOP_TOKEN),
            WorkerDispatchAction::Stop,
        );
    }

    /// `worker_dispatch_event` routes KICK_TOKEN with EventSet::IN
    /// to `Drain { throttle_token_fired: false }`. The drain is
    /// guarded by the `last_known_blocked` skip in the worker
    /// loop, so a kick that arrives while the throttle is stalled
    /// must NOT force the drain — only THROTTLE_TOKEN does.
    #[test]
    fn worker_dispatch_kick_token_clean() {
        assert_eq!(
            worker_dispatch_event(EventSet::IN, KICK_TOKEN),
            WorkerDispatchAction::Drain {
                throttle_token_fired: false,
            },
        );
    }

    /// `worker_dispatch_event` routes THROTTLE_TOKEN with
    /// EventSet::IN to `Drain { throttle_token_fired: true }`.
    /// Setting the flag is load-bearing for liveness: it forces
    /// the drain past `last_known_blocked` so the rolled-back
    /// chain is retried after the bucket refill timer expires.
    #[test]
    fn worker_dispatch_throttle_token_sets_flag() {
        assert_eq!(
            worker_dispatch_event(EventSet::IN, THROTTLE_TOKEN),
            WorkerDispatchAction::Drain {
                throttle_token_fired: true,
            },
        );
    }

    /// `worker_dispatch_event` routes PAUSE_TOKEN with EventSet::IN
    /// to `Pause`. The pause action drives the freeze-coordinator
    /// rendezvous: the worker drains pause_fd, stores `paused=true`
    /// (Release), and parks until the coordinator clears the flag.
    /// Pins the dispatch contract so a regression that drops the
    /// PAUSE_TOKEN arm (or routes it to `Skip`) breaks this test
    /// before the freeze rendezvous breaks in production.
    #[test]
    fn worker_dispatch_pause_token_clean() {
        assert_eq!(
            worker_dispatch_event(EventSet::IN, PAUSE_TOKEN),
            WorkerDispatchAction::Pause,
        );
    }

    /// EPOLLERR | IN on PAUSE_TOKEN still pauses. eventfd EPOLLERR
    /// fires only on counter saturation (count == ULLONG_MAX), which
    /// is implausible for the pause path because every `pause()` is
    /// paired with a worker-side entry drain. Mirrors the
    /// KICK/STOP/THROTTLE EPOLLERR sibling tests so a future change
    /// that short-circuits on EPOLLERR before reaching the token
    /// match would break this test before it broke the rendezvous.
    #[test]
    fn worker_dispatch_pause_token_with_epollerr_still_pauses() {
        let event_set = EventSet::IN | EventSet::ERROR;
        assert_eq!(
            worker_dispatch_event(event_set, PAUSE_TOKEN),
            WorkerDispatchAction::Pause,
            "EPOLLERR on pause_fd must fall through to the pause arm \
             so the entry-side drain clears the saturated counter",
        );
    }

    /// Unknown token dispatches to `Skip` and the worker loop
    /// continues without draining. Defends against a future
    /// regression that registers an additional fd on the same
    /// epoll without extending the dispatch match.
    /// Tokens 0 and 5..=u64::MAX are guaranteed unknown; tokens
    /// 1..=4 are KICK_TOKEN/STOP_TOKEN/THROTTLE_TOKEN/PAUSE_TOKEN
    /// respectively and are excluded.
    #[test]
    fn worker_dispatch_unknown_token_skips() {
        for token in [0u64, 5, 99, u64::MAX] {
            assert_eq!(
                worker_dispatch_event(EventSet::IN, token),
                WorkerDispatchAction::Skip,
                "token {token} must dispatch to Skip",
            );
        }
    }

    /// EPOLLERR | IN on KICK_TOKEN still drains. Eventfd
    /// `eventfd_poll` co-sets EPOLLIN whenever count > 0, and
    /// EPOLLERR when count == ULLONG_MAX. The recovery is for the
    /// per-token handler's `kick_fd.read()` to drain the saturated
    /// counter — so the dispatch must still produce
    /// `Drain { throttle_token_fired: false }`.
    #[test]
    fn worker_dispatch_kick_token_with_epollerr_still_drains() {
        let event_set = EventSet::IN | EventSet::ERROR;
        assert_eq!(
            worker_dispatch_event(event_set, KICK_TOKEN),
            WorkerDispatchAction::Drain {
                throttle_token_fired: false,
            },
            "EPOLLERR on eventfd indicates counter saturation; \
             fall through to per-token drain so the read clears it",
        );
    }

    /// EPOLLERR | IN on THROTTLE_TOKEN still drains. Timerfd
    /// never sets EPOLLERR (timerfd_poll only checks ticks), so
    /// observing it means the kernel contract changed — but the
    /// dispatch still falls through and the timerfd read in the
    /// worker arm yields EAGAIN if no expiry is queued. Net
    /// effect: defensive log + no-op.
    #[test]
    fn worker_dispatch_throttle_token_with_epollerr_still_drains() {
        let event_set = EventSet::IN | EventSet::ERROR;
        assert_eq!(
            worker_dispatch_event(event_set, THROTTLE_TOKEN),
            WorkerDispatchAction::Drain {
                throttle_token_fired: true,
            },
        );
    }

    /// EPOLLERR | IN on STOP_TOKEN still stops. Stop-fd is an
    /// eventfd (same EFD_NONBLOCK flags as kick_fd); saturation
    /// is implausible because Drop writes 1 once. But if it ever
    /// happens (e.g. a regression hands the worker a long-lived
    /// stop_fd whose counter accumulated), Stop semantics
    /// dominate ERR — there's no useful recovery once we've
    /// decided to exit.
    #[test]
    fn worker_dispatch_stop_token_with_epollerr_still_stops() {
        let event_set = EventSet::IN | EventSet::ERROR;
        assert_eq!(
            worker_dispatch_event(event_set, STOP_TOKEN),
            WorkerDispatchAction::Stop,
        );
    }

    /// EPOLLHUP | IN on KICK_TOKEN still drains. eventfd_poll
    /// never sets POLLHUP, so observing it is structurally
    /// impossible for our owned eventfd — but we log defensively
    /// and the dispatch still falls through. The per-token
    /// handler's `kick_fd.read()` is harmless in any case.
    #[test]
    fn worker_dispatch_kick_token_with_epollhup_still_drains() {
        let event_set = EventSet::IN | EventSet::HANG_UP;
        assert_eq!(
            worker_dispatch_event(event_set, KICK_TOKEN),
            WorkerDispatchAction::Drain {
                throttle_token_fired: false,
            },
        );
    }

    /// EPOLLERR ALONE (no EPOLLIN) on KICK_TOKEN still drains.
    /// Reaching this state in production for eventfd is
    /// structurally impossible (count==ULLONG_MAX implies
    /// count>0 implies EPOLLIN is also set per eventfd_poll), but
    /// the dispatch must remain robust if a future kernel patch
    /// changes the contract or a different fd type is registered.
    /// Falling through to the per-token handler's read is the
    /// canonical recovery; the read returns EAGAIN harmlessly if
    /// no data is queued.
    #[test]
    fn worker_dispatch_kick_token_epollerr_alone_still_drains() {
        let event_set = EventSet::ERROR;
        assert_eq!(
            worker_dispatch_event(event_set, KICK_TOKEN),
            WorkerDispatchAction::Drain {
                throttle_token_fired: false,
            },
        );
    }

    /// EPOLLERR | EPOLLHUP | IN on THROTTLE_TOKEN: every defensive
    /// flag is set at once. The dispatch still drains and sets
    /// the throttle-fired marker. Catches a regression that
    /// short-circuits on EPOLLHUP before reaching the token
    /// match.
    #[test]
    fn worker_dispatch_all_flags_throttle_still_drains() {
        let event_set = EventSet::IN | EventSet::ERROR | EventSet::HANG_UP;
        assert_eq!(
            worker_dispatch_event(event_set, THROTTLE_TOKEN),
            WorkerDispatchAction::Drain {
                throttle_token_fired: true,
            },
        );
    }

    /// EpollEvent::new + event_set roundtrip pin. Defends against
    /// a vmm-sys-util regression that lost the EventSet::ERROR or
    /// EventSet::HANG_UP bit-mapping — without it, the dispatch
    /// helper's `event_set.contains(EventSet::ERROR)` checks would
    /// silently fail and the warn log would never fire on
    /// saturation.
    #[test]
    fn epoll_event_set_roundtrip_pin() {
        let combo = EventSet::IN | EventSet::ERROR | EventSet::HANG_UP;
        let ev = EpollEvent::new(combo, KICK_TOKEN);
        assert_eq!(ev.data(), KICK_TOKEN);
        assert!(ev.event_set().contains(EventSet::IN));
        assert!(ev.event_set().contains(EventSet::ERROR));
        assert!(ev.event_set().contains(EventSet::HANG_UP));
    }
}
