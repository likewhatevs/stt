//! Per-device host-side observability counters for virtio-block.
//!
//! Pure atomic counters + their `record_*` mutator helpers and `pub fn`
//! readers. No MMIO, no FSM, no IO — split out from `device.rs` for
//! module locality so the counter taxonomy doc and the per-helper
//! invariants (per-event vs per-request vs gauge) sit together.
//!
//! See `super::drain_bracket_impl` and the per-handler `handle_*_impl`
//! functions for the writer sites; see `VirtioBlk::counters()` for the
//! external Arc handle the host monitor uses to read without locking
//! the device struct.

use std::sync::atomic::{AtomicU64, Ordering};

// ----------------------------------------------------------------------------
// Counters (host-side observability)
// ----------------------------------------------------------------------------

/// Per-device counters surfaced to the host monitor. All atomic so
/// the monitor can read them without locking the device struct.
///
/// Mutation goes through the `record_*` helper methods, NOT direct
/// `field.fetch_add(...)` calls. The helpers enforce the
/// "completion + bytes" pairing for reads and writes — every
/// `record_read(bytes)` increments both `reads_completed` AND
/// `bytes_read` in one call. A bare `reads_completed.fetch_add(1)`
/// without a paired `bytes_read.fetch_add(n)` would let the
/// failure-dump renderer compute a misleading bytes-per-op
/// average. The helpers also keep the call site one line each,
/// matching the SPSC-style accounting common in network/block
/// device fast paths.
///
/// Fields are `pub(crate)` so the helper-mutation rule is enforced
/// across the crate by visibility. External consumers reach in via
/// the per-field `pub fn` accessors below — each performs a
/// `Relaxed` load and returns the current value as `u64`.
///
/// # Counter taxonomy: events vs requests vs gauges
///
/// Counters fall into three semantic categories. Operators
/// reading the failure-dump must understand which is which to
/// avoid drawing wrong conclusions:
///
/// - **Per-event cumulative counters** (`io_errors`,
///   `throttled_count`): bumped each time the underlying event
///   fires, with no per-request deduplication. A single hostile
///   request can produce multiple `io_errors` bumps if it trips
///   several gates in sequence (see `io_errors` doc below for the
///   double-bump scenarios). Use these to compare event rates
///   over time, not to count requests.
/// - **Per-request cumulative counters** (`reads_completed`,
///   `writes_completed`, `flushes_completed`, `bytes_read`,
///   `bytes_written`): bumped exactly once per successfully
///   serviced request. Each surfaces a one-to-one mapping with
///   guest-observable completions. Use these to compute
///   throughput, average request size, and per-direction IO
///   share.
/// - **Per-request live gauges** (`currently_throttled_gauge`):
///   "how many requests are RIGHT NOW in this state." Increments
///   when a request enters the state, decrements when it exits.
///   The cumulative event counter for the same condition lives
///   in `throttled_count` (events, not requests). Reading
///   `currently_throttled_gauge == 5` means 5 chains are pinned
///   in the avail ring at this instant; `throttled_count == 100`
///   over the same period means 100 stall events have occurred.
///   The two answer different questions and operators MUST NOT
///   compare or sum them.
///
/// # Lifetime semantics
///
/// Counters are **cumulative for the device's lifetime** —
/// `VirtioBlk::reset()` does NOT zero them. A guest issuing
/// STATUS=0 (driver re-bind) re-uses the existing counter Arc; an
/// operator monitoring `reads_completed` etc. observes a
/// monotonically non-decreasing series across resets. Only
/// destruction of the device (`Drop`) reclaims the counters Arc.
/// This matches operator expectation that failure-dump counters
/// reflect the device's full IO history, not just the post-reset
/// fragment.
///
/// Per-request live gauges (`currently_throttled_gauge`) decrement
/// across the device's lifetime as requests exit the gauged
/// state, but the gauge value itself is "right now," not
/// cumulative. A reset that strands a chain in the
/// "currently throttled" state would leak the gauge increment;
/// the production reset path joins the worker thread before
/// rebuilding the queue, and the worker decrements the gauge on
/// any subsequent successful drain — but a worker that never
/// observes a successful drain (e.g. the device is destroyed
/// while the chain is still rolled back) leaves the increment
/// pinned for the device's lifetime. This is acceptable because
/// the gauge is informational and the device is going away
/// anyway; downstream consumers must not depend on a strictly
/// zero-on-shutdown property.
///
/// We diverge from virtio-v1.2 §2.1 ("device returned to its
/// initial state") for counters because operator-side
/// failure-dump observability requires cumulative IO history
/// spanning the device's full lifetime, not just the post-reset
/// fragment.
#[derive(Debug, Default)]
pub struct VirtioBlkCounters {
    pub(crate) reads_completed: AtomicU64,
    pub(crate) writes_completed: AtomicU64,
    pub(crate) flushes_completed: AtomicU64,
    pub(crate) bytes_read: AtomicU64,
    pub(crate) bytes_written: AtomicU64,
    /// Cumulative throttle-stall **events** for the device's
    /// lifetime. Bumped each time `drain_bracket_impl` returns
    /// `DrainOutcome::ThrottleStalled`. A single chain that
    /// stalls, refills, stalls again, and finally completes
    /// produces TWO `throttled_count` bumps but ONE
    /// `reads_completed` (or `writes_completed`/etc.) bump on
    /// final success.
    ///
    /// To answer "how many requests are stuck right now," read
    /// `currently_throttled_gauge` instead — the per-event
    /// cumulative counter and the per-request live gauge are
    /// distinct semantics and answer different questions.
    pub(crate) throttled_count: AtomicU64,
    pub(crate) io_errors: AtomicU64,
    /// Live "how many requests are currently waiting for tokens"
    /// gauge. Incremented when a chain transitions into the
    /// stalled state; decremented when the next successful drain
    /// confirms the chain has been serviced.
    ///
    /// On a single-queue virtio-blk device the gauge is bounded
    /// at 0 or 1 in practice — only the head-of-queue chain can
    /// be stalled at a time, because the FIFO drain rolls back
    /// the popped chain on stall and the next successful drain
    /// always processes that same chain first before any newer
    /// arrivals. A multi-queue extension would lift the bound to
    /// "1 per queue currently stalled."
    ///
    /// Distinct from `throttled_count` (cumulative events): the
    /// gauge tracks the live state, the counter tracks the
    /// historical event rate. See the type-level "Counter
    /// taxonomy" doc for why operators must not conflate the
    /// two.
    pub(crate) currently_throttled_gauge: AtomicU64,
    /// Cumulative count of `Error::InvalidAvailRingIndex` events
    /// observed by `drain_bracket_impl`. Bumped each time the
    /// virtio-queue iter() rejects an avail.idx whose distance
    /// from `next_avail` exceeds the queue size — a hostile or
    /// buggy guest condition that, if not detected, would loop
    /// the worker forever (the swallowed-error livelock fixed by
    /// the queue_poisoned gate).
    ///
    /// Per-event counter (NOT per-request): a single drain pass
    /// produces at most one bump (the poison flag short-circuits
    /// further attempts on the same queue). Successive
    /// QUEUE_NOTIFY kicks against an unresetted poisoned queue
    /// take the early-return path and produce zero additional
    /// bumps until the guest performs a virtio reset.
    pub(crate) invalid_avail_idx_count: AtomicU64,
}

impl VirtioBlkCounters {
    /// Record one completed read: bumps `reads_completed` and adds
    /// `bytes` to `bytes_read`. The pairing is enforced — bare
    /// reads_completed bumps without the paired bytes_read add are
    /// caught at refactor time.
    ///
    /// `bytes` MUST be the count actually returned by `read_at`
    /// summed across the request's data segments — NOT the
    /// descriptor length. On a short read the zero-padded tail is
    /// delivered to the guest but does not count here; see
    /// [`Self::bytes_read`] for the rationale.
    pub(crate) fn record_read(&self, bytes: u64) {
        self.reads_completed.fetch_add(1, Ordering::Relaxed);
        self.bytes_read.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record one completed write: bumps `writes_completed` and
    /// adds `bytes` to `bytes_written`.
    pub(crate) fn record_write(&self, bytes: u64) {
        self.writes_completed.fetch_add(1, Ordering::Relaxed);
        self.bytes_written.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record one completed flush.
    pub(crate) fn record_flush(&self) {
        self.flushes_completed.fetch_add(1, Ordering::Relaxed);
    }

    /// Bumped on every host-observed IO failure **event**, whether
    /// the guest saw S_IOERR or not (e.g. unmapped status-byte
    /// address that prevented the status write). Covers spec
    /// violations, backend IO errors, malformed chains, add_used
    /// failures, and status-write failures where the chain stays
    /// in the avail ring (no S_IOERR ever reaches the guest, but
    /// the host still counts the silent-stall event).
    ///
    /// # Events, not requests
    ///
    /// `io_errors` is an **events** counter, not a per-request
    /// counter. A single hostile request can produce multiple
    /// `io_errors` bumps if it trips several gates in sequence.
    /// Concretely:
    ///
    /// - **Pre-publish gates that bump io_errors then call
    ///   `publish_completion`**: SEG_MAX reject, bad header,
    ///   header-read failure, SIZE_MAX reject, zero-data,
    ///   sub-sector data_len, direction violation. Each of these
    ///   records one io_errors event for the validation
    ///   rejection. If the subsequent `publish_completion`'s
    ///   status-byte write or `add_used` then fails (e.g. the
    ///   guest also placed the status descriptor at unmapped
    ///   GPA), `publish_completion` records a SECOND io_errors
    ///   event for the silent-stall failure mode. A pathological
    ///   chain with a malformed header AND an unmapped status
    ///   descriptor surfaces as `io_errors += 2` for one chain.
    /// - **Handler error paths**: `handle_read_impl` /
    ///   `handle_write_impl` / `handle_get_id_impl` /
    ///   `handle_flush_impl` each record io_errors on backing-file
    ///   error or guest-memory access failure. The handler
    ///   produces an S_IOERR status which `process_requests`
    ///   passes to `publish_completion`. If the status-write or
    ///   add_used then fails, `publish_completion` records a
    ///   SECOND io_errors event for that request.
    /// - **publish_completion's own failure modes**: status-write
    ///   failure or add_used failure each record one io_errors
    ///   event independently of any prior caller bump.
    ///
    /// The double-bump under hostile-guest scenarios is
    /// **intentional**. Hoisting all error bumps to a single
    /// outermost catch site would lose the "silent-stall failure
    /// distinct from validation rejection" signal: an operator
    /// reading io_errors needs to see a separate event each time
    /// the device hits a failure mode, even if multiple events
    /// happen on the same request.
    ///
    /// Operators who want a per-request error count must not
    /// derive it from io_errors — they need a separate counter
    /// (deliberately not provided here; the per-request semantic
    /// is reachable via `reads_completed + writes_completed +
    /// flushes_completed` for the success side, with the failure
    /// side inferable from `total_chains_observed - success_count`
    /// once a `total_chains_observed` counter is added).
    ///
    /// See also `currently_throttled_gauge` (per-request live
    /// gauge) and `throttled_count` (per-event cumulative
    /// counter) for the throttle-side distinction; the same
    /// events-vs-requests split applies there.
    pub(crate) fn record_io_error(&self) {
        self.io_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Record one throttle-stall **event**. virtio-spec doesn't
    /// reserve a "throttled" status code; on stall the device
    /// rolls back the pop and arms a retry timer (see
    /// `drain_bracket_impl` and `worker_thread_main`) — the chain
    /// stays invisible to the guest until enough tokens refill.
    /// Retry fires within `RETRY_TIMER_MAX_NANOS` (1 s);
    /// pathological refill rates re-stall at the cap. The
    /// counter is separate from `io_errors` so operators can
    /// distinguish "real IO problem" from "throttle bucket
    /// drained, request deferred."
    ///
    /// # Events, not requests
    ///
    /// `throttled_count` is the cumulative event rate, not the
    /// number of stuck requests. A single chain that stalls
    /// twice (initial stall + premature retry that re-stalls)
    /// bumps `throttled_count` twice but represents one stuck
    /// request. To answer "how many requests are stuck right
    /// now," read `currently_throttled_gauge` instead.
    pub(crate) fn record_throttled(&self) {
        self.throttled_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the live "currently waiting for tokens" gauge.
    /// Called by `drain_bracket_impl` when a chain transitions
    /// from "running" to "stalled" — i.e. the per-worker
    /// `currently_stalled` flag was false before this stall.
    /// Idempotent stall observations (same chain, multiple
    /// retries that all re-stall) MUST NOT double-increment; the
    /// caller gates this on the per-worker flag transition.
    pub(crate) fn record_throttle_pending_inc(&self) {
        self.currently_throttled_gauge
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement the live "currently waiting for tokens" gauge,
    /// saturating at 0. Called by `drain_bracket_impl` when the
    /// worker observes a successful drain after a prior stall, by
    /// `reset_engine_*` on a reset that strands a stalled chain,
    /// and by `Drop` on device destruction while the
    /// rollback-stalled flag is still set. The per-worker
    /// `currently_stalled` flag gates the transition so a paired
    /// inc precedes every dec under correct operation; the
    /// saturating CAS exists as a defence-in-depth against any
    /// future caller that decrements an already-zero gauge —
    /// vanilla `fetch_sub(1)` would wrap to `u64::MAX` and the
    /// failure-dump renderer would then surface a 17-exabyte
    /// "currently stalled" reading.
    pub(crate) fn record_throttle_pending_dec(&self) {
        let _ = self.currently_throttled_gauge.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |v| v.checked_sub(1),
        );
    }

    /// Record one observed `Error::InvalidAvailRingIndex` event
    /// from `Queue::iter`. Called by `drain_bracket_impl` when the
    /// avail ring's `idx` is more than `queue.size` ahead of
    /// `next_avail` — a virtio-spec violation by the guest. The
    /// caller also sets `BlkWorkerState::queue_poisoned` so a
    /// single hostile-guest event produces exactly one bump,
    /// regardless of how many subsequent kicks land before the
    /// next reset (subsequent drains short-circuit on the poison
    /// flag and never re-call `iter`).
    pub(crate) fn record_invalid_avail_idx(&self) {
        self.invalid_avail_idx_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Read the cumulative count of successfully completed read
    /// requests for this device's lifetime. Per-request counter:
    /// bumped exactly once per successful read via
    /// [`Self::record_read`] (paired with a `bytes_read` add).
    /// `Relaxed` ordering matches the writer side — counters are
    /// publish-only observability and do not establish
    /// happens-before with other operations.
    pub fn reads_completed(&self) -> u64 {
        self.reads_completed.load(Ordering::Relaxed)
    }

    /// Read the cumulative count of successfully completed write
    /// requests for this device's lifetime. Per-request counter:
    /// bumped exactly once per successful write via
    /// [`Self::record_write`] (paired with a `bytes_written` add).
    pub fn writes_completed(&self) -> u64 {
        self.writes_completed.load(Ordering::Relaxed)
    }

    /// Read the cumulative count of successfully completed flush
    /// requests for this device's lifetime. Per-request counter:
    /// bumped once per successful flush via
    /// [`Self::record_flush`].
    pub fn flushes_completed(&self) -> u64 {
        self.flushes_completed.load(Ordering::Relaxed)
    }

    /// Read the cumulative number of bytes the device's backing
    /// file actually returned for read requests. Per-request
    /// counter: incremented in lockstep with `reads_completed`.
    ///
    /// This counts the `n` returned by each `read_at` call (i.e.
    /// the bytes actually sourced from the backing file), NOT the
    /// full descriptor length delivered to the guest. On a short
    /// read at backing-file EOF, the device zero-pads the
    /// remaining bytes of the descriptor (sparse-file semantics)
    /// and delivers them to the guest, but those zero-pad bytes
    /// do not count here — they were not "read" from any source.
    /// The virtio-spec used.elem.len reported via `add_used`
    /// includes the zero-pad (per virtio-v1.2 §2.7.7.2 it counts
    /// bytes written to device-writable buffers); operators
    /// comparing `bytes_read` to guest-side accounting must
    /// account for the zero-pad gap in sparse-file scenarios.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read.load(Ordering::Relaxed)
    }

    /// Read the cumulative number of bytes successfully written
    /// from guest memory to the backing file. Per-request counter:
    /// incremented in lockstep with `writes_completed`.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written.load(Ordering::Relaxed)
    }

    /// Read the cumulative count of throttle-stall **events** for
    /// this device's lifetime. Per-event counter (NOT per-request):
    /// a single chain that stalls multiple times produces multiple
    /// bumps. To answer "how many requests are stuck right now,"
    /// read [`Self::currently_throttled_gauge`] instead.
    pub fn throttled_count(&self) -> u64 {
        self.throttled_count.load(Ordering::Relaxed)
    }

    /// Read the cumulative count of host-observed IO failure
    /// **events**. Per-event counter (NOT per-request): a single
    /// hostile chain can produce multiple bumps if it trips
    /// several gates in sequence. See [`Self::record_io_error`]
    /// for the double-bump scenarios.
    pub fn io_errors(&self) -> u64 {
        self.io_errors.load(Ordering::Relaxed)
    }

    /// Read the live "how many requests are currently waiting for
    /// throttle tokens" gauge. NOT cumulative — increments when a
    /// chain enters the stalled state, decrements when it exits.
    /// On a single-queue device the value is bounded at 0 or 1 in
    /// practice.
    pub fn currently_throttled_gauge(&self) -> u64 {
        self.currently_throttled_gauge.load(Ordering::Relaxed)
    }

    /// Read the cumulative count of `Error::InvalidAvailRingIndex`
    /// events the device has observed. Per-event counter (NOT
    /// per-request): the queue-poison flag short-circuits
    /// subsequent kicks against the same hostile state, so one
    /// guest fault produces exactly one bump regardless of how
    /// many notifications follow before reset. A non-zero value
    /// means the guest violated virtio-v1.2 §2.7.13.3 — the
    /// device is in the "structurally broken queue" state and
    /// will not service IO until the guest issues a virtio reset.
    pub fn invalid_avail_idx_count(&self) -> u64 {
        self.invalid_avail_idx_count.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    //! Helper-level unit tests for the `record_*` mutators. These
    //! pin per-helper invariants (paired-counter lockstep,
    //! single-counter bumps, gauge inc/dec idempotence + saturating
    //! decrement) directly on `VirtioBlkCounters` without crossing
    //! the chain-parsing or worker-thread boundary. Cross-thread
    //! atomicity and end-to-end production-path coverage live in
    //! `tests_atomics.rs`; these helper-level tests catch regressions
    //! to the helpers themselves before the chain-level tests would.
    //!
    //! Each test starts from a fresh `VirtioBlkCounters::default()`
    //! so the pre-conditions are pinned at zero by the type
    //! contract — no shared state leaks across tests.
    use super::*;
    /// Fresh counters initialise every field to zero. Pinned
    /// here as a pre-condition for the rest of the helper tests
    /// — they all rely on `default()` producing an
    /// all-zero starting state. A regression that gave
    /// `AtomicU64::new(non_zero)` to any field would surface
    /// here before downstream tests' "increments by N" math
    /// silently reads a stale base.
    #[test]
    fn default_counters_are_all_zero() {
        let c = VirtioBlkCounters::default();
        assert_eq!(c.reads_completed(), 0, "reads_completed must default to 0");
        assert_eq!(
            c.writes_completed(),
            0,
            "writes_completed must default to 0"
        );
        assert_eq!(
            c.flushes_completed(),
            0,
            "flushes_completed must default to 0"
        );
        assert_eq!(c.bytes_read(), 0, "bytes_read must default to 0");
        assert_eq!(c.bytes_written(), 0, "bytes_written must default to 0");
        assert_eq!(c.throttled_count(), 0, "throttled_count must default to 0");
        assert_eq!(c.io_errors(), 0, "io_errors must default to 0");
        assert_eq!(
            c.currently_throttled_gauge(),
            0,
            "currently_throttled_gauge must default to 0",
        );
        assert_eq!(
            c.invalid_avail_idx_count(),
            0,
            "invalid_avail_idx_count must default to 0",
        );
    }

    /// `record_read(bytes)` bumps BOTH `reads_completed` AND
    /// `bytes_read` in one call. The pairing is the helper's
    /// reason to exist — a regression that dropped the
    /// `bytes_read.fetch_add(bytes)` line (e.g. a refactor that
    /// inlined just the completion bump) would let the
    /// failure-dump renderer compute a misleading bytes-per-op
    /// average. Pin both increments side-by-side so a half-fix
    /// can't pass.
    ///
    /// Also pins that `record_read` does NOT touch any other
    /// counter — write-side counters, flushes, throttle counters,
    /// io_errors, and the gauge must stay at zero. A regression
    /// that copy-pasted `record_read` from `record_write` and
    /// left the wrong field name would be caught by the
    /// "everything else stays zero" check.
    #[test]
    fn record_read_bumps_completion_and_bytes_in_lockstep() {
        let c = VirtioBlkCounters::default();
        c.record_read(512);
        assert_eq!(
            c.reads_completed(),
            1,
            "first record_read must bump reads_completed to 1",
        );
        assert_eq!(
            c.bytes_read(),
            512,
            "first record_read must add bytes to bytes_read",
        );
        // Second call: counters increment in lockstep.
        c.record_read(1024);
        assert_eq!(
            c.reads_completed(),
            2,
            "second record_read must bump reads_completed to 2",
        );
        assert_eq!(
            c.bytes_read(),
            512 + 1024,
            "second record_read must accumulate bytes",
        );
        // Untouched counters stay at zero.
        assert_eq!(
            c.writes_completed(),
            0,
            "record_read must NOT bump writes_completed",
        );
        assert_eq!(
            c.bytes_written(),
            0,
            "record_read must NOT bump bytes_written",
        );
        assert_eq!(
            c.flushes_completed(),
            0,
            "record_read must NOT bump flushes_completed",
        );
        assert_eq!(
            c.throttled_count(),
            0,
            "record_read must NOT bump throttled_count",
        );
        assert_eq!(c.io_errors(), 0, "record_read must NOT bump io_errors");
        assert_eq!(
            c.currently_throttled_gauge(),
            0,
            "record_read must NOT touch the throttle gauge",
        );
        assert_eq!(
            c.invalid_avail_idx_count(),
            0,
            "record_read must NOT bump invalid_avail_idx_count",
        );
    }

    /// Zero-byte reads are valid: the helper bumps
    /// `reads_completed` even when `bytes == 0`. The contract is
    /// "one completion, n bytes," not "one completion conditional
    /// on n > 0." A regression that gated the completion bump on
    /// `bytes > 0` would mis-count completions in scenarios where
    /// the chain returned zero data (e.g. an EOF-truncated read).
    #[test]
    fn record_read_zero_bytes_still_bumps_completion() {
        let c = VirtioBlkCounters::default();
        c.record_read(0);
        assert_eq!(
            c.reads_completed(),
            1,
            "zero-byte read must still increment reads_completed",
        );
        assert_eq!(
            c.bytes_read(),
            0,
            "zero-byte read must leave bytes_read at 0",
        );
    }

    /// `record_write(bytes)` bumps BOTH `writes_completed` AND
    /// `bytes_written`, mirroring `record_read`. Same paired-
    /// counter rationale: the failure-dump renderer's
    /// bytes-per-write average becomes misleading if either
    /// half is missing.
    #[test]
    fn record_write_bumps_completion_and_bytes_in_lockstep() {
        let c = VirtioBlkCounters::default();
        c.record_write(4096);
        assert_eq!(
            c.writes_completed(),
            1,
            "first record_write must bump writes_completed to 1",
        );
        assert_eq!(
            c.bytes_written(),
            4096,
            "first record_write must add bytes to bytes_written",
        );
        c.record_write(8192);
        assert_eq!(
            c.writes_completed(),
            2,
            "second record_write must bump writes_completed to 2",
        );
        assert_eq!(
            c.bytes_written(),
            4096 + 8192,
            "second record_write must accumulate bytes",
        );
        // Untouched counters stay at zero — pins that
        // record_write doesn't accidentally bump read-side
        // counters via a copy-paste regression.
        assert_eq!(
            c.reads_completed(),
            0,
            "record_write must NOT bump reads_completed",
        );
        assert_eq!(c.bytes_read(), 0, "record_write must NOT bump bytes_read");
        assert_eq!(
            c.flushes_completed(),
            0,
            "record_write must NOT bump flushes_completed",
        );
        assert_eq!(
            c.throttled_count(),
            0,
            "record_write must NOT bump throttled_count",
        );
        assert_eq!(c.io_errors(), 0, "record_write must NOT bump io_errors");
        assert_eq!(
            c.currently_throttled_gauge(),
            0,
            "record_write must NOT touch the throttle gauge",
        );
        assert_eq!(
            c.invalid_avail_idx_count(),
            0,
            "record_write must NOT bump invalid_avail_idx_count",
        );
    }

    /// Zero-byte writes parallel zero-byte reads: the completion
    /// counter advances regardless. A guest issuing a zero-data
    /// write (chain with header + status only and no data
    /// segments) is rejected upstream by the
    /// classify_pre_throttle gate, but the helper itself does
    /// not enforce a non-zero invariant — pinned here so a
    /// future refactor that adds defensive checks at the helper
    /// layer is a deliberate decision, not an accidental
    /// regression of the "one completion, n bytes" contract.
    #[test]
    fn record_write_zero_bytes_still_bumps_completion() {
        let c = VirtioBlkCounters::default();
        c.record_write(0);
        assert_eq!(
            c.writes_completed(),
            1,
            "zero-byte write must still increment writes_completed",
        );
        assert_eq!(
            c.bytes_written(),
            0,
            "zero-byte write must leave bytes_written at 0",
        );
    }

    /// `record_flush()` bumps `flushes_completed` and ONLY
    /// `flushes_completed`. Distinct from read/write because
    /// flush has no associated byte count — there's no paired
    /// counter to keep in lockstep, only a single completion.
    /// A regression that conflated flush with write (e.g. a
    /// refactor that routed flush through `record_write(0)`)
    /// would surface here as `writes_completed == 1` instead of
    /// `flushes_completed == 1`.
    #[test]
    fn record_flush_bumps_only_flushes_completed() {
        let c = VirtioBlkCounters::default();
        c.record_flush();
        assert_eq!(
            c.flushes_completed(),
            1,
            "record_flush must bump flushes_completed to 1",
        );
        c.record_flush();
        c.record_flush();
        assert_eq!(
            c.flushes_completed(),
            3,
            "three record_flush calls must accumulate to 3",
        );
        // Every other counter stays at zero — flush has no
        // paired bytes counter and must not splash onto any
        // other field.
        assert_eq!(
            c.reads_completed(),
            0,
            "record_flush must NOT bump reads_completed",
        );
        assert_eq!(c.bytes_read(), 0, "record_flush must NOT bump bytes_read");
        assert_eq!(
            c.writes_completed(),
            0,
            "record_flush must NOT bump writes_completed",
        );
        assert_eq!(
            c.bytes_written(),
            0,
            "record_flush must NOT bump bytes_written",
        );
        assert_eq!(
            c.throttled_count(),
            0,
            "record_flush must NOT bump throttled_count",
        );
        assert_eq!(c.io_errors(), 0, "record_flush must NOT bump io_errors");
        assert_eq!(
            c.currently_throttled_gauge(),
            0,
            "record_flush must NOT touch the throttle gauge",
        );
        assert_eq!(
            c.invalid_avail_idx_count(),
            0,
            "record_flush must NOT bump invalid_avail_idx_count",
        );
    }

    /// `record_throttle_pending_inc()` bumps the live gauge by
    /// exactly one per call. The helper itself is NOT idempotent
    /// — back-to-back calls increment twice (gauge 0→1→2). The
    /// production caller (`drain_bracket_impl`) gates each
    /// invocation on the per-worker `currently_stalled` flag's
    /// false→true transition; the helper relies on the caller
    /// to enforce idempotence and faithfully bumps every time
    /// it's invoked. Pinning this distinction matters: a
    /// regression that pushed the flag-gate INTO the helper
    /// would break the helper's contract with cross-cutting
    /// callers (e.g. a future test seam that simulates back-
    /// to-back stalls without going through the production
    /// gate). The events-vs-requests semantic is a property of
    /// the CALLER (which only invokes `record_throttle_pending_inc`
    /// on transitions), NOT of the helper.
    ///
    /// The complementary "no double-inc on re-stall via the
    /// production gate" invariant is pinned by
    /// `currently_throttled_gauge_no_double_inc_on_re_stall` in
    /// tests_atomics.rs which exercises the full
    /// drain_bracket_impl path.
    #[test]
    fn record_throttle_pending_inc_increments_each_call() {
        let c = VirtioBlkCounters::default();
        c.record_throttle_pending_inc();
        assert_eq!(
            c.currently_throttled_gauge(),
            1,
            "first inc must bump gauge from 0 to 1",
        );
        // Helper is not idempotent — the production caller's
        // currently_stalled flag prevents the second call from
        // happening, but the helper itself does increment again
        // when invoked.
        c.record_throttle_pending_inc();
        assert_eq!(
            c.currently_throttled_gauge(),
            2,
            "second inc must bump gauge from 1 to 2 (helper itself \
                 is not idempotent — caller must gate)",
        );
        c.record_throttle_pending_inc();
        assert_eq!(
            c.currently_throttled_gauge(),
            3,
            "third inc must bump gauge from 2 to 3",
        );
        // The other counters stay at zero — gauge ops must not
        // splash onto throttled_count (events) or any other
        // field. throttled_count is bumped by `record_throttled`,
        // a SEPARATE helper.
        assert_eq!(
            c.throttled_count(),
            0,
            "record_throttle_pending_inc must NOT bump throttled_count \
                 (events vs gauge are separate counters with separate helpers)",
        );
        assert_eq!(
            c.reads_completed(),
            0,
            "record_throttle_pending_inc must NOT bump reads_completed",
        );
        assert_eq!(
            c.io_errors(),
            0,
            "record_throttle_pending_inc must NOT bump io_errors",
        );
    }

    /// `record_throttle_pending_dec()` decrements the gauge by
    /// one when it is non-zero, mirror of inc.
    #[test]
    fn record_throttle_pending_dec_decrements_when_positive() {
        let c = VirtioBlkCounters::default();
        c.record_throttle_pending_inc();
        c.record_throttle_pending_inc();
        c.record_throttle_pending_inc();
        assert_eq!(c.currently_throttled_gauge(), 3, "pre-cond: gauge at 3");
        c.record_throttle_pending_dec();
        assert_eq!(
            c.currently_throttled_gauge(),
            2,
            "first dec must drop gauge from 3 to 2",
        );
        c.record_throttle_pending_dec();
        assert_eq!(
            c.currently_throttled_gauge(),
            1,
            "second dec must drop gauge from 2 to 1",
        );
        c.record_throttle_pending_dec();
        assert_eq!(
            c.currently_throttled_gauge(),
            0,
            "third dec must drop gauge from 1 to 0",
        );
    }

    /// `record_throttle_pending_dec()` SATURATES at zero. The
    /// implementation uses `fetch_update(|v| v.checked_sub(1))`
    /// — if the gauge is already 0, the update returns `Err`
    /// and the helper drops the result via `let _`. A regression
    /// that swapped `checked_sub` for plain `fetch_sub(1)` would
    /// wrap to `u64::MAX` and the failure-dump renderer would
    /// surface a 17-exabyte "currently stalled" reading.
    ///
    /// Pin the saturating contract: dec on an already-zero gauge
    /// MUST leave the gauge at 0, not wrap to u64::MAX.
    #[test]
    fn record_throttle_pending_dec_saturates_at_zero() {
        let c = VirtioBlkCounters::default();
        // Gauge starts at 0; multiple decs must NOT wrap.
        c.record_throttle_pending_dec();
        assert_eq!(
            c.currently_throttled_gauge(),
            0,
            "dec on a zero gauge MUST saturate at 0, not wrap to u64::MAX \
                 (regression: fetch_sub instead of fetch_update + checked_sub)",
        );
        // Repeated dec stays at 0 — the failure mode is "wraps
        // to u64::MAX on the first underflowing dec," so multiple
        // decs each pin that the saturate-at-zero contract holds
        // across consecutive calls.
        for i in 0..5 {
            c.record_throttle_pending_dec();
            assert_eq!(
                c.currently_throttled_gauge(),
                0,
                "dec on a zero gauge must stay 0 across {} repeated calls",
                i + 1,
            );
        }
    }

    /// Inc-then-dec pair returns the gauge to zero. Pins the
    /// matching-pair invariant the production caller depends on:
    /// every chain that stalls (inc) and later succeeds (dec)
    /// must net to a delta of zero on the gauge. A regression
    /// to the inc/dec arithmetic that failed to undo the inc
    /// would surface as a non-zero residual gauge after the
    /// pair.
    #[test]
    fn record_throttle_pending_inc_then_dec_nets_to_zero() {
        let c = VirtioBlkCounters::default();
        c.record_throttle_pending_inc();
        c.record_throttle_pending_dec();
        assert_eq!(
            c.currently_throttled_gauge(),
            0,
            "inc-then-dec must net to 0 on the gauge",
        );
        // Also check N inc / N dec for N > 1 — pins that the
        // counter-style accounting holds regardless of pair
        // count.
        for _ in 0..10 {
            c.record_throttle_pending_inc();
        }
        assert_eq!(c.currently_throttled_gauge(), 10, "10 incs → gauge=10");
        for _ in 0..10 {
            c.record_throttle_pending_dec();
        }
        assert_eq!(
            c.currently_throttled_gauge(),
            0,
            "10 incs + 10 decs must net to 0",
        );
    }

    /// `record_io_error()` bumps `io_errors` and ONLY `io_errors`.
    /// The events-counter contract is at the call sites (a single
    /// hostile chain can produce multiple bumps if it trips
    /// several gates in sequence — pinned by the doc comment on
    /// `record_io_error`); the helper itself faithfully bumps
    /// per call. Pin that the bump lands on the right field and
    /// no other counter is touched: a regression that copy-pasted
    /// the helper from `record_throttled` and left the wrong
    /// field name would surface as `throttled_count == 1` in
    /// place of the expected `io_errors == 1`.
    #[test]
    fn record_io_error_increments_only_io_errors() {
        let c = VirtioBlkCounters::default();
        c.record_io_error();
        assert_eq!(
            c.io_errors(),
            1,
            "first record_io_error must bump io_errors to 1",
        );
        c.record_io_error();
        c.record_io_error();
        assert_eq!(
            c.io_errors(),
            3,
            "three record_io_error calls must accumulate to 3 \
                 (events counter, no per-request dedup)",
        );
        // Every other counter stays at zero — io_errors must not
        // splash onto throttled_count, gauges, or completion
        // counters.
        assert_eq!(
            c.reads_completed(),
            0,
            "record_io_error must NOT bump reads_completed",
        );
        assert_eq!(
            c.writes_completed(),
            0,
            "record_io_error must NOT bump writes_completed",
        );
        assert_eq!(
            c.flushes_completed(),
            0,
            "record_io_error must NOT bump flushes_completed",
        );
        assert_eq!(c.bytes_read(), 0, "record_io_error must NOT bump bytes_read");
        assert_eq!(
            c.bytes_written(),
            0,
            "record_io_error must NOT bump bytes_written",
        );
        assert_eq!(
            c.throttled_count(),
            0,
            "record_io_error must NOT bump throttled_count \
                 (events-vs-events distinction — IO errors and \
                 throttle stalls are separately classified)",
        );
        assert_eq!(
            c.currently_throttled_gauge(),
            0,
            "record_io_error must NOT touch the throttle gauge",
        );
        assert_eq!(
            c.invalid_avail_idx_count(),
            0,
            "record_io_error must NOT bump invalid_avail_idx_count",
        );
    }

    /// `record_throttled()` bumps `throttled_count` and ONLY
    /// `throttled_count`. Per-event counter, not per-request:
    /// a single chain that stalls multiple times produces
    /// multiple bumps. The events-vs-requests distinction lives
    /// at the CALLER (drain_bracket_impl); the helper itself is
    /// just an unconditional bump. Pin parity with the other
    /// "single-counter" helpers — io_errors, flushes — so a
    /// copy-paste regression that wrote to the wrong field
    /// surfaces here.
    #[test]
    fn record_throttled_increments_only_throttled_count() {
        let c = VirtioBlkCounters::default();
        c.record_throttled();
        assert_eq!(
            c.throttled_count(),
            1,
            "first record_throttled must bump throttled_count to 1",
        );
        c.record_throttled();
        assert_eq!(
            c.throttled_count(),
            2,
            "second record_throttled must bump throttled_count to 2 \
                 (events counter — same chain re-stalling produces \
                 multiple bumps in production)",
        );
        // Crucially, the gauge is NOT touched — gauge has its own
        // helper (record_throttle_pending_inc/dec). A regression
        // that conflated the two would surface as gauge != 0.
        assert_eq!(
            c.currently_throttled_gauge(),
            0,
            "record_throttled (events counter) must NOT touch \
                 currently_throttled_gauge (live gauge — separate helper)",
        );
        // Other counters stay at zero.
        assert_eq!(c.io_errors(), 0, "record_throttled must NOT bump io_errors");
        assert_eq!(
            c.reads_completed(),
            0,
            "record_throttled must NOT bump reads_completed",
        );
        assert_eq!(
            c.writes_completed(),
            0,
            "record_throttled must NOT bump writes_completed",
        );
        assert_eq!(
            c.flushes_completed(),
            0,
            "record_throttled must NOT bump flushes_completed",
        );
        assert_eq!(
            c.invalid_avail_idx_count(),
            0,
            "record_throttled must NOT bump invalid_avail_idx_count",
        );
    }

    /// `record_invalid_avail_idx()` bumps `invalid_avail_idx_count`
    /// and ONLY that field. Per-event counter; the production
    /// caller's queue-poison flag short-circuits subsequent kicks
    /// so one guest fault produces exactly one bump regardless of
    /// notification count. The helper itself is just an
    /// unconditional bump; the no-double-bump invariant is a
    /// property of the CALLER (gated on queue_poisoned), pinned
    /// by `inflated_avail_idx_poisons_queue_no_livelock` and
    /// `poisoned_queue_clears_on_reset` in tests_atomics.rs.
    #[test]
    fn record_invalid_avail_idx_increments_only_that_field() {
        let c = VirtioBlkCounters::default();
        c.record_invalid_avail_idx();
        assert_eq!(
            c.invalid_avail_idx_count(),
            1,
            "first record_invalid_avail_idx must bump counter to 1",
        );
        c.record_invalid_avail_idx();
        assert_eq!(
            c.invalid_avail_idx_count(),
            2,
            "second record_invalid_avail_idx must bump counter to 2 \
                 (helper itself does not enforce single-bump; the \
                 caller's poison gate does)",
        );
        // Every other counter stays at zero.
        assert_eq!(
            c.io_errors(),
            0,
            "record_invalid_avail_idx must NOT bump io_errors \
                 (separate event class — guest spec violation \
                 vs IO failure)",
        );
        assert_eq!(
            c.throttled_count(),
            0,
            "record_invalid_avail_idx must NOT bump throttled_count",
        );
        assert_eq!(
            c.currently_throttled_gauge(),
            0,
            "record_invalid_avail_idx must NOT touch the throttle gauge",
        );
        assert_eq!(
            c.reads_completed(),
            0,
            "record_invalid_avail_idx must NOT bump reads_completed",
        );
        assert_eq!(
            c.writes_completed(),
            0,
            "record_invalid_avail_idx must NOT bump writes_completed",
        );
        assert_eq!(
            c.flushes_completed(),
            0,
            "record_invalid_avail_idx must NOT bump flushes_completed",
        );
    }
}
