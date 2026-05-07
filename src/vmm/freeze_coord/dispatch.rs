//! Per-frame dispatch for the virtio-console port-1 bulk TLV stream.
//!
//! The freeze coordinator's TOKEN_TX epoll branch drives this module:
//! after `bulk_assembler.feed(...)` returns a [`BulkMessages`] vec, the
//! coordinator iterates each [`BulkMessage`] through
//! [`dispatch_bulk_message`] and either pushes a verdict-bearing
//! [`crate::vmm::wire::ShmEntry`] into the run-wide bucket OR triggers
//! one of three coordinator-internal side effects (kill flag + eventfd
//! flip on `SchedExit`, sys-rdy eventfd fire-once on `SysRdy`, decode
//! and stash for later dispatch on `SnapshotRequest`).
//!
//! Splitting the dispatch out of the run-loop closure body lets test
//! code drive arbitrary CRC-mangled frame sequences against a pure
//! function — no VM boot required, no Arc plumbing beyond the sinks
//! a test wants to observe. Production behaviour is byte-for-byte
//! preserved; the only logic change relative to the inline code is
//! the function boundary.
//!
//! Hostile-guest discipline is identical to the inline arms: every
//! CRC-bearing promotion gates on `msg.crc_ok`, the SysRdy promotion
//! is fire-once via [`Option::take`], unknown msg_type entries warn-
//! and-drop without polluting the verdict, and SnapshotReply / other
//! coordinator-internal variants are filtered via
//! [`crate::vmm::wire::MsgType::is_coordinator_internal`].

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use vmm_sys_util::eventfd::EventFd;

use super::snapshot::decode_snapshot_request;
use super::state::SnapshotRequest;

/// Aggregate of the coordinator-side sinks the TOKEN_TX dispatch can
/// touch. Bundling them keeps the [`dispatch_bulk_message`] signature
/// readable (one `&mut` arg instead of four) and makes the test
/// fixture explicit: a test sets up exactly the sinks it wants to
/// observe, runs the dispatch, then asserts the post-state.
///
/// `sys_rdy_evt` is `&mut Option<...>` so [`Option::take`] can fire
/// the SysRdy eventfd exactly once per coordinator lifetime — the
/// inline production code uses the same `Option::take` to drop the
/// host-side handle after the first promotion, and the function-
/// extracted form preserves that behaviour by mutating through the
/// passed reference.
pub(super) struct BulkDispatchSinks<'a> {
    /// Run-wide kill flag flipped on a CRC-valid `MSG_TYPE_SCHED_EXIT`.
    /// Loaded by the BSP run loop, the watchdog, and the freeze
    /// coordinator's outer `while` predicate.
    pub kill: &'a Arc<AtomicBool>,
    /// Wake fd paired with [`Self::kill`]. Written `1` immediately
    /// after the AtomicBool flip so any consumer blocked in
    /// `epoll_wait` returns within microseconds rather than waiting
    /// up to one full poll interval.
    pub kill_evt: &'a Arc<EventFd>,
    /// Boot-complete signal. Promoted exactly once on the first
    /// CRC-valid empty-payload `MSG_TYPE_SYS_RDY` frame; the
    /// `Option::take` retains the host-side handle until that point
    /// and drops it after firing so subsequent SYS_RDY frames (a
    /// hostile guest could in principle resend) skip the eventfd
    /// write.
    pub sys_rdy_evt: &'a mut Option<Arc<EventFd>>,
    /// Per-iteration accumulator for decoded
    /// `MSG_TYPE_SNAPSHOT_REQUEST` frames. Drained later in the run-
    /// loop body where `freeze_and_capture` /
    /// `arm_user_watchpoint` are in scope. CRC-bad frames and
    /// malformed payloads (size mismatch, KIND_NONE, request_id == 0)
    /// never reach this Vec — [`decode_snapshot_request`] returns
    /// `None` and the entry is dropped without observable side effect.
    pub snapshot_requests_pending: &'a mut Vec<SnapshotRequest>,
    /// Guest-reported `phys_base + 1`. Stored by the KERN_ADDRS arm
    /// so the monitor thread can pick it up via Acquire load.
    pub kern_phys_base: &'a Arc<std::sync::atomic::AtomicU64>,
    /// Watchdog reset atomic + workload duration. SCENARIO_START
    /// stores `(now - run_start + duration).as_nanos()` so the
    /// watchdog starts the workload clock from scenario start, not
    /// from boot or SYS_RDY.
    pub watchdog_reset: Option<(
        &'a std::sync::atomic::AtomicU64,
        std::time::Duration,
        std::time::Instant,
    )>,
}

/// Classify and dispatch a single [`BulkMessage`] from the port-1
/// TLV stream. Returns the verdict-bearing [`crate::vmm::wire::ShmEntry`]
/// to push into the run-wide bucket, or `None` for coordinator-
/// internal frames whose only effect was on `sinks`.
///
/// # Promotion gates (hostile-guest defence)
///
/// * `MSG_TYPE_SCHED_EXIT` flips `kill` and writes `kill_evt` ONLY
///   when `msg.crc_ok`. A torn frame would otherwise let a hostile
///   guest force a false early exit. CRC-bad SchedExit also does NOT
///   land in the verdict bucket — the per-type contract requires
///   `crc_ok` for SchedExit to be observable.
/// * `MSG_TYPE_SYS_RDY` fires its eventfd ONLY when `msg.crc_ok &&
///   msg.payload.is_empty()`. The empty-payload gate is the safety
///   net against a hostile guest tacking smuggle bytes onto a SysRdy
///   frame past the [`crate::vmm::wire::MsgType::is_coordinator_internal`]
///   filter. Promotion is fire-once via [`Option::take`].
/// * `MSG_TYPE_SNAPSHOT_REQUEST` decodes via [`decode_snapshot_request`]
///   ONLY when `msg.crc_ok`. The decoder additionally rejects
///   `request_id == 0`, `kind == SNAPSHOT_KIND_NONE`, and any
///   payload whose size does not match the typed wire layout.
/// * Every other variant: pushes verbatim if not coordinator-
///   internal, drops silently if it is. Unknown msg_type values
///   warn-and-drop so a future guest variant cannot synthesise a
///   phantom verdict entry on the host.
///
/// # CRC handling on verdict-bearing arms
///
/// Non-SchedExit verdict frames (Stimulus, ScenarioStart,
/// ScenarioEnd, Exit, TestResult, Crash, PayloadMetrics,
/// RawPayloadOutput, Profraw, Stdout, Stderr, SchedLog, Lifecycle,
/// ExecExit, Dmesg, ProbeOutput) accumulate even when `crc_ok` is
/// false — the host-side consumers filter on per-type contract.
/// SchedExit is the lone exception: its kill-flag promotion makes a
/// torn-frame leak load-bearing for a hostile guest, so we gate the
/// bucket push on the same `crc_ok` flag the promotion checks.
pub(super) fn dispatch_bulk_message(
    msg: &crate::vmm::bulk::BulkMessage,
    sinks: &mut BulkDispatchSinks<'_>,
) -> Option<crate::vmm::wire::ShmEntry> {
    let kind = crate::vmm::wire::MsgType::from_wire(msg.msg_type);
    match kind {
        Some(crate::vmm::wire::MsgType::SchedExit) => {
            // Promote a guest-side SCHED_EXIT into the run-wide kill
            // flag so the BSP loop and the watchdog exit promptly
            // instead of running until the watchdog deadline. CRC
            // failures DO NOT promote — a torn frame would otherwise
            // let a hostile guest force a false early exit.
            if msg.crc_ok {
                sinks.kill.store(true, Ordering::Release);
                // EFD_NONBLOCK on a freshly-created eventfd never
                // legitimately fails; log unconditionally so a future
                // regression (e.g. the eventfd was closed by another
                // owner) surfaces in the host log instead of silently
                // swallowing the kill edge.
                if let Err(e) = sinks.kill_evt.write(1) {
                    tracing::warn!(
                        err = %e,
                        "freeze_coord: kill_evt write on SCHED_EXIT \
                         promotion failed; the kill AtomicBool above is \
                         still authoritative"
                    );
                }
            }
            // SchedExit is verdict data — bucket only on CRC-valid
            // frames so a torn or hostile-guest tag never surfaces as
            // a phantom verdict entry in `BulkDrainResult`. The
            // promotion gate above already short-circuits on
            // crc_ok=false; mirror the same discipline here so the
            // verdict-side filter is not deferred to a downstream
            // consumer that does not exist.
            if msg.crc_ok {
                Some(crate::vmm::wire::ShmEntry {
                    msg_type: msg.msg_type,
                    payload: msg.payload.to_vec(),
                    crc_ok: msg.crc_ok,
                })
            } else {
                None
            }
        }
        Some(crate::vmm::wire::MsgType::SysRdy) => {
            // Promote a CRC-valid, empty-payload SysRdy into the
            // monitor's boot-complete eventfd so the monitor thread's
            // pre-sample `epoll_wait` returns within microseconds
            // rather than waiting for the 5 s fallback. CRC failures
            // DO NOT promote — a torn frame would let a hostile guest
            // forge a fake boot signal that races ahead of
            // `setup_per_cpu_areas` / KASLR. The `Option::take` makes
            // promotion fire-once: a resent SysRdy skips the eventfd
            // write so the counter does not pump. SysRdy must carry
            // no payload — a hostile guest tacking bytes on would
            // otherwise smuggle data past the
            // is_coordinator_internal filter; this strict shape gate
            // is the safety net.
            if msg.crc_ok
                && msg.payload.is_empty()
                && let Some(evt) = sinks.sys_rdy_evt.take()
                && let Err(e) = evt.write(1)
            {
                tracing::warn!(
                    err = %e,
                    "freeze_coord: sys_rdy write failed; monitor will \
                     rely on kill_evt or 5 s timeout to leave its \
                     pre-sample wait"
                );
            }
            // SysRdy is coordinator-internal — do NOT bucket.
            None
        }
        _ if msg.msg_type == crate::vmm::wire::MSG_TYPE_KERN_ADDRS => {
            // Payload carries phys_base + 1 (biased to avoid
            // the 0 sentinel). Subtract 1 to recover.
            if msg.crc_ok && msg.payload.len() >= 8 {
                let biased = u64::from_le_bytes(msg.payload[..8].try_into().unwrap_or([0; 8]));
                if biased != 0 {
                    sinks
                        .kern_phys_base
                        .store(biased, std::sync::atomic::Ordering::Release);
                }
            }
            None
        }
        Some(crate::vmm::wire::MsgType::SnapshotRequest) => {
            // Decode and stash a CRC-valid SnapshotRequest for
            // dispatch later in this iteration's body.
            // `freeze_and_capture` / `thaw_and_barrier` /
            // `arm_user_watchpoint` are not in scope here. CRC-bad
            // frames are ignored (a torn frame would otherwise let a
            // hostile guest force a capture). Malformed payloads
            // (size mismatch, KIND_NONE, request_id == 0) decode to
            // `None` and drop.
            if msg.crc_ok
                && let Some(req) = decode_snapshot_request(&msg.payload[..])
            {
                sinks.snapshot_requests_pending.push(req);
            }
            // SnapshotRequest is coordinator-internal — its matching
            // reply ships over port-1 RX. Do NOT bucket.
            None
        }
        Some(crate::vmm::wire::MsgType::ScenarioStart) => {
            if msg.crc_ok
                && let Some((reset_ns, duration, run_start)) = sinks.watchdog_reset.as_ref()
            {
                let elapsed = run_start.elapsed();
                let target_ns = elapsed.as_nanos().saturating_add(duration.as_nanos());
                let encoded = u64::try_from(target_ns).unwrap_or(u64::MAX).max(1);
                reset_ns.store(encoded, std::sync::atomic::Ordering::Release);
            }
            Some(crate::vmm::wire::ShmEntry {
                msg_type: msg.msg_type,
                payload: msg.payload.to_vec(),
                crc_ok: msg.crc_ok,
            })
        }
        Some(crate::vmm::wire::MsgType::ScenarioEnd) => {
            if msg.crc_ok
                && let Some((reset_ns, duration, run_start)) = sinks.watchdog_reset.as_ref()
            {
                let elapsed = run_start.elapsed();
                let target_ns = elapsed
                    .as_nanos()
                    .saturating_add(duration.as_nanos());
                let encoded = u64::try_from(target_ns).unwrap_or(u64::MAX).max(1);
                reset_ns.store(encoded, std::sync::atomic::Ordering::Release);
            }
            Some(crate::vmm::wire::ShmEntry {
                msg_type: msg.msg_type,
                payload: msg.payload.to_vec(),
                crc_ok: msg.crc_ok,
            })
        }
        Some(other) if !other.is_coordinator_internal() => {
            // Every other typed verdict-bearing variant (Stimulus,
            // ScenarioEnd, Exit, TestResult, Crash,
            // PayloadMetrics, RawPayloadOutput, Profraw, Stdout,
            // Stderr, SchedLog, Lifecycle, ExecExit, Dmesg,
            // ProbeOutput) accumulates into the bucket verbatim.
            // SnapshotReply is host→guest only and is filtered out
            // by the `is_coordinator_internal` guard above; a guest
            // TX frame stamped with that tag falls through to the
            // `Some(_)` arm below and is dropped silently. CRC-bad
            // entries still land here — the host-side consumers
            // filter on `crc_ok` per their own per-type contract.
            Some(crate::vmm::wire::ShmEntry {
                msg_type: msg.msg_type,
                payload: msg.payload.to_vec(),
                crc_ok: msg.crc_ok,
            })
        }
        Some(_) => {
            // Coordinator-internal variant with no inline side-effect
            // arm above (e.g. a future is_coordinator_internal entry).
            // Drop silently — by definition this variant should not
            // surface as a verdict entry, and any side effect must be
            // added here explicitly.
            None
        }
        None => {
            // Unknown msg_type — log once and drop. A future guest
            // variant the host does not know about would otherwise
            // produce a phantom verdict entry.
            tracing::warn!(
                msg_type = msg.msg_type,
                len = msg.payload.len(),
                crc_ok = msg.crc_ok,
                "freeze_coord: unknown MSG_TYPE_* on bulk port; dropping"
            );
            None
        }
    }
}
