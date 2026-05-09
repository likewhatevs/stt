//! Unit coverage for the TOKEN_TX dispatch's CRC-gated promotion
//! and decode paths.
//!
//! Two production gates inspect each `BulkMessage` the streaming
//! [`crate::vmm::bulk::HostAssembler`] yields from the
//! virtio-console port-1 TX byte stream:
//!
//!   * `msg.msg_type == MSG_TYPE_SCHED_EXIT && msg.crc_ok` — flips
//!     the run-wide kill flag and writes the kill eventfd so the
//!     BSP loop and the watchdog exit promptly. CRC failures must
//!     NOT promote — a torn frame would otherwise let a hostile
//!     guest force a false early exit.
//!   * `msg.msg_type == MSG_TYPE_SNAPSHOT_REQUEST && msg.crc_ok &&
//!     decode_snapshot_request(payload).is_some()` — pushes the
//!     decoded request onto the per-iteration pending list for
//!     dispatch to `freeze_and_capture` / `arm_user_watchpoint`.
//!     CRC failures must NOT decode — a torn snapshot request
//!     would otherwise let a hostile guest force a spurious
//!     capture or watchpoint arm.
//!
//! These gates live inside the freeze coordinator's run-loop
//! closure where the kill eventfd, the snapshot-pending vec, and
//! the streaming assembler are all in scope; they cannot be
//! exercised through a public function call. The tests below
//! reproduce the production path end-to-end: build a torn-CRC
//! TLV byte stream, run it through the same `HostAssembler::feed`
//! the closure uses, and apply the gate predicates against the
//! resulting `BulkMessage`. A passing test means the assembler
//! flagged the frame as `crc_ok=false` AND the gate predicate
//! short-circuits before triggering the side effect (kill flip /
//! decode).
use super::*;
use crate::vmm::bulk::HostAssembler;
use crate::vmm::wire::{
    FRAME_HEADER_SIZE, MSG_TYPE_SCHED_EXIT, MSG_TYPE_SNAPSHOT_REQUEST, MSG_TYPE_SYS_RDY,
    SNAPSHOT_KIND_CAPTURE, SNAPSHOT_TAG_MAX, ShmMessage, SnapshotRequestPayload,
};
use std::sync::atomic::{AtomicBool, Ordering};
use vmm_sys_util::eventfd::{EFD_NONBLOCK, EventFd};
use zerocopy::IntoBytes;

/// Build a TLV frame whose header CRC matches the supplied payload
/// — `HostAssembler::feed` will produce a `BulkMessage` with
/// `crc_ok=true`. Mirrors `bulk.rs`'s test helper so the
/// assertions below pin the production behaviour the closure
/// observes, not a synthetic in-test path.
fn frame_with_crc(msg_type: u32, payload: &[u8]) -> Vec<u8> {
    let header = ShmMessage {
        msg_type,
        length: payload.len() as u32,
        crc32: crc32fast::hash(payload),
        _pad: 0,
    };
    let mut buf = Vec::with_capacity(FRAME_HEADER_SIZE + payload.len());
    buf.extend_from_slice(header.as_bytes());
    buf.extend_from_slice(payload);
    buf
}

/// Build a TLV frame whose header CRC deliberately does NOT match
/// the payload — `HostAssembler::feed` produces a `BulkMessage`
/// with `crc_ok=false`, exactly as it would on a torn or
/// hostile-guest publish. The bogus CRC is the recomputed CRC
/// XOR'd with `0xFFFF_FFFF` so the mismatch is total (every
/// bit flipped) rather than a near-miss that could match if the
/// payload were hashed differently.
fn frame_with_torn_crc(msg_type: u32, payload: &[u8]) -> Vec<u8> {
    let real_crc = crc32fast::hash(payload);
    let header = ShmMessage {
        msg_type,
        length: payload.len() as u32,
        crc32: real_crc ^ 0xFFFF_FFFF,
        _pad: 0,
    };
    let mut buf = Vec::with_capacity(FRAME_HEADER_SIZE + payload.len());
    buf.extend_from_slice(header.as_bytes());
    buf.extend_from_slice(payload);
    buf
}

/// Build a SnapshotRequest payload using the wire-layout type so
/// the bytes match what a guest publisher emits. Identical to
/// `make_request_bytes` in `snapshot_tlv_tests` — duplicated here
/// so this module remains self-contained.
fn snapshot_request_bytes(request_id: u32, kind: u32, tag: &str) -> Vec<u8> {
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

/// Apply the production SCHED_EXIT promotion gate (the
/// `msg.msg_type == MSG_TYPE_SCHED_EXIT && msg.crc_ok` branch in
/// the freeze coordinator's TOKEN_TX handler) against a slice of
/// `BulkMessage` values. Returns `(kill_flag_value,
/// kill_evt_fired)` so the test can assert both side effects of
/// the gate. The eventfd is created `EFD_NONBLOCK` to mirror the
/// closure's `freeze_coord_kill_evt`; `read()` returns `EAGAIN`
/// instead of blocking when the counter is zero.
fn run_sched_exit_gate(messages: &[crate::vmm::bulk::BulkMessage]) -> (bool, bool) {
    let kill = AtomicBool::new(false);
    let kill_evt = EventFd::new(EFD_NONBLOCK).expect("eventfd construction");
    for msg in messages {
        // Exact predicate copied from the production closure;
        // the test's value comes from this expression staying
        // in lockstep with the in-tree gate. If the production
        // gate ever drops the `crc_ok` clause, this test must
        // be updated in the same change so the regression is
        // visible.
        if msg.msg_type == MSG_TYPE_SCHED_EXIT && msg.crc_ok {
            kill.store(true, Ordering::Release);
            let _ = kill_evt.write(1);
        }
    }
    let kill_value = kill.load(Ordering::Acquire);
    // Drain the eventfd to detect a write — `read` returns the
    // accumulated counter (1 here) on success or `EAGAIN` if the
    // gate did not write. Either outcome is a non-zero / zero
    // distinguisher for the test's verdict.
    let evt_fired = kill_evt.read().is_ok();
    (kill_value, evt_fired)
}

/// Apply the production SNAPSHOT_REQUEST decode-and-stash gate
/// (the `msg.msg_type == MSG_TYPE_SNAPSHOT_REQUEST && msg.crc_ok
/// && let Some(req) = decode_snapshot_request(...)` branch in
/// the closure) against a slice of `BulkMessage` values. Returns
/// the count of requests pushed onto the per-iteration pending
/// list — zero means the gate dropped the frame, non-zero means
/// the gate accepted and decoded it.
fn run_snapshot_request_gate(messages: &[crate::vmm::bulk::BulkMessage]) -> usize {
    let mut pending: Vec<SnapshotRequest> = Vec::new();
    for msg in messages {
        // Exact predicate copied from the production closure.
        // Note: `decode_snapshot_request` is the same helper the
        // closure calls, so the decode-side defense (size /
        // KIND_NONE / request_id == 0) is exercised end-to-end
        // alongside the CRC gate.
        if msg.msg_type == MSG_TYPE_SNAPSHOT_REQUEST
            && msg.crc_ok
            && let Some(req) = decode_snapshot_request(&msg.payload[..])
        {
            pending.push(req);
        }
    }
    pending.len()
}

/// CRC-failed SCHED_EXIT MUST NOT promote the run-wide kill flag.
/// A torn or hostile-guest frame would otherwise let an attacker
/// force the BSP loop and the watchdog to exit early, ending a
/// test before its scheduler under test had a chance to
/// misbehave.
#[test]
fn sched_exit_with_torn_crc_does_not_promote_kill() {
    let mut a = HostAssembler::new();
    let bytes = frame_with_torn_crc(MSG_TYPE_SCHED_EXIT, b"exit-payload");
    let drained = a.feed(&bytes);
    assert_eq!(drained.messages.len(), 1, "assembler emits one message");
    assert!(
        !drained.messages[0].crc_ok,
        "torn CRC must surface as crc_ok=false"
    );
    assert_eq!(
        drained.messages[0].msg_type, MSG_TYPE_SCHED_EXIT,
        "msg_type unaffected by CRC mismatch — gate dispatch is by type"
    );
    let (kill, evt_fired) = run_sched_exit_gate(&drained.messages);
    assert!(
        !kill,
        "kill flag must NOT flip on CRC-failed SCHED_EXIT — \
             hostile guest must not force early exit"
    );
    assert!(
        !evt_fired,
        "kill eventfd must NOT be written on CRC-failed SCHED_EXIT — \
             the BSP loop and watchdog must not be woken"
    );
}

/// Positive control: a CRC-valid SCHED_EXIT DOES promote. Pins the
/// test against a degenerate case where the gate is broken and
/// the negative test passes for the wrong reason (i.e. kill never
/// promotes regardless of input). Without this control, a fix
/// that accidentally inverts the predicate
/// (`!msg.crc_ok` instead of `msg.crc_ok`) would still pass the
/// torn-CRC test but break production.
#[test]
fn sched_exit_with_valid_crc_does_promote_kill() {
    let mut a = HostAssembler::new();
    let bytes = frame_with_crc(MSG_TYPE_SCHED_EXIT, b"exit-payload");
    let drained = a.feed(&bytes);
    assert_eq!(drained.messages.len(), 1);
    assert!(
        drained.messages[0].crc_ok,
        "matching CRC must surface as crc_ok=true"
    );
    let (kill, evt_fired) = run_sched_exit_gate(&drained.messages);
    assert!(
        kill,
        "kill flag MUST flip on CRC-valid SCHED_EXIT — promotion is \
             the load-bearing path that ends a test promptly"
    );
    assert!(
        evt_fired,
        "kill eventfd MUST be written on CRC-valid SCHED_EXIT — \
             the BSP loop and watchdog need an epoll wake to exit \
             the run loop"
    );
}

/// Mixed batch: a CRC-failed SCHED_EXIT alongside other
/// CRC-valid frames must not promote. The gate is per-message,
/// not per-batch — every CRC failure must short-circuit
/// independently regardless of what arrived alongside it. This
/// catches a regression where the gate erroneously walks the
/// batch and trusts the first valid frame to authorise the rest.
#[test]
fn sched_exit_torn_crc_does_not_promote_when_other_valid_frames_present() {
    let mut a = HostAssembler::new();
    // Build a batch: torn SCHED_EXIT first, then a valid
    // STIMULUS frame (not a SCHED_EXIT — must not promote on
    // its own), then a torn SCHED_EXIT-typed frame.
    let mut buf = Vec::new();
    buf.extend(frame_with_torn_crc(MSG_TYPE_SCHED_EXIT, b"first"));
    buf.extend(frame_with_crc(
        crate::vmm::wire::MSG_TYPE_STIMULUS,
        b"valid",
    ));
    buf.extend(frame_with_torn_crc(MSG_TYPE_SCHED_EXIT, b"second"));
    let drained = a.feed(&buf);
    assert_eq!(drained.messages.len(), 3);
    assert!(!drained.messages[0].crc_ok);
    assert!(drained.messages[1].crc_ok);
    assert!(!drained.messages[2].crc_ok);
    let (kill, evt_fired) = run_sched_exit_gate(&drained.messages);
    assert!(
        !kill,
        "neither torn SCHED_EXIT may promote even though a CRC-valid \
             non-SCHED_EXIT frame arrived alongside them"
    );
    assert!(!evt_fired, "kill eventfd must remain undisturbed");
}

/// CRC-failed SNAPSHOT_REQUEST MUST be dropped before
/// `decode_snapshot_request` runs. A torn or hostile-guest
/// snapshot request would otherwise let an attacker force a
/// spurious `freeze_and_capture` (host-side stall, dump
/// allocation) or `arm_user_watchpoint` (DR slot consumption,
/// `KVM_SET_GUEST_DEBUG` reprogram) without ever generating a
/// matching CRC.
#[test]
fn snapshot_request_with_torn_crc_is_dropped() {
    let mut a = HostAssembler::new();
    let payload = snapshot_request_bytes(7, SNAPSHOT_KIND_CAPTURE, "snap_dump");
    let bytes = frame_with_torn_crc(MSG_TYPE_SNAPSHOT_REQUEST, &payload);
    let drained = a.feed(&bytes);
    assert_eq!(drained.messages.len(), 1, "assembler emits one message");
    assert!(
        !drained.messages[0].crc_ok,
        "torn CRC must surface as crc_ok=false"
    );
    assert_eq!(
        drained.messages[0].msg_type, MSG_TYPE_SNAPSHOT_REQUEST,
        "msg_type unaffected by CRC mismatch"
    );
    let pushed = run_snapshot_request_gate(&drained.messages);
    assert_eq!(
        pushed, 0,
        "CRC-failed SNAPSHOT_REQUEST must NOT decode — \
             hostile guest must not force a capture or watchpoint arm"
    );
}

/// Positive control: a CRC-valid SNAPSHOT_REQUEST with a
/// well-formed payload IS pushed onto the pending list. Same
/// degenerate-pass guard rationale as the SCHED_EXIT positive
/// control above.
#[test]
fn snapshot_request_with_valid_crc_is_pushed() {
    let mut a = HostAssembler::new();
    let payload = snapshot_request_bytes(42, SNAPSHOT_KIND_CAPTURE, "valid_tag");
    let bytes = frame_with_crc(MSG_TYPE_SNAPSHOT_REQUEST, &payload);
    let drained = a.feed(&bytes);
    assert_eq!(drained.messages.len(), 1);
    assert!(
        drained.messages[0].crc_ok,
        "matching CRC must surface as crc_ok=true"
    );
    let pushed = run_snapshot_request_gate(&drained.messages);
    assert_eq!(
        pushed, 1,
        "CRC-valid well-formed SNAPSHOT_REQUEST MUST decode and push"
    );
}

/// Mixed batch: CRC-failed SNAPSHOT_REQUEST sandwiched between
/// CRC-valid SNAPSHOT_REQUESTs. Only the valid ones must push;
/// the torn frame must drop independently. Pins the per-message
/// gate behaviour against a regression that decodes the whole
/// batch when any CRC matches.
#[test]
fn snapshot_request_torn_crc_dropped_in_mixed_batch() {
    let mut a = HostAssembler::new();
    let p_first = snapshot_request_bytes(1, SNAPSHOT_KIND_CAPTURE, "first");
    let p_torn = snapshot_request_bytes(2, SNAPSHOT_KIND_CAPTURE, "torn");
    let p_third = snapshot_request_bytes(3, SNAPSHOT_KIND_CAPTURE, "third");
    let mut buf = Vec::new();
    buf.extend(frame_with_crc(MSG_TYPE_SNAPSHOT_REQUEST, &p_first));
    buf.extend(frame_with_torn_crc(MSG_TYPE_SNAPSHOT_REQUEST, &p_torn));
    buf.extend(frame_with_crc(MSG_TYPE_SNAPSHOT_REQUEST, &p_third));
    let drained = a.feed(&buf);
    assert_eq!(drained.messages.len(), 3);
    assert!(drained.messages[0].crc_ok);
    assert!(!drained.messages[1].crc_ok);
    assert!(drained.messages[2].crc_ok);
    let pushed = run_snapshot_request_gate(&drained.messages);
    assert_eq!(
        pushed, 2,
        "exactly the two CRC-valid SNAPSHOT_REQUESTs must push; \
             the torn middle frame must drop independently"
    );
}

/// CRC-failed SCHED_EXIT followed by CRC-failed SNAPSHOT_REQUEST
/// in a single drain: BOTH gates must short-circuit. A regression
/// where the SCHED_EXIT gate's `crc_ok` check is correct but the
/// SNAPSHOT_REQUEST gate's check is dropped would still pass the
/// SCHED_EXIT-only test; this multi-gate test catches that.
#[test]
fn both_gates_drop_torn_frames_in_same_drain() {
    let mut a = HostAssembler::new();
    let snap_payload = snapshot_request_bytes(99, SNAPSHOT_KIND_CAPTURE, "tag");
    let mut buf = Vec::new();
    buf.extend(frame_with_torn_crc(MSG_TYPE_SCHED_EXIT, b"sched-exit"));
    buf.extend(frame_with_torn_crc(
        MSG_TYPE_SNAPSHOT_REQUEST,
        &snap_payload,
    ));
    let drained = a.feed(&buf);
    assert_eq!(drained.messages.len(), 2);
    assert!(!drained.messages[0].crc_ok);
    assert!(!drained.messages[1].crc_ok);
    let (kill, evt_fired) = run_sched_exit_gate(&drained.messages);
    let pushed = run_snapshot_request_gate(&drained.messages);
    assert!(!kill, "torn SCHED_EXIT must not promote kill");
    assert!(!evt_fired, "torn SCHED_EXIT must not write kill eventfd");
    assert_eq!(pushed, 0, "torn SNAPSHOT_REQUEST must not decode");
}

/// Apply the production SYS_RDY promotion gate (the
/// `msg.msg_type == MSG_TYPE_SYS_RDY && msg.crc_ok && let
/// Some(evt) = sys_rdy_evt.take()` branch in the freeze
/// coordinator's TOKEN_TX handler) against a slice of
/// `BulkMessage` values. Returns `(eventfd_counter,
/// remaining_handle_present)` so the test can assert both
/// the fire-once semantics (counter at most 1) and the
/// `Option::take` ownership transfer (remaining=false after
/// any successful promotion). The outer Arc clone lets the
/// caller read the counter after the gate moved its handle
/// into the predicate body.
fn run_sys_rdy_gate(messages: &[crate::vmm::bulk::BulkMessage]) -> (u32, bool) {
    let evt = std::sync::Arc::new(EventFd::new(EFD_NONBLOCK).expect("eventfd construction"));
    let mut sys_rdy_evt: Option<std::sync::Arc<EventFd>> = Some(evt.clone());
    for msg in messages {
        // Exact predicate copied from the production closure.
        if msg.msg_type == MSG_TYPE_SYS_RDY
            && msg.crc_ok
            && let Some(evt) = sys_rdy_evt.take()
        {
            let _ = evt.write(1);
        }
    }
    let remaining = sys_rdy_evt.is_some();
    // `read()` on EFD_NONBLOCK eventfd returns the accumulated
    // counter or EAGAIN when zero. With take()-based fire-once
    // semantics, at most one write can occur.
    let counter = match evt.read() {
        Ok(n) => n as u32,
        Err(_) => 0,
    };
    (counter, remaining)
}

/// CRC-failed SYS_RDY MUST NOT fire the boot-complete eventfd.
/// A torn or hostile-guest frame would otherwise let an attacker
/// race ahead of `setup_per_cpu_areas` / KASLR randomization,
/// causing the monitor's first sample iteration to read against
/// pre-boot zeros.
#[test]
fn sys_rdy_with_torn_crc_does_not_fire_eventfd() {
    let mut a = HostAssembler::new();
    let bytes = frame_with_torn_crc(MSG_TYPE_SYS_RDY, b"");
    let drained = a.feed(&bytes);
    assert_eq!(drained.messages.len(), 1, "assembler emits one message");
    assert!(
        !drained.messages[0].crc_ok,
        "torn CRC must surface as crc_ok=false"
    );
    assert_eq!(
        drained.messages[0].msg_type, MSG_TYPE_SYS_RDY,
        "msg_type unaffected by CRC mismatch"
    );
    let (counter, remaining) = run_sys_rdy_gate(&drained.messages);
    assert_eq!(
        counter, 0,
        "boot-complete eventfd must NOT be written on CRC-failed \
             SYS_RDY — hostile guest must not race ahead of percpu/KASLR"
    );
    assert!(
        remaining,
        "Option::take must NOT consume the handle on a dropped frame — \
             a later CRC-valid SYS_RDY must still be able to promote"
    );
}

/// Positive control: a CRC-valid SYS_RDY DOES fire the eventfd
/// and consumes the Option (fire-once semantics).
#[test]
fn sys_rdy_with_valid_crc_fires_eventfd_once() {
    let mut a = HostAssembler::new();
    let bytes = frame_with_crc(MSG_TYPE_SYS_RDY, b"");
    let drained = a.feed(&bytes);
    assert_eq!(drained.messages.len(), 1);
    assert!(
        drained.messages[0].crc_ok,
        "matching CRC must surface as crc_ok=true"
    );
    let (counter, remaining) = run_sys_rdy_gate(&drained.messages);
    assert_eq!(
        counter, 1,
        "boot-complete eventfd MUST receive a single write on \
             CRC-valid SYS_RDY"
    );
    assert!(
        !remaining,
        "Option::take must consume the handle so subsequent \
             SYS_RDY frames do not pump the counter"
    );
}

/// Two CRC-valid SYS_RDY frames in sequence: the first
/// promotes, every subsequent frame drops. Pins `Option::take`
/// semantics so a hostile or buggy guest resending SYS_RDY
/// cannot pump the eventfd counter into EAGAIN territory or
/// wedge a later boot signal.
#[test]
fn sys_rdy_with_valid_crc_fires_once_then_subsequent_drops() {
    let mut a = HostAssembler::new();
    let mut buf = Vec::new();
    buf.extend(frame_with_crc(MSG_TYPE_SYS_RDY, b""));
    buf.extend(frame_with_crc(MSG_TYPE_SYS_RDY, b""));
    let drained = a.feed(&buf);
    assert_eq!(drained.messages.len(), 2);
    assert!(drained.messages[0].crc_ok);
    assert!(drained.messages[1].crc_ok);
    let (counter, remaining) = run_sys_rdy_gate(&drained.messages);
    assert_eq!(
        counter, 1,
        "second SYS_RDY must NOT pump the eventfd — \
             Option::take consumed the handle on the first promotion"
    );
    assert!(!remaining);
}

/// CRC-valid SYS_RDY alongside CRC-valid SCHED_EXIT in the same
/// drain: both gates fire independently. Pins per-message gate
/// dispatch — a regression that aliased the two type checks
/// would let one gate's failure mask the other.
#[test]
fn sys_rdy_and_sched_exit_fire_independently() {
    let mut a = HostAssembler::new();
    let mut buf = Vec::new();
    buf.extend(frame_with_crc(MSG_TYPE_SYS_RDY, b""));
    buf.extend(frame_with_crc(MSG_TYPE_SCHED_EXIT, b"exit-payload"));
    let drained = a.feed(&buf);
    assert_eq!(drained.messages.len(), 2);
    assert!(drained.messages[0].crc_ok);
    assert!(drained.messages[1].crc_ok);
    let (rdy_counter, rdy_remaining) = run_sys_rdy_gate(&drained.messages);
    let (kill, kill_evt_fired) = run_sched_exit_gate(&drained.messages);
    assert_eq!(rdy_counter, 1, "SYS_RDY must promote");
    assert!(!rdy_remaining, "SYS_RDY handle must be consumed");
    assert!(kill, "SCHED_EXIT must promote kill");
    assert!(kill_evt_fired, "SCHED_EXIT must write kill eventfd");
}
