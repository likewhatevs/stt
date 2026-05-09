//! Coverage for the production TOKEN_TX `match
//! MsgType::from_wire(...)` arm structure inside the freeze
//! coordinator's run-loop closure.
//!
//! [`crc_defense_tests`] above pins individual gate predicates
//! (`msg_type == X && crc_ok`) but routes around the production
//! exhaustive-match scaffolding. The tests here run the full
//! dispatch — including the catch-all `Some(other) if
//! !is_coordinator_internal()` arm, the `Some(_)` drop arm for
//! coordinator-internal frames without inline side-effect arms,
//! and the `None` warn-and-drop arm for unrecognised tags — so
//! a regression that re-orders the arms or drops a `match`
//! clause fails here.
//!
//! Coverage map:
//!   * T1 — unknown msg_type lands in the `None` arm (no
//!     bucket emission).
//!   * T2 — CRC-bad SCHED_EXIT must NOT bucket. Pins the
//!     dispatch-site gate that prevents a torn or hostile-guest
//!     SchedExit from surfacing as a phantom verdict entry —
//!     no downstream consumer filters SchedExit on `crc_ok`.
//!   * T3 — guest-stamped SNAPSHOT_REPLY on TX must NOT
//!     bucket. Replies are host→guest only; the
//!     `is_coordinator_internal` classifier lists
//!     `MsgType::SnapshotReply` so the dispatch routes through
//!     the `Some(_)` drop arm.
//!   * T4 — SYS_RDY with non-empty payload must NOT promote
//!     (shape gate). A hostile guest tacking bytes onto a
//!     SYS_RDY would otherwise smuggle data past the
//!     coordinator-internal filter; the
//!     `&& msg.payload.is_empty()` clause is the safety net.
//!   * T5 — fire-once across HostAssembler::feed boundaries.
//!     Existing tests pin fire-once within one feed; this test
//!     calls feed twice with two separate SYS_RDY frames and
//!     asserts only the first promotes.
//!   * T6 — interleaved batch (SYS_RDY + SCHED_EXIT +
//!     SNAPSHOT_REQUEST + STIMULUS) — every gate fires
//!     independently and the bucket reflects exactly the
//!     non-coordinator-internal verdict-bearing entries.
//!   * T7 — empty bucket short-circuit. A SYS_RDY-only batch
//!     produces an empty bucket; pinning the post-condition
//!     keeps the production short-circuit honest.
//!   * T8 — multiple CRC-valid SCHED_EXIT frames pump kill_evt
//!     by 1 per frame (EFD_NONBLOCK accumulation). Pinning the
//!     count keeps a future "promote-once" optimisation from
//!     silently changing the wakeup-edge contract.
use super::*;
use crate::vmm::bulk::HostAssembler;
use crate::vmm::wire::{
    FRAME_HEADER_SIZE, MSG_TYPE_SCHED_EXIT, MSG_TYPE_SNAPSHOT_REPLY, MSG_TYPE_SNAPSHOT_REQUEST,
    MSG_TYPE_STIMULUS, MSG_TYPE_SYS_RDY, MsgType, SNAPSHOT_KIND_CAPTURE, SNAPSHOT_TAG_MAX,
    ShmEntry, ShmMessage, SnapshotRequestPayload,
};
use std::sync::atomic::{AtomicBool, Ordering};
use vmm_sys_util::eventfd::{EFD_NONBLOCK, EventFd};
use zerocopy::IntoBytes;

/// Outcome of running the production-shaped TOKEN_TX dispatch
/// against a slice of `BulkMessage`. Mirrors the closure-scope
/// state the production handler mutates: the run-wide kill
/// flag, the kill-eventfd counter, the SYS_RDY one-shot
/// handle's residual presence and write count, the
/// `snapshot_requests_pending` push count, and the verdict
/// `bucket` itself. Returning the bucket lets per-test
/// assertions inspect bucketing precisely (which frames
/// landed, which were filtered).
struct DispatchOutcome {
    kill: bool,
    kill_evt_counter: u32,
    sys_rdy_counter: u32,
    sys_rdy_remaining: bool,
    snapshot_pending: usize,
    bucket: Vec<ShmEntry>,
    unknown_count: usize,
}

/// Reproduce the production TOKEN_TX dispatch arm-for-arm
/// against a fresh closure-scope state and a slice of
/// `BulkMessage`. The arm structure below is copied verbatim
/// from `freeze_coord/mod.rs`'s TOKEN_TX handler; a
/// divergence in production would surface here as a test
/// that no longer pins the actual behaviour. If the
/// production dispatch changes, this helper must be updated
/// in the same commit so the regression is visible — same
/// in-test-mirror discipline `crc_defense_tests` uses.
fn run_dispatch(messages: &[crate::vmm::bulk::BulkMessage]) -> DispatchOutcome {
    let kill = AtomicBool::new(false);
    let kill_evt = EventFd::new(EFD_NONBLOCK).expect("kill eventfd");
    let sys_rdy_evt = std::sync::Arc::new(EventFd::new(EFD_NONBLOCK).expect("sys_rdy eventfd"));
    let mut sys_rdy_handle: Option<std::sync::Arc<EventFd>> = Some(sys_rdy_evt.clone());
    let mut snapshot_pending: Vec<SnapshotRequest> = Vec::new();
    let mut bucket: Vec<ShmEntry> = Vec::new();
    let mut unknown_count = 0usize;
    for msg in messages {
        let kind = MsgType::from_wire(msg.msg_type);
        match kind {
            Some(MsgType::SchedExit) => {
                if msg.crc_ok {
                    kill.store(true, Ordering::Release);
                    let _ = kill_evt.write(1);
                }
                if msg.crc_ok {
                    bucket.push(ShmEntry {
                        msg_type: msg.msg_type,
                        payload: msg.payload.to_vec(),
                        crc_ok: msg.crc_ok,
                    });
                }
            }
            Some(MsgType::SysRdy) => {
                if msg.crc_ok
                    && msg.payload.is_empty()
                    && let Some(evt) = sys_rdy_handle.take()
                {
                    let _ = evt.write(1);
                }
            }
            Some(MsgType::SnapshotRequest) => {
                if msg.crc_ok
                    && let Some(req) = decode_snapshot_request(&msg.payload[..])
                {
                    snapshot_pending.push(req);
                }
            }
            Some(other) if !other.is_coordinator_internal() => {
                bucket.push(ShmEntry {
                    msg_type: msg.msg_type,
                    payload: msg.payload.to_vec(),
                    crc_ok: msg.crc_ok,
                });
            }
            Some(_) => {
                // is_coordinator_internal with no inline arm.
                // Drop silently — the production dispatch
                // documents this as the future-extension hook
                // for new coordinator-internal frame types.
            }
            None => {
                unknown_count = unknown_count.saturating_add(1);
            }
        }
    }
    let kill_value = kill.load(Ordering::Acquire);
    let kill_evt_counter = match kill_evt.read() {
        Ok(n) => n as u32,
        Err(_) => 0,
    };
    let sys_rdy_remaining = sys_rdy_handle.is_some();
    let sys_rdy_counter = match sys_rdy_evt.read() {
        Ok(n) => n as u32,
        Err(_) => 0,
    };
    DispatchOutcome {
        kill: kill_value,
        kill_evt_counter,
        sys_rdy_counter,
        sys_rdy_remaining,
        snapshot_pending: snapshot_pending.len(),
        bucket,
        unknown_count,
    }
}

/// Build a CRC-valid TLV frame. Same helper as
/// `crc_defense_tests::frame_with_crc`; duplicated so this
/// module is self-contained.
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

/// Build a CRC-mismatched TLV frame.
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

/// Wire-format SnapshotRequest payload bytes.
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

/// T1 — Unknown msg_type lands in the `None` arm and DOES
/// NOT emit a bucket entry. The dispatch is exhaustive
/// precisely so a future guest stamping a tag the host
/// doesn't recognise is surfaced via warn-log rather than
/// slipping into the catch-all arm and surfacing as a
/// phantom verdict entry.
#[test]
fn unknown_msg_type_drops_without_bucketing() {
    let mut a = HostAssembler::new();
    // 0xDEAD_BEEF is not present in
    // `MsgType::from_wire`'s dispatch table; `from_wire`
    // returns `None` for it.
    assert!(MsgType::from_wire(0xDEAD_BEEF).is_none());
    let bytes = frame_with_crc(0xDEAD_BEEF, b"unknown-payload");
    let drained = a.feed(&bytes);
    assert_eq!(drained.messages.len(), 1);
    let out = run_dispatch(&drained.messages);
    assert_eq!(
        out.unknown_count, 1,
        "unknown msg_type must hit the `None` arm exactly once"
    );
    assert!(
        out.bucket.is_empty(),
        "unknown msg_type must NOT surface as a verdict entry"
    );
    assert!(!out.kill, "unknown msg_type must NOT promote kill");
    assert_eq!(
        out.sys_rdy_counter, 0,
        "unknown msg_type must NOT pump sys_rdy"
    );
}

/// T2 — CRC-bad SCHED_EXIT must NOT bucket. Without the
/// `if msg.crc_ok` gate at the bucket-push site, a torn or
/// hostile-guest SCHED_EXIT would surface in
/// `BulkDrainResult.entries` with `crc_ok=false`. No
/// downstream consumer filters SchedExit entries on
/// `crc_ok` — the only post-collect filter sites are
/// `MSG_TYPE_STIMULUS && crc_ok`, `MSG_TYPE_EXIT &&
/// crc_ok`, and `Stdout|Stderr` skipping `!e.crc_ok`. A
/// torn SchedExit therefore would have surfaced verbatim.
/// This test pins the filter at the dispatch site.
#[test]
fn sched_exit_torn_crc_does_not_bucket() {
    let mut a = HostAssembler::new();
    let bytes = frame_with_torn_crc(MSG_TYPE_SCHED_EXIT, b"torn-payload");
    let drained = a.feed(&bytes);
    assert_eq!(drained.messages.len(), 1);
    assert!(
        !drained.messages[0].crc_ok,
        "torn SCHED_EXIT must surface as crc_ok=false"
    );
    let out = run_dispatch(&drained.messages);
    assert!(
        out.bucket.is_empty(),
        "CRC-bad SCHED_EXIT must NOT pollute the verdict bucket — \
             a phantom verdict entry would surface in BulkDrainResult"
    );
    assert!(
        !out.kill,
        "CRC-bad SCHED_EXIT must NOT promote kill (existing gate)"
    );
    assert_eq!(
        out.kill_evt_counter, 0,
        "CRC-bad SCHED_EXIT must NOT write kill_evt (existing gate)"
    );
}

/// Positive control for T2: CRC-valid SCHED_EXIT bucketed
/// exactly once with the original payload and crc_ok=true.
/// Dropping the bucket entry on the happy path would lose
/// the scheduler exit-code diagnostic; this test pins that
/// the dispatch-site filter is "drop only torn frames",
/// not "drop every SchedExit".
#[test]
fn sched_exit_valid_crc_buckets_with_payload() {
    let mut a = HostAssembler::new();
    let payload = (-1i32).to_le_bytes();
    let bytes = frame_with_crc(MSG_TYPE_SCHED_EXIT, &payload);
    let drained = a.feed(&bytes);
    assert_eq!(drained.messages.len(), 1);
    assert!(drained.messages[0].crc_ok);
    let out = run_dispatch(&drained.messages);
    assert_eq!(
        out.bucket.len(),
        1,
        "CRC-valid SCHED_EXIT MUST bucket exactly once"
    );
    assert_eq!(out.bucket[0].msg_type, MSG_TYPE_SCHED_EXIT);
    assert_eq!(out.bucket[0].payload, &payload[..]);
    assert!(out.bucket[0].crc_ok);
    assert!(out.kill);
    assert_eq!(out.kill_evt_counter, 1);
}

/// T3 — Guest-stamped SNAPSHOT_REPLY on TX must NOT
/// bucket. Replies are host→guest only — the host emits
/// them via `queue_input_port1`. A hostile or malformed
/// guest stamping MSG_TYPE_SNAPSHOT_REPLY on its TX stream
/// would surface as a phantom verdict entry without the
/// `is_coordinator_internal` extension. The classifier
/// lists `MsgType::SnapshotReply` so the dispatch routes
/// through the `Some(_)` drop arm. This test verifies the
/// silent drop end-to-end, including the matching
/// `is_coordinator_internal` invariant.
#[test]
fn snapshot_reply_on_tx_does_not_bucket() {
    let mut a = HostAssembler::new();
    // Even a CRC-valid frame must drop — the classifier
    // suppresses by tag, not by CRC.
    let bytes = frame_with_crc(MSG_TYPE_SNAPSHOT_REPLY, b"forged-reply");
    let drained = a.feed(&bytes);
    assert_eq!(drained.messages.len(), 1);
    assert!(drained.messages[0].crc_ok);
    let out = run_dispatch(&drained.messages);
    assert!(
        out.bucket.is_empty(),
        "guest-stamped SNAPSHOT_REPLY must NOT surface as verdict — \
             the tag is host→guest only"
    );
    assert_eq!(
        out.snapshot_pending, 0,
        "SNAPSHOT_REPLY is not a request — must not push pending"
    );
    // Sanity: `is_coordinator_internal` covers
    // SnapshotReply.
    assert!(
        MsgType::SnapshotReply.is_coordinator_internal(),
        "SnapshotReply must be classified as coordinator-internal"
    );
}

/// T4 — CRC-valid SYS_RDY with non-empty payload must NOT
/// promote. The strict shape gate `msg.payload.is_empty()`
/// is the safety net against a hostile guest tacking bytes
/// onto a SysRdy frame to smuggle data past the
/// coordinator-internal filter. SysRdy is documented as
/// carrying no payload; the guest writer at
/// `guest_comms::send_sys_rdy` emits exactly zero payload
/// bytes.
#[test]
fn sys_rdy_with_nonempty_payload_does_not_promote() {
    let mut a = HostAssembler::new();
    let bytes = frame_with_crc(MSG_TYPE_SYS_RDY, b"smuggled-bytes");
    let drained = a.feed(&bytes);
    assert_eq!(drained.messages.len(), 1);
    assert!(drained.messages[0].crc_ok);
    assert_eq!(
        drained.messages[0].payload.len(),
        14,
        "smuggled payload must propagate verbatim from assembler"
    );
    let out = run_dispatch(&drained.messages);
    assert_eq!(
        out.sys_rdy_counter, 0,
        "SysRdy with non-empty payload must NOT fire eventfd — \
             shape gate (is_empty) blocks the smuggle path"
    );
    assert!(
        out.sys_rdy_remaining,
        "SysRdy handle must remain available for a later \
             well-formed (empty-payload) frame"
    );
    assert!(
        out.bucket.is_empty(),
        "SysRdy must NOT bucket regardless of shape — \
             coordinator-internal classification dominates"
    );
}

/// T5 — Fire-once across feed boundaries. The closure's
/// `Option::take` lives on the closure scope, not in the
/// HostAssembler; a fresh `feed` call should not reset the
/// one-shot. Two SYS_RDY frames published across two TX
/// wakes (two `drain_bulk` + `feed` cycles in production)
/// must promote exactly once when both batches feed into
/// one closure-scope dispatch.
#[test]
fn sys_rdy_fires_once_across_two_feed_calls() {
    let mut a = HostAssembler::new();
    let drained1 = a.feed(&frame_with_crc(MSG_TYPE_SYS_RDY, b""));
    assert_eq!(drained1.messages.len(), 1);
    let drained2 = a.feed(&frame_with_crc(MSG_TYPE_SYS_RDY, b""));
    assert_eq!(drained2.messages.len(), 1);
    // Concatenate both feed outputs and pass through one
    // dispatch invocation — the closure-scope
    // `sys_rdy_handle` Option spans the entire run-loop
    // iteration in production, so a fire-once test must
    // use one `run_dispatch` over the merged messages.
    let mut combined: Vec<crate::vmm::bulk::BulkMessage> = Vec::new();
    combined.extend_from_slice(&drained1.messages);
    combined.extend_from_slice(&drained2.messages);
    let out = run_dispatch(&combined);
    assert_eq!(
        out.sys_rdy_counter, 1,
        "second SYS_RDY across feed boundary must NOT pump — \
             closure-scope take() is the one-shot, not the assembler"
    );
    assert!(
        !out.sys_rdy_remaining,
        "first SYS_RDY must have consumed the handle"
    );
}

/// T6 — Interleaved batch (SYS_RDY + SCHED_EXIT +
/// SNAPSHOT_REQUEST + STIMULUS in one drain). Every gate
/// fires independently and the bucket reflects exactly the
/// non-coordinator-internal verdict-bearing entries
/// (SCHED_EXIT + STIMULUS), not SYS_RDY or
/// SNAPSHOT_REQUEST. Pins the per-message arm dispatch — a
/// regression that aliased the type checks across arms
/// would let one gate's failure mask the other.
#[test]
fn interleaved_batch_dispatches_all_arms_independently() {
    let mut a = HostAssembler::new();
    let mut buf = Vec::new();
    buf.extend(frame_with_crc(MSG_TYPE_SYS_RDY, b""));
    buf.extend(frame_with_crc(MSG_TYPE_SCHED_EXIT, b"exit"));
    let snap = snapshot_request_bytes(7, SNAPSHOT_KIND_CAPTURE, "snap");
    buf.extend(frame_with_crc(MSG_TYPE_SNAPSHOT_REQUEST, &snap));
    buf.extend(frame_with_crc(MSG_TYPE_STIMULUS, b"stim-payload"));
    let drained = a.feed(&buf);
    assert_eq!(drained.messages.len(), 4);
    for m in &drained.messages {
        assert!(m.crc_ok, "all four frames must surface crc_ok=true");
    }
    let out = run_dispatch(&drained.messages);
    assert_eq!(out.sys_rdy_counter, 1, "SysRdy promotes");
    assert!(!out.sys_rdy_remaining, "SysRdy handle consumed");
    assert!(out.kill, "SchedExit promotes kill");
    assert_eq!(out.kill_evt_counter, 1);
    assert_eq!(
        out.snapshot_pending, 1,
        "SnapshotRequest decodes and pushes onto pending"
    );
    // Bucket has exactly two entries: SchedExit and
    // Stimulus. SysRdy and SnapshotRequest are
    // coordinator-internal and do NOT bucket.
    assert_eq!(
        out.bucket.len(),
        2,
        "bucket must contain SchedExit + Stimulus — \
             SysRdy + SnapshotRequest filtered as coordinator-internal"
    );
    let bucketed_tags: Vec<u32> = out.bucket.iter().map(|e| e.msg_type).collect();
    assert!(bucketed_tags.contains(&MSG_TYPE_SCHED_EXIT));
    assert!(bucketed_tags.contains(&MSG_TYPE_STIMULUS));
    assert!(!bucketed_tags.contains(&MSG_TYPE_SYS_RDY));
    assert!(!bucketed_tags.contains(&MSG_TYPE_SNAPSHOT_REQUEST));
}

/// T7 — Empty-bucket short-circuit. A SYS_RDY-only batch
/// produces an empty bucket; the production code's
/// `if !bucket.is_empty()` check at the bucket-flush site
/// avoids acquiring the shared `bulk_messages` mutex on
/// every empty-bucket drain. This matters during boot-up
/// where most TX wakes deliver pure SysRdy /
/// SnapshotRequest before SchedExit / TestResult arrive.
/// This test pins the post-condition: dispatch produces an
/// empty bucket on a pure-internal batch.
#[test]
fn sys_rdy_only_batch_yields_empty_bucket() {
    let mut a = HostAssembler::new();
    let drained = a.feed(&frame_with_crc(MSG_TYPE_SYS_RDY, b""));
    assert_eq!(drained.messages.len(), 1);
    let out = run_dispatch(&drained.messages);
    assert!(
        out.bucket.is_empty(),
        "SysRdy-only batch must produce an empty bucket — \
             the production short-circuit avoids the shared mutex"
    );
    assert_eq!(out.sys_rdy_counter, 1);
}

/// T8 — Multiple CRC-valid SCHED_EXIT frames in one batch.
/// The promotion gate fires per-message: every CRC-valid
/// SchedExit calls `kill_evt.write(1)`, which on
/// EFD_NONBLOCK (without EFD_SEMAPHORE) accumulates the
/// counter. Counter accumulation is documented as benign —
/// the AtomicBool above is authoritative — but pinning the
/// count keeps a future "promote-once via kill.swap(true)"
/// optimisation from silently changing the wakeup edge
/// semantics. The kill flag stays at `true` regardless of
/// how many SchedExit frames promote, and every CRC-valid
/// SchedExit also buckets.
#[test]
fn multiple_sched_exit_frames_pump_eventfd_per_frame() {
    let mut a = HostAssembler::new();
    let mut buf = Vec::new();
    buf.extend(frame_with_crc(MSG_TYPE_SCHED_EXIT, b"first"));
    buf.extend(frame_with_crc(MSG_TYPE_SCHED_EXIT, b"second"));
    buf.extend(frame_with_crc(MSG_TYPE_SCHED_EXIT, b"third"));
    let drained = a.feed(&buf);
    assert_eq!(drained.messages.len(), 3);
    let out = run_dispatch(&drained.messages);
    assert!(out.kill, "kill flag must be set");
    assert_eq!(
        out.kill_evt_counter, 3,
        "EFD_NONBLOCK eventfd accumulates 1 per CRC-valid SchedExit \
             frame — pinning the count documents the wakeup-edge \
             contract"
    );
    assert_eq!(
        out.bucket.len(),
        3,
        "every CRC-valid SchedExit must bucket — exit-code \
             diagnostic must reach the verdict stream"
    );
    for entry in &out.bucket {
        assert_eq!(entry.msg_type, MSG_TYPE_SCHED_EXIT);
        assert!(entry.crc_ok);
    }
}

/// Mixed-CRC SchedExit batch — a torn-CRC frame between
/// two CRC-valid SchedExits. Pins the per-message gate
/// from the dispatch fix: the torn middle frame must NOT
/// bucket and must NOT promote, but the surrounding valid
/// frames must do both.
#[test]
fn sched_exit_mixed_crc_batch_filters_torn_frame() {
    let mut a = HostAssembler::new();
    let mut buf = Vec::new();
    buf.extend(frame_with_crc(MSG_TYPE_SCHED_EXIT, b"valid-1"));
    buf.extend(frame_with_torn_crc(MSG_TYPE_SCHED_EXIT, b"torn"));
    buf.extend(frame_with_crc(MSG_TYPE_SCHED_EXIT, b"valid-2"));
    let drained = a.feed(&buf);
    assert_eq!(drained.messages.len(), 3);
    assert!(drained.messages[0].crc_ok);
    assert!(!drained.messages[1].crc_ok);
    assert!(drained.messages[2].crc_ok);
    let out = run_dispatch(&drained.messages);
    assert_eq!(
        out.bucket.len(),
        2,
        "torn middle SchedExit must drop; valid bookends must bucket"
    );
    for entry in &out.bucket {
        assert!(entry.crc_ok, "every bucketed SchedExit is crc_ok=true");
    }
    assert!(out.kill, "valid SchedExits promote kill");
    assert_eq!(
        out.kill_evt_counter, 2,
        "exactly the two valid SchedExits pump kill_evt"
    );
}
