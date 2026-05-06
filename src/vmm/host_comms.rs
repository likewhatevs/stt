//! Host-only typed consumers for the guest-bound bulk TLV stream and
//! virtio-console wake-byte control plane.
//!
//! These helpers run inside the host VMM (the freeze coordinator,
//! the watchdog, the monitor thread) and never inside a guest.
//!
//! - [`drain_bulk`] takes the accumulated port-1 TX bytes from a
//!   [`super::virtio_console::VirtioConsole`] device and parses them
//!   into a [`BulkDrainResult`]. Internally it delegates to
//!   [`parse_tlv_stream`].
//!
//! - [`parse_tlv_stream`] is the byte-level parser. It walks a buffer
//!   of [`super::wire::ShmMessage`]-prefixed frames and yields one
//!   [`super::wire::ShmEntry`] per complete frame, tolerating
//!   per-frame CRC mismatches (`crc_ok=false` on the affected entry,
//!   parsing continues for subsequent frames).
//!
//! - [`request_dump`] and [`request_shutdown`] push virtio-console
//!   RX wake bytes that the guest's `hvc0_poll_loop` recognises
//!   directly. The SysRq-D dispatch is triggered by the
//!   `SIGNAL_VC_DUMP` wake byte alone.
//!
//! # No drop counter
//!
//! The bulk channel uses the kernel virtio_console TX path:
//! `port_fops_write` blocks the writer until the host's `add_used`
//! rate catches up. Backpressure replaces drops, so
//! [`BulkDrainResult`] carries no `drops` field — every byte the
//! guest emitted is delivered, in order.

use std::sync::Arc;
use zerocopy::FromBytes;

use super::bulk::MAX_BULK_FRAME_PAYLOAD;
use super::pi_mutex::PiMutex;
use super::virtio_console::{
    SIGNAL_BPF_WRITE_DONE, SIGNAL_VC_DUMP, SIGNAL_VC_SHUTDOWN, VirtioConsole,
};
use super::wire::{FRAME_HEADER_SIZE, ShmEntry, ShmMessage};

/// Result of one bulk-channel drain.
///
/// Holds every complete frame parsed from the most recent drain. The
/// caller may filter on [`super::wire::ShmEntry::crc_ok`] to drop
/// corrupted entries; CRC mismatches do not stop the walk.
///
/// No `drops` field: the bulk channel uses the kernel virtio_console
/// TX path which blocks on backpressure rather than dropping. Every
/// byte the guest emitted is delivered, in order.
#[derive(Debug, Clone, Default)]
pub struct BulkDrainResult {
    /// One entry per complete frame parsed from the input bytes.
    pub entries: Vec<ShmEntry>,
}

/// Drain accumulated port-1 TX bytes from `dev` and parse them into a
/// [`BulkDrainResult`].
///
/// Calls [`VirtioConsole::drain_bulk`] (which clears the device's
/// internal buffer) and feeds the bytes through [`parse_tlv_stream`].
/// Partial trailing frames are dropped by the one-shot parser; the
/// freeze coordinator's mid-run streaming consumer must use
/// [`super::bulk::HostAssembler`] directly when partial-frame
/// retention across drains is required.
#[allow(dead_code)]
pub fn drain_bulk(dev: &mut VirtioConsole) -> BulkDrainResult {
    let bytes = dev.drain_bulk();
    parse_tlv_stream(&bytes)
}

/// Parse a TLV byte stream into a [`BulkDrainResult`].
///
/// Walks `buf` end-to-end, decoding one [`ShmMessage`] header per
/// 16-byte chunk and consuming the trailing `length` payload bytes.
/// CRC32 is computed over the payload and compared against the
/// header's stored value; mismatches surface as
/// [`super::wire::ShmEntry::crc_ok`]`=false` but do not break the
/// walk — subsequent frames are still parsed.
///
/// Trailing partial frames (header truncated, or payload shorter
/// than `length`) are silently dropped: the parser stops at the
/// first incomplete frame. Callers that need partial-frame retention
/// across multiple drain calls should use
/// [`super::bulk::HostAssembler`] instead.
///
/// # Hostile-input guard
///
/// Frames whose announced `length` exceeds
/// [`MAX_BULK_FRAME_PAYLOAD`] are rejected. The on-wire `length`
/// field is `u32` so a malformed or hostile guest can announce
/// up to 4 GiB of payload; the cap rejects such headers before
/// allocating the per-frame `Vec<u8>` payload. Mirrors the same
/// check applied by [`super::bulk::HostAssembler`] so the
/// streaming and one-shot consumers agree on what counts as a
/// legitimate frame. On rejection, the walk stops — subsequent
/// bytes cannot be trusted (the announced length cannot be relied
/// on to advance past the bogus payload).
pub fn parse_tlv_stream(buf: &[u8]) -> BulkDrainResult {
    let mut entries: Vec<ShmEntry> = Vec::new();
    let mut pos = 0usize;
    while pos.saturating_add(FRAME_HEADER_SIZE) <= buf.len() {
        let hdr_end = pos + FRAME_HEADER_SIZE;
        let hdr_slice = &buf[pos..hdr_end];
        let Ok(msg) = ShmMessage::read_from_bytes(hdr_slice) else {
            // Cannot fail — slice is exactly FRAME_HEADER_SIZE bytes
            // and ShmMessage is FromBytes. Defensive break to avoid
            // an infinite loop if the invariant is ever violated.
            break;
        };
        // Hostile-guest covert-channel guard: `_pad` is reserved
        // and writers MUST set it to 0 (see `ShmMessage` doc).
        // A non-zero value cannot come from a legitimate guest
        // producer — surface it for diagnostics rather than
        // silently ignoring potential covert-channel bytes. The
        // walk continues; the field is not load-bearing for frame
        // dispatch so a malformed value does not corrupt parsing.
        if msg._pad != 0 {
            tracing::warn!(
                msg_type = msg.msg_type,
                length = msg.length,
                pad = msg._pad,
                "parse_tlv_stream: non-zero _pad in frame header; possible hostile guest covert channel"
            );
        }
        // Hostile-input guard: a `length` above the per-frame cap
        // cannot come from any legitimate producer (every real
        // payload sits well below 256 KiB) and would trigger an
        // oversized `Vec<u8>` allocation. Reject before the
        // payload-length sanity check below so a malformed
        // `u32::MAX` length does not even reach the
        // `saturating_sub` arithmetic. Stop parsing — the
        // announced length cannot be trusted to advance the
        // cursor past the bogus payload, so any trailing bytes
        // are unparseable.
        if msg.length > MAX_BULK_FRAME_PAYLOAD {
            tracing::warn!(
                msg_type = msg.msg_type,
                length = msg.length,
                cap = MAX_BULK_FRAME_PAYLOAD,
                "parse_tlv_stream: dropping oversized frame; stopping walk"
            );
            break;
        }
        // Guard against a torn frame whose `length` claims more
        // payload than the buffer holds. Stop parsing rather than
        // attempting an over-read or oversized allocation.
        if (msg.length as usize) > buf.len().saturating_sub(hdr_end) {
            break;
        }
        let payload_end = hdr_end + msg.length as usize;
        let payload = buf[hdr_end..payload_end].to_vec();
        let computed_crc = crc32fast::hash(&payload);
        let crc_ok = computed_crc == msg.crc32;
        if !crc_ok {
            // Surface per-frame CRC mismatches for diagnostics —
            // Mirrors the assembly path so a corrupted bulk-stream frame
            // appears in the operator log instead of being dropped
            // silently when downstream consumers filter on
            // `crc_ok`. The walk continues regardless: subsequent
            // frames are still parsed.
            tracing::warn!(
                msg_type = msg.msg_type,
                length = msg.length,
                expected_crc = msg.crc32,
                computed_crc,
                "parse_tlv_stream: per-frame CRC mismatch; surfacing crc_ok=false"
            );
        }
        entries.push(ShmEntry {
            msg_type: msg.msg_type,
            payload,
            crc_ok,
        });
        pos = payload_end;
    }
    BulkDrainResult { entries }
}

/// Push a SysRq-D dump request to the guest by sending the
/// `SIGNAL_VC_DUMP` byte through the virtio-console RX queue. The
/// guest's `hvc0_poll_loop` blocks on `/dev/hvc0`, recognises the byte
/// directly, and triggers SysRq-D — no SHM control slot involved.
pub fn request_dump(virtio_con: &Arc<PiMutex<VirtioConsole>>) {
    virtio_con.lock().queue_input(&[SIGNAL_VC_DUMP]);
}

/// Push a graceful-shutdown request to the guest by sending the
/// `SIGNAL_VC_SHUTDOWN` byte through the virtio-console RX queue.
/// The guest's `hvc0_poll_loop` recognises the byte directly and
/// drives the graceful-shutdown teardown — no SHM signal slot
/// involved.
pub fn request_shutdown(virtio_con: &Arc<PiMutex<VirtioConsole>>) {
    virtio_con.lock().queue_input(&[SIGNAL_VC_SHUTDOWN]);
}

/// Notify the guest that the host's `bpf-map-write` thread finished
/// applying every queued `bpf_map_write`. Pushes
/// `SIGNAL_BPF_WRITE_DONE` through the virtio-console RX queue; the
/// guest's `hvc0_poll_loop` recognises the byte and sets the
/// `bpf_map_write_done` latch so a scenario blocked on
/// [`crate::scenario::Ctx::wait_for_map_write`] resumes. Replaces the
/// legacy SHM signal-slot rendezvous (host writes slot 0, guest blocks
/// on slot 0) with a single wake byte.
pub fn request_bpf_map_write_done(virtio_con: &Arc<PiMutex<VirtioConsole>>) {
    virtio_con.lock().queue_input(&[SIGNAL_BPF_WRITE_DONE]);
}

#[cfg(test)]
mod tests {
    //! Unit coverage for the typed host-side wrappers.
    //!
    //! `parse_tlv_stream` is the load-bearing entry point —
    //! `drain_bulk` is a thin adapter that delegates to it.

    use super::super::wire::{MSG_TYPE_EXIT, MSG_TYPE_STIMULUS};
    use super::*;
    use zerocopy::IntoBytes;

    fn frame_bytes(msg_type: u32, payload: &[u8]) -> Vec<u8> {
        let f = ShmMessage {
            msg_type,
            length: payload.len() as u32,
            crc32: crc32fast::hash(payload),
            _pad: 0,
        };
        let mut v = Vec::with_capacity(FRAME_HEADER_SIZE + payload.len());
        v.extend_from_slice(f.as_bytes());
        v.extend_from_slice(payload);
        v
    }

    /// `parse_tlv_stream` on an empty buffer yields no entries.
    #[test]
    fn parse_empty_buffer_yields_no_entries() {
        let r = parse_tlv_stream(&[]);
        assert!(r.entries.is_empty());
    }

    /// One complete frame produces one entry with `crc_ok=true`.
    #[test]
    fn parse_single_frame_one_entry() {
        let bytes = frame_bytes(MSG_TYPE_EXIT, &42i32.to_le_bytes());
        let r = parse_tlv_stream(&bytes);
        assert_eq!(r.entries.len(), 1);
        assert_eq!(r.entries[0].msg_type, MSG_TYPE_EXIT);
        assert!(r.entries[0].crc_ok);
        assert_eq!(r.entries[0].payload, 42i32.to_le_bytes());
    }

    /// Multiple frames in one buffer parse in order.
    #[test]
    fn parse_multiple_frames_preserve_order() {
        let mut buf = Vec::new();
        buf.extend(frame_bytes(MSG_TYPE_STIMULUS, b"first"));
        buf.extend(frame_bytes(MSG_TYPE_EXIT, b"second"));
        buf.extend(frame_bytes(MSG_TYPE_STIMULUS, b"third"));
        let r = parse_tlv_stream(&buf);
        assert_eq!(r.entries.len(), 3);
        assert_eq!(r.entries[0].payload, b"first");
        assert_eq!(r.entries[1].payload, b"second");
        assert_eq!(r.entries[2].payload, b"third");
    }

    /// A truncated trailing frame is dropped silently — partial
    /// frame retention is the streaming-assembler's job, not the
    /// one-shot parser's.
    #[test]
    fn parse_drops_trailing_partial_frame() {
        let bytes = frame_bytes(MSG_TYPE_EXIT, b"complete");
        let truncated = &bytes[..bytes.len() - 2];
        let r = parse_tlv_stream(truncated);
        assert!(r.entries.is_empty());
    }

    /// CRC mismatch surfaces via `crc_ok=false` but does not break
    /// the walk — the next frame is still parsed.
    #[test]
    fn parse_crc_mismatch_marks_entry_continues_walk() {
        let mut bad = frame_bytes(MSG_TYPE_EXIT, b"payload");
        // Mutate the payload AFTER frame_bytes computed CRC.
        bad[FRAME_HEADER_SIZE] ^= 0xFF;
        let mut good = frame_bytes(MSG_TYPE_STIMULUS, b"valid");
        let mut combined = Vec::new();
        combined.append(&mut bad);
        combined.append(&mut good);
        let r = parse_tlv_stream(&combined);
        assert_eq!(r.entries.len(), 2);
        assert!(!r.entries[0].crc_ok);
        assert!(r.entries[1].crc_ok);
    }

    /// Zero-length payload: header followed by no bytes still
    /// produces a valid empty-payload entry.
    #[test]
    fn parse_zero_length_payload() {
        let bytes = frame_bytes(MSG_TYPE_EXIT, b"");
        assert_eq!(bytes.len(), FRAME_HEADER_SIZE);
        let r = parse_tlv_stream(&bytes);
        assert_eq!(r.entries.len(), 1);
        assert!(r.entries[0].payload.is_empty());
        assert!(r.entries[0].crc_ok);
    }

    /// `drain_bulk` against a freshly-constructed VirtioConsole
    /// yields no entries — the device's TX buffer is empty.
    #[test]
    fn drain_bulk_empty_device_yields_no_entries() {
        let mut dev = VirtioConsole::new();
        let r = drain_bulk(&mut dev);
        assert!(r.entries.is_empty());
    }

    /// Hostile-input guard: a header announcing
    /// `length > MAX_BULK_FRAME_PAYLOAD` is rejected before any
    /// per-frame allocation. The walk stops at the bogus header —
    /// any trailing bytes are unparseable because the bogus length
    /// cannot be trusted to advance the cursor past the claimed
    /// payload.
    #[test]
    fn parse_rejects_oversized_announced_length() {
        use zerocopy::IntoBytes;
        let bad = ShmMessage {
            msg_type: MSG_TYPE_STIMULUS,
            length: u32::MAX,
            crc32: 0,
            _pad: 0,
        };
        let r = parse_tlv_stream(bad.as_bytes());
        assert!(
            r.entries.is_empty(),
            "header announcing u32::MAX must be rejected without producing entries"
        );

        // Boundary: `length == MAX_BULK_FRAME_PAYLOAD + 1` must
        // also be rejected. The cap check uses strict
        // greater-than, so a frame at exactly the cap is allowed
        // (covered by `parse_accepts_at_cap_payload`).
        let just_over = ShmMessage {
            msg_type: MSG_TYPE_STIMULUS,
            length: MAX_BULK_FRAME_PAYLOAD + 1,
            crc32: 0,
            _pad: 0,
        };
        // Construct a buffer that DOES contain enough bytes to
        // satisfy the announced length, so the cap check, not the
        // truncation check, is the rejection path under test.
        let mut buf = Vec::with_capacity(FRAME_HEADER_SIZE + just_over.length as usize);
        buf.extend_from_slice(just_over.as_bytes());
        buf.resize(FRAME_HEADER_SIZE + just_over.length as usize, 0xAA);
        let r2 = parse_tlv_stream(&buf);
        assert!(
            r2.entries.is_empty(),
            "header announcing cap + 1 must be rejected by the per-frame cap check"
        );
    }

    /// Boundary: a frame with `length == MAX_BULK_FRAME_PAYLOAD`
    /// must be accepted. The cap check uses strict greater-than
    /// so a payload that exactly hits the cap is legitimate.
    #[test]
    fn parse_accepts_at_cap_payload() {
        let max_payload = vec![0x55u8; MAX_BULK_FRAME_PAYLOAD as usize];
        let bytes = frame_bytes(MSG_TYPE_STIMULUS, &max_payload);
        let r = parse_tlv_stream(&bytes);
        assert_eq!(
            r.entries.len(),
            1,
            "frame with length == cap must be accepted"
        );
        assert_eq!(r.entries[0].payload.len(), MAX_BULK_FRAME_PAYLOAD as usize);
        assert!(r.entries[0].crc_ok);
    }

    /// A valid frame followed by an oversized header: the walk
    /// returns the valid frame and stops at the oversized header.
    /// Pins that the cap check breaks the walk cleanly without
    /// dropping already-parsed entries.
    #[test]
    fn parse_returns_valid_frame_then_stops_at_oversized() {
        use zerocopy::IntoBytes;
        let mut combined = Vec::new();
        combined.extend_from_slice(&frame_bytes(MSG_TYPE_EXIT, b"valid"));
        let bad = ShmMessage {
            msg_type: MSG_TYPE_STIMULUS,
            length: u32::MAX,
            crc32: 0,
            _pad: 0,
        };
        combined.extend_from_slice(bad.as_bytes());
        // Trailing bytes — a hostile guest can pad whatever it
        // likes after the bogus header.
        combined.extend_from_slice(b"residue");
        let r = parse_tlv_stream(&combined);
        assert_eq!(
            r.entries.len(),
            1,
            "valid frame must be returned even though the next header is bogus"
        );
        assert_eq!(r.entries[0].payload, b"valid");
        assert!(r.entries[0].crc_ok);
    }

    /// An oversized header followed by a fully-valid frame: the
    /// walk stops at the oversized header and the trailing valid
    /// frame is NOT returned. The bogus `length` cannot be trusted
    /// to advance the cursor past its claimed payload, so any
    /// bytes that follow are unparseable — even when those bytes
    /// happen to encode a structurally legitimate frame.
    #[test]
    fn parse_stops_at_oversized_does_not_return_subsequent_valid() {
        use zerocopy::IntoBytes;
        let bad = ShmMessage {
            msg_type: MSG_TYPE_STIMULUS,
            length: u32::MAX,
            crc32: 0,
            _pad: 0,
        };
        let mut combined = Vec::new();
        combined.extend_from_slice(bad.as_bytes());
        // A perfectly-formed frame after the bogus header. If the
        // parser were to skip past the oversized header instead of
        // stopping, this frame would be picked up.
        combined.extend_from_slice(&frame_bytes(MSG_TYPE_EXIT, b"valid"));
        let r = parse_tlv_stream(&combined);
        assert!(
            r.entries.is_empty(),
            "no entries: parser must stop at the oversized header and not resume on the trailing valid frame"
        );
    }

    /// Every new bincode-migration MsgType variant round-trips
    /// through frame-bytes → `parse_tlv_stream` → `MsgType::from_wire`
    /// without dropping the payload. A regression that diverged the
    /// guest writer's tag from the host parser's recogniser would
    /// trip the `from_wire` lookup and surface as `None`.
    #[test]
    fn parse_recognises_all_new_msg_type_variants() {
        use super::super::wire::{
            MSG_TYPE_DMESG, MSG_TYPE_EXEC_EXIT, MSG_TYPE_LIFECYCLE, MSG_TYPE_PROBE_OUTPUT,
            MSG_TYPE_SCHED_LOG, MSG_TYPE_STDERR, MSG_TYPE_STDOUT, MsgType,
        };
        let cases: &[(u32, MsgType, &[u8])] = &[
            (MSG_TYPE_STDOUT, MsgType::Stdout, b"hello\n"),
            (MSG_TYPE_STDERR, MsgType::Stderr, b"error\n"),
            (MSG_TYPE_SCHED_LOG, MsgType::SchedLog, b"---SCHED---\n"),
            // Lifecycle payload layout: 1-byte phase + reason
            // bytes. Use the InitStarted phase (=1) here.
            (MSG_TYPE_LIFECYCLE, MsgType::Lifecycle, &[1u8]),
            (MSG_TYPE_EXEC_EXIT, MsgType::ExecExit, &0i32.to_le_bytes()),
            (MSG_TYPE_DMESG, MsgType::Dmesg, b"[    0.000000] Linux\n"),
            (MSG_TYPE_PROBE_OUTPUT, MsgType::ProbeOutput, b"{\"k\":1}\n"),
        ];
        for (raw, typed, payload) in cases {
            let bytes = frame_bytes(*raw, payload);
            let r = parse_tlv_stream(&bytes);
            assert_eq!(
                r.entries.len(),
                1,
                "single-frame parse failed for {typed:?}",
            );
            assert!(r.entries[0].crc_ok, "CRC must round-trip for {typed:?}");
            assert_eq!(
                r.entries[0].payload, *payload,
                "payload byte mismatch for {typed:?}",
            );
            assert_eq!(
                MsgType::from_wire(r.entries[0].msg_type),
                Some(*typed),
                "from_wire decode mismatch for {typed:?}",
            );
        }
    }

    /// `is_coordinator_internal` flips on for the two control frames
    /// every host-side bucketing path filters out. This is a
    /// classifier-level test (mirrors the one in `wire::tests`) but
    /// pinning it here too guards the host_comms consumer from a
    /// future MsgType variant addition that silently joined the
    /// internal set.
    #[test]
    fn parsed_entries_match_is_coordinator_internal_classifier() {
        use super::super::wire::{
            MSG_TYPE_SNAPSHOT_REQUEST, MSG_TYPE_SYS_RDY, MSG_TYPE_TEST_RESULT, MsgType,
        };
        let internal_raw = frame_bytes(MSG_TYPE_SNAPSHOT_REQUEST, &[0u8; 72]);
        let r = parse_tlv_stream(&internal_raw);
        assert_eq!(r.entries.len(), 1);
        let typed = MsgType::from_wire(r.entries[0].msg_type).unwrap();
        assert!(typed.is_coordinator_internal());

        let internal_sys_rdy = frame_bytes(MSG_TYPE_SYS_RDY, b"");
        let r = parse_tlv_stream(&internal_sys_rdy);
        assert_eq!(r.entries.len(), 1);
        let typed = MsgType::from_wire(r.entries[0].msg_type).unwrap();
        assert!(typed.is_coordinator_internal());

        // Verdict-bearing — must NOT be classified as internal.
        let verdict = frame_bytes(MSG_TYPE_TEST_RESULT, b"\x00");
        let r = parse_tlv_stream(&verdict);
        assert_eq!(r.entries.len(), 1);
        let typed = MsgType::from_wire(r.entries[0].msg_type).unwrap();
        assert!(!typed.is_coordinator_internal());
    }
}
