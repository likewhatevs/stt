//! Host-side TLV stream assembler for the virtio-console port-1 bulk
//! channel.
//!
//! The guest writes [`super::wire::ShmMessage`]-prefixed messages through
//! `/dev/vport0p1`. The host VMM accumulates the byte stream into
//! [`super::virtio_console::VirtioConsole::drain_bulk`] and feeds each
//! drain into [`HostAssembler::feed`] which yields complete
//! `BulkMessage` values; partial trailing bytes are preserved across
//! calls so a frame split across multiple wakes (descriptor-sized
//! page boundaries, kernel write_all chunking) is recovered without
//! loss.
//!
//! # Why not just call `parse_tlv_stream` once at end-of-run
//!
//! The freeze coordinator promotes a guest-side
//! [`super::wire::MSG_TYPE_SCHED_EXIT`] frame into the run-wide kill
//! flag so a scheduler that exits early ends the test promptly
//! instead of waiting for the watchdog. Mid-run frame visibility
//! requires streaming assembly with partial-frame retention.

use crc32fast;
use zerocopy::FromBytes;

use super::wire::{FRAME_HEADER_SIZE, ShmMessage};

/// Per-frame payload cap enforced by the host assembler.
///
/// The on-wire `length` field is `u32` so a malformed or hostile guest
/// can announce a 4 GiB payload. Without this cap [`HostAssembler::feed`]
/// would buffer the announced bytes (waiting for the frame to complete)
/// up to that 4 GiB, opening an OOM vector. 256 KiB exceeds every real
/// payload type — coverage `Profraw` blobs and LLM
/// `RawPayloadOutput` JSON are well below this — and is large enough
/// that legitimate guest traffic never trips it. A frame whose
/// announced length exceeds the cap is dropped and the assembler
/// resyncs by clearing residue (the announced length cannot be trusted
/// to advance past the bogus payload).
pub const MAX_BULK_FRAME_PAYLOAD: u32 = 256 * 1024;

/// One complete message extracted from the bulk byte stream.
///
/// `payload` is read by external consumers via the public bulk API;
/// internal lib code only inspects `msg_type` + `crc_ok` for the
/// SCHED_EXIT promotion gate. `#[allow(dead_code)]` mirrors the
/// pattern on `VmResult::stimulus_events` and `Snapshot` accessor
/// types — fields part of the public surface that the lib build
/// does not internally read.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BulkMessage {
    pub msg_type: u32,
    pub payload: Vec<u8>,
    /// True when the per-frame CRC matched the recomputed payload
    /// CRC. Mirrors [`super::shm_ring::ShmEntry::crc_ok`] so the
    /// downstream consumer can apply the same crc-gate that the
    /// SHM ring path used.
    pub crc_ok: bool,
}

/// Output of one [`HostAssembler::feed`] call.
#[derive(Debug, Default)]
pub struct BulkMessages {
    /// Complete messages assembled from this drain plus any leftover
    /// from the previous drain.
    pub messages: Vec<BulkMessage>,
}

/// Streaming assembler for the bulk TLV byte stream.
///
/// Holds an internal buffer that accumulates incomplete frames across
/// successive `feed` calls. Every call drains every complete frame
/// from the front of the buffer and returns them via [`BulkMessages`];
/// any trailing partial frame stays in the buffer for the next call.
///
/// # Memory bounds
///
/// The buffer grows with the largest in-flight payload, capped by
/// [`MAX_BULK_FRAME_PAYLOAD`]. A frame whose announced `length`
/// exceeds the cap is dropped (the bogus header plus any buffered
/// residue is cleared) and the assembler resyncs on the next feed.
/// Without the cap a hostile guest could announce a 4 GiB payload via
/// the on-wire `u32` length field and the assembler would buffer up
/// to that bound waiting for completion — an OOM vector. Mirrors the
/// `max_payload` gate in [`super::shm_ring::shm_drain`] which rejects
/// implausible lengths before allocation.
#[derive(Debug, Default)]
pub struct HostAssembler {
    /// Bytes received but not yet assembled into a complete frame.
    /// On a clean stream this is empty after every `feed`; only a
    /// drain whose tail bytes contain a partial header / partial
    /// payload leaves residue.
    buf: Vec<u8>,
}

impl HostAssembler {
    /// Construct a fresh assembler with an empty buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed `bytes` into the assembler, parse every complete frame
    /// from the front of the accumulated buffer, and return them.
    /// Partial trailing frames stay in the buffer for the next
    /// call — the caller may invoke [`Self::feed`] with an empty
    /// slice to retry assembly without producing additional input.
    pub fn feed(&mut self, bytes: &[u8]) -> BulkMessages {
        if !bytes.is_empty() {
            self.buf.extend_from_slice(bytes);
        }
        let mut out: Vec<BulkMessage> = Vec::new();
        let mut consumed = 0usize;
        let mut resync = false;
        while consumed.saturating_add(FRAME_HEADER_SIZE) <= self.buf.len() {
            let hdr_end = consumed + FRAME_HEADER_SIZE;
            let hdr_slice = &self.buf[consumed..hdr_end];
            let Ok(frame) = ShmMessage::read_from_bytes(hdr_slice) else {
                // Cannot fail — slice is exactly FRAME_HEADER_SIZE
                // bytes and ShmMessage is FromBytes. Defensive bail.
                break;
            };
            if frame.length > MAX_BULK_FRAME_PAYLOAD {
                // Hostile-guest defense: an announced length above the
                // cap (4 GiB max via u32) would buffer until OOM while
                // we wait for the frame to complete. The announced
                // length cannot be trusted to advance past the bogus
                // payload, so drop the entire buffer and resync.
                tracing::warn!(
                    msg_type = frame.msg_type,
                    length = frame.length,
                    cap = MAX_BULK_FRAME_PAYLOAD,
                    "bulk assembler: dropping oversized frame; resyncing"
                );
                resync = true;
                break;
            }
            let payload_len = frame.length as usize;
            let payload_end = hdr_end.saturating_add(payload_len);
            if payload_end > self.buf.len() {
                // Incomplete payload — wait for more bytes.
                break;
            }
            let payload = self.buf[hdr_end..payload_end].to_vec();
            let computed = crc32fast::hash(&payload);
            out.push(BulkMessage {
                msg_type: frame.msg_type,
                payload,
                crc_ok: computed == frame.crc32,
            });
            consumed = payload_end;
        }
        if resync {
            // Stream sync is lost after a corrupt frame; drop every
            // buffered byte rather than risk re-interpreting payload
            // bytes as a fresh header.
            self.buf.clear();
        } else if consumed > 0 {
            self.buf.drain(..consumed);
        }
        BulkMessages { messages: out }
    }

    /// Bytes still buffered (not yet assembled into a complete
    /// frame). Diagnostic accessor; production callers do not need
    /// this.
    #[cfg(test)]
    pub fn pending(&self) -> usize {
        self.buf.len()
    }

    /// Take the residual partial-frame bytes, leaving the assembler
    /// empty.
    ///
    /// Called by the freeze coordinator at thread exit so a partial
    /// TLV frame (header or payload split across the last few
    /// drains) is not lost when the assembler is dropped. The
    /// returned bytes are pushed back onto the device's
    /// `port1_tx_buf` via
    /// [`super::virtio_console::VirtioConsole::push_back_bulk`] so
    /// `collect_results`'s end-of-run `drain_bulk` +
    /// `parse_tlv_stream` path completes the frame.
    pub fn take_residual(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.buf)
    }
}

#[cfg(test)]
mod tests {
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

    /// One complete frame in one feed produces one message.
    #[test]
    fn single_frame_one_feed() {
        let mut a = HostAssembler::new();
        let bytes = frame_bytes(MSG_TYPE_EXIT, &42i32.to_le_bytes());
        let r = a.feed(&bytes);
        assert_eq!(r.messages.len(), 1);
        assert_eq!(r.messages[0].msg_type, MSG_TYPE_EXIT);
        assert!(r.messages[0].crc_ok);
        assert_eq!(a.pending(), 0);
    }

    /// Multiple complete frames in one feed produce multiple messages
    /// in order.
    #[test]
    fn multiple_frames_one_feed_preserve_order() {
        let mut a = HostAssembler::new();
        let mut buf = Vec::new();
        buf.extend(frame_bytes(MSG_TYPE_STIMULUS, b"first"));
        buf.extend(frame_bytes(MSG_TYPE_EXIT, b"second"));
        buf.extend(frame_bytes(MSG_TYPE_STIMULUS, b"third"));
        let r = a.feed(&buf);
        assert_eq!(r.messages.len(), 3);
        assert_eq!(r.messages[0].payload, b"first");
        assert_eq!(r.messages[1].payload, b"second");
        assert_eq!(r.messages[2].payload, b"third");
    }

    /// A frame split across two feed calls is recovered intact —
    /// this is THE invariant that justifies the streaming
    /// assembler's existence vs. a one-shot end-of-run parse.
    #[test]
    fn frame_split_across_two_feeds() {
        let mut a = HostAssembler::new();
        let bytes = frame_bytes(MSG_TYPE_EXIT, b"payload-data");
        // Split mid-payload (after header + 3 bytes of payload).
        let split = FRAME_HEADER_SIZE + 3;
        let r1 = a.feed(&bytes[..split]);
        assert!(r1.messages.is_empty(), "partial frame must yield nothing");
        assert_eq!(a.pending(), split);
        let r2 = a.feed(&bytes[split..]);
        assert_eq!(r2.messages.len(), 1, "completing bytes must yield 1 frame");
        assert_eq!(r2.messages[0].payload, b"payload-data");
        assert_eq!(a.pending(), 0);
    }

    /// Partial header (fewer than 16 bytes) yields no message and
    /// leaves the bytes buffered.
    #[test]
    fn partial_header_buffered() {
        let mut a = HostAssembler::new();
        let r = a.feed(&[0xAA, 0xBB, 0xCC]);
        assert!(r.messages.is_empty());
        assert_eq!(a.pending(), 3);
    }

    /// Empty feed is a no-op and produces no messages.
    #[test]
    fn empty_feed_noop() {
        let mut a = HostAssembler::new();
        let r = a.feed(&[]);
        assert!(r.messages.is_empty());
        assert_eq!(a.pending(), 0);
    }

    /// CRC mismatch surfaces via `crc_ok=false` but does not break
    /// the walk — the next frame is still parsed.
    #[test]
    fn crc_mismatch_marks_entry_continues_walk() {
        let mut a = HostAssembler::new();
        // Hand-craft a frame whose CRC does NOT match the payload.
        let mut bad = frame_bytes(MSG_TYPE_EXIT, b"payload");
        // Mutate the payload AFTER frame_bytes computed CRC.
        bad[FRAME_HEADER_SIZE] ^= 0xFF;
        let mut good = frame_bytes(MSG_TYPE_STIMULUS, b"valid");
        let mut combined = Vec::new();
        combined.append(&mut bad);
        combined.append(&mut good);
        let r = a.feed(&combined);
        assert_eq!(r.messages.len(), 2);
        assert!(!r.messages[0].crc_ok);
        assert!(r.messages[1].crc_ok);
    }

    /// Byte-by-byte feed reconstructs the same frame as a one-shot
    /// feed — the assembler must not lose bytes at any chunking
    /// boundary.
    #[test]
    fn byte_by_byte_feed_reconstructs_frame() {
        let mut a = HostAssembler::new();
        let bytes = frame_bytes(MSG_TYPE_EXIT, b"hello-world");
        let mut total = Vec::new();
        for b in &bytes {
            let r = a.feed(std::slice::from_ref(b));
            total.extend(r.messages);
        }
        assert_eq!(total.len(), 1);
        assert_eq!(total[0].payload, b"hello-world");
    }

    /// Zero-length payload: header followed by no bytes still
    /// produces a valid empty-payload message.
    #[test]
    fn zero_length_payload() {
        let mut a = HostAssembler::new();
        let bytes = frame_bytes(MSG_TYPE_EXIT, b"");
        assert_eq!(bytes.len(), FRAME_HEADER_SIZE);
        let r = a.feed(&bytes);
        assert_eq!(r.messages.len(), 1);
        assert!(r.messages[0].payload.is_empty());
        assert!(r.messages[0].crc_ok);
    }

    /// Hostile-guest defense: a frame header announcing
    /// `length = u32::MAX` must be rejected without growing the
    /// buffer toward 4 GiB. Without [`MAX_BULK_FRAME_PAYLOAD`] the
    /// assembler would buffer subsequent bytes until OOM waiting for
    /// the impossibly large frame to complete.
    #[test]
    fn assembler_rejects_enormous_frame_length() {
        let mut a = HostAssembler::new();
        // Hand-craft a header with u32::MAX length — exceeds the cap.
        let bad = ShmMessage {
            msg_type: MSG_TYPE_EXIT,
            length: u32::MAX,
            crc32: 0,
            _pad: 0,
        };
        let r = a.feed(bad.as_bytes());
        // No message produced — frame is dropped.
        assert!(
            r.messages.is_empty(),
            "oversized frame must not yield any message"
        );
        // Buffer must not retain the bogus header (otherwise the
        // assembler would re-warn on every subsequent feed and could
        // grow without bound as the guest streams more bytes).
        assert_eq!(
            a.pending(),
            0,
            "buffer must be cleared after dropping an oversized frame"
        );
        // Subsequent bytes must not be appended to a pre-allocated
        // 4 GiB buffer. Even after streaming additional bytes, the
        // assembler stays bounded — the corrupt frame did not cause
        // capacity inflation toward `length`.
        let mut blast = Vec::with_capacity(64 * 1024);
        blast.resize(64 * 1024, 0xAAu8);
        let _ = a.feed(&blast);
        assert!(
            a.buf.capacity() < 1024 * 1024 * 1024,
            "buffer capacity must not approach the announced 4 GiB length \
             (saw {} bytes)",
            a.buf.capacity()
        );
    }

    /// Cap boundary: a frame with `length == MAX_BULK_FRAME_PAYLOAD`
    /// is accepted; one byte over the cap is rejected. Pins the
    /// strict-greater-than comparison.
    #[test]
    fn assembler_accepts_at_cap_rejects_above() {
        let mut a = HostAssembler::new();
        let max_payload = vec![0x55u8; MAX_BULK_FRAME_PAYLOAD as usize];
        let at_cap = frame_bytes(MSG_TYPE_STIMULUS, &max_payload);
        let r = a.feed(&at_cap);
        assert_eq!(
            r.messages.len(),
            1,
            "frame with length == cap must be accepted"
        );
        assert_eq!(r.messages[0].payload.len(), MAX_BULK_FRAME_PAYLOAD as usize);
        assert!(r.messages[0].crc_ok);

        // Now feed a header announcing one byte over the cap.
        let mut b = HostAssembler::new();
        let over = ShmMessage {
            msg_type: MSG_TYPE_STIMULUS,
            length: MAX_BULK_FRAME_PAYLOAD + 1,
            crc32: 0,
            _pad: 0,
        };
        let r2 = b.feed(over.as_bytes());
        assert!(
            r2.messages.is_empty(),
            "frame with length == cap + 1 must be rejected"
        );
        assert_eq!(b.pending(), 0);
    }

    /// A valid frame followed by an oversized frame: the valid frame
    /// is returned, the oversized frame is dropped, and the buffer
    /// is fully cleared (resync drops residue too).
    #[test]
    fn good_frame_then_bad_frame_returns_good_drops_bad() {
        let mut a = HostAssembler::new();
        let good = frame_bytes(MSG_TYPE_EXIT, b"valid");
        let bad = ShmMessage {
            msg_type: MSG_TYPE_STIMULUS,
            length: u32::MAX,
            crc32: 0,
            _pad: 0,
        };
        let mut combined = Vec::new();
        combined.extend_from_slice(&good);
        combined.extend_from_slice(bad.as_bytes());
        // Add tail bytes so resync clearing the residue is observable.
        combined.extend_from_slice(b"residue-bytes");
        let r = a.feed(&combined);
        assert_eq!(r.messages.len(), 1, "valid frame must still be returned");
        assert_eq!(r.messages[0].payload, b"valid");
        assert!(r.messages[0].crc_ok);
        assert_eq!(
            a.pending(),
            0,
            "resync must clear the bogus header AND the trailing residue"
        );
    }
}
