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

use std::sync::Arc;

use zerocopy::FromBytes;

use super::wire::{FRAME_HEADER_SIZE, ShmMessage};

/// Per-frame payload cap enforced by the host assembler.
///
/// The on-wire `length` field is `u32` so a malformed or hostile guest
/// can announce a 4 GiB payload. 256 KiB exceeds every real payload
/// type — coverage `Profraw` blobs and LLM `RawPayloadOutput` JSON
/// are well below this — and is large enough that legitimate guest
/// traffic never trips it. The cap check fires only after the
/// complete frame has accumulated in the buffer (header +
/// `length` payload bytes), so a header arriving ahead of its
/// payload via split writev is not falsely rejected. Once the full
/// frame is observed and `length > cap`, the assembler resyncs by
/// clearing the entire buffer (the announced length cannot be
/// trusted to advance past the bogus payload). The buffer itself
/// only grows by bytes the caller fed, never by the announced
/// length, so a hostile header does not inflate
/// [`HostAssembler::buf`]'s capacity toward 4 GiB. The
/// complementary residual-buffer cap inside [`HostAssembler::feed`]
/// (2 × this value) closes the gap when announced lengths sit
/// below the per-frame cap but exceed delivered bytes, or when
/// hostile traffic dribbles many partial frames whose payloads
/// never arrive.
///
/// The same cap also guards [`super::host_comms::parse_tlv_stream`]:
/// the one-shot parser rejects frames whose announced `length`
/// exceeds the cap so a corrupt or hostile buffer cannot trigger
/// an oversized per-frame allocation downstream.
pub const MAX_BULK_FRAME_PAYLOAD: u32 = 256 * 1024;

/// One complete message extracted from the bulk byte stream.
///
/// `payload` is read by external consumers via the public bulk API;
/// internal lib code only inspects `msg_type` + `crc_ok` for the
/// SCHED_EXIT promotion gate. `#[allow(dead_code)]` mirrors the
/// pattern on `VmResult::stimulus_events` and `Snapshot` accessor
/// types — fields part of the public surface that the lib build
/// does not internally read.
///
/// `payload` is `Arc<[u8]>` so cloning a `BulkMessage` (e.g. when
/// the freeze coordinator stashes parsed frames into the shared
/// `bulk_messages` buffer for `collect_results`) is a refcount bump
/// rather than a heap allocation + memcpy of the full payload bytes.
/// The single per-frame allocation still occurs inside
/// [`HostAssembler::feed`] when the assembler materialises the
/// payload from its accumulator buffer; downstream cloning is
/// O(1).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BulkMessage {
    pub msg_type: u32,
    pub payload: Arc<[u8]>,
    /// True when the per-frame CRC matched the recomputed payload
    /// CRC. Mirrors [`super::wire::ShmEntry::crc_ok`] so the
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
/// The buffer grows with the bytes the host has actually received,
/// not with announced frame lengths. Two complementary caps protect
/// against unbounded growth:
///
/// 1. **Per-frame cap** ([`MAX_BULK_FRAME_PAYLOAD`]): a frame whose
///    announced `length` exceeds the cap is dropped (the header,
///    payload bytes, and any trailing residue cleared) only after
///    the full announced length has accumulated — the cap check
///    fires post-completion so a partial header that arrives ahead
///    of its payload (split writev: the writer's
///    `write_to_bulk_port` emits two iovecs and the kernel
///    virtio_console driver's `port_fops_write` may flush them on
///    separate trips through the device's `port1_tx_buf`) is not
///    misclassified as oversized.
///
/// 2. **Residual-buffer cap** (`2 × MAX_BULK_FRAME_PAYLOAD`): closes
///    the gap left by the post-completion check. A hostile guest
///    can dribble headers (16 bytes each) announcing payloads it
///    never intends to send, or announce `length` values below the
///    per-frame cap that exceed what it actually delivers; the
///    partial-frame retention path would otherwise let the buffer
///    grow toward `port1_tx_buf`'s capacity. Once the residual
///    exceeds twice the per-frame cap, the buffer cannot contain a
///    legitimate single frame plus a partial follow-on — it must
///    be junk or attack traffic — and the assembler clears it and
///    resyncs.
///
/// The buffer never pre-allocates against the announced length —
/// `extend_from_slice` only grows by what the caller fed, so a
/// header announcing 4 GiB does not inflate `Vec::capacity` by
/// 4 GiB. Mirrors the `MAX_BULK_FRAME_PAYLOAD` cap in
/// [`super::host_comms::parse_tlv_stream`] which rejects implausible
/// lengths before allocating the per-frame payload buffer.
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
            let payload_len = frame.length as usize;
            let payload_end = hdr_end.saturating_add(payload_len);
            if payload_end > self.buf.len() {
                // Incomplete payload — wait for more bytes. Defer the
                // cap check below until we have observed the full
                // frame: a partial header arriving via split writev
                // (the writer's `write_to_bulk_port` emits two iovecs
                // and the kernel virtio_console driver's
                // `port_fops_write` may flush them on separate trips
                // through the device's `port1_tx_buf`) cannot be
                // distinguished from a corrupt header until the
                // payload bytes either arrive or fail to. Resyncing
                // here would discard a legitimate large frame whose
                // header happens to drain before the rest of its
                // bytes; legitimate frames have `length` ≤ cap so
                // the post-completion check will accept them.
                break;
            }
            if frame.length > MAX_BULK_FRAME_PAYLOAD {
                // Hostile-guest defense: an announced length above
                // the cap (4 GiB max via u32) cannot be a legitimate
                // frame — every real producer's payload sits well
                // below 256 KiB. We have now seen every byte the
                // header claimed; drop the entire buffer (header +
                // payload + any trailing residue) and resync. The
                // announced length cannot be trusted to advance past
                // the bogus payload, so partial bytes after this
                // frame are also unparsable.
                tracing::warn!(
                    msg_type = frame.msg_type,
                    length = frame.length,
                    cap = MAX_BULK_FRAME_PAYLOAD,
                    "bulk assembler: dropping oversized frame; resyncing"
                );
                resync = true;
                break;
            }
            // Materialise the payload as `Arc<[u8]>` so subsequent
            // clones (the freeze coordinator's stash from
            // `BulkMessage` into the shared `ShmEntry` buffer) are
            // refcount bumps rather than heap allocations + memcpy.
            // `Arc::<[u8]>::from(&[u8])` performs a single
            // allocation + copy of the slice contents into a
            // refcounted boxed slice — same cost as the previous
            // `to_vec()`, but every downstream `clone()` becomes
            // O(1).
            let payload: Arc<[u8]> = Arc::from(&self.buf[hdr_end..payload_end]);
            let computed = crc32fast::hash(&payload);
            let crc_ok = computed == frame.crc32;
            if !crc_ok {
                // Surface the per-frame CRC mismatch for diagnostics.
                // Mirrors `parse_tlv_stream`, which writes the same
                // observation into `ShmEntry::crc_ok`. Downstream
                // consumers may still filter on `crc_ok`, but the
                // operator now sees the corruption in the log rather
                // than learning of it only when a downstream parse
                // fails.
                tracing::warn!(
                    msg_type = frame.msg_type,
                    length = frame.length,
                    expected_crc = frame.crc32,
                    computed_crc = computed,
                    "bulk assembler: per-frame CRC mismatch; surfacing crc_ok=false"
                );
            }
            out.push(BulkMessage {
                msg_type: frame.msg_type,
                payload,
                crc_ok,
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
        // Hostile-guest defense: bound the assembler's residual
        // buffer size. The post-completion `length > cap` check
        // above only fires once a full frame has accumulated; a
        // hostile guest can announce `length = u32::MAX` (or any
        // value below the cap but above the bytes it ever intends
        // to send), then dribble headers without payloads. The
        // partial-frame retention path would let the buffer grow
        // unbounded toward `port1_tx_buf`'s capacity. Once the
        // residual exceeds twice the per-frame cap, the buffer
        // cannot contain a legitimate single frame plus a
        // partial follow-on — it must be junk or attack traffic.
        // Drop everything and resync.
        if self.buf.len() > 2 * MAX_BULK_FRAME_PAYLOAD as usize {
            tracing::warn!(
                pending = self.buf.len(),
                cap = 2 * MAX_BULK_FRAME_PAYLOAD as usize,
                "bulk assembler: pending buffer exceeded 2× per-frame cap; \
                 resyncing to prevent unbounded growth"
            );
            self.buf.clear();
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
        assert_eq!(&*r.messages[0].payload, b"first");
        assert_eq!(&*r.messages[1].payload, b"second");
        assert_eq!(&*r.messages[2].payload, b"third");
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
        assert_eq!(&*r2.messages[0].payload, b"payload-data");
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
        assert_eq!(&*total[0].payload, b"hello-world");
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
    /// `length = u32::MAX` must not pre-allocate the buffer toward
    /// 4 GiB. The assembler waits for the full frame before
    /// applying the cap so a header that drains ahead of its
    /// payload (split writev, partial drain) does not get
    /// misclassified as oversized; once enough bytes arrive to
    /// complete a frame whose announced length exceeds the cap,
    /// the assembler resyncs and drops the buffer. Until then the
    /// buffer holds only what the host received — no
    /// `with_capacity(announced_length)` allocation.
    #[test]
    fn assembler_does_not_pre_allocate_for_enormous_frame_length() {
        let mut a = HostAssembler::new();
        // Hand-craft a header with u32::MAX length — exceeds the cap.
        let bad = ShmMessage {
            msg_type: MSG_TYPE_EXIT,
            length: u32::MAX,
            crc32: 0,
            _pad: 0,
        };
        let r = a.feed(bad.as_bytes());
        // No message produced — payload incomplete (we only fed the
        // header).
        assert!(
            r.messages.is_empty(),
            "header-only feed must not yield any message"
        );
        // Buffer still holds the 16-byte header. The cap check is
        // deferred until the full frame arrives, so the buffer is
        // not cleared yet.
        assert_eq!(
            a.pending(),
            FRAME_HEADER_SIZE,
            "header bytes must remain buffered; cap check is deferred"
        );
        // Subsequent bytes must not be appended to a pre-allocated
        // 4 GiB buffer. Even after streaming additional bytes, the
        // assembler stays bounded — the corrupt header did not cause
        // capacity inflation toward `length`.
        let mut blast = Vec::with_capacity(64 * 1024);
        blast.resize(64 * 1024, 0xAAu8);
        let _ = a.feed(&blast);
        assert!(
            a.buf.capacity() < 1024 * 1024,
            "buffer capacity must not approach the announced 4 GiB length \
             (saw {} bytes)",
            a.buf.capacity()
        );
    }

    /// When a frame whose announced length exceeds the cap is
    /// completed (header + that many payload bytes accumulated),
    /// the assembler resyncs and clears the buffer. Pins the
    /// post-completion cap-check path against an OOM regression
    /// where the buffer would simply keep growing.
    #[test]
    fn assembler_drops_oversized_frame_once_complete() {
        let mut a = HostAssembler::new();
        // Choose a `length` that exceeds the cap but is small enough
        // to fit in test memory (cap + 1 bytes of payload). The
        // post-completion cap check must reject it.
        let oversized_len = MAX_BULK_FRAME_PAYLOAD + 1;
        let bad = ShmMessage {
            msg_type: MSG_TYPE_EXIT,
            length: oversized_len,
            crc32: 0,
            _pad: 0,
        };
        let mut bytes = Vec::new();
        bytes.extend_from_slice(bad.as_bytes());
        bytes.resize(FRAME_HEADER_SIZE + oversized_len as usize, 0xCC);
        // Feed the full frame plus residue.
        bytes.extend_from_slice(b"residue");
        let r = a.feed(&bytes);
        assert!(
            r.messages.is_empty(),
            "oversized frame must not yield any message"
        );
        assert_eq!(
            a.pending(),
            0,
            "resync must clear bogus header, payload, and residue"
        );
    }

    /// Cap boundary: a complete frame with
    /// `length == MAX_BULK_FRAME_PAYLOAD` is accepted; a complete
    /// frame one byte over the cap is rejected. Pins the
    /// strict-greater-than comparison applied after frame
    /// completion.
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

        // Now feed a complete frame announcing one byte over the
        // cap — header + (cap + 1) payload bytes. The
        // post-completion cap check must reject it and resync the
        // buffer.
        let mut b = HostAssembler::new();
        let over_payload = vec![0xAAu8; (MAX_BULK_FRAME_PAYLOAD + 1) as usize];
        let over = ShmMessage {
            msg_type: MSG_TYPE_STIMULUS,
            length: MAX_BULK_FRAME_PAYLOAD + 1,
            crc32: 0,
            _pad: 0,
        };
        let mut over_bytes = Vec::new();
        over_bytes.extend_from_slice(over.as_bytes());
        over_bytes.extend_from_slice(&over_payload);
        let r2 = b.feed(&over_bytes);
        assert!(
            r2.messages.is_empty(),
            "frame with length == cap + 1 must be rejected"
        );
        assert_eq!(
            b.pending(),
            0,
            "resync must clear the bogus frame after observing its full length"
        );
    }

    /// A valid frame followed by an oversized-but-incomplete frame:
    /// the valid frame is returned, and the bogus header bytes plus
    /// any trailing residue stay buffered (the cap check is deferred
    /// until the full frame arrives). The valid frame must still
    /// drain even though the next frame's parse stalls on incomplete
    /// payload — `consumed` advances past the good frame and the
    /// `drain(..consumed)` call removes those bytes from the front.
    #[test]
    fn good_frame_then_oversized_incomplete_frame_returns_good() {
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
        // Add tail bytes — they cannot satisfy the bogus header's
        // u32::MAX length, so the loop stalls on incomplete payload.
        combined.extend_from_slice(b"residue-bytes");
        let r = a.feed(&combined);
        assert_eq!(r.messages.len(), 1, "valid frame must still be returned");
        assert_eq!(&*r.messages[0].payload, b"valid");
        assert!(r.messages[0].crc_ok);
        // Buffer retains the bogus 16-byte header plus 13 residue
        // bytes — the cap check defers until the announced 4 GiB of
        // payload has arrived (it never will), and the post-good-frame
        // bytes stay buffered for the next feed without growing the
        // capacity toward the announced length.
        assert_eq!(
            a.pending(),
            FRAME_HEADER_SIZE + b"residue-bytes".len(),
            "bogus header + residue stay buffered until the announced length \
             is observed"
        );
        assert!(
            a.buf.capacity() < 1024 * 1024 * 1024,
            "buffer capacity must not approach the announced 4 GiB length \
             (saw {} bytes)",
            a.buf.capacity()
        );
    }

    /// Hostile-guest defense: a hostile producer that announces a
    /// length far above the per-frame cap (so the parser stalls
    /// on incomplete payload forever) cannot grow the assembler
    /// buffer past `2 × MAX_BULK_FRAME_PAYLOAD`. The
    /// post-completion per-frame cap check (see
    /// [`good_frame_then_oversized_incomplete_frame_returns_good`])
    /// only fires on full-length arrival; this test pins the
    /// complementary residual-buffer cap that closes the gap by
    /// rejecting buffers that grow past 2× the cap regardless of
    /// announced length.
    #[test]
    fn assembler_clears_buffer_when_residual_exceeds_cap() {
        let mut a = HostAssembler::new();
        // Header announcing `u32::MAX` payload — the parser will
        // stall on incomplete payload indefinitely because no
        // legitimate `port1_tx_buf` can deliver 4 GiB of bytes.
        // Without the residual-buffer cap, every subsequent feed
        // would just grow the buffer.
        let bad = ShmMessage {
            msg_type: MSG_TYPE_STIMULUS,
            length: u32::MAX,
            crc32: 0,
            _pad: 0,
        };
        let mut buf = Vec::new();
        buf.extend_from_slice(bad.as_bytes());
        // Pad past `2 × MAX_BULK_FRAME_PAYLOAD` so the
        // strict-greater-than residual cap triggers.
        let target_len = 2 * MAX_BULK_FRAME_PAYLOAD as usize + 1;
        buf.resize(target_len, 0xCD);
        let r = a.feed(&buf);
        assert!(
            r.messages.is_empty(),
            "no message produced — payload incomplete from the parser's view"
        );
        assert_eq!(
            a.pending(),
            0,
            "residual buffer cap must trigger a clear once the buffer \
             exceeds 2× the per-frame cap"
        );
    }

    /// Boundary case: residual exactly at `2 × MAX_BULK_FRAME_PAYLOAD`
    /// must NOT trip the cap. The strict-greater-than comparison is
    /// load-bearing: a hostile producer that drains the host's
    /// view of `port1_tx_buf` to exactly `2 × cap` bytes — short
    /// of triggering the resync — must leave the buffer
    /// untouched, so a follow-on legitimate frame is not lost
    /// alongside the attack residue.
    #[test]
    fn assembler_accepts_residual_at_2x_cap() {
        let mut a = HostAssembler::new();
        // Header announcing `u32::MAX` so the parser stalls on
        // incomplete payload and does not consume any bytes; pad
        // the buffer to exactly `2 × cap` total bytes so the
        // residual sits AT the bound but not above it.
        let bad = ShmMessage {
            msg_type: MSG_TYPE_STIMULUS,
            length: u32::MAX,
            crc32: 0,
            _pad: 0,
        };
        let total_residual = 2 * MAX_BULK_FRAME_PAYLOAD as usize;
        let mut buf = Vec::new();
        buf.extend_from_slice(bad.as_bytes());
        buf.resize(total_residual, 0xCE);
        let r = a.feed(&buf);
        assert!(
            r.messages.is_empty(),
            "no message produced — parser stalls on incomplete u32::MAX payload"
        );
        assert_eq!(
            a.pending(),
            total_residual,
            "residual at exactly 2× cap must NOT trigger the resync clear"
        );
    }

    /// Payload type round-trip: the assembler emits each
    /// `BulkMessage::payload` as `Arc<[u8]>` so downstream clones
    /// (e.g. the freeze coordinator's
    /// `BulkMessage` → `ShmEntry` stash) become refcount bumps
    /// rather than per-frame heap allocations. Pin the contract
    /// so a future revert to `Vec<u8>` re-introduces the per-clone
    /// allocation cost it was migrated away from.
    #[test]
    fn payload_is_arc_slice() {
        let mut a = HostAssembler::new();
        let bytes = frame_bytes(MSG_TYPE_EXIT, b"arc-payload");
        let r = a.feed(&bytes);
        assert_eq!(r.messages.len(), 1);
        // Clone is a refcount bump; both the original and the
        // clone point at the same allocation.
        let m0 = &r.messages[0];
        let cloned: Arc<[u8]> = m0.payload.clone();
        assert!(
            Arc::ptr_eq(&m0.payload, &cloned),
            "cloning Arc<[u8]> must share the underlying allocation, \
             not deep-copy the bytes"
        );
        assert_eq!(&*cloned, b"arc-payload");
    }
}
