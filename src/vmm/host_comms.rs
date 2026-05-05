//! Host-only typed consumers for the guest-bound bulk TLV stream and
//! the SHM doorbell + virtio-console wake-byte control plane.
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
//!   RX wake bytes that the guest's `shm_poll_loop` recognises
//!   directly; `request_dump` additionally writes the SHM
//!   `DUMP_REQ_OFFSET` byte so the SysRq-D path triggers on the
//!   next loop iteration.
//!
//! - [`snapshot_request_id`] reads the SHM doorbell-paired request
//!   id; the doorbell page itself survives the SHM-ring elimination
//!   because it is the kernel ioeventfd target.
//!
//! # No drop counter
//!
//! The bulk channel uses the kernel virtio_console TX path:
//! `port_fops_write` blocks the writer until the host's `add_used`
//! rate catches up. Backpressure replaces drops, so
//! [`BulkDrainResult`] carries no `drops` field — every byte the
//! guest emitted is delivered, in order. Compare to the legacy SHM
//! ring's [`super::shm_ring::ShmDrainResult`] which exposed a `drops`
//! counter for ring-full conditions.
//!
//! The lib build does not yet exercise every public helper in this
//! module — the freeze coordinator's TX_EVT epoll arm uses the
//! [`super::bulk::HostAssembler`] streaming path directly, and the
//! `snapshot_request_id` / `drain_bulk` / `parse_tlv_stream` entries
//! are present for downstream test code. `#[allow(dead_code)]`
//! matches the public-surface pattern used on `VmResult` and the
//! legacy `ShmDrainResult`.

#![allow(dead_code)]

use std::sync::Arc;
use zerocopy::FromBytes;

use super::pi_mutex::PiMutex;
use super::shm_ring;
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
        // Guard against a torn frame whose `length` claims more
        // payload than the buffer holds. Stop parsing rather than
        // attempting an over-read or oversized allocation.
        if (msg.length as usize) > buf.len().saturating_sub(hdr_end) {
            break;
        }
        let payload_end = hdr_end + msg.length as usize;
        let payload = buf[hdr_end..payload_end].to_vec();
        let computed_crc = crc32fast::hash(&payload);
        entries.push(ShmEntry {
            msg_type: msg.msg_type,
            payload,
            crc_ok: computed_crc == msg.crc32,
        });
        pos = payload_end;
    }
    BulkDrainResult { entries }
}

/// Push a SysRq-D dump request to the guest. Combines the SHM
/// control-byte write (`DUMP_REQ_OFFSET`) with a virtio-console
/// wake byte so the guest's `shm_poll_loop` re-reads the control
/// byte without waiting for its 200 ms poll cadence.
///
/// `mem` is the host's view of guest DRAM (via `GuestMem`).
/// `shm_base` is the offset of the SHM region.
/// `virtio_con` carries the wake-byte transport.
///
/// Lock acquisition: SHM byte first, then the wake byte; this order
/// guarantees the guest reads the new control byte after observing
/// the wake. Reversed order would let the guest re-read the SHM
/// byte BEFORE we wrote it, miss the new value, and stall until the
/// next 200 ms poll.
pub fn request_dump(
    mem: &crate::monitor::reader::GuestMem,
    shm_base: u64,
    virtio_con: &Arc<PiMutex<VirtioConsole>>,
) {
    mem.write_u8(
        shm_base,
        shm_ring::DUMP_REQ_OFFSET,
        shm_ring::DUMP_REQ_SYSRQ_D,
    );
    virtio_con.lock().queue_input(&[SIGNAL_VC_DUMP]);
}

/// Push a graceful-shutdown request to the guest by sending the
/// `SIGNAL_VC_SHUTDOWN` byte through the virtio-console RX queue.
/// The guest's `shm_poll_loop` recognises the byte directly and
/// drives the graceful-shutdown teardown — no SHM signal slot
/// involved.
pub fn request_shutdown(virtio_con: &Arc<PiMutex<VirtioConsole>>) {
    virtio_con.lock().queue_input(&[SIGNAL_VC_SHUTDOWN]);
}

/// Notify the guest that the host's `bpf-map-write` thread finished
/// applying every queued `bpf_map_write`. Pushes
/// `SIGNAL_BPF_WRITE_DONE` through the virtio-console RX queue; the
/// guest's `shm_poll_loop` recognises the byte and sets the
/// `bpf_map_write_done` latch so a scenario blocked on
/// [`crate::scenario::Ctx::wait_for_map_write`] resumes. Replaces the
/// legacy SHM signal-slot rendezvous (host writes slot 0, guest blocks
/// on slot 0) with a single wake byte.
pub fn request_bpf_map_write_done(virtio_con: &Arc<PiMutex<VirtioConsole>>) {
    virtio_con.lock().queue_input(&[SIGNAL_BPF_WRITE_DONE]);
}

/// Read the host-side view of an in-flight snapshot request. Returns
/// the request id the guest most recently published in
/// [`shm_ring::SNAPSHOT_REQUEST_ID_OFFSET`], 0 when no request is
/// pending.
///
/// The freeze coordinator polls this slot from its doorbell-event
/// dispatch path; non-zero means there is work to do.
pub fn snapshot_request_id(mem: &crate::monitor::reader::GuestMem, shm_base: u64) -> u32 {
    mem.read_u32(shm_base, shm_ring::SNAPSHOT_REQUEST_ID_OFFSET)
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
}
