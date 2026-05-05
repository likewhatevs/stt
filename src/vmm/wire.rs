//! Shared wire-format types for the host/guest virtio-console port-1
//! TLV stream and the multiport control protocol.
//!
//! Both [`super::guest_comms`] (guest-only senders) and
//! [`super::host_comms`] (host-only consumers) reference this module.
//! Splitting the wire format out of the transport modules keeps the
//! frame layout authoritative — a producer change here lands in both
//! the guest writer and the host parser without a hand-sync step.
//!
//! # Frame layout
//!
//! Each guest→host bulk message is a 16-byte [`ShmMessage`] header
//! followed by `length` payload bytes. The host's
//! [`super::shm_ring::parse_tlv_stream`] consumes this format. CRC32
//! covers payload bytes only, not the header.
//!
//! ```text
//! offset  size  field
//! ------  ----  ----------------------------------------------
//!   0      4    msg_type (u32 LE)  — see [`MsgType`]
//!   4      4    length   (u32 LE)  — payload bytes following
//!   8      4    crc32    (u32 LE)  — crc32fast over payload
//!  12      4    _pad     (u32 LE)  — reserved, MUST be zero
//!  16      N    payload  (N=length bytes)
//! ```
//!
//! # Control protocol
//!
//! [`VirtioConsoleControl`] mirrors the kernel uapi `struct
//! virtio_console_control` for multiport handshake messages on the
//! c_ivq / c_ovq queues (8 bytes: id u32, event u16, value u16).
//! [`ControlEvent`] enumerates the event discriminants the kernel and
//! the host VMM exchange during port enumeration.
//!
//! Many of the typed wrappers and constants in this module are part
//! of the public bulk API surface; the lib build does not yet read
//! every variant from internal call sites (the typed `MsgType` enum,
//! `ControlEvent`, `VirtioConsoleControl`, `NUM_PORTS`, `PORT1_NAME`,
//! and the `from_wire` reverse mappings are reachable via the public
//! crate path for downstream test code and wire-format tests). The
//! module-level `#[allow(dead_code)]` matches the `VmResult` field
//! pattern in `result.rs` — public surface that the in-tree readers
//! do not exercise without the unused-X lint firing.

#![allow(dead_code)]

use zerocopy::{FromBytes, IntoBytes};

// ---------------------------------------------------------------------------
// MsgType — typed message-type discriminant
// ---------------------------------------------------------------------------

/// Message-type discriminant for the bulk TLV stream.
///
/// Each variant maps to a 32-bit on-wire value via [`Self::wire_value`].
/// The values are 4-character ASCII tags so a hex dump of a captured
/// frame reads as the tag it represents (e.g. `0x4558_4954` = `"EXIT"`).
///
/// On-wire values are stable across host/guest builds — adding a new
/// variant requires picking a fresh ASCII tag and updating
/// [`Self::from_wire`] to recognise it. Existing tags must never be
/// repurposed.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum MsgType {
    /// Stimulus event from the guest step executor.
    Stimulus,
    /// Scenario start marker.
    ScenarioStart,
    /// Scenario end marker (payload: 8-byte LE u64 elapsed_ms).
    ScenarioEnd,
    /// Guest exit code (payload: 4-byte LE i32).
    Exit,
    /// Test result (payload: JSON-encoded `AssertResult`).
    TestResult,
    /// Scheduler process exit (payload: 4-byte LE i32 exit code).
    SchedExit,
    /// Guest crash diagnostic (payload: UTF-8 panic + backtrace).
    /// Only `Crash` may travel via the COM2 fallback transport; the
    /// panic hook cannot block on virtio backpressure.
    Crash,
    /// Per-payload-invocation metrics (payload: JSON-encoded
    /// `PayloadMetrics`).
    PayloadMetrics,
    /// Raw stdout/stderr captured from an LlmExtract payload (payload:
    /// JSON-encoded `RawPayloadOutput`).
    RawPayloadOutput,
    /// Coverage profraw blob.
    Profraw,
    /// Guest→host on-demand snapshot request (payload:
    /// [`SnapshotRequestPayload`]). The freeze coordinator's bulk-drain
    /// path intercepts this frame, runs the CAPTURE / WATCH dispatch,
    /// and replies with [`MsgType::SnapshotReply`] on port 1 RX.
    SnapshotRequest,
    /// Host→guest snapshot reply (payload: [`SnapshotReplyPayload`]).
    /// Sent on port 1 RX so the guest's blocking read on
    /// `/dev/vport0p1` wakes within microseconds. Reply payload
    /// carries the matching request_id, the status, and a UTF-8
    /// reason buffer for the failure path.
    SnapshotReply,
}

impl MsgType {
    /// 32-bit on-wire discriminant for this message type. The value is
    /// the big-endian ASCII representation of a 4-character tag.
    pub const fn wire_value(self) -> u32 {
        match self {
            MsgType::Stimulus => MSG_TYPE_STIMULUS,
            MsgType::ScenarioStart => MSG_TYPE_SCENARIO_START,
            MsgType::ScenarioEnd => MSG_TYPE_SCENARIO_END,
            MsgType::Exit => MSG_TYPE_EXIT,
            MsgType::TestResult => MSG_TYPE_TEST_RESULT,
            MsgType::SchedExit => MSG_TYPE_SCHED_EXIT,
            MsgType::Crash => MSG_TYPE_CRASH,
            MsgType::PayloadMetrics => MSG_TYPE_PAYLOAD_METRICS,
            MsgType::RawPayloadOutput => MSG_TYPE_RAW_PAYLOAD_OUTPUT,
            MsgType::Profraw => MSG_TYPE_PROFRAW,
            MsgType::SnapshotRequest => MSG_TYPE_SNAPSHOT_REQUEST,
            MsgType::SnapshotReply => MSG_TYPE_SNAPSHOT_REPLY,
        }
    }

    /// Reverse the wire mapping. Returns `None` when `value` is not a
    /// recognised discriminant — callers can either skip the frame or
    /// surface the unknown tag for diagnostics.
    pub const fn from_wire(value: u32) -> Option<Self> {
        match value {
            MSG_TYPE_STIMULUS => Some(MsgType::Stimulus),
            MSG_TYPE_SCENARIO_START => Some(MsgType::ScenarioStart),
            MSG_TYPE_SCENARIO_END => Some(MsgType::ScenarioEnd),
            MSG_TYPE_EXIT => Some(MsgType::Exit),
            MSG_TYPE_TEST_RESULT => Some(MsgType::TestResult),
            MSG_TYPE_SCHED_EXIT => Some(MsgType::SchedExit),
            MSG_TYPE_CRASH => Some(MsgType::Crash),
            MSG_TYPE_PAYLOAD_METRICS => Some(MsgType::PayloadMetrics),
            MSG_TYPE_RAW_PAYLOAD_OUTPUT => Some(MsgType::RawPayloadOutput),
            MSG_TYPE_PROFRAW => Some(MsgType::Profraw),
            MSG_TYPE_SNAPSHOT_REQUEST => Some(MsgType::SnapshotRequest),
            MSG_TYPE_SNAPSHOT_REPLY => Some(MsgType::SnapshotReply),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// On-wire u32 discriminants
// ---------------------------------------------------------------------------
//
// Kept as `pub const` for callers that compare a parsed frame's
// `msg_type` field directly (e.g. the freeze coordinator's stream
// filter). [`MsgType::wire_value`] is the typed entry point; the
// constants are the same values exposed for raw-byte comparisons.

/// Stimulus event from the guest step executor.
pub const MSG_TYPE_STIMULUS: u32 = 0x5354_494D; // "STIM"

/// Scenario start marker.
pub const MSG_TYPE_SCENARIO_START: u32 = 0x5343_5354; // "SCST"

/// Scenario end marker.
pub const MSG_TYPE_SCENARIO_END: u32 = 0x5343_454E; // "SCEN"

/// Guest exit code (payload: 4-byte i32).
pub const MSG_TYPE_EXIT: u32 = 0x4558_4954; // "EXIT"

/// Test result (payload: JSON-encoded AssertResult).
pub const MSG_TYPE_TEST_RESULT: u32 = 0x5445_5354; // "TEST"

/// Scheduler process exit (payload: 4-byte i32 exit code).
pub const MSG_TYPE_SCHED_EXIT: u32 = 0x5343_4458; // "SCDX"

/// Guest crash diagnostic (payload: UTF-8 panic + backtrace).
pub const MSG_TYPE_CRASH: u32 = 0x4352_5348; // "CRSH"

/// Per-payload-invocation metrics
/// (payload: JSON-encoded `crate::test_support::PayloadMetrics`).
pub const MSG_TYPE_PAYLOAD_METRICS: u32 = 0x504d_4554; // "PMET"

/// Raw stdout/stderr captured from an LlmExtract payload
/// (payload: JSON-encoded `crate::test_support::RawPayloadOutput`).
pub const MSG_TYPE_RAW_PAYLOAD_OUTPUT: u32 = 0x5241_574f; // "RAWO"

/// Coverage profraw blob (payload: raw `.profraw` bytes from
/// `__llvm_profile_get_data`).
pub const MSG_TYPE_PROFRAW: u32 = 0x5052_4157; // "PRAW"

/// Guest→host on-demand snapshot request
/// (payload: [`SnapshotRequestPayload`]).
pub const MSG_TYPE_SNAPSHOT_REQUEST: u32 = 0x534e_5251; // "SNRQ"

/// Host→guest on-demand snapshot reply
/// (payload: [`SnapshotReplyPayload`]).
pub const MSG_TYPE_SNAPSHOT_REPLY: u32 = 0x534e_5250; // "SNRP"

// ---------------------------------------------------------------------------
// ShmMessage — TLV header
// ---------------------------------------------------------------------------

/// 16-byte TLV header preceding each payload on the wire.
///
/// Both the legacy SHM ring and the bulk virtio-console port-1 channel
/// use this exact layout. CRC32 covers payload bytes only (not the
/// header).
///
/// SAFETY: `repr(C)` with four `u32` fields produces a 16-byte struct
/// with no padding (every field is 4-aligned). `_pad` is reserved for
/// future schema use; current writers MUST set it to 0 and current
/// readers ignore it. zerocopy derives produce no panics — every bit
/// pattern is valid for `u32`.
#[repr(C)]
#[derive(
    Clone, Copy, Default, Debug, FromBytes, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout,
)]
pub struct ShmMessage {
    pub msg_type: u32,
    pub length: u32,
    pub crc32: u32,
    pub _pad: u32,
}

const _SHM_MESSAGE_SIZE: () = assert!(std::mem::size_of::<ShmMessage>() == 16);

/// Size in bytes of the on-wire [`ShmMessage`] header.
pub const FRAME_HEADER_SIZE: usize = std::mem::size_of::<ShmMessage>();

// ---------------------------------------------------------------------------
// ShmEntry — parsed TLV entry
// ---------------------------------------------------------------------------

/// A single parsed message extracted from the bulk byte stream.
///
/// `crc_ok` is `true` when the recomputed payload CRC matched the
/// guest's stored value. CRC mismatches do not stop the walk — the
/// parser yields the entry with `crc_ok=false` and continues with the
/// next frame. Downstream consumers may filter on `crc_ok` to drop
/// corrupted entries.
#[derive(Debug, Clone)]
pub struct ShmEntry {
    pub msg_type: u32,
    pub payload: Vec<u8>,
    /// `true` when the recomputed payload CRC matched the on-wire CRC.
    pub crc_ok: bool,
}

// ---------------------------------------------------------------------------
// Snapshot request/reply TLV payloads
// ---------------------------------------------------------------------------

/// Maximum length, in bytes, of a snapshot tag (capture name or
/// watchpoint symbol path) carried inside the
/// [`SnapshotRequestPayload`]. Tags longer than this bound are
/// truncated by the guest before publishing; the host treats the
/// first NUL as the boundary, or stops at this size if no NUL is
/// present.
pub const SNAPSHOT_TAG_MAX: usize = 64;

/// Maximum length, in bytes, of a host-supplied reason string carried
/// inside the [`SnapshotReplyPayload`]. Same semantics as the tag
/// buffer (NUL-terminated when shorter, truncated when longer).
pub const SNAPSHOT_REASON_MAX: usize = 64;

/// Snapshot request kind: no request pending. Used as the sentinel
/// value for an uninitialised request slot (this discriminant must
/// not appear on the wire — the framing of a TLV with
/// `MSG_TYPE_SNAPSHOT_REQUEST` already implies a request).
pub const SNAPSHOT_KIND_NONE: u32 = 0;

/// Snapshot request kind: capture-now. The host runs
/// `freeze_and_capture(false)` and stores the resulting
/// `FailureDumpReport` on the bridge keyed by the request tag.
pub const SNAPSHOT_KIND_CAPTURE: u32 = 1;

/// Snapshot request kind: hardware-watchpoint registration. The host
/// resolves the symbol path through the vmlinux ELF symtab,
/// allocates a free user watchpoint slot, programs the hardware
/// watchpoint via `KVM_SET_GUEST_DEBUG`, and replies. A future
/// guest write to the resolved KVA fires the corresponding debug
/// exit and synthesises a snapshot tagged by the symbol.
pub const SNAPSHOT_KIND_WATCH: u32 = 2;

/// Reply status: success — the host completed the requested action
/// (capture stored, or watchpoint armed).
pub const SNAPSHOT_STATUS_OK: u32 = 1;

/// Reply status: failure — the host rejected or could not complete
/// the request. The reason buffer carries a UTF-8 diagnostic.
pub const SNAPSHOT_STATUS_ERR: u32 = 2;

/// Snapshot request payload (72 bytes).
///
/// Sent guest→host as the payload of a [`MsgType::SnapshotRequest`]
/// frame on virtio-console port 1 TX. The guest fills every field
/// before publishing; the trailing zeros in `tag` form the NUL
/// terminator when the supplied tag is shorter than
/// [`SNAPSHOT_TAG_MAX`].
///
/// SAFETY: `repr(C)` with `u32 + u32 + [u8; 64]` produces a 72-byte
/// struct with no padding (every field is naturally aligned;
/// trailing array of `u8` requires no end-of-struct padding).
/// Every bit pattern is valid for `u32` and `u8`. zerocopy derives
/// produce no panics.
#[repr(C)]
#[derive(Copy, Clone, Debug, FromBytes, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
pub struct SnapshotRequestPayload {
    /// Monotonic request id the guest stamped before publishing.
    /// The host echoes this value into the matching
    /// [`SnapshotReplyPayload::request_id`] so the guest's blocking
    /// reader can pair against the original request.
    pub request_id: u32,
    /// Request kind: one of [`SNAPSHOT_KIND_CAPTURE`] /
    /// [`SNAPSHOT_KIND_WATCH`]. [`SNAPSHOT_KIND_NONE`] is invalid on
    /// the wire — the host rejects it with [`SNAPSHOT_STATUS_ERR`].
    pub kind: u32,
    /// Tag — UTF-8, NUL-terminated when shorter than the buffer;
    /// truncated to [`SNAPSHOT_TAG_MAX`] when longer. For
    /// [`SNAPSHOT_KIND_CAPTURE`] the tag is the snapshot name (key
    /// the bridge stores the report under); for
    /// [`SNAPSHOT_KIND_WATCH`] the tag is the symbol path the host
    /// resolves through vmlinux ELF.
    pub tag: [u8; SNAPSHOT_TAG_MAX],
}

const _SNAPSHOT_REQUEST_PAYLOAD_SIZE: () =
    assert!(std::mem::size_of::<SnapshotRequestPayload>() == 8 + SNAPSHOT_TAG_MAX);

/// Snapshot reply payload (72 bytes).
///
/// Sent host→guest as the payload of a [`MsgType::SnapshotReply`]
/// frame on virtio-console port 1 RX. Mirrors the request layout —
/// the guest matches `request_id` against its outstanding request
/// and reads `status`/`reason` to surface the host's verdict.
///
/// SAFETY: identical layout reasoning as [`SnapshotRequestPayload`].
#[repr(C)]
#[derive(Copy, Clone, Debug, FromBytes, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
pub struct SnapshotReplyPayload {
    /// Echo of the request's `request_id`. The guest's blocking
    /// reader spins until it observes this value match its
    /// outstanding request.
    pub request_id: u32,
    /// Reply status: [`SNAPSHOT_STATUS_OK`] when the host completed
    /// the request, [`SNAPSHOT_STATUS_ERR`] otherwise.
    pub status: u32,
    /// Reason — UTF-8, NUL-terminated when shorter than the buffer;
    /// truncated to [`SNAPSHOT_REASON_MAX`] when longer. Empty
    /// (all-zero) on the success path.
    pub reason: [u8; SNAPSHOT_REASON_MAX],
}

const _SNAPSHOT_REPLY_PAYLOAD_SIZE: () =
    assert!(std::mem::size_of::<SnapshotReplyPayload>() == 8 + SNAPSHOT_REASON_MAX);

// ---------------------------------------------------------------------------
// ControlEvent — multiport control protocol discriminants
// ---------------------------------------------------------------------------

/// Multiport control-event discriminant. Mirrors the kernel uapi
/// `enum virtio_console_event` in `include/uapi/linux/virtio_console.h`.
///
/// The on-wire value is a u16. [`Self::wire_value`] returns the value
/// the kernel and the host VMM exchange on the c_ivq / c_ovq queues;
/// [`Self::from_wire`] reverses the mapping for a host-side parser.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum ControlEvent {
    /// Guest-side: driver finished probing, host may begin port
    /// enumeration.
    DeviceReady,
    /// Host-side: announce a new port to the guest.
    PortAdd,
    /// Host-side: tear down a port.
    PortRemove,
    /// Guest-side: per-port driver finished setup.
    PortReady,
    /// Host-side: mark a port as the system console.
    ConsolePort,
    /// Host-side: terminal resize event.
    Resize,
    /// Bidirectional: open/close indication for a port.
    PortOpen,
    /// Host-side: PORT_NAME header followed by name bytes.
    PortName,
}

impl ControlEvent {
    /// 16-bit on-wire discriminant. Values match the kernel uapi
    /// constants `VIRTIO_CONSOLE_*`.
    pub const fn wire_value(self) -> u16 {
        match self {
            ControlEvent::DeviceReady => 0,
            ControlEvent::PortAdd => 1,
            ControlEvent::PortRemove => 2,
            ControlEvent::PortReady => 3,
            ControlEvent::ConsolePort => 4,
            ControlEvent::Resize => 5,
            ControlEvent::PortOpen => 6,
            ControlEvent::PortName => 7,
        }
    }

    /// Reverse the wire mapping. Returns `None` for unknown
    /// discriminants — the host parser is expected to log + skip such
    /// frames rather than panic.
    pub const fn from_wire(value: u16) -> Option<Self> {
        match value {
            0 => Some(ControlEvent::DeviceReady),
            1 => Some(ControlEvent::PortAdd),
            2 => Some(ControlEvent::PortRemove),
            3 => Some(ControlEvent::PortReady),
            4 => Some(ControlEvent::ConsolePort),
            5 => Some(ControlEvent::Resize),
            6 => Some(ControlEvent::PortOpen),
            7 => Some(ControlEvent::PortName),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// VirtioConsoleControl — wire-format control message
// ---------------------------------------------------------------------------

/// Wire-format control message exchanged on c_ivq / c_ovq.
///
/// Mirrors `struct virtio_console_control` in
/// `include/uapi/linux/virtio_console.h`: id (u32), event (u16),
/// value (u16). The kernel's wire format is little-endian; on the LE
/// hosts ktstr targets (x86_64, aarch64), `repr(C)` produces the
/// correct byte order via zerocopy `IntoBytes` / `FromBytes`.
///
/// SAFETY: `repr(C)` produces an 8-byte struct with no padding when
/// every field is naturally aligned (u32 at offset 0, u16 at offset
/// 4, u16 at offset 6). The `packed` qualifier is unnecessary because
/// the natural alignment matches the kernel's expected wire layout
/// and is checked by [`std::mem::size_of`] below. Every bit pattern
/// is valid for u32/u16. zerocopy derives produce no panics.
#[repr(C)]
#[derive(Copy, Clone, Debug, FromBytes, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
pub struct VirtioConsoleControl {
    pub id: u32,
    pub event: u16,
    pub value: u16,
}

const _VIRTIO_CONSOLE_CONTROL_SIZE: () = assert!(std::mem::size_of::<VirtioConsoleControl>() == 8);

// ---------------------------------------------------------------------------
// Multiport device constants
// ---------------------------------------------------------------------------

/// Number of multiport ports the device exposes.
///
/// Port 0 is the kernel console (`/dev/hvc0`); port 1 is the
/// host-bound bulk TLV stream (`/dev/vport0p1`). Two ports → six
/// queues per virtio-v1.2 §5.3.5 (`2 + 2 * num_ports`).
pub const NUM_PORTS: u32 = 2;

/// Port-1 device-name advertised to the guest. The kernel exposes
/// this as `/sys/class/virtio-ports/vport0p1/name`; the guest init
/// reads from this path to discover the bulk channel device node.
pub const PORT1_NAME: &str = "ktstr-bulk";

#[cfg(test)]
mod tests {
    use super::*;

    /// `ShmMessage` round-trips through bytes — guards against an
    /// accidental field reorder or a stray padding byte that would
    /// shift the on-wire layout for both guest writer and host
    /// reader.
    #[test]
    fn shm_message_round_trip_through_bytes() {
        let f = ShmMessage {
            msg_type: MSG_TYPE_EXIT,
            length: 4,
            crc32: 0xDEAD_BEEF,
            _pad: 0,
        };
        let bytes = f.as_bytes();
        assert_eq!(bytes.len(), FRAME_HEADER_SIZE);
        let back = ShmMessage::read_from_bytes(bytes).expect("16-byte slice deserializes");
        let msg_type = back.msg_type;
        let length = back.length;
        let crc32 = back.crc32;
        let pad = back._pad;
        assert_eq!(msg_type, MSG_TYPE_EXIT);
        assert_eq!(length, 4);
        assert_eq!(crc32, 0xDEAD_BEEF);
        assert_eq!(pad, 0);
    }

    /// Every msg_type constant is distinct — a copy/paste error
    /// that aliased two ids would silently misroute messages.
    #[test]
    fn msg_type_constants_are_unique() {
        let ids = [
            MSG_TYPE_STIMULUS,
            MSG_TYPE_SCENARIO_START,
            MSG_TYPE_SCENARIO_END,
            MSG_TYPE_EXIT,
            MSG_TYPE_TEST_RESULT,
            MSG_TYPE_SCHED_EXIT,
            MSG_TYPE_CRASH,
            MSG_TYPE_PAYLOAD_METRICS,
            MSG_TYPE_RAW_PAYLOAD_OUTPUT,
            MSG_TYPE_PROFRAW,
            MSG_TYPE_SNAPSHOT_REQUEST,
            MSG_TYPE_SNAPSHOT_REPLY,
        ];
        for (i, a) in ids.iter().enumerate() {
            for b in &ids[i + 1..] {
                assert_ne!(a, b, "duplicate MSG_TYPE id 0x{a:08x}");
            }
        }
    }

    /// `ShmMessage` header is exactly 16 bytes with no padding.
    #[test]
    fn shm_message_size_is_16() {
        assert_eq!(FRAME_HEADER_SIZE, 16);
        assert_eq!(std::mem::size_of::<ShmMessage>(), 16);
    }

    /// Every [`MsgType`] variant round-trips through
    /// `wire_value` → `from_wire`.
    #[test]
    fn msg_type_round_trips() {
        let all = [
            MsgType::Stimulus,
            MsgType::ScenarioStart,
            MsgType::ScenarioEnd,
            MsgType::Exit,
            MsgType::TestResult,
            MsgType::SchedExit,
            MsgType::Crash,
            MsgType::PayloadMetrics,
            MsgType::RawPayloadOutput,
            MsgType::Profraw,
            MsgType::SnapshotRequest,
            MsgType::SnapshotReply,
        ];
        for variant in all {
            let v = variant.wire_value();
            assert_eq!(MsgType::from_wire(v), Some(variant));
        }
    }

    /// `MsgType::from_wire` returns `None` for an unrecognised
    /// discriminant — the bulk parser must surface unknown tags as
    /// errors rather than treat them as a known variant.
    #[test]
    fn msg_type_from_wire_unknown_returns_none() {
        assert_eq!(MsgType::from_wire(0xDEAD_BEEF), None);
        assert_eq!(MsgType::from_wire(0), None);
    }

    /// `MsgType::wire_value` matches the corresponding `MSG_TYPE_*`
    /// constant — guards against a typo that would diverge the typed
    /// API from the on-wire constant.
    #[test]
    fn msg_type_wire_value_matches_constants() {
        assert_eq!(MsgType::Stimulus.wire_value(), MSG_TYPE_STIMULUS);
        assert_eq!(MsgType::ScenarioStart.wire_value(), MSG_TYPE_SCENARIO_START);
        assert_eq!(MsgType::ScenarioEnd.wire_value(), MSG_TYPE_SCENARIO_END);
        assert_eq!(MsgType::Exit.wire_value(), MSG_TYPE_EXIT);
        assert_eq!(MsgType::TestResult.wire_value(), MSG_TYPE_TEST_RESULT);
        assert_eq!(MsgType::SchedExit.wire_value(), MSG_TYPE_SCHED_EXIT);
        assert_eq!(MsgType::Crash.wire_value(), MSG_TYPE_CRASH);
        assert_eq!(
            MsgType::PayloadMetrics.wire_value(),
            MSG_TYPE_PAYLOAD_METRICS
        );
        assert_eq!(
            MsgType::RawPayloadOutput.wire_value(),
            MSG_TYPE_RAW_PAYLOAD_OUTPUT
        );
        assert_eq!(MsgType::Profraw.wire_value(), MSG_TYPE_PROFRAW);
        assert_eq!(
            MsgType::SnapshotRequest.wire_value(),
            MSG_TYPE_SNAPSHOT_REQUEST
        );
        assert_eq!(
            MsgType::SnapshotReply.wire_value(),
            MSG_TYPE_SNAPSHOT_REPLY
        );
    }

    /// `SnapshotRequestPayload` round-trips through bytes — guards
    /// against an accidental field reorder or a stray padding byte
    /// that would shift the on-wire layout for both guest writer
    /// and host parser.
    #[test]
    fn snapshot_request_payload_round_trip_through_bytes() {
        let mut tag = [0u8; SNAPSHOT_TAG_MAX];
        tag[..6].copy_from_slice(b"hello!");
        let p = SnapshotRequestPayload {
            request_id: 0xDEAD_BEEF,
            kind: SNAPSHOT_KIND_CAPTURE,
            tag,
        };
        let bytes = p.as_bytes();
        assert_eq!(bytes.len(), 8 + SNAPSHOT_TAG_MAX);
        let back =
            SnapshotRequestPayload::read_from_bytes(bytes).expect("payload deserializes");
        let request_id = back.request_id;
        let kind = back.kind;
        assert_eq!(request_id, 0xDEAD_BEEF);
        assert_eq!(kind, SNAPSHOT_KIND_CAPTURE);
        assert_eq!(&back.tag[..6], b"hello!");
    }

    /// `SnapshotReplyPayload` round-trips through bytes.
    #[test]
    fn snapshot_reply_payload_round_trip_through_bytes() {
        let mut reason = [0u8; SNAPSHOT_REASON_MAX];
        reason[..4].copy_from_slice(b"oops");
        let p = SnapshotReplyPayload {
            request_id: 0xCAFE_BABE,
            status: SNAPSHOT_STATUS_ERR,
            reason,
        };
        let bytes = p.as_bytes();
        assert_eq!(bytes.len(), 8 + SNAPSHOT_REASON_MAX);
        let back =
            SnapshotReplyPayload::read_from_bytes(bytes).expect("payload deserializes");
        let request_id = back.request_id;
        let status = back.status;
        assert_eq!(request_id, 0xCAFE_BABE);
        assert_eq!(status, SNAPSHOT_STATUS_ERR);
        assert_eq!(&back.reason[..4], b"oops");
    }

    /// Snapshot kind constants are distinct.
    #[test]
    fn snapshot_kind_constants_are_unique() {
        assert_ne!(SNAPSHOT_KIND_NONE, SNAPSHOT_KIND_CAPTURE);
        assert_ne!(SNAPSHOT_KIND_NONE, SNAPSHOT_KIND_WATCH);
        assert_ne!(SNAPSHOT_KIND_CAPTURE, SNAPSHOT_KIND_WATCH);
    }

    /// Snapshot status constants are distinct.
    #[test]
    fn snapshot_status_constants_are_unique() {
        assert_ne!(SNAPSHOT_STATUS_OK, SNAPSHOT_STATUS_ERR);
    }

    /// Every [`ControlEvent`] variant round-trips through
    /// `wire_value` → `from_wire`.
    #[test]
    fn control_event_round_trips() {
        let all = [
            ControlEvent::DeviceReady,
            ControlEvent::PortAdd,
            ControlEvent::PortRemove,
            ControlEvent::PortReady,
            ControlEvent::ConsolePort,
            ControlEvent::Resize,
            ControlEvent::PortOpen,
            ControlEvent::PortName,
        ];
        for variant in all {
            let v = variant.wire_value();
            assert_eq!(ControlEvent::from_wire(v), Some(variant));
        }
    }

    /// `ControlEvent::from_wire` returns `None` for unknown values.
    #[test]
    fn control_event_from_wire_unknown_returns_none() {
        assert_eq!(ControlEvent::from_wire(8), None);
        assert_eq!(ControlEvent::from_wire(0xFFFF), None);
    }

    /// `ControlEvent` discriminants match the kernel uapi numbers
    /// (`VIRTIO_CONSOLE_*` in `include/uapi/linux/virtio_console.h`).
    #[test]
    fn control_event_discriminants_match_uapi() {
        assert_eq!(ControlEvent::DeviceReady.wire_value(), 0);
        assert_eq!(ControlEvent::PortAdd.wire_value(), 1);
        assert_eq!(ControlEvent::PortRemove.wire_value(), 2);
        assert_eq!(ControlEvent::PortReady.wire_value(), 3);
        assert_eq!(ControlEvent::ConsolePort.wire_value(), 4);
        assert_eq!(ControlEvent::Resize.wire_value(), 5);
        assert_eq!(ControlEvent::PortOpen.wire_value(), 6);
        assert_eq!(ControlEvent::PortName.wire_value(), 7);
    }

    /// `VirtioConsoleControl` is exactly 8 bytes — matches the
    /// kernel uapi struct.
    #[test]
    fn virtio_console_control_size_is_8() {
        assert_eq!(std::mem::size_of::<VirtioConsoleControl>(), 8);
    }

    /// `VirtioConsoleControl` round-trips through bytes — pins the
    /// repr(C) layout against an accidental field reorder that would
    /// produce malformed control frames on the c_ivq / c_ovq queues.
    #[test]
    fn virtio_console_control_round_trip() {
        let c = VirtioConsoleControl {
            id: 1,
            event: ControlEvent::PortOpen.wire_value(),
            value: 1,
        };
        let bytes = c.as_bytes();
        assert_eq!(bytes.len(), 8);
        let back = VirtioConsoleControl::read_from_bytes(bytes).unwrap();
        let id = back.id;
        let event = back.event;
        let value = back.value;
        assert_eq!(id, 1);
        assert_eq!(event, ControlEvent::PortOpen.wire_value());
        assert_eq!(value, 1);
    }
}
