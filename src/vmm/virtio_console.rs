//! Three-port virtio-console with inline MMIO transport.
//!
//! Eight virtqueues per virtio-v1.2 §5.3.5 with `VIRTIO_CONSOLE_F_MULTIPORT`:
//!   q0 in0  — host→guest, port 0 (console / hvc0 stdin)
//!   q1 out0 — guest→host, port 0 (console / hvc0 stdout)
//!   q2 c_ivq — host→guest control (PORT_ADD, PORT_OPEN, etc.)
//!   q3 c_ovq — guest→host control (DEVICE_READY, PORT_READY, PORT_OPEN ack)
//!   q4 in1  — host→guest, port 1 (snapshot reply payloads from the
//!            freeze coordinator; see `queue_input_port1`)
//!   q5 out1 — guest→host, port 1 (bulk TLV stream)
//!   q6 in2  — host→guest, port 2 (scheduler-stats requests from the
//!            host's [`super::sched_stats::SchedStatsClient`]; see
//!            `queue_input_port2`)
//!   q7 out2 — guest→host, port 2 (scheduler-stats responses; raw
//!            byte passthrough, no TLV)
//!
//! Port 0 carries the interactive console (stdout/stdin via `/dev/hvc0`).
//! Port 1 carries the TLV stream written by
//! `guest_comms::send_*` — exit code, test result, per-payload
//! metrics, raw payload outputs, profraw, scheduler exit
//! notifications, stimulus events, scenario start/end markers.
//! Port 2 is a transparent byte pipe: the host pushes scx_stats
//! request bytes; the guest's relay thread forwards them to
//! `/var/run/scx/root/stats` and pumps the socket response back via
//! port 2 TX. scx_stats is already newline-delimited JSON so no
//! framing layer is added.
//! Crash payloads travel over COM2. Backpressure is asymmetric:
//!   * Guest→host TX (port 1, port 2): the host's `add_used` rate
//!     gates the guest's writes; when the host lags, the guest
//!     blocks in `wait_port_writable` instead of dropping. Per-call
//!     drain is also capped (`TX_PER_CALL_MAX`) so a hostile guest
//!     cannot grow the host accumulator without bound on a single
//!     notify.
//!   * Host→guest RX (port 0 + port 1 + port 2): the per-port
//!     `pending_rx` accumulators are unbounded by design — the host
//!     alone produces these bytes (kernel scheduler signals, terminal
//!     paste, snapshot replies, stats requests), so a hostile guest
//!     cannot grow them; losing a host→guest byte would silently
//!     strand a wake signal or truncate a reply, which is worse than
//!     a host-side OOM. The per-call CHAIN drain is capped
//!     (`RX_CHAINS_PER_CALL_MAX`) so a hostile guest publishing many
//!     zero-progress descriptor chains cannot hold the vCPU MMIO
//!     handler in `drain_pending_rx` for an unbounded number of
//!     iterations on a single notify.
//!
//! Features: `VIRTIO_F_VERSION_1 | VIRTIO_CONSOLE_F_MULTIPORT`.
//! Config space: `cols=0, rows=0, max_nr_ports=3, emerg_wr=0` (cols/rows
//! valid only with F_SIZE which we do not advertise; the kernel reads
//! `max_nr_ports` via `virtio_cread_feature(F_MULTIPORT, max_nr_ports)`,
//! offset 4 in `struct virtio_console_config`).
//!
//! MMIO register layout per virtio-v1.2 §4.2.2. Interrupt delivery via
//! irqfd (eventfd → KVM GSI). TX data on port 0 or port 1 signals
//! `tx_evt`; TX data on port 2 signals a separate `stats_tx_evt` so
//! the host's [`super::sched_stats::SchedStatsClient`] wakes only on
//! its own port without contending with the freeze coordinator's
//! port-1 drain path.

use std::collections::VecDeque;

use virtio_bindings::virtio_config::{
    VIRTIO_CONFIG_S_ACKNOWLEDGE, VIRTIO_CONFIG_S_DRIVER, VIRTIO_CONFIG_S_DRIVER_OK,
    VIRTIO_CONFIG_S_FAILED, VIRTIO_CONFIG_S_FEATURES_OK, VIRTIO_F_VERSION_1,
};
use virtio_bindings::virtio_ids::VIRTIO_ID_CONSOLE;

/// Multiport feature bit per `include/uapi/linux/virtio_console.h`.
/// virtio-bindings 0.2.7 does not expose virtio_console.h constants
/// (the crate's per-arch `bindings/` tree only carries blk/config/
/// gpu/ids/input/mmio/net/ring/scsi), so the spec-defined value lives
/// here as a single source of truth.
const VIRTIO_CONSOLE_F_MULTIPORT: u32 = 1;
use virtio_bindings::virtio_mmio::{
    VIRTIO_MMIO_CONFIG_GENERATION, VIRTIO_MMIO_DEVICE_FEATURES, VIRTIO_MMIO_DEVICE_FEATURES_SEL,
    VIRTIO_MMIO_DEVICE_ID, VIRTIO_MMIO_DRIVER_FEATURES, VIRTIO_MMIO_DRIVER_FEATURES_SEL,
    VIRTIO_MMIO_INT_VRING, VIRTIO_MMIO_INTERRUPT_ACK, VIRTIO_MMIO_INTERRUPT_STATUS,
    VIRTIO_MMIO_MAGIC_VALUE, VIRTIO_MMIO_QUEUE_AVAIL_HIGH, VIRTIO_MMIO_QUEUE_AVAIL_LOW,
    VIRTIO_MMIO_QUEUE_DESC_HIGH, VIRTIO_MMIO_QUEUE_DESC_LOW, VIRTIO_MMIO_QUEUE_NOTIFY,
    VIRTIO_MMIO_QUEUE_NUM, VIRTIO_MMIO_QUEUE_NUM_MAX, VIRTIO_MMIO_QUEUE_READY,
    VIRTIO_MMIO_QUEUE_SEL, VIRTIO_MMIO_QUEUE_USED_HIGH, VIRTIO_MMIO_QUEUE_USED_LOW,
    VIRTIO_MMIO_STATUS, VIRTIO_MMIO_VENDOR_ID, VIRTIO_MMIO_VERSION,
};
use virtio_queue::{Queue, QueueT};
use vm_memory::{Bytes, GuestMemoryMmap};
use vmm_sys_util::eventfd::EventFd;
use zerocopy::{FromBytes, IntoBytes};

const MMIO_MAGIC: u32 = 0x7472_6976; // "virt" in LE
const MMIO_VERSION: u32 = 2; // virtio 1.x MMIO
const VENDOR_ID: u32 = 0;

/// MMIO region size: 4 KB (one page).
pub const VIRTIO_MMIO_SIZE: u64 = 0x1000;

/// RX wake byte: host requested a SysRq-D dump. The guest's
/// `hvc0_poll_loop` blocks on `/dev/hvc0`, scans every drained byte
/// for this value, and triggers SysRq-D directly via
/// `/proc/sysrq-trigger` when it is observed. Distinct from
/// `SIGNAL_VC_SHUTDOWN` and `SIGNAL_BPF_WRITE_DONE` so stack traces
/// and tcpdump-style captures can distinguish the trigger source.
pub const SIGNAL_VC_DUMP: u8 = 0xD1;

/// RX wake byte: host pushed a graceful-shutdown request through
/// the virtio-console RX queue.
pub const SIGNAL_VC_SHUTDOWN: u8 = 0xD3;

/// RX wake byte: host's `bpf-map-write` thread finished applying
/// every queued `bpf_map_write` to the BPF maps inside the guest's
/// kernel. The guest's `hvc0_poll_loop` recognises the byte and
/// sets the `bpf_map_write_done` latch so a scenario blocked on
/// [`crate::scenario::Ctx::wait_for_map_write`] resumes. Replaces
/// the legacy SHM signal-slot rendezvous (host writes slot 0, guest
/// blocks on slot 0) with a virtio-console wake byte. Host side:
/// `host_comms::request_bpf_map_write_done`.
pub const SIGNAL_BPF_WRITE_DONE: u8 = 0xBF;

// `NUM_PORTS` lives in [`super::wire`]; re-exported here so existing
// call sites keep working. Port 0 = console (hvc0); port 1 = bulk
// TLV stream (`/dev/vport0p1`); port 2 = scheduler-stats relay
// (`/dev/vport0p2`). Three ports → eight queues per virtio-v1.2 §5.3.5
// (`2 + 2 * num_ports`).
pub use super::wire::NUM_PORTS;

const NUM_QUEUES: usize = 2 + 2 * NUM_PORTS as usize;
const QUEUE_MAX_SIZE: u16 = 256;

// Per port_id_to_queue_idx in libkrun (mirrored here):
//   port 0: rx=0, tx=1
//   control: c_ivq=2 (host→guest), c_ovq=3 (guest→host)
//   port N>=1: rx = 2+2N, tx = 2+2N+1
// So port 1: rx=4, tx=5; port 2: rx=6, tx=7.
const PORT0_RXQ: usize = 0;
const PORT0_TXQ: usize = 1;
const C_IVQ: usize = 2; // host pushes control msgs to guest
const C_OVQ: usize = 3; // guest sends control msgs to host
const PORT1_RXQ: usize = 4;
const PORT1_TXQ: usize = 5;
const PORT2_RXQ: usize = 6;
const PORT2_TXQ: usize = 7;

/// Maximum bytes accepted from a single TX descriptor. The kernel's
/// virtio-console driver sends PAGE_SIZE chunks; this cap prevents a
/// malformed descriptor (len=0xFFFFFFFF) from triggering a ~4GB alloc.
const TX_DESC_MAX: usize = 32 * 1024;

/// Maximum cumulative bytes accepted by a single `process_tx`
/// call. The per-descriptor `TX_DESC_MAX` cap bounds individual
/// descriptors, but a hostile guest can publish thousands of valid
/// descriptors back-to-back and grow the per-port `tx_buf` without
/// bound. Capping the per-call drain at 256 KiB keeps the per-vCPU
/// MMIO-handler latency budget bounded — once the cap is hit we
/// stop popping chains and let the next QUEUE_NOTIFY drain the
/// rest. Backpressure on the guest's TX queue is the natural
/// consequence: a chain that has not been add_used yet stays in
/// the avail ring for the next call.
const TX_PER_CALL_MAX: usize = 256 * 1024;

/// Maximum control-queue chains drained per `process_control_tx`
/// call. The c_ovq's payload is a fixed 8-byte
/// `VirtioConsoleControl` frame — a hostile guest publishing
/// thousands of small chains would otherwise let one notify hold the
/// vCPU thread in `process_control_tx` for an unbounded duration and
/// grow the `events` Vec without bound. Mirrors the TX byte-cap
/// pattern: chains beyond the cap stay in the avail ring for the
/// next QUEUE_NOTIFY. 32 is enough headroom for the legitimate
/// handshake (DEVICE_READY + per-port PORT_READY + per-port
/// PORT_OPEN = ~5 events) with margin while still bounding the
/// adversarial case.
const CONTROL_CHAINS_PER_CALL_MAX: usize = 32;

/// Maximum host→guest RX chains drained per `drain_pending_rx`
/// call. Unlike TX (byte-driven via `TX_PER_CALL_MAX`), RX progress
/// is chain-shaped: each chain absorbs
/// `min(pending_rx_len, sum_of_write_only_desc_lens)` bytes. A
/// hostile guest publishing many zero-length write-only descriptors
/// (or chains lacking any write-only desc — unusual but legal)
/// makes `consumed_offset` stay 0; the `drain(..0)` at the bottom
/// of the loop is then a no-op and the outer
/// `while !pending_rx.is_empty()` reissues `pop_descriptor_chain`
/// without progress until the avail ring is exhausted. With
/// `QUEUE_MAX_SIZE = 256` chains × 256 descriptors per chain that's
/// ~65k iterations per notify, parked on a vCPU thread that is
/// expected to bound MMIO-handler latency. Cap drains at 64 chains
/// per call: legitimate traffic posts a small number of multi-KB
/// chains (kernel virtio-console driver allocates PAGE_SIZE buffers
/// per chain for hvc0, larger for `/dev/vport0p1`); 64 is well
/// above any single-notify legitimate fan-out while still bounding
/// the adversarial latency. Remaining chains stay in the avail
/// ring for the next QUEUE_NOTIFY (or the next host-side push).
const RX_CHAINS_PER_CALL_MAX: usize = 64;

/// Status bits required before each phase.
const S_ACK: u32 = VIRTIO_CONFIG_S_ACKNOWLEDGE;
const S_DRV: u32 = S_ACK | VIRTIO_CONFIG_S_DRIVER;
const S_FEAT: u32 = S_DRV | VIRTIO_CONFIG_S_FEATURES_OK;
/// Test helper — terminal state bits with DRIVER_OK set.
#[cfg(test)]
const S_OK: u32 = S_FEAT | VIRTIO_CONFIG_S_DRIVER_OK;

// ----- virtio-console control protocol -----------------------------
//
// `VirtioConsoleControl` and its u16 event discriminants live in
// [`super::wire`]. The constants here are convenience aliases for
// `ControlEvent::*.wire_value()` so the existing call sites read
// the same as the kernel uapi names; new code should prefer the
// typed `ControlEvent` enum directly. The wire format is 8 bytes
// little-endian: id (u32), event (u16), value (u16). LE on the
// wire is x86_64 / aarch64 native.

pub use super::wire::VirtioConsoleControl;

pub const VIRTIO_CONSOLE_DEVICE_READY: u16 = super::wire::ControlEvent::DeviceReady.wire_value();
pub const VIRTIO_CONSOLE_PORT_ADD: u16 = super::wire::ControlEvent::PortAdd.wire_value();
// PORT_REMOVE and RESIZE are kernel uapi event ids the lib does not
// yet generate or consume — kept as named constants so the public
// surface mirrors `enum virtio_console_event` 1:1. `#[allow(dead_code)]`
// matches the `KVM_INTERESTING_STATS` pattern in `result.rs`.
#[allow(dead_code)]
pub const VIRTIO_CONSOLE_PORT_REMOVE: u16 = super::wire::ControlEvent::PortRemove.wire_value();
pub const VIRTIO_CONSOLE_PORT_READY: u16 = super::wire::ControlEvent::PortReady.wire_value();
pub const VIRTIO_CONSOLE_CONSOLE_PORT: u16 = super::wire::ControlEvent::ConsolePort.wire_value();
#[allow(dead_code)]
pub const VIRTIO_CONSOLE_RESIZE: u16 = super::wire::ControlEvent::Resize.wire_value();
pub const VIRTIO_CONSOLE_PORT_OPEN: u16 = super::wire::ControlEvent::PortOpen.wire_value();
pub const VIRTIO_CONSOLE_PORT_NAME: u16 = super::wire::ControlEvent::PortName.wire_value();

const VC_CONTROL_SIZE: usize = std::mem::size_of::<VirtioConsoleControl>();
const _: () = assert!(VC_CONTROL_SIZE == 8);

// `PORT1_NAME` and `PORT2_NAME` live in [`super::wire`]; re-exported
// here for the existing call sites in this module.
pub use super::wire::PORT1_NAME;
pub use super::wire::PORT2_NAME;

/// Port-0 device-name advertised to the guest. The kernel's
/// `handle_control_message` PORT_NAME case
/// (drivers/char/virtio_console.c) creates the sysfs
/// `/sys/class/virtio-ports/vport0p0/name` attribute when the host
/// sends PORT_NAME; without that emission the attribute does not
/// exist and tooling that scans `/sys/class/virtio-ports/*/name` to
/// disambiguate port 0 (console) from port 1 (bulk) cannot
/// distinguish them. QEMU's `add_port` (hw/char/virtio-console.c)
/// sets a name on the chardev (`chardev-id` derived) and the
/// virtio-serial PORT_NAME emission in
/// `virtio_serial_post_load_timer_cb` / `send_control_event`
/// emits it for every port that has one — including the console
/// port. Mirror that here.
pub const PORT0_NAME: &str = "ktstr-console";

/// Outbound (host→guest) control payload kinds. The host serialises
/// these into 8-byte wire frames (plus optional name bytes) for the
/// c_ivq.
#[derive(Debug, Clone)]
enum ControlOut {
    /// Fixed 8-byte command.
    Cmd(VirtioConsoleControl),
    /// 8-byte PORT_NAME header followed by name bytes and a trailing
    /// NUL terminator. QEMU's PORT_NAME emitter
    /// (hw/char/virtio-serial-bus.c, `buffer_len = sizeof(cpkt) +
    /// strlen(port->name) + 1; ... buffer[buffer_len - 1] = 0;`)
    /// includes the NUL; the kernel parser
    /// (drivers/char/virtio_console.c `handle_control_message`
    /// PORT_NAME case) computes `name_size = buf->len - buf->offset
    /// - sizeof(*cpkt) + 1` and `strscpy`s into a kmalloc'd buffer,
    ///   which works either way but expects the QEMU layout. Sending
    ///   the NUL keeps the wire format byte-identical to QEMU so any
    ///   downstream tooling that snoops the frame sees the same shape.
    Name { id: u32, name: &'static str },
}

impl ControlOut {
    fn len(&self) -> usize {
        match self {
            ControlOut::Cmd(_) => VC_CONTROL_SIZE,
            // +1 for the trailing NUL terminator (see Name doc).
            ControlOut::Name { name, .. } => VC_CONTROL_SIZE + name.len() + 1,
        }
    }

    fn write_into(&self, dst: &mut Vec<u8>) {
        match self {
            ControlOut::Cmd(c) => dst.extend_from_slice(c.as_bytes()),
            ControlOut::Name { id, name } => {
                let hdr = VirtioConsoleControl {
                    id: *id,
                    event: VIRTIO_CONSOLE_PORT_NAME,
                    value: 1, // libkrun / qemu both pass value=1 here.
                };
                dst.extend_from_slice(hdr.as_bytes());
                dst.extend_from_slice(name.as_bytes());
                // Trailing NUL — matches QEMU's wire layout.
                dst.push(0);
            }
        }
    }
}

/// Per-port state for the three virtio-console ports. Indexed by
/// `port_id` (0 = console / hvc0, 1 = bulk TLV stream / vport0p1,
/// 2 = scheduler-stats relay / vport0p2).
///
/// `tx_buf` accumulates guest→host TX bytes pending host drain;
/// `pending_rx` accumulates host→guest RX bytes pending delivery
/// into the guest's RX ring. Both are unbounded by design — a host
/// alone produces RX bytes (so a hostile guest cannot grow
/// `pending_rx`), and TX bytes are bounded per-call by
/// `TX_PER_CALL_MAX`.
struct Port {
    /// Accumulated guest→host TX output. `VecDeque` so port 1's
    /// [`VirtioConsole::push_back_bulk`] can prepend the freeze
    /// coordinator's `bulk_assembler` residual via `push_front` in
    /// O(bytes); other ports drain via `Vec::from(VecDeque)` (no
    /// reallocation; at worst an O(N) rotate when the ring is split).
    tx_buf: VecDeque<u8>,
    /// Pending host→guest RX bytes. Drained into write-only
    /// descriptors on the matching RX queue notify (or on a
    /// PORT_OPEN closed→open transition for ports 1 and 2).
    pending_rx: VecDeque<u8>,
    /// Set when the guest sends `PORT_OPEN(value=1)` on c_ovq for
    /// this port. The RX drain gates on this flag for ports 1 and 2 —
    /// pushing bytes before the guest opens the port lets the kernel
    /// discard them with no userspace reader. Port 0 starts implicitly
    /// open (the kernel's hvc-console path does not require a
    /// control-protocol open before TX/RX).
    opened: bool,
    /// Set when the guest sends `PORT_READY(value=1)` on c_ovq for
    /// this port. Gates the host-side CONSOLE_PORT / PORT_OPEN /
    /// PORT_NAME enqueues — repeat PORT_READY messages from a
    /// hostile guest would otherwise grow `control_out` without
    /// bound, exhausting host memory. Each port may be readied
    /// exactly once per device lifecycle; `reset()` clears this
    /// back to false.
    readied: bool,
    /// Device name advertised to the guest via the PORT_NAME
    /// control message. Becomes the sysfs
    /// `/sys/class/virtio-ports/vport0pN/name` attribute that udev
    /// rules consume to symlink the port.
    name: &'static str,
}

impl Port {
    const fn new(name: &'static str) -> Self {
        Port {
            tx_buf: VecDeque::new(),
            pending_rx: VecDeque::new(),
            opened: false,
            readied: false,
            name,
        }
    }
}

/// Map a queue index to (port_id, is_tx). Returns None for the
/// control queues (C_IVQ / C_OVQ) and any out-of-range index.
const fn queue_to_port(queue_idx: usize) -> Option<(usize, bool)> {
    match queue_idx {
        PORT0_RXQ => Some((0, false)),
        PORT0_TXQ => Some((0, true)),
        PORT1_RXQ => Some((1, false)),
        PORT1_TXQ => Some((1, true)),
        PORT2_RXQ => Some((2, false)),
        PORT2_TXQ => Some((2, true)),
        _ => None,
    }
}

/// Map a port id to its (rxq, txq) queue indices. Inverse of
/// `queue_to_port` for the data direction.
const fn port_queues(port_id: usize) -> (usize, usize) {
    match port_id {
        0 => (PORT0_RXQ, PORT0_TXQ),
        1 => (PORT1_RXQ, PORT1_TXQ),
        2 => (PORT2_RXQ, PORT2_TXQ),
        _ => panic!("port_queues: port id out of range"),
    }
}

/// Static log-friendly label for a port id. Used in tracing fields
/// across the TX / RX / reset-drain paths so structured logs carry a
/// stable port identifier without per-call allocation.
const fn port_label(port_id: usize) -> &'static str {
    match port_id {
        0 => "port0",
        1 => "port1",
        2 => "port2",
        _ => "port?",
    }
}

/// Three-port virtio-console MMIO device.
///
/// Single-struct state — no separate transport layer. Caller holds
/// this in a `PiMutex` and dispatches MMIO reads/writes from the vCPU
/// run loop.
pub struct VirtioConsole {
    queues: [Queue; NUM_QUEUES],
    queue_select: u32,
    device_features_sel: u32,
    driver_features_sel: u32,
    driver_features: u64,
    device_status: u32,
    interrupt_status: u32,
    config_generation: u32,
    /// Eventfd for KVM irqfd — signals guest interrupt.
    irq_evt: EventFd,
    /// Eventfd signaled when TX data is available on port 0 or port 1.
    /// The host's stdout drain thread polls this to wake on port-0
    /// console bytes; the freeze coordinator's TOKEN_TX handler reads
    /// `ports[1].tx_buf` after a generic notification (the eventfd
    /// does not carry per-port granularity between ports 0 and 1, but
    /// the cost of an extra empty drain is negligible). Port 2 TX is
    /// signaled separately via [`Self::stats_tx_evt`] so the stats
    /// client wakes only on its own port.
    tx_evt: EventFd,
    /// Eventfd signaled when TX data is available on port 2 (scheduler
    /// stats relay). Distinct from [`Self::tx_evt`] so the host's
    /// [`super::sched_stats::SchedStatsClient`] poll wakes only on a
    /// stats-response edge — the freeze coordinator's TOKEN_TX
    /// handler does not contend on this fd, and the stats client
    /// does not get spurious wakes from port-0 console / port-1 bulk
    /// traffic.
    stats_tx_evt: EventFd,
    /// Guest memory reference. Set before starting vCPUs.
    mem: Option<GuestMemoryMmap>,
    /// Per-port state, indexed by port id. Replaces the prior
    /// `port{0,1,2}_tx_buf`, `port{0,1,2}_pending_rx`,
    /// `port_opened`, `port_readied` per-port fields with a single
    /// indexed array. See [`Port`] for field semantics.
    ports: [Port; NUM_PORTS as usize],
    /// Scratch staging for TX descriptor reads. `read_slice`
    /// writes into a contiguous `&mut [u8]`; the per-port `tx_buf`
    /// is a `VecDeque` so we read into this scratch first, then
    /// `extend` the deque from it. Shared mutex with the rest of
    /// the device (no concurrent TX/RX), so a single per-device
    /// scratch is safe and avoids per-descriptor heap churn.
    tx_scratch: Vec<u8>,
    /// Per-device reusable scratch for RX delivery; same justification
    /// as the original single-port impl — avoids per-descriptor heap
    /// churn under high paste rates.
    rx_scratch: Vec<u8>,
    /// Outbound control queue: messages waiting for the next c_ivq
    /// descriptor chain (PORT_ADD, PORT_OPEN, CONSOLE_PORT, PORT_NAME).
    /// Serviced FIFO; the host pushes during the
    /// DEVICE_READY/PORT_READY handshake, the guest publishes c_ivq
    /// buffers and we copy one message per chain.
    control_out: VecDeque<ControlOut>,
    /// True once the guest has sent `DEVICE_READY(value=1)` on c_ovq.
    /// Gates the host-side PORT_ADD enqueues — emitting them before
    /// DEVICE_READY would be ignored by the kernel and a per-port
    /// PORT_READY handshake would never start.
    device_ready: bool,
}

impl Default for VirtioConsole {
    fn default() -> Self {
        Self::new()
    }
}

impl VirtioConsole {
    /// Create a new virtio-console device.
    pub fn new() -> Self {
        let irq_evt =
            EventFd::new(libc::EFD_NONBLOCK).expect("failed to create virtio-console irq eventfd");
        let tx_evt =
            EventFd::new(libc::EFD_NONBLOCK).expect("failed to create virtio-console tx eventfd");
        let stats_tx_evt = EventFd::new(libc::EFD_NONBLOCK)
            .expect("failed to create virtio-console stats_tx eventfd");
        VirtioConsole {
            queues: [
                Queue::new(QUEUE_MAX_SIZE).expect("valid queue size"),
                Queue::new(QUEUE_MAX_SIZE).expect("valid queue size"),
                Queue::new(QUEUE_MAX_SIZE).expect("valid queue size"),
                Queue::new(QUEUE_MAX_SIZE).expect("valid queue size"),
                Queue::new(QUEUE_MAX_SIZE).expect("valid queue size"),
                Queue::new(QUEUE_MAX_SIZE).expect("valid queue size"),
                Queue::new(QUEUE_MAX_SIZE).expect("valid queue size"),
                Queue::new(QUEUE_MAX_SIZE).expect("valid queue size"),
            ],
            queue_select: 0,
            device_features_sel: 0,
            driver_features_sel: 0,
            driver_features: 0,
            device_status: 0,
            interrupt_status: 0,
            config_generation: 0,
            irq_evt,
            tx_evt,
            stats_tx_evt,
            mem: None,
            ports: [
                Port::new(PORT0_NAME),
                Port::new(PORT1_NAME),
                Port::new(PORT2_NAME),
            ],
            tx_scratch: Vec::new(),
            rx_scratch: Vec::new(),
            control_out: VecDeque::new(),
            device_ready: false,
        }
    }

    /// Eventfd for KVM irqfd registration.
    pub fn irq_evt(&self) -> &EventFd {
        &self.irq_evt
    }

    /// Eventfd signaled when new TX data arrives on port 0 or port 1.
    /// Use in the host-side stdout / bulk drain thread's poll set for
    /// zero-latency wakeup. Port 2 TX wakes are delivered separately
    /// via [`Self::stats_tx_evt`] so the stats client does not
    /// contend on this fd.
    pub fn tx_evt(&self) -> &EventFd {
        &self.tx_evt
    }

    /// Eventfd signaled when new TX data arrives on port 2 (scheduler
    /// stats relay). Used by
    /// [`super::sched_stats::SchedStatsClient`] to wake on a
    /// stats-response edge without seeing port-0 console / port-1
    /// bulk wakes.
    pub fn stats_tx_evt(&self) -> &EventFd {
        &self.stats_tx_evt
    }

    /// Set guest memory reference. Must be called before starting vCPUs.
    pub fn set_mem(&mut self, mem: GuestMemoryMmap) {
        self.mem = Some(mem);
    }

    fn device_features(&self) -> u64 {
        (1u64 << VIRTIO_F_VERSION_1) | (1u64 << (VIRTIO_CONSOLE_F_MULTIPORT as u64))
    }

    fn selected_queue(&self) -> Option<usize> {
        let idx = self.queue_select as usize;
        if idx < NUM_QUEUES { Some(idx) } else { None }
    }

    // Console does not negotiate VIRTIO_RING_F_EVENT_IDX so the
    // combined bit+eventfd pattern is correct here. virtio_blk
    // splits the two because it negotiates EVENT_IDX.
    fn signal_used(&mut self) {
        self.interrupt_status |= VIRTIO_MMIO_INT_VRING;
        // Success path is silent (high-volume hot path: signal_used
        // fires per drained chain, tens to hundreds of times per
        // console burst); failure logs once per occurrence so a
        // genuine eventfd-write breakage surfaces in tracing rather
        // than silently disappearing.
        if let Err(e) = self.irq_evt.write(1) {
            tracing::warn!(%e, "virtio-console irq_evt.write failed");
        }
    }

    /// True iff the driver negotiated `VIRTIO_CONSOLE_F_MULTIPORT`.
    /// The multiport-only queues (c_ivq, c_ovq, port-1 RX, port-1
    /// TX) are valid only after this feature is acked. Without
    /// F_MULTIPORT the kernel's `init_vqs` allocates only the first
    /// two queues (drivers/char/virtio_console.c — the legacy
    /// single-console path), so any QUEUE_NOTIFY for queues 2-5 in
    /// that case is a guest protocol violation.
    fn multiport_negotiated(&self) -> bool {
        self.driver_features & (1u64 << (VIRTIO_CONSOLE_F_MULTIPORT as u64)) != 0
    }

    /// True when device_status has progressed past FEATURES_OK but not
    /// yet reached DRIVER_OK — the window where queue config is valid.
    fn queue_config_allowed(&self) -> bool {
        self.device_status & S_FEAT == S_FEAT && self.device_status & VIRTIO_CONFIG_S_DRIVER_OK == 0
    }

    /// True when driver features may be written: DRIVER set, FEATURES_OK
    /// not yet set.
    fn features_write_allowed(&self) -> bool {
        self.device_status & S_DRV == S_DRV && self.device_status & VIRTIO_CONFIG_S_FEATURES_OK == 0
    }

    // ------------------------------------------------------------------
    // Port 0 console: guest → host (TX) and host → guest (RX)
    // ------------------------------------------------------------------

    /// Process a TX queue: drain descriptor data into the matching
    /// port's `tx_buf`. TX descriptors are device-readable (guest
    /// wrote them); the device writes nothing back, so add_used len
    /// is 0.
    ///
    /// Returns true when at least one byte was successfully copied —
    /// the caller uses that to gate `signal_used` + `tx_evt.write`.
    fn process_tx(&mut self, port_id: usize) -> bool {
        let port_label = port_label(port_id);
        let (_, queue_idx) = port_queues(port_id);
        // Spec gate: virtio-v1.2 §3.1.1 forbids the device from
        // accessing virtqueue memory before DRIVER_OK. A QUEUE_NOTIFY
        // arriving in a transient state would otherwise let the
        // device read addresses the guest hasn't validated.
        if self.device_status & VIRTIO_CONFIG_S_DRIVER_OK == 0 {
            tracing::debug!(
                queue = queue_idx,
                status = self.device_status,
                "virtio-console process_tx: DRIVER_OK not set; ignoring notify"
            );
            return false;
        }
        // F_MULTIPORT runtime gate: ports 1 and 2 are multiport-only.
        // Port 0 TX is valid in both single-console and multiport
        // configurations.
        if port_id != 0 && !self.multiport_negotiated() {
            tracing::warn!(
                port = port_label,
                "virtio-console process_tx: F_MULTIPORT not \
                 negotiated; ignoring notify on multiport-only TX queue"
            );
            return false;
        }
        let mem = match self.mem.as_ref() {
            Some(m) => m,
            None => return false,
        };
        let mut had_data = false;
        // Cumulative bytes copied into the per-port accumulator
        // during this call. Compared against `TX_PER_CALL_MAX` after
        // every chain so a hostile guest publishing many small
        // chains cannot grow the host buffer without bound on a
        // single notify.
        let mut cumulative_bytes: usize = 0;
        // Disjoint-field borrows: the queue (`self.queues[queue_idx]`)
        // and the per-port accumulator (`self.ports[port_id].tx_buf`,
        // `self.tx_scratch`) are reborrowed independently inside the
        // loop. `tx_buf` is `VecDeque<u8>` and `read_slice` needs a
        // contiguous `&mut [u8]`, so we stage descriptor bytes in
        // `tx_scratch` then `extend` the deque. The deque lets port
        // 1's [`Self::push_back_bulk`] prepend the freeze
        // coordinator's residual in O(bytes) without an
        // O(buf_len + bytes) `Vec::splice(0..0, _)`.
        let q = &mut self.queues[queue_idx];
        while let Some(chain) = q.pop_descriptor_chain(mem) {
            let head = chain.head_index();
            for desc in chain {
                if !desc.is_write_only() {
                    let guest_addr = desc.addr();
                    let dlen = (desc.len() as usize).min(TX_DESC_MAX);
                    self.tx_scratch.clear();
                    self.tx_scratch.resize(dlen, 0);
                    let read_ok = match mem.read_slice(&mut self.tx_scratch, guest_addr) {
                        Ok(()) => {
                            self.ports[port_id]
                                .tx_buf
                                .extend(self.tx_scratch.iter().copied());
                            true
                        }
                        Err(e) => {
                            tracing::warn!(
                                port = port_label,
                                head,
                                dlen,
                                %e,
                                "virtio-console process_tx: read_slice failed \
                                 (descriptor addr likely unmapped); dropping \
                                 segment from this chain"
                            );
                            false
                        }
                    };
                    // Gate had_data and cumulative_bytes on dlen > 0.
                    // A zero-length descriptor's read_slice trivially
                    // succeeds (read_ok = true) but contributes no
                    // bytes — neither cumulative_bytes (which feeds
                    // the per-call cap at TX_PER_CALL_MAX) nor the
                    // accumulator grows. Setting had_data=true on a
                    // zero-byte chain would still trigger
                    // signal_used (IRQ to guest) and the tx_evt
                    // wake — a hostile guest publishing N
                    // zero-length chains gets N IRQs without any
                    // host-observable byte progress, an
                    // amplification primitive that bypasses the
                    // TX_PER_CALL_MAX bound (cumulative_bytes stays
                    // 0 forever). The kernel's virtio_console TX
                    // path always emits non-empty data segments
                    // (drivers/char/virtio_console.c
                    // __send_to_port via sg_set_buf with non-zero
                    // length), so legitimate guests are unaffected.
                    if read_ok && dlen > 0 {
                        had_data = true;
                        cumulative_bytes = cumulative_bytes.saturating_add(dlen);
                    }
                }
            }
            // TX add_used: on failure the descriptor leaks (guest's
            // TX queue eventually starves and the port stops). Log so
            // the silent-stop is observable rather than disappearing
            // into a swallowed Result. Mirrors virtio-blk's
            // `publish_completion` and virtio-net's TX add_used
            // logging pattern.
            if let Err(e) = q.add_used(mem, head, 0) {
                tracing::warn!(
                    port = port_label,
                    head,
                    %e,
                    "virtio-console TX add_used failed (used-ring address \
                     likely unmapped); guest TX queue will eventually \
                     starve and this port will stop"
                );
            }
            // Per-call drain cap: stop popping chains once the
            // cumulative byte total crosses the budget. Remaining
            // chains stay in the avail ring; the next QUEUE_NOTIFY
            // (or the next host-side drain) picks them up. A hostile
            // guest publishing thousands of valid PAGE_SIZE chains
            // would otherwise let one notify allocate hundreds of
            // megabytes into the host accumulator before returning
            // to the vCPU loop.
            if cumulative_bytes >= TX_PER_CALL_MAX {
                tracing::debug!(
                    port = port_label,
                    cumulative_bytes,
                    cap = TX_PER_CALL_MAX,
                    "virtio-console process_tx: per-call byte cap reached; \
                     remaining chains deferred to next notify"
                );
                break;
            }
        }
        if had_data {
            self.signal_used();
            // Wake the matching host poll thread. Ports 0 and 1 share
            // `tx_evt` (the freeze coordinator's TOKEN_TX handler
            // drains both); port 2 fires its own `stats_tx_evt` so
            // [`super::sched_stats::SchedStatsClient`] wakes only on
            // a stats edge. A missed write means the host poll
            // absorbs the latency next cycle — not a correctness
            // failure. Silent swallow is intentional (in contrast to
            // signal_used's irq_evt write, which logs because a
            // missed IRQ stalls the GUEST, not just a host poll
            // cadence).
            if port_id == 2 {
                let _ = self.stats_tx_evt.write(1);
            } else {
                let _ = self.tx_evt.write(1);
            }
        }
        had_data
    }

    /// Drain-only TX walk for the device-reset path. Pops pending
    /// avail-ring chains for `queue_idx`, copies their bytes into the
    /// matching `port{0,1}_tx_buf`, and `add_used`s each chain — but
    /// emits NO `signal_used` (no IRQ to a guest that is rebooting)
    /// and NO `tx_evt.write` (no host wake; `collect_results`'s
    /// `final_drain` walks the buffer synchronously after the
    /// coordinator has already exited).
    ///
    /// Reset is called from `mmio_write` while the vCPU thread holds
    /// the device's outer mutex. Calling `process_tx` here would
    /// fire `tx_evt`, waking the freeze coordinator's TOKEN_TX handler
    /// which races to acquire the same mutex — a lock-contention
    /// feedback loop under burst workloads where 16 workers / 8 vCPUs
    /// flood the bulk port. The drain-only variant breaks the loop:
    /// the coord is never woken about a drain it doesn't need to do
    /// (the bytes are already captured for `final_drain` to surface).
    ///
    /// The DRIVER_OK and F_MULTIPORT gates from `process_tx` are
    /// preserved verbatim — `reset()` invokes this BEFORE clearing
    /// `device_status` and `driver_features`, so both gates pass on
    /// the legitimate reboot path. Calling this method post-reset
    /// (when both gates would fail) is a no-op, mirroring
    /// `process_tx`'s behaviour.
    ///
    /// Per-call byte cap (`TX_PER_CALL_MAX`) is preserved as a
    /// defensive bound against a hostile guest that publishes a
    /// massive avail-ring backlog right before the reboot — the
    /// remainder is left in the avail ring and is unreachable post-
    /// reset (queue GPAs get reset to sentinels), exactly the same
    /// data-loss surface as before this change. Test runs with the
    /// kernel's virtio_console driver never exercise this bound on
    /// the reset path; the cap is hostile-guest defence-in-depth.
    fn drain_tx_into_capture_buf(&mut self, port_id: usize) {
        let port_label = port_label(port_id);
        let (_, queue_idx) = port_queues(port_id);
        if self.device_status & VIRTIO_CONFIG_S_DRIVER_OK == 0 {
            return;
        }
        if port_id != 0 && !self.multiport_negotiated() {
            return;
        }
        let mem = match self.mem.as_ref() {
            Some(m) => m,
            None => return,
        };
        let mut cumulative_bytes: usize = 0;
        let q = &mut self.queues[queue_idx];
        while let Some(chain) = q.pop_descriptor_chain(mem) {
            let head = chain.head_index();
            for desc in chain {
                if !desc.is_write_only() {
                    let guest_addr = desc.addr();
                    let dlen = (desc.len() as usize).min(TX_DESC_MAX);
                    self.tx_scratch.clear();
                    self.tx_scratch.resize(dlen, 0);
                    let read_ok = match mem.read_slice(&mut self.tx_scratch, guest_addr) {
                        Ok(()) => {
                            self.ports[port_id]
                                .tx_buf
                                .extend(self.tx_scratch.iter().copied());
                            true
                        }
                        Err(e) => {
                            tracing::warn!(
                                port = port_label,
                                head,
                                dlen,
                                %e,
                                "virtio-console reset-drain: read_slice failed \
                                 (descriptor addr likely unmapped); dropping \
                                 segment from this chain"
                            );
                            false
                        }
                    };
                    if read_ok && dlen > 0 {
                        cumulative_bytes = cumulative_bytes.saturating_add(dlen);
                    }
                }
            }
            if let Err(e) = q.add_used(mem, head, 0) {
                tracing::warn!(
                    port = port_label,
                    head,
                    %e,
                    "virtio-console reset-drain: add_used failed (used-ring \
                     address likely unmapped); descriptor leaks but the guest \
                     is rebooting so the leak has no observer"
                );
            }
            if cumulative_bytes >= TX_PER_CALL_MAX {
                tracing::debug!(
                    port = port_label,
                    cumulative_bytes,
                    cap = TX_PER_CALL_MAX,
                    "virtio-console reset-drain: per-call byte cap reached; \
                     remaining chains lost to queue reset"
                );
                break;
            }
        }
    }

    /// Drain a port's `tx_buf` and return its bytes as a contiguous
    /// `Vec<u8>`. Capacity-preserving swap: the replacement deque
    /// keeps the drained deque's capacity (capped at 256 KiB so a
    /// single hostile-guest burst that grew the accumulator beyond
    /// the per-call cap does not retain that capacity for the
    /// lifetime of the device). `Vec::from(VecDeque)` reuses the
    /// deque's allocation when the ring is contiguous (O(1)) and
    /// rotates in place when it isn't (O(N) but no realloc).
    fn drain_port_tx(&mut self, port_id: usize) -> Vec<u8> {
        let buf = &mut self.ports[port_id].tx_buf;
        let cap = buf.capacity().min(256 * 1024);
        let old = std::mem::replace(buf, VecDeque::with_capacity(cap));
        Vec::from(old)
    }

    /// Return and clear accumulated port-0 TX output (guest console →
    /// host stdout).
    pub fn drain_output(&mut self) -> Vec<u8> {
        self.drain_port_tx(0)
    }

    /// Return and clear accumulated port-1 TX output (guest bulk TLV
    /// stream). Host-side TLV parsing is in
    /// [`crate::vmm::host_comms::parse_tlv_stream`].
    pub fn drain_bulk(&mut self) -> Vec<u8> {
        self.drain_port_tx(1)
    }

    /// Final post-VM-exit port-1 TX drain. Walks the avail ring once
    /// to convert any descriptor chains the guest published without
    /// a corresponding QUEUE_NOTIFY MMIO arrival into bytes in port
    /// 1's `tx_buf`, then returns the accumulated buffer.
    ///
    /// `pub fn drain_bulk` alone catches only what `process_tx`
    /// has already deposited; on the eevdf-style failure path the
    /// guest writes a `MSG_TYPE_LIFECYCLE` and a `MSG_TYPE_EXIT`
    /// frame to `/dev/vport0p1` and immediately calls `force_reboot`,
    /// and the userspace write's `virtqueue_kick` MMIO can lag
    /// behind so the chains land in the avail ring without a
    /// matching host-side notify. A single explicit `process_tx`
    /// call from the host side picks them up.
    ///
    /// Synchronous call from `collect_results`; the underlying
    /// `process_tx` is the same code path MMIO QUEUE_NOTIFY
    /// uses, including the per-call `TX_PER_CALL_MAX` byte cap and
    /// the `DRIVER_OK` / `F_MULTIPORT` gates. If the guest already
    /// wrote 0 to VIRTIO_MMIO_STATUS, [`Self::reset`] has run; that
    /// path drains every port's TX queue via
    /// [`Self::drain_tx_into_capture_buf`] at its top (before the
    /// `device_status = 0` clobber and the `Queue::reset` GPA
    /// clobber) so any chains pending at reset time are already in
    /// `ports[1].tx_buf`. The `process_tx(1)` call below short-
    /// circuits on `DRIVER_OK == 0` and `drain_bulk` returns the
    /// captured bytes.
    pub(crate) fn final_drain(&mut self) -> Vec<u8> {
        let _ = self.process_tx(1);
        self.drain_bulk()
    }

    /// Push raw bytes back onto the head of the port-1 TX buffer.
    ///
    /// The freeze coordinator's mid-run `bulk_assembler` (see
    /// `crate::vmm::bulk::HostAssembler`) drains port 1's `tx_buf`
    /// via `drain_bulk` and assembles complete TLV frames. Trailing
    /// bytes of a partial frame stay buffered inside the assembler.
    /// When the coordinator thread exits (kill or BSP-done), those
    /// residual bytes would be dropped on the floor — `collect_results`
    /// then calls `drain_bulk` and parses with `parse_tlv_stream`,
    /// but the assembler-buffered tail never reaches that path. This
    /// method lets the coordinator push the residual back so
    /// `collect_results`'s `drain_bulk` returns it and `parse_tlv_stream`
    /// completes the frame. Bytes prepend to preserve the on-wire
    /// chronological order: the assembler's residual was at the head
    /// of the bulk stream relative to any bytes the device accumulated
    /// after the last coordinator drain.
    pub fn push_back_bulk(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        // The port-1 `tx_buf` is a VecDeque — push_front is O(1) per
        // byte (amortised); iterating in reverse and pushing each
        // lands `bytes[0]` at the very front of the deque, preserving
        // the chronological order described by the doc comment above.
        // This replaces a `Vec::splice(0..0, _)` whose cost was
        // O(buf_len + bytes) — under bursty TLV traffic that already
        // grew the buffer to hundreds of KiB the splice would
        // memmove every byte in the buffer to make room, which a
        // residual push at the very end of the run was not worth.
        // `reserve` lifts the capacity once so the push_front loop
        // does not trigger N small grows.
        let buf = &mut self.ports[1].tx_buf;
        buf.reserve(bytes.len());
        for &b in bytes.iter().rev() {
            buf.push_front(b);
        }
    }

    /// Return and clear accumulated port-2 TX output (guest stats
    /// relay → host stats client). Raw byte passthrough — no TLV
    /// parsing is applied. Same capacity-preserving swap as
    /// [`Self::drain_bulk`].
    pub fn drain_port2_bulk(&mut self) -> Vec<u8> {
        self.drain_port_tx(2)
    }

    /// Test helper — return all accumulated port-0 TX output as a string.
    #[cfg(test)]
    pub fn output(&self) -> String {
        let bytes: Vec<u8> = self.ports[0].tx_buf.iter().copied().collect();
        String::from_utf8_lossy(&bytes).to_string()
    }

    /// Test helper — return a copy of the pending port-0 RX bytes
    /// (host → guest direction) that have not yet been delivered to
    /// the guest. Tests that exercise the host-side wake-byte
    /// pushers without a fully-wired guest queue use this to inspect
    /// what would have been delivered.
    #[cfg(test)]
    pub fn pending_rx_bytes(&self) -> Vec<u8> {
        self.ports[0].pending_rx.iter().copied().collect()
    }

    // ------------------------------------------------------------------
    // Port 0 RX: host → guest console
    // ------------------------------------------------------------------

    /// Push host data into the guest's port-0 RX buffers. Same
    /// semantics as the original single-port `queue_input` —
    /// undelivered bytes accumulate in port 0's `pending_rx` and
    /// drain on the next QUEUE_NOTIFY for q0.
    pub fn queue_input(&mut self, data: &[u8]) {
        tracing::debug!(bytes = data.len(), "virtio-console queue_input");
        self.ports[0].pending_rx.extend(data);
        self.drain_pending_rx(0);
    }

    // ------------------------------------------------------------------
    // Port 1 RX: host → guest bulk channel (TLV reply frames)
    // ------------------------------------------------------------------

    /// Push host data into the guest's port-1 RX buffers. Used by the
    /// freeze coordinator's snapshot-request handler to deliver a
    /// [`super::wire::SnapshotReplyPayload`] back to the in-guest
    /// `request_snapshot` blocking reader. Bytes that cannot be
    /// delivered immediately (no chain available, port not opened
    /// yet, DRIVER_OK not set) accumulate in port 1's `pending_rx`
    /// and drain on the next q4 (`PORT1_RXQ`) notify.
    pub(crate) fn queue_input_port1(&mut self, data: &[u8]) {
        tracing::debug!(bytes = data.len(), "virtio-console queue_input_port1");
        self.ports[1].pending_rx.extend(data);
        self.drain_pending_rx(1);
    }

    // ------------------------------------------------------------------
    // Port 2 RX: host → guest scheduler-stats relay
    // ------------------------------------------------------------------

    /// Push host data into the guest's port-2 RX buffers. Used by the
    /// host's [`super::sched_stats::SchedStatsClient`] to deliver
    /// scx_stats request bytes to the in-guest relay thread that
    /// forwards them to `/var/run/scx/root/stats`. Bytes that cannot
    /// be delivered immediately accumulate in port 2's `pending_rx`
    /// and drain on the next q6 (`PORT2_RXQ`) notify. Mirrors
    /// [`Self::queue_input_port1`].
    pub(crate) fn queue_input_port2(&mut self, data: &[u8]) {
        tracing::debug!(bytes = data.len(), "virtio-console queue_input_port2");
        self.ports[2].pending_rx.extend(data);
        self.drain_pending_rx(2);
    }

    /// Drop any host→guest port-2 request bytes that have not yet
    /// been consumed by the guest. Called by
    /// [`super::sched_stats::SchedStatsClient::request_raw`] at the
    /// start of every fresh request: a freeze rendezvous that
    /// landed mid-request can leave half a JSON request line in port
    /// 2's `pending_rx`. If the next request just pushed onto the
    /// deque, the guest relay would read the previous-request tail
    /// concatenated with the new request, producing torn JSON the
    /// scheduler can't parse. Returns the number of bytes
    /// discarded so the caller can log/account for them.
    pub(crate) fn clear_port2_pending_rx(&mut self) -> usize {
        let pending = &mut self.ports[2].pending_rx;
        let n = pending.len();
        pending.clear();
        n
    }

    /// Drain a port's pending RX bytes into guest write-only
    /// descriptors. Replaces the prior per-port
    /// `drain_port{0,1,2}_pending_rx` triplet — port 1 and port 2
    /// add F_MULTIPORT and `opened` gates atop the port-0 baseline.
    /// Only publish a chain when ALL writes for that chain
    /// succeeded; otherwise keep bytes in `pending_rx` for retry.
    fn drain_pending_rx(&mut self, port_id: usize) {
        let port_label = port_label(port_id);
        let (queue_idx, _) = port_queues(port_id);
        if self.ports[port_id].pending_rx.is_empty() {
            return;
        }
        if self.device_status & VIRTIO_CONFIG_S_DRIVER_OK == 0 {
            tracing::debug!(
                port = port_label,
                pending = self.ports[port_id].pending_rx.len(),
                status = self.device_status,
                "virtio-console drain_pending_rx: DRIVER_OK not set; deferring"
            );
            return;
        }
        // F_MULTIPORT runtime gate: ports 1 and 2 are multiport-only.
        // The legacy single-console path never sees their traffic, so
        // any QUEUE_NOTIFY for a multiport-only RX queue without
        // F_MULTIPORT is a guest protocol violation. Bytes stay in
        // `pending_rx` for a future probe.
        if port_id != 0 && !self.multiport_negotiated() {
            tracing::warn!(
                port = port_label,
                pending = self.ports[port_id].pending_rx.len(),
                "virtio-console drain_pending_rx: F_MULTIPORT \
                 not negotiated; deferring host→guest bytes"
            );
            return;
        }
        // Per-port `opened` gate: ports 1 and 2 are multiport
        // channels that the kernel only opens after the host's
        // `PORT_OPEN` control message completes the handshake.
        // Pushing bytes before the open landed would let the kernel
        // discard them with no userspace reader. Port 0 (console)
        // starts implicitly open in the hvc-console path.
        if port_id != 0 && !self.ports[port_id].opened {
            tracing::debug!(
                port = port_label,
                pending = self.ports[port_id].pending_rx.len(),
                "virtio-console drain_pending_rx: port not yet opened by guest; deferring"
            );
            return;
        }
        let mem = match self.mem.as_ref() {
            Some(m) => m,
            None => {
                tracing::debug!(
                    port = port_label,
                    pending = self.ports[port_id].pending_rx.len(),
                    "virtio-console drain_pending_rx: no mem"
                );
                return;
            }
        };
        if !self.queues[queue_idx].ready() {
            tracing::debug!(
                port = port_label,
                pending = self.ports[port_id].pending_rx.len(),
                "virtio-console drain_pending_rx: RX queue not ready"
            );
            return;
        }
        let q = &mut self.queues[queue_idx];
        let mut total_written = 0u32;
        let mut chains_drained = 0usize;
        while !self.ports[port_id].pending_rx.is_empty() {
            let Some(chain) = q.pop_descriptor_chain(mem) else {
                break;
            };
            let head = chain.head_index();
            let mut consumed_offset = 0usize;
            let mut written = 0u32;
            let mut chain_torn = false;
            for desc in chain {
                if desc.is_write_only() && consumed_offset < self.ports[port_id].pending_rx.len() {
                    let guest_addr = desc.addr();
                    let avail = desc.len() as usize;
                    let remaining = self.ports[port_id].pending_rx.len() - consumed_offset;
                    let chunk = remaining.min(avail);
                    self.rx_scratch.clear();
                    let (head_slice, tail_slice) = self.ports[port_id].pending_rx.as_slices();
                    let head_skip = consumed_offset.min(head_slice.len());
                    let tail_skip = consumed_offset - head_skip;
                    let head_avail = &head_slice[head_skip..];
                    let tail_avail = if tail_skip < tail_slice.len() {
                        &tail_slice[tail_skip..]
                    } else {
                        &[][..]
                    };
                    let h = head_avail.len().min(chunk);
                    self.rx_scratch.extend_from_slice(&head_avail[..h]);
                    if h < chunk {
                        let t = (chunk - h).min(tail_avail.len());
                        self.rx_scratch.extend_from_slice(&tail_avail[..t]);
                    }
                    if mem.write_slice(&self.rx_scratch, guest_addr).is_ok() {
                        let n = self.rx_scratch.len();
                        consumed_offset += n;
                        written += n as u32;
                    } else {
                        tracing::warn!(
                            port = port_label,
                            head,
                            written,
                            "virtio-console drain_pending_rx: write_slice failed \
                             mid-chain; breaking out to avoid partial-fill corruption"
                        );
                        chain_torn = true;
                        break;
                    }
                }
            }
            if chain_torn {
                // Publish the head with len=0 so the guest reclaims
                // the descriptor instead of leaking it (mirrors the
                // c_ivq torn-write path). Bytes stay in `pending_rx`
                // for retry on the next chain. Without this add_used
                // the chain head is consumed from avail but never
                // returned to the used ring, and the descriptor index
                // leaks until reset.
                if let Err(e) = q.add_used(mem, head, 0) {
                    tracing::warn!(
                        port = port_label,
                        head,
                        %e,
                        "virtio-console drain_pending_rx: add_used(0) \
                         after torn write failed; chain head leaked"
                    );
                }
                break;
            }
            if let Err(e) = q.add_used(mem, head, written) {
                tracing::warn!(
                    port = port_label,
                    head,
                    written,
                    %e,
                    "virtio-console RX add_used failed (used-ring address \
                     likely unmapped); bytes preserved in pending_rx for \
                     retry on the next drain cycle"
                );
                break;
            }
            self.ports[port_id].pending_rx.drain(..consumed_offset);
            total_written += written;
            chains_drained += 1;
            // Per-call chain count cap: bound the per-vCPU
            // MMIO-handler latency under a hostile guest that
            // publishes many zero-progress chains (zero-length or
            // missing-write-only descriptors). Remaining chains
            // stay in the avail ring; the next QUEUE_NOTIFY (or the
            // next host-side push) picks them up. Mirrors the
            // `CONTROL_CHAINS_PER_CALL_MAX` pattern in
            // `process_control_tx`.
            if chains_drained >= RX_CHAINS_PER_CALL_MAX {
                tracing::debug!(
                    port = port_label,
                    chains_drained,
                    cap = RX_CHAINS_PER_CALL_MAX,
                    pending = self.ports[port_id].pending_rx.len(),
                    "virtio-console drain_pending_rx: per-call chain \
                     cap reached; remaining chains deferred to next notify"
                );
                break;
            }
        }
        if total_written > 0 {
            tracing::debug!(
                port = port_label,
                delivered = total_written,
                pending = self.ports[port_id].pending_rx.len(),
                "virtio-console drain_pending_rx: delivered to guest",
            );
            self.signal_used();
        }
    }

    // ------------------------------------------------------------------
    // Control protocol (multiport)
    // ------------------------------------------------------------------

    /// Process the c_ovq (queue 3): guest-originated control messages.
    /// Reads each chain's head descriptor as a `VirtioConsoleControl`
    /// frame, dispatches by `event`, and add_useds the chain.
    fn process_control_tx(&mut self) {
        if self.device_status & VIRTIO_CONFIG_S_DRIVER_OK == 0 {
            tracing::debug!("virtio-console c_ovq: DRIVER_OK not set; ignoring notify");
            return;
        }
        // F_MULTIPORT runtime gate: c_ovq exists only when multiport
        // is negotiated. A guest that didn't ack F_MULTIPORT but
        // still fires QUEUE_NOTIFY for queue 3 is misbehaving;
        // refuse to walk the queue.
        if !self.multiport_negotiated() {
            tracing::warn!(
                "virtio-console c_ovq: F_MULTIPORT not negotiated; \
                 ignoring notify on multiport-only queue"
            );
            return;
        }
        // Move work into local Vec so we can release the queue borrow
        // before calling back into self for control_out enqueue.
        let mut events: Vec<VirtioConsoleControl> = Vec::new();
        // Track whether any add_used call succeeded so the
        // unconditional `signal_used` below becomes conditional —
        // mirrors `drain_control_in`'s `any_published` discipline.
        // A QUEUE_NOTIFY that finds the avail ring empty pops zero
        // chains, publishes zero used entries, and must NOT raise
        // an interrupt: an unprovoked irq turns every spurious
        // notify into work for the guest's vring_interrupt path
        // (which then re-checks an empty used ring and returns).
        let mut any_published = false;
        {
            let mem = match self.mem.as_ref() {
                Some(m) => m,
                None => return,
            };
            let q = &mut self.queues[C_OVQ];
            let mut chains_drained = 0usize;
            while let Some(chain) = q.pop_descriptor_chain(mem) {
                let head = chain.head_index();
                let mut total = 0u32;
                let mut buf = [0u8; VC_CONTROL_SIZE];
                let mut need = VC_CONTROL_SIZE;
                let mut filled = 0usize;
                for desc in chain {
                    if desc.is_write_only() || need == 0 {
                        continue;
                    }
                    let take = desc.len().min(need as u32) as usize;
                    if take == 0 {
                        continue;
                    }
                    if let Err(e) = mem.read_slice(&mut buf[filled..filled + take], desc.addr()) {
                        tracing::warn!(%e, head, "c_ovq read_slice failed");
                        break;
                    }
                    filled += take;
                    need -= take;
                    total += take as u32;
                }
                if filled == VC_CONTROL_SIZE
                    && let Ok(c) = VirtioConsoleControl::read_from_bytes(&buf)
                {
                    events.push(c);
                }
                match q.add_used(mem, head, total) {
                    Ok(()) => any_published = true,
                    Err(e) => {
                        tracing::warn!(
                            head,
                            total,
                            %e,
                            "virtio-console c_ovq add_used failed"
                        );
                    }
                }
                chains_drained += 1;
                // Per-call chain count cap: bound the per-vCPU
                // MMIO-handler latency under a hostile guest that
                // publishes thousands of control chains. Remaining
                // chains stay in the avail ring; the next
                // QUEUE_NOTIFY picks them up. Mirrors the
                // `TX_PER_CALL_MAX` pattern in `process_tx`.
                if chains_drained >= CONTROL_CHAINS_PER_CALL_MAX {
                    tracing::debug!(
                        chains_drained,
                        cap = CONTROL_CHAINS_PER_CALL_MAX,
                        "virtio-console process_control_tx: per-call chain \
                         cap reached; remaining chains deferred to next notify"
                    );
                    break;
                }
            }
        }
        for c in events {
            self.handle_control_event(c);
        }
        // Only kick the guest when the used ring actually advanced —
        // if no chain was popped (empty avail ring) or every add_used
        // failed, there is nothing for the guest to consume. Mirrors
        // the `if any_published { self.signal_used(); }` gate at the
        // tail of `drain_control_in`. Then attempt to push pending
        // outbound control messages onto c_ivq (which may have been
        // refilled by the guest after this notify) — `drain_control_in`
        // owns its own conditional `signal_used` so it remains correct
        // even when this branch skipped the kick.
        if any_published {
            self.signal_used();
        }
        self.drain_control_in();
    }

    fn handle_control_event(&mut self, c: VirtioConsoleControl) {
        let id = c.id;
        let event = c.event;
        let value = c.value;
        match event {
            VIRTIO_CONSOLE_DEVICE_READY => {
                if value != 1 {
                    tracing::warn!(value, "virtio-console DEVICE_READY value != 1");
                    return;
                }
                // Reject repeats: a hostile or buggy guest sending
                // DEVICE_READY a second time would re-enqueue PORT_ADD
                // for every port and grow `control_out` without
                // bound. The kernel sends DEVICE_READY exactly once
                // per probe (drivers/char/virtio_console.c
                // `virtcons_probe`), so any subsequent message is a
                // guest protocol violation.
                if self.device_ready {
                    tracing::warn!("virtio-console DEVICE_READY repeat ignored");
                    return;
                }
                self.device_ready = true;
                // Send PORT_ADD for every port we expose. value=1
                // matches QEMU (hw/char/virtio-serial-bus.c
                // `send_control_event(... PORT_ADD, 1)` at the
                // probe-time fanout); the kernel parser
                // (`handle_control_message`) ignores `value` for
                // PORT_ADD but the wire convention pins value=1.
                for port_id in 0..NUM_PORTS {
                    self.control_out
                        .push_back(ControlOut::Cmd(VirtioConsoleControl {
                            id: port_id,
                            event: VIRTIO_CONSOLE_PORT_ADD,
                            value: 1,
                        }));
                }
            }
            VIRTIO_CONSOLE_PORT_READY => {
                if value != 1 {
                    // value=0 is the kernel's add_port-failed signal
                    // (drivers/char/virtio_console.c `add_port` fail
                    // path: `__send_control_msg(portdev, id,
                    // VIRTIO_CONSOLE_PORT_READY, 0)`). Surface as
                    // tracing::error so a guest probe failure is
                    // visible at the host log level rather than
                    // wedging silently behind a debug log. Other
                    // values are protocol violations from a
                    // misbehaving guest; warn covers them.
                    if value == 0 {
                        tracing::error!(
                            id,
                            "virtio-console PORT_READY value=0: guest \
                             reports add_port failure for this port \
                             (kernel virtio_console.c add_port error \
                             path). The port will not function and the \
                             control handshake for this port will not \
                             complete."
                        );
                    } else {
                        tracing::warn!(id, value, "virtio-console PORT_READY != 1");
                    }
                    return;
                }
                if id >= NUM_PORTS {
                    tracing::warn!(id, "virtio-console PORT_READY for unknown port");
                    return;
                }
                // Reject repeats: a hostile or buggy guest sending
                // PORT_READY a second time for the same port would
                // re-enqueue CONSOLE_PORT / PORT_OPEN / PORT_NAME
                // and grow `control_out` without bound. The kernel
                // sends PORT_READY exactly once per port
                // (drivers/char/virtio_console.c `add_port`), so any
                // subsequent message for the same port is a guest
                // protocol violation.
                if self.ports[id as usize].readied {
                    tracing::warn!(id, "virtio-console PORT_READY repeat ignored");
                    return;
                }
                self.ports[id as usize].readied = true;
                let name = self.ports[id as usize].name;
                if id == 0 {
                    // Console port: announce, name, then open. The
                    // CONSOLE_PORT marker tells the guest this port
                    // is the system console (drivers/char/virtio_console.c
                    // `handle_control_message` CONSOLE_PORT case sets
                    // `port->cons.hvc` and calls `init_port_console`).
                    // PORT_NAME creates the sysfs `name` attribute
                    // (`/sys/class/virtio-ports/vport0p0/name`) so
                    // tooling that scans port names can find port 0;
                    // without it the attribute does not exist. PORT_OPEN
                    // matches QEMU's emission order
                    // (hw/char/virtio-serial-bus.c — PORT_NAME before
                    // PORT_OPEN), keeping udev symlink creation ahead
                    // of any userspace open of `/dev/hvc0`.
                    self.control_out
                        .push_back(ControlOut::Cmd(VirtioConsoleControl {
                            id,
                            event: VIRTIO_CONSOLE_CONSOLE_PORT,
                            value: 1,
                        }));
                    self.control_out.push_back(ControlOut::Name { id, name });
                    self.control_out
                        .push_back(ControlOut::Cmd(VirtioConsoleControl {
                            id,
                            event: VIRTIO_CONSOLE_PORT_OPEN,
                            value: 1,
                        }));
                } else {
                    // Bulk data port: name then open. Order matches
                    // QEMU's PORT_READY handler in
                    // hw/char/virtio-serial-bus.c (PORT_NAME goes out
                    // before PORT_OPEN at lines 425 / 430). The
                    // kernel's `handle_control_message` PORT_NAME
                    // case creates the sysfs `name` attribute, which
                    // udev rules consume to symlink the port; sending
                    // PORT_OPEN first races udev's symlink creation
                    // against userspace opens of /dev/vport0p{1,2}.
                    self.control_out.push_back(ControlOut::Name { id, name });
                    self.control_out
                        .push_back(ControlOut::Cmd(VirtioConsoleControl {
                            id,
                            event: VIRTIO_CONSOLE_PORT_OPEN,
                            value: 1,
                        }));
                }
            }
            VIRTIO_CONSOLE_PORT_OPEN => {
                if id >= NUM_PORTS {
                    tracing::warn!(id, "virtio-console PORT_OPEN for unknown port");
                    return;
                }
                let now_open = value == 1;
                let was_open = self.ports[id as usize].opened;
                self.ports[id as usize].opened = now_open;
                // When ports 1 or 2 transition closed→open, kick the
                // matching pending-RX drain. The host may have queued
                // snapshot replies (port 1) or stats requests (port 2)
                // before the guest finished its PORT_OPEN handshake
                // (the bulk port appears asynchronously after
                // multiport completes); without a drain trigger here
                // those bytes would sit in the port's `pending_rx`
                // until the next q4 / q6 notify, which a guest still
                // in `read()` may not generate promptly. Port 0
                // (console) does not require this kick because port-0
                // RX has no `opened` gate.
                if now_open && !was_open && id != 0 {
                    self.drain_pending_rx(id as usize);
                }
            }
            other => {
                tracing::debug!(
                    id,
                    event = other,
                    value,
                    "virtio-console: unhandled c_ovq event"
                );
            }
        }
    }

    /// Push pending control messages onto c_ivq (host→guest control).
    /// One message per descriptor chain — we publish only when the
    /// chain has enough write-only space to hold the whole message.
    fn drain_control_in(&mut self) {
        if self.control_out.is_empty() {
            return;
        }
        if self.device_status & VIRTIO_CONFIG_S_DRIVER_OK == 0 {
            return;
        }
        // F_MULTIPORT runtime gate: c_ivq is multiport-only. Without
        // negotiation the kernel never allocates this queue, so any
        // descriptor chain we'd find here is a guest protocol
        // violation. Defer the messages — they'll wait in
        // `control_out` until either F_MULTIPORT is negotiated on a
        // future probe (after reset) or the device is reset.
        if !self.multiport_negotiated() {
            tracing::warn!(
                pending = self.control_out.len(),
                "virtio-console c_ivq: F_MULTIPORT not negotiated; \
                 deferring control messages"
            );
            return;
        }
        let mem = match self.mem.as_ref() {
            Some(m) => m,
            None => return,
        };
        if !self.queues[C_IVQ].ready() {
            return;
        }
        let q = &mut self.queues[C_IVQ];
        // Tracks whether any used-ring entry was published this
        // call (including add_used(0) for too-small / torn chains)
        // so signal_used fires. Without this, a sequence of
        // too-small chains would publish add_used(0) entries that
        // the guest never sees an interrupt for.
        let mut any_published = false;
        let mut scratch: Vec<u8> = Vec::with_capacity(64);
        while let Some(msg) = self.control_out.front() {
            let need = msg.len();
            let Some(chain) = q.pop_descriptor_chain(mem) else {
                break;
            };
            let head = chain.head_index();
            // Tally chain capacity (write-only bytes available).
            let segs: Vec<(u64, usize)> = chain
                .filter(|d| d.is_write_only())
                .map(|d| (d.addr().0, d.len() as usize))
                .collect();
            let avail: usize = segs.iter().map(|(_, l)| *l).sum();
            if avail < need {
                // Chain cannot hold this message — push it back via
                // add_used(0) so the guest reclaims it; the message
                // stays in `control_out` for the next chain. Try
                // the next chain instead of breaking out — the guest
                // may have published a mix of small (single-byte
                // probe) and properly-sized (PAGE_SIZE) chains, and
                // a too-small chain at the front of avail must not
                // strand a large message that a later chain in the
                // ring could hold.
                tracing::warn!(
                    head,
                    avail,
                    need,
                    "virtio-console c_ivq: chain too small for control \
                     message; trying next chain"
                );
                if let Err(e) = q.add_used(mem, head, 0) {
                    tracing::warn!(head, %e, "virtio-console c_ivq add_used(0) failed");
                } else {
                    any_published = true;
                }
                continue;
            }
            scratch.clear();
            msg.write_into(&mut scratch);
            let mut written = 0u32;
            let mut idx = 0usize;
            let mut torn = false;
            for (gpa, seg_len) in &segs {
                if idx >= scratch.len() {
                    break;
                }
                let chunk = (*seg_len).min(scratch.len() - idx);
                if let Err(e) =
                    mem.write_slice(&scratch[idx..idx + chunk], vm_memory::GuestAddress(*gpa))
                {
                    tracing::warn!(
                        head,
                        %e,
                        "virtio-console c_ivq write_slice failed mid-chain"
                    );
                    torn = true;
                    break;
                }
                idx += chunk;
                written += chunk as u32;
            }
            if torn {
                // Torn write: publish the head with len=0 so the
                // guest reclaims the descriptor without leaking it.
                // The kernel does NOT gate cpkt parsing on buf->len
                // — drivers/char/virtio_console.c
                // `handle_control_message` reads `cpkt = (... *)(buf
                // ->buf + buf->offset)` unconditionally, and
                // `control_work_handler` only uses `len` to clamp
                // `buf->len` (used later for PORT_NAME's name_size
                // computation). The actual protection against
                // dispatching a truncated frame is `find_port_by_id`
                // rejecting unknown port ids — for non-PORT_ADD
                // events with a stale/garbage id the kernel drops
                // the message before the switch. For PORT_NAME a
                // buf->len=0 produces an absurd name_size that
                // typically fails the kmalloc and skips the strscpy.
                // None of this is bulletproof; the safe path is to
                // never produce a torn write in the first place.
                // The control message stays at the front of
                // `control_out` for retry on the next chain.
                if let Err(e) = q.add_used(mem, head, 0) {
                    tracing::warn!(head, %e, "virtio-console c_ivq add_used after torn failed");
                } else {
                    any_published = true;
                }
                break;
            }
            if let Err(e) = q.add_used(mem, head, written) {
                tracing::warn!(
                    head,
                    written,
                    %e,
                    "virtio-console c_ivq add_used failed; control message lost"
                );
                break;
            }
            any_published = true;
            // Now safe to consume from the front.
            self.control_out.pop_front();
        }
        // Fire signal_used whenever any used-ring entry was
        // published this call. `any_published` covers the data
        // path AND the add_used(0) paths (too-small / torn) so a
        // sequence of failures still kicks an irq for the guest.
        if any_published {
            self.signal_used();
        }
    }

    // ------------------------------------------------------------------
    // MMIO register dispatch
    // ------------------------------------------------------------------

    /// Handle MMIO read at `offset` within the device's MMIO region.
    pub fn mmio_read(&self, offset: u64, data: &mut [u8]) {
        // Config space lives at 0x100..0x110 in the MMIO layout per
        // virtio-v1.2 §4.2.2; `struct virtio_console_config` is 12
        // bytes (cols u16, rows u16, max_nr_ports u32, emerg_wr u32).
        const CFG_BASE: u64 = 0x100;
        const CFG_END: u64 = CFG_BASE + 12;
        if (CFG_BASE..CFG_END).contains(&offset) {
            self.config_read(offset - CFG_BASE, data);
            return;
        }
        if data.len() != 4 {
            for b in data.iter_mut() {
                *b = 0xff;
            }
            return;
        }
        let val: u32 = match offset as u32 {
            VIRTIO_MMIO_MAGIC_VALUE => MMIO_MAGIC,
            VIRTIO_MMIO_VERSION => MMIO_VERSION,
            VIRTIO_MMIO_DEVICE_ID => VIRTIO_ID_CONSOLE,
            VIRTIO_MMIO_VENDOR_ID => VENDOR_ID,
            VIRTIO_MMIO_DEVICE_FEATURES => {
                let page = self.device_features_sel;
                if page == 0 {
                    self.device_features() as u32
                } else if page == 1 {
                    (self.device_features() >> 32) as u32
                } else {
                    0
                }
            }
            VIRTIO_MMIO_QUEUE_NUM_MAX => self
                .selected_queue()
                .map(|i| self.queues[i].max_size() as u32)
                .unwrap_or(0),
            VIRTIO_MMIO_QUEUE_READY => self
                .selected_queue()
                .map(|i| self.queues[i].ready() as u32)
                .unwrap_or(0),
            VIRTIO_MMIO_INTERRUPT_STATUS => self.interrupt_status,
            VIRTIO_MMIO_STATUS => self.device_status,
            VIRTIO_MMIO_CONFIG_GENERATION => self.config_generation,
            _ => 0,
        };
        tracing::debug!(offset, val, "virtio-console mmio_read");
        data.copy_from_slice(&val.to_le_bytes());
    }

    /// Read from device config space. Layout matches
    /// `struct virtio_console_config` byte-for-byte:
    ///   off 0  u16 cols       — only valid with F_SIZE
    ///   off 2  u16 rows       — only valid with F_SIZE
    ///   off 4  u32 max_nr_ports — only valid with F_MULTIPORT
    ///   off 8  u32 emerg_wr   — only valid with F_EMERG_WRITE
    /// We advertise F_MULTIPORT only, so cols/rows/emerg_wr return 0
    /// (the kernel reads them via `virtio_cread_feature` which is a
    /// no-op when the feature bit is not set).
    fn config_read(&self, offset: u64, data: &mut [u8]) {
        let mut cfg = [0u8; 12];
        // max_nr_ports at offset 4, LE u32.
        cfg[4..8].copy_from_slice(&NUM_PORTS.to_le_bytes());
        let start = offset as usize;
        let end = start.saturating_add(data.len());
        if end > cfg.len() {
            // Out-of-range read: fill with 0xff so a misbehaving guest
            // gets explicit garbage rather than stale stack values.
            for b in data.iter_mut() {
                *b = 0xff;
            }
            return;
        }
        data.copy_from_slice(&cfg[start..end]);
    }

    /// Handle MMIO write at `offset` within the device's MMIO region.
    pub fn mmio_write(&mut self, offset: u64, data: &[u8]) {
        // Config space at 0x100..0x10c is read-only for this device
        // (we do not advertise F_EMERG_WRITE). Drop writes silently
        // and log so a guest stack writing into config surfaces.
        if (0x100..0x10c).contains(&offset) {
            tracing::warn!(
                offset,
                len = data.len(),
                "virtio-console: guest write to read-only config space ignored"
            );
            return;
        }
        if data.len() != 4 {
            return;
        }
        let val = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        tracing::debug!(offset, val, "virtio-console mmio_write");
        match offset as u32 {
            VIRTIO_MMIO_DEVICE_FEATURES_SEL => self.device_features_sel = val,
            VIRTIO_MMIO_DRIVER_FEATURES_SEL => self.driver_features_sel = val,
            VIRTIO_MMIO_DRIVER_FEATURES => {
                if !self.features_write_allowed() {
                    return;
                }
                let page = self.driver_features_sel;
                if page == 0 {
                    self.driver_features =
                        (self.driver_features & 0xFFFF_FFFF_0000_0000) | val as u64;
                } else if page == 1 {
                    self.driver_features =
                        (self.driver_features & 0x0000_0000_FFFF_FFFF) | ((val as u64) << 32);
                }
            }
            VIRTIO_MMIO_QUEUE_SEL => self.queue_select = val,
            VIRTIO_MMIO_QUEUE_NUM if self.queue_config_allowed() => {
                if let Some(i) = self.selected_queue() {
                    self.queues[i].set_size(val as u16);
                }
            }
            VIRTIO_MMIO_QUEUE_READY if self.queue_config_allowed() => {
                if let Some(i) = self.selected_queue() {
                    self.queues[i].set_ready(val == 1);
                }
            }
            VIRTIO_MMIO_QUEUE_NOTIFY => {
                let idx = val as usize;
                match idx {
                    C_IVQ => self.drain_control_in(),
                    C_OVQ => self.process_control_tx(),
                    _ => match queue_to_port(idx) {
                        // Guest published RX buffers; drain any
                        // pending host→guest bytes (port-0 console
                        // wakes, port-1 TLV replies, port-2 stats
                        // requests) into the freshly available
                        // descriptors. When no bytes are pending the
                        // drain is a quick no-op — the guest
                        // publishes empty buffers as flow control
                        // even when the host has nothing to send.
                        Some((port_id, false)) => self.drain_pending_rx(port_id),
                        Some((port_id, true)) => {
                            let _ = self.process_tx(port_id);
                        }
                        None => {
                            tracing::debug!(idx, "virtio-console: notify on unused queue");
                        }
                    },
                }
            }
            VIRTIO_MMIO_INTERRUPT_ACK => {
                self.interrupt_status &= !val;
            }
            VIRTIO_MMIO_STATUS => {
                if val == 0 {
                    self.reset();
                } else {
                    self.set_status(val);
                }
            }
            VIRTIO_MMIO_QUEUE_DESC_LOW if self.queue_config_allowed() => {
                if let Some(i) = self.selected_queue() {
                    self.queues[i].set_desc_table_address(Some(val), None);
                }
            }
            VIRTIO_MMIO_QUEUE_DESC_HIGH if self.queue_config_allowed() => {
                if let Some(i) = self.selected_queue() {
                    self.queues[i].set_desc_table_address(None, Some(val));
                }
            }
            VIRTIO_MMIO_QUEUE_AVAIL_LOW if self.queue_config_allowed() => {
                if let Some(i) = self.selected_queue() {
                    self.queues[i].set_avail_ring_address(Some(val), None);
                }
            }
            VIRTIO_MMIO_QUEUE_AVAIL_HIGH if self.queue_config_allowed() => {
                if let Some(i) = self.selected_queue() {
                    self.queues[i].set_avail_ring_address(None, Some(val));
                }
            }
            VIRTIO_MMIO_QUEUE_USED_LOW if self.queue_config_allowed() => {
                if let Some(i) = self.selected_queue() {
                    self.queues[i].set_used_ring_address(Some(val), None);
                }
            }
            VIRTIO_MMIO_QUEUE_USED_HIGH if self.queue_config_allowed() => {
                if let Some(i) = self.selected_queue() {
                    self.queues[i].set_used_ring_address(None, Some(val));
                }
            }
            _ => {}
        }
    }

    /// Validate and apply a status transition per virtio-v1.2 §3.1.1.
    /// The driver must not clear bits. Each phase requires the previous
    /// phase's bits to be set. Invalid transitions are ignored.
    fn set_status(&mut self, val: u32) {
        let old = self.device_status;
        // Driver must not clear bits (except via reset, which writes 0).
        if val & self.device_status != self.device_status {
            tracing::warn!(
                old,
                val,
                "virtio-console set_status: rejected (clears bits) — \
                 virtio-v1.2 §3.1.1 status bits are monotone within a \
                 driver session. Hostile-guest FSM violations surface at \
                 this log level (matches virtio-blk's set_status \
                 rejection-warn pattern)."
            );
            return;
        }
        let new_bits = val & !self.device_status;
        // FAILED (virtio-v1.2 §2.1.1 bit 0x80) is the driver's
        // "I give up" signal — `virtio_add_status(dev,
        // VIRTIO_CONFIG_S_FAILED)` is the kernel's exit path on probe
        // failure (drivers/virtio/virtio.c:363, 570, 606, 643). It
        // can land at any FSM state, alone or alongside other bits
        // already set (the call is `dev->config->set_status(dev,
        // get_status() | FAILED)` so `val` is `current_status |
        // 0x80`). Accept the bit and store; do not gate on the FSM
        // ladder. Reject only if FAILED appears together with
        // unrecognised new bits — those are protocol violations
        // unrelated to the legitimate FAILED signal. Log warn so a
        // guest probe failure surfaces visibly rather than silently
        // wedging the device behind a rejection.
        if new_bits == VIRTIO_CONFIG_S_FAILED {
            self.device_status = val;
            tracing::warn!(
                old,
                new = val,
                "virtio-console set_status: guest set FAILED status \
                 (virtio-v1.2 §2.1.1 bit 0x80 — driver gave up on \
                 device probe). Stored without further FSM advance."
            );
            return;
        }
        let valid = match new_bits {
            VIRTIO_CONFIG_S_ACKNOWLEDGE => self.device_status == 0,
            VIRTIO_CONFIG_S_DRIVER => self.device_status == S_ACK,
            VIRTIO_CONFIG_S_FEATURES_OK => self.device_status == S_DRV,
            VIRTIO_CONFIG_S_DRIVER_OK => self.device_status == S_FEAT,
            _ => false,
        };
        if valid {
            self.device_status = val;
            tracing::debug!(old, new = val, "virtio-console set_status: accepted");
            // Once DRIVER_OK lands, drain any host bytes that arrived
            // during initialization.
            if new_bits == VIRTIO_CONFIG_S_DRIVER_OK {
                // F_MULTIPORT must be negotiated for the multiport
                // control protocol to function. Without it the
                // kernel's virtcons_probe (drivers/char/virtio_console.c)
                // takes the legacy single-console path
                // (`add_port(portdev, 0)` then no DEVICE_READY) and
                // never sends a control-queue handshake. Port 1
                // bytes would then sit in `ports[1].tx_buf` /
                // `ports[1].pending_rx` unconsumed, and the c_ivq /
                // c_ovq queues would never receive driver buffers.
                // Surface this loudly so the failure is visible
                // rather than wedging behind a silent fallback.
                if self.driver_features & (1u64 << (VIRTIO_CONSOLE_F_MULTIPORT as u64)) == 0 {
                    tracing::warn!(
                        driver_features = self.driver_features,
                        "virtio-console set_status DRIVER_OK: \
                         F_MULTIPORT (bit 1) not negotiated by \
                         driver. Multiport control protocol will \
                         not run; port 1 bulk channel will not \
                         function. Verify the guest kernel has \
                         CONFIG_VIRTIO_CONSOLE enabled and that \
                         feature negotiation completed before \
                         DRIVER_OK."
                    );
                }
                self.drain_pending_rx(0);
                self.drain_control_in();
            }
        } else {
            tracing::warn!(
                old,
                val,
                "virtio-console set_status: rejected (invalid transition) — \
                 virtio-v1.2 §3.1.1 ordering: ACK → DRIVER → FEATURES_OK \
                 → DRIVER_OK, one bit at a time. Hostile-guest FSM \
                 violations surface at this log level."
            );
        }
    }

    fn reset(&mut self) {
        // Drain pending guest TX chains BEFORE clearing device state.
        // The kernel's `kernel_restart` → `device_shutdown` path on
        // `force_reboot()` writes 0 to `VIRTIO_MMIO_STATUS` (the
        // caller of this reset) after the userspace `send_exit` /
        // `send_lifecycle` writes have published descriptor chains
        // into the avail ring but before the host has observed the
        // matching QUEUE_NOTIFY MMIO. Once we run the field-clear
        // sequence below, the queue's `desc_table` / `avail_ring`
        // GuestAddresses are reset to virtio-queue 0.17.0's
        // `DEFAULT_*_ADDR` sentinels (see `Queue::reset` in the
        // virtio-queue crate), making the original avail-ring GPA
        // unreachable. Walk every port's TX queue here while their
        // GPAs and the DRIVER_OK / F_MULTIPORT gates are still
        // valid; the bytes land in each port's `tx_buf`, which we
        // deliberately do NOT clear below so `collect_results`'s
        // `final_drain` can still surface them.
        //
        // `drain_tx_into_capture_buf` is the side-effect-free variant
        // of `process_tx` — no `signal_used` (the rebooting guest
        // will not consume the IRQ), no `tx_evt.write` (the freeze
        // coordinator must NOT be woken to drain the bulk port here:
        // this reset call site already holds the device's outer
        // mutex, and the coord's TOKEN_TX handler races to acquire
        // the same mutex when `tx_evt` fires — under burst workloads
        // where 16 workers / 8 vCPUs flood the port, that wake-and-
        // block pattern stalls coord teardown for the duration of
        // every reset-time drain). The captured bytes are surfaced
        // post-reset via `final_drain` instead.
        for port_id in 0..NUM_PORTS as usize {
            self.drain_tx_into_capture_buf(port_id);
        }
        self.device_status = 0;
        self.interrupt_status = 0;
        self.queue_select = 0;
        self.device_features_sel = 0;
        self.driver_features_sel = 0;
        self.driver_features = 0;
        // Per-port `tx_buf` is a host-side capture buffer; clearing
        // it here would discard bytes the guest already published
        // but the host hasn't drained yet. `pending_rx` is queue-side
        // (host-prepared bytes waiting to travel back into the
        // guest's RX ring) and has no post-reset consumer, so it is
        // still cleared. `opened` and `readied` reset to false so
        // a fresh probe re-runs the per-port handshake.
        for port in &mut self.ports {
            port.pending_rx.clear();
            port.opened = false;
            port.readied = false;
        }
        self.control_out.clear();
        self.device_ready = false;
        for q in &mut self.queues {
            q.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::io::AsRawFd;
    use virtio_bindings::bindings::virtio_ring::{VRING_DESC_F_NEXT, VRING_DESC_F_WRITE};
    use virtio_bindings::virtio_mmio::VIRTIO_MMIO_INT_CONFIG;
    use virtio_queue::desc::{RawDescriptor, split::Descriptor as SplitDescriptor};
    use virtio_queue::mock::MockSplitQueue;
    use vm_memory::{Address, GuestAddress};

    fn read_reg(dev: &VirtioConsole, offset: u32) -> u32 {
        let mut buf = [0u8; 4];
        dev.mmio_read(offset as u64, &mut buf);
        u32::from_le_bytes(buf)
    }

    fn write_reg(dev: &mut VirtioConsole, offset: u32, val: u32) {
        dev.mmio_write(offset as u64, &val.to_le_bytes());
    }

    /// Drive the device through the full init sequence up to DRIVER_OK,
    /// negotiating both VIRTIO_F_VERSION_1 and F_MULTIPORT (the kernel
    /// always negotiates every feature bit it supports that we
    /// advertise).
    fn init_device(dev: &mut VirtioConsole) {
        write_reg(dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(dev, VIRTIO_MMIO_STATUS, S_DRV);
        // F_MULTIPORT (bit 1) lives in the low 32-bit page.
        write_reg(dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
        write_reg(
            dev,
            VIRTIO_MMIO_DRIVER_FEATURES,
            1 << VIRTIO_CONSOLE_F_MULTIPORT,
        );
        // VIRTIO_F_VERSION_1 (bit 32) lives in the high 32-bit page.
        write_reg(dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
        write_reg(
            dev,
            VIRTIO_MMIO_DRIVER_FEATURES,
            1 << (VIRTIO_F_VERSION_1 - 32),
        );
        write_reg(dev, VIRTIO_MMIO_STATUS, S_FEAT);
        write_reg(dev, VIRTIO_MMIO_STATUS, S_OK);
    }

    /// Build a guest memory map sized to host the chain test rings
    /// (one or two queues at low GPAs) plus per-test data buffers
    /// (high GPAs). 2 MiB is generous enough that buffers up to
    /// 64 KiB (the oversize-truncation test) fit above the queue
    /// regions without collision. Queue rings live near GPA 0; the
    /// per-test descriptor data addresses sit at 0x10000+ which
    /// leaves the entire first 64 KiB free for ring placement.
    fn make_chain_test_mem() -> GuestMemoryMmap {
        GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), 2 << 20)])
            .expect("create chain test guest mem")
    }

    /// Walk the FSM up to FEATURES_OK, configure `queue_idx` to
    /// point at `mock`'s desc/avail/used rings with size 16, mark
    /// the queue ready, then transition to DRIVER_OK. Mirrors
    /// `virtio_blk::testing::wire_device_to_mock` adapted to the
    /// 6-queue console layout.
    ///
    /// `init_device` walks the FSM all the way to DRIVER_OK with no
    /// queue config — the post-DRIVER_OK gate then rejects any
    /// queue address writes (`queue_config_allowed` requires
    /// `S_FEAT && !DRIVER_OK`). For chain-level tests we must
    /// configure the queue BEFORE DRIVER_OK lands, so this helper
    /// stops at S_FEAT to install the queue addresses then advances
    /// to S_OK once.
    ///
    /// The MockSplitQueue size (16) matches the queue size we tell
    /// the device via `QUEUE_NUM`; the mock's ring layout
    /// (desc table → avail → used) is what the device's
    /// `pop_descriptor_chain` walks once it sees the
    /// driver-published avail.idx.
    fn wire_console_queue_to_mock(
        dev: &mut VirtioConsole,
        mock: &MockSplitQueue<GuestMemoryMmap>,
        queue_idx: u32,
    ) {
        write_reg(dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(dev, VIRTIO_MMIO_STATUS, S_DRV);
        // F_MULTIPORT (bit 1) low half + VIRTIO_F_VERSION_1 (bit 32)
        // high half — same negotiation as `init_device`.
        write_reg(dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
        write_reg(
            dev,
            VIRTIO_MMIO_DRIVER_FEATURES,
            1 << VIRTIO_CONSOLE_F_MULTIPORT,
        );
        write_reg(dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
        write_reg(
            dev,
            VIRTIO_MMIO_DRIVER_FEATURES,
            1 << (VIRTIO_F_VERSION_1 - 32),
        );
        write_reg(dev, VIRTIO_MMIO_STATUS, S_FEAT);

        write_reg(dev, VIRTIO_MMIO_QUEUE_SEL, queue_idx);
        write_reg(dev, VIRTIO_MMIO_QUEUE_NUM, 16);
        let desc = mock.desc_table_addr().0;
        let avail = mock.avail_addr().0;
        let used = mock.used_addr().0;
        write_reg(dev, VIRTIO_MMIO_QUEUE_DESC_LOW, desc as u32);
        write_reg(dev, VIRTIO_MMIO_QUEUE_DESC_HIGH, (desc >> 32) as u32);
        write_reg(dev, VIRTIO_MMIO_QUEUE_AVAIL_LOW, avail as u32);
        write_reg(dev, VIRTIO_MMIO_QUEUE_AVAIL_HIGH, (avail >> 32) as u32);
        write_reg(dev, VIRTIO_MMIO_QUEUE_USED_LOW, used as u32);
        write_reg(dev, VIRTIO_MMIO_QUEUE_USED_HIGH, (used >> 32) as u32);
        write_reg(dev, VIRTIO_MMIO_QUEUE_READY, 1);
        write_reg(dev, VIRTIO_MMIO_STATUS, S_OK);
        // Sanity: a regression in feature negotiation that wedged
        // the FSM at FEATURES_OK would otherwise produce confusing
        // "process_tx sees an empty queue" failures from every
        // chain test downstream.
        assert_eq!(
            dev.device_status, S_OK,
            "wire_console_queue_to_mock: FSM did not reach DRIVER_OK \
             (got {:#x}) — feature negotiation likely regressed",
            dev.device_status,
        );
        assert!(
            dev.queues[queue_idx as usize].ready(),
            "wire_console_queue_to_mock: queue {queue_idx} did not \
             become ready after QUEUE_READY=1",
        );
    }

    #[test]
    fn magic_version_device_id() {
        let dev = VirtioConsole::new();
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_MAGIC_VALUE), 0x7472_6976);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_VERSION), 2);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_DEVICE_ID), VIRTIO_ID_CONSOLE);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_VENDOR_ID), 0);
    }

    #[test]
    fn device_features_advertises_multiport_and_v1() {
        let mut dev = VirtioConsole::new();
        write_reg(&mut dev, VIRTIO_MMIO_DEVICE_FEATURES_SEL, 0);
        let lo = read_reg(&dev, VIRTIO_MMIO_DEVICE_FEATURES);
        write_reg(&mut dev, VIRTIO_MMIO_DEVICE_FEATURES_SEL, 1);
        let hi = read_reg(&dev, VIRTIO_MMIO_DEVICE_FEATURES);
        let features = (hi as u64) << 32 | lo as u64;
        assert_ne!(features & (1 << VIRTIO_F_VERSION_1), 0);
        assert_ne!(features & (1u64 << (VIRTIO_CONSOLE_F_MULTIPORT as u64)), 0);
    }

    #[test]
    fn config_space_max_nr_ports_at_offset_4() {
        let dev = VirtioConsole::new();
        // Read max_nr_ports (u32 at offset 4 inside config space, which
        // starts at 0x100).
        let mut buf = [0u8; 4];
        dev.mmio_read(0x100 + 4, &mut buf);
        assert_eq!(u32::from_le_bytes(buf), NUM_PORTS);
    }

    #[test]
    fn config_space_cols_rows_zero_without_f_size() {
        let dev = VirtioConsole::new();
        let mut buf = [0u8; 4];
        // cols (u16) + rows (u16) at offsets 0..4 inside config.
        dev.mmio_read(0x100, &mut buf);
        assert_eq!(buf, [0, 0, 0, 0]);
    }

    #[test]
    fn queue_num_max_for_six_queues() {
        let mut dev = VirtioConsole::new();
        for q in 0..NUM_QUEUES {
            write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, q as u32);
            assert_eq!(
                read_reg(&dev, VIRTIO_MMIO_QUEUE_NUM_MAX),
                QUEUE_MAX_SIZE as u32,
                "queue {q} should be available",
            );
        }
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, NUM_QUEUES as u32);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_QUEUE_NUM_MAX), 0);
    }

    #[test]
    fn queue_ready_requires_features_ok() {
        let mut dev = VirtioConsole::new();
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 0);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_READY, 1);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_QUEUE_READY), 0);

        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
        write_reg(
            &mut dev,
            VIRTIO_MMIO_DRIVER_FEATURES,
            1 << (VIRTIO_F_VERSION_1 - 32),
        );
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_FEAT);

        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 0);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_READY, 1);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_QUEUE_READY), 1);
    }

    #[test]
    fn status_state_machine() {
        let mut dev = VirtioConsole::new();
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_STATUS), 0);

        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        assert_eq!(dev.device_status, S_ACK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
        assert_eq!(dev.device_status, S_DRV);

        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);
        assert_eq!(
            dev.device_status, S_DRV,
            "skip FEATURES_OK must be rejected"
        );

        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        assert_eq!(
            dev.device_status, S_DRV,
            "clearing DRIVER bit must be rejected"
        );
    }

    #[test]
    fn status_reset_via_zero() {
        let mut dev = VirtioConsole::new();
        init_device(&mut dev);
        assert_eq!(dev.device_status, S_OK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, 0);
        assert_eq!(dev.device_status, 0);
    }

    #[test]
    fn interrupt_status_and_ack() {
        let mut dev = VirtioConsole::new();
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS), 0);
        dev.interrupt_status = VIRTIO_MMIO_INT_VRING;
        assert_eq!(
            read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS),
            VIRTIO_MMIO_INT_VRING
        );
    }

    #[test]
    fn interrupt_ack_clears_bits() {
        let mut dev = VirtioConsole::new();
        dev.interrupt_status = VIRTIO_MMIO_INT_VRING | VIRTIO_MMIO_INT_CONFIG;
        write_reg(&mut dev, VIRTIO_MMIO_INTERRUPT_ACK, VIRTIO_MMIO_INT_VRING);
        assert_eq!(
            read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS),
            VIRTIO_MMIO_INT_CONFIG
        );
    }

    #[test]
    fn non_4byte_read_returns_ff() {
        let dev = VirtioConsole::new();
        let mut buf = [0u8; 2];
        dev.mmio_read(0, &mut buf);
        assert_eq!(buf, [0xff, 0xff]);
    }

    #[test]
    fn non_4byte_write_ignored() {
        let mut dev = VirtioConsole::new();
        dev.mmio_write(VIRTIO_MMIO_STATUS as u64, &[0x01, 0x00]);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_STATUS), 0);
    }

    #[test]
    fn driver_features_gated_by_status() {
        let mut dev = VirtioConsole::new();
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES, 0xDEAD);
        assert_eq!(dev.driver_features, 0);

        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES, 0xDEAD_BEEF);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES, 0xCAFE_BABE);
        assert_eq!(dev.driver_features, 0xCAFE_BABE_DEAD_BEEF);
    }

    #[test]
    fn features_rejected_after_features_ok() {
        let mut dev = VirtioConsole::new();
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
        write_reg(
            &mut dev,
            VIRTIO_MMIO_DRIVER_FEATURES,
            1 << (VIRTIO_F_VERSION_1 - 32),
        );
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_FEAT);

        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES, 0xFFFF);
        assert_eq!(dev.driver_features & 0xFFFF_FFFF, 0);
    }

    #[test]
    fn queue_desc_addr_requires_features_ok() {
        let mut dev = VirtioConsole::new();
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 0);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_LOW, 0x1000);
        assert_ne!(dev.queues[0].desc_table(), 0x1000);

        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
        write_reg(
            &mut dev,
            VIRTIO_MMIO_DRIVER_FEATURES,
            1 << (VIRTIO_F_VERSION_1 - 32),
        );
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_FEAT);

        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 0);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_LOW, 0x1000);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_HIGH, 0);
        assert_eq!(dev.queues[0].desc_table(), 0x1000);
    }

    #[test]
    fn reset_clears_all_state() {
        let mut dev = VirtioConsole::new();
        init_device(&mut dev);
        dev.interrupt_status = 0xFF;
        dev.ports[0].tx_buf.extend(b"leftover0".iter().copied());
        // VecDeque has no `extend_from_slice`; copy the byte iterator.
        dev.ports[1].tx_buf.extend(b"leftover1".iter().copied());
        dev.ports[2].tx_buf.extend(b"leftover2".iter().copied());
        // Pending-RX buffers are queue-side scratch — populate
        // them so the test verifies reset clears them.
        dev.ports[0].pending_rx.extend(b"pending0".iter().copied());
        dev.ports[1].pending_rx.extend(b"pending1".iter().copied());
        dev.ports[2].pending_rx.extend(b"pending2".iter().copied());
        dev.ports[0].opened = true;
        dev.ports[1].opened = true;
        dev.ports[2].opened = true;
        dev.device_ready = true;
        dev.ports[0].readied = true;
        dev.ports[1].readied = true;
        dev.ports[2].readied = true;
        dev.control_out
            .push_back(ControlOut::Cmd(VirtioConsoleControl {
                id: 0,
                event: VIRTIO_CONSOLE_PORT_ADD,
                value: 0,
            }));

        write_reg(&mut dev, VIRTIO_MMIO_STATUS, 0);

        assert_eq!(read_reg(&dev, VIRTIO_MMIO_STATUS), 0);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS), 0);
        assert_eq!(dev.queue_select, 0);
        assert_eq!(dev.device_features_sel, 0);
        assert_eq!(dev.driver_features, 0);
        // Per-port `tx_buf` survives reset by design — these are
        // host-side capture buffers `collect_results` (and the
        // stats client's drainer) consume after the guest's reboot
        // path has clobbered device state. The pre-existing
        // leftover bytes must remain observable.
        assert_eq!(
            dev.ports[0].tx_buf.iter().copied().collect::<Vec<u8>>(),
            b"leftover0",
            "ports[0].tx_buf must survive reset (host-side capture buffer)"
        );
        assert_eq!(
            dev.ports[1].tx_buf.iter().copied().collect::<Vec<u8>>(),
            b"leftover1",
            "ports[1].tx_buf must survive reset (host-side capture buffer)"
        );
        assert_eq!(
            dev.ports[2].tx_buf.iter().copied().collect::<Vec<u8>>(),
            b"leftover2",
            "ports[2].tx_buf must survive reset (host-side capture buffer)"
        );
        // Per-port `pending_rx` is cleared at reset because the
        // guest's RX queues are about to be reset and the bytes
        // would otherwise sit in the deque indefinitely.
        assert!(dev.ports[0].pending_rx.is_empty());
        assert!(dev.ports[1].pending_rx.is_empty());
        assert!(dev.ports[2].pending_rx.is_empty());
        for p in &dev.ports {
            assert!(!p.opened);
        }
        assert!(!dev.device_ready);
        for p in &dev.ports {
            assert!(!p.readied);
        }
        assert!(dev.control_out.is_empty());
    }

    #[test]
    fn config_generation_initially_zero() {
        let dev = VirtioConsole::new();
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_CONFIG_GENERATION), 0);
    }

    #[test]
    fn new_creates_eventfds() {
        let dev = VirtioConsole::new();
        assert!(dev.irq_evt().as_raw_fd() >= 0);
        assert!(dev.tx_evt().as_raw_fd() >= 0);
        assert!(dev.stats_tx_evt().as_raw_fd() >= 0);
        // All three eventfds must be distinct so wakes on one fd
        // do not bleed into another. A regression that aliased
        // `tx_evt` and `stats_tx_evt` would let port-1 bulk
        // traffic wake the stats drainer (and vice versa),
        // defeating the orthogonality the dedicated stats path
        // depends on.
        let irq = dev.irq_evt().as_raw_fd();
        let tx = dev.tx_evt().as_raw_fd();
        let stats = dev.stats_tx_evt().as_raw_fd();
        assert_ne!(irq, tx);
        assert_ne!(irq, stats);
        assert_ne!(tx, stats);
    }

    #[test]
    fn output_empty_initially() {
        let dev = VirtioConsole::new();
        assert!(dev.output().is_empty());
    }

    #[test]
    fn drain_output_empty() {
        let mut dev = VirtioConsole::new();
        assert!(dev.drain_output().is_empty());
    }

    #[test]
    fn drain_bulk_empty() {
        let mut dev = VirtioConsole::new();
        assert!(dev.drain_bulk().is_empty());
    }

    #[test]
    fn set_mem_stores_reference() {
        let mut dev = VirtioConsole::new();
        assert!(dev.mem.is_none());
        let mem = GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), 0x10000)]).unwrap();
        dev.set_mem(mem);
        assert!(dev.mem.is_some());
    }

    #[test]
    fn queue_input_no_mem_no_panic() {
        let mut dev = VirtioConsole::new();
        dev.queue_input(b"hello");
    }

    #[test]
    fn unknown_register_returns_zero() {
        let dev = VirtioConsole::new();
        assert_eq!(read_reg(&dev, 0x300), 0);
    }

    #[test]
    fn unknown_register_write_ignored() {
        let mut dev = VirtioConsole::new();
        write_reg(&mut dev, 0x300, 0xDEAD);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_STATUS), 0);
    }

    #[test]
    fn invalid_queue_select_returns_zero() {
        let mut dev = VirtioConsole::new();
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 99);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_QUEUE_NUM_MAX), 0);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_QUEUE_READY), 0);
    }

    #[test]
    fn signal_used_sets_interrupt_and_writes_eventfd() {
        let mut dev = VirtioConsole::new();
        assert_eq!(dev.interrupt_status, 0);
        dev.signal_used();
        assert_ne!(dev.interrupt_status & VIRTIO_MMIO_INT_VRING, 0);
        let val = dev.irq_evt.read().unwrap();
        assert!(val > 0);
    }

    #[test]
    fn features_page_2_returns_zero() {
        let mut dev = VirtioConsole::new();
        write_reg(&mut dev, VIRTIO_MMIO_DEVICE_FEATURES_SEL, 2);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_DEVICE_FEATURES), 0);
    }

    #[test]
    fn tx_evt_silent_on_empty_process_tx() {
        let mut dev = VirtioConsole::new();
        let mem = GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), 0x10000)]).unwrap();
        dev.set_mem(mem);
        let _ = dev.process_tx(0);
        let _ = dev.process_tx(1);
        assert!(dev.tx_evt.read().is_err());
    }

    #[test]
    fn status_skip_acknowledge_rejected() {
        let mut dev = VirtioConsole::new();
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, VIRTIO_CONFIG_S_DRIVER);
        assert_eq!(dev.device_status, 0);
    }

    #[test]
    fn queue_config_rejected_after_driver_ok() {
        let mut dev = VirtioConsole::new();
        init_device(&mut dev);
        assert_eq!(dev.device_status, S_OK);

        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 0);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NUM, 64);
        assert_eq!(dev.queues[0].size(), QUEUE_MAX_SIZE);
    }

    #[test]
    fn config_space_write_ignored() {
        let mut dev = VirtioConsole::new();
        // Try to write to max_nr_ports (offset 4 in config space).
        let buf = 99u32.to_le_bytes();
        dev.mmio_write(0x104, &buf);
        // Read back — must still be NUM_PORTS.
        let mut out = [0u8; 4];
        dev.mmio_read(0x104, &mut out);
        assert_eq!(u32::from_le_bytes(out), NUM_PORTS);
    }

    #[test]
    fn handle_device_ready_enqueues_port_adds() {
        let mut dev = VirtioConsole::new();
        dev.handle_control_event(VirtioConsoleControl {
            id: 0xFFFF_FFFF,
            event: VIRTIO_CONSOLE_DEVICE_READY,
            value: 1,
        });
        assert!(dev.device_ready);
        assert_eq!(dev.control_out.len(), NUM_PORTS as usize);
        for (i, msg) in dev.control_out.iter().enumerate() {
            match msg {
                ControlOut::Cmd(c) => {
                    let id = c.id;
                    let event = c.event;
                    let value = c.value;
                    assert_eq!(id, i as u32);
                    assert_eq!(event, VIRTIO_CONSOLE_PORT_ADD);
                    // PORT_ADD value=1 matches QEMU
                    // (hw/char/virtio-serial-bus.c
                    // `send_control_event(... PORT_ADD, 1)`).
                    assert_eq!(value, 1);
                }
                _ => panic!("unexpected msg variant"),
            }
        }
    }

    #[test]
    fn handle_port_ready_port0_console_announce() {
        let mut dev = VirtioConsole::new();
        dev.handle_control_event(VirtioConsoleControl {
            id: 0,
            event: VIRTIO_CONSOLE_PORT_READY,
            value: 1,
        });
        // Order: CONSOLE_PORT, PORT_NAME, PORT_OPEN. PORT_NAME between
        // CONSOLE_PORT and PORT_OPEN keeps the sysfs `name` attribute
        // (`/sys/class/virtio-ports/vport0p0/name`) created before any
        // userspace `/dev/hvc0` open races with udev symlink creation.
        assert_eq!(dev.control_out.len(), 3);
        let m0 = &dev.control_out[0];
        let m1 = &dev.control_out[1];
        let m2 = &dev.control_out[2];
        match m0 {
            ControlOut::Cmd(c) => {
                let event = c.event;
                let value = c.value;
                assert_eq!(event, VIRTIO_CONSOLE_CONSOLE_PORT);
                assert_eq!(value, 1);
            }
            _ => panic!("expected Cmd"),
        }
        match m1 {
            ControlOut::Name { id, name } => {
                assert_eq!(*id, 0);
                assert_eq!(*name, PORT0_NAME);
            }
            _ => panic!("expected Name"),
        }
        match m2 {
            ControlOut::Cmd(c) => {
                let event = c.event;
                let value = c.value;
                assert_eq!(event, VIRTIO_CONSOLE_PORT_OPEN);
                assert_eq!(value, 1);
            }
            _ => panic!("expected Cmd"),
        }
    }

    #[test]
    fn handle_port_ready_port1_name_then_open() {
        let mut dev = VirtioConsole::new();
        dev.handle_control_event(VirtioConsoleControl {
            id: 1,
            event: VIRTIO_CONSOLE_PORT_READY,
            value: 1,
        });
        assert_eq!(dev.control_out.len(), 2);
        // Order matches QEMU's PORT_READY handler
        // (hw/char/virtio-serial-bus.c): PORT_NAME first, PORT_OPEN
        // second. Sending OPEN before NAME would race udev symlink
        // creation against the userspace open of /dev/vport0p1.
        match &dev.control_out[0] {
            ControlOut::Name { id, name } => {
                assert_eq!(*id, 1);
                assert_eq!(*name, PORT1_NAME);
            }
            _ => panic!("expected Name"),
        }
        match &dev.control_out[1] {
            ControlOut::Cmd(c) => {
                let event = c.event;
                let value = c.value;
                let id = c.id;
                assert_eq!(id, 1);
                assert_eq!(event, VIRTIO_CONSOLE_PORT_OPEN);
                assert_eq!(value, 1);
            }
            _ => panic!("expected Cmd"),
        }
    }

    #[test]
    fn handle_port_open_tracks_state() {
        let mut dev = VirtioConsole::new();
        assert!(!dev.ports[1].opened);
        dev.handle_control_event(VirtioConsoleControl {
            id: 1,
            event: VIRTIO_CONSOLE_PORT_OPEN,
            value: 1,
        });
        assert!(dev.ports[1].opened);
        dev.handle_control_event(VirtioConsoleControl {
            id: 1,
            event: VIRTIO_CONSOLE_PORT_OPEN,
            value: 0,
        });
        assert!(!dev.ports[1].opened);
    }

    #[test]
    fn handle_port_ready_unknown_port_ignored() {
        let mut dev = VirtioConsole::new();
        dev.handle_control_event(VirtioConsoleControl {
            id: 99,
            event: VIRTIO_CONSOLE_PORT_READY,
            value: 1,
        });
        assert!(dev.control_out.is_empty());
    }

    #[test]
    fn vc_control_size_is_eight_bytes() {
        assert_eq!(VC_CONTROL_SIZE, 8);
        assert_eq!(std::mem::size_of::<VirtioConsoleControl>(), 8);
    }

    #[test]
    fn vc_control_round_trip_through_bytes() {
        let c = VirtioConsoleControl {
            id: 0xDEAD_BEEF,
            event: VIRTIO_CONSOLE_PORT_OPEN,
            value: 1,
        };
        let bytes = c.as_bytes();
        let back = VirtioConsoleControl::read_from_bytes(bytes).unwrap();
        let id = back.id;
        let event = back.event;
        let value = back.value;
        assert_eq!(id, 0xDEAD_BEEF);
        assert_eq!(event, VIRTIO_CONSOLE_PORT_OPEN);
        assert_eq!(value, 1);
    }

    // ----------------------------------------------------------------
    // Chain-level MockSplitQueue tests for the port 1 TX path.
    //
    // These exercise `process_tx_into(PORT1_TXQ, ...)` end-to-end via
    // the real virtio-queue descriptor walker — MockSplitQueue plants
    // a chain in guest memory, MMIO QUEUE_NOTIFY fires, and the
    // device's process_port1_tx walks the chain, copies device-readable
    // descriptor data into `port1_tx_buf`, and add_useds the chain.
    //
    // The handler-level tests above bypass the queue walker and only
    // pin MMIO/FSM/control surface; these tests pin the production
    // bulk TX path the host-side `bulk_drain` consumer depends on.
    // Chain-level MockSplitQueue coverage is mandatory because the
    // chain-parsing logic is the highest-risk code on the bulk path:
    // a hostile or malformed chain that handler-level tests can't
    // construct (multi-segment, mixed-direction, length-cap edges)
    // is exactly what the production `process_tx_into` walker has
    // to reject without panicking. Without these chain tests every
    // chain-shape regression has to wait for an end-to-end VM run
    // to surface.
    // ----------------------------------------------------------------

    /// Single-descriptor TX chain on port 1: one device-readable
    /// segment with a known byte pattern lands verbatim in
    /// `port1_tx_buf`. Pins the simplest happy-path: chain pop →
    /// non-write_only branch → `mem.read_slice` → append to
    /// `port1_tx_buf` → add_used → `signal_used`. `drain_bulk()`
    /// returns the bytes to confirm the routing accumulator was the
    /// port-1 buffer (not port 0).
    #[test]
    fn port1_tx_single_descriptor_lands_in_port1_buf() {
        let mut dev = VirtioConsole::new();
        let mem = make_chain_test_mem();
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
        let data_addr = GuestAddress(0x10000);
        let payload = b"port1 single descriptor TX";
        mem.write_slice(payload, data_addr).expect("plant payload");
        // Single device-readable descriptor (flags=0 → guest wrote
        // the bytes; device reads them).
        let descs = [RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            payload.len() as u32,
            0,
            0,
        ))];
        mock.build_desc_chain(&descs).expect("build chain");
        dev.set_mem(mem.clone());
        wire_console_queue_to_mock(&mut dev, &mock, PORT1_TXQ as u32);

        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT1_TXQ as u32);

        // Bytes must have landed in port1_tx_buf, observable via
        // drain_bulk.
        let drained = dev.drain_bulk();
        assert_eq!(
            drained,
            payload.to_vec(),
            "port 1 TX must deliver the descriptor's bytes to drain_bulk verbatim",
        );
        // Port 0 buffer must be untouched — the routing match on
        // queue_idx (PORT0_TXQ vs PORT1_TXQ in process_tx_into)
        // must have steered to port1_tx_buf.
        assert!(
            dev.drain_output().is_empty(),
            "port 0 TX buffer must remain empty when only port 1 was notified",
        );
        // Used ring reflects exactly one completion. UsedRing.idx
        // sits at used_addr+2 after the 2-byte flags field
        // (virtio-v1.2 §2.7.8).
        let used_idx: u16 = mem
            .read_obj(mock.used_addr().checked_add(2).unwrap())
            .expect("read used.idx");
        assert_eq!(used_idx, 1, "exactly one used-ring entry expected");
        // signal_used must have set INT_VRING in interrupt_status
        // and written the irq eventfd.
        assert_ne!(
            dev.interrupt_status & VIRTIO_MMIO_INT_VRING,
            0,
            "INT_VRING must be set after a successful TX drain",
        );
        let irq_count = dev.irq_evt.read().expect("irq_evt was written");
        assert!(
            irq_count > 0,
            "irq_evt counter must be non-zero after signal_used",
        );
    }

    /// Multi-descriptor TX chain: four 4 KiB device-readable
    /// segments concatenate in `port1_tx_buf`, in chain order. Pins
    /// that the descriptor-walker honours the chain order and that
    /// the per-descriptor `dst.resize + read_slice` append logic
    /// preserves byte boundaries. The kernel's virtio-console driver
    /// sends PAGE_SIZE chunks as multi-segment chains under high
    /// volume — losing chain order or dropping the second-N segments
    /// would corrupt the host-side TLV stream.
    #[test]
    fn port1_tx_multi_descriptor_chain_concatenates() {
        let mut dev = VirtioConsole::new();
        let mem = make_chain_test_mem();
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
        // PAGE_SIZE matches the kernel's typical TX descriptor size
        // (drivers/char/virtio_console.c uses PAGE_SIZE buffers).
        // Four distinct GPAs spaced 8 KiB apart so each descriptor
        // points at a non-overlapping region.
        const PAGE: u32 = 4096;
        let segs: [(GuestAddress, u8); 4] = [
            (GuestAddress(0x10000), 0xA1),
            (GuestAddress(0x12000), 0xA2),
            (GuestAddress(0x14000), 0xA3),
            (GuestAddress(0x16000), 0xA4),
        ];
        for (addr, fill) in &segs {
            let buf = vec![*fill; PAGE as usize];
            mem.write_slice(&buf, *addr).expect("plant segment");
        }
        let descs = [
            RawDescriptor::from(SplitDescriptor::new(segs[0].0.0, PAGE, 0, 0)),
            RawDescriptor::from(SplitDescriptor::new(segs[1].0.0, PAGE, 0, 0)),
            RawDescriptor::from(SplitDescriptor::new(segs[2].0.0, PAGE, 0, 0)),
            RawDescriptor::from(SplitDescriptor::new(segs[3].0.0, PAGE, 0, 0)),
        ];
        mock.build_desc_chain(&descs).expect("build chain");
        dev.set_mem(mem.clone());
        wire_console_queue_to_mock(&mut dev, &mock, PORT1_TXQ as u32);

        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT1_TXQ as u32);

        // Expected: 4 × PAGE concatenation, segment N filled with
        // 0xA(N+1).
        let drained = dev.drain_bulk();
        assert_eq!(
            drained.len(),
            4 * PAGE as usize,
            "drain_bulk length must equal sum of segment lengths",
        );
        for (i, (_, fill)) in segs.iter().enumerate() {
            let start = i * PAGE as usize;
            let end = start + PAGE as usize;
            assert!(
                drained[start..end].iter().all(|&b| b == *fill),
                "segment {i} must hold fill {fill:#x} verbatim — chain order \
                 or per-descriptor append regressed",
            );
        }
        // Used ring reflects exactly one completion — the chain is
        // a single descriptor chain (head + 3 next links), not 4
        // separate chains.
        let used_idx: u16 = mem
            .read_obj(mock.used_addr().checked_add(2).unwrap())
            .expect("read used.idx");
        assert_eq!(
            used_idx, 1,
            "one chain → one used-ring entry, regardless of segment count",
        );
    }

    /// Oversize descriptor truncates to `TX_DESC_MAX`. A guest that
    /// publishes `len > 32 KiB` (max u32 = 4 GiB worst case) is
    /// hostile or buggy; the device caps each descriptor at 32 KiB
    /// to bound the per-chain heap allocation. Pins the
    /// `(desc.len() as usize).min(TX_DESC_MAX)` clamp at
    /// `process_tx_into` line ~379 and the `dst.resize(start +
    /// dlen, 0)` allocation. Without the cap, a single bogus
    /// descriptor could trigger a multi-GiB Vec allocation.
    #[test]
    fn port1_tx_oversize_descriptor_truncates_to_tx_desc_max() {
        let mut dev = VirtioConsole::new();
        let mem = make_chain_test_mem();
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
        let data_addr = GuestAddress(0x10000);
        // Plant a buffer twice TX_DESC_MAX so the truncation can be
        // distinguished from "no data planted past the cap." First
        // half filled with 0x55, second half with 0x99 — the cap
        // should keep only the 0x55 region.
        const OVERSIZE: usize = TX_DESC_MAX * 2;
        let mut payload = vec![0x55u8; TX_DESC_MAX];
        payload.extend_from_slice(&vec![0x99u8; TX_DESC_MAX]);
        assert_eq!(payload.len(), OVERSIZE);
        mem.write_slice(&payload, data_addr).expect("plant payload");
        // One descriptor with len=OVERSIZE > TX_DESC_MAX. The
        // device must cap reads at TX_DESC_MAX.
        let descs = [RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            OVERSIZE as u32,
            0,
            0,
        ))];
        mock.build_desc_chain(&descs).expect("build chain");
        dev.set_mem(mem.clone());
        wire_console_queue_to_mock(&mut dev, &mock, PORT1_TXQ as u32);

        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT1_TXQ as u32);

        let drained = dev.drain_bulk();
        // Length pinned to TX_DESC_MAX, not OVERSIZE.
        assert_eq!(
            drained.len(),
            TX_DESC_MAX,
            "oversize descriptor (len > TX_DESC_MAX) must truncate to TX_DESC_MAX",
        );
        // The bytes that DID land must be the first TX_DESC_MAX of
        // the planted buffer (0x55 fill); a regression that read
        // past the cap would surface 0x99 bytes here.
        assert!(
            drained.iter().all(|&b| b == 0x55),
            "truncated bytes must be the FIRST TX_DESC_MAX bytes \
             of the descriptor (0x55), not anything past the cap",
        );
    }

    /// Port 1 TX with DRIVER_OK NOT set: the device must drop the
    /// notify silently. Pins the spec gate at the head of
    /// `process_tx_into`: virtio-v1.2 §3.1.1 forbids the device
    /// from accessing virtqueue memory before DRIVER_OK because
    /// queue addresses written during FEATURES_OK are not yet
    /// committed by the driver. A regression that lifted the gate
    /// would let the device walk a queue the guest hasn't fully
    /// validated.
    ///
    /// Setup walks the FSM only to FEATURES_OK + queue ready (NOT
    /// DRIVER_OK), plants a chain, fires the notify, and verifies
    /// (a) port1_tx_buf untouched (drain_bulk empty), (b) used.idx
    /// untouched (no add_used), (c) interrupt_status untouched (no
    /// signal_used), (d) irq_evt unwritten.
    #[test]
    fn port1_tx_rejected_without_driver_ok() {
        let mut dev = VirtioConsole::new();
        let mem = make_chain_test_mem();
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
        let data_addr = GuestAddress(0x10000);
        let payload = b"this must NOT reach port1_tx_buf";
        mem.write_slice(payload, data_addr).expect("plant payload");
        let descs = [RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            payload.len() as u32,
            0,
            0,
        ))];
        mock.build_desc_chain(&descs).expect("build chain");
        dev.set_mem(mem.clone());

        // Walk FSM only up to S_FEAT — STOP before S_OK. Configure
        // the queue and mark it ready (which is allowed in the
        // S_FEAT..S_OK window per `queue_config_allowed`). The
        // process_tx_into gate should still drop the notify because
        // VIRTIO_CONFIG_S_DRIVER_OK is not in device_status.
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
        write_reg(
            &mut dev,
            VIRTIO_MMIO_DRIVER_FEATURES,
            1 << VIRTIO_CONSOLE_F_MULTIPORT,
        );
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
        write_reg(
            &mut dev,
            VIRTIO_MMIO_DRIVER_FEATURES,
            1 << (VIRTIO_F_VERSION_1 - 32),
        );
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_FEAT);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, PORT1_TXQ as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NUM, 16);
        let desc = mock.desc_table_addr().0;
        let avail = mock.avail_addr().0;
        let used = mock.used_addr().0;
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_LOW, desc as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_HIGH, (desc >> 32) as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_AVAIL_LOW, avail as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_AVAIL_HIGH, (avail >> 32) as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_USED_LOW, used as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_USED_HIGH, (used >> 32) as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_READY, 1);
        // Precondition: device_status reflects FEATURES_OK but NOT
        // DRIVER_OK; the queue is ready.
        assert_eq!(
            dev.device_status & VIRTIO_CONFIG_S_DRIVER_OK,
            0,
            "precondition: DRIVER_OK must NOT be set",
        );
        assert!(
            dev.queues[PORT1_TXQ].ready(),
            "precondition: port 1 TX queue must be ready (the gate \
             we are testing is the DRIVER_OK gate, not a not-ready \
             gate)",
        );

        // Fire the notify. The DRIVER_OK gate at the top of
        // process_tx_into must early-return with no observable side
        // effects.
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT1_TXQ as u32);

        // No bytes must have landed in port1_tx_buf.
        assert!(
            dev.drain_bulk().is_empty(),
            "port1_tx_buf must remain empty — DRIVER_OK gate must \
             reject pre-DRIVER_OK notify",
        );
        // used.idx must remain 0 — no add_used.
        let used_idx: u16 = mem
            .read_obj(mock.used_addr().checked_add(2).unwrap())
            .expect("read used.idx");
        assert_eq!(
            used_idx, 0,
            "used.idx must be 0 — DRIVER_OK gate must skip add_used",
        );
        // interrupt_status must remain 0 — no signal_used.
        assert_eq!(
            dev.interrupt_status, 0,
            "interrupt_status must be 0 — DRIVER_OK gate must skip signal_used",
        );
        // irq_evt counter must be drained / never written. A
        // non-blocking read on an unwritten EFD_NONBLOCK eventfd
        // returns WouldBlock.
        match dev.irq_evt.read() {
            Ok(n) => panic!("irq_evt must NOT have been written, but read returned {n}"),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(e) => panic!("unexpected irq_evt read error: {e}"),
        }
    }

    /// Port 0 vs port 1 TX routing: bytes from port 0 land in
    /// port0_tx_buf and bytes from port 1 land in port1_tx_buf.
    /// Pins the queue_idx → buffer dispatch in process_tx_into:
    /// `match queue_idx { PORT0_TXQ => &mut self.port0_tx_buf,
    /// PORT1_TXQ => &mut self.port1_tx_buf, ... }`. A regression
    /// that swapped the buffers would corrupt the host-side
    /// stdout stream with TLV bytes (or vice versa) — neither
    /// surface would parse correctly.
    ///
    /// Two MockSplitQueues are used at distinct GPAs because each
    /// queue needs its own desc/avail/used rings. Port 0 mock at
    /// GPA 0x0; port 1 mock at GPA 0x1000 (4 KiB above port 0,
    /// which keeps a comfortable margin: a queue of size 16 uses
    /// roughly 16*16 + 4 + 16*2 + 2 + 16*8 + 6 ≈ 432 bytes, well
    /// under one 4 KiB page).
    #[test]
    fn port0_tx_vs_port1_tx_routes_to_correct_buffer() {
        let mut dev = VirtioConsole::new();
        let mem = make_chain_test_mem();
        let mock0 = MockSplitQueue::create(&mem, GuestAddress(0x0), 16);
        let mock1 = MockSplitQueue::create(&mem, GuestAddress(0x1000), 16);
        let port0_data_addr = GuestAddress(0x10000);
        let port1_data_addr = GuestAddress(0x20000);
        let port0_payload = b"port0 console bytes";
        let port1_payload = b"port1 bulk TLV bytes";
        mem.write_slice(port0_payload, port0_data_addr)
            .expect("plant port0 payload");
        mem.write_slice(port1_payload, port1_data_addr)
            .expect("plant port1 payload");

        let port0_descs = [RawDescriptor::from(SplitDescriptor::new(
            port0_data_addr.0,
            port0_payload.len() as u32,
            0,
            0,
        ))];
        let port1_descs = [RawDescriptor::from(SplitDescriptor::new(
            port1_data_addr.0,
            port1_payload.len() as u32,
            0,
            0,
        ))];
        mock0
            .build_desc_chain(&port0_descs)
            .expect("build port0 chain");
        mock1
            .build_desc_chain(&port1_descs)
            .expect("build port1 chain");
        dev.set_mem(mem.clone());

        // Walk FSM up to FEATURES_OK, configure BOTH queues, then
        // transition to DRIVER_OK. We can't reuse
        // `wire_console_queue_to_mock` for two queues (it wraps a
        // single FSM walk); inline the multi-queue version here.
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
        write_reg(
            &mut dev,
            VIRTIO_MMIO_DRIVER_FEATURES,
            1 << VIRTIO_CONSOLE_F_MULTIPORT,
        );
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
        write_reg(
            &mut dev,
            VIRTIO_MMIO_DRIVER_FEATURES,
            1 << (VIRTIO_F_VERSION_1 - 32),
        );
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_FEAT);
        // Configure port 0 TX queue (idx 1).
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, PORT0_TXQ as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NUM, 16);
        let d0 = mock0.desc_table_addr().0;
        let a0 = mock0.avail_addr().0;
        let u0 = mock0.used_addr().0;
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_LOW, d0 as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_HIGH, (d0 >> 32) as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_AVAIL_LOW, a0 as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_AVAIL_HIGH, (a0 >> 32) as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_USED_LOW, u0 as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_USED_HIGH, (u0 >> 32) as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_READY, 1);
        // Configure port 1 TX queue (idx 5).
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, PORT1_TXQ as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NUM, 16);
        let d1 = mock1.desc_table_addr().0;
        let a1 = mock1.avail_addr().0;
        let u1 = mock1.used_addr().0;
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_LOW, d1 as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_HIGH, (d1 >> 32) as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_AVAIL_LOW, a1 as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_AVAIL_HIGH, (a1 >> 32) as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_USED_LOW, u1 as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_USED_HIGH, (u1 >> 32) as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_READY, 1);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);
        assert_eq!(
            dev.device_status, S_OK,
            "FSM did not reach DRIVER_OK after both queues configured",
        );

        // Notify port 0 then port 1; observe the routing.
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT0_TXQ as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT1_TXQ as u32);

        // Port 0 bytes must land in port0_tx_buf, observable via
        // drain_output (which returns port0_tx_buf).
        let port0_drained = dev.drain_output();
        assert_eq!(
            port0_drained,
            port0_payload.to_vec(),
            "port 0 TX bytes must route to port0_tx_buf — drain_output \
             returns port0 bytes verbatim",
        );
        // Port 1 bytes must land in port1_tx_buf, observable via
        // drain_bulk (which returns port1_tx_buf).
        let port1_drained = dev.drain_bulk();
        assert_eq!(
            port1_drained,
            port1_payload.to_vec(),
            "port 1 TX bytes must route to port1_tx_buf — drain_bulk \
             returns port1 bytes verbatim",
        );
        // Each port's used.idx reflects exactly one completion.
        let port0_used_idx: u16 = mem
            .read_obj(mock0.used_addr().checked_add(2).unwrap())
            .expect("read port0 used.idx");
        let port1_used_idx: u16 = mem
            .read_obj(mock1.used_addr().checked_add(2).unwrap())
            .expect("read port1 used.idx");
        assert_eq!(
            port0_used_idx, 1,
            "port 0 used.idx must reflect 1 completion"
        );
        assert_eq!(
            port1_used_idx, 1,
            "port 1 used.idx must reflect 1 completion"
        );
    }

    /// Port 1 TX `process_tx_into` honours the per-call cumulative
    /// byte cap (`TX_PER_CALL_MAX`). With 9 chains of `TX_DESC_MAX`
    /// bytes each (= 9 × 32 KiB = 288 KiB), the first 8 chains
    /// drain to `port1_tx_buf` (cumulative 256 KiB hits the cap)
    /// and the 9th chain remains unconsumed in the avail ring for
    /// the next notify. Pins the per-call drain cap that bounds
    /// the per-vCPU MMIO-handler latency budget against a hostile
    /// guest publishing thousands of valid PAGE_SIZE chains.
    ///
    /// Verifies:
    /// (a) `port1_tx_buf` length == 8 × TX_DESC_MAX (cap drain only);
    /// (b) `used.idx` == 8 (only the 8 drained chains were add_used);
    /// (c) the 9th descriptor's bytes are NOT in `port1_tx_buf`
    ///     (each chain's payload uses a distinguishable fill byte
    ///     so a regression that drained past the cap surfaces here);
    /// (d) a second `QUEUE_NOTIFY` drains the remaining 9th chain
    ///     (cumulative 32 KiB on the second call, well under cap).
    #[test]
    fn port1_tx_per_call_cap_partial_drain() {
        let mut dev = VirtioConsole::new();
        let mem = make_chain_test_mem();
        // Queue size 16: enough headroom for 9 standalone chains
        // (the test's published count) without wrap-around.
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
        // Buffer count: 9 chains × TX_DESC_MAX bytes = 288 KiB
        // total. With TX_PER_CALL_MAX = 256 KiB, 8 chains hit the
        // cap (cumulative 256 KiB after the 8th chain triggers the
        // break check). The 9th must stay unconsumed.
        const N_CHAINS: usize = 9;
        // Per-chain GPA spacing: each buffer is TX_DESC_MAX (32
        // KiB) bytes, placed at 0x10000, 0x18000, 0x20000, ...
        // (8 KiB stride between starts is too tight; use
        // TX_DESC_MAX stride so buffers don't overlap). Total
        // memory usage: 0x10000 + 9 × TX_DESC_MAX = 64 KiB + 288
        // KiB = 352 KiB, well under the 2 MiB test memory map.
        let mut descs: Vec<RawDescriptor> = Vec::with_capacity(N_CHAINS);
        for i in 0..N_CHAINS {
            let buf_addr = GuestAddress(0x10000 + (i as u64) * (TX_DESC_MAX as u64));
            // Fill byte i+1 (1..=9) so a regression that drained
            // past the cap surfaces an unexpected fill byte. Using
            // (i+1) instead of i keeps every chain's fill byte
            // non-zero so a buffer of zeros from an unrelated
            // region can't pass for chain bytes.
            let fill = (i + 1) as u8;
            let payload = vec![fill; TX_DESC_MAX];
            mem.write_slice(&payload, buf_addr)
                .expect("plant per-chain payload");
            descs.push(RawDescriptor::from(SplitDescriptor::new(
                buf_addr.0,
                TX_DESC_MAX as u32,
                0, // flags: no NEXT — each desc is its own chain
                0,
            )));
        }
        // `add_desc_chains` writes descs at indices 0..N and
        // increments avail.idx for each chain head (every desc
        // here is a standalone chain because none carry the NEXT
        // flag).
        mock.add_desc_chains(&descs, 0)
            .expect("publish 9 standalone chains");
        dev.set_mem(mem.clone());
        wire_console_queue_to_mock(&mut dev, &mock, PORT1_TXQ as u32);

        // First notify: cap drains 8 chains (256 KiB), 9th left.
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT1_TXQ as u32);

        let drained_first = dev.drain_bulk();
        assert_eq!(
            drained_first.len(),
            TX_PER_CALL_MAX,
            "first notify must drain exactly TX_PER_CALL_MAX bytes \
             (8 × TX_DESC_MAX) — the per-call cap stops popping after \
             the 8th chain"
        );
        // First 8 chains drained: bytes are 32 KiB of fill=1 then
        // 32 KiB of fill=2 ... up to 32 KiB of fill=8. The 9th
        // chain (fill=9) must NOT appear.
        for i in 0..8 {
            let start = i * TX_DESC_MAX;
            let end = start + TX_DESC_MAX;
            let expected_fill = (i + 1) as u8;
            assert!(
                drained_first[start..end]
                    .iter()
                    .all(|&b| b == expected_fill),
                "chain {i} bytes must be fill={expected_fill}; \
                 a regression that drained past the cap (or out of \
                 chain order) would surface a different byte here"
            );
        }
        assert!(
            !drained_first.contains(&9u8),
            "9th chain (fill=9) must NOT appear in the first drain \
             — the per-call cap must hold the 9th chain back"
        );
        // used.idx must reflect exactly 8 completions. The 9th
        // chain stayed in the avail ring; the device never
        // add_used'd it.
        let used_idx_first: u16 = mem
            .read_obj(mock.used_addr().checked_add(2).unwrap())
            .expect("read used.idx after first notify");
        assert_eq!(
            used_idx_first, 8,
            "used.idx must be 8 after first notify — the cap stopped \
             popping after the 8th chain so only 8 add_used calls \
             happened"
        );

        // Second notify: drains the remaining 9th chain.
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT1_TXQ as u32);

        let drained_second = dev.drain_bulk();
        assert_eq!(
            drained_second.len(),
            TX_DESC_MAX,
            "second notify must drain the remaining 9th chain \
             (TX_DESC_MAX bytes) — the cap is per-call, not per-run"
        );
        assert!(
            drained_second.iter().all(|&b| b == 9u8),
            "second drain must contain the 9th chain's bytes (fill=9)"
        );
        let used_idx_second: u16 = mem
            .read_obj(mock.used_addr().checked_add(2).unwrap())
            .expect("read used.idx after second notify");
        assert_eq!(
            used_idx_second, 9,
            "used.idx must be 9 after second notify — every chain \
             eventually drains, the cap only spreads them across \
             multiple notifies"
        );
    }

    // ----------------------------------------------------------------
    // Hostile-guest control message defenses (handle_control_event).
    //
    // The kernel's virtio-console driver
    // (drivers/char/virtio_console.c `virtcons_probe`) sends
    // DEVICE_READY exactly once and PORT_READY exactly once per port.
    // A hostile or buggy guest that re-sends either message would
    // re-enqueue PORT_ADD / CONSOLE_PORT / PORT_OPEN / PORT_NAME and
    // grow `control_out` without bound, exhausting host memory. The
    // device gates each repeat behind `device_ready` / `port_readied`
    // flags; these tests pin the gates against regressions.
    // ----------------------------------------------------------------

    /// DEVICE_READY repeats must be ignored — the second message
    /// must NOT enqueue a second batch of PORT_ADD frames. Pins the
    /// `if self.device_ready` early-return at handle_control_event
    /// (the DEVICE_READY arm). Without the gate, a guest spamming
    /// DEVICE_READY would grow `control_out` by NUM_PORTS entries
    /// per message until the host OOMs.
    #[test]
    fn handle_device_ready_repeat_ignored() {
        let mut dev = VirtioConsole::new();
        // First DEVICE_READY: enqueues NUM_PORTS PORT_ADD frames.
        dev.handle_control_event(VirtioConsoleControl {
            id: 0,
            event: VIRTIO_CONSOLE_DEVICE_READY,
            value: 1,
        });
        assert!(
            dev.device_ready,
            "device_ready must be set after first message"
        );
        let after_first = dev.control_out.len();
        assert_eq!(
            after_first, NUM_PORTS as usize,
            "first DEVICE_READY must enqueue exactly NUM_PORTS PORT_ADD frames",
        );

        // Second DEVICE_READY: the gate must reject; control_out
        // must NOT grow.
        dev.handle_control_event(VirtioConsoleControl {
            id: 0,
            event: VIRTIO_CONSOLE_DEVICE_READY,
            value: 1,
        });
        assert_eq!(
            dev.control_out.len(),
            after_first,
            "DEVICE_READY repeat must be a no-op — control_out length must \
             remain at NUM_PORTS, otherwise a hostile guest can grow it \
             unboundedly",
        );
        // device_ready stays true — the repeat does not flip it back.
        assert!(
            dev.device_ready,
            "device_ready must remain set after repeat"
        );
    }

    /// PORT_READY repeat for the same port must be ignored — the
    /// second message must NOT re-enqueue
    /// CONSOLE_PORT/PORT_NAME/PORT_OPEN. Pins the
    /// `if self.port_readied[id as usize]` early-return at
    /// handle_control_event (the PORT_READY arm). Mirrors the
    /// DEVICE_READY gate but per-port — readying port 0 twice would
    /// otherwise enqueue 6 frames (3 per call) instead of 3.
    #[test]
    fn handle_port_ready_repeat_ignored_port0() {
        let mut dev = VirtioConsole::new();
        // First PORT_READY for port 0: enqueues 3 frames
        // (CONSOLE_PORT, PORT_NAME, PORT_OPEN).
        dev.handle_control_event(VirtioConsoleControl {
            id: 0,
            event: VIRTIO_CONSOLE_PORT_READY,
            value: 1,
        });
        assert!(
            dev.ports[0].readied,
            "port_readied[0] must be set after first PORT_READY"
        );
        let after_first = dev.control_out.len();
        assert_eq!(
            after_first, 3,
            "first PORT_READY for port 0 must enqueue 3 frames \
             (CONSOLE_PORT, PORT_NAME, PORT_OPEN)",
        );

        // Second PORT_READY for port 0: the gate must reject;
        // control_out must NOT grow.
        dev.handle_control_event(VirtioConsoleControl {
            id: 0,
            event: VIRTIO_CONSOLE_PORT_READY,
            value: 1,
        });
        assert_eq!(
            dev.control_out.len(),
            after_first,
            "PORT_READY repeat for the same port must be a no-op — \
             control_out length must remain at 3, otherwise a hostile \
             guest can re-enqueue announce frames unboundedly",
        );
    }

    /// PORT_READY repeat for port 1 must be ignored — symmetric to
    /// the port-0 case but exercises the port-1 branch (PORT_NAME
    /// then PORT_OPEN, 2 frames per legitimate message). A regression
    /// that scoped the gate per port-0 only would surface here.
    #[test]
    fn handle_port_ready_repeat_ignored_port1() {
        let mut dev = VirtioConsole::new();
        dev.handle_control_event(VirtioConsoleControl {
            id: 1,
            event: VIRTIO_CONSOLE_PORT_READY,
            value: 1,
        });
        assert!(dev.ports[1].readied);
        let after_first = dev.control_out.len();
        assert_eq!(
            after_first, 2,
            "first PORT_READY for port 1 must enqueue 2 frames (PORT_NAME, PORT_OPEN)",
        );

        dev.handle_control_event(VirtioConsoleControl {
            id: 1,
            event: VIRTIO_CONSOLE_PORT_READY,
            value: 1,
        });
        assert_eq!(
            dev.control_out.len(),
            after_first,
            "PORT_READY repeat for port 1 must be a no-op",
        );
    }

    /// PORT_READY for port 0 must NOT inhibit a subsequent PORT_READY
    /// for port 1 — the gate is per-port, not global. Pins the array
    /// indexing in `port_readied[id as usize]`. A regression that
    /// used a single global flag would let only one port's announce
    /// frames go through.
    #[test]
    fn handle_port_ready_per_port_not_global() {
        let mut dev = VirtioConsole::new();
        dev.handle_control_event(VirtioConsoleControl {
            id: 0,
            event: VIRTIO_CONSOLE_PORT_READY,
            value: 1,
        });
        let after_port0 = dev.control_out.len();
        assert_eq!(after_port0, 3, "port 0 announce: 3 frames");

        dev.handle_control_event(VirtioConsoleControl {
            id: 1,
            event: VIRTIO_CONSOLE_PORT_READY,
            value: 1,
        });
        // Now both ports have enqueued: 3 (port 0) + 2 (port 1) = 5.
        assert_eq!(
            dev.control_out.len(),
            5,
            "PORT_READY for port 1 after PORT_READY for port 0 must \
             enqueue port 1's announce frames — the gate is per-port",
        );
        assert!(dev.ports[0].readied);
        assert!(dev.ports[1].readied);
    }

    /// PORT_READY with value=0 must log an error and skip the
    /// announce-frame enqueue. value=0 is the kernel's
    /// `add_port` failed signal (drivers/char/virtio_console.c
    /// `add_port` error path). Pins the early-return without
    /// setting `port_readied[id]` so a future legitimate PORT_READY
    /// (after recovery) is not blocked by the gate.
    #[test]
    fn handle_port_ready_value_zero_skipped() {
        let mut dev = VirtioConsole::new();
        dev.handle_control_event(VirtioConsoleControl {
            id: 0,
            event: VIRTIO_CONSOLE_PORT_READY,
            value: 0,
        });
        assert!(
            dev.control_out.is_empty(),
            "PORT_READY value=0 must NOT enqueue announce frames",
        );
        // port_readied[0] must remain false — the early-return for
        // value=0 happens BEFORE the gate flag is set, so a future
        // PORT_READY value=1 can still complete.
        assert!(
            !dev.ports[0].readied,
            "PORT_READY value=0 must NOT set port_readied — the kernel \
             may legitimately retry with value=1 after the host fixes \
             the underlying issue",
        );
    }

    /// PORT_READY value=0 must NOT block a subsequent PORT_READY
    /// value=1 from completing the handshake. Pins the rule that
    /// the value=0 early-return precedes the `port_readied` flag
    /// flip.
    #[test]
    fn handle_port_ready_value_zero_then_one_completes() {
        let mut dev = VirtioConsole::new();
        // value=0 first: skipped, no announce frames.
        dev.handle_control_event(VirtioConsoleControl {
            id: 0,
            event: VIRTIO_CONSOLE_PORT_READY,
            value: 0,
        });
        assert!(dev.control_out.is_empty());
        assert!(!dev.ports[0].readied);
        // value=1 next: must fire the announce as normal.
        dev.handle_control_event(VirtioConsoleControl {
            id: 0,
            event: VIRTIO_CONSOLE_PORT_READY,
            value: 1,
        });
        assert!(dev.ports[0].readied);
        assert_eq!(
            dev.control_out.len(),
            3,
            "PORT_READY value=1 after value=0 must enqueue the announce \
             — the value=0 path must not poison the per-port gate",
        );
    }

    /// PORT_READY for an unknown port id (>= NUM_PORTS) must be
    /// ignored. The existing `handle_port_ready_unknown_port_ignored`
    /// covers value=1; this pins the port id bounds check is the
    /// outer gate (rejected before the value-check or repeat-check).
    /// Verifies port_readied stays all-false and control_out is empty.
    #[test]
    fn handle_port_ready_unknown_port_state_unchanged() {
        let mut dev = VirtioConsole::new();
        dev.handle_control_event(VirtioConsoleControl {
            id: NUM_PORTS, // first invalid id (NUM_PORTS == 2)
            event: VIRTIO_CONSOLE_PORT_READY,
            value: 1,
        });
        assert!(dev.control_out.is_empty());
        for p in &dev.ports {
            assert!(
                !p.readied,
                "unknown-port PORT_READY must not flip any port readied flag",
            );
        }
    }

    /// PORT_OPEN for an unknown port id (>= NUM_PORTS) must be
    /// ignored. Pins the `if id >= NUM_PORTS` gate at the head of
    /// the PORT_OPEN arm. Without the gate, an out-of-bounds
    /// `port_opened[id as usize]` index would panic with
    /// "index out of bounds" — far worse than a tracing warning.
    #[test]
    fn handle_port_open_unknown_port_ignored() {
        let mut dev = VirtioConsole::new();
        // Try multiple invalid ids — anything >= NUM_PORTS.
        dev.handle_control_event(VirtioConsoleControl {
            id: NUM_PORTS,
            event: VIRTIO_CONSOLE_PORT_OPEN,
            value: 1,
        });
        dev.handle_control_event(VirtioConsoleControl {
            id: 0xFFFF_FFFF,
            event: VIRTIO_CONSOLE_PORT_OPEN,
            value: 1,
        });
        // No panic. Every port's `opened` stays at the initial false.
        for p in &dev.ports {
            assert!(
                !p.opened,
                "unknown-port PORT_OPEN must not flip any port opened \
                 flag — the gate prevents out-of-bounds array access on \
                 a hostile id",
            );
        }
    }

    /// Unhandled c_ovq event values must be absorbed silently — the
    /// `other` arm in handle_control_event logs a debug line and
    /// returns. The kernel today only sends DEVICE_READY, PORT_READY,
    /// and PORT_OPEN on c_ovq (drivers/char/virtio_console.c
    /// `send_control_msg` callers); a future kernel may add new
    /// event types. Pins that an unrecognised event does not panic,
    /// does not enqueue control_out, and does not flip any FSM flag.
    #[test]
    fn handle_unhandled_event_absorbed() {
        let mut dev = VirtioConsole::new();
        // Pick events that are valid wire values but the device
        // does not act on. PORT_ADD is the kernel's PORT_ADD —
        // the host emits it (PORT_ADD on c_ivq), the guest does not
        // send it back on c_ovq, so seeing it here is "the guest
        // sent something we don't handle." PORT_REMOVE / RESIZE
        // / PORT_NAME / CONSOLE_PORT round out the wire-defined
        // events that hit the `other` arm.
        let unhandled = [
            VIRTIO_CONSOLE_PORT_ADD,
            VIRTIO_CONSOLE_PORT_REMOVE,
            VIRTIO_CONSOLE_CONSOLE_PORT,
            VIRTIO_CONSOLE_RESIZE,
            VIRTIO_CONSOLE_PORT_NAME,
            // Completely synthetic event id — pins that even values
            // beyond the wire-defined set fall through cleanly.
            0xBEEF,
        ];
        for ev in unhandled {
            dev.handle_control_event(VirtioConsoleControl {
                id: 0,
                event: ev,
                value: 1,
            });
        }
        // Nothing must have been enqueued, no FSM flag must have
        // been flipped.
        assert!(
            dev.control_out.is_empty(),
            "unhandled events must NOT enqueue control_out",
        );
        assert!(
            !dev.device_ready,
            "unhandled events must NOT flip device_ready",
        );
        for p in &dev.ports {
            assert!(
                !p.opened,
                "unhandled events must NOT flip any port opened flag"
            );
            assert!(
                !p.readied,
                "unhandled events must NOT flip any port readied flag"
            );
        }
    }

    // ----------------------------------------------------------------
    // set_status monotone defense.
    //
    // virtio-v1.2 §3.1.1 requires status bits to advance monotonically
    // within a driver session — the driver MUST NOT clear bits except
    // by writing 0 (which triggers reset). A hostile or buggy guest
    // that clears bits mid-session would otherwise let the device
    // backslide through the FSM (e.g. drop FEATURES_OK while
    // queues are configured) and produce undefined behaviour.
    //
    // The handler enforces monotonicity via:
    //   `if val & self.device_status != self.device_status { reject }`
    // i.e. `val` must be a SUPERSET of `device_status`. These tests
    // pin every facet of that check.
    // ----------------------------------------------------------------

    /// Clearing ACKNOWLEDGE alone must be rejected. Driver advances
    /// to ACK, then writes 0 of all other bits while clearing ACK —
    /// the monotone gate must reject because the new value is not
    /// a superset of the current device_status.
    #[test]
    fn set_status_clear_acknowledge_rejected() {
        let mut dev = VirtioConsole::new();
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        assert_eq!(dev.device_status, S_ACK);
        // Try to clear ACK by writing DRIVER alone (no ACK bit).
        // val = DRIVER; current = ACK. val & current = 0 != ACK → reject.
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, VIRTIO_CONFIG_S_DRIVER);
        assert_eq!(
            dev.device_status, S_ACK,
            "writing a value that clears ACKNOWLEDGE must be rejected — \
             monotone gate (virtio-v1.2 §3.1.1)",
        );
    }

    /// Clearing DRIVER while keeping ACKNOWLEDGE must be rejected.
    /// Driver advances to ACK | DRIVER, then writes ACK alone — the
    /// monotone gate must reject because DRIVER would silently drop.
    #[test]
    fn set_status_clear_driver_keeps_ack_rejected() {
        let mut dev = VirtioConsole::new();
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
        assert_eq!(dev.device_status, S_DRV);
        // val = S_ACK; current = S_DRV (= ACK | DRIVER).
        // val & current = ACK != DRV → reject.
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        assert_eq!(
            dev.device_status, S_DRV,
            "writing ACK alone after advancing to DRIVER must be \
             rejected — clears DRIVER bit",
        );
    }

    /// Clearing FEATURES_OK from a fully-initialised device must be
    /// rejected. Driver advances all the way to S_OK, then tries to
    /// write S_FEAT (= ACK|DRIVER|FEATURES_OK, no DRIVER_OK) — the
    /// monotone gate rejects because DRIVER_OK would be cleared.
    #[test]
    fn set_status_clear_driver_ok_rejected() {
        let mut dev = VirtioConsole::new();
        init_device(&mut dev);
        assert_eq!(dev.device_status, S_OK);
        // val = S_FEAT; current = S_OK. S_FEAT is missing DRIVER_OK.
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_FEAT);
        assert_eq!(
            dev.device_status, S_OK,
            "writing S_FEAT after S_OK must be rejected — \
             clears DRIVER_OK",
        );
    }

    /// Writing the SAME value (no new bits, no cleared bits) must
    /// not panic and must leave device_status unchanged. This is
    /// not a hostile case but it exercises the boundary: `val ==
    /// device_status` is a superset of itself, so the monotone
    /// gate does not reject; `new_bits == 0` then falls through to
    /// the `_ => false` arm in the valid-transition match, which
    /// rejects the write. Pins that the device is idempotent
    /// against duplicate writes — the kernel does not retry, but
    /// hostile guests might.
    #[test]
    fn set_status_idempotent_same_value_no_change() {
        let mut dev = VirtioConsole::new();
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        assert_eq!(dev.device_status, S_ACK);
        // Re-write S_ACK — superset check passes (val==current),
        // but no NEW bits, so the valid-transition match falls
        // through to the catch-all rejection.
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        assert_eq!(
            dev.device_status, S_ACK,
            "re-writing the same status must leave device_status \
             unchanged — no advance, no regression",
        );
    }

    /// Setting two new bits in one write (e.g. FEATURES_OK + DRIVER_OK
    /// jumped together) must be rejected — the FSM advances one bit
    /// at a time. Pins the `match new_bits` valid-transition arm:
    /// only single-bit advances are accepted.
    #[test]
    fn set_status_two_bits_at_once_rejected() {
        let mut dev = VirtioConsole::new();
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
        // val = S_OK = ACK|DRIVER|FEATURES_OK|DRIVER_OK; current = S_DRV.
        // new_bits = FEATURES_OK | DRIVER_OK. The valid-transition
        // match has no arm for the union; falls through to false.
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);
        assert_eq!(
            dev.device_status, S_DRV,
            "advancing two FSM bits at once (FEATURES_OK + DRIVER_OK) \
             must be rejected — the FSM advances one bit at a time",
        );
    }

    /// Writing a status value with an unrecognised bit set (e.g. an
    /// undefined flag at 0x40) must be rejected. virtio-v1.2 §2.1
    /// defines bits 0x01 (ACK), 0x02 (DRIVER), 0x04 (DRIVER_OK),
    /// 0x08 (FEATURES_OK), 0x40 (NEEDS_RESET, device-set only),
    /// 0x80 (FAILED). 0x10 and 0x20 are reserved. NEEDS_RESET (0x40)
    /// is device-set only, so a guest that writes it is misbehaving;
    /// the value is also a non-canonical FSM advance. Pins that
    /// these unknown bits land in the `_ => false` valid arm and
    /// device_status remains unchanged.
    #[test]
    fn set_status_unknown_bit_rejected() {
        let mut dev = VirtioConsole::new();
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        assert_eq!(dev.device_status, S_ACK);
        // val = S_ACK | 0x10 (reserved/unknown bit). new_bits = 0x10.
        // The valid-transition match has no arm for 0x10 → reject.
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK | 0x10);
        assert_eq!(
            dev.device_status, S_ACK,
            "writing a status with an unrecognised bit (0x10) must \
             be rejected — only the defined ACK/DRIVER/FEATURES_OK/\
             DRIVER_OK/FAILED transitions are accepted",
        );
    }

    /// FAILED (bit 0x80) must be accepted on top of any FSM state.
    /// virtio-v1.2 §2.1.1 — `virtio_add_status(dev,
    /// VIRTIO_CONFIG_S_FAILED)` is the kernel's exit path on probe
    /// failure (drivers/virtio/virtio.c). Pins the FAILED early-accept
    /// branch in `set_status`. Verified at every FSM rung.
    #[test]
    fn set_status_failed_accepted_at_every_fsm_state() {
        // From device_status = 0.
        let mut dev = VirtioConsole::new();
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, VIRTIO_CONFIG_S_FAILED);
        assert_eq!(
            dev.device_status, VIRTIO_CONFIG_S_FAILED,
            "FAILED from status=0 must be accepted",
        );

        // From device_status = ACK.
        let mut dev = VirtioConsole::new();
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK | VIRTIO_CONFIG_S_FAILED);
        assert_eq!(
            dev.device_status,
            S_ACK | VIRTIO_CONFIG_S_FAILED,
            "FAILED from status=ACK must be accepted (ACK preserved)",
        );

        // From device_status = DRV.
        let mut dev = VirtioConsole::new();
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV | VIRTIO_CONFIG_S_FAILED);
        assert_eq!(
            dev.device_status,
            S_DRV | VIRTIO_CONFIG_S_FAILED,
            "FAILED from status=DRV must be accepted (DRV preserved)",
        );

        // From device_status = S_OK.
        let mut dev = VirtioConsole::new();
        init_device(&mut dev);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK | VIRTIO_CONFIG_S_FAILED);
        assert_eq!(
            dev.device_status,
            S_OK | VIRTIO_CONFIG_S_FAILED,
            "FAILED from status=S_OK must be accepted (S_OK preserved)",
        );
    }

    /// FAILED combined with a non-FAILED unrecognised new bit must
    /// be rejected — the FAILED early-accept only triggers when
    /// `new_bits == VIRTIO_CONFIG_S_FAILED` (FAILED alone, no other
    /// new bits). A guest mixing FAILED with garbage extra bits is
    /// misbehaving in a way unrelated to the legitimate FAILED signal.
    #[test]
    fn set_status_failed_plus_unknown_bit_rejected() {
        let mut dev = VirtioConsole::new();
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        // val = ACK | FAILED | 0x10. new_bits = FAILED | 0x10 ≠
        // FAILED alone, so the early-accept branch does NOT trigger;
        // the regular valid-transition match has no arm for the
        // union → reject.
        write_reg(
            &mut dev,
            VIRTIO_MMIO_STATUS,
            S_ACK | VIRTIO_CONFIG_S_FAILED | 0x10,
        );
        assert_eq!(
            dev.device_status, S_ACK,
            "FAILED combined with a non-FAILED unknown bit must be \
             rejected — the early-accept is gated on FAILED alone",
        );
    }

    // ----------------------------------------------------------------
    // mmio_read non-4-byte and config_read out-of-range defenses.
    //
    // virtio-v1.2 §4.2.2 specifies 4-byte register access for the MMIO
    // control registers. A misbehaving guest issuing 1/2/8-byte reads
    // would otherwise let the device return stale stack contents (the
    // `data` buffer is the caller's, not zero-initialised by the
    // device path). The defensive 0xff-fill makes the protocol
    // violation visible — distinct from a register that legitimately
    // reads as zero.
    // ----------------------------------------------------------------

    /// 8-byte mmio_read (above the 4-byte register width) must fill
    /// every byte with 0xff. The existing `non_4byte_read_returns_ff`
    /// covers the 2-byte case at offset 0; this extends coverage to
    /// the upper boundary so a regression that only checked
    /// `data.len() < 4` (instead of `!= 4`) would surface here.
    #[test]
    fn mmio_read_oversized_fills_0xff() {
        let dev = VirtioConsole::new();
        let mut buf = [0u8; 8];
        dev.mmio_read(VIRTIO_MMIO_VERSION as u64, &mut buf);
        assert_eq!(
            buf, [0xff; 8],
            "8-byte read must fill with 0xff — the device is 4-byte \
             register width per virtio-v1.2 §4.2.2",
        );
    }

    /// 1-byte mmio_read must fill the byte with 0xff. Pins the
    /// boundary at the low end (data.len() == 1 < 4). A regression
    /// that special-cased 0-length or skipped the fill on 1-byte
    /// reads would surface here.
    #[test]
    fn mmio_read_1byte_fills_0xff() {
        let dev = VirtioConsole::new();
        let mut buf = [0u8; 1];
        dev.mmio_read(VIRTIO_MMIO_VERSION as u64, &mut buf);
        assert_eq!(buf, [0xff], "1-byte read must fill with 0xff",);
    }

    /// 3-byte mmio_read (one short of the 4-byte register width)
    /// must fill every byte with 0xff. Pins the strict equality
    /// against `data.len() != 4` — a regression using `< 4` would
    /// reject this but the test still passes; using `> 4` would
    /// accept this and copy stale data.
    #[test]
    fn mmio_read_3byte_fills_0xff() {
        let dev = VirtioConsole::new();
        let mut buf = [0u8; 3];
        dev.mmio_read(VIRTIO_MMIO_VERSION as u64, &mut buf);
        assert_eq!(
            buf,
            [0xff, 0xff, 0xff],
            "3-byte read must fill with 0xff — the device is exactly \
             4-byte register width, not 'at least 4'",
        );
    }

    /// 4-byte mmio_read at a known register (VERSION) returns the
    /// register's actual value — pins that the 0xff-fill defense
    /// is gated on len != 4 specifically, NOT applied to legitimate
    /// 4-byte reads. Without this control test, a regression that
    /// always filled 0xff would still pass the misalignment tests.
    #[test]
    fn mmio_read_4byte_returns_register_value() {
        let dev = VirtioConsole::new();
        let mut buf = [0u8; 4];
        dev.mmio_read(VIRTIO_MMIO_VERSION as u64, &mut buf);
        assert_eq!(
            u32::from_le_bytes(buf),
            MMIO_VERSION,
            "4-byte read at VIRTIO_MMIO_VERSION must return the \
             actual register value, NOT the 0xff-fill defense",
        );
    }

    /// `config_read` out-of-range fill: a read that starts inside
    /// the 12-byte config struct but extends past byte 11 must fill
    /// with 0xff. Pins the `if end > cfg.len()` defense — without
    /// it, the `data.copy_from_slice(&cfg[start..end])` would
    /// panic with "index out of bounds." A guest that issues a
    /// 4-byte read at config offset 10 (i.e. straddling the end of
    /// the struct) is misbehaving; the device must absorb without
    /// panicking.
    #[test]
    fn config_read_out_of_range_fills_0xff() {
        let dev = VirtioConsole::new();
        let mut buf = [0u8; 4];
        // Config space starts at 0x100; offset 10 inside config
        // (i.e. mmio offset 0x10A) puts start=10, end=14 → out of
        // range (cfg.len() = 12).
        dev.mmio_read(0x100 + 10, &mut buf);
        assert_eq!(
            buf, [0xff; 4],
            "config_read past byte 11 must fill 0xff — the defense \
             prevents a panic from cfg[start..end] when end > 12",
        );
    }

    /// `config_read` at the exact end boundary (read straddles
    /// struct end by 1 byte) still triggers the 0xff fill. Pins
    /// the `>` (strict) comparison — `end == cfg.len()` is in
    /// range; `end == cfg.len() + 1` is not.
    #[test]
    fn config_read_one_byte_past_end_fills_0xff() {
        let dev = VirtioConsole::new();
        let mut buf = [0u8; 4];
        // Offset 9 inside config: start=9, end=13. cfg.len()=12,
        // so end=13 > 12 → 0xff fill.
        dev.mmio_read(0x100 + 9, &mut buf);
        assert_eq!(
            buf, [0xff; 4],
            "config_read with end one byte past struct must fill 0xff",
        );
    }

    /// `config_read` reading the LAST 4 valid bytes (offset 8..12
    /// inside config, the emerg_wr field) must return the actual
    /// data (zeros, since we don't advertise F_EMERG_WRITE), NOT the
    /// 0xff fill. Pins the strict-greater-than boundary in the
    /// reverse direction: end == 12 is allowed.
    #[test]
    fn config_read_at_exact_end_returns_data() {
        let dev = VirtioConsole::new();
        let mut buf = [0u8; 4];
        // Offset 8 inside config: start=8, end=12. cfg.len()=12,
        // so end == cfg.len() → in range, returns emerg_wr (0).
        dev.mmio_read(0x100 + 8, &mut buf);
        assert_eq!(
            buf, [0; 4],
            "read at the last 4-byte slot of the config struct \
             (emerg_wr, offset 8..12) must return actual data, NOT \
             the 0xff out-of-range fill",
        );
    }

    /// `config_read` 1-byte read at offset 0 (cols low byte) must
    /// return the actual data (0). Pins that the out-of-range
    /// defense does not over-trigger on small reads inside the
    /// struct. Without F_SIZE we never populate cols, so the byte
    /// is 0 by initialisation in `let mut cfg = [0u8; 12];`.
    #[test]
    fn config_read_1byte_inside_struct_returns_data() {
        let dev = VirtioConsole::new();
        let mut buf = [0u8; 1];
        // Offset 0 inside config (cols low byte). start=0, end=1.
        // 1 <= 12, so in range; returns cfg[0] = 0.
        dev.mmio_read(0x100, &mut buf);
        assert_eq!(
            buf,
            [0],
            "1-byte read inside config struct must return actual \
             data, not 0xff fill",
        );
    }

    // ----------------------------------------------------------------
    // Chain-level MockSplitQueue tests for `drain_port1_pending_rx`.
    //
    // These pin the host->guest port-1 RX path: snapshot reply payloads
    // the freeze coordinator queues via `queue_input_port1` get
    // delivered into guest write-only descriptors on q4 (PORT1_RXQ).
    //
    // Two concerns covered:
    //
    //   1. Deferral gates. Each early-return must hold pending bytes
    //      in `port1_pending_rx` without touching the queue, the used
    //      ring, or `interrupt_status`. A regression that lifted any
    //      gate would either:
    //        - publish a chain the guest has not committed (pre-
    //          DRIVER_OK / queue-not-ready);
    //        - publish bytes the kernel discards (port not yet
    //          opened by the guest's PORT_OPEN handshake);
    //        - walk a queue that does not exist for the negotiated
    //          features (F_MULTIPORT not negotiated).
    //
    //   2. Torn-write recovery. When a multi-descriptor write-only
    //      chain has one descriptor pointing at unmapped guest memory,
    //      `mem.write_slice` fails mid-chain. The function must:
    //        - publish the chain head with `len=0` (so the guest
    //          reclaims the descriptor; without this the head leaks
    //          from the avail ring until reset);
    //        - leave bytes in `port1_pending_rx` unchanged (the
    //          per-chain `drain(..consumed_offset)` is in the
    //          non-torn branch, so a torn chain does NOT consume
    //          bytes from the deque);
    //        - break out of the drain loop (further chains for this
    //          notify are NOT processed; the next notify retries).
    //
    // The torn-write fixture exploits write_slice's bounds check by
    // pointing one descriptor at a GPA past the 2 MiB
    // `make_chain_test_mem` map.
    // ----------------------------------------------------------------

    /// Wire q4 (PORT1_RXQ) to a MockSplitQueue and drive the FSM to
    /// DRIVER_OK with F_MULTIPORT negotiated. Variant of
    /// `wire_console_queue_to_mock` that targets q4 specifically; used
    /// by every drain_port1 test below.
    fn wire_port1_rxq_to_mock(dev: &mut VirtioConsole, mock: &MockSplitQueue<GuestMemoryMmap>) {
        wire_console_queue_to_mock(dev, mock, PORT1_RXQ as u32);
    }

    /// Mark port 1 as opened by sending PORT_OPEN(value=1) on c_ovq.
    /// This sets `port_opened[1] = true` so the gate in
    /// `drain_port1_pending_rx` lets bytes through. The same call
    /// invokes `drain_port1_pending_rx` internally on the open
    /// transition, but with no pending bytes that is a no-op — the
    /// caller pushes pending bytes AFTER opening the port (or relies
    /// on the open handler's own deferred-drain trigger).
    fn open_port1(dev: &mut VirtioConsole) {
        dev.handle_control_event(VirtioConsoleControl {
            id: 1,
            event: VIRTIO_CONSOLE_PORT_OPEN,
            value: 1,
        });
        assert!(
            dev.ports[1].opened,
            "open_port1 helper precondition: PORT_OPEN(value=1) must \
             set port_opened[1]"
        );
    }

    /// Empty pending-rx → drain is a no-op. No queue access, no
    /// add_used, no signal_used. Pins the `if pending.is_empty()`
    /// fast-exit at the head of `drain_port1_pending_rx`.
    #[test]
    fn drain_port1_pending_rx_empty_pending_is_noop() {
        let mut dev = VirtioConsole::new();
        let mem = make_chain_test_mem();
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
        dev.set_mem(mem.clone());
        wire_port1_rxq_to_mock(&mut dev, &mock);
        open_port1(&mut dev);

        // Plant a write-only chain that would be popped IF the
        // function walked the queue; the empty-pending fast-exit
        // must skip the queue entirely.
        let data_addr = GuestAddress(0x10000);
        let descs = [RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            64,
            VRING_DESC_F_WRITE as u16,
            0,
        ))];
        mock.build_desc_chain(&descs).expect("build chain");

        assert!(
            dev.ports[1].pending_rx.is_empty(),
            "precondition: port1_pending_rx must start empty"
        );
        let int_before = dev.interrupt_status;

        dev.drain_pending_rx(1);

        // No add_used → used.idx still 0.
        let used_idx: u16 = mem
            .read_obj(mock.used_addr().checked_add(2).unwrap())
            .expect("read used.idx");
        assert_eq!(
            used_idx, 0,
            "empty-pending fast-exit must not touch the queue"
        );
        assert_eq!(
            dev.interrupt_status, int_before,
            "empty-pending fast-exit must not call signal_used"
        );
    }

    /// DRIVER_OK gate: bytes pushed before DRIVER_OK stay in
    /// `port1_pending_rx`. A regression that walked the queue
    /// pre-DRIVER_OK would let the device read descriptor addresses
    /// the driver has not yet committed (virtio-v1.2 §3.1.1).
    #[test]
    fn drain_port1_pending_rx_defers_without_driver_ok() {
        let mut dev = VirtioConsole::new();
        let mem = make_chain_test_mem();
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
        let data_addr = GuestAddress(0x10000);
        let descs = [RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            64,
            VRING_DESC_F_WRITE as u16,
            0,
        ))];
        mock.build_desc_chain(&descs).expect("build chain");
        dev.set_mem(mem.clone());
        // Walk FSM up to S_FEAT only — STOP before S_OK. Configure
        // q4 and mark it ready (allowed in S_FEAT..S_OK per
        // `queue_config_allowed`). The DRIVER_OK gate at the head of
        // `drain_port1_pending_rx` must still defer.
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
        write_reg(
            &mut dev,
            VIRTIO_MMIO_DRIVER_FEATURES,
            1 << VIRTIO_CONSOLE_F_MULTIPORT,
        );
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
        write_reg(
            &mut dev,
            VIRTIO_MMIO_DRIVER_FEATURES,
            1 << (VIRTIO_F_VERSION_1 - 32),
        );
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_FEAT);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, PORT1_RXQ as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NUM, 16);
        let desc = mock.desc_table_addr().0;
        let avail = mock.avail_addr().0;
        let used = mock.used_addr().0;
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_LOW, desc as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_HIGH, (desc >> 32) as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_AVAIL_LOW, avail as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_AVAIL_HIGH, (avail >> 32) as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_USED_LOW, used as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_USED_HIGH, (used >> 32) as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_READY, 1);
        // Cannot call `open_port1` — `handle_control_event(PORT_OPEN)`
        // works regardless of FSM state but unrelated to the gate
        // under test. Set the field directly.
        dev.ports[1].opened = true;

        let payload = b"snapshot reply bytes";
        dev.ports[1].pending_rx.extend(payload.iter().copied());

        dev.drain_pending_rx(1);

        // Bytes preserved verbatim in pending_rx — no consumption.
        assert_eq!(
            dev.ports[1].pending_rx.len(),
            payload.len(),
            "DRIVER_OK gate must hold bytes in pending_rx"
        );
        // No add_used → used.idx still 0.
        let used_idx: u16 = mem
            .read_obj(mock.used_addr().checked_add(2).unwrap())
            .expect("read used.idx");
        assert_eq!(used_idx, 0, "DRIVER_OK gate must skip add_used");
    }

    /// F_MULTIPORT runtime gate: even with DRIVER_OK and the queue
    /// ready, if the driver did not negotiate F_MULTIPORT then q4
    /// should not be walked. Pins the
    /// `if !self.multiport_negotiated()` guard at line ~903.
    #[test]
    fn drain_port1_pending_rx_defers_without_multiport() {
        let mut dev = VirtioConsole::new();
        let mem = make_chain_test_mem();
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
        let data_addr = GuestAddress(0x10000);
        let descs = [RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            64,
            VRING_DESC_F_WRITE as u16,
            0,
        ))];
        mock.build_desc_chain(&descs).expect("build chain");
        dev.set_mem(mem.clone());

        // Walk FSM to DRIVER_OK negotiating ONLY VIRTIO_F_VERSION_1
        // (no F_MULTIPORT). `wire_console_queue_to_mock` always
        // negotiates both, so inline a custom version here.
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
        write_reg(
            &mut dev,
            VIRTIO_MMIO_DRIVER_FEATURES,
            1 << (VIRTIO_F_VERSION_1 - 32),
        );
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_FEAT);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, PORT1_RXQ as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NUM, 16);
        let desc = mock.desc_table_addr().0;
        let avail = mock.avail_addr().0;
        let used = mock.used_addr().0;
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_LOW, desc as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_HIGH, (desc >> 32) as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_AVAIL_LOW, avail as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_AVAIL_HIGH, (avail >> 32) as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_USED_LOW, used as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_USED_HIGH, (used >> 32) as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_READY, 1);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);
        assert_eq!(
            dev.device_status, S_OK,
            "precondition: FSM must reach DRIVER_OK"
        );
        assert!(
            !dev.multiport_negotiated(),
            "precondition: F_MULTIPORT must NOT be negotiated"
        );
        // Set port_opened[1] directly so the multiport gate is the
        // ONLY gate the test exercises (hostile guest pretend-state).
        dev.ports[1].opened = true;

        let payload = b"reply bytes that must not leak";
        dev.ports[1].pending_rx.extend(payload.iter().copied());

        dev.drain_pending_rx(1);

        assert_eq!(
            dev.ports[1].pending_rx.len(),
            payload.len(),
            "F_MULTIPORT gate must hold bytes in pending_rx"
        );
        let used_idx: u16 = mem
            .read_obj(mock.used_addr().checked_add(2).unwrap())
            .expect("read used.idx");
        assert_eq!(used_idx, 0, "F_MULTIPORT gate must skip add_used");
    }

    /// `port_opened[1]` gate: with DRIVER_OK + F_MULTIPORT but BEFORE
    /// the guest has sent `PORT_OPEN(id=1, value=1)` on c_ovq, port 1
    /// has no userspace reader. The kernel's port-1 buffer-pool
    /// allocation only completes after PORT_OPEN; pushing bytes
    /// through descriptors that exist before the open lets the
    /// kernel discard them with no userspace consumer (per
    /// drivers/char/virtio_console.c `port_fops_open`). Pins the
    /// `if !self.port_opened[1]` guard at line ~911.
    ///
    /// After the guest opens the port via PORT_OPEN, the deferred
    /// drain runs (the open transition itself triggers it at line
    /// ~1268); the bytes must then land in the queue.
    #[test]
    fn drain_port1_pending_rx_defers_until_port_open() {
        let mut dev = VirtioConsole::new();
        let mem = make_chain_test_mem();
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
        let data_addr = GuestAddress(0x10000);
        let payload = b"deferred snapshot reply";
        let descs = [RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            64,
            VRING_DESC_F_WRITE as u16,
            0,
        ))];
        mock.build_desc_chain(&descs).expect("build chain");
        dev.set_mem(mem.clone());
        wire_port1_rxq_to_mock(&mut dev, &mock);

        // Port 1 not yet opened — the `port_opened[1]` gate must
        // defer.
        assert!(
            !dev.ports[1].opened,
            "precondition: port_opened[1] must be false"
        );

        dev.ports[1].pending_rx.extend(payload.iter().copied());
        dev.drain_pending_rx(1);

        // Bytes still pending; queue untouched.
        assert_eq!(
            dev.ports[1].pending_rx.len(),
            payload.len(),
            "port_opened[1] gate must defer when guest has not opened port 1"
        );
        let used_idx_before: u16 = mem
            .read_obj(mock.used_addr().checked_add(2).unwrap())
            .expect("read used.idx before open");
        assert_eq!(used_idx_before, 0, "port_opened[1] gate must skip add_used");

        // Now drive PORT_OPEN(id=1, value=1). The handler at line
        // ~1268 calls drain_port1_pending_rx on the closed→open
        // transition, which now must drain.
        open_port1(&mut dev);

        assert!(
            dev.ports[1].pending_rx.is_empty(),
            "after PORT_OPEN, deferred bytes must drain"
        );
        let used_idx_after: u16 = mem
            .read_obj(mock.used_addr().checked_add(2).unwrap())
            .expect("read used.idx after open");
        assert_eq!(
            used_idx_after, 1,
            "after PORT_OPEN, the deferred chain must add_used"
        );
        // Bytes landed in the guest descriptor buffer.
        let mut readback = vec![0u8; payload.len()];
        mem.read_slice(&mut readback, data_addr)
            .expect("read back delivered payload");
        assert_eq!(
            readback, payload,
            "delivered bytes must match the queued payload verbatim"
        );
    }

    /// No-mem gate: a device whose `mem` field is None (the freeze
    /// coordinator may push reply bytes during the brief window
    /// before `set_mem` lands) must not crash and must hold the
    /// bytes for retry. Pins the `match self.mem.as_ref() { ... None
    /// => return }` arm at line ~918.
    #[test]
    fn drain_port1_pending_rx_defers_without_mem() {
        let mut dev = VirtioConsole::new();
        // Walk FSM to DRIVER_OK with F_MULTIPORT but DO NOT call
        // set_mem. `init_device` reaches S_OK without touching mem.
        init_device(&mut dev);
        assert!(dev.mem.is_none(), "precondition: mem must be None");
        dev.ports[1].opened = true;

        let payload = b"reply bytes pre-set_mem";
        dev.ports[1].pending_rx.extend(payload.iter().copied());

        // No panic, no crash.
        dev.drain_pending_rx(1);

        assert_eq!(
            dev.ports[1].pending_rx.len(),
            payload.len(),
            "no-mem gate must hold bytes in pending_rx"
        );
    }

    /// Queue-not-ready gate: bytes stay pending if PORT1_RXQ has not
    /// been marked ready by the driver. Pins the
    /// `!self.queues[PORT1_RXQ].ready()` guard at line ~928. The
    /// driver writes QUEUE_READY=1 only after the desc/avail/used
    /// addresses are committed; reading the queue before that point
    /// would walk uninitialized state.
    #[test]
    fn drain_port1_pending_rx_defers_when_queue_not_ready() {
        let mut dev = VirtioConsole::new();
        let mem = make_chain_test_mem();
        dev.set_mem(mem);
        // Walk FSM to DRIVER_OK with F_MULTIPORT, but DO NOT
        // configure or ready q4. `init_device` reaches DRIVER_OK
        // without configuring any queue.
        init_device(&mut dev);
        assert!(
            !dev.queues[PORT1_RXQ].ready(),
            "precondition: q4 must NOT be ready"
        );
        dev.ports[1].opened = true;

        let payload = b"reply bytes before queue ready";
        dev.ports[1].pending_rx.extend(payload.iter().copied());

        dev.drain_pending_rx(1);

        assert_eq!(
            dev.ports[1].pending_rx.len(),
            payload.len(),
            "queue-not-ready gate must hold bytes in pending_rx"
        );
    }

    /// Single-descriptor write-only chain: happy-path baseline for
    /// the torn-write tests below. Pins that a normal drain delivers
    /// the payload to the descriptor buffer, drains
    /// `port1_pending_rx`, advances `used.idx`, and signals the
    /// guest via INT_VRING + irq_evt.
    #[test]
    fn drain_port1_pending_rx_single_descriptor_happy_path() {
        let mut dev = VirtioConsole::new();
        let mem = make_chain_test_mem();
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
        let data_addr = GuestAddress(0x10000);
        let payload = b"snapshot reply payload bytes";
        let descs = [RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            payload.len() as u32,
            VRING_DESC_F_WRITE as u16,
            0,
        ))];
        mock.build_desc_chain(&descs).expect("build chain");
        dev.set_mem(mem.clone());
        wire_port1_rxq_to_mock(&mut dev, &mock);
        open_port1(&mut dev);

        dev.ports[1].pending_rx.extend(payload.iter().copied());
        dev.drain_pending_rx(1);

        // pending_rx fully drained.
        assert!(
            dev.ports[1].pending_rx.is_empty(),
            "happy-path drain must consume all pending bytes"
        );
        // Bytes landed in the guest descriptor.
        let mut readback = vec![0u8; payload.len()];
        mem.read_slice(&mut readback, data_addr)
            .expect("read back delivered payload");
        assert_eq!(
            readback, payload,
            "delivered bytes must equal the queued payload"
        );
        // Used ring reflects exactly one completion.
        let used_idx: u16 = mem
            .read_obj(mock.used_addr().checked_add(2).unwrap())
            .expect("read used.idx");
        assert_eq!(used_idx, 1, "happy-path drain must add_used exactly once");
        // signal_used set INT_VRING.
        assert_ne!(
            dev.interrupt_status & VIRTIO_MMIO_INT_VRING,
            0,
            "INT_VRING must be set after a non-zero drain"
        );
        let irq_count = dev.irq_evt.read().expect("irq_evt was written");
        assert!(
            irq_count > 0,
            "irq_evt counter must be non-zero after signal_used"
        );
    }

    /// Multi-descriptor torn-write recovery: a chain with two
    /// write-only descriptors where the second points at unmapped
    /// guest memory (GPA past `make_chain_test_mem`'s 2 MiB cap).
    /// `mem.write_slice` fails on the second descriptor, triggering
    /// the torn-write branch.
    ///
    /// Pins the four invariants of the torn-write recovery (lines
    /// ~983-997):
    /// (a) `chain_torn = true` triggers `q.add_used(mem, head, 0)`
    ///     — used.idx advances to 1, but the published `len` is 0;
    /// (b) bytes stay in `port1_pending_rx` (the
    ///     `drain(..consumed_offset)` is in the non-torn branch, so
    ///     a torn chain does NOT consume bytes);
    /// (c) the drain loop breaks (no further chains processed for
    ///     this notify even if more are available);
    /// (d) `total_written` stays 0 for the torn chain — combined
    ///     with the absence of any prior successful chain in this
    ///     test, signal_used is NOT called and INT_VRING stays 0.
    ///
    /// Without `add_used(0)` the chain head would leak from the
    /// avail ring until reset (the c_ivq, port-0 RX, and port-1 RX
    /// torn paths all share this recovery convention).
    #[test]
    fn drain_port1_pending_rx_torn_write_publishes_head_with_zero_len() {
        let mut dev = VirtioConsole::new();
        let mem = make_chain_test_mem();
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
        // Valid GPA for the first descriptor.
        let valid_addr = GuestAddress(0x10000);
        // GPA past the 2 MiB map → write_slice fails. 4 MiB chosen
        // for clarity; any value >= 2 MiB works.
        let unmapped_addr = GuestAddress(4 << 20);
        // Two descriptors, both write-only, chained via NEXT.
        // First desc at index 0 (chain head); second at index 1.
        let descs = [
            RawDescriptor::from(SplitDescriptor::new(
                valid_addr.0,
                32,
                (VRING_DESC_F_WRITE | VRING_DESC_F_NEXT) as u16,
                1,
            )),
            RawDescriptor::from(SplitDescriptor::new(
                unmapped_addr.0,
                32,
                VRING_DESC_F_WRITE as u16,
                0,
            )),
        ];
        mock.build_desc_chain(&descs).expect("build torn chain");
        dev.set_mem(mem.clone());
        wire_port1_rxq_to_mock(&mut dev, &mock);
        open_port1(&mut dev);

        // 64-byte payload: first 32 bytes will write to valid_addr,
        // the next 32 will attempt to write to unmapped_addr and
        // fail.
        let payload: Vec<u8> = (0..64u8).collect();
        dev.ports[1].pending_rx.extend(payload.iter().copied());
        let int_before = dev.interrupt_status;
        // Drain irq_evt so the post-drain assertion can detect a
        // spurious signal_used call.
        let _ = dev.irq_evt.read();

        dev.drain_pending_rx(1);

        // (a) used.idx == 1: torn chain head was add_used'd with
        // len=0.
        let used_idx: u16 = mem
            .read_obj(mock.used_addr().checked_add(2).unwrap())
            .expect("read used.idx");
        assert_eq!(
            used_idx, 1,
            "torn-write recovery must add_used the chain head (with len=0)"
        );
        // The published len for that head must be 0. used-ring
        // entry layout: u16 flags, u16 idx, then array of
        // VRING_USED_ELEM { u32 id; u32 len }. The first entry's
        // `len` field sits at used_addr + 4 (flags+idx) + 4 (id).
        let used_elem_len: u32 = mem
            .read_obj(mock.used_addr().checked_add(8).unwrap())
            .expect("read used elem 0 len");
        assert_eq!(
            used_elem_len, 0,
            "torn-write recovery must publish len=0 for the chain head \
             — a non-zero len would tell the guest the descriptor was \
             fully filled, leading to data corruption"
        );

        // (b) Bytes stay in pending_rx. The first descriptor's
        // partial write does not consume bytes from the deque
        // because the torn branch skips the
        // `drain(..consumed_offset)` call (line ~1009 is in the
        // success branch, after the `if chain_torn { ... break; }`
        // arm at line ~983).
        assert_eq!(
            dev.ports[1].pending_rx.len(),
            payload.len(),
            "torn-write recovery must preserve bytes in pending_rx \
             for retry on the next drain cycle"
        );
        // Verify byte identity, not just length: a regression that
        // partially drained then re-extended could mask itself.
        let preserved: Vec<u8> = dev.ports[1].pending_rx.iter().copied().collect();
        assert_eq!(
            preserved, payload,
            "preserved bytes must be the original payload verbatim"
        );

        // (d) signal_used must NOT have been called: total_written
        // stays 0 for the torn chain, so the
        // `if total_written > 0 { signal_used() }` branch at line
        // ~1012 is skipped.
        assert_eq!(
            dev.interrupt_status, int_before,
            "torn-only chain must not trigger signal_used (total_written=0)"
        );
        match dev.irq_evt.read() {
            Ok(n) => panic!("irq_evt must NOT have been written, got {n}"),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(e) => panic!("unexpected irq_evt read error: {e}"),
        }
    }

    /// Torn-write breaks the drain loop: even when MORE chains are
    /// available in the avail ring, a torn first chain stops the
    /// drain immediately. Pins the `break` after `add_used(0)` at
    /// line ~996. A regression that continued to the next chain
    /// would (1) interleave torn-recovery with success traffic,
    /// confusing failure-mode analysis, and (2) potentially deliver
    /// bytes out of order if the torn chain's bytes were retried
    /// after a later chain landed.
    ///
    /// Two chains:
    ///   - chain 0: torn (second desc unmapped)
    ///   - chain 1: fully valid single descriptor
    ///
    /// After drain, used.idx must be exactly 1 (only the torn head),
    /// chain 1 must remain in the avail ring un-consumed (its data
    /// region still holds zeros, never the payload), and a follow-up
    /// drain cycle must process chain 1.
    #[test]
    fn drain_port1_pending_rx_torn_write_breaks_drain_loop() {
        let mut dev = VirtioConsole::new();
        let mem = make_chain_test_mem();
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
        let valid_addr_a = GuestAddress(0x10000);
        let unmapped_addr = GuestAddress(4 << 20);
        let valid_addr_b = GuestAddress(0x20000);

        // `add_desc_chains` publishes one chain head per descriptor
        // entry in the slice that does NOT carry F_NEXT (entries
        // with F_NEXT are linked tails of the prior head). With
        // descs[0] carrying F_NEXT->1 and descs[1]/descs[2] without
        // F_NEXT, two chain heads land in the avail ring:
        //   chain 0 = descs[0] -> descs[1]   (torn: descs[1] unmapped)
        //   chain 1 = descs[2]               (valid, single desc)
        let descs = [
            // Chain 0 head (idx 0): valid + NEXT to idx 1.
            RawDescriptor::from(SplitDescriptor::new(
                valid_addr_a.0,
                32,
                (VRING_DESC_F_WRITE | VRING_DESC_F_NEXT) as u16,
                1,
            )),
            // Chain 0 tail (idx 1): unmapped.
            RawDescriptor::from(SplitDescriptor::new(
                unmapped_addr.0,
                32,
                VRING_DESC_F_WRITE as u16,
                0,
            )),
            // Chain 1 (idx 2): single valid descriptor.
            RawDescriptor::from(SplitDescriptor::new(
                valid_addr_b.0,
                32,
                VRING_DESC_F_WRITE as u16,
                0,
            )),
        ];
        mock.add_desc_chains(&descs, 0).expect("publish two chains");
        dev.set_mem(mem.clone());
        wire_port1_rxq_to_mock(&mut dev, &mock);
        open_port1(&mut dev);

        // 64-byte payload — enough that the second descriptor of
        // chain 0 would be filled if the unmapped write succeeded.
        let payload: Vec<u8> = (0..64u8).collect();
        dev.ports[1].pending_rx.extend(payload.iter().copied());

        dev.drain_pending_rx(1);

        // used.idx == 1: ONLY chain 0's head was add_used'd (with
        // len=0); chain 1 was NOT processed because the torn-write
        // recovery broke the drain loop.
        let used_idx: u16 = mem
            .read_obj(mock.used_addr().checked_add(2).unwrap())
            .expect("read used.idx");
        assert_eq!(
            used_idx, 1,
            "torn-write recovery must break the drain loop — chain 1 \
             must remain unconsumed even though its descriptor is valid"
        );

        // Chain 1's data region must be untouched: read back
        // valid_addr_b — it must be all zeros (the test memory's
        // initial state), NOT any byte from `payload`.
        let mut readback_b = vec![0u8; 32];
        mem.read_slice(&mut readback_b, valid_addr_b)
            .expect("read back chain 1 data region");
        assert!(
            readback_b.iter().all(|&b| b == 0),
            "chain 1's data region must be untouched — the drain loop \
             must NOT have reached chain 1 after the torn break"
        );

        // Bytes still in pending_rx for retry.
        assert_eq!(
            dev.ports[1].pending_rx.len(),
            payload.len(),
            "all bytes must remain in pending_rx (torn chain consumed nothing)"
        );

        // A second drain cycle must process chain 1 successfully —
        // chain 0's head was already add_used'd, so chain 1 is the
        // next chain to pop.
        dev.drain_pending_rx(1);

        let used_idx_after: u16 = mem
            .read_obj(mock.used_addr().checked_add(2).unwrap())
            .expect("read used.idx after second drain");
        assert_eq!(
            used_idx_after, 2,
            "second drain must process chain 1 (used.idx 1 -> 2)"
        );
        // Chain 1 received the first 32 bytes of the still-pending
        // payload.
        let mut readback_b2 = vec![0u8; 32];
        mem.read_slice(&mut readback_b2, valid_addr_b)
            .expect("read chain 1 data after second drain");
        assert_eq!(
            readback_b2,
            payload[..32],
            "chain 1 must hold the first 32 bytes of the preserved payload"
        );
        // 32 bytes consumed from pending_rx; 32 remain.
        assert_eq!(
            dev.ports[1].pending_rx.len(),
            payload.len() - 32,
            "second drain must consume only chain 1's capacity (32 bytes)"
        );
    }

    /// Torn-write after a successful prior chain in the SAME drain
    /// call: the prior chain's `total_written` accumulates, then the
    /// torn chain breaks the loop. signal_used MUST be called
    /// because `total_written > 0` from the prior chain. Pins that
    /// the torn-write break does not suppress the signal for
    /// successfully-delivered chains earlier in the same call.
    ///
    /// Two chains:
    ///   - chain 0: valid single descriptor → drains 32 bytes
    ///   - chain 1: torn (second desc unmapped) → publishes head=0,
    ///     breaks loop
    ///
    /// After drain: used.idx==2, INT_VRING set, irq_evt non-zero,
    /// 32 bytes consumed from pending_rx (chain 0 only).
    #[test]
    fn drain_port1_pending_rx_torn_after_success_still_signals() {
        let mut dev = VirtioConsole::new();
        let mem = make_chain_test_mem();
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
        let valid_addr_a = GuestAddress(0x10000);
        let valid_addr_b = GuestAddress(0x20000);
        let unmapped_addr = GuestAddress(4 << 20);

        // Layout (no F_NEXT means chain end; F_NEXT means link to
        // the named index):
        //   idx 0 = chain 0 (valid, single, no NEXT)
        //   idx 1 = chain 1 head (valid + NEXT -> idx 2)
        //   idx 2 = chain 1 tail (unmapped)
        let descs = [
            RawDescriptor::from(SplitDescriptor::new(
                valid_addr_a.0,
                32,
                VRING_DESC_F_WRITE as u16,
                0,
            )),
            RawDescriptor::from(SplitDescriptor::new(
                valid_addr_b.0,
                32,
                (VRING_DESC_F_WRITE | VRING_DESC_F_NEXT) as u16,
                2,
            )),
            RawDescriptor::from(SplitDescriptor::new(
                unmapped_addr.0,
                32,
                VRING_DESC_F_WRITE as u16,
                0,
            )),
        ];
        mock.add_desc_chains(&descs, 0)
            .expect("publish success+torn chains");
        dev.set_mem(mem.clone());
        wire_port1_rxq_to_mock(&mut dev, &mock);
        open_port1(&mut dev);

        let payload: Vec<u8> = (0..96u8).collect();
        dev.ports[1].pending_rx.extend(payload.iter().copied());
        // Drain irq_evt so the post-drain check sees only the new
        // signal_used call.
        let _ = dev.irq_evt.read();

        dev.drain_pending_rx(1);

        // used.idx == 2: chain 0 add_used with len=32, chain 1
        // (torn) add_used with len=0.
        let used_idx: u16 = mem
            .read_obj(mock.used_addr().checked_add(2).unwrap())
            .expect("read used.idx");
        assert_eq!(
            used_idx, 2,
            "drain must add_used both chain 0 (len=32) and chain 1 \
             (torn, len=0)"
        );
        // INT_VRING must be set: total_written > 0 from chain 0.
        assert_ne!(
            dev.interrupt_status & VIRTIO_MMIO_INT_VRING,
            0,
            "signal_used must have been called because chain 0 \
             delivered 32 bytes"
        );
        let irq_count = dev.irq_evt.read().expect("irq_evt was written");
        assert!(
            irq_count > 0,
            "irq_evt counter must be non-zero after signal_used"
        );

        // Chain 0 received the first 32 bytes.
        let mut readback_a = vec![0u8; 32];
        mem.read_slice(&mut readback_a, valid_addr_a)
            .expect("read chain 0 data");
        assert_eq!(
            readback_a,
            payload[..32],
            "chain 0 must hold the first 32 bytes of the payload"
        );
        // pending_rx: 32 bytes from chain 0 consumed; bytes 32-95
        // remain (chain 1's torn write consumed nothing).
        assert_eq!(
            dev.ports[1].pending_rx.len(),
            payload.len() - 32,
            "only chain 0's bytes were consumed from pending_rx"
        );
        let preserved: Vec<u8> = dev.ports[1].pending_rx.iter().copied().collect();
        assert_eq!(
            preserved,
            payload[32..],
            "preserved bytes must be exactly the suffix not delivered \
             to chain 0 (chain 1's first descriptor's partial write \
             does NOT consume from the deque)"
        );
    }

    // ----------------------------------------------------------------
    // Port 2 chain-level tests (scheduler-stats relay).
    //
    // Port 2 mirrors port 1's TX/RX paths line-for-line in production
    // — only the queue index, the buffer field, and the wake eventfd
    // differ. Without explicit port-2 coverage the queue_idx → buffer
    // routing match in `process_tx_into`, the `port_opened[2]` gate in
    // `drain_port2_pending_rx`, and the `stats_tx_evt` vs `tx_evt`
    // dispatch could regress to incorrect targets and only surface in
    // an end-to-end VM run. These tests pin every divergence between
    // port 2 and the other two ports.
    // ----------------------------------------------------------------

    /// Mark port 2 as opened by sending PORT_OPEN(value=1) on c_ovq.
    /// Mirrors `open_port1` for the port-2 RX gate in
    /// `drain_port2_pending_rx`.
    fn open_port2(dev: &mut VirtioConsole) {
        dev.handle_control_event(VirtioConsoleControl {
            id: 2,
            event: VIRTIO_CONSOLE_PORT_OPEN,
            value: 1,
        });
        assert!(
            dev.ports[2].opened,
            "open_port2 helper precondition: PORT_OPEN(value=1) must \
             set port_opened[2]"
        );
    }

    /// Wire q6 (PORT2_RXQ) to a MockSplitQueue and drive the FSM to
    /// DRIVER_OK with F_MULTIPORT negotiated. Variant of
    /// `wire_port1_rxq_to_mock` for port 2.
    fn wire_port2_rxq_to_mock(dev: &mut VirtioConsole, mock: &MockSplitQueue<GuestMemoryMmap>) {
        wire_console_queue_to_mock(dev, mock, PORT2_RXQ as u32);
    }

    // ----------------------------------------------------------------
    // Port 2 TX (PORT2_TXQ) chain-level tests.
    // ----------------------------------------------------------------

    /// Single-descriptor TX chain on port 2: one device-readable
    /// segment with a known byte pattern lands verbatim in
    /// `port2_tx_buf` (observable via `drain_port2_bulk`). Pins the
    /// PORT2_TXQ branch of the queue_idx routing match in
    /// `process_tx_into` (line ~668): a regression that mis-routed
    /// to `port0_tx_buf` or `port1_tx_buf` would surface here.
    #[test]
    fn port2_tx_single_descriptor_lands_in_port2_buf() {
        let mut dev = VirtioConsole::new();
        let mem = make_chain_test_mem();
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
        let data_addr = GuestAddress(0x10000);
        let payload = b"port2 stats relay TX bytes";
        mem.write_slice(payload, data_addr).expect("plant payload");
        let descs = [RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            payload.len() as u32,
            0,
            0,
        ))];
        mock.build_desc_chain(&descs).expect("build chain");
        dev.set_mem(mem.clone());
        wire_console_queue_to_mock(&mut dev, &mock, PORT2_TXQ as u32);

        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT2_TXQ as u32);

        // Bytes must have landed in port2_tx_buf, observable via
        // drain_port2_bulk.
        let drained = dev.drain_port2_bulk();
        assert_eq!(
            drained,
            payload.to_vec(),
            "port 2 TX must deliver the descriptor's bytes to \
             drain_port2_bulk verbatim",
        );
        // Port 0 buffer must be untouched.
        assert!(
            dev.drain_output().is_empty(),
            "port 0 TX buffer must remain empty when only port 2 was notified",
        );
        // Port 1 buffer must be untouched — the routing match must
        // not have spilled into the port-1 deque.
        assert!(
            dev.drain_bulk().is_empty(),
            "port 1 TX buffer must remain empty when only port 2 was notified",
        );
        let used_idx: u16 = mem
            .read_obj(mock.used_addr().checked_add(2).unwrap())
            .expect("read used.idx");
        assert_eq!(used_idx, 1, "exactly one used-ring entry expected");
        assert_ne!(
            dev.interrupt_status & VIRTIO_MMIO_INT_VRING,
            0,
            "INT_VRING must be set after a successful TX drain",
        );
        let irq_count = dev.irq_evt.read().expect("irq_evt was written");
        assert!(
            irq_count > 0,
            "irq_evt counter must be non-zero after signal_used",
        );
    }

    /// Multi-descriptor port-2 TX chain: four 4 KiB segments
    /// concatenate in `port2_tx_buf` in chain order. Mirrors
    /// `port1_tx_multi_descriptor_chain_concatenates` — pins that
    /// the per-descriptor staged-scratch append path for PORT2_TXQ
    /// (line ~668-694) preserves byte boundaries and chain order.
    /// The host's stats client expects scx_stats responses to arrive
    /// as contiguous newline-delimited JSON; chain-order corruption
    /// would tear the JSON across response boundaries.
    #[test]
    fn port2_tx_multi_descriptor_chain_concatenates() {
        let mut dev = VirtioConsole::new();
        let mem = make_chain_test_mem();
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
        const PAGE: u32 = 4096;
        let segs: [(GuestAddress, u8); 4] = [
            (GuestAddress(0x10000), 0xB1),
            (GuestAddress(0x12000), 0xB2),
            (GuestAddress(0x14000), 0xB3),
            (GuestAddress(0x16000), 0xB4),
        ];
        for (addr, fill) in &segs {
            let buf = vec![*fill; PAGE as usize];
            mem.write_slice(&buf, *addr).expect("plant segment");
        }
        let descs = [
            RawDescriptor::from(SplitDescriptor::new(segs[0].0.0, PAGE, 0, 0)),
            RawDescriptor::from(SplitDescriptor::new(segs[1].0.0, PAGE, 0, 0)),
            RawDescriptor::from(SplitDescriptor::new(segs[2].0.0, PAGE, 0, 0)),
            RawDescriptor::from(SplitDescriptor::new(segs[3].0.0, PAGE, 0, 0)),
        ];
        mock.build_desc_chain(&descs).expect("build chain");
        dev.set_mem(mem.clone());
        wire_console_queue_to_mock(&mut dev, &mock, PORT2_TXQ as u32);

        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT2_TXQ as u32);

        let drained = dev.drain_port2_bulk();
        assert_eq!(
            drained.len(),
            4 * PAGE as usize,
            "drain_port2_bulk length must equal sum of segment lengths",
        );
        for (i, (_, fill)) in segs.iter().enumerate() {
            let start = i * PAGE as usize;
            let end = start + PAGE as usize;
            assert!(
                drained[start..end].iter().all(|&b| b == *fill),
                "segment {i} must hold fill {fill:#x} verbatim — chain order \
                 or per-descriptor append regressed",
            );
        }
        let used_idx: u16 = mem
            .read_obj(mock.used_addr().checked_add(2).unwrap())
            .expect("read used.idx");
        assert_eq!(used_idx, 1, "one chain → one used-ring entry on port 2 TX",);
    }

    /// Oversize port-2 TX descriptor truncates to `TX_DESC_MAX`. A
    /// hostile guest publishing `len > 32 KiB` on port 2 must not
    /// trigger a multi-GiB scratch alloc. Mirrors
    /// `port1_tx_oversize_descriptor_truncates_to_tx_desc_max` for
    /// the PORT2_TXQ scratch path.
    #[test]
    fn port2_tx_oversize_descriptor_truncates_to_tx_desc_max() {
        let mut dev = VirtioConsole::new();
        let mem = make_chain_test_mem();
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
        let data_addr = GuestAddress(0x10000);
        const OVERSIZE: usize = TX_DESC_MAX * 2;
        let mut payload = vec![0x55u8; TX_DESC_MAX];
        payload.extend_from_slice(&vec![0x99u8; TX_DESC_MAX]);
        assert_eq!(payload.len(), OVERSIZE);
        mem.write_slice(&payload, data_addr).expect("plant payload");
        let descs = [RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            OVERSIZE as u32,
            0,
            0,
        ))];
        mock.build_desc_chain(&descs).expect("build chain");
        dev.set_mem(mem.clone());
        wire_console_queue_to_mock(&mut dev, &mock, PORT2_TXQ as u32);

        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT2_TXQ as u32);

        let drained = dev.drain_port2_bulk();
        assert_eq!(
            drained.len(),
            TX_DESC_MAX,
            "oversize port-2 descriptor must truncate to TX_DESC_MAX",
        );
        assert!(
            drained.iter().all(|&b| b == 0x55),
            "truncated bytes must be the FIRST TX_DESC_MAX bytes \
             (0x55), not anything past the cap",
        );
    }

    /// Port 2 TX rejected without DRIVER_OK: mirrors
    /// `port1_tx_rejected_without_driver_ok`. Walks the FSM only to
    /// FEATURES_OK, configures PORT2_TXQ, plants a chain, fires the
    /// notify; the gate at the head of `process_tx_into` must drop
    /// the notify with no observable side effects.
    #[test]
    fn port2_tx_rejected_without_driver_ok() {
        let mut dev = VirtioConsole::new();
        let mem = make_chain_test_mem();
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
        let data_addr = GuestAddress(0x10000);
        let payload = b"this must NOT reach port2_tx_buf";
        mem.write_slice(payload, data_addr).expect("plant payload");
        let descs = [RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            payload.len() as u32,
            0,
            0,
        ))];
        mock.build_desc_chain(&descs).expect("build chain");
        dev.set_mem(mem.clone());

        // Walk FSM to S_FEAT only — STOP before S_OK.
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
        write_reg(
            &mut dev,
            VIRTIO_MMIO_DRIVER_FEATURES,
            1 << VIRTIO_CONSOLE_F_MULTIPORT,
        );
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
        write_reg(
            &mut dev,
            VIRTIO_MMIO_DRIVER_FEATURES,
            1 << (VIRTIO_F_VERSION_1 - 32),
        );
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_FEAT);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, PORT2_TXQ as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NUM, 16);
        let desc = mock.desc_table_addr().0;
        let avail = mock.avail_addr().0;
        let used = mock.used_addr().0;
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_LOW, desc as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_HIGH, (desc >> 32) as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_AVAIL_LOW, avail as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_AVAIL_HIGH, (avail >> 32) as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_USED_LOW, used as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_USED_HIGH, (used >> 32) as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_READY, 1);
        assert_eq!(
            dev.device_status & VIRTIO_CONFIG_S_DRIVER_OK,
            0,
            "precondition: DRIVER_OK must NOT be set",
        );
        assert!(
            dev.queues[PORT2_TXQ].ready(),
            "precondition: port 2 TX queue must be ready",
        );

        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT2_TXQ as u32);

        assert!(
            dev.drain_port2_bulk().is_empty(),
            "port2_tx_buf must remain empty — DRIVER_OK gate must \
             reject pre-DRIVER_OK notify",
        );
        let used_idx: u16 = mem
            .read_obj(mock.used_addr().checked_add(2).unwrap())
            .expect("read used.idx");
        assert_eq!(
            used_idx, 0,
            "used.idx must be 0 — DRIVER_OK gate must skip add_used",
        );
        assert_eq!(
            dev.interrupt_status, 0,
            "interrupt_status must be 0 — DRIVER_OK gate must skip signal_used",
        );
    }

    /// Port 2 TX rejected without F_MULTIPORT: PORT2_TXQ is
    /// multiport-only (line 576 in `process_tx_into`). A guest that
    /// reaches DRIVER_OK without negotiating F_MULTIPORT must not
    /// have its PORT2_TXQ notifies serviced — the legacy single-
    /// console path never exercises port 2.
    #[test]
    fn port2_tx_rejected_without_multiport() {
        let mut dev = VirtioConsole::new();
        let mem = make_chain_test_mem();
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
        let data_addr = GuestAddress(0x10000);
        let payload = b"port2 bytes that must not leak without multiport";
        mem.write_slice(payload, data_addr).expect("plant payload");
        let descs = [RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            payload.len() as u32,
            0,
            0,
        ))];
        mock.build_desc_chain(&descs).expect("build chain");
        dev.set_mem(mem.clone());

        // Walk FSM to DRIVER_OK negotiating ONLY VIRTIO_F_VERSION_1
        // (no F_MULTIPORT). `wire_console_queue_to_mock` always
        // negotiates both, so inline the custom version here.
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
        write_reg(
            &mut dev,
            VIRTIO_MMIO_DRIVER_FEATURES,
            1 << (VIRTIO_F_VERSION_1 - 32),
        );
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_FEAT);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, PORT2_TXQ as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NUM, 16);
        let desc = mock.desc_table_addr().0;
        let avail = mock.avail_addr().0;
        let used = mock.used_addr().0;
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_LOW, desc as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_HIGH, (desc >> 32) as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_AVAIL_LOW, avail as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_AVAIL_HIGH, (avail >> 32) as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_USED_LOW, used as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_USED_HIGH, (used >> 32) as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_READY, 1);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);
        assert_eq!(
            dev.device_status, S_OK,
            "precondition: FSM must reach DRIVER_OK"
        );
        assert!(
            !dev.multiport_negotiated(),
            "precondition: F_MULTIPORT must NOT be negotiated"
        );

        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT2_TXQ as u32);

        assert!(
            dev.drain_port2_bulk().is_empty(),
            "port2_tx_buf must remain empty — F_MULTIPORT gate must \
             reject the notify",
        );
        let used_idx: u16 = mem
            .read_obj(mock.used_addr().checked_add(2).unwrap())
            .expect("read used.idx");
        assert_eq!(
            used_idx, 0,
            "used.idx must be 0 — F_MULTIPORT gate must skip add_used",
        );
    }

    /// Port-0 vs port-1 vs port-2 TX routing: bytes from each port
    /// land in the matching `port{0,1,2}_tx_buf`. Extends
    /// `port0_tx_vs_port1_tx_routes_to_correct_buffer` with port 2,
    /// pinning the three-way queue_idx → buffer dispatch. A
    /// regression that mis-routed any pair would let the stats
    /// stream corrupt the bulk TLV stream or the console output.
    #[test]
    fn port0_vs_port1_vs_port2_tx_routes_to_correct_buffer() {
        let mut dev = VirtioConsole::new();
        let mem = make_chain_test_mem();
        // Three MockSplitQueues at distinct GPAs (each ring fits in
        // < 1 KiB at queue size 16; 4 KiB stride is comfortable).
        let mock0 = MockSplitQueue::create(&mem, GuestAddress(0x0), 16);
        let mock1 = MockSplitQueue::create(&mem, GuestAddress(0x1000), 16);
        let mock2 = MockSplitQueue::create(&mem, GuestAddress(0x2000), 16);
        let port0_data_addr = GuestAddress(0x10000);
        let port1_data_addr = GuestAddress(0x20000);
        let port2_data_addr = GuestAddress(0x30000);
        let port0_payload = b"port0 console bytes";
        let port1_payload = b"port1 bulk TLV bytes";
        let port2_payload = b"port2 stats relay bytes";
        mem.write_slice(port0_payload, port0_data_addr)
            .expect("plant port0 payload");
        mem.write_slice(port1_payload, port1_data_addr)
            .expect("plant port1 payload");
        mem.write_slice(port2_payload, port2_data_addr)
            .expect("plant port2 payload");

        let port0_descs = [RawDescriptor::from(SplitDescriptor::new(
            port0_data_addr.0,
            port0_payload.len() as u32,
            0,
            0,
        ))];
        let port1_descs = [RawDescriptor::from(SplitDescriptor::new(
            port1_data_addr.0,
            port1_payload.len() as u32,
            0,
            0,
        ))];
        let port2_descs = [RawDescriptor::from(SplitDescriptor::new(
            port2_data_addr.0,
            port2_payload.len() as u32,
            0,
            0,
        ))];
        mock0
            .build_desc_chain(&port0_descs)
            .expect("build port0 chain");
        mock1
            .build_desc_chain(&port1_descs)
            .expect("build port1 chain");
        mock2
            .build_desc_chain(&port2_descs)
            .expect("build port2 chain");
        dev.set_mem(mem.clone());

        // Walk FSM up to FEATURES_OK, configure all three TX queues,
        // then transition to DRIVER_OK.
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
        write_reg(
            &mut dev,
            VIRTIO_MMIO_DRIVER_FEATURES,
            1 << VIRTIO_CONSOLE_F_MULTIPORT,
        );
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
        write_reg(
            &mut dev,
            VIRTIO_MMIO_DRIVER_FEATURES,
            1 << (VIRTIO_F_VERSION_1 - 32),
        );
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_FEAT);
        // Configure each TX queue.
        for (q_idx, mock_ref) in [
            (PORT0_TXQ, &mock0),
            (PORT1_TXQ, &mock1),
            (PORT2_TXQ, &mock2),
        ] {
            write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, q_idx as u32);
            write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NUM, 16);
            let d = mock_ref.desc_table_addr().0;
            let a = mock_ref.avail_addr().0;
            let u = mock_ref.used_addr().0;
            write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_LOW, d as u32);
            write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_HIGH, (d >> 32) as u32);
            write_reg(&mut dev, VIRTIO_MMIO_QUEUE_AVAIL_LOW, a as u32);
            write_reg(&mut dev, VIRTIO_MMIO_QUEUE_AVAIL_HIGH, (a >> 32) as u32);
            write_reg(&mut dev, VIRTIO_MMIO_QUEUE_USED_LOW, u as u32);
            write_reg(&mut dev, VIRTIO_MMIO_QUEUE_USED_HIGH, (u >> 32) as u32);
            write_reg(&mut dev, VIRTIO_MMIO_QUEUE_READY, 1);
        }
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);
        assert_eq!(
            dev.device_status, S_OK,
            "FSM did not reach DRIVER_OK after all three queues configured",
        );

        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT0_TXQ as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT1_TXQ as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT2_TXQ as u32);

        let port0_drained = dev.drain_output();
        assert_eq!(
            port0_drained,
            port0_payload.to_vec(),
            "port 0 TX must route to port0_tx_buf",
        );
        let port1_drained = dev.drain_bulk();
        assert_eq!(
            port1_drained,
            port1_payload.to_vec(),
            "port 1 TX must route to port1_tx_buf",
        );
        let port2_drained = dev.drain_port2_bulk();
        assert_eq!(
            port2_drained,
            port2_payload.to_vec(),
            "port 2 TX must route to port2_tx_buf",
        );
        let port0_used_idx: u16 = mem
            .read_obj(mock0.used_addr().checked_add(2).unwrap())
            .expect("read port0 used.idx");
        let port1_used_idx: u16 = mem
            .read_obj(mock1.used_addr().checked_add(2).unwrap())
            .expect("read port1 used.idx");
        let port2_used_idx: u16 = mem
            .read_obj(mock2.used_addr().checked_add(2).unwrap())
            .expect("read port2 used.idx");
        assert_eq!(port0_used_idx, 1);
        assert_eq!(port1_used_idx, 1);
        assert_eq!(
            port2_used_idx, 1,
            "port 2 used.idx must reflect 1 completion"
        );
    }

    /// Port 2 TX wakes `stats_tx_evt` (NOT `tx_evt`). Pins the
    /// queue_idx == PORT2_TXQ branch in `process_tx_into` (lines
    /// 768-772). A regression that fired `tx_evt` for port 2 would
    /// wake the freeze coordinator's TOKEN_TX handler on every
    /// scheduler-stats response, contending on the device mutex.
    /// A regression that fired `stats_tx_evt` for ports 0/1 would
    /// wake the stats client on every console / bulk byte.
    #[test]
    fn port2_tx_wakes_stats_tx_evt_only() {
        let mut dev = VirtioConsole::new();
        let mem = make_chain_test_mem();
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
        let data_addr = GuestAddress(0x10000);
        let payload = b"port2 tx wakes stats_tx_evt";
        mem.write_slice(payload, data_addr).expect("plant payload");
        let descs = [RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            payload.len() as u32,
            0,
            0,
        ))];
        mock.build_desc_chain(&descs).expect("build chain");
        dev.set_mem(mem.clone());
        wire_console_queue_to_mock(&mut dev, &mock, PORT2_TXQ as u32);

        // Drain any prior counter so the post-notify reads measure
        // only this notify's effect.
        let _ = dev.tx_evt.read();
        let _ = dev.stats_tx_evt.read();

        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT2_TXQ as u32);

        // stats_tx_evt must have been written.
        let stats_count = dev
            .stats_tx_evt
            .read()
            .expect("stats_tx_evt was written by port-2 TX");
        assert!(
            stats_count > 0,
            "port 2 TX must wake stats_tx_evt (count > 0)",
        );
        // tx_evt must NOT have been written. EFD_NONBLOCK returns
        // WouldBlock on an empty eventfd.
        match dev.tx_evt.read() {
            Ok(n) => {
                panic!("tx_evt must NOT have been written by port-2 TX, but read returned {n}")
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(e) => panic!("unexpected tx_evt read error: {e}"),
        }
    }

    /// Port 1 TX wakes `tx_evt` (NOT `stats_tx_evt`). Symmetric to
    /// `port2_tx_wakes_stats_tx_evt_only`. Without this control
    /// test, a regression that swapped the eventfd dispatch would
    /// only surface on the port-2 side.
    #[test]
    fn port1_tx_wakes_tx_evt_only_not_stats_tx_evt() {
        let mut dev = VirtioConsole::new();
        let mem = make_chain_test_mem();
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
        let data_addr = GuestAddress(0x10000);
        let payload = b"port1 tx wakes tx_evt only";
        mem.write_slice(payload, data_addr).expect("plant payload");
        let descs = [RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            payload.len() as u32,
            0,
            0,
        ))];
        mock.build_desc_chain(&descs).expect("build chain");
        dev.set_mem(mem.clone());
        wire_console_queue_to_mock(&mut dev, &mock, PORT1_TXQ as u32);

        let _ = dev.tx_evt.read();
        let _ = dev.stats_tx_evt.read();

        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT1_TXQ as u32);

        let tx_count = dev.tx_evt.read().expect("tx_evt was written by port-1 TX");
        assert!(tx_count > 0, "port 1 TX must wake tx_evt");
        match dev.stats_tx_evt.read() {
            Ok(n) => panic!(
                "stats_tx_evt must NOT have been written by port-1 TX, but read returned {n}"
            ),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(e) => panic!("unexpected stats_tx_evt read error: {e}"),
        }
    }

    // ----------------------------------------------------------------
    // Eventfd distinctness for the three signal channels.
    // ----------------------------------------------------------------

    /// `irq_evt`, `tx_evt`, and `stats_tx_evt` are three distinct
    /// eventfds (distinct file descriptors). Extends
    /// `new_creates_eventfds` with the third signal channel. A
    /// regression that aliased any pair (e.g. cloned `tx_evt` into
    /// `stats_tx_evt`) would let port-2 wakes contend on the
    /// freeze coordinator's TOKEN_TX handler — the exact contention
    /// the separate `stats_tx_evt` was introduced to avoid.
    #[test]
    fn three_eventfds_are_distinct_fds() {
        let dev = VirtioConsole::new();
        let irq_fd = dev.irq_evt().as_raw_fd();
        let tx_fd = dev.tx_evt().as_raw_fd();
        let stats_fd = dev.stats_tx_evt().as_raw_fd();
        assert!(irq_fd >= 0, "irq_evt fd must be valid");
        assert!(tx_fd >= 0, "tx_evt fd must be valid");
        assert!(stats_fd >= 0, "stats_tx_evt fd must be valid");
        assert_ne!(irq_fd, tx_fd, "irq_evt and tx_evt must be distinct fds");
        assert_ne!(
            irq_fd, stats_fd,
            "irq_evt and stats_tx_evt must be distinct fds"
        );
        assert_ne!(
            tx_fd, stats_fd,
            "tx_evt and stats_tx_evt must be distinct fds — \
             aliasing them would defeat the per-port wake \
             separation",
        );
    }

    // ----------------------------------------------------------------
    // Port-2 control message handshake (id=2).
    // ----------------------------------------------------------------

    /// `DEVICE_READY` enqueues `PORT_ADD` for every port id including
    /// id=2. The existing `handle_device_ready_enqueues_port_adds`
    /// covers all ports via a loop over `NUM_PORTS`; this pins the
    /// id=2 entry specifically because port 2 was added later than
    /// ports 0/1 and a regression that hardcoded the loop bound
    /// (e.g. `for id in 0..2`) would skip port 2's PORT_ADD without
    /// any other test surfacing the regression.
    #[test]
    fn handle_device_ready_enqueues_port_add_for_port_2() {
        let mut dev = VirtioConsole::new();
        dev.handle_control_event(VirtioConsoleControl {
            id: 0xFFFF_FFFF,
            event: VIRTIO_CONSOLE_DEVICE_READY,
            value: 1,
        });
        assert_eq!(dev.control_out.len(), NUM_PORTS as usize);
        // Find the PORT_ADD for id=2.
        let port2_add = dev.control_out.iter().find(|m| match m {
            ControlOut::Cmd(c) => c.id == 2 && c.event == VIRTIO_CONSOLE_PORT_ADD,
            _ => false,
        });
        match port2_add {
            Some(ControlOut::Cmd(c)) => {
                let value = c.value;
                assert_eq!(
                    value, 1,
                    "PORT_ADD value=1 matches QEMU \
                     (hw/char/virtio-serial-bus.c)",
                );
            }
            Some(_) => panic!("PORT_ADD for id=2 must be a Cmd variant"),
            None => panic!(
                "DEVICE_READY must enqueue PORT_ADD for id=2 (port 2 is \
                 the scheduler-stats relay; missing it strands the port \
                 in the kernel's port-add-pending state)"
            ),
        }
    }

    /// `PORT_READY` for id=2 enqueues `PORT_NAME` (with `PORT2_NAME`)
    /// then `PORT_OPEN` — same order as port 1 (PORT_NAME before
    /// PORT_OPEN keeps udev sysfs symlink creation ahead of any
    /// userspace `/dev/vport0p2` open). Pins the `id == 1 || id == 2`
    /// branch in `handle_control_event` PORT_READY (line ~1782) and
    /// the `match id { ... 2 => PORT2_NAME, ... }` lookup at
    /// line ~1794.
    #[test]
    fn handle_port_ready_port2_name_then_open() {
        let mut dev = VirtioConsole::new();
        dev.handle_control_event(VirtioConsoleControl {
            id: 2,
            event: VIRTIO_CONSOLE_PORT_READY,
            value: 1,
        });
        assert!(
            dev.ports[2].readied,
            "port_readied[2] must be set after PORT_READY for id=2"
        );
        // Bulk-port pattern: PORT_NAME then PORT_OPEN, 2 frames.
        assert_eq!(
            dev.control_out.len(),
            2,
            "PORT_READY for id=2 must enqueue exactly 2 frames \
             (PORT_NAME, PORT_OPEN) — same shape as port 1, distinct \
             from port 0's 3-frame announce",
        );
        match &dev.control_out[0] {
            ControlOut::Name { id, name } => {
                assert_eq!(*id, 2);
                assert_eq!(
                    *name, PORT2_NAME,
                    "PORT_NAME for id=2 must use PORT2_NAME (not \
                     PORT1_NAME or PORT0_NAME) — wrong name strands \
                     udev rules that disambiguate ports by name",
                );
            }
            _ => panic!("first control_out frame must be Name for port 2"),
        }
        match &dev.control_out[1] {
            ControlOut::Cmd(c) => {
                let id = c.id;
                let event = c.event;
                let value = c.value;
                assert_eq!(id, 2);
                assert_eq!(event, VIRTIO_CONSOLE_PORT_OPEN);
                assert_eq!(value, 1);
            }
            _ => panic!("second control_out frame must be Cmd for port 2 OPEN"),
        }
    }

    /// `PORT_OPEN(id=2, value=1)` flips `port_opened[2]` from false
    /// to true; `value=0` flips it back. Pins the array-index path
    /// in `handle_control_event` PORT_OPEN (line ~1814-1816) for
    /// id=2 specifically. Without this test, a regression that
    /// scoped the index update to `id < 2` would leave port_opened[2]
    /// stuck at false and the port-2 RX gate would defer all bytes
    /// indefinitely.
    #[test]
    fn handle_port_open_tracks_state_for_port_2() {
        let mut dev = VirtioConsole::new();
        assert!(
            !dev.ports[2].opened,
            "precondition: port_opened[2] must start false"
        );
        dev.handle_control_event(VirtioConsoleControl {
            id: 2,
            event: VIRTIO_CONSOLE_PORT_OPEN,
            value: 1,
        });
        assert!(
            dev.ports[2].opened,
            "PORT_OPEN(value=1) for id=2 must set port_opened[2]"
        );
        dev.handle_control_event(VirtioConsoleControl {
            id: 2,
            event: VIRTIO_CONSOLE_PORT_OPEN,
            value: 0,
        });
        assert!(
            !dev.ports[2].opened,
            "PORT_OPEN(value=0) for id=2 must clear port_opened[2]"
        );
    }

    /// `PORT_READY` for id=2 with value=0 must NOT set
    /// `port_readied[2]` and must NOT enqueue announce frames —
    /// same semantics as the port-0 / port-1 value=0 paths. Pins
    /// that the value=0 early-return precedes the per-port gate
    /// flag for id=2.
    #[test]
    fn handle_port_ready_port2_value_zero_skipped() {
        let mut dev = VirtioConsole::new();
        dev.handle_control_event(VirtioConsoleControl {
            id: 2,
            event: VIRTIO_CONSOLE_PORT_READY,
            value: 0,
        });
        assert!(
            dev.control_out.is_empty(),
            "PORT_READY(id=2, value=0) must NOT enqueue announce frames",
        );
        assert!(
            !dev.ports[2].readied,
            "PORT_READY(id=2, value=0) must NOT set port_readied[2] — \
             a future legitimate value=1 must still complete",
        );
    }

    /// `PORT_READY` repeat for port 2 must be ignored — symmetric
    /// to the port-0 and port-1 repeat tests. A regression that
    /// scoped the per-port gate to ids {0, 1} would let port 2
    /// re-enqueue PORT_NAME / PORT_OPEN unboundedly.
    #[test]
    fn handle_port_ready_repeat_ignored_port2() {
        let mut dev = VirtioConsole::new();
        dev.handle_control_event(VirtioConsoleControl {
            id: 2,
            event: VIRTIO_CONSOLE_PORT_READY,
            value: 1,
        });
        assert!(dev.ports[2].readied);
        let after_first = dev.control_out.len();
        assert_eq!(
            after_first, 2,
            "first PORT_READY for port 2 must enqueue 2 frames (PORT_NAME, PORT_OPEN)",
        );

        dev.handle_control_event(VirtioConsoleControl {
            id: 2,
            event: VIRTIO_CONSOLE_PORT_READY,
            value: 1,
        });
        assert_eq!(
            dev.control_out.len(),
            after_first,
            "PORT_READY repeat for port 2 must be a no-op",
        );
    }

    // ----------------------------------------------------------------
    // Reset coverage for port-2 specific state.
    // ----------------------------------------------------------------

    /// `reset()` clears `port_opened[2]`, `port_readied[2]`, and
    /// `port2_pending_rx`, but PRESERVES `port2_tx_buf` (host-side
    /// capture buffer, mirrors port 0 / port 1). Extends
    /// `reset_clears_all_state` with explicit port-2 coverage. A
    /// regression that cleared `port2_tx_buf` would discard
    /// scheduler-stats responses captured at reset time before
    /// `final_drain` could surface them; a regression that did NOT
    /// clear `port2_pending_rx` would let stale request bytes leak
    /// into a fresh probe. The `device_ready` reset is implicit in
    /// `reset_clears_all_state`; this test focuses on the port-2
    /// specific fields.
    #[test]
    fn reset_clears_port2_state_preserves_port2_tx_buf() {
        let mut dev = VirtioConsole::new();
        init_device(&mut dev);
        // Plant port-2 specific state.
        dev.ports[2].tx_buf.extend(b"leftover2".iter().copied());
        dev.ports[2]
            .pending_rx
            .extend(b"stale request bytes".iter().copied());
        dev.ports[2].opened = true;
        dev.ports[2].readied = true;

        // Reset.
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, 0);

        // port2_tx_buf survives — it's the host-side capture buffer
        // for `final_drain`. Mirrors `port{0,1}_tx_buf` semantics in
        // the existing `reset_clears_all_state` test.
        assert_eq!(
            dev.ports[2].tx_buf.iter().copied().collect::<Vec<u8>>(),
            b"leftover2",
            "port2_tx_buf must survive reset (host-side capture buffer)",
        );
        // port2_pending_rx is cleared: pending host→guest bytes have
        // no post-reset consumer (the kernel's port-2 reader has
        // already torn down).
        assert!(
            dev.ports[2].pending_rx.is_empty(),
            "port2_pending_rx must be cleared on reset (no post-reset consumer)",
        );
        // port_opened[2] and port_readied[2] reset to false. The
        // existing array assertion in `reset_clears_all_state` covers
        // this implicitly via `[false; NUM_PORTS as usize]`; this
        // explicit pin avoids dependence on NUM_PORTS happening to
        // equal 3.
        assert!(!dev.ports[2].opened, "port_opened[2] must reset to false",);
        assert!(!dev.ports[2].readied, "port_readied[2] must reset to false",);
    }

    // ----------------------------------------------------------------
    // Port 2 RX (PORT2_RXQ / drain_port2_pending_rx) chain-level tests.
    // ----------------------------------------------------------------

    /// Empty pending-rx → drain is a no-op. Pins the
    /// `if pending.is_empty()` fast-exit at the head of
    /// `drain_port2_pending_rx`.
    #[test]
    fn drain_port2_pending_rx_empty_pending_is_noop() {
        let mut dev = VirtioConsole::new();
        let mem = make_chain_test_mem();
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
        dev.set_mem(mem.clone());
        wire_port2_rxq_to_mock(&mut dev, &mock);
        open_port2(&mut dev);

        let data_addr = GuestAddress(0x10000);
        let descs = [RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            64,
            VRING_DESC_F_WRITE as u16,
            0,
        ))];
        mock.build_desc_chain(&descs).expect("build chain");

        assert!(
            dev.ports[2].pending_rx.is_empty(),
            "precondition: port2_pending_rx must start empty"
        );
        let int_before = dev.interrupt_status;

        dev.drain_pending_rx(2);

        let used_idx: u16 = mem
            .read_obj(mock.used_addr().checked_add(2).unwrap())
            .expect("read used.idx");
        assert_eq!(
            used_idx, 0,
            "empty-pending fast-exit must not touch the queue"
        );
        assert_eq!(
            dev.interrupt_status, int_before,
            "empty-pending fast-exit must not call signal_used"
        );
    }

    /// `port_opened[2]` gate: with DRIVER_OK + F_MULTIPORT but BEFORE
    /// the guest has sent `PORT_OPEN(id=2, value=1)` on c_ovq, port 2
    /// has no userspace reader. Pins the
    /// `if !self.port_opened[2]` guard in `drain_port2_pending_rx`.
    /// After the guest opens port 2 via PORT_OPEN, the deferred
    /// drain runs (the open transition itself triggers it via the
    /// `2 => self.drain_port2_pending_rx()` arm at line ~1829);
    /// the bytes must then land in the queue.
    #[test]
    fn drain_port2_pending_rx_defers_until_port_open() {
        let mut dev = VirtioConsole::new();
        let mem = make_chain_test_mem();
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
        let data_addr = GuestAddress(0x10000);
        let payload = b"deferred scx_stats request";
        let descs = [RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            64,
            VRING_DESC_F_WRITE as u16,
            0,
        ))];
        mock.build_desc_chain(&descs).expect("build chain");
        dev.set_mem(mem.clone());
        wire_port2_rxq_to_mock(&mut dev, &mock);

        // Port 2 not yet opened — gate must defer.
        assert!(
            !dev.ports[2].opened,
            "precondition: port_opened[2] must be false"
        );

        dev.ports[2].pending_rx.extend(payload.iter().copied());
        dev.drain_pending_rx(2);

        assert_eq!(
            dev.ports[2].pending_rx.len(),
            payload.len(),
            "port_opened[2] gate must defer when guest has not opened port 2"
        );
        let used_idx_before: u16 = mem
            .read_obj(mock.used_addr().checked_add(2).unwrap())
            .expect("read used.idx before open");
        assert_eq!(used_idx_before, 0, "port_opened[2] gate must skip add_used");

        // Now drive PORT_OPEN(id=2, value=1). The handler at line
        // ~1829 calls drain_port2_pending_rx on the closed→open
        // transition.
        open_port2(&mut dev);

        assert!(
            dev.ports[2].pending_rx.is_empty(),
            "after PORT_OPEN(id=2), deferred bytes must drain"
        );
        let used_idx_after: u16 = mem
            .read_obj(mock.used_addr().checked_add(2).unwrap())
            .expect("read used.idx after open");
        assert_eq!(
            used_idx_after, 1,
            "after PORT_OPEN(id=2), the deferred chain must add_used"
        );
        let mut readback = vec![0u8; payload.len()];
        mem.read_slice(&mut readback, data_addr)
            .expect("read back delivered payload");
        assert_eq!(
            readback, payload,
            "delivered bytes must match the queued payload verbatim"
        );
    }

    /// Single-descriptor write-only chain on port 2: happy-path
    /// baseline. Pins that a normal drain delivers the payload to
    /// the descriptor buffer, drains `port2_pending_rx`, advances
    /// `used.idx`, and signals the guest via INT_VRING + irq_evt.
    /// Mirrors `drain_port1_pending_rx_single_descriptor_happy_path`.
    #[test]
    fn drain_port2_pending_rx_single_descriptor_happy_path() {
        let mut dev = VirtioConsole::new();
        let mem = make_chain_test_mem();
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
        let data_addr = GuestAddress(0x10000);
        let payload = b"scx_stats request line\n";
        let descs = [RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            payload.len() as u32,
            VRING_DESC_F_WRITE as u16,
            0,
        ))];
        mock.build_desc_chain(&descs).expect("build chain");
        dev.set_mem(mem.clone());
        wire_port2_rxq_to_mock(&mut dev, &mock);
        open_port2(&mut dev);

        dev.ports[2].pending_rx.extend(payload.iter().copied());
        dev.drain_pending_rx(2);

        assert!(
            dev.ports[2].pending_rx.is_empty(),
            "happy-path drain must consume all pending bytes"
        );
        let mut readback = vec![0u8; payload.len()];
        mem.read_slice(&mut readback, data_addr)
            .expect("read back delivered payload");
        assert_eq!(
            readback, payload,
            "delivered bytes must equal the queued payload"
        );
        let used_idx: u16 = mem
            .read_obj(mock.used_addr().checked_add(2).unwrap())
            .expect("read used.idx");
        assert_eq!(used_idx, 1, "happy-path drain must add_used exactly once");
        assert_ne!(
            dev.interrupt_status & VIRTIO_MMIO_INT_VRING,
            0,
            "INT_VRING must be set after a non-zero drain"
        );
        let irq_count = dev.irq_evt.read().expect("irq_evt was written");
        assert!(
            irq_count > 0,
            "irq_evt counter must be non-zero after signal_used"
        );
    }

    /// Multi-descriptor torn-write recovery on port 2: a chain with
    /// two write-only descriptors where the second points at
    /// unmapped guest memory. `mem.write_slice` fails on the second
    /// descriptor, triggering the torn-write branch in
    /// `drain_port2_pending_rx`.
    ///
    /// Pins:
    /// (a) chain head add_used'd with len=0 (used.idx == 1);
    /// (b) bytes preserved in `port2_pending_rx` for retry;
    /// (c) drain loop breaks (any further chain remains unconsumed);
    /// (d) signal_used NOT called (total_written stays 0).
    ///
    /// Mirrors
    /// `drain_port1_pending_rx_torn_write_publishes_head_with_zero_len`
    /// for the port-2 torn-write recovery branch.
    #[test]
    fn drain_port2_pending_rx_torn_write_publishes_head_with_zero_len() {
        let mut dev = VirtioConsole::new();
        let mem = make_chain_test_mem();
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
        let valid_addr = GuestAddress(0x10000);
        let unmapped_addr = GuestAddress(4 << 20);
        let descs = [
            RawDescriptor::from(SplitDescriptor::new(
                valid_addr.0,
                32,
                (VRING_DESC_F_WRITE | VRING_DESC_F_NEXT) as u16,
                1,
            )),
            RawDescriptor::from(SplitDescriptor::new(
                unmapped_addr.0,
                32,
                VRING_DESC_F_WRITE as u16,
                0,
            )),
        ];
        mock.build_desc_chain(&descs).expect("build torn chain");
        dev.set_mem(mem.clone());
        wire_port2_rxq_to_mock(&mut dev, &mock);
        open_port2(&mut dev);

        let payload: Vec<u8> = (0..64u8).collect();
        dev.ports[2].pending_rx.extend(payload.iter().copied());
        let int_before = dev.interrupt_status;
        let _ = dev.irq_evt.read();

        dev.drain_pending_rx(2);

        // (a) used.idx == 1, head published with len=0.
        let used_idx: u16 = mem
            .read_obj(mock.used_addr().checked_add(2).unwrap())
            .expect("read used.idx");
        assert_eq!(
            used_idx, 1,
            "torn-write recovery must add_used the chain head (with len=0)"
        );
        let used_elem_len: u32 = mem
            .read_obj(mock.used_addr().checked_add(8).unwrap())
            .expect("read used elem 0 len");
        assert_eq!(
            used_elem_len, 0,
            "torn-write recovery must publish len=0 for the chain head",
        );

        // (b) Bytes stay in pending_rx.
        assert_eq!(
            dev.ports[2].pending_rx.len(),
            payload.len(),
            "torn-write recovery must preserve bytes in pending_rx"
        );
        let preserved: Vec<u8> = dev.ports[2].pending_rx.iter().copied().collect();
        assert_eq!(
            preserved, payload,
            "preserved bytes must be the original payload verbatim"
        );

        // (d) signal_used NOT called.
        assert_eq!(
            dev.interrupt_status, int_before,
            "torn-only chain must not trigger signal_used (total_written=0)"
        );
        match dev.irq_evt.read() {
            Ok(n) => panic!("irq_evt must NOT have been written, got {n}"),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(e) => panic!("unexpected irq_evt read error: {e}"),
        }
    }
}
