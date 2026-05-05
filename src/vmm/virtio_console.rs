//! Two-port virtio-console with inline MMIO transport.
//!
//! Six virtqueues per virtio-v1.2 §5.3.5 with `VIRTIO_CONSOLE_F_MULTIPORT`:
//!   q0 in0  — host→guest, port 0 (console / hvc0 stdin)
//!   q1 out0 — guest→host, port 0 (console / hvc0 stdout)
//!   q2 c_ivq — host→guest control (PORT_ADD, PORT_OPEN, etc.)
//!   q3 c_ovq — guest→host control (DEVICE_READY, PORT_READY, PORT_OPEN ack)
//!   q4 in1  — host→guest, port 1 (bulk; unused — TLV is guest→host only)
//!   q5 out1 — guest→host, port 1 (bulk TLV stream)
//!
//! Port 0 carries the interactive console (stdout/stdin via `/dev/hvc0`).
//! Port 1 carries the TLV stream written by `shm_ring::write_msg` —
//! exit code, test result, per-payload metrics, raw payload outputs,
//! profraw, scheduler exit notifications. Stimulus events, scenario
//! start/end markers, and crash payloads still travel over the SHM
//! ring. Backpressure: the host's `add_used` rate on port 1 TX gates
//! the guest's writes; when the host lags, the guest blocks in
//! `wait_port_writable` instead of dropping.
//!
//! Features: `VIRTIO_F_VERSION_1 | VIRTIO_CONSOLE_F_MULTIPORT`.
//! Config space: `cols=0, rows=0, max_nr_ports=2, emerg_wr=0` (cols/rows
//! valid only with F_SIZE which we do not advertise; the kernel reads
//! `max_nr_ports` via `virtio_cread_feature(F_MULTIPORT, max_nr_ports)`,
//! offset 4 in `struct virtio_console_config`).
//!
//! MMIO register layout per virtio-v1.2 §4.2.2. Interrupt delivery via
//! irqfd (eventfd → KVM GSI). TX data arrival on either port signals
//! `tx_evt` for zero-latency host-side wakeup.

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

/// RX wake byte: host pushed a SHM dump request. The guest's
/// `shm_poll_loop` blocks on `/dev/hvc0`; on any byte it re-checks
/// the SHM `DUMP_REQ_OFFSET` byte. The byte VALUE is informational
/// only — any non-zero byte would trigger the same re-check.
/// Distinct values let stack traces and tcpdump-style captures
/// distinguish the trigger source.
pub const SIGNAL_VC_DUMP: u8 = 0xD1;

/// RX wake byte: host pushed a graceful-shutdown request through
/// the virtio-console RX queue.
pub const SIGNAL_VC_SHUTDOWN: u8 = 0xD3;

/// RX wake byte: host's `bpf-map-write` thread finished applying
/// every queued `bpf_map_write` to the BPF maps inside the guest's
/// kernel. The guest's `shm_poll_loop` recognises the byte and
/// sets the `bpf_map_write_done` latch so a scenario blocked on
/// [`crate::scenario::Ctx::wait_for_map_write`] resumes. Replaces
/// the legacy SHM signal-slot rendezvous (host writes slot 0, guest
/// blocks on slot 0) with a virtio-console wake byte. Host side:
/// `host_comms::request_bpf_map_write_done`.
pub const SIGNAL_BPF_WRITE_DONE: u8 = 0xBF;

// `NUM_PORTS` lives in [`super::wire`]; re-exported here so existing
// call sites keep working. Port 0 = console (hvc0); port 1 = bulk
// TLV stream (`/dev/vport0p1`). Two ports → six queues per
// virtio-v1.2 §5.3.5 (`2 + 2 * num_ports`).
pub use super::wire::NUM_PORTS;

const NUM_QUEUES: usize = 2 + 2 * NUM_PORTS as usize;
const QUEUE_MAX_SIZE: u16 = 256;

// Per port_id_to_queue_idx in libkrun (mirrored here):
//   port 0: rx=0, tx=1
//   control: c_ivq=2 (host→guest), c_ovq=3 (guest→host)
//   port N>=1: rx = 2+2N, tx = 2+2N+1
// So port 1: rx=4, tx=5.
const PORT0_RXQ: usize = 0;
const PORT0_TXQ: usize = 1;
const C_IVQ: usize = 2; // host pushes control msgs to guest
const C_OVQ: usize = 3; // guest sends control msgs to host
const PORT1_RXQ: usize = 4;
const PORT1_TXQ: usize = 5;

/// Maximum bytes accepted from a single TX descriptor. The kernel's
/// virtio-console driver sends PAGE_SIZE chunks; this cap prevents a
/// malformed descriptor (len=0xFFFFFFFF) from triggering a ~4GB alloc.
const TX_DESC_MAX: usize = 32 * 1024;

/// Maximum cumulative bytes accepted by a single `process_tx_into`
/// call. The per-descriptor `TX_DESC_MAX` cap bounds individual
/// descriptors, but a hostile guest can publish thousands of valid
/// descriptors back-to-back and grow `port0_tx_buf` / `port1_tx_buf`
/// without bound. Capping the per-call drain at 256 KiB keeps the
/// per-vCPU MMIO-handler latency budget bounded — once the cap is
/// hit we stop popping chains and let the next QUEUE_NOTIFY drain
/// the rest. Backpressure on the guest's TX queue is the natural
/// consequence: a chain that has not been add_used yet stays in
/// the avail ring for the next call.
const TX_PER_CALL_MAX: usize = 256 * 1024;

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

// `PORT1_NAME` lives in [`super::wire`]; re-exported here for the
// existing call sites in this module.
pub use super::wire::PORT1_NAME;

/// Outbound (host→guest) control payload kinds. The host serialises
/// these into 8-byte wire frames (plus optional name bytes) for the
/// c_ivq.
#[derive(Debug, Clone)]
enum ControlOut {
    /// Fixed 8-byte command.
    Cmd(VirtioConsoleControl),
    /// 8-byte PORT_NAME header followed by name bytes (no NUL — the
    /// kernel pulls the trailing bytes as the name itself).
    Name { id: u32, name: &'static str },
}

impl ControlOut {
    fn len(&self) -> usize {
        match self {
            ControlOut::Cmd(_) => VC_CONTROL_SIZE,
            ControlOut::Name { name, .. } => VC_CONTROL_SIZE + name.len(),
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
            }
        }
    }
}

/// Two-port virtio-console MMIO device.
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
    /// Eventfd signaled when TX data is available on EITHER port. The
    /// host's stdout drain thread polls this to wake on port-0 console
    /// bytes; the host's bulk-data drain reads `port1_tx_buf` after a
    /// generic notification (the eventfd does not carry per-port
    /// granularity but the cost of an extra empty drain is negligible).
    tx_evt: EventFd,
    /// Guest memory reference. Set before starting vCPUs.
    mem: Option<GuestMemoryMmap>,
    /// Accumulated port-0 TX output (guest console → host stdout).
    port0_tx_buf: Vec<u8>,
    /// Accumulated port-1 TX output (guest TLV stream → host bulk
    /// drain). The guest writes TLV-framed messages here; the host
    /// parses them via `bulk_drain` into the same `ShmDrainResult`
    /// shape that the SHM ring used.
    port1_tx_buf: Vec<u8>,
    /// Pending port-0 RX (host stdin / wake bytes → guest /dev/hvc0).
    /// Unbounded: test framework must never lose host→guest data on
    /// the console. A host-side OOM is preferable to a silent dropped
    /// byte that makes a wake signal disappear or a shell paste
    /// truncate. Bursts are bounded by the producer (kernel scheduler
    /// bytes, terminal paste limits, SHM control bytes that fire at
    /// most a few times per VM run); a hostile guest cannot grow this
    /// unboundedly because the host alone produces the bytes.
    port0_pending_rx: VecDeque<u8>,
    /// Pending port-1 RX (host TLV reply frames → guest
    /// `/dev/vport0p1`). The freeze coordinator's snapshot-request
    /// handler frames a [`super::wire::SnapshotReplyPayload`] and
    /// pushes the bytes here; the next q4 (`PORT1_RXQ`) notify drains
    /// the buffer into the guest's read-side descriptors. Unbounded
    /// for the same reason as `port0_pending_rx`: a dropped reply
    /// strands the guest's `request_snapshot` blocking reader until
    /// its timeout expires, mirroring a transport failure.
    port1_pending_rx: VecDeque<u8>,
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
    /// Per-port "guest opened" state. Set when the guest sends
    /// `PORT_OPEN(value=1)` on c_ovq. The host gates port-1 RX
    /// delivery on this flag — pushing bytes before the guest opens
    /// the port lets the kernel discard them with no userspace reader.
    /// Port 0 starts implicitly open (the kernel's hvc-console path
    /// does not require a control-protocol open before TX/RX).
    port_opened: [bool; NUM_PORTS as usize],
    /// True once the guest has sent `DEVICE_READY(value=1)` on c_ovq.
    /// Gates the host-side PORT_ADD enqueues — emitting them before
    /// DEVICE_READY would be ignored by the kernel and a per-port
    /// PORT_READY handshake would never start.
    device_ready: bool,
    /// Per-port "ready" state. Set when the guest sends
    /// `PORT_READY(value=1)` on c_ovq. Gates the host-side
    /// CONSOLE_PORT / PORT_OPEN / PORT_NAME enqueues — repeat
    /// PORT_READY messages from a hostile guest would otherwise
    /// grow `control_out` without bound, exhausting host memory.
    /// Each port may be readied exactly once per device lifecycle;
    /// `reset()` clears this back to all-false.
    port_readied: [bool; NUM_PORTS as usize],
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
        VirtioConsole {
            queues: [
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
            mem: None,
            port0_tx_buf: Vec::new(),
            port1_tx_buf: Vec::new(),
            port0_pending_rx: VecDeque::new(),
            port1_pending_rx: VecDeque::new(),
            rx_scratch: Vec::new(),
            control_out: VecDeque::new(),
            port_opened: [false; NUM_PORTS as usize],
            device_ready: false,
            port_readied: [false; NUM_PORTS as usize],
        }
    }

    /// Eventfd for KVM irqfd registration.
    pub fn irq_evt(&self) -> &EventFd {
        &self.irq_evt
    }

    /// Eventfd signaled when new TX data arrives on either port. Use
    /// in the host-side stdout / bulk drain thread's poll set for
    /// zero-latency wakeup.
    pub fn tx_evt(&self) -> &EventFd {
        &self.tx_evt
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

    /// Process a TX queue: drain descriptor data into `dst`. Common
    /// path shared by port 0 (console) and port 1 (bulk). TX
    /// descriptors are device-readable (guest wrote them); the device
    /// writes nothing back, so add_used len is 0.
    ///
    /// Returns true when at least one byte was successfully copied —
    /// the caller uses that to gate `signal_used` + `tx_evt.write`.
    fn process_tx_into(&mut self, queue_idx: usize, port_label: &'static str) -> bool {
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
        let q = &mut self.queues[queue_idx];
        while let Some(chain) = q.pop_descriptor_chain(mem) {
            let head = chain.head_index();
            for desc in chain {
                if !desc.is_write_only() {
                    let guest_addr = desc.addr();
                    let dlen = (desc.len() as usize).min(TX_DESC_MAX);
                    // Append into the matching per-port accumulator.
                    let dst = match queue_idx {
                        PORT0_TXQ => &mut self.port0_tx_buf,
                        PORT1_TXQ => &mut self.port1_tx_buf,
                        _ => unreachable!("process_tx_into called on non-tx queue {queue_idx}"),
                    };
                    let start = dst.len();
                    dst.resize(start + dlen, 0);
                    if let Err(e) = mem.read_slice(&mut dst[start..], guest_addr) {
                        // Drop the descriptor's bytes from the
                        // accumulator and log so a structurally
                        // broken TX descriptor address surfaces in
                        // tracing rather than silently disappearing.
                        dst.truncate(start);
                        tracing::warn!(
                            port = port_label,
                            head,
                            dlen,
                            %e,
                            "virtio-console process_tx: read_slice failed \
                             (descriptor addr likely unmapped); dropping \
                             segment from this chain"
                        );
                    } else {
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
            // tx_evt is a wake hint to the host stdout / bulk drain
            // thread; a missed write means the host poll absorbs the
            // latency next cycle — not a correctness failure. Silent
            // swallow is intentional (in contrast to signal_used's
            // irq_evt write, which logs because a missed IRQ stalls
            // the GUEST, not just a host poll cadence).
            let _ = self.tx_evt.write(1);
        }
        had_data
    }

    fn process_port0_tx(&mut self) {
        let _ = self.process_tx_into(PORT0_TXQ, "port0");
    }

    fn process_port1_tx(&mut self) {
        let _ = self.process_tx_into(PORT1_TXQ, "port1");
    }

    /// Return and clear accumulated port-0 TX output (guest console →
    /// host stdout).
    pub fn drain_output(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.port0_tx_buf)
    }

    /// Return and clear accumulated port-1 TX output (guest bulk TLV
    /// stream). Host-side TLV parsing is in `crate::vmm::shm_ring`'s
    /// `parse_tlv_stream`.
    pub fn drain_bulk(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.port1_tx_buf)
    }

    /// Push raw bytes back onto the head of the port-1 TX buffer.
    ///
    /// The freeze coordinator's mid-run `bulk_assembler` (see
    /// `crate::vmm::bulk::HostAssembler`) drains `port1_tx_buf` via
    /// `drain_bulk` and assembles complete TLV frames. Trailing
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
        // splice prepends without allocating a new Vec.
        self.port1_tx_buf.splice(0..0, bytes.iter().copied());
    }

    /// Test helper — return all accumulated port-0 TX output as a string.
    #[cfg(test)]
    pub fn output(&self) -> String {
        String::from_utf8_lossy(&self.port0_tx_buf).to_string()
    }

    // ------------------------------------------------------------------
    // Port 0 RX: host → guest console
    // ------------------------------------------------------------------

    /// Push host data into the guest's port-0 RX buffers. Same
    /// semantics as the original single-port `queue_input` —
    /// undelivered bytes accumulate in `port0_pending_rx` and drain
    /// on the next QUEUE_NOTIFY for q0.
    pub fn queue_input(&mut self, data: &[u8]) {
        tracing::debug!(bytes = data.len(), "virtio-console queue_input");
        self.port0_pending_rx.extend(data);
        self.drain_port0_pending_rx();
    }

    /// Drain port-0 pending RX into guest buffers. Called from
    /// `queue_input` and on q0 notify. Mirrors the original
    /// single-port logic: only publish a chain when ALL writes for
    /// that chain succeeded; otherwise keep bytes in `pending_rx` for
    /// retry.
    fn drain_port0_pending_rx(&mut self) {
        if self.port0_pending_rx.is_empty() {
            return;
        }
        if self.device_status & VIRTIO_CONFIG_S_DRIVER_OK == 0 {
            tracing::debug!(
                pending = self.port0_pending_rx.len(),
                status = self.device_status,
                "virtio-console drain_port0_pending_rx: DRIVER_OK not set; deferring"
            );
            return;
        }
        let mem = match self.mem.as_ref() {
            Some(m) => m,
            None => {
                tracing::debug!(
                    pending = self.port0_pending_rx.len(),
                    "virtio-console drain_port0_pending_rx: no mem"
                );
                return;
            }
        };
        if !self.queues[PORT0_RXQ].ready() {
            tracing::debug!(
                pending = self.port0_pending_rx.len(),
                "virtio-console drain_port0_pending_rx: RX queue not ready"
            );
            return;
        }
        let q = &mut self.queues[PORT0_RXQ];
        let mut total_written = 0u32;
        while !self.port0_pending_rx.is_empty() {
            let Some(chain) = q.pop_descriptor_chain(mem) else {
                break;
            };
            let head = chain.head_index();
            let mut consumed_offset = 0usize;
            let mut written = 0u32;
            let mut chain_torn = false;
            for desc in chain {
                if desc.is_write_only() && consumed_offset < self.port0_pending_rx.len() {
                    let guest_addr = desc.addr();
                    let avail = desc.len() as usize;
                    let remaining = self.port0_pending_rx.len() - consumed_offset;
                    let chunk = remaining.min(avail);
                    self.rx_scratch.clear();
                    let (head_slice, tail_slice) = self.port0_pending_rx.as_slices();
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
                            head,
                            written,
                            "virtio-console drain_port0_pending_rx: write_slice failed \
                             mid-chain; breaking out to avoid partial-fill corruption"
                        );
                        chain_torn = true;
                        break;
                    }
                }
            }
            if chain_torn {
                break;
            }
            if let Err(e) = q.add_used(mem, head, written) {
                tracing::warn!(
                    head,
                    written,
                    %e,
                    "virtio-console RX add_used failed (used-ring address \
                     likely unmapped); bytes preserved in pending_rx for \
                     retry on the next drain cycle"
                );
                break;
            }
            self.port0_pending_rx.drain(..consumed_offset);
            total_written += written;
        }
        if total_written > 0 {
            tracing::debug!(
                delivered = total_written,
                pending = self.port0_pending_rx.len(),
                "virtio-console drain_port0_pending_rx: delivered to guest",
            );
            self.signal_used();
        }
    }

    // ------------------------------------------------------------------
    // Port 1 RX: host → guest bulk channel (TLV reply frames)
    // ------------------------------------------------------------------

    /// Push host data into the guest's port-1 RX buffers. Used by the
    /// freeze coordinator's snapshot-request handler to deliver a
    /// [`super::wire::SnapshotReplyPayload`] back to the in-guest
    /// `request_snapshot` blocking reader. Bytes that cannot be
    /// delivered immediately (no chain available, port not opened
    /// yet, DRIVER_OK not set) accumulate in `port1_pending_rx` and
    /// drain on the next q4 (`PORT1_RXQ`) notify.
    #[allow(dead_code)]
    pub(crate) fn queue_input_port1(&mut self, data: &[u8]) {
        tracing::debug!(bytes = data.len(), "virtio-console queue_input_port1");
        self.port1_pending_rx.extend(data);
        self.drain_port1_pending_rx();
    }

    /// Drain port-1 pending RX into guest buffers. Mirrors
    /// [`Self::drain_port0_pending_rx`] but gated additionally on
    /// `port_opened[1]` because port 1 is a multiport channel that the
    /// kernel only opens after the host's `PORT_OPEN` control message
    /// completes the handshake. Pushing bytes before the open landed
    /// would let the kernel discard them with no userspace reader.
    fn drain_port1_pending_rx(&mut self) {
        if self.port1_pending_rx.is_empty() {
            return;
        }
        if self.device_status & VIRTIO_CONFIG_S_DRIVER_OK == 0 {
            tracing::debug!(
                pending = self.port1_pending_rx.len(),
                status = self.device_status,
                "virtio-console drain_port1_pending_rx: DRIVER_OK not set; deferring"
            );
            return;
        }
        if !self.port_opened[1] {
            tracing::debug!(
                pending = self.port1_pending_rx.len(),
                "virtio-console drain_port1_pending_rx: port 1 not yet opened by guest; deferring"
            );
            return;
        }
        let mem = match self.mem.as_ref() {
            Some(m) => m,
            None => {
                tracing::debug!(
                    pending = self.port1_pending_rx.len(),
                    "virtio-console drain_port1_pending_rx: no mem"
                );
                return;
            }
        };
        if !self.queues[PORT1_RXQ].ready() {
            tracing::debug!(
                pending = self.port1_pending_rx.len(),
                "virtio-console drain_port1_pending_rx: RX queue not ready"
            );
            return;
        }
        let q = &mut self.queues[PORT1_RXQ];
        let mut total_written = 0u32;
        while !self.port1_pending_rx.is_empty() {
            let Some(chain) = q.pop_descriptor_chain(mem) else {
                break;
            };
            let head = chain.head_index();
            let mut consumed_offset = 0usize;
            let mut written = 0u32;
            let mut chain_torn = false;
            for desc in chain {
                if desc.is_write_only() && consumed_offset < self.port1_pending_rx.len() {
                    let guest_addr = desc.addr();
                    let avail = desc.len() as usize;
                    let remaining = self.port1_pending_rx.len() - consumed_offset;
                    let chunk = remaining.min(avail);
                    self.rx_scratch.clear();
                    let (head_slice, tail_slice) = self.port1_pending_rx.as_slices();
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
                            head,
                            written,
                            "virtio-console drain_port1_pending_rx: write_slice failed \
                             mid-chain; breaking out to avoid partial-fill corruption"
                        );
                        chain_torn = true;
                        break;
                    }
                }
            }
            if chain_torn {
                break;
            }
            if let Err(e) = q.add_used(mem, head, written) {
                tracing::warn!(
                    head,
                    written,
                    %e,
                    "virtio-console port-1 RX add_used failed (used-ring \
                     address likely unmapped); bytes preserved in \
                     pending_rx for retry on the next drain cycle"
                );
                break;
            }
            self.port1_pending_rx.drain(..consumed_offset);
            total_written += written;
        }
        if total_written > 0 {
            tracing::debug!(
                delivered = total_written,
                pending = self.port1_pending_rx.len(),
                "virtio-console drain_port1_pending_rx: delivered to guest",
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
        // Move work into local Vec so we can release the queue borrow
        // before calling back into self for control_out enqueue.
        let mut events: Vec<VirtioConsoleControl> = Vec::new();
        {
            let mem = match self.mem.as_ref() {
                Some(m) => m,
                None => return,
            };
            let q = &mut self.queues[C_OVQ];
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
                if let Err(e) = q.add_used(mem, head, total) {
                    tracing::warn!(
                        head,
                        total,
                        %e,
                        "virtio-console c_ovq add_used failed"
                    );
                }
            }
        }
        for c in events {
            self.handle_control_event(c);
        }
        // Always ensure the guest sees an irq for any used-ring
        // entries we just published, then attempt to push pending
        // outbound control messages onto c_ivq (which may have been
        // refilled by the guest after this notify).
        self.signal_used();
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
                // `init_vqs`), so any subsequent message is a guest
                // protocol violation.
                if self.device_ready {
                    tracing::warn!("virtio-console DEVICE_READY repeat ignored");
                    return;
                }
                self.device_ready = true;
                // Send PORT_ADD for every port we expose.
                for port_id in 0..NUM_PORTS {
                    self.control_out
                        .push_back(ControlOut::Cmd(VirtioConsoleControl {
                            id: port_id,
                            event: VIRTIO_CONSOLE_PORT_ADD,
                            value: 0,
                        }));
                }
            }
            VIRTIO_CONSOLE_PORT_READY => {
                if value != 1 {
                    tracing::warn!(id, value, "virtio-console PORT_READY != 1");
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
                if self.port_readied[id as usize] {
                    tracing::warn!(id, "virtio-console PORT_READY repeat ignored");
                    return;
                }
                self.port_readied[id as usize] = true;
                if id == 0 {
                    // Console port: announce and open.
                    self.control_out
                        .push_back(ControlOut::Cmd(VirtioConsoleControl {
                            id,
                            event: VIRTIO_CONSOLE_CONSOLE_PORT,
                            value: 1,
                        }));
                    self.control_out
                        .push_back(ControlOut::Cmd(VirtioConsoleControl {
                            id,
                            event: VIRTIO_CONSOLE_PORT_OPEN,
                            value: 1,
                        }));
                } else {
                    // Bulk data port: open + name.
                    self.control_out
                        .push_back(ControlOut::Cmd(VirtioConsoleControl {
                            id,
                            event: VIRTIO_CONSOLE_PORT_OPEN,
                            value: 1,
                        }));
                    self.control_out.push_back(ControlOut::Name {
                        id,
                        name: PORT1_NAME,
                    });
                }
            }
            VIRTIO_CONSOLE_PORT_OPEN => {
                if id >= NUM_PORTS {
                    tracing::warn!(id, "virtio-console PORT_OPEN for unknown port");
                    return;
                }
                let now_open = value == 1;
                let was_open = self.port_opened[id as usize];
                self.port_opened[id as usize] = now_open;
                // When port 1 transitions closed→open, kick the
                // pending-RX drain. The freeze coordinator may have
                // queued snapshot replies before the guest finished
                // its PORT_OPEN handshake (the bulk port appears
                // asynchronously after multiport completes); without
                // a drain trigger here those bytes would sit in
                // `port1_pending_rx` until the next q4 notify, which
                // a guest still in `read()` may not generate
                // promptly.
                if id == 1 && now_open && !was_open {
                    self.drain_port1_pending_rx();
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
        let mem = match self.mem.as_ref() {
            Some(m) => m,
            None => return,
        };
        if !self.queues[C_IVQ].ready() {
            return;
        }
        let q = &mut self.queues[C_IVQ];
        let mut total_written = 0u32;
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
                // stays in `control_out` for the next chain.
                tracing::warn!(
                    head,
                    avail,
                    need,
                    "virtio-console c_ivq: chain too small for control message"
                );
                if let Err(e) = q.add_used(mem, head, 0) {
                    tracing::warn!(head, %e, "virtio-console c_ivq add_used(0) failed");
                }
                break;
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
                // Torn write: report 0 bytes used so the guest's
                // virtio_console driver does not parse the partial
                // frame as a complete control message. Reporting
                // `written` here would let the guest dispatch on a
                // truncated header (e.g. PORT_NAME with a half-
                // copied name) and corrupt its multiport state.
                // The control message stays at the front of
                // `control_out` for retry on the next chain.
                if let Err(e) = q.add_used(mem, head, 0) {
                    tracing::warn!(head, %e, "virtio-console c_ivq add_used after torn failed");
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
            total_written += written;
            // Now safe to consume from the front.
            self.control_out.pop_front();
        }
        if total_written > 0 {
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
                    PORT0_RXQ => self.drain_port0_pending_rx(),
                    PORT0_TXQ => self.process_port0_tx(),
                    C_IVQ => self.drain_control_in(),
                    C_OVQ => self.process_control_tx(),
                    PORT1_RXQ => {
                        // Guest published RX buffers on port 1; drain
                        // any pending host→guest TLV reply bytes
                        // (e.g. snapshot replies) into the freshly
                        // available descriptors. When no bytes are
                        // pending the drain is a quick no-op — the
                        // guest publishes empty buffers as flow
                        // control even when the host has nothing to
                        // send.
                        self.drain_port1_pending_rx();
                    }
                    PORT1_TXQ => self.process_port1_tx(),
                    _ => {
                        tracing::debug!(idx, "virtio-console: notify on unused queue");
                    }
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
                self.drain_port0_pending_rx();
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
        self.device_status = 0;
        self.interrupt_status = 0;
        self.queue_select = 0;
        self.device_features_sel = 0;
        self.driver_features_sel = 0;
        self.driver_features = 0;
        self.port0_tx_buf.clear();
        self.port1_tx_buf.clear();
        self.port0_pending_rx.clear();
        self.port1_pending_rx.clear();
        self.control_out.clear();
        self.port_opened = [false; NUM_PORTS as usize];
        self.device_ready = false;
        self.port_readied = [false; NUM_PORTS as usize];
        for q in &mut self.queues {
            q.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::io::AsRawFd;
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
        dev.port0_tx_buf.extend_from_slice(b"leftover0");
        dev.port1_tx_buf.extend_from_slice(b"leftover1");
        dev.port_opened[0] = true;
        dev.port_opened[1] = true;
        dev.device_ready = true;
        dev.port_readied[0] = true;
        dev.port_readied[1] = true;
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
        assert!(dev.port0_tx_buf.is_empty(), "port0_tx_buf must be cleared");
        assert!(dev.port1_tx_buf.is_empty(), "port1_tx_buf must be cleared");
        assert_eq!(dev.port_opened, [false; NUM_PORTS as usize]);
        assert!(!dev.device_ready);
        assert_eq!(dev.port_readied, [false; NUM_PORTS as usize]);
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
        assert_ne!(dev.irq_evt().as_raw_fd(), dev.tx_evt().as_raw_fd());
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
        dev.process_port0_tx();
        dev.process_port1_tx();
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
                    assert_eq!(id, i as u32);
                    assert_eq!(event, VIRTIO_CONSOLE_PORT_ADD);
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
        assert_eq!(dev.control_out.len(), 2);
        // CONSOLE_PORT then PORT_OPEN
        let m0 = &dev.control_out[0];
        let m1 = &dev.control_out[1];
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
    fn handle_port_ready_port1_open_then_name() {
        let mut dev = VirtioConsole::new();
        dev.handle_control_event(VirtioConsoleControl {
            id: 1,
            event: VIRTIO_CONSOLE_PORT_READY,
            value: 1,
        });
        assert_eq!(dev.control_out.len(), 2);
        match &dev.control_out[0] {
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
        match &dev.control_out[1] {
            ControlOut::Name { id, name } => {
                assert_eq!(*id, 1);
                assert_eq!(*name, PORT1_NAME);
            }
            _ => panic!("expected Name"),
        }
    }

    #[test]
    fn handle_port_open_tracks_state() {
        let mut dev = VirtioConsole::new();
        assert!(!dev.port_opened[1]);
        dev.handle_control_event(VirtioConsoleControl {
            id: 1,
            event: VIRTIO_CONSOLE_PORT_OPEN,
            value: 1,
        });
        assert!(dev.port_opened[1]);
        dev.handle_control_event(VirtioConsoleControl {
            id: 1,
            event: VIRTIO_CONSOLE_PORT_OPEN,
            value: 0,
        });
        assert!(!dev.port_opened[1]);
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
    // Per CLAUDE.md VMM convergence: "chain-level MockSplitQueue
    // tests mandatory — exercises the actual process_requests
    // production path through descriptor chains."
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
}
