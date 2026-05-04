//! Device-side virtio-net: MMIO dispatch, FSM, counters, in-VMM
//! loopback. See the parent module `super` for the execution-model and
//! "why" doc — header-size invariant, loopback rationale, no-worker
//! decision.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use virtio_bindings::virtio_config::{
    VIRTIO_CONFIG_S_ACKNOWLEDGE, VIRTIO_CONFIG_S_DRIVER, VIRTIO_CONFIG_S_DRIVER_OK,
    VIRTIO_CONFIG_S_FAILED, VIRTIO_CONFIG_S_FEATURES_OK, VIRTIO_F_VERSION_1,
};
use virtio_bindings::virtio_ids::VIRTIO_ID_NET;
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
use virtio_bindings::virtio_net::VIRTIO_NET_F_MAC;
use virtio_queue::{Queue, QueueT};
use vm_memory::{Address, ByteValued, Bytes, GuestMemoryMmap};
use vmm_sys_util::eventfd::EventFd;

use crate::vmm::net_config::NetConfig;

pub(crate) const MMIO_MAGIC: u32 = 0x7472_6976; // "virt" in LE
pub(crate) const MMIO_VERSION: u32 = 2; // virtio 1.x MMIO
pub(crate) const VENDOR_ID: u32 = 0;

/// MMIO region size: 4 KB (one page). Matches virtio-console and
/// virtio-blk so the FDT/cmdline emitter and the MMIO range checks in
/// `exit_dispatch` can use a single constant per device class.
pub const VIRTIO_MMIO_SIZE: u64 = 0x1000;

/// Two queues: RX index 0, TX index 1. Order is the kernel's
/// `init_vqs` order (`drivers/net/virtio_net.c`); changing the order
/// would have the guest probe mismatched queues.
pub(crate) const NUM_QUEUES: usize = 2;
pub(crate) const QUEUE_MAX_SIZE: u16 = 256;
pub(crate) const RXQ: usize = 0;
pub(crate) const TXQ: usize = 1;

/// Header length the guest expects on every RX delivery and emits on
/// every TX request. `VIRTIO_F_VERSION_1` negotiation forces
/// `vi->hdr_len = sizeof(virtio_net_hdr_mrg_rxbuf) = 12 bytes` in
/// `drivers/net/virtio_net.c::virtnet_probe`, even when
/// `VIRTIO_NET_F_MRG_RXBUF` is NOT negotiated. The mrg_rxbuf form
/// flattens to `virtio_net_hdr_v1` (10 bytes of GSO/csum fields) plus
/// a 2-byte `num_buffers`. The field is only read on RX (the device
/// emits it); on TX the guest writes a copy that the device strips.
pub const VIRTIO_NET_HDR_LEN: usize = 12;

/// Maximum L2 frame size the device accepts on TX or emits on RX.
/// 64 KiB is the largest standard MTU + jumbo headroom; bounds the
/// per-request scratch allocation against a hostile guest constructing
/// a chain that totals 4 GiB worth of descriptor lengths. Frames
/// longer than this are dropped (TX path) or refused (RX path).
pub(crate) const MAX_FRAME_SIZE: usize = 65_536;

/// Maximum bytes accepted from a single descriptor on TX. Mirrors the
/// virtio-console `TX_DESC_MAX` cap. A guest sending one descriptor
/// of `len = 0xFFFF_FFFF` would otherwise force the device to size a
/// `Vec<u8>` against an attacker-controlled value.
pub(crate) const TX_DESC_MAX: usize = MAX_FRAME_SIZE;

/// Status bits required before each phase. Mirrors virtio_console.
pub(crate) const S_ACK: u32 = VIRTIO_CONFIG_S_ACKNOWLEDGE;
pub(crate) const S_DRV: u32 = S_ACK | VIRTIO_CONFIG_S_DRIVER;
pub(crate) const S_FEAT: u32 = S_DRV | VIRTIO_CONFIG_S_FEATURES_OK;
/// Test helper — terminal state bits with DRIVER_OK set.
#[cfg(test)]
pub(crate) const S_OK: u32 = S_FEAT | VIRTIO_CONFIG_S_DRIVER_OK;

// ---------------------------------------------------------------------------
// Config space
// ---------------------------------------------------------------------------

/// Net device config space (virtio-v1.2 §5.1.4). Mirrors the kernel
/// uapi `struct virtio_net_config` field-for-field up through `mtu`
/// (the last field whose feature bit governs reads we serve). Trailing
/// fields (`speed`, `duplex`, RSS) are gated on feature bits we don't
/// advertise, so the guest driver's `virtio_cread_feature` returns
/// `-ENOENT` for those reads and never depends on the device-side
/// bytes — we serve zeros for any read past `size_of::<VirtioNetConfig>()`,
/// matching virtio-v1.2 §4.2.2.2 ("reads past the populated config
/// layout return zero").
///
/// The kernel struct is `__attribute__((packed))` (see
/// `include/uapi/linux/virtio_net.h`), so this redeclaration uses
/// `repr(C, packed)` to match the wire layout byte-for-byte. Without
/// the `packed` attribute the compiler would insert padding after
/// `mac` to align `status` (which contains a `u16`) — that padding
/// would shift `status` from offset 0x06 to 0x08 and serve the guest
/// a wrong link-status value silently.
#[repr(C, packed)]
#[derive(Copy, Clone, Default, Debug)]
pub(crate) struct VirtioNetConfig {
    /// MAC address. Always populated; gated on `VIRTIO_NET_F_MAC` from
    /// the guest's perspective (without the bit it generates a random
    /// MAC and never reads this field). v0 always advertises F_MAC.
    pub(crate) mac: [u8; 6],
    /// Link status. `VIRTIO_NET_S_LINK_UP = 1` means the carrier is up.
    /// Gated on `VIRTIO_NET_F_STATUS`. v0 does NOT advertise STATUS,
    /// so the kernel driver assumes link up unconditionally
    /// (`virtnet_probe`: "Assume link up if device can't report link
    /// status"). The field stays zero in this struct; reads past the
    /// populated layout return zero anyway.
    pub(crate) status: u16,
    /// Multiqueue pair count. Gated on `VIRTIO_NET_F_MQ`. v0 does NOT
    /// advertise MQ, so this field is unread.
    pub(crate) max_virtqueue_pairs: u16,
    /// Initial MTU. Gated on `VIRTIO_NET_F_MTU`. v0 does NOT advertise
    /// MTU, so this field is unread.
    pub(crate) mtu: u16,
}

// SAFETY: `VirtioNetConfig` is `repr(C, packed)`. With `packed` the
// alignment is 1 and there is no inter-field padding by definition
// (every field is byte-aligned). All fields are integer / fixed-size
// byte-array types for which every bit pattern is a valid value, so
// reading arbitrary bytes into the struct yields a well-defined
// value. The struct is `Copy`, `Send`, and `Sync` (all primitives),
// satisfying the `ByteValued` supertrait bounds. Total size is
// verified against the kernel uapi layout by the
// `VIRTIO_NET_CONFIG_SIZE` const assertion below.
unsafe impl ByteValued for VirtioNetConfig {}

/// Size of the populated portion of net config space (12 bytes:
/// mac 6 + status 2 + max_virtqueue_pairs 2 + mtu 2). Reads at
/// config-space offsets `>= VIRTIO_NET_CONFIG_SIZE` return zero per
/// virtio-v1.2 §4.2.2.2.
pub(crate) const VIRTIO_NET_CONFIG_SIZE: usize = std::mem::size_of::<VirtioNetConfig>();
// Compile-time field-offset checks against the kernel uapi
// `struct virtio_net_config` layout. A mismatch here means either
// Rust's `repr(C, packed)` introduced a divergence from the kernel's
// `__attribute__((packed))` layout, or a field was added/removed —
// in either case the guest would read garbage from a misaligned
// field. Failing to compile is preferable to silently serving wrong
// bytes. Citations: `include/uapi/linux/virtio_net.h` and the
// `virtio_bindings::virtio_net` mod whose own `_padding` static
// assertions pin the same offsets.
const _: () = assert!(std::mem::offset_of!(VirtioNetConfig, mac) == 0x00);
const _: () = assert!(std::mem::offset_of!(VirtioNetConfig, status) == 0x06);
const _: () = assert!(std::mem::offset_of!(VirtioNetConfig, max_virtqueue_pairs) == 0x08);
const _: () = assert!(std::mem::offset_of!(VirtioNetConfig, mtu) == 0x0A);
const _: () = assert!(VIRTIO_NET_CONFIG_SIZE == 12);

// ---------------------------------------------------------------------------
// Counters (host-side observability)
// ---------------------------------------------------------------------------

/// Per-device counters surfaced to the host monitor. All atomic so
/// the monitor can read them without locking the device struct.
///
/// Mirrors the [`super::super::virtio_blk::VirtioBlkCounters`] pattern:
/// `record_*` helper methods enforce field-pairing invariants, and
/// per-field `pub fn` accessors perform `Relaxed` loads. Counters are
/// cumulative for the device's lifetime — `VirtioNet::reset()` does
/// NOT zero them, so an operator monitoring `tx_packets` etc. observes
/// a monotonically non-decreasing series across guest re-binds.
///
/// # Counter taxonomy
///
/// All counters here are **per-event cumulative**. There are no
/// per-request live gauges in v0 — the loopback path is synchronous
/// (no deferred RX, no throttle) so there is no "currently waiting"
/// state to gauge. A future async backend (TAP, AF_PACKET) would add
/// a `currently_deferred_rx_gauge` mirroring virtio-blk's
/// `currently_throttled_gauge`.
#[derive(Debug, Default)]
pub struct VirtioNetCounters {
    /// Cumulative count of TX chains the device accepted from the
    /// guest, parsed cleanly, AND successfully marked used (TX-side
    /// `add_used` returned Ok). A TX chain rejected for malformed
    /// shape (short header, wrong direction) bumps `tx_chain_invalid`
    /// only. A parsed TX chain whose `add_used` then fails bumps
    /// `tx_add_used_failures` only. So `tx_packets` reflects chains
    /// the guest can actually observe as completed.
    ///
    /// In v0's pure-loopback mode with the per-failure counters
    /// taken into account, the conservation identity is:
    ///
    /// ```text
    /// tx_packets =
    ///     rx_packets
    ///   + tx_dropped_no_rx_buffer
    ///   + rx_chain_invalid
    ///   + rx_add_used_failures
    /// ```
    ///
    /// Each TX chain that completed lands in exactly one of:
    /// successfully delivered RX (rx_packets), no RX buffer
    /// (tx_dropped_no_rx_buffer), RX chain shape rejected
    /// (rx_chain_invalid), or RX add_used failed
    /// (rx_add_used_failures). The simple subtraction
    /// `tx_packets - rx_packets` only works when the latter three
    /// are all zero — it's NOT a generic shortfall formula.
    pub(crate) tx_packets: AtomicU64,
    /// Cumulative bytes of L2 frame data accepted from successfully
    /// completed TX chains (i.e. those that bumped `tx_packets`).
    /// Excludes the 12-byte virtio header. Paired with `tx_packets`
    /// via [`Self::record_tx_completed`].
    pub(crate) tx_bytes: AtomicU64,
    /// Cumulative count of RX chains the device successfully wrote
    /// (header + frame) AND successfully marked used (`add_used`
    /// returned Ok AND the used-ring index advanced). RX chains
    /// where `add_used` failed bump `rx_add_used_failures` only —
    /// the guest never observes the publish, so it would be wrong
    /// to count it as a delivery.
    /// Paired with `rx_bytes` via [`Self::record_rx_delivered`].
    pub(crate) rx_packets: AtomicU64,
    /// Cumulative bytes of L2 frame data successfully delivered to
    /// the guest's RX chains (i.e. paired with `rx_packets`).
    /// Excludes the 12-byte virtio header. On a chain whose RX
    /// buffer was smaller than `header + frame`, this counter
    /// reflects the actual bytes written into the descriptor minus
    /// the header — NOT the source `frame_len`. An operator sees
    /// the real bytes the guest can read, not the bytes the device
    /// intended to deliver.
    pub(crate) rx_bytes: AtomicU64,
    /// Cumulative count of successfully-captured TX frames the
    /// device could not deliver to RX because the RX queue was
    /// empty. Per-event counter; a guest that never posts RX buffers
    /// and floods TX produces one bump per dropped TX frame. The TX
    /// chain is still marked used (the guest sees TX completion via
    /// `tx_packets`); the frame never arrives on RX (no `rx_packets`
    /// bump). Distinct from `tx_chain_invalid` (TX chain shape
    /// rejected before any RX delivery was attempted).
    pub(crate) tx_dropped_no_rx_buffer: AtomicU64,
    /// Cumulative count of TX chains rejected for malformed shape:
    /// missing header, write-only descriptor in TX (TX descriptors
    /// must be device-readable), header-read failure. The TX chain
    /// is still marked used so the guest doesn't hang on the
    /// request, but the frame is dropped without an RX delivery and
    /// neither `tx_packets` nor `rx_packets` is bumped. Per-event
    /// counter.
    pub(crate) tx_chain_invalid: AtomicU64,
    /// Cumulative count of RX chains rejected for malformed shape on
    /// the loopback delivery side: read-only descriptor in RX (RX
    /// descriptors must be device-writable), or guest-memory
    /// write-failure on a write-only descriptor. The RX chain is
    /// still marked used (with `len = 0`) so the guest's network-
    /// stack equivalent of a hung-task watchdog doesn't fire on a
    /// stuck request. Per-event counter; bumped exactly once per
    /// malformed RX chain (the `tx_dropped_no_rx_buffer` counter is
    /// NOT also bumped — they are mutually exclusive failure modes,
    /// see [`Self::record_rx_chain_invalid`]).
    pub(crate) rx_chain_invalid: AtomicU64,
    /// Cumulative count of `add_used` failures on the TX queue. A
    /// non-zero value means the queue's used-ring address is
    /// unmapped or otherwise inaccessible — distinct from a chain-
    /// shape rejection (which uses `tx_chain_invalid`). Per-event
    /// counter. Operators monitoring `tx_add_used_failures > 0`
    /// know the queue itself is broken and the guest has not seen
    /// any TX completion since the failure started; the typical
    /// recovery path is a virtio reset (write `STATUS=0`). Distinct
    /// from `tx_chain_invalid` so an operator can tell "guest sent
    /// malformed frame" from "queue itself is broken".
    pub(crate) tx_add_used_failures: AtomicU64,
    /// Cumulative count of `add_used` failures on the RX queue. As
    /// with `tx_add_used_failures`, indicates a queue-state failure
    /// (used-ring unmapped) distinct from chain-shape rejection.
    /// Bumped on the RX side from both the malformed-chain branch
    /// and the successful-frame-write branch when the trailing
    /// `add_used` fails — both branches mean the device tried to
    /// publish a used-ring entry and the publish itself failed.
    pub(crate) rx_add_used_failures: AtomicU64,
}

impl VirtioNetCounters {
    /// Record TX-side completion: a parsed TX chain whose
    /// `add_used` returned Ok. Bumps `tx_packets` + `tx_bytes`.
    /// MUST be called AFTER the TX `add_used` succeeds — calling
    /// it before would let the counter lie if the publish fails
    /// (the guest would never observe the completion).
    pub(crate) fn record_tx_completed(&self, frame_bytes: u64) {
        self.tx_packets.fetch_add(1, Ordering::Relaxed);
        self.tx_bytes.fetch_add(frame_bytes, Ordering::Relaxed);
    }

    /// Record successful RX delivery (frame written to a guest
    /// descriptor chain, `add_used` returned Ok). Bumps
    /// `rx_packets` + `rx_bytes`. MUST be called AFTER the RX
    /// `add_used` succeeds — if the publish fails, the guest never
    /// observes the frame and the counter would lie. The byte count
    /// is the actual L2 bytes written into the descriptor (i.e.
    /// `bytes_written - VIRTIO_NET_HDR_LEN`), which differs from
    /// the source `frame_len` when the guest's RX buffer was
    /// smaller than `header + frame`.
    pub(crate) fn record_rx_delivered(&self, frame_bytes: u64) {
        self.rx_packets.fetch_add(1, Ordering::Relaxed);
        self.rx_bytes.fetch_add(frame_bytes, Ordering::Relaxed);
    }

    /// Record one TX chain dropped because the RX queue is empty
    /// (the TX-side already completed via [`Self::record_tx_completed`];
    /// this counter records the RX-delivery failure).
    pub(crate) fn record_tx_dropped_no_rx_buffer(&self) {
        self.tx_dropped_no_rx_buffer.fetch_add(1, Ordering::Relaxed);
    }

    /// Record one TX chain rejected for malformed shape (short
    /// header, wrong direction, header-read failure). The TX chain
    /// is marked used but neither `tx_packets` nor `rx_packets` is
    /// bumped — this is the protocol-violation path.
    pub(crate) fn record_tx_chain_invalid(&self) {
        self.tx_chain_invalid.fetch_add(1, Ordering::Relaxed);
    }

    /// Record one RX chain rejected for malformed shape on the
    /// loopback delivery side. Mutually exclusive with
    /// [`Self::record_tx_dropped_no_rx_buffer`]: a chain is either
    /// missing entirely (queue empty → `tx_dropped_no_rx_buffer`) or
    /// present but malformed (this counter). The caller honors that
    /// invariant by routing each TX→RX delivery failure to exactly
    /// one of the two counters.
    pub(crate) fn record_rx_chain_invalid(&self) {
        self.rx_chain_invalid.fetch_add(1, Ordering::Relaxed);
    }

    /// Record one `add_used` failure on the TX queue. Distinct from
    /// `record_tx_chain_invalid` so operators can tell queue-state
    /// breakage from chain-shape rejection.
    pub(crate) fn record_tx_add_used_failure(&self) {
        self.tx_add_used_failures.fetch_add(1, Ordering::Relaxed);
    }

    /// Record one `add_used` failure on the RX queue. Distinct from
    /// `record_rx_chain_invalid` so operators can tell queue-state
    /// breakage from chain-shape rejection.
    pub(crate) fn record_rx_add_used_failure(&self) {
        self.rx_add_used_failures.fetch_add(1, Ordering::Relaxed);
    }

    /// Read the cumulative count of TX chains successfully looped to
    /// RX. Per-event counter: bumped exactly once per TX chain that
    /// completed both halves of the loopback.
    pub fn tx_packets(&self) -> u64 {
        self.tx_packets.load(Ordering::Relaxed)
    }

    /// Read the cumulative bytes of L2 frame data successfully looped
    /// to RX. Excludes the 12-byte virtio header.
    pub fn tx_bytes(&self) -> u64 {
        self.tx_bytes.load(Ordering::Relaxed)
    }

    /// Read the cumulative count of RX chains delivered to the guest.
    /// Equal to `tx_packets()` in v0's pure-loopback mode.
    pub fn rx_packets(&self) -> u64 {
        self.rx_packets.load(Ordering::Relaxed)
    }

    /// Read the cumulative bytes of L2 frame data delivered to the
    /// guest's RX chains. Excludes the 12-byte virtio header.
    pub fn rx_bytes(&self) -> u64 {
        self.rx_bytes.load(Ordering::Relaxed)
    }

    /// Read the cumulative count of TX chains dropped because the RX
    /// queue had no buffer.
    pub fn tx_dropped_no_rx_buffer(&self) -> u64 {
        self.tx_dropped_no_rx_buffer.load(Ordering::Relaxed)
    }

    /// Read the cumulative count of TX chains rejected for malformed
    /// shape (missing/short header, wrong direction, header read
    /// failure).
    pub fn tx_chain_invalid(&self) -> u64 {
        self.tx_chain_invalid.load(Ordering::Relaxed)
    }

    /// Read the cumulative count of RX chains rejected for malformed
    /// shape (write-only direction violated).
    pub fn rx_chain_invalid(&self) -> u64 {
        self.rx_chain_invalid.load(Ordering::Relaxed)
    }

    /// Read the cumulative count of TX `add_used` failures (queue's
    /// used-ring address unmapped or otherwise inaccessible).
    /// Non-zero means the TX queue itself is structurally broken;
    /// distinct from `tx_chain_invalid` (chain-shape rejection).
    pub fn tx_add_used_failures(&self) -> u64 {
        self.tx_add_used_failures.load(Ordering::Relaxed)
    }

    /// Read the cumulative count of RX `add_used` failures.
    /// Non-zero means the RX queue itself is structurally broken;
    /// distinct from `rx_chain_invalid` (chain-shape rejection).
    pub fn rx_add_used_failures(&self) -> u64 {
        self.rx_add_used_failures.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Device struct
// ---------------------------------------------------------------------------

/// Virtio-net MMIO device with in-VMM loopback backend.
///
/// All state behind a single struct — no separate transport layer.
/// The caller holds this in a `PiMutex` and dispatches MMIO
/// reads/writes; the loopback work runs inline on the vCPU thread
/// inside `mmio_write(QUEUE_NOTIFY)`. See parent module docs for the
/// no-worker-thread rationale.
pub struct VirtioNet {
    queues: [Queue; NUM_QUEUES],
    queue_select: u32,
    device_features_sel: u32,
    driver_features_sel: u32,
    driver_features: u64,
    device_status: u32,
    interrupt_status: u32,
    config_generation: AtomicU32,
    /// Eventfd for KVM irqfd — signals guest interrupt.
    irq_evt: EventFd,
    /// Guest memory reference. Set before starting vCPUs.
    mem: Option<GuestMemoryMmap>,
    /// Static config-space content (mac + zeroed STATUS/MQ/MTU).
    /// Built at construction from `NetConfig`; the bytes are
    /// `byte_valued` and copied directly into the MMIO read response
    /// when the guest reads at offsets `0x100..0x100+config_size`.
    config: VirtioNetConfig,
    /// Cumulative event counters. `Arc` so external monitor observers
    /// can read them without holding any device borrow.
    counters: Arc<VirtioNetCounters>,
    /// Per-device reusable scratch buffer for one TX frame. Sized by
    /// `resize` to the actual frame length on each TX iteration.
    /// Allocated once and reused across all TX requests; the
    /// underlying capacity grows monotonically up to `MAX_FRAME_SIZE`,
    /// at which point all subsequent TX is amortized to zero
    /// allocation.
    tx_frame_scratch: Vec<u8>,
}

impl VirtioNet {
    /// Create a new virtio-net device with the given configuration.
    pub fn new(config: NetConfig) -> Self {
        let irq_evt =
            EventFd::new(libc::EFD_NONBLOCK).expect("failed to create virtio-net irq eventfd");
        VirtioNet {
            queues: [
                Queue::new(QUEUE_MAX_SIZE).expect("valid queue size"),
                Queue::new(QUEUE_MAX_SIZE).expect("valid queue size"),
            ],
            queue_select: 0,
            device_features_sel: 0,
            driver_features_sel: 0,
            driver_features: 0,
            device_status: 0,
            interrupt_status: 0,
            config_generation: AtomicU32::new(0),
            irq_evt,
            mem: None,
            config: VirtioNetConfig {
                mac: config.mac,
                status: 0,
                max_virtqueue_pairs: 0,
                mtu: 0,
            },
            counters: Arc::new(VirtioNetCounters::default()),
            tx_frame_scratch: Vec::with_capacity(MAX_FRAME_SIZE),
        }
    }

    /// Eventfd for KVM irqfd registration.
    pub fn irq_evt(&self) -> &EventFd {
        &self.irq_evt
    }

    /// Set guest memory reference. Must be called before starting vCPUs.
    pub fn set_mem(&mut self, mem: GuestMemoryMmap) {
        self.mem = Some(mem);
    }

    /// Cloneable handle to the host-observability counters. The
    /// monitor thread holds an Arc to read counters without locking
    /// the device.
    pub fn counters(&self) -> Arc<VirtioNetCounters> {
        Arc::clone(&self.counters)
    }

    /// Feature bits advertised to the guest.
    ///
    /// - `VIRTIO_F_VERSION_1`: modern virtio. Mandatory for the
    ///   12-byte mrg_rxbuf header semantics described at module level.
    /// - `VIRTIO_NET_F_MAC`: device provides the MAC. Without this
    ///   bit the kernel generates a random MAC and the
    ///   `eth_hw_addr_random` path runs; the deterministic MAC from
    ///   `NetConfig` is one of the few values an operator wants to
    ///   pin across runs (for AF_PACKET capture correlation).
    fn device_features(&self) -> u64 {
        (1u64 << VIRTIO_F_VERSION_1) | (1u64 << VIRTIO_NET_F_MAC)
    }

    fn selected_queue(&self) -> Option<usize> {
        let idx = self.queue_select as usize;
        if idx < NUM_QUEUES { Some(idx) } else { None }
    }

    // Net does not negotiate VIRTIO_RING_F_EVENT_IDX so the combined
    // bit+eventfd pattern is correct here. virtio_blk splits the two
    // because it negotiates EVENT_IDX. Without EVENT_IDX there is no
    // guest-published suppression threshold to consult, so the kick
    // is at the device's discretion. We coalesce to one kick per
    // drain (kick-per-drain, not kick-per-chain): the caller's
    // `had_used_ring_publish` flag accumulates across the whole
    // drain loop and `signal_used` runs once at the end. NAPI on the
    // guest side polls the used ring until empty, so coalescing
    // multiple chain advances under one IRQ is correct and reduces
    // vCPU exits proportional to the burst size.
    //
    // The eventfd write below has two possible errno paths,
    // both recoverable:
    //
    //   - `EAGAIN` is impossible at runtime. The eventfd is created
    //     in counter mode (no `EFD_SEMAPHORE`) with `EFD_NONBLOCK`,
    //     so EAGAIN only fires when the internal u64 is at
    //     `u64::MAX - 1` and adding 1 would overflow. That requires
    //     ~2^64 unread kicks in a row — implausible under any
    //     workload because the guest's NAPI consumes (read()s) the
    //     eventfd before the next batch.
    //
    //   - `EBADF` means the device is being torn down: the irqfd
    //     was unregistered or the EventFd dropped. There is no
    //     useful recovery — the VM is shutting down.
    //
    // Either way, the bit-set on `interrupt_status` is the
    // IRQ-handler handshake target — `vm_interrupt`
    // (drivers/virtio/virtio_mmio.c) reads and acks it on each IRQ
    // delivery. The guest does NOT poll this register. We log any
    // errno so a failed write surfaces in tracing rather than
    // silently disappearing.
    fn signal_used(&mut self) {
        self.interrupt_status |= VIRTIO_MMIO_INT_VRING;
        if let Err(e) = self.irq_evt.write(1) {
            tracing::warn!(%e, "virtio-net irq_evt.write failed");
        }
    }

    /// True when device_status has progressed past FEATURES_OK but
    /// not yet reached DRIVER_OK — the window where queue config is
    /// valid.
    fn queue_config_allowed(&self) -> bool {
        self.device_status & S_FEAT == S_FEAT
            && self.device_status & VIRTIO_CONFIG_S_DRIVER_OK == 0
    }

    /// True when driver features may be written: DRIVER set,
    /// FEATURES_OK not yet set.
    fn features_write_allowed(&self) -> bool {
        self.device_status & S_DRV == S_DRV
            && self.device_status & VIRTIO_CONFIG_S_FEATURES_OK == 0
    }

    // ------------------------------------------------------------------
    // Loopback: TX → RX byte echo
    // ------------------------------------------------------------------

    /// Drive the TX queue. For each TX chain, captures the L2 frame
    /// (after stripping the 12-byte virtio header), marks the chain
    /// used, then synthesizes an RX delivery for the same frame.
    ///
    /// vCPU-thread bounded work: the inner loop executes guest-memory
    /// reads + writes (no syscalls, no blocking) plus one irqfd write
    /// per delivered RX. Each TX chain processed contributes
    /// O(`frame_bytes`) memory copy. The MMIO QUEUE_NOTIFY handler
    /// invokes this function and returns; the freeze-rendezvous
    /// timeout is never at risk because there is no syscall to block
    /// SIGRTMIN delivery on.
    fn process_tx_loopback(&mut self) {
        // DRIVER_OK gate per virtio-v1.2 §2.1.2: the device MUST NOT
        // process virtqueue requests until the driver has finished
        // initialisation by writing DRIVER_OK. A guest writing
        // QUEUE_NOTIFY while still in the FEATURES_OK..DRIVER_OK
        // window is either buggy or hostile; either way, ignore the
        // kick. virtio_blk and virtio_console both honor this gate
        // in practice via the queue-ready check (Queue::ready
        // returns false until the address registers have been
        // written, which happens between FEATURES_OK and DRIVER_OK)
        // — but our pop_descriptor_chain path would happily drain
        // a queue whose addresses had been written but DRIVER_OK
        // not yet set, so we add the explicit status check here
        // rather than rely on queue-ready as a proxy.
        if self.device_status & VIRTIO_CONFIG_S_DRIVER_OK == 0 {
            return;
        }
        let Some(mem) = self.mem.clone() else {
            return;
        };
        // `had_used_ring_publish` tracks whether ANY queue's
        // used-ring index advanced during this drain (TX add_used
        // OR RX add_used succeeded somewhere). The irqfd kick at
        // the end is gated on this flag rather than on RX delivery
        // alone: a malformed RX chain whose `add_used(head, 0)`
        // succeeded ALSO needs a kick, otherwise the guest's NAPI
        // never observes the empty completion and the descriptor
        // sits unrecycled in the used ring until a virtio reset.
        let mut had_used_ring_publish = false;

        // Borrow-split: the TX queue iterator and the RX queue side
        // both need `&mut self.queues[...]` at non-overlapping times.
        // We iterate TX chains, capture frame bytes into the per-device
        // scratch (releasing the TX borrow), walk RX queue inside
        // `try_loopback_to_rx` (taking the RX borrow), then close
        // the loop iteration with a TX `add_used`.
        loop {
            let Some(chain_outcome) = self.pop_and_capture_tx(&mem) else {
                break;
            };
            let TxChainOutcome { head, frame_len } = chain_outcome;

            if let Some(len) = frame_len {
                // Frame captured into self.tx_frame_scratch[..len].
                // Run the RX half before recording any TX-completion
                // counter — the RX outcome determines what byte
                // count we use for rx_bytes (truncation vs full),
                // and the TX add_used at the end of this iteration
                // determines whether tx_packets bumps at all.
                match self.try_loopback_to_rx(&mem, len) {
                    LoopbackOutcome::Delivered { l2_bytes_written } => {
                        // RX add_used Ok, used-ring advanced.
                        // `l2_bytes_written` reflects actual bytes
                        // the guest can read past the virtio
                        // header — on a too-small RX buffer this
                        // is < the source `len`, so rx_bytes never
                        // overstates delivery.
                        self.counters.record_rx_delivered(l2_bytes_written);
                        had_used_ring_publish = true;
                    }
                    LoopbackOutcome::DeliveredButAddUsedFailed => {
                        // Header + frame DID land in the descriptor
                        // but the trailing `add_used` failed.
                        // `rx_add_used_failures` was already bumped
                        // inside try_loopback_to_rx. Do NOT bump
                        // rx_packets — the guest never observes the
                        // publish — and do NOT mark the used-ring
                        // as advanced (it didn't).
                    }
                    LoopbackOutcome::NoRxBuffer => {
                        // No chain popped — the RX queue was empty
                        // or not ready. The TX-captured frame is
                        // dropped on the floor.
                        self.counters.record_tx_dropped_no_rx_buffer();
                    }
                    LoopbackOutcome::RxChainInvalid { add_used_ok } => {
                        // Chain shape rejected. `rx_chain_invalid`
                        // already bumped inside try_loopback_to_rx.
                        // Whether the used-ring advanced depends on
                        // whether the recycle-add_used succeeded; if
                        // it did, the guest's NAPI must wake to see
                        // the empty completion (otherwise the buffer
                        // sits unrecycled until a virtio reset).
                        if add_used_ok {
                            had_used_ring_publish = true;
                        }
                    }
                }
            }
            // else: chain was malformed and tx_chain_invalid was
            // already bumped inside `pop_and_capture_tx`. Neither
            // `tx_packets` nor `rx_packets` advances on this path.
            // Still mark used so the guest doesn't hang.

            // Mark the TX chain used. TX descriptors are
            // device-readable, so used_len is 0 — the device wrote
            // nothing back to guest memory on the TX side. tx_packets
            // is bumped ONLY on TX add_used success — calling
            // `record_tx_completed` before this point would let the
            // counter lie if the publish fails (the guest never sees
            // the completion). Failed TX add_used bumps
            // `tx_add_used_failures` instead, keeping the per-event
            // counter taxonomy 1:1 with observable events.
            let q = &mut self.queues[TXQ];
            match q.add_used(&mem, head, 0) {
                Ok(()) => {
                    if let Some(len) = frame_len {
                        self.counters.record_tx_completed(len as u64);
                    }
                    had_used_ring_publish = true;
                }
                Err(e) => {
                    self.counters.record_tx_add_used_failure();
                    tracing::warn!(
                        head,
                        %e,
                        "virtio-net TX add_used failed (used-ring address \
                         likely unmapped); bumped tx_add_used_failures, \
                         will NOT bump tx_packets"
                    );
                }
            }
        }

        // Single irqfd kick after the full drain. Gated on
        // `had_used_ring_publish` rather than on RX delivery alone:
        // ANY used-ring advance (TX completion, RX delivery, or RX
        // malformed-chain recycle) must be observed by the guest's
        // NAPI; otherwise the guest waits indefinitely on a
        // descriptor sitting in the used ring. Without
        // VIRTIO_RING_F_EVENT_IDX, the kernel's `vring_interrupt`
        // always wakes NAPI, so coalescing multiple advances into
        // one irqfd write is correct (NAPI polls until empty).
        if had_used_ring_publish {
            self.signal_used();
        }
    }

    /// Pop one TX chain, capture the L2 frame bytes (after the
    /// 12-byte virtio header) into `self.tx_frame_scratch`, and
    /// return the chain head index plus the captured frame length.
    ///
    /// Returns `None` when the TX queue is empty (caller breaks the
    /// drain loop). Returns `Some(TxChainOutcome { frame_len: None })`
    /// when the chain is malformed — the caller must still
    /// `add_used` the head so the guest doesn't hang. Returns
    /// `Some(TxChainOutcome { frame_len: Some(n) })` on success;
    /// `self.tx_frame_scratch[..n]` holds the captured bytes.
    fn pop_and_capture_tx(&mut self, mem: &GuestMemoryMmap) -> Option<TxChainOutcome> {
        let q = &mut self.queues[TXQ];
        let chain = q.pop_descriptor_chain(mem)?;
        let head = chain.head_index();

        // Reset scratch; capacity stays. `clear` is O(1) — it just
        // zeroes the len.
        self.tx_frame_scratch.clear();

        // Track how many of the 12 virtio-net header bytes we've
        // already absorbed across the chain's leading descriptors.
        // The kernel TX path may emit the header in its own
        // descriptor (any_header_sg = true on VERSION_1, but the
        // pushed-into-skb-data path also uses a single combined
        // descriptor when headroom is sufficient). Either layout is
        // legal per virtio-v1.2 §5.1.6.5; the device must skip the
        // first 12 bytes of the chain regardless of how they're
        // distributed.
        let mut hdr_remaining: usize = VIRTIO_NET_HDR_LEN;
        let mut total_data_bytes: usize = 0;
        let mut chain_invalid = false;

        for desc in chain {
            if desc.is_write_only() {
                // TX descriptors must be device-readable. A
                // write-only descriptor in a TX chain is a guest
                // protocol violation. Stop reading; the chain is
                // dropped.
                chain_invalid = true;
                break;
            }
            let mut desc_len = (desc.len() as usize).min(TX_DESC_MAX);
            let mut desc_addr = desc.addr();

            // Skip / consume any remaining header bytes from this
            // descriptor first. `checked_add` here is defense in depth
            // against an attacker-controlled `desc.addr() = u64::MAX`:
            // an in-bounds descriptor read would have already failed
            // at `read_slice` below, but a hostile guest could place
            // the header AT a sub-page address near `u64::MAX` whose
            // `+skip` arithmetic wraps. Drop the chain on overflow
            // instead of panicking the vCPU thread (a panic on the
            // vCPU would propagate via `vcpu_panic::install_once` and
            // tear down the VM mid-test).
            if hdr_remaining > 0 {
                let skip = hdr_remaining.min(desc_len);
                let Some(new_addr) = desc_addr.checked_add(skip as u64) else {
                    chain_invalid = true;
                    break;
                };
                hdr_remaining -= skip;
                desc_len -= skip;
                desc_addr = new_addr;
            }

            if desc_len == 0 {
                continue;
            }

            // Cap total captured bytes at MAX_FRAME_SIZE so a hostile
            // chain summing to gigabytes is bounded. Any overflow is
            // dropped silently (the chain is still marked used).
            let remaining = MAX_FRAME_SIZE.saturating_sub(total_data_bytes);
            let take = desc_len.min(remaining);
            if take == 0 {
                // Frame already at MAX_FRAME_SIZE; ignore tail.
                break;
            }

            let start = self.tx_frame_scratch.len();
            self.tx_frame_scratch.resize(start + take, 0);
            if mem
                .read_slice(&mut self.tx_frame_scratch[start..start + take], desc_addr)
                .is_err()
            {
                // Guest-memory read failed (unmapped GPA). Drop the
                // chain; the rest of the descriptors are likely also
                // unmapped.
                self.tx_frame_scratch.truncate(start);
                chain_invalid = true;
                break;
            }
            total_data_bytes += take;
        }

        if chain_invalid || hdr_remaining != 0 {
            // hdr_remaining > 0 means the chain was shorter than 12
            // bytes total — the guest didn't even include the full
            // virtio header. That's a protocol violation per
            // virtio-v1.2 §5.1.6.5 ("A driver MUST set num_buffers
            // to 0" — implies the header is present in full).
            self.counters.record_tx_chain_invalid();
            return Some(TxChainOutcome { head, frame_len: None });
        }

        Some(TxChainOutcome {
            head,
            frame_len: Some(total_data_bytes),
        })
    }

    /// Deliver `self.tx_frame_scratch[..frame_len]` into one RX chain
    /// with a 12-byte virtio header (num_buffers=1, all other fields
    /// zero) prepended.
    ///
    /// Returns one of three outcomes per [`LoopbackOutcome`] so the
    /// caller can route to the correct counter:
    ///   - `Delivered`: RX chain popped, header + frame written,
    ///     `add_used` issued.
    ///   - `NoRxBuffer`: RX queue not ready or empty — caller bumps
    ///     `tx_dropped_no_rx_buffer`.
    ///   - `RxChainInvalid`: chain popped but malformed (read-only
    ///     direction, write failure) — this fn bumps
    ///     `rx_chain_invalid` AND marks the chain used so the guest
    ///     doesn't hang. Caller does NOT also bump
    ///     `tx_dropped_no_rx_buffer` (the two failure modes are
    ///     mutually exclusive).
    fn try_loopback_to_rx(
        &mut self,
        mem: &GuestMemoryMmap,
        frame_len: usize,
    ) -> LoopbackOutcome {
        let q = &mut self.queues[RXQ];
        if !q.ready() {
            // Driver hasn't published RX buffers yet (init not
            // complete). Drop the frame; future TX after RX is set
            // up will succeed.
            return LoopbackOutcome::NoRxBuffer;
        }
        let Some(chain) = q.pop_descriptor_chain(mem) else {
            return LoopbackOutcome::NoRxBuffer;
        };
        let head = chain.head_index();

        // Walk RX descriptors. Must be device-writable. Place the
        // 12-byte zero header first, then the captured frame bytes.
        // We do not split the header across descriptors — every
        // reference VMM (libkrun, firecracker, cloud-hypervisor,
        // qemu) and the kernel driver assume the header lives in a
        // single descriptor large enough to hold it. The guest
        // posts RX buffers each at least PAGE_SIZE in practice so
        // the assumption holds; on the rare case of an under-12
        // first descriptor we still try to write whatever fits and
        // walk forward — the resulting chain advertises `used_len =
        // hdr+frame` whether the bytes were split or contiguous.
        let mut bytes_written: u32 = 0;
        let mut hdr_remaining: usize = VIRTIO_NET_HDR_LEN;
        let mut frame_pos: usize = 0;
        let mut chain_invalid = false;

        for desc in chain {
            if !desc.is_write_only() {
                // RX descriptors must be device-writable. A
                // read-only descriptor in an RX chain is a guest
                // protocol violation.
                chain_invalid = true;
                break;
            }
            let mut desc_addr = desc.addr();
            let mut desc_len = desc.len() as usize;

            // First, drain any remaining header bytes into this
            // descriptor. The `mrg_rxbuf` header layout (12 bytes
            // matching `struct virtio_net_hdr_v1`): bytes 0..10 are
            // GSO/csum fields the device leaves at zero (no
            // negotiated offload features → `flags=0`,
            // `gso_type=GSO_NONE=0`, csum/hdr_len fields irrelevant);
            // bytes 10..12 are `num_buffers` LE u16 = 1, signalling
            // the kernel's `virtnet_receive_mergeable` /
            // `virtnet_receive_done` "single-buffer frame" path. A
            // zero `num_buffers` would make
            // `drivers/net/virtio_net.c::receive_mergeable` treat the
            // frame as the head of a multi-buffer chain and either
            // wait forever for the next buffer or panic on the
            // shouldn't-happen branch. Pinned at 1 because we never
            // negotiate `VIRTIO_NET_F_MRG_RXBUF`.
            //
            // `checked_add` is defense in depth against an attacker-
            // controlled `desc.addr()` near `u64::MAX`. Drop the
            // chain on overflow instead of panicking the vCPU
            // (a panic propagates via `vcpu_panic::install_once`).
            if hdr_remaining > 0 {
                let take = hdr_remaining.min(desc_len);
                const RX_HDR: [u8; VIRTIO_NET_HDR_LEN] = {
                    let mut h = [0u8; VIRTIO_NET_HDR_LEN];
                    // num_buffers = 1 (LE u16 at offset 10)
                    h[10] = 1;
                    h[11] = 0;
                    h
                };
                let hdr_start = VIRTIO_NET_HDR_LEN - hdr_remaining;
                let hdr_slice = &RX_HDR[hdr_start..hdr_start + take];
                if mem.write_slice(hdr_slice, desc_addr).is_err() {
                    chain_invalid = true;
                    break;
                }
                let Some(new_addr) = desc_addr.checked_add(take as u64) else {
                    chain_invalid = true;
                    break;
                };
                bytes_written = bytes_written
                    .checked_add(take as u32)
                    .expect("bytes_written cannot overflow u32 — capped by MAX_FRAME_SIZE+12");
                hdr_remaining -= take;
                desc_len -= take;
                desc_addr = new_addr;
            }

            if desc_len == 0 || frame_pos == frame_len {
                continue;
            }

            // Then frame bytes.
            let take = desc_len.min(frame_len - frame_pos);
            if mem
                .write_slice(
                    &self.tx_frame_scratch[frame_pos..frame_pos + take],
                    desc_addr,
                )
                .is_err()
            {
                chain_invalid = true;
                break;
            }
            bytes_written = bytes_written
                .checked_add(take as u32)
                .expect("bytes_written cannot overflow u32 — capped by MAX_FRAME_SIZE+12");
            frame_pos += take;

            if frame_pos == frame_len && hdr_remaining == 0 {
                break;
            }
        }

        if chain_invalid {
            // Malformed RX chain: the frame is dropped, the chain
            // is marked used with `len=0` so the guest can recycle
            // its descriptor (without `add_used` the kernel's
            // virtio core would never recover the buffer until a
            // virtio reset). `record_rx_chain_invalid` and the
            // `RxChainInvalid` outcome together signal the caller
            // NOT to also bump `tx_dropped_no_rx_buffer` — both
            // counters would mean different things and the
            // failure-classification taxonomy MUST stay 1:1 with
            // events.
            self.counters.record_rx_chain_invalid();
            // If `add_used` itself fails after a chain-direction
            // violation, the guest's used-ring is broken at the
            // same address the malformed chain came from. Record
            // the queue-state failure separately from
            // `rx_chain_invalid` so operators can distinguish "RX
            // chain shape was bad" (which we already counted) from
            // "RX queue is structurally broken" (this site). Both
            // counters can fire on the same chain because the
            // failure modes describe different problems.
            //
            // `add_used_ok` is propagated to the caller so it can
            // decide whether to kick: if `add_used` succeeded the
            // used-ring advanced and the guest's NAPI must wake to
            // observe the empty completion and recycle the buffer.
            let add_used_ok = match q.add_used(mem, head, 0) {
                Ok(()) => true,
                Err(e) => {
                    self.counters.record_rx_add_used_failure();
                    tracing::warn!(
                        head,
                        %e,
                        "virtio-net RX add_used failed after malformed-chain \
                         reject (used-ring address likely unmapped); bumped \
                         rx_add_used_failures"
                    );
                    false
                }
            };
            return LoopbackOutcome::RxChainInvalid { add_used_ok };
        }

        if frame_pos < frame_len || hdr_remaining != 0 {
            // RX descriptor chain was too small to hold the full
            // header + frame. virtio-v1.2 §5.1.6.4: the driver
            // SHOULD always provide an RX buffer of at least
            // `vi->hdr_len + 1500` (default MTU) bytes; a chain
            // smaller than that is the guest's fault. Drop the
            // remainder of the frame; the `bytes_written` we
            // already issued is what `add_used` records.
            //
            // Without VIRTIO_NET_F_MRG_RXBUF, frame fragmentation
            // across multiple posted buffers is NOT permitted —
            // each frame must fit in one popped chain. We intentionally
            // do not pop a second RX chain for the spillover.
            tracing::debug!(
                frame_len,
                bytes_written,
                hdr_remaining,
                "virtio-net RX buffer too small for full frame; truncating"
            );
        }

        // Compute actual L2 bytes delivered (i.e. the bytes the
        // guest can actually read past the virtio header). On a
        // too-small RX buffer this is `bytes_written - hdr_taken`
        // where `hdr_taken = VIRTIO_NET_HDR_LEN - hdr_remaining`;
        // when the buffer truncated mid-header even the header is
        // partial, in which case the L2 byte count is zero.
        // `saturating_sub` covers both cases without an explicit
        // branch.
        let hdr_taken = (VIRTIO_NET_HDR_LEN - hdr_remaining) as u32;
        let l2_bytes = bytes_written.saturating_sub(hdr_taken) as u64;

        // The guest cannot recover from an `add_used` failure
        // without a virtio reset. Bump `rx_add_used_failures`
        // (queue-state breakage) and route to a distinct outcome
        // so the caller does NOT bump `rx_packets` — the guest
        // never observes the publish. A counter that lies during
        // queue-state breakage would mislead operators into
        // thinking delivery worked.
        match q.add_used(mem, head, bytes_written) {
            Ok(()) => LoopbackOutcome::Delivered {
                l2_bytes_written: l2_bytes,
            },
            Err(e) => {
                self.counters.record_rx_add_used_failure();
                tracing::warn!(
                    head,
                    %e,
                    "virtio-net RX add_used failed after successful frame \
                     write (used-ring address likely unmapped); guest will \
                     see the frame in its descriptor but not in the ring; \
                     bumped rx_add_used_failures, will NOT bump rx_packets"
                );
                LoopbackOutcome::DeliveredButAddUsedFailed
            }
        }
    }
}

/// Outcome classification for `try_loopback_to_rx`. Each variant
/// describes both the data-side outcome and whether the RX
/// used-ring advanced — the latter governs whether the irqfd
/// kick is needed.
///
/// Variants:
///   - `Delivered { l2_bytes_written }`: header + frame written,
///     `add_used` returned Ok, used-ring advanced. Caller bumps
///     `rx_packets` / `rx_bytes` and kicks the guest. The byte
///     count is the actual L2 bytes (i.e. total descriptor bytes
///     minus the 12-byte virtio header), NOT the source frame
///     length, so a too-small RX buffer reports the truncated
///     count truthfully.
///   - `DeliveredButAddUsedFailed`: header + frame DID land in
///     the descriptor but the trailing `add_used` failed —
///     used-ring did NOT advance. `rx_add_used_failures` was
///     already bumped inside the function. Caller does NOT bump
///     `rx_packets` (the guest never observes the publish) and
///     does NOT kick (there's nothing in the ring to kick about).
///   - `NoRxBuffer`: RX queue not ready or empty, no chain
///     popped. Caller bumps `tx_dropped_no_rx_buffer`. No kick
///     (no descriptor was used).
///   - `RxChainInvalid { add_used_ok }`: chain popped but
///     malformed (read-only direction or write failure).
///     `rx_chain_invalid` was already bumped. The function tried
///     `add_used(head, 0)` to recycle the descriptor:
///     - If `add_used_ok = true`, the used-ring advanced — caller
///       must kick so the guest's NAPI observes the empty
///       completion and recycles the buffer, otherwise the guest
///       waits indefinitely.
///     - If `add_used_ok = false`, the publish failed too,
///       `rx_add_used_failures` was bumped, and there is nothing
///       to kick about.
enum LoopbackOutcome {
    Delivered { l2_bytes_written: u64 },
    DeliveredButAddUsedFailed,
    NoRxBuffer,
    RxChainInvalid { add_used_ok: bool },
}

/// Outcome of `pop_and_capture_tx`.
struct TxChainOutcome {
    head: u16,
    /// `Some(n)` when the chain was valid and `n` L2 bytes (excluding
    /// the 12-byte virtio header) were captured into
    /// `self.tx_frame_scratch[..n]`. `None` when the chain was
    /// malformed — the caller still `add_used`s the head so the guest
    /// can't hang on a malformed request.
    frame_len: Option<usize>,
}

// ---------------------------------------------------------------------------
// MMIO register dispatch
// ---------------------------------------------------------------------------

impl VirtioNet {
    /// Handle MMIO read at `offset` within the device's MMIO region.
    pub fn mmio_read(&self, offset: u64, data: &mut [u8]) {
        // Config-space reads (offsets 0x100..) may be 1, 2, 4, or 8
        // bytes wide depending on the field's type per virtio-v1.2
        // §4.2.2.2; serve them from the static config struct's bytes
        // first so a 1-byte MAC read or 2-byte STATUS read returns
        // the right value rather than the 0xff "non-4-byte" sentinel.
        if offset >= 0x100 {
            self.read_config_space(offset - 0x100, data);
            return;
        }

        // Register-space reads are 4 bytes wide. Anything else is a
        // protocol violation — return 0xff bytes (matches virtio-blk
        // and virtio-console).
        if data.len() != 4 {
            for b in data.iter_mut() {
                *b = 0xff;
            }
            return;
        }
        let val: u32 = match offset as u32 {
            VIRTIO_MMIO_MAGIC_VALUE => MMIO_MAGIC,
            VIRTIO_MMIO_VERSION => MMIO_VERSION,
            VIRTIO_MMIO_DEVICE_ID => VIRTIO_ID_NET,
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
            VIRTIO_MMIO_CONFIG_GENERATION => self.config_generation.load(Ordering::Relaxed),
            _ => 0,
        };
        tracing::debug!(offset, val, "virtio-net mmio_read");
        data.copy_from_slice(&val.to_le_bytes());
    }

    /// Serve `data.len()` bytes from config space at `offset` within
    /// the config region (offset 0 = `mac[0]`, offset 6 = `status`
    /// low byte, etc.). Reads past the populated layout return zero
    /// per virtio-v1.2 §4.2.2.2.
    fn read_config_space(&self, offset: u64, data: &mut [u8]) {
        // SAFETY: `VirtioNetConfig` is `ByteValued` — every bit
        // pattern of the underlying bytes is a valid value, so
        // viewing it as a byte slice is sound.
        let config_bytes = self.config.as_slice();
        let start = offset as usize;
        for (i, byte) in data.iter_mut().enumerate() {
            let cfg_idx = start + i;
            *byte = config_bytes.get(cfg_idx).copied().unwrap_or(0);
        }
    }

    /// Handle MMIO write at `offset` within the device's MMIO region.
    pub fn mmio_write(&mut self, offset: u64, data: &[u8]) {
        // Config-space writes are silently ignored (this device is
        // not driver-configurable; STATUS/MQ/MTU are read-only).
        // Matches virtio-console; virtio-v1.2 §4.2.2.2 ("the device
        // MAY ignore writes to config space").
        if offset >= 0x100 {
            tracing::debug!(
                offset,
                len = data.len(),
                "virtio-net config-space write ignored"
            );
            return;
        }

        if data.len() != 4 {
            return;
        }
        let val = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        tracing::debug!(offset, val, "virtio-net mmio_write");
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
                if idx == TXQ {
                    self.process_tx_loopback();
                }
                // RXQ notify (guest posted new RX buffers): no
                // immediate work — the next TX will pick up any new
                // buffer. virtio-blk and virtio-console drain their
                // pending data on the matching queue notify, but
                // here there is no pending RX to deliver outside a
                // TX-induced loopback. A future TAP/AF_PACKET
                // backend would drain pending host->guest frames on
                // RXQ notify.
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
    /// The driver must not clear bits. Each phase requires the
    /// previous phase's bits to be set. Invalid transitions are
    /// ignored.
    ///
    /// **Feature gates on FEATURES_OK**: per virtio-v1.2 §3.1.1
    /// step 6 + §2.2.1, when the driver writes FEATURES_OK the
    /// device MUST verify that:
    ///   1. All features the device requires were negotiated. This
    ///      device requires `VIRTIO_F_VERSION_1` because it emits a
    ///      12-byte `mrg_rxbuf` header on every RX delivery —
    ///      pre-1.0 transitional drivers expect the 10-byte
    ///      `virtio_net_hdr` (no `num_buffers`) and would treat the
    ///      last 2 bytes of our header as the first 2 bytes of L2
    ///      frame data, silently corrupting every received packet.
    ///   2. The negotiated set is a subset of the offered set —
    ///      i.e. `driver_features & !device_features() == 0`.
    ///      virtio-v1.2 §2.2.1: "the driver MUST NOT accept a
    ///      feature which was not offered by the device". A guest
    ///      that accepts an un-offered bit might enable code paths
    ///      we never tested (e.g. setting the F_MQ bit even though
    ///      we didn't advertise multiqueue would have the kernel
    ///      driver read `max_virtqueue_pairs` from config space,
    ///      which we leave at zero — the kernel's `if
    ///      (max_queue_pairs < MIN || max_queue_pairs > MAX)` branch
    ///      then resets it to 1, but the principle stands).
    ///
    /// On either violation the device sets `VIRTIO_CONFIG_S_FAILED`
    /// and refuses to advance to FEATURES_OK. The kernel driver's
    /// `virtio_features_ok` path (drivers/virtio/virtio.c:204-235)
    /// observes that FEATURES_OK didn't stick on the post-write
    /// STATUS read-back and aborts probe with `-ENODEV`. The FAILED
    /// bit we set is informational; the kernel's check is
    /// `!(status & FEATURES_OK)`, not `status & FAILED`.
    fn set_status(&mut self, val: u32) {
        let old = self.device_status;
        // Driver must not clear bits (except via reset, which writes 0).
        if val & self.device_status != self.device_status {
            tracing::debug!(old, val, "virtio-net set_status: rejected (clears bits)");
            return;
        }
        let new_bits = val & !self.device_status;
        let valid = match new_bits {
            VIRTIO_CONFIG_S_ACKNOWLEDGE => self.device_status == 0,
            VIRTIO_CONFIG_S_DRIVER => self.device_status == S_ACK,
            VIRTIO_CONFIG_S_FEATURES_OK => self.device_status == S_DRV,
            VIRTIO_CONFIG_S_DRIVER_OK => self.device_status == S_FEAT,
            _ => false,
        };
        if !valid {
            tracing::debug!(
                old,
                val,
                "virtio-net set_status: rejected (invalid transition)"
            );
            return;
        }
        // Feature gates on the FEATURES_OK transition.
        if new_bits == VIRTIO_CONFIG_S_FEATURES_OK {
            let device_features = self.device_features();
            // Subset rule (virtio-v1.2 §2.2.1): driver must not
            // accept any bit the device did not offer. The bitwise
            // AND-NOT extracts driver-only bits; non-zero means
            // the guest violated the protocol.
            let unoffered = self.driver_features & !device_features;
            if unoffered != 0 {
                self.device_status |= VIRTIO_CONFIG_S_FAILED;
                tracing::warn!(
                    old,
                    attempted = val,
                    driver_features = self.driver_features,
                    device_features,
                    unoffered,
                    "virtio-net set_status: driver accepted features not \
                     offered by device; rejecting FEATURES_OK and setting \
                     FAILED bit"
                );
                return;
            }
            // VERSION_1 requirement: the kernel driver MUST
            // negotiate VERSION_1 — without it our 12-byte header
            // would be interpreted as 10 bytes by the guest.
            if (self.driver_features & (1u64 << VIRTIO_F_VERSION_1)) == 0 {
                self.device_status |= VIRTIO_CONFIG_S_FAILED;
                tracing::warn!(
                    old,
                    attempted = val,
                    "virtio-net set_status: VIRTIO_F_VERSION_1 not \
                     negotiated; rejecting FEATURES_OK and setting FAILED bit"
                );
                return;
            }
        }
        self.device_status = val;
        tracing::debug!(old, new = val, "virtio-net set_status: accepted");
    }

    /// Reset the device to the post-construction state. Clears all
    /// MMIO-side state (status, features, queue config, interrupt
    /// status) and rebuilds the queues. Counters are NOT zeroed —
    /// they persist across re-binds for monotonic operator
    /// observability, matching the virtio-blk pattern.
    fn reset(&mut self) {
        self.device_status = 0;
        self.interrupt_status = 0;
        self.queue_select = 0;
        self.device_features_sel = 0;
        self.driver_features_sel = 0;
        self.driver_features = 0;
        self.tx_frame_scratch.clear();
        for q in &mut self.queues {
            q.reset();
        }
    }
}
