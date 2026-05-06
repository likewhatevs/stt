//! Tests for virtio-net device.
//!
//! Three layers, each isolating a different failure surface:
//!   - MMIO state-machine tests (status FSM, feature negotiation,
//!     queue config gating).
//!   - Config-space layout tests (offsets, byte ordering, MAC delivery).
//!   - Loopback-path tests (TX → RX byte echo via real
//!     `virtio_queue::Queue` + `GuestMemoryMmap`).
//!
//! The loopback tests construct a small guest-memory region, lay out
//! one TX chain and one RX chain at known offsets, drive the device
//! through `mmio_write(QUEUE_NOTIFY, TXQ)`, and read back the RX
//! chain's bytes plus the device's counter state to verify both the
//! header invariant and the byte-echo correctness.

use super::device::*;
use crate::vmm::net_config::NetConfig;

use virtio_bindings::virtio_config::{
    VIRTIO_CONFIG_S_DRIVER, VIRTIO_CONFIG_S_FEATURES_OK, VIRTIO_F_VERSION_1,
};
use virtio_bindings::virtio_ids::VIRTIO_ID_NET;
use virtio_bindings::virtio_mmio::{
    VIRTIO_MMIO_DEVICE_FEATURES, VIRTIO_MMIO_DEVICE_FEATURES_SEL, VIRTIO_MMIO_DEVICE_ID,
    VIRTIO_MMIO_DRIVER_FEATURES, VIRTIO_MMIO_DRIVER_FEATURES_SEL, VIRTIO_MMIO_INT_VRING,
    VIRTIO_MMIO_INTERRUPT_ACK, VIRTIO_MMIO_INTERRUPT_STATUS, VIRTIO_MMIO_MAGIC_VALUE,
    VIRTIO_MMIO_QUEUE_AVAIL_LOW, VIRTIO_MMIO_QUEUE_DESC_LOW, VIRTIO_MMIO_QUEUE_NOTIFY,
    VIRTIO_MMIO_QUEUE_NUM, VIRTIO_MMIO_QUEUE_NUM_MAX, VIRTIO_MMIO_QUEUE_READY,
    VIRTIO_MMIO_QUEUE_SEL, VIRTIO_MMIO_QUEUE_USED_LOW, VIRTIO_MMIO_STATUS, VIRTIO_MMIO_VENDOR_ID,
    VIRTIO_MMIO_VERSION,
};
use virtio_bindings::virtio_net::VIRTIO_NET_F_MAC;
use vm_memory::{Bytes, GuestAddress, GuestMemoryMmap};

// ---------------------------------------------------------------------------
// MMIO helpers
// ---------------------------------------------------------------------------

fn read_reg(dev: &VirtioNet, offset: u32) -> u32 {
    let mut buf = [0u8; 4];
    dev.mmio_read(offset as u64, &mut buf);
    u32::from_le_bytes(buf)
}

fn write_reg(dev: &mut VirtioNet, offset: u32, val: u32) {
    dev.mmio_write(offset as u64, &val.to_le_bytes());
}

/// Drive the device through ACK → DRIVER → negotiate VERSION_1 + MAC →
/// FEATURES_OK. Stops short of DRIVER_OK so callers can program queue
/// addresses (which is only allowed in the FEATURES_OK..DRIVER_OK
/// window).
fn init_until_features_ok(dev: &mut VirtioNet) {
    write_reg(dev, VIRTIO_MMIO_STATUS, S_ACK);
    write_reg(dev, VIRTIO_MMIO_STATUS, S_DRV);
    // Negotiate VERSION_1 (bit 32) + F_MAC (bit 5).
    write_reg(dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
    write_reg(dev, VIRTIO_MMIO_DRIVER_FEATURES, 1u32 << VIRTIO_NET_F_MAC);
    write_reg(dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
    write_reg(
        dev,
        VIRTIO_MMIO_DRIVER_FEATURES,
        1u32 << (VIRTIO_F_VERSION_1 - 32),
    );
    write_reg(dev, VIRTIO_MMIO_STATUS, S_FEAT);
}

// ---------------------------------------------------------------------------
// Identification + features
// ---------------------------------------------------------------------------

#[test]
fn magic_version_device_id() {
    let dev = VirtioNet::new(NetConfig::default());
    assert_eq!(read_reg(&dev, VIRTIO_MMIO_MAGIC_VALUE), 0x7472_6976);
    assert_eq!(read_reg(&dev, VIRTIO_MMIO_VERSION), 2);
    assert_eq!(read_reg(&dev, VIRTIO_MMIO_DEVICE_ID), VIRTIO_ID_NET);
    assert_eq!(read_reg(&dev, VIRTIO_MMIO_VENDOR_ID), 0);
}

#[test]
fn device_features_advertises_version_1_and_mac() {
    let mut dev = VirtioNet::new(NetConfig::default());
    write_reg(&mut dev, VIRTIO_MMIO_DEVICE_FEATURES_SEL, 0);
    let lo = read_reg(&dev, VIRTIO_MMIO_DEVICE_FEATURES);
    write_reg(&mut dev, VIRTIO_MMIO_DEVICE_FEATURES_SEL, 1);
    let hi = read_reg(&dev, VIRTIO_MMIO_DEVICE_FEATURES);
    let features = ((hi as u64) << 32) | lo as u64;
    assert_ne!(
        features & (1u64 << VIRTIO_F_VERSION_1),
        0,
        "VIRTIO_F_VERSION_1 must be advertised (forces 12-byte mrg_rxbuf hdr)",
    );
    assert_ne!(
        features & (1u64 << VIRTIO_NET_F_MAC),
        0,
        "VIRTIO_NET_F_MAC must be advertised (deterministic MAC)",
    );
}

#[test]
fn device_features_does_not_advertise_unsupported_bits() {
    // Pin the negative side of feature negotiation: bits we
    // intentionally do NOT advertise (CSUM, MRG_RXBUF, CTRL_VQ, MQ,
    // STATUS, MTU). A regression that quietly added one of these
    // would change the kernel driver's hdr_len computation,
    // num_buffers handling, or queue layout — silent corruption.
    use virtio_bindings::virtio_net::{
        VIRTIO_NET_F_CSUM, VIRTIO_NET_F_CTRL_VQ, VIRTIO_NET_F_MQ, VIRTIO_NET_F_MRG_RXBUF,
        VIRTIO_NET_F_MTU, VIRTIO_NET_F_STATUS,
    };
    let mut dev = VirtioNet::new(NetConfig::default());
    write_reg(&mut dev, VIRTIO_MMIO_DEVICE_FEATURES_SEL, 0);
    let lo = read_reg(&dev, VIRTIO_MMIO_DEVICE_FEATURES);
    write_reg(&mut dev, VIRTIO_MMIO_DEVICE_FEATURES_SEL, 1);
    let hi = read_reg(&dev, VIRTIO_MMIO_DEVICE_FEATURES);
    let features = ((hi as u64) << 32) | lo as u64;
    for (bit, name) in [
        (VIRTIO_NET_F_CSUM, "CSUM"),
        (VIRTIO_NET_F_MRG_RXBUF, "MRG_RXBUF"),
        (VIRTIO_NET_F_STATUS, "STATUS"),
        (VIRTIO_NET_F_CTRL_VQ, "CTRL_VQ"),
        (VIRTIO_NET_F_MQ, "MQ"),
        (VIRTIO_NET_F_MTU, "MTU"),
    ] {
        assert_eq!(
            features & (1u64 << bit),
            0,
            "v0 must not advertise VIRTIO_NET_F_{name}",
        );
    }
}

// ---------------------------------------------------------------------------
// Status FSM (mirrors virtio_console test surface)
// ---------------------------------------------------------------------------

#[test]
fn status_state_machine_walks_phases() {
    let mut dev = VirtioNet::new(NetConfig::default());
    assert_eq!(read_reg(&dev, VIRTIO_MMIO_STATUS), 0);

    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
    assert_eq!(read_reg(&dev, VIRTIO_MMIO_STATUS), S_ACK);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
    assert_eq!(read_reg(&dev, VIRTIO_MMIO_STATUS), S_DRV);
    // Skipping FEATURES_OK to DRIVER_OK is rejected.
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);
    assert_eq!(
        read_reg(&dev, VIRTIO_MMIO_STATUS),
        S_DRV,
        "skip FEATURES_OK must be rejected"
    );
    // Clearing bits is rejected.
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
    assert_eq!(
        read_reg(&dev, VIRTIO_MMIO_STATUS),
        S_DRV,
        "clearing DRIVER bit must be rejected"
    );
}

#[test]
fn status_skip_acknowledge_rejected() {
    let mut dev = VirtioNet::new(NetConfig::default());
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, VIRTIO_CONFIG_S_DRIVER);
    assert_eq!(
        read_reg(&dev, VIRTIO_MMIO_STATUS),
        0,
        "DRIVER without prior ACKNOWLEDGE must be rejected"
    );
}

#[test]
fn status_reset_via_zero() {
    let mut dev = VirtioNet::new(NetConfig::default());
    init_until_features_ok(&mut dev);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);
    assert_eq!(read_reg(&dev, VIRTIO_MMIO_STATUS), S_OK);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, 0);
    assert_eq!(read_reg(&dev, VIRTIO_MMIO_STATUS), 0);
}

#[test]
fn driver_features_gated_by_status() {
    let mut dev = VirtioNet::new(NetConfig::default());
    // Before DRIVER status, features writes are rejected.
    write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
    write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES, 0xDEAD);
    // After ACKNOWLEDGE + DRIVER, features writes are accepted —
    // BUT the FEATURES_OK transition is gated on VERSION_1 being
    // negotiated (see F7 in adversarial review). A driver that
    // writes feature page 0 only without VERSION_1 (which lives at
    // bit 32 in feature page 1) hits the FAILED path. This test
    // negotiates VERSION_1 properly first so the FSM advances; the
    // FAILED path is exercised by `features_ok_rejected_without_version_1`
    // below.
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
    write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
    write_reg(
        &mut dev,
        VIRTIO_MMIO_DRIVER_FEATURES,
        1u32 << (VIRTIO_F_VERSION_1 - 32),
    );
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_FEAT);
    assert_eq!(read_reg(&dev, VIRTIO_MMIO_STATUS), S_FEAT);
}

#[test]
fn features_ok_rejected_without_version_1() {
    // F7: a guest that fails to negotiate VIRTIO_F_VERSION_1 then
    // writes FEATURES_OK MUST get the FAILED bit set, NOT silent
    // acceptance. Without VERSION_1 the wire format would be the
    // legacy 10-byte virtio_net_hdr (no num_buffers); our device
    // emits 12 bytes, so the kernel driver's frame-delivery path
    // would silently misalign and corrupt every received packet.
    use virtio_bindings::virtio_config::VIRTIO_CONFIG_S_FAILED;
    let mut dev = VirtioNet::new(NetConfig::default());
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
    // Write some bits, but NOT VIRTIO_F_VERSION_1.
    write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
    write_reg(
        &mut dev,
        VIRTIO_MMIO_DRIVER_FEATURES,
        1u32 << VIRTIO_NET_F_MAC,
    );
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_FEAT);
    let status = read_reg(&dev, VIRTIO_MMIO_STATUS);
    assert_eq!(
        status & VIRTIO_CONFIG_S_FEATURES_OK,
        0,
        "FEATURES_OK must NOT be set when VERSION_1 is missing",
    );
    assert_ne!(
        status & VIRTIO_CONFIG_S_FAILED,
        0,
        "FAILED bit must be set when the driver fails to negotiate VERSION_1",
    );
}

#[test]
fn queue_config_rejected_after_driver_ok() {
    let mut dev = VirtioNet::new(NetConfig::default());
    init_until_features_ok(&mut dev);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 0);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NUM, 64);
    // QUEUE_NUM is not externally readable, but QUEUE_READY toggling
    // is gated by the same `queue_config_allowed()` predicate. Verify
    // the gate via QUEUE_READY: written 1 after DRIVER_OK should not
    // take effect.
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_READY, 1);
    assert_eq!(
        read_reg(&dev, VIRTIO_MMIO_QUEUE_READY),
        0,
        "queue config writes after DRIVER_OK must not take effect"
    );
}

// ---------------------------------------------------------------------------
// Queue identification
// ---------------------------------------------------------------------------

#[test]
fn queue_num_max_is_256_for_both_queues() {
    let mut dev = VirtioNet::new(NetConfig::default());
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 0);
    assert_eq!(
        read_reg(&dev, VIRTIO_MMIO_QUEUE_NUM_MAX),
        QUEUE_MAX_SIZE as u32
    );
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 1);
    assert_eq!(
        read_reg(&dev, VIRTIO_MMIO_QUEUE_NUM_MAX),
        QUEUE_MAX_SIZE as u32
    );
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 2);
    assert_eq!(
        read_reg(&dev, VIRTIO_MMIO_QUEUE_NUM_MAX),
        0,
        "queue index >= NUM_QUEUES must report max=0"
    );
}

// ---------------------------------------------------------------------------
// Config space
// ---------------------------------------------------------------------------

#[test]
fn config_space_serves_mac_at_offset_0() {
    let mac = [0x52, 0x54, 0x00, 0x12, 0x34, 0x56];
    let dev = VirtioNet::new(NetConfig::default().mac(mac));
    let mut buf = [0u8; 6];
    dev.mmio_read(0x100, &mut buf);
    assert_eq!(
        buf, mac,
        "config offset 0x100 must serve the configured MAC bytes",
    );
}

#[test]
fn config_space_serves_zeros_past_layout() {
    let dev = VirtioNet::new(NetConfig::default());
    // Config space size is 12 bytes (mac6 + status2 + mq2 + mtu2).
    // Reads at offset 0x100+12 and beyond must return zero.
    let mut buf = [0u8; 4];
    dev.mmio_read(0x100 + VIRTIO_NET_CONFIG_SIZE as u64, &mut buf);
    assert_eq!(buf, [0, 0, 0, 0], "reads past populated layout return zero");
    let mut buf = [0u8; 8];
    dev.mmio_read(0x100 + VIRTIO_NET_CONFIG_SIZE as u64 + 16, &mut buf);
    assert_eq!(buf, [0u8; 8], "reads far past populated layout return zero");
}

#[test]
fn config_space_mac_byte_order_matches_kernel_uapi() {
    // mac[0..6] occupies offsets 0x00..0x06 verbatim — there is no
    // byte-swapping. Pin the explicit byte-by-byte mapping against
    // a regression that introduced `to_le_bytes` or similar.
    let mac = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];
    let dev = VirtioNet::new(NetConfig::default().mac(mac));
    let mut buf0 = [0u8; 1];
    dev.mmio_read(0x100, &mut buf0);
    assert_eq!(buf0[0], 0xAA);
    let mut buf5 = [0u8; 1];
    dev.mmio_read(0x100 + 5, &mut buf5);
    assert_eq!(buf5[0], 0xFF);
}

#[test]
fn config_space_writes_silently_ignored() {
    let mac = [0x02, 0x00, 0x00, 0x00, 0x00, 0x01];
    let mut dev = VirtioNet::new(NetConfig::default().mac(mac));
    // Try to overwrite the MAC via config-space write.
    dev.mmio_write(0x100, &[0xff, 0xff, 0xff, 0xff]);
    let mut buf = [0u8; 6];
    dev.mmio_read(0x100, &mut buf);
    assert_eq!(
        buf, mac,
        "config-space writes must be silently ignored (device is not driver-configurable)",
    );
}

// ---------------------------------------------------------------------------
// Interrupt status / ACK
// ---------------------------------------------------------------------------

#[test]
fn interrupt_ack_clears_bits() {
    let mut dev = VirtioNet::new(NetConfig::default());
    // Drive the device through a loopback delivery so the
    // interrupt-status bit is set the same way the production path
    // sets it. (We can't poke `interrupt_status` directly because
    // the field is private to the device module.)
    let (mem, layout) = build_test_memory();
    dev.set_mem(mem.clone());
    init_until_features_ok(&mut dev);
    program_queues(&mut dev, &layout);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);
    place_tx_chain(&mem, &layout, &payload_42_bytes());
    place_rx_chain(&mem, &layout);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, TXQ as u32);

    let before = read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS);
    assert_ne!(
        before & VIRTIO_MMIO_INT_VRING,
        0,
        "INT_VRING must be set after a successful loopback delivery"
    );
    write_reg(&mut dev, VIRTIO_MMIO_INTERRUPT_ACK, VIRTIO_MMIO_INT_VRING);
    assert_eq!(
        read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS) & VIRTIO_MMIO_INT_VRING,
        0,
        "INT_VRING must be cleared after ACK"
    );
}

// ---------------------------------------------------------------------------
// Loopback (TX → RX byte echo)
// ---------------------------------------------------------------------------

const GUEST_MEM_SIZE: usize = 0x10_0000; // 1 MB
const TX_DESC_TABLE_BASE: u64 = 0x1000;
const TX_AVAIL_RING_BASE: u64 = 0x2000;
const TX_USED_RING_BASE: u64 = 0x3000;
const TX_HEADER_BUF: u64 = 0x4000;
const TX_FRAME_BUF: u64 = 0x5000;
const RX_DESC_TABLE_BASE: u64 = 0x6000;
const RX_AVAIL_RING_BASE: u64 = 0x7000;
const RX_USED_RING_BASE: u64 = 0x8000;
const RX_BUF: u64 = 0x9000;

struct TestLayout {
    tx_desc: u64,
    tx_avail: u64,
    tx_used: u64,
    // `dead_code` allow: populated by `test_layout()` for
    // symmetry with the rest of the layout constants. No
    // current test reads this through the struct (callers
    // reference `TX_HEADER_BUF` directly when needed). Kept
    // so the struct remains a complete inventory of
    // guest-memory regions for tests that grow into using it.
    #[allow(dead_code)]
    tx_hdr_buf: u64,
    tx_frame_buf: u64,
    rx_desc: u64,
    rx_avail: u64,
    rx_used: u64,
    rx_buf: u64,
}

fn test_layout() -> TestLayout {
    TestLayout {
        tx_desc: TX_DESC_TABLE_BASE,
        tx_avail: TX_AVAIL_RING_BASE,
        tx_used: TX_USED_RING_BASE,
        tx_hdr_buf: TX_HEADER_BUF,
        tx_frame_buf: TX_FRAME_BUF,
        rx_desc: RX_DESC_TABLE_BASE,
        rx_avail: RX_AVAIL_RING_BASE,
        rx_used: RX_USED_RING_BASE,
        rx_buf: RX_BUF,
    }
}

fn build_test_memory() -> (GuestMemoryMmap, TestLayout) {
    let mem = GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), GUEST_MEM_SIZE)]).unwrap();
    let layout = test_layout();
    (mem, layout)
}

fn program_queues(dev: &mut VirtioNet, layout: &TestLayout) {
    // RX queue (idx 0)
    write_reg(dev, VIRTIO_MMIO_QUEUE_SEL, RXQ as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_NUM, 4);
    write_reg(dev, VIRTIO_MMIO_QUEUE_DESC_LOW, layout.rx_desc as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_AVAIL_LOW, layout.rx_avail as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_USED_LOW, layout.rx_used as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_READY, 1);
    // TX queue (idx 1)
    write_reg(dev, VIRTIO_MMIO_QUEUE_SEL, TXQ as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_NUM, 4);
    write_reg(dev, VIRTIO_MMIO_QUEUE_DESC_LOW, layout.tx_desc as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_AVAIL_LOW, layout.tx_avail as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_USED_LOW, layout.tx_used as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_READY, 1);
}

/// virtio split-ring descriptor layout (16 bytes per entry):
/// addr u64 | len u32 | flags u16 | next u16
fn write_desc(
    mem: &GuestMemoryMmap,
    table_base: u64,
    idx: u16,
    addr: u64,
    len: u32,
    flags: u16,
    next: u16,
) {
    let off = table_base + (idx as u64) * 16;
    let mut buf = [0u8; 16];
    buf[0..8].copy_from_slice(&addr.to_le_bytes());
    buf[8..12].copy_from_slice(&len.to_le_bytes());
    buf[12..14].copy_from_slice(&flags.to_le_bytes());
    buf[14..16].copy_from_slice(&next.to_le_bytes());
    mem.write_slice(&buf, GuestAddress(off)).unwrap();
}

/// virtio split-ring avail layout: flags u16 | idx u16 | ring[u16; N]
fn publish_avail(mem: &GuestMemoryMmap, avail_base: u64, head_idx: u16, ring_pos: u16) {
    // Publish the head descriptor at ring[ring_pos], then update idx.
    let ring_off = avail_base + 4 + (ring_pos as u64) * 2;
    mem.write_slice(&head_idx.to_le_bytes(), GuestAddress(ring_off))
        .unwrap();
    let idx_off = avail_base + 2;
    mem.write_slice(&(ring_pos + 1).to_le_bytes(), GuestAddress(idx_off))
        .unwrap();
}

fn payload_42_bytes() -> Vec<u8> {
    (0..42u8).collect()
}

/// Place one TX chain: a single descriptor covering [12-byte header,
/// `payload`-byte frame] back-to-back. Mirrors the kernel's
/// `can_push` path where the header lives in the same buffer as the
/// skb data.
fn place_tx_chain(mem: &GuestMemoryMmap, layout: &TestLayout, payload: &[u8]) {
    // Write the 12-byte zero header + payload contiguously starting
    // at TX_FRAME_BUF.
    let zero_hdr = [0u8; VIRTIO_NET_HDR_LEN];
    mem.write_slice(&zero_hdr, GuestAddress(layout.tx_frame_buf))
        .unwrap();
    mem.write_slice(
        payload,
        GuestAddress(layout.tx_frame_buf + VIRTIO_NET_HDR_LEN as u64),
    )
    .unwrap();
    // Single read-only descriptor (flags=0, no NEXT) covering both.
    let total = (VIRTIO_NET_HDR_LEN + payload.len()) as u32;
    write_desc(mem, layout.tx_desc, 0, layout.tx_frame_buf, total, 0, 0);
    publish_avail(mem, layout.tx_avail, 0, 0);
}

/// Place one RX chain: a single write-only descriptor at RX_BUF
/// covering 64 bytes (large enough for the 12-byte header + ~52 byte
/// payload).
fn place_rx_chain(mem: &GuestMemoryMmap, layout: &TestLayout) {
    // VRING_DESC_F_WRITE = 2: device-writable.
    write_desc(mem, layout.rx_desc, 0, layout.rx_buf, 256, 2, 0);
    publish_avail(mem, layout.rx_avail, 0, 0);
}

#[test]
fn loopback_delivers_tx_payload_to_rx_with_zero_header() {
    let (mem, layout) = build_test_memory();
    let mut dev = VirtioNet::new(NetConfig::default());
    dev.set_mem(mem.clone());
    init_until_features_ok(&mut dev);
    program_queues(&mut dev, &layout);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);

    let payload = payload_42_bytes();
    place_tx_chain(&mem, &layout, &payload);
    place_rx_chain(&mem, &layout);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, TXQ as u32);

    // Read back the RX buffer: 12-byte virtio header + payload.
    // Header layout per `virtio_net_hdr_v1` (12 bytes): bytes 0..10
    // zero (no GSO/csum/data-valid), bytes 10..12 LE u16 = 1
    // (`num_buffers`). num_buffers MUST be 1 because we don't
    // negotiate VIRTIO_NET_F_MRG_RXBUF — a num_buffers=0 header
    // would make virtnet_receive_mergeable wait for a non-existent
    // continuation buffer.
    let mut delivered = vec![0u8; VIRTIO_NET_HDR_LEN + payload.len()];
    mem.read_slice(&mut delivered, GuestAddress(layout.rx_buf))
        .unwrap();
    let mut expected_hdr = [0u8; VIRTIO_NET_HDR_LEN];
    expected_hdr[10] = 1;
    expected_hdr[11] = 0;
    assert_eq!(
        &delivered[..VIRTIO_NET_HDR_LEN],
        &expected_hdr,
        "RX virtio header must be zero-filled with num_buffers=1 LE u16 at offset 10"
    );
    assert_eq!(
        &delivered[VIRTIO_NET_HDR_LEN..],
        payload.as_slice(),
        "RX frame bytes must match the TX payload"
    );

    let counters = dev.counters();
    assert_eq!(counters.tx_packets(), 1);
    assert_eq!(counters.rx_packets(), 1);
    assert_eq!(counters.tx_bytes(), payload.len() as u64);
    assert_eq!(counters.rx_bytes(), payload.len() as u64);
    assert_eq!(counters.tx_chain_invalid(), 0);
    assert_eq!(counters.rx_chain_invalid(), 0);
    assert_eq!(counters.tx_dropped_no_rx_buffer(), 0);
}

#[test]
fn loopback_drops_tx_when_rx_queue_empty() {
    let (mem, layout) = build_test_memory();
    let mut dev = VirtioNet::new(NetConfig::default());
    dev.set_mem(mem.clone());
    init_until_features_ok(&mut dev);
    program_queues(&mut dev, &layout);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);

    let payload = payload_42_bytes();
    place_tx_chain(&mem, &layout, &payload);
    // Do NOT post an RX buffer.

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, TXQ as u32);

    let counters = dev.counters();
    // Counter taxonomy: TX add_used succeeded (the chain was
    // popped, parsed, and marked used) — `tx_packets` advances.
    // The RX half found no buffer to deliver into, so
    // `tx_dropped_no_rx_buffer` advances and `rx_packets` stays
    // at zero. The conservation identity holds:
    //   tx_packets (1) = rx_packets (0) + tx_dropped_no_rx_buffer (1)
    //                  + rx_chain_invalid (0) + rx_add_used_failures (0)
    assert_eq!(
        counters.tx_packets(),
        1,
        "TX add_used succeeded → tx_packets bumps"
    );
    assert_eq!(counters.tx_bytes(), payload.len() as u64);
    assert_eq!(counters.rx_packets(), 0, "no RX delivery when queue empty");
    assert_eq!(counters.rx_bytes(), 0);
    assert_eq!(
        counters.tx_dropped_no_rx_buffer(),
        1,
        "must record TX drop when RX queue empty"
    );
    assert_eq!(
        counters.rx_chain_invalid(),
        0,
        "queue was empty, not malformed"
    );
    assert_eq!(counters.tx_add_used_failures(), 0);
    assert_eq!(counters.rx_add_used_failures(), 0);
    // The TX add_used DID advance the used-ring (TX completion is
    // observable to the guest), so a single irqfd kick fires at
    // the end of the drain. The guest's NAPI / virtqueue thread
    // wakes to observe the TX completion, even though no RX
    // delivery happened.
    let kicks = dev.irq_evt().read().unwrap_or(0);
    assert_eq!(kicks, 1, "TX add_used advance must produce one irqfd kick");
}

#[test]
fn tx_chain_with_only_header_produces_zero_frame_loopback() {
    // Edge case: TX chain whose total length is exactly 12 bytes
    // (just the virtio header, no L2 frame). Device should:
    //   - capture frame_len = 0
    //   - deliver an RX chain with 12-byte zero header + 0 payload
    //   - count it as one tx_packet / rx_packet with 0 bytes
    let (mem, layout) = build_test_memory();
    let mut dev = VirtioNet::new(NetConfig::default());
    dev.set_mem(mem.clone());
    init_until_features_ok(&mut dev);
    program_queues(&mut dev, &layout);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);

    // Single 12-byte descriptor — header only.
    let zero_hdr = [0u8; VIRTIO_NET_HDR_LEN];
    mem.write_slice(&zero_hdr, GuestAddress(layout.tx_frame_buf))
        .unwrap();
    write_desc(
        &mem,
        layout.tx_desc,
        0,
        layout.tx_frame_buf,
        VIRTIO_NET_HDR_LEN as u32,
        0,
        0,
    );
    publish_avail(&mem, layout.tx_avail, 0, 0);
    place_rx_chain(&mem, &layout);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, TXQ as u32);

    let counters = dev.counters();
    assert_eq!(counters.tx_packets(), 1);
    assert_eq!(counters.rx_packets(), 1);
    assert_eq!(counters.tx_bytes(), 0);
    assert_eq!(counters.rx_bytes(), 0);
    assert_eq!(counters.tx_chain_invalid(), 0);
}

#[test]
fn tx_chain_shorter_than_header_marked_invalid() {
    // Chain of 8 bytes total — less than the 12-byte virtio header.
    // Per virtio-v1.2 §5.1.6.5 the driver MUST emit the full header,
    // so this is a guest protocol violation. Device drops the chain
    // (counts tx_chain_invalid) and still marks the head used so the
    // guest doesn't hang.
    let (mem, layout) = build_test_memory();
    let mut dev = VirtioNet::new(NetConfig::default());
    dev.set_mem(mem.clone());
    init_until_features_ok(&mut dev);
    program_queues(&mut dev, &layout);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);

    write_desc(&mem, layout.tx_desc, 0, layout.tx_frame_buf, 8, 0, 0);
    publish_avail(&mem, layout.tx_avail, 0, 0);
    place_rx_chain(&mem, &layout);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, TXQ as u32);

    let counters = dev.counters();
    assert_eq!(counters.tx_chain_invalid(), 1);
    assert_eq!(counters.tx_packets(), 0);
}

#[test]
fn tx_chain_with_write_only_descriptor_marked_invalid() {
    // TX chain with a write-only descriptor — RX-direction violation.
    // VRING_DESC_F_WRITE = 2.
    let (mem, layout) = build_test_memory();
    let mut dev = VirtioNet::new(NetConfig::default());
    dev.set_mem(mem.clone());
    init_until_features_ok(&mut dev);
    program_queues(&mut dev, &layout);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);

    write_desc(
        &mem,
        layout.tx_desc,
        0,
        layout.tx_frame_buf,
        100,
        2, // F_WRITE — wrong direction for TX
        0,
    );
    publish_avail(&mem, layout.tx_avail, 0, 0);
    place_rx_chain(&mem, &layout);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, TXQ as u32);

    let counters = dev.counters();
    assert_eq!(counters.tx_chain_invalid(), 1);
    assert_eq!(counters.tx_packets(), 0);
}

#[test]
fn rx_chain_with_read_only_descriptor_marked_invalid() {
    // RX chain with a read-only descriptor — TX-direction violation
    // on the receive side. RX descriptors must be device-writable.
    let (mem, layout) = build_test_memory();
    let mut dev = VirtioNet::new(NetConfig::default());
    dev.set_mem(mem.clone());
    init_until_features_ok(&mut dev);
    program_queues(&mut dev, &layout);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);

    place_tx_chain(&mem, &layout, &payload_42_bytes());
    // RX descriptor with flags=0 (read-only) — wrong direction.
    write_desc(&mem, layout.rx_desc, 0, layout.rx_buf, 256, 0, 0);
    publish_avail(&mem, layout.rx_avail, 0, 0);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, TXQ as u32);

    let counters = dev.counters();
    // RX direction violation routes to rx_chain_invalid ONLY (no
    // double-bump on tx_dropped_no_rx_buffer — a chain WAS popped,
    // just rejected for shape; the queue was not empty). The
    // failure-classification taxonomy stays 1:1 with events.
    assert_eq!(
        counters.rx_chain_invalid(),
        1,
        "RX direction violation must be flagged"
    );
    // Mutually-exclusive partner counter `rx_write_failed` MUST
    // stay at zero — a read-only descriptor is a chain-shape
    // rejection, not a guest-memory write failure. A regression
    // that bumped both counters on the same chain would violate
    // the per-event 1:1 taxonomy.
    assert_eq!(
        counters.rx_write_failed(),
        0,
        "shape rejection must NOT also bump rx_write_failed",
    );
    assert_eq!(
        counters.tx_dropped_no_rx_buffer(),
        0,
        "must NOT also bump tx_dropped_no_rx_buffer (RX queue was non-empty, just malformed)",
    );
    // tx_packets advances because TX add_used succeeded; rx_packets
    // does NOT advance because the RX delivery itself failed.
    assert_eq!(
        counters.tx_packets(),
        1,
        "TX add_used succeeded → tx_packets bumps"
    );
    assert_eq!(counters.rx_packets(), 0, "no successful RX delivery");
    assert_eq!(
        counters.rx_add_used_failures(),
        0,
        "RX add_used succeeded (recycled with len=0)"
    );
    // The malformed-RX path issued add_used(head, 0) successfully
    // to recycle the descriptor — that advances the used-ring, so
    // the guest's NAPI must wake to observe the empty completion.
    // Plus the TX add_used also advanced the TX used-ring. Either
    // alone would suffice; both together still produce one
    // coalesced kick (counter-mode eventfd).
    let kicks = dev.irq_evt().read().unwrap_or(0);
    assert_eq!(
        kicks, 1,
        "used-ring advance from RX recycle + TX completion must trigger one coalesced kick",
    );
}

#[test]
fn rx_chain_with_unmapped_gpa_bumps_rx_write_failed() {
    // RX chain whose shape is valid (write-only flag set) but
    // whose `addr` points BEYOND the guest-memory region. The
    // header `write_slice` returns Err on the first attempt, so
    // the descriptor walk hits the GPA-write-failure path inside
    // `try_loopback_to_rx`. The split counter must route this to
    // `rx_write_failed`, NOT `rx_chain_invalid`. This is the
    // direct regression guard for the counter split: chain-shape
    // rejection and GPA write failure are now distinct event
    // counters because operators reading a failure dump need to
    // tell "guest violated the RX descriptor-direction rule"
    // from "guest posted a buffer at an unmapped GPA".
    let (mem, layout) = build_test_memory();
    let mut dev = VirtioNet::new(NetConfig::default());
    dev.set_mem(mem.clone());
    init_until_features_ok(&mut dev);
    program_queues(&mut dev, &layout);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);

    place_tx_chain(&mem, &layout, &payload_42_bytes());
    // Write-only descriptor (correct shape) pointing past the
    // 1 MiB guest-memory region (`GUEST_MEM_SIZE = 0x10_0000`).
    // Capacity 256 bytes is large enough that the descriptor's
    // `addr + take` doesn't overflow u64 — so the chain is NOT
    // shape-rejected via the address-overflow arm. The
    // `write_slice` for the 12-byte header is the first observable
    // failure: the GPA is unmapped, vm-memory's `write_slice`
    // returns Err. VRING_DESC_F_WRITE = 2.
    let unmapped_gpa: u64 = (GUEST_MEM_SIZE as u64) + 0x1000;
    write_desc(&mem, layout.rx_desc, 0, unmapped_gpa, 256, 2, 0);
    publish_avail(&mem, layout.rx_avail, 0, 0);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, TXQ as u32);

    let counters = dev.counters();
    assert_eq!(
        counters.rx_write_failed(),
        1,
        "GPA-unmapped header write must bump rx_write_failed",
    );
    // Mutually-exclusive partner counter MUST stay at zero —
    // chain shape was acceptable (write-only flag set, address
    // didn't overflow). Routing to BOTH counters would violate
    // the per-event 1:1 taxonomy and mislead operators.
    assert_eq!(
        counters.rx_chain_invalid(),
        0,
        "GPA write failure must NOT also bump rx_chain_invalid \
         (chain shape was valid)",
    );
    // `tx_dropped_no_rx_buffer` MUST also stay at zero — the RX
    // queue was non-empty, just write-broken. Same mutual-
    // exclusion rule the read-only-descriptor test pins.
    assert_eq!(
        counters.tx_dropped_no_rx_buffer(),
        0,
        "RX queue was non-empty (just write-broken) — must NOT \
         bump tx_dropped_no_rx_buffer",
    );
    // tx_packets advances because the TX add_used path
    // succeeded; rx_packets does NOT because the RX delivery
    // failed. Same shape as `rx_chain_with_read_only_descriptor_*`.
    assert_eq!(
        counters.tx_packets(),
        1,
        "TX add_used succeeded → tx_packets bumps",
    );
    assert_eq!(counters.rx_packets(), 0, "no successful RX delivery");
    // The recycle add_used(head, 0) succeeded (the RX queue's
    // used ring is in mapped memory; only the descriptor's
    // payload GPA is unmapped). So rx_add_used_failures stays
    // at zero and the used-ring advance kicks the guest.
    assert_eq!(
        counters.rx_add_used_failures(),
        0,
        "RX recycle add_used succeeded — used-ring is in mapped \
         memory, only the descriptor's payload GPA was unmapped",
    );
    let kicks = dev.irq_evt().read().unwrap_or(0);
    assert_eq!(
        kicks, 1,
        "TX completion + RX recycle used-ring advances coalesce \
         into a single irqfd kick",
    );
}

#[test]
fn rx_chain_with_unmapped_gpa_on_frame_bumps_rx_write_failed() {
    // Variant of the GPA-unmapped test where the HEADER
    // write_slice succeeds (descriptor #0 is mapped + 12 bytes
    // long) but the FRAME write_slice fails on a SECOND
    // descriptor pointing at unmapped memory. Exercises the
    // frame-walk write-failure arm distinctly from the
    // header-walk arm; both arms must route to rx_write_failed,
    // never to rx_chain_invalid.
    let (mem, layout) = build_test_memory();
    let mut dev = VirtioNet::new(NetConfig::default());
    dev.set_mem(mem.clone());
    init_until_features_ok(&mut dev);
    program_queues(&mut dev, &layout);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);

    place_tx_chain(&mem, &layout, &payload_42_bytes());
    // Two-descriptor chain. desc 0: mapped, write-only, exactly
    // 12 bytes (the virtio header lands here cleanly). desc 1:
    // write-only, NEXT bit clear (terminator), points at
    // unmapped GPA — frame write_slice fails. VRING_DESC_F_NEXT
    // = 1, VRING_DESC_F_WRITE = 2.
    let unmapped_gpa: u64 = (GUEST_MEM_SIZE as u64) + 0x1000;
    write_desc(
        &mem,
        layout.rx_desc,
        0,
        layout.rx_buf,
        VIRTIO_NET_HDR_LEN as u32,
        1 | 2, // F_NEXT | F_WRITE
        1,     // next = 1
    );
    write_desc(&mem, layout.rx_desc, 1, unmapped_gpa, 256, 2, 0);
    publish_avail(&mem, layout.rx_avail, 0, 0);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, TXQ as u32);

    let counters = dev.counters();
    assert_eq!(
        counters.rx_write_failed(),
        1,
        "frame-walk write_slice failure must bump rx_write_failed",
    );
    assert_eq!(
        counters.rx_chain_invalid(),
        0,
        "frame-walk write failure must NOT bump rx_chain_invalid \
         (chain shape was valid)",
    );
    assert_eq!(counters.rx_packets(), 0, "no successful RX delivery");
    // tx_packets bumps because TX add_used succeeded.
    assert_eq!(counters.tx_packets(), 1);
}

#[test]
fn rx_write_failed_initially_zero() {
    // Pin the new counter's initial state. Distinct from
    // rx_chain_invalid; operators monitoring guest-memory write
    // breakage need to see this counter at zero on a healthy
    // device.
    let dev = VirtioNet::new(NetConfig::default());
    let counters = dev.counters();
    assert_eq!(counters.rx_write_failed(), 0);
}

#[test]
fn loopback_two_frames_in_one_kick() {
    // Two TX chains queued; one QUEUE_NOTIFY drains both. Verify
    // both deliver to two RX buffers and counters reflect 2 packets.
    let (mem, layout) = build_test_memory();
    let mut dev = VirtioNet::new(NetConfig::default());
    dev.set_mem(mem.clone());
    init_until_features_ok(&mut dev);
    program_queues(&mut dev, &layout);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);

    // TX chain 0: header+payload at offset 0x5000
    let payload0: Vec<u8> = (10..30u8).collect();
    let zero_hdr = [0u8; VIRTIO_NET_HDR_LEN];
    mem.write_slice(&zero_hdr, GuestAddress(layout.tx_frame_buf))
        .unwrap();
    mem.write_slice(
        &payload0,
        GuestAddress(layout.tx_frame_buf + VIRTIO_NET_HDR_LEN as u64),
    )
    .unwrap();
    write_desc(
        &mem,
        layout.tx_desc,
        0,
        layout.tx_frame_buf,
        (VIRTIO_NET_HDR_LEN + payload0.len()) as u32,
        0,
        0,
    );

    // TX chain 1: separate buffer at 0x5800 (still inside guest mem)
    let chain1_buf = layout.tx_frame_buf + 0x800;
    let payload1: Vec<u8> = (50..70u8).collect();
    mem.write_slice(&zero_hdr, GuestAddress(chain1_buf))
        .unwrap();
    mem.write_slice(
        &payload1,
        GuestAddress(chain1_buf + VIRTIO_NET_HDR_LEN as u64),
    )
    .unwrap();
    write_desc(
        &mem,
        layout.tx_desc,
        1,
        chain1_buf,
        (VIRTIO_NET_HDR_LEN + payload1.len()) as u32,
        0,
        0,
    );

    // Publish both heads.
    let avail_idx_off = layout.tx_avail + 2;
    let ring_off = layout.tx_avail + 4;
    mem.write_slice(&0u16.to_le_bytes(), GuestAddress(ring_off))
        .unwrap();
    mem.write_slice(&1u16.to_le_bytes(), GuestAddress(ring_off + 2))
        .unwrap();
    mem.write_slice(&2u16.to_le_bytes(), GuestAddress(avail_idx_off))
        .unwrap();

    // Two RX buffers at offsets 0x9000 and 0x9400.
    write_desc(&mem, layout.rx_desc, 0, layout.rx_buf, 256, 2, 0);
    write_desc(&mem, layout.rx_desc, 1, layout.rx_buf + 0x400, 256, 2, 0);
    let avail_idx_off = layout.rx_avail + 2;
    let ring_off = layout.rx_avail + 4;
    mem.write_slice(&0u16.to_le_bytes(), GuestAddress(ring_off))
        .unwrap();
    mem.write_slice(&1u16.to_le_bytes(), GuestAddress(ring_off + 2))
        .unwrap();
    mem.write_slice(&2u16.to_le_bytes(), GuestAddress(avail_idx_off))
        .unwrap();

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, TXQ as u32);

    let counters = dev.counters();
    assert_eq!(counters.tx_packets(), 2);
    assert_eq!(counters.rx_packets(), 2);
    assert_eq!(
        counters.tx_bytes(),
        (payload0.len() + payload1.len()) as u64
    );

    // Verify RX0 holds payload0, RX1 holds payload1.
    let mut rx0 = vec![0u8; VIRTIO_NET_HDR_LEN + payload0.len()];
    mem.read_slice(&mut rx0, GuestAddress(layout.rx_buf))
        .unwrap();
    assert_eq!(&rx0[VIRTIO_NET_HDR_LEN..], payload0.as_slice());

    let mut rx1 = vec![0u8; VIRTIO_NET_HDR_LEN + payload1.len()];
    mem.read_slice(&mut rx1, GuestAddress(layout.rx_buf + 0x400))
        .unwrap();
    assert_eq!(&rx1[VIRTIO_NET_HDR_LEN..], payload1.as_slice());
}

#[test]
fn loopback_emits_single_irqfd_kick_for_drain() {
    // Multiple TX chains in one drain produce ONE irqfd write
    // (counter-mode coalescing). The test reads the eventfd and
    // checks the value.
    let (mem, layout) = build_test_memory();
    let mut dev = VirtioNet::new(NetConfig::default());
    dev.set_mem(mem.clone());
    init_until_features_ok(&mut dev);
    program_queues(&mut dev, &layout);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);

    place_tx_chain(&mem, &layout, &payload_42_bytes());
    place_rx_chain(&mem, &layout);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, TXQ as u32);
    let kick_count = dev.irq_evt().read().unwrap();
    assert_eq!(
        kick_count, 1,
        "single drain must produce exactly one irqfd write"
    );
}

#[test]
fn reset_clears_state_but_preserves_counters() {
    let (mem, layout) = build_test_memory();
    let mut dev = VirtioNet::new(NetConfig::default());
    dev.set_mem(mem.clone());
    init_until_features_ok(&mut dev);
    program_queues(&mut dev, &layout);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);
    place_tx_chain(&mem, &layout, &payload_42_bytes());
    place_rx_chain(&mem, &layout);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, TXQ as u32);
    assert_eq!(dev.counters().tx_packets(), 1);
    // Reset.
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, 0);
    assert_eq!(read_reg(&dev, VIRTIO_MMIO_STATUS), 0);
    assert_eq!(
        dev.counters().tx_packets(),
        1,
        "counters survive reset (operator-observability invariant)",
    );
}

// ---------------------------------------------------------------------------
// MMIO size + dispatch boundaries
// ---------------------------------------------------------------------------

#[test]
fn non_4byte_register_read_returns_ff() {
    let dev = VirtioNet::new(NetConfig::default());
    let mut buf = [0u8; 2];
    dev.mmio_read(0, &mut buf);
    assert_eq!(buf, [0xff, 0xff]);
}

#[test]
fn non_4byte_register_write_ignored() {
    let mut dev = VirtioNet::new(NetConfig::default());
    dev.mmio_write(VIRTIO_MMIO_STATUS as u64, &[0x01, 0x00]);
    assert_eq!(read_reg(&dev, VIRTIO_MMIO_STATUS), 0);
}

#[test]
fn unknown_register_returns_zero() {
    let dev = VirtioNet::new(NetConfig::default());
    // 0xC0 is in register space (< 0x100) and not a valid register.
    assert_eq!(read_reg(&dev, 0xC0), 0);
}

#[test]
fn unknown_register_write_ignored() {
    let mut dev = VirtioNet::new(NetConfig::default());
    write_reg(&mut dev, 0xC0, 0xDEAD);
    assert_eq!(read_reg(&dev, VIRTIO_MMIO_STATUS), 0);
}

// ---------------------------------------------------------------------------
// F4: add_used counter taxonomy
// ---------------------------------------------------------------------------

#[test]
fn tx_add_used_failures_initially_zero() {
    // Pin the new counters' initial state. Distinct from
    // tx_chain_invalid (chain-shape rejection); operators monitoring
    // queue-state breakage need to see this counter at zero on a
    // healthy device.
    let dev = VirtioNet::new(NetConfig::default());
    let counters = dev.counters();
    assert_eq!(counters.tx_add_used_failures(), 0);
    assert_eq!(counters.rx_add_used_failures(), 0);
}

#[test]
fn tx_add_used_failures_distinct_from_tx_chain_invalid() {
    // F4: a malformed-chain rejection bumps tx_chain_invalid but
    // NOT tx_add_used_failures. The two counters describe distinct
    // failure modes (chain shape vs queue state) and an operator
    // reading them must see them advance independently.
    let (mem, layout) = build_test_memory();
    let mut dev = VirtioNet::new(NetConfig::default());
    dev.set_mem(mem.clone());
    init_until_features_ok(&mut dev);
    program_queues(&mut dev, &layout);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);

    // Submit a TX chain with a write-only descriptor (wrong direction).
    write_desc(&mem, layout.tx_desc, 0, layout.tx_frame_buf, 100, 2, 0);
    publish_avail(&mem, layout.tx_avail, 0, 0);
    place_rx_chain(&mem, &layout);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, TXQ as u32);

    let counters = dev.counters();
    assert_eq!(
        counters.tx_chain_invalid(),
        1,
        "chain-shape rejection counted"
    );
    assert_eq!(
        counters.tx_add_used_failures(),
        0,
        "add_used succeeded; queue-state counter must NOT bump",
    );
    assert_eq!(counters.rx_add_used_failures(), 0);
}

// ---------------------------------------------------------------------------
// F7: feature-subset rule on FEATURES_OK
// ---------------------------------------------------------------------------

#[test]
fn features_ok_rejected_when_driver_accepts_unoffered_bit() {
    // F7 (subset rule, virtio-v1.2 §2.2.1): the driver MUST NOT
    // accept any feature bit that the device did not offer. Our
    // device only offers VIRTIO_F_VERSION_1 + VIRTIO_NET_F_MAC.
    // A guest that accepts VIRTIO_NET_F_MQ (=22, not offered)
    // should hit the FAILED bit and the FEATURES_OK transition
    // should be rejected.
    use virtio_bindings::virtio_config::VIRTIO_CONFIG_S_FAILED;
    use virtio_bindings::virtio_net::VIRTIO_NET_F_MQ;
    let mut dev = VirtioNet::new(NetConfig::default());
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
    // Negotiate VERSION_1 (so the VERSION_1 gate would not fire) +
    // F_MQ (which we did NOT offer) — the subset gate must catch this.
    write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
    write_reg(
        &mut dev,
        VIRTIO_MMIO_DRIVER_FEATURES,
        1u32 << VIRTIO_NET_F_MQ,
    );
    write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
    write_reg(
        &mut dev,
        VIRTIO_MMIO_DRIVER_FEATURES,
        1u32 << (VIRTIO_F_VERSION_1 - 32),
    );
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_FEAT);
    let status = read_reg(&dev, VIRTIO_MMIO_STATUS);
    assert_eq!(
        status & VIRTIO_CONFIG_S_FEATURES_OK,
        0,
        "FEATURES_OK must NOT be set when driver accepts an unoffered bit",
    );
    assert_ne!(
        status & VIRTIO_CONFIG_S_FAILED,
        0,
        "FAILED bit must be set on subset-rule violation",
    );
}

#[test]
fn features_ok_accepted_with_only_offered_bits() {
    // Positive control for the subset-rule gate: with VERSION_1 +
    // F_MAC (both offered) and nothing else, FEATURES_OK should be
    // accepted with no FAILED bit.
    use virtio_bindings::virtio_config::VIRTIO_CONFIG_S_FAILED;
    use virtio_bindings::virtio_net::VIRTIO_NET_F_MAC;
    let mut dev = VirtioNet::new(NetConfig::default());
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
    write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
    write_reg(
        &mut dev,
        VIRTIO_MMIO_DRIVER_FEATURES,
        1u32 << VIRTIO_NET_F_MAC,
    );
    write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
    write_reg(
        &mut dev,
        VIRTIO_MMIO_DRIVER_FEATURES,
        1u32 << (VIRTIO_F_VERSION_1 - 32),
    );
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_FEAT);
    let status = read_reg(&dev, VIRTIO_MMIO_STATUS);
    assert_ne!(
        status & VIRTIO_CONFIG_S_FEATURES_OK,
        0,
        "FEATURES_OK must be set when driver accepts only offered bits",
    );
    assert_eq!(
        status & VIRTIO_CONFIG_S_FAILED,
        0,
        "FAILED bit must NOT be set on a clean subset",
    );
}

// ---------------------------------------------------------------------------
// G4: regression test for the GPA-overflow fix (F3)
// ---------------------------------------------------------------------------

#[test]
fn tx_chain_with_address_overflow_dropped_gracefully() {
    // F3 regression guard: a TX descriptor whose `addr + len` would
    // overflow `u64` MUST drop the chain rather than panic. Our
    // captured-frame loop's `desc_addr.checked_add(skip as u64)`
    // returns None on overflow; this test wires up a descriptor
    // near `u64::MAX` to hit that path and asserts the device
    // counts a tx_chain_invalid event without panicking.
    //
    // The descriptor's read_slice WOULD also fail (the GPA is
    // outside guest memory), but the test is robust to either
    // outcome path: the device must end with the chain invalid
    // counter bumped and tx_packets at zero. Without the
    // checked_add fix, this test would have panicked the test
    // thread via .expect(), which Rust's test harness catches and
    // reports as failure — so a future revert of F3 trips here.
    let (mem, layout) = build_test_memory();
    let mut dev = VirtioNet::new(NetConfig::default());
    dev.set_mem(mem.clone());
    init_until_features_ok(&mut dev);
    program_queues(&mut dev, &layout);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);

    // Descriptor at addr = u64::MAX - 11, len = 24. The first 12
    // bytes are the virtio header (skip path → checked_add(skip)
    // tries u64::MAX - 11 + 12 = wrap). desc.len cap of TX_DESC_MAX
    // doesn't matter here — overflow is on addr arithmetic.
    write_desc(
        &mem,
        layout.tx_desc,
        0,
        u64::MAX - 11,
        24, // 12-byte hdr + 12-byte payload
        0,  // read-only (TX direction)
        0,
    );
    publish_avail(&mem, layout.tx_avail, 0, 0);
    place_rx_chain(&mem, &layout);

    // Must not panic.
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, TXQ as u32);

    let counters = dev.counters();
    assert_eq!(
        counters.tx_chain_invalid(),
        1,
        "GPA-overflow chain must bump tx_chain_invalid (graceful drop)",
    );
    assert_eq!(counters.tx_packets(), 0, "no TX completion on overflow");
    assert_eq!(counters.rx_packets(), 0, "no RX delivery on dropped chain");
}

// ---------------------------------------------------------------------------
// G12: regression test for the DRIVER_OK gate (F1)
// ---------------------------------------------------------------------------

#[test]
fn tx_kick_before_driver_ok_ignored() {
    // F1 regression guard: a guest that writes QUEUE_NOTIFY before
    // DRIVER_OK has been set MUST be ignored — the device is not
    // yet authorised to process virtqueue requests per virtio-v1.2
    // §2.1.2. Our guard at the top of process_tx_loopback returns
    // early on `device_status & DRIVER_OK == 0`. Without it, the
    // pop_descriptor_chain path would happily drain a queue whose
    // addresses were programmed in the FEATURES_OK..DRIVER_OK
    // window. This test posts a chain, kicks BEFORE DRIVER_OK,
    // asserts every counter stays at zero, then writes DRIVER_OK
    // and re-kicks to verify the device resumes.
    let (mem, layout) = build_test_memory();
    let mut dev = VirtioNet::new(NetConfig::default());
    dev.set_mem(mem.clone());
    init_until_features_ok(&mut dev);
    // Note: program_queues runs in init's window between
    // FEATURES_OK and DRIVER_OK, which is the only legal window
    // for queue config. We program the queues but DO NOT advance
    // status to DRIVER_OK yet.
    program_queues(&mut dev, &layout);

    place_tx_chain(&mem, &layout, &payload_42_bytes());
    place_rx_chain(&mem, &layout);

    // Kick before DRIVER_OK. Device must ignore.
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, TXQ as u32);

    let counters = dev.counters();
    assert_eq!(
        counters.tx_packets(),
        0,
        "kick before DRIVER_OK must not advance counters"
    );
    assert_eq!(counters.rx_packets(), 0);
    assert_eq!(counters.tx_chain_invalid(), 0);
    assert_eq!(counters.rx_chain_invalid(), 0);
    assert_eq!(counters.tx_dropped_no_rx_buffer(), 0);
    assert_eq!(counters.tx_add_used_failures(), 0);
    assert_eq!(counters.rx_add_used_failures(), 0);
    assert!(
        dev.irq_evt().read().is_err(),
        "no irqfd kick when device is pre-DRIVER_OK"
    );

    // Now advance to DRIVER_OK and re-kick. The same chain (still
    // present in the avail ring) should now be processed.
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, TXQ as u32);
    let counters = dev.counters();
    assert_eq!(
        counters.tx_packets(),
        1,
        "post-DRIVER_OK kick processes the queued chain",
    );
}
