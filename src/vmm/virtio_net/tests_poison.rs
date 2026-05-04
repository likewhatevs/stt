//! Hostile-guest avail.idx defense for virtio-net. Mirrors
//! virtio_blk's tests_poison.rs: plant a bogus avail.idx, kick, and
//! assert the poison gate fires (counter bumped, NEEDS_RESET set,
//! irqfd written) and that subsequent kicks against the poisoned
//! queue do NOT re-bump the counter, re-fire the irqfd, or
//! re-flood the host log with the same error line. The harm the
//! gate prevents is per-event-counter taxonomy violation +
//! NEEDS_RESET observability + log spam, NOT unbounded CPU
//! consumption — virtio-net has no enable/disable_notification
//! bracket, so each kick re-trips the error once per MMIO exit
//! and returns. The gate keeps the per-event semantics honest
//! and stops the log from filling up.
//!
//! Signal sequence parity with virtio-blk: poison sets both
//! `VIRTIO_CONFIG_S_NEEDS_RESET` (in device_status) AND
//! `VIRTIO_MMIO_INT_CONFIG` (in interrupt_status), then writes
//! the irqfd. Spec-compliant per virtio-v1.2 (config interrupt
//! paired with NEEDS_RESET) and matches cloud-hypervisor.
//! virtio-net's kernel callback `virtnet_config_changed_work`
//! cread-fails F_STATUS and no-ops, so the INT_CONFIG dispatch
//! costs one harmless guest workqueue wake on device death —
//! accepted cost for spec-compliance and cross-VMM convergence.
//! Tests assert both bits are set on poison.
//!
//! virtio-net has two queues (RX and TX). The kick path
//! (`process_tx_loopback` driven by `mmio_write(QUEUE_NOTIFY, TXQ)`)
//! reads the TX queue first, then the RX queue per chain. A poison
//! event on EITHER queue must short-circuit the drain. Two test
//! cases cover both sides; per-queue independence and signal
//! sequence parity get their own cases.

#![cfg(test)]

use super::device::*;
use crate::vmm::net_config::NetConfig;
use virtio_bindings::virtio_config::{
    VIRTIO_CONFIG_S_NEEDS_RESET, VIRTIO_F_VERSION_1,
};
use virtio_bindings::virtio_mmio::{
    VIRTIO_MMIO_DRIVER_FEATURES, VIRTIO_MMIO_DRIVER_FEATURES_SEL, VIRTIO_MMIO_INT_CONFIG,
    VIRTIO_MMIO_INTERRUPT_STATUS, VIRTIO_MMIO_QUEUE_AVAIL_LOW, VIRTIO_MMIO_QUEUE_DESC_LOW,
    VIRTIO_MMIO_QUEUE_NOTIFY, VIRTIO_MMIO_QUEUE_NUM, VIRTIO_MMIO_QUEUE_READY,
    VIRTIO_MMIO_QUEUE_SEL, VIRTIO_MMIO_QUEUE_USED_LOW, VIRTIO_MMIO_STATUS,
};
use virtio_bindings::virtio_net::VIRTIO_NET_F_MAC;
use vm_memory::{Bytes, GuestAddress, GuestMemoryMmap};

// ---------------------------------------------------------------------------
// Test memory layout — chosen so descriptor tables / avail rings / used
// rings / payload buffers all live at distinct, page-aligned addresses
// inside a 1 MiB guest memory region. Independent of tests.rs/tests_proptest.rs
// addresses so the poison tests are self-contained.
// ---------------------------------------------------------------------------
const GUEST_MEM_SIZE: usize = 0x10_0000; // 1 MiB
const TX_DESC_BASE: u64 = 0x1000;
const TX_AVAIL_BASE: u64 = 0x2000;
const TX_USED_BASE: u64 = 0x3000;
const TX_FRAME_BUF: u64 = 0x4000;
const RX_DESC_BASE: u64 = 0x6000;
const RX_AVAIL_BASE: u64 = 0x7000;
const RX_USED_BASE: u64 = 0x8000;
const RX_BUF: u64 = 0x9000;

/// Per-queue size programmed into the device. Matches the
/// production default `QUEUE_MAX_SIZE` (256) so the poison threshold
/// the kernel driver is up against in real workloads is what the
/// test exercises. A planted `avail.idx = 1000` still trips the
/// `(1000 - 0).0 > 256` check in `AvailIter::new` (queue.rs:707-709)
/// — distance 1000 against the 256 threshold, well over the limit.
const QUEUE_SIZE: u16 = 256;

const VRING_DESC_F_WRITE: u16 = 2;

fn read_reg(dev: &VirtioNet, offset: u32) -> u32 {
    let mut buf = [0u8; 4];
    dev.mmio_read(offset as u64, &mut buf);
    u32::from_le_bytes(buf)
}

fn write_reg(dev: &mut VirtioNet, offset: u32, val: u32) {
    dev.mmio_write(offset as u64, &val.to_le_bytes());
}

/// Drive the device through ACK → DRIVER → negotiate VERSION_1+MAC →
/// FEATURES_OK. Stops short of DRIVER_OK so callers can program queue
/// addresses (the only legal window per virtio-v1.2 §3.1.1).
fn init_until_features_ok(dev: &mut VirtioNet) {
    write_reg(dev, VIRTIO_MMIO_STATUS, S_ACK);
    write_reg(dev, VIRTIO_MMIO_STATUS, S_DRV);
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

fn program_queues(dev: &mut VirtioNet) {
    // RX queue (idx 0)
    write_reg(dev, VIRTIO_MMIO_QUEUE_SEL, RXQ as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_NUM, QUEUE_SIZE as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_DESC_LOW, RX_DESC_BASE as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_AVAIL_LOW, RX_AVAIL_BASE as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_USED_LOW, RX_USED_BASE as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_READY, 1);
    // TX queue (idx 1)
    write_reg(dev, VIRTIO_MMIO_QUEUE_SEL, TXQ as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_NUM, QUEUE_SIZE as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_DESC_LOW, TX_DESC_BASE as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_AVAIL_LOW, TX_AVAIL_BASE as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_USED_LOW, TX_USED_BASE as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_READY, 1);
}

fn build_fixture() -> (VirtioNet, GuestMemoryMmap) {
    let mem =
        GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), GUEST_MEM_SIZE)])
            .expect("create poison-test guest mem");
    let mut dev = VirtioNet::new(NetConfig::default());
    dev.set_mem(mem.clone());
    init_until_features_ok(&mut dev);
    program_queues(&mut dev);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);
    (dev, mem)
}

/// virtio split-ring descriptor layout: addr u64 | len u32 | flags u16 | next u16
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
    mem.write_slice(&buf, GuestAddress(off))
        .expect("plant descriptor");
}

/// Place one well-formed TX chain so `pop_descriptor_chain` would
/// return a chain (if not for the poisoned avail.idx). The chain
/// covers a 12-byte virtio header + 12-byte payload at TX_FRAME_BUF.
fn place_tx_chain(mem: &GuestMemoryMmap) {
    let zero_hdr = [0u8; VIRTIO_NET_HDR_LEN];
    mem.write_slice(&zero_hdr, GuestAddress(TX_FRAME_BUF)).unwrap();
    let payload: [u8; 12] = [0xAA; 12];
    mem.write_slice(&payload, GuestAddress(TX_FRAME_BUF + VIRTIO_NET_HDR_LEN as u64))
        .unwrap();
    let total = (VIRTIO_NET_HDR_LEN + payload.len()) as u32;
    write_desc(mem, TX_DESC_BASE, 0, TX_FRAME_BUF, total, 0, 0);
    // Publish via avail ring at ring_pos=0, idx=1.
    let ring_off = TX_AVAIL_BASE + 4;
    mem.write_slice(&0u16.to_le_bytes(), GuestAddress(ring_off))
        .unwrap();
    mem.write_slice(&1u16.to_le_bytes(), GuestAddress(TX_AVAIL_BASE + 2))
        .unwrap();
}

/// Place one well-formed RX chain so `try_loopback_to_rx` would have
/// somewhere to deliver into. Single device-writable descriptor of
/// 64 bytes at RX_BUF — large enough for header + tiny payload.
fn place_rx_chain(mem: &GuestMemoryMmap) {
    write_desc(mem, RX_DESC_BASE, 0, RX_BUF, 64, VRING_DESC_F_WRITE, 0);
    let ring_off = RX_AVAIL_BASE + 4;
    mem.write_slice(&0u16.to_le_bytes(), GuestAddress(ring_off))
        .unwrap();
    mem.write_slice(&1u16.to_le_bytes(), GuestAddress(RX_AVAIL_BASE + 2))
        .unwrap();
}

/// Plant a bogus avail.idx far ahead of `next_avail` (which starts
/// at 0) so `(idx - next_avail).0 > queue.size` fires in
/// `AvailIter::new` (virtio-queue queue.rs:707-709). Planting 1000
/// against `queue.size = 256` (production default) makes the
/// violation unambiguous (1000 - 0 = 1000, well over 256).
fn poison_avail_idx(mem: &GuestMemoryMmap, avail_base: u64, bogus_idx: u16) {
    mem.write_obj(bogus_idx, GuestAddress(avail_base + 2))
        .expect("plant bogus avail.idx");
}

// ---------------------------------------------------------------------------
// TX-side poison
// ---------------------------------------------------------------------------

/// Hostile-guest avail.idx on the TX queue. The drain MUST detect
/// the iter() error, set the poison flag, bump
/// `invalid_avail_idx_count`, set NEEDS_RESET in device_status,
/// write the irqfd, and bail. A re-kick MUST short-circuit at the
/// entry gate without re-bumping the counter or re-firing the
/// irqfd. A virtio reset MUST clear the poison and the device must
/// resume servicing kicks.
///
/// INT_CONFIG is intentionally NOT set — see the module-level
/// divergence-from-virtio-blk doc. The test asserts INT_CONFIG
/// stays 0 throughout to pin the kernel-source-grounded decision.
#[test]
fn tx_hostile_avail_idx_poisons_queue_and_signals() {
    let (mut dev, mem) = build_fixture();

    // Phase A — sanity: counter starts at zero, no signal bits set.
    assert_eq!(
        dev.counters().invalid_avail_idx_count(),
        0,
        "fresh device must have zero InvalidAvailRingIndex events",
    );
    assert_eq!(
        read_reg(&dev, VIRTIO_MMIO_STATUS) & VIRTIO_CONFIG_S_NEEDS_RESET,
        0,
        "fresh device must not have NEEDS_RESET set",
    );
    assert_eq!(
        read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS) & VIRTIO_MMIO_INT_CONFIG,
        0,
        "fresh device must not have INT_CONFIG set",
    );

    // Phase B — plant a bogus TX avail.idx (1000 vs queue.size 256).
    // The check `(1000 - 0).0 > 256` fires immediately on iter().
    // Place a real TX chain too so the descriptor table has data —
    // the poison fires before iter() yields a chain head, but it's
    // important that the rest of the queue state is sane to isolate
    // the poison-path behaviour.
    place_tx_chain(&mem);
    place_rx_chain(&mem);
    poison_avail_idx(&mem, TX_AVAIL_BASE, 1000);

    // Phase C — kick. The drain must observe InvalidAvailRingIndex
    // on the TX queue, poison, and signal.
    let pre_tx_packets = dev.counters().tx_packets();
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, TXQ as u32);

    assert_eq!(
        dev.counters().invalid_avail_idx_count(),
        1,
        "first hostile-idx kick must bump invalid_avail_idx_count exactly once",
    );
    assert_eq!(
        dev.counters().tx_packets(),
        pre_tx_packets,
        "no TX must be serviced — the poisoned queue is structurally broken",
    );
    assert_ne!(
        read_reg(&dev, VIRTIO_MMIO_STATUS) & VIRTIO_CONFIG_S_NEEDS_RESET,
        0,
        "queue-poison path must set VIRTIO_CONFIG_S_NEEDS_RESET",
    );
    // INT_CONFIG must remain 0 — virtio-net doesn't advertise
    // F_STATUS so the kernel callback would no-op anyway.
    // Pinning this keeps the divergence-from-virtio-blk
    // intentional (catches a regression that copy-pasted blk's
    // INT_CONFIG-set into virtio-net by mistake).
    assert_eq!(
        read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS) & VIRTIO_MMIO_INT_CONFIG,
        0,
        "virtio-net poison path must NOT set INT_CONFIG (kernel \
         driver doesn't negotiate F_STATUS; setting INT_CONFIG \
         would just produce a wasted vCPU exit — see \
         signal_queue_poisoned doc)",
    );
    // irqfd: one Ok read drains the counter set by the poison signal.
    assert!(
        dev.irq_evt().read().is_ok(),
        "queue-poison path must signal irq_evt; a missed write would \
         prevent the guest's vm_interrupt from running",
    );

    // Phase D — re-kick the poisoned queue. The early-return gate
    // at the top of process_tx_loopback must short-circuit before
    // re-reading avail.idx, so neither counter re-bumps nor irqfd
    // re-fires. STATUS bits remain set (cumulative).
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, TXQ as u32);
    assert_eq!(
        dev.counters().invalid_avail_idx_count(),
        1,
        "subsequent kicks against a poisoned queue MUST NOT \
         re-bump the counter — the queue_poisoned gate \
         short-circuits before iter()",
    );
    assert!(
        dev.irq_evt().read().is_err(),
        "re-kick of a poisoned queue must NOT re-fire the irqfd \
         — the poison gate short-circuits before signal_queue_poisoned",
    );
    assert_ne!(
        read_reg(&dev, VIRTIO_MMIO_STATUS) & VIRTIO_CONFIG_S_NEEDS_RESET,
        0,
        "NEEDS_RESET stays set across re-kicks until reset",
    );

    // Phase E — virtio reset clears the poison. Walk the FSM from
    // STATUS=0 back through DRIVER_OK and verify the device
    // resumes servicing kicks.
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, 0);
    assert_eq!(
        read_reg(&dev, VIRTIO_MMIO_STATUS) & VIRTIO_CONFIG_S_NEEDS_RESET,
        0,
        "STATUS=0 reset must clear NEEDS_RESET",
    );
    // INT_CONFIG was never set on the poison path (virtio-net
    // divergence from virtio-blk); reset just clears
    // interrupt_status to zero on principle.
    assert_eq!(
        read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS) & VIRTIO_MMIO_INT_CONFIG,
        0,
        "INT_CONFIG must be 0 post-reset (it was never set)",
    );
    // The cumulative counter persists across reset — operators
    // need lifetime-event visibility to detect repeated hostile
    // behavior. Same invariant virtio-blk's tests_poison.rs pins.
    assert_eq!(
        dev.counters().invalid_avail_idx_count(),
        1,
        "invalid_avail_idx_count is cumulative across reset; only \
         the per-device poison flag clears",
    );

    // Re-init, plant a clean chain, and verify the device drains.
    init_until_features_ok(&mut dev);
    program_queues(&mut dev);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);
    // Clear the prior planted avail.idx fields and re-publish.
    mem.write_obj(0u16, GuestAddress(TX_AVAIL_BASE + 2)).unwrap();
    mem.write_obj(0u16, GuestAddress(TX_USED_BASE + 2)).unwrap();
    mem.write_obj(0u16, GuestAddress(RX_AVAIL_BASE + 2)).unwrap();
    mem.write_obj(0u16, GuestAddress(RX_USED_BASE + 2)).unwrap();
    place_tx_chain(&mem);
    place_rx_chain(&mem);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, TXQ as u32);
    assert_eq!(
        dev.counters().tx_packets(),
        pre_tx_packets + 1,
        "post-reset chain must service normally — queue_poisoned cleared",
    );
}

// ---------------------------------------------------------------------------
// RX-side poison
// ---------------------------------------------------------------------------

/// Hostile-guest avail.idx on the RX queue. The TX side runs
/// normally (TX queue's avail.idx is sane), captures a frame, then
/// hands off to `try_loopback_to_rx` — which calls `iter()` on the
/// RX queue and observes `InvalidAvailRingIndex`. The drain must
/// poison the queue, bump the counter, fire the signals, complete
/// the TX `add_used` (so the in-flight TX request doesn't hang),
/// and bail.
#[test]
fn rx_hostile_avail_idx_poisons_queue_and_signals() {
    let (mut dev, mem) = build_fixture();

    place_tx_chain(&mem);
    place_rx_chain(&mem);
    // Plant the bogus avail.idx on the RX queue ONLY. TX queue
    // remains sane so the TX `iter()` succeeds and a chain pops.
    poison_avail_idx(&mem, RX_AVAIL_BASE, 1000);

    let pre_invalid = dev.counters().invalid_avail_idx_count();
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, TXQ as u32);

    assert_eq!(
        dev.counters().invalid_avail_idx_count(),
        pre_invalid + 1,
        "RX-side hostile avail.idx must bump invalid_avail_idx_count exactly once",
    );
    // The TX side completed: the TX chain was popped, captured,
    // and add_used was issued. tx_packets bumps because the TX
    // add_used succeeded — the RX poison didn't prevent that.
    assert_eq!(
        dev.counters().tx_packets(),
        1,
        "TX add_used succeeded before the poison-signal bail; \
         tx_packets must bump",
    );
    // RX delivery did NOT happen — the chain was poisoned before
    // pop.
    assert_eq!(
        dev.counters().rx_packets(),
        0,
        "RX poison prevents delivery; rx_packets must stay zero",
    );
    assert_ne!(
        read_reg(&dev, VIRTIO_MMIO_STATUS) & VIRTIO_CONFIG_S_NEEDS_RESET,
        0,
        "RX poison must set NEEDS_RESET",
    );
    // INT_CONFIG must remain 0 — see module-level
    // divergence-from-virtio-blk doc.
    assert_eq!(
        read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS) & VIRTIO_MMIO_INT_CONFIG,
        0,
        "virtio-net poison path must NOT set INT_CONFIG",
    );
    // irqfd was written by signal_used (TX completion) and
    // signal_queue_poisoned. counter-mode coalesces both writes
    // into one read.
    assert!(
        dev.irq_evt().read().is_ok(),
        "RX poison must signal irq_evt (signal_used + \
         signal_queue_poisoned coalesced)",
    );

    // Re-kick: poison gate short-circuits — counter stays at the
    // post-poison value, irqfd doesn't refire.
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, TXQ as u32);
    assert_eq!(
        dev.counters().invalid_avail_idx_count(),
        pre_invalid + 1,
        "re-kick of a poisoned queue MUST NOT re-bump the counter",
    );
    assert!(
        dev.irq_evt().read().is_err(),
        "re-kick of a poisoned queue MUST NOT re-fire the irqfd",
    );
}

// ---------------------------------------------------------------------------
// Per-queue independence (F17-2)
// ---------------------------------------------------------------------------

/// Adversarial PhD F17-2 invariant: per-queue poison flags allow
/// TX to keep servicing kicks while RX is poisoned. With a
/// device-level flag, this scenario would short-circuit BOTH
/// halves and the operator would lose visibility on which queue
/// was actually broken. With per-queue flags, the post-RX-poison
/// kick still drains the TX side; tx_packets advances and the
/// guest sees TX completions — RX delivery is the only thing that
/// stops working.
#[test]
fn rx_poison_does_not_halt_tx_progress() {
    let (mut dev, mem) = build_fixture();

    // Phase 1: poison RX. TX queue stays clean.
    place_tx_chain(&mem);
    place_rx_chain(&mem);
    poison_avail_idx(&mem, RX_AVAIL_BASE, 1000);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, TXQ as u32);

    // After phase 1: TX serviced one chain (add_used succeeded),
    // RX is poisoned. The signal_queue_poisoned + signal_used
    // coalesce into one irq_evt write; consume it.
    assert_eq!(dev.counters().tx_packets(), 1);
    assert_eq!(dev.counters().invalid_avail_idx_count(), 1);
    let _ = dev.irq_evt().read();

    // Phase 2: re-kick TX with another well-formed chain. RX is
    // still poisoned — try_loopback_to_rx returns
    // RxAlreadyPoisoned (gate short-circuits, no counter bump,
    // no signal). TX side still pops the chain and add_used's
    // it. tx_packets must advance.
    //
    // To plant a second TX chain at ring[1], idx=2: the queue's
    // next_avail is now 1 (one chain consumed). Append a second
    // chain at ring[1] and bump avail.idx to 2.
    mem.write_slice(&0u16.to_le_bytes(), GuestAddress(TX_AVAIL_BASE + 4 + 2))
        .unwrap();
    mem.write_slice(&2u16.to_le_bytes(), GuestAddress(TX_AVAIL_BASE + 2))
        .unwrap();

    let pre_tx = dev.counters().tx_packets();
    let pre_invalid = dev.counters().invalid_avail_idx_count();
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, TXQ as u32);

    assert_eq!(
        dev.counters().tx_packets(),
        pre_tx + 1,
        "TX must continue servicing kicks even when RX is poisoned \
         — per-queue poison flags isolate the failure",
    );
    assert_eq!(
        dev.counters().invalid_avail_idx_count(),
        pre_invalid,
        "RxAlreadyPoisoned arm must NOT re-bump invalid_avail_idx_count \
         — counter is event-once per false→true transition",
    );
    // signal_queue_poisoned must NOT re-fire (RX poison flag was
    // already true). signal_used DOES fire (TX completion is a new
    // used-ring advance the guest must observe). irq_evt counter
    // = 1 (just signal_used).
    let kicks = dev.irq_evt().read().unwrap_or(0);
    assert_eq!(
        kicks, 1,
        "TX completion in a kick where RX is already-poisoned must \
         fire signal_used exactly once and signal_queue_poisoned \
         zero times",
    );
}

// ---------------------------------------------------------------------------
// Poison signal-sequence shape (kernel-source-grounded divergence)
// ---------------------------------------------------------------------------

/// The poison signal sequence on virtio-net is INTENTIONALLY
/// shorter than virtio-blk's: NEEDS_RESET in device_status +
/// irqfd write, and that's it. virtio-blk additionally sets
/// `VIRTIO_MMIO_INT_CONFIG` so its `vp_config_changed` callback
/// runs and inspects (e.g.) capacity. virtio-net does NOT
/// negotiate `VIRTIO_NET_F_STATUS`, so the kernel's
/// `virtnet_config_changed_work` (drivers/net/virtio_net.c:6208-6239)
/// `virtio_cread_feature(F_STATUS, ...)`-fails and bails;
/// raising INT_CONFIG would just produce a wasted vCPU exit.
/// This test pins that divergence: a regression that copy-pasted
/// virtio-blk's INT_CONFIG-set into virtio-net's poison path
/// would fail this assertion.
#[test]
fn rx_poison_signal_sequence_no_int_config() {
    let (mut dev, mem) = build_fixture();

    place_tx_chain(&mem);
    place_rx_chain(&mem);
    poison_avail_idx(&mem, RX_AVAIL_BASE, 1000);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, TXQ as u32);

    // Two signal effects on the device side:
    //   1. NEEDS_RESET set in device_status (operator detects via
    //      mmio_read(STATUS))
    //   2. irqfd written
    // INT_VRING also gets set by signal_used because the TX
    // chain completed before the RX poison, but that's per-event,
    // not part of the poison sequence per se.
    assert_ne!(
        read_reg(&dev, VIRTIO_MMIO_STATUS) & VIRTIO_CONFIG_S_NEEDS_RESET,
        0,
        "NEEDS_RESET must be set",
    );
    // INT_CONFIG must remain 0 — kernel-source-grounded decision
    // (virtnet_config_changed_work cread-fails F_STATUS).
    assert_eq!(
        read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS) & VIRTIO_MMIO_INT_CONFIG,
        0,
        "virtio-net poison path must NOT set INT_CONFIG (divergence \
         from virtio-blk; see signal_queue_poisoned doc)",
    );
    assert!(
        dev.irq_evt().read().is_ok(),
        "irq_evt must be signaled",
    );

    // Reset clears NEEDS_RESET and the per-queue flag.
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, 0);
    assert_eq!(
        read_reg(&dev, VIRTIO_MMIO_STATUS) & VIRTIO_CONFIG_S_NEEDS_RESET,
        0,
    );
    assert_eq!(
        read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS) & VIRTIO_MMIO_INT_CONFIG,
        0,
    );
}
