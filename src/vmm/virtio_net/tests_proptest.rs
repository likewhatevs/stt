// ----------------------------------------------------------------------------
// proptest fuzz suite for process_tx_loopback (and the in-VMM loopback path
// it drives via try_loopback_to_rx).
//
// Property-driven coverage of the TX/RX descriptor-chain parsing paths:
// generate arbitrary sequences of descriptors (random addr/len/flags/next)
// plus arbitrary header bytes, feed them through the device via
// QUEUE_NOTIFY, and assert the hostile-input contract:
//
//   1. No panic, OOB index, or unwrap-on-None — process_tx_loopback must
//      handle every input without crashing the vCPU thread that called
//      mmio_write(QUEUE_NOTIFY). A panic on the vCPU thread propagates
//      via vcpu_panic::install_once and tears down the VM mid-test.
//   2. Forward progress: for every TX kick, at least one of
//      `used.idx` advance (TX or RX), `tx_packets`, `rx_packets`,
//      `tx_chain_invalid`, `rx_chain_invalid`, `tx_dropped_no_rx_buffer`,
//      `tx_add_used_failures`, or `rx_add_used_failures` shows movement.
//      A silent stall (no advance, no counter) would let a hostile
//      guest pin the queue indefinitely.
//   3. Counter monotonicity: counters never decrement.
//   4. Defined post-state: post-notify, the device's queue cursors and
//      counter Arcs remain consistent with the pinned event taxonomy
//      documented on `VirtioNetCounters`.
//
// Mirrors the firecracker pattern of systematic chain corruption used by
// virtio_blk/tests_proptest.rs — every randomly-generated descriptor
// element exercises a code path the hand-curated tests don't reach.
// ----------------------------------------------------------------------------

use super::device::*;
use crate::vmm::net_config::NetConfig;
use proptest::prelude::*;
use std::sync::atomic::Ordering;
use virtio_bindings::virtio_config::VIRTIO_F_VERSION_1;
use virtio_bindings::virtio_mmio::{
    VIRTIO_MMIO_DRIVER_FEATURES, VIRTIO_MMIO_DRIVER_FEATURES_SEL, VIRTIO_MMIO_QUEUE_AVAIL_LOW,
    VIRTIO_MMIO_QUEUE_DESC_LOW, VIRTIO_MMIO_QUEUE_NOTIFY, VIRTIO_MMIO_QUEUE_NUM,
    VIRTIO_MMIO_QUEUE_READY, VIRTIO_MMIO_QUEUE_SEL, VIRTIO_MMIO_QUEUE_USED_LOW,
    VIRTIO_MMIO_STATUS,
};
use virtio_bindings::virtio_net::VIRTIO_NET_F_MAC;
use vm_memory::{Bytes, GuestAddress, GuestMemoryMmap};

// ----------------------------------------------------------------------------
// Memory layout — picked so descriptor tables / avail rings / used rings /
// header buffers / frame buffers / RX buffers all live at distinct,
// page-aligned addresses inside a 1 MiB guest memory region. Mirrors the
// hand-curated tests.rs layout but the addresses themselves are local to
// this module so the proptest is self-contained.
// ----------------------------------------------------------------------------
const GUEST_MEM_SIZE: usize = 0x10_0000; // 1 MiB
const TX_DESC_BASE: u64 = 0x1000;
const TX_AVAIL_BASE: u64 = 0x2000;
const TX_USED_BASE: u64 = 0x3000;
const TX_FRAME_BUF_BASE: u64 = 0x4000; // 0x4000..0x6000 — TX header + payload
const RX_DESC_BASE: u64 = 0x6000;
const RX_AVAIL_BASE: u64 = 0x7000;
const RX_USED_BASE: u64 = 0x8000;
const RX_BUF_BASE: u64 = 0x9000; // 0x9000..0xC000 — RX descriptor target buffers

/// Per-queue size for proptest fixtures. 16 is enough to hold every
/// proptest-generated chain (≤8 descriptors per chain by `MAX_CHAIN_LEN`)
/// with room to spare for the rings, and small enough that the avail-ring
/// modulo (queue_size - 1) wraps cleanly when the proptest fires
/// repeated cases inside a single shrink-loop.
const PROPTEST_QUEUE_SIZE: u16 = 16;

/// Maximum chain length per proptest case. The kernel virtio-net driver
/// emits TX chains of at most a handful of descriptors (the header plus
/// 1-2 fragments in practice), and RX buffers are typically a single
/// descriptor (the kernel posts pre-allocated RX buffers as one
/// PAGE_SIZE descriptor each). 8 is a generous upper bound that exercises
/// the multi-descriptor walk without forcing the proptest to plant
/// chains larger than realistic guest behaviour. The mutation surface is
/// adequate at this length: every `next` link, `flags` bit, and `len`
/// boundary gets stress.
const MAX_CHAIN_LEN: usize = 8;

// virtio_ring descriptor flag bits.
const VRING_DESC_F_NEXT: u16 = 1;
const VRING_DESC_F_WRITE: u16 = 2;

// ----------------------------------------------------------------------------
// Fuzz strategies
// ----------------------------------------------------------------------------

/// Shape of one random descriptor. `flags` is restricted to the three
/// bits the device cares about (NEXT, WRITE, INDIRECT); higher bits
/// would be silently masked by the parser anyway, so generating them
/// adds no coverage. `next` is a full `u16` because out-of-range values
/// (>= queue_size) are part of the test surface — the queue iterator
/// must stop without panicking when `next >= queue_size`.
#[derive(Debug, Clone, Copy)]
struct FuzzDesc {
    addr: u64,
    len: u32,
    flags: u16,
    next: u16,
}

/// Strategy for a single descriptor on the TX side.
///
/// `addr` ranges far beyond the 1 MiB guest-memory region so a
/// substantial fraction of generated descriptors point at unmapped GPA.
/// The device must reject those via `mem.read_slice` errors rather than
/// panic. `0..2^24` covers the entire 1 MiB region in-range plus 15 MiB
/// beyond it (unmapped) — roughly 1:15 valid-to-invalid ratio.
///
/// `len` ranges past `TX_DESC_MAX = 64 KiB` so the per-descriptor cap is
/// exercised, plus `0` (zero-length descriptor). 0..=8 MiB generates
/// enough over-cap descriptors without making every chain trivially over-cap.
///
/// `flags` is `0..8` (3 bits), giving every combination of NEXT/WRITE/INDIRECT.
fn fuzz_desc_strategy() -> impl Strategy<Value = FuzzDesc> {
    (
        0u64..(1u64 << 24),
        0u32..(8 * 1024 * 1024),
        0u16..8,
        any::<u16>(),
    )
        .prop_map(|(addr, len, flags, next)| FuzzDesc {
            addr,
            len,
            flags,
            next,
        })
}

/// Strategy for a chain of 1..=MAX_CHAIN_LEN descriptors. The lower bound
/// of 1 ensures the avail ring always has at least one chain head so
/// `process_tx_loopback` always traverses at least one iteration.
fn fuzz_chain_strategy() -> impl Strategy<Value = Vec<FuzzDesc>> {
    prop::collection::vec(fuzz_desc_strategy(), 1..=MAX_CHAIN_LEN)
}

/// Strategy for the 12-byte `virtio_net_hdr_v1` header bytes. Random
/// values across all fields exercise the path where a hostile guest
/// embeds nonsense in the header — the device reads the header from
/// guest memory but ignores all fields except the first 12 bytes
/// boundary itself. A regression that interpreted any of these fields
/// (e.g. `gso_type`, `flags`) without honoring the negotiated feature
/// bits would surface as a panic or wrong-byte-count completion.
///
/// Layout per `struct virtio_net_hdr_v1`:
///   bytes 0:    flags (u8)
///   bytes 1:    gso_type (u8)
///   bytes 2..4: hdr_len (u16 LE)
///   bytes 4..6: gso_size (u16 LE)
///   bytes 6..8: csum_start (u16 LE)
///   bytes 8..10: csum_offset (u16 LE)
///   bytes 10..12: num_buffers (u16 LE) — RX-side only; TX writes here too
fn fuzz_header_strategy() -> impl Strategy<Value = [u8; 12]> {
    any::<[u8; 12]>()
}

// ----------------------------------------------------------------------------
// Fixture builder + queue programming
// ----------------------------------------------------------------------------

/// Build a `VirtioNet` device + 1 MiB guest memory and drive the FSM up
/// through DRIVER_OK with both queues programmed at the canonical
/// proptest layout addresses.
fn build_fuzz_fixture() -> (VirtioNet, GuestMemoryMmap) {
    let mem = GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), GUEST_MEM_SIZE)])
        .expect("create proptest guest mem");
    let mut dev = VirtioNet::new(NetConfig::default());
    dev.set_mem(mem.clone());
    init_until_features_ok(&mut dev);
    program_queues(&mut dev);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);
    (dev, mem)
}

/// Drive the device through ACK → DRIVER → negotiate VERSION_1+MAC →
/// FEATURES_OK. Stops short of DRIVER_OK so the caller can program queue
/// addresses (only allowed in the FEATURES_OK..DRIVER_OK window per
/// virtio-v1.2 §3.1.1).
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

/// Program both queues at the canonical proptest layout addresses with
/// `PROPTEST_QUEUE_SIZE` slots each.
fn program_queues(dev: &mut VirtioNet) {
    // RX queue (idx 0)
    write_reg(dev, VIRTIO_MMIO_QUEUE_SEL, RXQ as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_NUM, PROPTEST_QUEUE_SIZE as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_DESC_LOW, RX_DESC_BASE as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_AVAIL_LOW, RX_AVAIL_BASE as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_USED_LOW, RX_USED_BASE as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_READY, 1);
    // TX queue (idx 1)
    write_reg(dev, VIRTIO_MMIO_QUEUE_SEL, TXQ as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_NUM, PROPTEST_QUEUE_SIZE as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_DESC_LOW, TX_DESC_BASE as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_AVAIL_LOW, TX_AVAIL_BASE as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_USED_LOW, TX_USED_BASE as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_READY, 1);
}

fn write_reg(dev: &mut VirtioNet, offset: u32, val: u32) {
    dev.mmio_write(offset as u64, &val.to_le_bytes());
}

/// Read the used-ring `idx` field for the given queue. The used ring
/// layout is `flags u16 | idx u16 | ring[u16; N]`, so `+ 2` skips the
/// flags field.
fn read_used_idx(mem: &GuestMemoryMmap, used_base: u64) -> u16 {
    mem.read_obj::<u16>(GuestAddress(used_base + 2))
        .expect("read used.idx")
}

/// Plant a descriptor into a queue's descriptor table at the given index.
/// Layout per virtio split-ring: `addr u64 | len u32 | flags u16 | next u16`
/// (16 bytes per entry).
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

/// Publish a chain head into the avail ring at `ring_pos` and bump
/// `avail.idx` to `ring_pos + 1`. Avail layout: `flags u16 | idx u16 | ring[u16; N]`.
fn publish_avail(mem: &GuestMemoryMmap, avail_base: u64, head_idx: u16, ring_pos: u16) {
    let ring_off = avail_base + 4 + (ring_pos as u64) * 2;
    mem.write_slice(&head_idx.to_le_bytes(), GuestAddress(ring_off))
        .expect("write avail.ring entry");
    let idx_off = avail_base + 2;
    mem.write_slice(&(ring_pos + 1).to_le_bytes(), GuestAddress(idx_off))
        .expect("write avail.idx");
}

/// Plant a multi-descriptor TX chain from a list of `FuzzDesc`. Wires
/// the chain via the in-order `NEXT` link convention: descriptor `i`
/// links to descriptor `i+1` if and only if its `flags` had the NEXT
/// bit set in the strategy output. We honor the strategy's chosen
/// `flags` and `next` verbatim — including out-of-range `next` values
/// — so the proptest can stress the parser's "next out of bounds" arm.
fn plant_tx_chain(mem: &GuestMemoryMmap, descs: &[FuzzDesc]) {
    for (i, d) in descs.iter().enumerate() {
        write_desc(mem, TX_DESC_BASE, i as u16, d.addr, d.len, d.flags, d.next);
    }
    publish_avail(mem, TX_AVAIL_BASE, 0, 0);
}

/// Plant a multi-descriptor RX chain at the canonical RX_DESC_BASE.
/// Same `flags`/`next` honoring as `plant_tx_chain`.
fn plant_rx_chain(mem: &GuestMemoryMmap, descs: &[FuzzDesc]) {
    for (i, d) in descs.iter().enumerate() {
        write_desc(mem, RX_DESC_BASE, i as u16, d.addr, d.len, d.flags, d.next);
    }
    publish_avail(mem, RX_AVAIL_BASE, 0, 0);
}

/// Plant a well-formed RX chain consisting of a single device-writable
/// descriptor at RX_BUF_BASE with `len` bytes — large enough to hold a
/// 12-byte virtio header + a small frame. Used by tests that fuzz the
/// TX side and want a known-good RX target.
fn plant_well_formed_rx_chain(mem: &GuestMemoryMmap) {
    write_desc(
        mem,
        RX_DESC_BASE,
        0,
        RX_BUF_BASE,
        2048,
        VRING_DESC_F_WRITE,
        0,
    );
    publish_avail(mem, RX_AVAIL_BASE, 0, 0);
}

// ----------------------------------------------------------------------------
// Counter snapshot
// ----------------------------------------------------------------------------

/// Snapshot of every counter the device mutates. Captures the full event
/// taxonomy from `VirtioNetCounters` so the "something happened" check
/// stays exhaustive — a regression that introduced a new code path
/// without bumping any existing counter would fail the forward-progress
/// invariant.
#[derive(Default, Clone, Copy, Debug)]
struct CounterSnapshot {
    tx_packets: u64,
    tx_bytes: u64,
    rx_packets: u64,
    rx_bytes: u64,
    tx_dropped_no_rx_buffer: u64,
    tx_chain_invalid: u64,
    rx_chain_invalid: u64,
    tx_add_used_failures: u64,
    rx_add_used_failures: u64,
    invalid_avail_idx_count: u64,
}

fn snapshot_counters(dev: &VirtioNet) -> CounterSnapshot {
    let c = dev.counters();
    CounterSnapshot {
        tx_packets: c.tx_packets.load(Ordering::Relaxed),
        tx_bytes: c.tx_bytes.load(Ordering::Relaxed),
        rx_packets: c.rx_packets.load(Ordering::Relaxed),
        rx_bytes: c.rx_bytes.load(Ordering::Relaxed),
        tx_dropped_no_rx_buffer: c.tx_dropped_no_rx_buffer.load(Ordering::Relaxed),
        tx_chain_invalid: c.tx_chain_invalid.load(Ordering::Relaxed),
        rx_chain_invalid: c.rx_chain_invalid.load(Ordering::Relaxed),
        tx_add_used_failures: c.tx_add_used_failures.load(Ordering::Relaxed),
        rx_add_used_failures: c.rx_add_used_failures.load(Ordering::Relaxed),
        invalid_avail_idx_count: c.invalid_avail_idx_count.load(Ordering::Relaxed),
    }
}

/// Total observable progress across all event counters (excludes
/// paired byte aggregates `tx_bytes` / `rx_bytes` which would
/// double-count — every successful packet bumps `tx_packets` /
/// `rx_packets` AND adds to `tx_bytes` / `rx_bytes`, so summing
/// both pairs would inflate the delta). A regression that added a
/// silent-drop path (no counter, no used.idx) would leave this at
/// zero, failing the forward-progress invariant.
fn counter_delta(before: &CounterSnapshot, after: &CounterSnapshot) -> u64 {
    (after.tx_packets - before.tx_packets)
        + (after.rx_packets - before.rx_packets)
        + (after.tx_dropped_no_rx_buffer - before.tx_dropped_no_rx_buffer)
        + (after.tx_chain_invalid - before.tx_chain_invalid)
        + (after.rx_chain_invalid - before.rx_chain_invalid)
        + (after.tx_add_used_failures - before.tx_add_used_failures)
        + (after.rx_add_used_failures - before.rx_add_used_failures)
        + (after.invalid_avail_idx_count - before.invalid_avail_idx_count)
}

/// Assert the monotonicity invariant — every counter only ever increases.
/// A regression that subtracted from a counter (e.g. an over-eager
/// rollback path) would surface here regardless of whether the chain
/// itself made progress.
fn assert_counter_monotonicity(
    before: &CounterSnapshot,
    after: &CounterSnapshot,
) -> Result<(), TestCaseError> {
    prop_assert!(after.tx_packets >= before.tx_packets);
    prop_assert!(after.tx_bytes >= before.tx_bytes);
    prop_assert!(after.rx_packets >= before.rx_packets);
    prop_assert!(after.rx_bytes >= before.rx_bytes);
    prop_assert!(after.tx_dropped_no_rx_buffer >= before.tx_dropped_no_rx_buffer);
    prop_assert!(after.tx_chain_invalid >= before.tx_chain_invalid);
    prop_assert!(after.rx_chain_invalid >= before.rx_chain_invalid);
    prop_assert!(after.tx_add_used_failures >= before.tx_add_used_failures);
    prop_assert!(after.rx_add_used_failures >= before.rx_add_used_failures);
    prop_assert!(after.invalid_avail_idx_count >= before.invalid_avail_idx_count);
    Ok(())
}

// ----------------------------------------------------------------------------
// proptest cases
// ----------------------------------------------------------------------------

proptest! {
    // 256 matches the proptest default; explicit so a future
    // PROPTEST_CASES env override is the only knob that changes
    // behavior. `max_shrink_iters` capped at a moderate value
    // because shrunken cases mostly help debug failures, not
    // detect them.
    #![proptest_config(ProptestConfig {
        cases: 256,
        max_shrink_iters: 1024,
        .. ProptestConfig::default()
    })]

    /// Random TX descriptor chains with a well-formed RX target MUST
    /// produce forward progress: for every notify, at least one of
    /// `used.idx` advance (TX or RX), `tx_packets`, `rx_packets`,
    /// `tx_chain_invalid`, `rx_chain_invalid`,
    /// `tx_dropped_no_rx_buffer`, `tx_add_used_failures`,
    /// `rx_add_used_failures`, or `invalid_avail_idx_count` must
    /// show movement. A chain that left every counter and used.idx
    /// static would represent a silent stall — the guest's
    /// network-stack equivalent of virtio_blk's hung-task watchdog
    /// (the kernel's tx_hang detection) would eventually fire, but
    /// the host has no visibility until then.
    ///
    /// `invalid_avail_idx_count` is included because the queue-poison
    /// path returns from `process_tx_loopback` after only setting
    /// the per-queue poison flag, bumping that counter, and firing
    /// signal_queue_poisoned — `tx_packets` / `rx_packets` /
    /// `*_chain_invalid` stay static on a poisoned-iter() drain.
    /// (In practice the proptest fixture's `publish_avail` writes
    /// avail.idx=1 against queue.size=16, so the poison gate never
    /// fires; the inclusion is defense-in-depth for shrunken cases
    /// that might trip a future iter() error path.)
    ///
    /// Critically: this also pins panic-freeness. The proptest runner
    /// catches panics; a panic in `process_tx_loopback` or
    /// `try_loopback_to_rx` under any input crashes the test with
    /// the offending shrunken case.
    #[test]
    fn tx_chain_progress_under_random_descriptors(
        descs in fuzz_chain_strategy(),
    ) {
        let (mut dev, mem) = build_fuzz_fixture();
        plant_tx_chain(&mem, &descs);
        plant_well_formed_rx_chain(&mem);

        let before_tx_used = read_used_idx(&mem, TX_USED_BASE);
        let before_rx_used = read_used_idx(&mem, RX_USED_BASE);
        let before = snapshot_counters(&dev);

        // Fire the kick. process_tx_loopback is the system under
        // test. A panic here propagates up and proptest shrinks to
        // the minimal offending input. A hang (e.g. infinite chain
        // loop) surfaces as the test runner's per-case timeout.
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, TXQ as u32);

        let after_tx_used = read_used_idx(&mem, TX_USED_BASE);
        let after_rx_used = read_used_idx(&mem, RX_USED_BASE);
        let after = snapshot_counters(&dev);

        assert_counter_monotonicity(&before, &after)?;

        let used_delta = (after_tx_used - before_tx_used) as u64
            + (after_rx_used - before_rx_used) as u64;
        let cdelta = counter_delta(&before, &after);
        let progress = used_delta + cdelta;
        prop_assert!(
            progress >= 1,
            "no visible progress: tx_used_delta={} rx_used_delta={} \
             counter_delta={} (chain len={}, first_desc=({:#x},{},{:#x},{}))",
            (after_tx_used - before_tx_used) as u64,
            (after_rx_used - before_rx_used) as u64,
            cdelta,
            descs.len(),
            descs[0].addr,
            descs[0].len,
            descs[0].flags,
            descs[0].next,
        );
    }

    /// Random RX descriptor chain with a well-formed TX chain MUST
    /// produce forward progress. Mirrors the TX-side property but
    /// targets the RX-walk path inside `try_loopback_to_rx`. An
    /// arbitrary RX descriptor (random addr/len/flags) must either
    /// successfully accept the loopback delivery (rx_packets bumps),
    /// reject it as malformed (rx_chain_invalid bumps), or fail
    /// add_used (rx_add_used_failures bumps) — but never panic and
    /// never silently lose the frame.
    #[test]
    fn rx_chain_progress_under_random_descriptors(
        rx_descs in fuzz_chain_strategy(),
    ) {
        let (mut dev, mem) = build_fuzz_fixture();
        // Plant a well-formed TX chain: a single read-only descriptor
        // covering 12-byte zero header + 16-byte payload back-to-back
        // at TX_FRAME_BUF_BASE. The TX side is well-known so the only
        // variable in the test is the RX descriptor shape.
        let zero_hdr = [0u8; 12];
        let payload: [u8; 16] = [0x42; 16];
        mem.write_slice(&zero_hdr, GuestAddress(TX_FRAME_BUF_BASE))
            .expect("plant zero header");
        mem.write_slice(&payload, GuestAddress(TX_FRAME_BUF_BASE + 12))
            .expect("plant payload");
        let tx_total = (12 + payload.len()) as u32;
        write_desc(&mem, TX_DESC_BASE, 0, TX_FRAME_BUF_BASE, tx_total, 0, 0);
        publish_avail(&mem, TX_AVAIL_BASE, 0, 0);

        plant_rx_chain(&mem, &rx_descs);

        let before_tx_used = read_used_idx(&mem, TX_USED_BASE);
        let before_rx_used = read_used_idx(&mem, RX_USED_BASE);
        let before = snapshot_counters(&dev);

        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, TXQ as u32);

        let after_tx_used = read_used_idx(&mem, TX_USED_BASE);
        let after_rx_used = read_used_idx(&mem, RX_USED_BASE);
        let after = snapshot_counters(&dev);

        assert_counter_monotonicity(&before, &after)?;

        // The TX side is well-formed, so process_tx_loopback always
        // pops the TX chain and either calls add_used on TX (success
        // or tx_chain_invalid path), advancing tx used.idx by 1.
        // Even when the RX chain is malformed and the TX add_used
        // succeeds, at minimum tx_used advances.
        let tx_used_delta = (after_tx_used - before_tx_used) as u64;
        let rx_used_delta = (after_rx_used - before_rx_used) as u64;
        let cdelta = counter_delta(&before, &after);
        prop_assert!(
            tx_used_delta + rx_used_delta + cdelta >= 1,
            "no visible progress: tx_used_delta={} \
             rx_used_delta={} counter_delta={} \
             (rx chain len={}, first_rx_desc=({:#x},{},{:#x},{}))",
            tx_used_delta,
            rx_used_delta,
            cdelta,
            rx_descs.len(),
            rx_descs[0].addr,
            rx_descs[0].len,
            rx_descs[0].flags,
            rx_descs[0].next,
        );
    }

    /// Random `virtio_net_hdr_v1` header bytes — fuzz the TX header-
    /// strip path. The 12 leading bytes of the TX chain are
    /// attacker-controlled; the device must skip them regardless of
    /// the byte values (no fields are interpreted because
    /// VIRTIO_NET_F_CSUM and VIRTIO_NET_F_GUEST_TSO* are not
    /// negotiated). A regression that interpreted `gso_type` or
    /// `flags` and panicked on an invalid combination would surface
    /// here.
    ///
    /// The chain shape is fixed (single read-only descriptor covering
    /// header + 16-byte payload) so the only variable is the header
    /// content. The RX side is well-formed.
    #[test]
    fn random_tx_header_either_loops_or_records_failure(
        hdr_bytes in fuzz_header_strategy(),
    ) {
        let (mut dev, mem) = build_fuzz_fixture();

        // Plant the random header + a 16-byte 0xAB payload at
        // TX_FRAME_BUF_BASE.
        mem.write_slice(&hdr_bytes, GuestAddress(TX_FRAME_BUF_BASE))
            .expect("plant fuzzed header");
        let payload: [u8; 16] = [0xAB; 16];
        mem.write_slice(&payload, GuestAddress(TX_FRAME_BUF_BASE + 12))
            .expect("plant payload");

        // Single read-only descriptor covering the full 28 bytes.
        let tx_total = (12 + payload.len()) as u32;
        write_desc(&mem, TX_DESC_BASE, 0, TX_FRAME_BUF_BASE, tx_total, 0, 0);
        publish_avail(&mem, TX_AVAIL_BASE, 0, 0);
        plant_well_formed_rx_chain(&mem);

        let before_tx_used = read_used_idx(&mem, TX_USED_BASE);
        let before = snapshot_counters(&dev);

        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, TXQ as u32);

        let after_tx_used = read_used_idx(&mem, TX_USED_BASE);
        let after = snapshot_counters(&dev);

        assert_counter_monotonicity(&before, &after)?;

        // The TX chain is well-formed (single read-only descriptor,
        // ≥12 bytes). The TX add_used MUST succeed and advance the
        // TX used-ring by exactly 1 — no header byte combination can
        // change this because the device never interprets the
        // header bytes for accept/reject decisions.
        prop_assert_eq!(
            after_tx_used - before_tx_used,
            1,
            "TX chain with valid shape and arbitrary header bytes \
             must always complete TX add_used; header={:?}",
            hdr_bytes,
        );
        // tx_packets bumps because TX add_used succeeded (frame_len
        // was Some — the chain was well-formed).
        prop_assert_eq!(
            after.tx_packets - before.tx_packets,
            1,
            "well-formed TX chain must bump tx_packets exactly once",
        );
    }

    /// Random `len` on the TX descriptor — fuzz the per-descriptor cap
    /// (`TX_DESC_MAX = 64 KiB`) and the total-frame cap
    /// (`MAX_FRAME_SIZE = 64 KiB`). The device caps each descriptor at
    /// `TX_DESC_MAX` and the cumulative captured bytes at
    /// `MAX_FRAME_SIZE`; a regression that didn't apply either cap
    /// would let a hostile guest force an attacker-sized scratch
    /// allocation. The 0..=8 MiB strategy generates well over the
    /// caps so the boundary is exercised without making every chain
    /// trivially over-cap.
    #[test]
    fn random_tx_desc_len_either_truncates_or_records_failure(
        len in 0u32..(8u32 * 1024 * 1024),
    ) {
        let (mut dev, mem) = build_fuzz_fixture();

        // Plant a zero header + 0xBB filler at TX_FRAME_BUF_BASE,
        // sized to the smaller of `len` and the remaining guest-mem
        // window. We can't write `8 MiB` of bytes since the guest mem
        // is only 1 MiB, but the descriptor's `len` can claim
        // arbitrary values — the device's cap must reject the read
        // before the cap-imposed read fails on the unmapped pages.
        let safe_fill_len = (len as usize).min(0x10_000); // up to 64 KiB
        let zero_hdr = [0u8; 12];
        mem.write_slice(&zero_hdr, GuestAddress(TX_FRAME_BUF_BASE))
            .expect("plant zero header");
        if safe_fill_len > 12 {
            let filler = vec![0xBBu8; safe_fill_len - 12];
            mem.write_slice(&filler, GuestAddress(TX_FRAME_BUF_BASE + 12))
                .expect("plant filler");
        }

        write_desc(&mem, TX_DESC_BASE, 0, TX_FRAME_BUF_BASE, len, 0, 0);
        publish_avail(&mem, TX_AVAIL_BASE, 0, 0);
        plant_well_formed_rx_chain(&mem);

        let before_tx_used = read_used_idx(&mem, TX_USED_BASE);
        let before = snapshot_counters(&dev);

        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, TXQ as u32);

        let after_tx_used = read_used_idx(&mem, TX_USED_BASE);
        let after = snapshot_counters(&dev);

        assert_counter_monotonicity(&before, &after)?;

        // TX add_used must fire exactly once: `process_tx_loopback`
        // always marks the popped TX chain used regardless of whether
        // the inner read failed (the chain is dropped on read failure,
        // but the device still advances used.idx so the guest doesn't
        // hang waiting on the slot).
        prop_assert_eq!(
            after_tx_used - before_tx_used,
            1,
            "TX must advance used.idx by 1 per popped chain regardless of \
             read failure; len={}",
            len,
        );

        // Either tx_packets bumped (chain was processable, even if
        // truncated) OR tx_chain_invalid bumped (chain was malformed
        // because hdr_remaining > 0 from too-short descriptor at len < 12).
        // The two are mutually exclusive: a successfully-captured frame
        // bumps tx_packets, a malformed shape bumps tx_chain_invalid.
        let tx_pkt_delta = after.tx_packets - before.tx_packets;
        let tx_inv_delta = after.tx_chain_invalid - before.tx_chain_invalid;
        prop_assert_eq!(
            tx_pkt_delta + tx_inv_delta,
            1,
            "exactly one of tx_packets/tx_chain_invalid must bump per \
             popped TX chain; len={} pkt_delta={} inv_delta={}",
            len,
            tx_pkt_delta,
            tx_inv_delta,
        );
    }

    /// Random `flags` on the TX descriptor — fuzz the direction-
    /// violation gate (TX descriptors must be device-readable, so
    /// `VRING_DESC_F_WRITE` is invalid) and the INDIRECT path. The
    /// device's parser handles both: write-only triggers
    /// `tx_chain_invalid`, INDIRECT pointing at unmapped GPA fails the
    /// indirect-table read and yields no descriptors → chain dropped
    /// (still marked used).
    #[test]
    fn random_tx_desc_flags_either_loops_or_records_failure(
        flags in 0u16..16,
    ) {
        let (mut dev, mem) = build_fuzz_fixture();

        // Plant a zero header + small payload at TX_FRAME_BUF_BASE.
        let zero_hdr = [0u8; 12];
        let payload: [u8; 16] = [0xCC; 16];
        mem.write_slice(&zero_hdr, GuestAddress(TX_FRAME_BUF_BASE))
            .expect("plant zero header");
        mem.write_slice(&payload, GuestAddress(TX_FRAME_BUF_BASE + 12))
            .expect("plant payload");

        let tx_total = (12 + payload.len()) as u32;
        write_desc(&mem, TX_DESC_BASE, 0, TX_FRAME_BUF_BASE, tx_total, flags, 0);
        publish_avail(&mem, TX_AVAIL_BASE, 0, 0);
        plant_well_formed_rx_chain(&mem);

        let before_tx_used = read_used_idx(&mem, TX_USED_BASE);
        let before = snapshot_counters(&dev);

        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, TXQ as u32);

        let after_tx_used = read_used_idx(&mem, TX_USED_BASE);
        let after = snapshot_counters(&dev);

        assert_counter_monotonicity(&before, &after)?;

        // For any `flags` value, either:
        //   (a) the chain was processed (TX used advances by 1, AND
        //       at least one of tx_packets/tx_chain_invalid bumps) —
        //       the typical case for legal flag combinations
        //   (b) the parser observed an invalid chain shape (e.g.
        //       INDIRECT pointing at unmapped GPA, write-only TX
        //       descriptor) and dropped/marked-used the chain.
        // Either way SOME observable progress must occur.
        let tx_used_delta = (after_tx_used - before_tx_used) as u64;
        let cdelta = counter_delta(&before, &after);
        prop_assert!(
            tx_used_delta + cdelta >= 1,
            "no progress with TX flags={:#x}: tx_used_delta={} \
             counter_delta={}",
            flags,
            tx_used_delta,
            cdelta,
        );
    }

    /// Random `next` link on the first of a 2-descriptor TX chain —
    /// fuzz the parser's "next out of bounds" arm. The first
    /// descriptor sets NEXT and points at the random `next` value;
    /// the second descriptor (planted at idx 1) holds the rest of
    /// the payload. When `next == 1` the chain is well-formed; when
    /// `next != 1` (in particular, `next >= queue_size`), the parser
    /// must stop without panicking. The proptest pins panic-freeness
    /// across the entire u16 range.
    #[test]
    fn random_tx_next_link_either_loops_or_truncates(
        next in any::<u16>(),
    ) {
        let (mut dev, mem) = build_fuzz_fixture();

        // Plant a 2-descriptor chain: header in desc 0, payload in
        // desc 1. The first descriptor's `next` is fuzzed.
        let zero_hdr = [0u8; 12];
        let payload: [u8; 16] = [0xDD; 16];
        mem.write_slice(&zero_hdr, GuestAddress(TX_FRAME_BUF_BASE))
            .expect("plant zero header");
        mem.write_slice(
            &payload,
            GuestAddress(TX_FRAME_BUF_BASE + 0x100),
        )
        .expect("plant payload");

        // desc 0: header descriptor, NEXT to fuzzed `next`.
        write_desc(
            &mem,
            TX_DESC_BASE,
            0,
            TX_FRAME_BUF_BASE,
            12,
            VRING_DESC_F_NEXT,
            next,
        );
        // desc 1: payload descriptor (well-formed; the parser only
        // reaches it when next==1).
        write_desc(
            &mem,
            TX_DESC_BASE,
            1,
            TX_FRAME_BUF_BASE + 0x100,
            payload.len() as u32,
            0,
            0,
        );
        publish_avail(&mem, TX_AVAIL_BASE, 0, 0);
        plant_well_formed_rx_chain(&mem);

        let before_tx_used = read_used_idx(&mem, TX_USED_BASE);
        let before = snapshot_counters(&dev);

        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, TXQ as u32);

        let after_tx_used = read_used_idx(&mem, TX_USED_BASE);
        let after = snapshot_counters(&dev);

        assert_counter_monotonicity(&before, &after)?;

        // TX always advances by 1 per popped chain.
        prop_assert_eq!(
            after_tx_used - before_tx_used,
            1,
            "TX must advance used.idx by 1 regardless of next link; next={}",
            next,
        );

        // Exactly one of tx_packets / tx_chain_invalid bumps. When
        // next==1 the chain is well-formed and tx_packets bumps;
        // when next!=1 the parser's iterator terminates early — if
        // the leading 12 header bytes were captured, hdr_remaining=0
        // and tx_packets bumps with frame_len==0 (an empty L2
        // payload, valid but pointless); otherwise tx_chain_invalid
        // bumps because hdr_remaining > 0.
        let tx_pkt_delta = after.tx_packets - before.tx_packets;
        let tx_inv_delta = after.tx_chain_invalid - before.tx_chain_invalid;
        prop_assert_eq!(
            tx_pkt_delta + tx_inv_delta,
            1,
            "exactly one of tx_packets/tx_chain_invalid must bump; \
             next={} pkt_delta={} inv_delta={}",
            next,
            tx_pkt_delta,
            tx_inv_delta,
        );
    }
}
