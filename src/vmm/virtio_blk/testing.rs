//! Shared test fixtures for the virtio-blk module's test files.
//!
//! Tier-1 of the test co-location split: every helper that more than
//! one test file would need to construct a `VirtioBlk`, build a backing
//! file, plant a chain, or drive the FSM lives here. Each helper is
//! `pub(super)` so it is visible to sibling test modules
//! (`device::tests`, `worker::tests`, `throttle::tests`, the integration
//! tests in `mod.rs`) without leaking outside the `virtio_blk` module.
//!
//! No test bodies live here — only fixtures. Tests that own a fixture
//! exclusively (e.g. `setup_iops1_drained_chain` is read by the
//! gauge-transition tests in `tests_atomics`) still live here because
//! `pub(super)` keeps the door open for a future test in another sibling
//! to consume the same fixture without a copy.
//!
//! `cfg(test)` gated at the module-declaration site (`mod testing;`
//! in `mod.rs`); this file itself is not gated so rust-analyzer can
//! still resolve the module path during cfg(test) builds.
#![cfg(test)]
#![allow(dead_code)]

use std::fs::File;
use std::io::{Seek, Write};

use tempfile::tempfile;
use virtio_bindings::bindings::virtio_ring::VRING_DESC_F_WRITE;
use virtio_queue::desc::{RawDescriptor, split::Descriptor as SplitDescriptor};
use virtio_queue::mock::MockSplitQueue;
use vm_memory::{Address, Bytes, GuestAddress, GuestMemoryMmap};

use super::{
    DiskThrottle, QUEUE_MAX_SIZE, S_ACK, S_DRV, S_FEAT, S_OK, VIRTIO_BLK_OUTHDR_SIZE,
    VIRTIO_BLK_T_IN, VIRTIO_F_VERSION_1, VIRTIO_MMIO_DRIVER_FEATURES,
    VIRTIO_MMIO_DRIVER_FEATURES_SEL, VIRTIO_MMIO_QUEUE_AVAIL_HIGH, VIRTIO_MMIO_QUEUE_AVAIL_LOW,
    VIRTIO_MMIO_QUEUE_DESC_HIGH, VIRTIO_MMIO_QUEUE_DESC_LOW, VIRTIO_MMIO_QUEUE_NUM,
    VIRTIO_MMIO_QUEUE_READY, VIRTIO_MMIO_QUEUE_SEL, VIRTIO_MMIO_QUEUE_USED_HIGH,
    VIRTIO_MMIO_QUEUE_USED_LOW, VIRTIO_MMIO_STATUS, VIRTIO_RING_F_EVENT_IDX, VirtioBlk,
    VirtioBlkOutHdr,
};

/// Build a `VirtioBlk` backed by an empty tempfile sized to
/// `capacity_bytes`. The default fixture for any test that doesn't
/// care about the backing-file contents.
pub(super) fn make_device(capacity_bytes: u64, throttle: DiskThrottle) -> VirtioBlk {
    let mut f = tempfile().expect("create tempfile for virtio-blk test backing");
    f.set_len(capacity_bytes)
        .expect("set tempfile length to capacity_bytes — usually fails when TMPDIR is full");
    f.rewind().expect("rewind tempfile after set_len");
    VirtioBlk::new(f, capacity_bytes, throttle)
}

/// MMIO read sugar: read 4 bytes at `offset` and decode as little-endian u32.
pub(super) fn read_reg(dev: &VirtioBlk, offset: u32) -> u32 {
    let mut buf = [0u8; 4];
    dev.mmio_read(offset as u64, &mut buf);
    u32::from_le_bytes(buf)
}

/// MMIO write sugar: encode `val` as little-endian u32 and write at `offset`.
pub(super) fn write_reg(dev: &mut VirtioBlk, offset: u32, val: u32) {
    dev.mmio_write(offset as u64, &val.to_le_bytes());
}

/// Drive the device through the full virtio init sequence up to
/// `DRIVER_OK`. Mirrors the virtio_console `init_device` helper.
/// Used by tests that need a fully negotiated device.
pub(super) fn init_device(dev: &mut VirtioBlk) {
    write_reg(dev, VIRTIO_MMIO_STATUS, S_ACK);
    write_reg(dev, VIRTIO_MMIO_STATUS, S_DRV);
    // Negotiate VIRTIO_F_VERSION_1 — the device requires this and
    // the FSM rejects FEATURES_OK if the modern bit is missing
    // from the driver-acknowledged set.
    write_reg(dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
    write_reg(
        dev,
        VIRTIO_MMIO_DRIVER_FEATURES,
        1 << (VIRTIO_F_VERSION_1 - 32),
    );
    write_reg(dev, VIRTIO_MMIO_STATUS, S_FEAT);
    write_reg(dev, VIRTIO_MMIO_STATUS, S_OK);
}

/// Single-region GuestMemoryMmap at GPA 0 — sufficient for direct
/// handler testing where the test owns the GPAs.
pub(super) fn make_guest_mem(bytes: usize) -> GuestMemoryMmap {
    GuestMemoryMmap::from_ranges(&[(GuestAddress(0), bytes)]).expect("create test guest mem")
}

/// Build a backing file pre-populated with a fixed pattern so a
/// `handle_read` can verify the file contents propagate to guest
/// memory.
pub(super) fn make_backed_file_with_pattern(capacity: u64, pattern: u8) -> File {
    let mut f = tempfile().unwrap();
    f.set_len(capacity).unwrap();
    f.rewind().unwrap();
    let buf = vec![pattern; capacity as usize];
    f.write_all(&buf).unwrap();
    f.rewind().unwrap();
    f
}

/// Plant a `VirtioBlkOutHdr` at `header_addr` in `mem` so a
/// chain-level test can build a request with the correct header
/// type/sector. The header_addr is the GPA the header descriptor
/// will point at.
pub(super) fn write_blk_header(
    mem: &GuestMemoryMmap,
    header_addr: GuestAddress,
    req_type: u32,
    sector: u64,
) {
    let hdr = VirtioBlkOutHdr {
        type_: req_type,
        _ioprio: 0,
        sector,
    };
    // `VirtioBlkOutHdr` implements `ByteValued`, so `write_obj`
    // serialises the struct into guest memory directly without
    // any unsafe pointer casts.
    mem.write_obj(hdr, header_addr).expect("plant header");
}

/// Configure the device's queue to point at the mock's
/// desc/avail/used addresses, then drive the FSM to DRIVER_OK.
/// After this call, MMIO writes of QUEUE_NOTIFY fire
/// `process_requests` which sees whatever chain `mock` has set
/// up.
///
/// Asserts the FSM actually reached DRIVER_OK before returning
/// — a feature-negotiation regression that wedged the device
/// at FEATURES_OK would otherwise produce confusing
/// "process_requests sees an empty queue" failures from every
/// chain test downstream. Asserting here surfaces the FSM bug
/// at its source.
pub(super) fn wire_device_to_mock(dev: &mut VirtioBlk, mock: &MockSplitQueue<GuestMemoryMmap>) {
    // Walk the FSM up to FEATURES_OK so queue config is accepted.
    // DRIVER_OK is set last because queue config is rejected after.
    write_reg(dev, VIRTIO_MMIO_STATUS, S_ACK);
    write_reg(dev, VIRTIO_MMIO_STATUS, S_DRV);
    write_reg(dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
    write_reg(
        dev,
        VIRTIO_MMIO_DRIVER_FEATURES,
        1 << (VIRTIO_F_VERSION_1 - 32),
    );
    write_reg(dev, VIRTIO_MMIO_STATUS, S_FEAT);

    write_reg(dev, VIRTIO_MMIO_QUEUE_SEL, 0);
    write_reg(dev, VIRTIO_MMIO_QUEUE_NUM, QUEUE_MAX_SIZE as u32);
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
    // Sanity: the FSM must have actually accepted DRIVER_OK.
    // A regression in `set_status` that rejects the final
    // transition would otherwise leave the device wedged at
    // FEATURES_OK and every chain test downstream would see
    // an empty queue.
    assert_eq!(
        dev.device_status, S_OK,
        "wire_device_to_mock: FSM did not reach DRIVER_OK \
         (got {:#x}) — feature negotiation likely regressed",
        dev.device_status,
    );
}

/// Same as `wire_device_to_mock` but additionally negotiates
/// VIRTIO_RING_F_EVENT_IDX (bit 29 in the low feature half) and
/// places the used ring at a separate GPA (`used_override_addr`)
/// far from the avail ring's `used_event` field. After this
/// call, `process_requests` consults the avail ring's
/// `used_event` field via `Queue::needs_notification` and may
/// suppress the irqfd write — the rest of the suite uses
/// `wire_device_to_mock` (legacy path) where every drain
/// unconditionally fires.
///
/// `queue_size` is load-bearing for EVENT_IDX correctness:
/// `Queue::used_event` (the private helper that the public
/// `Queue::needs_notification` delegates to) computes the
/// avail-ring field offset as `VIRTQ_AVAIL_RING_HEADER_SIZE
/// + size * VIRTQ_AVAIL_ELEMENT_SIZE = 4 + size * 2`. The
/// device's negotiated queue size must match the mock's queue
/// size or the device reads `used_event` from the wrong GPA.
/// Existing legacy-path tests don't care because
/// `needs_notification` returns Ok(true) without consulting
/// `used_event` when `event_idx_enabled=false`.
///
/// `used_override_addr`: where the device should place the
/// used ring. The MockSplitQueue's default used ring address
/// overlaps the avail ring's `used_event` field (the mock
/// computes `used_addr = avail.end().align_up(4)` where
/// `avail.end()` does NOT include the trailing used_event
/// field — so add_used writes clobber the planted threshold).
/// Pass an address well above the avail ring's footprint
/// (`avail_addr + 4 + size*2 + 2 + slack`) to avoid the
/// collision.
pub(super) fn wire_device_to_mock_with_event_idx(
    dev: &mut VirtioBlk,
    mock: &MockSplitQueue<GuestMemoryMmap>,
    queue_size: u16,
    used_override_addr: GuestAddress,
) {
    write_reg(dev, VIRTIO_MMIO_STATUS, S_ACK);
    write_reg(dev, VIRTIO_MMIO_STATUS, S_DRV);
    // Low half: VIRTIO_RING_F_EVENT_IDX is bit 29.
    write_reg(dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
    write_reg(
        dev,
        VIRTIO_MMIO_DRIVER_FEATURES,
        1u32 << VIRTIO_RING_F_EVENT_IDX,
    );
    // High half: VIRTIO_F_VERSION_1 is bit 32, i.e. bit 0 of
    // the high page.
    write_reg(dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
    write_reg(
        dev,
        VIRTIO_MMIO_DRIVER_FEATURES,
        1 << (VIRTIO_F_VERSION_1 - 32),
    );
    write_reg(dev, VIRTIO_MMIO_STATUS, S_FEAT);

    write_reg(dev, VIRTIO_MMIO_QUEUE_SEL, 0);
    write_reg(dev, VIRTIO_MMIO_QUEUE_NUM, queue_size as u32);
    let desc = mock.desc_table_addr().0;
    let avail = mock.avail_addr().0;
    let used = used_override_addr.0;
    write_reg(dev, VIRTIO_MMIO_QUEUE_DESC_LOW, desc as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_DESC_HIGH, (desc >> 32) as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_AVAIL_LOW, avail as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_AVAIL_HIGH, (avail >> 32) as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_USED_LOW, used as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_USED_HIGH, (used >> 32) as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_READY, 1);
    write_reg(dev, VIRTIO_MMIO_STATUS, S_OK);
    assert_eq!(
        dev.device_status, S_OK,
        "wire_device_to_mock_with_event_idx: FSM did not reach \
         DRIVER_OK (got {:#x})",
        dev.device_status,
    );
    // Sanity: the device must have observed and stored the
    // EVENT_IDX bit. Without this assertion, a regression in
    // driver_features wiring would silently downgrade every
    // EVENT_IDX test to the legacy path.
    assert_ne!(
        dev.driver_features & (1u64 << VIRTIO_RING_F_EVENT_IDX),
        0,
        "VIRTIO_RING_F_EVENT_IDX missing from driver_features after \
         wire_device_to_mock_with_event_idx",
    );
}

/// Compute the GPA of the avail ring's `used_event` field for a
/// given queue size. Layout per virtio-v1.2 §2.7.6: the avail
/// ring is `flags(2) + idx(2) + ring[size]*2 + used_event(2)`.
/// Mirrors the offset arithmetic in
/// `virtio-queue::queue::Queue::used_event` which uses
/// `VIRTQ_AVAIL_RING_HEADER_SIZE + size * VIRTQ_AVAIL_ELEMENT_SIZE`.
pub(super) fn used_event_addr(avail_addr: GuestAddress, queue_size: u16) -> GuestAddress {
    // Header (4 bytes: flags + idx) + ring entries (2 bytes each).
    avail_addr
        .checked_add(4 + queue_size as u64 * 2)
        .expect("used_event_addr overflow")
}

/// Build a guest memory map sized to host both the queue
/// descriptor/avail/used rings (placed at GPA 0..) and the
/// chain's data buffers (placed above the ring region).
/// 1 MB total — generous so neither the rings nor the test
/// payloads collide.
pub(super) fn make_chain_test_mem() -> GuestMemoryMmap {
    GuestMemoryMmap::from_ranges(&[(GuestAddress(0), 1 << 20)])
        .expect("create chain test guest mem")
}

/// Build a `(VirtioBlk, MockSplitQueue)` pair sharing the same
/// guest-memory borrow, so a chain test can be set up in one
/// helper call instead of three. `cap` is fixed at 4 KiB (the
/// established small-disk size used by the surrounding chain
/// tests), backing pre-filled with `0xAB` so reads see a
/// deterministic pattern. Queue size is 16 — matches the
/// existing chain-test default.
///
/// `mem` is owned by the caller because `MockSplitQueue<'a, M>`
/// borrows `mem` (mock.rs `pub struct MockSplitQueue<'a, M> {
/// mem: &'a M, ... }`); bundling the owner and the borrower in
/// one return tuple is a self-referential struct, which Rust
/// does not support. Caller pattern:
/// ```ignore
/// let mem = make_chain_test_mem();
/// let (mut dev, mock) = setup_blk(&mem, false, DiskThrottle::default());
/// ```
pub(super) fn setup_blk<'a>(
    mem: &'a GuestMemoryMmap,
    read_only: bool,
    throttle: DiskThrottle,
) -> (VirtioBlk, MockSplitQueue<'a, GuestMemoryMmap>) {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let dev = VirtioBlk::with_options(f, cap, throttle, read_only);
    let mock = MockSplitQueue::create(mem, GuestAddress(0), 16);
    (dev, mock)
}

/// Build a `VirtioBlk` ready to drive the throttle-stall gauge
/// path: capacity 4 KiB, `iops=1` rate (1 token/sec), bucket
/// drained, a single 1-sector READ chain (header at `0x4000`,
/// data at `0x5000`, status at `0x6000`) planted in the avail
/// ring, FSM walked to DRIVER_OK, and `last_refill` pinned at
/// `Instant::now()` so any in-place refill yields zero tokens.
///
/// Multiple gauge-transition tests share this exact setup;
/// extracting it here prevents drift between them — when the
/// gauge invariant or the chain shape changes, this one site
/// updates instead of every call site. Each call site adds only
/// the per-test action sequence (MMIO QUEUE_NOTIFY versus direct
/// `drain_bracket_impl`, pre-write of a status sentinel, reset,
/// etc.) and the per-test assertions.
///
/// Why iops=1 (not iops=N): a 1-token bucket plus a planted
/// 1-sector READ chain forces the second consume-attempt to
/// stall exactly once, which is the gauge state-transition the
/// tests pin (0 → 1 on first stall, 1 → 0 on retry success).
/// Higher rates would refill mid-test and the deficit math
/// (`nanos_until_n_tokens` = `1_000_000_000` ns at rate=1) would
/// shift, breaking the assertions in the inline-redrain tests.
///
/// Uses `setup_blk` for the device + mock construction, then
/// extends it with the throttle drain + chain plant + FSM walk.
/// `mem` is borrowed by `MockSplitQueue` only during chain
/// construction; once `wire_device_to_mock` has copied the
/// queue addresses into the device, the mock is dropped here
/// and the helper returns just the device. Caller still owns
/// `mem` for the duration of the test (the device's
/// `OnceLock<GuestMemoryMmap>` holds a separate `clone()` of
/// the same backing).
pub(super) fn setup_iops1_drained_chain(mem: &GuestMemoryMmap) -> VirtioBlk {
    let throttle = DiskThrottle {
        iops: std::num::NonZeroU64::new(1),
        bytes_per_sec: None,
        iops_burst_capacity: None,
        bytes_burst_capacity: None,
    };
    let (mut dev, mock) = setup_blk(mem, false, throttle);

    // Drain the 1-token bucket so the next consume(1) stalls.
    // Pinning `last_refill` on both sides of the consume keeps
    // wall-clock drift at rate=1/sec (one token every full
    // second) from leaking even a partial token in.
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(std::time::Instant::now());
    assert!(
        dev.worker.state_mut().ops_bucket.consume(1),
        "drain the 1-token bucket on the freshly-built device",
    );
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(std::time::Instant::now());

    // Plant a standard 3-desc T_IN (read 1 sector) chain. The
    // addresses are fixed across all gauge tests — chain shape
    // is incidental to what the tests pin (gauge transitions),
    // so a single canonical chain is enough.
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    write_blk_header(mem, header_addr, VIRTIO_BLK_T_IN, 0);
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            512,
            VRING_DESC_F_WRITE as u16,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            status_addr.0,
            1,
            VRING_DESC_F_WRITE as u16,
            0,
        )),
    ];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_device_to_mock(&mut dev, &mock);

    // Re-pin after the FSM walk — `wire_device_to_mock`'s
    // MMIO writes take measurable wall time; at rate=1/sec one
    // token requires 1 s of elapsed time so realistically no
    // refill leaks through the floor-divide, but pinning here
    // matches what every existing call site did manually.
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(std::time::Instant::now());
    dev
}

/// Build a `VirtioBlk` ready to drive the throttle-stall gauge
/// path on the BYTES bucket (the iops bucket has tokens). Mirror
/// of `setup_iops1_drained_chain` for the bytes-only variant —
/// extracted so the bytes-only gauge tests share the same
/// chain-shape and pin-points as their iops-only counterparts,
/// preventing drift between the two transition surfaces.
///
/// Parameters:
/// * `iops_rate`: tokens/sec for the iops bucket. Must be large
///   enough to satisfy `consume(1)` against any reasonable wall
///   time (16/sec is plenty — the 1-token-per-request charge
///   never exhausts the bucket in tests).
/// * `bytes_rate`: tokens/sec for the bytes bucket. The bucket
///   is pre-drained via `consume(bytes_rate)` so the next
///   `consume(bytes_rate)` stalls; pick a value matching the
///   chain's `data_len` to land the deficit math at exactly
///   `1_000_000_000 ns` (deficit==capacity, rate==capacity → 1 s).
///
/// The chain is a 1-segment T_IN read of `bytes_rate` bytes:
/// header at `0x4000`, data at `0x5000`, status at `0x6000`.
/// Both buckets' `last_refill` are pinned at `Instant::now()`
/// after the FSM walk so the elapsed wall time during MMIO
/// writes does not passively grant tokens before the first
/// drain.
///
/// Why bytes_rate matters for the deficit math:
/// `nanos_until_n_tokens(bytes_rate)` against an empty bucket
/// at `refill_rate=bytes_rate` returns `bytes_rate * 1e9 /
/// bytes_rate = 1_000_000_000` ns. A test that pins
/// `wait_nanos=1_000_000_000` depends on this equality.
pub(super) fn setup_bytes_only_drained_chain(
    mem: &GuestMemoryMmap,
    iops_rate: u64,
    bytes_rate: u64,
) -> VirtioBlk {
    let throttle = DiskThrottle {
        iops: std::num::NonZeroU64::new(iops_rate),
        bytes_per_sec: std::num::NonZeroU64::new(bytes_rate),
        iops_burst_capacity: None,
        bytes_burst_capacity: None,
    };
    let (mut dev, mock) = setup_blk(mem, false, throttle);

    // Drain ONLY the bytes bucket so the first drain stalls on
    // bytes alone. Pin both buckets' last_refill so the
    // bucket arithmetic doesn't passively grant or revoke
    // tokens between assertions.
    let now0 = std::time::Instant::now();
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(now0);
    dev.worker
        .state_mut()
        .bytes_bucket
        .set_last_refill_for_test(now0);
    assert!(
        dev.worker.state_mut().bytes_bucket.consume(bytes_rate),
        "drain the bytes bucket on the freshly-built device",
    );
    // Re-pin AFTER consume so the next can_consume sees the
    // drained state at exactly t=now0.
    dev.worker
        .state_mut()
        .bytes_bucket
        .set_last_refill_for_test(now0);

    // Plant the standard 3-desc T_IN chain. data_len matches
    // bytes_rate so the chain's bytes-bucket charge equals the
    // bucket capacity — landing the deficit at 1_000_000_000 ns
    // when the bucket is empty.
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    write_blk_header(mem, header_addr, VIRTIO_BLK_T_IN, 0);
    let data_len_u32 =
        u32::try_from(bytes_rate).expect("bytes_rate fits in a single descriptor for tests");
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            data_len_u32,
            VRING_DESC_F_WRITE as u16,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            status_addr.0,
            1,
            VRING_DESC_F_WRITE as u16,
            0,
        )),
    ];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_device_to_mock(&mut dev, &mock);

    // Re-pin both buckets after the FSM walk — wire_device_to_mock's
    // MMIO writes take measurable wall time.
    let now1 = std::time::Instant::now();
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(now1);
    dev.worker
        .state_mut()
        .bytes_bucket
        .set_last_refill_for_test(now1);
    dev
}

