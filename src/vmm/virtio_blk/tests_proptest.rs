// ----------------------------------------------------------------------------
// proptest fuzz suite for process_requests.
//
// Property-driven coverage of the descriptor-chain parsing path: generate
// arbitrary sequences of descriptors (random `addr`/`len`/`flags`/`next`)
// and feed them through `process_requests` via `MockSplitQueue` +
// QUEUE_NOTIFY. Mirrors the firecracker pattern of systematic chain
// corruption: every randomly-generated chain element exercises a code
// path the hand-curated tests don't reach.
//
// The harness asserts the device's hostile-input contract:
//   1. No panic, OOB index, or unwrap-on-None — process_requests must
//      handle every input without crashing the thread running drain_bracket_impl.
//   2. Forward progress: for every chain that reaches `process_requests`,
//      the device either advances `used.idx` (status published) OR
//      bumps `io_errors` (chain dropped because no observable status
//      descriptor exists). Silent stalls — used.idx unchanged AND
//      no counter bump — would let a hostile guest pin the queue
//      indefinitely.
//   3. Counter monotonicity: counters never decrement.
//
// Counter assertions reference the same `VirtioBlkCounters` fields the
// production failure-dump renderer reads, so a regression that adds a
// new code path which neither bumps a counter nor advances used.idx
// surfaces as a property violation.
// ----------------------------------------------------------------------------

use super::{
    DiskThrottle, REQ_QUEUE, VIRTIO_BLK_OUTHDR_SIZE, VIRTIO_BLK_S_IOERR, VIRTIO_BLK_S_OK,
    VIRTIO_BLK_S_UNSUPP, VIRTIO_BLK_T_FLUSH, VIRTIO_BLK_T_IN, VIRTIO_BLK_T_OUT,
    VIRTIO_MMIO_QUEUE_NOTIFY, VirtioBlk, VirtioBlkOutHdr,
};
use proptest::prelude::*;
use std::num::NonZeroU64;
use std::os::unix::fs::FileExt;
use std::sync::atomic::Ordering;
use tempfile::tempfile;
use virtio_bindings::bindings::virtio_ring::{VRING_DESC_F_NEXT, VRING_DESC_F_WRITE};
use virtio_queue::QueueT;
use virtio_queue::desc::{RawDescriptor, split::Descriptor as SplitDescriptor};
use virtio_queue::mock::MockSplitQueue;
use vm_memory::{Address, Bytes, GuestAddress, GuestMemoryMmap};

/// Shape of one random descriptor. `flags` is restricted to the three
/// bits the device cares about (NEXT, WRITE, INDIRECT); higher bits
/// would be silently masked by the `virtio-queue` parser anyway, so
/// generating them adds no coverage. `next` is a full `u16` because
/// out-of-range values are part of the test surface — the queue
/// iterator must stop without panicking when `next >= queue_size`.
#[derive(Debug, Clone, Copy)]
struct FuzzDesc {
    addr: u64,
    len: u32,
    flags: u16,
    next: u16,
}

/// Strategy for a single descriptor.
///
/// `addr` ranges far beyond the 1 MiB guest-memory region so a
/// substantial fraction of generated descriptors point at unmapped
/// guest physical addresses — the device must reject those via
/// `mem.read_slice`/`write_slice` errors rather than panic.
/// Specifically we span `0..2^24` which covers the entire 1 MiB
/// region (in-range) plus 15 MiB beyond it (unmapped) — a roughly
/// 1:15 valid-to-invalid ratio that keeps both happy and sad paths
/// well-exercised.
///
/// `len` ranges past `VIRTIO_BLK_SIZE_MAX = 1 MiB` so the SIZE_MAX
/// gate is exercised. The `0..=8 MiB` range generates enough
/// over-cap descriptors to randomly trip the gate without making
/// every chain trivially over-cap.
///
/// `flags` is `0..8` (3 bits), giving every combination of
/// NEXT/WRITE/INDIRECT.
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

/// Strategy for a chain of 1..=200 descriptors. Includes an upper
/// bound on chain length matching the task's "1-200" requirement;
/// the lower bound of 1 ensures the avail ring always has at least
/// one chain head so `process_requests` always traverses at least
/// one iteration of its drain loop (the test's progress invariant
/// presumes drain occurred).
fn fuzz_chain_strategy() -> impl Strategy<Value = Vec<FuzzDesc>> {
    prop::collection::vec(fuzz_desc_strategy(), 1..=200)
}

/// Build the device + 1 MiB guest memory + mock queue with a
/// 256-slot descriptor table (`QUEUE_MAX_SIZE`). 256 matches the
/// device's advertised maximum and is large enough to hold the
/// maximum proptest-generated chain (200 descriptors) with room to
/// spare for the rings.
fn build_fuzz_fixture() -> (VirtioBlk, GuestMemoryMmap) {
    let cap = 4096u64;
    let f = tempfile().expect("create tempfile for fuzz backing");
    f.set_len(cap).expect("set tempfile length to fuzz cap");
    // Write a sentinel pattern so `T_IN` reads see deterministic
    // backing data; not load-bearing for the test invariants but
    // useful when debugging counter-exemplar failures.
    f.write_at(&[0xAB; 4096], 0).expect("seed backing pattern");
    let dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    // 1 MiB guest memory at GPA 0 — same sizing as the
    // hand-curated chain tests' `make_chain_test_mem`. Generated
    // addresses span 0..2^24, so guest-mem-bound addresses
    // resolve to in-range reads/writes while the rest hit the
    // 16 MiB-wide invalid zone.
    let mem = GuestMemoryMmap::from_ranges(&[(GuestAddress(0), 1 << 20)])
        .expect("create proptest guest mem");
    (dev, mem)
}

/// Drive the device through the full FSM up to DRIVER_OK with the
/// mock queue pinned. Mirrors `wire_device_to_mock` from the
/// hand-curated chain tests, but inlined here so the proptest
/// module is self-contained (no super-private helper imports).
fn wire_fuzz_device(dev: &mut VirtioBlk, mock: &MockSplitQueue<GuestMemoryMmap>) {
    use super::{
        QUEUE_MAX_SIZE, S_ACK, S_DRV, S_FEAT, S_OK, VIRTIO_MMIO_DRIVER_FEATURES,
        VIRTIO_MMIO_DRIVER_FEATURES_SEL, VIRTIO_MMIO_QUEUE_AVAIL_HIGH,
        VIRTIO_MMIO_QUEUE_AVAIL_LOW, VIRTIO_MMIO_QUEUE_DESC_HIGH, VIRTIO_MMIO_QUEUE_DESC_LOW,
        VIRTIO_MMIO_QUEUE_NUM, VIRTIO_MMIO_QUEUE_READY, VIRTIO_MMIO_QUEUE_SEL,
        VIRTIO_MMIO_QUEUE_USED_HIGH, VIRTIO_MMIO_QUEUE_USED_LOW, VIRTIO_MMIO_STATUS,
    };
    use virtio_bindings::virtio_config::VIRTIO_F_VERSION_1;
    let write_reg = |dev: &mut VirtioBlk, offset: u32, val: u32| {
        dev.mmio_write(offset as u64, &val.to_le_bytes());
    };
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
}

/// Read the used-ring `idx` field. Mirrors the inline
/// `read_obj(used_addr + 2)` pattern used by hand-curated tests;
/// extracted to a helper so the proptest assertions stay
/// declarative. `+ 2` skips the 2-byte `flags` field at the head
/// of the used ring (`virtq_used.flags`, `virtq_used.idx`).
fn read_used_idx(mem: &GuestMemoryMmap, mock: &MockSplitQueue<GuestMemoryMmap>) -> u16 {
    mem.read_obj::<u16>(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx")
}

/// Snapshot of the counters used as a per-iteration progress
/// witness. Captures every counter the device mutates so the
/// "something happened" check stays exhaustive.
#[derive(Default, Clone, Copy)]
struct CounterSnapshot {
    reads: u64,
    writes: u64,
    flushes: u64,
    bytes_read: u64,
    bytes_written: u64,
    throttled: u64,
    io_errors: u64,
}

fn snapshot_counters(dev: &VirtioBlk) -> CounterSnapshot {
    let c = dev.counters();
    CounterSnapshot {
        reads: c.reads_completed.load(Ordering::Relaxed),
        writes: c.writes_completed.load(Ordering::Relaxed),
        flushes: c.flushes_completed.load(Ordering::Relaxed),
        bytes_read: c.bytes_read.load(Ordering::Relaxed),
        bytes_written: c.bytes_written.load(Ordering::Relaxed),
        throttled: c.throttled_count.load(Ordering::Relaxed),
        io_errors: c.io_errors.load(Ordering::Relaxed),
    }
}

/// Build a fuzz fixture whose throttle is configured at iops=1
/// AND drained-at-construction so any chain reaching the
/// per-request throttle gate stalls. Used by the throttle-stall
/// proptest below to exercise the rollback path
/// (`set_next_avail` rewind, `currently_stalled` true→true /
/// false→true transitions, `throttled_count` event recording)
/// against random well-formed chains.
///
/// Mirrors `build_fuzz_fixture` but swaps the throttle and
/// drains the bucket via the test-only `set_last_refill_for_test`
/// + `consume(1)` seam used by the hand-curated stall tests.
fn build_throttled_fuzz_fixture() -> (VirtioBlk, GuestMemoryMmap) {
    let cap = 4096u64;
    let f = tempfile().expect("create tempfile for throttled fuzz backing");
    f.set_len(cap).expect("set tempfile length to fuzz cap");
    f.write_at(&[0xAB; 4096], 0).expect("seed backing pattern");
    let throttle = DiskThrottle {
        iops: NonZeroU64::new(1),
        bytes_per_sec: None,
        iops_burst_capacity: None,
        bytes_burst_capacity: None,
    };
    let mut dev = VirtioBlk::new(f, cap, throttle);
    // Drain the bucket and pin last_refill so refill on the
    // next consume yields 0 tokens. The proptest fires a
    // single QUEUE_NOTIFY per case; pinning here keeps the
    // bucket empty for the duration of the case regardless of
    // how long the test runs.
    let now = std::time::Instant::now();
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(now);
    assert!(dev.worker.state_mut().ops_bucket.consume(1));
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(now);
    let mem = GuestMemoryMmap::from_ranges(&[(GuestAddress(0), 1 << 20)])
        .expect("create proptest guest mem");
    (dev, mem)
}

/// One well-formed virtio-blk request chain shape: a request
/// type plus 1..=8 data segments. The proptest strategy
/// `well_formed_chain_strategy` materialises this into a
/// header + N data + status descriptor sequence in guest
/// memory at deterministic, well-mapped addresses.
///
/// Distinct from `FuzzDesc` — that strategy generates
/// arbitrary RAW descriptors (random `addr`/`len`/`flags`/`next`)
/// to fuzz the chain-shape parser. This strategy instead
/// generates VALID chain shapes with random multiplicities
/// to fuzz the post-parse stall path: every chain produced
/// here is well-formed, so the throttle gate is the
/// dominant rejection point.
#[derive(Debug, Clone)]
struct WellFormedChain {
    /// Request type. Restricted to T_IN/T_OUT/T_FLUSH so the
    /// chain has predictable direction-flag requirements.
    /// T_GET_ID is omitted because it's a metadata read with
    /// a fixed 20-byte payload requirement that doesn't
    /// stress the throttle dimensions tested here.
    req_type: u32,
    /// Starting sector. Bounded at `0..8` since the fuzz
    /// fixture's capacity is 4096 bytes = 8 sectors. Out-of-
    /// range sectors would surface as IOERR from the handler
    /// (after throttle), but the throttle gate runs BEFORE
    /// the handler — so an out-of-range sector still exercises
    /// the stall path. Bounding the strategy keeps the fuzz
    /// signal focused.
    sector: u64,
    /// Data-segment count. 1..=8 is the practical range that
    /// stresses the data-length aggregation (`data_len.iter().sum()`)
    /// and the throttle's bytes-bucket path. T_FLUSH ignores
    /// this — it gets header + status only.
    ///
    /// Capped at 8 because the fuzz fixture's 4 KiB capacity
    /// limits useful payload to 8 sectors (8 * 512 = 4096
    /// bytes); larger counts would either overlap addresses
    /// or trip the data-len > capacity gate before the
    /// throttle fires.
    n_data_segments: u32,
    /// Per-segment length in 512-byte sectors (1..=4). The
    /// total payload is bounded above by 8 sectors via the
    /// strategy's interaction (n_data_segments × seg_sectors
    /// ≤ 8 enforced at materialisation time by clamping the
    /// final segment).
    seg_sectors: u32,
}

fn well_formed_chain_strategy() -> impl Strategy<Value = WellFormedChain> {
    // Use prop_oneof so each case has a clean mapping from
    // the random input to a request type — distributing across
    // the three types we care about uniformly.
    let req_type = prop_oneof![
        Just(VIRTIO_BLK_T_IN),
        Just(VIRTIO_BLK_T_OUT),
        Just(VIRTIO_BLK_T_FLUSH),
    ];
    (req_type, 0u64..8u64, 1u32..=8u32, 1u32..=4u32).prop_map(
        |(req_type, sector, n_data_segments, seg_sectors)| WellFormedChain {
            req_type,
            sector,
            n_data_segments,
            seg_sectors,
        },
    )
}

/// Plant a `WellFormedChain` into guest memory + the mock
/// queue's descriptor table at well-mapped addresses. Returns
/// the status descriptor's GPA so the caller can verify
/// post-notify whether the device wrote to it (sentinel
/// survival).
///
/// Memory layout (deterministic so failure shrinking is
/// reproducible):
///   - 0x4000: header (16 bytes)
///   - 0x5000: data segments (back-to-back, 0x200-aligned)
///   - 0xC000: status byte (sentinel-pre-fill 0xEE)
///
/// All within the 1 MiB guest memory region so the device
/// reaches the throttle gate without earlier guest-memory
/// rejection paths firing.
fn plant_well_formed_chain(
    mem: &GuestMemoryMmap,
    mock: &MockSplitQueue<GuestMemoryMmap>,
    chain: &WellFormedChain,
) -> GuestAddress {
    let header_addr = GuestAddress(0x4000);
    let status_addr = GuestAddress(0xC000);
    // Plant the header.
    let hdr = VirtioBlkOutHdr {
        type_: chain.req_type,
        _ioprio: 0,
        sector: chain.sector,
    };
    mem.write_obj(hdr, header_addr).expect("plant header");
    // Plant the status sentinel so post-notify we can
    // detect whether the device wrote to it.
    mem.write_slice(&[0xEEu8], status_addr)
        .expect("plant status sentinel");

    // Build the descriptor list. T_FLUSH carries no data
    // segments — header + status only. T_IN/T_OUT carry
    // chain.n_data_segments data descriptors of
    // chain.seg_sectors * 512 bytes each, capped at the fuzz
    // fixture's 4 KiB capacity.
    let mut descs: Vec<RawDescriptor> = Vec::new();
    let header_link_to = if chain.req_type == VIRTIO_BLK_T_FLUSH {
        // Flush: header → status, single link.
        1u16
    } else {
        // Read/write: header → data[0] → ... → data[N-1] → status.
        1u16
    };
    descs.push(RawDescriptor::from(SplitDescriptor::new(
        header_addr.0,
        VIRTIO_BLK_OUTHDR_SIZE as u32,
        VRING_DESC_F_NEXT as u16,
        header_link_to,
    )));

    if chain.req_type != VIRTIO_BLK_T_FLUSH {
        // Cap total payload at 8 sectors (4 KiB). The fuzz
        // fixture's capacity is 4096 bytes; a chain whose
        // data_len exceeds capacity would IOERR before the
        // throttle gate. Keep the throttle gate as the
        // dominant rejection so the test signal is clean.
        let max_seg_count = (8u32)
            .saturating_div(chain.seg_sectors)
            .max(1)
            .min(chain.n_data_segments);
        // Direction flag: T_IN data segments are device-
        // writable; T_OUT data segments are device-readable.
        let data_flag = if chain.req_type == VIRTIO_BLK_T_IN {
            VRING_DESC_F_WRITE as u16
        } else {
            0u16
        };
        for i in 0..max_seg_count {
            let seg_addr = 0x5000u64 + (i as u64 * 0x800);
            let seg_len = chain.seg_sectors * 512;
            let next_idx = i + 2; // header is 0, data starts at 1
            descs.push(RawDescriptor::from(SplitDescriptor::new(
                seg_addr,
                seg_len,
                data_flag | VRING_DESC_F_NEXT as u16,
                next_idx as u16,
            )));
        }
    }
    // Status descriptor — always device-writable, length 1.
    descs.push(RawDescriptor::from(SplitDescriptor::new(
        status_addr.0,
        1,
        VRING_DESC_F_WRITE as u16,
        0,
    )));

    mock.build_desc_chain(&descs).expect("build chain");
    status_addr
}

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

    /// Random descriptor chains via `add_desc_chains` MUST produce
    /// forward progress: for every notify, at least one of
    /// `used.idx` advance, `io_errors`, `reads_completed`,
    /// `writes_completed`, `flushes_completed`, or
    /// `throttled_count` must show movement. A chain that left
    /// every counter and used.idx static would represent a silent
    /// stall — virtio_blk has no `mq_ops->timeout`, so blk-mq
    /// alone never surfaces it; the guest only sees the stall
    /// once the hung-task watchdog fires
    /// (`kernel.hung_task_timeout_secs`, default 120 s) without
    /// the host having any visibility.
    ///
    /// Critically: this also pins panic-freeness. The proptest
    /// runner catches panics; a panic in process_requests under
    /// any input crashes the test with the offending shrunken
    /// case.
    #[test]
    fn process_requests_progress_under_random_chains(
        descs in fuzz_chain_strategy(),
    ) {
        let (mut dev, mem) = build_fuzz_fixture();
        // Mock with 256 slots — exactly QUEUE_MAX_SIZE, larger
        // than the 200-descriptor chain max.
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 256);
        dev.set_mem(mem.clone());
        wire_fuzz_device(&mut dev, &mock);

        // Convert FuzzDesc -> RawDescriptor.
        let raw_descs: Vec<RawDescriptor> = descs
            .iter()
            .map(|d| {
                RawDescriptor::from(SplitDescriptor::new(
                    d.addr,
                    d.len,
                    d.flags,
                    d.next,
                ))
            })
            .collect();

        // Prime the avail ring + descriptor table. Using
        // add_desc_chains rather than build_desc_chain so the
        // generated `next`/`flags` fields are preserved verbatim
        // — `build_desc_chain` would auto-fix links and erase
        // the test's mutation of those fields.
        mock.add_desc_chains(&raw_descs, 0)
            .expect("plant descriptors into avail ring");

        let before_used = read_used_idx(&mem, &mock);
        let before = snapshot_counters(&dev);

        // Fire QUEUE_NOTIFY. process_requests is the system
        // under test. A panic here would propagate up and fail
        // the proptest, with shrinking pinpointing the minimal
        // offending input. A hang (e.g. infinite chain loop)
        // would surface as the test runner's wall-clock timeout.
        dev.mmio_write(
            VIRTIO_MMIO_QUEUE_NOTIFY as u64,
            &(REQ_QUEUE as u32).to_le_bytes(),
        );

        let after_used = read_used_idx(&mem, &mock);
        let after = snapshot_counters(&dev);

        // Counter monotonicity: every counter only ever
        // increases. A regression that subtracted from a counter
        // (e.g. on rollback) would surface here regardless of
        // whether progress overall happened. used.idx advances
        // monotonically modulo wrap; with at most 200 chains and
        // a 256-slot queue the wrap never triggers, so we can
        // assert plain >=.
        prop_assert!(after.reads >= before.reads);
        prop_assert!(after.writes >= before.writes);
        prop_assert!(after.flushes >= before.flushes);
        prop_assert!(after.bytes_read >= before.bytes_read);
        prop_assert!(after.bytes_written >= before.bytes_written);
        prop_assert!(after.throttled >= before.throttled);
        prop_assert!(after.io_errors >= before.io_errors);
        prop_assert!(after_used >= before_used);

        // Forward-progress invariant. With at least one
        // descriptor in the avail ring (chain length >= 1
        // guaranteed by fuzz_chain_strategy), process_requests
        // ALWAYS reaches at least one of:
        //   (a) `publish_completion` with a successful status
        //       write → used.idx advances by >= 1
        //   (b) the no-status-descriptor drop branch →
        //       io_errors bumps without used.idx advancing
        //   (c) a successful happy-path completion (read /
        //       write / flush / throttle / unsupp), each of
        //       which advances used.idx and bumps a counter
        //
        // The `progress` sum captures every visible side effect.
        // A regression that introduced a fourth code path
        // (silent drop with no counter and no used.idx advance)
        // would fail this assertion — exactly the silent-stall
        // class of bug the property is designed to catch.
        let used_delta = (after_used - before_used) as u64;
        let counter_delta = (after.reads - before.reads)
            + (after.writes - before.writes)
            + (after.flushes - before.flushes)
            + (after.throttled - before.throttled)
            + (after.io_errors - before.io_errors);
        let progress = used_delta + counter_delta;
        prop_assert!(
            progress >= 1,
            "no visible progress: used_delta={} counter_delta={} \
             (chain len={}, first_desc=({:#x},{},{:#x},{}))",
            used_delta,
            counter_delta,
            descs.len(),
            descs[0].addr,
            descs[0].len,
            descs[0].flags,
            descs[0].next,
        );
    }

    /// Random `addr` of the FIRST descriptor (treated as the
    /// header) — fuzz the header read path. Plants a syntactically
    /// minimal chain (header + status, header pointed at random
    /// guest addresses including unmapped regions) and asserts
    /// that the device either successfully decodes the header (if
    /// the random bytes happen to deserialize cleanly into a
    /// `VirtioBlkOutHdr`) OR rejects with S_IOERR. Either way the
    /// chain must complete (used.idx advances by 1) since the
    /// status descriptor is well-formed.
    ///
    /// This complements the broad chain-mutation property by
    /// pinning a specific high-risk path: every byte read by
    /// `mem.read_obj::<VirtioBlkOutHdr>(header_addr)` is
    /// attacker-controlled; a parser bug (e.g. assuming a valid
    /// req_type) would surface as a panic.
    #[test]
    fn random_header_addr_either_succeeds_or_ioerrs(
        header_addr_low in 0u64..(1u64 << 24),
    ) {
        let (mut dev, mem) = build_fuzz_fixture();
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 256);
        dev.set_mem(mem.clone());
        wire_fuzz_device(&mut dev, &mock);

        // Status_addr at 0x6000 — well within the 1 MiB region
        // and clear of the queue rings (which sit at GPA 0..a
        // few KiB).
        let status_addr = GuestAddress(0x6000);
        // Pre-fill status with a sentinel so we can detect
        // whether the device wrote a status byte. 0xEE is
        // distinct from S_OK (0), S_IOERR (1), S_UNSUPP (2).
        mem.write_slice(&[0xEEu8], status_addr).unwrap();

        let descs = [
            RawDescriptor::from(SplitDescriptor::new(
                header_addr_low,
                VIRTIO_BLK_OUTHDR_SIZE as u32,
                0, // device-readable, no NEXT — actually need NEXT
                1,
            )),
            RawDescriptor::from(SplitDescriptor::new(
                status_addr.0,
                1,
                VRING_DESC_F_WRITE as u16,
                0,
            )),
        ];
        // Use build_desc_chain so the NEXT/next links are
        // auto-set correctly — for this targeted test we want a
        // valid chain shape with only the header_addr fuzzed.
        mock.build_desc_chain(&descs).expect("build chain");
        dev.mmio_write(
            VIRTIO_MMIO_QUEUE_NOTIFY as u64,
            &(REQ_QUEUE as u32).to_le_bytes(),
        );

        // Status byte must be one of the canonical virtio-blk
        // status values OR remain the sentinel (the latter only
        // if status_addr write failed — impossible here since
        // status_addr = 0x6000 is in-range and writable).
        let mut s = [0u8; 1];
        mem.read_slice(&mut s, status_addr).unwrap();
        prop_assert!(
            s[0] == VIRTIO_BLK_S_OK as u8
                || s[0] == VIRTIO_BLK_S_IOERR as u8
                || s[0] == VIRTIO_BLK_S_UNSUPP as u8,
            "status byte {:#x} is not a valid virtio-blk status",
            s[0],
        );

        // used.idx advanced by exactly 1 — exactly one chain in
        // the avail ring, the device produced exactly one
        // completion. A chain-drop path (used.idx stays 0)
        // would mean the device skipped the chain entirely;
        // for this test shape that's impossible because
        // status_addr is mapped.
        let used_idx = read_used_idx(&mem, &mock);
        prop_assert_eq!(
            used_idx,
            1,
            "well-formed chain shape with random header_addr must \
             produce exactly one used-ring entry; got {}",
            used_idx,
        );
    }

    /// Random `len` on a single data descriptor — fuzz the
    /// SIZE_MAX gate and downstream length-arithmetic paths.
    /// Builds a valid header + 1 data segment + status chain
    /// where the data segment's length is randomised across the
    /// full u32 range (with bias toward the SIZE_MAX boundary).
    /// Asserts the chain always completes with a defined status
    /// byte and used.idx advances by 1.
    ///
    /// A regression that didn't cap data_len before computing
    /// `data_len * something` would surface as an integer
    /// overflow panic in debug builds; this property exercises
    /// the boundary where SIZE_MAX (1 MiB) is exceeded.
    #[test]
    fn random_data_len_either_succeeds_or_ioerrs(
        data_len in 0u32..(8u32 * 1024 * 1024),
        req_type in 0u32..=8u32,
    ) {
        let (mut dev, mem) = build_fuzz_fixture();
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 256);
        dev.set_mem(mem.clone());
        wire_fuzz_device(&mut dev, &mock);

        let header_addr = GuestAddress(0x4000);
        let data_addr = GuestAddress(0x5000);
        let status_addr = GuestAddress(0x6000);

        // Plant a header with the random req_type. ByteValued
        // serialisation matches the wire format.
        let hdr = VirtioBlkOutHdr {
            type_: req_type,
            _ioprio: 0,
            sector: 0,
        };
        mem.write_obj(hdr, header_addr).expect("plant header");
        // Pre-fill status with sentinel so an unwritten-status
        // case is detectable.
        mem.write_slice(&[0xEEu8], status_addr).unwrap();

        // Use WRITE flag for data so T_IN succeeds for valid
        // sector-aligned lengths within capacity. T_OUT requires
        // device-readable (no WRITE flag); we cover both
        // directions across the random req_type space.
        let data_flags = if req_type == 1 /* T_OUT */ {
            0
        } else {
            VRING_DESC_F_WRITE as u16
        };
        let descs = [
            RawDescriptor::from(SplitDescriptor::new(
                header_addr.0,
                VIRTIO_BLK_OUTHDR_SIZE as u32,
                0,
                1,
            )),
            RawDescriptor::from(SplitDescriptor::new(
                data_addr.0,
                data_len,
                data_flags,
                2,
            )),
            RawDescriptor::from(SplitDescriptor::new(
                status_addr.0,
                1,
                VRING_DESC_F_WRITE as u16,
                0,
            )),
        ];
        mock.build_desc_chain(&descs).expect("build chain");
        dev.mmio_write(
            VIRTIO_MMIO_QUEUE_NOTIFY as u64,
            &(REQ_QUEUE as u32).to_le_bytes(),
        );

        let mut s = [0u8; 1];
        mem.read_slice(&mut s, status_addr).unwrap();
        prop_assert!(
            s[0] == VIRTIO_BLK_S_OK as u8
                || s[0] == VIRTIO_BLK_S_IOERR as u8
                || s[0] == VIRTIO_BLK_S_UNSUPP as u8,
            "status byte {:#x} is not a valid virtio-blk status",
            s[0],
        );

        let used_idx = read_used_idx(&mem, &mock);
        prop_assert_eq!(
            used_idx,
            1,
            "fuzzed data_len chain must produce exactly one \
             used-ring entry; got {}",
            used_idx,
        );
    }

    /// Random `flags` on the data descriptor — fuzz the
    /// direction-violation gate and the INDIRECT path. The
    /// device must reject INDIRECT chains gracefully (the
    /// `virtio-queue` parser switches to indirect-table mode
    /// pointed at `addr`, which for this test is unmapped, so
    /// `read_obj` fails and the iterator yields no descs →
    /// chain dropped with io_errors). Direction-mismatch
    /// flags are caught by the production direction gate.
    ///
    /// All paths must produce a defined status byte (S_OK,
    /// S_IOERR, or S_UNSUPP) OR a chain drop (used.idx
    /// unchanged + io_errors bumped). The combined invariant:
    /// progress >= 1.
    #[test]
    fn random_flags_either_succeeds_or_ioerrs(
        data_flags in 0u16..16,
    ) {
        let (mut dev, mem) = build_fuzz_fixture();
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 256);
        dev.set_mem(mem.clone());
        wire_fuzz_device(&mut dev, &mock);

        let header_addr = GuestAddress(0x4000);
        let data_addr = GuestAddress(0x5000);
        let status_addr = GuestAddress(0x6000);

        // T_IN header, sector 0, valid 512-byte data length.
        // The variable is the data segment's `flags`.
        let hdr = VirtioBlkOutHdr {
            type_: super::VIRTIO_BLK_T_IN,
            _ioprio: 0,
            sector: 0,
        };
        mem.write_obj(hdr, header_addr).expect("plant header");
        mem.write_slice(&[0xEEu8], status_addr).unwrap();

        let descs = [
            RawDescriptor::from(SplitDescriptor::new(
                header_addr.0,
                VIRTIO_BLK_OUTHDR_SIZE as u32,
                VRING_DESC_F_NEXT as u16,
                1,
            )),
            RawDescriptor::from(SplitDescriptor::new(
                data_addr.0,
                512,
                data_flags | VRING_DESC_F_NEXT as u16,
                2,
            )),
            RawDescriptor::from(SplitDescriptor::new(
                status_addr.0,
                1,
                VRING_DESC_F_WRITE as u16,
                0,
            )),
        ];
        // add_desc_chains preserves flags verbatim so we can
        // observe the device's response to arbitrary flag bits
        // on the data descriptor.
        mock.add_desc_chains(&descs, 0).expect("plant descriptors");

        let before_used = read_used_idx(&mem, &mock);
        let before = snapshot_counters(&dev);
        dev.mmio_write(
            VIRTIO_MMIO_QUEUE_NOTIFY as u64,
            &(REQ_QUEUE as u32).to_le_bytes(),
        );
        let after_used = read_used_idx(&mem, &mock);
        let after = snapshot_counters(&dev);

        let used_delta = (after_used - before_used) as u64;
        let counter_delta = (after.reads - before.reads)
            + (after.writes - before.writes)
            + (after.flushes - before.flushes)
            + (after.throttled - before.throttled)
            + (after.io_errors - before.io_errors);
        prop_assert!(
            used_delta + counter_delta >= 1,
            "no progress with data_flags={:#x}: \
             used_delta={} counter_delta={}",
            data_flags,
            used_delta,
            counter_delta,
        );
    }

    /// Throttle-stall property: a well-formed chain dispatched
    /// against a drained iops=1 throttle MUST stall (or be
    /// rejected by a pre-throttle gate) without panicking,
    /// without livelocking, and without publishing a status
    /// byte — and the queue cursor MUST be rewound so the
    /// chain re-pops on the next refill.
    ///
    /// This complements the hand-curated stall tests
    /// (`enable_notification_err_on_stall_path_breaks_outer_cleanly`,
    /// the `apply_ops`-style throttle tests) by sweeping the
    /// chain-shape parameter space — varying request type
    /// (T_IN/T_OUT/T_FLUSH), sector value, segment count, and
    /// per-segment length — to surface invariant violations
    /// only specific shape combinations would expose.
    ///
    /// # u16 wrap coverage
    ///
    /// `next_avail` rollback uses
    /// `set_next_avail(prev.wrapping_sub(1))` — the wrap
    /// arithmetic is covered by the dedicated unit test
    /// `next_avail_zero_rollback_wraps_to_u16_max` in the
    /// parent tests module. That test pre-positions the
    /// cursor at 0 and asserts the post-rollback value is
    /// u16::MAX, which is the wrap edge case
    /// `set_next_avail(prev.wrapping_sub(1))` is designed to
    /// handle. This proptest cannot stress the wrap edge
    /// because `MockSplitQueue::build_desc_chain` only works
    /// against a fresh avail ring (the mock plants at
    /// avail.ring[0] and bumps avail.idx from 0 to 1, which
    /// would panic with overflow at avail.idx=u16::MAX). The
    /// proptest therefore runs at next_avail=0 → 1 → 0 and
    /// focuses on chain-shape variation.
    ///
    /// # Pinned invariants per case
    ///
    /// 1. `dev.mmio_write(QUEUE_NOTIFY, ...)` returns within
    ///    the proptest wall-clock budget (no infinite loop).
    ///    A panic propagates up and shrinks to the minimal
    ///    offending chain; a hang surfaces as the proptest
    ///    runner's per-case timeout.
    /// 2. `throttled_count` advanced by 1 if the chain reached
    ///    the throttle gate, OR `io_errors` advanced by 1 if a
    ///    pre-throttle gate (zero-data, sub-sector, direction)
    ///    rejected it first. Either outcome is correct under
    ///    the hostile-shape framing — what matters is that
    ///    SOME counter moved, no silent stall.
    /// 3. `reads_completed`, `writes_completed`,
    ///    `flushes_completed` UNCHANGED (the bucket is drained
    ///    so no chain successfully consumed tokens).
    /// 4. If throttled_count fired: status sentinel (0xEE)
    ///    UNCHANGED at status_addr (no publish_completion
    ///    ran); used.idx UNCHANGED (no add_used); next_avail
    ///    rewound to the pre-notify cursor value (the
    ///    wrap-aware rollback).
    /// 5. If io_errors fired (pre-throttle gate): status byte
    ///    is one of {0xEE sentinel if status_addr drop path,
    ///    S_IOERR otherwise}, and used.idx advanced by AT MOST
    ///    1.
    #[test]
    fn throttle_stall_under_random_chain_shapes_holds_invariants(
        chain in well_formed_chain_strategy(),
    ) {
        // u16-wrap stressor: the dedicated unit test
        // `next_avail_zero_rollback_wraps_to_u16_max`
        // (in the parent tests module) pins the modular
        // arithmetic edge directly via set_next_avail.
        // This proptest cannot stress the wrap edge because
        // `MockSplitQueue::build_desc_chain` only operates
        // on a fresh avail ring (the mock's chain-planting
        // helper writes to avail.ring[0] and bumps avail.idx
        // from 0 to 1, which would panic with overflow at
        // avail.idx=u16::MAX). The proptest therefore runs
        // at next_avail=0 → 1 → 0 (no wrap) and focuses on
        // chain-shape variation; the wrap arithmetic is
        // covered by the dedicated unit test alongside.
        let (mut dev, mem) = build_throttled_fuzz_fixture();
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 256);
        dev.set_mem(mem.clone());
        wire_fuzz_device(&mut dev, &mock);

        let status_addr = plant_well_formed_chain(&mem, &mock, &chain);
        // After plant: avail.idx is 1, avail.ring[0] holds
        // the chain head. The device's pop_descriptor_chain
        // reads slot 0, advances next_avail from 0 to 1,
        // hits the throttle gate (iops bucket drained at
        // construction), and rolls back via
        // `set_next_avail(prev.wrapping_sub(1))` — landing
        // back at 0.

        let before = snapshot_counters(&dev);

        // Fire QUEUE_NOTIFY. process_requests under a drained
        // throttle either stalls (most cases) or rejects
        // pre-throttle (cases that violate a pre-throttle
        // gate, e.g. T_OUT with zero data segments because
        // chain.n_data_segments was clamped).
        //
        // A panic propagates and proptest shrinks; a hang
        // surfaces as the per-case timeout. Counter
        // monotonicity and rollback-correctness are pinned
        // by the assertions below.
        dev.mmio_write(
            VIRTIO_MMIO_QUEUE_NOTIFY as u64,
            &(REQ_QUEUE as u32).to_le_bytes(),
        );

        let after = snapshot_counters(&dev);

        // Counter monotonicity (parity with the existing
        // fuzz tests).
        prop_assert!(after.reads >= before.reads);
        prop_assert!(after.writes >= before.writes);
        prop_assert!(after.flushes >= before.flushes);
        prop_assert!(after.bytes_read >= before.bytes_read);
        prop_assert!(after.bytes_written >= before.bytes_written);
        prop_assert!(after.throttled >= before.throttled);
        prop_assert!(after.io_errors >= before.io_errors);

        let throttled_delta = after.throttled - before.throttled;
        let io_errors_delta = after.io_errors - before.io_errors;

        // Forward progress: SOME counter moved. A drained
        // throttle MUST cause the chain to either stall
        // (throttled++) or reject pre-throttle (io_errors++);
        // a silent no-op would mean the chain was popped and
        // forgotten without observability.
        prop_assert!(
            throttled_delta + io_errors_delta >= 1,
            "drained throttle must produce a stall or pre-throttle reject; \
             throttled_delta={throttled_delta} io_errors_delta={io_errors_delta} \
             chain={chain:?}",
        );

        // No completion: every drained-throttle case must
        // leave the success counters at zero, regardless of
        // whether the rejection was throttle or pre-throttle.
        prop_assert_eq!(
            after.reads - before.reads, 0,
            "drained throttle must not produce a successful read"
        );
        prop_assert_eq!(
            after.writes - before.writes, 0,
            "drained throttle must not produce a successful write"
        );
        prop_assert_eq!(
            after.flushes - before.flushes, 0,
            "drained throttle must not produce a successful flush"
        );

        // Stall-only invariants (apply when the throttle gate
        // fired, not when a pre-throttle gate fired).
        if throttled_delta == 1 {
            // Status sentinel survives — no publish_completion
            // ran on a throttle stall.
            let mut s = [0u8; 1];
            mem.read_slice(&mut s, status_addr)
                .expect("read status sentinel");
            prop_assert_eq!(
                s[0], 0xEE,
                "stalled chain must not write status byte; chain={:?}",
                chain,
            );

            // Queue cursor rewound: post-stall next_avail
            // matches the pre-notify value (0).
            // pop_descriptor_chain advanced it from 0 to 1;
            // the stall rolled it back via wrapping_sub(1)
            // to 0.
            let post_stall_next_avail = dev.worker.queues[REQ_QUEUE].next_avail();
            prop_assert_eq!(
                post_stall_next_avail, 0u16,
                "post-stall next_avail must rewind to 0; got {}",
                post_stall_next_avail,
            );

            // currently_throttled_gauge incremented (false→true).
            let gauge = dev.counters().currently_throttled_gauge.load(Ordering::Relaxed);
            prop_assert_eq!(
                gauge, 1,
                "stalled-chain gauge must show 1 (false→true transition)",
            );
        }

        // Pre-throttle reject invariants (when io_errors
        // fired but throttled didn't).
        if io_errors_delta >= 1 && throttled_delta == 0 {
            // Pre-throttle rejection writes status byte
            // S_IOERR via publish_completion (when
            // status_addr is mapped, which it always is in
            // this fixture). used.idx advances by 1.
            let mut s = [0u8; 1];
            mem.read_slice(&mut s, status_addr)
                .expect("read status byte");
            prop_assert!(
                s[0] == VIRTIO_BLK_S_IOERR as u8 || s[0] == VIRTIO_BLK_S_OK as u8
                    || s[0] == VIRTIO_BLK_S_UNSUPP as u8,
                "pre-throttle reject must write a defined virtio-blk status; \
                 got status={:#x} chain={:?}",
                s[0], chain,
            );
        }
    }
}
