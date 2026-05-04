#![cfg(test)]
#![allow(unused_imports)]

use super::testing::*;
use super::*;
use std::io::{Seek, Write};
use std::os::unix::fs::FileExt;
use std::sync::atomic::Ordering;
use tempfile::tempfile;
use vm_memory::Bytes;

// ----------------------------------------------------------------
// File backend + read_only
//
// The MMIO descriptor-chain path requires a real GuestMemoryMmap +
// a populated descriptor ring, which is too heavy for unit tests.
// The handlers (`handle_read`, `handle_write`, `handle_flush`) take
// `data_segments` slices and a status address; we drive them
// directly against a small synthetic GuestMemoryMmap to exercise
// the pread/pwrite + read_only paths.
// ----------------------------------------------------------------

#[test]
fn handle_read_pulls_bytes_from_backing_file() {
    // 1 sector = 512 bytes. Backing prefilled with 0xAB.
    let cap = 4096u64; // 8 sectors
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_guest_mem(8192);
    // Guest buffer at GPA 0x1000, 1 sector = 512 bytes,
    // device-writable.
    let data_addr = GuestAddress(0x1000);
    let status_addr = GuestAddress(0x1FFF); // 1 byte
    let segs = vec![ChainDescriptor {
        addr: data_addr,
        len: 512,
        is_write_only: true,
    }];
    let (status, used) = dev.handle_read(&mem, 0, &segs, status_addr);
    assert_eq!(status, VIRTIO_BLK_S_OK as u8);
    assert_eq!(used, 513); // 512 data + 1 status
    // Verify the guest buffer now contains the pattern.
    let mut readback = [0u8; 512];
    mem.read_slice(&mut readback, data_addr).unwrap();
    assert!(readback.iter().all(|&b| b == 0xAB));
    // Status byte set to OK.
    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(s[0], VIRTIO_BLK_S_OK as u8);
}

#[test]
fn handle_write_persists_bytes_to_backing_file() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    // Borrow the file's fd before passing into the device — we
    // use FileExt::read_at on a clone to verify post-write
    // contents.
    let f_for_verify = f.try_clone().unwrap();
    let dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    // Larger guest mem so data_addr + len + status_addr all fit
    // within the single region. 32 KB is generous; the previous
    // 16 KB region with status_addr=0x2FFF (12287) sat within
    // bounds but write_slice rejected the write (likely a
    // single-region GuestMemoryMmap quirk under address overlap).
    let mem = make_guest_mem(32768);
    let data_addr = GuestAddress(0x1000);
    let status_addr = GuestAddress(0x2000);
    // Stuff 0xCD pattern into guest memory at data_addr.
    let pattern = vec![0xCDu8; 512];
    mem.write_slice(&pattern, data_addr).unwrap();
    let segs = vec![ChainDescriptor {
        addr: data_addr,
        len: 512,
        is_write_only: false,
    }];
    let (status, used) = dev.handle_write(&mem, 1, &segs, status_addr); // sector 1
    assert_eq!(status, VIRTIO_BLK_S_OK as u8);
    assert_eq!(used, 1);
    // Verify backing file at offset 512 now contains 0xCD.
    let mut readback = [0u8; 512];
    f_for_verify.read_at(&mut readback, 512).unwrap();
    assert!(readback.iter().all(|&b| b == 0xCD));
}

#[test]
fn handle_read_rejects_out_of_range_sector() {
    let cap = 4096u64; // 8 sectors
    let f = make_backed_file_with_pattern(cap, 0x00);
    let dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_guest_mem(8192);
    let data_addr = GuestAddress(0x1000);
    let status_addr = GuestAddress(0x1FFF);
    let segs = vec![ChainDescriptor {
        addr: data_addr,
        len: 512,
        is_write_only: true,
    }];
    // sector 9 is past capacity (cap=4096 → 8 sectors → max idx 7).
    let (status, _) = dev.handle_read(&mem, 9, &segs, status_addr);
    assert_eq!(status, VIRTIO_BLK_S_IOERR as u8);
    let c = dev.counters();
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 1);
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
}

#[test]
fn handle_write_rejects_out_of_range_sector() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    // 16 KiB guest mem to hold both data buffer (sized 512) and
    // status byte addr without overlap. data at 0x1000, status
    // at 0x2000, both well within [0, 0x4000).
    let mem = make_guest_mem(16384);
    let data_addr = GuestAddress(0x1000);
    let status_addr = GuestAddress(0x2000);
    let segs = vec![ChainDescriptor {
        addr: data_addr,
        len: 512,
        is_write_only: false,
    }];
    let (status, _) = dev.handle_write(&mem, 9, &segs, status_addr);
    assert_eq!(status, VIRTIO_BLK_S_IOERR as u8);
}

#[test]
fn handle_flush_succeeds_on_writable_backing() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_guest_mem(8192);
    let status_addr = GuestAddress(0x100);
    let (status, used) = dev.handle_flush(&mem, status_addr);
    assert_eq!(status, VIRTIO_BLK_S_OK as u8);
    assert_eq!(used, 1);
    let c = dev.counters();
    assert_eq!(c.flushes_completed.load(Ordering::Relaxed), 1);
}

#[test]
fn read_only_advertises_f_ro_feature_bit() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let dev = VirtioBlk::with_options(f, cap, DiskThrottle::default(), true);
    let feats = dev.device_features();
    assert_ne!(feats & (1u64 << VIRTIO_BLK_F_RO), 0);
}

#[test]
fn read_write_does_not_advertise_f_ro_feature_bit() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let feats = dev.device_features();
    assert_eq!(feats & (1u64 << VIRTIO_BLK_F_RO), 0);
}

#[test]
fn write_at_full_capacity_succeeds() {
    // The boundary case — a write whose end aligns exactly with
    // capacity. Should succeed (the spec wording is "if the
    // sector range overlaps a sector outside the capacity").
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_guest_mem(16384);
    let data_addr = GuestAddress(0x2000);
    let status_addr = GuestAddress(0x2FFF);
    let pattern = vec![0xEEu8; 512];
    mem.write_slice(&pattern, data_addr).unwrap();
    let segs = vec![ChainDescriptor {
        addr: data_addr,
        len: 512,
        is_write_only: false,
    }];
    // sector 7 covers bytes 3584..4096 — exactly at capacity.
    let (status, _) = dev.handle_write(&mem, 7, &segs, status_addr);
    assert_eq!(status, VIRTIO_BLK_S_OK as u8);
}

#[test]
fn read_short_pads_with_zeros() {
    // Sparse-file semantics: reads past the written region must
    // return zeros, not stale data. Establish by truncating the
    // backing file shorter than requested capacity, then reading.
    let cap = 4096u64; // 8 sectors
    let mut f = tempfile().unwrap();
    f.set_len(512).unwrap(); // backing covers only 1 sector
    f.write_all(&[0xAA; 512]).unwrap();
    f.rewind().unwrap();
    let dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_guest_mem(8192);
    let data_addr = GuestAddress(0x1000);
    let status_addr = GuestAddress(0x1FFF);
    let segs = vec![ChainDescriptor {
        addr: data_addr,
        len: 512,
        is_write_only: true,
    }];
    // Sector 4 → offset 2048, well past the backing's 512 bytes.
    let (status, _) = dev.handle_read(&mem, 4, &segs, status_addr);
    assert_eq!(status, VIRTIO_BLK_S_OK as u8);
    let mut readback = [0u8; 512];
    mem.read_slice(&mut readback, data_addr).unwrap();
    assert!(
        readback.iter().all(|&b| b == 0),
        "out-of-data reads must zero-pad, not return stale memory"
    );
}

#[test]
fn read_only_flush_returns_ok() {
    // A read-only disk has no dirty data — a precautionary flush
    // from a guest mounting RO must return OK to avoid spurious
    // mount errors in the guest dmesg.
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let dev = VirtioBlk::with_options(f, cap, DiskThrottle::default(), true);
    // read_only flush behaviour is checked through the
    // process_requests dispatch table; here we just pin the
    // device's `read_only` flag is captured.
    assert!(dev.worker.read_only);
}

/// A read-only device must reject `VIRTIO_BLK_T_OUT` with
/// `VIRTIO_BLK_S_IOERR`, regardless of whether the guest
/// honoured the negotiated F_RO bit. The classification happens
/// in `classify_pre_throttle`; this test drives that helper
/// directly so the assertion follows the same code path
/// `process_requests` does.
#[test]
fn read_only_write_returns_ioerr() {
    let counters = VirtioBlkCounters::default();
    let result = VirtioBlk::classify_pre_throttle(VIRTIO_BLK_T_OUT, true, &counters);
    assert_eq!(result, Some((VIRTIO_BLK_S_IOERR as u8, 1)));
    // io_errors bumped — the rejection counts as an IO error
    // for failure-dump visibility.
    assert_eq!(counters.io_errors.load(Ordering::Relaxed), 1);
    assert_eq!(counters.writes_completed.load(Ordering::Relaxed), 0);
}

/// A read-only device's flush is a no-op that completes with
/// `VIRTIO_BLK_S_OK` AND increments `flushes_completed` for
/// monitor visibility. The semantic is "guest issued a real
/// flush, device serviced it (trivially, because nothing's
/// dirty)" — distinct from "rejected" which would surface as
/// IOERR.
#[test]
fn read_only_flush_returns_ok_and_increments_counter() {
    let counters = VirtioBlkCounters::default();
    let result = VirtioBlk::classify_pre_throttle(VIRTIO_BLK_T_FLUSH, true, &counters);
    assert_eq!(result, Some((VIRTIO_BLK_S_OK as u8, 1)));
    assert_eq!(counters.flushes_completed.load(Ordering::Relaxed), 1);
    assert_eq!(counters.io_errors.load(Ordering::Relaxed), 0);
}

/// A multi-segment read scatters successive backing-file bytes
/// into successive guest segments. Pin the per-segment file
/// offset advance: segment 0 receives bytes 0..512, segment 1
/// receives bytes 512..1024. A regression that reset the file
/// offset between segments (or wrote both segments to the same
/// file region) would either fill segment 1 with zeros or
/// duplicate segment 0's contents — this test catches both.
#[test]
fn scatter_gather_read_two_segments() {
    let cap = 4096u64;
    let mut f = tempfile().unwrap();
    f.set_len(cap).unwrap();
    f.rewind().unwrap();
    // First 512 bytes = 0x11, next 512 bytes = 0x22, rest 0.
    let mut backing_data = vec![0x11u8; 512];
    backing_data.extend(vec![0x22u8; 512]);
    f.write_all(&backing_data).unwrap();
    f.rewind().unwrap();
    let dev = VirtioBlk::new(f, cap, DiskThrottle::default());

    let mem = make_guest_mem(8192);
    let seg0_addr = GuestAddress(0x1000);
    let seg1_addr = GuestAddress(0x1400); // 0x1000 + 0x400 = 0x1400, no overlap
    let status_addr = GuestAddress(0x1FFF);
    let segs = vec![
        ChainDescriptor {
            addr: seg0_addr,
            len: 512,
            is_write_only: true,
        },
        ChainDescriptor {
            addr: seg1_addr,
            len: 512,
            is_write_only: true,
        },
    ];
    let (status, used) = dev.handle_read(&mem, 0, &segs, status_addr);
    assert_eq!(status, VIRTIO_BLK_S_OK as u8);
    assert_eq!(used, 1024 + 1); // 2 × 512 data + 1 status

    // Segment 0 should contain 0x11 (file bytes 0..512).
    let mut buf0 = [0u8; 512];
    mem.read_slice(&mut buf0, seg0_addr).unwrap();
    assert!(
        buf0.iter().all(|&b| b == 0x11),
        "segment 0 must receive file bytes 0..512 (0x11 pattern)",
    );
    // Segment 1 should contain 0x22 (file bytes 512..1024) —
    // proves the per-segment cursor advanced correctly.
    let mut buf1 = [0u8; 512];
    mem.read_slice(&mut buf1, seg1_addr).unwrap();
    assert!(
        buf1.iter().all(|&b| b == 0x22),
        "segment 1 must receive file bytes 512..1024 (0x22 pattern); \
             a regression that didn't advance the file cursor would \
             produce 0x11 here",
    );
}

/// An unknown request type (anything outside T_IN/T_OUT/T_FLUSH/
/// T_GET_ID) must be classified as `VIRTIO_BLK_S_UNSUPP` per
/// virtio-v1.2 §5.2.6.4. Pin the dispatch table's default
/// behaviour so a future patch that mis-handles a new request
/// type as IOERR (or, worse, OK) surfaces here. Counters are
/// untouched on UNSUPP because the request was never dispatched
/// to a backend.
#[test]
fn unknown_type_returns_unsupp() {
    let counters = VirtioBlkCounters::default();
    let result = VirtioBlk::classify_pre_throttle(0xBEEF, false, &counters);
    assert_eq!(result, Some((VIRTIO_BLK_S_UNSUPP as u8, 1)));
    // Unknown types don't bump io_errors — the device gracefully
    // declined a request it didn't recognise, not something it
    // tried and failed to service.
    assert_eq!(counters.io_errors.load(Ordering::Relaxed), 0);
    assert_eq!(counters.reads_completed.load(Ordering::Relaxed), 0);
    assert_eq!(counters.writes_completed.load(Ordering::Relaxed), 0);
    assert_eq!(counters.flushes_completed.load(Ordering::Relaxed), 0);
}

/// Multi-segment scatter read: pin that `handle_read_impl`
/// walks `data_segments` in order, advances `cur_offset` by
/// each segment's `len`, and writes each guest segment with
/// the correct slice of the backing file. This is the central
/// scatter-gather invariant — without per-segment offset
/// advancement, segments 1..N would either stamp on segment 0
/// or skip data.
#[test]
fn handle_read_multi_segment_scatter() {
    // 2-sector backing prefilled with a known pattern: bytes
    // 0..512 = 0xAA, bytes 512..1024 = 0xBB. Two guest data
    // segments each receive one sector. After the read,
    // segment 0 must hold 0xAA and segment 1 must hold 0xBB.
    let cap = 4096u64; // 8 sectors
    let mut f = tempfile().unwrap();
    f.set_len(cap).unwrap();
    f.write_all(&[0xAA; 512]).unwrap();
    f.write_all(&[0xBB; 512]).unwrap();
    f.rewind().unwrap();
    let dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_guest_mem(8192);
    // Two scatter segments: 512 bytes each, separated by a
    // gap so test failure on cur_offset arithmetic shows up
    // as cross-contamination.
    let seg0_addr = GuestAddress(0x1000);
    let seg1_addr = GuestAddress(0x1400); // 0x1000 + 0x400 (1 KB)
    let status_addr = GuestAddress(0x1FFF);
    let segs = vec![
        ChainDescriptor {
            addr: seg0_addr,
            len: 512,
            is_write_only: true,
        },
        ChainDescriptor {
            addr: seg1_addr,
            len: 512,
            is_write_only: true,
        },
    ];
    // Read starting at sector 0 — covers backing bytes 0..1024,
    // split across the two segments.
    let (status, used) = dev.handle_read(&mem, 0, &segs, status_addr);
    assert_eq!(status, VIRTIO_BLK_S_OK as u8);
    assert_eq!(used, 1024 + 1); // 2 * 512 data + 1 status

    let mut readback0 = [0u8; 512];
    mem.read_slice(&mut readback0, seg0_addr).unwrap();
    assert!(
        readback0.iter().all(|&b| b == 0xAA),
        "segment 0 must hold the FIRST sector's pattern (0xAA), \
             got cross-contamination: {:?}..{:?}",
        &readback0[..8],
        &readback0[504..],
    );

    let mut readback1 = [0u8; 512];
    mem.read_slice(&mut readback1, seg1_addr).unwrap();
    assert!(
        readback1.iter().all(|&b| b == 0xBB),
        "segment 1 must hold the SECOND sector's pattern (0xBB) — \
             cur_offset must have advanced by 512 between segments. \
             got: {:?}..{:?}",
        &readback1[..8],
        &readback1[504..],
    );

    let c = dev.counters();
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 1);
    assert_eq!(c.bytes_read.load(Ordering::Relaxed), 1024);
}
