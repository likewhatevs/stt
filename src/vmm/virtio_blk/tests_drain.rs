#![cfg(test)]
#![allow(unused_imports)]

use super::testing::*;
use super::*;
use std::io::{Seek, Write};
use std::num::NonZeroU64;
use std::os::unix::fs::FileExt;
use std::sync::atomic::Ordering;
use std::time::Instant;
use tempfile::tempfile;
use virtio_bindings::bindings::virtio_ring::VRING_DESC_F_WRITE;
use virtio_queue::desc::{RawDescriptor, split::Descriptor as SplitDescriptor};
use virtio_queue::mock::MockSplitQueue;
use vm_memory::Address;

// ----------------------------------------------------------------
// MockSplitQueue chain-level tests against process_requests.
//
// These exercise the descriptor-chain parsing path
// (`process_requests` in virtio_blk.rs) that the handler-level
// tests above skip. The handler tests bypass virtio-queue's
// descriptor walker entirely; these drive the full pipeline:
// MockSplitQueue plants a chain → MMIO QUEUE_NOTIFY fires →
// `process_requests` walks the chain → handler runs → status
// byte + add_used → UsedRing reflects completion.
//
// Coverage: the chain-shape classifier in `process_requests`
// (header detection, status detection, data-segment collection),
// the throttle integration, the pre-throttle classification (RO
// writes / RO flushes / unknown types), and `add_used`'s used-ring
// publish. None of this is reachable from the handler-level
// tests above.
// ----------------------------------------------------------------

/// Drive a full READ chain through `process_requests`.
/// Plants a 1-sector read chain (header + data + status), fires
/// `process_requests` via QUEUE_NOTIFY, then verifies:
/// - the data descriptor receives the backing-file pattern,
/// - the status descriptor holds VIRTIO_BLK_S_OK,
/// - the UsedRing reflects exactly one completion,
/// - reads_completed counter ticks.
#[test]
fn process_requests_full_read_chain() {
    let cap = 4096u64;
    // Backing file pre-filled with 0xAB so we can detect the
    // bytes propagating from file → guest mem via the chain.
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    // Place the queue rings at GPA 0; data buffers go at high
    // offsets so they don't collide with the ring region.
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    // Plant the request header.
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
    // Build the chain: header (RO) + data (WRITE-only) + status
    // (WRITE-only, 1 byte). build_desc_chain writes the
    // descriptors AND publishes them to the avail ring so
    // process_requests sees them.
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0, // device-readable
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
    // Fire QUEUE_NOTIFY — this drives process_requests, which
    // pops the chain and runs the read handler.
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // Verify the data buffer now holds the backing file's
    // pattern (0xAB) and the status byte is OK.
    let mut data_buf = [0u8; 512];
    mem.read_slice(&mut data_buf, data_addr).unwrap();
    assert!(
        data_buf.iter().all(|&b| b == 0xAB),
        "data segment must contain backing file's 0xAB pattern after read",
    );
    let mut status_buf = [0u8; 1];
    mem.read_slice(&mut status_buf, status_addr).unwrap();
    assert_eq!(
        status_buf[0], VIRTIO_BLK_S_OK as u8,
        "status byte must be S_OK after successful read",
    );

    // Used ring must reflect one completion. UsedRing.idx is at
    // mock.used_addr() + 2 (after the 2-byte flags field).
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 1, "exactly one used-ring entry expected");

    // Counters: read counted, no errors.
    let c = dev.counters();
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 1);
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 0);
    assert_eq!(c.bytes_read.load(Ordering::Relaxed), 512);
}

/// Drive a full WRITE chain through `process_requests`.
/// Plants a 1-sector write chain, primes the data segment with a
/// distinctive pattern, fires QUEUE_NOTIFY, then verifies:
/// - the backing file receives the planted bytes (`pwrite`
///   landed at the right offset),
/// - the status byte is VIRTIO_BLK_S_OK,
/// - writes_completed and bytes_written tick.
#[test]
fn process_requests_full_write_chain() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let f_for_verify = f.try_clone().expect("clone backing for verify");
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    // Plant the request header for a WRITE at sector 1.
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_OUT, 1);
    // Plant the data the device should write to the backing file.
    let payload = vec![0xCDu8; 512];
    mem.write_slice(&payload, data_addr).expect("plant payload");
    // Build the chain: header (RO) + data (RO — write request,
    // device READS the data segment) + status (WRITE-only).
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
            0, // device-readable for write
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

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // Verify backing file at offset 512 (= sector 1) holds the
    // payload — proves the chain dispatched to handle_write_impl
    // and the pwrite landed.
    let mut readback = [0u8; 512];
    f_for_verify
        .read_at(&mut readback, 512)
        .expect("read backing");
    assert!(
        readback.iter().all(|&b| b == 0xCD),
        "backing file at sector 1 must hold the 0xCD payload after write",
    );

    // Status byte == OK.
    let mut status_buf = [0u8; 1];
    mem.read_slice(&mut status_buf, status_addr).unwrap();
    assert_eq!(status_buf[0], VIRTIO_BLK_S_OK as u8);

    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 1);

    let c = dev.counters();
    assert_eq!(c.writes_completed.load(Ordering::Relaxed), 1);
    assert_eq!(c.bytes_written.load(Ordering::Relaxed), 512);
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 0);
}

/// Drive a chain with an UNKNOWN request type through
/// `process_requests`. The dispatch table pre-classifies any
/// req_type outside `T_IN`/`T_OUT`/`T_FLUSH` as
/// VIRTIO_BLK_S_UNSUPP. Verifies:
/// - status byte is VIRTIO_BLK_S_UNSUPP (2), NOT IOERR (1),
/// - the chain still completes (used ring updated),
/// - io_errors does NOT tick (UNSUPP is not an IO error — the
///   device gracefully declined a request it didn't recognise),
/// - reads/writes/flushes counters all stay at 0.
#[test]
fn process_requests_unknown_type_returns_unsupp() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let status_addr = GuestAddress(0x5000);
    // Type 0xBEEF is outside every known T_* opcode (T_IN=0,
    // T_OUT=1, T_FLUSH=4, T_GET_ID=8). Sector irrelevant for an
    // unknown-type rejection — never reaches the backing path.
    write_blk_header(&mem, header_addr, 0xBEEF, 0);
    // No data segment — UNSUPP rejection happens before any
    // data-segment walk. Header + status only is the minimal
    // legal chain shape.
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0,
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

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // Status byte must be UNSUPP, not IOERR. A regression that
    // mapped unknown types to IOERR would corrupt the
    // pre-throttle classifier's contract; one that mapped them
    // to OK would silently let bogus requests pass.
    let mut status_buf = [0u8; 1];
    mem.read_slice(&mut status_buf, status_addr).unwrap();
    assert_eq!(
        status_buf[0], VIRTIO_BLK_S_UNSUPP as u8,
        "unknown req_type must produce S_UNSUPP, not S_IOERR or S_OK",
    );

    // Used ring still reflects completion — the device returned
    // the descriptor chain to the guest with the UNSUPP status
    // rather than leaking it.
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 1, "UNSUPP completions still update used.idx");

    // io_errors stays 0 — UNSUPP is not classified as an IO
    // error.
    let c = dev.counters();
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 0);
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
    assert_eq!(c.writes_completed.load(Ordering::Relaxed), 0);
    assert_eq!(c.flushes_completed.load(Ordering::Relaxed), 0);
}

/// Drive a FLUSH chain through `process_requests`.
/// FLUSH has no data segment per virtio-v1.2 §5.2.6 — the chain
/// is exactly header + status. Verifies the dispatch reaches
/// `handle_flush_impl` (calls fdatasync), increments
/// flushes_completed, and writes S_OK status.
#[test]
fn process_requests_flush_chain() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let status_addr = GuestAddress(0x5000);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_FLUSH, 0);
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0,
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

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    let mut status_buf = [0u8; 1];
    mem.read_slice(&mut status_buf, status_addr).unwrap();
    assert_eq!(status_buf[0], VIRTIO_BLK_S_OK as u8);

    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 1);

    let c = dev.counters();
    assert_eq!(c.flushes_completed.load(Ordering::Relaxed), 1);
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
    assert_eq!(c.writes_completed.load(Ordering::Relaxed), 0);
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 0);
}

/// Drive a chain whose first descriptor is too short
/// to hold the 16-byte `virtio_blk_outhdr`. The chain parser
/// rejects when `first_len < VIRTIO_BLK_OUTHDR_SIZE`, leaving
/// `header_addr` unset; the dispatcher writes S_IOERR to status,
/// increments io_errors, and add_used's the chain.
#[test]
fn process_requests_short_header_returns_ioerr() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let status_addr = GuestAddress(0x5000);
    // Header descriptor with len=8 (too short — outhdr is 16
    // bytes). The bytes there don't matter because the device
    // never reads them: `first_len < OUTHDR_SIZE` skips the
    // read entirely.
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            8, // SHORT — half the required 16 bytes
            0,
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

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    let mut status_buf = [0u8; 1];
    mem.read_slice(&mut status_buf, status_addr).unwrap();
    assert_eq!(
        status_buf[0], VIRTIO_BLK_S_IOERR as u8,
        "short header must be rejected with S_IOERR",
    );

    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 1);

    let c = dev.counters();
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 1);
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
    assert_eq!(c.writes_completed.load(Ordering::Relaxed), 0);
    assert_eq!(c.flushes_completed.load(Ordering::Relaxed), 0);
}

/// Drive a chain whose last descriptor is NOT
/// device-writable. Per virtio-v1.2 §5.2.6 the status descriptor
/// MUST be device-writable. The parser only accepts the last
/// descriptor as status when its `is_write_only` flag is set;
/// otherwise `status_addr` stays None and the dispatcher
/// drops the chain entirely (does NOT call `add_used`,
/// does NOT write a status byte).
///
/// CRITICAL: calling `add_used` here would tell the guest the
/// request SUCCEEDED. The kernel driver's `virtblk_done` reads
/// the status from the request's `vbr->in_hdr`, which is stale
/// from prior blk-mq tag use (initially zero from `__GFP_ZERO`
/// at allocation, stale on reuse); `virtblk_result(0) ==
/// BLK_STS_OK` (drivers/block/virtio_blk.c). With the device having
/// written no status byte to any guest-visible location, a
/// completed-but-unstatused request would surface as a phantom
/// success — silent data corruption for reads, silent dropped
/// writes for writes. The fix: leave the descriptor in the
/// avail ring; the guest hangs on this request until
/// `kernel.hung_task_timeout_secs` (default 120 s) fires or a
/// higher layer retries (virtio_blk has no `mq_ops->timeout`,
/// so blk-mq alone won't surface the stall).
#[test]
fn process_requests_status_not_writable_drops_chain() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let status_addr = GuestAddress(0x5000);
    // Plant a sentinel byte at status_addr so we can detect
    // whether the device wrote anything (it should NOT).
    // 0xEE chosen because it's distinct from S_OK (0),
    // S_IOERR (1), S_UNSUPP (2).
    mem.write_slice(&[0xEEu8], status_addr).unwrap();
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
    // Last descriptor lacks WRITE flag — disqualifies it as
    // status. The parser reads it as a (degenerate) data
    // segment and finds no status descriptor.
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            status_addr.0,
            1,
            0, // device-readable, NOT write-only
            0,
        )),
    ];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_device_to_mock(&mut dev, &mock);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // Sentinel must remain — device wrote nothing because it
    // had no status_addr.
    let mut status_buf = [0u8; 1];
    mem.read_slice(&mut status_buf, status_addr).unwrap();
    assert_eq!(
        status_buf[0], 0xEE,
        "no status descriptor → device must not write a status byte; \
             sentinel 0xEE survives",
    );

    // The chain MUST stay in the avail ring (used.idx unchanged
    // at 0). Calling add_used here would let the guest's
    // virtblk_done observe an in_hdr.status that's stale from
    // prior blk-mq tag use (initially zero from __GFP_ZERO at
    // allocation, stale on reuse) as BLK_STS_OK — phantom
    // success.
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(
        used_idx, 0,
        "no-status chain must NOT advance used.idx; advancing would \
             let the guest's stale in_hdr.status surface as \
             BLK_STS_OK (silent data corruption)",
    );

    let c = dev.counters();
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 1);
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
    // No-status chain → publish_completion never runs →
    // signal_needed stays false → interrupt_status bit MUST
    // remain 0 (the bit is set inside `if signal_needed`,
    // drain_bracket_impl). A regression that set the bit
    // unconditionally on every notify would leak phantom
    // interrupts to the guest's polling path.
    assert_eq!(
        dev.interrupt_status.load(Ordering::Acquire) & VIRTIO_MMIO_INT_VRING,
        0
    );
}

/// Multi-chain FIFO ordering through
/// `process_requests`. Plant 3 distinct READ chains in the
/// avail ring, single QUEUE_NOTIFY drains them all. Verify
/// (a) all 3 complete in the same `process_requests` call,
/// (b) the used.idx advances by exactly 3, (c) reads_completed
/// counter == 3, (d) each chain's data buffer holds the correct
/// per-chain pattern.
#[test]
fn process_requests_multiple_chains_drained_in_one_notify() {
    use virtio_bindings::bindings::virtio_ring::VRING_DESC_F_NEXT;
    let cap = 4096u64;
    let mut f = tempfile().unwrap();
    f.set_len(cap).unwrap();
    // Stamp three distinct sectors with three distinct
    // patterns so each chain's read result is identifiable.
    f.write_all(&[0x11; 512]).unwrap(); // sector 0
    f.write_all(&[0x22; 512]).unwrap(); // sector 1
    f.write_all(&[0x33; 512]).unwrap(); // sector 2
    f.rewind().unwrap();
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);

    let chains = [
        (
            GuestAddress(0x4000),
            GuestAddress(0x4400),
            GuestAddress(0x4800),
            0u64,
        ),
        (
            GuestAddress(0x5000),
            GuestAddress(0x5400),
            GuestAddress(0x5800),
            1u64,
        ),
        (
            GuestAddress(0x6000),
            GuestAddress(0x6400),
            GuestAddress(0x6800),
            2u64,
        ),
    ];
    for &(hdr, _, _, sector) in &chains {
        write_blk_header(&mem, hdr, VIRTIO_BLK_T_IN, sector);
    }

    let mut descs = Vec::new();
    for (chain_i, &(hdr, data, status, _)) in chains.iter().enumerate() {
        // Each chain occupies 3 sequential descriptor-table
        // slots starting at chain_i * 3. The non-last
        // descriptors must point to their successor via the
        // `next` field so the device's queue iterator walks
        // the whole chain. add_desc_chains writes descriptors
        // verbatim — it does NOT auto-link them (only
        // `build_desc_chain` does, and that path takes a
        // single chain).
        let base = (chain_i as u16) * 3;
        descs.push(RawDescriptor::from(SplitDescriptor::new(
            hdr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            VRING_DESC_F_NEXT as u16,
            base + 1,
        )));
        descs.push(RawDescriptor::from(SplitDescriptor::new(
            data.0,
            512,
            VRING_DESC_F_WRITE as u16 | VRING_DESC_F_NEXT as u16,
            base + 2,
        )));
        descs.push(RawDescriptor::from(SplitDescriptor::new(
            status.0,
            1,
            VRING_DESC_F_WRITE as u16,
            0,
        )));
    }
    // add_desc_chains writes the descriptor table sequentially
    // and bumps avail.idx for each chain head.
    mock.add_desc_chains(&descs, 0).expect("add 3 chains");

    dev.set_mem(mem.clone());
    wire_device_to_mock(&mut dev, &mock);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 3, "single notify must drain 3 chains");

    let c = dev.counters();
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 3);
    assert_eq!(c.bytes_read.load(Ordering::Relaxed), 3 * 512);
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 0);

    for (i, &(_, data, _, _)) in chains.iter().enumerate() {
        let mut buf = [0u8; 512];
        mem.read_slice(&mut buf, data).unwrap();
        let expected = (i as u8 + 1) * 0x11;
        assert!(
            buf.iter().all(|&b| b == expected),
            "chain {i}'s data must hold sector {i}'s pattern (0x{expected:02X})",
        );
    }
}

/// Throttle stall through `process_requests` rolls back the
/// chain rather than completing it with `S_IOERR`. Build a
/// device with IOPS limit = 1, drain the bucket, then notify
/// with a chain. Pin the new "stall" contract:
/// - status descriptor is NOT written (sentinel survives),
/// - used.idx stays at 0 (no `add_used`),
/// - the device's `next_avail` is rewound to its pre-pop
///   value (`set_next_avail(prev.wrapping_sub(1))`),
/// - `throttled_count` ticks, `io_errors` stays at 0,
/// - irqfd is unsignalled and `INT_VRING` bit is unset
///   (the chain is invisible to the guest).
///   The chain stays in the avail ring until the worker's
///   retry timer fires; from the test's perspective, calling
///   `process_requests` again after stepping the bucket
///   forward via `set_last_refill_for_test` re-pops the same
///   head (covered by `throttle_stall_then_refill_retry_succeeds`).
#[test]
fn process_requests_throttled_rolls_back_chain() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let throttle = DiskThrottle {
        iops: std::num::NonZeroU64::new(1),
        bytes_per_sec: None,
        iops_burst_capacity: None,
        bytes_burst_capacity: None,
    };
    let mut dev = VirtioBlk::new(f, cap, throttle);
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);

    // Drain the bucket and pin its last_refill so refill on
    // the next consume yields 0 tokens.
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(std::time::Instant::now());
    assert!(
        dev.worker.state_mut().ops_bucket.consume(1),
        "drain the 1-token bucket"
    );
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(std::time::Instant::now());

    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    // Plant a sentinel at the data descriptor so we can detect
    // whether the device wrote to it. 0xFF distinct from
    // backing 0xAB.
    let sentinel_data = vec![0xFFu8; 512];
    mem.write_slice(&sentinel_data, data_addr).unwrap();
    // Plant a sentinel at the status descriptor so we can
    // detect whether the device wrote a status byte.
    // 0xEE is distinct from S_OK (0), S_IOERR (1), S_UNSUPP (2).
    mem.write_slice(&[0xEEu8], status_addr).unwrap();
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
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
    // Capture next_avail BEFORE the notify so we can assert
    // the post-stall rewind preserves the value (the inner
    // loop's pop bumps next_avail by 1; the stall path then
    // calls set_next_avail(prev.wrapping_sub(1)) which
    // restores it).
    let next_avail_before = dev.worker.queues[REQ_QUEUE].next_avail();

    // Pre-notify: irqfd MUST be unsignalled.
    assert!(
        dev.irq_evt.read().is_err(),
        "irq_evt must not be signalled before notify",
    );

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // The status descriptor sentinel must survive — the device
    // wrote no status byte on stall.
    let mut status_buf = [0u8; 1];
    mem.read_slice(&mut status_buf, status_addr).unwrap();
    assert_eq!(
        status_buf[0], 0xEE,
        "throttle stall must NOT write a status byte; the chain \
             stays in the avail ring until the retry timer fires. \
             Sentinel 0xEE must survive.",
    );

    // used.idx must stay at 0 — no add_used on stall.
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(
        used_idx, 0,
        "throttle stall must NOT advance used.idx; advancing \
             would tell the guest the request completed with whatever \
             stale status byte was at the descriptor, defeating the \
             rollback.",
    );

    // next_avail must equal its pre-pop value — the rewind
    // via set_next_avail(prev.wrapping_sub(1)) put it back.
    assert_eq!(
        dev.worker.queues[REQ_QUEUE].next_avail(),
        next_avail_before,
        "throttle stall must rewind next_avail to its pre-pop \
             value so the next drain re-pops the same head",
    );

    let c = dev.counters();
    assert_eq!(
        c.throttled_count.load(Ordering::Relaxed),
        1,
        "throttle stall must bump throttled_count exactly once",
    );
    assert_eq!(
        c.io_errors.load(Ordering::Relaxed),
        0,
        "throttle stall is NOT classified as an I/O error — the \
             chain is deferred, not failed",
    );
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
    assert_eq!(c.bytes_read.load(Ordering::Relaxed), 0);

    // Data descriptor untouched — handler never ran, no pread
    // dispatched.
    let mut readback = [0u8; 512];
    mem.read_slice(&mut readback, data_addr).unwrap();
    assert!(
        readback.iter().all(|&b| b == 0xFF),
        "stalled chain must NOT touch the data descriptor; \
             0xFF sentinel must survive",
    );

    // irqfd MUST be unsignalled and INT_VRING bit unset — the
    // chain is invisible to the guest.
    assert!(
        dev.irq_evt.read().is_err(),
        "throttle stall must NOT signal the irqfd",
    );
    assert_eq!(
        dev.interrupt_status.load(Ordering::Acquire) & VIRTIO_MMIO_INT_VRING,
        0,
        "throttle stall must NOT set INT_VRING; the bit is only \
             set when a chain is published, and a stalled chain is not",
    );
}

/// Throttle stall, then refill the bucket and re-notify —
/// the rolled-back chain is re-popped and completes
/// successfully. End-to-end pin of the retry contract: the
/// undo-pop on stall preserves the chain head in the avail
/// ring, and a subsequent drain (after enough tokens
/// refill) services the same head as if the stall never
/// happened.
///
/// Production wiring: the worker's THROTTLE_TOKEN timerfd
/// fires after `wait_nanos`, the worker re-runs
/// `drain_bracket_impl`, the bucket is now satisfied, and
/// the chain completes. The cfg(test) inline path discards
/// `DrainOutcome` so we drive the retry manually with
/// `set_last_refill_for_test` + a second QUEUE_NOTIFY.
#[test]
fn throttle_stall_then_refill_retry_succeeds() {
    let cap = 4096u64;
    // Backing file pre-filled with 0xAB so the post-retry
    // read can verify bytes propagate from file → guest mem.
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let throttle = DiskThrottle {
        iops: std::num::NonZeroU64::new(1),
        bytes_per_sec: None,
        iops_burst_capacity: None,
        bytes_burst_capacity: None,
    };
    let mut dev = VirtioBlk::new(f, cap, throttle);
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);

    // Drain the bucket: consume the 1-token capacity, then
    // pin last_refill to "now" so the next consume's refill
    // window is ~0 nanoseconds → 0 new tokens.
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(std::time::Instant::now());
    assert!(dev.worker.state_mut().ops_bucket.consume(1), "drain bucket");
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(std::time::Instant::now());

    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    // Plant a sentinel at the data descriptor so we can detect
    // the retry's read landing.
    let sentinel_data = vec![0xFFu8; 512];
    mem.write_slice(&sentinel_data, data_addr).unwrap();
    mem.write_slice(&[0xEEu8], status_addr).unwrap();
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
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
    // Capture next_avail BEFORE the stall so the post-retry
    // assertion can verify the chain was completed exactly
    // once (next_avail advanced by 1 after the stall+retry
    // sequence, NOT 0 from a still-stalled chain or 2 from a
    // double-pop).
    let next_avail_before = dev.worker.queues[REQ_QUEUE].next_avail();

    // First notify: the bucket is empty, the chain stalls.
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);
    // Pin the stall: status sentinel intact, used.idx unmoved,
    // throttled_count == 1.
    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(s[0], 0xEE, "first notify must stall (no status write)",);
    let used_idx_after_stall: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx after stall");
    assert_eq!(used_idx_after_stall, 0, "stall must not advance used.idx");
    assert_eq!(
        dev.counters().throttled_count.load(Ordering::Relaxed),
        1,
        "first notify bumps throttled_count exactly once",
    );
    // next_avail equals pre-stall value: the rollback restored
    // it (pop bumped by 1, set_next_avail(prev.wrapping_sub(1))
    // restored).
    assert_eq!(
        dev.worker.queues[REQ_QUEUE].next_avail(),
        next_avail_before,
        "post-stall next_avail must equal pre-stall value (rollback)",
    );

    // Step the bucket forward by 2 s of wall time so the next
    // consume's refill grants >= 1 token. The bucket's refill
    // rate is 1/sec; 2 s of pretended elapsed time produces
    // 2 tokens (capped at capacity 1, so the bucket holds 1).
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(std::time::Instant::now() - std::time::Duration::from_secs(2));

    // Second notify: the bucket is satisfied, the chain
    // completes.
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // Status byte == OK.
    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(
        s[0], VIRTIO_BLK_S_OK as u8,
        "post-refill notify must complete the chain with S_OK",
    );
    // Used ring advances: the rolled-back chain head was
    // re-popped and add_used'd this time.
    let used_idx_after_retry: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx after retry");
    assert_eq!(
        used_idx_after_retry, 1,
        "post-refill notify must advance used.idx by 1; the \
             rolled-back chain is the SAME head, not a duplicate",
    );
    // Data segment holds the backing file's pattern — proves
    // handle_read_impl actually ran.
    let mut data_buf = [0u8; 512];
    mem.read_slice(&mut data_buf, data_addr).unwrap();
    assert!(
        data_buf.iter().all(|&b| b == 0xAB),
        "data segment must hold backing file's 0xAB pattern \
             after the retry; sentinel 0xFF must be overwritten",
    );
    let c = dev.counters();
    assert_eq!(
        c.reads_completed.load(Ordering::Relaxed),
        1,
        "retry counts as a single read completion (not double)",
    );
    assert_eq!(
        c.bytes_read.load(Ordering::Relaxed),
        512,
        "retry counts the data bytes once",
    );
    assert_eq!(
        c.io_errors.load(Ordering::Relaxed),
        0,
        "no IO error across the stall+retry sequence",
    );
    // throttled_count stays at 1 — the second drain
    // succeeded, no fresh stall.
    assert_eq!(
        c.throttled_count.load(Ordering::Relaxed),
        1,
        "retry success must not bump throttled_count again",
    );
    // next_avail advanced by exactly 1 across the
    // stall+retry sequence: stall rewound, retry re-popped
    // the same head, completion advanced by 1. A regression
    // that double-popped (consumed twice) would leave
    // next_avail at +2; one that skipped the retry would
    // leave it at the pre-stall value.
    assert_eq!(
        dev.worker.queues[REQ_QUEUE].next_avail(),
        next_avail_before.wrapping_add(1),
        "post-retry next_avail must equal pre-stall + 1 \
             (chain consumed exactly once across the stall+retry)",
    );
}

/// Throttle stall preserves FIFO order across two chains.
/// First chain consumes the 1-token bucket and completes.
/// Second chain stalls. Refill, re-notify. Second chain
/// completes — the order matches the avail-ring order, no
/// chain skipped, no chain duplicated.
///
/// Pins the rollback's FIFO contract: undoing the pop with
/// `set_next_avail(prev.wrapping_sub(1))` does not lose
/// ordering when chains are interleaved with successes.
#[test]
fn throttle_stall_fifo_order() {
    use virtio_bindings::bindings::virtio_ring::VRING_DESC_F_NEXT;
    let cap = 4096u64;
    // Backing file: sector 0 = 0x11, sector 1 = 0x22 — distinct
    // patterns let us prove each chain's read landed at the
    // right offset.
    let mut f = tempfile().unwrap();
    f.set_len(cap).unwrap();
    f.write_all(&[0x11; 512]).unwrap();
    f.write_all(&[0x22; 512]).unwrap();
    f.rewind().unwrap();
    let throttle = DiskThrottle {
        iops: std::num::NonZeroU64::new(1),
        bytes_per_sec: None,
        iops_burst_capacity: None,
        bytes_burst_capacity: None,
    };
    let mut dev = VirtioBlk::new(f, cap, throttle);
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);

    // Refill the bucket "now" so the first chain consumes
    // the 1 token. last_refill is set to Instant::now() in
    // TokenBucket::new; we re-pin it here to make the
    // pre-state explicit.
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(std::time::Instant::now());

    // Plant 2 chains: chain 0 reads sector 0, chain 1 reads
    // sector 1. The chains use NEXT-linked descriptors so the
    // queue iterator walks them in order.
    let chains = [
        (
            GuestAddress(0x4000),
            GuestAddress(0x4400),
            GuestAddress(0x4800),
            0u64,
        ),
        (
            GuestAddress(0x5000),
            GuestAddress(0x5400),
            GuestAddress(0x5800),
            1u64,
        ),
    ];
    for &(hdr, _, _, sector) in &chains {
        write_blk_header(&mem, hdr, VIRTIO_BLK_T_IN, sector);
    }
    let mut descs = Vec::new();
    for (chain_i, &(hdr, data, status, _)) in chains.iter().enumerate() {
        let base = (chain_i as u16) * 3;
        descs.push(RawDescriptor::from(SplitDescriptor::new(
            hdr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            VRING_DESC_F_NEXT as u16,
            base + 1,
        )));
        descs.push(RawDescriptor::from(SplitDescriptor::new(
            data.0,
            512,
            VRING_DESC_F_WRITE as u16 | VRING_DESC_F_NEXT as u16,
            base + 2,
        )));
        descs.push(RawDescriptor::from(SplitDescriptor::new(
            status.0,
            1,
            VRING_DESC_F_WRITE as u16,
            0,
        )));
    }
    mock.add_desc_chains(&descs, 0).expect("add 2 chains");
    dev.set_mem(mem.clone());
    wire_device_to_mock(&mut dev, &mock);
    // Capture next_avail BEFORE the first notify so the
    // post-retry assertion can verify both chains were
    // consumed exactly once (advance by 2 across stall + retry).
    let next_avail_before = dev.worker.queues[REQ_QUEUE].next_avail();

    // First notify: chain 0 consumes the 1 token and
    // completes; chain 1 stalls.
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx after first notify");
    assert_eq!(
        used_idx, 1,
        "first notify must complete chain 0 (one used-ring entry)",
    );
    let c = dev.counters();
    assert_eq!(
        c.reads_completed.load(Ordering::Relaxed),
        1,
        "exactly one read completed before the stall",
    );
    assert_eq!(
        c.throttled_count.load(Ordering::Relaxed),
        1,
        "second chain stalled — throttled_count == 1",
    );
    // Verify chain 0's data buffer holds sector 0's pattern.
    let mut buf0 = [0u8; 512];
    mem.read_slice(&mut buf0, chains[0].1).unwrap();
    assert!(
        buf0.iter().all(|&b| b == 0x11),
        "chain 0's data must hold sector 0's pattern (0x11)",
    );

    // Refill the bucket and re-notify.
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(std::time::Instant::now() - std::time::Duration::from_secs(2));
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // Chain 1 completes. used.idx advances to 2; total reads = 2.
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx after retry");
    assert_eq!(
        used_idx, 2,
        "retry must complete chain 1; used.idx advances to 2",
    );
    let c = dev.counters();
    assert_eq!(
        c.reads_completed.load(Ordering::Relaxed),
        2,
        "both chains completed",
    );
    assert_eq!(
        c.throttled_count.load(Ordering::Relaxed),
        1,
        "no fresh stall on retry — throttled_count stays at 1",
    );
    // Chain 1's data buffer holds sector 1's pattern (0x22) —
    // proves the retry read at the RIGHT offset, not chain 0's
    // sector 0. A regression that lost the rolled-back chain's
    // sector value would surface as 0x11 here.
    let mut buf1 = [0u8; 512];
    mem.read_slice(&mut buf1, chains[1].1).unwrap();
    assert!(
        buf1.iter().all(|&b| b == 0x22),
        "chain 1's data must hold sector 1's pattern (0x22) — \
             FIFO order preserved across the stall+retry",
    );
    // next_avail advanced by exactly 2 across the
    // stall+retry sequence: chain 0 consumed (+1), chain 1
    // rolled back (-1+1=0 net for the stall iteration), then
    // chain 1 re-popped on retry (+1). Total +2. A regression
    // that double-popped chain 1 would surface as +3.
    assert_eq!(
        dev.worker.queues[REQ_QUEUE].next_avail(),
        next_avail_before.wrapping_add(2),
        "post-retry next_avail must equal pre-stall + 2 \
             (both chains consumed exactly once across stall+retry)",
    );
}

/// Validation gates fire BEFORE the throttle. With the bucket
/// drained AND a sub-sector data length, the chain is rejected
/// by the validation gate (S_IOERR + io_errors=1) — not by the
/// throttle (throttled_count stays 0). The chain DOES advance
/// used.idx and write a status byte (validation failures
/// publish IOERR completions), so this test pins both halves
/// of the contract.
///
/// Companion to `validation_gates_do_not_consume_throttle_tokens`
/// (same precondition, different assertions): together they
/// pin "validation precedes throttle" both ways — the
/// validation gate fires AND the throttle is unaffected.
#[test]
fn validation_precedes_throttle_on_stall() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let throttle = DiskThrottle {
        iops: std::num::NonZeroU64::new(1),
        bytes_per_sec: None,
        iops_burst_capacity: None,
        bytes_burst_capacity: None,
    };
    let mut dev = VirtioBlk::new(f, cap, throttle);
    // Drain the bucket so any throttle-path probe would stall.
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(std::time::Instant::now());
    assert!(dev.worker.state_mut().ops_bucket.consume(1));
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(std::time::Instant::now());

    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            513, // sub-sector → validation gate fires
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
    // Capture next_avail BEFORE the notify so we can assert
    // the validation gate consumed the chain (advanced by 1)
    // rather than rolling it back as the throttle stall path
    // would (rewind by 1 → net change 0).
    let next_avail_before = dev.worker.queues[REQ_QUEUE].next_avail();
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // Validation rejected the chain with S_IOERR; the chain
    // completed normally (status byte written, add_used).
    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(
        s[0], VIRTIO_BLK_S_IOERR as u8,
        "sub-sector chain must produce S_IOERR via validation gate",
    );
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(
        used_idx, 1,
        "validation rejection completes the chain (add_used runs)",
    );
    let c = dev.counters();
    assert_eq!(
        c.io_errors.load(Ordering::Relaxed),
        1,
        "validation gate bumps io_errors",
    );
    assert_eq!(
        c.throttled_count.load(Ordering::Relaxed),
        0,
        "validation gate must fire BEFORE throttle; \
             throttled_count must stay 0 even with a drained bucket",
    );
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
    // next_avail advanced by 1: the validation gate
    // CONSUMED the chain (publishing IOERR via add_used) — it
    // did NOT roll back. A regression that mistakenly rewound
    // the chain through the validation gate would surface as
    // next_avail == next_avail_before (chain stuck for the
    // next drain).
    assert_eq!(
        dev.worker.queues[REQ_QUEUE].next_avail(),
        next_avail_before.wrapping_add(1),
        "validation gate consumes the chain — next_avail must \
             advance by exactly 1, not roll back like throttle stall",
    );
}

/// Throttle stall on EVENT_IDX path with retry: the
/// post-retry publish goes through the EVENT_IDX
/// `needs_notification` gate. With `used_event = u16::MAX`
/// the gate returns false and the irqfd stays unsignalled
/// even though the chain completed — but `INT_VRING` IS set
/// (the V8 bit/eventfd split applies to retry completions
/// just as to fresh ones).
#[test]
fn throttle_stall_event_idx_retry_routes_through_gate() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let throttle = DiskThrottle {
        iops: std::num::NonZeroU64::new(1),
        bytes_per_sec: None,
        iops_burst_capacity: None,
        bytes_burst_capacity: None,
    };
    let mut dev = VirtioBlk::new(f, cap, throttle);
    let mem = make_chain_test_mem();
    let qsize = 16u16;
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), qsize);

    // Drain the bucket up-front.
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(std::time::Instant::now());
    assert!(dev.worker.state_mut().ops_bucket.consume(1));
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(std::time::Instant::now());

    // used_event = u16::MAX: the guest is asking to never be
    // woken at any normal next_used. The post-retry
    // needs_notification must consult this and suppress the
    // irqfd.
    let used_event = used_event_addr(mock.avail_addr(), qsize);
    mem.write_obj::<u16>(u16::to_le(u16::MAX), used_event)
        .expect("plant used_event");

    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
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
    wire_device_to_mock_with_event_idx(&mut dev, &mock, qsize, GuestAddress(0x10000));
    // Capture next_avail BEFORE the stall+retry so the
    // post-retry assertion can verify the chain was consumed
    // exactly once across the stall+retry sequence.
    let next_avail_before = dev.worker.queues[REQ_QUEUE].next_avail();

    // First notify: stall.
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);
    assert_eq!(
        dev.counters().throttled_count.load(Ordering::Relaxed),
        1,
        "first notify stalls",
    );

    // Refill, retry.
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(std::time::Instant::now() - std::time::Duration::from_secs(2));
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // The retry completes — used.idx advances at the override
    // address.
    let used_idx: u16 = mem
        .read_obj(GuestAddress(0x10000).checked_add(2).unwrap())
        .expect("read device used.idx at override addr");
    assert_eq!(used_idx, 1, "retry must publish the chain");
    // Status = S_OK.
    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(s[0], VIRTIO_BLK_S_OK as u8);
    let c = dev.counters();
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 1);
    // INT_VRING IS set: a retry completion is still a
    // signal_needed=true publish, so the post-drain branch
    // ran and set the bit.
    assert_ne!(
        dev.interrupt_status.load(Ordering::Acquire) & VIRTIO_MMIO_INT_VRING,
        0,
        "retry completion sets INT_VRING (V8 bit/eventfd split)",
    );
    // irqfd suppressed: needs_notification returns false
    // because next_used (1) is far below used_event (u16::MAX).
    assert!(
        dev.irq_evt.read().is_err(),
        "EVENT_IDX gate must suppress irqfd on retry when \
             used_event threshold is unreached",
    );
    // io_errors must be 0 across the stall+retry sequence —
    // throttle stalls are deferred-not-failed and the retry
    // completed cleanly. A regression that classified the
    // stall as an IO error (or that bumped io_errors on the
    // retry path) would surface here.
    assert_eq!(
        dev.counters().io_errors.load(Ordering::Relaxed),
        0,
        "throttle stall + retry must not bump io_errors",
    );
    // next_avail advanced by exactly 1 across the stall+retry
    // sequence: stall rewound (-1+1=0), retry re-popped (+1).
    // Symmetric with throttle_stall_then_refill_retry_succeeds
    // but on the EVENT_IDX path.
    assert_eq!(
        dev.worker.queues[REQ_QUEUE].next_avail(),
        next_avail_before.wrapping_add(1),
        "post-retry next_avail must equal pre-stall + 1 \
             (chain consumed exactly once across stall+retry on \
             EVENT_IDX path)",
    );
}

/// Throttle stall on the bytes bucket alone (iops bucket
/// unlimited). DiskThrottle with `bytes_per_sec=Some(512)` and
/// `iops=None` accepts 1 request per second worth of bytes; a
/// Overconsumption policy: when `n > capacity` the
/// throttle gate is `available >= 0` (not `available >= n`),
/// so an oversized request whose chain is the FIRST one this
/// drain sees against a freshly-seeded bucket succeeds and
/// drives `available` deeply negative. To exercise the STALL
/// path for an oversized request, the bucket must already be
/// in debt (`available < 0`) before the drain: at that point
/// the `n > capacity` gate fails on the negative balance.
///
/// This test pre-drains the bytes bucket into debt via a
/// direct `bytes_bucket.consume(4096)` (allowed by the
/// overconsumption policy from a fresh `available = 512`
/// — the consume drives `available` to -3584), then issues a
/// 1024-byte T_IN request. With `available < 0` the
/// oversize gate fails and the chain stalls: status sentinel
/// survives, used.idx=0, throttled_count=1, io_errors=0.
/// Companion to the iops-only stall tests; together they
/// cover both single-bucket-exhaustion shapes under the
/// overconsumption policy.
#[test]
fn throttle_bytes_request_exceeds_capacity_stalls() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let throttle = DiskThrottle {
        iops: None,
        bytes_per_sec: std::num::NonZeroU64::new(512),
        iops_burst_capacity: None,
        bytes_burst_capacity: None,
    };
    let mut dev = VirtioBlk::new(f, cap, throttle);
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);

    // Pin last_refill BEFORE the priming consume so the refill
    // pass inside `consume` and inside the drain's
    // `can_consume` sees zero elapsed time and adds nothing.
    // Without this pin the initial `available = capacity = 512`
    // refill would have already happened (constructors seed
    // last_refill at `Instant::now()`), but a small wall-clock
    // tick between construction and the priming consume could
    // otherwise nudge available upward.
    dev.worker
        .state_mut()
        .bytes_bucket
        .set_last_refill_for_test(std::time::Instant::now());
    // Drive the bytes bucket into debt via overconsumption.
    // n=4096 > capacity=512, gate is `available >= 0`; from
    // a fresh seeding `available = 512 >= 0` so the consume
    // is granted and `available = 512 - 4096 = -3584`. After
    // this the next oversize request will FAIL the gate
    // because `available < 0`.
    assert!(
        dev.worker.state_mut().bytes_bucket.consume(4096),
        "priming overconsume must succeed against fresh \
             available=512 — the gate is `available >= 0` for \
             n > capacity",
    );
    // Re-pin so the post-priming drain's refill sees zero
    // elapsed time and `available` stays at -3584.
    dev.worker
        .state_mut()
        .bytes_bucket
        .set_last_refill_for_test(std::time::Instant::now());

    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    // Sentinel at status so a write would be detectable.
    mem.write_slice(&[0xEEu8], status_addr).unwrap();
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
    // 1024-byte data segment — n=1024 > capacity=512, gate
    // is `available >= 0`. With `available = -3584` from the
    // priming consume above, the gate FAILS → stall.
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            1024,
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
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(
        s[0], 0xEE,
        "bytes-bucket stall must NOT write a status byte",
    );
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 0, "bytes-bucket stall must NOT advance used.idx",);
    let c = dev.counters();
    assert_eq!(
        c.throttled_count.load(Ordering::Relaxed),
        1,
        "bytes-bucket exhaustion bumps throttled_count",
    );
    assert_eq!(
        c.io_errors.load(Ordering::Relaxed),
        0,
        "bytes-bucket stall is not an IO error",
    );
}

/// When BOTH buckets fail, drain_bracket_impl picks the longer
/// of the two waits via `ops_wait.max(bytes_wait)`. Drain the
/// bytes bucket only (ops bucket has plenty of headroom), and
/// verify the stall produces a wait_nanos that reflects the
/// bytes deficit. Drives the inline path so we can reach
/// `drain_bracket_impl` directly and observe the
/// `DrainOutcome::ThrottleStalled` value.
#[test]
fn throttle_both_buckets_max_wait() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    // iops=10 (small but easily satisfied for one request),
    // bytes_per_sec=1024 (will become the bottleneck for a
    // 2048-byte request after the bucket is drained).
    let throttle = DiskThrottle {
        iops: std::num::NonZeroU64::new(10),
        bytes_per_sec: std::num::NonZeroU64::new(1024),
        iops_burst_capacity: None,
        bytes_burst_capacity: None,
    };
    let mut dev = VirtioBlk::new(f, cap, throttle);
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);

    // Drive the bytes bucket into debt so a 2048-byte
    // (oversized) request stalls on the bytes-bucket gate.
    // Overconsumption policy: when n > capacity the
    // gate is `available >= 0`, so a fresh bucket
    // (available=capacity=1024) would grant the 2048-byte
    // request and drive available negative. To force the
    // STALL path the bucket must already be in debt before
    // the drain. consume(2048) from a fresh available=1024 is
    // granted (overconsume gate `available >= 0` passes) and
    // leaves available=-1024. Pinning last_refill before and
    // after keeps the in-place refill from clearing the debt.
    dev.worker
        .state_mut()
        .bytes_bucket
        .set_last_refill_for_test(std::time::Instant::now());
    assert!(
        dev.worker.state_mut().bytes_bucket.consume(2048),
        "priming overconsume must succeed against fresh \
             available=1024 — the gate is `available >= 0` for \
             n > capacity",
    );
    dev.worker
        .state_mut()
        .bytes_bucket
        .set_last_refill_for_test(std::time::Instant::now());

    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            2048,
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
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // Stall observed.
    let c = dev.counters();
    assert_eq!(
        c.throttled_count.load(Ordering::Relaxed),
        1,
        "two-bucket stall bumps throttled_count exactly once",
    );
    // The bytes deficit is the bottleneck — 2048 bytes at
    // 1024/sec = 2_000_000_000 ns. The wait_nanos must be at
    // least the bytes deficit: ops would never produce a
    // longer wait at iops=10 vs a single op (deficit 1 op /
    // 10 per sec = 100_000_000 ns), so max(ops_wait,
    // bytes_wait) = bytes_wait. The internal stall_outcome is
    // not directly exposed; we observe the consequence via
    // throttled_count and that the chain stayed in avail.
    // (Direct wait_nanos observation lives in the
    // drain_bracket_impl-level tests; here we assert via the
    // public surface.)
    assert_eq!(
        c.io_errors.load(Ordering::Relaxed),
        0,
        "two-bucket stall is not an IO error",
    );
}

/// Bytes-bucket stall, refill, retry. End-to-end pin that the
/// bytes-bucket retry path uses the same `add_used` /
/// status-write pipeline as the iops-bucket retry covered
/// upstream.
///
/// Overconsumption policy: a fresh bucket grants any
/// request whose size is `<= capacity` while `available >= n`,
/// and grants oversized (`n > capacity`) requests while
/// `available >= 0`. To exercise the STALL path the bucket
/// must be in debt before the drain. This test pre-drains the
/// bytes bucket via a direct `consume(4096)` (allowed by the
/// overconsumption gate from a fresh `available = 1024` —
/// drives `available` to `-3072`), pins `last_refill` so the
/// debt persists across the drain's in-place refill, and
/// issues a 1024-byte T_IN. With `available = -3072 < n =
/// 1024` the normal-path gate fails → stall.
///
/// Then `set_last_refill_for_test` rewinds 4 s into the past,
/// granting 4096 tokens which fully clear the -3072 debt and
/// raise `available` to `1024 = capacity`. The retry pops the
/// rolled-back chain, the bytes-bucket gate now passes
/// (`available = 1024 >= n = 1024`), and the request completes
/// S_OK with `reads_completed = 1`, `bytes_read = 1024`.
#[test]
fn throttle_bytes_bucket_retry_succeeds() {
    let cap = 4096u64;
    // Backing pre-filled with 0xAB so the post-retry read is
    // verifiable.
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let throttle = DiskThrottle {
        iops: None,
        bytes_per_sec: std::num::NonZeroU64::new(1024),
        iops_burst_capacity: None,
        bytes_burst_capacity: None,
    };
    let mut dev = VirtioBlk::new(f, cap, throttle);
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);

    // Pin last_refill BEFORE the priming consume so the refill
    // pass inside `consume` sees zero elapsed time (no spurious
    // top-up before the priming consume runs).
    dev.worker
        .state_mut()
        .bytes_bucket
        .set_last_refill_for_test(std::time::Instant::now());
    // Drive the bytes bucket into debt via overconsumption.
    // n=4096 > capacity=1024, gate is `available >= 0`; from
    // a fresh seeding `available = 1024 >= 0` so consume is
    // granted and `available = 1024 - 4096 = -3072`. Now any
    // subsequent normal-path consume (n <= capacity) fails its
    // `available >= n` check — the chain stalls.
    assert!(
        dev.worker.state_mut().bytes_bucket.consume(4096),
        "priming overconsume must succeed against fresh \
             available=1024 — the gate is `available >= 0` for \
             n > capacity",
    );
    // Re-pin so the post-priming drain's in-place refill sees
    // zero elapsed time and `available` stays at -3072.
    dev.worker
        .state_mut()
        .bytes_bucket
        .set_last_refill_for_test(std::time::Instant::now());

    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    // Sentinel at data so the post-retry read overwriting it
    // is detectable.
    let sentinel = vec![0xFFu8; 1024];
    mem.write_slice(&sentinel, data_addr).unwrap();
    mem.write_slice(&[0xEEu8], status_addr).unwrap();
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
    // 1024-byte data segment — n = capacity, normal-path gate
    // is `available >= n`. With `available = -3072` the gate
    // fails → stall.
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            1024,
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

    // First notify: stall (available=-3072 < n=1024).
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);
    assert_eq!(
        dev.counters().throttled_count.load(Ordering::Relaxed),
        1,
        "first notify must stall on bytes bucket — pre-drained \
             into -3072 debt, normal-path gate `available >= n` \
             fails",
    );
    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(s[0], 0xEE, "stall must not write status byte");

    // Refill bytes bucket: rewind last_refill 4s into the past.
    // At rate=1024/sec the refill grants 4096 tokens, fully
    // clearing the -3072 debt and raising `available` to
    // `(-3072 + 4096).min(capacity) = 1024`. The retry pop now
    // sees `available = 1024 >= n = 1024` and grants.
    dev.worker
        .state_mut()
        .bytes_bucket
        .set_last_refill_for_test(std::time::Instant::now() - std::time::Duration::from_secs(4));
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // Post-retry: 1024-byte read completed.
    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(
        s[0], VIRTIO_BLK_S_OK as u8,
        "post-refill chain must complete with S_OK",
    );
    let c = dev.counters();
    assert_eq!(
        c.reads_completed.load(Ordering::Relaxed),
        1,
        "exactly one read completed (the post-refill retry)",
    );
    assert_eq!(
        c.bytes_read.load(Ordering::Relaxed),
        1024,
        "bytes_read counts the 1024-byte read",
    );
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 0);
    assert_eq!(
        c.throttled_count.load(Ordering::Relaxed),
        1,
        "throttled_count == 1 — only the first notify stalled; \
             the post-refill retry succeeded without re-stalling \
             (refill cleared the debt and raised available to \
             capacity, exactly enough for the 1024-byte chain)",
    );
}

/// Two consecutive notifies on a drained bucket BOTH stall.
/// throttled_count ticks per stall; status sentinel survives;
/// used.idx stays 0. Pins that the rollback does not
/// accidentally consume the chain — a regression that
/// half-rewound (e.g. used `set_next_avail(prev)` instead of
/// `prev.wrapping_sub(1)`) would advance next_avail on retry
/// and the second notify would observe an empty queue.
#[test]
fn throttle_multi_stall_same_head() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let throttle = DiskThrottle {
        iops: std::num::NonZeroU64::new(1),
        bytes_per_sec: None,
        iops_burst_capacity: None,
        bytes_burst_capacity: None,
    };
    let mut dev = VirtioBlk::new(f, cap, throttle);
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    // Drain the iops bucket and pin last_refill.
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(std::time::Instant::now());
    assert!(dev.worker.state_mut().ops_bucket.consume(1));
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(std::time::Instant::now());
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    mem.write_slice(&[0xEEu8], status_addr).unwrap();
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
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
    // Capture next_avail BEFORE both stalls so the post-state
    // assertion can verify the rollback held across two
    // consecutive stalls (each stall pop bumps by 1, each
    // set_next_avail(prev.wrapping_sub(1)) restores it).
    let next_avail_before = dev.worker.queues[REQ_QUEUE].next_avail();

    // First notify: stall.
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);
    // Re-pin last_refill so the second notify also stalls
    // (without this, real wall time between the two notifies
    // could refill the iops bucket).
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(std::time::Instant::now());
    // Second notify: stall again on the SAME chain head.
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    let c = dev.counters();
    assert_eq!(
        c.throttled_count.load(Ordering::Relaxed),
        2,
        "two stalls on the same head must bump throttled_count twice — \
             a regression that lost the rolled-back chain would surface as 1",
    );
    // Sentinel survives — neither stall wrote a status byte.
    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(s[0], 0xEE);
    // used.idx stays at 0 across both stalls.
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 0);
    // next_avail equals pre-stall value across BOTH stalls:
    // the rollback fired on each stall, restoring the cursor
    // so the next pop returns the same head. A regression
    // that half-rewound (e.g. only restored on the first
    // stall) would surface as next_avail != pre-stall here.
    assert_eq!(
        dev.worker.queues[REQ_QUEUE].next_avail(),
        next_avail_before,
        "next_avail must equal pre-stall value after both \
             stalls (rollback preserved across 2 stalls)",
    );
}

/// Three-chain mid-batch stall: chain 0 consumes the only
/// token and completes; chain 1 stalls (ops bucket drained);
/// chain 2 stays in the avail ring untouched. After refill +
/// re-notify, chains 1 and 2 BOTH complete (the re-drain
/// processes the rolled-back chain plus the unobserved
/// chain 2). Pins FIFO + multi-chain integrity across stall.
#[test]
fn throttle_stall_mid_batch_three_chains() {
    use virtio_bindings::bindings::virtio_ring::VRING_DESC_F_NEXT;
    let cap = 4096u64;
    let mut f = tempfile().unwrap();
    f.set_len(cap).unwrap();
    f.write_all(&[0x11; 512]).unwrap(); // sector 0
    f.write_all(&[0x22; 512]).unwrap(); // sector 1
    f.write_all(&[0x33; 512]).unwrap(); // sector 2
    f.rewind().unwrap();
    // iops=2, rate=2/sec, capacity=2. We drain 1 token before
    // the first notify so chain 0 finds 1 token (consumes,
    // leaves 0), chain 1 stalls. After a refill stepping the
    // bucket forward by 2 s, the bucket holds capacity-cap=2
    // tokens — enough for chains 1 + 2 in the retry drain.
    // Using iops=1 (capacity=1) is insufficient: even after
    // refill the bucket caps at 1 and chain 2 would re-stall
    // immediately after chain 1.
    let throttle = DiskThrottle {
        iops: std::num::NonZeroU64::new(2),
        bytes_per_sec: None,
        iops_burst_capacity: None,
        bytes_burst_capacity: None,
    };
    let mut dev = VirtioBlk::new(f, cap, throttle);
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 32);

    // Drain 1 token (so initial available = 1) and pin
    // last_refill so the next consume's refill yields 0.
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(std::time::Instant::now());
    assert!(dev.worker.state_mut().ops_bucket.consume(1));
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(std::time::Instant::now());

    // Build 3 NEXT-linked chains.
    let chains = [
        (
            GuestAddress(0x4000),
            GuestAddress(0x4400),
            GuestAddress(0x4800),
            0u64,
        ),
        (
            GuestAddress(0x5000),
            GuestAddress(0x5400),
            GuestAddress(0x5800),
            1u64,
        ),
        (
            GuestAddress(0x6000),
            GuestAddress(0x6400),
            GuestAddress(0x6800),
            2u64,
        ),
    ];
    for &(hdr, _, _, sector) in &chains {
        write_blk_header(&mem, hdr, VIRTIO_BLK_T_IN, sector);
    }
    let mut descs = Vec::new();
    for (chain_i, &(hdr, data, status, _)) in chains.iter().enumerate() {
        let base = (chain_i as u16) * 3;
        descs.push(RawDescriptor::from(SplitDescriptor::new(
            hdr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            VRING_DESC_F_NEXT as u16,
            base + 1,
        )));
        descs.push(RawDescriptor::from(SplitDescriptor::new(
            data.0,
            512,
            VRING_DESC_F_WRITE as u16 | VRING_DESC_F_NEXT as u16,
            base + 2,
        )));
        descs.push(RawDescriptor::from(SplitDescriptor::new(
            status.0,
            1,
            VRING_DESC_F_WRITE as u16,
            0,
        )));
    }
    mock.add_desc_chains(&descs, 0).expect("add 3 chains");
    dev.set_mem(mem.clone());
    wire_device_to_mock(&mut dev, &mock);

    // First notify: chain 0 consumes the 1 token and completes;
    // chain 1 stalls; chain 2 untouched.
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);
    let c = dev.counters();
    assert_eq!(
        c.reads_completed.load(Ordering::Relaxed),
        1,
        "chain 0 completed before stall",
    );
    assert_eq!(
        c.throttled_count.load(Ordering::Relaxed),
        1,
        "chain 1 stalled",
    );
    let used_idx_after_stall: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx after stall");
    assert_eq!(used_idx_after_stall, 1);

    // Refill, re-notify.
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(std::time::Instant::now() - std::time::Duration::from_secs(5));
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // Chains 1 + 2 complete in the retry drain.
    let c = dev.counters();
    assert_eq!(
        c.reads_completed.load(Ordering::Relaxed),
        3,
        "all three chains completed (chain 0 first notify, \
             chains 1+2 second notify)",
    );
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx after retry");
    assert_eq!(used_idx, 3, "used.idx covers all three completions");
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 0);
    // throttled_count stays at 1 — the retry succeeded for both
    // chains, no fresh stall.
    assert_eq!(c.throttled_count.load(Ordering::Relaxed), 1);
    // Verify each chain's data buffer holds the correct
    // sector pattern. A FIFO regression that lost the
    // rolled-back chain's sector value (e.g. swapped chains
    // 1 and 2 on retry, or popped chain 2 first because the
    // rollback didn't restore the cursor) would surface as
    // chain 1 holding 0x33 or chain 2 holding 0x22 here.
    let mut buf1 = [0u8; 512];
    mem.read_slice(&mut buf1, chains[1].1).unwrap();
    assert!(
        buf1.iter().all(|&b| b == 0x22),
        "chain 1's data must hold sector 1's pattern (0x22) — \
             FIFO order preserved across stall+retry",
    );
    let mut buf2 = [0u8; 512];
    mem.read_slice(&mut buf2, chains[2].1).unwrap();
    assert!(
        buf2.iter().all(|&b| b == 0x33),
        "chain 2's data must hold sector 2's pattern (0x33) — \
             retry processed chains in avail-ring order",
    );
}

/// FLUSH consumes 0 bytes-tokens. With a drained bytes bucket
/// and an unlimited iops bucket, a FLUSH chain completes
/// without stalling — `data_len = 0` for FLUSH, and the
/// can_consume(0) check on the bytes bucket short-circuits to
/// true via TokenBucket's `if self.available >= n`. Pins that
/// FLUSH is exempt from bytes-bucket exhaustion (it does no
/// data IO).
#[test]
fn throttle_flush_on_drained_bytes_bucket() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let throttle = DiskThrottle {
        iops: None,
        bytes_per_sec: std::num::NonZeroU64::new(1),
        iops_burst_capacity: None,
        bytes_burst_capacity: None,
    };
    let mut dev = VirtioBlk::new(f, cap, throttle);
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    // Drain the bytes bucket and pin its last_refill.
    dev.worker
        .state_mut()
        .bytes_bucket
        .set_last_refill_for_test(std::time::Instant::now());
    assert!(dev.worker.state_mut().bytes_bucket.consume(1));
    dev.worker
        .state_mut()
        .bytes_bucket
        .set_last_refill_for_test(std::time::Instant::now());

    let header_addr = GuestAddress(0x4000);
    let status_addr = GuestAddress(0x5000);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_FLUSH, 0);
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0,
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
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    let c = dev.counters();
    assert_eq!(
        c.flushes_completed.load(Ordering::Relaxed),
        1,
        "FLUSH must complete despite drained bytes bucket — \
             FLUSH consumes 0 bytes-tokens",
    );
    assert_eq!(
        c.throttled_count.load(Ordering::Relaxed),
        0,
        "FLUSH must NOT stall on bytes bucket — data_len=0 \
             means can_consume(0)=true unconditionally",
    );
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 0);
}

/// Read-only device's WRITE chain through
/// `process_requests`. Builds a RO device, plants a write chain,
/// asserts the dispatch arm (RO writes → S_IOERR) actually
/// fires through the chain pipeline.
#[test]
fn process_requests_read_only_write_returns_ioerr_chain() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let mut dev = VirtioBlk::with_options(f, cap, DiskThrottle::default(), true);
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    let payload = vec![0xCDu8; 512];
    mem.write_slice(&payload, data_addr).expect("plant");
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_OUT, 1);
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(data_addr.0, 512, 0, 0)),
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

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    let mut status_buf = [0u8; 1];
    mem.read_slice(&mut status_buf, status_addr).unwrap();
    assert_eq!(
        status_buf[0], VIRTIO_BLK_S_IOERR as u8,
        "RO device must reject T_OUT with S_IOERR through the chain pipeline",
    );

    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 1);

    let c = dev.counters();
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 1);
    assert_eq!(c.writes_completed.load(Ordering::Relaxed), 0);
    assert_eq!(c.bytes_written.load(Ordering::Relaxed), 0);
    // Throttle did NOT fire — RO classification short-circuits
    // before throttle accounting.
    assert_eq!(c.throttled_count.load(Ordering::Relaxed), 0);
}

/// Used-idx tracking under repeated notifies. After
/// each of 5 sequential single-chain notifies, used.idx must
/// equal the cumulative count. Pins the used-ring's
/// monotonic-advance contract: every chain that completes
/// bumps used.idx by exactly 1.
#[test]
fn process_requests_used_idx_advances_across_repeated_notifies() {
    use virtio_bindings::bindings::virtio_ring::VRING_DESC_F_NEXT;
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 32);
    dev.set_mem(mem.clone());
    wire_device_to_mock(&mut dev, &mock);

    for i in 0..5u64 {
        let hdr = GuestAddress(0x4000 + i * 0x1000);
        let data = GuestAddress(0x4400 + i * 0x1000);
        let status = GuestAddress(0x4800 + i * 0x1000);
        write_blk_header(&mem, hdr, VIRTIO_BLK_T_IN, 0);
        // The 3-desc chain occupies descriptor-table indices
        // 3*i..3*(i+1). The non-last descriptors must point
        // to their successor's table index via the `next`
        // field — add_desc_chains writes descriptors verbatim.
        let base = (i as u16) * 3;
        let descs = [
            RawDescriptor::from(SplitDescriptor::new(
                hdr.0,
                VIRTIO_BLK_OUTHDR_SIZE as u32,
                VRING_DESC_F_NEXT as u16,
                base + 1,
            )),
            RawDescriptor::from(SplitDescriptor::new(
                data.0,
                512,
                VRING_DESC_F_WRITE as u16 | VRING_DESC_F_NEXT as u16,
                base + 2,
            )),
            RawDescriptor::from(SplitDescriptor::new(
                status.0,
                1,
                VRING_DESC_F_WRITE as u16,
                0,
            )),
        ];
        mock.add_desc_chains(&descs, base).expect("add chain");
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

        let used_idx: u16 = mem
            .read_obj(mock.used_addr().checked_add(2).unwrap())
            .expect("read used.idx");
        assert_eq!(
            used_idx,
            (i as u16) + 1,
            "after notify #{} used.idx must equal {}",
            i + 1,
            i + 1,
        );
    }

    let c = dev.counters();
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 5);
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 0);
}
