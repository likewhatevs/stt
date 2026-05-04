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
// Validation-gap tests against process_requests.
//
// Every test below pins a code path that production exercises
// in real-world failure modes but no upstream test covered. All
// are driven through the PUBLIC interface (`process_requests`
// via QUEUE_NOTIFY + MockSplitQueue) — internal helpers may
// change shape, but the chain-walking → classify → dispatch →
// publish-completion contract is the invariant.
// ----------------------------------------------------------------

/// SEG_MAX enforcement. The device advertises
/// `VIRTIO_BLK_F_SEG_MAX = 128`, so a chain with more than
/// 128 + 2 (header + status) descriptors is malformed.
/// Without this gate, a hostile guest could submit thousands
/// of descriptors and force the device to allocate matching
/// scratch storage per request (heap blowup).
///
/// The gate runs AFTER status_addr identification so the
/// rejection produces a normal IOERR completion (status byte
/// + add_used) — not a chain drop. Earlier-positioned drop
/// behaviour was the original design but left the chain stuck
/// in the avail ring until the guest's hung-task watchdog
/// fired (`kernel.hung_task_timeout_secs`, default 120 s —
/// virtio_blk has no `mq_ops->timeout`), hiding the rejection
/// from operators.
#[test]
fn seg_max_rejected_with_ioerr() {
    use virtio_bindings::bindings::virtio_ring::VRING_DESC_F_NEXT;
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    // Need queue size > 130 so the chain fits in the table even
    // though the device's SEG_MAX gate rejects it.
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 256);
    dev.set_mem(mem.clone());
    wire_device_to_mock(&mut dev, &mock);

    // Build 132 descriptors total: 1 header + 130 data + 1 status
    // = 132 > VIRTIO_BLK_SEG_MAX (128) + 2 = 130.
    let header_addr = GuestAddress(0x10000);
    let status_addr = GuestAddress(0x20000);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
    let total_descs: u16 = 132;
    let data_descs: u16 = total_descs - 2;
    let mut descs = Vec::new();
    descs.push(RawDescriptor::from(SplitDescriptor::new(
        header_addr.0,
        VIRTIO_BLK_OUTHDR_SIZE as u32,
        VRING_DESC_F_NEXT as u16,
        1,
    )));
    for i in 0..data_descs {
        descs.push(RawDescriptor::from(SplitDescriptor::new(
            0x40000 + i as u64 * 8,
            8,
            VRING_DESC_F_WRITE as u16 | VRING_DESC_F_NEXT as u16,
            i + 2,
        )));
    }
    descs.push(RawDescriptor::from(SplitDescriptor::new(
        status_addr.0,
        1,
        VRING_DESC_F_WRITE as u16,
        0,
    )));
    // Pre-fill status_addr with 0xEE — a value distinct from
    // S_OK (0), S_IOERR (1), S_UNSUPP (2). The post-notify
    // assertion expects the device to overwrite this with
    // S_IOERR.
    mem.write_slice(&[0xEEu8], status_addr).unwrap();
    mock.add_desc_chains(&descs, 0).expect("add chain");
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // Used ring advances — SEG_MAX rejection returns the chain
    // via add_used so the guest sees an immediate completion.
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 1, "SEG_MAX rejection still updates used.idx");

    let c = dev.counters();
    assert!(c.io_errors.load(Ordering::Relaxed) >= 1);
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
    // Throttle untouched — gate fires before token consumption.
    assert_eq!(c.throttled_count.load(Ordering::Relaxed), 0);

    // Status byte is S_IOERR — not the 0xEE sentinel and not
    // a stale 0 (which would be S_OK silent corruption).
    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(
        s[0], VIRTIO_BLK_S_IOERR as u8,
        "SEG_MAX rejection must write S_IOERR to status descriptor",
    );
}

/// Header read_obj failure. The header descriptor's
/// `addr` points at unmapped guest memory, so `mem.read_obj`
/// fails. The device writes IOERR to status, increments
/// io_errors, calls add_used.
#[test]
fn header_read_obj_failure_returns_ioerr() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    // Header at GPA 0x200000 — past the 1 MiB region's end.
    // status_addr stays inside the region so the IOERR write
    // can succeed.
    let header_addr = GuestAddress(0x200000);
    let status_addr = GuestAddress(0x4000);
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

    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(
        s[0], VIRTIO_BLK_S_IOERR as u8,
        "header read failure must surface as S_IOERR",
    );
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 1);

    let c = dev.counters();
    assert!(c.io_errors.load(Ordering::Relaxed) >= 1);
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
}

/// SIZE_MAX enforcement. A single data descriptor longer
/// than VIRTIO_BLK_SIZE_MAX (1 MB) is malformed. Without the
/// gate, a guest can force `vec![0u8; 4 GB]` heap allocations.
#[test]
fn size_max_oversized_data_desc_rejected() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x8000);
    let status_addr = GuestAddress(0x9000);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
    // Data desc len = 1 MB + 1.
    let oversize: u32 = (1u32 << 20) + 1;
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            oversize,
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
    assert_eq!(s[0], VIRTIO_BLK_S_IOERR as u8);
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 1);

    let c = dev.counters();
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 1);
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
    assert_eq!(c.throttled_count.load(Ordering::Relaxed), 0);
}

/// Zero-data T_IN. virtio-v1.2 §5.2.6 defines IN/OUT as
/// carrying a non-empty data payload; cloud-hypervisor
/// explicitly rejects header+status-only chains for these
/// request types.
#[test]
fn zero_data_t_in_returns_ioerr() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let status_addr = GuestAddress(0x5000);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
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

    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(s[0], VIRTIO_BLK_S_IOERR as u8);
    let c = dev.counters();
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 1);
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
    assert_eq!(c.throttled_count.load(Ordering::Relaxed), 0);
}

/// Sub-sector data_len. virtio-v1.2 §5.2.6 defines
/// T_IN/T_OUT as sector-aligned transfers; firecracker's
/// `Request::parse` rejects sub-sector lengths.
#[test]
fn sub_sector_data_len_returns_ioerr() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
    // Data len = 513.
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            513,
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
    assert_eq!(s[0], VIRTIO_BLK_S_IOERR as u8);

    let c = dev.counters();
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 1);
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
    assert_eq!(c.throttled_count.load(Ordering::Relaxed), 0);
}

/// Direction violation. T_IN with a non-write-only data
/// descriptor is a spec violation. Must reject; throttle
/// untouched (gate fires pre-consume).
#[test]
fn direction_violation_t_in_with_ro_data_returns_ioerr() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
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
            512,
            0, // device-readable — wrong for T_IN
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
    assert_eq!(s[0], VIRTIO_BLK_S_IOERR as u8);

    let c = dev.counters();
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 1);
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
    assert_eq!(
        c.throttled_count.load(Ordering::Relaxed),
        0,
        "direction violation must NOT touch throttle bucket",
    );
}

/// Direction violation. T_OUT with a device-writable data
/// descriptor is a spec violation (virtio-v1.2 §5.2.6: T_OUT
/// data segments must be device-readable). Must reject;
/// throttle untouched (gate fires pre-consume); writes_completed
/// stays at 0; backing file untouched. Exercises the
/// `VIRTIO_BLK_T_OUT => data_segments.iter().any(|d| d.is_write_only)`
/// match arm in `process_requests`.
#[test]
fn direction_violation_t_out_with_writable_data_returns_ioerr() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    // Pre-fill the data segment with a sentinel so we can
    // verify the device did NOT read from the backing file
    // into it (which would indicate the wrong code path ran).
    let sentinel = vec![0xCDu8; 512];
    mem.write_slice(&sentinel, data_addr).unwrap();
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_OUT, 1);
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
            VRING_DESC_F_WRITE as u16, // device-writable — wrong for T_OUT
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
    assert_eq!(s[0], VIRTIO_BLK_S_IOERR as u8);

    let c = dev.counters();
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 1);
    assert_eq!(
        c.writes_completed.load(Ordering::Relaxed),
        0,
        "direction-violating T_OUT must NOT count as a completed write",
    );
    assert_eq!(
        c.bytes_written.load(Ordering::Relaxed),
        0,
        "direction-violating T_OUT must NOT touch the backing file",
    );
    assert_eq!(
        c.throttled_count.load(Ordering::Relaxed),
        0,
        "direction violation must NOT touch throttle bucket",
    );
    // Sentinel must remain — proves the device did not read
    // backing data into the data segment (T_IN handler path
    // would write into it, which would indicate the gate
    // failed and the wrong arm ran).
    let mut data_check = vec![0u8; 512];
    mem.read_slice(&mut data_check, data_addr).unwrap();
    assert!(
        data_check.iter().all(|&b| b == 0xCDu8),
        "data segment sentinel must be intact — device must not run the read or write path",
    );
}

/// Status write_slice failure. The status descriptor
/// points at unmapped guest memory. When status
/// write fails, the device does NOT call add_used. The
/// descriptor head stays in the avail ring; io_errors bumps.
#[test]
fn status_write_slice_failure_no_add_used() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    // Status_addr beyond the 1 MiB region → write_slice fails.
    let status_addr = GuestAddress(0x300000);
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

    // used.idx must NOT advance — no add_used when
    // status write fails.
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(
        used_idx, 0,
        "status write failure must skip add_used (status-write-success gate); used.idx stays at 0",
    );

    let c = dev.counters();
    assert!(
        c.io_errors.load(Ordering::Relaxed) >= 1,
        "status write failure bumps io_errors",
    );
}

/// add_used Err path baseline. A clean fixture cannot
/// force `add_used` to fail (virtio-queue's add_used returns
/// Ok unless the head is out of range or the used ring address
/// is invalid). Best-effort coverage: pin that successful
/// add_used does NOT bump io_errors. A regression that
/// introduced a new add_used-fail site would surface as
/// elevated io_errors here.
///
/// Production add_used Err arms (lines that bump io_errors
/// when add_used returns Err) are reviewed in code: every
/// arm matches the established `if let Err(e) = q.add_used(...)
/// { ... io_errors.fetch_add(1) }` shape.
#[test]
fn add_used_err_path_baseline_io_errors_zero() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
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
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 1);
    let c = dev.counters();
    assert_eq!(
        c.io_errors.load(Ordering::Relaxed),
        0,
        "successful add_used must NOT bump io_errors",
    );
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 1);
}

/// RO-flush through full chain. RO disks accept T_FLUSH
/// and return S_OK. flushes_completed increments.
#[test]
fn ro_flush_full_chain_returns_ok_increments_counter() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let mut dev = VirtioBlk::with_options(f, cap, DiskThrottle::default(), true);
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

    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(s[0], VIRTIO_BLK_S_OK as u8);

    let c = dev.counters();
    assert_eq!(c.flushes_completed.load(Ordering::Relaxed), 1);
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 0);
    assert_eq!(c.throttled_count.load(Ordering::Relaxed), 0);
}

/// Multi-byte status descriptor. Status byte goes at
/// the LAST byte of the descriptor (`addr + len - 1`) so the
/// kernel driver's `virtio_blk_outhdr` lookup lines up
/// regardless of leading padding. The status_addr arithmetic
/// in drain_bracket_impl's chain-shape walk implements this;
/// pin the offset.
#[test]
fn multi_byte_status_writes_to_last_byte() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
    // Plant a 4-byte sentinel at status_addr.
    mem.write_slice(&[0xAA, 0xBB, 0xCC, 0xDD], status_addr)
        .unwrap();
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
            4, // multi-byte status
            VRING_DESC_F_WRITE as u16,
            0,
        )),
    ];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_device_to_mock(&mut dev, &mock);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    let mut buf = [0u8; 4];
    mem.read_slice(&mut buf, status_addr).unwrap();
    assert_eq!(
        buf[0], 0xAA,
        "first byte of multi-byte status must be untouched"
    );
    assert_eq!(buf[1], 0xBB);
    assert_eq!(buf[2], 0xCC);
    assert_eq!(
        buf[3], VIRTIO_BLK_S_OK as u8,
        "status byte must be at the LAST byte (offset len-1)",
    );
}

/// Zero-capacity device. Any read with non-zero data
/// length must IOERR (`base_offset + total_data > 0`).
#[test]
fn zero_capacity_read_returns_ioerr() {
    let cap = 0u64;
    let f = tempfile().unwrap();
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
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
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(s[0], VIRTIO_BLK_S_IOERR as u8);

    let c = dev.counters();
    assert!(c.io_errors.load(Ordering::Relaxed) >= 1);
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
}

/// Partial-data + zero-pad read. Backing file 100 bytes,
/// device capacity 8 sectors, read 1 sector starting at sector
/// 0. Bytes 0..100 = file pattern, bytes 100..512 = zero-pad.
#[test]
fn partial_data_read_zero_pads_remainder() {
    let cap = 4096u64;
    let mut f = tempfile().unwrap();
    f.set_len(100).unwrap();
    f.write_all(&[0xA5; 100]).unwrap();
    f.rewind().unwrap();
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    // Pre-fill data buffer with sentinel — must not survive.
    let pre = vec![0xFFu8; 512];
    mem.write_slice(&pre, data_addr).unwrap();
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
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(s[0], VIRTIO_BLK_S_OK as u8);

    let mut buf = [0u8; 512];
    mem.read_slice(&mut buf, data_addr).unwrap();
    assert!(
        buf[..100].iter().all(|&b| b == 0xA5),
        "first 100 bytes must match backing file pattern",
    );
    assert!(
        buf[100..].iter().all(|&b| b == 0),
        "bytes 100..512 must be zero-padded",
    );
}

/// sector=u64::MAX overflow. `checked_mul` catches and
/// rejects with IOERR. Without the check, the wraparound
/// would silently land at a low offset.
#[test]
fn write_sector_overflow_returns_ioerr() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_OUT, u64::MAX);
    let payload = vec![0xCDu8; 512];
    mem.write_slice(&payload, data_addr).unwrap();
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

    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(s[0], VIRTIO_BLK_S_IOERR as u8);

    let c = dev.counters();
    assert!(c.io_errors.load(Ordering::Relaxed) >= 1);
    assert_eq!(c.writes_completed.load(Ordering::Relaxed), 0);
}

/// Flush sync_data baseline. Forcing a real sync_data
/// failure from a passing test fixture requires a closed fd
/// or fault injection (libfiu, fioz). Best-effort coverage:
/// pin the OK path increments flushes_completed and not
/// io_errors. The Err arm is reviewed by code inspection
/// (handle_flush_impl writes VIRTIO_BLK_S_IOERR on Err).
#[test]
fn flush_sync_data_baseline_ok_path() {
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

    let c = dev.counters();
    assert_eq!(c.flushes_completed.load(Ordering::Relaxed), 1);
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 0);
}

/// Validation-before-consumption invariant. Drain the
/// throttle to 0, submit a sub-sector chain (G5 gate). Pin:
/// io_errors ticks (gate fires) but throttled_count stays 0
/// (gate is pre-throttle; tokens NOT consumed).
#[test]
fn validation_gates_do_not_consume_throttle_tokens() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let throttle = DiskThrottle {
        iops: std::num::NonZeroU64::new(1),
        bytes_per_sec: None,
        iops_burst_capacity: None,
        bytes_burst_capacity: None,
    };
    let mut dev = VirtioBlk::new(f, cap, throttle);
    // Drain the bucket and pin its last_refill so refill on
    // the next consume yields 0 tokens.
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
            513, // sub-sector → G5 gate fires
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

    let c = dev.counters();
    assert_eq!(
        c.io_errors.load(Ordering::Relaxed),
        1,
        "sub-sector gate must bump io_errors",
    );
    assert_eq!(
        c.throttled_count.load(Ordering::Relaxed),
        0,
        "validation gate must NOT consume throttle tokens; \
             throttled_count must stay at 0 even with bucket drained",
    );
}

/// RO-flush vs normal-flush counter symmetry. Both
/// paths must increment flushes_completed by exactly 1.
#[test]
fn ro_flush_and_normal_flush_both_increment_counter() {
    // Normal flush.
    {
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
        assert_eq!(
            dev.counters().flushes_completed.load(Ordering::Relaxed),
            1,
            "normal flush must increment flushes_completed",
        );
    }
    // RO flush.
    {
        let cap = 4096u64;
        let f = make_backed_file_with_pattern(cap, 0x00);
        let mut dev = VirtioBlk::with_options(f, cap, DiskThrottle::default(), true);
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
        assert_eq!(
            dev.counters().flushes_completed.load(Ordering::Relaxed),
            1,
            "RO flush must increment flushes_completed (counter symmetry)",
        );
    }
}

/// Legacy-path irqfd delivery through a full chain. Before
/// process_requests, irq_evt is unsignalled (read returns
/// EAGAIN). After QUEUE_NOTIFY drains the chain, the post-drain
/// V8-split logic inlined in `process_requests`
/// (interrupt_status bit + needs_notification-gated eventfd
/// write) MUST leave irq_evt readable on the legacy path
/// because `Queue::needs_notification` returns Ok(true)
/// unconditionally when EVENT_IDX is not negotiated. This
/// pins the KVM irqfd delivery contract.
#[test]
fn process_requests_fires_irqfd_on_legacy_path() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
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
    // Before notify: irq_evt must NOT be readable.
    assert!(
        dev.irq_evt.read().is_err(),
        "before process_requests, irq_evt must not be signalled",
    );
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    let val = dev
        .irq_evt
        .read()
        .expect("irq_evt must be readable after notify");
    // Production fires `irq_evt.write(1)` exactly once per
    // signalling drain (process_requests post-drain branch).
    // A loose `> 0` would let a regression that fired the
    // eventfd twice slip through; pin the exact count.
    assert_eq!(
        val, 1,
        "irq_evt counter must be exactly 1 after a single chain drain"
    );
    assert_ne!(
        dev.interrupt_status.load(Ordering::Acquire) & VIRTIO_MMIO_INT_VRING,
        0
    );
}

/// EVENT_IDX path: when the guest's `used_event` threshold has
/// not been crossed by `next_used`, the device must NOT write
/// the irqfd, even though it advanced the used ring.
/// `Queue::needs_notification` returns false in that window —
/// its `event_idx_enabled` arm runs the
/// `used_idx - used_event - 1 < used_idx - old`
/// wrapping-arithmetic test, which is false when `used_event`
/// is well above `next_used`.
/// The `interrupt_status` bit must still be set so the guest's
/// MMIO read sees pending work — the V8 split between bit and
/// eventfd lets the guest poll without losing context if it
/// happens to read INTERRUPT_STATUS while suppressed.
#[test]
fn event_idx_suppresses_irqfd_when_threshold_unreached() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let qsize = 16u16;
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), qsize);
    // Plant `used_event = u16::MAX` BEFORE wiring the device:
    // the guest writes this before the first QUEUE_NOTIFY in
    // real life, and `Queue::needs_notification` reads it
    // every time it's called.
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
    // used_override: place the used ring at 0x10000, well above
    // the avail ring's used_event field at avail_addr + 36. The
    // mock's default used_addr collides with used_event; see
    // `wire_device_to_mock_with_event_idx` doc comment.
    wire_device_to_mock_with_event_idx(&mut dev, &mock, qsize, GuestAddress(0x10000));
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // The chain landed: status byte and counter ticked.
    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(s[0], VIRTIO_BLK_S_OK as u8);
    assert_eq!(dev.counters().reads_completed.load(Ordering::Relaxed), 1,);
    // V8: interrupt_status bit IS set even when irqfd is
    // suppressed. The guest reads INTERRUPT_STATUS during its
    // ISR (or polling); seeing the bit lets it know there's
    // work even if no IRQ delivered.
    assert_ne!(
        dev.interrupt_status.load(Ordering::Acquire) & VIRTIO_MMIO_INT_VRING,
        0,
        "interrupt_status bit must be set when chain published",
    );
    // T-GAP-F: same bit observable through the MMIO surface
    // (`read_reg` → `mmio_read` → `interrupt_status` value at
    // VIRTIO_MMIO_INTERRUPT_STATUS). VIRTIO_MMIO_INT_VRING is
    // bit 1 (vring buffer interrupt). Pins that the guest's
    // ISR/polling path sees the bit through the published
    // MMIO contract, not just via the host-internal field.
    let status = read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS);
    assert_eq!(status & 1, 1);
    // irqfd MUST be unsignalled — read returns EAGAIN
    // (counter is 0, eventfd in counter mode blocks/EAGAINs
    // on read of zero-value).
    assert!(
        dev.irq_evt.read().is_err(),
        "irq_evt must be unsignalled when used_event threshold not crossed",
    );
}

/// EVENT_IDX path: when the guest's `used_event` threshold IS
/// crossed (e.g. used_event = 0 and we publish a chain causing
/// next_used = 1), the device fires the irqfd. This is the
/// common case for the first request after the guest sets up
/// the queue.
#[test]
fn event_idx_fires_irqfd_when_threshold_reached() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let qsize = 16u16;
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), qsize);
    // used_event = 0: the guest is asking to be notified as
    // soon as next_used reaches 1. After one chain
    // completion, `needs_notification` returns true.
    let used_event = used_event_addr(mock.avail_addr(), qsize);
    mem.write_obj::<u16>(u16::to_le(0), used_event)
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
    // used_override: place the used ring at 0x10000, well above
    // the avail ring's used_event field at avail_addr + 36. The
    // mock's default used_addr collides with used_event; see
    // `wire_device_to_mock_with_event_idx` doc comment.
    wire_device_to_mock_with_event_idx(&mut dev, &mock, qsize, GuestAddress(0x10000));
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // irqfd fired exactly once (counter mode: a single write(1)
    // produces read returning 1).
    let val = dev
        .irq_evt
        .read()
        .expect("irq_evt must be readable when threshold reached");
    assert_eq!(
        val, 1,
        "irq_evt counter must be exactly 1 after a single chain completion",
    );
    assert_ne!(
        dev.interrupt_status.load(Ordering::Acquire) & VIRTIO_MMIO_INT_VRING,
        0
    );
}

/// EVENT_IDX path: a multi-chain drain consults
/// `needs_notification` exactly once at the END of the drain
/// (V6: only call on the signal_needed=true path), so the
/// irqfd fires at most ONCE regardless of chain count. This
/// is the IRQ-coalescing benefit of EVENT_IDX — without it
/// the legacy path would fire once per drain anyway, but
/// with EVENT_IDX the fire decision is held until the drain
/// completes so `needs_notification` sees the final
/// `next_used` value (`num_added` reflects all 3 chains).
#[test]
fn event_idx_multi_chain_drain_fires_once() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let qsize = 16u16;
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), qsize);
    // used_event = 0: notify when next_used reaches 1.
    let used_event = used_event_addr(mock.avail_addr(), qsize);
    mem.write_obj::<u16>(u16::to_le(0), used_event)
        .expect("plant used_event");
    // Build 3 chains, each its own header/data/status triple.
    for i in 0..3u64 {
        let header_addr = GuestAddress(0x4000 + i * 0x1000);
        let data_addr = GuestAddress(0x8000 + i * 0x1000);
        let status_addr = GuestAddress(0xC000 + i * 0x100);
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
    }
    dev.set_mem(mem.clone());
    // used_override: place the used ring at 0x10000, well above
    // the avail ring's used_event field at avail_addr + 36. The
    // mock's default used_addr collides with used_event; see
    // `wire_device_to_mock_with_event_idx` doc comment.
    wire_device_to_mock_with_event_idx(&mut dev, &mock, qsize, GuestAddress(0x10000));
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // 3 chains completed.
    assert_eq!(dev.counters().reads_completed.load(Ordering::Relaxed), 3,);
    // irqfd fired exactly once. EventFd in counter mode: one
    // write(1) → read returns 1; three writes → read returns
    // 3. The post-drain gate produces a single write, so
    // read must return 1.
    let val = dev
        .irq_evt
        .read()
        .expect("irq_evt must be readable after multi-chain drain");
    assert_eq!(
        val, 1,
        "irq_evt must fire exactly once for a multi-chain drain \
             (V6: needs_notification consulted once at end of drain)",
    );
}

/// EVENT_IDX path, multi-chain drain, threshold above the
/// post-drain `next_used` value: 3 chains complete but
/// `needs_notification` returns false because `used_event = 10`
/// (the guest is asking to be notified only once `next_used`
/// crosses 10). Pins suppression under multi-chain load — a
/// regression that fired the irqfd once per chain (or once per
/// drain regardless of threshold) would surface as a non-zero
/// `irq_evt.read()` here. Companion to
/// `event_idx_multi_chain_drain_fires_once` (used_event=0,
/// expected fire) — together the pair pin both halves of the
/// gate at multi-chain load.
#[test]
fn event_idx_multi_chain_drain_suppresses_below_threshold() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let qsize = 16u16;
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), qsize);
    // used_event = 10: the guest is asking for notification only
    // once next_used crosses 10. We're going to drain 3 chains
    // (next_used → 3) so the threshold is unreached and the
    // post-drain `needs_notification` returns false. Plant
    // BEFORE wiring the device per the existing EVENT_IDX
    // pattern (Queue reads used_event lazily on each
    // needs_notification call).
    let used_event = used_event_addr(mock.avail_addr(), qsize);
    mem.write_obj::<u16>(u16::to_le(10), used_event)
        .expect("plant used_event");
    // Build 3 read chains with disjoint addresses so the
    // descriptor table doesn't alias across iterations.
    for i in 0..3u64 {
        let header_addr = GuestAddress(0x4000 + i * 0x1000);
        let data_addr = GuestAddress(0x8000 + i * 0x1000);
        let status_addr = GuestAddress(0xC000 + i * 0x100);
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
    }
    dev.set_mem(mem.clone());
    // used_override: place the used ring at 0x10000, well above
    // the avail ring's used_event field at avail_addr + 36. The
    // mock's default used_addr collides with used_event; see
    // `wire_device_to_mock_with_event_idx` doc comment.
    wire_device_to_mock_with_event_idx(&mut dev, &mock, qsize, GuestAddress(0x10000));
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // All 3 chains landed.
    assert_eq!(
        dev.counters().reads_completed.load(Ordering::Relaxed),
        3,
        "all 3 chains must complete in the single QUEUE_NOTIFY drain",
    );
    // Used ring (at the override addr 0x10000) reflects exactly
    // 3 completions. Reads u16 used.idx at offset 2 of the
    // override addr.
    let used_idx: u16 = mem
        .read_obj(GuestAddress(0x10000).checked_add(2).unwrap())
        .expect("read device used.idx at override addr");
    assert_eq!(
        used_idx, 3,
        "exactly three used-ring entries expected after 3-chain drain",
    );
    // V8: interrupt_status bit IS set even when irqfd is
    // suppressed. The guest's ISR or polling path reads
    // INTERRUPT_STATUS to find work; the bit must be visible
    // independent of the irqfd gate.
    assert_ne!(
        dev.interrupt_status.load(Ordering::Acquire) & VIRTIO_MMIO_INT_VRING,
        0,
        "interrupt_status bit must be set after 3 completions \
             even when irqfd suppressed",
    );
    // T-GAP-F: same bit observable through MMIO surface.
    let status = read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS);
    assert_eq!(status & 1, 1);
    // irqfd MUST be unsignalled — `needs_notification` saw
    // next_used=3 < used_event=10 so the gate held.
    assert!(
        dev.irq_evt.read().is_err(),
        "irq_evt must be unsignalled when post-drain next_used \
             stays below used_event threshold",
    );
}

/// Legacy path (EVENT_IDX not negotiated):
/// `Queue::needs_notification` always returns Ok(true) (the
/// trailing `Ok(true)` after the `event_idx_enabled` branch),
/// so every drain that publishes any chain fires the irqfd.
/// This test pins the legacy contract — a regression that
/// gated the irqfd write on the wrong path would silently
/// break the legacy guest's IRQ delivery.
#[test]
fn legacy_path_fires_irqfd_every_drain() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let qsize = 16u16;
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), qsize);
    // Plant used_event = u16::MAX. In the EVENT_IDX path this
    // would suppress; in the legacy path it's IGNORED — proves
    // the test exercises the legacy path.
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
    // Legacy path: VIRTIO_RING_F_EVENT_IDX NOT negotiated.
    wire_device_to_mock(&mut dev, &mock);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // irqfd fired despite used_event=u16::MAX, because the
    // legacy path ignores the threshold.
    let val = dev
        .irq_evt
        .read()
        .expect("irq_evt must be readable on legacy path");
    assert_eq!(
        val, 1,
        "legacy path must fire irq_evt unconditionally — used_event \
             is irrelevant when EVENT_IDX is not negotiated",
    );
    // Symmetry with EVENT_IDX suppressed-irqfd tests: pin that
    // INTERRUPT_STATUS reflects the bit on the legacy fire path
    // too. Production sets the bit unconditionally on any
    // signalling drain (process_requests post-drain branch),
    // independent of which path drives the irqfd gate.
    let status = read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS);
    assert_eq!(status & 1, 1);
}

/// Outer-loop bracket: when 2 chains are queued before
/// QUEUE_NOTIFY, both complete in a single `process_requests`
/// call. This is a deterministic variant of the re-drain
/// coverage — see the doc note below for why the
/// `enable_notification → Ok(true) → continue 'outer` arm
/// itself can't be tested deterministically from a single
/// thread.
///
/// Race-window note: the production re-drain arm fires when
/// `avail_idx != next_avail` AT the moment `enable_notification`
/// runs. In a real VMM, that gap exists between the inner-loop
/// break (next_avail caught up to the avail_idx the device saw)
/// and the `set_avail_event` call inside `enable_notification`
/// — a vCPU can write a fresh chain head and bump `avail_idx`
/// in that window. In a single-threaded test there is no such
/// vCPU; `MockSplitQueue` is the only writer and we control
/// when it writes. To trigger Ok(true) deterministically would
/// require interposing on `enable_notification` itself
/// (e.g. a test-only `Queue` implementation) — too invasive
/// for the value gained. The deterministic variant here pins
/// the WEAKER property: 2 chains queued before notify both
/// complete in one process_requests call. The actual re-drain
/// arm is exercised by the existing
/// `event_idx_multi_chain_drain_fires_once` test which queues
/// 3 chains; both tests share the same single-process-requests
/// shape.
///
/// What this DOES guarantee: a 2-chain pre-notify queue drains
/// fully in one call. A regression that prematurely broke out
/// of the outer loop after the first chain (e.g. dropping
/// `continue 'outer` in favour of `break 'outer`) would leave
/// the second chain unprocessed — that regression IS caught
/// here even though the path through the Ok(true) arm itself
/// isn't directly observed.
#[test]
fn outer_loop_drains_two_pre_queued_chains_in_one_call() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let qsize = 16u16;
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), qsize);
    // used_event = 0: notify on first completion. After both
    // chains are processed the post-drain `needs_notification`
    // observes next_used=2, num_added=2, threshold-crossed →
    // fires. Either Ok(true)→Ok(false) (re-drain path) OR
    // Ok(false) directly leaves both chains processed and
    // signal_needed=true.
    let used_event = used_event_addr(mock.avail_addr(), qsize);
    mem.write_obj::<u16>(u16::to_le(0), used_event)
        .expect("plant used_event");
    let header_addr = GuestAddress(0x4000);
    let status_addr = GuestAddress(0x4100);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_FLUSH, 0);
    // Two FLUSH chains pre-queued. FLUSH carries no data
    // (header + status only — virtio-v1.2 §5.2.6.3). Both
    // chains share the same desc_table slots because
    // `MockSplitQueue::add_desc_chains` writes at offset 0
    // each call; the second build_desc_chain overwrites
    // descriptors 0..1 but the avail_ring grows by one each
    // call — so 2 chain heads point at desc_table[0] and the
    // device walks the same descriptors twice. fdatasync on a
    // tempfile is idempotent.
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
    mock.build_desc_chain(&descs).expect("build chain 1");
    mock.build_desc_chain(&descs).expect("build chain 2");
    dev.set_mem(mem.clone());
    wire_device_to_mock_with_event_idx(&mut dev, &mock, qsize, GuestAddress(0x10000));
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // Both chains completed. The bracket pattern guarantees
    // EITHER (a) inner loop drains both immediately and
    // enable_notification returns Ok(false) → break, OR (b)
    // inner drains chain 1, enable_notification returns Ok(true)
    // because chain 2's avail-idx advance was visible after the
    // bracket close → continue 'outer drains chain 2. Both end
    // states are observable as flushes_completed == 2.
    let c = dev.counters();
    assert_eq!(
        c.flushes_completed.load(Ordering::Relaxed),
        2,
        "both pre-queued FLUSH chains must complete in a single \
             process_requests call",
    );
    // Used ring (placed at the override addr 0x10000) reflects
    // exactly two completions. The mock's default used ring is
    // unused; read used.idx from the override location.
    let used_idx: u16 = mem
        .read_obj(GuestAddress(0x10000).checked_add(2).unwrap())
        .expect("read device used.idx at override addr");
    assert_eq!(
        used_idx, 2,
        "exactly two used-ring entries expected after two-chain drain",
    );
    // Single irqfd fire: V6 has needs_notification consulted
    // once at end of drain. Whether the path went through the
    // re-drain arm or broke out directly, the tail signal is
    // ONE eventfd write.
    let val = dev
        .irq_evt
        .read()
        .expect("irq_evt readable after two-chain drain");
    assert_eq!(
        val, 1,
        "exactly one irq_evt write expected — needs_notification \
             consulted once after the drain settles",
    );
}

/// Bail-out branch: when `enable_notification` returns Err
/// (the `set_avail_event` write to the used ring's
/// `avail_event` field hits unmapped guest memory), the outer
/// loop must break cleanly without hanging, the chain that
/// was already published before the failure stays published
/// (`add_used` succeeded; the failure is in the post-drain
/// notification arming), and the irqfd fires fail-safe via
/// the `unwrap_or(true)` on the post-drain `needs_notification`
/// call.
///
/// Test setup: a multi-region GuestMemoryMmap with a hole
/// straddling the device's `avail_event` GPA. The used ring is
/// placed via `used_override_addr` so its body
/// (header + ring elements at offsets 0..132) lives in the
/// first region and the trailing `avail_event` u16 at
/// `used_addr + 132` lands at the boundary, in the unmapped
/// gap. add_used (offsets 4..12 for index 0) succeeds;
/// `set_avail_event` writing 2 bytes at `used_addr + 132`
/// fails with InvalidGuestAddress.
///
/// Layout: `Queue::set_avail_event` writes at
/// `used_ring + VIRTQ_USED_RING_HEADER_SIZE
/// + VIRTQ_USED_ELEMENT_SIZE * size = used_ring + 4 + 8 * 16 =
/// used_ring + 132`.
#[test]
fn enable_notification_err_breaks_outer_and_fires_irqfd_fail_safe() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    // Multi-region mem: [0, 0x20000) and [0x30000, 0x40000).
    // The hole is [0x20000, 0x30000). With used_addr=0x1FF7C
    // and size=16: avail_event is at 0x20000 (start of the
    // hole), inaccessible. add_used at next_used=0 writes 8
    // bytes to 0x1FF80..0x1FF88 (in-range) plus the 2-byte
    // next_used u16 to 0x1FF7E (in-range).
    let mem = GuestMemoryMmap::from_ranges(&[
        (GuestAddress(0), 0x20000),
        (GuestAddress(0x30000), 0x10000),
    ])
    .expect("create multi-region guest mem");
    let qsize = 16u16;
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), qsize);
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
    // used_override = 0x1FF7C: with size=16 the used-ring body
    // (header + 16 * 8-byte elements = 132 bytes) ends exactly
    // at 0x20000 (the boundary), and the trailing avail_event
    // u16 store at 0x20000..0x20002 lies in the unmapped hole.
    wire_device_to_mock_with_event_idx(&mut dev, &mock, qsize, GuestAddress(0x1FF7C));

    // Pre-notify: irqfd MUST be unsignalled.
    assert!(
        dev.irq_evt.read().is_err(),
        "irq_evt must not be signalled before notify",
    );

    // Fire QUEUE_NOTIFY. Inner drain processes the chain
    // (add_used succeeds at offsets in the mapped region),
    // enable_notification returns Err on the unmapped
    // avail_event store, the outer loop breaks cleanly. If
    // the bail were missing (infinite outer loop on persistent
    // err), this call would hang and the test would time out.
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // Chain was published before the bail. flushes_completed
    // ticked, used.idx advanced to 1.
    let c = dev.counters();
    assert_eq!(
        c.flushes_completed.load(Ordering::Relaxed),
        1,
        "FLUSH must complete before the enable_notification bail",
    );
    let used_idx: u16 = mem
        .read_obj(GuestAddress(0x1FF7C).checked_add(2).unwrap())
        .expect("read device used.idx at override addr");
    assert_eq!(
        used_idx, 1,
        "add_used must have run before the enable_notification bail",
    );

    // V8 + fail-safe: the irqfd MUST fire. The post-drain
    // `needs_notification` reads `used_event` from the avail
    // ring (in the mapped region — only the USED ring's
    // `avail_event` is in the hole), so the call returns
    // Ok(true|false) cleanly. With used_event=0 (default mock
    // initialisation, mock.rs:151) and next_used=1, the
    // formula returns true → fire. Even if it returned an
    // Err, `unwrap_or(true)` would still fire fail-safe.
    let val = dev
        .irq_evt
        .read()
        .expect("irq_evt must fire fail-safe after enable_notification bail");
    assert_eq!(
        val, 1,
        "irq_evt must fire exactly once after the bail (V8 \
             interrupt_status bit + needs_notification gate)",
    );
    assert_ne!(
        dev.interrupt_status.load(Ordering::Acquire) & VIRTIO_MMIO_INT_VRING,
        0,
        "interrupt_status bit must be set when chain published, \
             independent of the enable_notification bail",
    );
}

/// Companion to `enable_notification_err_breaks_outer_and_fires_irqfd_fail_safe`:
/// pins the OTHER `enable_notification` call site, the
/// post-stall arm. When the chain stalls on throttle exhaustion,
/// the inner pop loop breaks WITHOUT publishing
/// (`signal_needed` stays false), the outer-loop stall arm calls
/// `enable_notification` to re-arm guest-side wakeups, and on
/// Err logs a warn and breaks 'outer cleanly. Distinct from the
/// Done-path enable_notification (covered above) because the
/// stall path skips the post-drain `signal_needed` block — no
/// `interrupt_status` bit set, no irqfd write, no `add_used`.
///
/// Setup mirrors the Done-path test for guest-memory layout —
/// multi-region GuestMemoryMmap with a hole at the
/// `avail_event` GPA, used ring placed via `used_override_addr`
/// so its trailing 2-byte `avail_event` write lands in the
/// unmapped hole — but adds a drained 1-iops throttle so the
/// chain stalls instead of completing. The single chain pops,
/// the throttle gate fails, the stall path calls
/// `enable_notification` whose `set_avail_event` write hits
/// the hole and returns InvalidGuestAddress.
///
/// Stall-path invariants:
///
///   - `throttled_count` == 1 — the stall event was recorded.
///   - `currently_throttled_gauge` == 1 — the false→true
///     transition fired (per the gauge transition table).
///   - `state.currently_stalled` == true — the head is pinned
///     in the avail ring awaiting refill.
///   - used.idx == 0 (no add_used).
///   - irq_evt unsignalled — `signal_needed` stayed false, so
///     the post-drain V8 block was not entered.
///   - interrupt_status MMIO bit clear (same reason).
///   - status sentinel survives — no publish_completion ran.
///   - Queue cursor rewound to 0 (set_next_avail rolled the
///     pop back so the chain re-pops on retry).
///
/// A regression that propagated the enable_notification Err
/// instead of swallowing-and-breaking would either re-enter the
/// outer loop (livelock) or fail to record the stall counter —
/// both observable via the assertions below.
#[test]
fn enable_notification_err_on_stall_path_breaks_outer_cleanly() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let throttle = DiskThrottle {
        iops: std::num::NonZeroU64::new(1),
        bytes_per_sec: None,
        iops_burst_capacity: None,
        bytes_burst_capacity: None,
    };
    let mut dev = VirtioBlk::new(f, cap, throttle);

    // Multi-region mem: [0, 0x20000) and [0x30000, 0x40000).
    // The hole [0x20000, 0x30000) covers the avail_event GPA.
    // Same layout as the Done-path test so the
    // `avail_event = used_addr + 132` (size=16) calculation
    // produces 0x20000 — the boundary, in the hole.
    let mem = GuestMemoryMmap::from_ranges(&[
        (GuestAddress(0), 0x20000),
        (GuestAddress(0x30000), 0x10000),
    ])
    .expect("create multi-region guest mem");
    let qsize = 16u16;
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), qsize);

    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    // Plant a sentinel at the status byte — survival of this
    // byte through the stall is the key invariant (no
    // publish_completion ran).
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
    // used_override = 0x1FF7C: with size=16, the used-ring body
    // (4-byte header + 16 * 8-byte elements = 132 bytes) ends
    // at exactly 0x20000, and the trailing avail_event u16
    // store at 0x20000..0x20002 lies in the unmapped hole.
    // Same address as the Done-path test by design.
    wire_device_to_mock_with_event_idx(&mut dev, &mock, qsize, GuestAddress(0x1FF7C));

    // Drain the iops bucket so the chain stalls. With iops=1
    // and capacity=1, a single consume(1) takes the only token;
    // pin last_refill so the next can_consume sees an empty
    // bucket (no passive wall-clock refill in microseconds
    // between this setup and the QUEUE_NOTIFY).
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
    assert!(
        !dev.worker.state_mut().ops_bucket.can_consume(1),
        "precondition: ops bucket must be drained so the chain stalls",
    );

    // Pre-notify: every observable surface is at its baseline.
    let c = dev.counters();
    assert_eq!(c.throttled_count.load(Ordering::Relaxed), 0);
    assert_eq!(c.currently_throttled_gauge.load(Ordering::Relaxed), 0);
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 0);
    assert!(!dev.worker.state().currently_stalled);
    assert!(
        dev.irq_evt.read().is_err(),
        "irq_evt must be unsignalled before notify",
    );

    // Fire QUEUE_NOTIFY. Inner pop returns the chain, throttle
    // gate fails, stall_outcome = Some(_), break inner. Outer
    // stall arm calls enable_notification → Err on the unmapped
    // avail_event store → log warn, break 'outer. No
    // publish_completion ran; signal_needed stayed false; the
    // post-drain V8 block did not fire. If the bail were
    // missing (continued outer loop on persistent Err), this
    // call would hang.
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // Status sentinel survives — no publish_completion ran.
    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(
        s[0], 0xEE,
        "status byte must remain at sentinel — stall must not write status",
    );

    // used.idx unchanged at 0 — no add_used.
    let used_idx: u16 = mem
        .read_obj(GuestAddress(0x1FF7C).checked_add(2).unwrap())
        .expect("read device used.idx at override addr");
    assert_eq!(used_idx, 0, "used.idx must be 0 — stall must skip add_used",);

    // Stall counters: event recorded, gauge incremented on
    // false→true, reads_completed untouched.
    assert_eq!(
        c.throttled_count.load(Ordering::Relaxed),
        1,
        "stall event must be recorded once",
    );
    assert_eq!(
        c.currently_throttled_gauge.load(Ordering::Relaxed),
        1,
        "gauge must increment on the false→true transition",
    );
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 0);
    assert!(
        dev.worker.state().currently_stalled,
        "currently_stalled flag must be true post-stall",
    );

    // V8 post-drain block did not run — signal_needed stayed
    // false. interrupt_status bit clear, irqfd unsignalled.
    assert_eq!(
        dev.interrupt_status.load(Ordering::Acquire) & VIRTIO_MMIO_INT_VRING,
        0,
        "interrupt_status bit must be clear — stall does not \
             enter the V8 post-drain block",
    );
    assert!(
        dev.irq_evt.read().is_err(),
        "irq_evt must be unsignalled — stall does not fire irqfd",
    );

    // Queue cursor rewound: stall path runs
    // `set_next_avail(prev.wrapping_sub(1))` so the next pop
    // returns the same head. After one pop+rewind on a queue
    // with one chain, next_avail is back at 0.
    assert_eq!(
        dev.worker.queues[REQ_QUEUE].next_avail(),
        0,
        "queue cursor must be rewound to 0 — set_next_avail \
             rolled the pop back so the chain re-pops on retry",
    );
}

/// Pin the u16-wrap arithmetic the throttle-stall rollback
/// depends on: `set_next_avail(prev.wrapping_sub(1))` at
/// `prev = 0` MUST land at `u16::MAX`, not panic via signed
/// underflow. The companion proptest
/// `throttle_stall_under_random_chain_shapes_holds_invariants`
/// runs at `next_avail = 0 → 1 → 0` (no wrap exercised
/// because `MockSplitQueue::build_desc_chain` only supports
/// fresh avail rings); this dedicated unit test pins the
/// wrap edge by directly calling `set_next_avail` on the
/// queue and asserting the contract `wrapping_sub` provides.
///
/// A regression that swapped `wrapping_sub` for plain `-` or
/// `checked_sub().unwrap()` would panic on this test
/// instead of silently corrupting the cursor in production.
/// `wrapping_sub` matches the virtio ring's u16 wrap
/// semantics (avail/used cursors are u16 modular per
/// virtio-v1.2 §2.7).
#[test]
fn next_avail_zero_rollback_wraps_to_u16_max() {
    let mut dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
    // Drive next_avail to 0 explicitly so the sub-1 wrap is
    // observable via wrapping arithmetic (not a "happened to
    // be at 0" coincidence).
    dev.worker.queues[REQ_QUEUE].set_next_avail(0);
    assert_eq!(dev.worker.queues[REQ_QUEUE].next_avail(), 0);

    // The exact arithmetic the production stall path uses:
    // `set_next_avail(prev.wrapping_sub(1))`.
    let prev = dev.worker.queues[REQ_QUEUE].next_avail();
    dev.worker.queues[REQ_QUEUE].set_next_avail(prev.wrapping_sub(1));

    assert_eq!(
        dev.worker.queues[REQ_QUEUE].next_avail(),
        u16::MAX,
        "next_avail rollback at prev=0 must wrap to u16::MAX, \
             matching the virtio ring's u16 modular semantics",
    );

    // Wrap-back: another wrapping_sub from u16::MAX lands at
    // u16::MAX - 1, no panic. Pins the arithmetic in both
    // directions so a regression that handled the 0→u16::MAX
    // case but broke u16::MAX→u16::MAX-1 surfaces here.
    let prev = dev.worker.queues[REQ_QUEUE].next_avail();
    dev.worker.queues[REQ_QUEUE].set_next_avail(prev.wrapping_sub(1));
    assert_eq!(
        dev.worker.queues[REQ_QUEUE].next_avail(),
        u16::MAX - 1,
        "subsequent rollback at prev=u16::MAX must land at u16::MAX-1",
    );
}

/// Fragmented header. The first descriptor is shorter
/// than VIRTIO_BLK_OUTHDR_SIZE — the device cannot read a
/// full header from desc[0] and must reject. Chain layout:
/// [8-byte-RO, 8-byte-RO, status] — the second descriptor's
/// 8 bytes do NOT count toward the header (per virtio_blk.rs's
/// "first_len < OUTHDR_SIZE" gate).
#[test]
fn fragmented_header_returns_ioerr() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_part1_addr = GuestAddress(0x4000);
    let header_part2_addr = GuestAddress(0x4008);
    let status_addr = GuestAddress(0x5000);
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_part1_addr.0,
            8, // SHORT
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(header_part2_addr.0, 8, 0, 0)),
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
        s[0], VIRTIO_BLK_S_IOERR as u8,
        "fragmented header (first desc < OUTHDR_SIZE) must IOERR",
    );

    let c = dev.counters();
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 1);
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
}

/// EVENT_IDX path with an error chain: the IOERR completion must
/// route through the SAME post-drain `needs_notification` gate
/// as success completions, so a guest that asks for suppression
/// (`used_event = u16::MAX`) does not get spuriously interrupted
/// by an error chain.
///
/// Setup mirrors `fragmented_header_returns_ioerr` (chain has a
/// short first descriptor of 8 bytes — less than
/// `VIRTIO_BLK_OUTHDR_SIZE` = 16 — so the device cannot read a
/// full header from desc[0] and rejects via
/// `publish_completion(..., VIRTIO_BLK_S_IOERR, ...)` at
/// `process_requests`'s "header missing/short" branch). The
/// publish_completion call returns true (status-byte write
/// succeeded, add_used succeeded), so `signal_needed = true` —
/// the chain reaches the post-drain notification arm.
///
/// With EVENT_IDX negotiated and `used_event = u16::MAX`, the
/// post-drain `needs_notification` returns false (next_used=1
/// nowhere near u16::MAX) so the irqfd MUST stay unsignalled.
/// `interrupt_status` is still set (the guest's ISR/polling
/// path needs to see there's work). Pins the contract that
/// error completions are NOT a special-case bypass of the
/// suppression gate.
#[test]
fn event_idx_error_chain_suppressed_when_threshold_unreached() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let qsize = 16u16;
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), qsize);
    // Plant `used_event = u16::MAX` BEFORE wiring the device:
    // the guest is asking to never be notified for any normal
    // post-drain `next_used` value (it polls instead). The
    // post-drain `needs_notification` reads `used_event`
    // lazily, so plant before notify, not before wire.
    let used_event = used_event_addr(mock.avail_addr(), qsize);
    mem.write_obj::<u16>(u16::to_le(u16::MAX), used_event)
        .expect("plant used_event");
    // Fragmented-header layout: desc[0] = 8 bytes (< OUTHDR_SIZE
    // = 16) → header rejected, IOERR published via
    // publish_completion. desc[1] is also 8 bytes RO so the
    // device cannot stitch a full header from desc[0]+desc[1]
    // (per the "first_len < OUTHDR_SIZE" gate). desc[2] is the
    // 1-byte writable status descriptor.
    let header_part1_addr = GuestAddress(0x4000);
    let header_part2_addr = GuestAddress(0x4008);
    let status_addr = GuestAddress(0x5000);
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_part1_addr.0,
            8, // SHORT — triggers IOERR via publish_completion
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(header_part2_addr.0, 8, 0, 0)),
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
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // The error chain landed: status=IOERR, io_errors=1.
    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(
        s[0], VIRTIO_BLK_S_IOERR as u8,
        "fragmented header must produce IOERR even on EVENT_IDX path",
    );
    let c = dev.counters();
    assert_eq!(
        c.io_errors.load(Ordering::Relaxed),
        1,
        "fragmented-header reject must bump io_errors exactly once",
    );
    // The chain WAS add_used'd: error completions reach the
    // post-drain gate via signal_needed=true. used.idx at the
    // override addr advances to 1.
    let used_idx: u16 = mem
        .read_obj(GuestAddress(0x10000).checked_add(2).unwrap())
        .expect("read device used.idx at override addr");
    assert_eq!(
        used_idx, 1,
        "error chain must still be add_used'd so the guest sees \
             the IOERR status — V8 + the publish_completion contract",
    );
    // V8: interrupt_status bit IS set on the error chain too —
    // the guest's polling path reads INTERRUPT_STATUS to learn
    // there's work, regardless of irqfd suppression.
    assert_ne!(
        dev.interrupt_status.load(Ordering::Acquire) & VIRTIO_MMIO_INT_VRING,
        0,
        "interrupt_status bit must be set after error chain \
             completes, independent of irqfd gate",
    );
    // T-GAP-F: same bit observable through MMIO surface.
    let status = read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS);
    assert_eq!(status & 1, 1);
    // The contract this test pins: irqfd suppressed for the
    // error chain because `used_event=u16::MAX` was unreached.
    // A regression that bypassed `needs_notification` for
    // error completions (e.g. firing the irqfd unconditionally
    // on signal_needed=true) would surface here.
    assert!(
        dev.irq_evt.read().is_err(),
        "irq_evt must be unsignalled — error completions route \
             through the same needs_notification gate as success \
             completions, and used_event=u16::MAX was unreached",
    );
}

/// SIZE_MAX advertised in config space. virtio-v1.2
/// §5.2.4: size_max field at config-space offset 0x08
/// (= MMIO offset 0x108) must hold the per-descriptor max
/// byte length. Without the correct value, the guest
/// driver may submit oversize descriptors.
#[test]
fn size_max_advertised_in_config_space() {
    let dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
    let mut buf = [0u8; 4];
    dev.mmio_read(0x100 + 0x08, &mut buf);
    assert_eq!(
        u32::from_le_bytes(buf),
        VIRTIO_BLK_SIZE_MAX,
        "config-space size_max must equal VIRTIO_BLK_SIZE_MAX (1 MB)",
    );
}
