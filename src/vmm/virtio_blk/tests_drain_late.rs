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
// T_GET_ID (virtio-v1.2 §5.2.6.4) coverage. The kernel driver's
// `virtblk_get_id` (drivers/block/virtio_blk.c) issues a single
// 20-byte request to populate `/sys/block/<dev>/serial`. Tests
// span the direct handler, classify_pre_throttle dispatch, and
// the full chain pipeline.
// ----------------------------------------------------------------

/// `T_GET_ID` is NOT a pre-throttle terminal classification; it
/// dispatches to `handle_get_id_impl`. Pin that
/// `classify_pre_throttle` returns `None` for both writable and
/// read-only disks (the metadata read is RO-safe).
#[test]
fn classify_get_id_returns_none_for_both_modes() {
    let counters = VirtioBlkCounters::default();
    assert_eq!(
        VirtioBlk::classify_pre_throttle(VIRTIO_BLK_T_GET_ID, false, &counters),
        None,
        "writable disk: T_GET_ID falls through to handler",
    );
    assert_eq!(
        VirtioBlk::classify_pre_throttle(VIRTIO_BLK_T_GET_ID, true, &counters),
        None,
        "read-only disk: T_GET_ID is metadata-read-only and \
             still falls through to handler",
    );
    assert_eq!(
        counters.io_errors.load(Ordering::Relaxed),
        0,
        "T_GET_ID classification never bumps io_errors",
    );
}

/// `handle_get_id_impl` writes the device serial into a
/// 20-byte device-writable data segment and returns
/// `(S_OK, VIRTIO_BLK_ID_BYTES + 1)`. The serial bytes must
/// equal `VIRTIO_BLK_SERIAL` exactly so the guest's
/// `/sys/block/<dev>/serial` reads back the same string.
#[test]
fn handle_get_id_writes_serial_and_returns_ok() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    // 16 KiB mem so status_addr=0x2000 is in-range (the
    // single-region GuestMemoryMmap exposes [0, len) — len=8192
    // would put 0x2000 at the exclusive upper bound and reject
    // write_slice).
    let mem = make_guest_mem(16384);
    let data_addr = GuestAddress(0x1000);
    let status_addr = GuestAddress(0x2000);
    // Pre-fill the data buffer with a sentinel so a regression
    // that wrote zero bytes (or the wrong number of bytes)
    // surfaces as residual sentinel rather than a silent
    // pass.
    mem.write_slice(&[0xCDu8; VIRTIO_BLK_ID_BYTES as usize], data_addr)
        .unwrap();
    let segs = vec![ChainDescriptor {
        addr: data_addr,
        len: VIRTIO_BLK_ID_BYTES,
        is_write_only: true,
    }];
    let (status, used) = dev.handle_get_id(&mem, &segs, status_addr);
    assert_eq!(status, VIRTIO_BLK_S_OK as u8);
    assert_eq!(
        used,
        VIRTIO_BLK_ID_BYTES + 1,
        "used_len = 20 data bytes + 1 status byte",
    );
    let mut buf = [0u8; VIRTIO_BLK_ID_BYTES as usize];
    mem.read_slice(&mut buf, data_addr).unwrap();
    assert_eq!(
        buf, VIRTIO_BLK_SERIAL,
        "data segment must hold the device serial verbatim",
    );
    // Status descriptor holds S_OK.
    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(s[0], VIRTIO_BLK_S_OK as u8);
}

/// A data buffer shorter than `VIRTIO_BLK_ID_BYTES` (20) is
/// rejected with `S_IOERR`. Matches firecracker /
/// cloud-hypervisor / libkrun. QEMU truncates here; we
/// deliberately diverge — a partial serial would silently
/// surface garbage in `/sys/block/<dev>/serial`. The kernel
/// driver always passes exactly 20 bytes
/// (`virtblk_get_id` → `blk_rq_map_kern(req, id_str,
/// VIRTIO_BLK_ID_BYTES, GFP_KERNEL)`), so the only producers
/// of sub-20 buffers are buggy or hostile.
#[test]
fn handle_get_id_rejects_short_buffer() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_guest_mem(16384);
    let data_addr = GuestAddress(0x1000);
    let status_addr = GuestAddress(0x2000);
    // 19 bytes — one short of the 20-byte minimum.
    let segs = vec![ChainDescriptor {
        addr: data_addr,
        len: VIRTIO_BLK_ID_BYTES - 1,
        is_write_only: true,
    }];
    let (status, used) = dev.handle_get_id(&mem, &segs, status_addr);
    assert_eq!(
        status, VIRTIO_BLK_S_IOERR as u8,
        "sub-20-byte buffer must IOERR, not truncate",
    );
    assert_eq!(used, 1, "IOERR used_len is 1 (status byte only)");
    assert_eq!(
        dev.counters().io_errors.load(Ordering::Relaxed),
        1,
        "short buffer rejection bumps io_errors",
    );
}

/// A device-readable data descriptor (direction violation) is
/// rejected. virtio-v1.2 §5.2.6.4 mandates the data SG be
/// device-writable for T_GET_ID. The outer
/// `direction_violation` gate in `process_requests` already
/// filters this; the handler-level check is defense-in-depth
/// for callers that bypass the gate.
#[test]
fn handle_get_id_rejects_readonly_data_segment() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_guest_mem(16384);
    let data_addr = GuestAddress(0x1000);
    let status_addr = GuestAddress(0x2000);
    let segs = vec![ChainDescriptor {
        addr: data_addr,
        len: VIRTIO_BLK_ID_BYTES,
        is_write_only: false, // wrong direction for GET_ID
    }];
    let (status, _) = dev.handle_get_id(&mem, &segs, status_addr);
    assert_eq!(status, VIRTIO_BLK_S_IOERR as u8);
    assert_eq!(dev.counters().io_errors.load(Ordering::Relaxed), 1);
}

/// Drive a full T_GET_ID chain through `process_requests` via
/// MockSplitQueue + QUEUE_NOTIFY. Verifies the request reaches
/// `handle_get_id_impl`, the 20-byte serial lands in the data
/// descriptor, the status byte is S_OK, and used.idx
/// advances. Mirrors the kernel's `virtblk_get_id` chain shape:
/// header (RO, 16B) + data (WO, 20B) + status (WO, 1B).
#[test]
fn process_requests_full_get_id_chain() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    // Pre-fill the data buffer so a regression that doesn't
    // write the serial leaves a detectable sentinel. 0xCD
    // is distinct from the serial bytes (ascii letters + NUL).
    mem.write_slice(&[0xCDu8; VIRTIO_BLK_ID_BYTES as usize], data_addr)
        .unwrap();
    // Plant the GET_ID header. Kernel driver sets sector=0
    // (`vbr->out_hdr.sector = 0;` in virtblk_get_id) — we
    // mirror that for fidelity.
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_GET_ID, 0);
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            VIRTIO_BLK_ID_BYTES,
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

    // Status byte landed S_OK.
    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(s[0], VIRTIO_BLK_S_OK as u8);

    // Data descriptor holds the device serial verbatim.
    let mut buf = [0u8; VIRTIO_BLK_ID_BYTES as usize];
    mem.read_slice(&mut buf, data_addr).unwrap();
    assert_eq!(
        buf, VIRTIO_BLK_SERIAL,
        "T_GET_ID chain must populate data segment with device serial",
    );

    // Used ring advanced by one.
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 1);

    // io_errors stays 0 — the request completed cleanly.
    let c = dev.counters();
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 0);
    // reads/writes/flushes counters all stay at 0 — GET_ID
    // is a metadata operation, not classified as any of those.
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
    assert_eq!(c.writes_completed.load(Ordering::Relaxed), 0);
    assert_eq!(c.flushes_completed.load(Ordering::Relaxed), 0);
}

/// `T_GET_ID` chain on a read-only disk must succeed. The
/// metadata read is RO-safe, and the kernel always issues
/// `virtblk_get_id` for `serial_show` regardless of the disk's
/// RO state — rejecting it would surface as an empty
/// `/sys/block/<dev>/serial` on every RO mount.
#[test]
fn process_requests_get_id_succeeds_on_ro_disk() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let mut dev = VirtioBlk::with_options(f, cap, DiskThrottle::default(), true);
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_GET_ID, 0);
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            VIRTIO_BLK_ID_BYTES,
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
        s[0], VIRTIO_BLK_S_OK as u8,
        "RO disk must accept T_GET_ID — serial is RO-safe metadata",
    );
    let mut buf = [0u8; VIRTIO_BLK_ID_BYTES as usize];
    mem.read_slice(&mut buf, data_addr).unwrap();
    assert_eq!(buf, VIRTIO_BLK_SERIAL);

    let c = dev.counters();
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 0);
}

/// Sub-20-byte data descriptor through the chain pipeline.
/// The handler rejects the chain with S_IOERR; used.idx still
/// advances (the chain completes normally with the error
/// status, blk-mq surfaces the error to userspace immediately).
#[test]
fn process_requests_get_id_short_buffer_returns_ioerr() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_GET_ID, 0);
    // 19-byte buffer — short.
    let short_len: u32 = VIRTIO_BLK_ID_BYTES - 1;
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            short_len,
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
}

/// Zero-data T_GET_ID chain (header + status only, no data
/// descriptor) is rejected by the upstream zero-data gate
/// before the handler dispatches. Matches the IN/OUT zero-data
/// rejection.
#[test]
fn process_requests_get_id_zero_data_returns_ioerr() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let status_addr = GuestAddress(0x5000);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_GET_ID, 0);
    // No data descriptor — chain is just header + status.
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
    // Throttle untouched — zero-data gate fires pre-throttle.
    assert_eq!(c.throttled_count.load(Ordering::Relaxed), 0);
}

/// Direction violation through the chain pipeline: T_GET_ID
/// with a device-readable data segment. Outer
/// `direction_violation` gate writes S_IOERR; throttle
/// untouched.
#[test]
fn process_requests_get_id_readonly_data_returns_ioerr() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_GET_ID, 0);
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            VIRTIO_BLK_ID_BYTES,
            0, // device-readable — wrong direction for GET_ID
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
        c.throttled_count.load(Ordering::Relaxed),
        0,
        "direction violation must not consume throttle tokens",
    );
}

/// `VIRTIO_BLK_SERIAL` is exactly 20 bytes (matches
/// `VIRTIO_BLK_ID_BYTES`). A regression that resized the
/// constant would silently truncate or pad the serial in
/// guest sysfs — this pin catches it at compile time of the
/// const init AND at the assertion below.
#[test]
fn serial_constant_is_id_bytes_long() {
    assert_eq!(
        VIRTIO_BLK_SERIAL.len(),
        VIRTIO_BLK_ID_BYTES as usize,
        "serial must be exactly VIRTIO_BLK_ID_BYTES (20) bytes",
    );
    // Last 4 bytes are NUL padding — the kernel's `serial_show`
    // does `buf[VIRTIO_BLK_ID_BYTES] = '\0'` THEN `strlen(buf)`,
    // so we want the embedded NUL inside the 20-byte payload to
    // truncate the string at the meaningful length.
    assert_eq!(
        &VIRTIO_BLK_SERIAL[..16],
        b"ktstr-virtio-blk",
        "serial payload prefix",
    );
    assert_eq!(
        &VIRTIO_BLK_SERIAL[16..],
        &[0u8; 4],
        "trailing 4 bytes are NUL padding",
    );
}

// ----------------------------------------------------------------
// T5/T7/T8/T10/T11/T12: notification suppression edge cases.
//
// T4 (multi-chain unreached threshold) and T9 (error-chain
// suppression) are pinned above. The remaining tests in this
// block cover the rest of the notification-bracket surface:
//   T5  — successive drains spanning the threshold (multi-notify
//         num_added accounting).
//   T7  — `needs_notification` Err fault injection: avail-ring
//         `used_event` GPA unmapped, fail-safe fires the irqfd
//         via `unwrap_or(true)`.
//   T8  — `disable_notification` / `enable_notification` toggle
//         the legacy `VRING_USED_F_NO_NOTIFY` bit observable in
//         `used.flags`.
//   T10 — status-write failure must NOT signal the irqfd: an
//         unmapped `status_addr` produces `publish_completion`
//         → false, signal_needed stays false, `add_used` is
//         skipped, the irqfd stays unsignalled.
//   T11 — multi-notify boundary: a chain enqueued AFTER an
//         earlier drain returned must not be stranded; a fresh
//         QUEUE_NOTIFY drains it. The actual `Ok(true)` re-drain
//         arm is documented as untestable single-threaded (see
//         `outer_loop_drains_two_pre_queued_chains_in_one_call`)
//         — this test pins the deterministic surrogate.
//   T12 — legacy path full chain: post-`process_requests`,
//         `used.flags` is back to 0, proving the
//         disable→drain→enable bracket cleared the suppression
//         flag the device set during the inner drain.
// ----------------------------------------------------------------

/// T5: successive `process_requests` drains spanning the
/// EVENT_IDX threshold. With `used_event=2`, drain 1 publishes
/// one chain (next_used=1, threshold unreached, irqfd
/// suppressed); drain 2 publishes two more chains
/// (next_used=3, threshold crossed, irqfd fires exactly once).
/// Pins the multi-drain accounting:
///
/// - `num_added` is reset by `needs_notification` after each
///   drain (queue.rs line 533), so drain 2's
///   `needs_notification` sees `num_added=2` and the wrapping
///   `used_idx - used_event - 1 < used_idx - old` formula
///   evaluates against the post-drain `next_used=3` against
///   `old = 3 - 2 = 1`. With `used_event=2`, the formula yields
///   `(3 - 2 - 1) < (3 - 1)` → `0 < 2` → true, so the irqfd
///   fires.
///
/// - On drain 1, `num_added=1` after the chain publishes;
///   `needs_notification` sees `next_used=1`, `old=0`,
///   `used_event=2`, formula: `(1 - 2 - 1) < (1 - 0)` →
///   `u16::MAX - 2 < 1` → false (wrapping arithmetic). So
///   drain 1 is suppressed.
///
/// A regression that didn't reset `num_added` (or that
///  reused stale `next_used` values across drains) would fire
/// the irqfd at the wrong time — this test catches both
/// classes.
#[test]
fn event_idx_successive_drains_span_threshold() {
    let mem = make_chain_test_mem();
    let (mut dev, mock) = setup_blk(&mem, false, DiskThrottle::default());
    // setup_blk fixes the mock at queue size 16; mirror that
    // here so used_event_addr's offset arithmetic lines up
    // with the device's negotiated queue.
    let qsize = 16u16;
    // used_event = 2: the guest is asking to be notified once
    // `next_used` reaches 3 (formula crosses the threshold).
    let used_event = used_event_addr(mock.avail_addr(), qsize);
    mem.write_obj::<u16>(u16::to_le(2), used_event)
        .expect("plant used_event");
    dev.set_mem(mem.clone());
    wire_device_to_mock_with_event_idx(&mut dev, &mock, qsize, GuestAddress(0x10000));

    // Drain 1: one read chain. Build then notify.
    {
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
        mock.build_desc_chain(&descs).expect("build chain 1");
    }
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // Post-drain-1: next_used=1 < used_event=2 → irqfd
    // suppressed. interrupt_status bit IS set.
    let used_idx_after_d1: u16 = mem
        .read_obj(GuestAddress(0x10000).checked_add(2).unwrap())
        .expect("read used.idx after drain 1");
    assert_eq!(
        used_idx_after_d1, 1,
        "drain 1 must publish exactly one chain",
    );
    assert_ne!(
        dev.interrupt_status.load(Ordering::Acquire) & VIRTIO_MMIO_INT_VRING,
        0,
        "interrupt_status bit must be set after drain 1 \
             (V8 split: bit set independent of irqfd)",
    );
    // T-GAP-F: same bit observable through the MMIO surface.
    let status = read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS);
    assert_eq!(status & 1, 1);
    assert!(
        dev.irq_evt.read().is_err(),
        "drain 1 irqfd must be suppressed: next_used=1 < used_event=2",
    );

    // Drain 2: two more read chains. Build with disjoint
    // addresses so the descriptor table doesn't alias. Note
    // build_desc_chain reuses descriptor table slots starting
    // at index 0 each call — but the avail ring grows, so the
    // device sees fresh chain heads. The chain CONTENTS at the
    // time of drain are what build_desc_chain wrote LAST, so
    // we plant identical chain shapes that point at distinct
    // data buffers per-chain.
    for i in 0..2u64 {
        let header_addr = GuestAddress(0x7000 + i * 0x1000);
        let data_addr = GuestAddress(0x9000 + i * 0x1000);
        let status_addr = GuestAddress(0xB000 + i * 0x100);
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
        mock.build_desc_chain(&descs)
            .expect("build chain in drain 2");
    }
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // All 3 reads completed across two drains.
    assert_eq!(
        dev.counters().reads_completed.load(Ordering::Relaxed),
        3,
        "1 chain in drain 1 + 2 chains in drain 2 = 3 total reads",
    );
    let used_idx_after_d2: u16 = mem
        .read_obj(GuestAddress(0x10000).checked_add(2).unwrap())
        .expect("read used.idx after drain 2");
    assert_eq!(
        used_idx_after_d2, 3,
        "used.idx must advance to 3 after both drains",
    );
    // irqfd MUST fire on drain 2: next_used=3 crossed
    // used_event=2 with num_added=2 (drain 2 added 2 chains
    // since the drain-1 needs_notification reset num_added).
    let val = dev
        .irq_evt
        .read()
        .expect("irq_evt must be readable after drain 2 crossed threshold");
    assert_eq!(
        val, 1,
        "drain 2 must fire the irqfd exactly once when used_event \
             threshold is crossed across multiple drains",
    );
}

/// T7: `needs_notification` Err fault injection. The post-drain
/// `needs_notification` reads `used_event` from the avail ring
/// (`avail_ring + 4 + size*2`). When that GPA is unmapped, the
/// call returns `Err(GuestMemory(...))`, and the production
/// code's `inspect_err(...).unwrap_or(true)` MUST fail safe to
/// firing the irqfd — a missed IRQ stalls the guest until the
/// hung-task watchdog fires (`kernel.hung_task_timeout_secs`,
/// default 120 s — virtio_blk has no `mq_ops->timeout`), while
/// a redundant IRQ wastes only a vCPU exit.
///
/// Setup approach: build the mock entirely inside the mapped
/// region (mock's `AvailRing::new` writes used_event=0 at
/// construction time, so its avail-ring location must be
/// mapped — we can't put the mock's avail straddling a hole
/// without panicking the mock constructor at mock.rs:151).
/// After wire-up, REDIRECT the device's `avail_ring` address
/// to a custom GPA where the used_event field lands in the
/// unmapped hole. The descriptor table stays at the mock's
/// location (mock owns those slots), but the device reads
/// avail.flags/idx/ring/used_event from the new custom
/// avail location. We manually plant avail.idx and the
/// ring[0] entry at the custom address pointing at desc[0]
/// in the mock's desc_table.
///
/// Memory layout: region 1 [0, 0xA000), region 2
/// [0xB000, 0x40000), hole [0xA000, 0xB000). Custom avail
/// at 0x9FDC: ring entries occupy 0x9FE0..0xA000 (in mapped
/// region), used_event at 0xA000 (start of hole) → reads
/// fail. Custom avail must be 2-byte aligned (Queue's avail
/// alignment check); 0x9FDC & 0x1 == 0 ✓.
///
/// Set-up sequencing: queue config writes (avail/used ring
/// addresses) are gated on the FEATURES_OK..DRIVER_OK FSM
/// window. We let `wire_device_to_mock_with_event_idx`
/// drive the FSM through DRIVER_OK with the mock's avail
/// addr, then directly call
/// `dev.worker.queues[REQ_QUEUE].set_avail_ring_address(...)` to
/// override post-FSM. The QueueT setter bypasses the FSM
/// gate (the FSM gate is in `mmio_write`, not in `Queue`).
#[test]
fn event_idx_needs_notification_err_fires_irqfd_fail_safe() {
    use virtio_queue::QueueT;
    use vm_memory::Bytes;
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    // Multi-region mem with hole [0xA000, 0xB000).
    let mem =
        GuestMemoryMmap::from_ranges(&[(GuestAddress(0), 0xA000), (GuestAddress(0xB000), 0x40000)])
            .expect("create multi-region guest mem with avail-event hole");
    let qsize = 16u16;
    // Mock at GPA 0 — entirely in region 1. Mock's
    // construction-time used_event store lands at the
    // mock's natural location (0x100 + 0x24 = 0x124),
    // safely mapped.
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), qsize);
    // Custom avail ring at 0x9FDC: flags/idx/ring all in
    // mapped region; used_event at 0xA000 in hole.
    let custom_avail = GuestAddress(0x9FDC);
    let custom_used_event = custom_avail
        .checked_add(4 + qsize as u64 * 2)
        .expect("custom used_event addr");
    assert_eq!(
        custom_used_event,
        GuestAddress(0xA000),
        "test layout error: custom used_event must land at hole boundary",
    );
    // Confirm the boundary is actually unmapped.
    let mut probe = [0u8; 2];
    assert!(
        mem.read_slice(&mut probe, custom_used_event).is_err(),
        "test layout error: custom used_event GPA must be unmapped",
    );
    // Plant a FLUSH chain (no data, header + status only)
    // via the mock — populates desc_table[0..1] and bumps
    // mock's natural avail.idx. We'll mirror the relevant
    // entries to the custom avail location below.
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
    mock.build_desc_chain(&descs).expect("build flush chain");
    // Mirror the chain's avail-ring state to the custom
    // location: flags=0, idx=1, ring[0]=0 (head index of
    // the chain just built — build_desc_chain allocates
    // descriptors starting at 0). Without this, the device
    // would read avail.idx=0 from the custom avail and
    // process zero chains.
    mem.write_obj::<u16>(u16::to_le(0), custom_avail)
        .expect("plant custom avail.flags");
    mem.write_obj::<u16>(
        u16::to_le(1),
        custom_avail.checked_add(2).expect("idx addr"),
    )
    .expect("plant custom avail.idx");
    mem.write_obj::<u16>(
        u16::to_le(0),
        custom_avail.checked_add(4).expect("ring[0] addr"),
    )
    .expect("plant custom avail.ring[0]");
    dev.set_mem(mem.clone());
    // used_override = 0xB000 (region 2): avoids any conflict
    // with the custom avail ring and gives set_avail_event a
    // valid mapped target. The set_avail_event write at
    // 0xB000 + 4 + 16*8 = 0xB084 is well inside region 2.
    wire_device_to_mock_with_event_idx(&mut dev, &mock, qsize, GuestAddress(0xB000));
    // Override the device's avail_ring AFTER wire-up so the
    // device reads from the custom location with the
    // unmapped used_event field. The mock's natural avail
    // ring is no longer consulted by the device; the desc
    // table at the mock's desc_table_addr remains in use
    // (the chain head index in custom avail.ring[0] points
    // there).
    dev.worker.queues[REQ_QUEUE].set_avail_ring_address(
        Some(custom_avail.0 as u32),
        Some((custom_avail.0 >> 32) as u32),
    );
    assert_eq!(
        dev.worker.queues[REQ_QUEUE].avail_ring(),
        custom_avail.0,
        "avail ring override did not take effect",
    );

    // Pre-notify: irqfd MUST be unsignalled.
    assert!(
        dev.irq_evt.read().is_err(),
        "irq_evt must not be signalled before notify",
    );

    // Fire QUEUE_NOTIFY. The chain processes (the device
    // walks desc_table[0] → header → status), add_used at
    // 0xB000 succeeds, inner loop returns None,
    // enable_notification: set_avail_event at 0xB084
    // (succeeds), avail_idx re-read from 0x9FDE (custom,
    // mapped, returns 1), Ok(false) → break. Post-drain
    // needs_notification reads used_event at 0xA000
    // (FAILS — unmapped), inspect_err logs warn,
    // unwrap_or(true) → fire.
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // FLUSH completed — the chain reached the handler
    // despite the avail-ring redirect, proving the chain
    // walk completes before the post-drain needs_notification
    // failure.
    let c = dev.counters();
    assert_eq!(
        c.flushes_completed.load(Ordering::Relaxed),
        1,
        "FLUSH chain must complete normally — failure is in \
             needs_notification, not request processing",
    );
    // used.idx advanced — add_used succeeded at the override.
    let used_idx: u16 = mem
        .read_obj(GuestAddress(0xB000).checked_add(2).unwrap())
        .expect("read used.idx at override addr");
    assert_eq!(
        used_idx, 1,
        "used.idx must advance to 1 — add_used path is independent \
             of needs_notification",
    );
    // V8: interrupt_status bit set independent of irqfd gate.
    assert_ne!(
        dev.interrupt_status.load(Ordering::Acquire) & VIRTIO_MMIO_INT_VRING,
        0,
        "interrupt_status bit must be set after publish, even \
             when needs_notification fails",
    );
    // Fail-safe fire: unwrap_or(true) on the Err return path
    // produces an irqfd write.
    let val = dev
        .irq_evt
        .read()
        .expect("irq_evt must fire fail-safe when needs_notification Err");
    assert_eq!(
        val, 1,
        "irq_evt must fire exactly once via unwrap_or(true) \
             when needs_notification returns Err",
    );
}

/// T8: `disable_notification` / `enable_notification` toggle
/// the legacy `VRING_USED_F_NO_NOTIFY` flag observable in
/// `used.flags`. Pins the QueueT API contract that the
/// production bracket relies on: when EVENT_IDX is NOT
/// negotiated, `disable_notification` writes
/// `VRING_USED_F_NO_NOTIFY` to `used.flags`, telling the
/// guest to skip QUEUE_NOTIFY MMIO writes during the drain;
/// `enable_notification` clears it.
///
/// Driving the device's queue directly (rather than going
/// through `process_requests`) is the only way to observe
/// the FLAG-SET state — `process_requests` calls disable
/// → drain → enable as one synchronous unit, and by the
/// time the test reads `used.flags` post-call, the flag is
/// already cleared. This test pins the toggle behaviour at
/// the bracket's primitive layer; T12 below pins the
/// process_requests integration.
///
/// Per `Queue::set_notification` (queue.rs):
/// - legacy + disable → write `VRING_USED_F_NO_NOTIFY` to
///   used.flags (line 240).
/// - legacy + enable → write 0 to used.flags (line 237).
/// - EVENT_IDX + disable → no-op (line 244).
/// - EVENT_IDX + enable → write `next_avail` to
///   used_ring's avail_event field (line 235).
#[test]
fn legacy_disable_enable_notification_toggles_used_flags() {
    use virtio_bindings::bindings::virtio_ring::VRING_USED_F_NO_NOTIFY;
    let mem = make_chain_test_mem();
    let (mut dev, mock) = setup_blk(&mem, false, DiskThrottle::default());
    dev.set_mem(mem.clone());
    // Wire the legacy path (no EVENT_IDX) so disable/enable
    // hit the flag-toggle branch. process_requests is NOT
    // called — we drive the queue directly.
    wire_device_to_mock(&mut dev, &mock);
    // Sanity: the queue must NOT have EVENT_IDX enabled. A
    // regression in wire_device_to_mock that accidentally
    // negotiated EVENT_IDX would route through the no-op
    // branch and break this test's premise.
    use virtio_queue::QueueT;
    assert!(
        !dev.worker.queues[REQ_QUEUE].event_idx_enabled(),
        "wire_device_to_mock must produce a legacy-path queue \
             (no EVENT_IDX); test premise depends on it",
    );

    // Initial: flags = 0 (mock initializes used.flags to 0,
    // mock.rs line 149).
    let flags0: u16 = mem
        .read_obj(mock.used_addr())
        .expect("read initial used.flags");
    assert_eq!(flags0, 0, "mock initializes used.flags to 0",);

    // disable_notification → VRING_USED_F_NO_NOTIFY in
    // used.flags.
    dev.worker.queues[REQ_QUEUE]
        .disable_notification(&mem)
        .expect("disable_notification on legacy queue");
    let flags1: u16 = mem
        .read_obj(mock.used_addr())
        .expect("read used.flags after disable");
    assert_eq!(
        flags1, VRING_USED_F_NO_NOTIFY as u16,
        "legacy disable_notification must set VRING_USED_F_NO_NOTIFY \
             ({:#x}); got {:#x}",
        VRING_USED_F_NO_NOTIFY, flags1,
    );

    // enable_notification → flag cleared, used.flags = 0.
    // The return Ok(_) value reflects whether avail_idx
    // changed during the disabled window; with no chains
    // queued by the test, it must be Ok(false).
    let re_drain = dev.worker.queues[REQ_QUEUE]
        .enable_notification(&mem)
        .expect("enable_notification on legacy queue");
    assert!(
        !re_drain,
        "no chains queued; enable_notification must return Ok(false)",
    );
    let flags2: u16 = mem
        .read_obj(mock.used_addr())
        .expect("read used.flags after enable");
    assert_eq!(
        flags2, 0,
        "legacy enable_notification must clear used.flags; got {:#x}",
        flags2,
    );

    // Idempotent re-toggle: a second disable→enable must
    // produce the same observed state. Catches a regression
    // that accumulated a stale bit or that latched the flag
    // after the first toggle.
    dev.worker.queues[REQ_QUEUE]
        .disable_notification(&mem)
        .expect("second disable");
    let flags3: u16 = mem
        .read_obj(mock.used_addr())
        .expect("read used.flags after second disable");
    assert_eq!(flags3, VRING_USED_F_NO_NOTIFY as u16);
    dev.worker.queues[REQ_QUEUE]
        .enable_notification(&mem)
        .expect("second enable");
    let flags4: u16 = mem
        .read_obj(mock.used_addr())
        .expect("read used.flags after second enable");
    assert_eq!(flags4, 0);
}

/// T10: status-write-failure path. When `publish_completion`
/// fails to write the status byte (status_addr unmapped),
/// it returns `false`, the chain is NOT add_used'd, and
/// `signal_needed` stays false — so the irqfd is NEVER
/// signalled for this chain.
///
/// This pins the F15 contract: NEVER advance the used ring
/// for a chain whose status byte the guest can't observe.
/// The guest's `virtblk_done` reads the status byte from
/// `vbr->in_hdr.status` — initially zero from `__GFP_ZERO`
/// or stale from prior blk-mq tag use — and `virtblk_result(0)`
/// → `BLK_STS_OK`, silently corrupting reads / dropping
/// writes. The chain stays in the avail ring; virtio_blk has
/// no `mq_ops->timeout` callback (drivers/block/virtio_blk.c
/// `virtio_mq_ops` has no `.timeout` field), so blk-mq alone
/// never surfaces an unpublished request. The guest only
/// sees the stall when the hung-task watchdog fires
/// (`kernel.hung_task_timeout_secs`, default 120 s) or a
/// higher-layer (filesystem, application) retries.
///
/// `io_errors` MUST be bumped: by the per-handler error path
/// before publish_completion, AND by publish_completion
/// itself on the status-write failure (intentional double-bump
/// — see publish_completion docs on the silent-stall counter
/// rationale). The test asserts `io_errors >= 1` because the
/// double-count behaviour is implementation detail; the
/// load-bearing assertion is "host operator sees the
/// silent-stall via a counter."
///
/// Setup: a multi-region mem with status_addr at 0x20000 (in
/// the hole [0x20000, 0x30000)). The chain's header and data
/// descriptors land in region 1; status_addr is unmapped, so
/// `mem.write_slice(status_byte, status_addr)` fails inside
/// publish_completion.
#[test]
fn status_write_failure_skips_add_used_and_irqfd() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    // Multi-region mem with a hole at [0x20000, 0x30000).
    // status_addr=0x20000 lands in the hole.
    let mem = GuestMemoryMmap::from_ranges(&[
        (GuestAddress(0), 0x20000),
        (GuestAddress(0x30000), 0x10000),
    ])
    .expect("create multi-region guest mem with status hole");
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x20000); // unmapped
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
    // Sanity: confirm status_addr is actually unmapped before
    // running the device — a layout regression that mapped it
    // would silently turn the test into a happy-path read.
    let mut probe = [0u8; 1];
    assert!(
        mem.write_slice(&[0u8], status_addr).is_err(),
        "test layout error: status_addr must be unmapped",
    );
    assert!(
        mem.read_slice(&mut probe, status_addr).is_err(),
        "test layout error: status_addr must be unmapped",
    );
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
    // Legacy path — used_event is irrelevant since the chain
    // is never add_used'd. Using legacy makes the test focus
    // on the publish_completion gate, not the EVENT_IDX
    // suppression logic (already covered by T9).
    wire_device_to_mock(&mut dev, &mock);

    // Pre-notify: irqfd MUST be unsignalled.
    assert!(
        dev.irq_evt.read().is_err(),
        "irq_evt must not be signalled before notify",
    );

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // The handler ran (e.g. handle_read_impl read the
    // backing file into `data_addr`) but publish_completion
    // failed to write the status byte. used.idx MUST stay 0
    // — no add_used, no observable completion.
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(
        used_idx, 0,
        "status-write failure must skip add_used; used.idx \
             must stay 0 so the chain remains in the avail ring \
             and the guest's hung-task watchdog \
             (kernel.hung_task_timeout_secs, default 120 s) \
             eventually fires — virtio_blk has no mq_ops->timeout",
    );
    // io_errors bumped — host operator sees the silent-stall
    // event via the counter even though the guest never sees
    // an S_IOERR (the status byte was never written).
    let c = dev.counters();
    assert!(
        c.io_errors.load(Ordering::Relaxed) >= 1,
        "io_errors must be bumped on status-write failure; got {}",
        c.io_errors.load(Ordering::Relaxed),
    );
    // irq_evt MUST be unsignalled. publish_completion
    // returned false → signal_needed stays false → no irqfd
    // write. The interrupt_status bit also stays 0 because
    // it's only set on the `if signal_needed` path.
    assert!(
        dev.irq_evt.read().is_err(),
        "irq_evt must be unsignalled when publish_completion fails — \
             a chain the guest can't observe must NOT trigger an IRQ",
    );
    assert_eq!(
        dev.interrupt_status.load(Ordering::Acquire) & VIRTIO_MMIO_INT_VRING,
        0,
        "interrupt_status bit must stay 0 when no chain is \
             published — signal_needed remained false throughout",
    );
    // Same property observable through the MMIO surface — the
    // bit is what the guest's ISR / polling path actually
    // reads (VIRTIO_MMIO_INTERRUPT_STATUS register).
    assert_eq!(read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS) & 1, 0);
}

/// T11: multi-notify boundary regression. A chain enqueued
/// AFTER an earlier QUEUE_NOTIFY drain has returned must not
/// be stranded; a fresh QUEUE_NOTIFY drains it cleanly. The
/// guest's ISR updates `used_event` between drains to
/// re-arm notifications — we mirror that with a host-side
/// write so drain 2's `needs_notification` evaluates against
/// the new threshold.
///
/// Note on coverage scope: the production
/// `enable_notification → Ok(true) → continue 'outer`
/// re-drain arm fires when `avail_idx` advances between the
/// inner-loop break and the `enable_notification` call's
/// re-read of `avail_idx`. In single-threaded test code,
/// there is no concurrent vCPU to bump `avail_idx` mid-call
/// — `process_requests` runs as one synchronous unit driven
/// by `mmio_write(QUEUE_NOTIFY, ...)`. See the existing
/// `outer_loop_drains_two_pre_queued_chains_in_one_call`
/// test's doc note for the same observation.
///
/// What this test guarantees: a chain enqueued AFTER drain
/// 1 returns is processed by drain 2. A regression that
/// stranded chains across QUEUE_NOTIFY boundaries (e.g. by
/// caching `next_avail` past the bracket close, or by
/// failing to re-arm notifications correctly) would surface
/// here as `flushes_completed=1` instead of `2`.
///
/// Together with `outer_loop_drains_two_pre_queued_chains_in_one_call`,
/// this pins both shapes of multi-chain delivery: pre-batched
/// (one notify, multiple chains) and incremental (multiple
/// notifies, one chain each). The actual `Ok(true)` re-drain
/// arm is exercised in production by the real-world avail-idx
/// bump from a concurrent vCPU; this pair pins the
/// observable equivalent under test conditions.
///
/// EVENT_IDX threshold semantics (queue.rs:535):
/// `Ok(used_idx - used_event - 1 < used_idx - old)`,
/// where `old = used_idx - num_added`. After drain 1
/// (used_event=0, used_idx=1, num_added=1, old=0):
/// `(1-0-1) < (1-0)` → `0 < 1` → true → fire. After drain 2
/// without updating used_event (used_event=0, used_idx=2,
/// num_added=1, old=1): `(2-0-1) < (2-1)` → `1 < 1` →
/// false → suppress. To pin "drain 2 fires," we update
/// used_event=1 between drains, simulating the guest's
/// ISR consuming chain 1 and re-arming the threshold:
/// `(2-1-1) < (2-1)` → `0 < 1` → true → fire.
#[test]
fn multi_notify_boundary_drains_subsequent_chain() {
    let mem = make_chain_test_mem();
    let (mut dev, mock) = setup_blk(&mem, false, DiskThrottle::default());
    dev.set_mem(mem.clone());
    // EVENT_IDX path with used_event=0: drain 1 fires
    // (next_used crosses 0+1). Between drains we'll bump
    // used_event=1 so drain 2 fires when next_used reaches 2.
    let qsize = 16u16;
    let used_event = used_event_addr(mock.avail_addr(), qsize);
    mem.write_obj::<u16>(u16::to_le(0), used_event)
        .expect("plant used_event=0 for drain 1");
    wire_device_to_mock_with_event_idx(&mut dev, &mock, qsize, GuestAddress(0x10000));

    // Drain 1: one FLUSH chain.
    {
        let header_addr = GuestAddress(0x4000);
        let status_addr = GuestAddress(0x4100);
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
        mock.build_desc_chain(&descs).expect("build chain 1");
    }
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);
    assert_eq!(
        dev.counters().flushes_completed.load(Ordering::Relaxed),
        1,
        "drain 1 must complete the first FLUSH",
    );
    let used_idx_d1: u16 = mem
        .read_obj(GuestAddress(0x10000).checked_add(2).unwrap())
        .expect("read used.idx after drain 1");
    assert_eq!(used_idx_d1, 1);
    // Drain 1's irqfd fired (used_event=0, next_used=1 → fire).
    // Read it now so the eventfd counter doesn't accumulate
    // and confuse the drain-2 read.
    let val1 = dev.irq_evt.read().expect("drain 1 irqfd must fire");
    assert_eq!(val1, 1, "drain 1 fires exactly once");

    // Simulate the guest's ISR: after consuming chain 1's
    // used-ring entry, the guest re-arms the threshold by
    // bumping used_event so the device knows where to next
    // signal. Without this update, drain 2 below would be
    // suppressed by the EVENT_IDX threshold (next_used=2 vs
    // stale used_event=0 → formula yields false).
    mem.write_obj::<u16>(u16::to_le(1), used_event)
        .expect("update used_event=1 for drain 2");

    // Drain 2: build a fresh chain AFTER drain 1 returned —
    // this models the production case where the guest enqueues
    // more work between QUEUE_NOTIFY events. The new chain's
    // descriptors share descriptor table slots with chain 1
    // (build_desc_chain reuses slots from index 0), but the
    // avail ring grows by one entry. The device sees a fresh
    // chain head pointing at the new addresses.
    {
        let header_addr = GuestAddress(0x5000);
        let status_addr = GuestAddress(0x5100);
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
        mock.build_desc_chain(&descs).expect("build chain 2");
    }
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // Both flushes completed. A regression that left chain 2
    // stranded would show flushes_completed=1 here.
    assert_eq!(
        dev.counters().flushes_completed.load(Ordering::Relaxed),
        2,
        "drain 2 must process the chain enqueued after drain 1 — \
             a stranded chain would leave flushes_completed at 1",
    );
    let used_idx_d2: u16 = mem
        .read_obj(GuestAddress(0x10000).checked_add(2).unwrap())
        .expect("read used.idx after drain 2");
    assert_eq!(
        used_idx_d2, 2,
        "used.idx must advance to 2 across the two notifies",
    );
    // Drain 2 fires the irqfd: with the updated used_event=1
    // and post-drain next_used=2, num_added=1, old=1, the
    // threshold formula evaluates to true.
    let val2 = dev.irq_evt.read().expect("drain 2 irqfd must fire");
    assert_eq!(
        val2, 1,
        "drain 2 fires the irqfd exactly once for the post-boundary chain",
    );
}

/// T12: legacy-path full-chain integration of the
/// disable→drain→enable bracket. After
/// `process_requests` returns, `used.flags` must be back
/// to 0 — proving `enable_notification` ran at the end of
/// the bracket and cleared the
/// `VRING_USED_F_NO_NOTIFY` bit the inner
/// `disable_notification` set during the drain.
///
/// `used.idx` advances to 1 — the chain completed
/// normally. `irq_evt` fires unconditionally on the legacy
/// path because `Queue::needs_notification` returns
/// `Ok(true)` whenever `event_idx_enabled=false`
/// (queue.rs line 538). T12's load-bearing assertion is the
/// post-bracket `used.flags == 0` — the rest of the state
/// is companion coverage to confirm the chain processed
/// correctly (so a flag-toggle bug isn't masked by a
/// chain-drop bug).
///
/// Distinct from T8 (which drives the QueueT API directly):
/// T12 verifies that `process_requests` invokes the bracket
/// in the correct order — `disable_notification` then
/// drain then `enable_notification` — so the observable
/// post-call state is the cleared flag.
#[test]
fn legacy_process_requests_clears_used_flags_post_bracket() {
    use virtio_bindings::bindings::virtio_ring::VRING_USED_F_NO_NOTIFY;
    let mem = make_chain_test_mem();
    let (mut dev, mock) = setup_blk(&mem, false, DiskThrottle::default());
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
    // Legacy path — no EVENT_IDX. `disable_notification`
    // writes the suppression flag, `enable_notification`
    // clears it.
    wire_device_to_mock(&mut dev, &mock);
    // Sanity: legacy path negotiated. A regression that
    // accidentally routed through EVENT_IDX would skip the
    // flag toggle (set_notification's EVENT_IDX-disable arm
    // is a no-op, queue.rs line 244) and the post-call
    // used.flags assertion would still pass — but for the
    // wrong reason. Pin the wiring premise.
    use virtio_queue::QueueT;
    assert!(
        !dev.worker.queues[REQ_QUEUE].event_idx_enabled(),
        "legacy wiring must not negotiate EVENT_IDX",
    );

    // Initial: flags = 0 (mock initializes used.flags to 0).
    let flags_before: u16 = mem
        .read_obj(mock.used_addr())
        .expect("read used.flags before notify");
    assert_eq!(flags_before, 0, "mock initializes used.flags to 0",);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // Post-call: bracket completed. `enable_notification`
    // cleared the flag.
    let flags_after: u16 = mem
        .read_obj(mock.used_addr())
        .expect("read used.flags after notify");
    assert_eq!(
        flags_after, 0,
        "process_requests bracket must end with used.flags=0; \
             VRING_USED_F_NO_NOTIFY ({:#x}) must NOT remain set after \
             enable_notification ran. Got {:#x}",
        VRING_USED_F_NO_NOTIFY, flags_after,
    );

    // Companion coverage: the chain processed and the irqfd
    // fired (legacy path always fires).
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 1, "chain must complete normally (legacy path)",);
    assert_eq!(dev.counters().reads_completed.load(Ordering::Relaxed), 1,);
    let val = dev.irq_evt.read().expect("legacy path must fire irq_evt");
    assert_eq!(
        val, 1,
        "legacy path fires irq_evt unconditionally — pinned to \
             confirm the bracket didn't suppress on legacy",
    );
}

/// Throttle stall on an EVENT_IDX-negotiated device leaves
/// the chain in the avail ring just like the legacy path —
/// EVENT_IDX is irrelevant to the stall contract because no
/// `add_used`, no status write, and no irqfd write happens.
/// Pins the absence of an EVENT_IDX-specific stall bypass:
/// the device must not "publish a throttled IOERR through
/// the suppression gate" (the prior IOERR-on-stall behavior)
/// and must not "fire the irqfd because EVENT_IDX is on"
/// (no signal_needed=true reaches the gate).
#[test]
fn throttle_event_idx_stall_leaves_chain_in_avail_ring() {
    let throttle = DiskThrottle {
        iops: std::num::NonZeroU64::new(1),
        bytes_per_sec: None,
        iops_burst_capacity: None,
        bytes_burst_capacity: None,
    };
    let mem = make_chain_test_mem();
    let qsize = 16u16;
    let (mut dev, mock) = setup_blk(&mem, false, throttle);
    // setup_blk's default queue size is 16; matches `qsize`
    // here so the avail-ring `used_event` offset arithmetic
    // (used_event_addr below) lines up with the device's
    // negotiated queue.
    // Drain the bucket and pin its last_refill so the next
    // consume yields 0 tokens (matches the existing throttle
    // tests' set_last_refill_for_test pattern).
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
    // used_event = 0: a low threshold the next published chain
    // would cross. Under the OLD behavior the throttled chain
    // would (a) publish IOERR via add_used (next_used → 1) and
    // (b) `needs_notification` formula `(1-0-1) < (1-0)` → true
    // → fire. Under the NEW behavior the chain isn't published
    // at all, so the gate is never consulted.
    let used_event = used_event_addr(mock.avail_addr(), qsize);
    mem.write_obj::<u16>(u16::to_le(0), used_event)
        .expect("plant used_event");
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    // Plant a status sentinel so the no-status-byte assertion
    // is unambiguous.
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
    wire_device_to_mock_with_event_idx(&mut dev, &mock, qsize, GuestAddress(0x10000));
    // Capture next_avail BEFORE the notify so we can assert
    // the rollback preserved the cursor (stall pop bumps by
    // 1, set_next_avail(prev.wrapping_sub(1)) restores it).
    let next_avail_before = dev.worker.queues[REQ_QUEUE].next_avail();
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // Status sentinel survives — no add_used, no status write
    // on stall, even when EVENT_IDX would otherwise cross
    // the threshold.
    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(
        s[0], 0xEE,
        "throttle stall must NOT write a status byte even on \
             EVENT_IDX path",
    );
    let c = dev.counters();
    assert_eq!(
        c.throttled_count.load(Ordering::Relaxed),
        1,
        "throttle stall bumps throttled_count exactly once",
    );
    assert_eq!(
        c.io_errors.load(Ordering::Relaxed),
        0,
        "throttle stall is not classified as an I/O error",
    );
    // Used ring (override addr 0x10000) stayed at 0 — no
    // add_used reached the override location either.
    let used_idx: u16 = mem
        .read_obj(GuestAddress(0x10000).checked_add(2).unwrap())
        .expect("read device used.idx at override addr");
    assert_eq!(
        used_idx, 0,
        "throttle stall must NOT advance used.idx even on \
             EVENT_IDX path",
    );
    // INT_VRING bit unset: signal_needed never went true on
    // stall, so the post-drain `if signal_needed` branch
    // (which sets the bit) was skipped.
    assert_eq!(
        dev.interrupt_status.load(Ordering::Acquire) & VIRTIO_MMIO_INT_VRING,
        0,
        "throttle stall must NOT set INT_VRING — signal_needed \
             stays false on the stall path",
    );
    // Same property observable through the MMIO surface.
    let status = read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS);
    assert_eq!(status & 1, 0);
    // irqfd unsignalled — independent of EVENT_IDX gate.
    assert!(
        dev.irq_evt.read().is_err(),
        "throttle stall must NOT signal the irqfd",
    );
    // next_avail equals pre-stall value: the rollback
    // (set_next_avail(prev.wrapping_sub(1))) restored it after
    // the inner-loop pop bumped it by 1. Catches a regression
    // that lost the rollback on the EVENT_IDX path.
    assert_eq!(
        dev.worker.queues[REQ_QUEUE].next_avail(),
        next_avail_before,
        "post-stall next_avail must equal pre-stall value \
             (rollback preserved on EVENT_IDX path)",
    );
}

/// G4: `mem_unset_warned` latch fires once across multiple
/// pre-`set_mem` notifies. The drain path drops requests when
/// the shared `mem` slot is None and emits one warn the first
/// time (in `drain_inline` and `worker_thread_main` via
/// `if !mem_unset_warned.swap(true, Relaxed)`).
/// Without the latch, a buggy caller that issues N notifies
/// before set_mem would flood the log with N copies.
///
/// The test asserts the AtomicBool state directly because the
/// warn itself is observable only via tracing-subscriber log
/// capture (overkill for this one-shot check). The swap
/// semantics encode "fire-once": first call returns false
/// (was false → flips to true → warn emitted); second call
/// returns true (was true → stays true → warn skipped). So
/// reading the bool across two notifies pins both halves of
/// the latch contract.
#[test]
fn mem_unset_warned_latch_fires_once() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    // Initial: latch is false (default-init by AtomicBool::new(false)).
    assert!(
        !dev.mem_unset_warned.load(Ordering::Relaxed),
        "fresh device must have mem_unset_warned=false",
    );

    // First QUEUE_NOTIFY without set_mem: process_requests's
    // early-return arm flips the latch from false to true.
    // mmio_write(QUEUE_NOTIFY, REQ_QUEUE) goes through unconditionally
    // — the FSM does not gate QUEUE_NOTIFY (the QUEUE_NOTIFY arm of mmio_write).
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);
    assert!(
        dev.mem_unset_warned.load(Ordering::Relaxed),
        "first pre-set_mem notify must flip the latch to true",
    );

    // Second QUEUE_NOTIFY without set_mem: latch stays true.
    // The swap returns the old value (true) and re-stores
    // true — no new warn emitted. We assert the post-state to
    // confirm no spurious flip-back.
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);
    assert!(
        dev.mem_unset_warned.load(Ordering::Relaxed),
        "second pre-set_mem notify must leave the latch at true",
    );
    // Counters stay at zero: no actual processing happened on
    // either notify (the early-return path skipped everything).
    let c = dev.counters();
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 0);
}

/// T-GAP-G: `INTERRUPT_ACK` clears the
/// `VIRTIO_MMIO_INT_VRING` bit set by a chain completion via
/// `process_requests`. End-to-end pin: drain a chain through
/// the public MMIO surface, confirm INTERRUPT_STATUS reflects
/// the bit, write INTERRUPT_ACK to clear, confirm
/// INTERRUPT_STATUS reads zero. Distinct from
/// `interrupt_ack_clears_bits` which manipulates
/// `dev.interrupt_status` directly — this test pins ACK
/// semantics on a real-world bit-set source.
///
/// Production path: `process_requests` post-drain branch sets
/// the bit (`self.interrupt_status |= VIRTIO_MMIO_INT_VRING`).
/// `mmio_write(INTERRUPT_ACK, val)` clears bits via
/// `interrupt_status &= !val` in the INTERRUPT_ACK arm of
/// `mmio_write`.
#[test]
fn interrupt_ack_clears_status_bit() {
    let mem = make_chain_test_mem();
    let (mut dev, mock) = setup_blk(&mem, false, DiskThrottle::default());
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
    // After drain: bit set.
    assert_eq!(
        read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS) & 1,
        1,
        "drained chain must set VIRTIO_MMIO_INT_VRING in INTERRUPT_STATUS",
    );
    // Write INTERRUPT_ACK with the bit set → clears it.
    write_reg(&mut dev, VIRTIO_MMIO_INTERRUPT_ACK, VIRTIO_MMIO_INT_VRING);
    assert_eq!(
        read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS) & 1,
        0,
        "INTERRUPT_ACK with VIRTIO_MMIO_INT_VRING must clear the bit",
    );
}
