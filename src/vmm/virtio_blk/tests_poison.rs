#![cfg(test)]

#![allow(unused_imports)]

use super::*;
use super::testing::*;
use std::io::Write;
use std::sync::atomic::Ordering;
use tempfile::tempfile;
use virtio_bindings::bindings::virtio_ring::VRING_DESC_F_WRITE;
use virtio_queue::desc::{RawDescriptor, split::Descriptor as SplitDescriptor};
use virtio_queue::mock::MockSplitQueue;
use vm_memory::Address;

    /// Hostile-guest avail.idx defense. The virtio spec
    /// (virtio-v1.2 §2.7.13.3) requires `avail.idx` to advance
    /// monotonically and stay at most `queue.size` ahead of the
    /// device's `next_avail` cursor. The virtio-queue crate's
    /// `AvailIter::new` enforces this with
    /// `(idx - queue.next_avail).0 > queue.size` → returns
    /// `Error::InvalidAvailRingIndex` (queue.rs:707-709).
    ///
    /// The crate's `pop_descriptor_chain` SWALLOWS that error
    /// (queue.rs:573-587), so a naive drain loop would observe
    /// `None`, fall through to `enable_notification` which re-reads
    /// the same hostile avail.idx, returns `Ok(true)`, and the
    /// outer loop would re-iterate forever — burning a host CPU on
    /// the worker thread. This test pins the defense:
    ///
    ///   1. Plant a bogus avail.idx (1000, well above the device's
    ///      queue.size of 256).
    ///   2. Kick QUEUE_NOTIFY → drain runs, calls `Queue::iter` via
    ///      `q.lock()`, observes `InvalidAvailRingIndex`, sets
    ///      `queue_poisoned=true`, bumps `invalid_avail_idx_count`,
    ///      returns Done WITHOUT calling enable_notification.
    ///   3. Re-kick the poisoned queue → early-return at the top of
    ///      drain produces ZERO additional bumps (per-event
    ///      counter).
    ///   4. No reads completed in either kick (the malformed chain
    ///      is never popped).
    ///   5. A virtio reset clears the poison: rebind, build a real
    ///      chain, kick → it services normally and bumps
    ///      `reads_completed`.
    ///
    /// The test is the only mechanical guarantee that an unbounded
    /// adversarial guest cannot livelock the device.
    #[test]
    fn hostile_avail_idx_poisons_queue_until_reset() {
        let cap = 4096u64;
        let f = make_backed_file_with_pattern(cap, 0xAB);
        let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
        let mem = make_chain_test_mem();
        // MockSplitQueue size and the device's negotiated queue
        // size are independent. The mock's allocations only need to
        // hold descriptor table entries for the planted chain; the
        // poison threshold is set by the device's negotiated
        // queue.size, which `wire_device_to_mock` sets to
        // `QUEUE_MAX_SIZE` (256). Pick a mock size that holds the
        // 3-descriptor chain we plant for the post-reset success
        // case.
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
        let header_addr = GuestAddress(0x4000);
        let data_addr = GuestAddress(0x5000);
        let status_addr = GuestAddress(0x6000);
        write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
        let descs = [
            RawDescriptor::from(SplitDescriptor::new(
                header_addr.0,
                VIRTIO_BLK_OUTHDR_SIZE as u32,
                virtio_bindings::bindings::virtio_ring::VRING_DESC_F_NEXT as u16,
                1,
            )),
            RawDescriptor::from(SplitDescriptor::new(
                data_addr.0,
                512,
                VRING_DESC_F_WRITE as u16
                    | virtio_bindings::bindings::virtio_ring::VRING_DESC_F_NEXT as u16,
                2,
            )),
            RawDescriptor::from(SplitDescriptor::new(
                status_addr.0,
                1,
                VRING_DESC_F_WRITE as u16,
                0,
            )),
        ];
        // Build a real chain so the descriptor table is populated.
        // We'll then overwrite the avail.idx with a bogus value to
        // trigger the bounds check; the chain's actual contents are
        // irrelevant because the poison fires before iter() yields
        // a chain head.
        mock.build_desc_chain(&descs)
            .expect("build chain (consumed by hostile-idx test)");
        dev.set_mem(mem.clone());
        wire_device_to_mock(&mut dev, &mock);

        // Phase A — sanity: counter starts at zero.
        assert_eq!(
            dev.counters().invalid_avail_idx_count(),
            0,
            "fresh device must have zero InvalidAvailRingIndex events",
        );

        // Phase B — plant a bogus avail.idx. avail.idx lives at
        // avail_addr + 2 (after the 2-byte flags field), per
        // virtio-v1.2 §2.7.6. The device's negotiated queue.size is
        // 256 (QUEUE_MAX_SIZE); planting 1000 makes the bounds
        // check `(1000 - next_avail).0 > 256` fire — even the
        // smallest possible difference (next_avail = 1 from the
        // build_desc_chain bump) gives 999 > 256, well clear of
        // the threshold.
        let avail_idx_addr = mock.avail_addr().checked_add(2).unwrap();
        mem.write_obj(1000u16, avail_idx_addr).unwrap();

        // Phase C — kick. The drain loop must detect the poison,
        // bump the counter, set the flag, and bail without looping.
        let pre_reads = dev.counters().reads_completed();
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);
        assert_eq!(
            dev.counters().invalid_avail_idx_count(),
            1,
            "first hostile-idx kick must bump invalid_avail_idx_count exactly once",
        );
        assert_eq!(
            dev.counters().reads_completed(),
            pre_reads,
            "no reads must be serviced — the poisoned queue is structurally broken",
        );

        // Phase D — re-kick the poisoned queue. The early-return
        // gate at the top of drain_bracket_impl must short-circuit
        // before re-reading avail.idx, so the counter does NOT
        // re-bump.
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);
        assert_eq!(
            dev.counters().invalid_avail_idx_count(),
            1,
            "subsequent kicks against a poisoned queue MUST NOT \
             re-bump the counter — the per-event semantics rely on \
             the queue_poisoned flag short-circuiting before the \
             iter() call",
        );

        // Phase E — virtio reset clears the poison. Model the
        // guest's re-bind: zero avail.idx and used.idx in guest
        // memory (per virtio-v1.2 §2.7.6/§2.7.8 ring layouts), walk
        // the FSM back to DRIVER_OK, plant a fresh chain, and kick.
        // The drain must service the chain normally — no poison,
        // no counter bumps for InvalidAvailRingIndex.
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, 0);
        let used_idx_addr = mock.used_addr().checked_add(2).unwrap();
        mem.write_obj(0u16, avail_idx_addr).unwrap();
        mem.write_obj(0u16, used_idx_addr).unwrap();
        // Plant a fresh status sentinel so we can detect the
        // post-reset write.
        mem.write_slice(&[0xEEu8], status_addr).unwrap();
        // Re-build the chain. With avail.idx zeroed,
        // build_desc_chain stores the chain at avail.ring[0] and
        // bumps avail.idx to 1 — what a freshly re-bound guest
        // does.
        mock.build_desc_chain(&descs)
            .expect("build chain post-reset");
        wire_device_to_mock(&mut dev, &mock);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

        let mut s = [0u8; 1];
        mem.read_slice(&mut s, status_addr).unwrap();
        assert_eq!(
            s[0], VIRTIO_BLK_S_OK as u8,
            "post-reset chain must complete S_OK — the queue_poisoned \
             flag must have cleared in reset_engine_inline",
        );
        assert_eq!(
            dev.counters().reads_completed(),
            pre_reads + 1,
            "post-reset chain must bump reads_completed",
        );
        // The cumulative counter for poison events persists across
        // reset — operators need lifetime-event visibility to detect
        // repeated hostile behavior.
        assert_eq!(
            dev.counters().invalid_avail_idx_count(),
            1,
            "invalid_avail_idx_count is cumulative across reset; only \
             the per-worker poison flag clears",
        );
    }
