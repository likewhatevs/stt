//! Device-side virtio-block: MMIO dispatch, FSM, counters, request
//! state, and `Drop`. Extracted from `mod.rs` so the device code lives
//! beside its tests rather than scattered through the worker and
//! throttle modules.
//!
//! See the parent module `super` for the full execution-model and
//! "why" doc — the module-level rationale for why
//! `add_used` is gated on status-write success, why throttle stalls
//! roll back the chain and arm a timerfd, and the backing-speed
//! caveat — lives there.

pub(crate) use std::fs::File;
use std::os::unix::fs::FileExt;
#[cfg(not(test))]
use std::os::unix::io::AsRawFd;
pub(crate) use std::sync::Arc;
pub(crate) use std::sync::OnceLock;
pub(crate) use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc;
use std::thread;
pub(crate) use std::time::Duration;

pub(crate) use virtio_bindings::virtio_blk::{
    VIRTIO_BLK_F_BLK_SIZE, VIRTIO_BLK_F_FLUSH, VIRTIO_BLK_F_RO, VIRTIO_BLK_F_SEG_MAX,
    VIRTIO_BLK_F_SIZE_MAX, VIRTIO_BLK_ID_BYTES, VIRTIO_BLK_S_IOERR, VIRTIO_BLK_S_OK,
    VIRTIO_BLK_S_UNSUPP, VIRTIO_BLK_T_FLUSH, VIRTIO_BLK_T_GET_ID, VIRTIO_BLK_T_IN,
    VIRTIO_BLK_T_OUT,
};
pub(crate) use virtio_bindings::virtio_config::{
    VIRTIO_CONFIG_S_ACKNOWLEDGE, VIRTIO_CONFIG_S_DRIVER, VIRTIO_CONFIG_S_DRIVER_OK,
    VIRTIO_CONFIG_S_FEATURES_OK, VIRTIO_CONFIG_S_NEEDS_RESET, VIRTIO_F_VERSION_1,
};
pub(crate) use virtio_bindings::virtio_ids::VIRTIO_ID_BLOCK;
pub(crate) use virtio_bindings::virtio_mmio::{
    VIRTIO_MMIO_CONFIG_GENERATION, VIRTIO_MMIO_DEVICE_FEATURES, VIRTIO_MMIO_DEVICE_FEATURES_SEL,
    VIRTIO_MMIO_DEVICE_ID, VIRTIO_MMIO_DRIVER_FEATURES, VIRTIO_MMIO_DRIVER_FEATURES_SEL,
    VIRTIO_MMIO_INT_CONFIG, VIRTIO_MMIO_INT_VRING, VIRTIO_MMIO_INTERRUPT_ACK,
    VIRTIO_MMIO_INTERRUPT_STATUS, VIRTIO_MMIO_MAGIC_VALUE, VIRTIO_MMIO_QUEUE_AVAIL_HIGH,
    VIRTIO_MMIO_QUEUE_AVAIL_LOW, VIRTIO_MMIO_QUEUE_DESC_HIGH, VIRTIO_MMIO_QUEUE_DESC_LOW,
    VIRTIO_MMIO_QUEUE_NOTIFY, VIRTIO_MMIO_QUEUE_NUM, VIRTIO_MMIO_QUEUE_NUM_MAX,
    VIRTIO_MMIO_QUEUE_READY, VIRTIO_MMIO_QUEUE_SEL, VIRTIO_MMIO_QUEUE_USED_HIGH,
    VIRTIO_MMIO_QUEUE_USED_LOW, VIRTIO_MMIO_STATUS, VIRTIO_MMIO_VENDOR_ID, VIRTIO_MMIO_VERSION,
};
pub(crate) use virtio_bindings::virtio_ring::VIRTIO_RING_F_EVENT_IDX;
pub(crate) use virtio_queue::Error as VirtioQueueError;
#[cfg(test)]
use virtio_queue::Queue;
pub(crate) use virtio_queue::QueueOwnedT;
#[cfg(not(test))]
use virtio_queue::QueueSync;
pub(crate) use virtio_queue::QueueT;
pub(crate) use vm_memory::{ByteValued, Bytes, GuestAddress, GuestMemoryMmap};
// `EpollEvent` / `EventSet` are re-exported because tests for the
// always-compiled `worker_dispatch_event` helper construct EventSet
// values directly via `super::*`, and the helper itself accepts an
// EventSet argument. clippy --lib doesn't see the test consumers,
// so the re-export looks unused without the allow.
#[allow(unused_imports)]
pub(crate) use vmm_sys_util::epoll::{EpollEvent, EventSet};
pub(crate) use vmm_sys_util::eventfd::EventFd;

pub(crate) use super::super::disk_config::DiskThrottle;

pub(crate) const MMIO_MAGIC: u32 = 0x7472_6976; // "virt" in LE
pub(crate) const MMIO_VERSION: u32 = 2; // virtio 1.x MMIO
pub(crate) const VENDOR_ID: u32 = 0;

/// MMIO region size: 4 KB (one page).
pub const VIRTIO_MMIO_SIZE: u64 = 0x1000;

/// Single request queue. virtio-spec §5.2.2 declares one request
/// queue plus optional multiqueue (`VIRTIO_BLK_F_MQ`); MQ deferred.
pub(crate) const NUM_QUEUES: usize = 1;
pub(crate) const QUEUE_MAX_SIZE: u16 = 256;
pub(crate) const REQ_QUEUE: usize = 0;

/// Queue type used for the request virtqueue. Production uses
/// `QueueSync` (`Arc<Mutex<Queue>>` internally) so the vCPU thread
/// (MMIO config writes — set_size/ready/desc/avail/used addresses)
/// and the dedicated worker thread (drain_bracket_impl — pop, add_used,
/// needs_notification) can share the same queue state safely. Tests
/// run drain_bracket_impl inline on the caller thread so the bare `Queue`
/// (single-threaded, no internal lock) is sufficient and avoids
/// changing the test surface that drives `Queue` methods directly
/// (`disable_notification`, `set_avail_ring_address`, etc.).
///
/// The `QueueT` trait is the single API both implementations honour;
/// every drain-side method this file calls (pop_descriptor_chain,
/// add_used, disable/enable_notification, needs_notification,
/// avail_ring, event_idx_enabled) is part of `QueueT`. Generic
/// helpers like `publish_completion` are bound by `QueueT` so they
/// compile against either alias without further indirection.
#[cfg(not(test))]
pub(crate) type BlkQueue = QueueSync;
#[cfg(test)]
pub(crate) type BlkQueue = Queue;

/// Logical block size advertised to the guest. 512 bytes matches
/// the virtio spec default.
pub const VIRTIO_BLK_SECTOR_SIZE: u32 = 512;

/// Default capacity (256 MB) used by virtio_blk tests. Mirrors the
/// 256-MB default in [`super::disk_config::DiskConfig::default`].
///
/// Sized for `mkfs.btrfs` minimum without `--mixed`: btrfs needs
/// ~109 MB for single-profile metadata and ~256 MB if it picks DUP
/// metadata (which is the default on a single-device fs). Sized
/// below 256 MB risks `mkfs.btrfs` failing at template-build time.
///
/// `dead_code` allow: only consumed by `#[cfg(test)]` modules
/// (every virtio_blk test fixture passes this as the device's
/// capacity); clippy --lib doesn't see those references.
#[allow(dead_code)]
pub const VIRTIO_BLK_DEFAULT_CAPACITY_BYTES: u64 = 256 * 1024 * 1024;

/// Maximum number of data segments per request the device supports.
/// virtio-v1.2 §5.2.4: `seg_max` is the max scatter-gather buffer
/// count, exclusive of the header and status descriptors. Without
/// `F_SEG_MAX` the guest defaults `max_segments` to 1, which forces
/// `bio_split` and serializes large requests; advertising 128 is the
/// firecracker default and ample for the small files this device
/// targets.
pub(crate) const VIRTIO_BLK_SEG_MAX: u32 = 128;

/// Maximum size in bytes of a single descriptor's data buffer.
/// virtio-v1.2 §5.2.4 (`size_max`): caps per-descriptor length so a
/// guest can't submit a single 4 GB descriptor and force the device
/// to allocate a matching `Vec<u8>` for `read_at`/`write_at`. 1 MB
/// matches firecracker's default and is far above what the guest's
/// blk-mq layer typically generates (max_sectors_kb defaults to
/// 512 KB). Without `F_SIZE_MAX` the guest treats per-descriptor
/// length as unbounded — host OOM hazard on a hostile guest.
pub(crate) const VIRTIO_BLK_SIZE_MAX: u32 = 1 << 20;

/// Device serial number returned by `VIRTIO_BLK_T_GET_ID`. Per
/// virtio-v1.2 §5.2.6.4 (and `virtio_blk.h` `VIRTIO_BLK_ID_BYTES`)
/// the kernel driver passes a 20-byte buffer (`virtblk_get_id` →
/// `blk_rq_map_kern(req, id_str, VIRTIO_BLK_ID_BYTES, GFP_KERNEL)`,
/// drivers/block/virtio_blk.c). The string is exposed at
/// `/sys/block/<dev>/serial` after `serial_show` reads it from the
/// device. The 16-byte payload `ktstr-virtio-blk` is null-padded to
/// 20 bytes; the trailing zeros let `serial_show`'s
/// `strlen(buf)` (after the kernel's `buf[VIRTIO_BLK_ID_BYTES] =
/// '\0'` sentinel) terminate at the first NUL.
pub(crate) const VIRTIO_BLK_SERIAL: [u8; VIRTIO_BLK_ID_BYTES as usize] =
    *b"ktstr-virtio-blk\0\0\0\0";

/// Request out-header. virtio-v1.2 §5.2.6: every request chain
/// starts with a device-readable, 16-byte header carrying the
/// request type, ioprio (ignored), and starting sector. The struct
/// matches virtio_bindings::virtio_blk::virtio_blk_outhdr field for
/// field — it is redeclared here so we can attach `ByteValued` (the
/// bindings struct does not implement it) and use `Bytes::read_obj`
/// directly. `repr(C)` + integer-only fields satisfy the
/// `ByteValued` invariants (§ vm-memory bytes.rs trait docs).
#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
pub(crate) struct VirtioBlkOutHdr {
    /// `VIRTIO_BLK_T_*`. LE per virtio-v1.2 §5.2.6.
    pub(crate) type_: u32,
    /// I/O priority, ignored on this device.
    pub(crate) _ioprio: u32,
    /// Starting sector (512-byte units).
    pub(crate) sector: u64,
}

// SAFETY: VirtioBlkOutHdr is `repr(C)`, contains only `u32` and `u64`
// (themselves `ByteValued`), has no padding (4+4+8 = 16, all aligned),
// and any byte pattern is a valid value (the type/ioprio fields are
// validated separately by the request dispatcher; sector is just a
// number). All `ByteValued` requirements are met.
unsafe impl vm_memory::ByteValued for VirtioBlkOutHdr {}

/// Header size for `VirtioBlkOutHdr`. virtio-v1.2 §5.2.6:
/// type:u32, ioprio:u32, sector:u64.
pub(crate) const VIRTIO_BLK_OUTHDR_SIZE: usize = std::mem::size_of::<VirtioBlkOutHdr>();

/// Legacy CHS geometry sub-struct of `VirtioBlkConfig`, gated on
/// `VIRTIO_BLK_F_GEOMETRY`. Mirrors the kernel uapi
/// `struct virtio_blk_geometry` (cylinders:u16, heads:u8, sectors:u8 —
/// 4 bytes total) at config-space offset 0x10. We don't advertise
/// `F_GEOMETRY` so the field is left zero; the guest driver reads it
/// via `virtio_cread_feature`, which returns `-ENOENT` when the
/// feature bit is not negotiated and the read is skipped.
#[repr(C, packed)]
#[derive(Copy, Clone, Default, Debug)]
pub(crate) struct VirtioBlkGeometry {
    pub(crate) cylinders: u16,
    pub(crate) heads: u8,
    pub(crate) sectors: u8,
}

/// Block-device config space (virtio-v1.2 §5.2.4). Mirrors the kernel
/// uapi `struct virtio_blk_config` field-for-field up through
/// `blk_size` (the last field whose feature bit this device
/// advertises). Trailing fields (topology, MQ, discard, write-zeroes,
/// secure-erase, zoned) are gated on feature bits we don't advertise,
/// so the guest driver's `virtio_cread_feature` returns `-ENOENT` for
/// those reads and never depends on the device-side bytes — we serve
/// zeros for any read past `size_of::<VirtioBlkConfig>()`, matching
/// virtio-v1.2 §4.2.2.2 ("reads past the populated config layout
/// return zero").
///
/// The kernel struct is `__attribute__((packed))` (see
/// `include/uapi/linux/virtio_blk.h`), so this redeclaration uses
/// `repr(C, packed)` to match the wire layout byte-for-byte. Without
/// the `packed` attribute the compiler would insert padding after
/// `seg_max` to align `geometry` (which contains a `u16`) — that
/// padding would shift `blk_size` from offset 0x14 to 0x18 and serve
/// the guest a wrong block-size value silently.
#[repr(C, packed)]
#[derive(Copy, Clone, Default, Debug)]
pub(crate) struct VirtioBlkConfig {
    /// Capacity in 512-byte sectors. Always populated; the kernel
    /// driver reads this unconditionally (no feature bit gates it).
    pub(crate) capacity: u64,
    /// Maximum per-descriptor data length, gated on
    /// `VIRTIO_BLK_F_SIZE_MAX`.
    pub(crate) size_max: u32,
    /// Maximum scatter-gather segments per request, gated on
    /// `VIRTIO_BLK_F_SEG_MAX`.
    pub(crate) seg_max: u32,
    /// Legacy CHS geometry, gated on `VIRTIO_BLK_F_GEOMETRY`. We
    /// don't advertise that bit so this field is left zero.
    pub(crate) geometry: VirtioBlkGeometry,
    /// Logical block size, gated on `VIRTIO_BLK_F_BLK_SIZE`.
    pub(crate) blk_size: u32,
}

// SAFETY: `VirtioBlkConfig` and `VirtioBlkGeometry` are
// `repr(C, packed)`. With `packed` the alignment is 1 and there is no
// inter-field padding by definition (every field is byte-aligned). All
// fields are integer types (`u8`, `u16`, `u32`, `u64`) for which every
// bit pattern is a valid value, so reading arbitrary bytes into the
// struct yields a well-defined value. The struct is `Copy`, `Send`,
// and `Sync` (all primitives), satisfying the `ByteValued` supertrait
// bounds. Total size is verified against the kernel uapi layout by
// the `VIRTIO_BLK_CONFIG_SIZE` const assertion below.
unsafe impl vm_memory::ByteValued for VirtioBlkConfig {}
// SAFETY: same justification as `VirtioBlkConfig`. `VirtioBlkGeometry`
// is `repr(C, packed)` with three integer fields (`u16`, `u8`, `u8`),
// no padding, all bit patterns valid, `Copy + Send + Sync`.
unsafe impl vm_memory::ByteValued for VirtioBlkGeometry {}

/// Size of the populated portion of block config space (24 bytes:
/// capacity 8 + size_max 4 + seg_max 4 + geometry 4 + blk_size 4).
/// Reads at config-space offsets `>= VIRTIO_BLK_CONFIG_SIZE` return
/// zero per virtio-v1.2 §4.2.2.2.
pub(crate) const VIRTIO_BLK_CONFIG_SIZE: usize = std::mem::size_of::<VirtioBlkConfig>();
// Compile-time check that the struct layout matches the kernel uapi
// byte budget (8+4+4+4+4 = 24). A mismatch here means either Rust's
// `repr(C, packed)` introduced a divergence from the kernel's
// `__attribute__((packed))` layout, or a field was added/removed —
// in either case the guest would read garbage from a misaligned
// field. Failing to compile is preferable to silently serving wrong
// bytes.
const _: () = assert!(VIRTIO_BLK_CONFIG_SIZE == 24);
// Field-offset checks: the kernel driver reads each field at a
// specific offset via `virtio_cread`. If `repr(C, packed)` ever
// drifts from the kernel's `__attribute__((packed))` layout, these
// asserts catch it at compile time before a wrong-offset bug ships
// to the guest.
const _: () = assert!(std::mem::offset_of!(VirtioBlkConfig, capacity) == 0x00);
const _: () = assert!(std::mem::offset_of!(VirtioBlkConfig, size_max) == 0x08);
const _: () = assert!(std::mem::offset_of!(VirtioBlkConfig, seg_max) == 0x0C);
const _: () = assert!(std::mem::offset_of!(VirtioBlkConfig, geometry) == 0x10);
const _: () = assert!(std::mem::offset_of!(VirtioBlkConfig, blk_size) == 0x14);

/// One descriptor from a virtio request chain. Used uniformly for
/// every chain role — header, data segments, and status — so the
/// chain-walk code can collect all descriptors into one buffer and
/// then index by position (first = header, middle = data, last =
/// status).
///
/// `is_write_only` mirrors `desc.is_write_only()` from
/// `virtio_queue` (i.e. VRING_DESC_F_WRITE set in the descriptor):
/// device-writable for read data segments and the status byte;
/// device-readable for the request header and write data segments.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ChainDescriptor {
    pub(crate) addr: GuestAddress,
    pub(crate) len: u32,
    pub(crate) is_write_only: bool,
}

/// Status bits required before each phase (mirrors virtio_console).
pub(crate) const S_ACK: u32 = VIRTIO_CONFIG_S_ACKNOWLEDGE;
pub(crate) const S_DRV: u32 = S_ACK | VIRTIO_CONFIG_S_DRIVER;
pub(crate) const S_FEAT: u32 = S_DRV | VIRTIO_CONFIG_S_FEATURES_OK;
/// Test helper — terminal state bits with DRIVER_OK set.
#[cfg(test)]
pub(crate) const S_OK: u32 = S_FEAT | VIRTIO_CONFIG_S_DRIVER_OK;

// Token-bucket throttle primitives live in `throttle`. Pulled out as
// a submodule so the throttle's adversarial test surface (which is
// the most-exercised piece of the device) sits next to its tests
// rather than scattered through the device + worker code. See
// `throttle.rs` for the full type-level rationale.
use super::throttle::*;
#[cfg(not(test))]
use super::worker::worker_thread_main;

/// Publish a chain completion: write the status byte and, on
/// success, mark the chain used. Returns `true` if the device
/// should signal the guest (used-ring index advanced); `false`
/// otherwise.
///
/// Status-write-success gate: `add_used` is called ONLY after a
/// successful status-byte write. Publishing a completion the guest
/// can't observe (status write failed but used-ring advanced) would
/// let the guest's `virtblk_done` read its `vbr->in_hdr.status`
/// byte that's stale from prior blk-mq tag use (initially zero from
/// `__GFP_ZERO` at allocation, stale on reuse) as `BLK_STS_OK`
/// (drivers/block/virtio_blk.c `virtblk_vbr_status` +
/// `virtblk_result(0)`) — silent data corruption for reads, silent
/// dropped writes for writes. On status-write failure the chain
/// stays in the avail ring; virtio_blk has no `mq_ops->timeout`
/// (drivers/block/virtio_blk.c `virtio_mq_ops`), so the guest hangs
/// on this request until `kernel.hung_task_timeout_secs` (default
/// 120 s) fires or a higher-layer retries.
///
/// `used_len` is what `add_used` records as the "bytes written by
/// the device into guest memory". Error paths pass `1` (just the
/// status byte). The success path passes the data-bytes-written
/// total + 1 (for reads) or `1` (for writes/flushes — the device
/// wrote no data back).
///
/// `label` is included in any tracing::warn from this function so
/// operators can identify which gate triggered the publish.
///
/// `too_many_arguments` allow: every parameter is independent
/// per-request state (queue/memory binding, head index, status
/// address+byte, used-len, label) sourced from a different point
/// in the chain-handling pipeline. Bundling would build a struct
/// for one call seam.
#[allow(clippy::too_many_arguments)]
pub(crate) fn publish_completion<Q: QueueT>(
    mem: &GuestMemoryMmap,
    q: &mut Q,
    counters: &VirtioBlkCounters,
    head: u16,
    status_addr: GuestAddress,
    status_byte: u8,
    used_len: u32,
    label: &'static str,
) -> bool {
    if mem.write_slice(&[status_byte], status_addr).is_err() {
        // Status-byte write failed — the chain stays in the avail
        // ring. virtio_blk has no `mq_ops->timeout`, so the guest
        // hangs on this request until `kernel.hung_task_timeout_secs`
        // (default 120 s) fires or a higher-layer retries.
        // Bump io_errors so the host operator sees a counter for
        // every silent-stall event. Error-site callers also bump
        // io_errors before reaching here; the double-count is
        // intentional under hostile-guest scenarios — a guest
        // constructing chains with systematically unmapped
        // status_addr will double-count every request, but the
        // silent-stall it prevents on the success path is the
        // worse failure mode. Silent-swallow on the success path
        // (FLUSH or T_IN/T_OUT/T_GET_ID succeeded but the status
        // descriptor itself is unmapped) would otherwise produce
        // a host-side silent stall — virtio_blk has no
        // `mq_ops->timeout` callback, so blk-mq alone never
        // surfaces the unpublished request as an error; only the
        // guest's hung-task watchdog fires (default 120 s) — and
        // without this counter bump operators would have no
        // host-side signal until the watchdog message hits dmesg.
        counters.record_io_error();
        return false;
    }
    match q.add_used(mem, head, used_len) {
        Ok(()) => true,
        Err(e) => {
            tracing::warn!(head, %e, label, "virtio-blk add_used failed");
            counters.record_io_error();
            false
        }
    }
}

// ----------------------------------------------------------------------------
// Counters (host-side observability)
// ----------------------------------------------------------------------------

/// Per-device counters surfaced to the host monitor. All atomic so
/// the monitor can read them without locking the device struct.
///
/// Mutation goes through the `record_*` helper methods, NOT direct
/// `field.fetch_add(...)` calls. The helpers enforce the
/// "completion + bytes" pairing for reads and writes — every
/// `record_read(bytes)` increments both `reads_completed` AND
/// `bytes_read` in one call. A bare `reads_completed.fetch_add(1)`
/// without a paired `bytes_read.fetch_add(n)` would let the
/// failure-dump renderer compute a misleading bytes-per-op
/// average. The helpers also keep the call site one line each,
/// matching the SPSC-style accounting common in network/block
/// device fast paths.
///
/// Fields are `pub(crate)` so the helper-mutation rule is enforced
/// across the crate by visibility. External consumers reach in via
/// the per-field `pub fn` accessors below — each performs a
/// `Relaxed` load and returns the current value as `u64`.
///
/// # Counter taxonomy: events vs requests vs gauges
///
/// Counters fall into three semantic categories. Operators
/// reading the failure-dump must understand which is which to
/// avoid drawing wrong conclusions:
///
/// - **Per-event cumulative counters** (`io_errors`,
///   `throttled_count`): bumped each time the underlying event
///   fires, with no per-request deduplication. A single hostile
///   request can produce multiple `io_errors` bumps if it trips
///   several gates in sequence (see `io_errors` doc below for the
///   double-bump scenarios). Use these to compare event rates
///   over time, not to count requests.
/// - **Per-request cumulative counters** (`reads_completed`,
///   `writes_completed`, `flushes_completed`, `bytes_read`,
///   `bytes_written`): bumped exactly once per successfully
///   serviced request. Each surfaces a one-to-one mapping with
///   guest-observable completions. Use these to compute
///   throughput, average request size, and per-direction IO
///   share.
/// - **Per-request live gauges** (`currently_throttled_gauge`):
///   "how many requests are RIGHT NOW in this state." Increments
///   when a request enters the state, decrements when it exits.
///   The cumulative event counter for the same condition lives
///   in `throttled_count` (events, not requests). Reading
///   `currently_throttled_gauge == 5` means 5 chains are pinned
///   in the avail ring at this instant; `throttled_count == 100`
///   over the same period means 100 stall events have occurred.
///   The two answer different questions and operators MUST NOT
///   compare or sum them.
///
/// # Lifetime semantics
///
/// Counters are **cumulative for the device's lifetime** —
/// `VirtioBlk::reset()` does NOT zero them. A guest issuing
/// STATUS=0 (driver re-bind) re-uses the existing counter Arc; an
/// operator monitoring `reads_completed` etc. observes a
/// monotonically non-decreasing series across resets. Only
/// destruction of the device (`Drop`) reclaims the counters Arc.
/// This matches operator expectation that failure-dump counters
/// reflect the device's full IO history, not just the post-reset
/// fragment.
///
/// Per-request live gauges (`currently_throttled_gauge`) decrement
/// across the device's lifetime as requests exit the gauged
/// state, but the gauge value itself is "right now," not
/// cumulative. A reset that strands a chain in the
/// "currently throttled" state would leak the gauge increment;
/// the production reset path joins the worker thread before
/// rebuilding the queue, and the worker decrements the gauge on
/// any subsequent successful drain — but a worker that never
/// observes a successful drain (e.g. the device is destroyed
/// while the chain is still rolled back) leaves the increment
/// pinned for the device's lifetime. This is acceptable because
/// the gauge is informational and the device is going away
/// anyway; downstream consumers must not depend on a strictly
/// zero-on-shutdown property.
///
/// We diverge from virtio-v1.2 §2.1 ("device returned to its
/// initial state") for counters because operator-side
/// failure-dump observability requires cumulative IO history
/// spanning the device's full lifetime, not just the post-reset
/// fragment.
#[derive(Debug, Default)]
pub struct VirtioBlkCounters {
    pub(crate) reads_completed: AtomicU64,
    pub(crate) writes_completed: AtomicU64,
    pub(crate) flushes_completed: AtomicU64,
    pub(crate) bytes_read: AtomicU64,
    pub(crate) bytes_written: AtomicU64,
    /// Cumulative throttle-stall **events** for the device's
    /// lifetime. Bumped each time `drain_bracket_impl` returns
    /// `DrainOutcome::ThrottleStalled`. A single chain that
    /// stalls, refills, stalls again, and finally completes
    /// produces TWO `throttled_count` bumps but ONE
    /// `reads_completed` (or `writes_completed`/etc.) bump on
    /// final success.
    ///
    /// To answer "how many requests are stuck right now," read
    /// `currently_throttled_gauge` instead — the per-event
    /// cumulative counter and the per-request live gauge are
    /// distinct semantics and answer different questions.
    pub(crate) throttled_count: AtomicU64,
    pub(crate) io_errors: AtomicU64,
    /// Live "how many requests are currently waiting for tokens"
    /// gauge. Incremented when a chain transitions into the
    /// stalled state; decremented when the next successful drain
    /// confirms the chain has been serviced.
    ///
    /// On a single-queue virtio-blk device the gauge is bounded
    /// at 0 or 1 in practice — only the head-of-queue chain can
    /// be stalled at a time, because the FIFO drain rolls back
    /// the popped chain on stall and the next successful drain
    /// always processes that same chain first before any newer
    /// arrivals. A multi-queue extension would lift the bound to
    /// "1 per queue currently stalled."
    ///
    /// Distinct from `throttled_count` (cumulative events): the
    /// gauge tracks the live state, the counter tracks the
    /// historical event rate. See the type-level "Counter
    /// taxonomy" doc for why operators must not conflate the
    /// two.
    pub(crate) currently_throttled_gauge: AtomicU64,
    /// Cumulative count of `Error::InvalidAvailRingIndex` events
    /// observed by `drain_bracket_impl`. Bumped each time the
    /// virtio-queue iter() rejects an avail.idx whose distance
    /// from `next_avail` exceeds the queue size — a hostile or
    /// buggy guest condition that, if not detected, would loop
    /// the worker forever (the swallowed-error livelock fixed by
    /// the queue_poisoned gate).
    ///
    /// Per-event counter (NOT per-request): a single drain pass
    /// produces at most one bump (the poison flag short-circuits
    /// further attempts on the same queue). Successive
    /// QUEUE_NOTIFY kicks against an unresetted poisoned queue
    /// take the early-return path and produce zero additional
    /// bumps until the guest performs a virtio reset.
    pub(crate) invalid_avail_idx_count: AtomicU64,
}

impl VirtioBlkCounters {
    /// Record one completed read: bumps `reads_completed` and adds
    /// `bytes` to `bytes_read`. The pairing is enforced — bare
    /// reads_completed bumps without the paired bytes_read add are
    /// caught at refactor time.
    pub(crate) fn record_read(&self, bytes: u64) {
        self.reads_completed.fetch_add(1, Ordering::Relaxed);
        self.bytes_read.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record one completed write: bumps `writes_completed` and
    /// adds `bytes` to `bytes_written`.
    pub(crate) fn record_write(&self, bytes: u64) {
        self.writes_completed.fetch_add(1, Ordering::Relaxed);
        self.bytes_written.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record one completed flush.
    pub(crate) fn record_flush(&self) {
        self.flushes_completed.fetch_add(1, Ordering::Relaxed);
    }

    /// Bumped on every host-observed IO failure **event**, whether
    /// the guest saw S_IOERR or not (e.g. unmapped status-byte
    /// address that prevented the status write). Covers spec
    /// violations, backend IO errors, malformed chains, add_used
    /// failures, and status-write failures where the chain stays
    /// in the avail ring (no S_IOERR ever reaches the guest, but
    /// the host still counts the silent-stall event).
    ///
    /// # Events, not requests
    ///
    /// `io_errors` is an **events** counter, not a per-request
    /// counter. A single hostile request can produce multiple
    /// `io_errors` bumps if it trips several gates in sequence.
    /// Concretely:
    ///
    /// - **Pre-publish gates that bump io_errors then call
    ///   `publish_completion`**: SEG_MAX reject, bad header,
    ///   header-read failure, SIZE_MAX reject, zero-data,
    ///   sub-sector data_len, direction violation. Each of these
    ///   records one io_errors event for the validation
    ///   rejection. If the subsequent `publish_completion`'s
    ///   status-byte write or `add_used` then fails (e.g. the
    ///   guest also placed the status descriptor at unmapped
    ///   GPA), `publish_completion` records a SECOND io_errors
    ///   event for the silent-stall failure mode. A pathological
    ///   chain with a malformed header AND an unmapped status
    ///   descriptor surfaces as `io_errors += 2` for one chain.
    /// - **Handler error paths**: `handle_read_impl` /
    ///   `handle_write_impl` / `handle_get_id_impl` /
    ///   `handle_flush_impl` each record io_errors on backing-file
    ///   error or guest-memory access failure. The handler
    ///   produces an S_IOERR status which `process_requests`
    ///   passes to `publish_completion`. If the status-write or
    ///   add_used then fails, `publish_completion` records a
    ///   SECOND io_errors event for that request.
    /// - **publish_completion's own failure modes**: status-write
    ///   failure or add_used failure each record one io_errors
    ///   event independently of any prior caller bump.
    ///
    /// The double-bump under hostile-guest scenarios is
    /// **intentional**. Hoisting all error bumps to a single
    /// outermost catch site would lose the "silent-stall failure
    /// distinct from validation rejection" signal: an operator
    /// reading io_errors needs to see a separate event each time
    /// the device hits a failure mode, even if multiple events
    /// happen on the same request.
    ///
    /// Operators who want a per-request error count must not
    /// derive it from io_errors — they need a separate counter
    /// (deliberately not provided here; the per-request semantic
    /// is reachable via `reads_completed + writes_completed +
    /// flushes_completed` for the success side, with the failure
    /// side inferable from `total_chains_observed - success_count`
    /// once a `total_chains_observed` counter is added).
    ///
    /// See also `currently_throttled_gauge` (per-request live
    /// gauge) and `throttled_count` (per-event cumulative
    /// counter) for the throttle-side distinction; the same
    /// events-vs-requests split applies there.
    pub(crate) fn record_io_error(&self) {
        self.io_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Record one throttle-stall **event**. virtio-spec doesn't
    /// reserve a "throttled" status code; on stall the device
    /// rolls back the pop and arms a retry timer (see
    /// `drain_bracket_impl` and `worker_thread_main`) — the chain
    /// stays invisible to the guest until enough tokens refill.
    /// Retry fires within `RETRY_TIMER_MAX_NANOS` (1 s);
    /// pathological refill rates re-stall at the cap. The
    /// counter is separate from `io_errors` so operators can
    /// distinguish "real IO problem" from "throttle bucket
    /// drained, request deferred."
    ///
    /// # Events, not requests
    ///
    /// `throttled_count` is the cumulative event rate, not the
    /// number of stuck requests. A single chain that stalls
    /// twice (initial stall + premature retry that re-stalls)
    /// bumps `throttled_count` twice but represents one stuck
    /// request. To answer "how many requests are stuck right
    /// now," read `currently_throttled_gauge` instead.
    pub(crate) fn record_throttled(&self) {
        self.throttled_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the live "currently waiting for tokens" gauge.
    /// Called by `drain_bracket_impl` when a chain transitions
    /// from "running" to "stalled" — i.e. the per-worker
    /// `currently_stalled` flag was false before this stall.
    /// Idempotent stall observations (same chain, multiple
    /// retries that all re-stall) MUST NOT double-increment; the
    /// caller gates this on the per-worker flag transition.
    pub(crate) fn record_throttle_pending_inc(&self) {
        self.currently_throttled_gauge
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement the live "currently waiting for tokens" gauge.
    /// Called by `drain_bracket_impl` when the worker observes a
    /// successful drain after a prior stall — i.e. the
    /// per-worker `currently_stalled` flag was true before this
    /// drain finished without re-stalling. Saturating sub on the
    /// underlying AtomicU64 would be safer against
    /// double-decrement bugs, but the per-worker flag gates the
    /// transition so a paired inc precedes every dec; a vanilla
    /// `fetch_sub(1)` is correct under that invariant.
    pub(crate) fn record_throttle_pending_dec(&self) {
        self.currently_throttled_gauge
            .fetch_sub(1, Ordering::Relaxed);
    }

    /// Record one observed `Error::InvalidAvailRingIndex` event
    /// from `Queue::iter`. Called by `drain_bracket_impl` when the
    /// avail ring's `idx` is more than `queue.size` ahead of
    /// `next_avail` — a virtio-spec violation by the guest. The
    /// caller also sets `BlkWorkerState::queue_poisoned` so a
    /// single hostile-guest event produces exactly one bump,
    /// regardless of how many subsequent kicks land before the
    /// next reset (subsequent drains short-circuit on the poison
    /// flag and never re-call `iter`).
    pub(crate) fn record_invalid_avail_idx(&self) {
        self.invalid_avail_idx_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Read the cumulative count of successfully completed read
    /// requests for this device's lifetime. Per-request counter:
    /// bumped exactly once per successful read via
    /// [`Self::record_read`] (paired with a `bytes_read` add).
    /// `Relaxed` ordering matches the writer side — counters are
    /// publish-only observability and do not establish
    /// happens-before with other operations.
    pub fn reads_completed(&self) -> u64 {
        self.reads_completed.load(Ordering::Relaxed)
    }

    /// Read the cumulative count of successfully completed write
    /// requests for this device's lifetime. Per-request counter:
    /// bumped exactly once per successful write via
    /// [`Self::record_write`] (paired with a `bytes_written` add).
    pub fn writes_completed(&self) -> u64 {
        self.writes_completed.load(Ordering::Relaxed)
    }

    /// Read the cumulative count of successfully completed flush
    /// requests for this device's lifetime. Per-request counter:
    /// bumped once per successful flush via
    /// [`Self::record_flush`].
    pub fn flushes_completed(&self) -> u64 {
        self.flushes_completed.load(Ordering::Relaxed)
    }

    /// Read the cumulative number of bytes successfully read from
    /// the backing file and delivered to the guest. Per-request
    /// counter: incremented in lockstep with `reads_completed`.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read.load(Ordering::Relaxed)
    }

    /// Read the cumulative number of bytes successfully written
    /// from guest memory to the backing file. Per-request counter:
    /// incremented in lockstep with `writes_completed`.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written.load(Ordering::Relaxed)
    }

    /// Read the cumulative count of throttle-stall **events** for
    /// this device's lifetime. Per-event counter (NOT per-request):
    /// a single chain that stalls multiple times produces multiple
    /// bumps. To answer "how many requests are stuck right now,"
    /// read [`Self::currently_throttled_gauge`] instead.
    pub fn throttled_count(&self) -> u64 {
        self.throttled_count.load(Ordering::Relaxed)
    }

    /// Read the cumulative count of host-observed IO failure
    /// **events**. Per-event counter (NOT per-request): a single
    /// hostile chain can produce multiple bumps if it trips
    /// several gates in sequence. See [`Self::record_io_error`]
    /// for the double-bump scenarios.
    pub fn io_errors(&self) -> u64 {
        self.io_errors.load(Ordering::Relaxed)
    }

    /// Read the live "how many requests are currently waiting for
    /// throttle tokens" gauge. NOT cumulative — increments when a
    /// chain enters the stalled state, decrements when it exits.
    /// On a single-queue device the value is bounded at 0 or 1 in
    /// practice.
    pub fn currently_throttled_gauge(&self) -> u64 {
        self.currently_throttled_gauge.load(Ordering::Relaxed)
    }

    /// Read the cumulative count of `Error::InvalidAvailRingIndex`
    /// events the device has observed. Per-event counter (NOT
    /// per-request): the queue-poison flag short-circuits
    /// subsequent kicks against the same hostile state, so one
    /// guest fault produces exactly one bump regardless of how
    /// many notifications follow before reset. A non-zero value
    /// means the guest violated virtio-v1.2 §2.7.13.3 — the
    /// device is in the "structurally broken queue" state and
    /// will not service IO until the guest issues a virtio reset.
    pub fn invalid_avail_idx_count(&self) -> u64 {
        self.invalid_avail_idx_count.load(Ordering::Relaxed)
    }
}

// ----------------------------------------------------------------------------
// Device struct
// ----------------------------------------------------------------------------

/// Worker-thread-owned mutable state. In production this lives on
/// the dedicated worker thread for the device's lifetime; in test
/// builds it lives directly inside `BlkWorker` (Inline mode) so the
/// existing test surface — which calls `process_requests`
/// synchronously and immediately reads back state via
/// `dev.worker.state_mut().ops_bucket` etc. — keeps working.
///
/// The MMIO-side state (interrupt_status, irq_evt, mem, FSM bits)
/// stays on `VirtioBlk` and is shared with the worker via Arc.
pub(crate) struct BlkWorkerState {
    /// Backing file. The worker reads and writes sectors via
    /// `pread`/`pwrite` and never inspects the on-disk contents.
    pub(crate) backing: File,
    /// Token-bucket for ops/sec.
    pub(crate) ops_bucket: TokenBucket,
    /// Token-bucket for bytes/sec.
    pub(crate) bytes_bucket: TokenBucket,
    /// Reusable scratch for the descriptor-walk in `drain_bracket_impl`.
    /// Allocated once at construction and `clear()`-ed each
    /// iteration so the underlying capacity (sized by the worst-case
    /// chain) is reused. Avoids one Vec allocation per request on
    /// the hot path. Capacity grows monotonically up to
    /// `VIRTIO_BLK_SEG_MAX + 2`. The data-segment slice given to
    /// the handlers is borrowed directly from
    /// `&state.all_descs_scratch[1..chain_len - 1]` once `status_addr`
    /// has been validated — no second Vec, no copy.
    pub(crate) all_descs_scratch: Vec<ChainDescriptor>,
    /// Reusable per-segment IO buffer. Sized by `resize(len, 0)`
    /// per segment in the read/write handlers. Allocated once and
    /// reused across all segments of all requests; the underlying
    /// `Vec`'s capacity grows monotonically up to
    /// `VIRTIO_BLK_SIZE_MAX` (the per-descriptor cap we advertise),
    /// at which point all subsequent IO is amortized to zero
    /// allocation.
    pub(crate) io_buf_scratch: Vec<u8>,
    /// Capacity in bytes. Computed once at construction
    /// (`capacity_sectors * VIRTIO_BLK_SECTOR_SIZE`) and threaded
    /// into handlers so the multiply isn't repeated per request and
    /// can never overflow on a malicious sector value (the multiply
    /// happens once on host-trusted input).
    pub(crate) capacity_bytes: u64,
    /// Read-only mode. When `true`, the device advertises
    /// `VIRTIO_BLK_F_RO`. `VIRTIO_BLK_T_OUT` requests are rejected
    /// with `VIRTIO_BLK_S_IOERR`; `VIRTIO_BLK_T_FLUSH` requests
    /// return `VIRTIO_BLK_S_OK` (a no-op flush — there's no dirty
    /// data to flush, and a guest issuing a precautionary flush
    /// during mount-readonly should not see an error). Per
    /// virtio-v1.2 §5.2.5.1, when `F_RO` is negotiated the device
    /// is read-only and the guest driver SHOULD treat the device
    /// as read-only; how the driver chooses to do that
    /// (read-only mount, error on `open(O_WRONLY)`, etc.) is
    /// driver business. The in-device rejection is defense
    /// against a malicious or buggy guest that ignores the
    /// negotiated feature bit.
    pub(crate) read_only: bool,
    /// Counters. `Arc` so external monitor observers can read them
    /// without holding any device borrow; the worker mutates via
    /// the same `Arc`.
    pub(crate) counters: Arc<VirtioBlkCounters>,
    /// Per-worker "is the head-of-queue chain currently stalled?"
    /// flag. Owned by `drain_bracket_impl`; the flag transitions
    /// gate the live gauge updates on the shared `counters` Arc:
    ///
    /// - `false → true` (transition into stall): bump
    ///   `currently_throttled_gauge` via
    ///   `counters.record_throttle_pending_inc()`.
    /// - `true → false` (transition out of stall): decrement via
    ///   `counters.record_throttle_pending_dec()`.
    /// - `true → true` (idempotent re-stall on the same head):
    ///   no gauge update; only `throttled_count` (events)
    ///   advances.
    /// - `false → false` (normal completion without prior stall):
    ///   no gauge update.
    ///
    /// Lives on `BlkWorkerState` (not on the shared counters Arc)
    /// because the transition logic is per-worker — only the
    /// thread that owns the drain knows which transition just
    /// happened. Reading the AtomicU64 gauge alone could not
    /// distinguish "first stall" from "re-stall on the same head"
    /// without per-worker state. Cfg-independent so both Inline
    /// and Spawned engines maintain the same invariant.
    pub(crate) currently_stalled: bool,
    /// Sticky "the queue is structurally broken; stop draining"
    /// flag. Set when the avail-ring iterator returns
    /// `Error::InvalidAvailRingIndex` — the avail.idx the guest
    /// published is more than `queue.size` ahead of the device's
    /// `next_avail`, which the virtio spec forbids
    /// (virtio-v1.2 §2.7.13.3: avail.idx advances monotonically
    /// at most `queue.size` ahead of the device-side cursor; an
    /// excursion beyond that distance is the structural-invariant
    /// violation `iter()` reports as `InvalidAvailRingIndex`).
    /// Without this flag, every subsequent `pop_descriptor_chain`
    /// would re-trip the same error and `enable_notification`
    /// would re-arm immediately, looping the worker forever
    /// against a hostile guest at full vCPU/host CPU cost.
    ///
    /// Once set, `drain_bracket_impl` short-circuits to `Done`
    /// without touching the queue at all — no
    /// `disable_notification`, no `iter`, no `enable_notification`.
    /// The flag clears only on a full virtio reset
    /// (`reset_engine_inline` / `respawn_worker` rebuilds the
    /// state with `queue_poisoned: false`), matching the device's
    /// `VIRTIO_CONFIG_S_NEEDS_RESET` (virtio-v1.2 §2.1.1 bit 0x40)
    /// behaviour: the device tells the guest "I need a reset before
    /// I can service IO" and the only escape is a STATUS=0 MMIO
    /// write. This converges with cloud-hypervisor's NEEDS_RESET
    /// path on hostile-guest queue corruption (NOT the FAILED status
    /// = 0x80, which is the orthogonal "driver gives up" exit per
    /// virtio-v1.2 §2.1.1 bit 0x80 — the framework is signalling
    /// "device needs reset", not "driver gave up").
    ///
    /// Per-worker (not on the shared counters Arc) because only
    /// the drain thread mutates it. Cfg-independent so both
    /// Inline and Spawned engines maintain the same invariant.
    pub(crate) queue_poisoned: bool,
}

/// Wraps the request-processing engine. In Inline mode (cfg(test))
/// the state lives in-line and `process_requests` runs the drain
/// synchronously on the caller thread — preserving the existing
/// 113-test surface that calls `process_requests` then immediately
/// reads back queue + counter state without crossing a thread
/// boundary. In Spawned mode (production) a dedicated worker thread
/// owns the state and is woken by `kick_fd`; the MMIO QUEUE_NOTIFY
/// handler writes 1 to `kick_fd` and returns immediately so the
/// vCPU thread is never blocked by the IO syscall.
///
/// `read_only` and `counters` are duplicated outside the engine so
/// MMIO accessors (`device_features` reads `read_only`, `counters()`
/// returns the Arc) can reach them without coordinating with the
/// worker. They are immutable after construction in Spawned mode and
/// kept in sync with the Inline branch's `BlkWorkerState`.
///
/// The shared resources the worker needs to drive the drain
/// (`Arc<BlkQueue>` queue, `Arc<EventFd>` irq_evt,
/// `Arc<AtomicU32>` interrupt_status, `Arc<OnceLock<GuestMemoryMmap>>` mem)
/// are stored on `VirtioBlk` and cloned into the worker thread at
/// spawn time; the worker holds independent Arc handles for the
/// duration of its run.
pub(crate) struct BlkWorker {
    pub(crate) queues: [BlkQueue; NUM_QUEUES],
    /// `read_only` flag, mirrored on the device side for
    /// `device_features` and direct test inspection
    /// (`dev.worker.read_only`). Set once at construction and never
    /// mutated.
    pub(crate) read_only: bool,
    /// Counters Arc shared with the worker thread; mirrored on the
    /// device side for `counters()` and direct test inspection.
    pub(crate) counters: Arc<VirtioBlkCounters>,
    /// Engine-mode-specific state.
    pub(crate) engine: WorkerEngine,
}

/// Implementation strategy for the request-processing engine.
pub(crate) enum WorkerEngine {
    /// Synchronous in-thread mode (cfg(test)). The drain runs on the
    /// caller thread when `process_requests` is called.
    #[cfg(test)]
    Inline(InlineEngine),
    /// Production mode: a dedicated worker thread owns the state
    /// and drives the drain on receipt of a kick eventfd write.
    #[cfg(not(test))]
    Spawned(SpawnedEngine),
}

/// Inline-mode engine state (cfg(test) only). Holds `BlkWorkerState`
/// directly so the existing test surface that reaches into
/// `dev.worker.<state field>` keeps compiling without renames.
#[cfg(test)]
pub(crate) struct InlineEngine {
    pub(crate) state: BlkWorkerState,
}

/// Test-only accessors: in `cfg(test)` the `BlkWorkerState` lives in
/// the Inline engine; tests reach in via `dev.worker.state_mut()` /
/// `dev.worker.state()` rather than walking the engine enum on every
/// access. The `match` is exhaustive against the single-variant cfg
/// — there is no Spawned variant to handle in test builds.
#[cfg(test)]
impl BlkWorker {
    pub(crate) fn state(&self) -> &BlkWorkerState {
        let WorkerEngine::Inline(engine) = &self.engine;
        &engine.state
    }
    pub(crate) fn state_mut(&mut self) -> &mut BlkWorkerState {
        let WorkerEngine::Inline(engine) = &mut self.engine;
        &mut engine.state
    }
}

/// Spawned-mode engine state (production only). The mutable
/// `BlkWorkerState` lives entirely on the worker thread; the device
/// retains only a kick eventfd, a stop eventfd, and the join handle.
/// The pause-eventfd write side lives on `VirtioBlk::pause_evt`
/// (cfg-independent) so `pause()` / `resume()` compile in `cfg(test)`
/// without an engine match — the worker's read clone is taken at
/// spawn time and consumed by `worker_thread_main`'s frame.
#[cfg(not(test))]
pub(crate) struct SpawnedEngine {
    /// Eventfd written by `mmio_write(QUEUE_NOTIFY, …)`; the worker
    /// epoll-waits on it and runs one drain iteration per signal.
    /// Counter-mode (no `EFD_SEMAPHORE` flag) so coalesced kicks
    /// produce one wakeup. Configured `EFD_NONBLOCK` so neither the
    /// vCPU `write(1)` nor the worker `read()` ever blocks.
    pub(crate) kick_fd: EventFd,
    /// Eventfd written by `Drop::drop`; worker reads it and exits.
    /// Counter-mode + `EFD_NONBLOCK`. The worker checks both fds in
    /// the same `epoll_wait` call so a stop signal supersedes any
    /// pending kick.
    pub(crate) stop_fd: EventFd,
    /// Worker thread join handle. Wrapped in `Option` so `Drop`
    /// and `reset()` can `take()` and `join()` it. None after the
    /// thread has been joined.
    ///
    /// The `BlkWorkerState` payload is yielded by
    /// `worker_thread_main` on STOP_TOKEN: `reset()` recovers it
    /// to rebuild fresh throttle buckets and re-spawn a worker
    /// against the post-`q.reset()` queue. `Drop` discards the
    /// returned state with `let _ = handle.join()`. Both paths
    /// observe the same return value; only the consumer differs.
    pub(crate) handle: Option<thread::JoinHandle<BlkWorkerState>>,
    /// State reclaimed from a quiesced worker, awaiting respawn at
    /// the next DRIVER_OK transition. `Some(_)` between
    /// `reset_engine_spawned` (which joins the old worker, captures
    /// its state, and stashes it here) and the guest's subsequent
    /// `STATUS = DRIVER_OK` MMIO write (which `set_status` consumes
    /// to re-spawn a fresh worker). `None` in all other steady
    /// states.
    ///
    /// # Why deferred
    ///
    /// Between `reset()` and DRIVER_OK the guest is rebinding —
    /// queue addresses are zeroed, `QUEUE_READY` is false, and any
    /// kick that lands hits the `queues[REQ_QUEUE].ready()` early
    /// return in `drain_bracket_impl`. A worker spawned eagerly in
    /// `reset()` would sit in `epoll_wait` consuming a thread for
    /// an indeterminate window — the guest's rebind sequence may
    /// take milliseconds to seconds depending on driver
    /// implementation. Deferring the spawn until DRIVER_OK lifts
    /// the cost only when there is real work to service. This
    /// matches cloud-hypervisor's "kill on reset, respawn on
    /// DRIVER_OK" pattern.
    ///
    /// # Race-free invariant
    ///
    /// Both `reset_engine_spawned` and `set_status` execute on the
    /// vCPU thread that received the MMIO write — `reset()` from
    /// `STATUS = 0` and `set_status` from `STATUS = …`. The two
    /// run sequentially within a single vCPU thread context, so
    /// the `respawn_pending` field has no concurrent reader/writer.
    /// A regression that moved either path off the vCPU thread
    /// would need to add explicit synchronisation.
    ///
    /// # Failure consequences
    ///
    /// If `reset_engine_spawned` populated `respawn_pending` but
    /// `respawn_worker` (called from `set_status` on DRIVER_OK)
    /// fails to construct fresh fds or spawn the thread, the
    /// device enters the same permanent-workerless state described
    /// in `respawn_worker`'s "Failure consequences" section. A
    /// reset that produces `respawn_pending = None` (the
    /// `stop_worker_and_reclaim_state` non-Joined outcomes) means
    /// no state to respawn from; the device is permanently dead.
    /// In either case `set_status` clears `respawn_pending` to
    /// avoid a stale state holding scratch buffers and the
    /// backing-file `File` handle alive past the device's
    /// effective lifetime.
    pub(crate) respawn_pending: Option<BlkWorkerState>,
}

/// Process-wide monotonic counter for VirtioBlk instance IDs. Used
/// to derive `instance_id` at construction so tracing logs name the
/// device with a stable small integer instead of a raw heap pointer.
/// Heap pointers expose ASLR offsets and process-layout details
/// (the `host_resource_snapshot` doc treats this kind of detail as
/// environment leakage); a per-process counter preserves the
/// "uniquely identify the device within this process run" property
/// that the diagnostics depend on without leaking the address.
pub(crate) static VIRTIO_BLK_INSTANCE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Virtio-block MMIO device.
pub struct VirtioBlk {
    pub(crate) queue_select: u32,
    pub(crate) device_features_sel: u32,
    pub(crate) driver_features_sel: u32,
    pub(crate) driver_features: u64,
    /// FSM state bits per virtio-v1.2 §3.1.1 plus the
    /// `VIRTIO_CONFIG_S_NEEDS_RESET` bit set by `drain_bracket_impl`
    /// when the avail ring becomes structurally invalid (the
    /// queue-poison path). `Arc<AtomicU32>` so the worker thread can
    /// fetch_or the NEEDS_RESET bit alongside its INT_CONFIG +
    /// `irq_evt.write(1)` poison-signal sequence; the vCPU thread
    /// reads `STATUS` via `load(Acquire)` from `mmio_read` and writes
    /// it via the FSM in `set_status` / `reset`. Atomic ordering
    /// taxonomy: `set_status` uses
    /// `compare_exchange(_, _, Release, Acquire)` for race-safe
    /// FSM advance against the worker's concurrent
    /// `fetch_or(NEEDS_RESET)` (the only RMW write site on
    /// device_status from the vCPU thread); `reset` uses
    /// `store(0, Release)`; vCPU reads use `load(Acquire)`
    /// (mmio_read, queue_config_allowed, features_write_allowed);
    /// the worker uses `fetch_or(NEEDS_RESET, SeqCst)` on the
    /// queue-poison path. Mirrors the [`Self::interrupt_status`]
    /// shape and rationale.
    pub(crate) device_status: Arc<AtomicU32>,
    /// Worker may be on a separate thread (production cfg) and the
    /// vCPU MMIO reader may race the worker's bit-set, so the value
    /// is wrapped in an `Arc` and updated with atomic ops. Worker
    /// writes the bit via `fetch_or(VIRTIO_MMIO_INT_VRING, Release)`
    /// alongside its `add_used` publish; vCPU `mmio_read` of
    /// `INTERRUPT_STATUS` does `load(Acquire)`; `INTERRUPT_ACK`
    /// clears bits via `fetch_and(!val, AcqRel)`. The Release/Acquire
    /// pair orders the bit set vs. the used-ring `add_used` (which
    /// itself publishes `used.idx` with Release internally), so a
    /// vCPU reading the bit set is guaranteed to also observe the
    /// freshly-published used.idx — no torn observation where the
    /// bit appears before the ring update.
    pub(crate) interrupt_status: Arc<AtomicU32>,
    /// `AtomicU32` for consistency with `interrupt_status`; v0 bumps
    /// only from `reset()` on the vCPU thread, not from any other
    /// thread (the worker thread does not touch config space). The
    /// atomic shape is defense-in-depth for future runtime config
    /// changes that might add a non-vCPU writer.
    pub(crate) config_generation: AtomicU32,
    /// Eventfd for KVM irqfd. Shared `Arc` so the worker thread
    /// (production cfg) can call `write(1)` to fire the IRQ without
    /// taking ownership away from the device. Tests run inline so
    /// the same Arc is read directly via `dev.irq_evt.read()`.
    pub(crate) irq_evt: Arc<EventFd>,
    /// Guest memory reference. Set before starting vCPUs via
    /// `set_mem`. Wrapped in `Arc<OnceLock<…>>` so the worker
    /// thread (production) can pick up `mem` post-construction
    /// without locking on every drain. `set_mem` is the only
    /// writer and KVM wiring guarantees it runs before any vCPU
    /// runs (i.e. before any QUEUE_NOTIFY can fire), so the
    /// reader-side `OnceLock::get` is lock-free in steady state
    /// and returns `&GuestMemoryMmap` directly — no clone needed.
    /// `reset()` does NOT clear `mem`: the same guest memory map
    /// is re-used across re-binds (matching the behaviour of the
    /// original `Mutex<Option<…>>` field, which `set_mem` only
    /// overwrote at boot).
    pub(crate) mem: Arc<OnceLock<GuestMemoryMmap>>,
    /// Capacity in 512-byte sectors. Determines what the guest sees
    /// in the config space's `capacity` field.
    pub(crate) capacity_sectors: u64,
    /// Request-processing state. In production a worker thread owns
    /// the underlying `BlkWorkerState`; in `cfg(test)` the state is
    /// inline so existing tests can read it back synchronously.
    pub(crate) worker: BlkWorker,
    /// One-shot guard so the "queue notify before set_mem"
    /// warning fires at most once per device instance.
    /// Without this, a buggy caller that issues N notifies before
    /// `set_mem` would flood the log with N copies of the same
    /// message. Latched with Relaxed because the order of the
    /// log message vs. other operations doesn't affect
    /// correctness. `Arc` so the worker thread can also
    /// access-and-latch the same flag (production: the warn fires
    /// the first time the worker observes the unset mem during a
    /// drain).
    pub(crate) mem_unset_warned: Arc<AtomicBool>,
    /// Original throttle configuration. Stored so `reset()` can
    /// rebuild fresh `TokenBucket`s on the respawned worker. Per
    /// virtio-v1.2 §2.1 a reset returns the device to its initial
    /// state, which includes the throttle bucket fill: an
    /// adversarial guest must not be able to drain the bucket and
    /// then issue a reset to bypass the rate limit. `DiskThrottle`
    /// is `Copy` (a pair of `Option<NonZeroU64>`) so this is cheap
    /// to keep around.
    pub(crate) throttle: DiskThrottle,
    /// Stable per-device monotonic identifier from
    /// [`VIRTIO_BLK_INSTANCE_COUNTER`]. Replaces the previous
    /// `self as *const _ as usize` heap-pointer field for tracing
    /// log correlation: pointers fingerprint the host's ASLR
    /// layout, an integer counter does not.
    pub(crate) instance_id: u64,
    /// Pause eventfd (host-side handle). [`Self::pause`] writes 1 to
    /// signal the worker; the worker reads the counter and parks on
    /// [`Self::paused`]. Shared `Arc` because the worker owns a clone
    /// for its epoll registration and the device retains this handle
    /// for `pause()`/`resume()` calls from the freeze coordinator.
    /// Cfg-independent so [`Self::pause`] / [`Self::resume`] compile
    /// in `cfg(test)` builds (where the inline engine is a no-op
    /// because the worker thread does not exist).
    pub(crate) pause_evt: Arc<EventFd>,
    /// Worker-parked indicator. Set to `true` by the worker thread
    /// after it drains `pause_fd` and is parked in the
    /// `park_timeout`-loop; the freeze coordinator polls this with
    /// `load(Acquire)` to confirm the worker has reached its parked
    /// state before reading guest memory. Cleared by [`Self::resume`]
    /// (Release-store of `false`); the worker's `park_timeout(10ms)`
    /// observes the clear within 10 ms and resumes its `epoll_wait`
    /// loop.
    pub(crate) paused: Arc<AtomicBool>,
    /// Optional shared parked_evt the worker writes to alongside
    /// the `paused.store(true, Release)` so the freeze
    /// coordinator's rendezvous wakes within microseconds of the
    /// last parker rather than spinning. `None` when no freeze
    /// coordinator is plumbed (test paths). The freeze coordinator
    /// sets this on every device via [`Self::set_parked_evt`]
    /// before the first `pause()` call. Counter-mode EventFd
    /// (NOT EFD_SEMAPHORE): a single drain absorbs any number of
    /// coalesced parker writes.
    pub(crate) parked_evt: Arc<std::sync::Mutex<Option<Arc<EventFd>>>>,
    /// Per-thread CPU placement applied at the top of
    /// `worker_thread_main` before the worker enters its `epoll_wait`
    /// loop. Mirrors the host topology's perf-mode (`pin_target`) and
    /// `--cpu-cap` no-perf (`no_perf_cpus`) split: at most one of the
    /// two is `Some`, both `None` means inherit the parent thread's
    /// affinity (no placement applied). Set via
    /// [`Self::set_worker_placement`] after `with_options`; defaults
    /// to all-`None` so the device works in test fixtures and call
    /// sites that don't supply topology data.
    pub(crate) worker_placement: WorkerPlacement,
}

/// CPU placement for the virtio-blk worker thread. Threaded into
/// `worker_thread_main` and applied via `pin_current_thread` /
/// `set_thread_cpumask` at the top of the worker before entering
/// `epoll_wait`. Mutually exclusive: perf-mode picks a single CPU,
/// `--cpu-cap` no-perf picks an LLC mask, both `None` means inherit
/// the parent thread's affinity (the test/inline path).
#[derive(Debug, Clone, Default)]
pub struct WorkerPlacement {
    /// Single CPU pin (perf-mode). Equivalent to
    /// `pin_current_thread(cpu, "virtio-blk worker")`.
    pub service_cpu: Option<usize>,
    /// CPU mask (no-perf + `--cpu-cap`). Equivalent to
    /// `set_thread_cpumask(cpus, "virtio-blk worker")`.
    pub no_perf_cpus: Option<Vec<usize>>,
}

impl VirtioBlk {
    /// Create a new virtio-block device.
    ///
    /// `backing` is an open File for read+write at sector
    /// granularity (the host formatted it before VM boot).
    /// `capacity_bytes` is the disk capacity advertised to the
    /// guest (rounded down to sector boundary). `throttle` carries
    /// optional IOPS / bandwidth limits.
    ///
    /// `dead_code` allow: only consumed by `#[cfg(test)]` modules;
    /// production callers go through [`Self::with_options`] to set
    /// the read-only flag explicitly.
    #[allow(dead_code)]
    pub fn new(backing: File, capacity_bytes: u64, throttle: DiskThrottle) -> Self {
        Self::with_options(backing, capacity_bytes, throttle, false)
    }

    /// Like [`Self::new`] plus a `read_only` knob. When `read_only`
    /// is `true`, the device advertises `VIRTIO_BLK_F_RO` and
    /// rejects writes regardless of guest behaviour (defense
    /// against a guest that ignores the negotiated feature bit).
    ///
    /// `capacity_bytes` smaller than one sector is clamped to
    /// `capacity_sectors = 0`, producing a 0-sector disk that
    /// IOERRs every request. The device cannot represent a
    /// fractional sector, so a sub-sector allocation is operator
    /// error — log it, continue, and let the existing zero-capacity
    /// reject path surface the failure to the guest.
    pub fn with_options(
        backing: File,
        capacity_bytes: u64,
        throttle: DiskThrottle,
        read_only: bool,
    ) -> Self {
        let irq_evt = Arc::new(
            EventFd::new(libc::EFD_NONBLOCK).expect("failed to create virtio-blk irq eventfd"),
        );
        if capacity_bytes < VIRTIO_BLK_SECTOR_SIZE as u64 && capacity_bytes != 0 {
            tracing::warn!(
                capacity_bytes,
                sector_size = VIRTIO_BLK_SECTOR_SIZE,
                "virtio-blk capacity_bytes smaller than one sector; clamping \
                 capacity_sectors to 0 (every IO will be rejected)"
            );
        }
        let capacity_sectors = capacity_bytes / VIRTIO_BLK_SECTOR_SIZE as u64;
        let capacity_bytes = capacity_sectors * VIRTIO_BLK_SECTOR_SIZE as u64;
        let (ops_bucket, bytes_bucket) = buckets_from_throttle(throttle);
        let counters = Arc::new(VirtioBlkCounters::default());

        let state = BlkWorkerState {
            backing,
            ops_bucket,
            bytes_bucket,
            all_descs_scratch: Vec::with_capacity(VIRTIO_BLK_SEG_MAX as usize + 2),
            io_buf_scratch: Vec::new(),
            capacity_bytes,
            read_only,
            counters: Arc::clone(&counters),
            currently_stalled: false,
            queue_poisoned: false,
        };

        let interrupt_status = Arc::new(AtomicU32::new(0));
        let device_status = Arc::new(AtomicU32::new(0));
        let mem = Arc::new(OnceLock::new());
        let mem_unset_warned = Arc::new(AtomicBool::new(false));
        // Pause primitives (failure-dump rendezvous). The
        // `pause_evt` host handle is kept on the `VirtioBlk` for
        // `pause()`/`resume()`; in production a clone of its read
        // side becomes the `pause_fd` registered in the worker's
        // epoll. `paused` is the worker-set / coordinator-cleared
        // ack flag the freeze rendezvous polls. Both Arcs are
        // cfg-independent so the test-mode `pause`/`resume`
        // accessors compile without engine-conditional plumbing
        // (the test-mode worker is inline, so they observe the
        // same eventfd state without an active worker thread).
        let pause_evt = Arc::new(
            EventFd::new(libc::EFD_NONBLOCK).expect("failed to create virtio-blk pause eventfd"),
        );
        // Initialise to `true` so the freeze coordinator's
        // `is_paused()` poll passes vacuously while no worker is
        // alive — the initial spawn is deferred to DRIVER_OK
        // (see `respawn_pending` engine plumbing below), so any
        // freeze that fires between `with_options` and the first
        // DRIVER_OK MMIO write would otherwise time out at
        // FREEZE_RENDEZVOUS_TIMEOUT (30 s) waiting for a worker
        // that does not exist. The worker's first action inside
        // `worker_thread_main` (after affinity setup, before
        // entering `epoll_wait`) is a Release-store of `false`,
        // which makes the rendezvous start observing real
        // worker state from the moment the worker is genuinely
        // ready to service kicks. Cloud-hypervisor uses the
        // same "paused on construction, cleared by activate"
        // invariant in epoll_helper.rs.
        let paused = Arc::new(AtomicBool::new(true));

        // Build the queue. Production uses `QueueSync` (Arc<Mutex<Queue>>
        // internally) so the vCPU MMIO config writes and the worker
        // thread's drain can share the same queue state. Tests use the
        // bare `Queue` so the existing test surface that drives queue
        // methods directly via `dev.worker.queues[REQ_QUEUE].…` keeps
        // working without a runtime lock.
        let queues = [BlkQueue::new(QUEUE_MAX_SIZE).expect("valid queue size")];

        // Build the engine. cfg(test) keeps the state inline so the
        // existing test surface drives drain_bracket_impl synchronously;
        // cfg(not(test)) spawns a dedicated worker thread that owns
        // the state and waits for kick eventfd writes from
        // `process_requests`.
        #[cfg(test)]
        let engine = WorkerEngine::Inline(InlineEngine { state });

        #[cfg(not(test))]
        let engine = {
            // Counter-mode eventfds (no EFD_SEMAPHORE). EFD_NONBLOCK so
            // the vCPU `write(1)` to kick_fd never blocks even under
            // pathological backpressure (the worker has fallen behind
            // by more than u64::MAX-1 kicks — implausible under any
            // realistic workload, but the non-blocking flag keeps the
            // failure mode "EAGAIN, drop the spurious kick" instead of
            // "vCPU thread blocks on eventfd write").
            let kick_fd =
                EventFd::new(libc::EFD_NONBLOCK).expect("failed to create virtio-blk kick eventfd");
            let stop_fd =
                EventFd::new(libc::EFD_NONBLOCK).expect("failed to create virtio-blk stop eventfd");
            // Defer the initial worker spawn to the guest's first
            // DRIVER_OK transition (set_status → consume_pending_respawn
            // → respawn_worker). Stashing the seed `BlkWorkerState` in
            // `respawn_pending` collapses the initial-spawn path into
            // the existing respawn path, which `respawn_worker` already
            // implements correctly (placement applied via
            // `self.worker_placement` clone, fresh kick/stop/pause fds
            // built per spawn). This gives `set_worker_placement` a
            // race-free window between construction and DRIVER_OK in
            // which to override the default placement; without
            // deferral the initial worker would spawn with the default
            // placement before setup.rs's setter call could land.
            //
            // Pre-DRIVER_OK kicks land on the now-detached `kick_fd`
            // and accumulate harmlessly; the first post-DRIVER_OK
            // worker observes the queue's `ready()` flag and processes
            // any pre-existing chain state. The kernel's virtio-mmio
            // bind sequence (drivers/virtio/virtio_mmio.c
            // `virtio_mmio_probe` → `vp_finalize_features` →
            // `vm_setup_vq` → `STATUS=DRIVER_OK`) does not fire
            // QUEUE_NOTIFY before DRIVER_OK, so accumulation is
            // bounded at zero in the production path.
            WorkerEngine::Spawned(SpawnedEngine {
                kick_fd,
                stop_fd,
                handle: None,
                respawn_pending: Some(state),
            })
        };

        let worker = BlkWorker {
            queues,
            read_only,
            counters,
            engine,
        };

        VirtioBlk {
            queue_select: 0,
            device_features_sel: 0,
            driver_features_sel: 0,
            driver_features: 0,
            device_status,
            interrupt_status,
            config_generation: AtomicU32::new(0),
            irq_evt,
            mem,
            capacity_sectors,
            worker,
            mem_unset_warned,
            throttle,
            instance_id: VIRTIO_BLK_INSTANCE_COUNTER.fetch_add(1, Ordering::Relaxed),
            pause_evt,
            paused,
            parked_evt: Arc::new(std::sync::Mutex::new(None)),
            worker_placement: WorkerPlacement::default(),
        }
    }

    /// Plumb the freeze coordinator's shared parked_evt into this
    /// device. The worker writes to this fd alongside its
    /// `paused.store(true, Release)` so the coordinator's
    /// rendezvous wakes within microseconds of the worker
    /// parking. Called once by `run_vm` before any pause()/resume()
    /// fires; subsequent worker respawns pick up the same fd via
    /// the shared `Arc`.
    ///
    /// `None` is the default — test paths and any future device
    /// without a freeze coordinator skip the wake. The worker
    /// reads through this slot lazily so a setter call AFTER worker
    /// spawn (e.g. plumbing arrives late) still takes effect on
    /// the next pause cycle.
    pub fn set_parked_evt(&self, evt: Arc<EventFd>) {
        if let Ok(mut guard) = self.parked_evt.lock() {
            *guard = Some(evt);
        }
    }

    /// Configure the per-thread CPU placement applied at the top of
    /// the worker's main loop. Mirrors the `set_mem` setter pattern:
    /// called once after `with_options` / `new`, before the device
    /// starts servicing kicks. The placement is captured by the
    /// next worker-thread spawn — and because the initial spawn is
    /// DEFERRED to the guest's first `STATUS = DRIVER_OK` MMIO
    /// write (the seed `BlkWorkerState` lives in `respawn_pending`
    /// until then), a setter call between `with_options` and that
    /// DRIVER_OK transition lands on the very first worker. After
    /// the worker has started, calling this has no effect on the
    /// running thread — only respawned workers pick up the new
    /// placement, matching cloud-hypervisor's "topology applied at
    /// thread start" pattern.
    ///
    /// `WorkerPlacement::service_cpu` and `no_perf_cpus` are mutually
    /// exclusive — the topology layer (perf-mode vs `--cpu-cap`)
    /// produces at most one. Both `None` means inherit the parent
    /// thread's affinity (the test/inline path and the no-topology
    /// fallback for ad-hoc fixtures).
    pub fn set_worker_placement(&mut self, placement: WorkerPlacement) {
        self.worker_placement = placement;
    }

    /// Eventfd for KVM irqfd registration.
    pub fn irq_evt(&self) -> &EventFd {
        &self.irq_evt
    }

    /// Set guest memory reference. Must be called before starting vCPUs.
    ///
    /// Stores the memory inside the device's shared `Arc<OnceLock<…>>`
    /// so the worker thread (production cfg) can observe the
    /// reference on its next drain via a lock-free
    /// `OnceLock::get`. Returning before the worker observes the
    /// value is safe because the worker only consults `mem` in
    /// response to a kick (driven by `mmio_write` of QUEUE_NOTIFY),
    /// and KVM wiring guarantees `set_mem` runs before any vCPU
    /// runs (i.e. before any QUEUE_NOTIFY can fire).
    ///
    /// `OnceLock::set` returns `Err` if the slot is already
    /// populated. The current production wiring (mod.rs `init_virtio_blk`)
    /// calls `set_mem` exactly once per device, so the `Err` branch
    /// is unreachable in normal operation; `reset()` does NOT clear
    /// `mem`, matching the prior `Mutex<Option<…>>` semantics where
    /// the slot was only written at boot. Log on `Err` rather than
    /// panic so a future re-wire bug surfaces as a warning instead
    /// of aborting (a panic here could land mid-teardown when the
    /// caller is already unwinding).
    pub fn set_mem(&mut self, mem: GuestMemoryMmap) {
        if self.mem.set(mem).is_err() {
            tracing::warn!(
                "virtio-blk: set_mem called on already-initialised \
                 device; guest memory binding unchanged (mem is set \
                 once at boot and preserved across reset())"
            );
        }
    }

    /// Advertised capacity in 512-byte sectors.
    ///
    /// `dead_code` allow: kept for forward use by callers that
    /// need to read back the rounded-to-sector capacity; the lib
    /// pipeline currently consumes the `capacity_bytes` input
    /// directly through the config-space rendering path.
    #[allow(dead_code)]
    pub fn capacity_sectors(&self) -> u64 {
        self.capacity_sectors
    }

    /// Cloneable handle to the host-observability counters. The
    /// monitor thread holds an Arc to read counters without locking
    /// the device.
    pub fn counters(&self) -> Arc<VirtioBlkCounters> {
        Arc::clone(&self.worker.counters)
    }

    pub(crate) fn device_features(&self) -> u64 {
        // VIRTIO_F_VERSION_1: modern virtio.
        // VIRTIO_BLK_F_BLK_SIZE: config carries the logical
        //   block size at offset 0x14 (4 bytes LE).
        // VIRTIO_BLK_F_SEG_MAX: config carries the per-request
        //   max scatter-gather segment count at offset 0x0C.
        //   Without this bit the guest defaults max_segments to 1
        //   and a multi-segment bio gets split serially —
        //   throughput-cripplingly slow for our use case.
        // VIRTIO_BLK_F_SIZE_MAX: config carries the per-descriptor
        //   max byte length at offset 0x08. Without this, a guest
        //   can submit a single 4 GB descriptor and force the
        //   device to allocate a matching `Vec<u8>` for I/O —
        //   host OOM hazard on a hostile guest.
        // VIRTIO_BLK_F_FLUSH: device honours T_FLUSH (fdatasync).
        //   Durability semantics depend on the host filesystem
        //   backing the per-test image: for the default
        //   `tempfile()` path on tmpfs, fdatasync is effectively a
        //   no-op (tmpfs doesn't persist across reboot anyway —
        //   only ordering semantics). For btrfs/ext4-backed run
        //   dirs, fdatasync provides the standard kernel-level
        //   ordering guarantee. Advertising F_FLUSH lets the guest
        //   block layer issue REQ_OP_FLUSH at metadata-commit
        //   boundaries (btrfs in the guest depends on this for
        //   tree consistency).
        // VIRTIO_BLK_F_RO: gated on `read_only`. The kernel block
        //   layer marks the disk read-only via `set_disk_ro` after
        //   F_RO negotiation; the device's defensive T_OUT
        //   rejection guards against uncovered write paths.
        // VIRTIO_RING_F_EVENT_IDX (transport feature, bit 29): the
        //   guest can place an "event index" in the avail ring's
        //   `used_event` field telling the device "do not interrupt
        //   me until used.idx reaches this value." `virtio_queue`
        //   tracks that field internally — when this bit is
        //   negotiated, `Queue::needs_notification` returns false
        //   while the guest's threshold is not reached and the
        //   device can suppress the irqfd write. Notification
        //   suppression on bursty IO (multi-chain queue drains)
        //   reduces vCPU exits proportional to the burst size and
        //   is required for high-throughput virtio-blk under
        //   blk-mq. Wire-up: this advertises the bit, set_status
        //   enables event-idx tracking on the queue when FEATURES_OK
        //   negotiates it, and process_requests consults
        //   `Queue::needs_notification` after each drain to decide
        //   whether to fire the irqfd. The V8 split: process_requests
        //   sets VIRTIO_MMIO_INT_VRING unconditionally on any chain
        //   publish, then consults needs_notification to decide
        //   whether to also fire the irqfd.
        let mut feats = (1u64 << VIRTIO_F_VERSION_1)
            | (1u64 << VIRTIO_BLK_F_BLK_SIZE)
            | (1u64 << VIRTIO_BLK_F_SEG_MAX)
            | (1u64 << VIRTIO_BLK_F_SIZE_MAX)
            | (1u64 << VIRTIO_BLK_F_FLUSH)
            | (1u64 << VIRTIO_RING_F_EVENT_IDX);
        if self.worker.read_only {
            feats |= 1u64 << VIRTIO_BLK_F_RO;
        }
        feats
    }

    pub(crate) fn selected_queue(&self) -> Option<usize> {
        let idx = self.queue_select as usize;
        if idx < NUM_QUEUES { Some(idx) } else { None }
    }

    pub(crate) fn queue_config_allowed(&self) -> bool {
        let status = self.device_status.load(Ordering::Acquire);
        status & S_FEAT == S_FEAT && status & VIRTIO_CONFIG_S_DRIVER_OK == 0
    }

    pub(crate) fn features_write_allowed(&self) -> bool {
        let status = self.device_status.load(Ordering::Acquire);
        status & S_DRV == S_DRV && status & VIRTIO_CONFIG_S_FEATURES_OK == 0
    }

    /// Classify a request type into a "pre-throttle terminal status"
    /// when applicable. Returns `Some((status_byte, used_len))` for
    /// requests the device decides without ever touching the backing
    /// file or the throttle (RO-mode writes, RO-mode flushes, unknown
    /// request types). Returns `None` for the normal
    /// IN/OUT/FLUSH/GET_ID paths that need the backend handlers.
    ///
    /// Side effects per branch:
    ///
    /// - **RO-mode `T_OUT` → `S_IOERR`**: increments `io_errors`.
    ///   The guest negotiated `F_RO` (or, defensively, ignored it
    ///   and tried to write anyway). Either way, the device rejected
    ///   real work — that's an IO error from a host-observability
    ///   standpoint, surfaced in the failure-dump counters.
    /// - **RO-mode `T_FLUSH` → `S_OK`**: increments
    ///   `flushes_completed`. The guest issued a real flush; the
    ///   device serviced it (trivially, because nothing's dirty on
    ///   a read-only disk). The counter records the delivery, not
    ///   the work — symmetric with how the writable-disk path
    ///   bumps `flushes_completed` after `fdatasync` returns Ok.
    /// - **`T_GET_ID` → None (regardless of `read_only`)**: T_GET_ID
    ///   is a metadata read that never touches the backing file, so
    ///   the RO disk accepts it the same as a writable disk. Per
    ///   virtio-v1.2 §5.2.6.4 GET_ID is not gated on any feature
    ///   bit and is always accepted.
    /// - **Unknown type → `S_UNSUPP`**: NO counter bump. UNSUPP is
    ///   a graceful decline ("the device doesn't speak this
    ///   request"), not a service failure — the device never tried
    ///   anything that could fail.
    ///
    /// Counter writes belong with the classification because the
    /// dispatch decision IS the moment that bookkeeping happens —
    /// hoisting them out would force the caller to re-derive the
    /// request type.
    ///
    /// Extracted into a free associated function (no `&self`) so it
    /// can be tested directly without constructing a fully-wired
    /// `VirtioBlk` and so `process_requests` can call it while
    /// holding `&mut self.queues[..]`.
    pub(crate) fn classify_pre_throttle(
        req_type: u32,
        read_only: bool,
        counters: &VirtioBlkCounters,
    ) -> Option<(u8, u32)> {
        match req_type {
            VIRTIO_BLK_T_OUT if read_only => {
                counters.record_io_error();
                Some((VIRTIO_BLK_S_IOERR as u8, 1))
            }
            VIRTIO_BLK_T_FLUSH if read_only => {
                // No-op flush on a read-only disk: nothing dirty to
                // flush, but count it as a completed flush for
                // visibility in the failure-dump counters — the
                // guest issued a real flush and the device serviced
                // it.
                counters.record_flush();
                Some((VIRTIO_BLK_S_OK as u8, 1))
            }
            VIRTIO_BLK_T_IN | VIRTIO_BLK_T_OUT | VIRTIO_BLK_T_FLUSH | VIRTIO_BLK_T_GET_ID => None,
            _ => Some((VIRTIO_BLK_S_UNSUPP as u8, 1)),
        }
    }

    // ------------------------------------------------------------------
    // Request queue processing
    // ------------------------------------------------------------------

    /// Drive the request queue. In `cfg(test)` the drain runs
    /// inline on the caller thread (preserving the synchronous
    /// test surface). In production this is a non-blocking
    /// kick of the worker thread's eventfd — `mmio_write` of
    /// `QUEUE_NOTIFY` returns immediately so the vCPU thread
    /// doesn't block on backing-file IO.
    pub(crate) fn process_requests(&mut self) {
        #[cfg(test)]
        {
            self.drain_inline();
        }
        #[cfg(not(test))]
        {
            // Non-blocking kick. The worker thread's epoll_wait
            // resumes and runs one drain iteration per kick. EAGAIN
            // (counter saturated at u64::MAX-1) is implausible under
            // any realistic workload — the worker would have to be
            // ~2^64 kicks behind — and on encountering it we drop
            // the spurious kick because counter-mode coalesces all
            // pending kicks into a single read by the worker on the
            // next wakeup, so no QUEUE_NOTIFY is permanently lost.
            let WorkerEngine::Spawned(eng) = &self.worker.engine;
            let _ = eng.kick_fd.write(1);
        }
    }

    /// Inline drain (test-mode only). Resolves the Inline engine,
    /// fetches a `&mem` reference from the shared `Arc<OnceLock<…>>`
    /// via a lock-free `OnceLock::get`, and calls `drain_bracket_impl`
    /// directly with the worker state + queue + irq + interrupt_status
    /// borrows. No clone is needed — `drain_bracket_impl` accepts
    /// `&GuestMemoryMmap` and the lifetime ends inside this fn.
    #[cfg(test)]
    pub(crate) fn drain_inline(&mut self) {
        let Some(mem) = self.mem.get() else {
            // Caller (kvm wiring in src/vmm/mod.rs) is supposed to
            // call `set_mem` before any vCPU runs. A queue-notify
            // before that is a wiring bug; surface it once per
            // device so the log isn't flooded if the guest spams
            // notifies on the broken setup.
            if !self.mem_unset_warned.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    "virtio-blk: queue notify before set_mem; \
                     dropping requests until guest memory is wired"
                );
            }
            return;
        };
        let WorkerEngine::Inline(engine) = &mut self.worker.engine;
        // The cfg(test) inline path discards the DrainOutcome —
        // tests step the throttle bucket forward via
        // `set_last_refill_for_test` and re-issue QUEUE_NOTIFY to
        // exercise the post-stall retry path. There is no timer
        // arming because there is no worker thread to wake.
        let _ = drain_bracket_impl(
            &mut engine.state,
            &mut self.worker.queues,
            mem,
            &self.irq_evt,
            &self.interrupt_status,
            &self.device_status,
        );
    }
}

/// Outcome of a single `drain_bracket_impl` invocation.
///
/// `Done` — the inner pop loop ran to None and `enable_notification`
/// settled (no pending chains; nothing to retry). The caller should
/// rest until the next kick.
///
/// `ThrottleStalled { wait_nanos }` — a chain was popped whose IO
/// budget the throttle bucket cannot satisfy; the chain has been
/// rolled back via `set_next_avail(prev.wrapping_sub(1))` (so the
/// next drain re-pops it) and `wait_nanos` is the worst-case
/// delay before the bucket holds enough tokens to satisfy it. The
/// worker thread arms a timerfd for this duration; tests step the
/// bucket forward and re-call `process_requests`. `wait_nanos ==
/// 0` means the bucket is unlimited or already refilled to
/// sufficiency — the caller should re-drain immediately rather
/// than waiting on a timer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DrainOutcome {
    Done,
    ThrottleStalled { wait_nanos: u64 },
}

/// Drain the request queue, processing reads/writes/flushes
/// against the backing file and respecting the throttle.
///
/// The chain is walked in one pass and `add_used` is called
/// in the same loop iteration that completes the request.
/// `pop_descriptor_chain` returns a chain whose lifetime ends
/// at the bottom of the iteration (after we've collected the
/// data-segment vector), so the borrow on the queue is released
/// before `add_used` re-borrows it. This mirrors the
/// virtio-console pattern (see `process_tx` in virtio_console.rs).
///
/// Free function (not a method) so the worker thread (production)
/// and the inline test harness (cfg(test)) can both invoke it
/// against a `BlkWorkerState` they own without taking a method
/// receiver — production owns `state` on the worker thread and the
/// inline path borrows it via `self.worker.engine`.
///
/// Borrows guest memory, the irqfd, and the interrupt-status atomic
/// from the device — those live on the MMIO side (`VirtioBlk`) and
/// are passed in. `queues` is borrowed mutably so the drain can
/// pop / add_used / disable+enable_notification / needs_notification
/// in lock-pop-unlock-walk-lock-add_used order without holding any
/// queue lock during IO.
///
/// Returns `DrainOutcome::ThrottleStalled` when a chain was popped
/// but its IO budget is exhausted: the chain is rolled back via
/// `set_next_avail(prev.wrapping_sub(1))` (so the next drain re-pops
/// it) and the returned wait duration tells the caller how long
/// until the bucket will hold enough tokens to satisfy the request.
/// The worker thread arms a timerfd from this duration; when the
/// timer fires, the drain re-runs. (`go_to_previous_position` from
/// the virtio-queue crate has the same effect, but it lives on the
/// `QueueOwnedT` trait which `QueueSync` does not implement;
/// `set_next_avail` is on the base `QueueT` and works for both
/// alias targets in this module.)
///
/// On stall, no S_IOERR / no add_used / no signal — the chain stays
/// invisible to the guest until the retry. `throttled_count` is bumped
/// per stall so operators can observe the rate. `Done` indicates
/// the queue was drained to None and re-enable settled (no pending
/// chains).
pub(crate) fn drain_bracket_impl(
    state: &mut BlkWorkerState,
    queues: &mut [BlkQueue; NUM_QUEUES],
    mem: &GuestMemoryMmap,
    irq_evt: &EventFd,
    interrupt_status: &AtomicU32,
    device_status: &AtomicU32,
) -> DrainOutcome {
    // Pre-rebind / post-reset gate. After `q.reset()` clears the
    // queue (zeroing desc/avail/used GPAs and `ready`), there is
    // a window before the guest re-publishes addresses and sets
    // `QUEUE_READY = 1`. A kick or timer wakeup that lands in
    // that window must not call `disable_notification` /
    // `enable_notification` — both write to the used ring's
    // `flags` / `avail_event` fields, and a used-ring GPA of 0
    // (the post-reset state) causes a spurious write to guest
    // physical address 0. Worse, `pop_descriptor_chain` against
    // a stale ring-cursor can mis-read descriptor entries.
    //
    // `QueueT::ready()` returns `true` only after the guest has
    // written `QUEUE_READY = 1` (post-rebind). In production the
    // worker may receive kicks routed through the device's
    // `kick_fd` between `respawn_worker` and the guest's first
    // post-reset `set_ready(true)` MMIO write — this gate makes
    // those drains a no-op until the guest finishes rebinding.
    if !queues[REQ_QUEUE].ready() {
        return DrainOutcome::Done;
    }

    // Hostile-guest defense gate. A previous drain observed
    // `Error::InvalidAvailRingIndex` from `Queue::iter` (the
    // guest's avail.idx was more than `queue.size` ahead of
    // `next_avail`, violating virtio-v1.2 §2.7.13.3 avail.idx
    // semantics). The
    // structural invariant the iterator depends on is broken;
    // every subsequent `iter()` call would re-trip the same
    // error, and `enable_notification` would re-arm
    // immediately, looping the worker forever at full
    // vCPU/host-CPU cost.
    //
    // Returning `Done` without touching the queue:
    // - skips `disable_notification` (no spurious used.flags
    //   write — the guest already poisoned the queue, more
    //   side effects make the symptom worse, not better),
    // - skips `iter()` (no second `invalid_avail_idx_count`
    //   bump per kick — the counter is per-event, the flag
    //   makes it event-once),
    // - skips `enable_notification` (no Ok(true) re-loop and
    //   no irqfd write).
    //
    // The flag clears only on a full virtio reset
    // (`reset_engine_inline` / `respawn_worker` rebuilds the
    // state with `queue_poisoned: false`). Until then the
    // device will not service IO — the guest's blk-mq layer
    // observes hangs and the operator sees a non-zero
    // `invalid_avail_idx_count` in the failure dump.
    if state.queue_poisoned {
        return DrainOutcome::Done;
    }

    // The request loop calls handlers (which take `&` borrows
    // of state.backing/state.counters) plus throttle bucket
    // mutation (`&mut state.ops_bucket` / `&mut state.bytes_bucket`).
    // To keep the borrow checker happy we materialise the queue
    // handle separately (`&mut queues[REQ_QUEUE]`) and reach
    // into `&mut state` only via the disjoint fields it owns.
    // The eventfd write that signals the guest is hoisted to
    // the end so it does not alias with the queue mutation in
    // the loop.
    let mut signal_needed = false;
    // Set when the throttle path stalls; carries the
    // worst-case wait time (in nanoseconds) before the bucket
    // refills enough to satisfy the rolled-back chain. None
    // when the drain reached the natural end (all chains
    // processed, queue empty, enable_notification settled).
    let mut stall_outcome: Option<u64> = None;
    // Outer bracket: disable_notification → drain → enable_notification.
    // Canonical virtio-queue pattern — the doctest on the
    // `Queue` struct in the virtio-queue crate spells out the
    // disable/drain/enable shape this loop mirrors.
    // `Queue::enable_notification` returns Ok(true) when new
    // chain heads appeared during the disabled window — re-drain
    // to avoid stranding chains the guest has enqueued without
    // a fresh QUEUE_NOTIFY MMIO exit. Its trait-level contract
    // on `QueueT::enable_notification` documents the
    // re-iteration semantics. Without re-checking, a chain
    // enqueued after our final `pop_descriptor_chain` returns
    // None but before notifications come back on would sit
    // unprocessed until the guest's hung-task watchdog fired
    // (`kernel.hung_task_timeout_secs`, default 120 s — virtio_blk
    // has no `mq_ops->timeout`, so blk-mq won't surface the stall).
    //
    // `Queue::disable_notification` semantics depend on whether
    // EVENT_IDX is negotiated (see `Queue::set_notification`,
    // which `disable_notification` and `enable_notification`
    // both delegate to):
    //   * legacy path (event_idx_enabled=false): writes the
    //     VRING_USED_F_NO_NOTIFY flag in used.flags, telling
    //     the guest to skip QUEUE_NOTIFY MMIO writes during
    //     the drain — removes redundant vCPU exits.
    //   * EVENT_IDX path (event_idx_enabled=true):
    //     disable_notification is a no-op (queue.rs:241-244).
    //     Suppression of guest kicks relies on NOT updating
    //     avail_event during the drain — avail_event stays at
    //     whatever the prior enable_notification wrote.
    // Either way, the bracket pattern is correct; both paths
    // route through the canonical disable/enable.
    'outer: loop {
        // Best-effort disable; failure is non-fatal — the worst
        // case is the guest issues a redundant QUEUE_NOTIFY
        // mid-drain that we'd absorb on the next call anyway.
        if let Err(e) = queues[REQ_QUEUE].disable_notification(mem) {
            tracing::warn!(%e, "virtio-blk disable_notification failed");
        }
        loop {
            // Pop one chain via `iter()`/`.next()` so we OBSERVE
            // `Error::InvalidAvailRingIndex` instead of swallowing
            // it. The bare `Queue::pop_descriptor_chain` impl
            // (queue.rs:573-587) calls iter() internally, logs any
            // error, and returns None — masking the structural
            // violation as "no chain available" and letting
            // `enable_notification` re-arm immediately, looping
            // the worker forever against a hostile guest.
            //
            // `iter()` is on `QueueOwnedT`, which only the bare
            // `Queue` implements; we reach it via `q.lock()` —
            // `&mut Queue` for `Queue` (cfg(test) alias) and
            // `MutexGuard<Queue>` for `QueueSync` (cfg(not(test))).
            // Both deref to `Queue`, so `guard.iter(mem)` compiles
            // for both alias targets. Two-step extraction keeps
            // the borrow tight: take the iter inside a block and
            // capture only the outcome (Some(chain)/None/Err(_))
            // before dropping the lock guard. The
            // `DescriptorChain<M>` owns its own `mem.clone()`
            // (queue.rs:761-766), so it does not borrow from the
            // iter or the guard — we can walk it freely after the
            // guard drops, and `add_used` etc. can re-borrow the
            // queue downstream.
            let iter_outcome = {
                let q = &mut queues[REQ_QUEUE];
                let mut guard = q.lock();
                match guard.iter(mem) {
                    Ok(mut iter) => Ok(iter.next()),
                    Err(e) => Err(e),
                }
            };
            let chain = match iter_outcome {
                Ok(Some(c)) => c,
                Ok(None) => break,
                Err(VirtioQueueError::InvalidAvailRingIndex) => {
                    // Hostile-guest poison. The avail.idx is more
                    // than `queue.size` ahead of the device's
                    // `next_avail` (virtio-v1.2 §2.7.13.3
                    // avail.idx-distance violation; check sits at
                    // queue.rs:707-709 in `AvailIter::new`). Mark
                    // the queue dead so future drains
                    // short-circuit, bump the per-event counter
                    // (gated by the flag — exactly one bump per
                    // poison event regardless of re-kicks), and
                    // bail without calling `enable_notification`.
                    // Re-enabling notifications would arm the
                    // next kick to re-trip the same error — a
                    // livelock. A full virtio reset is the only
                    // path back to service.
                    state.queue_poisoned = true;
                    state.counters.record_invalid_avail_idx();
                    tracing::warn!(
                        "virtio-blk avail.idx exceeds next_avail by more \
                         than queue.size (virtio-v1.2 §2.7.13.3 \
                         avail.idx-distance violation); poisoning queue \
                         until guest reset"
                    );
                    // Surface the structural failure as an
                    // observability signal:
                    //
                    // 1. NEEDS_RESET in the device_status FSM is
                    //    visible to the guest via
                    //    `mmio_read(VIRTIO_MMIO_STATUS)` and to the
                    //    host operator via sysfs / failure-dump. It
                    //    is the spec-compliant way (virtio-v1.2
                    //    §2.1.1, bit 0x40) for a device to advertise
                    //    "I need to be reset before I can service
                    //    IO." Cloud-hypervisor uses the same bit for
                    //    its hostile-guest shutdown path.
                    // 2. INT_CONFIG + irq_evt.write trigger the
                    //    guest IRQ path through
                    //    `vm_interrupt → virtio_config_changed →
                    //    __virtio_config_changed → drv->config_changed`
                    //    (drivers/virtio/virtio_mmio.c). For
                    //    virtio_blk the `config_changed` callback
                    //    (`virtblk_config_changed`) only re-reads
                    //    config-space CAPACITY — it does NOT
                    //    automatically surface NEEDS_RESET to blk-mq
                    //    or fail in-flight requests. So this
                    //    sequence is a SIGNAL to the guest, not a
                    //    recovery primitive: the guest's behavior
                    //    on a poisoned queue depends on whatever
                    //    higher-layer logic reads STATUS.
                    //
                    // The actual request-stopping defense is the
                    // `state.queue_poisoned` gate at the top of
                    // `drain_bracket_impl`: every subsequent drain
                    // short-circuits to `Done`, so no chain ever
                    // gets `add_used`-published. In-flight requests
                    // hang until the guest's hung-task watchdog
                    // fires (default 120 s, virtio_blk has no
                    // `mq_ops->timeout`) or a reset arrives. The
                    // NEEDS_RESET signal here gives operators the
                    // STATUS-read tool to detect the wedged state
                    // before the watchdog fires; it does not
                    // unwedge anything on its own.
                    //
                    // Fired exactly once at the poison transition:
                    // the queue_poisoned gate above this arm's
                    // entry returns `Done` for every subsequent
                    // kick, so re-kicks never re-enter this arm.
                    //
                    // SeqCst on the device_status fetch_or pairs
                    // with two reader sites:
                    //   1. The vCPU MMIO read of STATUS via
                    //      `load(Acquire)` in `mmio_read` — the
                    //      post-poison read reflects the bit so the
                    //      guest's STATUS query sees NEEDS_RESET.
                    //   2. `set_status`'s CAS retry loop. The
                    //      `compare_exchange` failure-side Acquire
                    //      synchronizes-with this SeqCst write so
                    //      the retry iteration's re-snapshot sees
                    //      the NEEDS_RESET bit and the monotone-bit
                    //      gate rejects the FSM advance instead of
                    //      clobbering the bit. This is the
                    //      load-bearing pairing — without it, a
                    //      vCPU set_status racing this fetch_or
                    //      would silently drop NEEDS_RESET on the
                    //      next FSM advance.
                    // The interrupt_status fetch_or uses Release
                    // ordering to mirror the existing INT_VRING
                    // write-side discipline at the V8 publish-path
                    // INTERRUPT_STATUS bit-set.
                    //
                    // INVARIANT: the worker may ONLY `fetch_or`
                    // `VIRTIO_CONFIG_S_NEEDS_RESET` into
                    // device_status — never `store`,
                    // `fetch_and`, `fetch_xor`, or `fetch_or` any
                    // OTHER bit. Termination of `set_status`'s
                    // CAS retry loop is bounded at AT MOST ONE
                    // worker-induced retry: NEEDS_RESET fetch_or
                    // is idempotent after the first call, so the
                    // worker can change `device_status` from one
                    // observable state to one other state and
                    // never again from the device side. The
                    // single-bit constraint makes that bound
                    // auditable; a future worker fetch_or'ing a
                    // different bit (e.g. a hypothetical
                    // VIRTIO_CONFIG_S_DEVICE_NEEDS_RESET-like
                    // extension) would expand the retry universe
                    // and the bound. A worker that cleared bits —
                    // store/fetch_and/fetch_xor — would let the
                    // retry loop spin indefinitely as the
                    // snapshot re-enters the previously-rejected
                    // state.
                    device_status.fetch_or(VIRTIO_CONFIG_S_NEEDS_RESET, Ordering::SeqCst);
                    interrupt_status.fetch_or(VIRTIO_MMIO_INT_CONFIG, Ordering::Release);
                    // SAFETY: EAGAIN requires counter saturation at
                    // u64::MAX-1 (~1.8e19 unobserved kicks) —
                    // implausible. EBADF means the fd closed during
                    // shutdown. The simultaneously-set INT_CONFIG
                    // bit above is the enduring guest-visible
                    // signal: `vm_interrupt`
                    // (drivers/virtio/virtio_mmio.c) reads
                    // INTERRUPT_STATUS on the next IRQ delivery and
                    // dispatches via the bit set — but on the
                    // poison path NO subsequent device IRQ fires.
                    // The queue_poisoned gate makes every later
                    // drain short-circuit to `Done` without ever
                    // calling `add_used` or triggering another
                    // signal, so a missed irqfd write here means
                    // the operator's only path to seeing the
                    // NEEDS_RESET state is `mmio_read(STATUS)` —
                    // which still works because the bit is on
                    // device_status. The guest's actual recovery
                    // path is a STATUS=0 reset, driven by the
                    // hung-task watchdog or operator action. We log
                    // any errno so a failed write surfaces in
                    // tracing rather than silently disappearing.
                    if let Err(e) = irq_evt.write(1) {
                        tracing::warn!(%e, "virtio-blk irq_evt.write failed");
                    }
                    break 'outer;
                }
                Err(e) => {
                    // Other iter() errors: `QueueNotReady` (the
                    // `ready()` gate above already filtered this;
                    // would only fire on a TOCTOU race with a
                    // vCPU-side reset MMIO write) or
                    // address-overflow on `avail_idx`. Log and
                    // bail — the kick is wasted but the device
                    // recovers on the next legitimate notify. Do
                    // NOT poison: these are not
                    // structural-invariant violations the way
                    // InvalidAvailRingIndex is, so a future
                    // legitimate kick may succeed.
                    tracing::warn!(%e, "virtio-blk iter() failed");
                    break 'outer;
                }
            };
            // Re-bind `q` after the iter-scoped guard drops so the
            // downstream `add_used` / `set_next_avail` /
            // `publish_completion` callers can hold a fresh mutable
            // borrow (the guard above released its lock when its
            // block expression returned).
            let q = &mut queues[REQ_QUEUE];
            let head = chain.head_index();

            // Walk the chain. Layout per virtio-v1.2 §5.2.6:
            //   - desc[0]: device-readable, 16-byte virtio_blk_outhdr
            //   - desc[1..N-1]: data segments (write-only for reads,
            //     read-only for writes; absent for flush)
            //   - desc[N-1]: device-writable, 1-byte status
            //
            // The kernel's `virtblk_add_req` always emits the status
            // descriptor last (drivers/block/virtio_blk.c). We rely
            // on that invariant: collect all descriptors, treat the
            // LAST one as the status candidate, the FIRST as the
            // header, everything in between as data segments. This
            // is simpler than the "first 1-byte write-only after
            // header" heuristic, which mis-classified chains
            // containing a 1-byte data descriptor.
            //
            // The first descriptor MUST be the header — it cannot
            // be write-only and cannot be shorter than the
            // `virtio_blk_outhdr` struct. A malformed first
            // descriptor must NOT silently fall through to a
            // later device-readable descriptor as the "header".
            // Re-use the device's scratch buffers across requests.
            // `clear()` keeps the underlying Vec capacity allocated
            // once at construction (sized by VIRTIO_BLK_SEG_MAX + 2),
            // so steady-state push/clear is amortized to zero
            // allocation. Hot-path optimization — drain_bracket_impl
            // runs on the worker thread in production (cfg(test): on
            // the test thread) and is invoked once per kick (one
            // per QUEUE_NOTIFY MMIO write in production).
            state.all_descs_scratch.clear();
            for desc in chain {
                state.all_descs_scratch.push(ChainDescriptor {
                    addr: desc.addr(),
                    len: desc.len(),
                    is_write_only: desc.is_write_only(),
                });
            }

            let chain_len = state.all_descs_scratch.len();

            let mut header_addr: Option<GuestAddress> = None;
            let mut status_addr: Option<GuestAddress> = None;
            if let Some((first, rest)) = state.all_descs_scratch.split_first() {
                if !first.is_write_only && (first.len as usize) >= VIRTIO_BLK_OUTHDR_SIZE {
                    header_addr = Some(first.addr);
                }
                if let Some((last, _middle)) = rest.split_last() {
                    // Status descriptor: device-writable, length >= 1.
                    // QEMU/firecracker/cloud-hypervisor all accept
                    // multi-byte status descriptors; the device
                    // writes the 1-byte status to the LAST byte of
                    // the descriptor (`last.addr + last.len - 1`)
                    // so the actual status-bearing position lines
                    // up with the kernel driver's `in_hdr` (the
                    // device-writable trailing buffer of the
                    // virtio_blk request, distinct from the
                    // device-readable `out_hdr` at desc[0]) per
                    // multi-byte in_hdr formats handled by
                    // `virtblk_vbr_status` at
                    // drivers/block/virtio_blk.c:329-332.
                    //
                    // `checked_add` defends against a hostile guest
                    // submitting `last.addr + last.len` near
                    // `u64::MAX`, which would wrap silently and let
                    // the device write a status byte at low GPA. On
                    // overflow `status_addr` stays None and the
                    // dispatcher drops the chain at the no-status
                    // gate.
                    if last.is_write_only && last.len >= 1 {
                        status_addr = last
                            .addr
                            .0
                            .checked_add(last.len as u64 - 1)
                            .map(GuestAddress);
                    }
                    // else: last descriptor isn't a valid status
                    // byte; status_addr stays None and the
                    // dispatcher's "no status descriptor" branch
                    // drops the chain. Data segments are not
                    // observed in the no-status path because the
                    // dispatcher returns before binding the
                    // data_segments slice.
                }
                // else: chain is exactly 1 descriptor → status
                // missing; both header (if valid) and status_addr
                // outcomes handled below.
            }

            // Validate chain shape and decode the header in one go.
            // Header missing or short → reject with S_IOERR if we
            // can identify the status descriptor; otherwise drop the
            // chain entirely (do NOT call `add_used`).
            //
            // A chain with no status descriptor MUST NOT be marked
            // used. The guest's `virtblk_done` reads the status from
            // `vbr->in_hdr.status` (drivers/block/virtio_blk.c
            // virtblk_vbr_status). That field is stale from prior
            // blk-mq tag use (initially zero from `__GFP_ZERO` at
            // allocation, stale on reuse), and `virtblk_result(0)`
            // maps to `BLK_STS_OK` — so calling `add_used` would
            // tell the guest the request SUCCEEDED when in fact the device
            // never wrote a status byte. That's a silent data
            // corruption vector for any guest read (the data buffer
            // is whatever was on the heap before the request) and a
            // silent dropped write for any guest write.
            //
            // Instead: leave the descriptor in the avail ring.
            // virtio_blk has no `mq_ops->timeout` callback (kernel
            // drivers/block/virtio_blk.c `virtio_mq_ops` has no
            // .timeout field), so blk-mq's per-request expiry path
            // (`blk_mq_rq_timed_out` in block/blk-mq.c) finds
            // `q->mq_ops->timeout == NULL`, skips the driver
            // callback, and falls through to `blk_add_timer` —
            // re-arming the same timer indefinitely. An unpublished
            // request therefore hangs the guest until either the
            // hung-task watchdog fires
            // (`kernel.hung_task_timeout_secs`, default 120s) or a
            // higher-layer (filesystem, application) retries. Hard
            // correctness requirement, not a performance trade-off.
            // Virtio-spec explicitly permits device-side stalls.
            // `io_errors` is bumped so the host operator sees the
            // malformed request.
            let Some(status_addr) = status_addr else {
                tracing::warn!(head, "virtio-blk request without status descriptor");
                state.counters.record_io_error();
                continue;
            };

            // SEG_MAX enforcement: the descriptor count includes the
            // header (1) + data segments (<= VIRTIO_BLK_SEG_MAX) +
            // status (1). Reject chains whose total count exceeds
            // `VIRTIO_BLK_SEG_MAX + 2`. Without this, the advertised
            // `seg_max` is a lie a hostile guest can ignore — it
            // could submit thousands of descriptors and force the
            // device to allocate matching scratch storage per
            // request. The check is placed AFTER status_addr
            // identification so the rejection produces a normal
            // IOERR completion (status byte write + add_used) rather
            // than dropping the chain entirely. Hoisting the check
            // earlier was the original design, but it left the
            // chain stuck in the avail ring with no path to error
            // surfacing — virtio_blk has no `mq_ops->timeout`
            // callback (drivers/block/virtio_blk.c `virtio_mq_ops`
            // has no `.timeout` field), so blk-mq alone never
            // surfaces the unpublished request; the guest only sees
            // the stall once the hung-task watchdog fires
            // (`kernel.hung_task_timeout_secs`, default 120 s).
            // Standard IOERR completion gives the guest's block
            // layer an immediate error to surface.
            if chain_len > VIRTIO_BLK_SEG_MAX as usize + 2 {
                tracing::warn!(
                    head,
                    desc_count = chain_len,
                    "virtio-blk chain exceeds seg_max + 2"
                );
                state.counters.record_io_error();
                if publish_completion(
                    mem,
                    q,
                    &state.counters,
                    head,
                    status_addr,
                    VIRTIO_BLK_S_IOERR as u8,
                    1,
                    "seg_max reject",
                ) {
                    signal_needed = true;
                }
                continue;
            }

            // When the header is missing/short but the status
            // descriptor is valid, publish IOERR via
            // `publish_completion` so the guest sees an immediate
            // error rather than hanging until the hung-task
            // watchdog fires (virtio_blk has no `mq_ops->timeout`).
            // `publish_completion` itself gates `add_used` on a
            // successful status-byte write — so a chain whose
            // status_addr is unmapped still ends up in the
            // "drop chain, request hangs the guest" branch via the
            // `false` return path (no add_used, no signal).
            // `io_errors` is bumped so the host operator sees the
            // malformed request.
            let Some(header_addr) = header_addr else {
                tracing::warn!(head, "virtio-blk request without valid header descriptor");
                state.counters.record_io_error();
                if publish_completion(
                    mem,
                    q,
                    &state.counters,
                    head,
                    status_addr,
                    VIRTIO_BLK_S_IOERR as u8,
                    1,
                    "bad header",
                ) {
                    signal_needed = true;
                }
                continue;
            };
            let hdr: VirtioBlkOutHdr = match mem.read_obj(header_addr) {
                Ok(h) => h,
                Err(_) => {
                    tracing::warn!(head, "virtio-blk header read failed");
                    state.counters.record_io_error();
                    if publish_completion(
                        mem,
                        q,
                        &state.counters,
                        head,
                        status_addr,
                        VIRTIO_BLK_S_IOERR as u8,
                        1,
                        "bad hdr read",
                    ) {
                        signal_needed = true;
                    }
                    continue;
                }
            };
            let req_type = hdr.type_;
            let sector = hdr.sector;
            // Borrow the chain's data-segment slice once. Sliced
            // directly from `all_descs_scratch[1..chain_len - 1]`
            // — header is at index 0, status is at index
            // `chain_len - 1` (we just unwrapped status_addr from
            // that descriptor), so everything in between is the
            // data payload. No separate Vec or copy.
            //
            // chain_len >= 2 here because status_addr is Some
            // (`split_last` produced a `last` element, which means
            // `rest.len() >= 1`, which means `chain_len >= 2`).
            // The slice is therefore always in-bounds.
            //
            // The borrow is immutable; `&state.all_descs_scratch[..]`
            // is disjoint from `&mut queues[..]` (the `q` borrow)
            // and `&mut state.ops_bucket` / `&mut state.bytes_bucket`,
            // so split-borrow lets all coexist.
            let data_segments: &[ChainDescriptor] = &state.all_descs_scratch[1..chain_len - 1];

            // SIZE_MAX enforcement: reject any chain that violates
            // the per-descriptor cap we advertised. A guest that
            // submits a descriptor longer than VIRTIO_BLK_SIZE_MAX
            // is either buggy or hostile; rejecting up-front
            // prevents the I/O handlers from `vec![0u8; len]`-ing
            // multi-gigabyte buffers under host control.
            if data_segments.iter().any(|d| d.len > VIRTIO_BLK_SIZE_MAX) {
                tracing::warn!(head, "virtio-blk descriptor exceeds size_max");
                state.counters.record_io_error();
                if publish_completion(
                    mem,
                    q,
                    &state.counters,
                    head,
                    status_addr,
                    VIRTIO_BLK_S_IOERR as u8,
                    1,
                    "size_max reject",
                ) {
                    signal_needed = true;
                }
                continue;
            }

            // Compute total data length (used for both throttle
            // accounting and the `add_used` length).
            let data_len: u64 = data_segments.iter().map(|d| d.len as u64).sum();

            // Zero-data T_IN/T_OUT/T_GET_ID must IOERR. virtio-v1.2
            // §5.2.6 defines IN/OUT as carrying a non-empty data
            // payload; §5.2.6.4 defines GET_ID as writing a 20-byte
            // string into a device-writable data segment — a chain
            // with only header + status has no destination buffer.
            // cloud-hypervisor explicitly rejects this for
            // IN/OUT; firecracker rejects sub-20-byte GET_ID via the
            // handler's `data_len < VIRTIO_BLK_ID_BYTES` arm. We
            // hoist the empty case here so the throttle bucket is
            // never charged for a request the handler will reject
            // anyway. T_FLUSH is exempt — flush carries no data by
            // design (kernel `virtblk_setup_cmd` sets
            // `vbr->in_hdr_len = sizeof(status)` for flushes).
            if matches!(
                req_type,
                VIRTIO_BLK_T_IN | VIRTIO_BLK_T_OUT | VIRTIO_BLK_T_GET_ID
            ) && data_segments.is_empty()
            {
                tracing::warn!(
                    head,
                    req_type,
                    "virtio-blk T_IN/T_OUT/T_GET_ID with no data segments"
                );
                state.counters.record_io_error();
                if publish_completion(
                    mem,
                    q,
                    &state.counters,
                    head,
                    status_addr,
                    VIRTIO_BLK_S_IOERR as u8,
                    1,
                    "zero-data",
                ) {
                    signal_needed = true;
                }
                continue;
            }

            // Sector-granular transfer requirement. virtio-v1.2
            // §5.2.6 defines T_IN/T_OUT in terms of sector-aligned
            // transfers; a sub-sector data length is malformed.
            // firecracker rejects this in
            // src/vmm/src/devices/virtio/block/virtio/request.rs
            // (Request::parse). A buggy or malicious guest that
            // submits e.g. 513 bytes would otherwise reach
            // handle_read_impl/handle_write_impl, which compute
            // offsets in 512-byte units but transfer arbitrary
            // byte counts — the resulting access straddles a
            // sector boundary in a way the host filesystem and
            // backing-file accounting do not expect. Reject up
            // front so the throttle bucket is never charged.
            if matches!(req_type, VIRTIO_BLK_T_IN | VIRTIO_BLK_T_OUT)
                && !data_len.is_multiple_of(VIRTIO_BLK_SECTOR_SIZE as u64)
            {
                tracing::warn!(
                    head,
                    req_type,
                    data_len,
                    "virtio-blk T_IN/T_OUT data_len not a multiple of 512"
                );
                state.counters.record_io_error();
                if publish_completion(
                    mem,
                    q,
                    &state.counters,
                    head,
                    status_addr,
                    VIRTIO_BLK_S_IOERR as u8,
                    1,
                    "sub-sector",
                ) {
                    signal_needed = true;
                }
                continue;
            }

            // Pre-throttle terminal classifications: read-only-mode
            // writes, no-op read-only-mode flushes, and unsupported
            // request types are decided BEFORE consuming throttle
            // tokens. Burning IOPS/bytes budget on a request the
            // device is going to reject anyway is a correctness
            // hazard for tight throttle limits — the guest sees
            // intermittent IOERR on legitimate retries because the
            // bucket was drained by a request that never had a
            // chance to succeed.
            //
            // read_only is checked against the host-owned
            // `self.read_only` field, NOT against re-read guest
            // memory. The header was read once into `hdr` above and
            // not consulted again — no TOCTOU.
            let backing = &state.backing;
            let counters = state.counters.as_ref();
            let cap_bytes = state.capacity_bytes;
            let read_only = state.read_only;
            let pre_throttle = VirtioBlk::classify_pre_throttle(req_type, read_only, counters);

            // Direction validation, hoisted out of
            // handle_read_impl/handle_write_impl/handle_get_id_impl
            // so it runs BEFORE the throttle bucket is consumed.
            // virtio-v1.2 §5.2.6: T_IN data segments must be
            // device-writable (is_write_only); T_OUT data segments
            // must be device-readable (!is_write_only). T_GET_ID
            // (§5.2.6.4) writes a 20-byte string into a
            // device-writable data segment, matching T_IN's
            // direction (cloud-hypervisor and firecracker both
            // reject non-write-only data segments for GET_ID). A
            // request whose data SG direction violates the spec is
            // rejected unconditionally — running it would either
            // read host data into a guest-readable-only buffer
            // (T_IN/T_GET_ID) or write guest-writable buffers to
            // the backing file (T_OUT), neither of which the
            // kernel driver expects. Pre-throttle classifications
            // skip this — RO writes and unsupported requests are
            // already terminal and never dispatch. The redundant
            // per-segment check remains in
            // handle_read_impl/handle_write_impl as
            // defense-in-depth in case a future caller bypasses
            // this gate.
            let direction_violation = pre_throttle.is_none()
                && match req_type {
                    VIRTIO_BLK_T_IN | VIRTIO_BLK_T_GET_ID => {
                        data_segments.iter().any(|d| !d.is_write_only)
                    }
                    VIRTIO_BLK_T_OUT => data_segments.iter().any(|d| d.is_write_only),
                    _ => false,
                };
            if direction_violation {
                tracing::warn!(
                    head,
                    req_type,
                    "virtio-blk T_IN/T_OUT/T_GET_ID data segment direction mismatch"
                );
                state.counters.record_io_error();
                if publish_completion(
                    mem,
                    q,
                    &state.counters,
                    head,
                    status_addr,
                    VIRTIO_BLK_S_IOERR as u8,
                    1,
                    "direction",
                ) {
                    signal_needed = true;
                }
                continue;
            }

            // Throttle: consume 1 op + data_len bytes. If either
            // bucket fails, undo the pop with
            // `set_next_avail(prev.wrapping_sub(1))`, bump
            // `throttled_count`, compute a `wait_nanos` from the
            // bucket's refill rate, and return
            // `DrainOutcome::ThrottleStalled`. The chain stays
            // invisible to the guest (no add_used, no status byte,
            // no irqfd, no `io_errors` bump) until the worker's
            // retry timer fires (`THROTTLE_TOKEN`). The bucket
            // never sleeps — `can_consume` always returns
            // promptly, so the worker stays responsive to
            // STOP_TOKEN and KICK_TOKEN. virtio-spec doesn't
            // reserve a "throttled" status code; deferring the
            // chain is preferable to surfacing transient errors
            // to the guest (which would otherwise see spurious
            // S_IOERRs that confuse the guest's filesystem or
            // application retry semantics).
            //
            // Both buckets are checked first via `can_consume` and
            // only consumed once both pass. Short-circuiting on
            // `consume()` would burn the ops token whenever the
            // bytes check failed (or vice versa), depending on
            // operand order — losing budget to a request that
            // never serviced.
            //
            // FLUSH counts against IOPS, but only when FLUSH
            // actually dispatches to the backend. RO-mode flushes
            // are pre-classified above and never reach here, so
            // they don't touch the bucket.
            if pre_throttle.is_none() {
                let ops_ok = state.ops_bucket.can_consume(1);
                let bytes_ok = state.bytes_bucket.can_consume(data_len);
                if !ops_ok || !bytes_ok {
                    // Throttle exhausted: undo the pop and stall the
                    // drain. The chain stays invisible to the guest
                    // (no add_used, no S_IOERR, no irqfd) until the
                    // worker's retry timer fires and re-drains. The
                    // `wait_nanos` value covers both buckets — pick
                    // the longer of the two waits because both must
                    // hold enough tokens before the request can run.
                    // `set_next_avail(prev - 1)` rewinds the queue's
                    // tracking cursor by one, so the next pop returns
                    // this same chain head — preserving FIFO order
                    // across the stall.  We use this instead of
                    // `go_to_previous_position` because that helper
                    // is on `QueueOwnedT`, which `QueueSync` does not
                    // implement; `set_next_avail` is on the base
                    // `QueueT` and works for both alias targets.
                    // `wrapping_sub` matches the queue's u16 wrap
                    // semantics (next_avail wraps modulo 2^16, the
                    // virtio ring counter width).
                    state.counters.record_throttled();
                    // Live gauge: only increment on the
                    // false → true transition. Re-stalls on the
                    // same head (currently_stalled already true)
                    // bump throttled_count (events) but do NOT
                    // double-bump the gauge. See the
                    // BlkWorkerState::currently_stalled doc for
                    // the transition table.
                    if !state.currently_stalled {
                        state.currently_stalled = true;
                        state.counters.record_throttle_pending_inc();
                    }
                    let prev = queues[REQ_QUEUE].next_avail();
                    queues[REQ_QUEUE].set_next_avail(prev.wrapping_sub(1));
                    let ops_wait = if !ops_ok {
                        state.ops_bucket.nanos_until_n_tokens(1)
                    } else {
                        0
                    };
                    let bytes_wait = if !bytes_ok {
                        state.bytes_bucket.nanos_until_n_tokens(data_len)
                    } else {
                        0
                    };
                    let wait_nanos = ops_wait.max(bytes_wait);
                    tracing::trace!(
                        head,
                        ops_ok,
                        bytes_ok,
                        wait_nanos,
                        "virtio-blk throttle stall; rolling back chain"
                    );
                    stall_outcome = Some(wait_nanos);
                    break;
                }
                // Both checks passed — consume now. Each bucket's
                // `consume` does its own refill+capacity check, so
                // the post-can_consume window can't see a smaller
                // bucket here (refills are monotone-non-negative).
                let ops_consumed = state.ops_bucket.consume(1);
                let bytes_consumed = state.bytes_bucket.consume(data_len);
                debug_assert!(
                    ops_consumed && bytes_consumed,
                    "throttle invariant: can_consume must imply consume",
                );
                // Live gauge: if a prior stall left the gauge
                // incremented, the chain that just satisfied the
                // throttle gate is the head-of-queue stalled
                // chain. Decrement the gauge once the tokens have
                // been consumed — from the throttle-pending
                // perspective, the chain has exited the "waiting
                // for tokens" state. Decrement BEFORE dispatch so
                // a backing-file IO error in the handler doesn't
                // leave the gauge pinned (success/IO-error
                // outcomes are accounted separately, downstream).
                if state.currently_stalled {
                    state.currently_stalled = false;
                    state.counters.record_throttle_pending_dec();
                }
            }

            // Service the request. Handlers compute the status
            // byte + used_len but do NOT write the status byte
            // themselves; this loop performs the status write +
            // add_used as a single "publish completion" step so
            // that a failed status write skips add_used.
            let (status_byte, used_len) = if let Some(out) = pre_throttle {
                out
            } else {
                // Pass `data_len` already computed above so handlers
                // don't re-derive it (was a third sum() pass each).
                // Pass `&mut state.io_buf_scratch` as a reusable
                // per-segment buffer; handlers `resize(len, 0)` it
                // per descriptor and the underlying `Vec<u8>`
                // capacity grows monotonically up to
                // VIRTIO_BLK_SIZE_MAX, then steady-state is zero
                // allocation per segment.
                match req_type {
                    VIRTIO_BLK_T_IN => VirtioBlk::handle_read_impl(
                        backing,
                        cap_bytes,
                        counters,
                        mem,
                        sector,
                        data_segments,
                        data_len,
                        &mut state.io_buf_scratch,
                    ),
                    VIRTIO_BLK_T_OUT => VirtioBlk::handle_write_impl(
                        backing,
                        cap_bytes,
                        counters,
                        mem,
                        sector,
                        data_segments,
                        data_len,
                        &mut state.io_buf_scratch,
                    ),
                    VIRTIO_BLK_T_FLUSH => VirtioBlk::handle_flush_impl(backing, counters),
                    VIRTIO_BLK_T_GET_ID => {
                        VirtioBlk::handle_get_id_impl(counters, mem, data_segments)
                    }
                    // Defense-in-depth fall-through. classify_pre_throttle's
                    // catch-all `_ => Some((VIRTIO_BLK_S_UNSUPP, 1))` arm
                    // means this branch is unreachable today — but a future
                    // patch that adds a new variant to the
                    // `T_IN | T_OUT | T_FLUSH | T_GET_ID => None` arm
                    // without updating this match would otherwise panic the
                    // thread running drain_bracket_impl. Return S_UNSUPP and
                    // bump io_errors so the
                    // regression surfaces as a guest-visible error and a
                    // counter bump rather than a panic that kills the VM.
                    _ => {
                        counters.record_io_error();
                        (VIRTIO_BLK_S_UNSUPP as u8, 1)
                    }
                }
            };
            // Per-request log line. Level is `trace!`, not `debug!`,
            // because the device handles thousands of requests
            // per second under load — emitting at debug! would
            // drown out everything else in the default
            // RUST_LOG=info,ktstr=debug operator setting. Anomaly
            // events (rejected request, IOERR) log at `warn!` so
            // they always surface; throttle stalls log at `trace!`
            // (see "throttle stall; rolling back chain" above)
            // because they are deferred-not-failed and would flood
            // logs on a tight throttle. This per-request line is
            // the "happy path" record. The failure-path warns
            // above use the same field set (head, sector, etc.)
            // so log-grep correlation works.
            //
            // Map `req_type` to a human-readable string (rather
            // than the bare u32). The numeric value is preserved
            // as `req_type_raw` for cases where an unknown variant
            // slipped past `classify_pre_throttle` and the
            // operator wants the wire value.
            let req_type_name = match req_type {
                VIRTIO_BLK_T_IN => "in",
                VIRTIO_BLK_T_OUT => "out",
                VIRTIO_BLK_T_FLUSH => "flush",
                VIRTIO_BLK_T_GET_ID => "get_id",
                _ => "unsupp",
            };
            tracing::trace!(
                req_type = req_type_name,
                req_type_raw = req_type,
                sector,
                head,
                status = status_byte,
                used_len,
                "virtio-blk request done"
            );
            // Write status, then add_used ONLY if the status write
            // succeeded. `Queue::add_used` writes the descriptor
            // head/len via write_obj, then publishes used.idx with
            // Ordering::Release, so the prior status-byte
            // write_slice is ordered before the guest sees the new
            // index. The chain has already been dropped (the for
            // loop above consumed it), so this `q` re-borrow is
            // legal.
            //
            // `used_len` from the handlers measures bytes the device
            // wrote into guest memory (data + 1 status byte for
            // reads; 1 status byte for writes/flushes). When the
            // status descriptor is multi-byte we still report only
            // the bytes we wrote, not the descriptor's full length.
            if publish_completion(
                mem,
                q,
                &state.counters,
                head,
                status_addr,
                status_byte,
                used_len,
                "publish completion",
            ) {
                signal_needed = true;
            }
        }
        // Throttle stall: the inner loop's `break` (without
        // continue) ran because of `stall_outcome = Some(_)`.
        // Re-enable notifications so the guest can wake the
        // device when it adds new chains, then break the outer
        // loop. Bail unconditionally on stall to keep the path
        // simple; the worker's retry timer drives the
        // re-attempt regardless of whether the bucket happens
        // to have refilled by then.
        if stall_outcome.is_some() {
            if let Err(e) = queues[REQ_QUEUE].enable_notification(mem) {
                tracing::warn!(
                    %e,
                    "virtio-blk enable_notification failed on throttle stall"
                );
            }
            break 'outer;
        }
        // Inner drain ran to None. Re-arm notifications and
        // check whether new chains arrived during the disabled
        // window. `enable_notification` returns Ok(true) when
        // `avail_idx != next_avail` after re-enabling — those
        // chains MUST be processed before exiting or they'll
        // be stranded (V3: honour the return value).
        match queues[REQ_QUEUE].enable_notification(mem) {
            Ok(true) => continue 'outer,
            Ok(false) => break 'outer,
            Err(e) => {
                // A persistent enable failure (e.g. used-ring
                // GPA unmapped) would otherwise spin the outer
                // loop forever. Bail to avoid a livelock; on
                // the next QUEUE_NOTIFY the guest may have
                // recovered guest memory layout.
                tracing::warn!(%e, "virtio-blk enable_notification failed");
                break 'outer;
            }
        }
    }
    if signal_needed {
        // V8: always set the interrupt_status MMIO bit when
        // anything was published. The bit-set on `interrupt_status`
        // is the IRQ-handler handshake target — `vm_interrupt`
        // (drivers/virtio/virtio_mmio.c) reads and acks it on each
        // IRQ delivery. The guest does NOT poll this register; it
        // only consults it from inside the IRQ handler. If the irqfd
        // write fails, the guest never enters `vm_interrupt` and the
        // queue stalls — the bit remaining set is harmless on its
        // own; only IRQ delivery makes the guest read it.
        // Release-ordered fetch_or so the bit-set happens-after
        // the chain's add_used publish. The SeqCst fence inside
        // needs_notification then orders all prior writes
        // (including add_used and this bit-set) against the
        // used_event read that drives the IRQ decision. Result:
        // a vCPU reading INTERRUPT_STATUS via Acquire-load and
        // finding INT_VRING set is guaranteed to also observe
        // the freshly-published used.idx — no torn observation
        // where the bit appears before the ring update.
        interrupt_status.fetch_or(VIRTIO_MMIO_INT_VRING, Ordering::Release);
        // `Queue::needs_notification` consults the guest's
        // `used_event` threshold (from the avail ring) when
        // EVENT_IDX is negotiated — returns false if the guest
        // hasn't asked to be woken yet, true otherwise. In the
        // legacy path (event_idx_enabled=false) it always
        // returns Ok(true) (the trailing `Ok(true)` arm of
        // `Queue::needs_notification`), so the eventfd fires
        // every time as before.
        //
        // V6: only call `needs_notification` on the
        // signal_needed=true path. The method has side effects
        // (resets `num_added` to zero — see the doc comment on
        // `QueueT::needs_notification`) so calling it
        // speculatively would corrupt the suppression state.
        //
        // unwrap_or(true): on guest-memory errors reading the
        // `used_event` field, fail-safe to firing the IRQ. A
        // missed IRQ stalls the guest until the hung-task
        // watchdog fires (`kernel.hung_task_timeout_secs`,
        // default 120 s — virtio_blk has no `mq_ops->timeout`
        // so blk-mq alone never surfaces the stall); a
        // redundant IRQ wastes a vCPU exit.
        let q = &mut queues[REQ_QUEUE];
        if q.needs_notification(mem)
            .inspect_err(
                |e| tracing::warn!(%e, "needs_notification failed; firing IRQ as fail-safe"),
            )
            .unwrap_or(true)
        {
            // SAFETY: EAGAIN requires counter saturation at u64::MAX-1
            // (~1.8e19 unobserved kicks) — implausible. EBADF means
            // the fd closed during shutdown. The simultaneously-set
            // INT_VRING bit at the `interrupt_status.fetch_or` above
            // is the next IRQ handler's read target — but only if a
            // SUBSEQUENT request fires `irq_evt.write` successfully.
            // For the last chain in a burst with no follow-on
            // traffic, a missed write means the queue stalls until
            // hung_task_timeout (default 120s; virtio_blk has no
            // `mq_ops->timeout`). The recovery path here is the
            // next add_used + needs_notification cycle (the next
            // request's publish reaches this site again), NOT a
            // kernel timer (virtio_mmio has no periodic wake
            // mechanism). We log any errno so a failed write
            // surfaces in tracing rather than silently disappearing.
            if let Err(e) = irq_evt.write(1) {
                tracing::warn!(%e, "virtio-blk irq_evt.write failed");
            }
        }
    }
    match stall_outcome {
        Some(wait_nanos) => DrainOutcome::ThrottleStalled { wait_nanos },
        None => DrainOutcome::Done,
    }
}
impl VirtioBlk {
    /// Service `VIRTIO_BLK_T_IN` (read). Reads bytes from the
    /// backing file at `sector * 512` into the device-writable
    /// guest segments (scatter). Returns `(status_byte, used_len)`;
    /// the CALLER is responsible for writing `status_byte` to the
    /// status descriptor and calling `add_used` only when the
    /// status write succeeded — publishing a completion the guest
    /// can't observe is worse than dropping the chain.
    ///
    /// `checked_mul` is defense-in-depth against a sector value
    /// large enough to overflow `sector * 512` as u64. The
    /// downstream out-of-range check (`base_offset + total_data <=
    /// capacity_bytes`) would also catch most overflow cases on a
    /// reasonable capacity, but a checked multiply costs nothing
    /// and removes any worry about wrap-then-underflow corner
    /// cases when computing the post-multiply offset.
    ///
    /// Free function (not `&self`-method) so the caller can pass
    /// disjoint field borrows individually — `&self.backing`,
    /// `&self.counters`, and `self.capacity_sectors` (Copy) — and
    /// hold a concurrent `&mut self.queues[..]` borrow for
    /// `add_used`. A `&self`-method would have to borrow the whole
    /// receiver and conflict with the queue mutation in
    /// `process_requests`.
    ///
    /// `too_many_arguments` allow: deliberate disjoint-borrow
    /// shape — every parameter is a separate `&self` field that
    /// must be passed by reference so the caller can hold a
    /// concurrent mutable borrow of the queues vec.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn handle_read_impl(
        backing: &File,
        capacity_bytes: u64,
        counters: &VirtioBlkCounters,
        mem: &GuestMemoryMmap,
        sector: u64,
        data_segments: &[ChainDescriptor],
        data_len: u64,
        scratch: &mut Vec<u8>,
    ) -> (u8, u32) {
        let mut total_read: u32 = 0;
        let Some(base_offset) = sector.checked_mul(VIRTIO_BLK_SECTOR_SIZE as u64) else {
            counters.record_io_error();
            return (VIRTIO_BLK_S_IOERR as u8, 1);
        };
        // Read past EOF always returns S_IOERR. Capacity is fixed
        // at construction; auto-grow is not a v0 behaviour. A read
        // whose byte range extends past `capacity_bytes` fails
        // entirely — no partial-success short-read model — and
        // bumps `io_errors`. `capacity_bytes` is computed once in
        // `with_options` and threaded down — no per-request multiply.
        if base_offset
            .checked_add(data_len)
            .is_none_or(|end| end > capacity_bytes)
        {
            counters.record_io_error();
            return (VIRTIO_BLK_S_IOERR as u8, 1);
        }

        // Zero-length data segment: the empty-slice path is
        // intentional. The for-loop body runs unconditionally so
        // direction-mismatch checks (`!is_write_only`) still
        // apply; `read_at` against a zero-length slice is `Ok(0)`,
        // so `total_read`/`cur_offset` are unchanged and the chain
        // proceeds to `S_OK` once all segments are walked. A guest
        // that submits a zero-length data descriptor has issued a
        // weird-but-legal request, not a malformed one — qemu and
        // firecracker behave the same way. This is an explicit
        // design choice, not an accidental fall-through.
        let mut cur_offset = base_offset;
        for seg in data_segments {
            if !seg.is_write_only {
                // Spec violation — a read request's data SGs must
                // be device-writable. Defense-in-depth: the outer
                // gate in process_requests already rejected this
                // chain before throttle. Kept in case a future
                // caller reaches handle_read_impl directly.
                counters.record_io_error();
                return (VIRTIO_BLK_S_IOERR as u8, total_read + 1);
            }
            // Reuse the device-owned scratch buffer.
            // `resize(len, 0)` zero-fills any new tail; the existing
            // capacity is preserved. Bytes leftover from the prior
            // segment are overwritten by `read_at`, then
            // zero-padded only on a short read (below).
            scratch.resize(seg.len as usize, 0);
            match backing.read_at(&mut scratch[..], cur_offset) {
                Ok(n) => {
                    if (n as u32) < seg.len {
                        // Short read — pad with zeros (sparse file
                        // semantics).
                        scratch[n..].fill(0);
                    }
                    if mem.write_slice(&scratch[..], seg.addr).is_err() {
                        counters.record_io_error();
                        return (VIRTIO_BLK_S_IOERR as u8, total_read + 1);
                    }
                    total_read += seg.len;
                    cur_offset += seg.len as u64;
                }
                Err(e) => {
                    tracing::warn!(sector, %e, "virtio-blk read error");
                    counters.record_io_error();
                    return (VIRTIO_BLK_S_IOERR as u8, total_read + 1);
                }
            }
        }
        counters.record_read(total_read as u64);
        // used_len: data bytes written to guest + 1 status byte.
        (VIRTIO_BLK_S_OK as u8, total_read + 1)
    }

    /// Service `VIRTIO_BLK_T_OUT` (write). Reads bytes from the
    /// device-readable guest segments (gather) and writes them to
    /// the backing file at `sector * 512`. Returns
    /// `(status_byte, used_len)`; caller writes the status byte
    /// to the status descriptor and gates `add_used` on a
    /// successful status write. `checked_mul` matches
    /// `handle_read_impl` — same overflow concern.
    ///
    /// `too_many_arguments` allow: same disjoint-borrow shape as
    /// [`Self::handle_read_impl`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn handle_write_impl(
        backing: &File,
        capacity_bytes: u64,
        counters: &VirtioBlkCounters,
        mem: &GuestMemoryMmap,
        sector: u64,
        data_segments: &[ChainDescriptor],
        data_len: u64,
        scratch: &mut Vec<u8>,
    ) -> (u8, u32) {
        let Some(base_offset) = sector.checked_mul(VIRTIO_BLK_SECTOR_SIZE as u64) else {
            counters.record_io_error();
            return (VIRTIO_BLK_S_IOERR as u8, 1);
        };
        // Write past EOF always returns S_IOERR. The disk is a
        // fixed-capacity virtio-blk device; auto-growing the
        // backing file would silently change the reported
        // config-space `capacity_sectors` and the guest partition
        // table would not see the new sectors without a
        // capacity-change notification path. Out-of-range writes
        // are a guest-side bug or a malicious request — fail
        // closed. `capacity_bytes` is computed once in
        // `with_options` and threaded down — no per-request multiply.
        if base_offset
            .checked_add(data_len)
            .is_none_or(|end| end > capacity_bytes)
        {
            counters.record_io_error();
            return (VIRTIO_BLK_S_IOERR as u8, 1);
        }

        let mut cur_offset = base_offset;
        let mut total_written: u32 = 0;
        for seg in data_segments {
            if seg.is_write_only {
                // Spec violation — a write request's data SGs must
                // be device-readable. Defense-in-depth: the outer
                // gate in process_requests already rejected this
                // chain before throttle. Kept in case a future
                // caller reaches handle_write_impl directly.
                counters.record_io_error();
                return (VIRTIO_BLK_S_IOERR as u8, 1);
            }
            // Reuse the device-owned scratch buffer.
            scratch.resize(seg.len as usize, 0);
            if mem.read_slice(&mut scratch[..], seg.addr).is_err() {
                counters.record_io_error();
                return (VIRTIO_BLK_S_IOERR as u8, 1);
            }
            match backing.write_at(&scratch[..], cur_offset) {
                Ok(n) if (n as u32) == seg.len => {
                    total_written += seg.len;
                    cur_offset += seg.len as u64;
                }
                // Both partial write (`Ok(n)` with `n < seg.len`) and
                // hard error (`Err(_)`) collapse to S_IOERR + an
                // `io_errors` bump. From the guest's perspective the
                // request was not fulfilled in full, which is the same
                // failure signal — and counting partial writes as
                // io_errors keeps failure dumps honest. Note this
                // differs from the unsupported-type path, which sets
                // S_UNSUPP without bumping any counter (see
                // `classify_pre_throttle`). A future change that
                // wants to retry partial writes internally must not
                // silently suppress the `io_errors` increment when
                // the retry eventually fails — that signal is what
                // surfaces backing-store distress in failure dumps.
                Ok(_) | Err(_) => {
                    counters.record_io_error();
                    return (VIRTIO_BLK_S_IOERR as u8, 1);
                }
            }
        }
        counters.record_write(total_written as u64);
        // used_len: 1 (status byte only — write data is not written
        // back into guest mem).
        (VIRTIO_BLK_S_OK as u8, 1)
    }

    /// Service `VIRTIO_BLK_T_FLUSH`. `fdatasync(2)` on the backing.
    /// Returns `(status_byte, used_len)`; caller writes the status
    /// byte and gates `add_used` on a successful status write.
    pub(crate) fn handle_flush_impl(backing: &File, counters: &VirtioBlkCounters) -> (u8, u32) {
        let status = match backing.sync_data() {
            Ok(()) => {
                counters.record_flush();
                VIRTIO_BLK_S_OK as u8
            }
            Err(e) => {
                tracing::warn!(%e, "virtio-blk flush error");
                counters.record_io_error();
                VIRTIO_BLK_S_IOERR as u8
            }
        };
        (status, 1)
    }

    /// Service `VIRTIO_BLK_T_GET_ID` (virtio-v1.2 §5.2.6.4). Writes
    /// the device's 20-byte serial string into the FIRST data
    /// descriptor and returns `(status_byte, used_len)` where
    /// `used_len = VIRTIO_BLK_ID_BYTES + 1` on success (20 data
    /// bytes + 1 status byte). Caller publishes the status byte and
    /// gates `add_used` on a successful status write.
    ///
    /// The kernel driver `virtblk_get_id`
    /// (drivers/block/virtio_blk.c) maps a single 20-byte buffer
    /// via `blk_rq_map_kern(req, id_str, VIRTIO_BLK_ID_BYTES,
    /// GFP_KERNEL)`, so a well-formed chain has exactly one data
    /// descriptor of length >= 20. Multi-descriptor chains are
    /// theoretically legal under the spec but never produced by
    /// the kernel driver; we honor the kernel's contract by
    /// writing into the first descriptor only — matching
    /// firecracker's `process_get_device_id` and libkrun's
    /// `worker.rs` arm. If the first data descriptor is shorter
    /// than 20 bytes the request is rejected with `S_IOERR`
    /// (firecracker, cloud-hypervisor, libkrun all reject;
    /// QEMU truncates instead — we diverge intentionally because
    /// a guest that hands us a too-small buffer is already buggy
    /// and partial-data is a silent footgun).
    ///
    /// The data descriptor's direction has already been validated
    /// by the outer `direction_violation` gate in
    /// `process_requests` (T_GET_ID requires write-only); the
    /// per-segment direction check below is defense-in-depth for
    /// callers that bypass the gate.
    ///
    /// Free function (not `&self`-method) so the caller can pass
    /// disjoint field borrows individually — matching
    /// `handle_read_impl` / `handle_write_impl` for the same
    /// borrow-checker reason (`process_requests` holds
    /// `&mut self.queues[..]`).
    pub(crate) fn handle_get_id_impl(
        counters: &VirtioBlkCounters,
        mem: &GuestMemoryMmap,
        data_segments: &[ChainDescriptor],
    ) -> (u8, u32) {
        // First data descriptor receives the serial. The empty
        // case is filtered upstream by the zero-data gate, so
        // `first()` is always Some at production reach.
        // Defense-in-depth: still handle the empty slice by
        // returning S_IOERR rather than panicking on
        // `data_segments[0]` indexing.
        let Some(first) = data_segments.first() else {
            counters.record_io_error();
            return (VIRTIO_BLK_S_IOERR as u8, 1);
        };
        if !first.is_write_only {
            // Spec violation — GET_ID's data SG must be
            // device-writable. Defense-in-depth: the outer gate in
            // process_requests already rejected this chain before
            // throttle. Kept in case a future caller reaches
            // handle_get_id_impl directly.
            counters.record_io_error();
            return (VIRTIO_BLK_S_IOERR as u8, 1);
        }
        if first.len < VIRTIO_BLK_ID_BYTES {
            // Buffer too small — kernel driver always passes
            // exactly VIRTIO_BLK_ID_BYTES (20). Reject rather than
            // truncate: matches firecracker / cloud-hypervisor /
            // libkrun. A truncated serial would surface as a
            // garbled `/sys/block/<dev>/serial` value, which is
            // worse than an explicit IOERR (the guest's
            // `serial_show` maps -EIO to an empty string).
            counters.record_io_error();
            return (VIRTIO_BLK_S_IOERR as u8, 1);
        }
        if mem.write_slice(&VIRTIO_BLK_SERIAL[..], first.addr).is_err() {
            counters.record_io_error();
            return (VIRTIO_BLK_S_IOERR as u8, 1);
        }
        // used_len: 20 data bytes written + 1 status byte. Symmetric
        // with handle_read_impl's `total_read + 1` accounting.
        (VIRTIO_BLK_S_OK as u8, VIRTIO_BLK_ID_BYTES + 1)
    }

    // ------------------------------------------------------------------
    // MMIO register dispatch
    // ------------------------------------------------------------------

    /// Test-only `&self` proxies for the request handlers. The
    /// production `process_requests` invokes the free-function
    /// associated forms (`Self::handle_*_impl`) so that the
    /// `&mut self.queues[..]` borrow in the request loop doesn't
    /// conflict with `&self`. Tests prefer the method form for
    /// brevity.
    ///
    /// Wrappers also write the status byte themselves before
    /// returning — the production caller (`process_requests`) does
    /// this as part of its publish-completion step, so test
    /// helpers replicate it for convenience.
    #[cfg(test)]
    pub(crate) fn handle_read(
        &self,
        mem: &GuestMemoryMmap,
        sector: u64,
        data_segments: &[ChainDescriptor],
        status_addr: GuestAddress,
    ) -> (u8, u32) {
        let data_len: u64 = data_segments.iter().map(|d| d.len as u64).sum();
        let mut scratch = Vec::new();
        let s = self.worker.state();
        let (status, used_len) = Self::handle_read_impl(
            &s.backing,
            s.capacity_bytes,
            s.counters.as_ref(),
            mem,
            sector,
            data_segments,
            data_len,
            &mut scratch,
        );
        mem.write_slice(&[status], status_addr)
            .expect("write status in test wrapper");
        (status, used_len)
    }

    #[cfg(test)]
    pub(crate) fn handle_write(
        &self,
        mem: &GuestMemoryMmap,
        sector: u64,
        data_segments: &[ChainDescriptor],
        status_addr: GuestAddress,
    ) -> (u8, u32) {
        let data_len: u64 = data_segments.iter().map(|d| d.len as u64).sum();
        let mut scratch = Vec::new();
        let s = self.worker.state();
        let (status, used_len) = Self::handle_write_impl(
            &s.backing,
            s.capacity_bytes,
            s.counters.as_ref(),
            mem,
            sector,
            data_segments,
            data_len,
            &mut scratch,
        );
        mem.write_slice(&[status], status_addr)
            .expect("write status in test wrapper");
        (status, used_len)
    }

    #[cfg(test)]
    pub(crate) fn handle_flush(
        &self,
        mem: &GuestMemoryMmap,
        status_addr: GuestAddress,
    ) -> (u8, u32) {
        let s = self.worker.state();
        let (status, used_len) = Self::handle_flush_impl(&s.backing, s.counters.as_ref());
        mem.write_slice(&[status], status_addr)
            .expect("write status in test wrapper");
        (status, used_len)
    }

    #[cfg(test)]
    pub(crate) fn handle_get_id(
        &self,
        mem: &GuestMemoryMmap,
        data_segments: &[ChainDescriptor],
        status_addr: GuestAddress,
    ) -> (u8, u32) {
        let s = self.worker.state();
        let (status, used_len) = Self::handle_get_id_impl(s.counters.as_ref(), mem, data_segments);
        mem.write_slice(&[status], status_addr)
            .expect("write status in test wrapper");
        (status, used_len)
    }

    /// Handle MMIO read at `offset` within the device's MMIO region.
    ///
    /// Two address ranges:
    /// - `offset >= 0x100`: device-specific config space, dispatched
    ///   to `read_blk_config`.
    /// - `offset < 0x100`: virtio-mmio common transport registers
    ///   (magic/version/device-id, status, queue config, interrupt
    ///   status). All transport registers are 4-byte u32; non-4-byte
    ///   reads here are guest bugs.
    ///
    /// Non-4-byte fallback fills `data` with `0xff` rather than 0
    /// because 0xff is far easier to spot in a guest crash dump or
    /// hex view than a successful 0 — it surfaces "the device
    /// declined to answer" instead of disguising it as a valid
    /// zero-valued register read. Config space (`offset >= 0x100`)
    /// uses 0-fill instead because virtio-v1.2 §4.2.2.2 specifies
    /// reads past the populated config layout return zero.
    pub fn mmio_read(&self, offset: u64, data: &mut [u8]) {
        if offset >= 0x100 {
            self.read_blk_config(offset - 0x100, data);
            return;
        }
        if data.len() != 4 {
            data.fill(0xff);
            return;
        }
        let val: u32 = match offset as u32 {
            VIRTIO_MMIO_MAGIC_VALUE => MMIO_MAGIC,
            VIRTIO_MMIO_VERSION => MMIO_VERSION,
            VIRTIO_MMIO_DEVICE_ID => VIRTIO_ID_BLOCK,
            VIRTIO_MMIO_VENDOR_ID => VENDOR_ID,
            VIRTIO_MMIO_DEVICE_FEATURES => {
                let page = self.device_features_sel;
                if page == 0 {
                    self.device_features() as u32
                } else if page == 1 {
                    (self.device_features() >> 32) as u32
                } else {
                    0
                }
            }
            VIRTIO_MMIO_QUEUE_NUM_MAX => self
                .selected_queue()
                .map(|i| self.worker.queues[i].max_size() as u32)
                .unwrap_or(0),
            VIRTIO_MMIO_QUEUE_READY => self
                .selected_queue()
                .map(|i| self.worker.queues[i].ready() as u32)
                .unwrap_or(0),
            VIRTIO_MMIO_INTERRUPT_STATUS => self.interrupt_status.load(Ordering::Acquire),
            VIRTIO_MMIO_STATUS => self.device_status.load(Ordering::Acquire),
            VIRTIO_MMIO_CONFIG_GENERATION => self.config_generation.load(Ordering::Acquire),
            _ => 0,
        };
        data.copy_from_slice(&val.to_le_bytes());
    }

    /// Read from block config space. virtio-v1.2 §5.2.4 layout, mirrored
    /// in [`VirtioBlkConfig`]:
    ///   - 0x00..0x08: capacity (u64 LE, sectors) — always
    ///   - 0x08..0x0C: size_max (u32 LE) — VIRTIO_BLK_F_SIZE_MAX
    ///   - 0x0C..0x10: seg_max (u32 LE) — VIRTIO_BLK_F_SEG_MAX
    ///   - 0x10..0x14: geometry (4 bytes) — VIRTIO_BLK_F_GEOMETRY (zero;
    ///     feature bit not advertised)
    ///   - 0x14..0x18: blk_size (u32 LE) — VIRTIO_BLK_F_BLK_SIZE
    ///
    /// Reads at offsets `>= VIRTIO_BLK_CONFIG_SIZE` return zero per
    /// virtio-v1.2 §4.2.2.2 ("reads past the populated config layout
    /// return zero") — guarded fields like topology / MQ / discard
    /// have feature bits we don't advertise, so the kernel driver's
    /// `virtio_cread_feature` skips them and never observes the
    /// zero-bytes we serve.
    pub(crate) fn read_blk_config(&self, offset: u64, data: &mut [u8]) {
        let cfg = VirtioBlkConfig {
            capacity: self.capacity_sectors,
            size_max: VIRTIO_BLK_SIZE_MAX,
            seg_max: VIRTIO_BLK_SEG_MAX,
            geometry: VirtioBlkGeometry::default(),
            blk_size: VIRTIO_BLK_SECTOR_SIZE,
        };
        // `as_slice()` returns the struct's wire-format byte
        // representation directly — `repr(C, packed)` guarantees no
        // padding and host-LE u32/u64 stores match the virtio LE wire
        // format on the supported (x86_64, aarch64) hosts. See
        // ByteValued impl SAFETY note above.
        let cfg_bytes = cfg.as_slice();
        let len = data.len();
        let start = offset as usize;
        if start >= cfg_bytes.len() {
            data.fill(0);
            return;
        }
        let end = (start + len).min(cfg_bytes.len());
        let n = end - start;
        data[..n].copy_from_slice(&cfg_bytes[start..end]);
        data[n..].fill(0);
    }

    /// Handle MMIO write at `offset` within the device's MMIO region.
    ///
    /// Same two address ranges as [`Self::mmio_read`]:
    /// - `offset >= 0x100`: device config space. Per virtio-v1.2
    ///   §4.2.2 the device owns this region — it's read-only from
    ///   the driver's perspective, populated by the device when
    ///   the driver reads. Guest writes are silently dropped (no
    ///   tracing::warn either; the kernel's virtio_mmio probe path
    ///   has been seen to issue speculative config-space writes
    ///   during feature negotiation, and warning on every one
    ///   would flood the log without identifying any real bug).
    /// - `offset < 0x100`: transport registers, dispatched per
    ///   `match`. Non-4-byte writes are silently dropped — same
    ///   "the spec mandates 4-byte access" reasoning as the read
    ///   path; the device acts on a partial register write at its
    ///   peril, so dropping is safer than wedging an MMIO FSM
    ///   with half-applied state.
    pub fn mmio_write(&mut self, offset: u64, data: &[u8]) {
        if offset >= 0x100 {
            // Config space writes are device-owned; drop silently.
            return;
        }
        if data.len() != 4 {
            return;
        }
        let val = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        match offset as u32 {
            VIRTIO_MMIO_DEVICE_FEATURES_SEL => self.device_features_sel = val,
            VIRTIO_MMIO_DRIVER_FEATURES_SEL => self.driver_features_sel = val,
            VIRTIO_MMIO_DRIVER_FEATURES => {
                if !self.features_write_allowed() {
                    return;
                }
                let page = self.driver_features_sel;
                if page == 0 {
                    self.driver_features =
                        (self.driver_features & 0xFFFF_FFFF_0000_0000) | val as u64;
                } else if page == 1 {
                    self.driver_features =
                        (self.driver_features & 0x0000_0000_FFFF_FFFF) | ((val as u64) << 32);
                }
            }
            VIRTIO_MMIO_QUEUE_SEL => self.queue_select = val,
            VIRTIO_MMIO_QUEUE_NUM if self.queue_config_allowed() => {
                if let Some(i) = self.selected_queue() {
                    self.worker.queues[i].set_size(val as u16);
                }
            }
            VIRTIO_MMIO_QUEUE_READY if self.queue_config_allowed() => {
                if let Some(i) = self.selected_queue() {
                    self.worker.queues[i].set_ready(val == 1);
                }
            }
            VIRTIO_MMIO_QUEUE_NOTIFY => {
                let idx = val as usize;
                if idx == REQ_QUEUE {
                    self.process_requests();
                }
            }
            VIRTIO_MMIO_INTERRUPT_ACK => {
                // Clear bits the guest ACKed. AcqRel: the Acquire
                // half pairs with the worker's Release fetch_or so
                // we don't lose a bit racing with worker bit-set;
                // the Release half publishes the cleared state.
                self.interrupt_status.fetch_and(!val, Ordering::AcqRel);
            }
            VIRTIO_MMIO_STATUS => {
                if val == 0 {
                    self.reset();
                } else {
                    self.set_status(val);
                }
            }
            // QUEUE_{DESC,AVAIL,USED}_{LOW,HIGH} write a 64-bit
            // guest physical address as two 32-bit halves. Per
            // virtio-v1.2 §4.2.2: writes are only valid while
            // FEATURES_OK is set and DRIVER_OK is NOT — i.e. the
            // window between feature negotiation and the driver
            // signalling "I'm done configuring." Outside that
            // window the write is silently dropped (the
            // `queue_config_allowed` guard returns false). The
            // virtio-queue crate accumulates the two halves
            // internally; the guest typically writes LOW first
            // then HIGH but the order is not load-bearing here.
            VIRTIO_MMIO_QUEUE_DESC_LOW if self.queue_config_allowed() => {
                if let Some(i) = self.selected_queue() {
                    self.worker.queues[i].set_desc_table_address(Some(val), None);
                }
            }
            VIRTIO_MMIO_QUEUE_DESC_HIGH if self.queue_config_allowed() => {
                if let Some(i) = self.selected_queue() {
                    self.worker.queues[i].set_desc_table_address(None, Some(val));
                }
            }
            VIRTIO_MMIO_QUEUE_AVAIL_LOW if self.queue_config_allowed() => {
                if let Some(i) = self.selected_queue() {
                    self.worker.queues[i].set_avail_ring_address(Some(val), None);
                }
            }
            VIRTIO_MMIO_QUEUE_AVAIL_HIGH if self.queue_config_allowed() => {
                if let Some(i) = self.selected_queue() {
                    self.worker.queues[i].set_avail_ring_address(None, Some(val));
                }
            }
            VIRTIO_MMIO_QUEUE_USED_LOW if self.queue_config_allowed() => {
                if let Some(i) = self.selected_queue() {
                    self.worker.queues[i].set_used_ring_address(Some(val), None);
                }
            }
            VIRTIO_MMIO_QUEUE_USED_HIGH if self.queue_config_allowed() => {
                if let Some(i) = self.selected_queue() {
                    self.worker.queues[i].set_used_ring_address(None, Some(val));
                }
            }
            _ => {}
        }
    }

    /// Validate and apply a status transition per virtio-v1.2 §3.1.1.
    ///
    /// FEATURES_OK additionally enforces two constraints:
    ///
    /// 1. VIRTIO_F_VERSION_1 must be in `driver_features`
    ///    (virtio-v1.2 §6.1: "A driver MUST accept VIRTIO_F_VERSION_1").
    ///    Modern devices require this bit; a driver that fails to ack
    ///    it (legacy/transitional driver against this modern-only
    ///    device) cannot operate.
    /// 2. `driver_features` must be a SUBSET of `device_features()`
    ///    (virtio-v1.2 §3.1.1 step 5: "the driver MUST NOT set any
    ///    feature bit that the device did not offer"). A driver that
    ///    acks an unadvertised bit has either misread the device
    ///    feature page or is buggy/hostile; either way the device
    ///    cannot honor the implied contract because none of the
    ///    backend code paths for the unadvertised feature exist.
    ///
    /// The kernel's `virtio_features_ok` (drivers/virtio/virtio.c)
    /// writes FEATURES_OK then re-reads STATUS to confirm the bit
    /// stuck — rejecting here clears the path: the FSM leaves
    /// FEATURES_OK unset, the kernel's read-back fails, and the
    /// driver bind surfaces -ENODEV without descending into queue
    /// config.
    ///
    /// Every rejection path emits a `tracing::warn!` with the
    /// `device_status` / requested `val` / `new_bits` payload so an
    /// operator debugging a failed-bind can see which step the FSM
    /// rejected — clearing-bit attempts, ordering violations, multi-
    /// bit transitions, and unknown bits all surface explicitly
    /// rather than as a silent return.
    ///
    /// Idempotent re-writes (the requested `val` equals the
    /// current `device_status`) are a NO-OP, not a rejection: the
    /// monotone-bit gate accepts them (no bits cleared) and the
    /// new_bits-zero short-circuit returns without logging.
    /// Standard drivers go through `virtio_add_status`
    /// (drivers/virtio/virtio.c:196-200), which writes
    /// `STATUS = old | NEW_BIT`; `virtio_features_ok`
    /// (drivers/virtio/virtio.c:230) re-reads via `get_status`
    /// to confirm the bit stuck. Warning on idempotent re-writes
    /// would pollute operator logs without surfacing real bugs.
    pub(crate) fn set_status(&mut self, val: u32) {
        // Snapshot the current FSM state. `set_status` runs on the
        // vCPU thread that received the MMIO write; the FSM walk
        // through ACK → DRIVER → FEATURES_OK → DRIVER_OK happens
        // sequentially within and across calls on that thread. The
        // production worker thread's only write site to
        // device_status is the `fetch_or(NEEDS_RESET, SeqCst)` on
        // the queue-poison path. Whether that write can race the
        // vCPU's FSM-advance store depends on the worker's
        // lifecycle:
        //
        // - **Pre-DRIVER_OK** (initial spawn deferred to the first
        //   `STATUS = DRIVER_OK` per `consume_pending_respawn`):
        //   no worker thread is alive yet, so no concurrent
        //   `fetch_or` can land. Single-writer device_status.
        // - **Between DRIVER_OK and reset**: the worker is alive
        //   and may queue-poison at any point; a vCPU-side
        //   set_status arriving in this window can race its
        //   `fetch_or(NEEDS_RESET)`.
        // - **Between reset and the next DRIVER_OK**: the worker
        //   has been joined (`reset_engine_spawned` →
        //   `stop_worker_and_reclaim_state`); single-writer.
        //
        // The middle bucket is the race that motivates the CAS
        // below. A naive `store(val, Release)` after the snapshot
        // would clobber a NEEDS_RESET bit the worker had just
        // fetch_or'd in — silently lying to the guest by reporting
        // a healthy FSM after the device had already declared
        // itself broken. The CAS below is **load-bearing for race
        // safety**, not defense-in-depth: the worker's
        // `fetch_or(NEEDS_RESET, SeqCst)` can set bits between this
        // load and the CAS attempt, and the CAS is the mechanism
        // that detects the contention. Replacing the store with a
        // compare_exchange against the snapshot detects the race:
        // if the worker advanced device_status concurrently, the
        // CAS fails and we re-snapshot + re-validate. Either the
        // re-validated transition still passes (worker added bits
        // we are about to set anyway — proceed) or it fails
        // (worker added NEEDS_RESET, which is not a legal
        // FSM-advance bit; the new snapshot rejects with the
        // monotone-bit gate or the `valid` match). The Acquire
        // load and the CAS's failure-side Acquire ordering
        // synchronise-with the worker's SeqCst fetch_or at
        // `drain_bracket_impl`'s queue-poison arm — Acquire
        // observation pairs with the SeqCst write side because
        // SeqCst is at least Release on the writer.
        //
        // Snapshot loaded outside the loop; on a CAS failure the
        // `Err(observed)` branch updates `current_status` directly
        // without re-issuing a `load` — saving one redundant
        // atomic read per retry while preserving the same
        // happens-before chain.
        let mut current_status = self.device_status.load(Ordering::Acquire);
        // CAS retry loop. Each iteration re-validates the proposed
        // transition against the freshly-snapshotted `current_status`
        // and attempts a `compare_exchange` to commit. On contention
        // (the worker fetch_or'd NEEDS_RESET between snapshot and
        // commit), the CAS returns `Err(observed)` and we restart
        // the loop with the observed value as the new snapshot.
        // Termination is bounded at AT MOST ONE worker-induced
        // retry: by the worker invariant (see the worker's
        // queue-poison fetch_or site), the worker may only
        // fetch_or `VIRTIO_CONFIG_S_NEEDS_RESET` and the operation
        // is idempotent after the first call. So the worker can
        // transition `device_status` from one observable state
        // (`current_status`) to one other state
        // (`current_status | NEEDS_RESET`) and never to a third
        // value while this set_status is running. After that
        // single retry the snapshot is stable: either the second
        // CAS succeeds, or the monotone-bit gate fires because
        // the new snapshot has NEEDS_RESET and `val` does not
        // include it.
        loop {
            if val & current_status != current_status {
                // CORRECT behavior — do NOT "fix" this gate to admit
                // the advance. After the worker's queue-poison path
                // fetch_or'd `VIRTIO_CONFIG_S_NEEDS_RESET` into
                // `current_status`, every subsequent guest STATUS
                // write whose `val` does NOT include the NEEDS_RESET
                // bit (drivers never set it — it is device-emitted
                // per virtio-v1.2 §2.1.1 bit 0x40) trips this check
                // and is rejected. That is the spec-mandated
                // behaviour: the device is dead until a STATUS=0
                // reset, and the kernel's `virtio_features_ok`-style
                // post-write `get_status` re-read sees the FSM bit
                // never stuck (because we rejected here) and
                // surfaces -ENODEV to the bind path. A future
                // refactor that loosens this gate to "allow the
                // advance and clear NEEDS_RESET silently" would
                // restore the silent-corruption hazard the CAS
                // exists to prevent.
                //
                // Distinguish the two failure modes that both surface
                // here as `val & current_status != current_status`:
                //
                // 1. NEEDS_RESET bit (0x40) is set in `current_status`
                //    but not in `val`. This happens when the worker's
                //    queue-poison path fetch_or'd NEEDS_RESET — either
                //    before this set_status call or during a CAS
                //    retry. The driver did NOT try to regress; the
                //    device set NEEDS_RESET on its own. Cite the
                //    queue-poison cause and the STATUS=0 recovery
                //    path so an operator reading the log knows the
                //    fix is a full reset, not a driver bug.
                //
                // 2. Otherwise: the driver attempted to clear a
                //    previously-set bit (per virtio-v1.2 §3.1.1
                //    status bits are monotone within a driver
                //    session) — a regress that surfaces a buggy
                //    driver clearing FEATURES_OK while keeping
                //    ACKNOWLEDGE.
                if current_status & VIRTIO_CONFIG_S_NEEDS_RESET != 0 {
                    tracing::warn!(
                        device_status = current_status,
                        requested = val,
                        "virtio-blk set_status rejected — device in \
                         NEEDS_RESET state from prior queue poison; \
                         guest must write STATUS=0 to reset before any \
                         further FSM advance can succeed"
                    );
                } else {
                    tracing::warn!(
                        device_status = current_status,
                        requested = val,
                        "virtio-blk set_status rejected — attempted to clear \
                         a previously-set status bit without a full reset \
                         (virtio-v1.2 §3.1.1: status bits are monotone within \
                         a driver session)"
                    );
                }
                return;
            }
            let new_bits = val & !current_status;
            // Idempotent re-write of the current device_status: the
            // monotone-bit gate above passed (val is a superset) AND
            // the requested value adds no new bits. This is a
            // legitimate driver pattern — the kernel's
            // `virtio_add_status` (drivers/virtio/virtio.c:196-200)
            // writes `STATUS = old | NEW_BIT` and a subsequent
            // `virtio_features_ok` (drivers/virtio/virtio.c:230)
            // `get_status` read may race a duplicate set, plus an
            // MMIO probe path may issue a duplicate STATUS write.
            // Treat as a no-op rather than a rejection so the
            // rejection-warn path stays a true signal.
            if new_bits == 0 {
                return;
            }
            let valid = match new_bits {
                VIRTIO_CONFIG_S_ACKNOWLEDGE => current_status == 0,
                VIRTIO_CONFIG_S_DRIVER => current_status == S_ACK,
                VIRTIO_CONFIG_S_FEATURES_OK => {
                    current_status == S_DRV
                        && self.driver_features & (1u64 << VIRTIO_F_VERSION_1) != 0
                        && self.driver_features & !self.device_features() == 0
                }
                VIRTIO_CONFIG_S_DRIVER_OK => current_status == S_FEAT,
                _ => false,
            };
            if valid {
                // compare_exchange against the snapshot. On success
                // the store lands with Release ordering (mirroring
                // the pre-CAS `store(val, Release)` semantics for
                // any vCPU reader doing `load(Acquire)`). On failure
                // the worker raced an additional bit (NEEDS_RESET on
                // queue poison) and we restart the outer loop with
                // the observed value. Acquire on the failure side
                // synchronizes-with the worker's SeqCst fetch_or
                // (which is at least Release on the writer side) so
                // the next iteration's re-validation sees the
                // worker's NEEDS_RESET bit.
                match self.device_status.compare_exchange(
                    current_status,
                    val,
                    Ordering::Release,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {}
                    Err(observed) => {
                        current_status = observed;
                        continue;
                    }
                }
                // Once FEATURES_OK is committed, feature negotiation
                // is closed (virtio-v1.2 §3.1.1) — the negotiated set
                // lives in `driver_features` and the device may rely
                // on it. If VIRTIO_RING_F_EVENT_IDX was negotiated,
                // enable event-idx tracking on the request queue so
                // `Queue::needs_notification` consults the guest's
                // `used_event` threshold instead of always returning
                // true. `QueueT::event_idx_enabled` is documented to
                // return the correct value only after FEATURES_OK,
                // so this is the earliest legal moment to flip it
                // on.
                if new_bits == VIRTIO_CONFIG_S_FEATURES_OK
                    && self.driver_features & (1u64 << VIRTIO_RING_F_EVENT_IDX) != 0
                {
                    self.worker.queues[REQ_QUEUE].set_event_idx(true);
                }
                // DRIVER_OK transition: consume any deferred respawn
                // state stashed by `reset_engine_spawned`. By the
                // time the guest reaches DRIVER_OK it has walked ACK
                // → DRIVER → FEATURES_OK, and the
                // queue_config_allowed gate (S_FEAT && !DRIVER_OK)
                // admitted any DESC/AVAIL/USED address writes plus
                // QUEUE_NUM / QUEUE_READY between FEATURES_OK and
                // now. The kernel virtio-mmio driver's `vm_setup_vq`
                // (drivers/virtio/virtio_mmio.c:346-444) publishes
                // the queue addresses and writes `QUEUE_READY=1` in
                // that window before the DRIVER_OK MMIO write, so
                // the worker spawned here will find a
                // fully-configured queue on its first drain attempt.
                // Production cfg only — the inline-engine test build
                // has no respawn machinery. See the
                // `SpawnedEngine::respawn_pending` doc for the full
                // rationale and race-free invariant.
                #[cfg(not(test))]
                if new_bits == VIRTIO_CONFIG_S_DRIVER_OK {
                    self.consume_pending_respawn();
                }
                return;
            }
            // Rejection paths. The FEATURES_OK case has the richest
            // diagnostic because it's the only transition with
            // sub-conditions beyond simple ordering (subset rule +
            // VERSION_1 mandate); other rejections cite the FSM
            // ordering violation directly.
            if new_bits == VIRTIO_CONFIG_S_FEATURES_OK && current_status == S_DRV {
                // FEATURES_OK with the right ordering but the driver
                // failed the feature-set rules. Report VERSION_1
                // missing first (most common failure mode for a
                // legacy/transitional driver); fall through to the
                // unadvertised-bit case if VERSION_1 is fine.
                if self.driver_features & (1u64 << VIRTIO_F_VERSION_1) == 0 {
                    tracing::warn!(
                        driver_features = ?self.driver_features,
                        "FEATURES_OK rejected — VIRTIO_F_VERSION_1 not negotiated; \
                         legacy/transitional driver against modern-only device",
                    );
                } else {
                    let unadvertised = self.driver_features & !self.device_features();
                    if unadvertised != 0 {
                        tracing::warn!(
                            driver_features = ?self.driver_features,
                            device_features = ?self.device_features(),
                            unadvertised = ?unadvertised,
                            "FEATURES_OK rejected — driver acked unadvertised \
                             feature bits; subset rule (virtio-v1.2 §3.1.1) \
                             violated",
                        );
                    }
                }
            } else if current_status & VIRTIO_CONFIG_S_NEEDS_RESET != 0 {
                // NEEDS_RESET-specific diagnostic — defense in depth
                // alongside the same gate at the monotone-bit branch
                // above. The monotone-bit branch fires for the
                // typical race (val omits NEEDS_RESET, current_status
                // has it), but a future caller that constructed
                // `val` to include NEEDS_RESET (e.g. an internal
                // helper that shouldn't exist but might be added)
                // would slip past the monotone-bit gate and reach
                // this rejection arm. Cite the queue-poison cause
                // here too so the diagnostic taxonomy stays
                // consistent.
                tracing::warn!(
                    device_status = current_status,
                    requested = val,
                    new_bits = new_bits,
                    "virtio-blk set_status rejected — device in \
                     NEEDS_RESET state from prior queue poison; \
                     guest must write STATUS=0 to reset before any \
                     further FSM advance can succeed",
                );
            } else {
                // Generic ordering or unknown-bit rejection: ACK
                // without device_status==0, DRIVER without ACK,
                // FEATURES_OK from the wrong predecessor, DRIVER_OK
                // without FEATURES_OK, or any new_bits that aren't a
                // single virtio-v1.2 status bit (multi-bit
                // transitions, reserved bits set). Citing
                // device_status + new_bits lets an operator identify
                // the ordering violation without rederiving the FSM.
                tracing::warn!(
                    device_status = current_status,
                    requested = val,
                    new_bits = new_bits,
                    "virtio-blk set_status rejected — illegal FSM transition \
                     (virtio-v1.2 §3.1.1 ordering: ACK → DRIVER → FEATURES_OK \
                     → DRIVER_OK, one bit at a time)",
                );
            }
            return;
        }
    }

    /// Reset the device to its initial state per virtio-v1.2 §2.1.
    ///
    /// Two race-free paths, gated by `cfg`:
    ///
    /// - **Production (`cfg(not(test))`):** the worker thread owns
    ///   the `BlkWorkerState` and may be mid-drain when the vCPU
    ///   MMIO write of `STATUS = 0` lands here. Issuing
    ///   `q.reset()` while the worker holds the QueueSync mutex
    ///   (during `pop_descriptor_chain` / `add_used`) would race —
    ///   even worse, the worker may be in `pread`/`pwrite` against
    ///   a soon-to-be-stale guest memory mapping or compute an
    ///   `add_used` against the post-reset queue with cleared
    ///   `next_avail`. We close that window by stopping the worker
    ///   first, joining it (so no concurrent reader exists), then
    ///   running `q.reset()` and re-spawning a fresh worker
    ///   against the post-reset queue.
    ///
    ///   We converge with cloud-hypervisor's pattern of stopping
    ///   the worker on reset and deferring the respawn to the
    ///   guest's next `DRIVER_OK` transition. We still diverge
    ///   from firecracker (whose virtio-block device does not
    ///   implement reset at all — `Reset` returns `None` from the
    ///   device shim and the transport marks the device FAILED).
    ///   The reclaimed `BlkWorkerState` is parked in
    ///   `SpawnedEngine::respawn_pending` until `set_status`
    ///   observes the `STATUS = DRIVER_OK` MMIO write and calls
    ///   `consume_pending_respawn`, which builds fresh kick/stop
    ///   eventfds and a fresh worker thread against the
    ///   re-bound queue. Between reset and DRIVER_OK no worker
    ///   thread is alive, so kicks landing on the stale
    ///   (now-detached) `kick_fd` accumulate harmlessly until the
    ///   re-bind completes — the fresh worker will iter() over
    ///   chains the guest enqueued, since chain state lives in
    ///   guest memory, not the eventfd counter. Deferring saves
    ///   a thread sitting in `epoll_wait` for the duration of the
    ///   guest's rebind sequence (queue addresses zeroed,
    ///   `QUEUE_READY` false) — a window driver implementations
    ///   can stretch into milliseconds.
    ///
    /// - **Tests (`cfg(test)`):** Inline mode runs `drain_inline`
    ///   synchronously on the caller thread, so by the time
    ///   `reset()` is invoked there is no concurrent reader on
    ///   `worker.queues[…]`. The test-mode reset
    ///   (`reset_engine_inline`) resets the queue in place,
    ///   rebuilds the throttle buckets from the captured
    ///   `self.throttle` (so an adversarial test cannot drain the
    ///   bucket and reset to bypass), and clears the scratch Vecs
    ///   (capacity retained).
    ///
    /// # Counter persistence
    ///
    /// `VirtioBlkCounters` (`reads_completed`, `bytes_read`,
    /// `throttled_count`, `io_errors`, etc.) persist across reset.
    /// They are cumulative for the device's lifetime — a guest
    /// re-bind preserves the counter Arc so an operator monitoring
    /// failure-dump counters observes a monotonically
    /// non-decreasing series spanning the device's full IO
    /// history.
    ///
    /// # vCPU thread blocking
    ///
    /// The production path's `handle.join()` runs on the vCPU
    /// thread that received the MMIO write. If the worker is
    /// mid-`pread`/`pwrite` when STOP_TOKEN is signaled, the
    /// syscall completes before the worker reaches the next
    /// `epoll_wait` and observes the stop signal. The vCPU thread
    /// blocks for the duration. This is bounded by the same
    /// backing-speed assumption documented at the module level
    /// (tmpfs / warm page cache). A `reset()` issued during a slow
    /// IO can stretch beyond the freeze coordinator's rendezvous
    /// timeout, so `reset()` caps the worker join at
    /// [`RESET_JOIN_TIMEOUT`] (1 s) via [`join_worker_with_timeout`]
    /// (see [`Self::stop_worker_and_reclaim_state`]); on timeout
    /// the worker is leaked into the permanent-workerless state
    /// rather than hanging the rendezvous indefinitely.
    pub(crate) fn reset(&mut self) {
        // Phase 1 — clear MMIO-side scalar device state. These
        // fields live on `VirtioBlk` only (not shared with the
        // worker thread), so they're safe to mutate before the
        // queue stop+respawn. `interrupt_status` is intentionally
        // NOT cleared here because the worker thread (production)
        // may still race-fire `irq_evt.write(1)` and bit-set
        // INT_VRING; we clear it only after the worker is joined.
        // `device_status` is also deferred to Phase 3 for the same
        // reason: the worker's queue-poison path can fetch_or
        // NEEDS_RESET concurrently with this reset(), and clearing
        // it before the worker is joined would let a phantom
        // NEEDS_RESET bit re-set itself between Phase 1 and Phase 2.
        self.queue_select = 0;
        self.device_features_sel = 0;
        self.driver_features_sel = 0;
        self.driver_features = 0;
        // Bump config_generation on every reset so a re-binding
        // driver observes a different value and re-reads config
        // space (per virtio-v1.2 §4.2.2.1: drivers MUST re-read
        // on changed generation). For v0 the capacity is fixed
        // for the device's lifetime — set once in `new()` and
        // never mutated — so the bump is purely defense-in-depth:
        // a future patch that resizes the disk between resets is
        // the case it guards. wrapping_add is implicit in
        // fetch_add's modular arithmetic.
        //
        // Release ordering: today the only writer is this
        // (vCPU-thread `reset()`), and the only reader is the
        // vCPU-thread `mmio_read(CONFIG_GENERATION)`, so
        // single-threaded access makes Release semantically
        // unnecessary. Release is defense-in-depth against future
        // cross-thread config writers (e.g. a follow-up that
        // resizes the disk from a worker thread or a host
        // monitor); pairs with the Acquire load in `mmio_read`.
        self.config_generation.fetch_add(1, Ordering::Release);
        // Re-arm the "queue notify before set_mem" warning so a
        // post-reset wiring bug surfaces (virtio-v1.2 §3.1.1: a
        // reset puts the device in a state where the driver must
        // rebind and re-publish queue addresses; if a kick reaches
        // us before the rebind completes, that's worth a fresh
        // log line, not a quiet drop based on a latch from a
        // previous lifetime).
        self.mem_unset_warned.store(false, Ordering::Relaxed);

        // Phase 2 — engine-specific quiesce and queue reset
        // (production); respawn deferred to DRIVER_OK via
        // `consume_pending_respawn`. The `cfg(test)` Inline path
        // performs an in-place state reset on the caller thread.
        // Both paths leave the engine in a state where no worker
        // is currently mutating `interrupt_status` / `irq_evt`.
        #[cfg(test)]
        self.reset_engine_inline();
        #[cfg(not(test))]
        self.reset_engine_spawned();

        // Phase 3 — quiesce the IRQ path. With the worker stopped
        // (production) or never-active (test), no new
        // `irq_evt.write(1)`, `interrupt_status` bit-set, or
        // `device_status` fetch_or(NEEDS_RESET) can race us. Drain
        // the eventfd's pending counter so a stale worker write
        // (delivered between the last add_used and the stop signal)
        // doesn't fire a phantom IRQ at the post-reset guest; zero
        // `interrupt_status` so the guest's MMIO read of
        // INTERRUPT_STATUS observes a clean slate; zero
        // `device_status` so the guest re-reads STATUS=0 and walks
        // the FSM from scratch (per virtio-v1.2 §3.1.1: a reset
        // returns the device to its initial state including all FSM
        // bits — the NEEDS_RESET bit set by the worker's
        // queue-poison path is part of that state and clears here).
        // Both stores are Release-ordered to pair with their
        // respective `mmio_read` Acquire loads.
        //
        // Race window: a worker that completed `add_used` +
        // `irq_evt.write(1)` after the vCPU latched STATUS=0 but
        // before the stop signal landed would otherwise leave a
        // pending eventfd counter; KVM's irqfd would deliver the
        // GSI to the guest after reset, with the used ring now
        // empty (post-`q.reset()`), causing the guest's
        // `virtblk_done` to spin chasing a non-existent
        // completion. Draining here closes that window. The
        // device_status store deferral closes the parallel window
        // for the queue-poison path: a worker that ran
        // `fetch_or(NEEDS_RESET)` after Phase 1 but before being
        // joined would otherwise leave the bit set after reset,
        // and the guest's FSM walk from STATUS=0 → ACK → DRIVER →
        // FEATURES_OK → DRIVER_OK would silently transition
        // through a "device still says NEEDS_RESET" state visible
        // through `mmio_read(STATUS)`.
        let _ = self.irq_evt.read();
        // Drain the pause eventfd counter so any `pause()` writes
        // that landed during this reset cycle (e.g. a freeze
        // coordinator that fired between `reset_engine_spawned`'s
        // join and this Phase 3) do not carry a stale tick across
        // the rebind. Without this drain, the next
        // `worker_thread_main` (spawned at the next DRIVER_OK)
        // would observe PAUSE_TOKEN on its first `epoll_wait`,
        // park immediately, and starve the guest's first kicks
        // until the coordinator's eventual `resume()`. The read
        // is best-effort — a `WouldBlock` (counter already 0)
        // is normal, any other error means the eventfd is
        // already torn down which the next worker spawn will
        // re-create.
        let _ = self.pause_evt.read();
        self.interrupt_status.store(0, Ordering::Release);
        self.device_status.store(0, Ordering::Release);
    }

    /// Test-mode engine reset: queue mutation and bucket rebuild
    /// happen on the caller thread (no worker exists). Scratches
    /// keep their capacity.
    #[cfg(test)]
    pub(crate) fn reset_engine_inline(&mut self) {
        for q in &mut self.worker.queues {
            q.reset();
        }
        let WorkerEngine::Inline(engine) = &mut self.worker.engine;
        let (ops_bucket, bytes_bucket) = buckets_from_throttle(self.throttle);
        engine.state.ops_bucket = ops_bucket;
        engine.state.bytes_bucket = bytes_bucket;
        engine.state.all_descs_scratch.clear();
        engine.state.io_buf_scratch.clear();
        // Reset throttle-stall gauge state. q.reset() above
        // cleared the queue cursor, so any chain that was
        // rolled-back-pending is now lost from the device's
        // perspective — the guest's re-bind will re-issue
        // chains from a fresh avail.idx=0. The currently_stalled
        // flag must clear and the gauge must decrement to match;
        // otherwise the gauge leaks one increment per reset that
        // happens during a stall window. The gauge is "currently
        // pending throttle-stalled requests"; post-reset there
        // are none until the guest re-issues IO.
        if engine.state.currently_stalled {
            engine.state.currently_stalled = false;
            engine.state.counters.record_throttle_pending_dec();
        }
        // Clear hostile-guest poison: the guest issued a virtio
        // reset, which is the only documented escape from the
        // queue-poisoned state. The `invalid_avail_idx_count`
        // counter is intentionally NOT cleared here — operators
        // need cumulative-event visibility across resets to detect
        // repeated hostile-guest behavior.
        engine.state.queue_poisoned = false;
    }

    /// Production engine reset: stop the worker, join, q.reset(),
    /// stash the reclaimed state in `respawn_pending` for
    /// `set_status` to consume on the next DRIVER_OK transition.
    /// The reclaimed state contributes its long-lived resources
    /// (backing File, scratch capacities, capacity_bytes,
    /// read_only, counters Arc) — only the throttle buckets are
    /// rebuilt by `respawn_worker` once DRIVER_OK fires.
    ///
    /// Why defer the respawn: between `reset()` and DRIVER_OK
    /// the guest is rebinding (queue addresses zeroed,
    /// QUEUE_READY false). A worker spawned eagerly here would
    /// sit in `epoll_wait` doing nothing for the duration of the
    /// rebind. See the `SpawnedEngine::respawn_pending` doc for
    /// the full rationale and race-free invariant.
    #[cfg(not(test))]
    pub(crate) fn reset_engine_spawned(&mut self) {
        // Detect a back-to-back reset (the guest issued STATUS=0
        // twice without an intervening DRIVER_OK). The first
        // reset stashed state in respawn_pending and joined the
        // worker; the second reset has no live worker to stop
        // and must NOT overwrite the pending state (the second
        // `stop_worker_and_reclaim_state` would return None and
        // clobber the first reset's reclaimed state — the
        // backing File and counter Arc would be lost). Skip the
        // worker-quiesce step in that case; the queue reset
        // below still runs because the guest expects a fresh
        // queue cursor.
        let already_pending = {
            let WorkerEngine::Spawned(eng) = &self.worker.engine;
            eng.respawn_pending.is_some()
        };
        if !already_pending {
            // If a freeze coordinator paused the worker via
            // `pause()` and a STATUS=0 reset arrives before
            // `resume()`, the worker is parked in its
            // `park_timeout(10ms)` Acquire-load loop and does NOT
            // observe `stop_fd` — `epoll_wait` is unreachable from
            // the park. Clear `paused` (Release) and unpark BEFORE
            // writing `stop_fd` so the worker wakes within 10 ms
            // (or immediately on the unpark hint), exits the park
            // loop, returns to `epoll_wait`, and observes
            // STOP_TOKEN. Without this, the
            // `join_worker_with_timeout(RESET_JOIN_TIMEOUT, 1s)`
            // would always fire the TimedOut diagnostic when reset
            // races a paused worker. Cloud-hypervisor's epoll-helper
            // teardown follows the same unpause-before-stop ordering
            // (clear the paused flag and wake before signalling the
            // kill eventfd) so a parked worker observes the kill on
            // its first epoll-wake rather than after a 10 ms
            // park-timeout tick.
            self.resume();
            let reclaimed = self.stop_worker_and_reclaim_state();
            // Re-arm the construction-time "paused" sentinel so a
            // freeze that fires between this stop and the next
            // DRIVER_OK respawn passes the rendezvous vacuously
            // (mirrors the `with_options` initialisation). Without
            // this, the prior `resume()` left `paused=false`, and
            // the rendezvous would block until the 30 s timeout
            // waiting for a worker that does not yet exist — the
            // freeze coordinator's failure-dump path would lose
            // the dump for any STALL_DETECTED that lands in the
            // rebind window.
            self.paused.store(true, Ordering::Release);
            // Stash the reclaimed state for the deferred respawn.
            // `set_status` consumes it on the next valid DRIVER_OK
            // transition. `None` (worker had panicked / timed out /
            // helper failed) means no state to respawn from — the
            // device is permanently workerless from this point. The
            // diagnostic was already logged by
            // `stop_worker_and_reclaim_state`; the WorkerEngine
            // remains in `Spawned` form with `handle: None` and
            // `respawn_pending: None`, so future kicks land on the
            // stale `kick_fd` and accumulate harmlessly until the
            // device is destroyed. Only constructing a fresh
            // `VirtioBlk` recovers IO service.
            let WorkerEngine::Spawned(eng) = &mut self.worker.engine;
            eng.respawn_pending = reclaimed;
        }
        // q.reset() runs uncontested: the worker thread is joined
        // (or was never alive in the back-to-back-reset case) and
        // no new one has been spawned yet, so the QueueSync mutex
        // has no other holder.
        for q in &mut self.worker.queues {
            q.reset();
        }
    }

    /// Production: send STOP_TOKEN to the worker, join the
    /// thread with a [`RESET_JOIN_TIMEOUT`] budget, return the
    /// worker state. Returns `None` if the worker had already been
    /// joined (Option already taken — a second `reset()` after a
    /// torn-down engine, or a concurrent Drop racing the MMIO
    /// writer; both are operator bugs but must not panic the vCPU
    /// thread), if the worker panicked, OR if the join timed out
    /// or the helper machinery itself failed.
    ///
    /// # vCPU thread protection
    ///
    /// The unbounded `handle.join()` this function previously used
    /// would block the vCPU thread that received the `STATUS = 0`
    /// MMIO write through any wedged backing-IO path the worker
    /// hit (NFS stall, slow page cache, hung block device). The
    /// freeze coordinator's SIGRTMIN-based rendezvous (30 s wall
    /// budget at the coordinator level) targets that same vCPU
    /// thread; an unbounded reset block would either time out the
    /// rendezvous empty or arrive minutes late. Routing through
    /// [`join_worker_with_timeout`] caps the vCPU's pre-rendezvous
    /// overhead at [`RESET_JOIN_TIMEOUT`] (1 s) — the same
    /// invariant `Drop` enforces via [`DROP_JOIN_TIMEOUT`].
    ///
    /// # Outcomes
    ///
    /// - [`JoinWithTimeoutOutcome::Joined`] → return `Some(state)`;
    ///   reset proceeds to `q.reset()` + respawn.
    /// - [`JoinWithTimeoutOutcome::Panicked`] → log structured
    ///   error (matching Drop's diagnostic), return `None`. Device
    ///   enters permanent-workerless state.
    /// - [`JoinWithTimeoutOutcome::TimedOut`] → log structured
    ///   warn (worker is wedged in a blocking syscall that does
    ///   not check stop_fd), return `None`. Helper retains the
    ///   `JoinHandle` and the underlying `BlkWorkerState`; the
    ///   wedged worker keeps running until its blocking syscall
    ///   returns. Device enters permanent-workerless state — the
    ///   resource-retention trade documented at
    ///   [`join_worker_with_timeout`] applies here too.
    /// - [`JoinWithTimeoutOutcome::HelperSpawnFailed`] /
    ///   [`JoinWithTimeoutOutcome::HelperDisconnected`] → log
    ///   structured error, return `None`. Outer worker is
    ///   detached.
    ///
    /// All four non-Joined outcomes funnel through the
    /// "permanent device death" path documented at
    /// [`VirtioBlk::reset_engine_spawned`] — `reclaimed = None`
    /// skips the respawn and the device serves no further IO
    /// until reconstruction.
    #[cfg(not(test))]
    pub(crate) fn stop_worker_and_reclaim_state(&mut self) -> Option<BlkWorkerState> {
        let WorkerEngine::Spawned(eng) = &mut self.worker.engine;
        // Capture device-identifier fields before the
        // `eng.handle.take()` consumes the Option, so the
        // diagnostic warns can name the wedged device without
        // re-borrowing `self`.
        let stop_fd = eng.stop_fd.as_raw_fd();
        let capacity_sectors = self.capacity_sectors;
        let instance_id = self.instance_id;
        // Signal the worker to exit via the stop_fd helper, which
        // retries on EAGAIN (eventfd counter saturation) up to
        // STOP_FD_WRITE_MAX_RETRIES times before giving up. On
        // exhaustion the worker may not observe the stop signal;
        // the subsequent join's RESET_JOIN_TIMEOUT budget bounds
        // the wait to 1 s and surfaces the stall through the
        // TimedOut diagnostic below.
        signal_worker_stop(&eng.stop_fd, stop_fd, instance_id, capacity_sectors);
        // Re-borrow eng after the immutable reads above — needed
        // because `take()` mutates the Option.
        let WorkerEngine::Spawned(eng) = &mut self.worker.engine;
        let handle = eng.handle.take()?;
        match join_worker_with_timeout(handle, RESET_JOIN_TIMEOUT) {
            JoinWithTimeoutOutcome::Joined(state) => Some(state),
            JoinWithTimeoutOutcome::Panicked(payload) => {
                tracing::error!(
                    panic = panic_payload_str(&*payload),
                    stop_fd,
                    capacity_sectors,
                    instance_id,
                    "virtio-blk worker thread panicked during reset; \
                     no state to reclaim — device will not service IO \
                     until a fresh VirtioBlk is constructed"
                );
                None
            }
            JoinWithTimeoutOutcome::TimedOut => {
                tracing::warn!(
                    timeout_s = RESET_JOIN_TIMEOUT.as_secs_f32(),
                    stop_fd,
                    capacity_sectors,
                    instance_id,
                    "virtio-blk worker did not exit within \
                     RESET_JOIN_TIMEOUT of stop_fd during reset; \
                     leaking the worker thread to avoid blocking the \
                     vCPU thread (which the freeze coordinator may \
                     target with SIGRTMIN). Device enters the \
                     permanent-workerless state — guests will hang \
                     on every request until \
                     kernel.hung_task_timeout_secs (default 120 s) \
                     fires, and only constructing a fresh VirtioBlk \
                     recovers IO service. \
                     hint: identify the wedged device by stop_fd / \
                     instance_id / capacity_sectors above. \
                     hint: check `dmesg` for the backing fd's \
                     storage path stalling on I/O, or kill -USR1 \
                     the host process to dump worker thread \
                     backtraces."
                );
                None
            }
            JoinWithTimeoutOutcome::HelperSpawnFailed => {
                tracing::error!(
                    stop_fd,
                    capacity_sectors,
                    instance_id,
                    "virtio-blk reset helper thread spawn failed; \
                     detaching worker without join — device enters \
                     the permanent-workerless state"
                );
                None
            }
            JoinWithTimeoutOutcome::HelperDisconnected => {
                tracing::error!(
                    stop_fd,
                    capacity_sectors,
                    instance_id,
                    "virtio-blk reset helper thread terminated \
                     without forwarding the worker join result; \
                     device enters the permanent-workerless state"
                );
                None
            }
        }
    }

    /// Drain any state stashed in `SpawnedEngine::respawn_pending`
    /// by a prior `reset_engine_spawned` call and pass it to
    /// `respawn_worker`. Called by `set_status` on the DRIVER_OK
    /// transition — the only legal point at which the guest has
    /// finished publishing fresh queue addresses and the worker
    /// has real work to service.
    ///
    /// `respawn_pending` is `take()`-ed unconditionally even when
    /// `respawn_worker` itself fails to construct fresh fds or
    /// spawn the thread. This avoids leaving stale state holding
    /// scratch buffers and the backing-file `File` handle alive
    /// past the device's effective lifetime — the failure
    /// diagnostics from `respawn_worker` already document the
    /// permanent-workerless outcome. A second DRIVER_OK with no
    /// pending state (e.g. the guest re-binds without an
    /// intervening reset) is a no-op.
    #[cfg(not(test))]
    pub(crate) fn consume_pending_respawn(&mut self) {
        let pending = {
            let WorkerEngine::Spawned(eng) = &mut self.worker.engine;
            eng.respawn_pending.take()
        };
        if let Some(state) = pending {
            self.respawn_worker(state);
        }
    }

    /// Production: build a fresh `SpawnedEngine` (new kick_fd,
    /// stop_fd, worker thread) seeded with the reclaimed
    /// `BlkWorkerState`, and replace `self.worker.engine`. The
    /// throttle buckets in `state` are reconstructed from the
    /// captured `self.throttle` so an adversarial guest cannot
    /// drain the bucket and issue a reset to bypass the rate
    /// limit (spec-compliant: virtio-v1.2 §2.1 requires reset to
    /// return the device to its initial state, and bucket fill is
    /// part of that state).
    ///
    /// Scratch buffers (`all_descs_scratch`, `io_buf_scratch`) are
    /// `clear()`-ed (length zeroed, capacity retained) so the
    /// next worker iteration starts with no stale entries but
    /// without paying re-allocation cost on the first request.
    ///
    /// # Failure consequences
    ///
    /// On any resource-creation failure inside this function
    /// (`EventFd::new`, `try_clone`, `thread::Builder::spawn`),
    /// the engine is left holding the *old* `SpawnedEngine` whose
    /// `handle` field is `None` (taken by
    /// `stop_worker_and_reclaim_state` before this respawn).
    /// Future kicks via `process_requests` write to the stale
    /// `kick_fd` that no live worker is reading; the eventfd's
    /// counter increments harmlessly, but no IO completes — the
    /// guest will hang on every request until
    /// `kernel.hung_task_timeout_secs` (default 120 s) fires or
    /// the host destroys the device. The error is logged but not
    /// propagated to the caller (`reset()` returns `()` and the
    /// vCPU thread continues). This is permanent device death;
    /// only constructing a fresh `VirtioBlk` recovers the disk.
    #[cfg(not(test))]
    pub(crate) fn respawn_worker(&mut self, mut state: BlkWorkerState) {
        let (ops_bucket, bytes_bucket) = buckets_from_throttle(self.throttle);
        state.ops_bucket = ops_bucket;
        state.bytes_bucket = bytes_bucket;
        state.all_descs_scratch.clear();
        state.io_buf_scratch.clear();
        // Reset throttle-stall gauge state. q.reset() (run by
        // the caller before this) cleared the queue cursor, so
        // any chain that was rolled-back-pending is now lost
        // from the device's perspective — the guest's re-bind
        // will re-issue chains from a fresh avail.idx=0. The
        // currently_stalled flag must clear and the gauge must
        // decrement to match; otherwise the gauge leaks one
        // increment per reset-while-stalled scenario across the
        // device's lifetime.
        if state.currently_stalled {
            state.currently_stalled = false;
            state.counters.record_throttle_pending_dec();
        }
        // Clear hostile-guest poison: the guest issued a virtio
        // reset, which is the only documented escape from the
        // queue-poisoned state. `invalid_avail_idx_count` stays
        // because it tracks cumulative events across the device's
        // lifetime, not per-rebind state.
        state.queue_poisoned = false;

        // Build fresh kick/stop fds — the previous worker's
        // counter values are stale (a kick that arrived during
        // the old worker's drain may have been read but never
        // serviced before the stop, and the stop counter is
        // already incremented), and a hung vCPU mid-write to the
        // old kick_fd has nothing to coalesce against. Fresh fds
        // give a clean slate.
        //
        // The OLD worker's timerfd is owned by `worker_thread_main`'s
        // stack frame and dropped on STOP_TOKEN exit; we do NOT
        // need to migrate it. By the time this respawn runs:
        //   * `q.reset()` (called by the parent `reset_engine_spawned`
        //     just above this respawn) cleared the queue cursor —
        //     any chain that was rolled back via `set_next_avail`
        //     is gone from the device's perspective.
        //   * `state.ops_bucket` and `state.bytes_bucket` are
        //     rebuilt from `self.throttle` to full capacity, so
        //     the new worker's first drain attempt will not stall
        //     on a refill deficit (no timerfd needs to be armed
        //     for a chain that never re-stalls).
        //   * The guest must rebind (publish fresh queue addresses
        //     and set `QUEUE_READY = 1`) before any kick can fire.
        //     Until then `drain_bracket_impl` short-circuits on
        //     the `queues[REQ_QUEUE].ready()` gate — no drain, no
        //     stall, no need for a pending timerfd.
        // The clean-state contract above means a new timerfd
        // arms naturally on the first post-rebind stall, exactly
        // when one is needed.
        let kick_fd = match EventFd::new(libc::EFD_NONBLOCK) {
            Ok(fd) => fd,
            Err(e) => {
                tracing::error!(
                    %e,
                    "virtio-blk reset: kick eventfd creation failed; \
                     leaving device without a worker — IO will not \
                     be serviced until reconstruction"
                );
                return;
            }
        };
        let stop_fd = match EventFd::new(libc::EFD_NONBLOCK) {
            Ok(fd) => fd,
            Err(e) => {
                tracing::error!(
                    %e,
                    "virtio-blk reset: stop eventfd creation failed; \
                     leaving device without a worker — IO will not \
                     be serviced until reconstruction"
                );
                return;
            }
        };
        let worker_kick = match kick_fd.try_clone() {
            Ok(fd) => fd,
            Err(e) => {
                tracing::error!(
                    %e,
                    "virtio-blk reset: kick eventfd clone failed; \
                     leaving device without a worker"
                );
                return;
            }
        };
        let worker_stop = match stop_fd.try_clone() {
            Ok(fd) => fd,
            Err(e) => {
                tracing::error!(
                    %e,
                    "virtio-blk reset: stop eventfd clone failed; \
                     leaving device without a worker"
                );
                return;
            }
        };
        // Worker-side read clone of the host-owned `pause_evt`.
        // `try_clone` is `dup(2)`: it produces a new file descriptor
        // that points at the SAME underlying eventfd kernel object,
        // so the counter and any pending POLLIN readiness are shared
        // with `self.pause_evt`. The clone exists not to give the
        // worker a private counter (it can't — the kernel object is
        // shared) but because each fd can be registered in only one
        // epoll set: the worker's epoll holds this fd, while the
        // host side keeps `self.pause_evt` for `pause()` /
        // `is_paused()`. Counter cleanliness across respawns is
        // handled separately by `reset_engine_spawned`'s Phase 3
        // `pause_evt.read()` drain (V3) — a stale `1` from a
        // pre-stop write would otherwise carry across to the new
        // worker and trigger an immediate spurious park.
        let pause_fd = match self.pause_evt.try_clone() {
            Ok(fd) => fd,
            Err(e) => {
                tracing::error!(
                    %e,
                    "virtio-blk reset: pause eventfd clone failed; \
                     leaving device without a worker"
                );
                return;
            }
        };
        // Clone the queue handles and Arcs the worker needs.
        // QueueSync is internally an `Arc<Mutex<Queue>>` so the
        // clone is cheap (refcount bump).
        let worker_queues = [self.worker.queues[REQ_QUEUE].clone()];
        let worker_mem = Arc::clone(&self.mem);
        let worker_irq = Arc::clone(&self.irq_evt);
        let worker_status = Arc::clone(&self.interrupt_status);
        let worker_device_status = Arc::clone(&self.device_status);
        let worker_warned = Arc::clone(&self.mem_unset_warned);
        let worker_paused = Arc::clone(&self.paused);
        let worker_parked_evt_slot = Arc::clone(&self.parked_evt);
        // Snapshot the placement at spawn time. A subsequent
        // `set_worker_placement` call only takes effect on the
        // NEXT respawn; the running worker observes the placement
        // captured here. This matches cloud-hypervisor's "topology
        // applied at activate()" pattern.
        let worker_placement = self.worker_placement.clone();

        let handle = match thread::Builder::new()
            .name("ktstr-vblk".to_string())
            .spawn(move || {
                worker_thread_main(
                    state,
                    worker_queues,
                    worker_mem,
                    worker_irq,
                    worker_status,
                    worker_device_status,
                    worker_warned,
                    worker_paused,
                    worker_placement,
                    worker_kick,
                    worker_stop,
                    pause_fd,
                    worker_parked_evt_slot,
                )
            }) {
            Ok(h) => h,
            Err(e) => {
                tracing::error!(
                    %e,
                    "virtio-blk reset: worker thread spawn failed; \
                     leaving device without a worker"
                );
                return;
            }
        };
        let WorkerEngine::Spawned(eng) = &mut self.worker.engine;
        *eng = SpawnedEngine {
            kick_fd,
            stop_fd,
            handle: Some(handle),
            respawn_pending: None,
        };
    }

    /// Signal the worker thread to park for a failure-dump
    /// rendezvous. Writes 1 to `pause_evt`; the worker's
    /// `epoll_wait` resumes on PAUSE_TOKEN, drains the eventfd
    /// counter, stores `paused=true` (Release), and parks in a
    /// 10 ms `park_timeout` loop until [`Self::resume`] clears
    /// the flag.
    ///
    /// The freeze coordinator polls `paused.load(Acquire)` after
    /// calling this to confirm the worker has reached the parked
    /// state before reading guest memory. The Release/Acquire
    /// pair provides the happens-before edge that makes the
    /// host-side post-rendezvous reads observe every queue
    /// mutation the worker performed pre-pause.
    ///
    /// Cfg-independent: `cfg(test)` builds use the inline engine,
    /// so `pause()` writes to the host eventfd but no worker is
    /// blocked on it; the test harness can inspect
    /// `self.paused.load()` directly to verify the host-side
    /// rendezvous machinery without a worker thread.
    ///
    /// On EAGAIN (counter saturation at u64::MAX-1) or EBADF
    /// (closed fd during shutdown), we log via `tracing::warn!`
    /// and return — the caller's downstream `paused.load(Acquire)`
    /// poll either succeeds (a prior pause ack is still latched) or
    /// times out at the 30s rendezvous deadline. Saturation is
    /// implausible in practice (every `pause()` is paired with a
    /// `resume()` that does NOT increment the counter; the worker's
    /// drain reads it back to 0 each cycle).
    pub fn pause(&self) {
        // No-live-worker fast path. With the deferred-spawn lifecycle
        // (initial worker created on the first DRIVER_OK), there is a
        // window between `with_options` and the guest's bind where no
        // thread is reading `pause_fd`. Writing the eventfd is still
        // safe — counter just accumulates harmlessly, and `reset`'s
        // Phase 3 drain (V3) clears it before the next worker spawns —
        // but the counter would otherwise carry a stale tick across
        // a respawn, and the rendezvous already passes vacuously
        // because `paused` was initialised to `true` and is never
        // cleared until the worker actually starts. Skip the write
        // and log at `debug` level so a misuse (pause without a
        // worker) is observable but not noisy.
        #[cfg(not(test))]
        {
            let WorkerEngine::Spawned(eng) = &self.worker.engine;
            if eng.handle.is_none() {
                tracing::debug!(
                    "virtio-blk pause() with no live worker; \
                     `paused` is already `true` from construction \
                     (or post-stop), rendezvous will pass vacuously"
                );
                return;
            }
        }
        if let Err(e) = self.pause_evt.write(1) {
            tracing::warn!(%e, "virtio-blk pause_evt.write failed");
        }
    }

    /// Clear the worker's parked state. Stores `paused=false`
    /// (Release); the worker's 10 ms `park_timeout` Acquire-load
    /// observes the clear within 10 ms and resumes its
    /// `epoll_wait` loop. The `unpark` call is a hint — the
    /// `park_timeout` already wakes periodically so a missed
    /// unpark is bounded at 10 ms latency, not unbounded.
    ///
    /// Cfg-independent for the same reason as [`Self::pause`].
    /// Returns `true` if a worker thread is alive and was
    /// unparked; `false` if the engine has no live worker (test
    /// mode, post-stop, post-failed-respawn). Callers use the
    /// return value to skip a `resume()` that has nothing to
    /// resume.
    pub fn resume(&self) -> bool {
        // No-live-worker fast path. Mirrors `pause()`'s early-return:
        // when the engine has no live thread (pre-DRIVER_OK, post-stop,
        // post-failed-respawn), preserve the V1 sentinel by RE-ARMING
        // `paused = true` instead of clearing it. Without this, a
        // dual-snapshot freeze (early + late) that calls
        // pause()/resume() across the rebind window would clear the
        // sentinel on the first resume(), and the second freeze's
        // is_paused() poll would observe `false` and time out at
        // FREEZE_RENDEZVOUS_TIMEOUT waiting for a worker that does
        // not exist. Re-arming preserves the vacuous-pass invariant
        // across consecutive freezes.
        #[cfg(not(test))]
        {
            let WorkerEngine::Spawned(eng) = &self.worker.engine;
            if let Some(ref handle) = eng.handle {
                self.paused.store(false, Ordering::Release);
                handle.thread().unpark();
                return true;
            }
            // No live worker — re-arm the sentinel.
            self.paused.store(true, Ordering::Release);
            false
        }
        #[cfg(test)]
        {
            // Inline engine: no worker thread to unpark; the
            // store(Release) above is the entire resume side. A
            // test harness driving pause/resume observes the
            // updated `paused` flag directly.
            self.paused.store(false, Ordering::Release);
            false
        }
    }

    /// Return `true` when the worker has acknowledged a prior
    /// [`Self::pause`] call by parking. The freeze coordinator's
    /// rendezvous loop uses this to wait for the worker's parked
    /// state before reading guest memory. Acquire ordering pairs
    /// with the worker's `paused.store(true, Release)` so the
    /// host-side reads happen-after every queue mutation the
    /// worker performed pre-pause.
    ///
    /// Cfg-independent for the same reason as [`Self::pause`].
    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Acquire)
    }
}

/// Maximum number of retries [`signal_worker_stop`] performs when
/// `EventFd::write` returns `WouldBlock` (EAGAIN). The eventfd
/// counter saturates at `u64::MAX - 1`; reaching that value
/// requires `~2^64` unbalanced writes, which the device never
/// emits — each `reset()`/`Drop` writes the stop_fd exactly once
/// per fresh fd allocation. The retry loop exists strictly as
/// defense-in-depth against a future regression that re-uses a
/// long-lived stop_fd (or any other path that could let the
/// counter accumulate). 4 retries with `thread::yield_now`
/// between each gives the worker thread (running on the same
/// CPU under contention) a chance to drain the counter via its
/// `epoll_wait → read` cycle.
#[cfg(not(test))]
const STOP_FD_WRITE_MAX_RETRIES: u32 = 4;

/// Best-effort signal to the worker thread to exit by writing 1
/// to its `stop_fd`. Retries up to [`STOP_FD_WRITE_MAX_RETRIES`]
/// times on `WouldBlock` (EAGAIN — counter saturation),
/// yielding the scheduler between attempts so a co-located
/// worker can drain the eventfd counter. Logs the per-attempt
/// failure so the operator can see the rare path even when the
/// retry succeeds.
///
/// On exhaustion: log a structured warn and return — the caller
/// (`Drop` / `stop_worker_and_reclaim_state`) proceeds to the
/// join-with-timeout path. If the stop signal never reaches the
/// worker the join will time out and the existing
/// permanent-workerless diagnostic surfaces. The retry exists to
/// surface the failure-path itself; it does NOT promise the
/// worker will exit (only the join timeout does).
///
/// `device_id` is the per-device tracing tuple (stop_fd raw fd,
/// instance_id, capacity_sectors) so a warn can correlate to
/// the wedged device without the caller plumbing the same
/// fields through. Free function (not method) so the borrow is
/// limited to the EventFd reference; the caller still owns
/// `&mut self.worker.engine`.
#[cfg(not(test))]
pub(crate) fn signal_worker_stop(
    stop_fd: &EventFd,
    raw_fd: std::os::unix::io::RawFd,
    instance_id: u64,
    capacity_sectors: u64,
) {
    for attempt in 0..STOP_FD_WRITE_MAX_RETRIES {
        match stop_fd.write(1) {
            Ok(()) => return,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                tracing::warn!(
                    attempt,
                    stop_fd = raw_fd,
                    instance_id,
                    capacity_sectors,
                    "virtio-blk stop_fd write returned WouldBlock; \
                     eventfd counter likely saturated. Yielding and retrying"
                );
                std::thread::yield_now();
            }
            Err(e) => {
                tracing::error!(
                    attempt,
                    stop_fd = raw_fd,
                    instance_id,
                    capacity_sectors,
                    %e,
                    "virtio-blk stop_fd write failed with non-EAGAIN error; \
                     worker may not observe the stop signal — \
                     downstream join will surface the timeout"
                );
                return;
            }
        }
    }
    tracing::error!(
        max_retries = STOP_FD_WRITE_MAX_RETRIES,
        stop_fd = raw_fd,
        instance_id,
        capacity_sectors,
        "virtio-blk stop_fd write exhausted retries on WouldBlock; \
         worker did not consume the eventfd counter in time — \
         downstream join will surface the timeout and the device \
         enters the permanent-workerless state"
    );
}

/// Upper bound on how long [`VirtioBlk::drop`] will block while
/// joining the worker thread.
///
/// 1 s is a deliberate trade between two failure modes. Below 1 s,
/// the timeout would fire on healthy shutdowns under load — the
/// worker may be mid-`pread`/`pwrite` when `stop_fd` is signalled,
/// and a fast-but-not-instant drain (cold page cache, contended
/// disk) can take tens to hundreds of milliseconds before the
/// worker reaches the next `epoll_wait` and observes the stop. A
/// budget shorter than typical drain latency would log false
/// "wedged worker" warnings and detach threads that were about to
/// exit. Above 1 s, the budget would risk vCPU thread starvation
/// during freeze rendezvous: the freeze coordinator's SIGRTMIN
/// rendezvous timeout is 30 s and the vCPU thread can be mid-`drop`
/// at that moment, so any `Drop` blocking budget compounds with
/// other pre-rendezvous overhead.
///
/// The 1 s value is large enough to absorb realistic drain
/// latency on warm caches and small enough to keep the `Drop`
/// completion well below the rendezvous threshold.
pub(crate) const DROP_JOIN_TIMEOUT: Duration = Duration::from_secs(1);

/// Upper bound on how long [`VirtioBlk::reset`] (production
/// `WorkerEngine::Spawned` path) will block while joining the
/// outgoing worker thread before declaring it wedged and entering
/// the permanent-device-death state documented at
/// [`VirtioBlk::reset_engine_spawned`].
///
/// The same budget as [`DROP_JOIN_TIMEOUT`] (1 s) and for the same
/// reasons: a `reset()` runs on the vCPU thread that received the
/// `STATUS = 0` MMIO write, and that vCPU thread can be the next
/// SIGRTMIN target the freeze coordinator picks for a
/// failure-dump rendezvous (30 s wall budget at the coordinator
/// level — see `FREEZE_RENDEZVOUS_TIMEOUT` in
/// `src/vmm/freeze_coord.rs`). An unbounded `handle.join()` here would
/// block the vCPU through the worker's wedged `pread`/`pwrite`
/// (NFS stall, slow page cache, hung block device) and the freeze
/// would either time out empty or arrive minutes late. Capping at
/// the same 1 s the Drop path uses keeps the "reset takes ≤ 1 s
/// of vCPU time" invariant uniform — a guest issuing a re-bind
/// burst (multiple resets in flight from a confused driver) does
/// not compound the per-reset cap into a multi-second freeze
/// blocker.
///
/// Below 1 s would fire false-positive timeouts on healthy resets
/// where the worker is mid-sync on a contended disk; above 1 s
/// would let a single hung worker pin the vCPU past the freeze
/// coordinator's rendezvous tolerance.
///
/// On timeout the device enters the same permanent-workerless
/// state described in [`VirtioBlk::respawn_worker`]'s "Failure
/// consequences" section: future kicks land on a stale `kick_fd`
/// and the guest hangs on every request until
/// `kernel.hung_task_timeout_secs` (default 120 s) fires. Only
/// constructing a fresh `VirtioBlk` recovers IO service. This is
/// the explicit trade chosen over blocking a vCPU thread
/// indefinitely — the same trade [`DROP_JOIN_TIMEOUT`] makes for
/// the destructor path.
///
/// Visible to `cfg(test)` builds so the unit-test module can pin
/// the constant's value via [`reset_join_timeout_matches_drop_budget`]
/// without duplicating the literal. The production callsite in
/// [`VirtioBlk::stop_worker_and_reclaim_state`] is itself
/// `cfg(not(test))`, so the const stays unread in test builds —
/// the test module references it explicitly.
pub(crate) const RESET_JOIN_TIMEOUT: Duration = Duration::from_secs(1);

/// Outcome of a bounded join attempt by [`join_worker_with_timeout`].
///
/// The variants distinguish observable shutdown states so callers
/// can log appropriately and unit tests can assert which path the
/// worker took. `Joined` carries the recovered `BlkWorkerState`;
/// the other variants are valueless because the state is either
/// lost (panic) or still owned by a detached helper / worker
/// thread (timeout, helper failure).
pub(crate) enum JoinWithTimeoutOutcome {
    /// Worker exited normally and yielded its `BlkWorkerState`.
    Joined(BlkWorkerState),
    /// Worker panicked. The variant carries the panic payload
    /// returned by `JoinHandle::join` so the caller can render it
    /// (commonly a `&'static str` or `String` from `panic!(…)`)
    /// into a log message via `Debug` or by downcasting.
    Panicked(Box<dyn std::any::Any + Send>),
    /// Worker did not exit within `timeout`. The original
    /// `JoinHandle` is held by the helper thread, which continues
    /// running until the worker finally exits.
    TimedOut,
    /// `thread::Builder::spawn` for the helper thread failed
    /// (typically `EAGAIN` from `RLIMIT_NPROC` or thread-count
    /// exhaustion). The original handle was dropped — the worker
    /// is detached.
    HelperSpawnFailed,
    /// Helper thread itself panicked before forwarding the join
    /// result. Worker's outcome is unknown.
    HelperDisconnected,
}

/// Best-effort conversion of a `JoinHandle::join` panic payload to
/// a borrowed `&str`. Matches the two variants `panic!(…)` emits
/// in safe code: `&'static str` for `panic!("literal")` and
/// `String` for `panic!("{}", x)` / `panic!(format!(…))`. Other
/// payload types fall through to the placeholder `<non-string panic>`.
pub(crate) fn panic_payload_str(payload: &(dyn std::any::Any + Send)) -> &str {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        s
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.as_str()
    } else {
        "<non-string panic>"
    }
}

/// Join `handle` with an upper bound on the calling thread's wait
/// time.
///
/// Spawns a short-lived `ktstr-vblk-drop` helper thread that
/// performs the blocking `JoinHandle::join` and forwards the
/// result on an `mpsc::channel`. The calling thread waits via
/// `recv_timeout`; on timeout the helper is left running with the
/// handle and the calling thread returns. This bounds the
/// worst-case duration even when the worker is wedged in a
/// blocking syscall that does not check `stop_fd`
/// (`pread`/`pwrite` on slow backing, hung NFS, etc.). The vCPU
/// thread — which calls `VirtioBlk::drop` post-reset — therefore
/// cannot miss a SIGRTMIN delivery during freeze rendezvous
/// because the worker is hung.
///
/// # Outcomes
///
/// - [`JoinWithTimeoutOutcome::Joined`] — worker exited within
///   `timeout`; state recovered.
/// - [`JoinWithTimeoutOutcome::Panicked`] — worker exited within
///   `timeout`, but with a panic; state lost. The `Box<dyn Any +
///   Send>` payload returned by `JoinHandle::join` is propagated
///   so the caller can render it via [`panic_payload_str`] or by
///   downcasting to a concrete type.
/// - [`JoinWithTimeoutOutcome::TimedOut`] — worker did not exit
///   within `timeout`. Helper retains the `JoinHandle` and (through
///   it) the worker's `BlkWorkerState` until the worker finally
///   exits; if the worker never exits (perpetually-stuck IO), the
///   state outlives the device.
/// - [`JoinWithTimeoutOutcome::HelperSpawnFailed`] — the helper
///   thread itself could not be created (`RLIMIT_NPROC`,
///   thread-count exhaustion). Falling back to a direct
///   `handle.join()` would re-introduce the unbounded block this
///   function exists to prevent, so the handle is dropped and the
///   worker is detached.
/// - [`JoinWithTimeoutOutcome::HelperDisconnected`] — the helper
///   thread panicked before forwarding the join result. Worker's
///   outcome is unknown; the helper's `JoinHandle<()>` is dropped
///   when this function returns, detaching it.
///
/// # Resource retention on timeout
///
/// `BlkWorkerState` owns a `File`, an `Arc<VirtioBlkCounters>`,
/// two scratch `Vec`s, and two `TokenBucket`s. On timeout these
/// are reclaimed only when the worker thread finally exits; if it
/// does not, they outlive the device. This is the explicit trade
/// chosen over blocking a vCPU thread indefinitely. (The worker
/// also retains an `Arc<GuestMemoryMmap>` and the queue Arc clones
/// it was spawned with; those are part of the worker thread's
/// stack frame, not `BlkWorkerState`, but the same retention
/// applies — they live until the worker exits.)
pub(crate) fn join_worker_with_timeout(
    handle: thread::JoinHandle<BlkWorkerState>,
    timeout: Duration,
) -> JoinWithTimeoutOutcome {
    let (tx, rx) = mpsc::channel();
    let spawn_result = thread::Builder::new()
        .name("ktstr-vblk-drop".to_string())
        .spawn(move || {
            // Forward the join result. `send` failure means the
            // calling thread already gave up on `recv_timeout`
            // and dropped `rx`; the helper still owns the joined
            // state until this closure returns.
            let _ = tx.send(handle.join());
        });
    let _helper = match spawn_result {
        Ok(h) => h,
        Err(_) => return JoinWithTimeoutOutcome::HelperSpawnFailed,
    };
    match rx.recv_timeout(timeout) {
        Ok(Ok(state)) => JoinWithTimeoutOutcome::Joined(state),
        Ok(Err(payload)) => JoinWithTimeoutOutcome::Panicked(payload),
        Err(mpsc::RecvTimeoutError::Timeout) => JoinWithTimeoutOutcome::TimedOut,
        Err(mpsc::RecvTimeoutError::Disconnected) => JoinWithTimeoutOutcome::HelperDisconnected,
    }
}

/// `Drop` matches on `WorkerEngine` rather than gating the entire
/// impl on `cfg(not(test))`: the Inline branch is a no-op (the
/// default Drop drops `BlkWorkerState` cleanly when the engine
/// goes out of scope), the Spawned branch signals via `stop_fd`
/// and joins the worker thread so its resources (state, queues,
/// Arcs, eventfd clones) are reclaimed before `VirtioBlk` is
/// fully torn down.
///
/// The unconditional impl removes a fragility: a cfg-gated Drop
/// silently disappears in `cfg(test)`, so any pre-Drop side effect
/// added later (e.g. `tracing::debug!` on shutdown) would be
/// missing in tests. Pattern-matching the engine variant inside a
/// single impl keeps the dispatch obvious and makes adding such
/// side effects symmetric across cfgs. A regression that detached
/// the worker thread without stopping it would leave a daemon
/// thread holding the queue Arcs and the backing file open after
/// the device is dropped — visible as "test process leaks fds and
/// threads under stress."
///
/// # Bounded join
///
/// The Spawned arm quiesces the worker thread (production
/// `WorkerEngine::Spawned` path) by writing the `stop_fd` and
/// joining the thread with [`DROP_JOIN_TIMEOUT`] via
/// [`join_worker_with_timeout`]. On timeout the helper thread
/// retains the `JoinHandle` and the calling thread returns
/// without blocking further. The match arms log per-outcome
/// diagnostics — every error arm emits a structured `tracing`
/// event so the operator can correlate a missing-VM teardown
/// against the originating device. `JoinWithTimeoutOutcome::Joined`
/// is silent (clean shutdown is not logged). See
/// [`join_worker_with_timeout`] for full outcome semantics and
/// resource-retention notes, and [`DROP_JOIN_TIMEOUT`] for why
/// the budget is set where it is.
///
/// # Resource retention on `TimedOut`
///
/// When the worker join exceeds [`DROP_JOIN_TIMEOUT`] (the
/// `JoinWithTimeoutOutcome::TimedOut` arm), the [`Drop`] returns
/// without calling [`std::thread::JoinHandle::join`] — the
/// helper thread is detached and the worker keeps running. Every
/// `Arc` the worker holds remains live until the worker thread
/// exits naturally (typically when its blocking syscall
/// returns) and its captured state finally drops.
///
/// The retained Arcs are:
/// - `Arc<OnceLock<GuestMemoryMmap>>` (the `mem` field;
///   cloned into the worker thread frame). The guest memory
///   mapping stays mapped on the host until the worker exits —
///   the parent VM's teardown does NOT free guest memory at the
///   `VirtioBlk::drop` site.
/// - `Arc<EventFd>` (the IRQ eventfd, `irq_evt`). The eventfd's
///   kernel object stays alive; the kvmfd irqfd binding the
///   parent VM held does not unwind synchronously.
/// - `Arc<AtomicU32>` (the `interrupt_status` register, used
///   for the worker's release-store of `VIRTIO_MMIO_INT_VRING`).
/// - `Arc<AtomicBool>` (the `mem_unset_warned` one-shot latch).
/// - `Arc<VirtioBlkCounters>` (the per-device counter Arc the
///   worker increments on each request).
///
/// Operationally: a wedged worker means the VM teardown returns
/// to the caller (the calling thread is freed promptly, which is
/// the [`DROP_JOIN_TIMEOUT`] mechanism's whole point — usually a
/// vCPU thread that the freeze coordinator must not pin) but
/// the per-device shared state stays mapped until the kernel
/// eventually unblocks the worker. For long-lived host
/// processes that build many VMs, this can accumulate retained
/// memory; restart the host process to flush all leaked
/// per-device state. Bug reports mentioning "host RSS keeps
/// climbing across many ktstr test runs even though no VM is
/// active" should investigate `tracing::warn!` lines from this
/// arm to identify the wedged device(s).
impl Drop for VirtioBlk {
    fn drop(&mut self) {
        // Snapshot the device-identifier fields BEFORE the
        // match so the per-arm logs can correlate the device
        // across multiple concurrent VirtioBlk drops without
        // borrowing `self` after the `&mut self.worker.engine`
        // mutable borrow lands. None of the three are stable
        // across host restarts (`stop_fd` recycles, `instance_id`
        // resets at process start) but together they uniquely
        // identify the device within this process run.
        // `instance_id` replaces an earlier `self as *const _`
        // pointer field — the pointer leaked the host's ASLR
        // layout into log output (environment leakage); the
        // process-local counter has the same uniqueness shape
        // without the leak.
        //
        // The cfg(test) Inline arm doesn't consume these
        // snapshots; the `let _ = (capacity_sectors, instance_id);`
        // reference inside that arm satisfies the
        // `unused_variables` lint under cfg(test) where the
        // Spawned arm is excluded. (`stop_fd` is read inside the
        // cfg(not(test)) Spawned arm directly, so it doesn't
        // need the same dead-code dance.)
        let capacity_sectors = self.capacity_sectors;
        let instance_id = self.instance_id;
        match &mut self.worker.engine {
            #[cfg(test)]
            WorkerEngine::Inline(_) => {
                // Default-drop the inline state when this fn returns.
                // Reference the snapshot vars to avoid `unused`
                // lints in cfg(test).
                let _ = (capacity_sectors, instance_id);
            }
            #[cfg(not(test))]
            WorkerEngine::Spawned(eng) => {
                // The third device-identifier field (`stop_fd`
                // raw fd) is only meaningful in the Spawned
                // arm — Inline mode has no eventfd to name.
                let stop_fd = eng.stop_fd.as_raw_fd();
                // Unpause first so a parked worker observes the
                // upcoming stop signal. Same rationale as
                // `reset_engine_spawned`: a worker stuck in its
                // `park_timeout(10ms)` Acquire-load loop is
                // unreachable from `epoll_wait`, so STOP_TOKEN
                // would block until the 10 ms tick + Acquire-load
                // sees the cleared flag. Clearing here makes the
                // worker exit the park within 10 ms (faster on
                // the unpark hint) so the join timeout window
                // (DROP_JOIN_TIMEOUT, 1 s) is not consumed by
                // park latency alone.
                self.paused.store(false, Ordering::Release);
                if let Some(ref handle) = eng.handle {
                    handle.thread().unpark();
                }
                // Signal the worker to exit via the stop_fd
                // helper, which retries on EAGAIN (eventfd
                // counter saturation) up to STOP_FD_WRITE_MAX_RETRIES
                // times before giving up. On exhaustion the join
                // below absorbs the failure via DROP_JOIN_TIMEOUT.
                signal_worker_stop(&eng.stop_fd, stop_fd, instance_id, capacity_sectors);
                if let Some(handle) = eng.handle.take() {
                    match join_worker_with_timeout(handle, DROP_JOIN_TIMEOUT) {
                        JoinWithTimeoutOutcome::Joined(_state) => {
                            // Clean shutdown: state drops at scope end.
                        }
                        JoinWithTimeoutOutcome::Panicked(payload) => {
                            tracing::error!(
                                panic = panic_payload_str(&*payload),
                                stop_fd,
                                capacity_sectors,
                                instance_id,
                                "virtio-blk worker thread panicked"
                            );
                        }
                        JoinWithTimeoutOutcome::TimedOut => {
                            tracing::warn!(
                                timeout_s = DROP_JOIN_TIMEOUT.as_secs_f32(),
                                stop_fd,
                                capacity_sectors,
                                instance_id,
                                "virtio-blk worker did not exit within \
                                 DROP_JOIN_TIMEOUT of stop_fd; leaking \
                                 the worker thread to avoid blocking the \
                                 calling thread (likely a vCPU). Worker \
                                 is wedged in a blocking syscall that \
                                 does not check stop_fd. \
                                 hint: identify the wedged device by \
                                 stop_fd / instance_id / capacity_sectors \
                                 above; per-device GuestMemoryMmap and \
                                 EventFd Arcs stay live until the worker \
                                 unblocks (see Drop's resource-retention \
                                 doc). hint: kill -USR1 the host process \
                                 to dump worker thread backtraces, OR \
                                 check `dmesg` for the backing fd's \
                                 storage path stalling on I/O."
                            );
                        }
                        JoinWithTimeoutOutcome::HelperSpawnFailed => {
                            tracing::error!(
                                stop_fd,
                                capacity_sectors,
                                instance_id,
                                "virtio-blk drop helper thread spawn \
                                 failed; detaching worker without join"
                            );
                        }
                        JoinWithTimeoutOutcome::HelperDisconnected => {
                            tracing::error!(
                                stop_fd,
                                capacity_sectors,
                                instance_id,
                                "virtio-blk drop helper thread \
                                 terminated without forwarding the \
                                 worker join result"
                            );
                        }
                    }
                }
            }
        }
    }
}
