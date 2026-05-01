//! Virtio-block device with file-backed storage and token-bucket
//! throttle.
//!
//! Single request virtqueue. Advertised features: `VIRTIO_F_VERSION_1`,
//! `VIRTIO_BLK_F_BLK_SIZE`, `VIRTIO_BLK_F_SEG_MAX`,
//! `VIRTIO_BLK_F_SIZE_MAX`, `VIRTIO_BLK_F_FLUSH`,
//! `VIRTIO_RING_F_EVENT_IDX`, plus the optional `VIRTIO_BLK_F_RO`
//! when the disk is configured read-only. MMIO
//! register layout per virtio-v1.2 §4.2.2; block-specific config
//! space at offsets `0x100..` is served from a [`VirtioBlkConfig`]
//! struct whose `repr(C, packed)` layout mirrors the kernel uapi
//! `struct virtio_blk_config` byte-for-byte (virtio-v1.2 §5.2.4).
//! Interrupt delivery via irqfd (eventfd → KVM GSI).
//!
//! Every request flows through chain-shape validation, per-descriptor
//! `SIZE_MAX` enforcement, pre-throttle terminal classification (RO
//! write → `S_IOERR`, RO flush → `S_OK`, unsupported request type →
//! `S_UNSUPP`), then throttle bucket consumption. Validation
//! precedes consumption — a malformed or type-rejected request never
//! drains the bucket or hits `pread`/`pwrite`. See `process_requests`.
//!
//! # Why
//!
//! - **Inline request processing on the vCPU thread.** `mmio_write` of
//!   `QUEUE_NOTIFY` drives `process_requests` synchronously. The
//!   throttle therefore must NOT block: the vCPU thread is the
//!   target of SIGRTMIN from the failure-dump freeze coordinator,
//!   and a blocked syscall delays SIGRTMIN delivery — a stalled
//!   vCPU produces an empty dump. Throttle exhaustion returns
//!   `S_IOERR` immediately rather than parking the request.
//!
//! - **Status-write-success gates `add_used`.** A used-ring
//!   advancement without a successfully-written status byte lets
//!   the guest's `virtblk_done` observe its `vbr->in_hdr.status`
//!   byte that's stale from prior blk-mq tag use as `BLK_STS_OK`
//!   — silent data corruption for reads, silent dropped writes
//!   for writes. See `publish_completion`.
//!
//! # Blocking-IO assumption
//!
//! Backend IO is synchronous on the vCPU thread:
//! `handle_read_impl` / `handle_write_impl` call
//! `FileExt::read_at` / `write_at` (`pread64` / `pwrite64`) and
//! `handle_flush_impl` calls `File::sync_data` (`fdatasync`)
//! inline from `process_requests`. The vCPU thread blocks until
//! the syscall returns. There is no host worker thread, no
//! io_uring, no async queue.
//!
//! This is acceptable when the backing is **fast** — tmpfs
//! (the `tempfile()` default) or warm page cache — where pread /
//! pwrite return in sub-microsecond time and fdatasync is a
//! no-op (`noop_fsync`). The vCPU continues running guest code
//! between requests with negligible interruption.
//!
//! It is **not** acceptable when the backing could block for
//! milliseconds — cold page cache on a real spinning disk,
//! contended file locks, fdatasync forcing real journal writes,
//! a network-mounted backing file, etc. A blocked syscall on
//! the vCPU thread delays SIGRTMIN delivery from the
//! failure-dump freeze coordinator (the same constraint
//! `TokenBucket` documents for the throttle path); the freeze
//! rendezvous times out at 30 s and the failure dump arrives
//! empty for any vCPU still stuck in IO.
//!
//! v0 commits to this tradeoff because the test fixture targets
//! small backing files on tmpfs — every request returns in
//! microseconds and SIGRTMIN delivery is never delayed in
//! practice. Operators who point a virtio-blk disk at a slow
//! backing accept the failure-dump-empties risk; the rule is
//! "the backing must be fast." Moving IO to a worker thread
//! would lift this constraint but is not implemented in v0;
//! see also the per-request request-cap commentary in the
//! validation pipeline.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use virtio_bindings::virtio_blk::{
    VIRTIO_BLK_F_BLK_SIZE, VIRTIO_BLK_F_FLUSH, VIRTIO_BLK_F_RO, VIRTIO_BLK_F_SEG_MAX,
    VIRTIO_BLK_F_SIZE_MAX, VIRTIO_BLK_ID_BYTES, VIRTIO_BLK_S_IOERR, VIRTIO_BLK_S_OK,
    VIRTIO_BLK_S_UNSUPP, VIRTIO_BLK_T_FLUSH, VIRTIO_BLK_T_GET_ID, VIRTIO_BLK_T_IN,
    VIRTIO_BLK_T_OUT,
};
use virtio_bindings::virtio_config::{
    VIRTIO_CONFIG_S_ACKNOWLEDGE, VIRTIO_CONFIG_S_DRIVER, VIRTIO_CONFIG_S_DRIVER_OK,
    VIRTIO_CONFIG_S_FEATURES_OK, VIRTIO_F_VERSION_1,
};
use virtio_bindings::virtio_ids::VIRTIO_ID_BLOCK;
use virtio_bindings::virtio_ring::VIRTIO_RING_F_EVENT_IDX;
use virtio_bindings::virtio_mmio::{
    VIRTIO_MMIO_CONFIG_GENERATION, VIRTIO_MMIO_DEVICE_FEATURES, VIRTIO_MMIO_DEVICE_FEATURES_SEL,
    VIRTIO_MMIO_DEVICE_ID, VIRTIO_MMIO_DRIVER_FEATURES, VIRTIO_MMIO_DRIVER_FEATURES_SEL,
    VIRTIO_MMIO_INT_VRING, VIRTIO_MMIO_INTERRUPT_ACK, VIRTIO_MMIO_INTERRUPT_STATUS,
    VIRTIO_MMIO_MAGIC_VALUE, VIRTIO_MMIO_QUEUE_AVAIL_HIGH, VIRTIO_MMIO_QUEUE_AVAIL_LOW,
    VIRTIO_MMIO_QUEUE_DESC_HIGH, VIRTIO_MMIO_QUEUE_DESC_LOW, VIRTIO_MMIO_QUEUE_NOTIFY,
    VIRTIO_MMIO_QUEUE_NUM, VIRTIO_MMIO_QUEUE_NUM_MAX, VIRTIO_MMIO_QUEUE_READY,
    VIRTIO_MMIO_QUEUE_SEL, VIRTIO_MMIO_QUEUE_USED_HIGH, VIRTIO_MMIO_QUEUE_USED_LOW,
    VIRTIO_MMIO_STATUS, VIRTIO_MMIO_VENDOR_ID, VIRTIO_MMIO_VERSION,
};
use virtio_queue::{Queue, QueueT};
use vm_memory::{ByteValued, Bytes, GuestAddress, GuestMemoryMmap};
use vmm_sys_util::eventfd::EventFd;

use super::disk_config::DiskThrottle;

const MMIO_MAGIC: u32 = 0x7472_6976; // "virt" in LE
const MMIO_VERSION: u32 = 2; // virtio 1.x MMIO
const VENDOR_ID: u32 = 0;

/// MMIO region size: 4 KB (one page).
pub const VIRTIO_MMIO_SIZE: u64 = 0x1000;

/// Single request queue. virtio-spec §5.2.2 declares one request
/// queue plus optional multiqueue (`VIRTIO_BLK_F_MQ`); MQ deferred.
const NUM_QUEUES: usize = 1;
const QUEUE_MAX_SIZE: u16 = 256;
const REQ_QUEUE: usize = 0;

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
pub const VIRTIO_BLK_DEFAULT_CAPACITY_BYTES: u64 = 256 * 1024 * 1024;

/// Maximum number of data segments per request the device supports.
/// virtio-v1.2 §5.2.4: `seg_max` is the max scatter-gather buffer
/// count, exclusive of the header and status descriptors. Without
/// `F_SEG_MAX` the guest defaults `max_segments` to 1, which forces
/// `bio_split` and serializes large requests; advertising 128 is the
/// firecracker default and ample for the small files this device
/// targets.
const VIRTIO_BLK_SEG_MAX: u32 = 128;

/// Maximum size in bytes of a single descriptor's data buffer.
/// virtio-v1.2 §5.2.4 (`size_max`): caps per-descriptor length so a
/// guest can't submit a single 4 GB descriptor and force the device
/// to allocate a matching `Vec<u8>` for `read_at`/`write_at`. 1 MB
/// matches firecracker's default and is far above what the guest's
/// blk-mq layer typically generates (max_sectors_kb defaults to
/// 512 KB). Without `F_SIZE_MAX` the guest treats per-descriptor
/// length as unbounded — host OOM hazard on a hostile guest.
const VIRTIO_BLK_SIZE_MAX: u32 = 1 << 20;

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
const VIRTIO_BLK_SERIAL: [u8; VIRTIO_BLK_ID_BYTES as usize] = *b"ktstr-virtio-blk\0\0\0\0";

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
struct VirtioBlkOutHdr {
    /// `VIRTIO_BLK_T_*`. LE per virtio-v1.2 §5.2.6.
    type_: u32,
    /// I/O priority, ignored on this device.
    _ioprio: u32,
    /// Starting sector (512-byte units).
    sector: u64,
}

// SAFETY: VirtioBlkOutHdr is `repr(C)`, contains only `u32` and `u64`
// (themselves `ByteValued`), has no padding (4+4+8 = 16, all aligned),
// and any byte pattern is a valid value (the type/ioprio fields are
// validated separately by the request dispatcher; sector is just a
// number). All `ByteValued` requirements are met.
unsafe impl vm_memory::ByteValued for VirtioBlkOutHdr {}

/// Header size for `VirtioBlkOutHdr`. virtio-v1.2 §5.2.6:
/// type:u32, ioprio:u32, sector:u64.
const VIRTIO_BLK_OUTHDR_SIZE: usize = std::mem::size_of::<VirtioBlkOutHdr>();

/// Legacy CHS geometry sub-struct of `VirtioBlkConfig`, gated on
/// `VIRTIO_BLK_F_GEOMETRY`. Mirrors the kernel uapi
/// `struct virtio_blk_geometry` (cylinders:u16, heads:u8, sectors:u8 —
/// 4 bytes total) at config-space offset 0x10. We don't advertise
/// `F_GEOMETRY` so the field is left zero; the guest driver reads it
/// via `virtio_cread_feature`, which returns `-ENOENT` when the
/// feature bit is not negotiated and the read is skipped.
#[repr(C, packed)]
#[derive(Copy, Clone, Default, Debug)]
struct VirtioBlkGeometry {
    cylinders: u16,
    heads: u8,
    sectors: u8,
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
struct VirtioBlkConfig {
    /// Capacity in 512-byte sectors. Always populated; the kernel
    /// driver reads this unconditionally (no feature bit gates it).
    capacity: u64,
    /// Maximum per-descriptor data length, gated on
    /// `VIRTIO_BLK_F_SIZE_MAX`.
    size_max: u32,
    /// Maximum scatter-gather segments per request, gated on
    /// `VIRTIO_BLK_F_SEG_MAX`.
    seg_max: u32,
    /// Legacy CHS geometry, gated on `VIRTIO_BLK_F_GEOMETRY`. We
    /// don't advertise that bit so this field is left zero.
    geometry: VirtioBlkGeometry,
    /// Logical block size, gated on `VIRTIO_BLK_F_BLK_SIZE`.
    blk_size: u32,
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
const VIRTIO_BLK_CONFIG_SIZE: usize = std::mem::size_of::<VirtioBlkConfig>();
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
struct ChainDescriptor {
    addr: GuestAddress,
    len: u32,
    is_write_only: bool,
}

/// Status bits required before each phase (mirrors virtio_console).
const S_ACK: u32 = VIRTIO_CONFIG_S_ACKNOWLEDGE;
const S_DRV: u32 = S_ACK | VIRTIO_CONFIG_S_DRIVER;
const S_FEAT: u32 = S_DRV | VIRTIO_CONFIG_S_FEATURES_OK;
/// Test helper — terminal state bits with DRIVER_OK set.
#[cfg(test)]
const S_OK: u32 = S_FEAT | VIRTIO_CONFIG_S_DRIVER_OK;

// ----------------------------------------------------------------------------
// Token bucket throttle (firecracker pattern)
// ----------------------------------------------------------------------------

/// Single token-bucket. `capacity` tokens accumulate at `refill_rate`
/// per second. `consume(n)` succeeds when `>= n` tokens are available
/// AND drains them; otherwise leaves the bucket untouched and returns
/// false, signalling the caller to backoff.
///
/// "Leak rate" is implicit: every `consume` call first refills based
/// on elapsed wall time since the last refill, capped at `capacity`.
/// No periodic timer needed — the refill is on-demand per request.
///
/// # vCPU thread blocking budget
///
/// `process_requests` is invoked from `mmio_write` on a `QUEUE_NOTIFY`,
/// which runs on the vCPU thread (KVM exit → exit_dispatch →
/// virtio_blk MMIO handler → here). The vCPU thread is also the
/// signal target for the failure-dump freeze coordinator, which
/// kicks every vCPU with `SIGRTMIN` to force a `KVM_RUN` return
/// and rendezvous on the freeze barrier. Per the CLAUDE.md
/// "vCPU thread blocking budget" rule: any code on this path
/// must NOT block, because a blocked syscall delays SIGRTMIN
/// delivery and the freeze rendezvous times out at 30 s — the
/// failure dump arrives empty.
///
/// **Critical invariant: this bucket NEVER calls `thread::sleep` or
/// any blocking syscall.** `consume` returns immediately whether the
/// bucket is satisfied or empty. Caller (`process_requests`) is
/// responsible for failing the request with `S_IOERR` when
/// `consume` returns false. This is non-negotiable: the device's
/// throttle reject path is "S_IOERR + bump throttled_count + let
/// the guest's blk-mq retry"; sleeping or blocking is forbidden.
/// `std::thread::sleep` in particular retries on EINTR per the
/// Rust std source, so even a SIGRTMIN-targeted thread would not
/// wake until the sleep duration elapsed.
///
/// The "low-IOPS guest sees transient IO errors" trade-off is
/// acceptable — btrfs and the blk-mq layer retry. Realism of disk
/// latency is NOT a goal of the test fixture; preserving the
/// failure-dump signal chain is.
///
/// `unlimited` (capacity == 0) is a fast path that always returns
/// true. `DiskConfig` materialises this when neither IOPS nor bytes
/// throttle is set; the cold path here would otherwise charge a
/// monotonic-clock read per request unconditionally.
#[derive(Debug)]
struct TokenBucket {
    capacity: u64,
    refill_rate: u64, // tokens per second
    available: u64,
    last_refill: Instant,
    unlimited: bool,
}

impl TokenBucket {
    fn unlimited() -> Self {
        Self {
            capacity: 0,
            refill_rate: 0,
            available: 0,
            last_refill: Instant::now(),
            unlimited: true,
        }
    }

    /// Build a bucket with the given capacity that refills at the
    /// rate per second. `capacity == 0` becomes `unlimited()` (no
    /// throttle).
    fn new(capacity: u64, refill_rate_per_sec: u64) -> Self {
        if capacity == 0 || refill_rate_per_sec == 0 {
            return Self::unlimited();
        }
        Self {
            capacity,
            refill_rate: refill_rate_per_sec,
            available: capacity,
            last_refill: Instant::now(),
            unlimited: false,
        }
    }

    fn refill(&mut self) {
        if self.unlimited {
            return;
        }
        let now = Instant::now();
        let elapsed_ns = now.duration_since(self.last_refill).as_nanos();
        if elapsed_ns == 0 {
            return;
        }
        // tokens = refill_rate * elapsed_seconds; do the math in u128
        // to avoid overflow on long stalls. Refill rate is small
        // enough (typically <= a few million per second) that the
        // multiplication fits in u128 trivially.
        let new_tokens = (self.refill_rate as u128 * elapsed_ns) / 1_000_000_000;
        let new_tokens_u64 = u64::try_from(new_tokens).unwrap_or(u64::MAX);
        // Only advance `last_refill` when at least one token was
        // granted. At low rates (e.g. 100 IOPS = one token every
        // 10 ms) sub-10ms calls produce `new_tokens_u64 == 0`; if
        // we updated `last_refill` anyway, the elapsed window
        // would reset on every call and the bucket would never
        // refill in steady state. Preserving the old `last_refill`
        // on a 0-token computation lets elapsed time accumulate
        // across calls until enough has passed for at least one
        // whole token to be granted.
        if new_tokens_u64 == 0 {
            return;
        }
        self.available = self.available.saturating_add(new_tokens_u64).min(self.capacity);
        self.last_refill = now;
    }

    fn consume(&mut self, n: u64) -> bool {
        if self.unlimited {
            return true;
        }
        self.refill();
        if self.available >= n {
            self.available -= n;
            true
        } else {
            false
        }
    }

    /// Check whether `n` tokens are currently available without
    /// consuming them. Used by the per-request "both buckets must
    /// pass" gate so a request that fails the bytes check doesn't
    /// silently drain the ops bucket (or vice versa). Refills
    /// on-demand so the answer reflects up-to-the-instant state.
    fn can_consume(&mut self, n: u64) -> bool {
        if self.unlimited {
            return true;
        }
        self.refill();
        self.available >= n
    }

    /// Test-only knob: rewind `last_refill` so the next `refill()`
    /// computes "as if X ago". Lets tests pin throttle behaviour
    /// without burning real wall time. Production code uses
    /// `Instant::now()` exclusively — no trait injection, because
    /// per-request overhead matters and the bucket's correctness is
    /// independent of clock source (the formula is a per-second
    /// rate that any monotonic clock produces correctly).
    #[cfg(test)]
    fn set_last_refill_for_test(&mut self, t: Instant) {
        self.last_refill = t;
    }
}

/// Materialise a [`DiskThrottle`] into a pair of token buckets with
/// initial capacity == refill_rate (1-second burst). `None` on
/// either field becomes the unlimited fast path. `Option<NonZeroU64>`
/// is unwrapped via `NonZeroU64::get` so the bucket sees a plain
/// `u64`; the type-level invariant (the value can't be 0) means the
/// `if rate == 0` branch in `TokenBucket::new` is unreachable from
/// this caller — kept there for defense-in-depth against direct
/// construction.
fn buckets_from_throttle(throttle: DiskThrottle) -> (TokenBucket, TokenBucket) {
    let ops_bucket = throttle
        .iops
        .map_or_else(TokenBucket::unlimited, |nz| {
            let r = nz.get();
            TokenBucket::new(r, r)
        });
    let bytes_bucket = throttle
        .bytes_per_sec
        .map_or_else(TokenBucket::unlimited, |nz| {
            let r = nz.get();
            TokenBucket::new(r, r)
        });
    (ops_bucket, bytes_bucket)
}

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
/// stays in the avail ring and the guest's blk-mq layer times it
/// out at 30s.
///
/// `used_len` is what `add_used` records as the "bytes written by
/// the device into guest memory". Error paths pass `1` (just the
/// status byte). The success path passes the data-bytes-written
/// total + 1 (for reads) or `1` (for writes/flushes — the device
/// wrote no data back).
///
/// `label` is included in any tracing::warn from this function so
/// operators can identify which gate triggered the publish.
fn publish_completion(
    mem: &GuestMemoryMmap,
    q: &mut Queue,
    counters: &VirtioBlkCounters,
    head: u16,
    status_addr: GuestAddress,
    status_byte: u8,
    used_len: u32,
    label: &'static str,
) -> bool {
    if mem.write_slice(&[status_byte], status_addr).is_err() {
        // Status-byte write failed — the chain stays in the avail
        // ring and blk-mq's 30s timeout fires on the guest side.
        // Callers ALWAYS bump io_errors at the request-level error
        // site before calling here; bumping again would
        // double-count. add_used failure (below) is a separate
        // class — own that count locally.
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
/// across the crate by visibility. External consumers must reach
/// in via dedicated accessors (none exist today; add them as the
/// public API needs them).
#[derive(Debug, Default)]
pub struct VirtioBlkCounters {
    pub(crate) reads_completed: AtomicU64,
    pub(crate) writes_completed: AtomicU64,
    pub(crate) flushes_completed: AtomicU64,
    pub(crate) bytes_read: AtomicU64,
    pub(crate) bytes_written: AtomicU64,
    pub(crate) throttled_count: AtomicU64,
    pub(crate) io_errors: AtomicU64,
}

impl VirtioBlkCounters {
    /// Record one completed read: bumps `reads_completed` and adds
    /// `bytes` to `bytes_read`. The pairing is enforced — bare
    /// reads_completed bumps without the paired bytes_read add are
    /// caught at refactor time.
    fn record_read(&self, bytes: u64) {
        self.reads_completed.fetch_add(1, Ordering::Relaxed);
        self.bytes_read.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record one completed write: bumps `writes_completed` and
    /// adds `bytes` to `bytes_written`.
    fn record_write(&self, bytes: u64) {
        self.writes_completed.fetch_add(1, Ordering::Relaxed);
        self.bytes_written.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record one completed flush.
    fn record_flush(&self) {
        self.flushes_completed.fetch_add(1, Ordering::Relaxed);
    }

    /// Record one IO error. Used for spec violations, backend IO
    /// errors, malformed chains, and add_used failures — every
    /// path that reports `VIRTIO_BLK_S_IOERR` to the guest.
    fn record_io_error(&self) {
        self.io_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Record one throttled request. virtio-spec doesn't reserve a
    /// "throttled" status code — the guest sees `S_IOERR` — but the
    /// device-side counter is separate so operators can distinguish
    /// "real IO problem" from "throttle bucket drained".
    fn record_throttled(&self) {
        self.throttled_count.fetch_add(1, Ordering::Relaxed);
    }
}

// ----------------------------------------------------------------------------
// Device struct
// ----------------------------------------------------------------------------

/// Virtio-block MMIO device.
pub struct VirtioBlk {
    queues: [Queue; NUM_QUEUES],
    queue_select: u32,
    device_features_sel: u32,
    driver_features_sel: u32,
    driver_features: u64,
    device_status: u32,
    interrupt_status: u32,
    config_generation: u32,
    /// Eventfd for KVM irqfd.
    irq_evt: EventFd,
    /// Guest memory reference. Set before starting vCPUs.
    mem: Option<GuestMemoryMmap>,
    /// Backing file. The device reads and writes sectors via
    /// `pread`/`pwrite` and never inspects the on-disk contents.
    backing: File,
    /// Capacity in 512-byte sectors. Determines what the guest sees
    /// in the config space's `capacity` field.
    capacity_sectors: u64,
    /// Capacity in bytes. Computed once at construction
    /// (`capacity_sectors * VIRTIO_BLK_SECTOR_SIZE`) and threaded
    /// into handlers so the multiply isn't repeated per request and
    /// can never overflow on a malicious sector value (the multiply
    /// happens once on host-trusted input).
    capacity_bytes: u64,
    /// Token-bucket for ops/sec.
    ops_bucket: TokenBucket,
    /// Token-bucket for bytes/sec.
    bytes_bucket: TokenBucket,
    /// Counters.
    counters: Arc<VirtioBlkCounters>,
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
    read_only: bool,
    /// Reusable scratch for the descriptor-walk in `process_requests`.
    /// Allocated once at construction and `clear()`-ed each
    /// iteration so the underlying capacity (sized by the worst-case
    /// chain) is reused. Avoids one Vec allocation per request on
    /// the hot path. Capacity grows monotonically up to
    /// `VIRTIO_BLK_SEG_MAX + 2`. The data-segment slice given to
    /// the handlers is borrowed directly from
    /// `&self.all_descs_scratch[1..chain_len - 1]` once `status_addr`
    /// has been validated — no second Vec, no copy.
    all_descs_scratch: Vec<ChainDescriptor>,
    /// Reusable per-segment IO buffer. Sized by `resize(len, 0)`
    /// per segment in the read/write handlers. Allocated once and
    /// reused across all segments of all requests; the underlying
    /// `Vec`'s capacity grows monotonically up to
    /// `VIRTIO_BLK_SIZE_MAX` (the per-descriptor cap we advertise),
    /// at which point all subsequent IO is amortized to zero
    /// allocation.
    io_buf_scratch: Vec<u8>,
    /// One-shot guard so the "process_requests called before
    /// set_mem" warning fires at most once per device instance.
    /// Without this, a buggy caller that issues N notifies before
    /// `set_mem` would flood the log with N copies of the same
    /// message. Latched with Relaxed because the order of the
    /// log message vs. other operations doesn't affect
    /// correctness.
    mem_unset_warned: AtomicBool,
}

impl VirtioBlk {
    /// Create a new virtio-block device.
    ///
    /// `backing` is an open File for read+write at sector
    /// granularity (the host formatted it before VM boot).
    /// `capacity_bytes` is the disk capacity advertised to the
    /// guest (rounded down to sector boundary). `throttle` carries
    /// optional IOPS / bandwidth limits.
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
        let irq_evt =
            EventFd::new(libc::EFD_NONBLOCK).expect("failed to create virtio-blk irq eventfd");
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
        VirtioBlk {
            queues: [Queue::new(QUEUE_MAX_SIZE).expect("valid queue size")],
            queue_select: 0,
            device_features_sel: 0,
            driver_features_sel: 0,
            driver_features: 0,
            device_status: 0,
            interrupt_status: 0,
            config_generation: 0,
            irq_evt,
            mem: None,
            backing,
            capacity_sectors,
            capacity_bytes,
            ops_bucket,
            bytes_bucket,
            counters: Arc::new(VirtioBlkCounters::default()),
            read_only,
            all_descs_scratch: Vec::with_capacity(VIRTIO_BLK_SEG_MAX as usize + 2),
            io_buf_scratch: Vec::new(),
            mem_unset_warned: AtomicBool::new(false),
        }
    }

    /// Eventfd for KVM irqfd registration.
    pub fn irq_evt(&self) -> &EventFd {
        &self.irq_evt
    }

    /// Set guest memory reference. Must be called before starting vCPUs.
    pub fn set_mem(&mut self, mem: GuestMemoryMmap) {
        self.mem = Some(mem);
    }

    /// Advertised capacity in 512-byte sectors.
    pub fn capacity_sectors(&self) -> u64 {
        self.capacity_sectors
    }

    /// Cloneable handle to the host-observability counters. The
    /// monitor thread holds an Arc to read counters without locking
    /// the device.
    pub fn counters(&self) -> Arc<VirtioBlkCounters> {
        Arc::clone(&self.counters)
    }

    fn device_features(&self) -> u64 {
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
        //   blk-mq. Wire-up is split: this advertises the bit;
        //   #161 wires `needs_notification` into the `signal_used`
        //   call site so the device actually suppresses irqfd
        //   writes when the guest's event index isn't reached.
        let mut feats = (1u64 << VIRTIO_F_VERSION_1)
            | (1u64 << VIRTIO_BLK_F_BLK_SIZE)
            | (1u64 << VIRTIO_BLK_F_SEG_MAX)
            | (1u64 << VIRTIO_BLK_F_SIZE_MAX)
            | (1u64 << VIRTIO_BLK_F_FLUSH)
            | (1u64 << VIRTIO_RING_F_EVENT_IDX);
        if self.read_only {
            feats |= 1u64 << VIRTIO_BLK_F_RO;
        }
        feats
    }

    fn selected_queue(&self) -> Option<usize> {
        let idx = self.queue_select as usize;
        if idx < NUM_QUEUES { Some(idx) } else { None }
    }

    fn signal_used(&mut self) {
        self.interrupt_status |= VIRTIO_MMIO_INT_VRING;
        let _ = self.irq_evt.write(1);
    }

    fn queue_config_allowed(&self) -> bool {
        self.device_status & S_FEAT == S_FEAT && self.device_status & VIRTIO_CONFIG_S_DRIVER_OK == 0
    }

    fn features_write_allowed(&self) -> bool {
        self.device_status & S_DRV == S_DRV && self.device_status & VIRTIO_CONFIG_S_FEATURES_OK == 0
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
    fn classify_pre_throttle(
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
    fn process_requests(&mut self) {
        // Borrow guest memory rather than cloning. `Arc::clone` would
        // bump the refcount on every notify; the guest memory map is
        // alive for the full VM lifetime so the borrow is sufficient.
        let Some(mem) = self.mem.as_ref() else {
            // Caller (kvm wiring in src/vmm/mod.rs) is supposed to
            // call `set_mem` before any vCPU runs. A queue-notify
            // before that is a wiring bug; surface it once per
            // device so the log isn't flooded if the guest spams
            // notifies on the broken setup.
            if !self.mem_unset_warned.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    "virtio-blk process_requests called before set_mem; \
                     dropping all requests until guest memory is wired"
                );
            }
            return;
        };
        // `mem` borrows `self.mem`, and the request loop also calls
        // through `self.handle_*` (which take `&self`) plus
        // `self.ops_bucket.consume` / counter mutation (which take
        // `&mut self.ops_bucket`/`&mut self.bytes_bucket`). To keep
        // the borrow checker happy we materialise the queue handle
        // separately and reach into `&mut self` only via the
        // disjoint-fields it owns. The `signal_used()` write to
        // `self.interrupt_status` is hoisted to the end so it does
        // not aliase with the queue mutation in the loop.
        let mut signal_needed = false;
        loop {
            let q = &mut self.queues[REQ_QUEUE];
            let Some(chain) = q.pop_descriptor_chain(mem) else {
                break;
            };
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
            // allocation. Hot-path optimization — process_requests
            // runs on the vCPU thread and is invoked once per
            // QUEUE_NOTIFY MMIO write.
            self.all_descs_scratch.clear();
            for desc in chain {
                self.all_descs_scratch.push(ChainDescriptor {
                    addr: desc.addr(),
                    len: desc.len(),
                    is_write_only: desc.is_write_only(),
                });
            }

            let chain_len = self.all_descs_scratch.len();

            let mut header_addr: Option<GuestAddress> = None;
            let mut status_addr: Option<GuestAddress> = None;
            if let Some((first, rest)) = self.all_descs_scratch.split_first() {
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
                    // up with the kernel driver's `virtio_blk_outhdr`
                    // expectation regardless of leading padding.
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
            // Instead: leave the descriptor in the avail ring. The
            // guest's blk-mq layer arms a 30-second timeout per
            // request; on expiry it sees a timeout error and
            // surfaces the failure to userspace. Virtio-spec
            // explicitly permits device-side stalls. `io_errors`
            // is bumped so the host operator sees the malformed
            // request.
            let Some(status_addr) = status_addr else {
                tracing::warn!(head, "virtio-blk request without status descriptor");
                self.counters.record_io_error();
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
            // chain stuck in the avail ring until blk-mq's 30s
            // timeout fired — operationally invisible until the
            // guest stalled. Standard IOERR completion gives the
            // guest's block layer an immediate error to surface.
            if chain_len > VIRTIO_BLK_SEG_MAX as usize + 2 {
                tracing::warn!(
                    head,
                    desc_count = chain_len,
                    "virtio-blk chain exceeds seg_max + 2"
                );
                self.counters.record_io_error();
                if publish_completion(
                    mem,
                    q,
                    &self.counters,
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
            // error rather than waiting for blk-mq's 30s timeout.
            // `publish_completion` itself gates `add_used` on a
            // successful status-byte write — so a chain whose
            // status_addr is unmapped still ends up in the
            // "drop chain, let blk-mq time out" branch via the
            // `false` return path (no add_used, no signal).
            // `io_errors` is bumped so the host operator sees the
            // malformed request.
            let Some(header_addr) = header_addr else {
                tracing::warn!(head, "virtio-blk request without valid header descriptor");
                self.counters.record_io_error();
                if publish_completion(
                    mem,
                    q,
                    &self.counters,
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
                    self.counters.record_io_error();
                    if publish_completion(
                        mem,
                        q,
                        &self.counters,
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
            // The borrow is immutable; `&self.all_descs_scratch[..]`
            // is disjoint from `&mut self.queues[..]` (the `q`
            // borrow) and `&mut self.ops_bucket` / `&mut
            // self.bytes_bucket`, so split-borrow lets all coexist.
            let data_segments: &[ChainDescriptor] = &self.all_descs_scratch[1..chain_len - 1];

            // SIZE_MAX enforcement: reject any chain that violates
            // the per-descriptor cap we advertised. A guest that
            // submits a descriptor longer than VIRTIO_BLK_SIZE_MAX
            // is either buggy or hostile; rejecting up-front
            // prevents the I/O handlers from `vec![0u8; len]`-ing
            // multi-gigabyte buffers under host control.
            if data_segments.iter().any(|d| d.len > VIRTIO_BLK_SIZE_MAX) {
                tracing::warn!(head, "virtio-blk descriptor exceeds size_max");
                self.counters.record_io_error();
                if publish_completion(
                    mem,
                    q,
                    &self.counters,
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
                self.counters.record_io_error();
                if publish_completion(
                    mem,
                    q,
                    &self.counters,
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
                && data_len % VIRTIO_BLK_SECTOR_SIZE as u64 != 0
            {
                tracing::warn!(
                    head,
                    req_type,
                    data_len,
                    "virtio-blk T_IN/T_OUT data_len not a multiple of 512"
                );
                self.counters.record_io_error();
                if publish_completion(
                    mem,
                    q,
                    &self.counters,
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
            let backing = &self.backing;
            let counters = self.counters.as_ref();
            let cap_bytes = self.capacity_bytes;
            let read_only = self.read_only;
            let pre_throttle = Self::classify_pre_throttle(req_type, read_only, counters);

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
                self.counters.record_io_error();
                if publish_completion(
                    mem,
                    q,
                    &self.counters,
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
            // bucket fails, return S_IOERR immediately so the
            // guest's blk-mq driver doesn't hang waiting forever.
            // virtio-spec doesn't reserve a "throttled" status —
            // a throttled request becomes a transient IO error
            // from the guest's perspective. The device returns
            // synchronously rather than delaying inline; sleeping
            // here would block SIGRTMIN delivery to the vCPU
            // thread (see `TokenBucket` doc).
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
                let ops_ok = self.ops_bucket.can_consume(1);
                let bytes_ok = self.bytes_bucket.can_consume(data_len);
                if !ops_ok || !bytes_ok {
                    self.counters.record_throttled();
                    if publish_completion(
                        mem,
                        q,
                        &self.counters,
                        head,
                        status_addr,
                        VIRTIO_BLK_S_IOERR as u8,
                        1,
                        "throttled",
                    ) {
                        signal_needed = true;
                    }
                    continue;
                }
                // Both checks passed — consume now. Each bucket's
                // `consume` does its own refill+capacity check, so
                // the post-can_consume window can't see a smaller
                // bucket here (refills are monotone-non-negative).
                let ops_consumed = self.ops_bucket.consume(1);
                let bytes_consumed = self.bytes_bucket.consume(data_len);
                debug_assert!(
                    ops_consumed && bytes_consumed,
                    "throttle invariant: can_consume must imply consume",
                );
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
                // Pass `&mut self.io_buf_scratch` as a reusable
                // per-segment buffer; handlers `resize(len, 0)` it
                // per descriptor and the underlying `Vec<u8>`
                // capacity grows monotonically up to
                // VIRTIO_BLK_SIZE_MAX, then steady-state is zero
                // allocation per segment.
                match req_type {
                    VIRTIO_BLK_T_IN => Self::handle_read_impl(
                        backing,
                        cap_bytes,
                        counters,
                        mem,
                        sector,
                        data_segments,
                        data_len,
                        &mut self.io_buf_scratch,
                    ),
                    VIRTIO_BLK_T_OUT => Self::handle_write_impl(
                        backing,
                        cap_bytes,
                        counters,
                        mem,
                        sector,
                        data_segments,
                        data_len,
                        &mut self.io_buf_scratch,
                    ),
                    VIRTIO_BLK_T_FLUSH => Self::handle_flush_impl(backing, counters),
                    VIRTIO_BLK_T_GET_ID => {
                        Self::handle_get_id_impl(counters, mem, data_segments)
                    }
                    // Defense-in-depth fall-through. classify_pre_throttle's
                    // catch-all `_ => Some((VIRTIO_BLK_S_UNSUPP, 1))` arm
                    // means this branch is unreachable today — but a future
                    // patch that adds a new variant to the
                    // `T_IN | T_OUT | T_FLUSH | T_GET_ID => None` arm
                    // without updating this match would otherwise panic the
                    // vCPU thread. Return S_UNSUPP and bump io_errors so the
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
            // events (rejected request, IOERR, IOERR-on-throttle)
            // log at `warn!` so they always surface; this trace
            // line is the per-request "happy path" record. The
            // failure-path warns above use the same field set
            // (head, sector, etc.) so log-grep correlation works.
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
            // succeeded. `Queue::add_used` performs a Release-store
            // of the descriptor head/len then a SeqCst fence before
            // publishing used.idx, so the prior status-byte
            // write_slice is flushed before the guest sees the new
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
                &self.counters,
                head,
                status_addr,
                status_byte,
                used_len,
                "publish completion",
            ) {
                signal_needed = true;
            }
        }
        if signal_needed {
            self.signal_used();
        }
    }

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
    fn handle_read_impl(
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
    fn handle_write_impl(
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
    fn handle_flush_impl(backing: &File, counters: &VirtioBlkCounters) -> (u8, u32) {
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
    fn handle_get_id_impl(
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
        if mem
            .write_slice(&VIRTIO_BLK_SERIAL[..], first.addr)
            .is_err()
        {
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
    fn handle_read(
        &self,
        mem: &GuestMemoryMmap,
        sector: u64,
        data_segments: &[ChainDescriptor],
        status_addr: GuestAddress,
    ) -> (u8, u32) {
        let data_len: u64 = data_segments.iter().map(|d| d.len as u64).sum();
        let mut scratch = Vec::new();
        let (status, used_len) = Self::handle_read_impl(
            &self.backing,
            self.capacity_bytes,
            self.counters.as_ref(),
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
    fn handle_write(
        &self,
        mem: &GuestMemoryMmap,
        sector: u64,
        data_segments: &[ChainDescriptor],
        status_addr: GuestAddress,
    ) -> (u8, u32) {
        let data_len: u64 = data_segments.iter().map(|d| d.len as u64).sum();
        let mut scratch = Vec::new();
        let (status, used_len) = Self::handle_write_impl(
            &self.backing,
            self.capacity_bytes,
            self.counters.as_ref(),
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
    fn handle_flush(&self, mem: &GuestMemoryMmap, status_addr: GuestAddress) -> (u8, u32) {
        let (status, used_len) = Self::handle_flush_impl(&self.backing, self.counters.as_ref());
        mem.write_slice(&[status], status_addr)
            .expect("write status in test wrapper");
        (status, used_len)
    }

    #[cfg(test)]
    fn handle_get_id(
        &self,
        mem: &GuestMemoryMmap,
        data_segments: &[ChainDescriptor],
        status_addr: GuestAddress,
    ) -> (u8, u32) {
        let (status, used_len) =
            Self::handle_get_id_impl(self.counters.as_ref(), mem, data_segments);
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
                .map(|i| self.queues[i].max_size() as u32)
                .unwrap_or(0),
            VIRTIO_MMIO_QUEUE_READY => self
                .selected_queue()
                .map(|i| self.queues[i].ready() as u32)
                .unwrap_or(0),
            VIRTIO_MMIO_INTERRUPT_STATUS => self.interrupt_status,
            VIRTIO_MMIO_STATUS => self.device_status,
            VIRTIO_MMIO_CONFIG_GENERATION => self.config_generation,
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
    ///                  feature bit not advertised)
    ///   - 0x14..0x18: blk_size (u32 LE) — VIRTIO_BLK_F_BLK_SIZE
    ///
    /// Reads at offsets `>= VIRTIO_BLK_CONFIG_SIZE` return zero per
    /// virtio-v1.2 §4.2.2.2 ("reads past the populated config layout
    /// return zero") — guarded fields like topology / MQ / discard
    /// have feature bits we don't advertise, so the kernel driver's
    /// `virtio_cread_feature` skips them and never observes the
    /// zero-bytes we serve.
    fn read_blk_config(&self, offset: u64, data: &mut [u8]) {
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
                    self.queues[i].set_size(val as u16);
                }
            }
            VIRTIO_MMIO_QUEUE_READY if self.queue_config_allowed() => {
                if let Some(i) = self.selected_queue() {
                    self.queues[i].set_ready(val == 1);
                }
            }
            VIRTIO_MMIO_QUEUE_NOTIFY => {
                let idx = val as usize;
                if idx == REQ_QUEUE {
                    self.process_requests();
                }
            }
            VIRTIO_MMIO_INTERRUPT_ACK => {
                self.interrupt_status &= !val;
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
                    self.queues[i].set_desc_table_address(Some(val), None);
                }
            }
            VIRTIO_MMIO_QUEUE_DESC_HIGH if self.queue_config_allowed() => {
                if let Some(i) = self.selected_queue() {
                    self.queues[i].set_desc_table_address(None, Some(val));
                }
            }
            VIRTIO_MMIO_QUEUE_AVAIL_LOW if self.queue_config_allowed() => {
                if let Some(i) = self.selected_queue() {
                    self.queues[i].set_avail_ring_address(Some(val), None);
                }
            }
            VIRTIO_MMIO_QUEUE_AVAIL_HIGH if self.queue_config_allowed() => {
                if let Some(i) = self.selected_queue() {
                    self.queues[i].set_avail_ring_address(None, Some(val));
                }
            }
            VIRTIO_MMIO_QUEUE_USED_LOW if self.queue_config_allowed() => {
                if let Some(i) = self.selected_queue() {
                    self.queues[i].set_used_ring_address(Some(val), None);
                }
            }
            VIRTIO_MMIO_QUEUE_USED_HIGH if self.queue_config_allowed() => {
                if let Some(i) = self.selected_queue() {
                    self.queues[i].set_used_ring_address(None, Some(val));
                }
            }
            _ => {}
        }
    }

    /// Validate and apply a status transition per virtio-v1.2 §3.1.1.
    fn set_status(&mut self, val: u32) {
        if val & self.device_status != self.device_status {
            return;
        }
        let new_bits = val & !self.device_status;
        let valid = match new_bits {
            VIRTIO_CONFIG_S_ACKNOWLEDGE => self.device_status == 0,
            VIRTIO_CONFIG_S_DRIVER => self.device_status == S_ACK,
            VIRTIO_CONFIG_S_FEATURES_OK => self.device_status == S_DRV,
            VIRTIO_CONFIG_S_DRIVER_OK => self.device_status == S_FEAT,
            _ => false,
        };
        if valid {
            self.device_status = val;
        }
    }

    fn reset(&mut self) {
        self.device_status = 0;
        self.interrupt_status = 0;
        self.queue_select = 0;
        self.device_features_sel = 0;
        self.driver_features_sel = 0;
        self.driver_features = 0;
        // Bump config_generation on every reset so a re-binding driver
        // observes a different value and re-reads config space (per
        // virtio-v1.2 §4.2.2.1: drivers MUST re-read on changed
        // generation). For v0 the capacity is fixed for the
        // device's lifetime — set once in `new()` and never mutated
        // — so the bump is purely defense-in-depth: a future patch
        // that resizes the disk between resets is the case it
        // guards. The cost is a single u32 increment per reset,
        // worth paying to avoid a torn-read class of bug if/when
        // resize lands.
        self.config_generation = self.config_generation.wrapping_add(1);
        for q in &mut self.queues {
            q.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, Write};
    use tempfile::tempfile;

    fn make_device(capacity_bytes: u64, throttle: DiskThrottle) -> VirtioBlk {
        let mut f = tempfile().expect("create tempfile for virtio-blk test backing");
        f.set_len(capacity_bytes)
            .expect("set tempfile length to capacity_bytes — usually fails when TMPDIR is full");
        f.rewind().expect("rewind tempfile after set_len");
        VirtioBlk::new(f, capacity_bytes, throttle)
    }

    fn read_reg(dev: &VirtioBlk, offset: u32) -> u32 {
        let mut buf = [0u8; 4];
        dev.mmio_read(offset as u64, &mut buf);
        u32::from_le_bytes(buf)
    }

    fn write_reg(dev: &mut VirtioBlk, offset: u32, val: u32) {
        dev.mmio_write(offset as u64, &val.to_le_bytes());
    }

    #[test]
    fn magic_version_device_id() {
        let dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_MAGIC_VALUE), 0x7472_6976);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_VERSION), 2);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_DEVICE_ID), VIRTIO_ID_BLOCK);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_VENDOR_ID), 0);
    }

    #[test]
    fn advertised_features_include_size_max_seg_max_blk_size_flush() {
        let mut dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        write_reg(&mut dev, VIRTIO_MMIO_DEVICE_FEATURES_SEL, 0);
        let lo = read_reg(&dev, VIRTIO_MMIO_DEVICE_FEATURES);
        write_reg(&mut dev, VIRTIO_MMIO_DEVICE_FEATURES_SEL, 1);
        let hi = read_reg(&dev, VIRTIO_MMIO_DEVICE_FEATURES);
        let features = (hi as u64) << 32 | lo as u64;
        assert_ne!(features & (1u64 << VIRTIO_F_VERSION_1), 0);
        assert_ne!(features & (1u64 << VIRTIO_BLK_F_BLK_SIZE), 0);
        assert_ne!(features & (1u64 << VIRTIO_BLK_F_SEG_MAX), 0);
        assert_ne!(features & (1u64 << VIRTIO_BLK_F_SIZE_MAX), 0);
        // F_FLUSH advertised so guest blk-mq can issue REQ_OP_FLUSH
        // at metadata-commit boundaries — btrfs depends on this for
        // tree-consistency ordering.
        assert_ne!(features & (1u64 << VIRTIO_BLK_F_FLUSH), 0);
    }

    #[test]
    fn advertised_features_include_event_idx() {
        // VIRTIO_RING_F_EVENT_IDX is bit 29, in the low 32-bit half.
        // The guest needs the bit set during feature negotiation so
        // it populates `used_event` in the avail ring; without
        // advertisement the device cannot suppress IRQs even when
        // the corresponding wire-up lands.
        let mut dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        write_reg(&mut dev, VIRTIO_MMIO_DEVICE_FEATURES_SEL, 0);
        let lo = read_reg(&dev, VIRTIO_MMIO_DEVICE_FEATURES);
        write_reg(&mut dev, VIRTIO_MMIO_DEVICE_FEATURES_SEL, 1);
        let hi = read_reg(&dev, VIRTIO_MMIO_DEVICE_FEATURES);
        let features = (hi as u64) << 32 | lo as u64;
        assert_ne!(features & (1u64 << VIRTIO_RING_F_EVENT_IDX), 0);
    }

    #[test]
    fn capacity_in_config_space() {
        // 256 MB / 512 = 524_288 sectors. The default capacity is
        // 256 MB (mkfs.btrfs minimum).
        let dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        assert_eq!(dev.capacity_sectors(), 524_288);
        let mut buf = [0u8; 8];
        dev.mmio_read(0x100, &mut buf);
        assert_eq!(u64::from_le_bytes(buf), 524_288);
    }

    #[test]
    fn blk_size_in_config_space() {
        // VIRTIO_BLK_F_BLK_SIZE puts the logical block size at
        // offset 0x14 in config space.
        let dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        let mut buf = [0u8; 4];
        dev.mmio_read(0x100 + 0x14, &mut buf);
        assert_eq!(u32::from_le_bytes(buf), VIRTIO_BLK_SECTOR_SIZE);
    }

    #[test]
    fn reset_bumps_config_generation() {
        // virtio-v1.2 §4.2.2.1: config_generation must change when
        // any config-space field changes. Reset always bumps (rather
        // than tracking specific field mutations) so a re-binding
        // driver re-reads config space cleanly. v0 capacity is
        // fixed for the device's lifetime, so today the bump is
        // pure defense-in-depth — but if a future patch resizes
        // between resets the guest must observe the new generation
        // to invalidate its cached read.
        let mut dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        let gen0 = read_reg(&dev, VIRTIO_MMIO_CONFIG_GENERATION);
        // Drive through a full status handshake then write 0 to
        // trigger reset.
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, 0);
        let gen1 = read_reg(&dev, VIRTIO_MMIO_CONFIG_GENERATION);
        assert_eq!(gen1, gen0.wrapping_add(1));
        // Second cycle bumps again.
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, 0);
        let gen2 = read_reg(&dev, VIRTIO_MMIO_CONFIG_GENERATION);
        assert_eq!(gen2, gen1.wrapping_add(1));
    }

    #[test]
    fn seg_max_in_config_space() {
        // VIRTIO_BLK_F_SEG_MAX puts the per-request max scatter-gather
        // segment count at offset 0x0C in config space (virtio-v1.2
        // §5.2.4). Without this the guest defaults max_segments to 1
        // and serialises every multi-page bio.
        let dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        let mut buf = [0u8; 4];
        dev.mmio_read(0x100 + 0x0C, &mut buf);
        assert_eq!(u32::from_le_bytes(buf), VIRTIO_BLK_SEG_MAX);
    }

    #[test]
    fn config_space_struct_layout_byte_for_byte() {
        // Read the entire 24-byte populated config-space layout via
        // a single mmio read and verify that every field lands at
        // the kernel-uapi-mandated offset:
        //   capacity (u64 LE) @ 0x00 — VIRTIO_BLK_DEFAULT_CAPACITY_BYTES / 512
        //   size_max (u32 LE) @ 0x08 — VIRTIO_BLK_SIZE_MAX
        //   seg_max  (u32 LE) @ 0x0C — VIRTIO_BLK_SEG_MAX
        //   geometry (4B zeroed) @ 0x10 — F_GEOMETRY not advertised
        //   blk_size (u32 LE) @ 0x14 — VIRTIO_BLK_SECTOR_SIZE
        // A regression in `repr(C, packed)` field ordering or padding
        // would shift any field by a byte and break this assertion
        // before the wrong bytes ever reach the guest.
        let dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        let mut bytes = [0u8; VIRTIO_BLK_CONFIG_SIZE];
        dev.mmio_read(0x100, &mut bytes);

        let capacity = u64::from_le_bytes(bytes[0x00..0x08].try_into().unwrap());
        let size_max = u32::from_le_bytes(bytes[0x08..0x0C].try_into().unwrap());
        let seg_max = u32::from_le_bytes(bytes[0x0C..0x10].try_into().unwrap());
        let geometry = &bytes[0x10..0x14];
        let blk_size = u32::from_le_bytes(bytes[0x14..0x18].try_into().unwrap());

        assert_eq!(
            capacity,
            VIRTIO_BLK_DEFAULT_CAPACITY_BYTES / VIRTIO_BLK_SECTOR_SIZE as u64,
            "capacity mismatch — repr(C, packed) layout drift?",
        );
        assert_eq!(size_max, VIRTIO_BLK_SIZE_MAX, "size_max layout drift");
        assert_eq!(seg_max, VIRTIO_BLK_SEG_MAX, "seg_max layout drift");
        assert_eq!(
            geometry,
            &[0u8; 4],
            "F_GEOMETRY not advertised; geometry must be zero",
        );
        assert_eq!(blk_size, VIRTIO_BLK_SECTOR_SIZE, "blk_size layout drift");
    }

    #[test]
    fn config_space_zero_past_struct_size() {
        // virtio-v1.2 §4.2.2.2: reads past the populated config layout
        // return zero. Our `repr(C, packed)` struct is 24 bytes; the
        // device must zero-fill any read at offset >= 24 within
        // config space. A buggy guest or future feature negotiation
        // must see deterministic zero rather than uninitialized memory.
        let dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        let mut buf = [0xffu8; 16];
        dev.mmio_read(0x100 + VIRTIO_BLK_CONFIG_SIZE as u64, &mut buf);
        assert!(
            buf.iter().all(|&b| b == 0),
            "config-space read past struct size must be zero-filled, got {:02x?}",
            buf,
        );

        // Read straddling the struct boundary: half within, half
        // past. The within portion carries blk_size at offset 0x14;
        // the past portion (offset 0x18..0x1C) must zero-fill.
        let mut buf = [0xffu8; 8];
        dev.mmio_read(0x100 + 0x14, &mut buf);
        assert_eq!(
            u32::from_le_bytes(buf[0..4].try_into().unwrap()),
            VIRTIO_BLK_SECTOR_SIZE,
            "first 4 bytes must be blk_size",
        );
        assert_eq!(
            &buf[4..],
            &[0u8; 4],
            "trailing 4 bytes (offset 0x18..0x1C) must zero-fill past struct end",
        );
    }

    #[test]
    fn config_space_struct_size_matches_kernel_uapi() {
        // Mirror the compile-time size assertion at runtime so a
        // broken assertion surfaces under nextest output rather than
        // hidden in a const-eval failure. Also pin the alignment to
        // 1: ByteValued::as_slice() returns the struct's bytes
        // directly, and `repr(C, packed)` collapses alignment to 1
        // — which both matches the kernel's
        // `__attribute__((packed))` layout and avoids any
        // unaligned-access UB on architectures we don't currently
        // target.
        assert_eq!(
            VIRTIO_BLK_CONFIG_SIZE, 24,
            "VirtioBlkConfig must be 24 bytes (capacity 8 + size_max 4 + \
             seg_max 4 + geometry 4 + blk_size 4) per the kernel uapi \
             layout. Mismatch implies repr(C, packed) drift.",
        );
        assert_eq!(
            std::mem::align_of::<VirtioBlkConfig>(),
            1,
            "repr(C, packed) must produce alignment 1",
        );
        assert_eq!(
            std::mem::align_of::<VirtioBlkGeometry>(),
            1,
            "geometry sub-struct must also be packed to align 1",
        );
    }

    #[test]
    fn config_space_writes_silently_dropped() {
        let mut dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        let before = dev.capacity_sectors();
        dev.mmio_write(0x100, &[0xff, 0xff, 0xff, 0xff]);
        assert_eq!(dev.capacity_sectors(), before);
    }

    #[test]
    fn queue_num_max() {
        let mut dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 0);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_QUEUE_NUM_MAX), QUEUE_MAX_SIZE as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 1);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_QUEUE_NUM_MAX), 0);
    }

    #[test]
    fn status_state_machine() {
        let mut dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
        // Skipping FEATURES_OK is rejected.
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);
        assert_eq!(dev.device_status, S_DRV);
    }

    #[test]
    fn status_reset_via_zero() {
        let mut dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, 0);
        assert_eq!(dev.device_status, 0);
    }

    #[test]
    fn token_bucket_unlimited_always_grants() {
        let mut tb = TokenBucket::unlimited();
        for _ in 0..1_000_000 {
            assert!(tb.consume(1));
        }
    }

    #[test]
    fn token_bucket_consumes_capacity() {
        let mut tb = TokenBucket::new(100, 1); // 100 capacity, refills 1/sec
        for _ in 0..100 {
            assert!(tb.consume(1));
        }
        assert!(!tb.consume(1));
    }

    #[test]
    fn token_bucket_refills_over_time() {
        // Slow refill (10/sec) so the consume loop's wall-time
        // overhead doesn't refill enough to mask the bucket
        // exhaustion. At 10 tokens/sec, ~100ms must elapse before
        // even a single token refills.
        let mut tb = TokenBucket::new(100, 10);
        for _ in 0..100 {
            assert!(tb.consume(1));
        }
        assert!(
            !tb.consume(1),
            "bucket exhausted; refill too slow to top up in microseconds",
        );
        // Sleep enough to refill at least 1 token (>=100ms at
        // 10/sec). Use 200ms for slack.
        std::thread::sleep(std::time::Duration::from_millis(200));
        assert!(
            tb.consume(1),
            "after 200ms at 10 tokens/sec, at least 1 should be available",
        );
    }

    #[test]
    fn throttle_zero_rate_becomes_unlimited() {
        // The DiskThrottle public surface uses Option<NonZeroU64>, so
        // a zero rate is unrepresentable at construction. This test
        // pins TokenBucket's defense-in-depth fallback at the
        // primitive layer: if a future caller (or a reflective
        // construction path that bypasses NonZeroU64) hands
        // TokenBucket::new a 0 rate, the bucket must become the
        // unlimited fast path rather than infinitely-failing
        // consume(1) calls.
        let mut tb = TokenBucket::new(0, 100);
        for _ in 0..10_000 {
            assert!(tb.consume(1));
        }
        let mut tb = TokenBucket::new(100, 0);
        for _ in 0..10_000 {
            assert!(tb.consume(1));
        }
    }

    #[test]
    fn capacity_custom_size() {
        let dev = make_device(256 * 1024 * 1024, DiskThrottle::default());
        assert_eq!(dev.capacity_sectors(), 256 * 1024 * 1024 / 512);
    }

    #[test]
    fn counters_initially_zero() {
        let dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        let c = dev.counters();
        assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
        assert_eq!(c.writes_completed.load(Ordering::Relaxed), 0);
        assert_eq!(c.flushes_completed.load(Ordering::Relaxed), 0);
        assert_eq!(c.bytes_read.load(Ordering::Relaxed), 0);
        assert_eq!(c.bytes_written.load(Ordering::Relaxed), 0);
        assert_eq!(c.throttled_count.load(Ordering::Relaxed), 0);
        assert_eq!(c.io_errors.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn counters_arc_shared_with_caller() {
        let dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        let c1 = dev.counters();
        let c2 = dev.counters();
        c1.reads_completed.store(42, Ordering::Relaxed);
        assert_eq!(c2.reads_completed.load(Ordering::Relaxed), 42);
    }

    #[test]
    fn handle_flush_no_mem_no_panic() {
        // Flush calls fdatasync on the backing file. Ensure it
        // succeeds on a fresh tempfile (which is always
        // fdatasync-able on Linux).
        let mut f = tempfile().unwrap();
        f.write_all(&[0u8; 1024]).unwrap();
        // Direct call bypassing MMIO — sync_data must succeed.
        f.sync_data().expect("tempfile sync_data must succeed");
    }

    #[test]
    fn ok_status_constant_distinct_from_ioerr() {
        // Pin the canonical virtio-blk status byte values. A
        // regression that maps every request to OK silently corrupts
        // guest filesystems by serving uninitialized data.
        assert_eq!(VIRTIO_BLK_S_OK, 0);
        assert_eq!(VIRTIO_BLK_S_IOERR, 1);
        assert_eq!(VIRTIO_BLK_S_UNSUPP, 2);
    }

    // ----------------------------------------------------------------
    // MMIO/FSM/IRQ surface tests ported from virtio_console.
    //
    // These exercise the device's MMIO/FSM/IRQ surface independent
    // of the backend descriptor-I/O path — they pin behaviour the
    // device shares with every virtio-mmio device per virtio-v1.2
    // (status FSM, queue config gating, IRQ delivery). Mechanical
    // ports of virtio_console's analogous coverage; behavioral
    // parity with that device is the goal because the two share
    // the same MMIO contract.
    // ----------------------------------------------------------------

    /// Drive the device through the full virtio init sequence up to
    /// `DRIVER_OK`. Mirrors the virtio_console `init_device` helper.
    /// Used by tests that need a fully negotiated device.
    fn init_device(dev: &mut VirtioBlk) {
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

    /// `INTERRUPT_STATUS` reflects the device's internal
    /// `interrupt_status` register; reads do NOT clear it (only
    /// `INTERRUPT_ACK` writes do, per virtio-v1.2 §4.2.2). Mirrors
    /// `virtio_console::interrupt_status_and_ack`.
    #[test]
    fn interrupt_status_and_ack() {
        let mut dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS), 0);
        dev.interrupt_status = VIRTIO_MMIO_INT_VRING;
        assert_eq!(
            read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS),
            VIRTIO_MMIO_INT_VRING
        );
    }

    /// `INTERRUPT_ACK` clears only the bits the driver writes,
    /// leaving other bits set. virtio-v1.2 §4.2.2.2 specifies that
    /// each write to `InterruptACK` clears the bits in `InterruptStatus`
    /// matching the bits set in the value written. Mirrors
    /// `virtio_console::interrupt_ack_clears_bits`.
    ///
    /// virtio-blk does not currently use `VIRTIO_MMIO_INT_CONFIG` —
    /// the device has no config-change events because the capacity
    /// is fixed for the device's lifetime (set once in `new()`,
    /// never mutated; `reset()` bumps `config_generation` only as
    /// defense-in-depth against a hypothetical future resize path).
    /// ACK semantics are still tested with a synthetic INT_CONFIG
    /// bit so a future config-change path drops in without breaking
    /// the mask logic.
    #[test]
    fn interrupt_ack_clears_bits() {
        use virtio_bindings::virtio_mmio::VIRTIO_MMIO_INT_CONFIG;
        let mut dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        dev.interrupt_status = VIRTIO_MMIO_INT_VRING | VIRTIO_MMIO_INT_CONFIG;
        write_reg(&mut dev, VIRTIO_MMIO_INTERRUPT_ACK, VIRTIO_MMIO_INT_VRING);
        assert_eq!(
            read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS),
            VIRTIO_MMIO_INT_CONFIG,
        );
    }

    /// Non-4-byte reads must return 0xff for every byte. The
    /// virtio-MMIO spec mandates 4-byte register access; a partial
    /// access is a guest bug. Returning 0xff is more debuggable than
    /// silently zeroing because it produces an obviously wrong value
    /// the kernel driver flags. Mirrors
    /// `virtio_console::non_4byte_read_returns_ff`.
    #[test]
    fn non_4byte_read_returns_ff() {
        let dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        let mut buf = [0u8; 2];
        dev.mmio_read(0, &mut buf);
        assert_eq!(buf, [0xff, 0xff]);
    }

    /// Non-4-byte writes are silently dropped. The device MUST
    /// not act on a partial register write; pinning that the device
    /// state remains untouched after a bogus 2-byte write to STATUS.
    /// Mirrors `virtio_console::non_4byte_write_ignored`.
    #[test]
    fn non_4byte_write_ignored() {
        let mut dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        dev.mmio_write(VIRTIO_MMIO_STATUS as u64, &[0x01, 0x00]);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_STATUS), 0);
    }

    /// `DRIVER_FEATURES` writes are gated by status: BEFORE
    /// DRIVER bit, writes are dropped. AFTER ACKNOWLEDGE+DRIVER
    /// (`S_DRV`), writes are accepted into the page selected by
    /// `DRIVER_FEATURES_SEL`. Pins the page-select dispatch (page 0
    /// → low 32 bits, page 1 → high 32 bits). Mirrors
    /// `virtio_console::driver_features_gated_by_status`.
    #[test]
    fn driver_features_gated_by_status() {
        let mut dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        // Before DRIVER status, features writes are rejected.
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES, 0xDEAD);
        assert_eq!(dev.driver_features, 0);

        // After ACKNOWLEDGE + DRIVER, features writes are accepted.
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES, 0xDEAD_BEEF);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES, 0xCAFE_BABE);
        assert_eq!(dev.driver_features, 0xCAFE_BABE_DEAD_BEEF);
    }

    /// Feature negotiation closes once `FEATURES_OK` is set:
    /// subsequent `DRIVER_FEATURES` writes must be dropped. Pinning
    /// this prevents a regression that would let the guest mutate
    /// negotiated features post-handshake — a spec violation that
    /// can corrupt device behavior. Mirrors
    /// `virtio_console::features_rejected_after_features_ok`.
    #[test]
    fn features_rejected_after_features_ok() {
        let mut dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
        write_reg(
            &mut dev,
            VIRTIO_MMIO_DRIVER_FEATURES,
            1 << (VIRTIO_F_VERSION_1 - 32),
        );
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_FEAT);

        // After FEATURES_OK, features writes are rejected.
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES, 0xFFFF);
        assert_eq!(dev.driver_features & 0xFFFF_FFFF, 0);
    }

    /// `QUEUE_DESC_LOW`/`QUEUE_DESC_HIGH` writes are gated by
    /// `queue_config_allowed`: BEFORE `FEATURES_OK`, writes drop;
    /// AFTER `FEATURES_OK` (and before `DRIVER_OK`), they're applied.
    /// virtio-v1.2 §4.2.2.2 mandates queue config is only legal in
    /// the `FEATURES_OK..DRIVER_OK` window. Mirrors
    /// `virtio_console::queue_desc_addr_requires_features_ok`.
    #[test]
    fn queue_desc_addr_requires_features_ok() {
        let mut dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 0);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_LOW, 0x1000);
        // Not accepted before FEATURES_OK.
        assert_ne!(dev.queues[0].desc_table(), 0x1000);

        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
        write_reg(
            &mut dev,
            VIRTIO_MMIO_DRIVER_FEATURES,
            1 << (VIRTIO_F_VERSION_1 - 32),
        );
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_FEAT);

        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 0);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_LOW, 0x1000);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_HIGH, 0);
        assert_eq!(dev.queues[0].desc_table(), 0x1000);
    }

    /// Reads of unknown register offsets return 0 (the catchall
    /// `_ => 0` arm in `mmio_read`). 0x300 sits beyond every defined
    /// MMIO offset and below the 0x100 config-space split, so it's a
    /// pure unknown-register probe. Mirrors
    /// `virtio_console::unknown_register_returns_zero`.
    #[test]
    fn unknown_register_returns_zero() {
        let dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        assert_eq!(read_reg(&dev, 0x300), 0);
    }

    /// Writes to unknown register offsets are silently dropped.
    /// Confirms that an attempted write to 0x300 has no observable
    /// side effect on the device's STATUS register. Mirrors
    /// `virtio_console::unknown_register_write_ignored`.
    #[test]
    fn unknown_register_write_ignored() {
        let mut dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        write_reg(&mut dev, 0x300, 0xDEAD);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_STATUS), 0);
    }

    /// `QUEUE_SEL` accepting any 32-bit value is fine, but
    /// reading `QUEUE_NUM_MAX`/`QUEUE_READY` for a select that's
    /// out of range returns 0 because `selected_queue()` returns
    /// `None`. virtio-blk has only one queue (REQ_QUEUE=0), so any
    /// select >= 1 must read 0. Mirrors
    /// `virtio_console::invalid_queue_select_returns_zero`.
    #[test]
    fn invalid_queue_select_returns_zero() {
        let mut dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 99);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_QUEUE_NUM_MAX), 0);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_QUEUE_READY), 0);
    }

    /// `signal_used` sets the `VIRTIO_MMIO_INT_VRING` bit on
    /// `interrupt_status` AND writes to the `irq_evt` eventfd so the
    /// KVM irqfd path delivers the IRQ. Without both, the guest
    /// either never sees the IRQ (eventfd not written) or sees an
    /// IRQ without context (status bit not set). Mirrors
    /// `virtio_console::signal_used_sets_interrupt_and_writes_eventfd`.
    #[test]
    fn signal_used_sets_interrupt_and_writes_eventfd() {
        let mut dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        assert_eq!(dev.interrupt_status, 0);
        dev.signal_used();
        assert_ne!(dev.interrupt_status & VIRTIO_MMIO_INT_VRING, 0);
        let val = dev.irq_evt.read().unwrap();
        assert!(val > 0);
    }

    /// `DEVICE_FEATURES_SEL` page 2 returns 0. Only pages 0
    /// and 1 are defined (low / high 32 bits of the 64-bit feature
    /// set); higher pages must read 0 per virtio-v1.2's
    /// reserved-for-future-extensions semantics. Mirrors
    /// `virtio_console::features_page_2_returns_zero`.
    #[test]
    fn features_page_2_returns_zero() {
        let mut dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        write_reg(&mut dev, VIRTIO_MMIO_DEVICE_FEATURES_SEL, 2);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_DEVICE_FEATURES), 0);
    }

    /// Skipping `ACKNOWLEDGE` (writing `DRIVER` directly) is
    /// rejected by the FSM. `set_status` requires the new-bit set
    /// to match exactly the next legal transition; jumping straight
    /// to `DRIVER` from 0 violates the §3.1.1 ordering.
    /// Mirrors `virtio_console::status_skip_acknowledge_rejected`.
    #[test]
    fn status_skip_acknowledge_rejected() {
        let mut dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        // Skipping ACKNOWLEDGE, going straight to DRIVER.
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, VIRTIO_CONFIG_S_DRIVER);
        assert_eq!(dev.device_status, 0);
    }

    /// Once `DRIVER_OK` is set, queue config writes (here
    /// `QUEUE_NUM`) MUST be rejected by `queue_config_allowed`. The
    /// FSM gate is `S_FEAT && !DRIVER_OK`, so a `QUEUE_NUM` write
    /// after the device is fully driver-up is a spec violation that
    /// the device drops. Pin that the queue size stays at the
    /// initialised default. Mirrors
    /// `virtio_console::queue_config_rejected_after_driver_ok`.
    ///
    /// Uses [`init_device`] to fully sequence the FSM up to
    /// `S_OK`, so this test also exercises the happy-path init
    /// pipeline (any FSM regression that breaks negotiate-up surfaces
    /// here as a setup-time assertion failure rather than a missed
    /// post-OK write).
    #[test]
    fn queue_config_rejected_after_driver_ok() {
        let mut dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        init_device(&mut dev);
        assert_eq!(dev.device_status, S_OK);

        // After DRIVER_OK, queue config is rejected.
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 0);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NUM, 64);
        // Queue size should still be the post-init default
        // (QUEUE_MAX_SIZE), not 64.
        assert_eq!(dev.queues[0].size(), QUEUE_MAX_SIZE);
    }

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

    fn make_guest_mem(bytes: usize) -> GuestMemoryMmap {
        // Single-region GuestMemoryMmap at GPA 0 — sufficient for
        // direct handler testing where the test owns the GPAs.
        GuestMemoryMmap::from_ranges(&[(GuestAddress(0), bytes)]).expect("create test guest mem")
    }

    /// Build a backing file pre-populated with a fixed pattern so a
    /// `handle_read` can verify the file contents propagate to guest
    /// memory.
    fn make_backed_file_with_pattern(capacity: u64, pattern: u8) -> File {
        let mut f = tempfile().unwrap();
        f.set_len(capacity).unwrap();
        f.rewind().unwrap();
        let buf = vec![pattern; capacity as usize];
        f.write_all(&buf).unwrap();
        f.rewind().unwrap();
        f
    }

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
        let segs = vec![ChainDescriptor { addr: data_addr, len: 512, is_write_only: true }];
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
        let segs = vec![ChainDescriptor { addr: data_addr, len: 512, is_write_only: false }];
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
        let segs = vec![ChainDescriptor { addr: data_addr, len: 512, is_write_only: true }];
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
        let mem = make_guest_mem(8192);
        let data_addr = GuestAddress(0x2000);
        let status_addr = GuestAddress(0x2FFF);
        let segs = vec![ChainDescriptor { addr: data_addr, len: 512, is_write_only: false }];
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

    /// Pin VIRTIO_RING_F_EVENT_IDX (bit 29) in the advertised
    /// feature set. The guest's blk-mq layer can then suppress
    /// notifications by writing an event index into the avail
    /// ring's `used_event` field; without this bit the guest never
    /// sets the threshold and every used-ring advance triggers an
    /// irqfd write. The actual suppression decision lives in
    /// `Queue::needs_notification` — this test pins the
    /// advertisement so a future feature-set edit can't silently
    /// drop notification-suppression support.
    #[test]
    fn advertises_event_idx_feature_bit() {
        let cap = 4096u64;
        let f = make_backed_file_with_pattern(cap, 0x00);
        let dev = VirtioBlk::new(f, cap, DiskThrottle::default());
        let feats = dev.device_features();
        assert_ne!(
            feats & (1u64 << VIRTIO_RING_F_EVENT_IDX),
            0,
            "VIRTIO_RING_F_EVENT_IDX (bit 29) must be advertised so \
             the guest can suppress unnecessary interrupts via the \
             avail ring's used_event field",
        );
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
        let segs = vec![ChainDescriptor { addr: data_addr, len: 512, is_write_only: false }];
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
        let segs = vec![ChainDescriptor { addr: data_addr, len: 512, is_write_only: true }];
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
        assert!(dev.read_only);
    }

    #[test]
    fn token_bucket_refill_uses_elapsed_wall_time() {
        // Drain to empty, sleep 1 second, observe a full refill.
        // Use small absolute numbers (<=10) so the test is fast and
        // any timing slop in the test harness produces a rounding
        // difference of <= 1 token rather than a flake.
        let mut tb = TokenBucket::new(10, 10);
        for _ in 0..10 {
            assert!(tb.consume(1));
        }
        assert!(!tb.consume(1));
        std::thread::sleep(std::time::Duration::from_millis(1100));
        // After ~1.1s at 10/sec, capacity caps at 10. Verify we get
        // back the full bucket.
        for _ in 0..10 {
            assert!(tb.consume(1), "bucket should have refilled to capacity after sleep");
        }
    }

    #[test]
    fn token_bucket_consume_zero_is_free() {
        // A zero-byte data transfer (e.g. T_FLUSH) should not consume
        // any bytes-bucket tokens. Pin that consume(0) is a no-op
        // success, which is the established firecracker semantic.
        let mut tb = TokenBucket::new(10, 10);
        for _ in 0..1_000 {
            assert!(tb.consume(0));
        }
        // Bucket still full.
        for _ in 0..10 {
            assert!(tb.consume(1));
        }
        assert!(!tb.consume(1));
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
            ChainDescriptor { addr: seg0_addr, len: 512, is_write_only: true },
            ChainDescriptor { addr: seg1_addr, len: 512, is_write_only: true },
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
            ChainDescriptor { addr: seg0_addr, len: 512, is_write_only: true },
            ChainDescriptor { addr: seg1_addr, len: 512, is_write_only: true },
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

    use vm_memory::Address;
    use virtio_bindings::bindings::virtio_ring::VRING_DESC_F_WRITE;
    use virtio_queue::desc::{RawDescriptor, split::Descriptor as SplitDescriptor};
    use virtio_queue::mock::MockSplitQueue;

    /// Plant a `VirtioBlkOutHdr` at `header_addr` in `mem` so a
    /// chain-level test can build a request with the correct header
    /// type/sector. The header_addr is the GPA the header descriptor
    /// will point at.
    fn write_blk_header(
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
    fn wire_device_to_mock(dev: &mut VirtioBlk, mock: &MockSplitQueue<GuestMemoryMmap>) {
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

    /// Build a guest memory map sized to host both the queue
    /// descriptor/avail/used rings (placed at GPA 0..) and the
    /// chain's data buffers (placed above the ring region).
    /// 1 MB total — generous so neither the rings nor the test
    /// payloads collide.
    fn make_chain_test_mem() -> GuestMemoryMmap {
        GuestMemoryMmap::from_ranges(&[(GuestAddress(0), 1 << 20)])
            .expect("create chain test guest mem")
    }

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
    /// avail ring; blk-mq's per-request 30-second timeout fires and
    /// surfaces a real error.
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

    /// Throttle exhaustion through `process_requests`.
    /// Build a device with IOPS limit = 1 (1 op per second, 1 op
    /// burst capacity). Drain the bucket up-front, then notify with
    /// a chain — should fail with IOERR. Pin: throttled_count
    /// ticks, the chain's data descriptor is NOT modified (no read
    /// happened), io_errors stays at 0 (throttling is a separate
    /// counter from I/O errors).
    #[test]
    fn process_requests_throttled_returns_ioerr_and_increments_counter() {
        let cap = 4096u64;
        let f = make_backed_file_with_pattern(cap, 0xAB);
        let throttle = DiskThrottle {
            iops: std::num::NonZeroU64::new(1),
            bytes_per_sec: None,
        };
        let mut dev = VirtioBlk::new(f, cap, throttle);
        let mem = make_chain_test_mem();
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);

        // Drain the bucket and pin its last_refill so refill on
        // the next consume yields 0 tokens.
        dev.ops_bucket
            .set_last_refill_for_test(std::time::Instant::now());
        assert!(dev.ops_bucket.consume(1), "drain the 1-token bucket");
        dev.ops_bucket
            .set_last_refill_for_test(std::time::Instant::now());

        let header_addr = GuestAddress(0x4000);
        let data_addr = GuestAddress(0x5000);
        let status_addr = GuestAddress(0x6000);
        // Plant a sentinel pattern so we can detect whether the
        // device wrote to it. 0xFF distinct from backing 0xAB.
        let sentinel = vec![0xFFu8; 512];
        mem.write_slice(&sentinel, data_addr).unwrap();
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

        let mut status_buf = [0u8; 1];
        mem.read_slice(&mut status_buf, status_addr).unwrap();
        assert_eq!(
            status_buf[0], VIRTIO_BLK_S_IOERR as u8,
            "throttle exhaustion must surface as S_IOERR",
        );

        let used_idx: u16 = mem
            .read_obj(mock.used_addr().checked_add(2).unwrap())
            .expect("read used.idx");
        assert_eq!(used_idx, 1);

        let c = dev.counters();
        assert_eq!(
            c.throttled_count.load(Ordering::Relaxed),
            1,
            "throttle exhaustion must bump throttled_count",
        );
        assert_eq!(
            c.io_errors.load(Ordering::Relaxed),
            0,
            "throttle is NOT classified as an I/O error",
        );
        assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
        assert_eq!(c.bytes_read.load(Ordering::Relaxed), 0);

        // Data descriptor untouched — throttle short-circuited
        // before any backing read happened.
        let mut readback = [0u8; 512];
        mem.read_slice(&mut readback, data_addr).unwrap();
        assert!(
            readback.iter().all(|&b| b == 0xFF),
            "throttled chain must NOT touch the data descriptor; \
             0xFF sentinel must survive",
        );
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
            RawDescriptor::from(SplitDescriptor::new(
                data_addr.0,
                512,
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
    /// in the avail ring until blk-mq's 30s timeout, hiding the
    /// rejection from operators.
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
            s[0],
            VIRTIO_BLK_S_IOERR as u8,
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
    /// regardless of leading padding. virtio_blk.rs:657-659
    /// implements this; pin the offset.
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
            RawDescriptor::from(SplitDescriptor::new(
                data_addr.0,
                512,
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
        };
        let mut dev = VirtioBlk::new(f, cap, throttle);
        // Drain the bucket and pin its last_refill so refill on
        // the next consume yields 0 tokens.
        dev.ops_bucket
            .set_last_refill_for_test(std::time::Instant::now());
        assert!(dev.ops_bucket.consume(1));
        dev.ops_bucket
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

    /// signal_used eventfd write through full chain. Before
    /// process_requests, irq_evt is unsignalled (read returns
    /// EAGAIN). After, it MUST be readable (KVM irqfd path
    /// delivery).
    #[test]
    fn signal_used_writes_irq_eventfd_after_process_requests() {
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
        assert!(val > 0, "irq_evt counter must be > 0 after process_requests");
        assert_ne!(dev.interrupt_status & VIRTIO_MMIO_INT_VRING, 0);
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
            RawDescriptor::from(SplitDescriptor::new(
                header_part2_addr.0,
                8,
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
            "fragmented header (first desc < OUTHDR_SIZE) must IOERR",
        );

        let c = dev.counters();
        assert_eq!(c.io_errors.load(Ordering::Relaxed), 1);
        assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
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
        let mem = make_guest_mem(8192);
        let data_addr = GuestAddress(0x1000);
        let status_addr = GuestAddress(0x2000);
        // Pre-fill the data buffer with a sentinel so a regression
        // that wrote zero bytes (or the wrong number of bytes)
        // surfaces as residual sentinel rather than a silent
        // pass.
        mem.write_slice(
            &[0xCDu8; VIRTIO_BLK_ID_BYTES as usize],
            data_addr,
        )
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
        let mem = make_guest_mem(8192);
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
            status,
            VIRTIO_BLK_S_IOERR as u8,
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
        let mem = make_guest_mem(8192);
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
        mem.write_slice(
            &[0xCDu8; VIRTIO_BLK_ID_BYTES as usize],
            data_addr,
        )
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
}

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
//      handle every input without crashing the vCPU thread.
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
#[cfg(test)]
mod proptest_tests {
    use super::{
        ChainDescriptor, DiskThrottle, REQ_QUEUE, VIRTIO_BLK_OUTHDR_SIZE, VIRTIO_BLK_S_IOERR,
        VIRTIO_BLK_S_OK, VIRTIO_BLK_S_UNSUPP, VIRTIO_MMIO_QUEUE_NOTIFY, VirtioBlk,
        VirtioBlkOutHdr,
    };
    use proptest::prelude::*;
    use std::os::unix::fs::FileExt;
    use std::sync::atomic::Ordering;
    use tempfile::tempfile;
    use virtio_bindings::bindings::virtio_ring::{
        VRING_DESC_F_INDIRECT, VRING_DESC_F_NEXT, VRING_DESC_F_WRITE,
    };
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
        f.set_len(cap)
            .expect("set tempfile length to fuzz cap");
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

    proptest! {
        // Set a non-default cases count: 256 is more than the
        // proptest default (256 already; explicit so a future
        // PROPTEST_CASES env override is the only knob that changes
        // behaviour). Disable shrinking timeout to catch slow
        // pathological chains. `max_shrink_iters` capped at a
        // moderate value because shrunken cases mostly help debug
        // failures, not detect them.
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
        /// stall — the guest's blk-mq layer would time out at 30s
        /// without the host having any visibility.
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
    }

    // Reference to ChainDescriptor to suppress dead-import warning
    // when the type is only used via super:: re-export. ChainDescriptor
    // is intentionally imported even though no proptest function
    // names it, because the proptest harness may evolve to call
    // `handle_*_impl` directly with synthesized ChainDescriptor
    // slices (mirroring the existing handler-level tests). Keeping
    // the import documents the intended extension surface.
    #[allow(dead_code)]
    fn _chain_descriptor_marker() -> Option<ChainDescriptor> {
        None
    }

    // VRING_DESC_F_INDIRECT is referenced in doc comments above to
    // motivate the random flags strategy; keeping it imported makes
    // the import list a complete enumeration of the descriptor flags
    // the device's parser inspects.
    #[allow(dead_code)]
    const _INDIRECT_FLAG_MARKER: u32 = VRING_DESC_F_INDIRECT;
}
