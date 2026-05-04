//! BTF offsets for `BPF_MAP_TYPE_RINGBUF` / `BPF_MAP_TYPE_USER_RINGBUF`
//! and `BPF_MAP_TYPE_STACK_TRACE` diagnostic-map rendering.
//!
//! Both groups exist to surface read-only state from kernel BPF
//! infrastructure that the standard `read_value` / `iter_hash_map`
//! paths can't decode. Their walkers live next to each other in
//! `dump/render_map.rs`; their offset definitions live next to each
//! other here.
//!
//! Verified against the kernel source:
//! - `kernel/bpf/ringbuf.c` for `bpf_ringbuf_map` (line 82) and
//!   `bpf_ringbuf` (line 28). `rb->mask = data_sz - 1` set in
//!   `bpf_ringbuf_alloc` (line 185) — capacity is `mask + 1`.
//!   `bpf_ringbuf_map` is BTF-listed via `BTF_ID_LIST_SINGLE` at
//!   `kernel/bpf/ringbuf.c:377`, which forces emission of `bpf_ringbuf`
//!   as a referenced type into vmlinux BTF.
//! - `kernel/bpf/stackmap.c` for `bpf_stack_map` (line 26) and
//!   `stack_map_bucket` (line 19). `n_buckets =
//!   roundup_pow_of_two(max_entries)` per `stack_map_alloc:122`, so
//!   the iteration bound differs from the user-declared `max_entries`.

use anyhow::Result;
use btf_rs::Btf;

use super::{find_struct, member_byte_offset};

/// Byte offsets within `struct bpf_ringbuf_map` and `struct bpf_ringbuf`
/// (`kernel/bpf/ringbuf.c`) needed to surface ringbuf occupancy from
/// guest memory without walking the records themselves.
///
/// The map type itself stores only a pointer to the heap-allocated
/// `bpf_ringbuf` (`bpf_ringbuf_map.rb`); the consumer/producer positions
/// and the data-region mask live on that secondary struct. The dump path
/// reads `rb` from the bpf_map base, then dereferences it via
/// `translate_any_kva` to read the four position/mask fields.
///
/// Capacity is derived from `mask + 1` — see
/// `bpf_ringbuf_area_alloc` in `kernel/bpf/ringbuf.c` which sets
/// `rb->mask = data_sz - 1` for a power-of-two `data_sz`. Pending
/// bytes is `producer_pos - consumer_pos` (both monotonically advancing
/// 64-bit counters; the kernel uses unsigned wraparound subtraction
/// to compute occupancy in the dispatch path).
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct BpfRingbufOffsets {
    /// Offset of `rb` (`struct bpf_ringbuf *`) within
    /// `struct bpf_ringbuf_map`. Dereferenced to reach the position
    /// counters that follow.
    pub rbm_rb: usize,
    /// Offset of `mask` (`u64`) within `struct bpf_ringbuf`. Set as
    /// `data_sz - 1` in `bpf_ringbuf_alloc` where `data_sz` is the
    /// map's `max_entries` (a power-of-two byte count). Capacity in
    /// bytes is `mask + 1`. (`kernel/bpf/ringbuf.c::struct bpf_ringbuf`
    /// declares `u64 mask` directly, distinct from the `unsigned long`
    /// position fields below.)
    pub rb_mask: usize,
    /// Offset of `consumer_pos` (`unsigned long`) within
    /// `struct bpf_ringbuf`. Userspace updates this; the kernel only
    /// reads it. Bytes pending = `producer_pos - consumer_pos`.
    pub rb_consumer_pos: usize,
    /// Offset of `producer_pos` (`unsigned long`) within
    /// `struct bpf_ringbuf`. Kernel updates this on each
    /// `bpf_ringbuf_reserve`.
    pub rb_producer_pos: usize,
    /// Offset of `pending_pos` (`unsigned long`) within
    /// `struct bpf_ringbuf`. Tracks the oldest in-flight reservation —
    /// records committed beyond this point are visible to the
    /// consumer; records below `producer_pos` but above `pending_pos`
    /// are still being filled by a producer.
    pub rb_pending_pos: usize,
}

/// Resolve BTF offsets for `bpf_ringbuf_map` + `bpf_ringbuf`.
/// Returns `Err` if either type or any required field is missing.
pub(crate) fn resolve_ringbuf_offsets(btf: &Btf) -> Result<BpfRingbufOffsets> {
    let (rbm, _) = find_struct(btf, "bpf_ringbuf_map")?;
    let rbm_rb = member_byte_offset(btf, &rbm, "rb")?;

    let (rb, _) = find_struct(btf, "bpf_ringbuf")?;
    let rb_mask = member_byte_offset(btf, &rb, "mask")?;
    let rb_consumer_pos = member_byte_offset(btf, &rb, "consumer_pos")?;
    let rb_producer_pos = member_byte_offset(btf, &rb, "producer_pos")?;
    let rb_pending_pos = member_byte_offset(btf, &rb, "pending_pos")?;

    Ok(BpfRingbufOffsets {
        rbm_rb,
        rb_mask,
        rb_consumer_pos,
        rb_producer_pos,
        rb_pending_pos,
    })
}

/// Byte offsets within `struct bpf_stack_map` and `struct stack_map_bucket`
/// (`kernel/bpf/stackmap.c`) needed to enumerate stored stack traces
/// from guest memory.
///
/// `bpf_stack_map.buckets[]` is a flex array of `struct stack_map_bucket *`
/// indexed 0..n_buckets (where n_buckets = roundup_pow_of_two(max_entries),
/// see `stack_map_alloc`). A non-null slot points to a bucket whose
/// `data[]` flex array holds `nr` u64 program counters (or
/// `bpf_stack_build_id` records when `BPF_F_STACK_BUILD_ID` is set on
/// the map; the dump path treats both as opaque trace bytes).
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct BpfStackmapOffsets {
    /// Offset of `n_buckets` (u32) within `struct bpf_stack_map`.
    /// Use this for the iteration bound rather than `max_entries`,
    /// because the kernel rounds up to the next power of two.
    pub smap_n_buckets: usize,
    /// Offset of `buckets` flex array within `struct bpf_stack_map`.
    /// Each slot is `sizeof(void *)` (8 bytes on 64-bit).
    pub smap_buckets: usize,
    /// Offset of `nr` (u32) within `struct stack_map_bucket`. Records
    /// how many trace entries (PCs) are populated in `data[]`.
    pub smb_nr: usize,
    /// Offset of `data` flex array within `struct stack_map_bucket`.
    /// Each PC is 8 bytes when `BPF_F_STACK_BUILD_ID` is unset; for
    /// build-id stacks each entry is a `struct bpf_stack_build_id`
    /// (32 bytes per the BUILD_BUG_ON in `stack_map_alloc:107`).
    pub smb_data: usize,
}

/// Resolve BTF offsets for `bpf_stack_map` + `stack_map_bucket`.
/// Returns `Err` if either type or any required field is missing.
pub(crate) fn resolve_stackmap_offsets(btf: &Btf) -> Result<BpfStackmapOffsets> {
    let (smap, _) = find_struct(btf, "bpf_stack_map")?;
    let smap_n_buckets = member_byte_offset(btf, &smap, "n_buckets")?;
    let smap_buckets = member_byte_offset(btf, &smap, "buckets")?;

    let (smb, _) = find_struct(btf, "stack_map_bucket")?;
    let smb_nr = member_byte_offset(btf, &smb, "nr")?;
    let smb_data = member_byte_offset(btf, &smb, "data")?;

    Ok(BpfStackmapOffsets {
        smap_n_buckets,
        smap_buckets,
        smb_nr,
        smb_data,
    })
}
