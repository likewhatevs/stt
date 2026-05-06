//! Host-side walker for `BPF_MAP_TYPE_HASH` /
//! `BPF_MAP_TYPE_LRU_HASH` / `BPF_MAP_TYPE_PERCPU_HASH` /
//! `BPF_MAP_TYPE_LRU_PERCPU_HASH`.
//!
//! All four variants share the `bpf_htab.buckets` /
//! `hlist_nulls_node` / `htab_elem` layout. The only per-type
//! difference is what the value slot of `htab_elem` contains:
//! HASH/LRU_HASH store inline value bytes; PERCPU_HASH /
//! LRU_PERCPU_HASH store a `void __percpu *` pointer that resolves
//! to per-CPU storage via `__per_cpu_offset[cpu]`. [`walk_htab`]
//! centralizes the bucket-array translation and chain walk; the
//! [`iter_htab_entries`] / [`iter_percpu_htab_entries`] entry points
//! supply per-element extractors.

use super::super::idr::translate_any_kva;
use super::{
    AccessorCtx, BPF_MAP_TYPE_HASH, BPF_MAP_TYPE_LRU_HASH, BPF_MAP_TYPE_LRU_PERCPU_HASH,
    BPF_MAP_TYPE_PERCPU_HASH, BpfMapInfo,
};

/// Maximum number of entries to iterate when walking a hash map.
/// Prevents unbounded iteration on corrupted or very large maps.
pub(super) const HTAB_ITER_MAX: usize = 1_000_000;

/// Maximum number of buckets walked per hash map.
///
/// Production maps cap n_buckets at `roundup_pow_of_two(max_entries)`
/// where `max_entries` is bounded by the kernel's BPF_MAP_CREATE
/// validation. This 16-bit cap is a hostile-guest safety bound:
/// a corrupted (uninitialized) u32 read of `bpf_htab.n_buckets`
/// could yield up to `u32::MAX`, which would otherwise attempt to
/// walk billions of buckets on the freeze hot path. Mirror of the
/// matching `TASK_STORAGE_BUCKETS_MAX` cap in
/// `bpf_map::local_storage`.
pub(super) const HTAB_BUCKETS_MAX: u32 = 1 << 16;

/// Iterate all entries in a `BPF_MAP_TYPE_HASH` or `BPF_MAP_TYPE_LRU_HASH`
/// map, yielding (key, value) byte pairs.
///
/// `HASH` and `LRU_HASH` share the same `htab_elem` layout: the
/// `lru_node` field on LRU lives in the same union slot as
/// `ptr_to_pptr`, and the kernel resolves both via
/// `htab_elem_value(l, key_size) = l->key + round_up(key_size, 8)`
/// (`kernel/bpf/hashtab.c:185`). Inline value bytes start at that
/// offset for both map types, so the walker is identical.
///
/// Walks the `bpf_htab.buckets` array, following `hlist_nulls` chains
/// in each bucket to reach `htab_elem` entries. Key bytes start at the
/// end of `struct htab_elem` (the `key[]` flex array), and value bytes
/// follow at `round_up(key_size, 8)` from the key start.
///
/// `buckets` is allocated via `bpf_map_area_alloc` (vmalloc for large
/// allocations, kmalloc for small), so addresses are translated via
/// `translate_any_kva`. Element pointers within bucket chains are
/// SLAB-allocated (direct mapping) or from `bpf_mem_alloc`.
///
/// Returns an empty vec if the map is neither `BPF_MAP_TYPE_HASH` nor
/// `BPF_MAP_TYPE_LRU_HASH`, htab offsets are unavailable, or the htab
/// struct itself is untranslatable. Untranslatable buckets are skipped;
/// an untranslatable element breaks the current bucket's chain and
/// advances to the next bucket.
///
/// Per-element short-read policy: `read_bytes` may return fewer bytes
/// than requested when the resolved PA is near the end of a guest
/// memory region (multi-region NUMA layouts can have non-contiguous
/// host mappings). The walker drops the entire element rather than
/// hand back a partially-zeroed key or value — the renderer never
/// sees a `(Vec<u8>, Vec<u8>)` whose bytes are a mix of guest data
/// and uninitialized scratch. The dropped element is silently
/// omitted; the bucket walk continues to the next chain entry.
pub(super) fn iter_htab_entries(
    ctx: &AccessorCtx<'_>,
    map: &BpfMapInfo,
) -> Vec<(Vec<u8>, Vec<u8>)> {
    if map.map_type != BPF_MAP_TYPE_HASH && map.map_type != BPF_MAP_TYPE_LRU_HASH {
        return Vec::new();
    }
    walk_htab(
        ctx,
        map,
        |elem_pa, key_off_in_elem, value_off_in_elem, key_size, value_size, mem| {
            // `Vec::with_capacity` skips the zero-fill that
            // `vec![0u8; n]` would emit; every byte is overwritten by
            // `read_bytes`, so the scribbled zeros were dead writes.
            // `set_len` is gated on the read returning the requested
            // length (mismatch surfaces as a dropped element rather
            // than a partial buffer reaching the renderer).
            let mut key_buf: Vec<u8> = Vec::with_capacity(key_size);
            // SAFETY: capacity is `key_size`; `read_bytes` writes the
            // full slice when its return equals `key_size`, and we
            // only `set_len` after asserting that.
            let key_slice =
                unsafe { std::slice::from_raw_parts_mut(key_buf.as_mut_ptr(), key_size) };
            let kn = mem.read_bytes(elem_pa + key_off_in_elem as u64, key_slice);
            if kn != key_size {
                return None;
            }
            // SAFETY: `kn == key_size`, so every byte in `0..key_size`
            // of `key_buf`'s backing storage was written.
            unsafe {
                key_buf.set_len(key_size);
            }

            let mut val_buf: Vec<u8> = Vec::with_capacity(value_size);
            // SAFETY: same contract as the key buffer above.
            let val_slice =
                unsafe { std::slice::from_raw_parts_mut(val_buf.as_mut_ptr(), value_size) };
            let vn = mem.read_bytes(elem_pa + value_off_in_elem as u64, val_slice);
            if vn != value_size {
                return None;
            }
            // SAFETY: `vn == value_size`, mirror of the key buffer.
            unsafe {
                val_buf.set_len(value_size);
            }
            Some((key_buf, val_buf))
        },
    )
}

/// Iterate all entries in a `BPF_MAP_TYPE_PERCPU_HASH` or
/// `BPF_MAP_TYPE_LRU_PERCPU_HASH` map, yielding `(key, per_cpu_values)`.
///
/// PERCPU hash variants store a `void __percpu *` pointer at the
/// `htab_elem_value` position rather than inline bytes
/// (`kernel/bpf/hashtab.c:198` `htab_elem_get_ptr`). Each per-CPU
/// value is reached via `pptr + __per_cpu_offset[cpu]`, identical to
/// the `PERCPU_ARRAY` deref path in [`super::read_percpu_array_value`].
///
/// `per_cpu_values` is one entry per CPU indexed by CPU number.
/// `Some(bytes)` when that CPU's slot translates and reads; `None`
/// when the per-CPU page is unmapped, the CPU is out of range
/// (cpu_off==0 && cpu_index>0; same guard as
/// [`super::read_percpu_array_value`]), or `read_bytes` returns
/// fewer bytes than `value_size` (short-read drop, mirroring the
/// plain-HASH walker's per-element policy: never hand back a
/// partially-zeroed value).
///
/// The key buffer follows the same short-read drop policy as
/// [`iter_htab_entries`]: a short key read drops the whole entry
/// (not just one CPU slot) and the bucket walk advances to the
/// next chain entry.
pub(super) fn iter_percpu_htab_entries(
    ctx: &AccessorCtx<'_>,
    map: &BpfMapInfo,
    per_cpu_offsets: &[u64],
) -> super::PerCpuHashEntries {
    if map.map_type != BPF_MAP_TYPE_PERCPU_HASH && map.map_type != BPF_MAP_TYPE_LRU_PERCPU_HASH {
        return Vec::new();
    }
    let value_size = map.value_size as usize;
    walk_htab(
        ctx,
        map,
        |elem_pa, key_off_in_elem, value_off_in_elem, key_size, _value_size_unused, mem| {
            // Skip the `vec![0u8; key_size]` zero-fill — the
            // `read_bytes` call below overwrites the entire slice and
            // any short read drops the element rather than handing a
            // partial buffer back to the renderer.
            let mut key_buf: Vec<u8> = Vec::with_capacity(key_size);
            // SAFETY: capacity == key_size; we set_len only after
            // `read_bytes` returns key_size.
            let key_slice =
                unsafe { std::slice::from_raw_parts_mut(key_buf.as_mut_ptr(), key_size) };
            let kn = mem.read_bytes(elem_pa + key_off_in_elem as u64, key_slice);
            if kn != key_size {
                return None;
            }
            // SAFETY: read_bytes wrote `kn == key_size` bytes.
            unsafe {
                key_buf.set_len(key_size);
            }

            // The "value" slot in a PERCPU htab_elem holds a percpu
            // base pointer, not data. Same shape as bpf_array.pptrs[k]
            // for PERCPU_ARRAY.
            let percpu_base = mem.read_u64(elem_pa, value_off_in_elem);
            if percpu_base == 0 {
                return Some((key_buf, Vec::new()));
            }

            let mut per_cpu = Vec::with_capacity(per_cpu_offsets.len());
            for (cpu_index, &cpu_off) in per_cpu_offsets.iter().enumerate() {
                // Same out-of-range guard as `read_percpu_array_value`:
                // cpu_off==0 && cpu_index>0 means the kernel's
                // `__per_cpu_offset[cpu]` is BSS-zero (cpu beyond
                // `nr_cpu_ids`). Treat as unmapped to avoid aliasing
                // CPU 0.
                if cpu_off == 0 && cpu_index > 0 {
                    per_cpu.push(None);
                    continue;
                }
                let cpu_kva = percpu_base.wrapping_add(cpu_off);
                match translate_any_kva(
                    ctx.mem,
                    ctx.cr3_pa.0,
                    ctx.page_offset.0,
                    cpu_kva,
                    ctx.l5,
                    ctx.tcr_el1,
                ) {
                    // `checked_add` against a pathological cpu_pa
                    // + value_size that would wrap u64 — without
                    // the guard, a wrap would silently make
                    // `<= mem.size()` true and the read_bytes call
                    // would walk past end-of-DRAM.
                    Some(cpu_pa)
                        if cpu_pa
                            .checked_add(value_size as u64)
                            .is_some_and(|end| end <= ctx.mem.size()) =>
                    {
                        // Same with-capacity-then-set-len trick as
                        // the key buffer; a short read leaves the
                        // slot as `None` so the renderer never sees a
                        // partially-zeroed value.
                        let mut buf: Vec<u8> = Vec::with_capacity(value_size);
                        // SAFETY: capacity == value_size; gated by
                        // the read returning value_size.
                        let slice =
                            unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr(), value_size) };
                        let n = ctx.mem.read_bytes(cpu_pa, slice);
                        if n == value_size {
                            // SAFETY: read_bytes filled value_size bytes.
                            unsafe {
                                buf.set_len(value_size);
                            }
                            per_cpu.push(Some(buf));
                        } else {
                            per_cpu.push(None);
                        }
                    }
                    _ => per_cpu.push(None),
                }
            }
            Some((key_buf, per_cpu))
        },
    )
}

/// Shared bpf_htab bucket walker. Calls `extract` for every reachable
/// `htab_elem`, collecting whatever the closure returns.
///
/// Centralizes the bucket-array translation, hlist_nulls chain walk,
/// and the [`HTAB_ITER_MAX`] cap so plain-HASH and PERCPU-HASH
/// variants share one traversal — the only difference between them
/// is what the per-element extractor reads.
fn walk_htab<T, F>(ctx: &AccessorCtx<'_>, map: &BpfMapInfo, mut extract: F) -> Vec<T>
where
    F: FnMut(u64, usize, usize, usize, usize, &super::super::reader::GuestMem) -> Option<T>,
{
    let Some(htab) = &ctx.offsets.htab_offsets else {
        return Vec::new();
    };

    // bpf_htab embeds bpf_map at offset 0, so map_kva == htab_kva.
    let htab_kva = map.map_kva;

    let Some(htab_pa) = translate_any_kva(
        ctx.mem,
        ctx.cr3_pa.0,
        ctx.page_offset.0,
        htab_kva,
        ctx.l5,
        ctx.tcr_el1,
    ) else {
        return Vec::new();
    };
    let n_buckets = ctx.mem.read_u32(htab_pa, htab.htab_n_buckets);
    let buckets_kva = ctx.mem.read_u64(htab_pa, htab.htab_buckets);
    if n_buckets == 0 || n_buckets > HTAB_BUCKETS_MAX || buckets_kva == 0 {
        // n_buckets > HTAB_BUCKETS_MAX surfaces a corrupted
        // (uninitialized) read on the freeze hot path; bail
        // rather than walk billions of buckets.
        return Vec::new();
    }

    let key_size = map.key_size as usize;
    let value_size = map.value_size as usize;
    if key_size > super::MAX_VALUE_SIZE || value_size > super::MAX_VALUE_SIZE {
        return Vec::new();
    }
    // Value follows key at round_up(key_size, 8) within htab_elem.
    let value_off_in_elem = htab.htab_elem_size_base + ((key_size + 7) & !7);
    let key_off_in_elem = htab.htab_elem_size_base;

    let mut out = Vec::new();
    let mut total_visited = 0usize;

    for i in 0..n_buckets {
        let bucket_kva = buckets_kva + (i as u64) * (htab.bucket_size as u64);
        let Some(bucket_pa) = translate_any_kva(
            ctx.mem,
            ctx.cr3_pa.0,
            ctx.page_offset.0,
            bucket_kva,
            ctx.l5,
            ctx.tcr_el1,
        ) else {
            continue;
        };

        let first_ptr = ctx
            .mem
            .read_u64(bucket_pa, htab.bucket_head + htab.hlist_nulls_head_first);

        let mut node_ptr = first_ptr;
        loop {
            if node_ptr & 1 != 0 || node_ptr == 0 {
                break;
            }
            total_visited += 1;
            if total_visited > HTAB_ITER_MAX {
                return out;
            }

            // node_ptr == KVA of the hlist_nulls_node == htab_elem
            // (hash_node first in the union).
            let elem_kva = node_ptr;
            let Some(elem_pa) = translate_any_kva(
                ctx.mem,
                ctx.cr3_pa.0,
                ctx.page_offset.0,
                elem_kva,
                ctx.l5,
                ctx.tcr_el1,
            ) else {
                break;
            };

            if let Some(item) = extract(
                elem_pa,
                key_off_in_elem,
                value_off_in_elem,
                key_size,
                value_size,
                ctx.mem,
            ) {
                out.push(item);
            }

            node_ptr = ctx.mem.read_u64(elem_pa, htab.hlist_nulls_node_next);
        }
    }

    out
}
