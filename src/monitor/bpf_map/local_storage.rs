//! Host-side walker for `bpf_local_storage_map` chains.
//!
//! Covers `BPF_MAP_TYPE_TASK_STORAGE` and the shape-identical
//! `INODE_STORAGE` / `SK_STORAGE` / `CGRP_STORAGE` variants. All four
//! kernel map types share the
//! `bpf_local_storage_map`/`bpf_local_storage_map_bucket`/
//! `bpf_local_storage_elem`/`bpf_local_storage_data`/
//! `bpf_local_storage` layout, so one walker plus
//! [`super::super::btf_offsets::TaskStorageOffsets`] handles them all
//! — the only per-type difference is what `bpf_local_storage.owner`
//! points at (task_struct vs inode vs sock vs cgroup), which the
//! walker treats as opaque.
//!
//! Walk shape (verified against the kernel source under
//! `kernel/bpf/bpf_local_storage.c`):
//!
//! 1. Read `bpf_local_storage_map.bucket_log` and
//!    `bpf_local_storage_map.buckets`. The bucket count is
//!    `1 << bucket_log`.
//! 2. For each `i in 0..nbuckets`, walk the regular hlist starting
//!    at `buckets[i].list.first`. NULL terminates the chain (no
//!    LSB-tagged sentinel — this is `hlist_head`, not
//!    `hlist_nulls_head`).
//! 3. The chain links via `bpf_local_storage_elem.map_node`. That
//!    field is at offset 0 of the elem (asserted at BTF resolve
//!    time in [`super::super::btf_offsets::resolve_task_storage_offsets`]),
//!    so each chain `node_kva` IS the elem KVA.
//! 4. For each elem: copy `value_size` bytes from
//!    `elem + elem_sdata + sdata_data` (the cacheline-aligned
//!    `sdata.data[]` flex array) and follow `elem.local_storage`
//!    (RCU pointer to `bpf_local_storage`) to read `owner` for the
//!    task_struct/inode/sock/cgroup KVA.
//! 5. Advance `node_ptr = elem + hlist_node_next` (since
//!    `map_node` is at offset 0, the link is at the elem base + the
//!    `next` offset within `hlist_node`).
//!
//! Hostile-input handling: untranslatable buckets are skipped (the
//! per-bucket chain breaks); untranslatable elems break the current
//! chain. A null `local_storage` pointer surfaces as `owner=0` rather
//! than dropping the entry — the value bytes are still useful even
//! when the owner identity is unrecoverable.

use super::super::idr::translate_any_kva;
use super::{
    AccessorCtx, BPF_MAP_TYPE_CGRP_STORAGE, BPF_MAP_TYPE_INODE_STORAGE, BPF_MAP_TYPE_SK_STORAGE,
    BPF_MAP_TYPE_TASK_STORAGE, BpfMapInfo,
};

/// Maximum number of buckets walked when iterating a
/// `bpf_local_storage_map`.
///
/// Production maps size buckets to
/// `roundup_pow_of_two(num_possible_cpus())`, so even a 4096-CPU
/// machine produces only 12 levels of `bucket_log`. This 16-bit cap
/// is a hostile-guest safety bound: a corrupted (uninitialized) u32
/// read of `bucket_log` could yield up to 31, which would otherwise
/// attempt to walk 2^31 buckets on the freeze hot path.
const TASK_STORAGE_BUCKETS_MAX: u32 = 1 << 16;

/// Maximum total selem visits across all buckets.
///
/// Mirrors the `HTAB_ITER_MAX` cycle defense in the bpf_htab walker:
/// a corrupted `next` pointer that loops back into the same chain
/// would otherwise hang the freeze hot path until the rendezvous
/// timeout.
const TASK_STORAGE_ITER_MAX: usize = 1_000_000;

/// Iterate every `bpf_local_storage_elem` registered against `map`.
///
/// Returns `(owner_kva_le_bytes, value_bytes)` per entry:
/// - `owner_kva_le_bytes` is the 8-byte little-endian encoding of
///   `bpf_local_storage.owner` reached by following each
///   `bpf_local_storage_elem.local_storage`. For TASK_STORAGE this
///   is the `task_struct` KVA; for INODE/SK/CGRP_STORAGE it is the
///   corresponding owner KVA. The walker treats it as opaque so the
///   same shape works for all four variants.
/// - `value_bytes` is `map.value_size` bytes copied from
///   `bpf_local_storage_elem.sdata.data[]`.
///
/// Returns an empty vec when:
/// - the map type is not one of the four local-storage variants;
/// - [`TaskStorageOffsets`] is unavailable (kernel BTF lacks the
///   local-storage subsystem types);
/// - the map's `buckets` pointer is null (map allocation failed
///   between create and freeze);
/// - the bucket count exceeds [`TASK_STORAGE_BUCKETS_MAX`] (corrupted
///   `bucket_log`).
///
/// Untranslatable buckets and elems are skipped — the corresponding
/// chain breaks but the walk continues into the next bucket.
///
/// Per-element short-read policy: a `read_bytes` call returning fewer
/// bytes than `value_size` drops the entire entry rather than
/// surfacing a partially-zeroed value buffer. Mirrors the
/// [`super::htab::iter_htab_entries`] walker's policy; the renderer
/// never sees mixed guest-data / scratch bytes for any element.
pub(super) fn iter_local_storage_entries(
    ctx: &AccessorCtx<'_>,
    map: &BpfMapInfo,
) -> Vec<(Vec<u8>, Vec<u8>)> {
    if map.map_type != BPF_MAP_TYPE_TASK_STORAGE
        && map.map_type != BPF_MAP_TYPE_INODE_STORAGE
        && map.map_type != BPF_MAP_TYPE_SK_STORAGE
        && map.map_type != BPF_MAP_TYPE_CGRP_STORAGE
    {
        return Vec::new();
    }
    let Some(ts) = &ctx.offsets.task_storage_offsets else {
        return Vec::new();
    };

    // bpf_local_storage_map embeds bpf_map at offset 0
    // (include/linux/bpf_local_storage.h struct bpf_local_storage_map
    // declaration), so map_kva == smap_kva.
    let smap_kva = map.map_kva;
    let Some(smap_pa) = translate_any_kva(
        ctx.mem,
        ctx.cr3_pa.0,
        ctx.page_offset.0,
        smap_kva,
        ctx.l5,
        ctx.tcr_el1,
    ) else {
        return Vec::new();
    };

    let bucket_log = ctx.mem.read_u32(smap_pa, ts.smap_bucket_log);
    let buckets_kva = ctx.mem.read_u64(smap_pa, ts.smap_buckets);
    if buckets_kva == 0 {
        return Vec::new();
    }
    // bucket_log is a u32 in BTF but the kernel only ever stores
    // `ilog2(roundup_pow_of_two(num_possible_cpus()))`. Cap the
    // bucket count so a corrupted (uninitialized) read can't induce
    // a 2^31-iteration walk on the freeze hot path. checked_shl
    // returns None when the shift is >= 32, which we also treat as
    // out-of-range. Log on the corrupt path so operators can
    // distinguish "map is empty" from "walker disabled by corrupt
    // bucket_log."
    let n_buckets = 1u32.checked_shl(bucket_log).unwrap_or(0);
    if n_buckets == 0 || n_buckets > TASK_STORAGE_BUCKETS_MAX {
        tracing::debug!(
            map_name = %map.name(),
            bucket_log,
            n_buckets,
            cap = TASK_STORAGE_BUCKETS_MAX,
            "local_storage walker: out-of-range bucket_log, returning empty"
        );
        return Vec::new();
    }

    let value_size = map.value_size as usize;
    if value_size > super::MAX_VALUE_SIZE {
        return Vec::new();
    }
    let value_off_in_elem = ts.elem_sdata + ts.sdata_data;

    let mut out = Vec::new();
    let mut total_visited = 0usize;

    // Per-walk owner cache: every selem links to a
    // `bpf_local_storage` whose `owner` field identifies the owning
    // task/cgroup/inode/sock. A single owner can appear behind many
    // selems (a task with N local-storage maps under it has N
    // selems all pointing at the same `bpf_local_storage`). Caching
    // `local_storage_kva -> owner_kva` for the duration of one walk
    // eliminates the redundant `translate_any_kva` page-table walk
    // and the redundant `read_u64(ls_pa, ts.ls_owner)` for repeat
    // owners. The cache is dropped at function return so a
    // subsequent dump rebuilds it from the freshly-frozen guest.
    let mut owner_cache: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();

    for i in 0..n_buckets {
        let bucket_kva = buckets_kva + (i as u64) * (ts.bucket_size as u64);
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
            .read_u64(bucket_pa, ts.bucket_list + ts.hlist_head_first);

        let mut node_ptr = first_ptr;
        loop {
            // Regular hlist (not hlist_nulls): NULL terminates.
            if node_ptr == 0 {
                break;
            }
            total_visited += 1;
            if total_visited > TASK_STORAGE_ITER_MAX {
                return out;
            }

            // node_ptr addresses bpf_local_storage_elem.map_node, and
            // map_node sits at offset 0 of bpf_local_storage_elem (the
            // BTF resolver asserts this), so the elem KVA equals the
            // node KVA — no container_of subtraction.
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

            // Owner KVA: elem.local_storage -> bpf_local_storage.owner.
            // A null local_storage pointer surfaces as owner=0 so the
            // value bytes still reach the consumer.
            let local_storage_kva = ctx.mem.read_u64(elem_pa, ts.elem_local_storage);
            let owner_kva = if local_storage_kva == 0 {
                0
            } else if let Some(&cached) = owner_cache.get(&local_storage_kva) {
                cached
            } else {
                let resolved = match translate_any_kva(
                    ctx.mem,
                    ctx.cr3_pa.0,
                    ctx.page_offset.0,
                    local_storage_kva,
                    ctx.l5,
                    ctx.tcr_el1,
                ) {
                    Some(ls_pa) => ctx.mem.read_u64(ls_pa, ts.ls_owner),
                    None => 0,
                };
                owner_cache.insert(local_storage_kva, resolved);
                resolved
            };

            // Skip the `vec![0u8; value_size]` zero-fill — every byte
            // is overwritten by `read_bytes` below; a short read drops
            // the entry to avoid handing a partial buffer to the
            // renderer.
            let mut val_buf: Vec<u8> = Vec::with_capacity(value_size);
            // SAFETY: capacity == value_size; we set_len only after
            // confirming `read_bytes` filled the requested length.
            let slice = unsafe { std::slice::from_raw_parts_mut(val_buf.as_mut_ptr(), value_size) };
            let n = ctx
                .mem
                .read_bytes(elem_pa + value_off_in_elem as u64, slice);
            if n == value_size {
                // SAFETY: `n == value_size`, so every byte in
                // `0..value_size` of the backing storage was written.
                unsafe {
                    val_buf.set_len(value_size);
                }
                out.push((owner_kva.to_le_bytes().to_vec(), val_buf));
            }

            // Advance via the hlist link. With map_node at offset 0
            // of the elem, the chain `next` pointer is at
            // `elem + hlist_node_next`.
            node_ptr = ctx.mem.read_u64(elem_pa, ts.hlist_node_next);
        }
    }

    out
}
