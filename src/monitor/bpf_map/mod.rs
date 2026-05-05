//! Host-side BPF map discovery, read/write, and iteration via guest physical memory.
//!
//! Walks the kernel's `map_idr` xarray from the host, finds a BPF map
//! by name suffix, and provides read/write access to the map's value
//! region. No guest cooperation is needed — all reads go through the
//! guest physical memory mapping.
//!
//! Address translation strategy:
//! - `map_idr` is a kernel BSS symbol: use `text_kva_to_pa_with_base`
//!   (or [`super::guest::GuestKernel::text_kva_to_pa`]).
//! - xa_node structs are SLAB-allocated (direct mapping): use `kva_to_pa`.
//! - bpf_map/bpf_array may be kmalloc'd or vmalloc'd: use `translate_any_kva`.
//! - .bss value region is vmalloc'd: use `translate_kva`.
//! - Per-CPU values (`BPF_MAP_TYPE_PERCPU_ARRAY`) are in the direct mapping:
//!   use `kva_to_pa` with `__per_cpu_offset[cpu]`.

use super::btf_offsets::BpfMapOffsets;
use super::idr::{translate_any_kva, xa_load};
use super::reader::GuestMem;
use super::symbols::text_kva_to_pa_with_base;
use super::{Cr3Pa, Kva, PageOffset};

mod htab;
mod local_storage;
#[cfg(test)]
mod tests;
use htab::{iter_htab_entries, iter_percpu_htab_entries};
use local_storage::iter_local_storage_entries;

/// Per-element row from a percpu-hash iteration: `(key_bytes,
/// per_cpu_values)` where `per_cpu_values[cpu]` is `Some(value_bytes)`
/// when the per-CPU slot is readable and `None` when the page is
/// unmapped or the CPU index is out of range. Returned by
/// [`BpfMapAccessor::iter_percpu_hash_map`] and the underlying walker
/// helpers in [`htab`].
pub(crate) type PerCpuHashEntries = Vec<(Vec<u8>, Vec<Option<Vec<u8>>>)>;

/// Bundle of borrow-held state every map-access routine threads
/// through the page-table walk, bounds check, and byte read/write path.
///
/// Every free function in this module previously took the same four-
/// to eight-argument fan of `mem`, `cr3_pa`, `page_offset`, `offsets`,
/// `l5` (some also took `map_idr_kva`); callers invariably forwarded
/// the same fields from their [`GuestMemMapAccessor`] because all six
/// originate on the accessor. Grouping them here drops the duplication
/// and lets additional shared context (per-CPU offset cache, BTF
/// cache, etc.) ride the same lifetime without touching every
/// signature. `cr3_pa` and `page_offset` are newtyped so the page-
/// walker can't silently swap them at a call site.
pub(crate) struct AccessorCtx<'a> {
    pub mem: &'a GuestMem,
    pub cr3_pa: Cr3Pa,
    pub page_offset: PageOffset,
    pub offsets: &'a BpfMapOffsets,
    pub l5: bool,
    /// Cached TCR_EL1 register; drives the aarch64 page-table walker's
    /// granule selection. Always 0 on x86_64 (the walker ignores it).
    pub tcr_el1: u64,
    /// Runtime kernel image base (`__START_KERNEL_map` on x86_64,
    /// `KIMAGE_VADDR` on aarch64). Used for translating
    /// kernel-text/data symbols (e.g. `map_idr`) to physical
    /// addresses. Mirrors [`super::guest::GuestKernel::start_kernel_map`].
    pub start_kernel_map: u64,
}

// Map type discriminants from `enum bpf_map_type` in
// `include/uapi/linux/bpf.h`. Kept as flat `pub const u32` rather
// than a Rust enum so a kernel that adds a new map type past this
// list still surfaces as a numeric `map_type` on the
// [`BpfMapInfo`] / [`super::dump::FailureDumpMap`] wire format —
// the dump renderer falls through to a generic
// "unknown map type {n}" arm rather than failing to deserialize.

/// `BPF_MAP_TYPE_HASH` — generic hash table. Inline value bytes at
/// `htab_elem_value` (`key + round_up(key_size, 8)`).
pub const BPF_MAP_TYPE_HASH: u32 = 1;

/// `BPF_MAP_TYPE_ARRAY` — fixed-size array of values. Inline values
/// at the `bpf_array.value` flex array.
pub const BPF_MAP_TYPE_ARRAY: u32 = 2;

/// `BPF_MAP_TYPE_PROG_ARRAY` — array of `struct bpf_prog *` slots
/// used by `bpf_tail_call`. Userspace-visible value is a program fd
/// (or its kernel pointer); the underlying program is not data.
pub const BPF_MAP_TYPE_PROG_ARRAY: u32 = 3;

/// `BPF_MAP_TYPE_PERF_EVENT_ARRAY` — array of perf event fds. Same
/// shape as `PROG_ARRAY` but stores perf event references.
pub const BPF_MAP_TYPE_PERF_EVENT_ARRAY: u32 = 4;

/// `BPF_MAP_TYPE_PERCPU_HASH` — like `HASH` but value is a
/// `void __percpu *` resolved per-CPU via `__per_cpu_offset[cpu]`.
pub const BPF_MAP_TYPE_PERCPU_HASH: u32 = 5;

/// `BPF_MAP_TYPE_PERCPU_ARRAY` — like `ARRAY` but each slot is a
/// `void __percpu *` resolved per-CPU.
pub const BPF_MAP_TYPE_PERCPU_ARRAY: u32 = 6;

/// `BPF_MAP_TYPE_STACK_TRACE` — kernel-side stack trace storage
/// keyed by stackid. Values are transient (cleared after read by
/// `bpf_get_stackid`); not a persistent state surface.
pub const BPF_MAP_TYPE_STACK_TRACE: u32 = 7;

/// `BPF_MAP_TYPE_CGROUP_ARRAY` — array of cgroup fds. FD-array shape
/// like `PROG_ARRAY`.
pub const BPF_MAP_TYPE_CGROUP_ARRAY: u32 = 8;

/// `BPF_MAP_TYPE_LRU_HASH` — `HASH` plus LRU eviction. Value layout
/// identical to `HASH` (inline value bytes); `htab_elem` carries
/// `lru_node` in the same union slot as `ptr_to_pptr`.
pub const BPF_MAP_TYPE_LRU_HASH: u32 = 9;

/// `BPF_MAP_TYPE_LRU_PERCPU_HASH` — `PERCPU_HASH` plus LRU eviction.
/// Same value-position-is-percpu-pointer layout as `PERCPU_HASH`.
pub const BPF_MAP_TYPE_LRU_PERCPU_HASH: u32 = 10;

/// `BPF_MAP_TYPE_LPM_TRIE` — longest-prefix-match trie. Keyed by
/// (prefixlen, data); values are bytes. Iteration requires the
/// trie's per-node walk, not provided here.
pub const BPF_MAP_TYPE_LPM_TRIE: u32 = 11;

/// `BPF_MAP_TYPE_ARRAY_OF_MAPS` — array slots store map fds.
pub const BPF_MAP_TYPE_ARRAY_OF_MAPS: u32 = 12;

/// `BPF_MAP_TYPE_HASH_OF_MAPS` — hash slots store map fds.
pub const BPF_MAP_TYPE_HASH_OF_MAPS: u32 = 13;

/// `BPF_MAP_TYPE_DEVMAP` — array of net_device fds for XDP
/// redirection.
pub const BPF_MAP_TYPE_DEVMAP: u32 = 14;

/// `BPF_MAP_TYPE_SOCKMAP` — array of socket fds.
pub const BPF_MAP_TYPE_SOCKMAP: u32 = 15;

/// `BPF_MAP_TYPE_CPUMAP` — array of cpumap entries for XDP
/// redirection.
pub const BPF_MAP_TYPE_CPUMAP: u32 = 16;

/// `BPF_MAP_TYPE_XSKMAP` — array of AF_XDP socket fds.
pub const BPF_MAP_TYPE_XSKMAP: u32 = 17;

/// `BPF_MAP_TYPE_SOCKHASH` — hash of socket fds.
pub const BPF_MAP_TYPE_SOCKHASH: u32 = 18;

/// `BPF_MAP_TYPE_CGROUP_STORAGE` — deprecated cgroup-attached
/// storage. Replaced by `CGRP_STORAGE`. Reading requires the
/// cgroup context the program was attached to.
pub const BPF_MAP_TYPE_CGROUP_STORAGE: u32 = 19;

/// `BPF_MAP_TYPE_REUSEPORT_SOCKARRAY` — array of SO_REUSEPORT
/// socket fds.
pub const BPF_MAP_TYPE_REUSEPORT_SOCKARRAY: u32 = 20;

/// `BPF_MAP_TYPE_PERCPU_CGROUP_STORAGE` — deprecated per-CPU
/// cgroup-attached storage.
pub const BPF_MAP_TYPE_PERCPU_CGROUP_STORAGE: u32 = 21;

/// `BPF_MAP_TYPE_QUEUE` — FIFO queue (no key). Values are popped
/// destructively by `bpf_map_pop_elem`.
pub const BPF_MAP_TYPE_QUEUE: u32 = 22;

/// `BPF_MAP_TYPE_STACK` — LIFO stack (no key). Same destructive
/// pop semantics as `QUEUE`.
pub const BPF_MAP_TYPE_STACK: u32 = 23;

/// `BPF_MAP_TYPE_SK_STORAGE` — per-socket storage. Reading requires
/// iterating sockets, not a flat key/value walk.
pub const BPF_MAP_TYPE_SK_STORAGE: u32 = 24;

/// `BPF_MAP_TYPE_DEVMAP_HASH` — hash of net_device fds.
pub const BPF_MAP_TYPE_DEVMAP_HASH: u32 = 25;

/// `BPF_MAP_TYPE_STRUCT_OPS` — kernel struct table (e.g.
/// `tcp_congestion_ops`, `sched_ext_ops`). The map holds a single
/// `bpf_struct_ops_value` whose `data` field is the registered
/// kernel struct. `lookup_elem` returns `-EINVAL`; the live-host
/// path reads via `BPF_MAP_LOOKUP_ELEM` at key=0 anyway because the
/// kernel's syscall ABI does the read for `STRUCT_OPS` maps.
pub const BPF_MAP_TYPE_STRUCT_OPS: u32 = 26;

/// `BPF_MAP_TYPE_RINGBUF` — single-producer/single-consumer ring
/// buffer for streaming events. No key/value access; consumers
/// poll via `bpf_ringbuf_poll`.
pub const BPF_MAP_TYPE_RINGBUF: u32 = 27;

/// `BPF_MAP_TYPE_INODE_STORAGE` — per-inode storage. Reading
/// requires iterating inodes.
pub const BPF_MAP_TYPE_INODE_STORAGE: u32 = 28;

/// `BPF_MAP_TYPE_TASK_STORAGE` — per-task storage. Reading
/// requires iterating tasks.
pub const BPF_MAP_TYPE_TASK_STORAGE: u32 = 29;

/// `BPF_MAP_TYPE_BLOOM_FILTER` — probabilistic set membership.
/// No key enumeration — only `bpf_map_peek_elem` returns whether
/// a probe value is "maybe present".
pub const BPF_MAP_TYPE_BLOOM_FILTER: u32 = 30;

/// `BPF_MAP_TYPE_USER_RINGBUF` — userspace producer / BPF
/// consumer ring buffer. Same transient nature as `RINGBUF`.
pub const BPF_MAP_TYPE_USER_RINGBUF: u32 = 31;

/// `BPF_MAP_TYPE_CGRP_STORAGE` — per-cgroup storage (replaces
/// `CGROUP_STORAGE`). Reading requires iterating cgroups.
pub const BPF_MAP_TYPE_CGRP_STORAGE: u32 = 32;

/// `BPF_MAP_TYPE_ARENA` — sparse, page-granular memory region
/// shared between BPF programs and userspace. The host-side
/// walker for arena pages lives in [`super::arena`].
pub const BPF_MAP_TYPE_ARENA: u32 = 33;

/// `BPF_MAP_TYPE_INSN_ARRAY` — array of bpf instructions used by
/// the verifier for indirect-jump targets. Values are kernel-side
/// program references, not application data.
pub const BPF_MAP_TYPE_INSN_ARRAY: u32 = 34;

/// BPF_OBJ_NAME_LEN from include/linux/bpf.h.
const BPF_OBJ_NAME_LEN: usize = 16;

/// Discovered BPF map metadata and value location.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct BpfMapInfo {
    /// Guest physical address of the `struct bpf_map`.
    pub map_pa: u64,
    /// Guest KVA of the `struct bpf_map` (or containing struct like
    /// `bpf_array`/`bpf_htab`). Needed for hash map iteration to
    /// read `bpf_htab` fields relative to this base.
    pub map_kva: u64,
    /// Map name (null-terminated, up to BPF_OBJ_NAME_LEN).
    pub name: String,
    /// `map_type` field value.
    pub map_type: u32,
    /// `map_flags` field value.
    pub map_flags: u32,
    /// `key_size` field value.
    pub key_size: u32,
    /// `value_size` field value.
    pub value_size: u32,
    /// `max_entries` field value.
    pub max_entries: u32,
    /// Guest KVA of the map's value region for single-buffer
    /// reads. `Some(kva)` when the renderer can read up to `value_size`
    /// bytes starting at this address; `None` when the map type
    /// requires a different walker (hash iteration, arena page
    /// snapshot, …) or the kva resolution failed.
    ///
    /// Populated for:
    /// * `BPF_MAP_TYPE_ARRAY` — points at `bpf_array.value` (the
    ///   inline flex array). Renderer reads `value_size` bytes.
    /// * `BPF_MAP_TYPE_STRUCT_OPS` — points at `kvalue.data` (the
    ///   embedded registered struct's bytes, after the
    ///   `bpf_struct_ops_common_value` header). Renderer reads
    ///   `value_size - data_off` bytes to match the size of the
    ///   `btf_value_type_id` type, which describes the data payload
    ///   only. `None` when struct_ops BTF offsets are unresolved.
    pub value_kva: Option<u64>,
    /// Guest KVA of the map's `struct btf` (guest-memory backend),
    /// or `btf_id` cast to u64 (live-host backend reading via the
    /// bpf(2) syscall: `BPF_OBJ_GET_INFO_BY_FD` returns `btf_id`,
    /// not a kernel pointer). The dump path treats the value as
    /// opaque — only `btf_kva == 0` is meaningful (no BTF
    /// associated with this map). Backend-specific consumers cast
    /// to the shape they need.
    /// 0 if the map has no BTF.
    pub btf_kva: u64,
    /// BTF type ID for the map's value type. 0 if the map has no BTF.
    pub btf_value_type_id: u32,
    /// BTF type ID for the kernel-side `bpf_struct_ops_<name>`
    /// wrapper in vmlinux BTF, populated for `BPF_MAP_TYPE_STRUCT_OPS`
    /// maps. libbpf zeros `btf_value_type_id` for STRUCT_OPS and
    /// passes the wrapper id via the kernel-only
    /// `btf_vmlinux_value_type_id` field on `struct bpf_map`. The
    /// dump path uses it to BTF-render the data payload by walking
    /// the wrapper's `data` member to the per-ops struct (e.g.
    /// `sched_ext_ops`). Zero on every other map type.
    pub btf_vmlinux_value_type_id: u32,
    /// BTF type ID for the map's key type. 0 when the map's BTF is
    /// missing or the map type does not record a key type id (most
    /// ARRAY-family maps store a synthetic `__u32` key implicitly).
    /// HASH maps populate this so the dump path can render keys via
    /// BTF instead of falling back to hex.
    pub btf_key_type_id: u32,
}

/// Enumerate all BPF maps in the kernel's `map_idr` xarray.
///
/// Returns metadata for every map whose KVA can be translated.
/// No filtering by type or name — callers select from the result.
///
/// `value_kva` is populated for `BPF_MAP_TYPE_ARRAY` (inline
/// `bpf_array.value`) and `BPF_MAP_TYPE_STRUCT_OPS`
/// (`kvalue.data` inside `bpf_struct_ops_map`). All other map types
/// resolve to `None` — they require dedicated walkers
/// ([`iter_htab_entries`] for HASH, [`super::arena::snapshot_arena`]
/// for ARENA, …).
pub(crate) fn find_all_bpf_maps(ctx: &AccessorCtx<'_>, map_idr_kva: u64) -> Vec<BpfMapInfo> {
    let idr_pa = text_kva_to_pa_with_base(map_idr_kva, ctx.start_kernel_map);
    let offsets = ctx.offsets;

    let xa_head = ctx.mem.read_u64(idr_pa, offsets.idr_xa_head);
    if xa_head == 0 {
        return Vec::new();
    }
    // idr_next is the next ID the kernel will allocate. All live entries
    // have IDs in 0..idr_next, so scanning beyond it only hits empty or
    // wrapped slots.
    let idr_next = ctx.mem.read_u32(idr_pa, offsets.idr_next);

    let mut maps = Vec::new();

    for id in 0..idr_next {
        let Some(entry) = xa_load(
            ctx.mem,
            ctx.page_offset.0,
            xa_head,
            id as u64,
            offsets.xa_node_slots,
            offsets.xa_node_shift,
        ) else {
            continue;
        };
        if entry == 0 {
            continue;
        }

        let Some(map_pa) = translate_any_kva(
            ctx.mem,
            ctx.cr3_pa.0,
            ctx.page_offset.0,
            entry,
            ctx.l5,
            ctx.tcr_el1,
        ) else {
            continue;
        };

        let mut name_buf = [0u8; BPF_OBJ_NAME_LEN];
        ctx.mem
            .read_bytes(map_pa + offsets.map_name as u64, &mut name_buf);
        let name_len = name_buf
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(BPF_OBJ_NAME_LEN);
        let name = String::from_utf8_lossy(&name_buf[..name_len]).to_string();

        let map_type = ctx.mem.read_u32(map_pa, offsets.map_type);
        let map_flags = ctx.mem.read_u32(map_pa, offsets.map_flags);
        let key_size = ctx.mem.read_u32(map_pa, offsets.key_size);
        let value_size = ctx.mem.read_u32(map_pa, offsets.value_size);
        let max_entries = ctx.mem.read_u32(map_pa, offsets.max_entries);

        // value_kva is the start KVA the renderer reads value bytes
        // from. Two map types populate it:
        //
        // * `BPF_MAP_TYPE_ARRAY`: `bpf_array` embeds `bpf_map` at
        //   offset 0 and the value flex array is inline at
        //   `bpf_array.value`.
        // * `BPF_MAP_TYPE_STRUCT_OPS`: `bpf_struct_ops_map` embeds
        //   `kvalue` (a `bpf_struct_ops_value`) inline; the registered
        //   kernel struct lives at `kvalue.data`. `map->btf_value_type_id`
        //   describes only the data payload, not the prefixing
        //   `bpf_struct_ops_common_value`, so value_kva points at
        //   `data` and the renderer reads `value_size - data_off` bytes
        //   to fit the typed shape.
        //
        // Other map types (HASH, RINGBUF, ARENA, …) have no contiguous
        // value region the renderer can read with a single offset/len
        // pair — they use dedicated walkers (`iter_hash_map`,
        // `read_arena_pages`, …).
        let value_kva = match map_type {
            BPF_MAP_TYPE_ARRAY => Some(entry + offsets.array_value as u64),
            BPF_MAP_TYPE_STRUCT_OPS => offsets
                .struct_ops_offsets
                .as_ref()
                .map(|so| entry + so.kvalue as u64 + so.value_data as u64),
            _ => None,
        };

        let btf_kva = ctx.mem.read_u64(map_pa, offsets.map_btf);
        let btf_value_type_id = ctx.mem.read_u32(map_pa, offsets.map_btf_value_type_id);
        // `btf_vmlinux_value_type_id` lives at offset 0 only when the
        // resolver couldn't locate the field (kernel built without
        // CONFIG_BPF_JIT). Treat offset 0 as "unresolved" — reading
        // u32 at offset 0 of `struct bpf_map` would alias `map_type`,
        // which is decidedly NOT a btf type id. The STRUCT_OPS arm
        // checks for non-zero before using.
        let btf_vmlinux_value_type_id = if offsets.map_btf_vmlinux_value_type_id != 0 {
            ctx.mem
                .read_u32(map_pa, offsets.map_btf_vmlinux_value_type_id)
        } else {
            0
        };
        let btf_key_type_id = ctx.mem.read_u32(map_pa, offsets.map_btf_key_type_id);

        maps.push(BpfMapInfo {
            map_pa,
            map_kva: entry,
            name,
            map_type,
            map_flags,
            key_size,
            value_size,
            max_entries,
            value_kva,
            btf_kva,
            btf_value_type_id,
            btf_vmlinux_value_type_id,
            btf_key_type_id,
        });
    }

    maps
}

/// Find the first BPF ARRAY map whose name ends with `name_suffix`.
///
/// Only returns `BPF_MAP_TYPE_ARRAY` maps. Use [`find_all_bpf_maps`]
/// to enumerate maps of all types.
pub(crate) fn find_bpf_map(
    ctx: &AccessorCtx<'_>,
    map_idr_kva: u64,
    name_suffix: &str,
) -> Option<BpfMapInfo> {
    find_all_bpf_maps(ctx, map_idr_kva)
        .into_iter()
        .find(|m| m.map_type == BPF_MAP_TYPE_ARRAY && m.name.ends_with(name_suffix))
}

/// Smallest page granule that `translate_kva` always resolves contiguously.
///
/// x86-64 and aarch64 both partition KVA into 4 KiB pages at the lowest
/// level; larger entries (2 MiB PMD block, 1 GiB PUD block, aarch64 64 KiB
/// base) are strictly coarser, so chunking at 4 KiB means a single
/// `translate_kva` call covers the rest of the page regardless of the
/// entry granule. Bumping this to a larger value would break 4 KiB
/// huge-page-absent paths because a single translate result would no
/// longer be guaranteed to span the chunk.
const BPF_MAP_PAGE_CHUNK: u64 = 4096;

/// Hostile-guest cap on a single value-region read. Bounds the
/// `vec![0u8; len]` allocation before it reaches the heap so a
/// corrupted (uninitialized) `bpf_map.value_size` read can't
/// induce a multi-gigabyte allocation on the freeze hot path.
/// 16 MiB covers every realistic BPF map's per-entry size; a
/// global-section ARRAY (`.bss` etc.) is the largest practical
/// value at scheduler scale, and the kernel itself caps `value_size`
/// well below this for ordinary map types.
const MAX_VALUE_SIZE: usize = 16 * 1024 * 1024;

/// Copy a contiguous byte range to or from a map's value region,
/// chunking at page boundaries so each chunk takes one `translate_kva`
/// call plus one bulk DRAM copy.
///
/// This replaces the former byte-by-byte loop that issued one
/// translate per byte — a 4 KiB value read translated 4096 times and
/// paid 4096 copy_nonoverlapping-of-one-byte calls. A full page now
/// takes one translate + one bulk copy (up to BPF_MAP_PAGE_CHUNK
/// bytes); a range that crosses a page boundary splits into N
/// translate+copy pairs where N is the number of pages touched.
///
/// `ctx` supplies the CR3 / L5 flag and the DRAM accessor. `target_kva`
/// is the starting guest virtual address; `len` is the total length.
/// `chunk_fn` receives the resolved guest PA and the chunk buffer
/// (mutable for reads, immutable for writes) and performs the actual
/// memcpy. Returns `false` when any chunk fails to translate.
fn chunked_kva_io<F>(ctx: &AccessorCtx<'_>, target_kva: u64, len: usize, mut chunk_fn: F) -> bool
where
    F: FnMut(u64, u64, usize),
{
    let mut consumed: u64 = 0;
    let total = len as u64;
    while consumed < total {
        let kva = target_kva + consumed;
        let Some(pa) = ctx
            .mem
            .translate_kva(ctx.cr3_pa.0, Kva(kva), ctx.l5, ctx.tcr_el1)
        else {
            return false;
        };
        // Advance at most to the next page boundary so the next
        // translate_kva lands on a fresh resolved page.
        let page_end = (kva & !(BPF_MAP_PAGE_CHUNK - 1)) + BPF_MAP_PAGE_CHUNK;
        let chunk_len = (page_end - kva).min(total - consumed) as usize;
        chunk_fn(pa, consumed, chunk_len);
        consumed += chunk_len as u64;
    }
    true
}

/// Write bytes to a BPF map's value region at `offset`.
///
/// Translates the value KVA (vmalloc'd for .bss maps) through the
/// page table to find the guest physical address, then writes directly.
/// Returns `false` if the map has no value KVA (non-ARRAY map),
/// `offset + data.len()` exceeds `value_size`, or any page in the
/// range is unmapped. Uses [`chunked_kva_io`] to pay one translate per
/// 4 KiB page rather than one per byte.
pub(crate) fn write_bpf_map_value(
    ctx: &AccessorCtx<'_>,
    map_info: &BpfMapInfo,
    offset: usize,
    data: &[u8],
) -> bool {
    let Some(base_kva) = map_info.value_kva else {
        return false;
    };
    // checked_add against pathological offset+len that would
    // wrap usize. Without the check, a wrap would silently make
    // `> value_size` false and the chunked write would walk
    // arbitrary KVAs.
    let Some(end) = offset.checked_add(data.len()) else {
        return false;
    };
    if end > map_info.value_size as usize {
        return false;
    }
    let target_kva = base_kva + offset as u64;

    chunked_kva_io(ctx, target_kva, data.len(), |pa, src_off, chunk_len| {
        // GuestMem exposes only byte-wise volatile writes for arbitrary
        // lengths; the savings come from paying one translate per page
        // instead of per byte. A block-write primitive in reader.rs
        // could let this become one copy_nonoverlapping per chunk.
        let src_off = src_off as usize;
        for (i, &byte) in data[src_off..src_off + chunk_len].iter().enumerate() {
            ctx.mem.write_u8(pa, i, byte);
        }
    })
}

/// Write a u32 to a BPF map's value region at `offset`.
pub(crate) fn write_bpf_map_value_u32(
    ctx: &AccessorCtx<'_>,
    map_info: &BpfMapInfo,
    offset: usize,
    val: u32,
) -> bool {
    write_bpf_map_value(ctx, map_info, offset, &val.to_ne_bytes())
}

/// Read bytes from a BPF map's value region at `offset`.
///
/// Translates the value KVA (vmalloc'd for .bss maps) through the
/// page table to find the guest physical address, then reads directly.
/// Returns `None` if the map has no value KVA (non-ARRAY map),
/// `offset + len` exceeds `value_size`, or any page in the range
/// is unmapped. Uses [`chunked_kva_io`] to pay one translate per 4 KiB
/// page plus one bulk [`GuestMem::read_bytes`] call, instead of one
/// translate and one-byte copy per byte.
pub(crate) fn read_bpf_map_value(
    ctx: &AccessorCtx<'_>,
    map_info: &BpfMapInfo,
    offset: usize,
    len: usize,
) -> Option<Vec<u8>> {
    let base_kva = map_info.value_kva?;
    // checked_add against pathological offset+len that would
    // wrap usize. See the matching guard on `write_bpf_map_value`
    // above for the rationale.
    let end = offset.checked_add(len)?;
    if end > map_info.value_size as usize {
        return None;
    }
    // Hostile-guest size cap before allocation: a corrupted
    // `value_size` (or a caller passing a huge `len`) would
    // otherwise allocate up to 4 GiB inside `vec![0u8; len]`.
    if len > MAX_VALUE_SIZE {
        return None;
    }
    let target_kva = base_kva + offset as u64;
    let mut buf = vec![0u8; len];

    // Safety / correctness: `chunked_kva_io` returns false when any
    // page in the range is unmapped; propagate that to None so callers
    // see "unreadable" rather than a partial buffer.
    let buf_ptr = buf.as_mut_ptr();
    let ok = chunked_kva_io(ctx, target_kva, len, |pa, dst_off, chunk_len| {
        // SAFETY: dst_off + chunk_len <= len <= buf.len(); the slice
        // borrows the heap-allocated Vec whose backing storage is live
        // for the duration of this call (the Vec is pinned in `buf`
        // above and reborrowed here only through its mutable pointer).
        let slice =
            unsafe { std::slice::from_raw_parts_mut(buf_ptr.add(dst_off as usize), chunk_len) };
        // GuestMem::read_bytes returns the count actually copied; the
        // caller has bounds-checked value_size and translate_kva has
        // confirmed the page is mapped, so a short read here means
        // the page crosses end-of-DRAM, which the original byte loop
        // would also have silently short-copied.
        let _ = ctx.mem.read_bytes(pa, slice);
    });
    if !ok {
        return None;
    }
    Some(buf)
}

/// Read a u32 from a BPF map's value region at `offset`.
pub(crate) fn read_bpf_map_value_u32(
    ctx: &AccessorCtx<'_>,
    map_info: &BpfMapInfo,
    offset: usize,
) -> Option<u32> {
    let bytes = read_bpf_map_value(ctx, map_info, offset, 4)?;
    Some(u32::from_ne_bytes(bytes.try_into().unwrap()))
}

/// Read the per-CPU values for a single key in a `BPF_MAP_TYPE_PERCPU_ARRAY` map.
///
/// `bpf_array.pptrs[key]` holds a `__percpu` pointer. Adding
/// `__per_cpu_offset[cpu]` yields the per-CPU KVA, which may live
/// either in the direct mapping (static percpu, kmalloc'd percpu)
/// or in vmalloc'd memory (large dynamic per-CPU allocations).
/// Address translation goes through [`translate_any_kva`], which
/// tries direct mapping first and falls through to a page-table
/// walk for vmalloc'd percpu — so a per-CPU value that misses the
/// direct mapping no longer reads as `None` simply because the
/// underlying allocation lives in vmalloc.
///
/// Returns one entry per CPU, indexed by CPU number. `Some(bytes)`
/// when the per-CPU PA falls within guest memory; `None` when it
/// does not. Returns an empty vec if the map is not
/// `BPF_MAP_TYPE_PERCPU_ARRAY`, `key >= max_entries`, or the percpu
/// pointer is zero.
fn read_percpu_array_value(
    ctx: &AccessorCtx<'_>,
    map: &BpfMapInfo,
    key: u32,
    per_cpu_offsets: &[u64],
) -> Vec<Option<Vec<u8>>> {
    if map.map_type != BPF_MAP_TYPE_PERCPU_ARRAY {
        return Vec::new();
    }
    if key >= map.max_entries {
        return Vec::new();
    }

    // pptrs is at the same offset as value (union in bpf_array).
    let pptrs_kva = map.map_kva + ctx.offsets.array_value as u64;
    // pptrs[key] is a void __percpu * — 8 bytes.
    let pptr_kva = pptrs_kva + (key as u64) * 8;

    // bpf_array may be kmalloc'd or vmalloc'd — try direct mapping first.
    let Some(pptr_pa) = translate_any_kva(
        ctx.mem,
        ctx.cr3_pa.0,
        ctx.page_offset.0,
        pptr_kva,
        ctx.l5,
        ctx.tcr_el1,
    ) else {
        return Vec::new();
    };
    let percpu_base = ctx.mem.read_u64(pptr_pa, 0);
    if percpu_base == 0 {
        return Vec::new();
    }

    let value_size = map.value_size as usize;
    let mut result = Vec::with_capacity(per_cpu_offsets.len());

    for (cpu_index, &cpu_off) in per_cpu_offsets.iter().enumerate() {
        // Out-of-range CPU detection: kernel `setup_per_cpu_areas`
        // (e.g. arch/x86/kernel/setup_percpu.c) only writes
        // `__per_cpu_offset[cpu]` for cpus in `for_each_possible_cpu`,
        // leaving slots beyond `nr_cpu_ids` at the BSS-initialized
        // value of 0. Real SMP kernels assign each possible CPU a
        // strictly-positive offset (`delta + unit_offsets[cpu]`) for
        // cpu > 0 because `unit_offsets[cpu]` is a positive multiple
        // of the per-CPU unit size — only the BSP (cpu_index == 0)
        // can legitimately observe a zero offset on systems where
        // the delta term is zero. Treating `cpu_off == 0 &&
        // cpu_index > 0` as out-of-range prevents the prior aliasing
        // bug where every out-of-range slot returned CPU 0's bytes
        // (because `percpu_base + 0` translated successfully to
        // whatever the bare percpu_base pointed at).
        if cpu_off == 0 && cpu_index > 0 {
            result.push(None);
            continue;
        }
        let cpu_kva = percpu_base.wrapping_add(cpu_off);
        // The percpu base + cpu_off may land in either the direct
        // mapping (per-CPU __percpu allocations from the static
        // percpu region or kmalloc'd percpu blocks) or vmalloc'd
        // percpu memory (large dynamic per-CPU allocations served
        // from pcpu_get_vm_areas). `translate_any_kva` tries direct
        // mapping first and falls through to a page-table walk for
        // vmalloc'd percpu, so it covers both.
        match translate_any_kva(
            ctx.mem,
            ctx.cr3_pa.0,
            ctx.page_offset.0,
            cpu_kva,
            ctx.l5,
            ctx.tcr_el1,
        ) {
            Some(cpu_pa)
                if cpu_pa
                    .checked_add(value_size as u64)
                    .is_some_and(|end| end <= ctx.mem.size()) =>
            {
                let mut buf = vec![0u8; value_size];
                ctx.mem.read_bytes(cpu_pa, &mut buf);
                result.push(Some(buf));
            }
            _ => result.push(None),
        }
    }

    result
}

/// Chase modifiers (Volatile, Const, Typedef, TypeTag, Restrict),
/// pointers, and typedefs from `type_id` to find a Struct or Union.
///
/// Returns `None` if the chain ends in a type that is neither Struct
/// nor Union, or exceeds depth 20. Also resolves through Ptr (for
/// pointer-to-struct members).
pub(crate) fn resolve_to_struct(btf: &btf_rs::Btf, type_id: u32) -> Option<btf_rs::Struct> {
    let mut t = btf.resolve_type_by_id(type_id).ok()?;
    for _ in 0..20 {
        match t {
            btf_rs::Type::Struct(s) | btf_rs::Type::Union(s) => return Some(s),
            btf_rs::Type::Ptr(_)
            | btf_rs::Type::Volatile(_)
            | btf_rs::Type::Const(_)
            | btf_rs::Type::Typedef(_)
            | btf_rs::Type::TypeTag(_)
            | btf_rs::Type::Restrict(_) => {
                t = btf.resolve_chained_type(t.as_btf_type()?).ok()?;
            }
            _ => return None,
        }
    }
    None
}

/// Read-only abstraction over BPF map enumeration and value reads
/// across data sources. Mutating operations (write_value etc.) are
/// inherent on each backend, NOT exposed here — the trait surface is
/// a snapshot-style read API used by the failure-dump renderer and
/// any future read-only consumer.
///
/// One implementation lives in this crate today; a second backend is
/// planned (live-host introspection via the `bpf()` syscall — see
/// the live-host introspection task in the project queue) and will
/// plug into the same trait surface.
///
/// - [`GuestMemMapAccessor`] — reads from a frozen guest VM's physical
///   memory via PTE walks against the frozen `init_mm`. Used by the
///   freeze-coordinator path ([`super::dump::dump_state`]) on the
///   in-VM scheduler test runs. Hash map iteration walks
///   `bpf_htab.buckets` directly without RCU; the freeze rendezvous
///   IS the ordering primitive (every CPU is parked at a known KVM
///   exit before the host begins reading memory). Per-CPU value
///   reads use the cached `__per_cpu_offset[cpu]` array; out-of-range
///   CPUs surface as `None` rather than aliasing CPU 0 (see
///   [`read_percpu_array_value`]).
///
/// The planned live-host backend will produce identical
/// [`BpfMapInfo`] / byte buffers, so the rendering pipeline
/// ([`super::btf_render::render_value`]) stays data-source-agnostic
/// and will consume either accessor through this trait. The
/// live-host backend's failure modes are different (e.g. hash reads
/// will rely on the kernel's RCU read-side critical section,
/// `bpf_map_lookup_elem` rejection for non-readable types) and
/// individual method docs spell those out where they matter.
///
/// `dump_state` currently takes a concrete
/// [`GuestMemMapAccessor`] because its sdt_alloc post-pass walks
/// the underlying [`super::guest::GuestKernel`] — that handle is
/// not part of the trait surface. When the live-host backend lands
/// (and sdt_alloc walking moves into a backend-specific path),
/// `dump_state` will switch to `&dyn BpfMapAccessor`. Other call
/// sites that need only the trait surface can already bind on
/// `&dyn BpfMapAccessor` (or `<A: BpfMapAccessor>`) without paying
/// virtual dispatch.
#[allow(dead_code)]
pub trait BpfMapAccessor {
    /// Enumerate every BPF map visible to this accessor.
    ///
    /// Order is implementation-defined: the guest-memory backend walks
    /// `map_idr` (allocation order); the planned bpf-syscall backend
    /// will walk the kernel's id space via `BPF_MAP_GET_NEXT_ID` (also
    /// allocation order, modulo concurrent destruction races on the
    /// live host). Callers that want a stable view should sort by name.
    fn maps(&self) -> Vec<BpfMapInfo>;

    /// Find the first BPF map whose name ends with `name_suffix`.
    ///
    /// Default impl walks [`Self::maps`]. Backends with cheaper
    /// targeted lookups can override (e.g. a libbpf-handle-backed
    /// accessor that already holds a name index).
    fn find_map(&self, name_suffix: &str) -> Option<BpfMapInfo> {
        self.maps()
            .into_iter()
            .find(|m| m.name.ends_with(name_suffix))
    }

    /// Read a contiguous byte range from a map's value region.
    ///
    /// Returns `None` for non-readable map types (e.g. ARENA — use
    /// [`Self::read_arena_pages`]; HASH — use [`Self::iter_hash_map`])
    /// or when the backing read fails. The guest-memory backend's
    /// failure modes are unmapped guest pages and out-of-range value
    /// regions; the planned bpf-syscall backend will additionally
    /// surface `bpf_map_lookup_elem` rejection (e.g. `-EINVAL` on
    /// arena maps, kernel-side ACL denials).
    fn read_value(&self, map: &BpfMapInfo, offset: usize, len: usize) -> Option<Vec<u8>>;

    /// Iterate every entry in a `BPF_MAP_TYPE_HASH` or
    /// `BPF_MAP_TYPE_LRU_HASH` map.
    ///
    /// Both share the inline-value `htab_elem` layout
    /// (`kernel/bpf/hashtab.c::htab_elem_value`); LRU adds an
    /// eviction policy but the value bytes still sit at
    /// `key + round_up(key_size, 8)`. Returns an empty vec for any
    /// other map type.
    ///
    /// Per-element atomicity is backend-specific: the guest-memory
    /// backend reads raw bytes at the freeze instant (the freeze
    /// rendezvous IS the synchronization — no concurrent writers
    /// exist while parked vCPUs stay parked); the bpf-syscall backend
    /// reads under the kernel's RCU read-side critical section
    /// (`bpf_map_lookup_elem` -> `htab_map_lookup_elem`). Both can
    /// produce torn views relative to a multi-element transaction
    /// the scheduler intended to commit atomically — that's a feature
    /// of reading without locking the whole table.
    fn iter_hash_map(&self, map: &BpfMapInfo) -> Vec<(Vec<u8>, Vec<u8>)>;

    /// Iterate every entry in a `BPF_MAP_TYPE_PERCPU_HASH` or
    /// `BPF_MAP_TYPE_LRU_PERCPU_HASH` map. Returns
    /// `(key_bytes, per_cpu_values)` where `per_cpu_values` is one
    /// entry per CPU indexed by CPU number; `Some(bytes)` when the
    /// CPU's slot is readable, `None` otherwise (unmapped page or
    /// out-of-range CPU).
    ///
    /// Returns an empty vec for any other map type. Default
    /// implementation returns empty so backends that haven't yet
    /// wired the percpu-hash path don't break trait dispatch — the
    /// dump renderer surfaces the resulting empty list as a
    /// "no entries" outcome rather than a panic.
    fn iter_percpu_hash_map(&self, _map: &BpfMapInfo, _num_cpus: u32) -> PerCpuHashEntries {
        Vec::new()
    }

    /// Iterate every entry in a `BPF_MAP_TYPE_TASK_STORAGE` map (and
    /// the shape-identical `INODE_STORAGE` / `SK_STORAGE` /
    /// `CGRP_STORAGE` variants — they all use
    /// [`super::btf_offsets::TaskStorageOffsets`]).
    ///
    /// Returned tuples are `(owner_kva_le_bytes, value_bytes)`:
    /// - `owner_kva_le_bytes` is the 8-byte little-endian encoding of
    ///   the `bpf_local_storage.owner` pointer reached by following
    ///   each `bpf_local_storage_elem.local_storage`. For
    ///   `TASK_STORAGE` this is the `task_struct` KVA; for the other
    ///   variants it is the inode/sock/cgroup KVA. The walker treats
    ///   it as opaque so the same shape works across all four map
    ///   types.
    /// - `value_bytes` is `value_size` bytes copied from
    ///   `bpf_local_storage_elem.sdata.data[]` — the value the
    ///   scheduler stored under this owner.
    ///
    /// Returns an empty vec for any other map type, when
    /// `task_storage_offsets` is unavailable, or when the map's
    /// `buckets` pointer cannot be translated. Returns an empty vec
    /// for any other map type. Default implementation returns empty
    /// so backends that haven't yet wired this path don't break
    /// trait dispatch — the dump renderer surfaces the resulting
    /// empty list as a "no entries" outcome rather than a panic.
    fn iter_task_storage(&self, _map: &BpfMapInfo) -> Vec<(Vec<u8>, Vec<u8>)> {
        Vec::new()
    }

    /// Read every CPU's value for a key in a `BPF_MAP_TYPE_PERCPU_ARRAY` map.
    ///
    /// Returns one entry per CPU, indexed by CPU number. `Some(bytes)`
    /// when the per-CPU slot is readable; `None` when it isn't (e.g.
    /// an out-of-range CPU index — `__per_cpu_offset[cpu]` reads as
    /// the BSS-zero sentinel — or an unmapped page on the
    /// guest-memory path; the planned bpf-syscall backend surfaces
    /// out-of-range CPU on `bpf_map_lookup_elem` failure). Returns an
    /// empty vec for non-PERCPU_ARRAY maps or `key >= max_entries`.
    fn read_percpu_array(&self, map: &BpfMapInfo, key: u32, num_cpus: u32) -> Vec<Option<Vec<u8>>>;

    /// Snapshot every mapped page of a `BPF_MAP_TYPE_ARENA` map.
    ///
    /// `arena_offsets` resolves kernel struct field offsets the
    /// guest-memory backend uses to walk `bpf_arena -> kern_vm ->
    /// vm_struct.addr`; the planned bpf-syscall backend will mmap the
    /// arena fd directly (the only data path the kernel exposes —
    /// arena's `lookup_elem` returns `-EINVAL`, see
    /// `kernel/bpf/arena.c`) and ignore `arena_offsets`. The default
    /// implementation returns an empty snapshot; backends override to
    /// produce real content.
    fn read_arena_pages(
        &self,
        _map: &BpfMapInfo,
        _arena_offsets: &super::arena::BpfArenaOffsets,
    ) -> super::arena::ArenaSnapshot {
        super::arena::ArenaSnapshot::default()
    }

    /// Load the program BTF object referenced by a map.
    ///
    /// `base_btf` is the host's vmlinux BTF used as the base for
    /// split-BTF parsing. Returns `None` when the map carries no
    /// program BTF (e.g. kernel-builtin maps), when the BTF blob can't
    /// be loaded, or when [`btf_rs::Btf::from_bytes`] /
    /// [`btf_rs::Btf::from_split_bytes`] reject the bytes.
    ///
    /// The default implementation returns `None`; backends override to
    /// hand back a parsed [`btf_rs::Btf`].
    fn load_program_btf(&self, _map: &BpfMapInfo, _base_btf: &btf_rs::Btf) -> Option<btf_rs::Btf> {
        None
    }
}

/// Host-side BPF map accessor backed by direct guest physical-memory
/// reads.
///
/// Resolves BTF offsets for BPF map structures and provides map
/// discovery, value read/write, hash iteration, and per-CPU reads.
/// Uses a [`GuestKernel`] for address translation (PTE walks against
/// the guest's frozen page tables).
///
/// Implements the [`BpfMapAccessor`] trait so [`super::dump::dump_state`]
/// can dispatch through it without committing to a backend at the call
/// site.
///
/// [`GuestKernel`]: super::guest::GuestKernel
pub struct GuestMemMapAccessor<'a> {
    kernel: &'a super::guest::GuestKernel<'a>,
    map_idr_kva: u64,
    /// Borrowed from the `GuestMemMapAccessorOwned` that produced this
    /// accessor via `as_accessor`, or provided by the caller to
    /// `from_guest_kernel`. Borrowing avoids the ~160-byte
    /// `BpfMapOffsets` clone that the old owned-field design paid
    /// on every `as_accessor()` call.
    offsets: &'a BpfMapOffsets,
}

#[allow(dead_code)]
impl<'a> GuestMemMapAccessor<'a> {
    /// Create from an existing [`GuestKernel`] and a caller-owned
    /// [`BpfMapOffsets`].
    ///
    /// The accessor borrows the offsets for its lifetime, so callers
    /// typically stash them in a `GuestMemMapAccessorOwned` (or another
    /// stable location) before calling this. Build `offsets` once via
    /// [`BpfMapOffsets::from_vmlinux`] and reuse — they're per-kernel,
    /// not per-call.
    ///
    /// [`GuestKernel`]: super::guest::GuestKernel
    pub fn from_guest_kernel(
        kernel: &'a super::guest::GuestKernel<'a>,
        offsets: &'a BpfMapOffsets,
    ) -> anyhow::Result<Self> {
        let map_idr_kva = kernel
            .symbol_kva("map_idr")
            .ok_or_else(|| anyhow::anyhow!("map_idr symbol not found in vmlinux"))?;

        Ok(Self {
            kernel,
            map_idr_kva,
            offsets,
        })
    }

    /// Build a `GuestMemMapAccessor` for unit tests, bypassing the
    /// `map_idr` symbol lookup `from_guest_kernel` performs.
    ///
    /// Cross-module tests for the per-map render helpers
    /// (`render_ringbuf_state`, `render_stack_traces`,
    /// `render_fd_array_slots`) and for `iter_percpu_hash_map` need
    /// an accessor over a synthetic `GuestKernel`. The production
    /// `from_guest_kernel` requires the kernel to expose a `map_idr`
    /// symbol, which synthetic kernels constructed via
    /// `GuestKernel::new_for_test` typically do not. This
    /// constructor takes `map_idr_kva` directly so the caller can
    /// pass `0` (the per-map render helpers never read through the
    /// map_idr) or a known-good KVA when exercising
    /// `find_all_bpf_maps`.
    #[cfg(test)]
    pub(crate) fn new_for_test(
        kernel: &'a super::guest::GuestKernel<'a>,
        offsets: &'a BpfMapOffsets,
        map_idr_kva: u64,
    ) -> Self {
        Self {
            kernel,
            map_idr_kva,
            offsets,
        }
    }

    /// Build the [`AccessorCtx`] used by every map-read/write routine.
    fn ctx(&self) -> AccessorCtx<'_> {
        AccessorCtx {
            mem: self.kernel.mem(),
            cr3_pa: Cr3Pa(self.kernel.cr3_pa()),
            page_offset: PageOffset(self.kernel.page_offset()),
            offsets: self.offsets,
            l5: self.kernel.l5(),
            tcr_el1: self.kernel.tcr_el1(),
            start_kernel_map: self.kernel.start_kernel_map(),
        }
    }

    /// Borrow the resolved BPF map field offsets. Used by callers
    /// that need to read kernel struct fields (e.g. `struct btf` for
    /// the program-BTF loader) without going through the
    /// map-access trait surface.
    pub fn offsets(&self) -> &BpfMapOffsets {
        self.offsets
    }

    /// Borrow the underlying [`super::guest::GuestKernel`] for callers
    /// that need direct access to symbol resolution / page-walk
    /// primitives outside the map-discovery surface (e.g. arena page
    /// enumeration in [`super::arena`], sdt_alloc tree walks).
    pub fn kernel(&self) -> &'a super::guest::GuestKernel<'a> {
        self.kernel
    }

    /// Find the first BPF ARRAY map whose name ends with `name_suffix`.
    ///
    /// Only returns `BPF_MAP_TYPE_ARRAY` maps. Use
    /// [`BpfMapAccessor::maps`] to enumerate maps of all types.
    pub fn find_map(&self, name_suffix: &str) -> Option<BpfMapInfo> {
        find_bpf_map(&self.ctx(), self.map_idr_kva, name_suffix)
    }

    /// Write bytes to a map's value region.
    ///
    /// Returns `false` if the map has no value KVA (non-ARRAY map)
    /// or any page in the range is unmapped.
    pub fn write_value(&self, map: &BpfMapInfo, offset: usize, data: &[u8]) -> bool {
        write_bpf_map_value(&self.ctx(), map, offset, data)
    }

    /// Write a u32 to a map's value region.
    pub fn write_value_u32(&self, map: &BpfMapInfo, offset: usize, val: u32) -> bool {
        write_bpf_map_value_u32(&self.ctx(), map, offset, val)
    }

    /// Read a u32 from a map's value region.
    pub fn read_value_u32(&self, map: &BpfMapInfo, offset: usize) -> Option<u32> {
        read_bpf_map_value_u32(&self.ctx(), map, offset)
    }
}

impl BpfMapAccessor for GuestMemMapAccessor<'_> {
    fn maps(&self) -> Vec<BpfMapInfo> {
        find_all_bpf_maps(&self.ctx(), self.map_idr_kva)
    }

    fn read_value(&self, map: &BpfMapInfo, offset: usize, len: usize) -> Option<Vec<u8>> {
        read_bpf_map_value(&self.ctx(), map, offset, len)
    }

    fn iter_hash_map(&self, map: &BpfMapInfo) -> Vec<(Vec<u8>, Vec<u8>)> {
        iter_htab_entries(&self.ctx(), map)
    }

    /// Read per-CPU values for a key in a `BPF_MAP_TYPE_PERCPU_ARRAY` map.
    ///
    /// Resolves `__per_cpu_offset` from the guest kernel and reads each
    /// CPU's slot via [`translate_any_kva`]. Out-of-range CPUs (those
    /// whose `__per_cpu_offset` slot reads as zero — including reads
    /// past the end of guest memory and BSS-zero slots beyond
    /// `nr_cpu_ids`) return `None` rather than aliasing CPU 0's bytes;
    /// see the cpu_off==0 guard in [`read_percpu_array_value`].
    fn read_percpu_array(&self, map: &BpfMapInfo, key: u32, num_cpus: u32) -> Vec<Option<Vec<u8>>> {
        let Some(pco_kva) = self.kernel.symbol_kva("__per_cpu_offset") else {
            return Vec::new();
        };
        let pco_pa = self.kernel.text_kva_to_pa(pco_kva);
        let per_cpu_offsets =
            super::symbols::read_per_cpu_offsets(self.kernel.mem(), pco_pa, num_cpus);
        read_percpu_array_value(&self.ctx(), map, key, &per_cpu_offsets)
    }

    /// Walk a `BPF_MAP_TYPE_PERCPU_HASH` or
    /// `BPF_MAP_TYPE_LRU_PERCPU_HASH` map, dereferencing each
    /// element's per-CPU pointer for every CPU.
    ///
    /// Reuses the same `__per_cpu_offset` resolution path as
    /// [`Self::read_percpu_array`].
    fn iter_percpu_hash_map(&self, map: &BpfMapInfo, num_cpus: u32) -> PerCpuHashEntries {
        let Some(pco_kva) = self.kernel.symbol_kva("__per_cpu_offset") else {
            return Vec::new();
        };
        let pco_pa = self.kernel.text_kva_to_pa(pco_kva);
        let per_cpu_offsets =
            super::symbols::read_per_cpu_offsets(self.kernel.mem(), pco_pa, num_cpus);
        iter_percpu_htab_entries(&self.ctx(), map, &per_cpu_offsets)
    }

    fn read_arena_pages(
        &self,
        map: &BpfMapInfo,
        arena_offsets: &super::arena::BpfArenaOffsets,
    ) -> super::arena::ArenaSnapshot {
        super::arena::snapshot_arena(self.kernel, map, arena_offsets)
    }

    /// Walk every selem of a TASK_STORAGE / INODE_STORAGE /
    /// SK_STORAGE / CGRP_STORAGE map. Returns
    /// `(owner_kva_le_bytes, value_bytes)` per entry — see
    /// [`iter_local_storage_entries`] for the kernel-side walk
    /// shape (`bpf_local_storage_map.buckets[i].list` — regular
    /// hlist, NULL termination — followed by `container_of` math
    /// from `map_node` back to the elem base).
    fn iter_task_storage(&self, map: &BpfMapInfo) -> Vec<(Vec<u8>, Vec<u8>)> {
        iter_local_storage_entries(&self.ctx(), map)
    }

    fn load_program_btf(&self, map: &BpfMapInfo, base_btf: &btf_rs::Btf) -> Option<btf_rs::Btf> {
        if map.btf_kva == 0 {
            return None;
        }
        super::dump::load_program_btf_kva(self, map.btf_kva, base_btf)
    }
}

/// Owns a [`GuestKernel`] and provides BPF map access through the
/// [`GuestMemMapAccessor`] borrow.
///
/// Returned by [`GuestMemMapAccessorOwned::new`] which builds the
/// `GuestKernel` internally. Borrow as [`GuestMemMapAccessor`] via
/// [`as_accessor`](Self::as_accessor) for map operations.
///
/// [`GuestKernel`]: super::guest::GuestKernel
pub struct GuestMemMapAccessorOwned<'a> {
    kernel: super::guest::GuestKernel<'a>,
    map_idr_kva: u64,
    offsets: BpfMapOffsets,
}

#[allow(dead_code)]
impl<'a> GuestMemMapAccessorOwned<'a> {
    /// Create from GuestMem and vmlinux path.
    ///
    /// One-shot constructor: builds a [`GuestKernel`] from `vmlinux`,
    /// parses BTF to resolve the map-related struct offsets, and
    /// locates the `map_idr` symbol. The resulting handle owns both
    /// the `GuestKernel` and the `BpfMapOffsets`.
    ///
    /// Prefer [`GuestMemMapAccessor::from_guest_kernel`] when you already
    /// hold a `GuestKernel` **and** a pre-built `&BpfMapOffsets` — it
    /// builds a borrowed accessor without taking ownership of either,
    /// so callers that maintain their own offsets cache (e.g. across
    /// multiple map probes in the same poll cycle) don't pay repeat
    /// BTF parses. `new` is the convenience path when you want the
    /// accessor to own its offsets.
    ///
    /// [`GuestKernel`]: super::guest::GuestKernel
    pub fn new(mem: &'a GuestMem, vmlinux: &std::path::Path, tcr_el1: u64) -> anyhow::Result<Self> {
        let kernel = super::guest::GuestKernel::new(mem, vmlinux, tcr_el1)?;
        let offsets = BpfMapOffsets::from_vmlinux(vmlinux)?;

        let map_idr_kva = kernel
            .symbol_kva("map_idr")
            .ok_or_else(|| anyhow::anyhow!("map_idr symbol not found in vmlinux"))?;

        Ok(Self {
            kernel,
            map_idr_kva,
            offsets,
        })
    }

    /// Borrow as a [`GuestMemMapAccessor`] for map operations.
    ///
    /// The returned accessor borrows `self.offsets`; no clone on
    /// the hot path.
    pub fn as_accessor(&self) -> GuestMemMapAccessor<'_> {
        GuestMemMapAccessor {
            kernel: &self.kernel,
            map_idr_kva: self.map_idr_kva,
            offsets: &self.offsets,
        }
    }

    /// Access the underlying [`GuestKernel`] for low-level memory reads.
    ///
    /// [`GuestKernel`]: super::guest::GuestKernel
    pub fn guest_kernel(&self) -> &super::guest::GuestKernel<'a> {
        &self.kernel
    }

    // Map operations live on [`GuestMemMapAccessor`]. Borrow via
    // [`as_accessor`] to call them: `owned.as_accessor().find_map(...)`.
    // The wrapper type exists only to own the `GuestKernel` and
    // `BpfMapOffsets`; it does not duplicate the accessor's surface.
}
