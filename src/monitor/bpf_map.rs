//! Host-side BPF map discovery, read/write, and iteration via guest physical memory.
//!
//! Walks the kernel's `map_idr` xarray from the host, finds a BPF map
//! by name suffix, and provides read/write access to the map's value
//! region. No guest cooperation is needed — all reads go through the
//! guest physical memory mapping.
//!
//! Address translation strategy:
//! - `map_idr` is a kernel BSS symbol: use `text_kva_to_pa`.
//! - xa_node structs are SLAB-allocated (direct mapping): use `kva_to_pa`.
//! - bpf_map/bpf_array may be kmalloc'd or vmalloc'd: use `translate_any_kva`.
//! - .bss value region is vmalloc'd: use `translate_kva`.
//! - Per-CPU values (`BPF_MAP_TYPE_PERCPU_ARRAY`) are in the direct mapping:
//!   use `kva_to_pa` with `__per_cpu_offset[cpu]`.

use super::btf_offsets::BpfMapOffsets;
use super::idr::{translate_any_kva, xa_load};
use super::reader::GuestMem;
use super::symbols::text_kva_to_pa;

/// Bundle of borrow-held state every map-access routine threads
/// through the page-table walk, bounds check, and byte read/write path.
///
/// Every free function in this module previously took the same four-
/// to eight-argument fan of `mem`, `cr3_pa`, `page_offset`, `offsets`,
/// `l5` (some also took `map_idr_kva`); callers invariably forwarded
/// the same fields from their [`BpfMapAccessor`] because all six
/// originate on the accessor. Grouping them here drops the duplication
/// and lets additional shared context (per-CPU offset cache, BTF
/// cache, etc.) ride the same lifetime without touching every
/// signature.
pub(crate) struct AccessorCtx<'a> {
    pub mem: &'a GuestMem,
    pub cr3_pa: u64,
    pub page_offset: u64,
    pub offsets: &'a BpfMapOffsets,
    pub l5: bool,
}

/// BPF_MAP_TYPE_HASH from include/uapi/linux/bpf.h.
pub const BPF_MAP_TYPE_HASH: u32 = 1;

/// BPF_MAP_TYPE_ARRAY from include/uapi/linux/bpf.h.
pub const BPF_MAP_TYPE_ARRAY: u32 = 2;

/// BPF_MAP_TYPE_PERCPU_ARRAY from include/uapi/linux/bpf.h.
pub const BPF_MAP_TYPE_PERCPU_ARRAY: u32 = 6;

/// BPF_OBJ_NAME_LEN from include/linux/bpf.h.
const BPF_OBJ_NAME_LEN: usize = 16;

/// Discovered BPF map metadata and value location.
#[derive(Debug, Clone)]
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
    /// Guest KVA of the value region start (bpf_array.value).
    /// Only set for BPF_MAP_TYPE_ARRAY maps where the value data
    /// is inline at the `bpf_array.value` flex array offset.
    /// `None` for non-ARRAY map types.
    pub value_kva: Option<u64>,
    /// Guest KVA of the map's `struct btf`. 0 if the map has no BTF.
    pub btf_kva: u64,
    /// BTF type ID for the map's value type. 0 if the map has no BTF.
    pub btf_value_type_id: u32,
}

/// Enumerate all BPF maps in the kernel's `map_idr` xarray.
///
/// Returns metadata for every map whose KVA can be translated.
/// No filtering by type or name — callers select from the result.
///
/// `value_kva` is `Some` only for `BPF_MAP_TYPE_ARRAY` maps where
/// the value data is inline at the `bpf_array.value` flex array offset.
pub(crate) fn find_all_bpf_maps(ctx: &AccessorCtx<'_>, map_idr_kva: u64) -> Vec<BpfMapInfo> {
    let idr_pa = text_kva_to_pa(map_idr_kva);
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
            ctx.page_offset,
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

        let Some(map_pa) = translate_any_kva(ctx.mem, ctx.cr3_pa, ctx.page_offset, entry, ctx.l5)
        else {
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

        // value_kva is only meaningful for ARRAY maps where bpf_array
        // embeds bpf_map at offset 0 and the value flex array is inline.
        let value_kva = if map_type == BPF_MAP_TYPE_ARRAY {
            Some(entry + offsets.array_value as u64)
        } else {
            None
        };

        let btf_kva = ctx.mem.read_u64(map_pa, offsets.map_btf);
        let btf_value_type_id = ctx.mem.read_u32(map_pa, offsets.map_btf_value_type_id);

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

/// Write bytes to a BPF map's value region at `offset`.
///
/// Translates the value KVA (vmalloc'd for .bss maps) through the
/// page table to find the guest physical address, then writes directly.
/// Returns `false` if the map has no value KVA (non-ARRAY map),
/// `offset + data.len()` exceeds `value_size`, or any page in the
/// range is unmapped.
pub(crate) fn write_bpf_map_value(
    ctx: &AccessorCtx<'_>,
    map_info: &BpfMapInfo,
    offset: usize,
    data: &[u8],
) -> bool {
    let Some(base_kva) = map_info.value_kva else {
        return false;
    };
    if offset + data.len() > map_info.value_size as usize {
        return false;
    }
    let target_kva = base_kva + offset as u64;

    // Write byte-by-byte across potential page boundaries.
    for (i, &byte) in data.iter().enumerate() {
        let kva = target_kva + i as u64;
        let Some(pa) = ctx.mem.translate_kva(ctx.cr3_pa, kva, ctx.l5) else {
            return false;
        };
        ctx.mem.write_u8(pa, 0, byte);
    }
    true
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
/// is unmapped.
pub(crate) fn read_bpf_map_value(
    ctx: &AccessorCtx<'_>,
    map_info: &BpfMapInfo,
    offset: usize,
    len: usize,
) -> Option<Vec<u8>> {
    let base_kva = map_info.value_kva?;
    if offset + len > map_info.value_size as usize {
        return None;
    }
    let target_kva = base_kva + offset as u64;
    let mut buf = vec![0u8; len];

    // Read byte-by-byte across potential page boundaries.
    for (i, byte) in buf.iter_mut().enumerate() {
        let kva = target_kva + i as u64;
        let pa = ctx.mem.translate_kva(ctx.cr3_pa, kva, ctx.l5)?;
        *byte = ctx.mem.read_u8(pa, 0);
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

/// Maximum number of entries to iterate when walking a hash map.
/// Prevents unbounded iteration on corrupted or very large maps.
const HTAB_ITER_MAX: usize = 1_000_000;

/// Iterate all entries in a BPF_MAP_TYPE_HASH map, yielding (key, value)
/// byte pairs.
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
/// Returns an empty vec if the map is not `BPF_MAP_TYPE_HASH`, htab
/// offsets are unavailable, or the htab struct itself is untranslatable.
/// Untranslatable buckets are skipped; an untranslatable element breaks
/// the current bucket's chain and advances to the next bucket.
fn iter_htab_entries(ctx: &AccessorCtx<'_>, map: &BpfMapInfo) -> Vec<(Vec<u8>, Vec<u8>)> {
    if map.map_type != BPF_MAP_TYPE_HASH {
        return Vec::new();
    }
    let Some(htab) = &ctx.offsets.htab_offsets else {
        return Vec::new();
    };

    // bpf_htab embeds bpf_map at offset 0, so map_kva == htab_kva.
    let htab_kva = map.map_kva;

    // Read n_buckets and buckets pointer from the bpf_htab struct.
    let Some(htab_pa) = translate_any_kva(ctx.mem, ctx.cr3_pa, ctx.page_offset, htab_kva, ctx.l5)
    else {
        return Vec::new();
    };
    let n_buckets = ctx.mem.read_u32(htab_pa, htab.htab_n_buckets);
    let buckets_kva = ctx.mem.read_u64(htab_pa, htab.htab_buckets);
    if n_buckets == 0 || buckets_kva == 0 {
        return Vec::new();
    }

    let key_size = map.key_size as usize;
    let value_size = map.value_size as usize;
    // Value follows key at round_up(key_size, 8) within htab_elem.
    let value_off_in_elem = htab.htab_elem_size_base + ((key_size + 7) & !7);
    let key_off_in_elem = htab.htab_elem_size_base;

    let mut entries = Vec::new();
    let mut total_visited = 0usize;

    for i in 0..n_buckets {
        // bucket[i] is at buckets_kva + i * bucket_size.
        let bucket_kva = buckets_kva + (i as u64) * (htab.bucket_size as u64);
        let Some(bucket_pa) =
            translate_any_kva(ctx.mem, ctx.cr3_pa, ctx.page_offset, bucket_kva, ctx.l5)
        else {
            continue;
        };

        // Read hlist_nulls_head.first from the bucket.
        let first_ptr = ctx
            .mem
            .read_u64(bucket_pa, htab.bucket_head + htab.hlist_nulls_head_first);

        // Walk the hlist_nulls chain.
        let mut node_ptr = first_ptr;
        loop {
            // hlist_nulls termination: bit 0 set means end-of-list marker.
            if node_ptr & 1 != 0 || node_ptr == 0 {
                break;
            }
            total_visited += 1;
            if total_visited > HTAB_ITER_MAX {
                return entries;
            }

            // node_ptr is the KVA of the hlist_nulls_node, which is at
            // offset 0 of htab_elem (hash_node is first in the union).
            // So elem_kva == node_ptr.
            let elem_kva = node_ptr;
            let Some(elem_pa) =
                translate_any_kva(ctx.mem, ctx.cr3_pa, ctx.page_offset, elem_kva, ctx.l5)
            else {
                break;
            };

            // Read key bytes.
            let mut key_buf = vec![0u8; key_size];
            ctx.mem
                .read_bytes(elem_pa + key_off_in_elem as u64, &mut key_buf);

            // Read value bytes.
            let mut val_buf = vec![0u8; value_size];
            ctx.mem
                .read_bytes(elem_pa + value_off_in_elem as u64, &mut val_buf);

            entries.push((key_buf, val_buf));

            // Follow next pointer. hlist_nulls_node.next is at
            // hlist_nulls_node_next offset within the node.
            node_ptr = ctx.mem.read_u64(elem_pa, htab.hlist_nulls_node_next);
        }
    }

    entries
}

/// Read the per-CPU values for a single key in a `BPF_MAP_TYPE_PERCPU_ARRAY` map.
///
/// `bpf_array.pptrs[key]` holds a `__percpu` pointer. Adding
/// `__per_cpu_offset[cpu]` yields the per-CPU KVA, which resides in
/// the direct mapping.
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
    let Some(pptr_pa) = translate_any_kva(ctx.mem, ctx.cr3_pa, ctx.page_offset, pptr_kva, ctx.l5)
    else {
        return Vec::new();
    };
    let percpu_base = ctx.mem.read_u64(pptr_pa, 0);
    if percpu_base == 0 {
        return Vec::new();
    }

    let value_size = map.value_size as usize;
    let mut result = Vec::with_capacity(per_cpu_offsets.len());

    for &cpu_off in per_cpu_offsets {
        let cpu_kva = percpu_base.wrapping_add(cpu_off);
        let cpu_pa = super::symbols::kva_to_pa(cpu_kva, ctx.page_offset);
        if cpu_pa + value_size as u64 <= ctx.mem.size() {
            let mut buf = vec![0u8; value_size];
            ctx.mem.read_bytes(cpu_pa, &mut buf);
            result.push(Some(buf));
        } else {
            result.push(None);
        }
    }

    result
}

/// Typed value read from or written to a BPF map field.
///
/// The variant must match the [`BpfFieldKind`] of the target field.
/// `write_field` returns `false` on mismatch.
#[derive(Debug, Clone, PartialEq)]
pub enum BpfValue {
    /// 1-byte boolean.
    Bool(bool),
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    /// Opaque byte buffer for non-standard sizes or unknown types.
    Bytes(Vec<u8>),
}

/// Discriminant for a BPF map field's type, resolved from BTF.
///
/// Determined by chasing the field's BTF type chain (through Volatile,
/// Const, Typedef, TypeTag, Restrict) to the underlying Int or Enum.
/// Falls back to [`Bytes`](Self::Bytes) for non-standard sizes.
#[derive(Debug, Clone, PartialEq)]
pub enum BpfFieldKind {
    /// 1-byte boolean.
    Bool,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    /// Opaque byte buffer; the `usize` is the field size in bytes.
    Bytes(usize),
}

/// Metadata for a single field within a BPF map's value struct.
///
/// Resolved from BTF by [`BpfValueLayout::from_btf`]. Bitfields and
/// unnamed fields are skipped.
#[derive(Debug, Clone)]
pub struct BpfFieldInfo {
    /// Field name from BTF.
    pub name: String,
    /// Byte offset from the start of the value region.
    pub offset: usize,
    /// Field size in bytes.
    pub size: usize,
    /// Resolved type discriminant.
    pub kind: BpfFieldKind,
}

/// Layout of a BPF map's value type, resolved from the map's BTF.
///
/// Built by [`from_btf`](Self::from_btf) which chases modifier chains
/// to reach the underlying Struct and extracts field metadata.
#[derive(Debug, Clone)]
pub struct BpfValueLayout {
    /// Fields in declaration order. Bitfields and unnamed fields are excluded.
    pub fields: Vec<BpfFieldInfo>,
    /// Total size of the value struct in bytes (from BTF `size()`).
    pub total_size: usize,
}

impl BpfValueLayout {
    /// Build a layout from a btf_rs Btf and a type ID.
    ///
    /// Chases Volatile/Const/Typedef/TypeTag/Restrict chains to reach
    /// the underlying Struct, then extracts field offsets and types.
    /// Returns `None` if the type ID does not resolve to a Struct or
    /// any named, byte-aligned member has an unresolvable type.
    pub fn from_btf(btf: &btf_rs::Btf, type_id: u32) -> Option<Self> {
        let s = resolve_to_struct(btf, type_id)?;
        let total_size = s.size();
        let mut fields = Vec::new();

        for member in &s.members {
            let name = btf.resolve_name(member).unwrap_or_default();
            if name.is_empty() {
                continue;
            }
            let bits = member.bit_offset();
            if bits % 8 != 0 {
                continue;
            }
            let offset = (bits / 8) as usize;
            let (kind, size) = resolve_field_kind(btf, member)?;
            fields.push(BpfFieldInfo {
                name,
                offset,
                size,
                kind,
            });
        }

        Some(BpfValueLayout { fields, total_size })
    }

    /// Find a field by name.
    pub fn field(&self, name: &str) -> Option<&BpfFieldInfo> {
        self.fields.iter().find(|f| f.name == name)
    }
}

/// Chase modifiers (Volatile, Const, Typedef, TypeTag, Restrict),
/// pointers, and typedefs from `type_id` to find a Struct.
///
/// Returns `None` if the chain ends in a non-Struct type or exceeds
/// depth 20. Also resolves through Ptr (for pointer-to-struct members).
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

/// Determine the BpfFieldKind and byte size for a struct member by
/// chasing its type chain to the underlying Int or falling back to Bytes.
fn resolve_field_kind(btf: &btf_rs::Btf, member: &btf_rs::Member) -> Option<(BpfFieldKind, usize)> {
    let mut t = btf.resolve_chained_type(member).ok()?;
    for _ in 0..20 {
        match t {
            btf_rs::Type::Int(ref i) => {
                let size = i.size();
                let kind = if i.is_bool() {
                    BpfFieldKind::Bool
                } else if i.is_signed() {
                    match size {
                        1 => BpfFieldKind::I8,
                        2 => BpfFieldKind::I16,
                        4 => BpfFieldKind::I32,
                        8 => BpfFieldKind::I64,
                        _ => BpfFieldKind::Bytes(size),
                    }
                } else {
                    match size {
                        1 => BpfFieldKind::U8,
                        2 => BpfFieldKind::U16,
                        4 => BpfFieldKind::U32,
                        8 => BpfFieldKind::U64,
                        _ => BpfFieldKind::Bytes(size),
                    }
                };
                return Some((kind, size));
            }
            btf_rs::Type::Enum(ref e) => {
                let size = e.size();
                let kind = if e.is_signed() {
                    match size {
                        1 => BpfFieldKind::I8,
                        2 => BpfFieldKind::I16,
                        4 => BpfFieldKind::I32,
                        8 => BpfFieldKind::I64,
                        _ => BpfFieldKind::Bytes(size),
                    }
                } else {
                    match size {
                        1 => BpfFieldKind::U8,
                        2 => BpfFieldKind::U16,
                        4 => BpfFieldKind::U32,
                        8 => BpfFieldKind::U64,
                        _ => BpfFieldKind::Bytes(size),
                    }
                };
                return Some((kind, size));
            }
            btf_rs::Type::Volatile(_)
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

/// Read a typed field from a BPF map's value region.
fn read_typed_field(
    ctx: &AccessorCtx<'_>,
    map: &BpfMapInfo,
    field: &BpfFieldInfo,
) -> Option<BpfValue> {
    let bytes = read_bpf_map_value(ctx, map, field.offset, field.size)?;
    Some(match &field.kind {
        BpfFieldKind::Bool => BpfValue::Bool(bytes[0] != 0),
        BpfFieldKind::U8 => BpfValue::U8(bytes[0]),
        BpfFieldKind::I8 => BpfValue::I8(bytes[0] as i8),
        BpfFieldKind::U16 => BpfValue::U16(u16::from_ne_bytes(bytes[..2].try_into().ok()?)),
        BpfFieldKind::I16 => BpfValue::I16(i16::from_ne_bytes(bytes[..2].try_into().ok()?)),
        BpfFieldKind::U32 => BpfValue::U32(u32::from_ne_bytes(bytes[..4].try_into().ok()?)),
        BpfFieldKind::I32 => BpfValue::I32(i32::from_ne_bytes(bytes[..4].try_into().ok()?)),
        BpfFieldKind::U64 => BpfValue::U64(u64::from_ne_bytes(bytes[..8].try_into().ok()?)),
        BpfFieldKind::I64 => BpfValue::I64(i64::from_ne_bytes(bytes[..8].try_into().ok()?)),
        BpfFieldKind::Bytes(_) => BpfValue::Bytes(bytes),
    })
}

/// Write a typed field to a BPF map's value region.
fn write_typed_field(
    ctx: &AccessorCtx<'_>,
    map: &BpfMapInfo,
    field: &BpfFieldInfo,
    val: BpfValue,
) -> bool {
    let bytes: Vec<u8> = match (&field.kind, &val) {
        (BpfFieldKind::Bool, BpfValue::Bool(v)) => vec![*v as u8],
        (BpfFieldKind::U8, BpfValue::U8(v)) => vec![*v],
        (BpfFieldKind::I8, BpfValue::I8(v)) => vec![*v as u8],
        (BpfFieldKind::U16, BpfValue::U16(v)) => v.to_ne_bytes().to_vec(),
        (BpfFieldKind::I16, BpfValue::I16(v)) => v.to_ne_bytes().to_vec(),
        (BpfFieldKind::U32, BpfValue::U32(v)) => v.to_ne_bytes().to_vec(),
        (BpfFieldKind::I32, BpfValue::I32(v)) => v.to_ne_bytes().to_vec(),
        (BpfFieldKind::U64, BpfValue::U64(v)) => v.to_ne_bytes().to_vec(),
        (BpfFieldKind::I64, BpfValue::I64(v)) => v.to_ne_bytes().to_vec(),
        (BpfFieldKind::Bytes(n), BpfValue::Bytes(v)) if v.len() == *n => v.clone(),
        _ => return false,
    };
    write_bpf_map_value(ctx, map, field.offset, &bytes)
}

/// Host-side BPF map accessor for a running guest VM.
///
/// Resolves BTF offsets for BPF map structures and provides
/// map discovery and value read/write. Uses a [`GuestKernel`]
/// for address translation.
///
/// [`GuestKernel`]: super::guest::GuestKernel
pub struct BpfMapAccessor<'a> {
    kernel: &'a super::guest::GuestKernel<'a>,
    map_idr_kva: u64,
    /// Borrowed from the `BpfMapAccessorOwned` that produced this
    /// accessor via `as_accessor`, or provided by the caller to
    /// `from_guest_kernel`. Borrowing avoids the ~160-byte
    /// `BpfMapOffsets` clone that the old owned-field design paid
    /// on every `as_accessor()` call.
    offsets: &'a BpfMapOffsets,
}

impl<'a> BpfMapAccessor<'a> {
    /// Create from an existing [`GuestKernel`] and a caller-owned
    /// [`BpfMapOffsets`].
    ///
    /// The accessor borrows the offsets for its lifetime, so callers
    /// typically stash them in a `BpfMapAccessorOwned` (or another
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

    /// Build the [`AccessorCtx`] used by every map-read/write routine.
    fn ctx(&self) -> AccessorCtx<'_> {
        AccessorCtx {
            mem: self.kernel.mem(),
            cr3_pa: self.kernel.cr3_pa(),
            page_offset: self.kernel.page_offset(),
            offsets: self.offsets,
            l5: self.kernel.l5(),
        }
    }

    /// Enumerate all BPF maps in the kernel's `map_idr`.
    ///
    /// Returns metadata for every map whose KVA can be translated.
    /// No filtering by type or name.
    pub fn maps(&self) -> Vec<BpfMapInfo> {
        find_all_bpf_maps(&self.ctx(), self.map_idr_kva)
    }

    /// Find the first BPF ARRAY map whose name ends with `name_suffix`.
    ///
    /// Only returns `BPF_MAP_TYPE_ARRAY` maps. Use [`maps`](Self::maps)
    /// to enumerate maps of all types.
    pub fn find_map(&self, name_suffix: &str) -> Option<BpfMapInfo> {
        find_bpf_map(&self.ctx(), self.map_idr_kva, name_suffix)
    }

    /// Read bytes from a map's value region.
    ///
    /// Returns `None` if the map has no value KVA (non-ARRAY map)
    /// or any page in the range is unmapped.
    pub fn read_value(&self, map: &BpfMapInfo, offset: usize, len: usize) -> Option<Vec<u8>> {
        read_bpf_map_value(&self.ctx(), map, offset, len)
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

    /// Iterate all entries in a `BPF_MAP_TYPE_HASH` map.
    pub fn iter_hash_map(&self, map: &BpfMapInfo) -> Vec<(Vec<u8>, Vec<u8>)> {
        iter_htab_entries(&self.ctx(), map)
    }

    /// Read per-CPU values for a key in a `BPF_MAP_TYPE_PERCPU_ARRAY` map.
    ///
    /// Returns one entry per CPU, indexed by CPU number. `Some(bytes)`
    /// when the per-CPU PA falls within guest memory; `None` when it
    /// does not. Resolves `__per_cpu_offset` from the guest kernel.
    ///
    /// Returns an empty vec if the map is not `BPF_MAP_TYPE_PERCPU_ARRAY`,
    /// `key >= max_entries`, or the `__per_cpu_offset` symbol is missing.
    pub fn read_percpu_array(
        &self,
        map: &BpfMapInfo,
        key: u32,
        num_cpus: u32,
    ) -> Vec<Option<Vec<u8>>> {
        let Some(pco_kva) = self.kernel.symbol_kva("__per_cpu_offset") else {
            return Vec::new();
        };
        let pco_pa = super::symbols::text_kva_to_pa(pco_kva);
        // read_per_cpu_offsets routes through GuestMem::read_u64 which
        // bounds-checks each element against the mapped size; out-of-
        // range PAs yield 0 rather than faulting. No pre-check needed.
        let per_cpu_offsets =
            super::symbols::read_per_cpu_offsets(self.kernel.mem(), pco_pa, num_cpus);

        read_percpu_array_value(&self.ctx(), map, key, &per_cpu_offsets)
    }

    /// Resolve the value layout from the map's BTF.
    ///
    /// Reads `struct btf` from guest memory (kmalloc'd, direct mapping),
    /// then reads `btf->data` (kvmalloc'd, page table walk), parses
    /// it with `btf_rs::Btf::from_bytes`, and resolves the value type.
    ///
    /// Returns `None` if the map has no BTF (`btf_kva == 0` or
    /// `btf_value_type_id == 0`), the BTF data cannot be read, or
    /// the value type cannot be resolved.
    pub fn resolve_value_layout(&self, map: &BpfMapInfo) -> Option<BpfValueLayout> {
        if map.btf_kva == 0 || map.btf_value_type_id == 0 {
            return None;
        }

        // struct btf is kmalloc'd (SLAB, direct mapping).
        let data_kva = self
            .kernel
            .read_direct_u64(map.btf_kva + self.offsets.btf_data as u64);
        let data_size = self
            .kernel
            .read_direct_u32(map.btf_kva + self.offsets.btf_data_size as u64);

        if data_kva == 0 || data_size == 0 {
            return None;
        }

        // btf->data is kvmalloc'd — could be direct or vmalloc.
        // Use page table walk which handles both.
        let raw_btf = self.kernel.read_kva_bytes(data_kva, data_size as usize)?;

        let btf = btf_rs::Btf::from_bytes(&raw_btf).ok()?;
        BpfValueLayout::from_btf(&btf, map.btf_value_type_id)
    }

    /// Read a typed field from a map's value region.
    ///
    /// Returns `None` if the field is not found in the layout, the map
    /// has no value KVA, or any page in the field's range is unmapped.
    pub fn read_field(
        &self,
        map: &BpfMapInfo,
        layout: &BpfValueLayout,
        field: &str,
    ) -> Option<BpfValue> {
        let fi = layout.field(field)?;
        read_typed_field(&self.ctx(), map, fi)
    }

    /// Write a typed field to a map's value region.
    ///
    /// Returns `false` if the field is not found, the value type
    /// doesn't match the field's kind, or the write fails.
    pub fn write_field(
        &self,
        map: &BpfMapInfo,
        layout: &BpfValueLayout,
        field: &str,
        val: BpfValue,
    ) -> bool {
        let Some(fi) = layout.field(field) else {
            return false;
        };
        write_typed_field(&self.ctx(), map, fi, val)
    }
}

/// Owns a [`GuestKernel`] and provides BPF map access.
///
/// Returned by [`BpfMapAccessorOwned::new`] which builds the
/// `GuestKernel` internally. Borrow as [`BpfMapAccessor`] via
/// [`as_accessor`](Self::as_accessor) for map operations.
///
/// [`GuestKernel`]: super::guest::GuestKernel
pub struct BpfMapAccessorOwned<'a> {
    kernel: super::guest::GuestKernel<'a>,
    map_idr_kva: u64,
    offsets: BpfMapOffsets,
}

impl<'a> BpfMapAccessorOwned<'a> {
    /// Create from GuestMem and vmlinux path.
    ///
    /// One-shot constructor: builds a [`GuestKernel`] from `vmlinux`,
    /// parses BTF to resolve the map-related struct offsets, and
    /// locates the `map_idr` symbol. The resulting handle owns both
    /// the `GuestKernel` and the `BpfMapOffsets`.
    ///
    /// Prefer [`BpfMapAccessor::from_guest_kernel`] when you already
    /// hold a `GuestKernel` **and** a pre-built `&BpfMapOffsets` — it
    /// builds a borrowed accessor without taking ownership of either,
    /// so callers that maintain their own offsets cache (e.g. across
    /// multiple map probes in the same poll cycle) don't pay repeat
    /// BTF parses. `new` is the convenience path when you want the
    /// accessor to own its offsets.
    ///
    /// [`GuestKernel`]: super::guest::GuestKernel
    pub fn new(mem: &'a GuestMem, vmlinux: &std::path::Path) -> anyhow::Result<Self> {
        let kernel = super::guest::GuestKernel::new(mem, vmlinux)?;
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

    /// Borrow as a [`BpfMapAccessor`] for map operations.
    ///
    /// The returned accessor borrows `self.offsets`; no clone on
    /// the hot path.
    pub fn as_accessor(&self) -> BpfMapAccessor<'_> {
        BpfMapAccessor {
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

    // Map operations live on [`BpfMapAccessor`]. Borrow via
    // [`as_accessor`] to call them: `owned.as_accessor().find_map(...)`.
    // The wrapper type exists only to own the `GuestKernel` and
    // `BpfMapOffsets`; it does not duplicate the accessor's surface.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::idr::{XA_CHUNK_SIZE, xa_node_shift};
    use crate::monitor::symbols::START_KERNEL_MAP;

    /// Test-only alias: many value-I/O tests don't thread an
    /// `&BpfMapOffsets` through, because `read_value` / `write_value`
    /// never touch one. Build the full [`AccessorCtx`] by borrowing
    /// [`BpfMapOffsets::EMPTY`] so those call sites stay terse.
    fn value_ctx<'a>(mem: &'a GuestMem, cr3_pa: u64, l5: bool) -> AccessorCtx<'a> {
        AccessorCtx {
            mem,
            cr3_pa,
            page_offset: 0,
            offsets: &BpfMapOffsets::EMPTY,
            l5,
        }
    }

    fn lookup_ctx<'a>(
        mem: &'a GuestMem,
        cr3_pa: u64,
        page_offset: u64,
        offsets: &'a BpfMapOffsets,
        l5: bool,
    ) -> AccessorCtx<'a> {
        AccessorCtx {
            mem,
            cr3_pa,
            page_offset,
            offsets,
            l5,
        }
    }

    // On aarch64, page table entries contain GPAs starting at DRAM_START.
    // The walker subtracts DRAM_START to produce GuestMem offsets. Test
    // page table entries must include this base so the subtraction yields
    // the correct buffer offset.
    #[cfg(target_arch = "x86_64")]
    const PTE_BASE: u64 = 0;
    #[cfg(target_arch = "aarch64")]
    const PTE_BASE: u64 = crate::vmm::kvm::DRAM_START;

    // Huge page (block) descriptor flags differ by architecture.
    // x86: PS(0x80) | present | rw | accessed | dirty = 0xE3.
    // aarch64: block descriptor bits [1:0] = 0b01 = 0x01.
    #[cfg(target_arch = "x86_64")]
    const BLOCK_FLAGS: u64 = 0xE3;
    #[cfg(target_arch = "aarch64")]
    #[allow(dead_code)] // used when aarch64 huge page tests are added
    const BLOCK_FLAGS: u64 = 0x01;

    // -- translate_kva tests --

    /// Build a minimal 4-level page table in a buffer, mapping a single
    /// 4KB page. Returns (buffer, cr3_pa, mapped_kva, mapped_pa).
    #[cfg(target_arch = "x86_64")]
    fn setup_page_table() -> (Vec<u8>, u64, u64, u64) {
        // Use a KVA and compute indices dynamically.
        let kva: u64 = 0xFFFF_8880_0000_5000;
        let pgd_idx = (kva >> 39) & 0x1FF;
        let pud_idx = (kva >> 30) & 0x1FF;
        let pmd_idx = (kva >> 21) & 0x1FF;
        let pte_idx = (kva >> 12) & 0x1FF;

        // Page table pages at fixed PAs. PGD needs to be large enough
        // for the highest index entry.
        let pgd_pa: u64 = 0x10000; // 64KB — enough for any index * 8
        let pud_pa: u64 = pgd_pa + 0x1000;
        let pmd_pa: u64 = pud_pa + 0x1000;
        let pte_pa: u64 = pmd_pa + 0x1000;
        let data_pa: u64 = pte_pa + 0x1000;

        let size = (data_pa + 0x1000) as usize;
        let mut buf = vec![0u8; size];

        let write_entry = |buf: &mut Vec<u8>, base: u64, idx: u64, val: u64| {
            let off = (base + idx * 8) as usize;
            buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
        };

        write_entry(&mut buf, pgd_pa, pgd_idx, (pud_pa + PTE_BASE) | 0x63);
        write_entry(&mut buf, pud_pa, pud_idx, (pmd_pa + PTE_BASE) | 0x63);
        write_entry(&mut buf, pmd_pa, pmd_idx, (pte_pa + PTE_BASE) | 0x63);
        write_entry(&mut buf, pte_pa, pte_idx, (data_pa + PTE_BASE) | 0x63);

        // Write known data at the target page.
        buf[data_pa as usize..data_pa as usize + 8]
            .copy_from_slice(&0xDEAD_BEEF_CAFE_1234u64.to_ne_bytes());

        (buf, pgd_pa, kva, data_pa)
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn translate_kva_basic() {
        let (buf, cr3_pa, kva, data_pa) = setup_page_table();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let pa = mem.translate_kva(cr3_pa, kva, false);
        assert_eq!(pa, Some(data_pa));
        // Read through the translated PA to verify correctness.
        assert_eq!(mem.read_u64(pa.unwrap(), 0), 0xDEAD_BEEF_CAFE_1234);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn translate_kva_with_offset() {
        let (buf, cr3_pa, kva, data_pa) = setup_page_table();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        // KVA + 0x100 should map to data_pa + 0x100
        let pa = mem.translate_kva(cr3_pa, kva + 0x100, false);
        assert_eq!(pa, Some(data_pa + 0x100));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn translate_kva_unmapped() {
        let (buf, cr3_pa, _, _) = setup_page_table();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        // A completely different address that has no PGD entry.
        let pa = mem.translate_kva(cr3_pa, 0xFFFF_FFFF_8000_0000, false);
        assert_eq!(pa, None);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn translate_kva_unmapped_pte() {
        let (buf, cr3_pa, kva, _) = setup_page_table();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        // Same PGD/PUD/PMD but next PTE index — not mapped.
        let unmapped_kva = kva + 0x1000;
        let pa = mem.translate_kva(cr3_pa, unmapped_kva, false);
        assert_eq!(pa, None);
    }

    // -- translate_kva: 2MB huge page --

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn translate_kva_2mb_huge_page() {
        // Map KVA via a 2MB page (PS bit set in PMD entry).
        let kva: u64 = 0xFFFF_8880_0020_0000; // 2MB-aligned
        let pgd_idx = (kva >> 39) & 0x1FF;
        let pud_idx = (kva >> 30) & 0x1FF;
        let pmd_idx = (kva >> 21) & 0x1FF;

        let pgd_pa: u64 = 0x10000;
        let pud_pa: u64 = pgd_pa + 0x1000;
        let pmd_pa: u64 = pud_pa + 0x1000;
        let huge_page_pa: u64 = 0x20_0000; // 2MB-aligned physical page

        let size = (huge_page_pa + 0x20_0000) as usize; // room for the 2MB page
        let mut buf = vec![0u8; size];

        let write_entry = |buf: &mut Vec<u8>, base: u64, idx: u64, val: u64| {
            let off = (base + idx * 8) as usize;
            buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
        };

        // PGD -> PUD
        write_entry(&mut buf, pgd_pa, pgd_idx, (pud_pa + PTE_BASE) | 0x63);
        // PUD -> PMD
        write_entry(&mut buf, pud_pa, pud_idx, (pmd_pa + PTE_BASE) | 0x63);
        // PMD entry with PS bit set (bit 7) = 2MB huge page
        write_entry(
            &mut buf,
            pmd_pa,
            pmd_idx,
            (huge_page_pa + PTE_BASE) | BLOCK_FLAGS,
        ); // present+rw+PS

        // Write marker data at the huge page base.
        buf[huge_page_pa as usize..huge_page_pa as usize + 8]
            .copy_from_slice(&0xCAFE_BABE_1234_5678u64.to_ne_bytes());

        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let pa = mem.translate_kva(pgd_pa, kva, false);
        assert_eq!(pa, Some(huge_page_pa));
        assert_eq!(mem.read_u64(pa.unwrap(), 0), 0xCAFE_BABE_1234_5678);

        // Offset within the 2MB page.
        let pa_off = mem.translate_kva(pgd_pa, kva + 0x1000, false);
        assert_eq!(pa_off, Some(huge_page_pa + 0x1000));
    }

    // -- translate_kva: 1GB huge page --

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn translate_kva_1gb_huge_page() {
        // Map KVA via a 1GB page (PS bit set in PUD entry).
        let kva: u64 = 0xFFFF_8880_4000_0000; // 1GB-aligned
        let pgd_idx = (kva >> 39) & 0x1FF;
        let pud_idx = (kva >> 30) & 0x1FF;

        let pgd_pa: u64 = 0x10000;
        let pud_pa: u64 = pgd_pa + 0x1000;
        let huge_page_pa: u64 = 0x4000_0000; // 1GB-aligned

        // Buffer must be large enough to hold PGD + PUD. We don't need
        // the actual 1GB page in the buffer — just verify the PA math.
        let size = (pud_pa + 0x1000) as usize;
        let mut buf = vec![0u8; size];

        let write_entry = |buf: &mut Vec<u8>, base: u64, idx: u64, val: u64| {
            let off = (base + idx * 8) as usize;
            buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
        };

        // PGD -> PUD
        write_entry(&mut buf, pgd_pa, pgd_idx, (pud_pa + PTE_BASE) | 0x63);
        // PUD entry with PS bit set = 1GB huge page
        write_entry(
            &mut buf,
            pud_pa,
            pud_idx,
            (huge_page_pa + PTE_BASE) | BLOCK_FLAGS,
        );

        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let pa = mem.translate_kva(pgd_pa, kva, false);
        assert_eq!(pa, Some(huge_page_pa));

        // Offset within the 1GB page.
        let pa_off = mem.translate_kva(pgd_pa, kva + 0x1234_5678, false);
        assert_eq!(pa_off, Some(huge_page_pa + 0x1234_5678));
    }

    // -- translate_kva: not-present at each level --

    #[test]
    fn translate_kva_pgd_not_present() {
        // PGD entry with present bit clear.
        let kva: u64 = 0xFFFF_8880_0000_5000;
        let pgd_idx = (kva >> 39) & 0x1FF;

        let pgd_pa: u64 = 0x10000;
        let size = (pgd_pa + 0x1000) as usize;
        let mut buf = vec![0u8; size];

        // Write PGD entry without present bit.
        let off = (pgd_pa + pgd_idx * 8) as usize;
        buf[off..off + 8].copy_from_slice(&0x2000u64.to_ne_bytes()); // no PRESENT

        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        assert_eq!(mem.translate_kva(pgd_pa, kva, false), None);
    }

    #[test]
    fn translate_kva_pud_not_present() {
        let kva: u64 = 0xFFFF_8880_0000_5000;
        let pgd_idx = (kva >> 39) & 0x1FF;
        let pud_idx = (kva >> 30) & 0x1FF;

        let pgd_pa: u64 = 0x10000;
        let pud_pa: u64 = pgd_pa + 0x1000;
        let size = (pud_pa + 0x1000) as usize;
        let mut buf = vec![0u8; size];

        let write_entry = |buf: &mut Vec<u8>, base: u64, idx: u64, val: u64| {
            let off = (base + idx * 8) as usize;
            buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
        };

        // PGD present -> PUD
        write_entry(&mut buf, pgd_pa, pgd_idx, (pud_pa + PTE_BASE) | 0x63);
        // PUD entry without present bit.
        write_entry(&mut buf, pud_pa, pud_idx, 0x3000); // no PRESENT

        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        assert_eq!(mem.translate_kva(pgd_pa, kva, false), None);
    }

    #[test]
    fn translate_kva_pmd_not_present() {
        let kva: u64 = 0xFFFF_8880_0000_5000;
        let pgd_idx = (kva >> 39) & 0x1FF;
        let pud_idx = (kva >> 30) & 0x1FF;
        let pmd_idx = (kva >> 21) & 0x1FF;

        let pgd_pa: u64 = 0x10000;
        let pud_pa: u64 = pgd_pa + 0x1000;
        let pmd_pa: u64 = pud_pa + 0x1000;
        let size = (pmd_pa + 0x1000) as usize;
        let mut buf = vec![0u8; size];

        let write_entry = |buf: &mut Vec<u8>, base: u64, idx: u64, val: u64| {
            let off = (base + idx * 8) as usize;
            buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
        };

        write_entry(&mut buf, pgd_pa, pgd_idx, (pud_pa + PTE_BASE) | 0x63);
        write_entry(&mut buf, pud_pa, pud_idx, (pmd_pa + PTE_BASE) | 0x63);
        // PMD entry without present bit.
        write_entry(&mut buf, pmd_pa, pmd_idx, 0x4000); // no PRESENT

        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        assert_eq!(mem.translate_kva(pgd_pa, kva, false), None);
    }

    #[test]
    fn translate_kva_pte_not_present() {
        let kva: u64 = 0xFFFF_8880_0000_5000;
        let pgd_idx = (kva >> 39) & 0x1FF;
        let pud_idx = (kva >> 30) & 0x1FF;
        let pmd_idx = (kva >> 21) & 0x1FF;
        let pte_idx = (kva >> 12) & 0x1FF;

        let pgd_pa: u64 = 0x10000;
        let pud_pa: u64 = pgd_pa + 0x1000;
        let pmd_pa: u64 = pud_pa + 0x1000;
        let pte_pa: u64 = pmd_pa + 0x1000;
        let size = (pte_pa + 0x1000) as usize;
        let mut buf = vec![0u8; size];

        let write_entry = |buf: &mut Vec<u8>, base: u64, idx: u64, val: u64| {
            let off = (base + idx * 8) as usize;
            buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
        };

        write_entry(&mut buf, pgd_pa, pgd_idx, (pud_pa + PTE_BASE) | 0x63);
        write_entry(&mut buf, pud_pa, pud_idx, (pmd_pa + PTE_BASE) | 0x63);
        write_entry(&mut buf, pmd_pa, pmd_idx, (pte_pa + PTE_BASE) | 0x63);
        // PTE entry without present bit.
        write_entry(&mut buf, pte_pa, pte_idx, 0x5000); // no PRESENT

        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        assert_eq!(mem.translate_kva(pgd_pa, kva, false), None);
    }

    // -- write_bpf_map_value tests --

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn write_bpf_map_value_u32_roundtrip() {
        let (mut buf, cr3_pa, kva, data_pa) = setup_page_table();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

        let info = BpfMapInfo {
            map_pa: 0,
            map_kva: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            key_size: 0,
            value_size: 64,
            max_entries: 0,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        // Write u32 at offset 4 within the value region.
        assert!(write_bpf_map_value_u32(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            4,
            0xABCD_1234,
        ));
        // Read it back via direct PA access.
        assert_eq!(mem.read_u32(data_pa, 4), 0xABCD_1234);
    }

    #[test]
    fn read_bytes_basic() {
        let buf = [1u8, 2, 3, 4, 5, 6, 7, 8];
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let mut out = [0u8; 4];
        let n = mem.read_bytes(2, &mut out);
        assert_eq!(n, 4);
        assert_eq!(out, [3, 4, 5, 6]);
    }

    #[test]
    fn read_bytes_past_end() {
        let buf = [1u8, 2, 3, 4];
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let mut out = [0u8; 8];
        let n = mem.read_bytes(2, &mut out);
        assert_eq!(n, 2); // Only 2 bytes available from PA 2.
        assert_eq!(out[..2], [3, 4]);
    }

    #[test]
    fn read_bytes_at_boundary() {
        let buf = [0xFFu8; 8];
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let mut out = [0u8; 8];
        let n = mem.read_bytes(8, &mut out);
        assert_eq!(n, 0); // PA == size, nothing to read.
    }

    #[test]
    fn write_u32_roundtrip() {
        let mut buf = [0u8; 16];
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        mem.write_u32(4, 0, 0xDEAD_BEEF);
        assert_eq!(mem.read_u32(4, 0), 0xDEAD_BEEF);
        assert_eq!(
            u32::from_ne_bytes(buf[4..8].try_into().unwrap()),
            0xDEAD_BEEF
        );
    }

    // -- xa_load tests --

    #[test]
    fn xa_load_zero_head() {
        let buf = [0u8; 64];
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        assert_eq!(xa_load(&mem, 0, 0, 0, 0, 0), Some(0));
        assert_eq!(xa_load(&mem, 0, 0, 5, 0, 0), Some(0));
    }

    #[test]
    fn xa_load_single_entry_index_zero() {
        // xa_head with bit 1 clear = single-entry xarray.
        // Only index 0 returns the head value; others return 0.
        let xa_head: u64 = 0xFFFF_8880_0001_0000; // bit 1 clear
        assert_eq!(xa_head & 2, 0);
        let buf = [0u8; 8];
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        assert_eq!(xa_load(&mem, 0, xa_head, 0, 0, 0), Some(xa_head));
    }

    #[test]
    fn xa_load_single_entry_index_nonzero() {
        let xa_head: u64 = 0xFFFF_8880_0001_0000;
        let buf = [0u8; 8];
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        assert_eq!(xa_load(&mem, 0, xa_head, 1, 0, 0), Some(0));
        assert_eq!(xa_load(&mem, 0, xa_head, 63, 0, 0), Some(0));
    }

    /// Build a single-level xa_node in a buffer. The node has shift=0
    /// (leaf level) and the given slots populated with entry pointers.
    /// Returns (buffer, xa_head pointing to the node, page_offset used).
    ///
    /// Layout: node at DRAM offset 0x1000, slots at 0x1000 + slots_off.
    /// kva_to_pa(node_kva, page_offset) = 0x1000.
    fn setup_xa_node(slots: &[(u64, u64)], slots_off: usize) -> (Vec<u8>, u64, u64) {
        let node_pa: u64 = 0x1000;
        let page_offset: u64 = crate::monitor::symbols::DEFAULT_PAGE_OFFSET;
        let node_kva = page_offset.wrapping_add(node_pa);

        let size = (node_pa as usize) + slots_off + XA_CHUNK_SIZE as usize * 8 + 8;
        let mut buf = vec![0u8; size];

        // xa_node.shift at offset 0 = 0 (leaf level).
        buf[node_pa as usize] = 0;

        // Populate slots.
        for &(idx, entry) in slots {
            let slot_pa = node_pa + slots_off as u64 + idx * 8;
            buf[slot_pa as usize..slot_pa as usize + 8].copy_from_slice(&entry.to_ne_bytes());
        }

        // xa_head = node_kva | 2 (internal node marker).
        let xa_head = node_kva | 2;
        (buf, xa_head, page_offset)
    }

    #[test]
    fn xa_load_multi_entry_populated_slot() {
        let slots_off = 16; // Simulated offset of slots within xa_node.
        let entry_ptr: u64 = 0xDEAD_0000; // Leaf entry (bit 1 clear).
        let (buf, xa_head, page_offset) = setup_xa_node(&[(3, entry_ptr)], slots_off);
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

        assert_eq!(
            xa_load(&mem, page_offset, xa_head, 3, slots_off, 0),
            Some(entry_ptr)
        );
    }

    #[test]
    fn xa_load_multi_entry_empty_slot() {
        let slots_off = 16;
        let (buf, xa_head, page_offset) = setup_xa_node(&[], slots_off);
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

        // All slots are zero.
        assert_eq!(
            xa_load(&mem, page_offset, xa_head, 0, slots_off, 0),
            Some(0)
        );
        assert_eq!(
            xa_load(&mem, page_offset, xa_head, 5, slots_off, 0),
            Some(0)
        );
    }

    #[test]
    fn xa_load_multi_entry_multiple_slots() {
        let slots_off = 16;
        let entries = [
            (0, 0xAAAA_0000u64),
            (7, 0xBBBB_0000u64),
            (63, 0xCCCC_0000u64),
        ];
        let (buf, xa_head, page_offset) = setup_xa_node(&entries, slots_off);
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

        assert_eq!(
            xa_load(&mem, page_offset, xa_head, 0, slots_off, 0),
            Some(0xAAAA_0000)
        );
        assert_eq!(
            xa_load(&mem, page_offset, xa_head, 7, slots_off, 0),
            Some(0xBBBB_0000)
        );
        assert_eq!(
            xa_load(&mem, page_offset, xa_head, 63, slots_off, 0),
            Some(0xCCCC_0000)
        );
        // Unpopulated slot.
        assert_eq!(
            xa_load(&mem, page_offset, xa_head, 1, slots_off, 0),
            Some(0)
        );
    }

    // -- find_bpf_map tests --

    /// Build a buffer with a mock IDR + bpf_map for find_bpf_map testing.
    ///
    /// Layout:
    /// - IDR at idr_pa (BSS region, translated via text_kva_to_pa)
    /// - bpf_map at map_pa (vmalloc'd, translated via page table walk)
    /// - Page table mapping map_kva -> map_pa
    #[cfg(target_arch = "x86_64")]
    fn setup_find_bpf_map(
        map_name: &str,
        map_type: u32,
        value_size: u32,
    ) -> (Vec<u8>, u64, u64, BpfMapOffsets) {
        // Offsets — use realistic values.
        let offsets = BpfMapOffsets {
            map_name: 32,
            map_type: 24,
            map_flags: 28,
            key_size: 44,
            value_size: 48,
            max_entries: 52,
            array_value: 256,
            xa_node_slots: 16,
            xa_node_shift: 0,
            idr_xa_head: 8,
            idr_next: 20,
            map_btf: 0,
            map_btf_value_type_id: 0,
            btf_data: 0,
            btf_data_size: 0,
            htab_offsets: None,
        };

        // Physical layout:
        // 0x0000..0x10000: padding / page tables
        // 0x10000: PGD
        // 0x11000: PUD
        // 0x12000: PMD
        // 0x13000: PTE
        // 0x14000: bpf_map/bpf_array data page
        // 0x15000: IDR data

        let pgd_pa: u64 = 0x10000;
        let pud_pa: u64 = 0x11000;
        let pmd_pa: u64 = 0x12000;
        let pte_pa: u64 = 0x13000;
        let map_pa: u64 = 0x14000;
        let idr_pa: u64 = 0x15000;

        // Choose a KVA for the bpf_map that will walk through our page table.
        let map_kva: u64 = 0xFFFF_C900_0000_0000;
        let pgd_idx = (map_kva >> 39) & 0x1FF;
        let pud_idx = (map_kva >> 30) & 0x1FF;
        let pmd_idx = (map_kva >> 21) & 0x1FF;
        let pte_idx = (map_kva >> 12) & 0x1FF;

        let size = 0x16000;
        let mut buf = vec![0u8; size];

        let write_u64 = |buf: &mut Vec<u8>, pa: u64, val: u64| {
            let off = pa as usize;
            buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
        };

        let write_u32 = |buf: &mut Vec<u8>, pa: u64, val: u32| {
            let off = pa as usize;
            buf[off..off + 4].copy_from_slice(&val.to_ne_bytes());
        };

        // Page table: PGD -> PUD -> PMD -> PTE -> map_pa.
        write_u64(&mut buf, pgd_pa + pgd_idx * 8, (pud_pa + PTE_BASE) | 0x63);
        write_u64(&mut buf, pud_pa + pud_idx * 8, (pmd_pa + PTE_BASE) | 0x63);
        write_u64(&mut buf, pmd_pa + pmd_idx * 8, (pte_pa + PTE_BASE) | 0x63);
        write_u64(&mut buf, pte_pa + pte_idx * 8, (map_pa + PTE_BASE) | 0x63);

        // IDR: xa_head is a single-entry xarray pointing directly to map_kva.
        // Single entry = bit 1 clear on map_kva (it has bit 1 clear: 0x...0000).
        write_u64(&mut buf, idr_pa + offsets.idr_xa_head as u64, map_kva);
        // idr_next = 1: one map at index 0.
        write_u32(&mut buf, idr_pa + offsets.idr_next as u64, 1);

        // bpf_map fields at map_pa.
        write_u32(&mut buf, map_pa + offsets.map_type as u64, map_type);
        write_u32(&mut buf, map_pa + offsets.value_size as u64, value_size);

        // Map name.
        let name_bytes = map_name.as_bytes();
        let name_pa = map_pa + offsets.map_name as u64;
        buf[name_pa as usize..name_pa as usize + name_bytes.len()].copy_from_slice(name_bytes);

        // IDR KVA: idr is in BSS, so text_kva_to_pa(idr_kva) = idr_pa.
        // text_kva_to_pa(kva) = kva - START_KERNEL_MAP.
        // So idr_kva = idr_pa + START_KERNEL_MAP.
        let start_kernel_map: u64 = START_KERNEL_MAP;
        let idr_kva = idr_pa + start_kernel_map;

        (buf, pgd_pa, idr_kva, offsets)
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn find_bpf_map_discovers_matching_map() {
        let (buf, cr3_pa, idr_kva, offsets) =
            setup_find_bpf_map("mitosis.bss", BPF_MAP_TYPE_ARRAY, 64);
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

        let result = find_bpf_map(
            &lookup_ctx(&mem, cr3_pa, 0xFFFF_8880_0000_0000, &offsets, false),
            idr_kva,
            ".bss",
        );

        let info = result.expect("should find the map");
        assert_eq!(info.name, "mitosis.bss");
        assert_eq!(info.map_type, BPF_MAP_TYPE_ARRAY);
        assert_eq!(info.value_size, 64);
        assert_eq!(info.map_pa, 0x14000);
        // value_kva = map_kva + array_value offset
        let map_kva: u64 = 0xFFFF_C900_0000_0000;
        assert_eq!(info.value_kva, Some(map_kva + offsets.array_value as u64));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn find_bpf_map_no_match_wrong_suffix() {
        let (buf, cr3_pa, idr_kva, offsets) =
            setup_find_bpf_map("mitosis.bss", BPF_MAP_TYPE_ARRAY, 64);
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

        let result = find_bpf_map(
            &lookup_ctx(&mem, cr3_pa, 0xFFFF_8880_0000_0000, &offsets, false),
            idr_kva,
            ".data",
        );
        assert!(result.is_none());
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn find_bpf_map_skips_non_array_type() {
        // map_type = 1 (BPF_MAP_TYPE_HASH), not BPF_MAP_TYPE_ARRAY.
        let (buf, cr3_pa, idr_kva, offsets) = setup_find_bpf_map("test.bss", 1, 64);
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

        let result = find_bpf_map(
            &lookup_ctx(&mem, cr3_pa, 0xFFFF_8880_0000_0000, &offsets, false),
            idr_kva,
            ".bss",
        );
        assert!(result.is_none());
    }

    #[test]
    fn find_bpf_map_empty_idr() {
        // IDR with xa_head = 0 (empty).
        let offsets = BpfMapOffsets {
            map_name: 32,
            map_type: 24,
            map_flags: 28,
            key_size: 44,
            value_size: 48,
            max_entries: 52,
            array_value: 256,
            xa_node_slots: 16,
            xa_node_shift: 0,
            idr_xa_head: 8,
            idr_next: 20,
            map_btf: 0,
            map_btf_value_type_id: 0,
            btf_data: 0,
            btf_data_size: 0,
            htab_offsets: None,
        };
        let idr_pa: u64 = 0x1000;
        let size = 0x2000;
        let buf = vec![0u8; size]; // All zeros, so xa_head = 0.

        let start_kernel_map: u64 = START_KERNEL_MAP;
        let idr_kva = idr_pa + start_kernel_map;

        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let result = find_bpf_map(
            &lookup_ctx(&mem, 0x10000, 0xFFFF_8880_0000_0000, &offsets, false),
            idr_kva,
            ".bss",
        );
        assert!(result.is_none());
    }

    // -- 5-level translate_kva tests --

    /// Build a 5-level page table mapping a single 4KB page.
    /// Returns (buffer, cr3_pa, mapped_kva, mapped_pa).
    #[cfg(target_arch = "x86_64")]
    fn setup_5level_page_table() -> (Vec<u8>, u64, u64, u64) {
        // Use a KVA with a non-zero PML5 index (bits 56:48).
        let kva: u64 = 0xFF11_8880_0000_5000;
        let pml5_idx = (kva >> 48) & 0x1FF;
        let pgd_idx = (kva >> 39) & 0x1FF;
        let pud_idx = (kva >> 30) & 0x1FF;
        let pmd_idx = (kva >> 21) & 0x1FF;
        let pte_idx = (kva >> 12) & 0x1FF;

        let pml5_pa: u64 = 0x10000;
        let p4d_pa: u64 = pml5_pa + 0x1000;
        let pud_pa: u64 = p4d_pa + 0x1000;
        let pmd_pa: u64 = pud_pa + 0x1000;
        let pte_pa: u64 = pmd_pa + 0x1000;
        let data_pa: u64 = pte_pa + 0x1000;

        let size = (data_pa + 0x1000) as usize;
        let mut buf = vec![0u8; size];

        let write_entry = |buf: &mut Vec<u8>, base: u64, idx: u64, val: u64| {
            let off = (base + idx * 8) as usize;
            buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
        };

        // PML5[pml5_idx] -> P4D
        write_entry(&mut buf, pml5_pa, pml5_idx, (p4d_pa + PTE_BASE) | 0x63);
        // P4D/PGD[pgd_idx] -> PUD
        write_entry(&mut buf, p4d_pa, pgd_idx, (pud_pa + PTE_BASE) | 0x63);
        // PUD[pud_idx] -> PMD
        write_entry(&mut buf, pud_pa, pud_idx, (pmd_pa + PTE_BASE) | 0x63);
        // PMD[pmd_idx] -> PTE
        write_entry(&mut buf, pmd_pa, pmd_idx, (pte_pa + PTE_BASE) | 0x63);
        // PTE[pte_idx] -> data page
        write_entry(&mut buf, pte_pa, pte_idx, (data_pa + PTE_BASE) | 0x63);

        // Write marker at data page.
        buf[data_pa as usize..data_pa as usize + 8]
            .copy_from_slice(&0x5555_AAAA_1234_5678u64.to_ne_bytes());

        (buf, pml5_pa, kva, data_pa)
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn translate_kva_5level_basic() {
        let (buf, cr3_pa, kva, data_pa) = setup_5level_page_table();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let pa = mem.translate_kva(cr3_pa, kva, true);
        assert_eq!(pa, Some(data_pa));
        assert_eq!(mem.read_u64(pa.unwrap(), 0), 0x5555_AAAA_1234_5678);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn translate_kva_5level_with_offset() {
        let (buf, cr3_pa, kva, data_pa) = setup_5level_page_table();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let pa = mem.translate_kva(cr3_pa, kva + 0x100, true);
        assert_eq!(pa, Some(data_pa + 0x100));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn translate_kva_5level_unmapped_pml5() {
        let (buf, cr3_pa, _, _) = setup_5level_page_table();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        // Different PML5 index — no entry mapped.
        let unmapped_kva: u64 = 0xFF22_8880_0000_5000;
        assert_eq!(mem.translate_kva(cr3_pa, unmapped_kva, true), None);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn translate_kva_5level_vs_4level_same_buffer() {
        // With l5=false on the same buffer, the walk starts at PGD (which
        // is our PML5). The PGD index from a 4-level perspective differs,
        // so it should fail to find a mapping.
        let (buf, cr3_pa, kva, _) = setup_5level_page_table();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        // 4-level walk uses bits 47:39 for PGD, not bits 56:48 for PML5.
        // The PGD index into our PML5 table won't find the right entry.
        let pa_4level = mem.translate_kva(cr3_pa, kva, false);
        // Should either be None (unmapped) or a different PA than 5-level.
        let pa_5level = mem.translate_kva(cr3_pa, kva, true);
        assert_ne!(pa_4level, pa_5level);
    }

    // -- write_bpf_map_value byte-by-byte across pages --

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn write_bpf_map_value_bytes_roundtrip() {
        let (mut buf, cr3_pa, kva, data_pa) = setup_page_table();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

        let info = BpfMapInfo {
            map_pa: 0,
            map_kva: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            key_size: 0,
            value_size: 16,
            max_entries: 0,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        let payload = [0xDE, 0xAD, 0xBE, 0xEF];
        assert!(write_bpf_map_value(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            0,
            &payload
        ));

        // Verify each byte was written.
        for (i, &expected) in payload.iter().enumerate() {
            assert_eq!(buf[data_pa as usize + i], expected);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn write_bpf_map_value_fails_on_unmapped_kva() {
        let (mut buf, cr3_pa, _, _) = setup_page_table();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

        let info = BpfMapInfo {
            map_pa: 0,
            map_kva: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            key_size: 0,
            value_size: 16,
            max_entries: 0,
            value_kva: Some(0xFFFF_FFFF_8000_0000), // Unmapped KVA.
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        assert!(!write_bpf_map_value(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            0,
            &[0xFF]
        ));
    }

    // -- two-level xarray traversal --

    /// Build a two-level xarray: root xa_node (shift=6) with one child
    /// xa_node (shift=0) containing a leaf entry. Exercises the xa_load
    /// loop's descent through internal nodes and the shift decrement.
    ///
    /// Layout:
    ///   root node at PA 0x1000, shift=6
    ///   child node at PA 0x2000, shift=0
    ///   root.slots[child_slot] = child_kva | 2 (internal marker)
    ///   child.slots[leaf_slot] = leaf_entry (bit 1 clear)
    ///
    /// Index = (child_slot << 6) | leaf_slot.
    fn setup_two_level_xarray(
        child_slot: u64,
        leaf_slot: u64,
        leaf_entry: u64,
        slots_off: usize,
    ) -> (Vec<u8>, u64, u64) {
        let root_pa: u64 = 0x1000;
        let child_pa: u64 = 0x2000;
        let page_offset: u64 = crate::monitor::symbols::DEFAULT_PAGE_OFFSET;
        let root_kva = page_offset.wrapping_add(root_pa);
        let child_kva = page_offset.wrapping_add(child_pa);

        let size = (child_pa as usize) + slots_off + XA_CHUNK_SIZE as usize * 8 + 8;
        let mut buf = vec![0u8; size];

        // Root node: shift=6 (one level above leaf).
        buf[root_pa as usize] = 6;
        // Root slot[child_slot] -> child node (internal marker: bit 1 set).
        let root_slot_pa = root_pa + slots_off as u64 + child_slot * 8;
        buf[root_slot_pa as usize..root_slot_pa as usize + 8]
            .copy_from_slice(&(child_kva | 2).to_ne_bytes());

        // Child node: shift=0 (leaf level).
        buf[child_pa as usize] = 0;
        // Child slot[leaf_slot] -> leaf entry (bit 1 clear).
        let child_slot_pa = child_pa + slots_off as u64 + leaf_slot * 8;
        buf[child_slot_pa as usize..child_slot_pa as usize + 8]
            .copy_from_slice(&leaf_entry.to_ne_bytes());

        let xa_head = root_kva | 2;
        (buf, xa_head, page_offset)
    }

    #[test]
    fn xa_load_two_level_finds_leaf() {
        let slots_off = 16;
        let child_slot = 1u64; // Root slot index for the child node.
        let leaf_slot = 5u64; // Child slot index for the leaf entry.
        let leaf_entry: u64 = 0xBEEF_0000; // Leaf (bit 1 clear).
        let index = (child_slot << 6) | leaf_slot; // = 69.

        let (buf, xa_head, page_offset) =
            setup_two_level_xarray(child_slot, leaf_slot, leaf_entry, slots_off);
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

        assert_eq!(
            xa_load(&mem, page_offset, xa_head, index, slots_off, 0),
            Some(leaf_entry)
        );
    }

    #[test]
    fn xa_load_two_level_empty_child_slot() {
        let slots_off = 16;
        let child_slot = 2u64;
        let leaf_slot = 10u64;
        let leaf_entry: u64 = 0xAAAA_0000;

        let (buf, xa_head, page_offset) =
            setup_two_level_xarray(child_slot, leaf_slot, leaf_entry, slots_off);
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

        // Index that hits root slot 2, child slot 10 -> populated.
        let populated_idx = (child_slot << 6) | leaf_slot;
        assert_eq!(
            xa_load(&mem, page_offset, xa_head, populated_idx, slots_off, 0),
            Some(leaf_entry)
        );

        // Index that hits root slot 2, but a different child slot -> 0.
        let empty_child_idx = child_slot << 6;
        assert_eq!(
            xa_load(&mem, page_offset, xa_head, empty_child_idx, slots_off, 0),
            Some(0)
        );
    }

    #[test]
    fn xa_load_two_level_empty_root_slot() {
        let slots_off = 16;
        let (buf, xa_head, page_offset) = setup_two_level_xarray(3, 0, 0xDEAD_0000, slots_off);
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

        // Index that maps to root slot 0 (empty, child is at slot 3).
        let empty_root_idx = 5u64; // root slot = 5 >> 6 = 0 (wait, index < 64 => root slot 0).
        // Actually: slot_idx = (index >> shift) & 63 = (5 >> 6) & 63 = 0.
        // Root slot 0 is empty (child is at slot 3).
        assert_eq!(
            xa_load(&mem, page_offset, xa_head, empty_root_idx, slots_off, 0),
            Some(0)
        );
    }

    #[test]
    fn xa_load_two_level_high_index() {
        let slots_off = 16;
        // Child at root slot 63, leaf at child slot 63. Max index for 2-level = 63*64+63 = 4095.
        let (buf, xa_head, page_offset) = setup_two_level_xarray(63, 63, 0xFFFF_0000, slots_off);
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

        let max_index = (63 << 6) | 63; // 4095
        assert_eq!(
            xa_load(&mem, page_offset, xa_head, max_index, slots_off, 0),
            Some(0xFFFF_0000)
        );
    }

    // -- find_bpf_map: multiple IDR entries --

    /// Build a buffer with multiple maps in the IDR (via xa_node).
    /// First map has wrong name, second map matches.
    #[cfg(target_arch = "x86_64")]
    fn setup_find_bpf_map_multi() -> (Vec<u8>, u64, u64, BpfMapOffsets) {
        let offsets = BpfMapOffsets {
            map_name: 32,
            map_type: 24,
            map_flags: 28,
            key_size: 44,
            value_size: 48,
            max_entries: 52,
            array_value: 256,
            xa_node_slots: 16,
            xa_node_shift: 0,
            idr_xa_head: 8,
            idr_next: 20,
            map_btf: 0,
            map_btf_value_type_id: 0,
            btf_data: 0,
            btf_data_size: 0,
            htab_offsets: None,
        };

        // Physical layout:
        // 0x10000: PGD
        // 0x11000: PUD
        // 0x12000: PMD
        // 0x13000: PTE (maps map1_kva -> map1_pa and map2_kva -> map2_pa)
        // 0x14000: bpf_map 1 (wrong name)
        // 0x15000: bpf_map 2 (correct name)
        // 0x16000: IDR data
        // 0x17000: xa_node for IDR

        let pgd_pa: u64 = 0x10000;
        let pud_pa: u64 = 0x11000;
        let pmd_pa: u64 = 0x12000;
        let pte_pa: u64 = 0x13000;
        let map1_pa: u64 = 0x14000;
        let map2_pa: u64 = 0x15000;
        let idr_pa: u64 = 0x16000;
        let xa_node_pa: u64 = 0x17000;

        // Two distinct KVAs with different PTE indices.
        let map1_kva: u64 = 0xFFFF_C900_0000_0000;
        let map2_kva: u64 = 0xFFFF_C900_0000_1000;
        let pgd_idx = (map1_kva >> 39) & 0x1FF;
        let pud_idx = (map1_kva >> 30) & 0x1FF;
        let pmd_idx = (map1_kva >> 21) & 0x1FF;
        let pte1_idx = (map1_kva >> 12) & 0x1FF;
        let pte2_idx = (map2_kva >> 12) & 0x1FF;

        let page_offset: u64 = 0xFFFF_8880_0000_0000;
        let xa_node_kva = xa_node_pa + page_offset;

        let size = 0x18000;
        let mut buf = vec![0u8; size];

        let write_u64 = |buf: &mut Vec<u8>, pa: u64, val: u64| {
            let off = pa as usize;
            buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
        };

        let write_u32 = |buf: &mut Vec<u8>, pa: u64, val: u32| {
            let off = pa as usize;
            buf[off..off + 4].copy_from_slice(&val.to_ne_bytes());
        };

        // Page table.
        write_u64(&mut buf, pgd_pa + pgd_idx * 8, (pud_pa + PTE_BASE) | 0x63);
        write_u64(&mut buf, pud_pa + pud_idx * 8, (pmd_pa + PTE_BASE) | 0x63);
        write_u64(&mut buf, pmd_pa + pmd_idx * 8, (pte_pa + PTE_BASE) | 0x63);
        write_u64(&mut buf, pte_pa + pte1_idx * 8, (map1_pa + PTE_BASE) | 0x63);
        write_u64(&mut buf, pte_pa + pte2_idx * 8, (map2_pa + PTE_BASE) | 0x63);

        // xa_node at xa_node_pa: shift=0 (leaf), with two entries.
        buf[xa_node_pa as usize] = 0; // shift=0
        // Slot 0 -> map1_kva (leaf, bit 1 clear).
        write_u64(
            &mut buf,
            xa_node_pa + offsets.xa_node_slots as u64,
            map1_kva,
        );
        // Slot 1 -> map2_kva (leaf, bit 1 clear).
        write_u64(
            &mut buf,
            xa_node_pa + offsets.xa_node_slots as u64 + 8,
            map2_kva,
        );

        // IDR xa_head -> xa_node (internal marker: bit 1 set).
        write_u64(
            &mut buf,
            idr_pa + offsets.idr_xa_head as u64,
            xa_node_kva | 2,
        );
        // idr_next = 2: two maps at indices 0 and 1.
        write_u32(&mut buf, idr_pa + offsets.idr_next as u64, 2);

        // Map 1: "other.data", BPF_MAP_TYPE_ARRAY.
        write_u32(
            &mut buf,
            map1_pa + offsets.map_type as u64,
            BPF_MAP_TYPE_ARRAY,
        );
        write_u32(&mut buf, map1_pa + offsets.value_size as u64, 32);
        let name1 = b"other.data";
        let name1_pa = map1_pa + offsets.map_name as u64;
        buf[name1_pa as usize..name1_pa as usize + name1.len()].copy_from_slice(name1);

        // Map 2: "mitosis.bss", BPF_MAP_TYPE_ARRAY.
        write_u32(
            &mut buf,
            map2_pa + offsets.map_type as u64,
            BPF_MAP_TYPE_ARRAY,
        );
        write_u32(&mut buf, map2_pa + offsets.value_size as u64, 128);
        let name2 = b"mitosis.bss";
        let name2_pa = map2_pa + offsets.map_name as u64;
        buf[name2_pa as usize..name2_pa as usize + name2.len()].copy_from_slice(name2);

        let start_kernel_map: u64 = START_KERNEL_MAP;
        let idr_kva = idr_pa + start_kernel_map;

        (buf, pgd_pa, idr_kva, offsets)
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn find_bpf_map_skips_wrong_name_finds_second() {
        let (buf, cr3_pa, idr_kva, offsets) = setup_find_bpf_map_multi();
        let page_offset: u64 = 0xFFFF_8880_0000_0000;
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

        let result = find_bpf_map(
            &lookup_ctx(&mem, cr3_pa, page_offset, &offsets, false),
            idr_kva,
            ".bss",
        );
        let info = result.expect("should find second map");
        assert_eq!(info.name, "mitosis.bss");
        assert_eq!(info.map_pa, 0x15000);
        assert_eq!(info.value_size, 128);
    }

    // -- find_bpf_map with full-length name (no null terminator) --

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn find_bpf_map_full_length_name() {
        // Map name fills all BPF_OBJ_NAME_LEN bytes with no null.
        let full_name = "0123456789a.bss"; // 15 bytes, fits in 16 with null.
        let (buf, cr3_pa, idr_kva, offsets) = setup_find_bpf_map(full_name, BPF_MAP_TYPE_ARRAY, 64);
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

        let result = find_bpf_map(
            &lookup_ctx(&mem, cr3_pa, 0xFFFF_8880_0000_0000, &offsets, false),
            idr_kva,
            ".bss",
        );
        let info = result.expect("should find map with 15-char name");
        assert_eq!(info.name, full_name);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn find_bpf_map_max_length_name_no_null() {
        // Exactly 16 bytes, no null terminator.
        let max_name = "0123456789a.bss!"; // 16 bytes
        assert_eq!(max_name.len(), BPF_OBJ_NAME_LEN);
        let (mut buf, cr3_pa, idr_kva, offsets) =
            setup_find_bpf_map("placeholder.bss", BPF_MAP_TYPE_ARRAY, 64);
        // Overwrite the name region with exactly 16 non-null bytes.
        let map_pa: u64 = 0x14000;
        let name_pa = (map_pa + offsets.map_name as u64) as usize;
        buf[name_pa..name_pa + 16].copy_from_slice(max_name.as_bytes());
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

        // The name doesn't end with ".bss" — the '!' is the 16th char.
        let result = find_bpf_map(
            &lookup_ctx(&mem, cr3_pa, 0xFFFF_8880_0000_0000, &offsets, false),
            idr_kva,
            ".bss",
        );
        assert!(
            result.is_none(),
            "16-byte name ending with '!' should not match .bss suffix"
        );
    }

    // -- write_bpf_map_value with nonzero offset --

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn write_bpf_map_value_nonzero_offset() {
        let (mut buf, cr3_pa, kva, data_pa) = setup_page_table();
        // Record the original bytes at data_pa before writing.
        let original_first_byte = buf[data_pa as usize];
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

        let info = BpfMapInfo {
            map_pa: 0,
            map_kva: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            key_size: 0,
            value_size: 64,
            max_entries: 0,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        // Write at offset 8 within the value region.
        let payload = [0x11, 0x22, 0x33, 0x44];
        assert!(write_bpf_map_value(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            8,
            &payload
        ));

        for (i, &expected) in payload.iter().enumerate() {
            assert_eq!(buf[data_pa as usize + 8 + i], expected);
        }
        // Bytes before offset should be untouched (still the marker data).
        assert_eq!(buf[data_pa as usize], original_first_byte);
    }

    // -- write_bpf_map_value with empty data --

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn write_bpf_map_value_empty_data() {
        let (mut buf, cr3_pa, kva, _) = setup_page_table();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

        let info = BpfMapInfo {
            map_pa: 0,
            map_kva: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            key_size: 0,
            value_size: 64,
            max_entries: 0,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        // Zero-length write should succeed without doing anything.
        assert!(write_bpf_map_value(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            0,
            &[]
        ));
    }

    // -- write_bpf_map_value_u32 with 5-level paging --

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn write_bpf_map_value_u32_5level() {
        let (mut buf, cr3_pa, kva, data_pa) = setup_5level_page_table();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

        let info = BpfMapInfo {
            map_pa: 0,
            map_kva: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            key_size: 0,
            value_size: 64,
            max_entries: 0,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        assert!(write_bpf_map_value_u32(
            &value_ctx(&mem, cr3_pa, true),
            &info,
            0,
            0xCAFE_BABE,
        ));
        assert_eq!(mem.read_u32(data_pa, 0), 0xCAFE_BABE);
    }

    // -- 5-level: not-present at P4D level --

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn translate_kva_5level_p4d_not_present() {
        // PML5 entry is present but the P4D (delegated to walk_4level as
        // PGD) has no entry for the requested PGD index.
        let kva: u64 = 0xFF11_8880_0000_5000;
        let pml5_idx = (kva >> 48) & 0x1FF;

        let pml5_pa: u64 = 0x10000;
        let p4d_pa: u64 = pml5_pa + 0x1000;

        // Buffer has PML5 -> P4D, but P4D is all zeros (no PGD entries).
        let size = (p4d_pa + 0x1000) as usize;
        let mut buf = vec![0u8; size];

        let off = (pml5_pa + pml5_idx * 8) as usize;
        buf[off..off + 8].copy_from_slice(&((p4d_pa + PTE_BASE) | 0x63).to_ne_bytes());

        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        assert_eq!(mem.translate_kva(pml5_pa, kva, true), None);
    }

    // -- 5-level: 2MB huge page --

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn translate_kva_5level_2mb_huge_page() {
        let kva: u64 = 0xFF11_8880_0020_0000; // 2MB-aligned
        let pml5_idx = (kva >> 48) & 0x1FF;
        let pgd_idx = (kva >> 39) & 0x1FF;
        let pud_idx = (kva >> 30) & 0x1FF;
        let pmd_idx = (kva >> 21) & 0x1FF;

        let pml5_pa: u64 = 0x10000;
        let p4d_pa: u64 = pml5_pa + 0x1000;
        let pud_pa: u64 = p4d_pa + 0x1000;
        let pmd_pa: u64 = pud_pa + 0x1000;
        let huge_page_pa: u64 = 0x20_0000;

        let size = (huge_page_pa + 0x20_0000) as usize;
        let mut buf = vec![0u8; size];

        let write_entry = |buf: &mut Vec<u8>, base: u64, idx: u64, val: u64| {
            let off = (base + idx * 8) as usize;
            buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
        };

        write_entry(&mut buf, pml5_pa, pml5_idx, (p4d_pa + PTE_BASE) | 0x63);
        write_entry(&mut buf, p4d_pa, pgd_idx, (pud_pa + PTE_BASE) | 0x63);
        write_entry(&mut buf, pud_pa, pud_idx, (pmd_pa + PTE_BASE) | 0x63);
        write_entry(
            &mut buf,
            pmd_pa,
            pmd_idx,
            (huge_page_pa + PTE_BASE) | BLOCK_FLAGS,
        ); // PS bit

        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let pa = mem.translate_kva(pml5_pa, kva, true);
        assert_eq!(pa, Some(huge_page_pa));

        let pa_off = mem.translate_kva(pml5_pa, kva + 0x1234, true);
        assert_eq!(pa_off, Some(huge_page_pa + 0x1234));
    }

    // -- 5-level: 1GB huge page --

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn translate_kva_5level_1gb_huge_page() {
        let kva: u64 = 0xFF11_8880_4000_0000; // 1GB-aligned
        let pml5_idx = (kva >> 48) & 0x1FF;
        let pgd_idx = (kva >> 39) & 0x1FF;
        let pud_idx = (kva >> 30) & 0x1FF;

        let pml5_pa: u64 = 0x10000;
        let p4d_pa: u64 = pml5_pa + 0x1000;
        let pud_pa: u64 = p4d_pa + 0x1000;
        let huge_page_pa: u64 = 0x4000_0000;

        let size = (pud_pa + 0x1000) as usize;
        let mut buf = vec![0u8; size];

        let write_entry = |buf: &mut Vec<u8>, base: u64, idx: u64, val: u64| {
            let off = (base + idx * 8) as usize;
            buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
        };

        write_entry(&mut buf, pml5_pa, pml5_idx, (p4d_pa + PTE_BASE) | 0x63);
        write_entry(&mut buf, p4d_pa, pgd_idx, (pud_pa + PTE_BASE) | 0x63);
        write_entry(
            &mut buf,
            pud_pa,
            pud_idx,
            (huge_page_pa + PTE_BASE) | BLOCK_FLAGS,
        ); // PS bit

        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let pa = mem.translate_kva(pml5_pa, kva, true);
        assert_eq!(pa, Some(huge_page_pa));

        let pa_off = mem.translate_kva(pml5_pa, kva + 0x1234_5678, true);
        assert_eq!(pa_off, Some(huge_page_pa + 0x1234_5678));
    }

    // -- find_bpf_map with translate_kva failure on first entry --

    #[test]
    fn find_bpf_map_skips_untranslatable_entry() {
        // IDR has a single entry whose KVA cannot be translated
        // (no page table mapping for it). find_bpf_map should continue
        // past it and return None (no other entries).
        let offsets = BpfMapOffsets {
            map_name: 32,
            map_type: 24,
            map_flags: 28,
            key_size: 44,
            value_size: 48,
            max_entries: 52,
            array_value: 256,
            xa_node_slots: 16,
            xa_node_shift: 0,
            idr_xa_head: 8,
            idr_next: 20,
            map_btf: 0,
            map_btf_value_type_id: 0,
            btf_data: 0,
            btf_data_size: 0,
            htab_offsets: None,
        };

        let idr_pa: u64 = 0x1000;
        let pgd_pa: u64 = 0x10000;
        let size = 0x12000;
        let mut buf = vec![0u8; size];

        // IDR xa_head = a KVA with no page table entry.
        // Single-entry xarray (bit 1 clear on the KVA).
        let unmappable_kva: u64 = 0xFFFF_C900_DEAD_0000;
        assert_eq!(unmappable_kva & 2, 0);
        let off = (idr_pa + offsets.idr_xa_head as u64) as usize;
        buf[off..off + 8].copy_from_slice(&unmappable_kva.to_ne_bytes());
        // idr_next = 1.
        let off_next = (idr_pa + offsets.idr_next as u64) as usize;
        buf[off_next..off_next + 4].copy_from_slice(&1u32.to_ne_bytes());

        // PGD exists but is all zeros — no entries.
        let start_kernel_map: u64 = START_KERNEL_MAP;
        let idr_kva = idr_pa + start_kernel_map;

        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let result = find_bpf_map(
            &lookup_ctx(&mem, pgd_pa, 0xFFFF_8880_0000_0000, &offsets, false),
            idr_kva,
            ".bss",
        );
        assert!(result.is_none());
    }

    // -- find_bpf_map with 5-level paging --

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn find_bpf_map_5level() {
        // Verify find_bpf_map works when l5=true by constructing a
        // 5-level page table mapping the bpf_map.
        let offsets = BpfMapOffsets {
            map_name: 32,
            map_type: 24,
            map_flags: 28,
            key_size: 44,
            value_size: 48,
            max_entries: 52,
            array_value: 256,
            xa_node_slots: 16,
            xa_node_shift: 0,
            idr_xa_head: 8,
            idr_next: 20,
            map_btf: 0,
            map_btf_value_type_id: 0,
            btf_data: 0,
            btf_data_size: 0,
            htab_offsets: None,
        };

        let map_kva: u64 = 0xFF11_C900_0000_0000;
        let pml5_idx = (map_kva >> 48) & 0x1FF;
        let pgd_idx = (map_kva >> 39) & 0x1FF;
        let pud_idx = (map_kva >> 30) & 0x1FF;
        let pmd_idx = (map_kva >> 21) & 0x1FF;
        let pte_idx = (map_kva >> 12) & 0x1FF;

        let pml5_pa: u64 = 0x10000;
        let p4d_pa: u64 = 0x11000;
        let pud_pa: u64 = 0x12000;
        let pmd_pa: u64 = 0x13000;
        let pte_pa: u64 = 0x14000;
        let map_pa: u64 = 0x15000;
        let idr_pa: u64 = 0x16000;

        let size = 0x17000;
        let mut buf = vec![0u8; size];

        let write_u64 = |buf: &mut Vec<u8>, pa: u64, val: u64| {
            let off = pa as usize;
            buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
        };
        let write_u32 = |buf: &mut Vec<u8>, pa: u64, val: u32| {
            let off = pa as usize;
            buf[off..off + 4].copy_from_slice(&val.to_ne_bytes());
        };

        // 5-level page table.
        write_u64(&mut buf, pml5_pa + pml5_idx * 8, (p4d_pa + PTE_BASE) | 0x63);
        write_u64(&mut buf, p4d_pa + pgd_idx * 8, (pud_pa + PTE_BASE) | 0x63);
        write_u64(&mut buf, pud_pa + pud_idx * 8, (pmd_pa + PTE_BASE) | 0x63);
        write_u64(&mut buf, pmd_pa + pmd_idx * 8, (pte_pa + PTE_BASE) | 0x63);
        write_u64(&mut buf, pte_pa + pte_idx * 8, (map_pa + PTE_BASE) | 0x63);

        // IDR: single-entry xarray.
        write_u64(&mut buf, idr_pa + offsets.idr_xa_head as u64, map_kva);
        // idr_next = 1.
        write_u32(&mut buf, idr_pa + offsets.idr_next as u64, 1);

        // bpf_map at map_pa.
        write_u32(
            &mut buf,
            map_pa + offsets.map_type as u64,
            BPF_MAP_TYPE_ARRAY,
        );
        write_u32(&mut buf, map_pa + offsets.value_size as u64, 96);
        let name = b"test.bss";
        let name_pa = (map_pa + offsets.map_name as u64) as usize;
        buf[name_pa..name_pa + name.len()].copy_from_slice(name);

        let start_kernel_map: u64 = START_KERNEL_MAP;
        let idr_kva = idr_pa + start_kernel_map;

        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let result = find_bpf_map(
            &lookup_ctx(&mem, pml5_pa, 0xFFFF_8880_0000_0000, &offsets, true),
            idr_kva,
            ".bss",
        );

        let info = result.expect("should find map via 5-level walk");
        assert_eq!(info.name, "test.bss");
        assert_eq!(info.map_pa, map_pa);
        assert_eq!(info.value_size, 96);
        assert_eq!(info.value_kva, Some(map_kva + offsets.array_value as u64));
    }

    // -- write_bpf_map_value across page boundary --

    /// Build a page table mapping two consecutive 4KB virtual pages to
    /// two physical pages. Returns (buffer, cr3_pa, base_kva, page1_pa, page2_pa).
    #[cfg(target_arch = "x86_64")]
    fn setup_two_page_table() -> (Vec<u8>, u64, u64, u64, u64) {
        let kva: u64 = 0xFFFF_8880_0000_5000;
        let kva2: u64 = kva + 0x1000;
        let pgd_idx = (kva >> 39) & 0x1FF;
        let pud_idx = (kva >> 30) & 0x1FF;
        let pmd_idx = (kva >> 21) & 0x1FF;
        let pte1_idx = (kva >> 12) & 0x1FF;
        let pte2_idx = (kva2 >> 12) & 0x1FF;

        let pgd_pa: u64 = 0x10000;
        let pud_pa: u64 = pgd_pa + 0x1000;
        let pmd_pa: u64 = pud_pa + 0x1000;
        let pte_pa: u64 = pmd_pa + 0x1000;
        let page1_pa: u64 = pte_pa + 0x1000;
        let page2_pa: u64 = page1_pa + 0x1000;

        let size = (page2_pa + 0x1000) as usize;
        let mut buf = vec![0u8; size];

        let write_entry = |buf: &mut Vec<u8>, base: u64, idx: u64, val: u64| {
            let off = (base + idx * 8) as usize;
            buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
        };

        write_entry(&mut buf, pgd_pa, pgd_idx, (pud_pa + PTE_BASE) | 0x63);
        write_entry(&mut buf, pud_pa, pud_idx, (pmd_pa + PTE_BASE) | 0x63);
        write_entry(&mut buf, pmd_pa, pmd_idx, (pte_pa + PTE_BASE) | 0x63);
        write_entry(&mut buf, pte_pa, pte1_idx, (page1_pa + PTE_BASE) | 0x63);
        write_entry(&mut buf, pte_pa, pte2_idx, (page2_pa + PTE_BASE) | 0x63);

        (buf, pgd_pa, kva, page1_pa, page2_pa)
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn write_bpf_map_value_across_page_boundary() {
        let (mut buf, cr3_pa, kva, page1_pa, page2_pa) = setup_two_page_table();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

        let info = BpfMapInfo {
            map_pa: 0,
            map_kva: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            key_size: 0,
            value_size: 0x2000,
            max_entries: 0,
            // value_kva at the start of page 1.
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        // Write a u32 at offset 0xFFE within the value region.
        // Bytes 0..2 land on page 1 (PA page1_pa + 0xFFE..0x1000),
        // bytes 2..4 land on page 2 (PA page2_pa + 0x000..0x002).
        let val: u32 = 0xAABB_CCDD;
        assert!(write_bpf_map_value_u32(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            0xFFE,
            val,
        ));

        // Verify bytes on page 1 (last 2 bytes of the page).
        let b = val.to_ne_bytes();
        assert_eq!(buf[page1_pa as usize + 0xFFE], b[0]);
        assert_eq!(buf[page1_pa as usize + 0xFFF], b[1]);
        // Verify bytes on page 2 (first 2 bytes).
        assert_eq!(buf[page2_pa as usize], b[2]);
        assert_eq!(buf[page2_pa as usize + 1], b[3]);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn write_bpf_map_value_single_byte_on_second_page() {
        let (mut buf, cr3_pa, kva, _, page2_pa) = setup_two_page_table();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

        let info = BpfMapInfo {
            map_pa: 0,
            map_kva: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            key_size: 0,
            value_size: 0x2000,
            max_entries: 0,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        // Write exactly at offset 0x1000 — first byte of page 2.
        assert!(write_bpf_map_value(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            0x1000,
            &[0x42],
        ));
        assert_eq!(buf[page2_pa as usize], 0x42);
    }

    // -- find_bpf_map: first entry untranslatable, second succeeds --

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn find_bpf_map_skips_untranslatable_finds_translatable() {
        let offsets = BpfMapOffsets {
            map_name: 32,
            map_type: 24,
            map_flags: 28,
            key_size: 44,
            value_size: 48,
            max_entries: 52,
            array_value: 256,
            xa_node_slots: 16,
            xa_node_shift: 0,
            idr_xa_head: 8,
            idr_next: 20,
            map_btf: 0,
            map_btf_value_type_id: 0,
            btf_data: 0,
            btf_data_size: 0,
            htab_offsets: None,
        };

        // Physical layout:
        // 0x10000: PGD
        // 0x11000: PUD
        // 0x12000: PMD
        // 0x13000: PTE (only maps map2_kva -> map2_pa; no entry for map1_kva)
        // 0x14000: bpf_map 2 (matching)
        // 0x15000: IDR data
        // 0x16000: xa_node

        let pgd_pa: u64 = 0x10000;
        let pud_pa: u64 = 0x11000;
        let pmd_pa: u64 = 0x12000;
        let pte_pa: u64 = 0x13000;
        let map2_pa: u64 = 0x14000;
        let idr_pa: u64 = 0x15000;
        let xa_node_pa: u64 = 0x16000;

        // map1_kva has no PTE entry; map2_kva does.
        let map1_kva: u64 = 0xFFFF_C900_0000_0000;
        let map2_kva: u64 = 0xFFFF_C900_0000_1000;
        let pgd_idx = (map2_kva >> 39) & 0x1FF;
        let pud_idx = (map2_kva >> 30) & 0x1FF;
        let pmd_idx = (map2_kva >> 21) & 0x1FF;
        let pte2_idx = (map2_kva >> 12) & 0x1FF;
        // map1_kva and map2_kva share PGD/PUD/PMD indices (they differ
        // only in bits 12..21). PTE index for map1_kva has no entry.

        let page_offset: u64 = 0xFFFF_8880_0000_0000;
        let xa_node_kva = xa_node_pa + page_offset;

        let size = 0x17000;
        let mut buf = vec![0u8; size];

        let write_u64 = |buf: &mut Vec<u8>, pa: u64, val: u64| {
            let off = pa as usize;
            buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
        };
        let write_u32 = |buf: &mut Vec<u8>, pa: u64, val: u32| {
            let off = pa as usize;
            buf[off..off + 4].copy_from_slice(&val.to_ne_bytes());
        };

        // Page table — only map2_kva is mapped.
        write_u64(&mut buf, pgd_pa + pgd_idx * 8, (pud_pa + PTE_BASE) | 0x63);
        write_u64(&mut buf, pud_pa + pud_idx * 8, (pmd_pa + PTE_BASE) | 0x63);
        write_u64(&mut buf, pmd_pa + pmd_idx * 8, (pte_pa + PTE_BASE) | 0x63);
        // Only PTE for map2_kva. map1_kva's PTE slot is zero (not present).
        write_u64(&mut buf, pte_pa + pte2_idx * 8, (map2_pa + PTE_BASE) | 0x63);

        // xa_node: slot 0 -> map1_kva (untranslatable), slot 1 -> map2_kva.
        buf[xa_node_pa as usize] = 0; // shift=0
        write_u64(
            &mut buf,
            xa_node_pa + offsets.xa_node_slots as u64,
            map1_kva,
        );
        write_u64(
            &mut buf,
            xa_node_pa + offsets.xa_node_slots as u64 + 8,
            map2_kva,
        );

        // IDR xa_head -> xa_node.
        write_u64(
            &mut buf,
            idr_pa + offsets.idr_xa_head as u64,
            xa_node_kva | 2,
        );
        // idr_next = 2: entries at slots 0 and 1.
        write_u32(&mut buf, idr_pa + offsets.idr_next as u64, 2);

        // Map 2: "target.bss", BPF_MAP_TYPE_ARRAY.
        write_u32(
            &mut buf,
            map2_pa + offsets.map_type as u64,
            BPF_MAP_TYPE_ARRAY,
        );
        write_u32(&mut buf, map2_pa + offsets.value_size as u64, 200);
        let name = b"target.bss";
        let name_pa = (map2_pa + offsets.map_name as u64) as usize;
        buf[name_pa..name_pa + name.len()].copy_from_slice(name);

        let start_kernel_map: u64 = START_KERNEL_MAP;
        let idr_kva = idr_pa + start_kernel_map;

        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let result = find_bpf_map(
            &lookup_ctx(&mem, pgd_pa, page_offset, &offsets, false),
            idr_kva,
            ".bss",
        );

        let info = result.expect("should skip untranslatable entry and find the second");
        assert_eq!(info.name, "target.bss");
        assert_eq!(info.map_pa, map2_pa);
        assert_eq!(info.value_size, 200);
    }

    // -- read_bpf_map_value tests --

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn read_bpf_map_value_u32_roundtrip() {
        let (mut buf, cr3_pa, kva, data_pa) = setup_page_table();
        // Write a known u32 at data_pa + 4.
        buf[data_pa as usize + 4..data_pa as usize + 8]
            .copy_from_slice(&0xCAFE_BABEu32.to_ne_bytes());
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

        let info = BpfMapInfo {
            map_pa: 0,
            map_kva: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            key_size: 0,
            value_size: 64,
            max_entries: 0,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        let val = read_bpf_map_value_u32(&value_ctx(&mem, cr3_pa, false), &info, 4);
        assert_eq!(val, Some(0xCAFE_BABE));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn read_bpf_map_value_bytes() {
        let (mut buf, cr3_pa, kva, data_pa) = setup_page_table();
        buf[data_pa as usize..data_pa as usize + 4].copy_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

        let info = BpfMapInfo {
            map_pa: 0,
            map_kva: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            key_size: 0,
            value_size: 64,
            max_entries: 0,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        let bytes = read_bpf_map_value(&value_ctx(&mem, cr3_pa, false), &info, 0, 4);
        assert_eq!(bytes, Some(vec![0xAA, 0xBB, 0xCC, 0xDD]));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn read_bpf_map_value_empty() {
        let (buf, cr3_pa, kva, _) = setup_page_table();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

        let info = BpfMapInfo {
            map_pa: 0,
            map_kva: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            key_size: 0,
            value_size: 64,
            max_entries: 0,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        let bytes = read_bpf_map_value(&value_ctx(&mem, cr3_pa, false), &info, 0, 0);
        assert_eq!(bytes, Some(vec![]));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn read_bpf_map_value_unmapped_returns_none() {
        let (buf, cr3_pa, _, _) = setup_page_table();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

        let info = BpfMapInfo {
            map_pa: 0,
            map_kva: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            key_size: 0,
            value_size: 16,
            max_entries: 0,
            value_kva: Some(0xFFFF_FFFF_8000_0000), // Unmapped KVA.
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        assert_eq!(
            read_bpf_map_value(&value_ctx(&mem, cr3_pa, false), &info, 0, 4),
            None
        );
        assert_eq!(
            read_bpf_map_value_u32(&value_ctx(&mem, cr3_pa, false), &info, 0),
            None
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn write_then_read_bpf_map_value_roundtrip() {
        let (mut buf, cr3_pa, kva, _) = setup_page_table();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

        let info = BpfMapInfo {
            map_pa: 0,
            map_kva: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            key_size: 0,
            value_size: 64,
            max_entries: 0,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        // Write then read u32.
        assert!(write_bpf_map_value_u32(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            8,
            0x1234_5678,
        ));
        assert_eq!(
            read_bpf_map_value_u32(&value_ctx(&mem, cr3_pa, false), &info, 8),
            Some(0x1234_5678)
        );

        // Write then read bytes.
        let payload = [0x11, 0x22, 0x33, 0x44, 0x55];
        assert!(write_bpf_map_value(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            16,
            &payload,
        ));
        assert_eq!(
            read_bpf_map_value(&value_ctx(&mem, cr3_pa, false), &info, 16, 5),
            Some(payload.to_vec()),
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn read_bpf_map_value_across_page_boundary() {
        let (mut buf, cr3_pa, kva, page1_pa, page2_pa) = setup_two_page_table();
        // Write known bytes at the page boundary.
        buf[page1_pa as usize + 0xFFE] = 0xAA;
        buf[page1_pa as usize + 0xFFF] = 0xBB;
        buf[page2_pa as usize] = 0xCC;
        buf[page2_pa as usize + 1] = 0xDD;

        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

        let info = BpfMapInfo {
            map_pa: 0,
            map_kva: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            key_size: 0,
            value_size: 0x2000,
            max_entries: 0,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        let bytes = read_bpf_map_value(&value_ctx(&mem, cr3_pa, false), &info, 0xFFE, 4);
        assert_eq!(bytes, Some(vec![0xAA, 0xBB, 0xCC, 0xDD]));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn read_bpf_map_value_u32_5level() {
        let (mut buf, cr3_pa, kva, data_pa) = setup_5level_page_table();
        buf[data_pa as usize..data_pa as usize + 4].copy_from_slice(&0xDEAD_BEEFu32.to_ne_bytes());
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

        let info = BpfMapInfo {
            map_pa: 0,
            map_kva: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            key_size: 0,
            value_size: 64,
            max_entries: 0,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        assert_eq!(
            read_bpf_map_value_u32(&value_ctx(&mem, cr3_pa, true), &info, 0),
            Some(0xDEAD_BEEF)
        );
    }

    // -- find_all_bpf_maps tests (fix #3) --

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn find_all_bpf_maps_returns_both_types() {
        // Reuse multi-map helper but change map1 to HASH type.
        let mut setup = setup_find_bpf_map_multi();
        let map1_pa: u64 = 0x14000;
        // Overwrite map1's map_type from ARRAY (2) to HASH (1).
        let map_type_off = setup.3.map_type;
        let off = (map1_pa + map_type_off as u64) as usize;
        setup.0[off..off + 4].copy_from_slice(&1u32.to_ne_bytes());

        let (buf, cr3_pa, idr_kva, offsets) = setup;
        let page_offset: u64 = 0xFFFF_8880_0000_0000;
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

        let maps = find_all_bpf_maps(
            &lookup_ctx(&mem, cr3_pa, page_offset, &offsets, false),
            idr_kva,
        );
        assert_eq!(maps.len(), 2);
        let hash_map = maps.iter().find(|m| m.name == "other.data");
        let array_map = maps.iter().find(|m| m.name == "mitosis.bss");
        assert!(hash_map.is_some(), "HASH map should be in results");
        assert!(array_map.is_some(), "ARRAY map should be in results");
        assert_eq!(hash_map.unwrap().map_type, 1); // BPF_MAP_TYPE_HASH
        assert!(hash_map.unwrap().value_kva.is_none());
        assert_eq!(array_map.unwrap().map_type, BPF_MAP_TYPE_ARRAY);
        assert!(array_map.unwrap().value_kva.is_some());
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn find_all_bpf_maps_single_entry() {
        let (buf, cr3_pa, idr_kva, offsets) =
            setup_find_bpf_map("test.bss", BPF_MAP_TYPE_ARRAY, 64);
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

        let maps = find_all_bpf_maps(
            &lookup_ctx(&mem, cr3_pa, 0xFFFF_8880_0000_0000, &offsets, false),
            idr_kva,
        );
        assert_eq!(maps.len(), 1);
        assert_eq!(maps[0].name, "test.bss");
    }

    #[test]
    fn find_all_bpf_maps_empty_idr() {
        let offsets = BpfMapOffsets {
            map_name: 32,
            map_type: 24,
            map_flags: 28,
            key_size: 44,
            value_size: 48,
            max_entries: 52,
            array_value: 256,
            xa_node_slots: 16,
            xa_node_shift: 0,
            idr_xa_head: 8,
            idr_next: 20,
            map_btf: 0,
            map_btf_value_type_id: 0,
            btf_data: 0,
            btf_data_size: 0,
            htab_offsets: None,
        };
        let buf = vec![0u8; 0x2000];
        let start_kernel_map: u64 = START_KERNEL_MAP;
        let idr_kva = 0x1000 + start_kernel_map;
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

        let maps = find_all_bpf_maps(
            &lookup_ctx(&mem, 0x10000, 0xFFFF_8880_0000_0000, &offsets, false),
            idr_kva,
        );
        assert!(maps.is_empty());
    }

    // -- value_kva Option tests (fix #4) --

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn read_value_returns_none_for_non_array_map() {
        let (buf, cr3_pa, _, _) = setup_page_table();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

        let info = BpfMapInfo {
            map_pa: 0,
            map_kva: 0,
            name: "hash.map".into(),
            map_type: 1, // HASH
            map_flags: 0,
            key_size: 0,
            value_size: 64,
            max_entries: 0,
            value_kva: None,
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        assert!(read_bpf_map_value(&value_ctx(&mem, cr3_pa, false), &info, 0, 4).is_none());
        assert!(read_bpf_map_value_u32(&value_ctx(&mem, cr3_pa, false), &info, 0).is_none());
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn write_value_returns_false_for_non_array_map() {
        let (mut buf, cr3_pa, _, _) = setup_page_table();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

        let info = BpfMapInfo {
            map_pa: 0,
            map_kva: 0,
            name: "hash.map".into(),
            map_type: 1, // HASH
            map_flags: 0,
            key_size: 0,
            value_size: 64,
            max_entries: 0,
            value_kva: None,
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        assert!(!write_bpf_map_value(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            0,
            &[1, 2, 3, 4],
        ));
        assert!(!write_bpf_map_value_u32(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            0,
            42
        ));
    }

    // -- map_flags test (fix #5) --

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn find_all_bpf_maps_reads_map_flags() {
        let (mut buf, cr3_pa, idr_kva, offsets) =
            setup_find_bpf_map("flagged.bss", BPF_MAP_TYPE_ARRAY, 64);
        // Write non-zero map_flags at the correct offset.
        let map_pa: u64 = 0x14000;
        let flags_pa = (map_pa + offsets.map_flags as u64) as usize;
        buf[flags_pa..flags_pa + 4].copy_from_slice(&0x0400u32.to_ne_bytes()); // BPF_F_MMAPABLE

        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let maps = find_all_bpf_maps(
            &lookup_ctx(&mem, cr3_pa, 0xFFFF_8880_0000_0000, &offsets, false),
            idr_kva,
        );
        assert_eq!(maps.len(), 1);
        assert_eq!(maps[0].map_flags, 0x0400);
    }

    // -- xa_node_shift non-zero offset test (fix #7) --

    #[test]
    fn xa_node_shift_nonzero_offset() {
        // Place shift at offset 8 within the xa_node instead of 0.
        let node_pa: u64 = 0x1000;
        let page_offset: u64 = crate::monitor::symbols::DEFAULT_PAGE_OFFSET;
        let node_kva = page_offset.wrapping_add(node_pa);
        let shift_off: usize = 8;

        let mut buf = vec![0u8; 0x2000];
        // Write shift=6 at node_pa + 8.
        buf[node_pa as usize + shift_off] = 6;

        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        assert_eq!(xa_node_shift(&mem, page_offset, node_kva, shift_off), 6);
        // With offset 0 (wrong), should read 0 (the byte at node_pa + 0).
        assert_eq!(xa_node_shift(&mem, page_offset, node_kva, 0), 0);
    }

    // -- xa_load continue past failed entry test (fix #12) --

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn find_all_bpf_maps_continues_past_untranslatable_entry() {
        // IDR with two entries via xa_node. First entry has an
        // untranslatable KVA (no page table mapping). Second entry
        // is a valid ARRAY map. find_all_bpf_maps should skip the
        // first and return the second.
        let offsets = BpfMapOffsets {
            map_name: 32,
            map_type: 24,
            map_flags: 28,
            key_size: 44,
            value_size: 48,
            max_entries: 52,
            array_value: 256,
            xa_node_slots: 16,
            xa_node_shift: 0,
            idr_xa_head: 8,
            idr_next: 20,
            map_btf: 0,
            map_btf_value_type_id: 0,
            btf_data: 0,
            btf_data_size: 0,
            htab_offsets: None,
        };

        let pgd_pa: u64 = 0x10000;
        let pud_pa: u64 = 0x11000;
        let pmd_pa: u64 = 0x12000;
        let pte_pa: u64 = 0x13000;
        let map_pa: u64 = 0x14000;
        let idr_pa: u64 = 0x15000;
        let xa_node_pa: u64 = 0x16000;

        let map_kva: u64 = 0xFFFF_C900_0000_0000;
        let pgd_idx = (map_kva >> 39) & 0x1FF;
        let pud_idx = (map_kva >> 30) & 0x1FF;
        let pmd_idx = (map_kva >> 21) & 0x1FF;
        let pte_idx = (map_kva >> 12) & 0x1FF;

        // Unmappable KVA: different PGD index, no page table entry.
        let bad_kva: u64 = 0xFFFF_C900_8000_0000;

        let page_offset: u64 = 0xFFFF_8880_0000_0000;
        let xa_node_kva = xa_node_pa + page_offset;

        let size = 0x17000;
        let mut buf = vec![0u8; size];

        let write_u64 = |buf: &mut Vec<u8>, pa: u64, val: u64| {
            let off = pa as usize;
            buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
        };
        let write_u32 = |buf: &mut Vec<u8>, pa: u64, val: u32| {
            let off = pa as usize;
            buf[off..off + 4].copy_from_slice(&val.to_ne_bytes());
        };

        // Page table for map_kva only.
        write_u64(&mut buf, pgd_pa + pgd_idx * 8, (pud_pa + PTE_BASE) | 0x63);
        write_u64(&mut buf, pud_pa + pud_idx * 8, (pmd_pa + PTE_BASE) | 0x63);
        write_u64(&mut buf, pmd_pa + pmd_idx * 8, (pte_pa + PTE_BASE) | 0x63);
        write_u64(&mut buf, pte_pa + pte_idx * 8, (map_pa + PTE_BASE) | 0x63);

        // xa_node with two entries: slot 0 = bad_kva, slot 1 = map_kva.
        buf[xa_node_pa as usize] = 0; // shift=0
        write_u64(&mut buf, xa_node_pa + offsets.xa_node_slots as u64, bad_kva);
        write_u64(
            &mut buf,
            xa_node_pa + offsets.xa_node_slots as u64 + 8,
            map_kva,
        );

        // IDR xa_head -> xa_node.
        write_u64(
            &mut buf,
            idr_pa + offsets.idr_xa_head as u64,
            xa_node_kva | 2,
        );
        // idr_next = 2: entries at slots 0 and 1.
        write_u32(&mut buf, idr_pa + offsets.idr_next as u64, 2);

        // Valid map at map_pa.
        write_u32(
            &mut buf,
            map_pa + offsets.map_type as u64,
            BPF_MAP_TYPE_ARRAY,
        );
        write_u32(&mut buf, map_pa + offsets.value_size as u64, 64);
        let name = b"good.bss";
        let name_pa = (map_pa + offsets.map_name as u64) as usize;
        buf[name_pa..name_pa + name.len()].copy_from_slice(name);

        let start_kernel_map: u64 = START_KERNEL_MAP;
        let idr_kva = idr_pa + start_kernel_map;

        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let maps = find_all_bpf_maps(
            &lookup_ctx(&mem, pgd_pa, page_offset, &offsets, false),
            idr_kva,
        );

        // Should find the second map despite the first being untranslatable.
        let good = maps.iter().find(|m| m.name == "good.bss");
        assert!(
            good.is_some(),
            "good.bss should be found despite bad entry at slot 0"
        );
        assert_eq!(good.unwrap().map_type, BPF_MAP_TYPE_ARRAY);
    }

    // -- bounds check tests (fix #12) --

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn read_value_rejects_out_of_bounds() {
        let (buf, cr3_pa, kva, _) = setup_page_table();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

        let info = BpfMapInfo {
            map_pa: 0,
            map_kva: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            key_size: 0,
            value_size: 8,
            max_entries: 0,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        // Exactly at boundary: offset=4, len=4 -> 4+4=8 == value_size, ok.
        assert!(read_bpf_map_value(&value_ctx(&mem, cr3_pa, false), &info, 4, 4).is_some());
        // One past: offset=4, len=5 -> 4+5=9 > 8, rejected.
        assert!(read_bpf_map_value(&value_ctx(&mem, cr3_pa, false), &info, 4, 5).is_none());
        // Offset past end: offset=9, len=1 -> 9+1=10 > 8, rejected.
        assert!(read_bpf_map_value(&value_ctx(&mem, cr3_pa, false), &info, 9, 1).is_none());
        // u32 past end: offset=6, 6+4=10 > 8, rejected.
        assert!(read_bpf_map_value_u32(&value_ctx(&mem, cr3_pa, false), &info, 6).is_none());
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn write_value_rejects_out_of_bounds() {
        let (mut buf, cr3_pa, kva, _) = setup_page_table();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

        let info = BpfMapInfo {
            map_pa: 0,
            map_kva: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            key_size: 0,
            value_size: 8,
            max_entries: 0,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        // Within bounds: offset=0, len=8.
        assert!(write_bpf_map_value(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            0,
            &[0u8; 8],
        ));
        // Past end: offset=0, len=9.
        assert!(!write_bpf_map_value(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            0,
            &[0u8; 9],
        ));
        // u32 past end: offset=6, 6+4=10 > 8.
        assert!(!write_bpf_map_value_u32(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            6,
            42
        ));
        // u32 at boundary: offset=4, 4+4=8, ok.
        assert!(write_bpf_map_value_u32(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            4,
            42
        ));
    }

    // -- BpfValueLayout tests --

    #[test]
    fn value_layout_from_vmlinux_btf() {
        // Resolve a known kernel struct via real BTF if available.
        let path = match crate::monitor::find_test_vmlinux() {
            Some(p) => p,
            None => return,
        };
        let btf = super::super::btf_offsets::load_btf_from_path(&path).unwrap();
        // struct bpf_map has well-known fields.
        let types = btf.resolve_types_by_name("bpf_map").unwrap();
        let struct_type = types.iter().find_map(|t| {
            if let btf_rs::Type::Struct(s) = t {
                Some(s)
            } else {
                None
            }
        });
        if struct_type.is_none() {}
    }

    #[test]
    fn value_layout_field_lookup() {
        // Build a BpfValueLayout manually and test field().
        let layout = BpfValueLayout {
            fields: vec![
                BpfFieldInfo {
                    name: "enabled".into(),
                    offset: 0,
                    size: 1,
                    kind: BpfFieldKind::Bool,
                },
                BpfFieldInfo {
                    name: "count".into(),
                    offset: 4,
                    size: 4,
                    kind: BpfFieldKind::U32,
                },
                BpfFieldInfo {
                    name: "total".into(),
                    offset: 8,
                    size: 8,
                    kind: BpfFieldKind::I64,
                },
            ],
            total_size: 16,
        };

        assert!(layout.field("enabled").is_some());
        assert_eq!(layout.field("enabled").unwrap().offset, 0);
        assert_eq!(layout.field("enabled").unwrap().kind, BpfFieldKind::Bool);

        assert!(layout.field("count").is_some());
        assert_eq!(layout.field("count").unwrap().offset, 4);
        assert_eq!(layout.field("count").unwrap().kind, BpfFieldKind::U32);

        assert!(layout.field("total").is_some());
        assert_eq!(layout.field("total").unwrap().kind, BpfFieldKind::I64);

        assert!(layout.field("missing").is_none());
    }

    // -- read_field / write_field tests --

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn read_write_field_typed_roundtrip() {
        let (mut buf, cr3_pa, kva, _) = setup_page_table();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

        let info = BpfMapInfo {
            map_pa: 0,
            map_kva: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            key_size: 0,
            value_size: 64,
            max_entries: 0,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        let field_u32 = BpfFieldInfo {
            name: "count".into(),
            offset: 0,
            size: 4,
            kind: BpfFieldKind::U32,
        };

        assert!(write_typed_field(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            &field_u32,
            BpfValue::U32(42),
        ));
        let val = read_typed_field(&value_ctx(&mem, cr3_pa, false), &info, &field_u32);
        assert_eq!(val, Some(BpfValue::U32(42)));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn read_write_field_all_types() {
        let (mut buf, cr3_pa, kva, _) = setup_page_table();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

        let info = BpfMapInfo {
            map_pa: 0,
            map_kva: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            key_size: 0,
            value_size: 64,
            max_entries: 0,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        // Bool
        let f = BpfFieldInfo {
            name: "b".into(),
            offset: 0,
            size: 1,
            kind: BpfFieldKind::Bool,
        };
        assert!(write_typed_field(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            &f,
            BpfValue::Bool(true),
        ));
        assert_eq!(
            read_typed_field(&value_ctx(&mem, cr3_pa, false), &info, &f),
            Some(BpfValue::Bool(true))
        );

        // U8
        let f = BpfFieldInfo {
            name: "u8".into(),
            offset: 1,
            size: 1,
            kind: BpfFieldKind::U8,
        };
        assert!(write_typed_field(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            &f,
            BpfValue::U8(0xAB),
        ));
        assert_eq!(
            read_typed_field(&value_ctx(&mem, cr3_pa, false), &info, &f),
            Some(BpfValue::U8(0xAB))
        );

        // I8
        let f = BpfFieldInfo {
            name: "i8".into(),
            offset: 2,
            size: 1,
            kind: BpfFieldKind::I8,
        };
        assert!(write_typed_field(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            &f,
            BpfValue::I8(-5),
        ));
        assert_eq!(
            read_typed_field(&value_ctx(&mem, cr3_pa, false), &info, &f),
            Some(BpfValue::I8(-5))
        );

        // U16
        let f = BpfFieldInfo {
            name: "u16".into(),
            offset: 4,
            size: 2,
            kind: BpfFieldKind::U16,
        };
        assert!(write_typed_field(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            &f,
            BpfValue::U16(1234),
        ));
        assert_eq!(
            read_typed_field(&value_ctx(&mem, cr3_pa, false), &info, &f),
            Some(BpfValue::U16(1234))
        );

        // I16
        let f = BpfFieldInfo {
            name: "i16".into(),
            offset: 6,
            size: 2,
            kind: BpfFieldKind::I16,
        };
        assert!(write_typed_field(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            &f,
            BpfValue::I16(-100),
        ));
        assert_eq!(
            read_typed_field(&value_ctx(&mem, cr3_pa, false), &info, &f),
            Some(BpfValue::I16(-100))
        );

        // U64
        let f = BpfFieldInfo {
            name: "u64".into(),
            offset: 8,
            size: 8,
            kind: BpfFieldKind::U64,
        };
        assert!(write_typed_field(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            &f,
            BpfValue::U64(0xDEAD_BEEF),
        ));
        assert_eq!(
            read_typed_field(&value_ctx(&mem, cr3_pa, false), &info, &f),
            Some(BpfValue::U64(0xDEAD_BEEF))
        );

        // I64
        let f = BpfFieldInfo {
            name: "i64".into(),
            offset: 16,
            size: 8,
            kind: BpfFieldKind::I64,
        };
        assert!(write_typed_field(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            &f,
            BpfValue::I64(-999),
        ));
        assert_eq!(
            read_typed_field(&value_ctx(&mem, cr3_pa, false), &info, &f),
            Some(BpfValue::I64(-999))
        );

        // I32
        let f = BpfFieldInfo {
            name: "i32".into(),
            offset: 24,
            size: 4,
            kind: BpfFieldKind::I32,
        };
        assert!(write_typed_field(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            &f,
            BpfValue::I32(-42),
        ));
        assert_eq!(
            read_typed_field(&value_ctx(&mem, cr3_pa, false), &info, &f),
            Some(BpfValue::I32(-42))
        );

        // Bytes
        let f = BpfFieldInfo {
            name: "data".into(),
            offset: 28,
            size: 3,
            kind: BpfFieldKind::Bytes(3),
        };
        assert!(write_typed_field(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            &f,
            BpfValue::Bytes(vec![1, 2, 3]),
        ));
        assert_eq!(
            read_typed_field(&value_ctx(&mem, cr3_pa, false), &info, &f),
            Some(BpfValue::Bytes(vec![1, 2, 3]))
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn write_field_type_mismatch_returns_false() {
        let (mut buf, cr3_pa, kva, _) = setup_page_table();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

        let info = BpfMapInfo {
            map_pa: 0,
            map_kva: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            key_size: 0,
            value_size: 64,
            max_entries: 0,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        // U32 field, try to write a U64 value.
        let field = BpfFieldInfo {
            name: "count".into(),
            offset: 0,
            size: 4,
            kind: BpfFieldKind::U32,
        };
        assert!(!write_typed_field(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            &field,
            BpfValue::U64(1),
        ));
        assert!(!write_typed_field(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            &field,
            BpfValue::Bool(true),
        ));
        assert!(!write_typed_field(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            &field,
            BpfValue::I32(-1),
        ));

        // Bytes field: wrong length.
        let field_bytes = BpfFieldInfo {
            name: "data".into(),
            offset: 4,
            size: 3,
            kind: BpfFieldKind::Bytes(3),
        };
        assert!(!write_typed_field(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            &field_bytes,
            BpfValue::Bytes(vec![1, 2]),
        ));
        assert!(!write_typed_field(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            &field_bytes,
            BpfValue::Bytes(vec![1, 2, 3, 4]),
        ));
        // Correct length works.
        assert!(write_typed_field(
            &value_ctx(&mem, cr3_pa, false),
            &info,
            &field_bytes,
            BpfValue::Bytes(vec![1, 2, 3]),
        ));
    }

    // -- BpfMapInfo btf fields --

    #[test]
    fn bpf_map_info_btf_fields_default_zero() {
        let info = BpfMapInfo {
            map_pa: 0x1000,
            map_kva: 0,
            name: "test".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            key_size: 0,
            value_size: 32,
            max_entries: 0,
            value_kva: None,
            btf_kva: 0,
            btf_value_type_id: 0,
        };
        assert_eq!(info.btf_kva, 0);
        assert_eq!(info.btf_value_type_id, 0);
    }

    #[test]
    fn bpf_map_info_btf_fields_populated() {
        let info = BpfMapInfo {
            map_pa: 0x1000,
            map_kva: 0,
            name: "test".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            key_size: 0,
            value_size: 32,
            max_entries: 0,
            value_kva: None,
            btf_kva: 0xFFFF_8880_0001_0000,
            btf_value_type_id: 42,
        };
        assert_eq!(info.btf_kva, 0xFFFF_8880_0001_0000);
        assert_eq!(info.btf_value_type_id, 42);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn find_all_bpf_maps_populates_btf_fields() {
        let (mut buf, cr3_pa, idr_kva, mut offsets) =
            setup_find_bpf_map("test.bss", BPF_MAP_TYPE_ARRAY, 64);

        // Place btf fields at offsets that don't overlap existing fields.
        offsets.map_btf = 56;
        offsets.map_btf_value_type_id = 64;

        let map_pa: u64 = 0x14000;
        let btf_off = (map_pa + offsets.map_btf as u64) as usize;
        let btf_tid_off = (map_pa + offsets.map_btf_value_type_id as u64) as usize;

        // Zero out the btf fields first — default from zeroed buf.
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let maps = find_all_bpf_maps(
            &lookup_ctx(&mem, cr3_pa, 0xFFFF_8880_0000_0000, &offsets, false),
            idr_kva,
        );
        assert_eq!(maps.len(), 1);
        assert_eq!(maps[0].btf_kva, 0);
        assert_eq!(maps[0].btf_value_type_id, 0);

        // Write nonzero values and re-scan.
        let btf_kva_val: u64 = 0xFFFF_8880_DEAD_0000;
        buf[btf_off..btf_off + 8].copy_from_slice(&btf_kva_val.to_ne_bytes());
        buf[btf_tid_off..btf_tid_off + 4].copy_from_slice(&7u32.to_ne_bytes());

        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let maps = find_all_bpf_maps(
            &lookup_ctx(&mem, cr3_pa, 0xFFFF_8880_0000_0000, &offsets, false),
            idr_kva,
        );
        assert_eq!(maps[0].btf_kva, btf_kva_val);
        assert_eq!(maps[0].btf_value_type_id, 7);
    }

    // -- idr_next scan bound --

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn find_all_bpf_maps_respects_idr_next_bound() {
        // Build IDR with 3 slots in xa_node, but set idr_next=2.
        // Only indices 0 and 1 should be scanned.
        let offsets = BpfMapOffsets {
            map_name: 32,
            map_type: 24,
            map_flags: 28,
            key_size: 44,
            value_size: 48,
            max_entries: 52,
            array_value: 256,
            xa_node_slots: 16,
            xa_node_shift: 0,
            idr_xa_head: 8,
            idr_next: 20,
            map_btf: 0,
            map_btf_value_type_id: 0,
            btf_data: 0,
            btf_data_size: 0,
            htab_offsets: None,
        };

        let pgd_pa: u64 = 0x10000;
        let pud_pa: u64 = 0x11000;
        let pmd_pa: u64 = 0x12000;
        let pte_pa: u64 = 0x13000;
        let map_pa: u64 = 0x14000;
        let map2_pa: u64 = 0x15000;
        let map3_pa: u64 = 0x16000;
        let idr_pa: u64 = 0x17000;
        let xa_node_pa: u64 = 0x18000;

        let map_kva: u64 = 0xFFFF_C900_0000_0000;
        let map2_kva: u64 = 0xFFFF_C900_0000_1000;
        let map3_kva: u64 = 0xFFFF_C900_0000_2000;
        let pgd_idx = (map_kva >> 39) & 0x1FF;
        let pud_idx = (map_kva >> 30) & 0x1FF;
        let pmd_idx = (map_kva >> 21) & 0x1FF;
        let pte1_idx = (map_kva >> 12) & 0x1FF;
        let pte2_idx = (map2_kva >> 12) & 0x1FF;
        let pte3_idx = (map3_kva >> 12) & 0x1FF;

        let page_offset: u64 = 0xFFFF_8880_0000_0000;
        let xa_node_kva = xa_node_pa + page_offset;

        let size = 0x19000;
        let mut buf = vec![0u8; size];

        let write_u64 = |buf: &mut Vec<u8>, pa: u64, val: u64| {
            let off = pa as usize;
            buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
        };
        let write_u32 = |buf: &mut Vec<u8>, pa: u64, val: u32| {
            let off = pa as usize;
            buf[off..off + 4].copy_from_slice(&val.to_ne_bytes());
        };

        // Page table for all three map KVAs.
        write_u64(&mut buf, pgd_pa + pgd_idx * 8, (pud_pa + PTE_BASE) | 0x63);
        write_u64(&mut buf, pud_pa + pud_idx * 8, (pmd_pa + PTE_BASE) | 0x63);
        write_u64(&mut buf, pmd_pa + pmd_idx * 8, (pte_pa + PTE_BASE) | 0x63);
        write_u64(&mut buf, pte_pa + pte1_idx * 8, (map_pa + PTE_BASE) | 0x63);
        write_u64(&mut buf, pte_pa + pte2_idx * 8, (map2_pa + PTE_BASE) | 0x63);
        write_u64(&mut buf, pte_pa + pte3_idx * 8, (map3_pa + PTE_BASE) | 0x63);

        // xa_node with 3 entries.
        buf[xa_node_pa as usize] = 0; // shift=0
        write_u64(&mut buf, xa_node_pa + offsets.xa_node_slots as u64, map_kva);
        write_u64(
            &mut buf,
            xa_node_pa + offsets.xa_node_slots as u64 + 8,
            map2_kva,
        );
        write_u64(
            &mut buf,
            xa_node_pa + offsets.xa_node_slots as u64 + 2 * 8,
            map3_kva,
        );

        // IDR: xa_head -> xa_node, idr_next = 2 (only scan 0..2).
        write_u64(
            &mut buf,
            idr_pa + offsets.idr_xa_head as u64,
            xa_node_kva | 2,
        );
        write_u32(&mut buf, idr_pa + offsets.idr_next as u64, 2);

        // Map 1 at slot 0.
        write_u32(
            &mut buf,
            map_pa + offsets.map_type as u64,
            BPF_MAP_TYPE_ARRAY,
        );
        write_u32(&mut buf, map_pa + offsets.value_size as u64, 32);
        let name = b"first.bss";
        let name_pa = (map_pa + offsets.map_name as u64) as usize;
        buf[name_pa..name_pa + name.len()].copy_from_slice(name);

        // Map 2 at slot 1.
        write_u32(
            &mut buf,
            map2_pa + offsets.map_type as u64,
            BPF_MAP_TYPE_ARRAY,
        );
        write_u32(&mut buf, map2_pa + offsets.value_size as u64, 64);
        let name = b"second.bss";
        let name_pa = (map2_pa + offsets.map_name as u64) as usize;
        buf[name_pa..name_pa + name.len()].copy_from_slice(name);

        // Map 3 at slot 2 — should NOT be found because idr_next=2.
        write_u32(
            &mut buf,
            map3_pa + offsets.map_type as u64,
            BPF_MAP_TYPE_ARRAY,
        );
        write_u32(&mut buf, map3_pa + offsets.value_size as u64, 128);
        let name = b"third.bss";
        let name_pa = (map3_pa + offsets.map_name as u64) as usize;
        buf[name_pa..name_pa + name.len()].copy_from_slice(name);

        let start_kernel_map: u64 = START_KERNEL_MAP;
        let idr_kva = idr_pa + start_kernel_map;

        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let maps = find_all_bpf_maps(
            &lookup_ctx(&mem, pgd_pa, page_offset, &offsets, false),
            idr_kva,
        );

        // Only 2 maps should be found (idr_next=2 means scan 0..2).
        assert_eq!(maps.len(), 2);
        assert!(maps.iter().any(|m| m.name == "first.bss"));
        assert!(maps.iter().any(|m| m.name == "second.bss"));
        assert!(!maps.iter().any(|m| m.name == "third.bss"));
    }

    // -- translate_kva in kernel image / vmalloc region --

    /// Build a page table mapping KVA 0xFFFF_8000_8400_5000 (KIMAGE_VADDR
    /// region on aarch64, vmalloc range where BPF maps live).
    ///
    /// x86_64: 4-level walk, 4KB pages, PGD index 256.
    /// aarch64 (64KB granule): 3-level walk, PGD index 32.
    #[cfg(target_arch = "x86_64")]
    fn setup_page_table_vmalloc() -> (Vec<u8>, u64, u64, u64) {
        let kva: u64 = 0xFFFF_8000_8400_5000;
        let pgd_idx = (kva >> 39) & 0x1FF;
        let pud_idx = (kva >> 30) & 0x1FF;
        let pmd_idx = (kva >> 21) & 0x1FF;
        let pte_idx = (kva >> 12) & 0x1FF;

        let pgd_pa: u64 = 0x10000;
        let pud_pa: u64 = pgd_pa + 0x1000;
        let pmd_pa: u64 = pud_pa + 0x1000;
        let pte_pa: u64 = pmd_pa + 0x1000;
        let data_pa: u64 = pte_pa + 0x1000;

        let size = (data_pa + 0x1000) as usize;
        let mut buf = vec![0u8; size];

        let write_entry = |buf: &mut Vec<u8>, base: u64, idx: u64, val: u64| {
            let off = (base + idx * 8) as usize;
            buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
        };

        write_entry(&mut buf, pgd_pa, pgd_idx, (pud_pa + PTE_BASE) | 0x63);
        write_entry(&mut buf, pud_pa, pud_idx, (pmd_pa + PTE_BASE) | 0x63);
        write_entry(&mut buf, pmd_pa, pmd_idx, (pte_pa + PTE_BASE) | 0x63);
        write_entry(&mut buf, pte_pa, pte_idx, (data_pa + PTE_BASE) | 0x63);

        // Write known data at the target page.
        buf[data_pa as usize..data_pa as usize + 8]
            .copy_from_slice(&0x1234_5678_ABCD_EF00u64.to_ne_bytes());

        (buf, pgd_pa, kva, data_pa)
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn translate_kva_l0_index_256() {
        let (buf, cr3_pa, kva, data_pa) = setup_page_table_vmalloc();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let pa = mem.translate_kva(cr3_pa, kva, false);
        assert_eq!(
            pa,
            Some(data_pa),
            "L0[256] walk should resolve to data page"
        );
        assert_eq!(mem.read_u64(pa.unwrap(), 0), 0x1234_5678_ABCD_EF00);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn translate_kva_l0_index_256_with_offset() {
        let (buf, cr3_pa, kva, data_pa) = setup_page_table_vmalloc();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let pa = mem.translate_kva(cr3_pa, kva + 0x100, false);
        assert_eq!(pa, Some(data_pa + 0x100));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn translate_kva_l0_index_256_unmapped_neighbor() {
        let (buf, cr3_pa, kva, _) = setup_page_table_vmalloc();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let kva_257 = kva + (1u64 << 39);
        assert_eq!(mem.translate_kva(cr3_pa, kva_257, false), None);
    }

    // -- aarch64 64KB granule vmalloc region tests --

    /// Build a 3-level page table for 64KB granule mapping KVA
    /// 0xFFFF_8000_8400_0000 (KIMAGE_VADDR region, PGD index 32).
    #[cfg(target_arch = "aarch64")]
    fn setup_page_table_vmalloc_64k() -> (Vec<u8>, u64, u64, u64) {
        let kva: u64 = 0xFFFF_8000_8400_0000;
        let pgd_idx = (kva >> 42) & 0x3F; // 32
        let pmd_idx = (kva >> 29) & 0x1FFF; // 4
        let pte_idx = (kva >> 16) & 0x1FFF; // 0

        let pgd_pa: u64 = 0x10000;
        let pmd_pa: u64 = 0x20000;
        let pte_pa: u64 = 0x30000;
        let data_pa: u64 = 0x40000;

        let size = (data_pa + 0x10000) as usize;
        let mut buf = vec![0u8; size];

        let write_entry = |buf: &mut Vec<u8>, base: u64, idx: u64, val: u64| {
            let off = (base + idx * 8) as usize;
            buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
        };

        write_entry(&mut buf, pgd_pa, pgd_idx, (pmd_pa + PTE_BASE) | 0x03);
        write_entry(&mut buf, pmd_pa, pmd_idx, (pte_pa + PTE_BASE) | 0x03);
        write_entry(&mut buf, pte_pa, pte_idx, (data_pa + PTE_BASE) | 0x03);

        buf[data_pa as usize..data_pa as usize + 8]
            .copy_from_slice(&0x1234_5678_ABCD_EF00u64.to_ne_bytes());

        (buf, pgd_pa, kva, data_pa)
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn translate_kva_vmalloc_64k() {
        let (buf, cr3_pa, kva, data_pa) = setup_page_table_vmalloc_64k();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let pa = mem.translate_kva(cr3_pa, kva, false);
        assert_eq!(pa, Some(data_pa), "64KB vmalloc walk should resolve");
        assert_eq!(mem.read_u64(pa.unwrap(), 0), 0x1234_5678_ABCD_EF00);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn translate_kva_vmalloc_64k_with_offset() {
        let (buf, cr3_pa, kva, data_pa) = setup_page_table_vmalloc_64k();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let pa = mem.translate_kva(cr3_pa, kva + 0x100, false);
        assert_eq!(pa, Some(data_pa + 0x100));
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn translate_kva_vmalloc_64k_unmapped_neighbor() {
        let (buf, cr3_pa, kva, _) = setup_page_table_vmalloc_64k();
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let unmapped = kva + (1u64 << 42);
        assert_eq!(mem.translate_kva(cr3_pa, unmapped, false), None);
    }

    // -- iter_htab_entries tests --

    use crate::monitor::btf_offsets::HtabOffsets;

    /// Simplified htab offsets for synthetic buffer tests.
    /// htab_elem_size_base=32 is a test value, not the real kernel size.
    fn test_htab_offsets() -> HtabOffsets {
        HtabOffsets {
            htab_buckets: 200,
            htab_n_buckets: 208,
            bucket_size: 16,
            bucket_head: 0,
            hlist_nulls_head_first: 0,
            hlist_nulls_node_next: 0,
            htab_elem_size_base: 32,
        }
    }

    fn test_htab_map_offsets() -> BpfMapOffsets {
        BpfMapOffsets {
            map_name: 32,
            map_type: 24,
            map_flags: 28,
            key_size: 44,
            value_size: 48,
            max_entries: 52,
            array_value: 256,
            xa_node_slots: 16,
            xa_node_shift: 0,
            idr_xa_head: 8,
            idr_next: 20,
            map_btf: 0,
            map_btf_value_type_id: 0,
            btf_data: 0,
            btf_data_size: 0,
            htab_offsets: Some(test_htab_offsets()),
        }
    }

    #[test]
    fn iter_htab_entries_non_hash_map_returns_empty() {
        let buf = [0u8; 256];
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let offsets = test_htab_map_offsets();
        let map = BpfMapInfo {
            map_pa: 0,
            map_kva: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            key_size: 4,
            value_size: 8,
            max_entries: 0,
            value_kva: None,
            btf_kva: 0,
            btf_value_type_id: 0,
        };
        let entries = iter_htab_entries(&lookup_ctx(&mem, 0, 0, &offsets, false), &map);
        assert!(entries.is_empty());
    }

    #[test]
    fn iter_htab_entries_no_htab_offsets_returns_empty() {
        let buf = [0u8; 256];
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let mut offsets = test_htab_map_offsets();
        offsets.htab_offsets = None;
        let map = BpfMapInfo {
            map_pa: 0,
            map_kva: 0,
            name: "test".into(),
            map_type: BPF_MAP_TYPE_HASH,
            map_flags: 0,
            key_size: 4,
            value_size: 8,
            max_entries: 0,
            value_kva: None,
            btf_kva: 0,
            btf_value_type_id: 0,
        };
        let entries = iter_htab_entries(&lookup_ctx(&mem, 0, 0, &offsets, false), &map);
        assert!(entries.is_empty());
    }

    /// Build a synthetic hash map in a flat buffer with direct-mapping
    /// address translation. All structures are laid out at known PAs;
    /// page_offset is chosen so kva = pa + page_offset.
    ///
    /// Layout:
    ///   PA 0x0000: bpf_htab struct (contains bpf_map + htab fields)
    ///   PA 0x1000: buckets array (n_buckets * bucket_size)
    ///   PA 0x2000+: htab_elem entries (elem_size each)
    ///
    /// Each htab_elem has: hlist_nulls_node at offset 0, key at
    /// htab_elem_size_base, value at htab_elem_size_base + round_up(key_size, 8).
    fn setup_htab_direct(
        key_size: u32,
        value_size: u32,
        entries: &[(&[u8], &[u8])],
        n_buckets: u32,
    ) -> (Vec<u8>, u64, BpfMapInfo, BpfMapOffsets) {
        let htab = test_htab_offsets();
        let offsets = test_htab_map_offsets();
        let page_offset: u64 = crate::monitor::symbols::DEFAULT_PAGE_OFFSET;
        // Direct-mapping KVA = PAGE_OFFSET + dram_offset.
        let pa_to_kva = |pa: u64| -> u64 { page_offset.wrapping_add(pa) };

        let htab_pa: u64 = 0x0000;
        let buckets_pa: u64 = 0x1000;
        let elems_start: u64 = 0x2000;
        let elem_data_size = htab.htab_elem_size_base
            + ((key_size as usize + 7) & !7)
            + ((value_size as usize + 7) & !7);
        let elem_stride = elem_data_size.max(64); // padding for safety

        let buf_size = elems_start as usize + entries.len() * elem_stride + 0x1000;
        let mut buf = vec![0u8; buf_size];

        let write_u32 = |buf: &mut Vec<u8>, pa: u64, val: u32| {
            let off = pa as usize;
            buf[off..off + 4].copy_from_slice(&val.to_ne_bytes());
        };
        let write_u64 = |buf: &mut Vec<u8>, pa: u64, val: u64| {
            let off = pa as usize;
            buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
        };

        // Write bpf_htab fields.
        write_u32(
            &mut buf,
            htab_pa + offsets.map_type as u64,
            BPF_MAP_TYPE_HASH,
        );
        write_u32(&mut buf, htab_pa + offsets.key_size as u64, key_size);
        write_u32(&mut buf, htab_pa + offsets.value_size as u64, value_size);
        write_u64(
            &mut buf,
            htab_pa + htab.htab_buckets as u64,
            pa_to_kva(buckets_pa),
        );
        write_u32(&mut buf, htab_pa + htab.htab_n_buckets as u64, n_buckets);

        // Initialize all bucket heads to nulls marker (bit 0 set = empty).
        for i in 0..n_buckets {
            let bucket_pa = buckets_pa + (i as u64) * (htab.bucket_size as u64);
            write_u64(
                &mut buf,
                bucket_pa + htab.bucket_head as u64 + htab.hlist_nulls_head_first as u64,
                (i as u64) << 1 | 1, // nulls marker with bucket index
            );
        }

        // Place all entries in bucket 0 as a linked list.
        let mut prev_node_pa: Option<u64> = None;
        for (idx, (key, val)) in entries.iter().enumerate().rev() {
            let elem_pa = elems_start + (idx as u64) * (elem_stride as u64);
            let elem_kva = pa_to_kva(elem_pa);

            // Write key at htab_elem_size_base offset.
            let key_off = elem_pa + htab.htab_elem_size_base as u64;
            buf[key_off as usize..key_off as usize + key.len()].copy_from_slice(key);

            // Write value at htab_elem_size_base + round_up(key_size, 8).
            let val_off = elem_pa + htab.htab_elem_size_base as u64 + ((key_size as u64 + 7) & !7);
            buf[val_off as usize..val_off as usize + val.len()].copy_from_slice(val);

            // Set next pointer: points to previous element or nulls marker.
            let next = match prev_node_pa {
                Some(prev_pa) => pa_to_kva(prev_pa), // KVA of previous elem
                None => 1u64,                        // nulls end marker
            };
            write_u64(&mut buf, elem_pa + htab.hlist_nulls_node_next as u64, next);

            prev_node_pa = Some(elem_pa);

            // First element in reverse order becomes the head.
            if idx == 0 {
                // Update bucket 0's head to point to this element.
                write_u64(
                    &mut buf,
                    buckets_pa + htab.bucket_head as u64 + htab.hlist_nulls_head_first as u64,
                    elem_kva,
                );
            }
        }

        // If entries is non-empty, fix the chain: bucket head -> entries[0],
        // entries[0].next -> entries[1], ..., entries[last].next -> nulls.
        // The reverse iteration above already built this correctly:
        // prev_node_pa tracks the previous elem for forward chaining.

        let map = BpfMapInfo {
            map_pa: htab_pa,
            map_kva: pa_to_kva(htab_pa),
            name: "test_hash".into(),
            map_type: BPF_MAP_TYPE_HASH,
            map_flags: 0,
            key_size,
            value_size,
            max_entries: 0,
            value_kva: None,
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        (buf, page_offset, map, offsets)
    }

    #[test]
    fn iter_htab_entries_empty_map() {
        let (buf, page_offset, map, offsets) = setup_htab_direct(4, 8, &[], 4);
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let entries = iter_htab_entries(&lookup_ctx(&mem, 0, page_offset, &offsets, false), &map);
        assert!(entries.is_empty());
    }

    #[test]
    fn iter_htab_entries_single_entry() {
        let key = 42u32.to_ne_bytes();
        let val = 0xDEAD_BEEF_CAFE_1234u64.to_ne_bytes();
        let (buf, page_offset, map, offsets) = setup_htab_direct(4, 8, &[(&key, &val)], 4);
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let entries = iter_htab_entries(&lookup_ctx(&mem, 0, page_offset, &offsets, false), &map);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, key);
        assert_eq!(entries[0].1, val);
    }

    #[test]
    fn iter_htab_entries_multiple_entries() {
        let k1 = 1u32.to_ne_bytes();
        let v1 = 100u64.to_ne_bytes();
        let k2 = 2u32.to_ne_bytes();
        let v2 = 200u64.to_ne_bytes();
        let k3 = 3u32.to_ne_bytes();
        let v3 = 300u64.to_ne_bytes();
        let (buf, page_offset, map, offsets) =
            setup_htab_direct(4, 8, &[(&k1, &v1), (&k2, &v2), (&k3, &v3)], 4);
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let entries = iter_htab_entries(&lookup_ctx(&mem, 0, page_offset, &offsets, false), &map);
        assert_eq!(entries.len(), 3);
        // All entries are in bucket 0, chained in order.
        assert_eq!(entries[0].0, k1);
        assert_eq!(entries[0].1, v1);
        assert_eq!(entries[1].0, k2);
        assert_eq!(entries[1].1, v2);
        assert_eq!(entries[2].0, k3);
        assert_eq!(entries[2].1, v3);
    }

    #[test]
    fn iter_htab_entries_zero_buckets() {
        let key = 1u32.to_ne_bytes();
        let val = 1u64.to_ne_bytes();
        let (mut buf, page_offset, map, offsets) = setup_htab_direct(4, 8, &[(&key, &val)], 4);
        // Override n_buckets to 0.
        let htab = test_htab_offsets();
        buf[htab.htab_n_buckets..htab.htab_n_buckets + 4].copy_from_slice(&0u32.to_ne_bytes());
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let entries = iter_htab_entries(&lookup_ctx(&mem, 0, page_offset, &offsets, false), &map);
        assert!(entries.is_empty());
    }

    #[test]
    fn iter_htab_entries_larger_key_and_value() {
        // 8-byte key, 16-byte value.
        let key = 0xAAAA_BBBB_CCCC_DDDDu64.to_ne_bytes();
        let val = [
            0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE,
            0xFF, 0x00,
        ];
        let (buf, page_offset, map, offsets) = setup_htab_direct(8, 16, &[(&key, &val)], 2);
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let entries = iter_htab_entries(&lookup_ctx(&mem, 0, page_offset, &offsets, false), &map);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, key);
        assert_eq!(entries[0].1, val);
    }

    #[test]
    fn iter_htab_entries_multi_bucket() {
        // Entry in bucket 2, buckets 0 and 1 empty. Exercises the
        // bucket stride calculation: buckets_kva + i * bucket_size.
        let htab = test_htab_offsets();
        let offsets = test_htab_map_offsets();
        let page_offset: u64 = crate::monitor::symbols::DEFAULT_PAGE_OFFSET;
        let pa_to_kva = |pa: u64| -> u64 { page_offset.wrapping_add(pa) };
        let key_size: u32 = 4;
        let value_size: u32 = 8;

        let htab_pa: u64 = 0x0000;
        let buckets_pa: u64 = 0x1000;
        let elem_pa: u64 = 0x2000;
        let n_buckets: u32 = 4;

        let buf_size = 0x3000;
        let mut buf = vec![0u8; buf_size];

        let write_u32 = |buf: &mut Vec<u8>, pa: u64, val: u32| {
            let off = pa as usize;
            buf[off..off + 4].copy_from_slice(&val.to_ne_bytes());
        };
        let write_u64 = |buf: &mut Vec<u8>, pa: u64, val: u64| {
            let off = pa as usize;
            buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
        };

        // bpf_htab fields.
        write_u32(
            &mut buf,
            htab_pa + offsets.map_type as u64,
            BPF_MAP_TYPE_HASH,
        );
        write_u32(&mut buf, htab_pa + offsets.key_size as u64, key_size);
        write_u32(&mut buf, htab_pa + offsets.value_size as u64, value_size);
        write_u64(
            &mut buf,
            htab_pa + htab.htab_buckets as u64,
            pa_to_kva(buckets_pa),
        );
        write_u32(&mut buf, htab_pa + htab.htab_n_buckets as u64, n_buckets);

        // All buckets get nulls markers (empty).
        for i in 0..n_buckets {
            let bp = buckets_pa + (i as u64) * (htab.bucket_size as u64);
            write_u64(&mut buf, bp, (i as u64) << 1 | 1);
        }

        // Place one entry in bucket 2.
        let bucket2_pa = buckets_pa + 2 * (htab.bucket_size as u64);
        let elem_kva = pa_to_kva(elem_pa);
        write_u64(&mut buf, bucket2_pa, elem_kva); // bucket 2 head -> elem

        // elem next = nulls marker (end).
        write_u64(&mut buf, elem_pa + htab.hlist_nulls_node_next as u64, 1);

        // key at htab_elem_size_base.
        let key_bytes = 99u32.to_ne_bytes();
        let key_off = elem_pa + htab.htab_elem_size_base as u64;
        buf[key_off as usize..key_off as usize + 4].copy_from_slice(&key_bytes);

        // value at htab_elem_size_base + round_up(key_size, 8).
        let val_bytes = 0xBEEF_CAFEu64.to_ne_bytes();
        let val_off = elem_pa + htab.htab_elem_size_base as u64 + ((key_size as u64 + 7) & !7);
        buf[val_off as usize..val_off as usize + 8].copy_from_slice(&val_bytes);

        let map = BpfMapInfo {
            map_pa: htab_pa,
            map_kva: pa_to_kva(htab_pa),
            name: "multi_bucket".into(),
            map_type: BPF_MAP_TYPE_HASH,
            map_flags: 0,
            key_size,
            value_size,
            max_entries: 0,
            value_kva: None,
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let entries = iter_htab_entries(&lookup_ctx(&mem, 0, page_offset, &offsets, false), &map);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, key_bytes);
        assert_eq!(entries[0].1, val_bytes);
    }

    // -- read_percpu_array_value tests --

    /// Build a buffer simulating a percpu array map with `num_cpus` CPUs
    /// and `max_entries` entries. Each per-CPU value region is `value_size`
    /// bytes. Uses direct-mapping (page_offset) for per-CPU addresses.
    ///
    /// Layout:
    ///   0x0000..0x1000: page table pages (PGD/PUD/PMD/PTE)
    ///   0x10000: bpf_array (containing pptrs at array_value offset)
    ///   0x11000+: per-CPU value regions
    ///   per_cpu_offsets[cpu] adjusts the percpu base to per-CPU data
    ///
    /// Returns (buffer, cr3_pa, page_offset, map_info, offsets, per_cpu_offsets).
    #[cfg(target_arch = "x86_64")]
    fn setup_percpu_array(
        num_cpus: u32,
        max_entries: u32,
        value_size: u32,
    ) -> (Vec<u8>, u64, u64, BpfMapInfo, BpfMapOffsets, Vec<u64>) {
        let offsets = BpfMapOffsets {
            map_name: 32,
            map_type: 24,
            map_flags: 28,
            key_size: 44,
            value_size: 48,
            max_entries: 52,
            array_value: 256,
            xa_node_slots: 16,
            xa_node_shift: 0,
            idr_xa_head: 8,
            idr_next: 20,
            map_btf: 0,
            map_btf_value_type_id: 0,
            btf_data: 0,
            btf_data_size: 0,
            htab_offsets: None,
        };

        let page_offset: u64 = 0xFFFF_8880_0000_0000;

        // Page table for translating the bpf_array KVA (vmalloc'd).
        let pgd_pa: u64 = 0x10000;
        let pud_pa: u64 = 0x11000;
        let pmd_pa: u64 = 0x12000;
        let pte_pa: u64 = 0x13000;
        let array_pa: u64 = 0x14000;

        let map_kva: u64 = 0xFFFF_C900_0000_0000;
        let pgd_idx = (map_kva >> 39) & 0x1FF;
        let pud_idx = (map_kva >> 30) & 0x1FF;
        let pmd_idx = (map_kva >> 21) & 0x1FF;
        let pte_idx = (map_kva >> 12) & 0x1FF;

        // Per-CPU data: each CPU gets value_size bytes per entry, at
        // fixed PAs separated by 0x1000 per CPU. The percpu base is
        // a direct-mapped KVA; per_cpu_offsets adjust it per CPU.
        let percpu_base_pa: u64 = 0x20000;
        let percpu_stride: u64 = 0x1000;
        let elem_size = ((value_size as u64 + 7) & !7) * max_entries as u64;

        let total_size = (percpu_base_pa + percpu_stride * num_cpus as u64 + elem_size) as usize;
        let mut buf = vec![0u8; total_size.max(0x30000)];

        let write_u64 = |buf: &mut Vec<u8>, pa: u64, val: u64| {
            let off = pa as usize;
            if off + 8 <= buf.len() {
                buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
            }
        };

        // Page table: PGD -> PUD -> PMD -> PTE -> array_pa.
        write_u64(&mut buf, pgd_pa + pgd_idx * 8, (pud_pa + PTE_BASE) | 0x63);
        write_u64(&mut buf, pud_pa + pud_idx * 8, (pmd_pa + PTE_BASE) | 0x63);
        write_u64(&mut buf, pmd_pa + pmd_idx * 8, (pte_pa + PTE_BASE) | 0x63);
        write_u64(&mut buf, pte_pa + pte_idx * 8, (array_pa + PTE_BASE) | 0x63);

        // percpu base KVA (direct-mapped).
        let percpu_base_kva = percpu_base_pa + page_offset;

        // per_cpu_offsets: CPU 0 at percpu_base, CPU 1 at +stride, etc.
        let per_cpu_offsets: Vec<u64> = (0..num_cpus)
            .map(|cpu| cpu as u64 * percpu_stride)
            .collect();

        // Write pptrs[0..max_entries] into the bpf_array at array_pa.
        let pptrs_pa = array_pa + offsets.array_value as u64;
        for entry in 0..max_entries {
            let pptr_value = percpu_base_kva + entry as u64 * ((value_size as u64 + 7) & !7);
            write_u64(&mut buf, pptrs_pa + entry as u64 * 8, pptr_value);
        }

        let info = BpfMapInfo {
            map_pa: array_pa,
            map_kva,
            name: "test_percpu".into(),
            map_type: BPF_MAP_TYPE_PERCPU_ARRAY,
            map_flags: 0,
            key_size: 4,
            value_size,
            max_entries,
            value_kva: None,
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        (buf, pgd_pa, page_offset, info, offsets, per_cpu_offsets)
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn read_percpu_array_basic() {
        let num_cpus = 4u32;
        let value_size = 8u32;
        let (mut buf, cr3_pa, page_offset, info, offsets, per_cpu_offsets) =
            setup_percpu_array(num_cpus, 1, value_size);

        // Write distinct u64 values for each CPU at key 0.
        let percpu_base_pa: u64 = 0x20000;
        let stride: u64 = 0x1000;
        for cpu in 0..num_cpus {
            let pa = percpu_base_pa + cpu as u64 * stride;
            buf[pa as usize..pa as usize + 8]
                .copy_from_slice(&((cpu as u64 + 1) * 0x1111).to_ne_bytes());
        }

        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let result = read_percpu_array_value(
            &lookup_ctx(&mem, cr3_pa, page_offset, &offsets, false),
            &info,
            0,
            &per_cpu_offsets,
        );

        assert_eq!(result.len(), num_cpus as usize);
        for (cpu, entry) in result.iter().enumerate() {
            let bytes = entry.as_ref().expect("CPU value should be Some");
            let val = u64::from_ne_bytes(bytes[..8].try_into().unwrap());
            assert_eq!(val, (cpu as u64 + 1) * 0x1111);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn read_percpu_array_key_out_of_bounds() {
        let (buf, cr3_pa, page_offset, info, offsets, per_cpu_offsets) =
            setup_percpu_array(2, 1, 8);
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

        // key=1 is out of bounds for max_entries=1.
        let result = read_percpu_array_value(
            &lookup_ctx(&mem, cr3_pa, page_offset, &offsets, false),
            &info,
            1,
            &per_cpu_offsets,
        );
        assert!(result.is_empty());
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn read_percpu_array_wrong_map_type() {
        let (buf, cr3_pa, page_offset, mut info, offsets, per_cpu_offsets) =
            setup_percpu_array(2, 1, 8);
        info.map_type = BPF_MAP_TYPE_ARRAY;
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

        let result = read_percpu_array_value(
            &lookup_ctx(&mem, cr3_pa, page_offset, &offsets, false),
            &info,
            0,
            &per_cpu_offsets,
        );
        assert!(result.is_empty());
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn read_percpu_array_zero_pptr() {
        let (mut buf, cr3_pa, page_offset, info, offsets, per_cpu_offsets) =
            setup_percpu_array(2, 1, 8);

        // Zero out pptrs[0] so the percpu base is 0.
        let pptrs_pa = (0x14000 + offsets.array_value as u64) as usize;
        buf[pptrs_pa..pptrs_pa + 8].copy_from_slice(&0u64.to_ne_bytes());

        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let result = read_percpu_array_value(
            &lookup_ctx(&mem, cr3_pa, page_offset, &offsets, false),
            &info,
            0,
            &per_cpu_offsets,
        );
        assert!(result.is_empty());
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn read_percpu_array_multiple_entries() {
        let num_cpus = 2u32;
        let value_size = 4u32;
        let max_entries = 3u32;
        let (mut buf, cr3_pa, page_offset, info, offsets, per_cpu_offsets) =
            setup_percpu_array(num_cpus, max_entries, value_size);

        // Write distinct u32 values for each CPU at each key.
        let percpu_base_pa: u64 = 0x20000;
        let stride: u64 = 0x1000;
        let elem_size = 8u64; // round_up(4, 8)
        for key in 0..max_entries {
            for cpu in 0..num_cpus {
                let pa = percpu_base_pa + cpu as u64 * stride + key as u64 * elem_size;
                let val: u32 = key * 100 + cpu;
                buf[pa as usize..pa as usize + 4].copy_from_slice(&val.to_ne_bytes());
            }
        }

        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

        for key in 0..max_entries {
            let result = read_percpu_array_value(
                &lookup_ctx(&mem, cr3_pa, page_offset, &offsets, false),
                &info,
                key,
                &per_cpu_offsets,
            );
            assert_eq!(result.len(), num_cpus as usize);
            for (cpu, entry) in result.iter().enumerate() {
                let bytes = entry.as_ref().expect("CPU value should be Some");
                let val = u32::from_ne_bytes(bytes[..4].try_into().unwrap());
                assert_eq!(val, key * 100 + cpu as u32);
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn read_percpu_array_cpu_out_of_guest_memory() {
        let (buf, cr3_pa, page_offset, info, offsets, _) = setup_percpu_array(2, 1, 8);

        // Craft per_cpu_offsets so CPU 1's PA exceeds guest memory size.
        let bad_offset = buf.len() as u64 + 0x10000;
        let per_cpu_offsets = vec![0u64, bad_offset];

        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let result = read_percpu_array_value(
            &lookup_ctx(&mem, cr3_pa, page_offset, &offsets, false),
            &info,
            0,
            &per_cpu_offsets,
        );

        assert_eq!(result.len(), 2);
        assert!(result[0].is_some(), "CPU 0 should be readable");
        assert!(
            result[1].is_none(),
            "CPU 1 should be None (out of guest memory)"
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn read_percpu_array_zero_cpus() {
        // Use setup_percpu_array with num_cpus=0 so the page table and
        // pptrs[0] are valid but per_cpu_offsets is empty. This exercises
        // the per-CPU loop with an empty slice (not the pptr translation
        // failure path).
        let (buf, cr3_pa, page_offset, info, offsets, per_cpu_offsets) =
            setup_percpu_array(0, 1, 8);
        assert!(per_cpu_offsets.is_empty());

        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let result = read_percpu_array_value(
            &lookup_ctx(&mem, cr3_pa, page_offset, &offsets, false),
            &info,
            0,
            &per_cpu_offsets,
        );
        assert!(result.is_empty(), "zero CPUs should produce empty result");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn read_percpu_array_mixed_translatable() {
        let num_cpus = 4u32;
        let value_size = 8u32;
        let (mut buf, cr3_pa, page_offset, info, offsets, _) =
            setup_percpu_array(num_cpus, 1, value_size);

        // Write known data at CPU 0 and CPU 2 (valid offsets).
        let percpu_base_pa: u64 = 0x20000;
        let stride: u64 = 0x1000;
        buf[percpu_base_pa as usize..percpu_base_pa as usize + 8]
            .copy_from_slice(&0xAAAAu64.to_ne_bytes());
        let cpu2_pa = percpu_base_pa + 2 * stride;
        buf[cpu2_pa as usize..cpu2_pa as usize + 8].copy_from_slice(&0xCCCCu64.to_ne_bytes());

        // CPU 0 and 2 have valid offsets; CPU 1 and 3 have offsets
        // that produce PAs beyond the buffer.
        let bad = buf.len() as u64 + 0x10000;
        let per_cpu_offsets = vec![0, bad, 2 * stride, bad + stride];

        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let result = read_percpu_array_value(
            &lookup_ctx(&mem, cr3_pa, page_offset, &offsets, false),
            &info,
            0,
            &per_cpu_offsets,
        );

        assert_eq!(result.len(), 4);
        // CPU 0: valid.
        let v0 = result[0].as_ref().expect("CPU 0 should be Some");
        assert_eq!(u64::from_ne_bytes(v0[..8].try_into().unwrap()), 0xAAAA);
        // CPU 1: out of bounds.
        assert!(result[1].is_none(), "CPU 1 should be None");
        // CPU 2: valid.
        let v2 = result[2].as_ref().expect("CPU 2 should be Some");
        assert_eq!(u64::from_ne_bytes(v2[..8].try_into().unwrap()), 0xCCCC);
        // CPU 3: out of bounds.
        assert!(result[3].is_none(), "CPU 3 should be None");
    }

    #[test]
    fn read_percpu_array_unmapped_bpf_array() {
        // bpf_array KVA that cannot be translated (no page table,
        // not in direct mapping) — translate_any_kva returns None.
        let buf = vec![0u8; 0x20000];
        // SAFETY: buf is a live Vec<u8> owned for the test's duration.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let offsets = BpfMapOffsets {
            map_name: 32,
            map_type: 24,
            map_flags: 28,
            key_size: 44,
            value_size: 48,
            max_entries: 52,
            array_value: 256,
            xa_node_slots: 16,
            xa_node_shift: 0,
            idr_xa_head: 8,
            idr_next: 20,
            map_btf: 0,
            map_btf_value_type_id: 0,
            btf_data: 0,
            btf_data_size: 0,
            htab_offsets: None,
        };

        // map_kva points to an untranslatable address: outside direct
        // mapping range and page table (cr3=0) is all zeros.
        let info = BpfMapInfo {
            map_pa: 0,
            map_kva: 0xFFFF_C900_DEAD_0000,
            name: "test_percpu".into(),
            map_type: BPF_MAP_TYPE_PERCPU_ARRAY,
            map_flags: 0,
            key_size: 4,
            value_size: 8,
            max_entries: 1,
            value_kva: None,
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        let per_cpu_offsets = vec![0u64, 0x1000];
        let result = read_percpu_array_value(
            &lookup_ctx(&mem, 0, 0xFFFF_8880_0000_0000, &offsets, false),
            &info,
            0,
            &per_cpu_offsets,
        );
        assert!(
            result.is_empty(),
            "unmapped bpf_array should return empty vec"
        );
    }
}
