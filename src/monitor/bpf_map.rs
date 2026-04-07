//! Host-side BPF map discovery and write via guest physical memory.
//!
//! Walks the kernel's `map_idr` xarray from the host, finds a BPF map
//! by name suffix, and provides read/write access to the map's value
//! region. No guest cooperation is needed — all reads go through the
//! guest physical memory mapping.
//!
//! Address translation strategy:
//! - `map_idr` is a kernel BSS symbol: use `text_kva_to_pa`.
//! - xa_node structs are SLAB-allocated (direct mapping): use `kva_to_pa`.
//! - bpf_map/bpf_array for MMAPABLE maps are vmalloc'd: use `translate_kva`.
//! - .bss value region is vmalloc'd: use `translate_kva`.

use super::btf_offsets::BpfMapOffsets;
use super::reader::GuestMem;
use super::symbols::{kva_to_pa, text_kva_to_pa};

/// BPF_MAP_TYPE_ARRAY from include/uapi/linux/bpf.h.
const BPF_MAP_TYPE_ARRAY: u32 = 2;

/// BPF_OBJ_NAME_LEN from include/linux/bpf.h.
const BPF_OBJ_NAME_LEN: usize = 16;

/// XA_CHUNK_SHIFT = 6, XA_CHUNK_SIZE = 64.
const XA_CHUNK_SIZE: u64 = 64;

/// Discovered BPF map metadata and value location.
#[derive(Debug, Clone)]
pub struct BpfMapInfo {
    /// Guest physical address of the `struct bpf_map`.
    pub map_pa: u64,
    /// Map name (null-terminated, up to BPF_OBJ_NAME_LEN).
    pub name: String,
    /// `map_type` field value.
    pub map_type: u32,
    /// `map_flags` field value.
    pub map_flags: u32,
    /// `value_size` field value.
    pub value_size: u32,
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
/// Translate a kernel virtual address to a GuestMem offset, trying
/// direct mapping first, then page table walk.
///
/// BPF map structs are SLAB-allocated (linear map) or vmalloc'd
/// (modules, BPF programs). The page table walk handles both vmalloc
/// and module addresses.
fn translate_any_kva(
    mem: &GuestMem,
    cr3_pa: u64,
    page_offset: u64,
    kva: u64,
    l5: bool,
) -> Option<u64> {
    // Linear map: PAGE_OFFSET..PAGE_END — SLAB allocations
    let direct_pa = kva_to_pa(kva, page_offset);
    if direct_pa < mem.size() {
        return Some(direct_pa);
    }
    // Vmalloc / module addresses: page table walk
    mem.translate_kva(cr3_pa, kva, l5)
}

pub(crate) fn find_all_bpf_maps(
    mem: &GuestMem,
    cr3_pa: u64,
    page_offset: u64,
    map_idr_kva: u64,
    offsets: &BpfMapOffsets,
    l5: bool,
) -> Vec<BpfMapInfo> {
    let idr_pa = text_kva_to_pa(map_idr_kva);

    let xa_head = mem.read_u64(idr_pa, offsets.idr_xa_head);
    if xa_head == 0 {
        return Vec::new();
    }
    // idr_next is the next ID the kernel will allocate. All live entries
    // have IDs in 0..idr_next, so scanning beyond it only hits empty or
    // wrapped slots.
    let idr_next = mem.read_u32(idr_pa, offsets.idr_next);

    let mut maps = Vec::new();

    for id in 0..idr_next {
        let Some(entry) = xa_load(
            mem,
            page_offset,
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

        let Some(map_pa) = translate_any_kva(mem, cr3_pa, page_offset, entry, l5) else {
            continue;
        };

        let mut name_buf = [0u8; BPF_OBJ_NAME_LEN];
        mem.read_bytes(map_pa + offsets.map_name as u64, &mut name_buf);
        let name_len = name_buf
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(BPF_OBJ_NAME_LEN);
        let name = String::from_utf8_lossy(&name_buf[..name_len]).to_string();

        let map_type = mem.read_u32(map_pa, offsets.map_type);
        let map_flags = mem.read_u32(map_pa, offsets.map_flags);
        let value_size = mem.read_u32(map_pa, offsets.value_size);

        // value_kva is only meaningful for ARRAY maps where bpf_array
        // embeds bpf_map at offset 0 and the value flex array is inline.
        let value_kva = if map_type == BPF_MAP_TYPE_ARRAY {
            Some(entry + offsets.array_value as u64)
        } else {
            None
        };

        let btf_kva = mem.read_u64(map_pa, offsets.map_btf);
        let btf_value_type_id = mem.read_u32(map_pa, offsets.map_btf_value_type_id);

        maps.push(BpfMapInfo {
            map_pa,
            name,
            map_type,
            map_flags,
            value_size,
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
    mem: &GuestMem,
    cr3_pa: u64,
    page_offset: u64,
    map_idr_kva: u64,
    name_suffix: &str,
    offsets: &BpfMapOffsets,
    l5: bool,
) -> Option<BpfMapInfo> {
    find_all_bpf_maps(mem, cr3_pa, page_offset, map_idr_kva, offsets, l5)
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
    mem: &GuestMem,
    cr3_pa: u64,
    map_info: &BpfMapInfo,
    offset: usize,
    data: &[u8],
    l5: bool,
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
        let Some(pa) = mem.translate_kva(cr3_pa, kva, l5) else {
            return false;
        };
        mem.write_u8(pa, 0, byte);
    }
    true
}

/// Write a u32 to a BPF map's value region at `offset`.
pub(crate) fn write_bpf_map_value_u32(
    mem: &GuestMem,
    cr3_pa: u64,
    map_info: &BpfMapInfo,
    offset: usize,
    val: u32,
    l5: bool,
) -> bool {
    write_bpf_map_value(mem, cr3_pa, map_info, offset, &val.to_ne_bytes(), l5)
}

/// Read bytes from a BPF map's value region at `offset`.
///
/// Translates the value KVA (vmalloc'd for .bss maps) through the
/// page table to find the guest physical address, then reads directly.
/// Returns `None` if the map has no value KVA (non-ARRAY map),
/// `offset + len` exceeds `value_size`, or any page in the range
/// is unmapped.
pub(crate) fn read_bpf_map_value(
    mem: &GuestMem,
    cr3_pa: u64,
    map_info: &BpfMapInfo,
    offset: usize,
    len: usize,
    l5: bool,
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
        let pa = mem.translate_kva(cr3_pa, kva, l5)?;
        *byte = mem.read_u8(pa, 0);
    }
    Some(buf)
}

/// Read a u32 from a BPF map's value region at `offset`.
pub(crate) fn read_bpf_map_value_u32(
    mem: &GuestMem,
    cr3_pa: u64,
    map_info: &BpfMapInfo,
    offset: usize,
    l5: bool,
) -> Option<u32> {
    let bytes = read_bpf_map_value(mem, cr3_pa, map_info, offset, 4, l5)?;
    Some(u32::from_ne_bytes(bytes.try_into().unwrap()))
}

/// Typed value read from or written to a BPF map field.
///
/// The variant must match the [`BpfFieldKind`] of the target field.
/// `write_field` returns `false` on mismatch.
#[derive(Debug, Clone, PartialEq)]
pub enum BpfValue {
    Bool(bool),
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    Bytes(Vec<u8>),
}

/// Discriminant for a BPF map field's type, resolved from BTF.
///
/// Determined by chasing the field's BTF type chain (through Volatile,
/// Const, Typedef, TypeTag, Restrict) to the underlying Int or Enum.
/// Falls back to [`Bytes`](Self::Bytes) for non-standard sizes.
#[derive(Debug, Clone, PartialEq)]
pub enum BpfFieldKind {
    Bool,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
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

/// Chase modifier chains (Volatile, Const, Typedef, TypeTag, Restrict)
/// to reach the underlying Struct.
fn resolve_to_struct(btf: &btf_rs::Btf, type_id: u32) -> Option<btf_rs::Struct> {
    let mut t = btf.resolve_type_by_id(type_id).ok()?;
    for _ in 0..20 {
        match t {
            btf_rs::Type::Struct(s) => return Some(s),
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
    mem: &GuestMem,
    cr3_pa: u64,
    map: &BpfMapInfo,
    field: &BpfFieldInfo,
    l5: bool,
) -> Option<BpfValue> {
    let bytes = read_bpf_map_value(mem, cr3_pa, map, field.offset, field.size, l5)?;
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
    mem: &GuestMem,
    cr3_pa: u64,
    map: &BpfMapInfo,
    field: &BpfFieldInfo,
    val: BpfValue,
    l5: bool,
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
    write_bpf_map_value(mem, cr3_pa, map, field.offset, &bytes, l5)
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
    offsets: BpfMapOffsets,
}

impl<'a> BpfMapAccessor<'a> {
    /// Create from an existing [`GuestKernel`] and vmlinux path.
    ///
    /// Only parses BTF from the vmlinux (not the full ELF symbol table,
    /// which the `GuestKernel` already has).
    ///
    /// [`GuestKernel`]: super::guest::GuestKernel
    pub fn from_guest_kernel(
        kernel: &'a super::guest::GuestKernel<'a>,
        vmlinux: &std::path::Path,
    ) -> anyhow::Result<Self> {
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

    /// Enumerate all BPF maps in the kernel's `map_idr`.
    ///
    /// Returns metadata for every map whose KVA can be translated.
    /// No filtering by type or name.
    pub fn maps(&self) -> Vec<BpfMapInfo> {
        find_all_bpf_maps(
            self.kernel.mem(),
            self.kernel.cr3_pa(),
            self.kernel.page_offset(),
            self.map_idr_kva,
            &self.offsets,
            self.kernel.l5(),
        )
    }

    /// Find the first BPF ARRAY map whose name ends with `name_suffix`.
    ///
    /// Only returns `BPF_MAP_TYPE_ARRAY` maps. Use [`maps`](Self::maps)
    /// to enumerate maps of all types.
    pub fn find_map(&self, name_suffix: &str) -> Option<BpfMapInfo> {
        find_bpf_map(
            self.kernel.mem(),
            self.kernel.cr3_pa(),
            self.kernel.page_offset(),
            self.map_idr_kva,
            name_suffix,
            &self.offsets,
            self.kernel.l5(),
        )
    }

    /// Read bytes from a map's value region.
    ///
    /// Returns `None` if the map has no value KVA (non-ARRAY map)
    /// or any page in the range is unmapped.
    pub fn read_value(&self, map: &BpfMapInfo, offset: usize, len: usize) -> Option<Vec<u8>> {
        read_bpf_map_value(
            self.kernel.mem(),
            self.kernel.cr3_pa(),
            map,
            offset,
            len,
            self.kernel.l5(),
        )
    }

    /// Write bytes to a map's value region.
    ///
    /// Returns `false` if the map has no value KVA (non-ARRAY map)
    /// or any page in the range is unmapped.
    pub fn write_value(&self, map: &BpfMapInfo, offset: usize, data: &[u8]) -> bool {
        write_bpf_map_value(
            self.kernel.mem(),
            self.kernel.cr3_pa(),
            map,
            offset,
            data,
            self.kernel.l5(),
        )
    }

    /// Write a u32 to a map's value region.
    pub fn write_value_u32(&self, map: &BpfMapInfo, offset: usize, val: u32) -> bool {
        write_bpf_map_value_u32(
            self.kernel.mem(),
            self.kernel.cr3_pa(),
            map,
            offset,
            val,
            self.kernel.l5(),
        )
    }

    /// Read a u32 from a map's value region.
    pub fn read_value_u32(&self, map: &BpfMapInfo, offset: usize) -> Option<u32> {
        read_bpf_map_value_u32(
            self.kernel.mem(),
            self.kernel.cr3_pa(),
            map,
            offset,
            self.kernel.l5(),
        )
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
        read_typed_field(
            self.kernel.mem(),
            self.kernel.cr3_pa(),
            map,
            fi,
            self.kernel.l5(),
        )
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
        write_typed_field(
            self.kernel.mem(),
            self.kernel.cr3_pa(),
            map,
            fi,
            val,
            self.kernel.l5(),
        )
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
    /// Builds a [`GuestKernel`] internally, resolves BTF offsets,
    /// and locates the `map_idr` symbol.
    ///
    /// Prefer [`BpfMapAccessor::from_guest_kernel`] when you already
    /// have a `GuestKernel` to avoid re-parsing the ELF.
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
    pub fn as_accessor(&self) -> BpfMapAccessor<'_> {
        BpfMapAccessor {
            kernel: &self.kernel,
            map_idr_kva: self.map_idr_kva,
            offsets: self.offsets.clone(),
        }
    }

    /// Access the underlying [`GuestKernel`] for low-level memory reads.
    ///
    /// [`GuestKernel`]: super::guest::GuestKernel
    pub fn guest_kernel(&self) -> &super::guest::GuestKernel<'a> {
        &self.kernel
    }

    /// Enumerate all BPF maps in the kernel's `map_idr`.
    pub fn maps(&self) -> Vec<BpfMapInfo> {
        self.as_accessor().maps()
    }

    /// Find the first BPF ARRAY map whose name ends with `name_suffix`.
    ///
    /// Only returns `BPF_MAP_TYPE_ARRAY` maps. Use [`maps`](Self::maps)
    /// to enumerate maps of all types.
    pub fn find_map(&self, name_suffix: &str) -> Option<BpfMapInfo> {
        self.as_accessor().find_map(name_suffix)
    }

    /// Read bytes from a map's value region.
    pub fn read_value(&self, map: &BpfMapInfo, offset: usize, len: usize) -> Option<Vec<u8>> {
        self.as_accessor().read_value(map, offset, len)
    }

    /// Write bytes to a map's value region.
    pub fn write_value(&self, map: &BpfMapInfo, offset: usize, data: &[u8]) -> bool {
        self.as_accessor().write_value(map, offset, data)
    }

    /// Write a u32 to a map's value region.
    pub fn write_value_u32(&self, map: &BpfMapInfo, offset: usize, val: u32) -> bool {
        self.as_accessor().write_value_u32(map, offset, val)
    }

    /// Read a u32 from a map's value region.
    pub fn read_value_u32(&self, map: &BpfMapInfo, offset: usize) -> Option<u32> {
        self.as_accessor().read_value_u32(map, offset)
    }

    /// Resolve the value layout from the map's BTF.
    pub fn resolve_value_layout(&self, map: &BpfMapInfo) -> Option<BpfValueLayout> {
        self.as_accessor().resolve_value_layout(map)
    }

    /// Read a typed field from a map's value region.
    pub fn read_field(
        &self,
        map: &BpfMapInfo,
        layout: &BpfValueLayout,
        field: &str,
    ) -> Option<BpfValue> {
        self.as_accessor().read_field(map, layout, field)
    }

    /// Write a typed field to a map's value region.
    pub fn write_field(
        &self,
        map: &BpfMapInfo,
        layout: &BpfValueLayout,
        field: &str,
        val: BpfValue,
    ) -> bool {
        self.as_accessor().write_field(map, layout, field, val)
    }
}

/// Load an entry from an xarray by index.
///
/// xa_node structs are SLAB-allocated and live in the direct mapping,
/// so their KVAs are translated via `kva_to_pa(kva, page_offset)`.
/// `slots_off` and `shift_off` are BTF-resolved byte offsets of
/// `slots` and `shift` within `struct xa_node`.
///
/// Returns `Some(0)` for empty slots or `Some(ptr)` for populated
/// entries. Out-of-bounds reads return 0 (empty slot).
fn xa_load(
    mem: &GuestMem,
    page_offset: u64,
    xa_head: u64,
    index: u64,
    slots_off: usize,
    shift_off: usize,
) -> Option<u64> {
    if xa_head == 0 {
        return Some(0);
    }

    // Check if xa_head is an internal node (bit 1 set) or a direct entry.
    if xa_head & 2 == 0 {
        // Single-entry xarray: only index 0 is valid.
        return if index == 0 { Some(xa_head) } else { Some(0) };
    }

    // xa_head is a node pointer. Clear the internal marker bits.
    let mut node_kva = xa_head & !3u64;
    let mut shift = xa_node_shift(mem, page_offset, node_kva, shift_off);

    loop {
        let slot_idx = (index >> shift) & (XA_CHUNK_SIZE - 1);
        let slot_pa = kva_to_pa(node_kva + slots_off as u64 + slot_idx * 8, page_offset);
        let entry = mem.read_u64(slot_pa, 0);

        if entry == 0 {
            return Some(0);
        }

        if entry & 2 == 0 {
            // Leaf entry — a pointer to a bpf_map (or other object).
            return Some(entry);
        }

        // Internal node — descend.
        node_kva = entry & !3u64;
        if shift < 6 {
            return Some(0);
        }
        shift -= 6; // XA_CHUNK_SHIFT
    }
}

/// Read the `shift` field from an xa_node (SLAB-allocated, direct mapping).
fn xa_node_shift(mem: &GuestMem, page_offset: u64, node_kva: u64, shift_off: usize) -> u64 {
    let pa = kva_to_pa(node_kva, page_offset);
    mem.read_u8(pa, shift_off) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::symbols::START_KERNEL_MAP;

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
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let pa = mem.translate_kva(cr3_pa, kva, false);
        assert_eq!(pa, Some(data_pa));
        // Read through the translated PA to verify correctness.
        assert_eq!(mem.read_u64(pa.unwrap(), 0), 0xDEAD_BEEF_CAFE_1234);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn translate_kva_with_offset() {
        let (buf, cr3_pa, kva, data_pa) = setup_page_table();
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        // KVA + 0x100 should map to data_pa + 0x100
        let pa = mem.translate_kva(cr3_pa, kva + 0x100, false);
        assert_eq!(pa, Some(data_pa + 0x100));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn translate_kva_unmapped() {
        let (buf, cr3_pa, _, _) = setup_page_table();
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        // A completely different address that has no PGD entry.
        let pa = mem.translate_kva(cr3_pa, 0xFFFF_FFFF_8000_0000, false);
        assert_eq!(pa, None);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn translate_kva_unmapped_pte() {
        let (buf, cr3_pa, kva, _) = setup_page_table();
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
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

        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
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

        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
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

        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
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

        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
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

        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
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

        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        assert_eq!(mem.translate_kva(pgd_pa, kva, false), None);
    }

    // -- write_bpf_map_value tests --

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn write_bpf_map_value_u32_roundtrip() {
        let (mut buf, cr3_pa, kva, data_pa) = setup_page_table();
        let mem = GuestMem::new(buf.as_mut_ptr(), buf.len() as u64);

        let info = BpfMapInfo {
            map_pa: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            value_size: 64,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        // Write u32 at offset 4 within the value region.
        assert!(write_bpf_map_value_u32(
            &mem,
            cr3_pa,
            &info,
            4,
            0xABCD_1234,
            false
        ));
        // Read it back via direct PA access.
        assert_eq!(mem.read_u32(data_pa, 4), 0xABCD_1234);
    }

    #[test]
    fn read_bytes_basic() {
        let buf = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let mut out = [0u8; 4];
        let n = mem.read_bytes(2, &mut out);
        assert_eq!(n, 4);
        assert_eq!(out, [3, 4, 5, 6]);
    }

    #[test]
    fn read_bytes_past_end() {
        let buf = [1u8, 2, 3, 4];
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let mut out = [0u8; 8];
        let n = mem.read_bytes(2, &mut out);
        assert_eq!(n, 2); // Only 2 bytes available from PA 2.
        assert_eq!(out[..2], [3, 4]);
    }

    #[test]
    fn read_bytes_at_boundary() {
        let buf = [0xFFu8; 8];
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let mut out = [0u8; 8];
        let n = mem.read_bytes(8, &mut out);
        assert_eq!(n, 0); // PA == size, nothing to read.
    }

    #[test]
    fn write_u32_roundtrip() {
        let mut buf = [0u8; 16];
        let mem = GuestMem::new(buf.as_mut_ptr(), buf.len() as u64);
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
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
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
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        assert_eq!(xa_load(&mem, 0, xa_head, 0, 0, 0), Some(xa_head));
    }

    #[test]
    fn xa_load_single_entry_index_nonzero() {
        let xa_head: u64 = 0xFFFF_8880_0001_0000;
        let buf = [0u8; 8];
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        assert_eq!(xa_load(&mem, 0, xa_head, 1, 0, 0), Some(0));
        assert_eq!(xa_load(&mem, 0, xa_head, 63, 0, 0), Some(0));
    }

    /// Build a single-level xa_node in a buffer. The node has shift=0
    /// (leaf level) and the given slots populated with entry pointers.
    /// Returns (buffer, xa_head pointing to the node, page_offset used).
    ///
    /// Layout: node at PA 0x1000, slots at PA 0x1000 + slots_off.
    /// page_offset chosen so kva_to_pa(node_kva, page_offset) = 0x1000.
    fn setup_xa_node(slots: &[(u64, u64)], slots_off: usize) -> (Vec<u8>, u64, u64) {
        let node_pa: u64 = 0x1000;
        let page_offset: u64 = 0xFFFF_8880_0000_0000;
        let node_kva = node_pa + page_offset;

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
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);

        assert_eq!(
            xa_load(&mem, page_offset, xa_head, 3, slots_off, 0),
            Some(entry_ptr)
        );
    }

    #[test]
    fn xa_load_multi_entry_empty_slot() {
        let slots_off = 16;
        let (buf, xa_head, page_offset) = setup_xa_node(&[], slots_off);
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);

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
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);

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
            value_size: 48,
            array_value: 256,
            xa_node_slots: 16,
            xa_node_shift: 0,
            idr_xa_head: 8,
            idr_next: 20,
            map_btf: 0,
            map_btf_value_type_id: 0,
            btf_data: 0,
            btf_data_size: 0,
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
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);

        let result = find_bpf_map(
            &mem,
            cr3_pa,
            0xFFFF_8880_0000_0000, // page_offset (unused for this path)
            idr_kva,
            ".bss",
            &offsets,
            false,
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
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);

        let result = find_bpf_map(
            &mem,
            cr3_pa,
            0xFFFF_8880_0000_0000,
            idr_kva,
            ".data",
            &offsets,
            false,
        );
        assert!(result.is_none());
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn find_bpf_map_skips_non_array_type() {
        // map_type = 1 (BPF_MAP_TYPE_HASH), not BPF_MAP_TYPE_ARRAY.
        let (buf, cr3_pa, idr_kva, offsets) = setup_find_bpf_map("test.bss", 1, 64);
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);

        let result = find_bpf_map(
            &mem,
            cr3_pa,
            0xFFFF_8880_0000_0000,
            idr_kva,
            ".bss",
            &offsets,
            false,
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
            value_size: 48,
            array_value: 256,
            xa_node_slots: 16,
            xa_node_shift: 0,
            idr_xa_head: 8,
            idr_next: 20,
            map_btf: 0,
            map_btf_value_type_id: 0,
            btf_data: 0,
            btf_data_size: 0,
        };
        let idr_pa: u64 = 0x1000;
        let size = 0x2000;
        let buf = vec![0u8; size]; // All zeros, so xa_head = 0.

        let start_kernel_map: u64 = START_KERNEL_MAP;
        let idr_kva = idr_pa + start_kernel_map;

        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let result = find_bpf_map(
            &mem,
            0x10000,
            0xFFFF_8880_0000_0000,
            idr_kva,
            ".bss",
            &offsets,
            false,
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
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let pa = mem.translate_kva(cr3_pa, kva, true);
        assert_eq!(pa, Some(data_pa));
        assert_eq!(mem.read_u64(pa.unwrap(), 0), 0x5555_AAAA_1234_5678);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn translate_kva_5level_with_offset() {
        let (buf, cr3_pa, kva, data_pa) = setup_5level_page_table();
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let pa = mem.translate_kva(cr3_pa, kva + 0x100, true);
        assert_eq!(pa, Some(data_pa + 0x100));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn translate_kva_5level_unmapped_pml5() {
        let (buf, cr3_pa, _, _) = setup_5level_page_table();
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
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
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
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
        let mem = GuestMem::new(buf.as_mut_ptr(), buf.len() as u64);

        let info = BpfMapInfo {
            map_pa: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            value_size: 16,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        let payload = [0xDE, 0xAD, 0xBE, 0xEF];
        assert!(write_bpf_map_value(&mem, cr3_pa, &info, 0, &payload, false));

        // Verify each byte was written.
        for (i, &expected) in payload.iter().enumerate() {
            assert_eq!(buf[data_pa as usize + i], expected);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn write_bpf_map_value_fails_on_unmapped_kva() {
        let (mut buf, cr3_pa, _, _) = setup_page_table();
        let mem = GuestMem::new(buf.as_mut_ptr(), buf.len() as u64);

        let info = BpfMapInfo {
            map_pa: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            value_size: 16,
            value_kva: Some(0xFFFF_FFFF_8000_0000), // Unmapped KVA.
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        assert!(!write_bpf_map_value(&mem, cr3_pa, &info, 0, &[0xFF], false));
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
        let page_offset: u64 = 0xFFFF_8880_0000_0000;
        let root_kva = root_pa + page_offset;
        let child_kva = child_pa + page_offset;

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
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);

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
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);

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
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);

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
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);

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
            value_size: 48,
            array_value: 256,
            xa_node_slots: 16,
            xa_node_shift: 0,
            idr_xa_head: 8,
            idr_next: 20,
            map_btf: 0,
            map_btf_value_type_id: 0,
            btf_data: 0,
            btf_data_size: 0,
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
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);

        let result = find_bpf_map(&mem, cr3_pa, page_offset, idr_kva, ".bss", &offsets, false);
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
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);

        let result = find_bpf_map(
            &mem,
            cr3_pa,
            0xFFFF_8880_0000_0000,
            idr_kva,
            ".bss",
            &offsets,
            false,
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
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);

        // The name doesn't end with ".bss" — the '!' is the 16th char.
        let result = find_bpf_map(
            &mem,
            cr3_pa,
            0xFFFF_8880_0000_0000,
            idr_kva,
            ".bss",
            &offsets,
            false,
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
        let mem = GuestMem::new(buf.as_mut_ptr(), buf.len() as u64);

        let info = BpfMapInfo {
            map_pa: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            value_size: 64,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        // Write at offset 8 within the value region.
        let payload = [0x11, 0x22, 0x33, 0x44];
        assert!(write_bpf_map_value(&mem, cr3_pa, &info, 8, &payload, false));

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
        let mem = GuestMem::new(buf.as_mut_ptr(), buf.len() as u64);

        let info = BpfMapInfo {
            map_pa: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            value_size: 64,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        // Zero-length write should succeed without doing anything.
        assert!(write_bpf_map_value(&mem, cr3_pa, &info, 0, &[], false));
    }

    // -- write_bpf_map_value_u32 with 5-level paging --

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn write_bpf_map_value_u32_5level() {
        let (mut buf, cr3_pa, kva, data_pa) = setup_5level_page_table();
        let mem = GuestMem::new(buf.as_mut_ptr(), buf.len() as u64);

        let info = BpfMapInfo {
            map_pa: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            value_size: 64,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        assert!(write_bpf_map_value_u32(
            &mem,
            cr3_pa,
            &info,
            0,
            0xCAFE_BABE,
            true
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

        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
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

        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
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

        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
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
            value_size: 48,
            array_value: 256,
            xa_node_slots: 16,
            xa_node_shift: 0,
            idr_xa_head: 8,
            idr_next: 20,
            map_btf: 0,
            map_btf_value_type_id: 0,
            btf_data: 0,
            btf_data_size: 0,
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

        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let result = find_bpf_map(
            &mem,
            pgd_pa,
            0xFFFF_8880_0000_0000,
            idr_kva,
            ".bss",
            &offsets,
            false,
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
            value_size: 48,
            array_value: 256,
            xa_node_slots: 16,
            xa_node_shift: 0,
            idr_xa_head: 8,
            idr_next: 20,
            map_btf: 0,
            map_btf_value_type_id: 0,
            btf_data: 0,
            btf_data_size: 0,
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

        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let result = find_bpf_map(
            &mem,
            pml5_pa,
            0xFFFF_8880_0000_0000,
            idr_kva,
            ".bss",
            &offsets,
            true,
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
        let mem = GuestMem::new(buf.as_mut_ptr(), buf.len() as u64);

        let info = BpfMapInfo {
            map_pa: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            value_size: 0x2000,
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
            &mem, cr3_pa, &info, 0xFFE, val, false
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
        let mem = GuestMem::new(buf.as_mut_ptr(), buf.len() as u64);

        let info = BpfMapInfo {
            map_pa: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            value_size: 0x2000,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        // Write exactly at offset 0x1000 — first byte of page 2.
        assert!(write_bpf_map_value(
            &mem,
            cr3_pa,
            &info,
            0x1000,
            &[0x42],
            false
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
            value_size: 48,
            array_value: 256,
            xa_node_slots: 16,
            xa_node_shift: 0,
            idr_xa_head: 8,
            idr_next: 20,
            map_btf: 0,
            map_btf_value_type_id: 0,
            btf_data: 0,
            btf_data_size: 0,
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

        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let result = find_bpf_map(&mem, pgd_pa, page_offset, idr_kva, ".bss", &offsets, false);

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
        let mem = GuestMem::new(buf.as_mut_ptr(), buf.len() as u64);

        let info = BpfMapInfo {
            map_pa: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            value_size: 64,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        let val = read_bpf_map_value_u32(&mem, cr3_pa, &info, 4, false);
        assert_eq!(val, Some(0xCAFE_BABE));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn read_bpf_map_value_bytes() {
        let (mut buf, cr3_pa, kva, data_pa) = setup_page_table();
        buf[data_pa as usize..data_pa as usize + 4].copy_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);
        let mem = GuestMem::new(buf.as_mut_ptr(), buf.len() as u64);

        let info = BpfMapInfo {
            map_pa: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            value_size: 64,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        let bytes = read_bpf_map_value(&mem, cr3_pa, &info, 0, 4, false);
        assert_eq!(bytes, Some(vec![0xAA, 0xBB, 0xCC, 0xDD]));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn read_bpf_map_value_empty() {
        let (buf, cr3_pa, kva, _) = setup_page_table();
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);

        let info = BpfMapInfo {
            map_pa: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            value_size: 64,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        let bytes = read_bpf_map_value(&mem, cr3_pa, &info, 0, 0, false);
        assert_eq!(bytes, Some(vec![]));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn read_bpf_map_value_unmapped_returns_none() {
        let (buf, cr3_pa, _, _) = setup_page_table();
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);

        let info = BpfMapInfo {
            map_pa: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            value_size: 16,
            value_kva: Some(0xFFFF_FFFF_8000_0000), // Unmapped KVA.
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        assert_eq!(read_bpf_map_value(&mem, cr3_pa, &info, 0, 4, false), None);
        assert_eq!(read_bpf_map_value_u32(&mem, cr3_pa, &info, 0, false), None);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn write_then_read_bpf_map_value_roundtrip() {
        let (mut buf, cr3_pa, kva, _) = setup_page_table();
        let mem = GuestMem::new(buf.as_mut_ptr(), buf.len() as u64);

        let info = BpfMapInfo {
            map_pa: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            value_size: 64,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        // Write then read u32.
        assert!(write_bpf_map_value_u32(
            &mem,
            cr3_pa,
            &info,
            8,
            0x1234_5678,
            false
        ));
        assert_eq!(
            read_bpf_map_value_u32(&mem, cr3_pa, &info, 8, false),
            Some(0x1234_5678)
        );

        // Write then read bytes.
        let payload = [0x11, 0x22, 0x33, 0x44, 0x55];
        assert!(write_bpf_map_value(
            &mem, cr3_pa, &info, 16, &payload, false
        ));
        assert_eq!(
            read_bpf_map_value(&mem, cr3_pa, &info, 16, 5, false),
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

        let mem = GuestMem::new(buf.as_mut_ptr(), buf.len() as u64);

        let info = BpfMapInfo {
            map_pa: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            value_size: 0x2000,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        let bytes = read_bpf_map_value(&mem, cr3_pa, &info, 0xFFE, 4, false);
        assert_eq!(bytes, Some(vec![0xAA, 0xBB, 0xCC, 0xDD]));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn read_bpf_map_value_u32_5level() {
        let (mut buf, cr3_pa, kva, data_pa) = setup_5level_page_table();
        buf[data_pa as usize..data_pa as usize + 4].copy_from_slice(&0xDEAD_BEEFu32.to_ne_bytes());
        let mem = GuestMem::new(buf.as_mut_ptr(), buf.len() as u64);

        let info = BpfMapInfo {
            map_pa: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            value_size: 64,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        assert_eq!(
            read_bpf_map_value_u32(&mem, cr3_pa, &info, 0, true),
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
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);

        let maps = find_all_bpf_maps(&mem, cr3_pa, page_offset, idr_kva, &offsets, false);
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
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);

        let maps = find_all_bpf_maps(
            &mem,
            cr3_pa,
            0xFFFF_8880_0000_0000,
            idr_kva,
            &offsets,
            false,
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
            value_size: 48,
            array_value: 256,
            xa_node_slots: 16,
            xa_node_shift: 0,
            idr_xa_head: 8,
            idr_next: 20,
            map_btf: 0,
            map_btf_value_type_id: 0,
            btf_data: 0,
            btf_data_size: 0,
        };
        let buf = vec![0u8; 0x2000];
        let start_kernel_map: u64 = START_KERNEL_MAP;
        let idr_kva = 0x1000 + start_kernel_map;
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);

        let maps = find_all_bpf_maps(
            &mem,
            0x10000,
            0xFFFF_8880_0000_0000,
            idr_kva,
            &offsets,
            false,
        );
        assert!(maps.is_empty());
    }

    // -- value_kva Option tests (fix #4) --

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn read_value_returns_none_for_non_array_map() {
        let (buf, cr3_pa, _, _) = setup_page_table();
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);

        let info = BpfMapInfo {
            map_pa: 0,
            name: "hash.map".into(),
            map_type: 1, // HASH
            map_flags: 0,
            value_size: 64,
            value_kva: None,
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        assert!(read_bpf_map_value(&mem, cr3_pa, &info, 0, 4, false).is_none());
        assert!(read_bpf_map_value_u32(&mem, cr3_pa, &info, 0, false).is_none());
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn write_value_returns_false_for_non_array_map() {
        let (mut buf, cr3_pa, _, _) = setup_page_table();
        let mem = GuestMem::new(buf.as_mut_ptr(), buf.len() as u64);

        let info = BpfMapInfo {
            map_pa: 0,
            name: "hash.map".into(),
            map_type: 1, // HASH
            map_flags: 0,
            value_size: 64,
            value_kva: None,
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        assert!(!write_bpf_map_value(
            &mem,
            cr3_pa,
            &info,
            0,
            &[1, 2, 3, 4],
            false
        ));
        assert!(!write_bpf_map_value_u32(&mem, cr3_pa, &info, 0, 42, false));
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

        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let maps = find_all_bpf_maps(
            &mem,
            cr3_pa,
            0xFFFF_8880_0000_0000,
            idr_kva,
            &offsets,
            false,
        );
        assert_eq!(maps.len(), 1);
        assert_eq!(maps[0].map_flags, 0x0400);
    }

    // -- xa_node_shift non-zero offset test (fix #7) --

    #[test]
    fn xa_node_shift_nonzero_offset() {
        // Place shift at offset 8 within the xa_node instead of 0.
        let node_pa: u64 = 0x1000;
        let page_offset: u64 = 0xFFFF_8880_0000_0000;
        let node_kva = node_pa + page_offset;
        let shift_off: usize = 8;

        let mut buf = vec![0u8; 0x2000];
        // Write shift=6 at node_pa + 8.
        buf[node_pa as usize + shift_off] = 6;

        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
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
            value_size: 48,
            array_value: 256,
            xa_node_slots: 16,
            xa_node_shift: 0,
            idr_xa_head: 8,
            idr_next: 20,
            map_btf: 0,
            map_btf_value_type_id: 0,
            btf_data: 0,
            btf_data_size: 0,
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

        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let maps = find_all_bpf_maps(&mem, pgd_pa, page_offset, idr_kva, &offsets, false);

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
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);

        let info = BpfMapInfo {
            map_pa: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            value_size: 8,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        // Exactly at boundary: offset=4, len=4 -> 4+4=8 == value_size, ok.
        assert!(read_bpf_map_value(&mem, cr3_pa, &info, 4, 4, false).is_some());
        // One past: offset=4, len=5 -> 4+5=9 > 8, rejected.
        assert!(read_bpf_map_value(&mem, cr3_pa, &info, 4, 5, false).is_none());
        // Offset past end: offset=9, len=1 -> 9+1=10 > 8, rejected.
        assert!(read_bpf_map_value(&mem, cr3_pa, &info, 9, 1, false).is_none());
        // u32 past end: offset=6, 6+4=10 > 8, rejected.
        assert!(read_bpf_map_value_u32(&mem, cr3_pa, &info, 6, false).is_none());
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn write_value_rejects_out_of_bounds() {
        let (mut buf, cr3_pa, kva, _) = setup_page_table();
        let mem = GuestMem::new(buf.as_mut_ptr(), buf.len() as u64);

        let info = BpfMapInfo {
            map_pa: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            value_size: 8,
            value_kva: Some(kva),
            btf_kva: 0,
            btf_value_type_id: 0,
        };

        // Within bounds: offset=0, len=8.
        assert!(write_bpf_map_value(
            &mem, cr3_pa, &info, 0, &[0u8; 8], false
        ));
        // Past end: offset=0, len=9.
        assert!(!write_bpf_map_value(
            &mem, cr3_pa, &info, 0, &[0u8; 9], false
        ));
        // u32 past end: offset=6, 6+4=10 > 8.
        assert!(!write_bpf_map_value_u32(&mem, cr3_pa, &info, 6, 42, false));
        // u32 at boundary: offset=4, 4+4=8, ok.
        assert!(write_bpf_map_value_u32(&mem, cr3_pa, &info, 4, 42, false));
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
        let mem = GuestMem::new(buf.as_mut_ptr(), buf.len() as u64);

        let info = BpfMapInfo {
            map_pa: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            value_size: 64,
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
            &mem,
            cr3_pa,
            &info,
            &field_u32,
            BpfValue::U32(42),
            false
        ));
        let val = read_typed_field(&mem, cr3_pa, &info, &field_u32, false);
        assert_eq!(val, Some(BpfValue::U32(42)));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn read_write_field_all_types() {
        let (mut buf, cr3_pa, kva, _) = setup_page_table();
        let mem = GuestMem::new(buf.as_mut_ptr(), buf.len() as u64);

        let info = BpfMapInfo {
            map_pa: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            value_size: 64,
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
            &mem,
            cr3_pa,
            &info,
            &f,
            BpfValue::Bool(true),
            false
        ));
        assert_eq!(
            read_typed_field(&mem, cr3_pa, &info, &f, false),
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
            &mem,
            cr3_pa,
            &info,
            &f,
            BpfValue::U8(0xAB),
            false
        ));
        assert_eq!(
            read_typed_field(&mem, cr3_pa, &info, &f, false),
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
            &mem,
            cr3_pa,
            &info,
            &f,
            BpfValue::I8(-5),
            false
        ));
        assert_eq!(
            read_typed_field(&mem, cr3_pa, &info, &f, false),
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
            &mem,
            cr3_pa,
            &info,
            &f,
            BpfValue::U16(1234),
            false
        ));
        assert_eq!(
            read_typed_field(&mem, cr3_pa, &info, &f, false),
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
            &mem,
            cr3_pa,
            &info,
            &f,
            BpfValue::I16(-100),
            false
        ));
        assert_eq!(
            read_typed_field(&mem, cr3_pa, &info, &f, false),
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
            &mem,
            cr3_pa,
            &info,
            &f,
            BpfValue::U64(0xDEAD_BEEF),
            false
        ));
        assert_eq!(
            read_typed_field(&mem, cr3_pa, &info, &f, false),
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
            &mem,
            cr3_pa,
            &info,
            &f,
            BpfValue::I64(-999),
            false
        ));
        assert_eq!(
            read_typed_field(&mem, cr3_pa, &info, &f, false),
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
            &mem,
            cr3_pa,
            &info,
            &f,
            BpfValue::I32(-42),
            false
        ));
        assert_eq!(
            read_typed_field(&mem, cr3_pa, &info, &f, false),
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
            &mem,
            cr3_pa,
            &info,
            &f,
            BpfValue::Bytes(vec![1, 2, 3]),
            false
        ));
        assert_eq!(
            read_typed_field(&mem, cr3_pa, &info, &f, false),
            Some(BpfValue::Bytes(vec![1, 2, 3]))
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn write_field_type_mismatch_returns_false() {
        let (mut buf, cr3_pa, kva, _) = setup_page_table();
        let mem = GuestMem::new(buf.as_mut_ptr(), buf.len() as u64);

        let info = BpfMapInfo {
            map_pa: 0,
            name: "test.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            value_size: 64,
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
            &mem,
            cr3_pa,
            &info,
            &field,
            BpfValue::U64(1),
            false
        ));
        assert!(!write_typed_field(
            &mem,
            cr3_pa,
            &info,
            &field,
            BpfValue::Bool(true),
            false
        ));
        assert!(!write_typed_field(
            &mem,
            cr3_pa,
            &info,
            &field,
            BpfValue::I32(-1),
            false
        ));

        // Bytes field: wrong length.
        let field_bytes = BpfFieldInfo {
            name: "data".into(),
            offset: 4,
            size: 3,
            kind: BpfFieldKind::Bytes(3),
        };
        assert!(!write_typed_field(
            &mem,
            cr3_pa,
            &info,
            &field_bytes,
            BpfValue::Bytes(vec![1, 2]),
            false
        ));
        assert!(!write_typed_field(
            &mem,
            cr3_pa,
            &info,
            &field_bytes,
            BpfValue::Bytes(vec![1, 2, 3, 4]),
            false
        ));
        // Correct length works.
        assert!(write_typed_field(
            &mem,
            cr3_pa,
            &info,
            &field_bytes,
            BpfValue::Bytes(vec![1, 2, 3]),
            false
        ));
    }

    // -- BpfMapInfo btf fields --

    #[test]
    fn bpf_map_info_btf_fields_default_zero() {
        let info = BpfMapInfo {
            map_pa: 0x1000,
            name: "test".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            value_size: 32,
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
            name: "test".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            map_flags: 0,
            value_size: 32,
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
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let maps = find_all_bpf_maps(
            &mem,
            cr3_pa,
            0xFFFF_8880_0000_0000,
            idr_kva,
            &offsets,
            false,
        );
        assert_eq!(maps.len(), 1);
        assert_eq!(maps[0].btf_kva, 0);
        assert_eq!(maps[0].btf_value_type_id, 0);

        // Write nonzero values and re-scan.
        let btf_kva_val: u64 = 0xFFFF_8880_DEAD_0000;
        buf[btf_off..btf_off + 8].copy_from_slice(&btf_kva_val.to_ne_bytes());
        buf[btf_tid_off..btf_tid_off + 4].copy_from_slice(&7u32.to_ne_bytes());

        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let maps = find_all_bpf_maps(
            &mem,
            cr3_pa,
            0xFFFF_8880_0000_0000,
            idr_kva,
            &offsets,
            false,
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
            value_size: 48,
            array_value: 256,
            xa_node_slots: 16,
            xa_node_shift: 0,
            idr_xa_head: 8,
            idr_next: 20,
            map_btf: 0,
            map_btf_value_type_id: 0,
            btf_data: 0,
            btf_data_size: 0,
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

        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let maps = find_all_bpf_maps(&mem, pgd_pa, page_offset, idr_kva, &offsets, false);

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
        let (buf, cr3_pa, kva, data_pa) = setup_page_table_l0_256();
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
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
        let (buf, cr3_pa, kva, data_pa) = setup_page_table_l0_256();
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let pa = mem.translate_kva(cr3_pa, kva + 0x100, false);
        assert_eq!(pa, Some(data_pa + 0x100));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn translate_kva_l0_index_256_unmapped_neighbor() {
        let (buf, cr3_pa, kva, _) = setup_page_table_l0_256();
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
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
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let pa = mem.translate_kva(cr3_pa, kva, false);
        assert_eq!(pa, Some(data_pa), "64KB vmalloc walk should resolve");
        assert_eq!(mem.read_u64(pa.unwrap(), 0), 0x1234_5678_ABCD_EF00);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn translate_kva_vmalloc_64k_with_offset() {
        let (buf, cr3_pa, kva, data_pa) = setup_page_table_vmalloc_64k();
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let pa = mem.translate_kva(cr3_pa, kva + 0x100, false);
        assert_eq!(pa, Some(data_pa + 0x100));
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn translate_kva_vmalloc_64k_unmapped_neighbor() {
        let (buf, cr3_pa, kva, _) = setup_page_table_vmalloc_64k();
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let unmapped = kva + (1u64 << 42);
        assert_eq!(mem.translate_kva(cr3_pa, unmapped, false), None);
    }
}
