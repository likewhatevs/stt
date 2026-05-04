//! BTF offsets for `BPF_MAP_TYPE_STRUCT_OPS` value rendering.
//!
//! `bpf_struct_ops_map` (`kernel/bpf/bpf_struct_ops.c::bpf_struct_ops_map`)
//! embeds a `struct bpf_struct_ops_value kvalue` directly — the registered
//! kernel struct's bytes (e.g. `sched_ext_ops` for ktstr's scx-ktstr
//! fixture) live inline at `kvalue.data`, prefixed by an
//! 8-byte `bpf_struct_ops_common_value` (refcnt + state) header.
//!
//! `map->btf_value_type_id` (set by libbpf to the user program's
//! interpretation of the registered struct — `tools/lib/bpf/libbpf.c`
//! around `map->btf_value_type_id = type_id` in `bpf_object__init_struct_ops`)
//! describes the `data` payload only; it does NOT cover the common
//! header. To render through the existing BTF path with that type_id,
//! the dump renderer reads from `kvalue + value_data` (the data start)
//! for `value_size - value_data` bytes.
//!
//! Verified against `kernel/bpf/bpf_struct_ops.c::bpf_struct_ops_map_alloc`:
//! `if (attr->value_size != vt->size) return -EINVAL` (line ~1090) —
//! `value_size` is the wrapper size (common + data). `st_map_size =
//! sizeof(*st_map) + (vt->size - sizeof(struct bpf_struct_ops_value))`
//! extends the trailing flex array so `kvalue.data` has exactly the
//! registered struct's worth of room.

use anyhow::{Context, Result};
use btf_rs::Btf;

use super::{
    find_struct, member_byte_offset, member_byte_offset_with_member, resolve_member_struct,
};

/// Byte offsets needed to read `BPF_MAP_TYPE_STRUCT_OPS` value bytes
/// from guest memory.
///
/// Resolution is optional — `resolve_struct_ops_offsets()` returns
/// `Err` when `bpf_struct_ops_map` or `bpf_struct_ops_value` is
/// missing from BTF (kernels built without struct_ops support).
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct StructOpsOffsets {
    /// Offset of the embedded `kvalue` (`struct bpf_struct_ops_value`)
    /// within `struct bpf_struct_ops_map`.
    pub kvalue: usize,
    /// Offset of `data` (the flex array) within
    /// `struct bpf_struct_ops_value`. `data` follows
    /// `bpf_struct_ops_common_value common` and is
    /// `____cacheline_aligned_in_smp`, so this offset equals the
    /// effective common-header size after alignment padding.
    pub value_data: usize,
}

/// Resolve `StructOpsOffsets` from a parsed BTF object.
pub(super) fn resolve_struct_ops_offsets(btf: &Btf) -> Result<StructOpsOffsets> {
    let (st_map, _) = find_struct(btf, "bpf_struct_ops_map")?;
    let (kvalue_off, kvalue_member) = member_byte_offset_with_member(btf, &st_map, "kvalue")?;
    let st_value = resolve_member_struct(btf, &kvalue_member)
        .context("btf: resolve bpf_struct_ops_map.kvalue type")?;
    let data_off = member_byte_offset(btf, &st_value, "data")?;

    Ok(StructOpsOffsets {
        kvalue: kvalue_off,
        value_data: data_off,
    })
}
