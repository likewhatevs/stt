//! BTF offsets for `BPF_MAP_TYPE_HASH` / `BPF_MAP_TYPE_PERCPU_HASH` /
//! `BPF_MAP_TYPE_LRU_HASH` / `BPF_MAP_TYPE_LRU_PERCPU_HASH` walking.
//!
//! All four map types share the `bpf_htab` / `bucket` /
//! `hlist_nulls_head` / `hlist_nulls_node` / `htab_elem` layout
//! declared in `kernel/bpf/hashtab.c`. The walker walks
//! `bpf_htab.buckets[0..n_buckets]` (a flex array of `struct bucket`),
//! follows each bucket's `head.first` `hlist_nulls` chain, and reads
//! key/value bytes off each `htab_elem`.
//!
//! Note: `htab_elem` lives at the chain `node` KVA itself — the
//! `hlist_node` is the first member of `htab_elem` in
//! `kernel/bpf/hashtab.c::htab_elem`, so the chain `node_kva` IS the
//! elem KVA — no `container_of` subtraction needed.

use anyhow::{Context, Result, bail};
use btf_rs::{Btf, Type};

use super::{find_struct, member_byte_offset};

/// Byte offsets within kernel BPF hash table structures needed for
/// host-side hash map iteration.
///
/// Resolution is optional — `resolve_htab_offsets()` returns `Err`
/// when the required types are missing from BTF.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct HtabOffsets {
    /// Offset of `buckets` pointer within `struct bpf_htab`.
    pub htab_buckets: usize,
    /// Offset of `n_buckets` (u32) within `struct bpf_htab`.
    pub htab_n_buckets: usize,
    /// Size of `struct bucket` in bytes.
    pub bucket_size: usize,
    /// Offset of `head` (`struct hlist_nulls_head`) within `struct bucket`.
    pub bucket_head: usize,
    /// Offset of `first` pointer within `struct hlist_nulls_head`.
    pub hlist_nulls_head_first: usize,
    /// Offset of `next` pointer within `struct hlist_nulls_node`.
    pub hlist_nulls_node_next: usize,
    /// Size of `struct htab_elem` (base size, before flex key[]).
    pub htab_elem_size_base: usize,
}

/// Find the BPF hashtab `struct bucket` among possibly multiple BTF
/// structs named `bucket`. Returns the struct and its `head` field offset.
/// The BPF bucket has a `head` field (`hlist_nulls_head`); other `bucket`
/// structs (e.g. bcache) do not.
fn find_bucket_struct(btf: &Btf) -> Result<(btf_rs::Struct, usize)> {
    let types = btf
        .resolve_types_by_name("bucket")
        .with_context(|| "btf: type 'bucket' not found")?;

    for t in &types {
        if let Type::Struct(s) = t
            && let Ok(head_off) = member_byte_offset(btf, s, "head")
        {
            return Ok((s.clone(), head_off));
        }
    }
    bail!("btf: no 'bucket' struct with 'head' field found");
}

/// Resolve BTF offsets for BPF hash table structures.
/// Returns Err if any required type/field is missing.
///
/// Asserts `htab_elem.hash_node` lives at offset 0. The walker
/// treats the chain `node_kva` (a `hlist_nulls_node *`) as the
/// elem KVA — `kernel/bpf/hashtab.c::htab_elem` declares
/// `hash_node` as the first member (inside an anonymous union),
/// so the offset is structurally 0. A future kernel that reorders
/// `htab_elem` would silently misread without the assertion (the
/// walker would compute key/value offsets relative to the wrong
/// base). Mirror of the matching invariant in
/// `local_storage::resolve_task_storage_offsets`.
pub(super) fn resolve_htab_offsets(btf: &Btf) -> Result<HtabOffsets> {
    let (bpf_htab, _) = find_struct(btf, "bpf_htab")?;
    let htab_buckets = member_byte_offset(btf, &bpf_htab, "buckets")?;
    let htab_n_buckets = member_byte_offset(btf, &bpf_htab, "n_buckets")?;

    // Multiple structs named `bucket` may exist in BTF (e.g. bcache).
    // Find the one with a `head` field (BPF hashtab's bucket).
    let (bucket_struct, bucket_head) = find_bucket_struct(btf)?;
    let bucket_size = bucket_struct.size();

    let (hlist_nulls_head, _) = find_struct(btf, "hlist_nulls_head")?;
    let hlist_nulls_head_first = member_byte_offset(btf, &hlist_nulls_head, "first")?;

    let (hlist_nulls_node, _) = find_struct(btf, "hlist_nulls_node")?;
    let hlist_nulls_node_next = member_byte_offset(btf, &hlist_nulls_node, "next")?;

    let (htab_elem, _) = find_struct(btf, "htab_elem")?;
    let htab_elem_size_base = htab_elem.size();
    // hash_node is the first member of htab_elem (inside an
    // anonymous union, per kernel source); its offset must be 0
    // for the walker's `elem_kva == node_kva` assumption to hold.
    let hash_node_off = member_byte_offset(btf, &htab_elem, "hash_node")?;
    if hash_node_off != 0 {
        anyhow::bail!(
            "htab_elem.hash_node at offset {} (expected 0): walker assumes \
             elem_kva == hash_node_kva. A kernel that reorders htab_elem \
             must teach the walker container_of math before this resolver \
             returns Ok.",
            hash_node_off,
        );
    }

    Ok(HtabOffsets {
        htab_buckets,
        htab_n_buckets,
        bucket_size,
        bucket_head,
        hlist_nulls_head_first,
        hlist_nulls_node_next,
        htab_elem_size_base,
    })
}
