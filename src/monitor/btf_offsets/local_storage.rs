//! BTF offsets for `bpf_local_storage_map` walking.
//!
//! Covers `BPF_MAP_TYPE_TASK_STORAGE` and the shape-identical
//! `INODE_STORAGE` / `SK_STORAGE` / `CGRP_STORAGE` variants — all four
//! share the `bpf_local_storage_map` / `bpf_local_storage_map_bucket`
//! / `bpf_local_storage_elem` / `bpf_local_storage_data` /
//! `bpf_local_storage` layout declared in
//! `include/linux/bpf_local_storage.h`.
//!
//! Walk shape (verified against the kernel source under
//! `kernel/bpf/bpf_local_storage.c`):
//!
//! 1. `bpf_local_storage_map.buckets` is an array of
//!    `1 << bucket_log` `bpf_local_storage_map_bucket` structs, each
//!    holding a regular `struct hlist_head list`. NOT `hlist_nulls`
//!    — `bpf_local_storage_map_alloc` initializes via
//!    `INIT_HLIST_HEAD`, so chain termination is `next == NULL` with
//!    no LSB-tagged sentinel.
//! 2. Each chain links via `bpf_local_storage_elem.map_node`. That
//!    field sits at offset 0 of `bpf_local_storage_elem` (the
//!    `hlist_node` is the first member of the struct in
//!    `include/linux/bpf_local_storage.h`), so the chain `node_kva`
//!    IS the elem KVA — no `container_of` subtraction needed.
//! 3. From the elem: value bytes live at
//!    `elem + elem_sdata + sdata_data` (the cacheline-aligned
//!    `sdata.data[]` flex array). `elem.local_storage` (RCU pointer
//!    to `bpf_local_storage`) points at the storage container
//!    holding `owner` (the task_struct / inode / sock / cgroup KVA).

use anyhow::{Context, Result};
use btf_rs::Btf;

use super::{
    find_struct, member_byte_offset, member_byte_offset_with_member, resolve_member_struct,
};

/// Byte offsets within kernel BPF local-storage structures needed for
/// host-side iteration of a `BPF_MAP_TYPE_TASK_STORAGE` map (and the
/// shape-identical `INODE_STORAGE` / `SK_STORAGE` / `CGRP_STORAGE`
/// variants).
///
/// Resolution is optional — [`resolve_task_storage_offsets`] returns
/// `Err` when any of `bpf_local_storage_map`,
/// `bpf_local_storage_map_bucket`, `bpf_local_storage_elem`,
/// `bpf_local_storage_data`, or `bpf_local_storage` is missing from
/// BTF (kernels built without `CONFIG_BPF_SYSCALL` or otherwise
/// lacking the local-storage subsystem). It also returns `Err` when
/// `bpf_local_storage_elem.map_node` is at any non-zero offset — the
/// walker assumes `elem_kva == node_kva` and a future kernel that
/// reorders `bpf_local_storage_elem` would silently misread without
/// the assertion. A fail-fast resolver lets the caller surface
/// "walker disabled on this kernel" instead of corrupted data.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct TaskStorageOffsets {
    /// Offset of `buckets` pointer (`struct bpf_local_storage_map_bucket *`)
    /// within `struct bpf_local_storage_map`.
    pub smap_buckets: usize,
    /// Offset of `bucket_log` (u32) within `struct bpf_local_storage_map`.
    /// The bucket count is `1 << bucket_log`.
    pub smap_bucket_log: usize,
    /// Size of `struct bpf_local_storage_map_bucket` in bytes — the
    /// stride between consecutive buckets in the array at
    /// `smap_buckets`.
    pub bucket_size: usize,
    /// Offset of `list` (`struct hlist_head`) within
    /// `struct bpf_local_storage_map_bucket`.
    pub bucket_list: usize,
    /// Offset of `first` pointer within `struct hlist_head`.
    pub hlist_head_first: usize,
    /// Offset of `next` pointer within `struct hlist_node`. Combined
    /// with `bpf_local_storage_elem.map_node`'s offset (which is 0
    /// — see [`Self`] doc) to address the chain link from the elem
    /// base.
    pub hlist_node_next: usize,
    /// Offset of `local_storage` (`struct bpf_local_storage __rcu *`)
    /// within `struct bpf_local_storage_elem`.
    pub elem_local_storage: usize,
    /// Offset of `sdata` (`struct bpf_local_storage_data`) within
    /// `struct bpf_local_storage_elem`. The kernel header marks
    /// `sdata` with `____cacheline_aligned`, so the offset is
    /// BTF-resolved (not hand-computed) to honor the alignment
    /// padding.
    pub elem_sdata: usize,
    /// Offset of `data` flex array (8-byte-aligned u8[]) within
    /// `struct bpf_local_storage_data`. Value bytes start here.
    pub sdata_data: usize,
    /// Offset of `owner` (void *) within `struct bpf_local_storage`.
    /// For TASK_STORAGE this points to the `task_struct`; for the
    /// other variants it points to the corresponding owner type. The
    /// walker treats it as an opaque KVA.
    pub ls_owner: usize,
}

/// Resolve BTF offsets for `bpf_local_storage_map` walking. Asserts
/// the `bpf_local_storage_elem.map_node` offset-0 invariant; returns
/// `Err` if a future kernel violates it so the walker doesn't read
/// silently-corrupted bytes.
pub(crate) fn resolve_task_storage_offsets(btf: &Btf) -> Result<TaskStorageOffsets> {
    let (smap_struct, _) = find_struct(btf, "bpf_local_storage_map")?;
    let smap_buckets = member_byte_offset(btf, &smap_struct, "buckets")?;
    let smap_bucket_log = member_byte_offset(btf, &smap_struct, "bucket_log")?;

    let (bucket_struct, _) = find_struct(btf, "bpf_local_storage_map_bucket")?;
    let bucket_size = bucket_struct.size();
    let (bucket_list, list_member) = member_byte_offset_with_member(btf, &bucket_struct, "list")?;

    let hlist_head_struct = resolve_member_struct(btf, &list_member)
        .context("btf: resolve type of bpf_local_storage_map_bucket.list")?;
    let hlist_head_first = member_byte_offset(btf, &hlist_head_struct, "first")?;

    let (hlist_node_struct, _) = find_struct(btf, "hlist_node")?;
    let hlist_node_next = member_byte_offset(btf, &hlist_node_struct, "next")?;
    // hlist_node.next is the first member of struct hlist_node
    // (`include/linux/types.h::struct hlist_node { struct
    // hlist_node *next, **pprev; }`); its offset is structurally 0.
    // Combined with the `map_node` offset-0 assertion below, the
    // walker's chain advance via `elem + hlist_node_next` reads
    // offset 0 of the elem — so a future kernel that flips the
    // hlist_node layout would silently misadvance the chain
    // without this assertion. Mirror of the matching invariants
    // on `htab_elem.hash_node` and
    // `bpf_local_storage_elem.map_node`.
    if hlist_node_next != 0 {
        anyhow::bail!(
            "hlist_node.next at offset {} (expected 0): walker advances \
             chain via `elem + hlist_node_next` and the map_node-at-0 \
             invariant assumes the link reads from elem base. A kernel \
             that reorders hlist_node must teach the walker an explicit \
             link offset before this resolver returns Ok.",
            hlist_node_next,
        );
    }

    let (elem_struct, _) = find_struct(btf, "bpf_local_storage_elem")?;
    let map_node_off = member_byte_offset(btf, &elem_struct, "map_node")?;
    if map_node_off != 0 {
        anyhow::bail!(
            "bpf_local_storage_elem.map_node at offset {} (expected 0): \
             walker assumes elem_kva == node_kva. A kernel that reorders \
             bpf_local_storage_elem must teach the walker container_of math \
             before this resolver returns Ok.",
            map_node_off,
        );
    }
    let elem_local_storage = member_byte_offset(btf, &elem_struct, "local_storage")?;
    let (elem_sdata, _) = member_byte_offset_with_member(btf, &elem_struct, "sdata")?;

    let (sdata_struct, _) = find_struct(btf, "bpf_local_storage_data")?;
    let sdata_data = member_byte_offset(btf, &sdata_struct, "data")?;

    let (ls_struct, _) = find_struct(btf, "bpf_local_storage")?;
    let ls_owner = member_byte_offset(btf, &ls_struct, "owner")?;

    Ok(TaskStorageOffsets {
        smap_buckets,
        smap_bucket_log,
        bucket_size,
        bucket_list,
        hlist_head_first,
        hlist_node_next,
        elem_local_storage,
        elem_sdata,
        sdata_data,
        ls_owner,
    })
}
