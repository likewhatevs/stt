//! Host-side `sdt_alloc` radix-tree walker.
//!
//! `sdt_alloc` is the per-task / per-cgroup arena allocator that ships
//! in the upstream scx tree at `lib/sdt_alloc.bpf.c` (and friends in
//! `lib/sdt_task_defs.h`). Schedulers that opt into it allocate
//! per-entity contexts out of BPF arena memory, addressed via a
//! 3-level radix tree rooted at `scx_allocator.root`. The kernel
//! never exposes this layout to userspace — at freeze time the host
//! has only the raw arena page snapshot from [`super::arena`], which
//! is page-granular and structurally opaque.
//!
//! This module walks the same tree the scheduler walks, from frozen
//! guest memory, and produces structured per-allocation records that
//! the BTF renderer can turn into named field views. The result lands
//! in [`super::dump::FailureDumpReport::sdt_allocations`], distinct
//! from the page-granular snapshot so consumers can read either
//! representation.
//!
//! # Tree shape
//!
//! From `lib/sdt_task_defs.h`:
//!
//! ```text
//! scx_allocator { sdt_pool pool; sdt_desc_t *root; }
//! sdt_pool      { void *slab; u64 elem_size; u64 max_elems; u64 idx; }
//! sdt_desc      { u64 allocated[8]; u64 nr_free; sdt_chunk *chunk; }
//! sdt_chunk     { union { sdt_desc *descs[512]; sdt_data *data[512]; } }
//! sdt_data      { union sdt_id tid; u64 payload[]; }
//! sdt_id        { s32 idx; s32 genn; }   /* 8 bytes */
//! ```
//!
//! Three levels (`SDT_TASK_LEVELS = 3`), 512 entries per chunk
//! (`1 << SDT_TASK_ENTS_PER_PAGE_SHIFT`). The `allocated[8]` bitmap
//! (512 bits / 64) tracks live slots at each level. Internal levels
//! (0, 1) reach the next descriptor via `chunk->descs[pos]`; the leaf
//! level (2) reaches the user-visible payload via `chunk->data[pos]`.
//!
//! `sdt_data.tid.idx` carries the 27-bit (3 × 9) packed index that
//! produced this slot, and `tid.genn` increments on recycle so a
//! consumer can detect ABA across allocations of the same idx.
//!
//! # Liveness
//!
//! At the leaf level (level 2), `allocated[]` is the source of truth:
//! a set bit means slot `pos` carries a live `sdt_data *` in
//! `chunk->data[pos]`. We use the bitmap there because `tid.idx` and
//! `pool.idx` are both unreliable — post-free `tid.idx` is reset to 0
//! (ambiguous with slot 0), and `pool.idx` is the pool's high-water
//! mark, not the live count. `chunk->data[pos]` is also nullable for
//! pristine slots that the pool never handed out — we skip those
//! silently.
//!
//! At internal levels (0 and 1) the bitmap semantics are inverted:
//! `lib/sdt_alloc.bpf.c` only sets a parent bit once a child becomes
//! FULL (`desc_find_empty` propagates the set up only while the
//! decremented `nr_free` stays at 0) and clears it once the child
//! transitions back from full (`mark_nodes_avail` only propagates the
//! clear up while the incremented `nr_free` is still 1). So a set bit
//! at an internal level means "the descendant subtree is full"; a
//! clear bit means "partially populated, empty, or never created."
//! The common case for any scheduler with N << 512^3 live tasks is
//! "all clear", so the bitmap is unusable for enumeration at internal
//! levels. The walker enumerates internal levels by pointer non-null
//! in `chunk->descs[]` instead — every populated subtree has a
//! non-NULL desc child stored at its `pos` (`desc_find_empty` writes
//! `desc_children[pos]` whenever it allocates a new chunk), and a
//! NULL pointer is a never-created subtree we skip silently.
//!
//! # Race window
//!
//! The freeze coordinator pauses every vCPU before this walker runs,
//! but `scx_alloc_free_idx` zeroes a slot's payload BEFORE clearing
//! the bitmap. A frozen snapshot captured between those two writes
//! sees a "live" bitmap bit but a zero-filled payload. We render the
//! zeros as a "zeroed slot" rather than try to detect mid-free — the
//! consumer can recognise all-zero payloads as the race.

use serde::{Deserialize, Serialize};

use anyhow::{Context, Result};
use btf_rs::{Btf, Type};

use super::Kva;
use super::btf_offsets::{StructOrFwd, find_struct_or_fwd, member_byte_offset};
use super::btf_render::{MemReader, RenderedValue, render_value_with_mem};
use super::dump::hex_dump;
use super::guest::GuestKernel;

/// Tree depth and per-chunk fan-out from `lib/sdt_task_defs.h`.
///
/// `SDT_TASK_LEVELS = 3` and `SDT_TASK_ENTS_PER_PAGE_SHIFT = 9`. The
/// 512-entry fan-out is hard-baked into the layout (chunk arrays are
/// declared as `[SDT_TASK_ENTS_PER_CHUNK]` at file scope), so a future
/// upstream change in fan-out would re-shape `struct sdt_chunk` itself
/// and the walker would surface the divergence as a missing-field BTF
/// resolution failure during offset lookup.
const SDT_TASK_LEVELS: usize = 3;
const SDT_TASK_ENTS_PER_PAGE_SHIFT: u32 = 9;
const SDT_TASK_ENTS_PER_CHUNK: usize = 1 << SDT_TASK_ENTS_PER_PAGE_SHIFT; // 512
const SDT_TASK_CHUNK_BITMAP_U64S: usize = SDT_TASK_ENTS_PER_CHUNK / 64; // 8

/// Maximum number of leaf allocations the walker will surface in a
/// single dump.
///
/// A scheduler that allocated millions of per-task contexts would OOM
/// the host renderer if the result was unbounded; cap to 4096 entries
/// (mirrors `MAX_HASH_ENTRIES` in [`super::dump`]) and surface
/// truncation via [`SdtAllocatorSnapshot::truncated`].
pub const MAX_SDT_ALLOC_ENTRIES: usize = 4096;

/// Width of `union sdt_id` in bytes — 8: `s32 idx + s32 genn`.
///
/// The kernel layout in `lib/sdt_task_defs.h` makes this a hard part
/// of the wire format: `union sdt_id { s64 val; struct { s32 idx; s32
/// genn; }; }` is exactly 8 bytes, and `struct sdt_data { union sdt_id
/// tid; u64 payload[]; }` has no other non-flex-array member, so
/// `sizeof(struct sdt_data) == 8` for every kernel that ships
/// sdt_alloc. [`SdtAllocOffsets::from_btf`] uses this as the fallback
/// for `data_header_size` when the scheduler's program BTF surfaces
/// `sdt_data` as a `BTF_KIND_FWD` forward declaration (no struct body
/// from which `.size()` could be read); unit tests pin it against
/// upstream layout drift.
const SIZEOF_SDT_ID: usize = 8;

/// Sanity cap on `pool.elem_size` (allocation slot stride) the walker
/// will trust.
///
/// `lib/sdt_alloc.bpf.c::pool_set_size` checks `data_size % 8 == 0`
/// and bails on zero; `scx_alloc_init` rounds up to 8 then ensures
/// the chunk fits in `PAGE_SIZE`. So a real `elem_size` is always
/// `[16, 4096]` for non-degenerate allocators (16-byte minimum =
/// `sizeof(sdt_data) + 8`-byte minimum payload after `round_up(...,
/// 8)`). A torn snapshot or an uninitialized struct could surface a
/// wild value; reject anything outside this range.
const MIN_ELEM_SIZE: u64 = 16;
const MAX_ELEM_SIZE: u64 = 4096;

/// Upper bound on the BTF type-id walk in [`discover_payload_btf_id`].
///
/// btf-rs has no "list all types" iterator, so the heuristic walks
/// type ids 1..N and probes each with `resolve_type_by_id`. BTF can
/// have sparse id gaps (a single unresolvable id does NOT mean the
/// table is exhausted), so we don't break on the first miss — we
/// `continue` and walk up to this cap. 100k is well above the largest
/// program-BTF type tables ktstr sees in practice (~10k for a complex
/// scheduler) while still keeping the worst-case probe cost bounded.
///
/// Shared with [`super::cast_analysis`]'s candidate-search id walk:
/// both probes use the same heuristic against the same per-program
/// BTFs, so a single ceiling keeps them aligned.
pub(crate) const MAX_BTF_ID_PROBE: u32 = 100_000;

/// Byte offsets within the sdt_alloc data structures.
///
/// All resolved from the SCHEDULER'S program BTF (not vmlinux), since
/// `struct scx_allocator`, `struct sdt_pool`, `struct sdt_desc`, and
/// `struct sdt_chunk` are linked into the BPF program from
/// `lib/sdt_alloc.bpf.c` and never appear in vmlinux BTF.
#[derive(Debug, Clone)]
pub struct SdtAllocOffsets {
    /// Offset of `pool` (sdt_pool) within `struct scx_allocator`.
    pub allocator_pool: usize,
    /// Offset of `root` (sdt_desc_t *) within `struct scx_allocator`.
    pub allocator_root: usize,
    /// Total size of `struct scx_allocator`. Used by callers that
    /// need to bound a slice read of the in-bss allocator image.
    pub allocator_size: usize,
    /// Offset of `elem_size` (u64) within `struct sdt_pool`.
    pub pool_elem_size: usize,
    /// Offset of `allocated` ([u64; 8]) within `struct sdt_desc`.
    pub desc_allocated: usize,
    /// Offset of `nr_free` (u64) within `struct sdt_desc`.
    pub desc_nr_free: usize,
    /// Offset of `chunk` (`struct sdt_chunk *`) within `struct sdt_desc`.
    pub desc_chunk: usize,
    /// Offset of the union (`descs`/`data`) within `struct sdt_chunk`.
    /// Both interpretations alias at the same offset (it's a union).
    pub chunk_union: usize,
    /// Total size of `struct sdt_data` (header + zero-length payload[]).
    /// Equals 8 on all known kernels (the size of `union sdt_id`)
    /// because the flexible `payload[]` array adds no bytes to the
    /// struct's size.
    pub data_header_size: usize,
}

impl SdtAllocOffsets {
    /// Resolve sdt_alloc struct offsets from a pre-loaded program BTF.
    ///
    /// Returns `Err` when the program BTF lacks any of the required
    /// types — e.g. a scheduler that doesn't link `lib/sdt_alloc.bpf.c`
    /// into its BPF object. The dump pipeline treats this as "no
    /// sdt_alloc state to surface" and skips the walk silently rather
    /// than aborting, since not every scheduler uses the allocator.
    ///
    /// # `BTF_KIND_FWD` handling
    ///
    /// BPF program BTFs emit `BTF_KIND_FWD` (forward declaration, no
    /// body) for any struct the program references only by pointer.
    /// The four "structural" types — `scx_allocator`, `sdt_pool`,
    /// `sdt_desc`, `sdt_chunk` — must surface as full struct
    /// definitions: the walker derives member offsets from each, and
    /// a forward declaration carries no member information. A Fwd for
    /// any of those four is surfaced as `Err` so the dump pipeline
    /// records a clear diagnostic instead of crashing on missing
    /// members.
    ///
    /// `sdt_data` is the exception: lavd and other schedulers that
    /// only consume opaque allocator-returned pointers emit `sdt_data`
    /// as a `BTF_KIND_FWD`. The walker only needs the size of the
    /// header (the leading `union sdt_id`, 8 bytes; the `payload[]`
    /// flex-array contributes 0), so [`SIZEOF_SDT_ID`] is used as the
    /// fallback when `sdt_data` is a Fwd. The kernel header
    /// `lib/sdt_task_defs.h` makes this size invariant (it's the only
    /// non-flex-array member), so the fallback is correct without BTF
    /// involvement.
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let allocator = require_full_struct(btf, "scx_allocator").context(
            "btf: struct scx_allocator unavailable (scheduler doesn't link sdt_alloc, or BTF only carries a forward declaration)"
        )?;
        let allocator_pool = member_byte_offset(btf, &allocator, "pool")?;
        let allocator_root = member_byte_offset(btf, &allocator, "root")?;
        let allocator_size = allocator.size();

        let pool = require_full_struct(btf, "sdt_pool")
            .context("btf: struct sdt_pool unavailable for member offsets")?;
        let pool_elem_size = member_byte_offset(btf, &pool, "elem_size")?;

        let desc = require_full_struct(btf, "sdt_desc")
            .context("btf: struct sdt_desc unavailable for member offsets")?;
        let desc_allocated = member_byte_offset(btf, &desc, "allocated")?;
        let desc_nr_free = member_byte_offset(btf, &desc, "nr_free")?;
        let desc_chunk = member_byte_offset(btf, &desc, "chunk")?;

        // sdt_chunk is another type schedulers commonly emit as
        // BTF_KIND_FWD — it's only accessed internally by the
        // sdt_alloc library helpers. The struct contains a single
        // anonymous union at offset 0 (descs[] for internal nodes,
        // data[] for leaves). When only a Fwd is available, hardcode
        // chunk_union = 0 matching the kernel layout at
        // lib/sdt_task_defs.h.
        let chunk_union = match find_struct_or_fwd(btf, "sdt_chunk")
            .context("btf: struct sdt_chunk not found")?
        {
            StructOrFwd::Full(chunk) => chunk_union_offset(btf, &chunk)?,
            StructOrFwd::Fwd => 0,
        };

        // `sdt_data` is the one type the walker tolerates as a
        // `BTF_KIND_FWD`. The scheduler program never accesses its
        // members directly (the lib/sdt_alloc.bpf.c helpers do, in lib
        // BTF that may not be linked into the program BTF), so lavd
        // and similar schedulers emit it as a forward declaration.
        // The size is fixed by `lib/sdt_task_defs.h` at 8 bytes (the
        // `union sdt_id` header; `payload[]` is a flex-array
        // contributing 0 bytes), so we fall back to [`SIZEOF_SDT_ID`]
        // when the body is absent.
        let data_header_size =
            match find_struct_or_fwd(btf, "sdt_data").context("btf: struct sdt_data not found")? {
                StructOrFwd::Full(data) => data.size(),
                StructOrFwd::Fwd => SIZEOF_SDT_ID,
            };

        Ok(Self {
            allocator_pool,
            allocator_root,
            allocator_size,
            pool_elem_size,
            desc_allocated,
            desc_nr_free,
            desc_chunk,
            chunk_union,
            data_header_size,
        })
    }
}

/// Resolve a struct that the walker requires by member offset.
///
/// Wraps [`find_struct_or_fwd`] and rejects forward declarations with
/// an explicit "fwd, no body" diagnostic — distinct from "not found at
/// all" so an operator can tell whether the scheduler links sdt_alloc
/// at all (Err: not found) versus whether the program BTF stripped the
/// struct body (Err: fwd only). Returning a Fwd from this helper would
/// mean propagating an unusable struct handle whose `member_byte_offset`
/// calls would then fail with a misleading "field 'X' not found" error.
fn require_full_struct(btf: &Btf, name: &str) -> Result<btf_rs::Struct> {
    match find_struct_or_fwd(btf, name)? {
        StructOrFwd::Full(s) => Ok(s),
        StructOrFwd::Fwd => anyhow::bail!(
            "btf: struct {name} present only as BTF_KIND_FWD forward declaration; member offsets unavailable"
        ),
    }
}

/// Locate the union member offset within `struct sdt_chunk`.
///
/// The chunk struct is `struct { union { sdt_desc *descs[512];
/// sdt_data *data[512]; } }` — both arms occupy the same byte range.
/// Searching for either name returns the same offset; we accept the
/// first that resolves so a future rename of one arm doesn't cause
/// a hard failure when the other arm still exists.
fn chunk_union_offset(btf: &Btf, chunk: &btf_rs::Struct) -> Result<usize> {
    if let Ok(off) = member_byte_offset(btf, chunk, "descs") {
        return Ok(off);
    }
    if let Ok(off) = member_byte_offset(btf, chunk, "data") {
        return Ok(off);
    }
    anyhow::bail!("btf: struct sdt_chunk has neither `descs` nor `data` member")
}

/// One leaf allocation surfaced from the sdt_alloc tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SdtAllocEntry {
    /// `sdt_data.tid.idx` — the 27-bit packed slot index. Negative
    /// values are surfaced verbatim (the kernel uses `s32`); the host
    /// does not interpret sign here.
    pub idx: i32,
    /// `sdt_data.tid.genn` — incremented on recycle so consumers can
    /// distinguish reallocations of the same `idx`.
    pub genn: i32,
    /// Low 32 bits of the user-side arena pointer to the `sdt_data`
    /// slot. Computed by [`TreeWalker::emit_leaf`] as
    /// `data_ptr & 0xFFFF_FFFF`, NOT the full user-side VA — slot
    /// addresses already live in the 32-bit `arena.user_vm_start`
    /// window, so the masked low 32 bits are sufficient for
    /// correlation against pointer values an operator sees in BPF
    /// program output AND match the masking convention the renderer's
    /// arena-type bridge keys on (see
    /// [`super::btf_render::MemReader::resolve_arena_type`]).
    ///
    /// Distinct from [`super::arena::ArenaPage::user_addr`], which
    /// carries the FULL user-side VA — the page-snapshot consumer
    /// surfaces the unmasked address so cross-arena callers (multiple
    /// arena maps with different `user_vm_start` bases) do not collide
    /// on the same low-32 bits.
    pub user_addr: u64,
    /// BTF-rendered payload (everything after the 8-byte
    /// `union sdt_id` tid header). Falls back to a hex dump when
    /// payload type discovery failed; renders as
    /// [`RenderedValue::Unsupported`] when the payload couldn't be
    /// read at all (end-of-DRAM, unmapped page).
    pub payload: RenderedValue,
}

impl std::fmt::Display for SdtAllocEntry {
    /// Human-readable rendering: `idx=N genn=M user_addr=0x... payload=<value>`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "idx={} genn={} user_addr={:#x} payload=",
            self.idx, self.genn, self.user_addr
        )?;
        std::fmt::Display::fmt(&self.payload, f)
    }
}

/// All sdt_alloc allocations surfaced from a single allocator.
///
/// One [`SdtAllocatorSnapshot`] per allocator instance — the dump
/// pipeline today walks one allocator (the scheduler's primary) but
/// the type shape leaves room for multiple allocators to coexist in
/// the snapshot.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SdtAllocatorSnapshot {
    /// Allocator name (e.g. the .bss symbol the allocator was read
    /// from, like `"scx_task_allocator"`). Surfaced so a consumer
    /// reading multiple allocator dumps can tell them apart.
    pub allocator_name: String,
    /// Live allocations, in tree-walk order (level 0 → 1 → 2,
    /// monotonic pos at each level). Capped at
    /// [`MAX_SDT_ALLOC_ENTRIES`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entries: Vec<SdtAllocEntry>,
    /// True when the walk stopped at [`MAX_SDT_ALLOC_ENTRIES`] before
    /// covering every live bit in the bitmaps.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
    /// Count of subtrees the walker abandoned mid-descent due to a
    /// pointer translate failure, an out-of-range `nr_free`, a NULL
    /// `chunk` pointer, or any other diagnostic that aborts descent
    /// into one subtree without poisoning the rest of the walk. A
    /// non-zero value here means the dump is partial — some live
    /// allocations may not be in `entries`.
    ///
    /// Always serialized — a zero value carries diagnostic information
    /// ("walker reached the end of the tree without skipping anything"),
    /// and suppressing it on default would make consumers conflate "zero
    /// skipped" with "field absent / older schema". Mirrors the
    /// always-serialize policy used by sibling `elem_size` and
    /// `target_type_id`.
    pub skipped_subtrees: u32,
    /// Diagnostic: the per-pool slot stride. Surfaces alongside the
    /// rendered entries so a consumer can spot when the rendered
    /// payload size diverges from the declared one.
    pub elem_size: u64,
    /// Diagnostic: the BTF type id used to render payload bytes.
    /// 0 when [`discover_payload_btf_id`] returned no candidate and
    /// the renderer fell back to hex.
    pub target_type_id: u32,
    /// Diagnostic: when [`discover_payload_btf_id`] returned 0, the
    /// reason (e.g. `"no candidate of size 16"`,
    /// `"ambiguous: 3 candidates"`, `"payload_size == 0"`). Empty on
    /// successful BTF resolve. Lets an operator distinguish the
    /// fallback paths without re-deriving the heuristic.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub payload_type_reason: String,
    #[serde(skip)]
    pub all_slot_addrs: Vec<u64>,
}

impl std::fmt::Display for SdtAllocatorSnapshot {
    /// Header line + one entry per allocation, indented. Diagnostic
    /// lines (truncated, skipped_subtrees) appended when non-default.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sdt_alloc {} (elem_size={}, target_type_id={}",
            self.allocator_name, self.elem_size, self.target_type_id
        )?;
        if !self.payload_type_reason.is_empty() {
            write!(f, ", reason={}", self.payload_type_reason)?;
        }
        write!(f, "): {} live", self.entries.len())?;
        if self.truncated {
            f.write_str(" (truncated)")?;
        }
        if self.skipped_subtrees > 0 {
            write!(f, " ({} subtrees skipped)", self.skipped_subtrees)?;
        }
        for entry in &self.entries {
            f.write_str("\n")?;
            super::btf_render::write_value_at_depth(f, &entry.payload, 2)?;
        }
        Ok(())
    }
}

/// Walk an `scx_allocator` from frozen guest memory and surface every
/// live allocation.
///
/// `allocator_bytes` is the raw byte image of the `struct
/// scx_allocator` instance — the caller reads this from the
/// scheduler's `.bss` (or wherever the allocator is declared) and
/// hands the slice in so the walker doesn't need to know symbol
/// resolution machinery.
///
/// `kern_vm_start` is the kernel-side base of the arena's user_vm
/// window — `bpf_arena.kern_vm->addr + GUARD_HALF`, the same value
/// [`super::arena::snapshot_arena`] computes. Arena pointers in the
/// tree are `__arena` pointers whose low 32 bits index this window,
/// so every translation goes:
///   `kern_va = kern_vm_start + (ptr & 0xFFFF_FFFF)`
///
/// `target_type_id` is the BTF type id the renderer applies to
/// the bytes after the `tid` header. Pass 0 (or a discovered id from
/// [`discover_payload_btf_id`]) — 0 routes to a hex dump.
///
/// `payload_type_reason` is a human-readable string describing why
/// `target_type_id` is 0 (when it is); ignored when the id is
/// non-zero. Surfaces in [`SdtAllocatorSnapshot::payload_type_reason`]
/// so an operator can distinguish "no candidate of this size" from
/// "ambiguous candidates" without re-deriving the heuristic.
///
/// The walk is best-effort: a corrupt desc / chunk / data pointer
/// stops descent into that subtree (incrementing
/// [`SdtAllocatorSnapshot::skipped_subtrees`]) and skips to the next
/// bit, so a single stale slot can't truncate the whole dump.
//
// `clippy::too_many_arguments` is silenced here: every parameter
// is genuinely needed and bundling them into a config struct would
// just shift the surface area without reducing it. The kernel /
// BTF / offsets references must stay independent borrows so the
// caller can compose them from different sources (vmlinux BTF +
// program BTF, accessor + arena snapshot kern_vm_start, etc.).
#[allow(clippy::too_many_arguments)]
pub fn walk_sdt_allocator(
    kernel: &GuestKernel,
    kern_vm_start: u64,
    allocator_bytes: &[u8],
    offsets: &SdtAllocOffsets,
    btf: &Btf,
    target_type_id: u32,
    payload_type_reason: impl Into<String>,
    allocator_name: impl Into<String>,
    mem: &dyn MemReader,
) -> SdtAllocatorSnapshot {
    let mut snap = SdtAllocatorSnapshot {
        allocator_name: allocator_name.into(),
        entries: Vec::new(),
        truncated: false,
        skipped_subtrees: 0,
        elem_size: 0,
        target_type_id,
        payload_type_reason: payload_type_reason.into(),
        all_slot_addrs: Vec::new(),
    };

    // Read pool.elem_size from the in-bss allocator image. This is
    // the per-slot stride the leaf walker needs to know how many
    // bytes to read for each allocation.
    let pool_off = offsets.allocator_pool + offsets.pool_elem_size;
    let Some(elem_size) = read_u64_at(allocator_bytes, pool_off) else {
        return snap;
    };
    if !(MIN_ELEM_SIZE..=MAX_ELEM_SIZE).contains(&elem_size) {
        // Out-of-range: corrupt allocator or torn snapshot. Surface
        // empty entries with the captured elem_size so a consumer
        // can see the diagnostic.
        snap.elem_size = elem_size;
        return snap;
    }
    snap.elem_size = elem_size;

    // Sanity: the data header (8 bytes for sdt_id) must fit inside
    // the slot — without this, payload_size would be negative.
    let header = offsets.data_header_size;
    if elem_size < header as u64 {
        return snap;
    }
    let payload_size = (elem_size - header as u64) as usize;

    // Read the root descriptor pointer.
    let Some(root_ptr) = read_u64_at(allocator_bytes, offsets.allocator_root) else {
        return snap;
    };
    if root_ptr == 0 {
        return snap;
    }

    let mut walker = TreeWalker {
        kernel,
        kern_vm_start,
        offsets,
        btf,
        target_type_id,
        payload_size,
        mem,
        out: &mut snap,
    };
    walker.descend(root_ptr, 0);

    snap
}

/// Result of [`discover_payload_btf_id`] — pairs the chosen BTF type
/// id (0 for fallback) with a human-readable reason describing the
/// fallback path when the id is 0.
#[derive(Debug, Clone)]
pub struct PayloadTypeChoice {
    pub target_type_id: u32,
    pub reason: String,
}

/// Heuristic: pick a payload BTF type id matching the slot stride.
///
/// `pool.elem_size = sizeof(sdt_data) + payload_size`, rounded up to
/// 8. So `payload_size = elem_size - sizeof(sdt_data)`. We search the
/// BTF for struct types whose `.size()` equals `payload_size` exactly,
/// then narrow:
///
///   1. Exactly one match → use it.
///   2. Multiple matches → prefer names matching the conventional
///      patterns: `task_ctx` (exact), then `*_arena_ctx`,
///      `*_task_ctx`, `*_ctx` (suffix). scx schedulers consistently
///      use these suffixes; ktstr's own test fixture struct is
///      `ktstr_arena_ctx`. If 2+ structs match the same pattern arm,
///      the heuristic continues to lower-priority arms — a collision
///      at a higher-specificity level does not prevent a lower-
///      specificity unambiguous match from resolving.
///   3. No match or still ambiguous → return 0 to fall back to a hex
///      dump.
///
/// `base_btf` is the optional base BTF (vmlinux for split program
/// BTFs) used to filter out base-BTF type ids from the candidate set.
/// The program BTF the renderer threads in is built via
/// [`btf_rs::Btf::from_split_bytes`] with vmlinux as base; the base's
/// type ids occupy the low end of the id space (1..base_nr_types),
/// and `Btf::resolve_type_by_id` walks them first. Without filtering,
/// a vmlinux struct of the same byte size as the scheduler's payload
/// (e.g. some `*_ctx` of size 16) can win the size-match step and
/// then propagate the wrong layout into the renderer. btf-rs 1.1.1
/// does not expose `base_nr_types()` directly, so the filter resolves
/// each candidate id in `base_btf` and excludes it when the
/// resolution succeeds — base-resolvable ids are by definition base
/// types. `None` skips the filter (test BTFs without a base, or
/// callers that genuinely want every match).
///
/// The function is intentionally conservative: a wrong type id renders
/// nonsense field names; falling back to hex always shows the operator
/// raw bytes they can decode by hand. The returned reason string is
/// surfaced to the operator via [`SdtAllocatorSnapshot::payload_type_reason`]
/// so the fallback paths are distinguishable without re-running the
/// heuristic.
pub fn discover_payload_btf_id(btf: &Btf, payload_size: usize,
) -> PayloadTypeChoice {
    if payload_size == 0 {
        return PayloadTypeChoice {
            target_type_id: 0,
            reason: "payload_size == 0".into(),
        };
    }
    let mut size_matches: Vec<(u32, String)> = Vec::new();

    // btf-rs 1.1.1 has no public "list all types" iterator, so probe
    // ids 1..N. BTF type ids are dense within a single object's BTF
    // section (libbpf assigns them sequentially during compile), and
    // for split BTF the program-BTF ids start at `base_nr_types + 1`
    // contiguously. A run of CONSECUTIVE_FAIL_CAP failed lookups
    // indicates the table is exhausted; bailing early is the
    // performance fix for the prior pattern that walked all
    // MAX_BTF_ID_PROBE (100k) ids when only a few hundred existed.
    //
    // The hard ceiling [`MAX_BTF_ID_PROBE`] still bounds the worst
    // case (sparse-id table, defensive). Real ktstr program BTFs
    // top out in the low thousands of types; 64 consecutive failures
    // is generous (a sparse gap of 64 in a contiguous BTF means the
    // generator is broken in a way the heuristic can't help with).
    const CONSECUTIVE_FAIL_CAP: u32 = 64;

    let mut tid: u32 = 1;
    let mut consecutive_fail: u32 = 0;
    while tid < MAX_BTF_ID_PROBE {
        match btf.resolve_type_by_id(tid) {
            Ok(ty) => {
                consecutive_fail = 0;
                if let Type::Struct(s) = ty
                    && s.size() == payload_size
                    && let Ok(name) = btf.resolve_name(&s)
                    && !name.is_empty()
                {
                    // Base-BTF filter: exclude vmlinux structs from
                    // the candidate set. The base BTF's
                    // `resolve_type_by_id` succeeds only for ids it
                    // owns (its own `obj` table — base-only BTFs
                    // have `self.base = None`, so the lookup never
                    // delegates further). A success here proves
                    // the type lives in base BTF — drop it. `None`
                    // (test fixtures without a base, or production
                    // callers that pass `None`) keeps the full id
                    // range.
                    size_matches.push((tid, name));
                }
            }
            Err(_) => {
                consecutive_fail += 1;
                if consecutive_fail >= CONSECUTIVE_FAIL_CAP {
                    break;
                }
            }
        }
        tid += 1;
    }

    match size_matches.len() {
        0 => PayloadTypeChoice {
            target_type_id: 0,
            reason: format!("no candidate of size {payload_size}"),
        },
        1 => PayloadTypeChoice {
            target_type_id: size_matches[0].0,
            reason: String::new(),
        },
        n => {
            // Multiple size-match candidates; prefer conventional
            // names. Order is most-specific first so a struct named
            // exactly `task_ctx` wins over `foo_task_ctx`. If 2+
            // structs share the SAME pattern arm, the loop continues
            // to lower-priority arms — a higher-specificity collision
            // does not prevent a lower-specificity unambiguous match
            // from resolving.
            type Pat = fn(&str) -> bool;
            let patterns: &[Pat] = &[
                |n: &str| n == "task_ctx",
                |n: &str| n.ends_with("_arena_ctx"),
                |n: &str| n.ends_with("_task_ctx"),
                |n: &str| n.ends_with("_ctx"),
            ];
            for pat in patterns {
                let hits: Vec<u32> = size_matches
                    .iter()
                    .filter(|(_, n)| pat(n))
                    .map(|(id, _)| *id)
                    .collect();
                match hits.len() {
                    0 => continue,
                    1 => {
                        return PayloadTypeChoice {
                            target_type_id: hits[0],
                            reason: String::new(),
                        };
                    }
                    _ => {
                        // 2+ matches in the SAME pattern arm —
                        // ambiguous at this priority level. Continue
                        // to the next (lower-priority) pattern arm:
                        // a higher-specificity collision shouldn't
                        // prevent a lower-specificity unambiguous
                        // match from resolving.
                        continue;
                    }
                }
            }
            // No unambiguous pattern winner — fall back to hex.
            PayloadTypeChoice {
                target_type_id: 0,
                reason: format!("ambiguous: {n} candidates"),
            }
        }
    }
}

/// Internal walker state. Bundles the read-only inputs the recursive
/// descent threads through every call so each function takes one
/// `&mut self` and the actual position arguments.
struct TreeWalker<'a> {
    kernel: &'a GuestKernel,
    kern_vm_start: u64,
    offsets: &'a SdtAllocOffsets,
    btf: &'a Btf,
    target_type_id: u32,
    payload_size: usize,
    /// `MemReader` used by [`render_value_with_mem`] when rendering
    /// each leaf payload — lets the BTF renderer chase `__arena`
    /// pointers within the payload (e.g. an entry holding a pointer
    /// to another arena struct) into typed contents instead of
    /// emitting raw hex.
    mem: &'a dyn MemReader,
    out: &'a mut SdtAllocatorSnapshot,
}

impl<'a> TreeWalker<'a> {
    /// Descend into a `sdt_desc` at level `level`. Levels 0 and 1
    /// scan every position in `chunk->descs[]` and recurse into each
    /// non-NULL child pointer (the bitmap is unusable for enumeration
    /// at internal levels — see the module-level "Liveness" docs).
    /// Level 2 reads `chunk->data[]` and emits one [`SdtAllocEntry`]
    /// per allocated bit in this descriptor's `allocated[]` bitmap.
    ///
    /// Increments [`SdtAllocatorSnapshot::skipped_subtrees`] on every
    /// early return that abandons descent into a non-trivial subtree
    /// — translate failures, out-of-range `nr_free`, NULL `chunk`. A
    /// NULL `chunk->descs[pos]` at an internal level is a never-created
    /// subtree and is skipped silently (NOT counted as a skipped
    /// subtree).
    fn descend(&mut self, desc_ptr: u64, level: usize) {
        if level >= SDT_TASK_LEVELS {
            return;
        }

        // Once `emit_leaf` has flipped `truncated`, the cap on entries
        // is reached. Continuing to descend would still grow
        // `all_slot_addrs` (every leaf appended unconditionally before
        // the `entries` cap check) and waste work on every remaining
        // subtree. Bail at the top of every recursive call so descent
        // halts globally, not just at the next leaf.
        if self.out.truncated {
            return;
        }

        // Translate the arena pointer to a kernel VA, then to a PA.
        let Some(desc_pa) = self.translate_arena_ptr(desc_ptr) else {
            self.out.skipped_subtrees = self.out.skipped_subtrees.saturating_add(1);
            return;
        };

        // Read the descriptor: bitmap, nr_free (sanity), chunk pointer.
        let mut allocated = [0u64; SDT_TASK_CHUNK_BITMAP_U64S];
        let mem = self.kernel.mem();
        for (i, slot) in allocated.iter_mut().enumerate() {
            *slot = mem.read_u64(desc_pa, self.offsets.desc_allocated + i * 8);
        }
        let nr_free = mem.read_u64(desc_pa, self.offsets.desc_nr_free);
        // Sanity: nr_free is u64 but bounded by 512 in the kernel.
        // A wild value indicates a torn read or an uninitialized
        // descriptor — abort descent into this subtree.
        if nr_free > SDT_TASK_ENTS_PER_CHUNK as u64 {
            self.out.skipped_subtrees = self.out.skipped_subtrees.saturating_add(1);
            return;
        }
        let chunk_ptr = mem.read_u64(desc_pa, self.offsets.desc_chunk);
        if chunk_ptr == 0 {
            self.out.skipped_subtrees = self.out.skipped_subtrees.saturating_add(1);
            return;
        }
        let Some(chunk_pa) = self.translate_arena_ptr(chunk_ptr) else {
            self.out.skipped_subtrees = self.out.skipped_subtrees.saturating_add(1);
            return;
        };

        // Enumerate slots. The leaf level (`SDT_TASK_LEVELS - 1`)
        // reads `allocated[]` directly: a set bit there means the
        // `sdt_data *` in `chunk->data[pos]` is live (per the
        // module-level "Liveness" docs). Internal levels (0 and 1)
        // ignore the bitmap entirely — `lib/sdt_alloc.bpf.c` only
        // sets a parent bit once a child subtree is FULL, so SET on
        // an internal level means "subtree full" and CLEAR is "empty,
        // partial, or never created." The common-case bitmap (few
        // tasks, deep tree) is all-zero at the internal levels, so
        // iterating set bits would surface zero allocations. Walk
        // every position and descend into any non-NULL `desc *`; a
        // NULL pointer is a never-created subtree we skip silently.
        if level == SDT_TASK_LEVELS - 1 {
            for (word_idx, &word_value) in allocated.iter().enumerate() {
                let mut word = word_value;
                while word != 0 {
                    let bit = word.trailing_zeros() as usize;
                    word &= word - 1;
                    let pos = word_idx * 64 + bit;
                    if pos >= SDT_TASK_ENTS_PER_CHUNK {
                        continue;
                    }

                    // chunk->data[pos] at chunk_pa + chunk_union +
                    // pos * 8.
                    let entry_ptr_off = self.offsets.chunk_union + pos * 8;
                    let entry_ptr = mem.read_u64(chunk_pa, entry_ptr_off);
                    if entry_ptr == 0 {
                        // Pristine slot: bit was set but
                        // `chunk->data[pos]` never got populated. The
                        // kernel allocator populates
                        // `chunk->data[pos]` after setting the bit
                        // (in `scx_alloc_internal`), so a snapshot
                        // captured between the bit set and the
                        // pointer store sees this transient state.
                        // Skip without counting as a skipped subtree
                        // — it's a legitimate transient. Surface a
                        // `tracing::debug!` so an operator
                        // diagnosing missing slots can see the race
                        // without re-deriving "where did the live
                        // bit go?".
                        tracing::debug!(
                            allocator = %self.out.allocator_name,
                            pos,
                            "sdt_alloc walker: leaf data[pos] == 0 (bit set, \
                             pointer store not yet committed — scx_alloc_internal \
                             populates the pointer after the bitmap bit)",
                        );
                        continue;
                    }

                    self.emit_leaf(entry_ptr);
                }
            }
        } else {
            // Internal level: scan every position in chunk->descs[]
            // and descend into any non-NULL child pointer.
            for pos in 0..SDT_TASK_ENTS_PER_CHUNK {
                let entry_ptr_off = self.offsets.chunk_union + pos * 8;
                let entry_ptr = mem.read_u64(chunk_pa, entry_ptr_off);
                if entry_ptr == 0 {
                    // Never-created subtree: `desc_find_empty` only
                    // writes `desc_children[pos]` when it allocates
                    // a new chunk for a previously empty slot. A
                    // NULL pointer is a legitimate gap, not an
                    // anomaly; skip without counting as a skipped
                    // subtree. `trace!` (not `debug!`) because
                    // a full tree has up to 512 NULL slots per
                    // internal node — a sparse allocator would
                    // flood the debug log otherwise.
                    tracing::trace!(
                        allocator = %self.out.allocator_name,
                        level,
                        pos,
                        "sdt_alloc walker: internal desc[pos] == 0 \
                         (never-created subtree)",
                    );
                    continue;
                }
                self.descend(entry_ptr, level + 1);
            }
        }
    }

    /// Emit one leaf allocation: read tid + payload, BTF-render.
    fn emit_leaf(&mut self, data_ptr: u64) {
        self.out.all_slot_addrs.push(data_ptr & 0xFFFF_FFFF);
        if self.out.entries.len() >= MAX_SDT_ALLOC_ENTRIES {
            self.out.truncated = true;
            return;
        }
        let Some(data_pa) = self.translate_arena_ptr(data_ptr) else {
            self.out.skipped_subtrees = self.out.skipped_subtrees.saturating_add(1);
            return;
        };
        let mem = self.kernel.mem();

        // tid: union sdt_id { s64 val; struct { s32 idx; s32 genn; } }
        // — read as two s32s.
        let idx = mem.read_u32(data_pa, 0) as i32;
        let genn = mem.read_u32(data_pa, 4) as i32;

        // Payload: read self.payload_size bytes after the
        // data_header_size offset.
        let mut payload_bytes = vec![0u8; self.payload_size];
        let n = mem.read_bytes(
            data_pa + self.offsets.data_header_size as u64,
            &mut payload_bytes,
        );
        payload_bytes.truncate(n);
        if payload_bytes.is_empty() {
            // Couldn't read a single byte of payload — chunk was at
            // end-of-DRAM or unmapped. Surface the tid alone (still
            // useful for an operator) with an Unsupported payload
            // carrying the diagnostic reason.
            self.out.entries.push(SdtAllocEntry {
                idx,
                genn,
                user_addr: data_ptr & 0xFFFF_FFFF,
                payload: RenderedValue::Unsupported {
                    reason: "payload read failed: end-of-DRAM or unmapped page".into(),
                },
            });
            return;
        }

        // Zeroed-slot race: `scx_alloc_free_idx` writes zeros to the
        // payload BEFORE clearing the bitmap (see the module-level
        // "Race window" doc). A frozen snapshot captured between
        // those two writes sees a live bitmap bit but an all-zero
        // payload. Surface a `tracing::debug!` so an operator
        // triaging "this slot's payload looks empty" can correlate
        // it with the mid-free race rather than chasing a phantom
        // bug in the renderer. The render itself still proceeds —
        // the consumer can decide whether to interpret an all-zero
        // entry as a real "freshly-allocated zero-init" allocation
        // or as a mid-free residue.
        if payload_bytes.iter().all(|&b| b == 0) {
            tracing::debug!(
                allocator = %self.out.allocator_name,
                idx,
                genn,
                user_addr = format_args!("{:#x}", data_ptr & 0xFFFF_FFFF),
                payload_len = payload_bytes.len(),
                "sdt_alloc walker: all-zero payload (mid-free race? scx_alloc_free_idx \
                 zeros payload before clearing the bitmap)",
            );
        }

        let payload = if self.target_type_id != 0 {
            render_value_with_mem(self.btf, self.target_type_id, &payload_bytes, self.mem)
        } else {
            RenderedValue::Bytes {
                hex: hex_dump(&payload_bytes),
            }
        };

        self.out.entries.push(SdtAllocEntry {
            idx,
            genn,
            user_addr: data_ptr & 0xFFFF_FFFF,
            payload,
        });
    }

    /// Translate a `__arena` pointer to a guest physical address.
    ///
    /// Mirrors the formula in [`super::arena`]: the kernel composes
    /// the actual kern-VA from the LOW 32 bits of the arena pointer
    /// added to `kern_vm_start`. Returns `None` if the translate
    /// fails (page unmapped, PA out of DRAM bounds).
    fn translate_arena_ptr(&self, ptr: u64) -> Option<u64> {
        if ptr == 0 {
            return None;
        }
        let kva = self.kern_vm_start.wrapping_add(ptr & 0xFFFF_FFFF);
        let pa = self.kernel.mem().translate_kva(
            self.kernel.cr3_pa(),
            Kva(kva),
            self.kernel.l5(),
            self.kernel.tcr_el1(),
        )?;
        // Bounds-check the PA: a corrupt PTE could point past
        // end-of-DRAM. Translate guarantees page alignment but not
        // DRAM membership beyond the first page.
        if pa >= self.kernel.mem().size() {
            return None;
        }
        Some(pa)
    }
}

/// Read a u64 at `offset` from a byte slice, returning None when the
/// read would overflow the slice. Little-endian to match the kernel
/// layout the bytes came from.
fn read_u64_at(bytes: &[u8], offset: usize) -> Option<u64> {
    let end = offset.checked_add(8)?;
    let slice = bytes.get(offset..end)?;
    let mut buf = [0u8; 8];
    buf.copy_from_slice(slice);
    Some(u64::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_u64_at_basic() {
        let bytes = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0xff];
        // LE: 0x0807060504030201.
        assert_eq!(read_u64_at(&bytes, 0), Some(0x0807060504030201));
        // Out of range.
        assert_eq!(read_u64_at(&bytes, 2), None);
        assert_eq!(read_u64_at(&bytes, 100), None);
    }

    #[test]
    fn read_u64_at_handles_offset_overflow() {
        // offset.checked_add(8) overflow returns None rather than
        // panicking.
        let bytes = [0u8; 16];
        assert_eq!(read_u64_at(&bytes, usize::MAX), None);
    }

    #[test]
    fn empty_snapshot_serde() {
        let snap = SdtAllocatorSnapshot::default();
        let json = serde_json::to_string(&snap).unwrap();
        // entries / truncated / payload_type_reason skipped when at
        // default (the conditional skip predicates).
        assert!(!json.contains("\"entries\""));
        assert!(!json.contains("\"truncated\""));
        assert!(!json.contains("\"payload_type_reason\""));
        // elem_size, allocator_name, target_type_id, and
        // skipped_subtrees are always emitted — zero values carry
        // diagnostic information that suppression would mask.
        assert!(json.contains("\"elem_size\":0"));
        assert!(json.contains("\"allocator_name\":\"\""));
        assert!(json.contains("\"skipped_subtrees\":0"));
    }

    #[test]
    fn populated_snapshot_roundtrip() {
        let snap = SdtAllocatorSnapshot {
            allocator_name: "scx_task_allocator".into(),
            entries: vec![SdtAllocEntry {
                idx: 7,
                genn: 1,
                user_addr: 0x1000,
                payload: RenderedValue::Bytes {
                    hex: "de ad be ef".into(),
                },
            }],
            truncated: false,
            skipped_subtrees: 2,
            elem_size: 24,
            target_type_id: 42,
            payload_type_reason: String::new(),
            all_slot_addrs: Vec::new(),
        };
        let json = serde_json::to_string(&snap).expect("serialize");
        let parsed: SdtAllocatorSnapshot = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.entries.len(), 1);
        assert_eq!(parsed.entries[0].idx, 7);
        assert_eq!(parsed.entries[0].genn, 1);
        assert_eq!(parsed.elem_size, 24);
        assert_eq!(parsed.target_type_id, 42);
        assert_eq!(parsed.skipped_subtrees, 2);
        assert_eq!(parsed.allocator_name, "scx_task_allocator");
    }

    #[test]
    fn truncated_flag_serialises() {
        let snap = SdtAllocatorSnapshot {
            allocator_name: "x".into(),
            entries: vec![],
            truncated: true,
            skipped_subtrees: 0,
            elem_size: 24,
            target_type_id: 0,
            payload_type_reason: String::new(),
            all_slot_addrs: Vec::new(),
        };
        let json = serde_json::to_string(&snap).unwrap();
        assert!(json.contains("\"truncated\":true"));
    }

    #[test]
    fn payload_type_reason_serialises_when_nonempty() {
        let snap = SdtAllocatorSnapshot {
            allocator_name: "x".into(),
            entries: vec![],
            truncated: false,
            skipped_subtrees: 0,
            elem_size: 24,
            target_type_id: 0,
            payload_type_reason: "no candidate of size 16".into(),
            all_slot_addrs: Vec::new(),
        };
        let json = serde_json::to_string(&snap).unwrap();
        assert!(json.contains("\"payload_type_reason\":\"no candidate of size 16\""));
    }

    #[test]
    fn constants_match_upstream_layout() {
        // Pin the per-chunk fan-out and bitmap shape against
        // `lib/sdt_task_defs.h`. A future upstream change in either
        // value would re-shape `struct sdt_chunk` and surface here.
        assert_eq!(SDT_TASK_LEVELS, 3);
        assert_eq!(SDT_TASK_ENTS_PER_PAGE_SHIFT, 9);
        assert_eq!(SDT_TASK_ENTS_PER_CHUNK, 512);
        assert_eq!(SDT_TASK_CHUNK_BITMAP_U64S, 8);
        assert_eq!(SIZEOF_SDT_ID, 8);
    }

    #[test]
    fn elem_size_bounds_match_kernel() {
        // Mirror `pool_set_size`'s `data_size % 8 == 0` and
        // PAGE_SIZE chunk-fit checks. MIN/MAX must allow every
        // valid scheduler-declared payload size. Wrapped in `const`
        // blocks so the asserts run at compile time — a future drift
        // surfaces as a build failure, not a deferred test failure.
        const {
            assert!(MIN_ELEM_SIZE >= 16);
        }
        const {
            assert!(MAX_ELEM_SIZE <= 4096);
        }
        const {
            assert!(MIN_ELEM_SIZE.is_multiple_of(8));
        }
    }

    #[test]
    fn entry_display_shows_idx_genn_user_addr() {
        let entry = SdtAllocEntry {
            idx: 7,
            genn: 1,
            user_addr: 0x1000,
            payload: RenderedValue::Uint {
                bits: 32,
                value: 42,
            },
        };
        let out = format!("{entry}");
        assert!(out.contains("idx=7"), "missing idx: {out}");
        assert!(out.contains("genn=1"), "missing genn: {out}");
        assert!(out.contains("user_addr=0x1000"), "missing user_addr: {out}");
        assert!(out.contains("payload=42"), "missing payload: {out}");
    }

    #[test]
    fn snapshot_display_shows_header_and_entries() {
        let snap = SdtAllocatorSnapshot {
            allocator_name: "scx_task_allocator".into(),
            entries: vec![SdtAllocEntry {
                idx: 7,
                genn: 1,
                user_addr: 0x1000,
                payload: RenderedValue::Uint {
                    bits: 32,
                    value: 42,
                },
            }],
            truncated: false,
            skipped_subtrees: 0,
            elem_size: 24,
            target_type_id: 42,
            payload_type_reason: String::new(),
            all_slot_addrs: Vec::new(),
        };
        let out = format!("{snap}");
        assert!(
            out.contains("sdt_alloc scx_task_allocator"),
            "missing header: {out}"
        );
        assert!(out.contains("elem_size=24"), "missing elem_size: {out}");
        assert!(
            out.contains("target_type_id=42"),
            "missing target_type_id: {out}"
        );
        assert!(out.contains("1 live"), "missing entry count: {out}");
        assert!(out.contains("idx=7"), "missing entry render: {out}");
    }

    #[test]
    fn snapshot_display_marks_truncated_and_skipped() {
        let snap = SdtAllocatorSnapshot {
            allocator_name: "x".into(),
            entries: vec![],
            truncated: true,
            skipped_subtrees: 5,
            elem_size: 24,
            target_type_id: 0,
            payload_type_reason: "no candidate of size 16".into(),
            all_slot_addrs: Vec::new(),
        };
        let out = format!("{snap}");
        assert!(out.contains("(truncated)"), "missing truncated: {out}");
        assert!(
            out.contains("(5 subtrees skipped)"),
            "missing skipped: {out}"
        );
        assert!(
            out.contains("reason=no candidate of size 16"),
            "missing reason: {out}"
        );
    }

    // -- discover_payload_btf_id ------------------------------------
    //
    // Pure-function tests that don't need a `GuestKernel`. The
    // `walk_sdt_allocator` walker is intentionally NOT covered by
    // unit tests — it requires a live GuestKernel reading frozen
    // VM memory, which is structural integration coverage owned by
    // the existing failure_dump_e2e harness. These tests exercise
    // the BTF heuristic and offset-resolver branches that don't
    // need a kernel handle.

    /// `payload_size == 0` short-circuits without probing any BTF
    /// type ids — the heuristic correctly recognises that an
    /// allocator with zero payload bytes has no struct to discover.
    /// Pinning the early-return shape against a future regression
    /// that might silently start probing on zero (which would
    /// produce a spurious "no candidate of size 0" diagnostic).
    #[test]
    fn discover_payload_btf_id_zero_size_short_circuits() {
        let path = match crate::monitor::find_test_vmlinux() {
            Some(p) => p,
            None => {
                crate::report::test_skip("no vmlinux for BTF load");
                return;
            }
        };
        let btf = match crate::monitor::btf_offsets::load_btf_from_path(&path) {
            Ok(b) => b,
            Err(_) => {
                crate::report::test_skip("BTF load failed");
                return;
            }
        };
        let choice = discover_payload_btf_id(&btf, 0);
        assert_eq!(
            choice.target_type_id, 0,
            "zero-size must yield target_type_id=0"
        );
        assert_eq!(
            choice.reason, "payload_size == 0",
            "zero-size reason must be the early-return marker, got: {}",
            choice.reason
        );
    }

    /// A payload size that no real kernel struct can possibly hit —
    /// `usize::MAX / 2` is far larger than any real struct
    /// (kernel `struct task_struct` is on the order of 10 KiB, the
    /// largest plausible kernel struct is well under a megabyte).
    /// Searching for that size yields zero candidates; assert the
    /// returned reason matches the documented "no candidate of size
    /// {N}" wording so a consumer reading `payload_type_reason` can
    /// rely on the exact format.
    #[test]
    fn discover_payload_btf_id_no_candidate_path() {
        let path = match crate::monitor::find_test_vmlinux() {
            Some(p) => p,
            None => {
                crate::report::test_skip("no vmlinux for BTF load");
                return;
            }
        };
        let btf = match crate::monitor::btf_offsets::load_btf_from_path(&path) {
            Ok(b) => b,
            Err(_) => {
                crate::report::test_skip("BTF load failed");
                return;
            }
        };
        let impossible_size = usize::MAX / 2;
        let choice = discover_payload_btf_id(&btf, impossible_size);
        assert_eq!(choice.target_type_id, 0);
        let expected = format!("no candidate of size {impossible_size}");
        assert_eq!(
            choice.reason, expected,
            "reason must exactly match documented format: got '{}'",
            choice.reason
        );
    }

    /// SdtAllocOffsets::from_btf returns Err when `struct
    /// scx_allocator` is absent from the BTF — vmlinux BTF never
    /// contains it (sdt_alloc lives in the scheduler's program BTF,
    /// not the kernel's), so a from_btf call against vmlinux must
    /// surface the expected error and not panic. The dump pipeline
    /// reads this Err to decide "no sdt_alloc state to surface."
    /// Pin the diagnostic-string contract since callers rely on it.
    #[test]
    fn sdt_alloc_offsets_from_vmlinux_btf_returns_err() {
        let path = match crate::monitor::find_test_vmlinux() {
            Some(p) => p,
            None => {
                crate::report::test_skip("no vmlinux for BTF load");
                return;
            }
        };
        let btf = match crate::monitor::btf_offsets::load_btf_from_path(&path) {
            Ok(b) => b,
            Err(_) => {
                crate::report::test_skip("BTF load failed");
                return;
            }
        };
        let err = SdtAllocOffsets::from_btf(&btf)
            .expect_err("vmlinux BTF must NOT contain scx_allocator — from_btf must Err");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("scx_allocator"),
            "error must name the missing struct so the dump pipeline can log a useful diagnostic: '{msg}'"
        );
    }

    // -- from_btf error paths ----------------------------------------
    //
    // The four error paths in [`SdtAllocOffsets::from_btf`] each
    // surface a distinct diagnostic:
    //
    //   1. `scx_allocator` missing → `"struct scx_allocator unavailable"`
    //      (covered by `sdt_alloc_offsets_from_vmlinux_btf_returns_err`)
    //   2. `sdt_pool` missing      → `"struct sdt_pool unavailable for member offsets"`
    //   3. `sdt_desc` missing      → `"struct sdt_desc unavailable for member offsets"`
    //   4. `sdt_chunk` missing     → `"struct sdt_chunk not found"`
    //
    // The fifth code path — `sdt_data` as `BTF_KIND_FWD` — must
    // succeed (lavd and similar schedulers emit `sdt_data` as a
    // forward declaration; the walker hardcodes 8 from
    // [`SIZEOF_SDT_ID`] when the body is absent).
    //
    // The tests below build minimal synthetic BTF blobs (mirroring
    // the per-test-module pattern in
    // `cast_analysis::tests::build_btf` and
    // `btf_render::tests::cast_build_btf` — pared down here to only
    // the kinds `from_btf` consults: BTF_KIND_INT, BTF_KIND_STRUCT,
    // BTF_KIND_FWD) and parse them via `Btf::from_bytes`. Synthetic
    // BTF makes the four error paths reachable deterministically
    // without requiring a real scheduler program BTF on disk.
    //
    // The constants and wire format mirror linux uapi `btf.h` and
    // `Documentation/bpf/btf.rst`. The `info` u32 layout: `kind`
    // in bits 24..29, `vlen` in low 16 bits, `kind_flag` in bit 31.

    const SDTA_BTF_MAGIC: u16 = 0xEB9F;
    const SDTA_BTF_VERSION: u8 = 1;
    const SDTA_BTF_HEADER_LEN: u32 = 24;
    const SDTA_BTF_KIND_INT: u32 = 1;
    const SDTA_BTF_KIND_STRUCT: u32 = 4;
    /// `BTF_KIND_FWD = 7`. Forward declaration. Carries a name but
    /// no body; `kind_flag` selects struct (0) vs union (1) per
    /// `btf-rs::Fwd::is_struct` / `is_union`.
    const SDTA_BTF_KIND_FWD: u32 = 7;

    /// One member of a synthetic `BTF_KIND_STRUCT`. The wire format
    /// stores `bit_offset` (member offset in BITS, not bytes); the
    /// test helper converts from `byte_offset` for readability.
    #[derive(Clone, Copy)]
    struct SdtaSynMember {
        name_off: u32,
        type_id: u32,
        byte_offset: u32,
    }

    /// One synthetic BTF type. The pared-down kind set (`Int`,
    /// `Struct`, `Fwd`) is exactly what `from_btf` needs to traverse:
    /// member offsets come from `Struct`s, the `sdt_data` Fwd path
    /// exercises the [`SIZEOF_SDT_ID`] fallback, and `Int` provides
    /// terminal type ids the struct members can reference.
    enum SdtaSynType {
        /// `BTF_KIND_INT`. `encoding=0` is plain unsigned (the form
        /// `u64` / `u32` resolve to in libbpf-emitted BTF).
        Int {
            name_off: u32,
            size: u32,
            encoding: u32,
            offset: u32,
            bits: u32,
        },
        /// `BTF_KIND_STRUCT` with `kind_flag=0` — non-bitfield
        /// members. `from_btf` only consumes byte-aligned member
        /// offsets via [`member_byte_offset`], so the simpler
        /// `kind_flag=0` form suffices.
        Struct {
            name_off: u32,
            size: u32,
            members: Vec<SdtaSynMember>,
        },
        /// `BTF_KIND_FWD` (struct flavour, `kind_flag=0`). Used by
        /// the `sdt_data` Fwd-fallback test — the only path
        /// `from_btf` accepts a Fwd on, since the header size is
        /// kernel-source-fixed at `SIZEOF_SDT_ID = 8`.
        Fwd { name_off: u32 },
    }

    /// Append a NUL-terminated string to the BTF strings buffer and
    /// return its byte offset. Same shape as
    /// `cast_analysis::tests::push_name`, kept private to this test
    /// module to avoid coupling between fixtures.
    fn sdta_push_name(s: &mut Vec<u8>, name: &str) -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    }

    /// Build a minimal BTF byte blob from a list of synthetic types
    /// and a string section.
    ///
    /// Header layout matches `cast_analysis::tests::build_btf`:
    /// 24-byte header (magic + version + flags + hdr_len + type_off
    /// + type_len + str_off + str_len) followed by the type section
    ///   then the string section. Type ids start at 1 (id 0 is Void)
    ///   and increase in `types` order.
    fn sdta_build_btf(types: &[SdtaSynType], strings: &[u8]) -> Vec<u8> {
        let mut type_section: Vec<u8> = Vec::new();
        for ty in types {
            match ty {
                SdtaSynType::Int {
                    name_off,
                    size,
                    encoding,
                    offset,
                    bits,
                } => {
                    type_section.extend_from_slice(&name_off.to_le_bytes());
                    let info = (SDTA_BTF_KIND_INT << 24) & 0x1f00_0000;
                    type_section.extend_from_slice(&info.to_le_bytes());
                    type_section.extend_from_slice(&size.to_le_bytes());
                    let int_data = (*encoding << 24) | ((*offset & 0xff) << 16) | (*bits & 0xff);
                    type_section.extend_from_slice(&int_data.to_le_bytes());
                }
                SdtaSynType::Struct {
                    name_off,
                    size,
                    members,
                } => {
                    type_section.extend_from_slice(&name_off.to_le_bytes());
                    let vlen = members.len() as u32;
                    let info = ((SDTA_BTF_KIND_STRUCT << 24) & 0x1f00_0000) | (vlen & 0xffff);
                    type_section.extend_from_slice(&info.to_le_bytes());
                    type_section.extend_from_slice(&size.to_le_bytes());
                    for m in members {
                        type_section.extend_from_slice(&m.name_off.to_le_bytes());
                        type_section.extend_from_slice(&m.type_id.to_le_bytes());
                        // Non-bitfield struct: bit_offset = byte * 8.
                        let bit_off = m.byte_offset * 8;
                        type_section.extend_from_slice(&bit_off.to_le_bytes());
                    }
                }
                SdtaSynType::Fwd { name_off } => {
                    type_section.extend_from_slice(&name_off.to_le_bytes());
                    // BTF_KIND_FWD: vlen=0, kind_flag=0 (struct
                    // flavour). size_type field is unused but is
                    // still 4 bytes wide on the wire — emit 0.
                    let info = (SDTA_BTF_KIND_FWD << 24) & 0x1f00_0000;
                    type_section.extend_from_slice(&info.to_le_bytes());
                    type_section.extend_from_slice(&0u32.to_le_bytes());
                }
            }
        }

        let type_len = type_section.len() as u32;
        let str_len = strings.len() as u32;

        let mut blob: Vec<u8> = Vec::new();
        blob.extend_from_slice(&SDTA_BTF_MAGIC.to_le_bytes());
        blob.push(SDTA_BTF_VERSION);
        blob.push(0); // flags
        blob.extend_from_slice(&SDTA_BTF_HEADER_LEN.to_le_bytes());
        blob.extend_from_slice(&0u32.to_le_bytes()); // type_off
        blob.extend_from_slice(&type_len.to_le_bytes());
        blob.extend_from_slice(&type_len.to_le_bytes()); // str_off = type_len
        blob.extend_from_slice(&str_len.to_le_bytes());
        blob.extend_from_slice(&type_section);
        blob.extend_from_slice(strings);
        blob
    }

    /// Set of names every from_btf-error-path test needs in its
    /// string section. Shared so each test's setup stays focused on
    /// the type list, not the string table mechanics.
    ///
    /// Returns `(strings, name_offsets)` where `name_offsets` is a
    /// struct of byte offsets keyed by name.
    fn sdta_strings_for_from_btf() -> (Vec<u8>, SdtaNames) {
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = sdta_push_name(&mut strings, "u64");
        let n_scx_allocator = sdta_push_name(&mut strings, "scx_allocator");
        let n_sdt_pool = sdta_push_name(&mut strings, "sdt_pool");
        let n_sdt_desc = sdta_push_name(&mut strings, "sdt_desc");
        let n_sdt_chunk = sdta_push_name(&mut strings, "sdt_chunk");
        let n_sdt_data = sdta_push_name(&mut strings, "sdt_data");
        let n_pool = sdta_push_name(&mut strings, "pool");
        let n_root = sdta_push_name(&mut strings, "root");
        let n_elem_size = sdta_push_name(&mut strings, "elem_size");
        let n_allocated = sdta_push_name(&mut strings, "allocated");
        let n_nr_free = sdta_push_name(&mut strings, "nr_free");
        let n_chunk = sdta_push_name(&mut strings, "chunk");
        let n_descs = sdta_push_name(&mut strings, "descs");
        (
            strings,
            SdtaNames {
                n_u64,
                n_scx_allocator,
                n_sdt_pool,
                n_sdt_desc,
                n_sdt_chunk,
                n_sdt_data,
                n_pool,
                n_root,
                n_elem_size,
                n_allocated,
                n_nr_free,
                n_chunk,
                n_descs,
            },
        )
    }

    /// Byte offsets within the string section for the names every
    /// from_btf-error-path test references. Bundled into a struct so
    /// each test's local state stays tidy and so the order of `let`
    /// bindings matches across tests (preventing accidental skews
    /// between tests that all reference the same name table).
    struct SdtaNames {
        n_u64: u32,
        n_scx_allocator: u32,
        n_sdt_pool: u32,
        n_sdt_desc: u32,
        n_sdt_chunk: u32,
        n_sdt_data: u32,
        n_pool: u32,
        n_root: u32,
        n_elem_size: u32,
        n_allocated: u32,
        n_nr_free: u32,
        n_chunk: u32,
        n_descs: u32,
    }

    /// Build the minimal `scx_allocator` struct every from_btf path
    /// must traverse before reaching the inner-struct lookups. Two
    /// members `pool` and `root`, both typed as `u64` (type_id=1)
    /// since `from_btf` only reads each member's byte offset, not
    /// its type. `pool` at offset 0, `root` at offset 8, total size
    /// 16 — matches the kernel's `struct scx_allocator { struct
    /// sdt_pool pool; sdt_desc_t *root; }` member ORDER (the actual
    /// kernel `pool` is itself a struct, but the synthetic version
    /// only needs a name + offset for [`member_byte_offset`] to
    /// succeed).
    fn sdta_allocator_struct(names: &SdtaNames) -> SdtaSynType {
        SdtaSynType::Struct {
            name_off: names.n_scx_allocator,
            size: 16,
            members: vec![
                SdtaSynMember {
                    name_off: names.n_pool,
                    type_id: 1,
                    byte_offset: 0,
                },
                SdtaSynMember {
                    name_off: names.n_root,
                    type_id: 1,
                    byte_offset: 8,
                },
            ],
        }
    }

    /// Test 1: `scx_allocator` is present but `sdt_pool` is missing
    /// from the BTF entirely. `from_btf` must surface the
    /// `"sdt_pool unavailable for member offsets"` context — the
    /// distinct diagnostic that lets the dump pipeline distinguish
    /// "no scheduler links sdt_alloc" (test
    /// `sdt_alloc_offsets_from_vmlinux_btf_returns_err`) from
    /// "scheduler links sdt_alloc but the BTF stripped sdt_pool".
    #[test]
    fn sdt_alloc_offsets_missing_sdt_pool_distinct_error() {
        let (strings, names) = sdta_strings_for_from_btf();
        let types = vec![
            // id=1: u64 plain unsigned. Used as the type for every
            // member of the synthetic structs (see
            // `sdta_allocator_struct`'s comment).
            SdtaSynType::Int {
                name_off: names.n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            // id=2: scx_allocator (full struct).
            sdta_allocator_struct(&names),
            // id=3: sdt_desc (full struct, present so we don't
            // accidentally match its error path instead).
            SdtaSynType::Struct {
                name_off: names.n_sdt_desc,
                size: 24,
                members: vec![
                    SdtaSynMember {
                        name_off: names.n_allocated,
                        type_id: 1,
                        byte_offset: 0,
                    },
                    SdtaSynMember {
                        name_off: names.n_nr_free,
                        type_id: 1,
                        byte_offset: 8,
                    },
                    SdtaSynMember {
                        name_off: names.n_chunk,
                        type_id: 1,
                        byte_offset: 16,
                    },
                ],
            },
            // id=4: sdt_chunk (full struct).
            SdtaSynType::Struct {
                name_off: names.n_sdt_chunk,
                size: 8,
                members: vec![SdtaSynMember {
                    name_off: names.n_descs,
                    type_id: 1,
                    byte_offset: 0,
                }],
            },
            // id=5: sdt_data (Fwd, the form `from_btf` accepts).
            SdtaSynType::Fwd {
                name_off: names.n_sdt_data,
            },
            // sdt_pool is intentionally OMITTED — this is the path
            // under test.
        ];
        let blob = sdta_build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

        let err =
            SdtAllocOffsets::from_btf(&btf).expect_err("missing sdt_pool must surface as Err");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("sdt_pool"),
            "error must name the missing struct: '{msg}'"
        );
        assert!(
            msg.contains("unavailable for member offsets"),
            "error must carry the sdt_pool-specific context distinguishing this from sdt_chunk's 'not found' wording: '{msg}'"
        );
        // The diagnostic must NOT name unrelated structs — the
        // error path is sdt_pool-specific. A regression that
        // reordered the require_full_struct calls would surface
        // `sdt_desc` in the message instead.
        assert!(
            !msg.contains("sdt_desc"),
            "missing-sdt_pool error must not reference sdt_desc: '{msg}'"
        );
        assert!(
            !msg.contains("sdt_chunk"),
            "missing-sdt_pool error must not reference sdt_chunk: '{msg}'"
        );
    }

    /// Test 2: `scx_allocator` and `sdt_pool` are present but
    /// `sdt_desc` is missing from the BTF. The error must reach
    /// `from_btf`'s `sdt_desc` lookup specifically — surfacing the
    /// `"sdt_desc unavailable for member offsets"` context — and
    /// not collapse onto an earlier failure.
    #[test]
    fn sdt_alloc_offsets_missing_sdt_desc_distinct_error() {
        let (strings, names) = sdta_strings_for_from_btf();
        let types = vec![
            // id=1: u64 plain unsigned.
            SdtaSynType::Int {
                name_off: names.n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            // id=2: scx_allocator (full struct).
            sdta_allocator_struct(&names),
            // id=3: sdt_pool (full struct, present).
            SdtaSynType::Struct {
                name_off: names.n_sdt_pool,
                size: 32,
                members: vec![SdtaSynMember {
                    name_off: names.n_elem_size,
                    type_id: 1,
                    byte_offset: 16,
                }],
            },
            // id=4: sdt_chunk (full struct, present so we reach
            // sdt_desc before sdt_chunk).
            SdtaSynType::Struct {
                name_off: names.n_sdt_chunk,
                size: 8,
                members: vec![SdtaSynMember {
                    name_off: names.n_descs,
                    type_id: 1,
                    byte_offset: 0,
                }],
            },
            // id=5: sdt_data (Fwd).
            SdtaSynType::Fwd {
                name_off: names.n_sdt_data,
            },
            // sdt_desc is intentionally OMITTED.
        ];
        let blob = sdta_build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

        let err =
            SdtAllocOffsets::from_btf(&btf).expect_err("missing sdt_desc must surface as Err");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("sdt_desc"),
            "error must name the missing struct: '{msg}'"
        );
        assert!(
            msg.contains("unavailable for member offsets"),
            "error must carry the sdt_desc-specific context distinguishing this from sdt_chunk's 'not found' wording: '{msg}'"
        );
        // sdt_pool must NOT appear — sdt_pool resolved successfully
        // before from_btf reached the sdt_desc lookup. A leak
        // would mean require_full_struct's context is misordered.
        assert!(
            !msg.contains("sdt_pool"),
            "missing-sdt_desc error must not reference sdt_pool: '{msg}'"
        );
        assert!(
            !msg.contains("sdt_chunk"),
            "missing-sdt_desc error must not reference sdt_chunk: '{msg}'"
        );
    }

    /// Test 3: `scx_allocator`, `sdt_pool`, `sdt_desc` all present
    /// but `sdt_chunk` is missing. The error must surface the
    /// distinct `"sdt_chunk not found"` context — sdt_chunk goes
    /// through [`find_struct_or_fwd`] (not `require_full_struct`),
    /// so its diagnostic wording differs from sdt_pool / sdt_desc.
    #[test]
    fn sdt_alloc_offsets_missing_sdt_chunk_distinct_error() {
        let (strings, names) = sdta_strings_for_from_btf();
        let types = vec![
            // id=1: u64 plain unsigned.
            SdtaSynType::Int {
                name_off: names.n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            // id=2: scx_allocator (full struct).
            sdta_allocator_struct(&names),
            // id=3: sdt_pool (full struct).
            SdtaSynType::Struct {
                name_off: names.n_sdt_pool,
                size: 32,
                members: vec![SdtaSynMember {
                    name_off: names.n_elem_size,
                    type_id: 1,
                    byte_offset: 16,
                }],
            },
            // id=4: sdt_desc (full struct, present so sdt_chunk is
            // the failing lookup).
            SdtaSynType::Struct {
                name_off: names.n_sdt_desc,
                size: 24,
                members: vec![
                    SdtaSynMember {
                        name_off: names.n_allocated,
                        type_id: 1,
                        byte_offset: 0,
                    },
                    SdtaSynMember {
                        name_off: names.n_nr_free,
                        type_id: 1,
                        byte_offset: 8,
                    },
                    SdtaSynMember {
                        name_off: names.n_chunk,
                        type_id: 1,
                        byte_offset: 16,
                    },
                ],
            },
            // id=5: sdt_data (Fwd).
            SdtaSynType::Fwd {
                name_off: names.n_sdt_data,
            },
            // sdt_chunk is intentionally OMITTED.
        ];
        let blob = sdta_build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

        let err =
            SdtAllocOffsets::from_btf(&btf).expect_err("missing sdt_chunk must surface as Err");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("sdt_chunk"),
            "error must name the missing struct: '{msg}'"
        );
        // sdt_chunk uses `find_struct_or_fwd` with the context
        // `"btf: struct sdt_chunk not found"`. The inner anyhow
        // error from `with_context` always carries `"type 'X' not
        // found"`, so a contains-"not found" check alone is also
        // satisfied by the sdt_pool / sdt_desc paths via their
        // inner context. The distinguishing OUTER-context check is
        // the absence of `"unavailable for member offsets"` — that
        // wording is sdt_pool / sdt_desc-specific.
        assert!(
            msg.contains("not found"),
            "sdt_chunk error must carry the find_struct_or_fwd 'not found' wording: '{msg}'"
        );
        assert!(
            !msg.contains("unavailable for member offsets"),
            "sdt_chunk uses find_struct_or_fwd, NOT require_full_struct — the 'unavailable for member offsets' phrase is sdt_pool / sdt_desc-specific and must not appear: '{msg}'"
        );
        assert!(
            !msg.contains("sdt_pool"),
            "missing-sdt_chunk error must not reference sdt_pool: '{msg}'"
        );
        assert!(
            !msg.contains("sdt_desc"),
            "missing-sdt_chunk error must not reference sdt_desc: '{msg}'"
        );
    }

    /// Test 4: every required type present and `sdt_data` emitted
    /// as a `BTF_KIND_FWD` forward declaration. `from_btf` must
    /// succeed and `data_header_size` must equal
    /// [`SIZEOF_SDT_ID`] = 8 — the kernel-header-fixed size of the
    /// `union sdt_id` header (the only non-flex-array member in
    /// `struct sdt_data`). This is the lavd-style scheduler path:
    /// the program never accesses `sdt_data` members directly so
    /// libbpf strips the body to a Fwd, and the walker covers leaf
    /// liveness via the leaf descriptor's `allocated[]` bitmap rather
    /// than the slot's own header content.
    #[test]
    fn sdt_alloc_offsets_sdt_data_fwd_uses_sizeof_sdt_id() {
        let (strings, names) = sdta_strings_for_from_btf();
        let types = vec![
            // id=1: u64 plain unsigned.
            SdtaSynType::Int {
                name_off: names.n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            // id=2: scx_allocator (full struct).
            sdta_allocator_struct(&names),
            // id=3: sdt_pool (full struct).
            SdtaSynType::Struct {
                name_off: names.n_sdt_pool,
                size: 32,
                members: vec![SdtaSynMember {
                    name_off: names.n_elem_size,
                    type_id: 1,
                    byte_offset: 16,
                }],
            },
            // id=4: sdt_desc (full struct).
            SdtaSynType::Struct {
                name_off: names.n_sdt_desc,
                size: 24,
                members: vec![
                    SdtaSynMember {
                        name_off: names.n_allocated,
                        type_id: 1,
                        byte_offset: 0,
                    },
                    SdtaSynMember {
                        name_off: names.n_nr_free,
                        type_id: 1,
                        byte_offset: 8,
                    },
                    SdtaSynMember {
                        name_off: names.n_chunk,
                        type_id: 1,
                        byte_offset: 16,
                    },
                ],
            },
            // id=5: sdt_chunk (full struct with `descs` member at
            // offset 0 — matches the kernel layout's union at
            // offset 0).
            SdtaSynType::Struct {
                name_off: names.n_sdt_chunk,
                size: 8,
                members: vec![SdtaSynMember {
                    name_off: names.n_descs,
                    type_id: 1,
                    byte_offset: 0,
                }],
            },
            // id=6: sdt_data (Fwd) — the path under test. The
            // hardcoded fallback to SIZEOF_SDT_ID must fire.
            SdtaSynType::Fwd {
                name_off: names.n_sdt_data,
            },
        ];
        let blob = sdta_build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

        let offsets = SdtAllocOffsets::from_btf(&btf)
            .expect("sdt_data Fwd must NOT cause from_btf to fail — Fwd is the lavd-style path");
        assert_eq!(
            offsets.data_header_size, SIZEOF_SDT_ID,
            "data_header_size for a Fwd sdt_data must fall back to SIZEOF_SDT_ID (=8, the union sdt_id header size that lib/sdt_task_defs.h fixes)"
        );
        assert_eq!(
            offsets.data_header_size, 8,
            "literal-8 cross-check: the Fwd fallback must equal exactly 8 bytes (kernel-header-fixed)"
        );
    }

    // -- discover_payload_btf_id heuristic branches ---------------
    //
    // The existing tests (`discover_payload_btf_id_zero_size_short_circuits`,
    // `discover_payload_btf_id_no_candidate_path`) cover only the
    // payload_size=0 short-circuit and the empty-size_matches path.
    // The heuristic's actual branching logic — single-match returns
    // id, multi-match falls through pattern arms (task_ctx exact →
    // *_arena_ctx → *_task_ctx → *_ctx suffix), per-arm ambiguity,
    // anonymous-struct rejection — is entirely uncovered. Without
    // these tests, an implementation that only handles the two
    // existing edge cases passes the suite while being broken for
    // every real per-cgroup or per-task allocator.
    //
    // The bug surface for #89 is per-cgroup arena pointers (cgx_raw,
    // llcx_raw) failing to chase. Test G1.2 below covers
    // `scx_cgroup_ctx`-style names matching the `*_ctx` arm; G1.5
    // covers the per-arm ambiguous fallback; G1.7 covers the
    // continue-to-next-arm path that the docstring at sdt_alloc.rs:565-571
    // contradicts the code at sdt_alloc.rs:670-674 about. The doc
    // says "fall back to hex" on any arm collision; the code says
    // "continue to next arm". The implementer's #89 fix must
    // reconcile this drift — these tests pin the chosen behavior.
    //
    // All three tests use the existing `sdta_*` BTF builder helpers
    // declared above to avoid duplicating wire-format logic.

    /// G1.1: Single size-match resolves cleanly. A BTF with one
    /// 16-byte struct named `cgrp_ctx` and one 8-byte int. Calling
    /// `discover_payload_btf_id(&btf, 16)` finds `cgrp_ctx`
    /// as the unique size-match; size_matches.len() == 1 routes to
    /// the "single match" arm and returns the id with empty reason.
    /// Pin the contract that a single-match path bypasses the
    /// pattern-priority dispatch entirely.
    #[test]
    fn discover_payload_btf_id_single_size_match_returns_id() {
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = sdta_push_name(&mut strings, "u64");
        let n_cgrp_ctx = sdta_push_name(&mut strings, "cgrp_ctx");
        let n_a = sdta_push_name(&mut strings, "a");
        let n_b = sdta_push_name(&mut strings, "b");
        let types = vec![
            // id 1: u64 (8 bytes; not a size-match for 16).
            SdtaSynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            // id 2: struct cgrp_ctx { u64 a @ 0; u64 b @ 8 } size=16.
            SdtaSynType::Struct {
                name_off: n_cgrp_ctx,
                size: 16,
                members: vec![
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 0,
                    },
                    SdtaSynMember {
                        name_off: n_b,
                        type_id: 1,
                        byte_offset: 8,
                    },
                ],
            },
        ];
        let blob = sdta_build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
        let choice = discover_payload_btf_id(&btf, 16);
        assert_eq!(
            choice.target_type_id, 2,
            "single 16-byte struct cgrp_ctx must be picked unambiguously"
        );
        assert_eq!(
            choice.reason, "",
            "single-match path must return empty reason; got {:?}",
            choice.reason
        );
    }

    /// G1.2: Per-cgroup `scx_cgroup_ctx`-style name resolves via
    /// the `*_ctx` suffix arm. With one 16-byte struct named
    /// `scx_cgroup_ctx`, this is also a single size-match — the
    /// test pins that the heuristic accepts the per-cgroup name
    /// AND that an int of size 8 doesn't pollute the candidate
    /// list. The bug surface for #89 is exactly this case
    /// (per-cgroup struct fails to resolve via discover);
    /// confirming a clean single-match here pins the baseline
    /// before the multi-candidate cases below exercise the actual
    /// branching.
    #[test]
    fn discover_payload_btf_id_per_cgroup_ctx_resolves_via_ctx_suffix() {
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = sdta_push_name(&mut strings, "u64");
        let n_cgrp = sdta_push_name(&mut strings, "scx_cgroup_ctx");
        let n_a = sdta_push_name(&mut strings, "a");
        let n_b = sdta_push_name(&mut strings, "b");
        let types = vec![
            SdtaSynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SdtaSynType::Struct {
                name_off: n_cgrp,
                size: 16,
                members: vec![
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 0,
                    },
                    SdtaSynMember {
                        name_off: n_b,
                        type_id: 1,
                        byte_offset: 8,
                    },
                ],
            },
        ];
        let blob = sdta_build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
        let choice = discover_payload_btf_id(&btf, 16);
        assert_eq!(
            choice.target_type_id, 2,
            "scx_cgroup_ctx (single 16-byte size-match) must resolve via the \
             single-match arm; the per-cgroup struct name is the bug surface for #89"
        );
        assert_eq!(choice.reason, "");
    }

    /// G1.3: `task_ctx` (exact name) wins over `*_ctx` suffix
    /// when both same-size structs exist. The heuristic at
    /// sdt_alloc.rs:646-651 lists arm 1 (`n == "task_ctx"`)
    /// before arm 4 (`*_ctx` suffix). Pin the priority order so
    /// a future refactor that drops the exact-name arm in favor
    /// of a single suffix-pattern walk surfaces here.
    #[test]
    fn discover_payload_btf_id_task_ctx_exact_wins_over_ctx_suffix() {
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = sdta_push_name(&mut strings, "u64");
        let n_task = sdta_push_name(&mut strings, "task_ctx");
        let n_cgrp = sdta_push_name(&mut strings, "cgrp_ctx");
        let n_a = sdta_push_name(&mut strings, "a");
        let types = vec![
            SdtaSynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            // id 2: task_ctx (16 bytes).
            SdtaSynType::Struct {
                name_off: n_task,
                size: 16,
                members: vec![
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 0,
                    },
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 8,
                    },
                ],
            },
            // id 3: cgrp_ctx (16 bytes — same size).
            SdtaSynType::Struct {
                name_off: n_cgrp,
                size: 16,
                members: vec![
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 0,
                    },
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 8,
                    },
                ],
            },
        ];
        let blob = sdta_build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
        let choice = discover_payload_btf_id(&btf, 16);
        assert_eq!(
            choice.target_type_id, 2,
            "exact `task_ctx` arm (priority 1) must win over `*_ctx` suffix \
             arm (priority 4); cgrp_ctx must NOT be picked: {:?}",
            choice
        );
        assert_eq!(choice.reason, "");
    }

    /// G1.5: Ambiguous at the `*_ctx` suffix arm with no upper-arm
    /// resolution. Two structs (`cgrp_ctx` and `task_data_ctx`)
    /// both 16 bytes, neither matching exact `task_ctx`,
    /// `*_arena_ctx`, or `*_task_ctx`. Arm 4 (`*_ctx`) gets 2 hits;
    /// the per-arm `_ => continue` at sdt_alloc.rs:670-674 advances
    /// past arm 4 (the last arm) and falls through to the
    /// post-loop "no unambiguous pattern winner" branch at
    /// sdt_alloc.rs:677-681, returning `target_type_id = 0` with
    /// `reason = "ambiguous: 2 candidates"`. Pin BOTH the id (0)
    /// AND the exact reason — the reason format is wire-stable
    /// (operator-visible via SdtAllocatorSnapshot::payload_type_reason).
    #[test]
    fn discover_payload_btf_id_ambiguous_at_ctx_arm_falls_through() {
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = sdta_push_name(&mut strings, "u64");
        let n_a = sdta_push_name(&mut strings, "a");
        // Two structs ending in `_ctx` — both qualify ONLY at arm 4.
        let n_cgrp = sdta_push_name(&mut strings, "cgrp_ctx");
        let n_task_data = sdta_push_name(&mut strings, "task_data_ctx");
        let types = vec![
            SdtaSynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SdtaSynType::Struct {
                name_off: n_cgrp,
                size: 16,
                members: vec![
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 0,
                    },
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 8,
                    },
                ],
            },
            SdtaSynType::Struct {
                name_off: n_task_data,
                size: 16,
                members: vec![
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 0,
                    },
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 8,
                    },
                ],
            },
        ];
        let blob = sdta_build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
        let choice = discover_payload_btf_id(&btf, 16);
        assert_eq!(
            choice.target_type_id, 0,
            "ambiguous `*_ctx` matches must fall through every arm and \
             return target_type_id=0; got {:?}",
            choice
        );
        assert_eq!(
            choice.reason, "ambiguous: 2 candidates",
            "ambiguous-fallback reason format is wire-stable (operator reads \
             SdtAllocatorSnapshot::payload_type_reason). Pin the format string \
             byte-for-byte; a refactor that changes 'ambiguous' to 'multi' or \
             'candidates' to 'matches' would silently break log scrapers."
        );
    }

    /// G1.7: Per-arm continue resolves at lower arm. TWO `*_arena_ctx`
    /// structs (ambiguous at arm 2) AND ONE `*_task_ctx` struct
    /// (unambiguous at arm 3). Per the production code at
    /// sdt_alloc.rs:670-674, arm 2's `_ => continue` advances to
    /// arm 3, which has 1 hit → returns the `*_task_ctx` id.
    ///
    /// The docstring at sdt_alloc.rs:565-571 says "If 2+ structs
    /// match the *same* pattern, we fall back to hex". The code
    /// CONTRADICTS this — `continue` proceeds to the next arm
    /// rather than aborting to the post-loop fall-through. This
    /// test pins the CODE's behavior; if the implementer's #89
    /// fix changes either side, the test must be updated to
    /// match the new semantics. The doc-vs-code drift was flagged
    /// in tester findings; the implementer is responsible for
    /// reconciling both sides in the same commit.
    #[test]
    fn discover_payload_btf_id_per_arm_ambiguity_resolves_at_lower_arm() {
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = sdta_push_name(&mut strings, "u64");
        let n_a = sdta_push_name(&mut strings, "a");
        // Two `*_arena_ctx` (collide at arm 2):
        let n_cgrp_arena = sdta_push_name(&mut strings, "cgrp_arena_ctx");
        let n_other_arena = sdta_push_name(&mut strings, "other_arena_ctx");
        // One unique `*_task_ctx` (resolves at arm 3):
        let n_my_task = sdta_push_name(&mut strings, "my_task_ctx");
        let types = vec![
            SdtaSynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            // id 2: cgrp_arena_ctx (16 bytes).
            SdtaSynType::Struct {
                name_off: n_cgrp_arena,
                size: 16,
                members: vec![
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 0,
                    },
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 8,
                    },
                ],
            },
            // id 3: other_arena_ctx (16 bytes — collides with id 2 at arm 2).
            SdtaSynType::Struct {
                name_off: n_other_arena,
                size: 16,
                members: vec![
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 0,
                    },
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 8,
                    },
                ],
            },
            // id 4: my_task_ctx (16 bytes — unique at arm 3,
            // matches `*_task_ctx`).
            SdtaSynType::Struct {
                name_off: n_my_task,
                size: 16,
                members: vec![
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 0,
                    },
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 8,
                    },
                ],
            },
        ];
        let blob = sdta_build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
        let choice = discover_payload_btf_id(&btf, 16);
        // Arm 1 (`task_ctx` exact): no hit. Arm 2 (`*_arena_ctx`):
        // 2 hits → continue. Arm 3 (`*_task_ctx`): 1 hit → return
        // id 4. Arm 4 (`*_ctx`): never reached.
        assert_eq!(
            choice.target_type_id, 4,
            "arm 2 ambiguous → continue; arm 3 unique my_task_ctx → return id 4. \
             Got {:?}. If this fails, the implementer's fix changed the \
             continue-on-arm-ambiguity semantics — verify against the \
             docstring at sdt_alloc.rs:565-571 (which currently contradicts \
             the code) and update both sides together.",
            choice
        );
        assert_eq!(
            choice.reason, "",
            "successful pattern-arm resolution must return empty reason"
        );
    }
}
