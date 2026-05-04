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
//! The walker uses the per-level `allocated[]` bitmap as the source
//! of truth, not `tid.idx` or `pool.idx` — both are unreliable:
//! post-free `tid.idx` is reset to 0 (ambiguous with slot 0), and
//! `pool.idx` is the pool's high-water mark, not the live count.
//! `chunk->data[pos]` is also nullable for pristine slots that the
//! pool never handed out — we skip those silently.
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
const MAX_BTF_ID_PROBE: u32 = 100_000;

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
    /// User-side virtual address of the `sdt_data` slot. 32-bit window
    /// matching `arena.user_vm_start`; lets a consumer correlate the
    /// rendered allocation against pointer values they see in BPF
    /// program output.
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
    /// `payload_btf_type_id`.
    pub skipped_subtrees: u32,
    /// Diagnostic: the per-pool slot stride. Surfaces alongside the
    /// rendered entries so a consumer can spot when the rendered
    /// payload size diverges from the declared one.
    pub elem_size: u64,
    /// Diagnostic: the BTF type id used to render payload bytes.
    /// 0 when [`discover_payload_btf_id`] returned no candidate and
    /// the renderer fell back to hex.
    pub payload_btf_type_id: u32,
    /// Diagnostic: when [`discover_payload_btf_id`] returned 0, the
    /// reason (e.g. `"no candidate of size 16"`,
    /// `"ambiguous: 3 candidates"`, `"payload_size == 0"`). Empty on
    /// successful BTF resolve. Lets an operator distinguish the
    /// fallback paths without re-deriving the heuristic.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub payload_type_reason: String,
}

impl std::fmt::Display for SdtAllocatorSnapshot {
    /// Header line + one entry per allocation, indented. Diagnostic
    /// lines (truncated, skipped_subtrees) appended when non-default.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sdt_alloc {} (elem_size={}, btf_type_id={}",
            self.allocator_name, self.elem_size, self.payload_btf_type_id
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
            f.write_str("\n  ")?;
            std::fmt::Display::fmt(entry, f)?;
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
/// `payload_btf_type_id` is the BTF type id the renderer applies to
/// the bytes after the `tid` header. Pass 0 (or a discovered id from
/// [`discover_payload_btf_id`]) — 0 routes to a hex dump.
///
/// `payload_type_reason` is a human-readable string describing why
/// `payload_btf_type_id` is 0 (when it is); ignored when the id is
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
    kernel: &GuestKernel<'_>,
    kern_vm_start: u64,
    allocator_bytes: &[u8],
    offsets: &SdtAllocOffsets,
    btf: &Btf,
    payload_btf_type_id: u32,
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
        payload_btf_type_id,
        payload_type_reason: payload_type_reason.into(),
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
        payload_btf_type_id,
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
    pub btf_type_id: u32,
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
///      `ktstr_arena_ctx`. If 2+ structs match the *same* pattern
///      (e.g. two `*_arena_ctx` of the same size), we fall back to
///      hex rather than guess between them.
///   3. No match or still ambiguous → return 0 to fall back to a hex
///      dump.
///
/// The function is intentionally conservative: a wrong type id renders
/// nonsense field names; falling back to hex always shows the operator
/// raw bytes they can decode by hand. The returned reason string is
/// surfaced to the operator via [`SdtAllocatorSnapshot::payload_type_reason`]
/// so the fallback paths are distinguishable without re-running the
/// heuristic.
pub fn discover_payload_btf_id(btf: &Btf, payload_size: usize) -> PayloadTypeChoice {
    if payload_size == 0 {
        return PayloadTypeChoice {
            btf_type_id: 0,
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
            btf_type_id: 0,
            reason: format!("no candidate of size {payload_size}"),
        },
        1 => PayloadTypeChoice {
            btf_type_id: size_matches[0].0,
            reason: String::new(),
        },
        n => {
            // Multiple size-match candidates; prefer conventional
            // names. Order is most-specific first so a struct named
            // exactly `task_ctx` wins over `foo_task_ctx`. If 2+
            // structs share the SAME pattern arm, we cannot pick
            // between them and fall back to hex.
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
                            btf_type_id: hits[0],
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
                btf_type_id: 0,
                reason: format!("ambiguous: {n} candidates"),
            }
        }
    }
}

/// Internal walker state. Bundles the read-only inputs the recursive
/// descent threads through every call so each function takes one
/// `&mut self` and the actual position arguments.
struct TreeWalker<'a> {
    kernel: &'a GuestKernel<'a>,
    kern_vm_start: u64,
    offsets: &'a SdtAllocOffsets,
    btf: &'a Btf,
    payload_btf_type_id: u32,
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
    /// recurse via `chunk->descs[pos]`; level 2 reads `chunk->data[pos]`
    /// and emits one [`SdtAllocEntry`] per allocated bit.
    ///
    /// Increments [`SdtAllocatorSnapshot::skipped_subtrees`] on every
    /// early return that abandons descent into a non-trivial subtree
    /// — translate failures, out-of-range `nr_free`, NULL `chunk`.
    fn descend(&mut self, desc_ptr: u64, level: usize) {
        if self.out.entries.len() >= MAX_SDT_ALLOC_ENTRIES {
            self.out.truncated = true;
            return;
        }
        if level >= SDT_TASK_LEVELS {
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

        // Walk each set bit in the bitmap.
        for (word_idx, &word_value) in allocated.iter().enumerate() {
            let mut word = word_value;
            while word != 0 {
                if self.out.entries.len() >= MAX_SDT_ALLOC_ENTRIES {
                    self.out.truncated = true;
                    return;
                }
                let bit = word.trailing_zeros() as usize;
                word &= word - 1;
                let pos = word_idx * 64 + bit;
                if pos >= SDT_TASK_ENTS_PER_CHUNK {
                    continue;
                }

                // chunk[pos] is a u64 pointer at chunk_pa +
                // chunk_union + pos * 8. Internal levels treat it as
                // a sdt_desc *; the leaf level treats it as a
                // sdt_data *.
                let entry_ptr_off = self.offsets.chunk_union + pos * 8;
                let entry_ptr = mem.read_u64(chunk_pa, entry_ptr_off);
                if entry_ptr == 0 {
                    // Pristine slot: bit was set but the chunk arm
                    // never got populated. The kernel allocator
                    // populates `chunk->data[pos]` after setting the
                    // bit (in `scx_alloc_internal`), so a snapshot
                    // captured between the bit set and the pointer
                    // store sees this transient state. Skip without
                    // counting as a skipped subtree — it's a
                    // legitimate transient.
                    continue;
                }

                if level == SDT_TASK_LEVELS - 1 {
                    self.emit_leaf(entry_ptr);
                } else {
                    self.descend(entry_ptr, level + 1);
                }
            }
        }
    }

    /// Emit one leaf allocation: read tid + payload, BTF-render.
    fn emit_leaf(&mut self, data_ptr: u64) {
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

        let payload = if self.payload_btf_type_id != 0 {
            render_value_with_mem(self.btf, self.payload_btf_type_id, &payload_bytes, self.mem)
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
        let pa =
            self.kernel
                .mem()
                .translate_kva(self.kernel.cr3_pa(), Kva(kva), self.kernel.l5())?;
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
        // elem_size, allocator_name, payload_btf_type_id, and
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
            payload_btf_type_id: 42,
            payload_type_reason: String::new(),
        };
        let json = serde_json::to_string(&snap).expect("serialize");
        let parsed: SdtAllocatorSnapshot = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.entries.len(), 1);
        assert_eq!(parsed.entries[0].idx, 7);
        assert_eq!(parsed.entries[0].genn, 1);
        assert_eq!(parsed.elem_size, 24);
        assert_eq!(parsed.payload_btf_type_id, 42);
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
            payload_btf_type_id: 0,
            payload_type_reason: String::new(),
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
            payload_btf_type_id: 0,
            payload_type_reason: "no candidate of size 16".into(),
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
            payload_btf_type_id: 42,
            payload_type_reason: String::new(),
        };
        let out = format!("{snap}");
        assert!(
            out.contains("sdt_alloc scx_task_allocator"),
            "missing header: {out}"
        );
        assert!(out.contains("elem_size=24"), "missing elem_size: {out}");
        assert!(out.contains("btf_type_id=42"), "missing btf_type_id: {out}");
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
            payload_btf_type_id: 0,
            payload_type_reason: "no candidate of size 16".into(),
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
        assert_eq!(choice.btf_type_id, 0, "zero-size must yield btf_type_id=0");
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
        assert_eq!(choice.btf_type_id, 0);
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
}
