//! Host-side `scx_static` bump-allocator walker.
//!
//! `scx_static` is the program-lifetime bump allocator declared in the
//! upstream scx tree at `lib/sdt_alloc.bpf.c` (line 577 — the
//! `struct scx_static scx_static;` global). It is distinct from the
//! per-instance `scx_allocator` walked by [`super::sdt_alloc`]: the
//! per-instance allocator hands out fixed-stride slots via a 3-level
//! radix tree with per-slot headers; `scx_static` hands out
//! variable-size, header-less allocations from a flat arena region by
//! advancing a single `off` pointer (see
//! `scx_static_alloc_internal` in `lib/sdt_alloc.bpf.c:580-657`).
//!
//! # Why a separate walker
//!
//! The renderer's deferred-resolve arena cast path
//! ([`crate::monitor::dump::render_map::resolve_arena_type_in_index`])
//! is backed by an [`crate::monitor::dump::render_map::ArenaTypeIndex`]
//! built from the per-instance sdt_alloc walk. That walk produces one
//! entry per allocator slot keyed on slot start. `scx_static` slots
//! have no per-slot metadata the host can recover at freeze time —
//! the bump allocator records nothing beyond `(memory, off)` for the
//! current backing region, and rolls forward to a fresh arena region
//! whenever it would overflow `max_alloc_bytes`. So a parallel walker
//! produces a coarser, region-granular index of the live-allocated
//! span, supporting:
//!
//! - "is this address in scx_static-allocated arena memory?"
//!   (membership check),
//! - the bookkeeping needed by the bridge to fail closed (return
//!   `None`) on a deferred-resolve chase whose target lives in
//!   `scx_static` memory but whose type cannot be recovered from
//!   cast-analysis evidence.
//!
//! # Type recovery
//!
//! The bump allocator has NO per-slot header — `scx_static_alloc_internal`
//! returns a raw `void __arena *` that the caller stores at a typed
//! site. To recover the per-allocation type at a given arena VA the
//! host needs evidence from cast analysis (which call site allocated
//! the bytes, and what BTF struct that site emitted). The cast
//! analyzer in [`crate::monitor::cast_analysis`] already tracks
//! `scx_static_alloc_internal` calls (see the
//! `ARENA_ALLOC_KFUNC_NAMES` allowlist at
//! `crate::vmm::cast_analysis_load:1034`) and emits a `CastHit` with
//! `target_type_id == 0` when shape inference is ambiguous — the
//! deferred-resolve path the renderer falls into. The analyzer does
//! NOT today emit a `(call_site_pc, expected_type_id, expected_size)`
//! map keyed by allocation site; without that map this walker cannot
//! recover the per-allocation type.
//!
//! This module therefore produces an UNTYPED [`ScxStaticRange`]
//! covering the live-allocated span of every `scx_static` instance.
//! When cast-analysis grows a per-call-site type hook in a future
//! revision, the walker can grow a typed companion index keyed on
//! observed call-site VAs without changing the membership-check
//! surface in this module.
//!
//! # Liveness model
//!
//! The bump allocator never frees individual allocations — it rolls
//! forward to a fresh `bpf_arena_alloc_pages` region when the current
//! `(memory, off + alloc_bytes)` would exceed `max_alloc_bytes` (see
//! `lib/sdt_alloc.bpf.c:609-649`). Old regions are abandoned in place
//! (the verifier comment at line 614-617 calls this out: "No free
//! operation so just forget about the previous allocation memory").
//! Only the CURRENT region is reachable through the live `memory`
//! pointer; old regions are unreachable arena pages whose pgoffs the
//! host snapshot would still capture but no live `__arena *` could
//! ever point into. The walker therefore indexes only
//! `[memory_low32, memory_low32 + off)` for the current region —
//! reading older regions is impossible without a separate audit
//! trail the kernel does not keep.
//!
//! # Race window
//!
//! The freeze coordinator pauses every vCPU before this walker runs,
//! so `scx_static.memory` and `scx_static.off` are stable. The
//! bump allocator updates `scx_static.off` AFTER deciding the
//! allocation will fit (see `lib/sdt_alloc.bpf.c:651-652`:
//! `ptr = (void __arena *)(addr + padding); scx_static.off += alloc_bytes;`),
//! so a frozen snapshot may observe a transient state where a caller
//! has just received a pointer but `off` has not yet been advanced.
//! In that case the just-allocated bytes lie at `memory + off_old`
//! and the walker reports a slightly-too-small range. Membership
//! checks are conservative on the small side: a chase against a
//! pointer just outside the reported range fails closed with
//! `is_arena_addr_in_scx_static_index` returning `false`, which is
//! the correct safe answer ("we cannot prove this address belongs
//! to scx_static memory").

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use btf_rs::Btf;
use serde::{Deserialize, Serialize};

use super::btf_offsets::{StructOrFwd, find_struct_or_fwd, member_byte_offset};
use super::reader::GuestMem;

/// Width of `struct scx_static` in bytes on every kernel that ships
/// the bump allocator.
///
/// The struct contains three pointer-/size_t-sized members (per
/// `lib/sdt_task_defs.h:104-108`):
///
/// ```text
/// struct scx_static {
///     size_t max_alloc_bytes;   // 8 bytes
///     void __arena *memory;     // 8 bytes
///     size_t off;               // 8 bytes
/// };
/// ```
///
/// Total: 24 bytes. Used as a sanity-cap on the .bss-slice the walker
/// reads — a far-larger size from BTF would indicate a header drift
/// and the walker bails closed (empty index) rather than reading
/// uninitialized bss bytes past the struct.
#[cfg(test)]
pub const SCX_STATIC_STRUCT_SIZE: usize = 24;

/// Sanity cap on `max_alloc_bytes` (per-region size) the walker will
/// trust.
///
/// `lib/sdt_alloc.bpf.c::scx_static_init` (line 660-678) takes
/// `alloc_pages` and computes `max_bytes = alloc_pages * PAGE_SIZE`,
/// then `bpf_arena_alloc_pages` allocates that many pages from the
/// arena. The arena window is at most 4 GiB
/// (`bpf_arena_map_alloc` clamps to `SZ_4G`), so any `max_alloc_bytes`
/// at or above 4 GiB is structurally impossible — a torn snapshot or
/// uninitialized struct could surface a wild value, and the walker
/// rejects anything at or above this bound.
const MAX_REASONABLE_REGION_BYTES: u64 = 1u64 << 32;

/// Sanity cap on `off` (high-water mark within a region) the walker
/// will trust. `off` is bounded above by `max_alloc_bytes` per the
/// bump allocator's overflow check at line 596-601 / 609. The walker
/// rejects `off > max_alloc_bytes` as a torn-snapshot signal.
///
/// Documented as a derived constant rather than a literal: the
/// authoritative bound is `max_alloc_bytes` from the same instance,
/// not a fixed number — different instances declare different region
/// sizes — so the function-level check uses the per-instance bound,
/// not this constant. The constant lives here only to surface the
/// upper-most reasonable value (the same 4 GiB cap as
/// [`MAX_REASONABLE_REGION_BYTES`]) for documentation parity.
#[allow(dead_code)]
const MAX_REASONABLE_OFF: u64 = MAX_REASONABLE_REGION_BYTES;

/// Byte offsets within `struct scx_static`, resolved from the
/// scheduler's program BTF.
///
/// All three fields are 8-byte members on every supported architecture
/// (`size_t` and pointer types both align to 8 bytes). Their offsets
/// in the struct could in principle change if upstream rearranges the
/// declaration; the walker re-resolves them at every dump pass so a
/// future reorder surfaces correctly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScxStaticOffsets {
    /// Offset of `max_alloc_bytes` (size_t) within `struct scx_static`.
    pub max_alloc_bytes: usize,
    /// Offset of `memory` (`void __arena *`) within `struct scx_static`.
    pub memory: usize,
    /// Offset of `off` (size_t) within `struct scx_static`.
    pub off: usize,
    /// Total declared size of `struct scx_static`. Used to bound the
    /// .bss-slice the walker reads. Equals 24 on every kernel that
    /// ships the upstream allocator.
    pub struct_size: usize,
}

impl ScxStaticOffsets {
    /// Resolve `struct scx_static` offsets from a pre-loaded program
    /// BTF.
    ///
    /// Returns `Err` when the program BTF lacks `struct scx_static` —
    /// e.g. a scheduler that doesn't link `lib/sdt_alloc.bpf.c`. The
    /// dump pipeline treats this as "no scx_static state to surface"
    /// and skips the walk silently.
    ///
    /// `struct scx_static` is a concrete type used directly by the
    /// allocator's init / alloc / free helpers, so it must surface as
    /// a full struct definition (not a `BTF_KIND_FWD`) — every member
    /// offset the walker reads is required and a forward declaration
    /// carries no member information. A Fwd surfaces as `Err` so the
    /// dump records a clear diagnostic instead of returning an
    /// unusable offsets struct.
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let scx_static = match find_struct_or_fwd(btf, "scx_static")
            .context("btf: struct scx_static not found (scheduler doesn't link sdt_alloc, or BTF only carries a forward declaration)")?
        {
            StructOrFwd::Full(s) => s,
            StructOrFwd::Fwd => anyhow::bail!(
                "btf: struct scx_static present only as BTF_KIND_FWD forward declaration; member offsets unavailable"
            ),
        };
        let max_alloc_bytes = member_byte_offset(btf, &scx_static, "max_alloc_bytes")?;
        let memory = member_byte_offset(btf, &scx_static, "memory")?;
        let off = member_byte_offset(btf, &scx_static, "off")?;
        let struct_size = scx_static.size();

        Ok(Self {
            max_alloc_bytes,
            memory,
            off,
            struct_size,
        })
    }
}

/// One live-allocated range surfaced from a single `scx_static`
/// instance.
///
/// Two fields capture the region:
///
/// - `start_low32` — the low 32 bits of the user-side arena address
///   of the region's first byte (`scx_static.memory`'s low 32 bits;
///   the high 32 bits are constant inside one arena window so the
///   low-32 keying matches the per-pass `ArenaTypeIndex` convention
///   in [`crate::monitor::dump::render_map::ArenaSlotInfo`]).
/// - `size` — the high-water mark `off`. Bytes in
///   `[start_low32, start_low32 + size)` are the live-allocated span;
///   bytes past `start_low32 + size` (within the same region's
///   `max_alloc_bytes`) are reserved-but-unallocated and a chase
///   against them must NOT report membership.
///
/// `instance_name` carries the .bss symbol the range was discovered
/// under (today the upstream scx tree declares one global,
/// `scx_static`, but the type leaves room for multiple instances).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub struct ScxStaticRange {
    /// .bss symbol name the range was read from
    /// (e.g. `"scx_static"`).
    pub instance_name: String,
    /// Low 32 bits of the region's user-side arena base address. See
    /// type doc for why low-32 is sufficient.
    pub start_low32: u32,
    /// High-water mark within the region in bytes (`scx_static.off`).
    /// Bytes in `[start_low32, start_low32 + size)` are live.
    pub size: u64,
    /// Region capacity in bytes (`scx_static.max_alloc_bytes`).
    /// Surfaced for diagnostic parity — operators reading the dump
    /// can see how full the region is by comparing `size` to
    /// `capacity`.
    pub capacity: u64,
}

/// All live `scx_static` ranges discovered in one freeze pass.
///
/// One [`ScxStaticRange`] per live allocator instance. The dump
/// pipeline today walks one instance (the upstream scx tree declares
/// only `scx_static`) but the type shape leaves room for multiple
/// instances to coexist.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub struct ScxStaticSnapshot {
    /// Live ranges, in tree-walk order. Empty when the walker found
    /// no `scx_static` instance in `.bss`, when every instance had
    /// `memory == 0` (allocator never initialised), or when every
    /// instance was rejected by the sanity caps.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ranges: Vec<ScxStaticRange>,
    /// Count of `scx_static` instances the walker located but
    /// rejected (e.g. `max_alloc_bytes` out of range, `off >
    /// max_alloc_bytes`, `memory == 0`). Always serialized so the
    /// dump consumer can distinguish "no instance found" from
    /// "instance found but unusable".
    pub skipped: u32,
}

impl ScxStaticSnapshot {
    /// True when the snapshot carries no live ranges and no skipped
    /// instances — equivalent to "scx_static walker did not run, or
    /// found nothing to surface". Used by the dump report's
    /// `skip_serializing_if` policy to keep the wire shape minimal
    /// for schedulers that don't link `lib/sdt_alloc.bpf.c`.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty() && self.skipped == 0
    }
}

impl std::fmt::Display for ScxStaticSnapshot {
    /// One line per range: `scx_static <name> [<lo>..<hi>) cap=<cap>`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.ranges.is_empty() && self.skipped == 0 {
            return write!(f, "scx_static: <none>");
        }
        for (i, r) in self.ranges.iter().enumerate() {
            if i > 0 {
                f.write_str("\n")?;
            }
            let end = (r.start_low32 as u64).saturating_add(r.size);
            write!(
                f,
                "scx_static {} [{:#x}..{:#x}) cap={}",
                r.instance_name, r.start_low32, end, r.capacity,
            )?;
        }
        if self.skipped > 0 {
            if !self.ranges.is_empty() {
                f.write_str("\n")?;
            }
            write!(f, "scx_static: {} instance(s) skipped", self.skipped)?;
        }
        Ok(())
    }
}

/// `start_low32 → size` lookup populated by the walker for every live
/// `scx_static` region.
///
/// Mirrors the `BTreeMap` keying convention of the per-instance
/// allocator's [`crate::monitor::dump::render_map::ArenaTypeIndex`]:
/// `start_low32` is the low 32 bits of the region's user-side base
/// address (4 GiB-alignment of the arena window keeps the high 32
/// bits constant), and `size` is the live-allocated span (`off`).
/// The lookup uses [`std::collections::BTreeMap::range`] to find the
/// range whose `[start_low32, start_low32 + size)` window contains
/// the chased address.
///
/// **Untyped**: this index reports membership only — the walker
/// cannot recover per-allocation BTF type ids from `scx_static`
/// memory (see the type-recovery section in the module-level doc).
/// The bridge consumer therefore answers "is this arena address
/// inside a live scx_static region?" with a boolean and treats the
/// per-type render as "not resolvable here, fall through to the
/// historical Fwd-skip behaviour or to cross-BTF resolution".
///
/// `BTreeMap` keeps the iteration order deterministic for tests
/// (matching the per-instance index's design choice) and bounds the
/// hot-path lookup at `O(log N)` against the small live-range count
/// (one entry per `scx_static` instance, today one).
pub type ScxStaticRangeIndex = BTreeMap<u32, u64>;

/// Build a [`ScxStaticRangeIndex`] from a [`ScxStaticSnapshot`].
///
/// The walker produces an ordered vector of ranges; the renderer
/// wants a `BTreeMap` keyed on `start_low32` for `O(log N)`
/// range-lookup against a chased address. Building the map once
/// outside the renderer (in [`crate::monitor::dump::dump_state`])
/// avoids a per-map rebuild on every `mem_reader` call.
///
/// Duplicate `start_low32` keys (two instances reporting the same
/// windowed start, indicating either a multi-instance setup that
/// happens to align both regions identically or a torn snapshot)
/// keep the FIRST entry seen and emit a `tracing::warn!` line so an
/// operator can diagnose the collision. The first-write-wins policy
/// matches the per-instance allocator index's collision policy in
/// [`crate::monitor::dump::render_map::append_arena_type_index_for_allocator`].
pub fn build_scx_static_range_index(snapshot: &ScxStaticSnapshot) -> ScxStaticRangeIndex {
    let mut index = ScxStaticRangeIndex::new();
    for range in &snapshot.ranges {
        match index.entry(range.start_low32) {
            std::collections::btree_map::Entry::Vacant(v) => {
                v.insert(range.size);
            }
            std::collections::btree_map::Entry::Occupied(o) => {
                tracing::warn!(
                    start_low32 = format_args!("{:#x}", range.start_low32),
                    first_size = *o.get(),
                    duplicate_size = range.size,
                    instance = %range.instance_name,
                    "scx_static index has duplicate start_low32 (multi-instance \
                     low-32 collision or torn snapshot); keeping first range",
                );
            }
        }
    }
    index
}

/// Range lookup: does `addr` (a chased arena address) fall within any
/// live `scx_static` region in `index`?
///
/// `addr` is a full 64-bit arena address; the helper masks with
/// `0xFFFF_FFFF` to match the index's low-32 keying. The lookup uses
/// [`std::collections::BTreeMap::range`] to find the range entry
/// whose `[start_low32, start_low32 + size)` window contains the
/// masked key.
///
/// Returns `false` when `index` is empty, when the address falls
/// outside every range, or when the masked address would land at
/// `start + size` exactly (the bound is `<`, not `<=`, mirroring
/// the per-instance allocator index's slot-end-excluded convention
/// in [`crate::monitor::dump::render_map::resolve_arena_type_in_index`]).
///
/// The function is gate-only: it does NOT consult an arena snapshot
/// to validate that `addr` lives in the arena window. Callers that
/// need that guarantee compose this helper with
/// [`crate::monitor::dump::render_map::is_arena_addr_in_snapshot`].
pub fn is_arena_addr_in_scx_static_index(index: &ScxStaticRangeIndex, addr: u64) -> bool {
    if index.is_empty() {
        return false;
    }
    let key = (addr & 0xFFFF_FFFF) as u32;
    let Some((&start, &size)) = index.range(..=key).next_back() else {
        return false;
    };
    let end = (start as u64) + size;
    (key as u64) < end
}

/// Walk every `scx_static` instance declared in `bss_bytes` and
/// surface its live-allocated range.
///
/// `bss_bytes` is the raw .bss image — the same buffer the per-instance
/// allocator walker consumes. `var_offset_provider` yields one
/// `(var_name, var_offset, var_type_id)` triple per `.bss` Var; the
/// caller is expected to filter to vars whose type is `struct
/// scx_static` (today the upstream scx tree declares one global of
/// this type, named `scx_static`, but the walker handles N).
/// `is_scx_static_var` decides which vars to include.
///
/// `offsets` carries the resolved struct member offsets from
/// [`ScxStaticOffsets::from_btf`].
///
/// The walk is best-effort: a sanity-rejected instance increments
/// [`ScxStaticSnapshot::skipped`] and skips to the next, so a single
/// torn-read can't truncate the whole walk.
pub fn walk_scx_static<I, F>(
    bss_bytes: &[u8],
    offsets: &ScxStaticOffsets,
    vars: I,
    is_scx_static_var: F,
) -> ScxStaticSnapshot
where
    I: IntoIterator<Item = (String, usize, u32)>,
    F: Fn(u32) -> bool,
{
    let mut snap = ScxStaticSnapshot::default();
    for (name, var_offset, type_id) in vars {
        if !is_scx_static_var(type_id) {
            continue;
        }
        match read_one_scx_static(bss_bytes, var_offset, offsets) {
            Some((memory_low32, off, capacity)) => {
                snap.ranges.push(ScxStaticRange {
                    instance_name: name,
                    start_low32: memory_low32,
                    size: off,
                    capacity,
                });
            }
            None => {
                snap.skipped = snap.skipped.saturating_add(1);
            }
        }
    }
    snap
}

/// Read one `scx_static` instance from the .bss-slice and return the
/// triple `(memory_low32, off, max_alloc_bytes)` the walker keys on,
/// or `None` when the instance is sanity-rejected.
///
/// Sanity gates (in order):
///
/// 1. The .bss-slice must be at least `var_offset + struct_size`
///    bytes; out-of-range slice → reject.
/// 2. `max_alloc_bytes` must be in `(0, MAX_REASONABLE_REGION_BYTES)`;
///    zero or near-4-GiB → reject (uninitialized or torn).
/// 3. `memory == 0` → reject (`scx_static_init` has not run).
/// 4. `off > max_alloc_bytes` → reject (impossible in the
///    bump allocator's invariant — the overflow check at
///    `lib/sdt_alloc.bpf.c:609` rejects allocations that would
///    cross `max_alloc_bytes`, so a torn snapshot is the only way
///    `off` can exceed the cap).
fn read_one_scx_static(
    bss_bytes: &[u8],
    var_offset: usize,
    offsets: &ScxStaticOffsets,
) -> Option<(u32, u64, u64)> {
    // Slice the in-bss bytes for one full `struct scx_static`. We use
    // the BTF-reported size so a future field appended to scx_static
    // surfaces correctly via the struct-size grow rather than reading
    // past the slice end.
    let slice_end = var_offset.checked_add(offsets.struct_size)?;
    let slice = bss_bytes.get(var_offset..slice_end)?;

    let max_alloc_bytes = read_u64_at(slice, offsets.max_alloc_bytes)?;
    if max_alloc_bytes == 0 || max_alloc_bytes >= MAX_REASONABLE_REGION_BYTES {
        return None;
    }
    let memory = read_u64_at(slice, offsets.memory)?;
    if memory == 0 {
        return None;
    }
    let off = read_u64_at(slice, offsets.off)?;
    if off > max_alloc_bytes {
        return None;
    }
    // The arena window is at most 4 GiB, so the low 32 bits of the
    // memory pointer are sufficient to address any byte inside the
    // region. Mirrors the masking convention in
    // `crate::monitor::sdt_alloc::TreeWalker::emit_leaf` (see its
    // `data_ptr & 0xFFFF_FFFF` comment).
    let memory_low32 = memory as u32;
    Some((memory_low32, off, max_alloc_bytes))
}

/// Read a u64 at `offset` from a byte slice, returning `None` when
/// the read would overflow the slice. Little-endian to match the
/// in-bss layout of the kernel struct.
///
/// Mirrors the `read_u64_at` helper in
/// [`super::sdt_alloc::read_u64_at`] (private there; duplicated here
/// to keep the modules independently testable). A future refactor
/// could lift both into a shared helper if the duplication grows.
fn read_u64_at(bytes: &[u8], offset: usize) -> Option<u64> {
    let end = offset.checked_add(8)?;
    let slice = bytes.get(offset..end)?;
    let mut buf = [0u8; 8];
    buf.copy_from_slice(slice);
    Some(u64::from_le_bytes(buf))
}

/// Read a `struct scx_static` instance's tuple direct from guest
/// memory via a [`GuestMem`] reader.
///
/// Convenience wrapper for callers that have a full guest physical
/// address rather than a pre-read .bss slice. Returns the same
/// `(memory_low32, off, max_alloc_bytes)` triple as
/// [`read_one_scx_static`] applies the same sanity gates.
///
/// `instance_pa` is the guest physical address of the
/// `struct scx_static` instance's first byte. Callers typically
/// resolve this via the kernel's direct-mapping translation (the
/// .bss is in the direct map) but the walker accepts any PA the
/// caller supplies.
#[allow(dead_code)]
pub fn read_scx_static_from_pa(
    mem: &GuestMem,
    instance_pa: u64,
    offsets: &ScxStaticOffsets,
) -> Option<(u32, u64, u64)> {
    let max_alloc_bytes = mem.read_u64(instance_pa, offsets.max_alloc_bytes);
    if max_alloc_bytes == 0 || max_alloc_bytes >= MAX_REASONABLE_REGION_BYTES {
        return None;
    }
    let memory = mem.read_u64(instance_pa, offsets.memory);
    if memory == 0 {
        return None;
    }
    let off = mem.read_u64(instance_pa, offsets.off);
    if off > max_alloc_bytes {
        return None;
    }
    Some((memory as u32, off, max_alloc_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compose a synthetic `struct scx_static` byte image for a single
    /// instance at the given offsets. Used by tests to drive the
    /// walker without needing a full guest VM.
    fn synth_scx_static(
        offsets: &ScxStaticOffsets,
        max_alloc_bytes: u64,
        memory: u64,
        off: u64,
    ) -> Vec<u8> {
        let mut bytes = vec![0u8; offsets.struct_size];
        bytes[offsets.max_alloc_bytes..offsets.max_alloc_bytes + 8]
            .copy_from_slice(&max_alloc_bytes.to_le_bytes());
        bytes[offsets.memory..offsets.memory + 8].copy_from_slice(&memory.to_le_bytes());
        bytes[offsets.off..offsets.off + 8].copy_from_slice(&off.to_le_bytes());
        bytes
    }

    /// Default test offsets matching the upstream layout in
    /// `lib/sdt_task_defs.h:104-108`. Pinning the values here makes
    /// every other test in this module read against the same
    /// canonical shape; a divergence between this fixture and the
    /// production resolver would surface as a member-offset mismatch
    /// in the live walker, not a test failure here.
    fn default_offsets() -> ScxStaticOffsets {
        ScxStaticOffsets {
            max_alloc_bytes: 0,
            memory: 8,
            off: 16,
            struct_size: SCX_STATIC_STRUCT_SIZE,
        }
    }

    #[test]
    fn read_u64_at_basic() {
        let bytes = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0xff];
        // LE: 0x0807060504030201.
        assert_eq!(read_u64_at(&bytes, 0), Some(0x0807060504030201));
        assert_eq!(read_u64_at(&bytes, 2), None);
        assert_eq!(read_u64_at(&bytes, 100), None);
    }

    #[test]
    fn read_u64_at_handles_offset_overflow() {
        let bytes = [0u8; 16];
        assert_eq!(read_u64_at(&bytes, usize::MAX), None);
    }

    /// Pin the canonical struct width: any drift from the upstream
    /// layout would surface as a behaviour-changing mismatch between
    /// the walker's slice-end check and the resolved `struct_size`.
    #[test]
    fn struct_size_matches_upstream_layout() {
        // Three 8-byte members.
        assert_eq!(SCX_STATIC_STRUCT_SIZE, 24);
    }

    /// Empty bss → empty snapshot, zero skipped. Pins the zero-input
    /// shape so a future regression where the walker spuriously emits
    /// a placeholder range surfaces here.
    #[test]
    fn walk_scx_static_empty_input() {
        let offsets = default_offsets();
        let snap = walk_scx_static(&[], &offsets, std::iter::empty(), |_| true);
        assert_eq!(snap.ranges.len(), 0);
        assert_eq!(snap.skipped, 0);
    }

    /// Var filter rejects every type → empty snapshot, zero skipped.
    /// The filter rejects upstream of the .bss-slice read so non-
    /// matching instances don't even count as "skipped" — they were
    /// never considered. Distinguishing these in
    /// [`ScxStaticSnapshot::skipped`] is important: `skipped` reports
    /// "instance found but unusable", not "instance not searched".
    #[test]
    fn walk_scx_static_filter_rejects_all() {
        let offsets = default_offsets();
        let bytes = synth_scx_static(&offsets, 4096, 0xDEAD_BEEF, 100);
        let snap = walk_scx_static(
            &bytes,
            &offsets,
            std::iter::once(("not_scx_static".to_string(), 0, 42)),
            |_| false,
        );
        assert_eq!(snap.ranges.len(), 0);
        assert_eq!(snap.skipped, 0);
    }

    /// Happy path: one well-formed instance produces one range with
    /// the right low-32 start, size, and capacity.
    #[test]
    fn walk_scx_static_single_instance_happy_path() {
        let offsets = default_offsets();
        let bytes = synth_scx_static(&offsets, 4096, 0x1234_5678_DEAD_BEEF, 100);
        let snap = walk_scx_static(
            &bytes,
            &offsets,
            std::iter::once(("scx_static".to_string(), 0, 1)),
            |_| true,
        );
        assert_eq!(snap.ranges.len(), 1);
        assert_eq!(snap.skipped, 0);
        assert_eq!(snap.ranges[0].start_low32, 0xDEAD_BEEF);
        assert_eq!(snap.ranges[0].size, 100);
        assert_eq!(snap.ranges[0].capacity, 4096);
        assert_eq!(snap.ranges[0].instance_name, "scx_static");
    }

    /// `memory == 0` (allocator never initialised) → instance is
    /// rejected by the sanity gate; `skipped` increments. The walker
    /// must not emit a range for an uninitialised instance — a
    /// `start_low32 == 0` range would falsely cover NULL pointer
    /// chases.
    #[test]
    fn walk_scx_static_rejects_zero_memory() {
        let offsets = default_offsets();
        let bytes = synth_scx_static(&offsets, 4096, 0, 100);
        let snap = walk_scx_static(
            &bytes,
            &offsets,
            std::iter::once(("scx_static".to_string(), 0, 1)),
            |_| true,
        );
        assert_eq!(snap.ranges.len(), 0);
        assert_eq!(snap.skipped, 1);
    }

    /// `max_alloc_bytes == 0` → reject. Zero capacity is the
    /// uninitialised state; emitting a range against it would
    /// shadow the invariant that any allocation is impossible.
    #[test]
    fn walk_scx_static_rejects_zero_capacity() {
        let offsets = default_offsets();
        let bytes = synth_scx_static(&offsets, 0, 0xDEAD_BEEF, 0);
        let snap = walk_scx_static(
            &bytes,
            &offsets,
            std::iter::once(("scx_static".to_string(), 0, 1)),
            |_| true,
        );
        assert_eq!(snap.ranges.len(), 0);
        assert_eq!(snap.skipped, 1);
    }

    /// `max_alloc_bytes >= 4 GiB` → reject. A near-4-GiB value is
    /// structurally impossible (the arena window is capped at 4 GiB
    /// by `bpf_arena_map_alloc::SZ_4G`); pinning the rejection so a
    /// future drift to a wider arena window must explicitly raise
    /// the cap rather than silently accept the wild value.
    #[test]
    fn walk_scx_static_rejects_oversized_capacity() {
        let offsets = default_offsets();
        let bytes = synth_scx_static(&offsets, MAX_REASONABLE_REGION_BYTES, 0xDEAD_BEEF, 0);
        let snap = walk_scx_static(
            &bytes,
            &offsets,
            std::iter::once(("scx_static".to_string(), 0, 1)),
            |_| true,
        );
        assert_eq!(snap.ranges.len(), 0);
        assert_eq!(snap.skipped, 1);
    }

    /// `off > max_alloc_bytes` → reject. The bump allocator's
    /// overflow check at `lib/sdt_alloc.bpf.c:609` makes this
    /// invariant impossible at quiescence; a violating snapshot is
    /// torn or corrupt and must fail closed rather than report a
    /// range past the region end.
    #[test]
    fn walk_scx_static_rejects_off_past_capacity() {
        let offsets = default_offsets();
        let bytes = synth_scx_static(&offsets, 4096, 0xDEAD_BEEF, 8192);
        let snap = walk_scx_static(
            &bytes,
            &offsets,
            std::iter::once(("scx_static".to_string(), 0, 1)),
            |_| true,
        );
        assert_eq!(snap.ranges.len(), 0);
        assert_eq!(snap.skipped, 1);
    }

    /// Fully-allocated region: `off == max_alloc_bytes` is the
    /// boundary case the gate must accept. Pinning the inclusive
    /// upper bound (`<=`) keeps a fully-used region observable.
    #[test]
    fn walk_scx_static_accepts_off_eq_capacity() {
        let offsets = default_offsets();
        let bytes = synth_scx_static(&offsets, 4096, 0xDEAD_BEEF, 4096);
        let snap = walk_scx_static(
            &bytes,
            &offsets,
            std::iter::once(("scx_static".to_string(), 0, 1)),
            |_| true,
        );
        assert_eq!(snap.ranges.len(), 1);
        assert_eq!(snap.ranges[0].size, 4096);
    }

    /// `off == 0` (region allocated but never used) → emit the
    /// range with size 0. Membership checks against any address in
    /// the region's reserved-but-unallocated tail must return
    /// `false` because the high-water mark is still 0.
    #[test]
    fn walk_scx_static_accepts_zero_off() {
        let offsets = default_offsets();
        let bytes = synth_scx_static(&offsets, 4096, 0xDEAD_BEEF, 0);
        let snap = walk_scx_static(
            &bytes,
            &offsets,
            std::iter::once(("scx_static".to_string(), 0, 1)),
            |_| true,
        );
        assert_eq!(snap.ranges.len(), 1);
        assert_eq!(snap.ranges[0].size, 0);
        // Build the index and verify membership behaviour.
        let index = build_scx_static_range_index(&snap);
        // Zero-size range: every address is past start + size = start + 0.
        assert!(!is_arena_addr_in_scx_static_index(&index, 0xDEAD_BEEF));
    }

    /// Fully-consumed region (`off > 0 && capacity == off`) walks
    /// without panic, the surfaced range covers exactly the
    /// allocated span, and the index admits no entries past the
    /// fully-allocated endpoint. Distinct from
    /// [`walk_scx_static_accepts_off_eq_capacity`], which only pins
    /// emission shape — this test composes walk + index + boundary
    /// to verify the end-of-region semantics: the address at
    /// `start + capacity` (== `start + size`, the byte just past the
    /// last live byte) must NOT match. Pins the invariant that the
    /// reserved-but-unallocatable tail of a fully-consumed region —
    /// i.e. nothing, since the region is full — does not falsely
    /// admit chases against the immediately-following byte.
    #[test]
    fn walk_scx_static_fully_consumed_excludes_endpoint() {
        let offsets = default_offsets();
        // off > 0 && capacity == off (fully consumed).
        let bytes = synth_scx_static(&offsets, 4096, 0xDEAD_BEEF, 4096);
        let snap = walk_scx_static(
            &bytes,
            &offsets,
            std::iter::once(("scx_static".to_string(), 0, 1)),
            |_| true,
        );
        assert_eq!(snap.ranges.len(), 1);
        assert_eq!(snap.skipped, 0);
        assert_eq!(snap.ranges[0].size, 4096);
        assert_eq!(snap.ranges[0].capacity, 4096);
        let index = build_scx_static_range_index(&snap);
        // Last live byte (start + size - 1) is IN.
        assert!(is_arena_addr_in_scx_static_index(
            &index,
            0xDEAD_BEEF_u64 + 4095
        ));
        // start + size (== start + capacity) is OUT — no entries
        // past the fully-consumed span.
        assert!(!is_arena_addr_in_scx_static_index(
            &index,
            0xDEAD_BEEF_u64 + 4096
        ));
    }

    /// .bss-slice too short to hold a full `struct scx_static` →
    /// reject. The walker must not read past the slice end (which
    /// would wrap into an unrelated var). `read_u64_at`'s slice
    /// bound is the gate.
    #[test]
    fn walk_scx_static_rejects_short_slice() {
        let offsets = default_offsets();
        // Slice has only 16 bytes; struct needs 24.
        let bytes = vec![0u8; 16];
        let snap = walk_scx_static(
            &bytes,
            &offsets,
            std::iter::once(("scx_static".to_string(), 0, 1)),
            |_| true,
        );
        assert_eq!(snap.ranges.len(), 0);
        assert_eq!(snap.skipped, 1);
    }

    /// `var_offset + struct_size` overflows or runs past the slice
    /// end → reject. Pins the `checked_add` overflow path against a
    /// regression that might silently treat a near-`usize::MAX`
    /// var_offset as zero.
    #[test]
    fn walk_scx_static_rejects_overflow_offset() {
        let offsets = default_offsets();
        let bytes = vec![0u8; 1024];
        // var_offset right at the edge of the slice; var would
        // extend past the end.
        let snap = walk_scx_static(
            &bytes,
            &offsets,
            std::iter::once(("scx_static".to_string(), 1010, 1)),
            |_| true,
        );
        assert_eq!(snap.ranges.len(), 0);
        assert_eq!(snap.skipped, 1);
    }

    /// Two instances at distinct .bss offsets → both surface in the
    /// snapshot. The walker iterates the var iterator linearly, so
    /// the test fixture pins ordering.
    #[test]
    fn walk_scx_static_multiple_instances() {
        let offsets = default_offsets();
        let mut bytes = vec![0u8; 64];
        // Instance A at offset 0.
        bytes[0..24].copy_from_slice(&synth_scx_static(&offsets, 4096, 0xAAAA_AAAA, 100));
        // Instance B at offset 32 (gap to keep this test honest).
        bytes[32..56].copy_from_slice(&synth_scx_static(&offsets, 8192, 0xBBBB_BBBB, 200));
        let vars = vec![
            ("scx_static_a".to_string(), 0, 1),
            ("scx_static_b".to_string(), 32, 1),
        ];
        let snap = walk_scx_static(&bytes, &offsets, vars, |_| true);
        assert_eq!(snap.ranges.len(), 2);
        assert_eq!(snap.skipped, 0);
        assert_eq!(snap.ranges[0].start_low32, 0xAAAA_AAAA);
        assert_eq!(snap.ranges[0].size, 100);
        assert_eq!(snap.ranges[1].start_low32, 0xBBBB_BBBB);
        assert_eq!(snap.ranges[1].size, 200);
    }

    /// Determinism: two walks of the same byte image produce
    /// identical snapshots. This is the explicit completion-criterion
    /// from the task spec ("walking the same arena state twice
    /// produces identical results").
    #[test]
    fn walk_scx_static_is_deterministic() {
        let offsets = default_offsets();
        let bytes = synth_scx_static(&offsets, 4096, 0xDEAD_BEEF, 100);
        let snap_a = walk_scx_static(
            &bytes,
            &offsets,
            vec![("scx_static".to_string(), 0, 1)],
            |_| true,
        );
        let snap_b = walk_scx_static(
            &bytes,
            &offsets,
            vec![("scx_static".to_string(), 0, 1)],
            |_| true,
        );
        assert_eq!(snap_a, snap_b);
    }

    // -- ScxStaticRangeIndex / membership ----------------------------

    /// Empty index → every membership check returns false. Pins the
    /// short-circuit so a default-empty index doesn't accidentally
    /// match on a chased address whose low 32 bits would map below
    /// the smallest valid key.
    #[test]
    fn is_arena_addr_in_scx_static_index_empty_returns_false() {
        let index = ScxStaticRangeIndex::new();
        assert!(!is_arena_addr_in_scx_static_index(&index, 0));
        assert!(!is_arena_addr_in_scx_static_index(&index, 0xDEAD_BEEF));
        assert!(!is_arena_addr_in_scx_static_index(&index, u64::MAX));
    }

    /// Address inside a single live range → `true`.
    #[test]
    fn is_arena_addr_in_scx_static_index_inside_range_returns_true() {
        let snap = ScxStaticSnapshot {
            ranges: vec![ScxStaticRange {
                instance_name: "scx_static".into(),
                start_low32: 0x1000,
                size: 100,
                capacity: 4096,
            }],
            skipped: 0,
        };
        let index = build_scx_static_range_index(&snap);
        // Start of range.
        assert!(is_arena_addr_in_scx_static_index(&index, 0x1000));
        // Last byte inside range (start + size - 1 = 0x1063).
        assert!(is_arena_addr_in_scx_static_index(&index, 0x1063));
        // Same address with high bits — the helper masks with
        // 0xFFFF_FFFF, so the high 32 bits are ignored.
        assert!(is_arena_addr_in_scx_static_index(
            &index,
            0xDEAD_BEEF_0000_1000
        ));
    }

    /// Boundary excluded: `start + size` is OUT of range. Pins the
    /// `<` (not `<=`) bound check that mirrors the per-instance
    /// allocator index's slot-end-excluded convention.
    #[test]
    fn is_arena_addr_in_scx_static_index_boundary_excluded() {
        let snap = ScxStaticSnapshot {
            ranges: vec![ScxStaticRange {
                instance_name: "scx_static".into(),
                start_low32: 0x1000,
                size: 100,
                capacity: 4096,
            }],
            skipped: 0,
        };
        let index = build_scx_static_range_index(&snap);
        // Exactly start + size.
        assert!(!is_arena_addr_in_scx_static_index(&index, 0x1064));
    }

    /// Address before any range → `false`. The
    /// `range(..=key).next_back()` lookup returns `None` when the
    /// chased key is below every entry; the helper must not panic.
    #[test]
    fn is_arena_addr_in_scx_static_index_below_range_returns_false() {
        let snap = ScxStaticSnapshot {
            ranges: vec![ScxStaticRange {
                instance_name: "scx_static".into(),
                start_low32: 0x1000,
                size: 100,
                capacity: 4096,
            }],
            skipped: 0,
        };
        let index = build_scx_static_range_index(&snap);
        assert!(!is_arena_addr_in_scx_static_index(&index, 0x0FFF));
        assert!(!is_arena_addr_in_scx_static_index(&index, 0));
    }

    /// Boundary at `start + size` excluded — second fixture pinning
    /// the same `<`-not-`<=` bound across a different range geometry
    /// (larger start, larger size) than
    /// [`is_arena_addr_in_scx_static_index_boundary_excluded`]. Two
    /// fixtures protect against a regression that might happen to
    /// pass the small-fixture test through accidental wrap or
    /// off-by-one luck.
    #[test]
    fn is_arena_addr_in_scx_static_index_boundary_at_start_plus_size_excluded() {
        let snap = ScxStaticSnapshot {
            ranges: vec![ScxStaticRange {
                instance_name: "scx_static".into(),
                start_low32: 0x10_0000,
                size: 0x4000,
                capacity: 0x4000,
            }],
            skipped: 0,
        };
        let index = build_scx_static_range_index(&snap);
        // Last live byte is IN.
        assert!(is_arena_addr_in_scx_static_index(&index, 0x10_3FFF));
        // start + size is OUT.
        assert!(!is_arena_addr_in_scx_static_index(&index, 0x10_4000));
    }

    /// Address at `start - 1` (one byte below range start) returns
    /// `false`. Second fixture (larger start than
    /// [`is_arena_addr_in_scx_static_index_below_range_returns_false`])
    /// pinning the lower-bound rejection across a different range
    /// geometry. The `range(..=key).next_back()` lookup must yield
    /// `None` for any key below every entry, and the helper must
    /// return `false` rather than panic.
    #[test]
    fn is_arena_addr_in_scx_static_index_just_below_start_returns_false() {
        let snap = ScxStaticSnapshot {
            ranges: vec![ScxStaticRange {
                instance_name: "scx_static".into(),
                start_low32: 0x10_0000,
                size: 0x4000,
                capacity: 0x4000,
            }],
            skipped: 0,
        };
        let index = build_scx_static_range_index(&snap);
        // start - 1 is OUT.
        assert!(!is_arena_addr_in_scx_static_index(&index, 0x0F_FFFF));
        // start is IN (sanity: confirms the boundary is on the
        // correct side of the rejection).
        assert!(is_arena_addr_in_scx_static_index(&index, 0x10_0000));
    }

    /// Range search across multiple ranges picks the range whose
    /// `[start, start + size)` contains the chased address.
    #[test]
    fn is_arena_addr_in_scx_static_index_picks_correct_range() {
        let snap = ScxStaticSnapshot {
            ranges: vec![
                ScxStaticRange {
                    instance_name: "scx_static_a".into(),
                    start_low32: 0x1000,
                    size: 100,
                    capacity: 4096,
                },
                ScxStaticRange {
                    instance_name: "scx_static_b".into(),
                    start_low32: 0x2000,
                    size: 100,
                    capacity: 4096,
                },
            ],
            skipped: 0,
        };
        let index = build_scx_static_range_index(&snap);
        // Inside range A.
        assert!(is_arena_addr_in_scx_static_index(&index, 0x1050));
        // Between ranges.
        assert!(!is_arena_addr_in_scx_static_index(&index, 0x1500));
        // Inside range B.
        assert!(is_arena_addr_in_scx_static_index(&index, 0x2050));
        // Past every range.
        assert!(!is_arena_addr_in_scx_static_index(&index, 0x3000));
    }

    /// Build_scx_static_range_index handles duplicate start_low32:
    /// keeps the first entry, emits a warn, second entry's size is
    /// ignored. Pins the first-write-wins policy.
    #[test]
    fn build_scx_static_range_index_keeps_first_on_duplicate_start() {
        let snap = ScxStaticSnapshot {
            ranges: vec![
                ScxStaticRange {
                    instance_name: "scx_static_a".into(),
                    start_low32: 0x1000,
                    size: 100,
                    capacity: 4096,
                },
                ScxStaticRange {
                    instance_name: "scx_static_b".into(),
                    start_low32: 0x1000, // same key
                    size: 200,
                    capacity: 4096,
                },
            ],
            skipped: 0,
        };
        let index = build_scx_static_range_index(&snap);
        assert_eq!(index.get(&0x1000), Some(&100));
    }

    /// Snapshot serde: empty default snapshot serialises minimally
    /// (skipped is always emitted; ranges is skipped when empty).
    #[test]
    fn snapshot_empty_serde_roundtrip() {
        let snap = ScxStaticSnapshot::default();
        let json = serde_json::to_string(&snap).unwrap();
        // ranges skipped on empty default.
        assert!(!json.contains("\"ranges\""));
        // skipped always emitted (zero value carries diagnostic info).
        assert!(json.contains("\"skipped\":0"));
        let parsed: ScxStaticSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, snap);
    }

    /// Snapshot serde: populated snapshot roundtrips cleanly with
    /// every field preserved.
    #[test]
    fn snapshot_populated_serde_roundtrip() {
        let snap = ScxStaticSnapshot {
            ranges: vec![ScxStaticRange {
                instance_name: "scx_static".into(),
                start_low32: 0x1000,
                size: 100,
                capacity: 4096,
            }],
            skipped: 1,
        };
        let json = serde_json::to_string(&snap).unwrap();
        let parsed: ScxStaticSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, snap);
    }

    /// Display formatting: empty snapshot → `<none>`. Populated →
    /// one line per range, range bounds in hex, capacity decimal.
    #[test]
    fn snapshot_display_formats() {
        let empty = ScxStaticSnapshot::default();
        assert_eq!(format!("{empty}"), "scx_static: <none>");

        let one = ScxStaticSnapshot {
            ranges: vec![ScxStaticRange {
                instance_name: "scx_static".into(),
                start_low32: 0x1000,
                size: 100,
                capacity: 4096,
            }],
            skipped: 0,
        };
        let s = format!("{one}");
        assert!(s.contains("scx_static"));
        assert!(s.contains("0x1000"));
        // 0x1000 + 100 = 0x1064.
        assert!(s.contains("0x1064"));
        assert!(s.contains("cap=4096"));

        let with_skipped = ScxStaticSnapshot {
            ranges: vec![],
            skipped: 3,
        };
        let s = format!("{with_skipped}");
        assert!(s.contains("3 instance(s) skipped"));
    }

    // -- ScxStaticOffsets::from_btf ---------------------------------

    /// `from_btf` returns Err when `struct scx_static` is absent —
    /// vmlinux BTF never contains it (scx_static lives in the
    /// scheduler's program BTF, not the kernel's), so a from_btf
    /// call against vmlinux must surface the expected error and not
    /// panic. The dump pipeline reads this Err to decide "no
    /// scx_static state to surface."
    #[test]
    fn from_btf_against_vmlinux_returns_err() {
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
        let err = ScxStaticOffsets::from_btf(&btf)
            .expect_err("vmlinux BTF must NOT contain scx_static — from_btf must Err");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("scx_static"),
            "error must name the missing struct so the dump pipeline can log a useful diagnostic: '{msg}'"
        );
    }
}
