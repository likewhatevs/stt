//! Host-side BPF arena page enumeration.
//!
//! `BPF_MAP_TYPE_ARENA` (kernel uapi value [`BPF_MAP_TYPE_ARENA`]) is
//! a sparse, page-granular memory region shared between BPF programs
//! and userspace. The kernel allocates a 4 GiB-plus-guard
//! (`KERN_VM_SZ`) `vm_struct` and lazily maps order-0 pages into it
//! on demand (see `kernel/bpf/arena.c::arena_alloc_pages` and
//! `arena_vm_fault`); the user-visible window is at
//! `[arena.user_vm_start .. arena.user_vm_end)`, a 32-bit-addressable
//! range whose lower 32 bits the BPF JIT uses as the arena pointer
//! payload. Translation kernel-side is:
//!
//! ```text
//! kern_vm_start = arena->kern_vm->addr + GUARD_SZ/2
//! kaddr         = kern_vm_start + (u32)user_addr
//! page          = vmalloc_to_page(kaddr)   // PTE walk on init_mm
//! ```
//!
//! The host-side walker mirrors this: read the arena's `kern_vm`
//! pointer, dereference to get `vm_struct.addr`, add `GUARD_SZ/2`,
//! then for each pgoff in `0..max_entries` compute `kaddr` and run
//! [`GuestMem::translate_kva`] (the existing PTE walker against
//! `init_mm`'s page table). `max_entries` is the BPF map's declared
//! page capacity from `bpf_map_create()` — it is the source of truth
//! for "how many pages this arena could hold", regardless of whether
//! the scheduler exposes a userspace mmap (some don't, leaving
//! `user_vm_start == user_vm_end == 0`). Pages whose translate fails
//! are simply "not faulted in" — arena maps are sparse by design.
//!
//! The walker does NOT consult `arena->rt` (the range_tree of free
//! pgoffs) — `range_tree` polarity is "set = free" / "clear =
//! allocated", reading it from a frozen snapshot would only tell
//! the host which pages the kernel *intended* to be allocated, not
//! which are actually mapped. The PTE walk is the source of truth.
//!
//! [`BPF_MAP_TYPE_ARENA`]: BPF_MAP_TYPE_ARENA

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

use btf_rs::Btf;

use super::Kva;
use super::bpf_map::{BPF_MAP_TYPE_ARENA, BpfMapInfo};
use super::btf_offsets::{find_struct, load_btf_from_path, member_byte_offset};
use super::guest::GuestKernel;

/// Page size used by the arena walker, derived from the GUEST
/// kernel's MMU configuration.
///
/// `arena_alloc_pages` and `arena_vm_fault` both call
/// `apply_to_page_range` on `PAGE_SIZE`-granular ranges where
/// `PAGE_SIZE` is the GUEST kernel's own MMU page size. The host's
/// page size is irrelevant — ktstr can run a 16 KiB-granule guest
/// on a 4 KiB-granule host (and vice versa), and the arena layout
/// must match the guest's view.
///
/// On x86_64 the guest page granule is fixed at 4 KiB. On aarch64
/// the granule is encoded in `TCR_EL1.TG1` (bits [31:30]):
///   - `0b10` → 4 KiB
///   - `0b01` → 16 KiB
///   - `0b11` → 64 KiB
///
/// Falls back to 4 KiB when the architecture branches reject the
/// register value (e.g. uninitialized `tcr_el1 == 0` on aarch64);
/// the fallback is conservative — at worst the walker overscans a
/// small arena and surfaces extra `pgoff` slots that translate to
/// `None`. A guest with non-4 KiB granule whose `tcr_el1` reads
/// zero would be a freeze-path bug elsewhere (the freeze
/// coordinator polls until `tcr_el1` populates before snapshotting).
fn guest_page_size(tcr_el1: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        let _ = tcr_el1;
        4096
    }
    #[cfg(target_arch = "aarch64")]
    {
        match (tcr_el1 >> 30) & 0x3 {
            0b10 => 4096,
            0b01 => 16384,
            0b11 => 65536,
            _ => 4096, // 0b00 reserved; conservative fallback
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = tcr_el1;
        4096
    }
}

/// `GUARD_SZ / 2` from `kernel/bpf/arena.c`.
///
/// Kernel formula:
///   `GUARD_SZ = round_up(1ull << sizeof_field(struct bpf_insn, off) * 8,
///                        PAGE_SIZE << 1)`
/// where `sizeof_field(struct bpf_insn, off) * 8 = 16` so the lower
/// term is `1 << 16 = 65536`. Result depends on the kernel's page
/// granule (`PAGE_SIZE << 1`):
///   - 4 KiB pages: `round_up(65536, 8192)` = 65536, GUARD_HALF = 32768.
///   - 16 KiB pages: `round_up(65536, 32768)` = 65536, GUARD_HALF = 32768.
///   - 64 KiB pages: `round_up(65536, 131072)` = 131072, GUARD_HALF = 65536.
///
/// `bpf_arena_get_kern_vm_start` returns `arena->kern_vm->addr +
/// GUARD_SZ/2`, so the kernel-side accessible region starts
/// `GUARD_HALF` past the raw `vm_struct.addr`. The walker must add
/// this offset when translating user-VA to kern-VA.
fn guard_half(page_size: u64) -> u64 {
    (1u64 << 16).next_multiple_of(page_size << 1) / 2
}

/// Maximum number of pages the walker will translate per arena
/// sequentially.
///
/// `KERN_VM_SZ = SZ_4G + GUARD_SZ` is the kernel's vmalloc reservation
/// (~1M pages) but most arenas use a small fraction. Cap the
/// sequential walk at 4096 pages (16 MiB) to bound report size and
/// freeze-path latency (a full 1M-page walk at ~1 µs per
/// translate_kva would burn ~1 s on the freeze hot path); truncation
/// is surfaced via [`ArenaSnapshot::truncated`] and a sparse stride
/// sweep (see [`MAX_ARENA_STRIDE_PROBES`]) catches mapped pages
/// beyond this cap.
const MAX_ARENA_PAGES: u64 = 4096;

/// Number of evenly-spaced stride probes the walker performs across
/// pgoffs [`MAX_ARENA_PAGES`]..`declared_pages` when `declared_pages`
/// exceeds the sequential cap. Lets the walker surface mapped pages
/// in sparse arenas (e.g. a scheduler that allocated pages near the
/// 4 GiB end of its user_vm window) without paying the full 1M-page
/// translate_kva cost.
///
/// 256 probes × ~1 µs per translate ≈ 0.25 ms — negligible on the
/// freeze hot path. Each hit lands in [`ArenaSnapshot::pages`]
/// alongside the sequential prefix, so the consumer sees both.
const MAX_ARENA_STRIDE_PROBES: u64 = 256;

/// Defensive cap on the arena's address-range span, in bytes.
///
/// The walker computes its span from `info.max_entries * page_size`
/// (the BPF map's declared page capacity, see [`snapshot_arena`]).
/// `bpf_arena_init` allows at most 4 GiB worth of pages by design —
/// the BPF JIT addresses arena pointers via the low 32 bits of the
/// user address, so anything wider than `0x1_0000_0000` cannot be a
/// real arena layout (see `bpf_arena_alloc_pages` in
/// `kernel/bpf/arena.c`). A torn / corrupt `bpf_map.max_entries` or
/// a freeze-time race against `arena_init` could yield a wild value;
/// cap it here so the walker never multiplies a near-`u64::MAX` page
/// count by the page size (overflow) or attempts to walk billions of
/// pgoffs (live-lock on the freeze path).
const MAX_VM_RANGE_BYTES: u64 = 0x1_0000_0000;

/// Byte offsets within `struct bpf_arena` and `struct vm_struct`
/// needed for the host-side arena walker.
///
/// Resolved from BTF at startup so the walker doesn't hardcode kernel
/// layout. Mirrors the [`super::btf_offsets::BpfMapOffsets`] pattern.
#[derive(Debug, Clone)]
pub struct BpfArenaOffsets {
    /// Offset of `kern_vm` (`struct vm_struct *`) within `struct bpf_arena`.
    pub arena_kern_vm: usize,
    /// Offset of `user_vm_start` (u64) within `struct bpf_arena`.
    pub arena_user_vm_start: usize,
    /// Offset of `addr` (`void *`) within `struct vm_struct`.
    pub vm_struct_addr: usize,
}

impl BpfArenaOffsets {
    /// Parse BTF from a vmlinux ELF and resolve arena field offsets.
    ///
    /// Returns Err on kernels whose BTF lacks `bpf_arena` (i.e. arena
    /// support is not built in) — the caller can treat the absent
    /// offsets as a signal to skip arena enumeration.
    ///
    /// Production callers (the freeze coordinator) reach this code
    /// via [`Self::from_btf`] on a pre-parsed `&Btf` to amortize the
    /// ELF parse — `from_vmlinux` stays public as the convenience
    /// entry point for direct-from-vmlinux callers (CLI tools, unit
    /// tests against a vmlinux on disk).
    #[allow(dead_code)]
    pub fn from_vmlinux(path: &Path) -> Result<Self> {
        let btf = load_btf_from_path(path).context("btf: open vmlinux")?;
        Self::from_btf(&btf)
    }

    /// Resolve arena struct offsets from a pre-loaded BTF object.
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let (bpf_arena, _) = find_struct(btf, "bpf_arena")
            .context("btf: struct bpf_arena not found (arena unsupported on this kernel?)")?;
        let arena_kern_vm = member_byte_offset(btf, &bpf_arena, "kern_vm")?;
        let arena_user_vm_start = member_byte_offset(btf, &bpf_arena, "user_vm_start")?;

        let (vm_struct, _) =
            find_struct(btf, "vm_struct").context("btf: struct vm_struct not found")?;
        let vm_struct_addr = member_byte_offset(btf, &vm_struct, "addr")?;

        Ok(Self {
            arena_kern_vm,
            arena_user_vm_start,
            vm_struct_addr,
        })
    }
}

/// One mapped arena page captured from guest memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ArenaPage {
    /// User-side virtual address (32-bit window starting at
    /// `arena.user_vm_start`). Operators correlate this with the
    /// pointer values they see in BPF program output.
    pub user_addr: u64,
    /// One arena page's worth of bytes read from the guest. Length
    /// matches the guest kernel's MMU page size: 4 KiB on x86_64
    /// and on aarch64 with `TCR_EL1.TG1=0b10`; 16 KiB on aarch64
    /// 16 KiB-granule kernels (Apple Silicon style); 64 KiB on
    /// aarch64 64 KiB-granule kernels. The resolution lives in
    /// [`guest_page_size`] — the snapshot stamps every captured
    /// page at that size.
    pub bytes: Vec<u8>,
}

/// Snapshot of one arena map's mapped pages.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ArenaSnapshot {
    /// Mapped pages, in pgoff order (skipped over unmapped pgoffs).
    /// Sequential prefix (pgoffs `0..MAX_ARENA_PAGES`) followed by any
    /// stride-probe hits in the sparse tail (pgoffs sampled across
    /// `MAX_ARENA_PAGES..declared_pages`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pages: Vec<ArenaPage>,
    /// True when the walker stopped sequential enumeration at
    /// [`MAX_ARENA_PAGES`] before finishing the user_vm window. The
    /// stride sweep that follows samples the tail at coarse intervals,
    /// so a hit reaches `pages` even when this flag is set; pgoffs
    /// between sampled positions are still silently skipped.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
    /// Total declared page count. Derived from
    /// `max_entries * page_size` (the BPF map's declared page
    /// capacity, with `page_size` resolved from the guest's
    /// TCR_EL1 via [`guest_page_size`]), not the user_vm window.
    /// Reflects any [`MAX_VM_RANGE_BYTES`] cap. Surfaced alongside
    /// `pages.len()` so consumers can see the
    /// allocated-vs-declared ratio.
    pub declared_pages: u64,
    /// True when `max_entries * page_size` exceeded
    /// [`MAX_VM_RANGE_BYTES`] (4 GiB) and the walker capped the span
    /// before computing `declared_pages`. Indicates a torn / corrupt
    /// `bpf_arena` struct or a freeze-time race against initialization;
    /// the rendered pages still come from valid translates, so the
    /// snapshot is usable.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub span_capped: bool,
    /// Kernel-side base of the arena's user_vm window:
    /// `bpf_arena.kern_vm->addr + GUARD_HALF`. Surfaces here so
    /// downstream consumers (notably the [`super::sdt_alloc`] tree
    /// walker) can translate `__arena` pointers without re-reading
    /// `struct bpf_arena` themselves. `0` when the snapshot bailed
    /// before computing the value (kern_vm_kva NULL, vm_addr NULL,
    /// or any of the upstream translates failed).
    ///
    /// Always serialized — the zero value carries diagnostic
    /// information ("walker reached this point but couldn't compute
    /// the base"), so suppressing it would mask the failure. Mirrors
    /// the policy used for the sibling `declared_pages` field.
    pub kern_vm_start: u64,
    /// User-side base of the arena window: the value of
    /// `bpf_arena.user_vm_start`, the address space the BPF program
    /// (and any captured `__arena` pointer) sees. `[user_vm_start ..
    /// user_vm_start + 4 GiB)` is the kernel-enforced upper bound
    /// (`bpf_arena_alloc_pages` clamps to `SZ_4G`). Consumers use it
    /// to classify a pointer as "lives in this arena" before chasing
    /// into [`Self::pages`].
    ///
    /// `0` when the snapshot bailed before reading
    /// `arena.user_vm_start` (e.g. `arena_pa` translate failed). On
    /// the syscall backend this comes from `bpf_map.map_extra` which
    /// the kernel pins at create time (`lib/arena_map.h` hardcodes
    /// `1<<44` on x86, `1<<32` on aarch64). On the guest-memory
    /// backend it's read directly from
    /// `bpf_arena.user_vm_start` via the resolved offset.
    ///
    /// Always serialized for the same diagnostic reason as
    /// [`Self::kern_vm_start`].
    pub user_vm_start: u64,
}

/// Walk the arena's mapped pages and return a snapshot.
///
/// Reads `kern_vm` from `struct bpf_arena` at `info.map_kva`,
/// dereferences to `vm_struct.addr`, computes
/// `kern_vm_start = addr + GUARD_HALF`, and for each pgoff in
/// `0..N` translates `kern_vm_start + (u32)user_addr` via
/// [`GuestMem::translate_kva`]. Pages that fail to translate are
/// "not faulted in" and silently skipped.
///
/// The walker is best-effort: any read failure on `bpf_arena` /
/// `vm_struct` itself yields an empty snapshot rather than an error,
/// so a corrupt arena can't break the broader failure dump.
pub fn snapshot_arena(
    kernel: &GuestKernel,
    info: &BpfMapInfo,
    offsets: &BpfArenaOffsets,
) -> ArenaSnapshot {
    if info.map_type != BPF_MAP_TYPE_ARENA {
        return ArenaSnapshot::default();
    }

    let mem = kernel.mem();
    let walk = kernel.walk_context();
    let page_size = guest_page_size(walk.tcr_el1);
    let guard_half_bytes = guard_half(page_size);

    // bpf_arena embeds bpf_map at offset 0, so map_kva == arena_kva.
    let arena_kva = info.map_kva;
    // Translate the arena struct itself — it may be kmalloc'd
    // (direct map) or vmalloc'd (`bpf_map_area_alloc`).
    let Some(arena_pa) = super::idr::translate_any_kva(
        mem,
        walk.cr3_pa,
        walk.page_offset,
        arena_kva,
        walk.l5,
        walk.tcr_el1,
    ) else {
        return ArenaSnapshot::default();
    };

    let user_vm_start = mem.read_u64(arena_pa, offsets.arena_user_vm_start);
    let kern_vm_kva = mem.read_u64(arena_pa, offsets.arena_kern_vm);
    // Preserve `user_vm_start` even when the kern-side walk fails:
    // the `MemReader::is_arena_addr` consumer needs it to classify
    // an `__arena` pointer as in-window (vs. a kernel kptr) so the
    // Ptr-deref path returns `None` cleanly instead of falling
    // through to the kernel-kptr cpumask probe. Without the anchor,
    // an arena pointer would be misread as a slab address — at best
    // garbage hex, at worst a translate against an unmapped page.
    if kern_vm_kva == 0 {
        return ArenaSnapshot {
            user_vm_start,
            ..ArenaSnapshot::default()
        };
    }

    // vm_struct lives in the kernel's slab/kmalloc area; direct or
    // vmalloc, so use translate_any_kva.
    let Some(vm_struct_pa) = super::idr::translate_any_kva(
        mem,
        walk.cr3_pa,
        walk.page_offset,
        kern_vm_kva,
        walk.l5,
        walk.tcr_el1,
    ) else {
        return ArenaSnapshot {
            user_vm_start,
            ..ArenaSnapshot::default()
        };
    };
    let vm_addr = mem.read_u64(vm_struct_pa, offsets.vm_struct_addr);
    if vm_addr == 0 {
        return ArenaSnapshot {
            user_vm_start,
            ..ArenaSnapshot::default()
        };
    }
    let kern_vm_start = vm_addr.wrapping_add(guard_half_bytes);

    // max_entries is the create-time page capacity; user_vm_end may
    // be 0 for arenas without userspace mmap.
    let plan = ArenaWalkPlan::new((info.max_entries as u64) * page_size, page_size);

    let mut snapshot = ArenaSnapshot {
        pages: Vec::new(),
        truncated: plan.truncated,
        declared_pages: plan.declared_pages,
        span_capped: plan.span_capped,
        kern_vm_start,
        user_vm_start,
    };

    // Reusable scratch buffer for the per-page read. Sized once at
    // `page_size` and reused across every captured page: on success
    // the buffer is moved into `ArenaPage` (one allocation per
    // captured page is unavoidable since each page owns its bytes),
    // then a fresh allocation refills the scratch on the next
    // `resize`. The win is the SKIP path — every translate-failure
    // or short-read pgoff used to allocate-and-discard a page-sized
    // zero-initialised buffer; now those paths reuse the existing
    // scratch capacity. On a sparse arena window (most pgoffs
    // unmapped) this collapses thousands of doomed allocations into
    // one. The hot path (freeze coordinator's dump pipeline) used
    // to dominate freeze-time wallclock on arenas with declared
    // pages > captured pages.
    let mut scratch: Vec<u8> = Vec::with_capacity(page_size as usize);

    // Closure: translate one pgoff to a page-content read; push
    // onto `snapshot.pages` if the translate + read succeed.
    // Captures `mem`, `walk`, `kern_vm_start`, `user_vm_start`,
    // `page_size`, and `scratch` (mutable — drained into the
    // captured page on success).
    let mut try_capture_page = |pgoff: u64, pages: &mut Vec<ArenaPage>| {
        // user_vm_start + pgoff*page_size is a 64-bit value, but the
        // kernel composes the kern-VA from the LOW 32 bits only —
        // `uaddr32 = (u32)(arena->user_vm_start + pgoff * PAGE_SIZE)`
        // in arena_alloc_pages — since the user_vm window is capped
        // at SZ_4G and aligned so the low 32 bits cover the whole
        // span uniquely. Match the same truncation here.
        //
        // pgoff and page_size both originate from BPF map metadata
        // and the guest TCR_EL1; pgoff*page_size in u64 can overflow
        // when a corrupt map advertises a huge declared_pages count.
        // Skip the page on multiplication overflow — wrapping_add on
        // user_vm_start is intentional (matches kernel truncation),
        // but only when the multiplicand was correctly computed.
        let Some(byte_off) = pgoff.checked_mul(page_size) else {
            return;
        };
        let user_addr = user_vm_start.wrapping_add(byte_off);
        let kaddr = kern_vm_start.wrapping_add(user_addr & 0xFFFF_FFFF);
        let Some(pa) = mem.translate_kva(walk.cr3_pa, Kva(kaddr), walk.l5, walk.tcr_el1) else {
            return;
        };
        // Translate guarantees a page-aligned PA; bound-check
        // against guest DRAM size in case a corrupt PTE points
        // past end-of-DRAM.
        if pa + page_size > mem.size() {
            return;
        }
        // Resize the reusable scratch to `page_size` and zero-fill.
        // After a previous capture moved the inner Vec out via
        // `mem::take`, `scratch` is empty with `page_size` capacity;
        // resize allocates exactly the new buffer's bytes, but
        // skipping iterations that hit the early returns above
        // never reach this line so their alloc is avoided entirely.
        scratch.clear();
        scratch.resize(page_size as usize, 0);
        // `GuestMem::read_bytes` returns the actual byte count copied
        // (may be short when the PA crosses end-of-DRAM, even after
        // the bounds check above — DRAM can have non-contiguous
        // regions). Truncate the buffer to that count so consumers
        // never see the zero-init tail of an unwritten range as
        // legitimate page bytes.
        let n = mem.read_bytes(pa, &mut scratch);
        scratch.truncate(n);
        if scratch.is_empty() {
            return;
        }
        // Move the populated buffer into the captured page; the
        // scratch falls back to empty (capacity preserved) for the
        // next iteration.
        pages.push(ArenaPage {
            user_addr,
            bytes: std::mem::take(&mut scratch),
        });
    };

    // Phase 1: sequential walk of the first MAX_ARENA_PAGES (16 MiB
    // window) — covers every scheduler today, where allocations cluster
    // near pgoff 0.
    for pgoff in 0..plan.sequential_to {
        try_capture_page(pgoff, &mut snapshot.pages);
    }

    // Phase 2: stride sweep over the sparse tail. Without this, a
    // scheduler that allocated even one page near the 4 GiB end of
    // its user_vm window would be invisible to the dump despite the
    // truncation flag. Mapped pages discovered here append to
    // `snapshot.pages` after the sequential prefix and are
    // discoverable by `user_addr` (the consumer correlates by user
    // pointer, not pgoff index, so out-of-order pgoffs are fine).
    if let Some(stride) = plan.stride {
        let mut pgoff = plan.sequential_to;
        while pgoff < plan.declared_pages {
            try_capture_page(pgoff, &mut snapshot.pages);
            // Saturate at declared_pages on the last step; without
            // this `pgoff += stride` could skip past the final page
            // when stride > 1.
            pgoff = pgoff.saturating_add(stride);
        }
    }

    snapshot
}

/// Pure computation that decides how many pgoffs the walker must
/// translate (sequential prefix + stride sweep). Extracted so the
/// span-cap, declared-page, and stride-derivation logic is unit-
/// testable without mocking a [`super::guest::GuestKernel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ArenaWalkPlan {
    /// Page count the snapshot reports as "declared". Reflects any
    /// [`MAX_VM_RANGE_BYTES`] cap.
    declared_pages: u64,
    /// True when [`MAX_VM_RANGE_BYTES`] capped the raw span.
    span_capped: bool,
    /// True when `declared_pages > MAX_ARENA_PAGES` and the walker
    /// will skip pgoffs in the sparse tail.
    truncated: bool,
    /// Sequential-walk endpoint: the walker enumerates
    /// `0..sequential_to` exhaustively.
    sequential_to: u64,
    /// Stride for the post-sequential sweep, or `None` when no tail
    /// remains. `Some(stride)` walks pgoffs
    /// `sequential_to, sequential_to + stride, ...` until
    /// `declared_pages`.
    stride: Option<u64>,
}

impl ArenaWalkPlan {
    fn new(raw_span: u64, page_size: u64) -> Self {
        let span_capped = raw_span > MAX_VM_RANGE_BYTES;
        let span = raw_span.min(MAX_VM_RANGE_BYTES);
        let declared_pages = span / page_size;
        let sequential_to = declared_pages.min(MAX_ARENA_PAGES);
        let truncated = declared_pages > sequential_to;
        let stride = if declared_pages > MAX_ARENA_PAGES {
            let tail_pages = declared_pages - MAX_ARENA_PAGES;
            // div_ceil so stride * MAX_ARENA_STRIDE_PROBES covers
            // the whole tail; .max(1) so a tail smaller than
            // MAX_ARENA_STRIDE_PROBES still walks every remaining
            // page sequentially.
            Some(tail_pages.div_ceil(MAX_ARENA_STRIDE_PROBES).max(1))
        } else {
            None
        };
        Self {
            declared_pages,
            span_capped,
            truncated,
            sequential_to,
            stride,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_arena_offsets_from_vmlinux() {
        let path = match crate::monitor::find_test_vmlinux() {
            Some(p) => p,
            None => return,
        };
        // Skip when find_test_vmlinux returns the raw BTF blob — the
        // vmlinux-ELF parse path inside `from_vmlinux` would fail on
        // it, but `from_btf` works directly. Tests in btf_offsets/tests.rs
        // skip the same way for the same reason.
        if path.starts_with("/sys/") {
            crate::report::test_skip("vmlinux is raw BTF (skipping ELF-only path)");
            return;
        }
        let offsets = match BpfArenaOffsets::from_vmlinux(&path) {
            Ok(o) => o,
            Err(e) => {
                // Older kernels without arena support: BTF lacks
                // `struct bpf_arena`. That's a valid configuration —
                // skip rather than fail.
                crate::report::test_skip(format!("arena BTF missing: {e}"));
                return;
            }
        };
        // bpf_arena starts with `struct bpf_map map`, so user_vm_*
        // come AFTER the embedded bpf_map; both must be at nonzero
        // offsets. kern_vm follows them in the kernel layout.
        assert!(
            offsets.arena_user_vm_start > 0,
            "user_vm_start follows embedded bpf_map"
        );
        assert_ne!(
            offsets.arena_kern_vm, offsets.arena_user_vm_start,
            "kern_vm distinct from user_vm_start"
        );
        // vm_struct.addr lives after the 8-byte union (next/llnode)
        // on 64-bit kernels.
        assert!(
            offsets.vm_struct_addr > 0,
            "vm_struct.addr follows the next/llnode union"
        );
    }

    // ---- ArenaWalkPlan: span cap + stride sweep -------------------
    //
    // The plan is a pure function of the raw user_vm span. Pin its
    // outputs against representative shapes so the snapshot_arena
    // call site stays tight against:
    //   - tiny arena (single page) — no truncation, no stride
    //   - mid arena (just under 16 MiB) — sequential only
    //   - large arena (declared > MAX_ARENA_PAGES) — sequential
    //     prefix + stride sweep
    //   - 4 GiB-cap (raw_span > MAX_VM_RANGE_BYTES) — span_capped flag,
    //     declared_pages clamped to MAX_VM_RANGE_BYTES / page_size
    //   - corrupt span (raw_span = u64::MAX) — capped, flag set,
    //     no overflow

    /// Page size used for ArenaWalkPlan unit tests. Production code
    /// resolves the page size from [`guest_page_size`] (which decodes
    /// the guest's `TCR_EL1.TG1`); the plan tests pin their math
    /// against an explicit 4 KiB so they exercise the same shapes
    /// regardless of the host the test runs on. Granule-specific
    /// shapes have their own dedicated tests
    /// (`arena_walk_plan_16k_granule_*`).
    const TEST_PAGE_SIZE: u64 = 4096;

    #[test]
    fn arena_walk_plan_constants_sane() {
        // The plan-derivation invariants depend on these constants.
        // Pin them so a future tightening surfaces here, not in
        // snapshot_arena's runtime behavior.
        assert_eq!(MAX_VM_RANGE_BYTES, 0x1_0000_0000);
        assert_eq!(MAX_ARENA_PAGES, 4096);
        assert_eq!(MAX_ARENA_STRIDE_PROBES, 256);
    }

    #[test]
    fn arena_walk_plan_single_page() {
        // Smallest non-empty arena: one page. Sequential walk covers
        // it; no stride needed; no truncation.
        let plan = ArenaWalkPlan::new(TEST_PAGE_SIZE, TEST_PAGE_SIZE);
        assert_eq!(plan.declared_pages, 1);
        assert!(!plan.span_capped);
        assert!(!plan.truncated);
        assert_eq!(plan.sequential_to, 1);
        assert_eq!(plan.stride, None);
    }

    #[test]
    fn arena_walk_plan_exactly_max_arena_pages() {
        // declared == MAX_ARENA_PAGES: still no stride, no truncation.
        // Boundary case: MAX_ARENA_PAGES walks sequentially.
        let plan = ArenaWalkPlan::new(MAX_ARENA_PAGES * TEST_PAGE_SIZE, TEST_PAGE_SIZE);
        assert_eq!(plan.declared_pages, MAX_ARENA_PAGES);
        assert!(!plan.truncated);
        assert_eq!(plan.sequential_to, MAX_ARENA_PAGES);
        assert_eq!(plan.stride, None);
    }

    #[test]
    fn arena_walk_plan_one_page_past_max() {
        // declared = MAX_ARENA_PAGES + 1: stride mode kicks in for
        // the single tail page; stride must be 1 (every page).
        let plan = ArenaWalkPlan::new((MAX_ARENA_PAGES + 1) * TEST_PAGE_SIZE, TEST_PAGE_SIZE);
        assert_eq!(plan.declared_pages, MAX_ARENA_PAGES + 1);
        assert!(plan.truncated);
        assert_eq!(plan.sequential_to, MAX_ARENA_PAGES);
        assert_eq!(plan.stride, Some(1));
    }

    #[test]
    fn arena_walk_plan_full_4gib() {
        // Largest legitimate arena: full 4 GiB user_vm window (1M pages).
        // Sequential covers first 16 MiB; stride sweeps the remaining
        // ~1M-4096 pages with 256 probes -> stride = ceil((1M - 4096) / 256).
        let raw = MAX_VM_RANGE_BYTES;
        let plan = ArenaWalkPlan::new(raw, TEST_PAGE_SIZE);
        assert_eq!(plan.declared_pages, raw / TEST_PAGE_SIZE);
        assert!(!plan.span_capped, "exactly 4 GiB is at the cap, not above");
        assert!(plan.truncated);
        assert_eq!(plan.sequential_to, MAX_ARENA_PAGES);
        let stride = plan.stride.expect("stride mode for >MAX_ARENA_PAGES");
        let tail = plan.declared_pages - MAX_ARENA_PAGES;
        // Verify stride covers the tail: stride * MAX_ARENA_STRIDE_PROBES
        // must reach `tail` with at most one slot of overshoot.
        assert!(stride * MAX_ARENA_STRIDE_PROBES >= tail);
        assert!((stride - 1) * MAX_ARENA_STRIDE_PROBES < tail);
    }

    #[test]
    fn arena_walk_plan_caps_at_4gib() {
        // Raw span 8 GiB (corrupt struct): span_capped flag set,
        // declared_pages clamped to MAX_VM_RANGE_BYTES / page_size.
        let plan = ArenaWalkPlan::new(2 * MAX_VM_RANGE_BYTES, TEST_PAGE_SIZE);
        assert!(plan.span_capped);
        assert_eq!(plan.declared_pages, MAX_VM_RANGE_BYTES / TEST_PAGE_SIZE);
        assert!(plan.truncated);
        assert!(plan.stride.is_some());
    }

    #[test]
    fn arena_walk_plan_caps_corrupt_u64_max_span() {
        // Pathological: raw_span = u64::MAX. The cap must apply
        // BEFORE the span-to-pages division; without the cap,
        // u64::MAX / page_size = ~4.5 quadrillion pages and the
        // pgoff loop would live-lock.
        let plan = ArenaWalkPlan::new(u64::MAX, TEST_PAGE_SIZE);
        assert!(plan.span_capped);
        assert_eq!(plan.declared_pages, MAX_VM_RANGE_BYTES / TEST_PAGE_SIZE);
        assert!(plan.truncated);
    }

    #[test]
    fn arena_walk_plan_zero_span() {
        // Edge: zero span. snapshot_arena can reach this with
        // max_entries=0; the plan must handle zero spans without
        // panicking or computing nonsense bounds.
        let plan = ArenaWalkPlan::new(0, TEST_PAGE_SIZE);
        assert_eq!(plan.declared_pages, 0);
        assert!(!plan.span_capped);
        assert!(!plan.truncated);
        assert_eq!(plan.sequential_to, 0);
        assert_eq!(plan.stride, None);
    }

    #[test]
    fn arena_walk_plan_stride_visits_every_pgoff_when_short_tail() {
        // tail < MAX_ARENA_STRIDE_PROBES: stride saturates at 1, so
        // the sweep walks every remaining page. Verify by simulating
        // the walk and counting positions.
        // declared = MAX_ARENA_PAGES + 50 -> tail = 50 -> stride = 1.
        let plan = ArenaWalkPlan::new((MAX_ARENA_PAGES + 50) * TEST_PAGE_SIZE, TEST_PAGE_SIZE);
        assert_eq!(plan.stride, Some(1));
        let mut pgoff = plan.sequential_to;
        let mut visited = 0u64;
        while pgoff < plan.declared_pages {
            visited += 1;
            pgoff = pgoff.saturating_add(plan.stride.unwrap());
        }
        assert_eq!(visited, 50, "every tail page should be visited");
    }

    #[test]
    fn arena_walk_plan_stride_distributes_probes_in_long_tail() {
        // tail >> MAX_ARENA_STRIDE_PROBES: stride > 1, fewer probes
        // than tail pages. Verify the sweep visits exactly
        // approximately MAX_ARENA_STRIDE_PROBES positions.
        let plan = ArenaWalkPlan::new(MAX_VM_RANGE_BYTES, TEST_PAGE_SIZE); // full 4 GiB
        let mut pgoff = plan.sequential_to;
        let mut visited = 0u64;
        while pgoff < plan.declared_pages {
            visited += 1;
            pgoff = pgoff.saturating_add(plan.stride.unwrap());
        }
        // The sweep visits ceil(tail / stride) positions; for the
        // 4 GiB case `stride * MAX_ARENA_STRIDE_PROBES >= tail` so
        // visited <= MAX_ARENA_STRIDE_PROBES, and `>= tail / stride`
        // ensures it's not zero.
        assert!(
            visited <= MAX_ARENA_STRIDE_PROBES + 1,
            "visited {visited}, expected ≤ {} probes",
            MAX_ARENA_STRIDE_PROBES + 1
        );
        assert!(
            visited >= MAX_ARENA_STRIDE_PROBES - 1,
            "visited {visited}, expected ≥ {}-ish probes",
            MAX_ARENA_STRIDE_PROBES - 1
        );
    }

    /// `guard_half` mirrors the kernel's `bpf_arena_get_kern_vm_start`
    /// `GUARD_SZ/2` formula. Pin the three legitimate page granules
    /// (4 KiB, 16 KiB, 64 KiB) against the hand-computed values from
    /// the doc comment so a regression in the
    /// `next_multiple_of(page_size << 1)` math surfaces here.
    #[test]
    fn guard_half_matches_kernel_formula() {
        // 4 KiB granule: round_up(65536, 8192) = 65536, /2 = 32768.
        assert_eq!(guard_half(4096), 32768);
        // 16 KiB granule: round_up(65536, 32768) = 65536, /2 = 32768.
        assert_eq!(guard_half(16384), 32768);
        // 64 KiB granule: round_up(65536, 131072) = 131072, /2 = 65536.
        assert_eq!(guard_half(65536), 65536);
    }

    /// `guest_page_size` decodes `TCR_EL1.TG1` (bits [31:30]) into
    /// the granule size on aarch64; on x86_64 it is fixed at 4 KiB
    /// regardless of the input. Pin the four encodings + the
    /// reserved fallback path so a regression in the bit math
    /// surfaces here.
    #[test]
    fn guest_page_size_decodes_tg1() {
        #[cfg(target_arch = "x86_64")]
        {
            // x86_64: page size is always 4 KiB, regardless of the
            // (ignored) `tcr_el1` argument.
            assert_eq!(guest_page_size(0), 4096);
            assert_eq!(guest_page_size(0b01u64 << 30), 4096);
            assert_eq!(guest_page_size(0b10u64 << 30), 4096);
            assert_eq!(guest_page_size(0b11u64 << 30), 4096);
        }
        #[cfg(target_arch = "aarch64")]
        {
            // TG1=0b10 → 4 KiB
            assert_eq!(guest_page_size(0b10u64 << 30), 4096);
            // TG1=0b01 → 16 KiB (Apple Silicon style)
            assert_eq!(guest_page_size(0b01u64 << 30), 16384);
            // TG1=0b11 → 64 KiB
            assert_eq!(guest_page_size(0b11u64 << 30), 65536);
            // TG1=0b00 (reserved) → conservative 4 KiB fallback
            assert_eq!(guest_page_size(0), 4096);
        }
    }

    /// 16 KiB-granule arena (Apple Silicon kernel build): a single
    /// declared page is 16 KiB. With raw_span = 16384 the plan must
    /// report `declared_pages = 1`, no stride. Pre-fix, `PAGE_SIZE`
    /// was hardcoded to 4096 so 16384 / 4096 = 4 pages — wrong.
    #[test]
    fn arena_walk_plan_16k_granule_single_page() {
        let plan = ArenaWalkPlan::new(16384, 16384);
        assert_eq!(plan.declared_pages, 1);
        assert!(!plan.span_capped);
        assert!(!plan.truncated);
        assert_eq!(plan.sequential_to, 1);
        assert_eq!(plan.stride, None);
    }

    /// 16 KiB-granule arena at the 4 GiB cap: `declared_pages` =
    /// 4 GiB / 16 KiB = 256 K. Pre-fix, the divisor was 4 KiB so
    /// the count would have been 4x too large.
    #[test]
    fn arena_walk_plan_16k_granule_full_cap() {
        let plan = ArenaWalkPlan::new(MAX_VM_RANGE_BYTES, 16384);
        assert_eq!(plan.declared_pages, MAX_VM_RANGE_BYTES / 16384);
        assert!(!plan.span_capped);
        assert!(plan.truncated);
        assert_eq!(plan.sequential_to, MAX_ARENA_PAGES);
    }
}
