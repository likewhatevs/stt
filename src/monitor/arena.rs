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
//! then for each pgoff in `0..N` compute `kaddr` and run
//! [`GuestMem::translate_kva`] (the existing PTE walker against
//! `init_mm`'s page table). Pages whose translate fails are simply
//! "not faulted in" — arena maps are sparse by design.
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

/// Page size assumed by the arena walker.
///
/// `arena_alloc_pages` and `arena_vm_fault` both call
/// `apply_to_page_range` on `PAGE_SIZE`-granular ranges, so 4 KiB is
/// the kernel's own page-granule for arena. Aarch64 with 16 KiB or
/// 64 KiB granules would diverge — those configurations are not
/// supported by ktstr today (see [`super::reader::GuestMem`] page
/// walker which also assumes 4 KiB lowest-level pages).
const PAGE_SIZE: u64 = 4096;

/// `GUARD_SZ / 2` from `kernel/bpf/arena.c`.
///
/// `GUARD_SZ = round_up(1ull << sizeof_field(struct bpf_insn, off) * 8, PAGE_SIZE << 1)`
/// = `round_up(65536, 8192)` = 65536 (64 KiB).
/// `bpf_arena_get_kern_vm_start` returns `arena->kern_vm->addr +
/// GUARD_SZ/2`, so the kernel-side accessible region starts 32 KiB
/// past the raw `vm_struct.addr`. The walker must add this offset
/// when translating user-VA to kern-VA.
const GUARD_HALF: u64 = 32768;

/// Maximum number of pages the walker will translate per arena.
///
/// `KERN_VM_SZ = SZ_4G + GUARD_SZ` is the kernel's vmalloc reservation
/// (~1M pages) but most arenas use a small fraction. Cap at 4096
/// pages (16 MiB) to bound report size; truncation is surfaced via
/// [`ArenaSnapshot::truncated`].
const MAX_ARENA_PAGES: u64 = 4096;

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
    /// Offset of `user_vm_end` (u64) within `struct bpf_arena`.
    pub arena_user_vm_end: usize,
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
        let arena_user_vm_end = member_byte_offset(btf, &bpf_arena, "user_vm_end")?;

        let (vm_struct, _) =
            find_struct(btf, "vm_struct").context("btf: struct vm_struct not found")?;
        let vm_struct_addr = member_byte_offset(btf, &vm_struct, "addr")?;

        Ok(Self {
            arena_kern_vm,
            arena_user_vm_start,
            arena_user_vm_end,
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
    /// 4 KiB of page contents read from the guest.
    pub bytes: Vec<u8>,
}

/// Snapshot of one arena map's mapped pages.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ArenaSnapshot {
    /// Mapped pages, in pgoff order (skipped over unmapped pgoffs).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pages: Vec<ArenaPage>,
    /// True when the walker stopped at [`MAX_ARENA_PAGES`] before
    /// finishing the user_vm window. The unrendered tail is silently
    /// dropped — recording it would itself need unbounded memory.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
    /// Total declared page count of the user_vm window
    /// (`(user_vm_end - user_vm_start) / PAGE_SIZE`). Surfaced
    /// alongside `pages.len()` so consumers can see the
    /// allocated-vs-declared ratio.
    pub declared_pages: u64,
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
/// so a corrupt arena can't break the broader stall dump.
pub fn snapshot_arena(
    kernel: &GuestKernel<'_>,
    info: &BpfMapInfo,
    offsets: &BpfArenaOffsets,
) -> ArenaSnapshot {
    if info.map_type != BPF_MAP_TYPE_ARENA {
        return ArenaSnapshot::default();
    }

    let mem = kernel.mem();
    let cr3_pa = kernel.cr3_pa();
    let page_offset = kernel.page_offset();
    let l5 = kernel.l5();

    // bpf_arena embeds bpf_map at offset 0, so map_kva == arena_kva.
    let arena_kva = info.map_kva;
    // Translate the arena struct itself — it may be kmalloc'd
    // (direct map) or vmalloc'd (`bpf_map_area_alloc`).
    let Some(arena_pa) = super::idr::translate_any_kva(mem, cr3_pa, page_offset, arena_kva, l5)
    else {
        return ArenaSnapshot::default();
    };

    let user_vm_start = mem.read_u64(arena_pa, offsets.arena_user_vm_start);
    let user_vm_end = mem.read_u64(arena_pa, offsets.arena_user_vm_end);
    let kern_vm_kva = mem.read_u64(arena_pa, offsets.arena_kern_vm);
    if kern_vm_kva == 0 || user_vm_end <= user_vm_start {
        return ArenaSnapshot::default();
    }

    // vm_struct lives in the kernel's slab/kmalloc area; direct or
    // vmalloc, so use translate_any_kva.
    let Some(vm_struct_pa) =
        super::idr::translate_any_kva(mem, cr3_pa, page_offset, kern_vm_kva, l5)
    else {
        return ArenaSnapshot::default();
    };
    let vm_addr = mem.read_u64(vm_struct_pa, offsets.vm_struct_addr);
    if vm_addr == 0 {
        return ArenaSnapshot::default();
    }
    let kern_vm_start = vm_addr.wrapping_add(GUARD_HALF);

    let span = user_vm_end - user_vm_start;
    let declared_pages = span / PAGE_SIZE;
    let to_walk = declared_pages.min(MAX_ARENA_PAGES);
    let truncated = declared_pages > to_walk;

    let mut snapshot = ArenaSnapshot {
        pages: Vec::new(),
        truncated,
        declared_pages,
    };

    for pgoff in 0..to_walk {
        // user_vm_start + pgoff*PAGE_SIZE is a 64-bit value, but the
        // kernel composes the kern-VA from the LOW 32 bits only —
        // `uaddr32 = (u32)(arena->user_vm_start + pgoff * PAGE_SIZE)`
        // in arena_alloc_pages — since the user_vm window is capped
        // at SZ_4G and aligned so the low 32 bits cover the whole
        // span uniquely. Match the same truncation here.
        let user_addr = user_vm_start.wrapping_add(pgoff * PAGE_SIZE);
        let kaddr = kern_vm_start.wrapping_add(user_addr & 0xFFFF_FFFF);
        let Some(pa) = mem.translate_kva(cr3_pa, Kva(kaddr), l5) else {
            continue;
        };
        // Translate guarantees a 4 KiB page-aligned PA; bound-check
        // against guest DRAM size in case a corrupt PTE points
        // past end-of-DRAM.
        if pa + PAGE_SIZE > mem.size() {
            continue;
        }
        let mut buf = vec![0u8; PAGE_SIZE as usize];
        // `GuestMem::read_bytes` returns the actual byte count copied
        // (may be short when the PA crosses end-of-DRAM, even after
        // the bounds check above — DRAM can have non-contiguous
        // regions). Truncate the buffer to that count so consumers
        // never see the zero-init tail of an unwritten range as
        // legitimate page bytes.
        let n = mem.read_bytes(pa, &mut buf);
        buf.truncate(n);
        if buf.is_empty() {
            continue;
        }
        snapshot.pages.push(ArenaPage {
            user_addr,
            bytes: buf,
        });
    }

    snapshot
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
        // it, but `from_btf` works directly. Tests in btf_offsets.rs
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
        assert!(
            offsets.arena_user_vm_end > offsets.arena_user_vm_start,
            "user_vm_end follows user_vm_start"
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
}
