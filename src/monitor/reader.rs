//! Guest physical memory access and monitor sampling loop.
//!
//! [`GuestMem`] wraps host pointers to guest DRAM regions and
//! provides bounds-checked volatile reads and writes for scalar types;
//! `read_bytes` uses `copy_nonoverlapping` for bulk copies. Multi-region
//! NUMA layouts are supported: each read/write resolves the target
//! region via binary search. It also implements 4-level and 5-level
//! x86-64 page table walks and a granule-agnostic aarch64 walker (4 KB
//! / 16 KB / 64 KB, level count derived from TCR_EL1.T1SZ and TG1) for
//! vmalloc'd addresses.
//!
//! The monitor loop (`monitor_loop`) periodically reads per-CPU
//! runqueue state from guest memory and collects `MonitorSample`s.

use super::btf_offsets::{
    CPU_MAX_IDLE_TYPES, KernelOffsets, SchedDomainOffsets, SchedDomainStatsOffsets,
    SchedstatOffsets, ScxEventOffsets,
};
use super::{
    CpuSnapshot, Kva, MonitorSample, RqSchedstat, SchedDomainSnapshot, SchedDomainStats,
    ScxEventCounters,
};
use std::os::unix::io::AsRawFd;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use vmm_sys_util::epoll::{ControlOperation, Epoll, EpollEvent, EventSet};
use vmm_sys_util::eventfd::EventFd;
use vmm_sys_util::timerfd::TimerFd;

/// Per-NUMA-node host memory region within a GuestMem.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MemRegion {
    /// Host pointer to the start of this region's mapping.
    pub(crate) host_ptr: *mut u8,
    /// DRAM-relative offset where this region starts.
    pub(crate) offset: u64,
    /// Size in bytes.
    pub(crate) size: u64,
}

/// Bundle of guest paging state threaded through every host-side KVA
/// translation call: the top-level page-table PA, the direct-map base,
/// the x86 5-level paging flag, and the cached aarch64 TCR_EL1.
///
/// Replaces the four-parameter fan (`cr3_pa`, `page_offset`, `l5`,
/// `tcr_el1`) that previously rode every page-walking signature in
/// [`super::bpf_prog`], [`super::task_enrichment`],
/// [`super::runnable_scan`], [`super::scx_walker`], and the freeze
/// coordinator's task / numa capture paths. The leaf walkers
/// [`GuestMem::translate_kva`] and [`super::idr::translate_any_kva`]
/// retain their unbundled signatures; intermediate callers pass
/// [`WalkContext`] and destructure at the leaf invocation. All four
/// fields originate in [`super::guest::GuestKernel`] (via its
/// accessors) and travel together; bundling them keeps signatures
/// terse without changing the values being passed.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct WalkContext {
    /// Guest physical address of the top-level page-table page (CR3 on
    /// x86, TTBR1 on aarch64). Threaded into [`GuestMem::translate_kva`]
    /// for vmalloc / module / per-CPU-percpu translations.
    pub cr3_pa: u64,
    /// Kernel direct-map base (`PAGE_OFFSET` on x86_64, linear-map
    /// base on aarch64). Adding this to a DRAM offset yields a KVA;
    /// subtracting it from a KVA yields the DRAM offset that
    /// [`GuestMem::read_bytes`] consumes.
    pub page_offset: u64,
    /// 5-level paging flag (x86 LA57). True when the guest enabled
    /// CR4.LA57 and CR3 points at a PML5 root. Ignored on aarch64.
    pub l5: bool,
    /// Cached aarch64 TCR_EL1 register (granule + T1SZ). Drives the
    /// page-table walker's granule and high-half-VA-width decoding.
    /// Always 0 on x86_64 where the register does not exist; the
    /// walker ignores the field on that arch.
    pub tcr_el1: u64,
}

/// Decoded aarch64 page-table walk parameters derived once from
/// `TCR_EL1`.
///
/// Every aarch64 translate previously re-decoded `T1SZ`, `TG1`,
/// `va_width`, `levels_below`, `stride`, `descaddrmask`, and the
/// first-level `indexmask` from a fresh `tcr_el1` argument. Those
/// fields are deterministic functions of `TCR_EL1`, and `TCR_EL1` is
/// immutable post-boot on the guest (the kernel writes it once
/// during early MMU bring-up in `__cpu_setup` and never modifies it
/// after — see arch/arm64/mm/proc.S). Caching the decoded form on
/// [`super::guest::GuestKernel`] (alongside the raw `tcr_el1`)
/// elides the per-call decode — meaningful on the freeze hot path
/// where each task enrichment performs dozens of translates per
/// task.
///
/// Construction goes through [`Self::from_tcr_el1`], which returns
/// `None` for the same configurations [`GuestMem::walk_aarch64`]
/// would reject mid-walk: `tcr_el1 == 0`, `T1SZ == 0`, reserved
/// `TG1 == 0b00`, `va_width < 4`, or `levels_below` outside `[1, 4]`.
/// On x86_64 the struct is unused (the walker ignores TCR_EL1) and
/// callers pass `None` through.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // used only by walk_aarch64; x86_64 builds elide
pub(crate) struct Aarch64WalkParams {
    /// Raw `TCR_EL1` value the params were decoded from. Used by
    /// [`GuestMem::translate_kva_with_aarch64_params`] as part of
    /// the page-TLB cache key so a translation performed under one
    /// TCR_EL1 cannot be reused under a different one (relevant on
    /// a future suspend/resume path that re-runs `__cpu_setup`;
    /// post-boot the value is immutable). Also lets a downstream
    /// consumer detect a TCR change by comparing against a fresh
    /// `tcr_el1` argument.
    pub tcr_el1: u64,
    /// Per-level index width in bits. 9 for 4 KiB, 11 for 16 KiB,
    /// 13 for 64 KiB.
    pub stride: u64,
    /// Top-level page-table level (0..=3). The cascade descends from
    /// here to level 3, the leaf.
    pub start_level: u64,
    /// Index mask for the first level: `!0 >> (64 - (va_width -
    /// stride * (4-start_level)))`.
    pub first_indexmask: u64,
    /// Index mask for every subsequent level: `(1 << (stride+3)) - 1`.
    pub indexmask_grainsize: u64,
    /// Descriptor address mask: `((1 << 50) - 1) & !indexmask_grainsize`.
    /// Strips the high attribute / SW bits at [63:50] and the low
    /// granule bits, preserving the low OA bits [49:granule_log2].
    /// Mirrors the kernel's `PTE_ADDR_LOW` from
    /// `arch/arm64/include/asm/pgtable-hwdef.h`
    /// (`((1 << (50 - PAGE_SHIFT)) - 1) << PAGE_SHIFT`).
    ///
    /// On non-LPA / non-LPA2 kernels bits [49:48] are RES0 by
    /// hardware contract (Arm ARM D5.3), so the wider mask is
    /// equivalent to a 48-bit mask in practice. The wider mask
    /// matches the kernel's accessor for forward-compatibility:
    /// FEAT_LPA on 64 KiB pages (which ktstr.kconfig leaves
    /// disabled) places OA bits [49:48] in this region, and only
    /// the high splice (bits [51:48] of OA in descriptor bits
    /// [15:12]) is then missing. FEAT_LPA2 (TCR_EL1.DS=1) is
    /// rejected by [`Self::from_tcr_el1`] because its OA encoding
    /// also requires a separate splice from descriptor bits [9:8].
    pub descaddrmask: u64,
}

#[cfg(target_arch = "aarch64")]
impl Aarch64WalkParams {
    /// Decode `TCR_EL1` into walk parameters, returning `None` for
    /// the same configurations [`GuestMem::walk_aarch64`] would
    /// reject mid-walk.
    ///
    /// Rejects FEAT_LPA2 (`TCR_EL1.DS == 1`, bit 59) because the
    /// walker's [`Self::descaddrmask`] cannot recover OA bits
    /// [51:50] from descriptor bits [9:8] — those bits are masked
    /// out as low descriptor metadata. Without that splice, valid
    /// LPA2 PTEs translate to wrong PAs that read attacker-controlled
    /// guest memory or out-of-range addresses. Returning `None` here
    /// makes the walker fail loudly instead — the caller surfaces it
    /// as "translation failed" rather than reading the wrong page.
    pub(crate) fn from_tcr_el1(tcr_el1: u64) -> Option<Self> {
        if tcr_el1 == 0 {
            return None;
        }
        // TCR_EL1.DS (bit 59) selects FEAT_LPA2 when set. The walker
        // does not support the 52-bit OA splice required for LPA2;
        // the kernel's `lpa2_is_enabled` (arch/arm64/include/asm/
        // pgtable-prot.h) reads exactly this bit.
        if tcr_el1 & (1u64 << 59) != 0 {
            return None;
        }
        let t1sz = (tcr_el1 >> 16) & 0x3F;
        if t1sz == 0 {
            return None;
        }
        let tg1 = (tcr_el1 >> 30) & 0x3;
        let stride: u64 = match tg1 {
            0b11 => 13,       // 64 KiB
            0b01 => 11,       // 16 KiB
            0b10 => 9,        // 4 KiB
            _ => return None, // 0b00 reserved
        };
        let va_width = 64u64.saturating_sub(t1sz);
        if va_width < 4 {
            return None;
        }
        let levels_below = (va_width - 4) / stride;
        if levels_below == 0 || levels_below > 4 {
            return None;
        }
        let start_level: u64 = 4 - levels_below;
        let indexmask_grainsize: u64 = (!0u64) >> (64 - (stride + 3));
        let first_indexmask: u64 = (!0u64) >> (64 - (va_width - stride * (4 - start_level)));
        // PTE_ADDR_LOW = bits [49:PAGE_SHIFT]. On non-LPA / non-LPA2
        // kernels bits [49:48] are RES0; matching the kernel's
        // accessor is forward-compatible at zero behaviour cost.
        let descaddrmask: u64 = ((!0u64) >> (64 - 50)) & !indexmask_grainsize;
        Some(Self {
            tcr_el1,
            stride,
            start_level,
            first_indexmask,
            indexmask_grainsize,
            descaddrmask,
        })
    }
}

/// Host pointer to the start of guest DRAM. Offsets passed to read/write
/// methods are DRAM-relative (x86_64: GPA 0, aarch64: GPA DRAM_START).
///
/// Carries per-NUMA-node region info (one region for single-node,
/// multiple for multi-node topologies). Each read/write resolves
/// the target region via binary search. With contiguous MAP_FIXED VA
/// (the current allocation strategy), the resolved pointer is
/// identical to `base.add(offset)`.
///
/// SAFETY: The pointer is valid for the lifetime of the KVM VM.
/// `ReservationGuard` owns the VA reservation (munmaps on drop);
/// per-node `MmapRegion`s have `owned=false` and do not munmap.
/// The guard outlives all threads that hold a `GuestMem`.
pub struct GuestMem {
    size: u64,
    regions: Vec<MemRegion>,
    /// Software TLB: caches the most recent successful KVA→PA
    /// translation. Cache key is `(cr3_pa, l5, tcr_el1, kva_page)`
    /// — every input the walker consumes (page-table root, walk
    /// mode, page-aligned KVA), see [`TlbEntry`].
    /// Sequential reads of structs that span a single page (e.g. a
    /// `bpf_map` struct, a `bpf_htab_elem`, a chained
    /// `sched_domain` walk) translate the same page repeatedly; the
    /// TLB collapses N walks into 1 walk + (N-1) compare-and-load.
    /// Unmapped pages are not cached — only successful walks update
    /// the slot, so a probe of an unmapped KVA still walks the page
    /// tables but does not pollute the TLB. A mismatch on any key
    /// field bypasses the cache.
    ///
    /// `Mutex<Option<TlbEntry>>` rather than `RefCell` because
    /// `GuestMem` is `Sync` (see the unsafe `Sync` impl below) and
    /// the freeze coordinator's read paths run from multiple
    /// threads (the monitor sampler, the freeze rendezvous, the
    /// dump renderer). `RefCell` would require `!Sync` which would
    /// break every existing caller. The lock is held only for the
    /// duration of one compare-and-load (lookup) or one struct
    /// store (populate), so contention is non-existent — the
    /// freeze coordinator is single-threaded for the per-dump read
    /// path that dominates this cache.
    page_tlb: std::sync::Mutex<Option<TlbEntry>>,
}

/// Single-slot software TLB entry. The walker only caches successful
/// translations: an unmapped KVA leaves the slot untouched.
///
/// The cache key is `(cr3_pa, l5, tcr_el1, kva_page)` — every input
/// to the underlying page walker that would change the resolution
/// must match for a hit. Without the walk-mode fields a 4-level
/// translation would be reused under a 5-level lookup (x86 LA57)
/// or under a different aarch64 TG1/T1SZ configuration, returning a
/// PA that the walker would never have produced for the new mode.
/// In practice `l5` and `tcr_el1` are immutable post-boot, but a
/// future suspend/resume-and-reconfigure path that re-runs
/// `__cpu_setup` would land here, and gating on the inputs costs
/// only two compares per hit.
#[derive(Debug, Clone, Copy)]
struct TlbEntry {
    /// Page-table root the translation was performed against.
    /// Mismatch with the next walk's `cr3_pa` invalidates the entry.
    cr3_pa: u64,
    /// 5-level paging flag (x86 LA57). Mismatch invalidates the
    /// entry — a PML5-rooted walk does not match a PML4-rooted walk
    /// even when the top-level table PA happens to coincide.
    l5: bool,
    /// Cached aarch64 TCR_EL1 (granule + T1SZ + DS bits). Mismatch
    /// invalidates the entry — different TG1/T1SZ produce different
    /// translations for the same KVA. Always 0 on x86_64 where the
    /// register does not exist; gating still costs nothing because
    /// every translate passes the same constant 0.
    tcr_el1: u64,
    /// Page-aligned KVA (low 12 bits cleared) the entry covers.
    /// Subsequent translates of any KVA in `[kva_page, kva_page +
    /// 4 KiB)` reuse the cached PA without walking.
    kva_page: u64,
    /// Page-aligned PA the cached KVA's page resolved to.
    pa_page: u64,
}

// SAFETY: `MemRegion::host_ptr` values point into KVM mmap'd
// regions whose lifetime is guaranteed by `ReservationGuard`. Reads
// and writes use volatile ops; concurrent access is acceptable
// because the monitor is a best-effort sampler of guest-owned data.
unsafe impl Send for GuestMem {}
unsafe impl Sync for GuestMem {}

impl GuestMem {
    /// Wrap the host-mapped guest DRAM region.
    ///
    /// # Safety
    ///
    /// `base` must point to the start of a valid, readable memory
    /// mapping at least `size` bytes long. The mapping MUST outlive
    /// every `GuestMem` access (the type holds no lifetime tying
    /// itself to the backing allocation). The caller is also
    /// responsible for ensuring concurrent writers do not shrink
    /// the mapping out from under the reader (e.g. via `ftruncate`
    /// on an underlying SHM fd).
    ///
    /// Marked `unsafe` because the raw-pointer contract is not
    /// expressible in the type system — `base: *mut u8` could be
    /// dangling, null, or into unmapped memory, and every subsequent
    /// `read_u64` / `read_slice` would miscompute or SIGSEGV. Every
    /// internal caller (including tests) constructs `base` from a
    /// live allocation whose lifetime is proven at the call site.
    ///
    /// # Memory ordering
    ///
    /// Reads go through `std::ptr::read_volatile` (see
    /// `read_volatile_bytes`), which
    /// disables compiler reordering and caching of the load but
    /// provides no hardware fence. Consequently, `GuestMem` offers
    /// no happens-before relationship with guest-side writes: a
    /// reader may observe torn writes, stale values, or partial
    /// updates from a concurrent guest mutator. Callers that
    /// require atomic snapshot semantics (e.g. double-check a
    /// CRC or re-read to confirm a stable value) must layer that
    /// logic themselves.
    pub unsafe fn new(base: *mut u8, size: u64) -> Self {
        Self {
            size,
            regions: vec![MemRegion {
                host_ptr: base,
                offset: 0,
                size,
            }],
            page_tlb: std::sync::Mutex::new(None),
        }
    }

    /// Test-only constructor: build a `GuestMem` from explicit
    /// `MemRegion` entries.
    ///
    /// `regions` must be sorted by ascending `offset` and each region's
    /// host pointer must address a live, readable mapping of at least
    /// `region.size` bytes. The reported `size()` is the largest
    /// `offset + size` across all regions (the DRAM-relative end).
    /// The lifetime of every backing mapping must outlive the
    /// returned `GuestMem`.
    ///
    /// # Safety
    /// Same constraints as [`GuestMem::new`], applied per region.
    #[cfg(test)]
    pub(crate) unsafe fn from_regions_for_test(regions: Vec<MemRegion>) -> Self {
        assert!(!regions.is_empty(), "at least one region required");
        let size = regions
            .iter()
            .map(|r| r.offset + r.size)
            .max()
            .expect("non-empty");
        Self {
            size,
            regions,
            page_tlb: std::sync::Mutex::new(None),
        }
    }

    /// Build a multi-region GuestMem from a NUMA memory layout.
    ///
    /// Each `NodeRegion` in the layout becomes a `MemRegion` with its
    /// host pointer resolved via `GuestMemoryMmap::get_host_address`.
    /// DRAM-relative offsets are computed by subtracting
    /// `layout.dram_base()` from each region's `gpa_start`.
    ///
    /// # Panics
    ///
    /// Panics if `get_host_address` fails for any region in the
    /// layout. This indicates the `GuestMemoryMmap` was not built
    /// from the same layout — a programming error in the caller.
    pub(crate) fn from_layout(
        layout: &crate::vmm::numa_mem::NumaMemoryLayout,
        guest_mem: &vm_memory::GuestMemoryMmap,
    ) -> Self {
        use vm_memory::GuestMemory;

        let dram_base = layout.dram_base();
        let total_size = layout.total_bytes();
        let mut regions = Vec::with_capacity(layout.regions().len());
        for nr in layout.regions() {
            let host_ptr = guest_mem
                .get_host_address(vm_memory::GuestAddress(nr.gpa_start))
                .unwrap();
            regions.push(MemRegion {
                host_ptr,
                offset: nr.gpa_start - dram_base,
                size: nr.size,
            });
        }

        Self {
            size: total_size,
            regions,
            page_tlb: std::sync::Mutex::new(None),
        }
    }

    /// Public host-pointer accessor for a DRAM-relative PA, returning
    /// `None` when the PA is out of bounds or the field would straddle
    /// a multi-region boundary.
    ///
    /// Used by callers that need a raw pointer for `read_volatile` /
    /// `write_volatile` outside the regular `read_*` / `write_*`
    /// helpers — most notably the failure-dump watchpoint, which
    /// caches a `*const u32` on the `WatchpointArm` so the vCPU
    /// thread can read `exit_kind` post-store without owning its own
    /// `GuestMem` clone. `field_size` is the width the caller will
    /// access; the returned pointer is rejected when it would walk
    /// past the resolved region's mmap, mirroring the bounds-check
    /// `read_scalar` performs.
    ///
    /// SAFETY of the returned pointer: valid for the lifetime of
    /// the underlying mapping (`ReservationGuard` outlives every
    /// `GuestMem` access). Callers MUST NOT cache the pointer past
    /// the VM run; cross-thread use requires the caller's own
    /// publication ordering (`AtomicPtr` Release/Acquire pair).
    pub fn host_ptr_for_pa(&self, pa: u64, field_size: u64) -> Option<*mut u8> {
        let end = pa.checked_add(field_size)?;
        if end > self.size {
            return None;
        }
        let (ptr, region_avail) = self.resolve_ptr(pa)?;
        if field_size > region_avail {
            return None;
        }
        Some(ptr)
    }

    /// Bytes remaining in the [`MemRegion`] that contains `pa`, from
    /// `pa` to that region's end. Returns 0 when `pa` falls outside
    /// every region (out-of-bounds PA or in a hole between regions —
    /// `from_layout` lays out one [`MemRegion`] per NUMA node, which
    /// can leave gaps in DRAM-relative space if the layout reserves
    /// holes).
    ///
    /// Callers that issue bulk reads / writes against a contiguous
    /// PA span use this to cap their per-call length so the underlying
    /// [`Self::read_bytes`] / [`Self::write_bytes`] does not silently
    /// short-return at a region boundary. The bulk methods clamp to
    /// the resolved region's `region_avail` themselves; this accessor
    /// lets callers detect the boundary up front and split their work
    /// across regions instead of treating the short return as an
    /// error.
    pub(crate) fn region_avail(&self, pa: u64) -> u64 {
        match self.resolve_ptr(pa) {
            Some((_, avail)) => avail,
            None => 0,
        }
    }

    /// Resolve a DRAM-relative byte offset to a host pointer plus the
    /// number of bytes remaining in the resolved region (i.e. how far
    /// the returned pointer can be advanced before leaving the mmap).
    ///
    /// Binary-searches the sorted region list. Returns `None` if the
    /// offset falls outside all regions.
    fn resolve_ptr(&self, offset: u64) -> Option<(*mut u8, u64)> {
        let idx = self
            .regions
            .partition_point(|r| r.offset <= offset)
            .checked_sub(1)?;
        let r = &self.regions[idx];
        let local = offset - r.offset;
        if local < r.size {
            // SAFETY: `local < r.size` ensures the offset is within
            // the mmap'd region that `host_ptr` points to.
            let ptr = unsafe { r.host_ptr.add(local as usize) };
            Some((ptr, r.size - local))
        } else {
            None
        }
    }

    /// Read `N` volatile bytes from `ptr`. The compiler must not
    /// cache or elide across the read (the guest writes to this
    /// memory concurrently and those writes are invisible to Rust's
    /// model). Returning a `[u8; N]` lets callers recompose the
    /// fundamental integer via `from_ne_bytes` without needing the
    /// caller's pointer alignment to match.
    ///
    /// Performance: when `N == 1, 2, 4, or 8` AND `ptr` is naturally
    /// aligned for that width, this issues a single
    /// `read_volatile::<uN>` instead of `N` per-byte volatiles. Misaligned
    /// pointers and other widths fall back to the per-byte loop —
    /// `read_volatile` requires natural alignment per the std
    /// contract, so misaligned u64 access would be UB (the original
    /// per-byte loop trades performance for portability across all
    /// alignments). Per-byte volatile reads have 1-byte alignment
    /// and are always safe; the fast paths preserve this guarantee
    /// by gating on `ptr.align_offset` returning 0.
    ///
    /// The fast path matters because the freeze coordinator runs
    /// dozens of `read_u64` calls per page-walk and per task in the
    /// dump path; reducing 8 volatile reads to 1 cuts that hot-path
    /// cost by ~8x. Most reads go through `host_ptr` plus a
    /// statically-known struct offset, so the alignment check
    /// resolves at runtime in the common case.
    ///
    /// # Safety
    /// `ptr..ptr+N` must be a valid, readable range in the mapped
    /// guest region. The caller (`read_u32`/`read_u64`/etc.) bounds
    /// checks before resolving the pointer.
    #[inline]
    unsafe fn read_volatile_bytes<const N: usize>(ptr: *const u8) -> [u8; N] {
        // Width-specific fast paths. The const arms compile-time
        // resolve so a non-matching `N` collapses straight to the
        // per-byte fallback with no branch in the generated code.
        match N {
            1 => {
                // SAFETY: 1-byte alignment is universal; ptr..ptr+1
                // is in-bounds per caller's check.
                let v = unsafe { std::ptr::read_volatile(ptr) };
                let mut bytes = [0u8; N];
                bytes[0] = v;
                return bytes;
            }
            2 => {
                if ptr.align_offset(std::mem::align_of::<u16>()) == 0 {
                    // SAFETY: align_offset == 0 proves ptr is u16-
                    // aligned; ptr..ptr+2 is in-bounds per caller's
                    // check; read_volatile::<u16> reads exactly 2
                    // bytes natively.
                    let v: u16 = unsafe { std::ptr::read_volatile(ptr as *const u16) };
                    let src = v.to_ne_bytes();
                    let mut bytes = [0u8; N];
                    bytes[..2].copy_from_slice(&src);
                    return bytes;
                }
            }
            4 => {
                if ptr.align_offset(std::mem::align_of::<u32>()) == 0 {
                    // SAFETY: align_offset == 0 proves ptr is u32-
                    // aligned; ptr..ptr+4 is in-bounds per caller's
                    // check.
                    let v: u32 = unsafe { std::ptr::read_volatile(ptr as *const u32) };
                    let src = v.to_ne_bytes();
                    let mut bytes = [0u8; N];
                    bytes[..4].copy_from_slice(&src);
                    return bytes;
                }
            }
            8 => {
                if ptr.align_offset(std::mem::align_of::<u64>()) == 0 {
                    // SAFETY: align_offset == 0 proves ptr is u64-
                    // aligned; ptr..ptr+8 is in-bounds per caller's
                    // check.
                    let v: u64 = unsafe { std::ptr::read_volatile(ptr as *const u64) };
                    let src = v.to_ne_bytes();
                    let mut bytes = [0u8; N];
                    bytes[..8].copy_from_slice(&src);
                    return bytes;
                }
            }
            _ => {}
        }
        // Fallback: per-byte volatile read. Always correct
        // regardless of alignment, but slow on the hot path.
        // Used for misaligned scalar reads and non-power-of-two
        // widths.
        let mut bytes = [0u8; N];
        for (i, slot) in bytes.iter_mut().enumerate() {
            // SAFETY: ptr..ptr+N is in-bounds per caller's check.
            *slot = unsafe { std::ptr::read_volatile(ptr.add(i)) };
        }
        bytes
    }

    /// Write `N` volatile bytes to `ptr`. Mirror of
    /// [`read_volatile_bytes`] for the store path, including the
    /// width-aligned fast paths.
    ///
    /// # Safety
    /// `ptr..ptr+N` must be a valid, writable range in the mapped
    /// guest region.
    #[inline]
    unsafe fn write_volatile_bytes<const N: usize>(ptr: *mut u8, bytes: [u8; N]) {
        match N {
            1 => {
                // SAFETY: 1-byte alignment is universal.
                unsafe { std::ptr::write_volatile(ptr, bytes[0]) };
                return;
            }
            2 => {
                if ptr.align_offset(std::mem::align_of::<u16>()) == 0 {
                    let mut le = [0u8; 2];
                    le.copy_from_slice(&bytes[..2]);
                    let v = u16::from_ne_bytes(le);
                    // SAFETY: alignment proven by align_offset == 0.
                    unsafe { std::ptr::write_volatile(ptr as *mut u16, v) };
                    return;
                }
            }
            4 => {
                if ptr.align_offset(std::mem::align_of::<u32>()) == 0 {
                    let mut le = [0u8; 4];
                    le.copy_from_slice(&bytes[..4]);
                    let v = u32::from_ne_bytes(le);
                    // SAFETY: alignment proven.
                    unsafe { std::ptr::write_volatile(ptr as *mut u32, v) };
                    return;
                }
            }
            8 => {
                if ptr.align_offset(std::mem::align_of::<u64>()) == 0 {
                    let mut le = [0u8; 8];
                    le.copy_from_slice(&bytes[..8]);
                    let v = u64::from_ne_bytes(le);
                    // SAFETY: alignment proven.
                    unsafe { std::ptr::write_volatile(ptr as *mut u64, v) };
                    return;
                }
            }
            _ => {}
        }
        for (i, &byte) in bytes.iter().enumerate() {
            // SAFETY: ptr..ptr+N is in-bounds per caller's check.
            unsafe { std::ptr::write_volatile(ptr.add(i), byte) };
        }
    }

    /// Bounds-checked volatile read of `N` little/native-endian bytes at
    /// DRAM offset `pa + offset`. Returns `[0; N]` if the range falls
    /// outside the mapped region, straddles a region boundary in a
    /// multi-region (NUMA) layout, or if the address arithmetic overflows
    /// (`pa` may be derived from an attacker-controlled guest page-table
    /// entry).
    ///
    /// `N` must match the width of the scalar caller. `read_volatile_bytes`
    /// reads byte-by-byte, so the access does not require `N`-alignment.
    fn read_scalar<const N: usize>(&self, pa: u64, offset: usize) -> [u8; N] {
        let Some(addr) = pa.checked_add(offset as u64) else {
            return [0; N];
        };
        let Some(end) = addr.checked_add(N as u64) else {
            return [0; N];
        };
        if end > self.size {
            return [0; N];
        }
        match self.resolve_ptr(addr) {
            Some((ptr, region_avail)) => {
                // Reject reads that would walk past the end of the
                // resolved region's mmap. Multi-region GuestMems can
                // have non-contiguous host mappings; reading off the
                // end of one region's mmap is undefined behavior even
                // if `addr + N <= self.size` overall.
                if (N as u64) > region_avail {
                    return [0; N];
                }
                // SAFETY: bounds checked above; resolve_ptr returned a
                // valid pointer into the mapped region and the read of
                // N bytes stays within `region_avail`.
                unsafe { Self::read_volatile_bytes::<N>(ptr as *const u8) }
            }
            None => [0; N],
        }
    }

    /// Bounds-checked volatile write of `bytes` at DRAM offset `pa + offset`.
    /// Silently no-ops if the range falls outside the mapped region,
    /// straddles a region boundary in a multi-region (NUMA) layout, or if
    /// the address arithmetic overflows (`pa` may be derived from an
    /// attacker-controlled guest page-table entry).
    fn write_scalar<const N: usize>(&self, pa: u64, offset: usize, bytes: [u8; N]) {
        let Some(addr) = pa.checked_add(offset as u64) else {
            return;
        };
        let Some(end) = addr.checked_add(N as u64) else {
            return;
        };
        if end > self.size {
            return;
        }
        if let Some((ptr, region_avail)) = self.resolve_ptr(addr) {
            // Reject writes that would walk past the end of the
            // resolved region's mmap (see `read_scalar` for the
            // rationale on multi-region GuestMems).
            if (N as u64) > region_avail {
                return;
            }
            // SAFETY: bounds checked above; the write of N bytes stays
            // within `region_avail`.
            unsafe { Self::write_volatile_bytes::<N>(ptr, bytes) };
        }
    }

    /// Read a u8 at DRAM offset `pa + offset`.
    pub fn read_u8(&self, pa: u64, offset: usize) -> u8 {
        u8::from_ne_bytes(self.read_scalar::<1>(pa, offset))
    }

    /// Read a u32 at DRAM offset `pa + offset`.
    pub fn read_u32(&self, pa: u64, offset: usize) -> u32 {
        u32::from_ne_bytes(self.read_scalar::<4>(pa, offset))
    }

    /// Read a u64 at DRAM offset `pa + offset`.
    pub fn read_u64(&self, pa: u64, offset: usize) -> u64 {
        u64::from_ne_bytes(self.read_scalar::<8>(pa, offset))
    }

    /// Read an i64 at DRAM offset `pa + offset`.
    pub fn read_i64(&self, pa: u64, offset: usize) -> i64 {
        self.read_u64(pa, offset) as i64
    }

    /// Write a u8 at DRAM offset `pa + offset`. Currently exercised
    /// only by unit tests that round-trip `read_u8` against the
    /// shared `write_scalar` path; production no longer writes any
    /// host→guest byte through this method (the SysRq-D dump
    /// signal moved to the virtio-console wake byte).
    #[allow(dead_code)]
    pub fn write_u8(&self, pa: u64, offset: usize, val: u8) {
        self.write_scalar::<1>(pa, offset, val.to_ne_bytes());
    }

    /// Write a u64 at DRAM offset `pa + offset`.
    pub fn write_u64(&self, pa: u64, offset: usize, val: u64) {
        self.write_scalar::<8>(pa, offset, val.to_ne_bytes());
    }

    /// Write `data` into DRAM starting at offset `pa`. Bounds-clipped
    /// the same way as [`Self::read_bytes`]: writes that would
    /// extend past the end of the resolved region are truncated.
    /// Used by the freeze coordinator's snapshot doorbell handler
    /// to stamp the reply reason buffer (UTF-8 NUL-terminated) into
    /// the guest's SHM tag area when a CAPTURE / WATCH request
    /// fails. Returns the number of bytes actually written.
    pub fn write_bytes(&self, pa: u64, data: &[u8]) -> usize {
        let len = data.len() as u64;
        if pa >= self.size {
            return 0;
        }
        let avail = (self.size - pa).min(len) as usize;
        match self.resolve_ptr(pa) {
            Some((ptr, region_avail)) => {
                let copy_len = avail.min(region_avail as usize);
                // SAFETY: `copy_len <= region_avail`, so the write
                // stays within the mmap that `ptr` points into. The
                // mapping is MAP_SHARED with the guest.
                //
                // `copy_nonoverlapping` is NOT a volatile store and
                // does not by itself synchronize the bytes with a
                // guest reader: the host compiler is free to reorder
                // it with surrounding accesses, and there is no
                // hardware fence. Production callers
                // (`vmm::freeze_coord::write_snapshot_reply`)
                // publish these bytes by following the bulk write
                // with an `std::sync::atomic::fence(Ordering::Release)`
                // and a volatile `write_u32` of the reply id; the
                // guest acquires via the matching volatile read of
                // the reply id, so the bulk bytes are observable
                // once the reply id transitions. Direct callers
                // without that downstream publish MUST use the
                // volatile `write_u32` / `write_u64` scalar helpers
                // instead.
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, copy_len);
                }
                copy_len
            }
            None => 0,
        }
    }

    /// Write `data` into DRAM starting at offset `pa + offset`.
    ///
    /// Same bounds-clipping contract as [`Self::write_bytes`]:
    /// writes that would extend past the end of the resolved region
    /// are truncated and the returned count reflects what actually
    /// landed. The bytes go through `copy_nonoverlapping`, NOT a
    /// volatile store — observability to the guest comes from a
    /// downstream publish (atomic fence + volatile u32) on the
    /// caller side; see [`Self::write_bytes`] for the production
    /// example.
    ///
    /// The `offset` parameter is preserved as a kernel-friendly API:
    /// every translated guest PA is paired with an in-page offset on
    /// the read path ([`Self::read_bytes`] callers add `offset` to
    /// `pa` themselves), and `checked_add` here rejects callers
    /// whose `pa + offset` would wrap. The current production caller
    /// ([`super::bpf_map::write_bpf_map_value`] via [`super::bpf_map::chunked_kva_io`])
    /// passes `offset=0` because the chunked walker already returns
    /// per-page PAs; future callers that already hold a base PA and
    /// want to splice in a field offset can pass non-zero without
    /// rewriting the wraparound guard.
    pub fn write_bytes_at(&self, pa: u64, offset: usize, data: &[u8]) -> usize {
        let Some(addr) = pa.checked_add(offset as u64) else {
            return 0;
        };
        self.write_bytes(addr, data)
    }

    /// Read `len` bytes from DRAM offset `pa` into `buf`.
    /// Returns the number of bytes actually read (may be less than `len`
    /// if the read would go past the end of guest memory or the end of
    /// the resolved region — multi-region NUMA layouts can have
    /// non-contiguous host mappings, so the copy must not extend past
    /// the region containing `pa`).
    pub fn read_bytes(&self, pa: u64, buf: &mut [u8]) -> usize {
        let len = buf.len() as u64;
        if pa >= self.size {
            return 0;
        }
        let avail = (self.size - pa).min(len) as usize;
        match self.resolve_ptr(pa) {
            Some((ptr, region_avail)) => {
                let copy_len = avail.min(region_avail as usize);
                // SAFETY: `copy_len <= region_avail`, so the read stays
                // within the mmap that `ptr` points into.
                unsafe {
                    std::ptr::copy_nonoverlapping(ptr, buf.as_mut_ptr(), copy_len);
                }
                copy_len
            }
            None => 0,
        }
    }

    /// Write a u32 at DRAM offset `pa + offset`. Production callers
    /// retired with the snapshot-doorbell TLV migration; preserved for
    /// the `write_scalar::<4>` boundary tests below and external test
    /// fixtures (`bpf_map`, `dump`) that splice u32 fields into mock
    /// guest memory.
    #[allow(dead_code)]
    pub fn write_u32(&self, pa: u64, offset: usize, val: u32) {
        self.write_scalar::<4>(pa, offset, val.to_ne_bytes());
    }

    /// Translate a kernel virtual address to guest physical address via
    /// page table walk.
    ///
    /// x86-64: supports 4-level (PGD -> PUD -> PMD -> PTE) and 5-level
    /// (PML5 -> P4D -> PUD -> PMD -> PTE) paging. `tcr_el1` is ignored.
    ///
    /// aarch64: granule-agnostic walker driven by TCR_EL1.TG1
    /// (4 KB / 16 KB / 64 KB) and TCR_EL1.T1SZ (high-half VA size).
    /// Handles block descriptors at intermediate levels (huge pages).
    /// `l5` is ignored.
    ///
    /// `cr3_pa` is the physical address of the top-level page table.
    /// `l5` selects 5-level paging (x86 LA57); use `resolve_pgtable_l5`
    /// to detect the guest's mode at runtime. `tcr_el1` is the
    /// guest's TCR_EL1 register (aarch64 only); read it via
    /// `KVM_GET_ONE_REG` once at coordinator start.
    /// Returns `None` if any level is not present or the address is
    /// out of guest memory bounds.
    ///
    /// Successful translations populate the single-slot software TLB
    /// (`page_tlb`) so the next translate of any KVA in the same
    /// 4 KiB page returns without walking. The TLB caches only
    /// successful walks — an unmapped KVA does not pollute the slot,
    /// so a re-probe of the same address still hits the page-table
    /// walker. Cross-`cr3_pa` access invalidates the slot implicitly
    /// (the entry's `cr3_pa` field is compared on lookup); the
    /// `l5` and `tcr_el1` walk-mode inputs are also part of the
    /// cache key so a switch between 4-level and 5-level paging
    /// (or a hypothetical aarch64 TCR_EL1 change) cannot return a
    /// translation built under a different walk configuration.
    pub(crate) fn translate_kva(
        &self,
        cr3_pa: u64,
        kva: Kva,
        l5: bool,
        tcr_el1: u64,
    ) -> Option<u64> {
        // 4 KiB software TLB: caches at the lowest sub-granule
        // common to every walker. The aarch64 walker's smallest
        // leaf is 16 KiB or 64 KiB depending on TCR_EL1.TG1, and
        // the x86 walker's smallest leaf is 4 KiB; caching at 4 KiB
        // is safe for both because any KVA in a larger leaf
        // projects 1:1 into the 4 KiB-aligned slice with the same
        // PA-to-KVA mapping (the leaf is contiguous PA-wise). 2 MiB
        // / 1 GiB block descriptors and the 16 KiB / 64 KiB
        // aarch64 granules all satisfy this property, so the
        // single-slot cache is correct even when the underlying
        // PTE covers more bytes than the cache key.
        const PAGE: u64 = 4096;
        let kva_bits = kva.0;
        let kva_page = kva_bits & !(PAGE - 1);
        // Recover from poison rather than silently bypassing the
        // cache: matches the `maps_cache` /
        // `per_cpu_offsets_cache` mutex handling in
        // [`super::bpf_map`]. The TLB is a pure cache of already-
        // computed PAs; a panic in a previous holder doesn't
        // corrupt the entry's bytes, and bypass-on-poison would
        // silently drop every subsequent hit.
        {
            let guard = self.page_tlb.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(entry) = guard.as_ref()
                && entry.cr3_pa == cr3_pa
                && entry.l5 == l5
                && entry.tcr_el1 == tcr_el1
                && entry.kva_page == kva_page
            {
                return Some(entry.pa_page | (kva_bits & (PAGE - 1)));
            }
        }

        let walk_result;
        #[cfg(target_arch = "x86_64")]
        {
            let _ = tcr_el1; // aarch64-only register; ignored on x86_64
            walk_result = if l5 {
                self.walk_5level(cr3_pa, kva)
            } else {
                self.walk_4level(cr3_pa, kva)
            };
        }
        #[cfg(target_arch = "aarch64")]
        {
            let _ = l5; // x86-only flag; aarch64 reads TCR_EL1 instead
            walk_result = self.walk_aarch64(cr3_pa, kva, tcr_el1);
        }

        if let Some(pa) = walk_result {
            let mut guard = self.page_tlb.lock().unwrap_or_else(|e| e.into_inner());
            *guard = Some(TlbEntry {
                cr3_pa,
                l5,
                tcr_el1,
                kva_page,
                pa_page: pa & !(PAGE - 1),
            });
        }
        walk_result
    }

    /// Translate a KVA using a pre-decoded [`Aarch64WalkParams`] on
    /// aarch64; uses the standard 4-level / 5-level walk on x86_64.
    ///
    /// The cached caller (`super::guest::GuestKernel`'s
    /// `read_kva_*` helpers) builds the params once at construction
    /// from the guest's TCR_EL1 and passes them on every translate
    /// to elide the per-call `T1SZ`/`TG1` decode. On x86_64 the
    /// `_params` argument is ignored; the path exists for cross-
    /// arch callers that build the cache once and want a single
    /// translate API. Callers without a cache should use
    /// [`Self::translate_kva`] which decodes from `tcr_el1` per
    /// call.
    #[allow(dead_code)]
    pub(crate) fn translate_kva_with_aarch64_params(
        &self,
        cr3_pa: u64,
        kva: Kva,
        l5: bool,
        params: &Aarch64WalkParams,
    ) -> Option<u64> {
        // Same 4 KiB software TLB sub-granule as `translate_kva`:
        // any larger leaf entry projects 1:1 into 4 KiB-aligned
        // slices with the same PA-to-KVA mapping. The TCR_EL1
        // captured at params-construction time is used as the cache
        // key alongside `cr3_pa` and `l5` so a translation
        // performed under one walk configuration cannot be reused
        // under a different one.
        const PAGE: u64 = 4096;
        let kva_bits = kva.0;
        let kva_page = kva_bits & !(PAGE - 1);
        // Recover from poison rather than silently bypass — see
        // [`Self::translate_kva`] for the rationale.
        {
            let guard = self.page_tlb.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(entry) = guard.as_ref()
                && entry.cr3_pa == cr3_pa
                && entry.l5 == l5
                && entry.tcr_el1 == params.tcr_el1
                && entry.kva_page == kva_page
            {
                return Some(entry.pa_page | (kva_bits & (PAGE - 1)));
            }
        }

        let walk_result;
        #[cfg(target_arch = "x86_64")]
        {
            // params.tcr_el1 is meaningful only on aarch64; on x86
            // the field is still part of the cache key (always 0)
            // and the walker itself ignores it.
            let _ = params;
            walk_result = if l5 {
                self.walk_5level(cr3_pa, kva)
            } else {
                self.walk_4level(cr3_pa, kva)
            };
        }
        #[cfg(target_arch = "aarch64")]
        {
            let _ = l5;
            walk_result = self.walk_aarch64_with_params(cr3_pa, kva, params);
        }

        if let Some(pa) = walk_result {
            let mut guard = self.page_tlb.lock().unwrap_or_else(|e| e.into_inner());
            *guard = Some(TlbEntry {
                cr3_pa,
                l5,
                tcr_el1: params.tcr_el1,
                kva_page,
                pa_page: pa & !(PAGE - 1),
            });
        }
        walk_result
    }

    /// 4-level page table walk (x86-64).
    ///
    /// CR3 -> PGD -> PUD -> PMD -> PTE. Uses PS bit (bit 7) for
    /// huge pages, OA in bits \[51:12\].
    #[cfg(target_arch = "x86_64")]
    fn walk_4level(&self, cr3_pa: u64, kva: Kva) -> Option<u64> {
        const PRESENT: u64 = 1;
        const PS: u64 = 1 << 7;
        const ADDR_MASK: u64 = 0x000F_FFFF_FFFF_F000;

        let kva_bits = kva.0;
        let pgd_idx = (kva_bits >> 39) & 0x1FF;
        let pud_idx = (kva_bits >> 30) & 0x1FF;
        let pmd_idx = (kva_bits >> 21) & 0x1FF;
        let pte_idx = (kva_bits >> 12) & 0x1FF;
        let page_off = kva_bits & 0xFFF;

        // PGD
        let pgd_pa = (cr3_pa & ADDR_MASK) + pgd_idx * 8;
        let pgde = self.read_u64(pgd_pa, 0);
        if pgde & PRESENT == 0 {
            return None;
        }

        // PUD
        let pud_pa = (pgde & ADDR_MASK) + pud_idx * 8;
        let pude = self.read_u64(pud_pa, 0);
        if pude & PRESENT == 0 {
            return None;
        }
        if pude & PS != 0 {
            let base = pude & 0x000F_FFFF_C000_0000;
            return Some(base | (kva_bits & 0x3FFF_FFFF));
        }

        // PMD
        let pmd_pa = (pude & ADDR_MASK) + pmd_idx * 8;
        let pmde = self.read_u64(pmd_pa, 0);
        if pmde & PRESENT == 0 {
            return None;
        }
        if pmde & PS != 0 {
            let base = pmde & 0x000F_FFFF_FFE0_0000;
            return Some(base | (kva_bits & 0x1F_FFFF));
        }

        // PTE
        let pte_pa = (pmde & ADDR_MASK) + pte_idx * 8;
        let ptee = self.read_u64(pte_pa, 0);
        if ptee & PRESENT == 0 {
            return None;
        }

        Some((ptee & ADDR_MASK) | page_off)
    }

    /// aarch64 page table walk, granule-agnostic.
    ///
    /// Reads TCR_EL1.TG1 to select the granule (4 KB / 16 KB / 64 KB)
    /// and TCR_EL1.T1SZ for the high-half VA size, then iterates the
    /// translation tables with the matching stride/level configuration.
    /// Handles block descriptors at intermediate levels for huge pages.
    ///
    /// TG1 encoding (TCR_EL1[31:30], distinct from TG0[15:14]):
    /// `0b01` = 16 KB (stride 11), `0b10` = 4 KB (stride 9),
    /// `0b11` = 64 KB (stride 13). `0b00` is reserved per Arm ARM
    /// D17.2.139 and the walker rejects it as unmapped. T1SZ in bits
    /// [21:16]; VA width = `64 - T1SZ`. Starting level computed as
    /// `4 - (va_width - 4) / stride` — the bottom of the descriptor
    /// cascade is always level 3.
    ///
    /// Descriptor format (ARMv8 D5.3):
    /// - bits [1:0] = 0b00: invalid
    /// - bits [1:0] = 0b01: block descriptor (intermediate levels) or
    ///   reserved (level 3, treated as invalid)
    /// - bits [1:0] = 0b11: table descriptor (intermediate levels) or
    ///   page descriptor (level 3)
    /// - OA layout: the walker recovers low OA bits [49:granule_log2]
    ///   via `descaddrmask` (matching the kernel's `PTE_ADDR_LOW`).
    ///   On non-LPA / non-LPA2 kernels bits [49:48] are RES0 by
    ///   hardware so the practical OA range is [47:granule_log2].
    ///   No high-OA splice is applied. FEAT_LPA2 (`TCR_EL1.DS=1`)
    ///   is rejected by [`Aarch64WalkParams::from_tcr_el1`] because
    ///   it requires the [9:8]→[51:50] splice; FEAT_LPA on the 64
    ///   KiB granule is undetectable from `TCR_EL1` alone but
    ///   ktstr.kconfig does not enable `CONFIG_ARM64_PA_BITS_52`,
    ///   so the assumption holds for the kernel under test.
    ///
    /// Page table entries contain guest physical addresses (GPAs).
    /// Since GuestMem offsets are DRAM-relative on aarch64 (DRAM_BASE
    /// at offset 0), every GPA is adjusted by `checked_sub(DRAM_START)`
    /// before use — descriptors whose payloads fall below DRAM_START
    /// (corrupt or attacker-controlled) are rejected as "not present"
    /// rather than silently wrapping to near-`u64::MAX`.
    ///
    /// `ttbr_pa` is the DRAM-relative offset of the top-level table
    /// (typically `text_kva_to_pa_with_base(swapper_pg_dir_kva,
    /// start_kernel_map)` for the kernel-half pgd at boot).
    ///
    /// FEAT_LPA / FEAT_LPA2 / 52-bit VA layouts are NOT supported.
    /// LPA2 is rejected up front (see `from_tcr_el1`); the others
    /// require kconfig opt-ins (`CONFIG_ARM64_PA_BITS_52`,
    /// `CONFIG_ARM64_VA_BITS_52`) that ktstr.kconfig does not enable.
    /// A future user that pins those configs must extend this walker
    /// with the high-OA splice (PTE_ADDR_HIGH bits [15:12] for
    /// FEAT_LPA on 64 KiB pages, [9:8] for FEAT_LPA2 on 4 KiB / 16
    /// KiB) before the monitor reads guest page tables.
    #[cfg(target_arch = "aarch64")]
    fn walk_aarch64(&self, ttbr_pa: u64, kva: Kva, tcr_el1: u64) -> Option<u64> {
        // Decode-once-per-call entry: every public translate path
        // funnels through here. Bookkeeping for the cached path
        // lives in [`Aarch64WalkParams::from_tcr_el1`], whose
        // failure modes match the historical short-circuits this
        // function previously inlined (T1SZ=0, reserved TG1, etc.).
        let params = Aarch64WalkParams::from_tcr_el1(tcr_el1)?;
        self.walk_aarch64_with_params(ttbr_pa, kva, &params)
    }

    /// Aarch64 page-table walk that consumes a pre-decoded
    /// [`Aarch64WalkParams`] instead of decoding `TCR_EL1` per call.
    /// The cached caller (`super::guest::GuestKernel`'s
    /// `read_kva_*` helpers) builds the params once at construction
    /// and reuses them for every translate, eliding the
    /// `T1SZ`/`TG1`/`va_width`/`levels_below`/index-mask math from
    /// the hot path.
    ///
    /// `params.tcr_el1` is ignored by the walk itself; it is only
    /// useful for downstream debugging when the cached value
    /// disagrees with a fresh `tcr_el1` argument (which would
    /// indicate an unexpected TCR change post-boot — see the
    /// `Aarch64WalkParams` doc).
    #[cfg(target_arch = "aarch64")]
    fn walk_aarch64_with_params(
        &self,
        ttbr_pa: u64,
        kva: Kva,
        params: &Aarch64WalkParams,
    ) -> Option<u64> {
        use crate::vmm::aarch64::kvm::DRAM_START;

        let stride = params.stride;
        let mut level: u64 = params.start_level;
        let indexmask_grainsize: u64 = params.indexmask_grainsize;
        let mut indexmask: u64 = params.first_indexmask;
        let descaddrmask: u64 = params.descaddrmask;

        // Translation table base: TTBR1_EL1 bits [47:0] — but the caller
        // already passed a DRAM-relative offset (text_kva_to_pa_with_base
        // of the pgd symbol), so we treat `ttbr_pa` directly as the
        // table's GuestMem offset and only mask off any ASID-style high
        // bits.
        let mut descaddr: u64 = ttbr_pa & ((!0u64) >> (64 - 48));

        // Convert a descriptor's GPA payload to a DRAM-relative offset.
        // `checked_sub` rejects descriptors whose payload addresses fall
        // below DRAM_START — a malicious or corrupted descriptor that
        // wraps would otherwise produce a near-u64::MAX offset and cause
        // out-of-bounds reads. Treat underflow as "not present" (None).
        let to_offset = |gpa: u64| -> Option<u64> { gpa.checked_sub(DRAM_START) };

        let kva_bits = kva.0;

        // Descriptor cascade. Each iteration reads one 8-byte table
        // entry, and either descends (table descriptor) or terminates
        // (block descriptor, page descriptor, or invalid).
        loop {
            let table_offset: u64 = (kva_bits >> (stride * (4 - level))) & indexmask;
            // Compose the byte offset of this level's descriptor inside
            // the table. The `& !7` aligns to 8-byte boundary; on the
            // first iteration `descaddr` is the (DRAM-relative) table
            // base, on subsequent iterations it's already the
            // descriptor's table base extracted from the previous
            // descriptor.
            descaddr |= table_offset;
            descaddr &= !7u64;

            let descriptor = self.read_u64(descaddr, 0);
            // bits [1:0] = 0b00 → invalid; 0b01 at level 3 is
            // reserved (also treated as invalid for kernel-half KVAs).
            if descriptor & 1 == 0 {
                return None;
            }
            // ARMv8 D5.3: at level 3, descriptor bit 1 must be set
            // (0b11 = page descriptor). 0b01 is reserved at the leaf
            // level and would be misread as a "block at level 3" by
            // the descend/leaf logic below — reject it explicitly.
            if level == 3 && (descriptor & 2) == 0 {
                return None;
            }

            // Extract the next-level / output address. `descaddrmask`
            // matches the kernel's PTE_ADDR_LOW (bits [49:granule_log2]),
            // stripping high attribute / SW bits at [63:50] and the
            // low granule bits. FEAT_LPA / FEAT_LPA2 high splices are
            // not applied here; LPA2 is rejected up front via the
            // TCR_EL1.DS gate in `Aarch64WalkParams::from_tcr_el1`,
            // and LPA on 64 KiB pages is undetectable from TCR_EL1
            // (documented limitation).
            let next = descriptor & descaddrmask;

            // Bit 1 distinguishes table (1) from block (0) at
            // intermediate levels; at level 3, bit 1 is always 1
            // (page descriptor) — a leaf entry.
            //
            // Per ARM ARM D5.3, block descriptors are allowed at
            // levels 1 and 2 only — and only for granules where
            // those levels hold output addresses larger than the
            // page granule:
            //   - 4 KiB granule (stride=9): blocks at L1 (1 GiB)
            //     and L2 (2 MiB); L0 is table-only.
            //   - 16 KiB granule (stride=11): blocks at L2 only
            //     (32 MiB); L0 and L1 are table-only.
            //   - 64 KiB granule (stride=13): blocks at L2 only
            //     (512 MiB); L1 is table-only.
            // A descriptor with bit 1 clear at any other level is
            // reserved per ARM ARM; reject rather than risk
            // composing a leaf address from a malformed entry.
            // (At level 3, the earlier check already rejected
            // bit-1-clear as a reserved encoding.)
            let is_block = (descriptor & 2) == 0;
            if is_block && level < 3 {
                let block_legal = match stride {
                    // 4 KiB granule: block descriptors valid at L1, L2.
                    9 => level == 1 || level == 2,
                    // 16 KiB granule: block descriptors valid at L2 only.
                    11 => level == 2,
                    // 64 KiB granule: block descriptors valid at L2 only.
                    13 => level == 2,
                    _ => false,
                };
                if !block_legal {
                    return None;
                }
            }

            if (descriptor & 2) != 0 && level < 3 {
                // Table descriptor — descend. The next iteration will
                // index into the table whose base GPA is `next`,
                // converted to a DRAM-relative offset.
                descaddr = to_offset(next)?;
                level += 1;
                indexmask = indexmask_grainsize;
                continue;
            }

            // Leaf entry: either a block descriptor at level 1 / 2
            // (huge page) or a page descriptor at level 3.
            descaddr = next;
            break;
        }

        // Compose the final guest physical address: page-aligned
        // descriptor output OR'd with the in-page offset of `kva`.
        // `page_size` covers the level's translation granularity:
        // 4 KB at level 3, larger blocks higher up.
        let page_size: u64 = 1u64 << ((stride * (4 - level)) + 3);
        let pa_gpa = (descaddr & !(page_size - 1)) | (kva_bits & (page_size - 1));
        to_offset(pa_gpa)
    }

    /// 5-level page table walk: CR3 -> PML5 -> P4D -> PUD -> PMD -> PTE.
    /// x86-64 only; aarch64 does not use 5-level paging.
    #[cfg(target_arch = "x86_64")]
    fn walk_5level(&self, cr3_pa: u64, kva: Kva) -> Option<u64> {
        const PRESENT: u64 = 1;
        const PS: u64 = 1 << 7;
        const ADDR_MASK: u64 = 0x000F_FFFF_FFFF_F000;

        // PML5 index: bits 56:48.
        let pml5_idx = (kva.0 >> 48) & 0x1FF;

        let pml5_pa = (cr3_pa & ADDR_MASK) + pml5_idx * 8;
        let pml5e = self.read_u64(pml5_pa, 0);
        if pml5e & PRESENT == 0 {
            return None;
        }
        // PML5 PS-bit huge page: 256 TiB region. Intel SDM Vol 3A
        // currently reserves PS=0 at this level, but the bit is
        // architecturally placed where future / non-Intel
        // implementations could enable it. Mirror the PUD/PMD
        // PS-bit handling in `walk_4level` so a future toolchain
        // running on extended-spec hardware doesn't silently mis-
        // descend into walk_4level on a leaf entry.
        //
        // Base mask clears the low 48 bits (256 TiB alignment); kva
        // splices in the low 48 bits as the page offset.
        if pml5e & PS != 0 {
            let base = pml5e & 0x000F_0000_0000_0000;
            return Some(base | (kva.0 & 0x0000_FFFF_FFFF_FFFF));
        }

        // P4D is the next level; continue with 4-level walk from there.
        let p4d_pa = pml5e & ADDR_MASK;
        self.walk_4level(p4d_pa, kva)
    }

    /// Guest memory size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }
}

/// Read scheduler stats from one CPU's struct rq at the given physical address.
///
/// Populates the core rq fields: `nr_running`, `scx_nr_running`,
/// `local_dsq_depth`, `rq_clock`, `scx_flags`. Leaves `event_counters`,
/// `schedstat`, `vcpu_cpu_time_ns`, `vcpu_perf`, and `sched_domains`
/// as `None` — those are filled in separately by `read_event_stats`,
/// `read_rq_schedstat`, vCPU stats collection, perf counter reads,
/// and sched_domain traversal respectively.
pub(crate) fn read_rq_stats(mem: &GuestMem, rq_pa: u64, offsets: &KernelOffsets) -> CpuSnapshot {
    CpuSnapshot {
        nr_running: mem.read_u32(rq_pa, offsets.rq_nr_running),
        scx_nr_running: mem.read_u32(rq_pa, offsets.rq_scx + offsets.scx_rq_nr_running),
        local_dsq_depth: mem.read_u32(
            rq_pa,
            offsets.rq_scx + offsets.scx_rq_local_dsq + offsets.dsq_nr,
        ),
        rq_clock: mem.read_u64(rq_pa, offsets.rq_clock),
        scx_flags: mem.read_u32(rq_pa, offsets.rq_scx + offsets.scx_rq_flags),
        event_counters: None,
        schedstat: None,
        vcpu_cpu_time_ns: None,
        vcpu_perf: None,
        sched_domains: None,
    }
}

/// Read scx event counters from one CPU's per-CPU event stats struct.
/// On 6.18+ (and 6.17.7+ stable), `pcpu_pa` points to `scx_sched_pcpu`;
/// on 6.16 through 6.17.6, it points directly to `scx_event_stats`
/// (`event_stats_off` = 0).
///
/// Version boundaries are approximate; see [`resolve_event_pcpu_pas`]
/// for the detection logic.
pub(crate) fn read_event_stats(
    mem: &GuestMem,
    pcpu_pa: u64,
    ev: &ScxEventOffsets,
) -> ScxEventCounters {
    let base = pcpu_pa + ev.event_stats_off as u64;
    let read_opt = |off: Option<usize>| off.map(|o| mem.read_i64(base, o)).unwrap_or(0);
    ScxEventCounters {
        select_cpu_fallback: mem.read_i64(base, ev.ev_select_cpu_fallback),
        dispatch_local_dsq_offline: mem.read_i64(base, ev.ev_dispatch_local_dsq_offline),
        dispatch_keep_last: mem.read_i64(base, ev.ev_dispatch_keep_last),
        enq_skip_exiting: mem.read_i64(base, ev.ev_enq_skip_exiting),
        enq_skip_migration_disabled: mem.read_i64(base, ev.ev_enq_skip_migration_disabled),
        reenq_immed: read_opt(ev.ev_reenq_immed),
        reenq_local_repeat: read_opt(ev.ev_reenq_local_repeat),
        refill_slice_dfl: read_opt(ev.ev_refill_slice_dfl),
        bypass_duration: read_opt(ev.ev_bypass_duration),
        bypass_dispatch: read_opt(ev.ev_bypass_dispatch),
        bypass_activate: read_opt(ev.ev_bypass_activate),
        insert_not_owned: read_opt(ev.ev_insert_not_owned),
        sub_bypass_dispatch: read_opt(ev.ev_sub_bypass_dispatch),
    }
}

/// Read schedstat fields from one CPU's struct rq at the given physical address.
///
/// Reads CONFIG_SCHEDSTATS counters: `run_delay` and `pcount` from the
/// embedded `sched_info` substruct, plus `yld_count`, `sched_count`,
/// `sched_goidle`, `ttwu_count`, and `ttwu_local` from the rq itself.
pub(crate) fn read_rq_schedstat(mem: &GuestMem, rq_pa: u64, ss: &SchedstatOffsets) -> RqSchedstat {
    let sched_info_pa = rq_pa + ss.rq_sched_info as u64;
    RqSchedstat {
        run_delay: mem.read_u64(sched_info_pa, ss.sched_info_run_delay),
        pcount: mem.read_u64(sched_info_pa, ss.sched_info_pcount),
        yld_count: mem.read_u32(rq_pa, ss.rq_yld_count),
        sched_count: mem.read_u32(rq_pa, ss.rq_sched_count),
        sched_goidle: mem.read_u32(rq_pa, ss.rq_sched_goidle),
        ttwu_count: mem.read_u32(rq_pa, ss.rq_ttwu_count),
        ttwu_local: mem.read_u32(rq_pa, ss.rq_ttwu_local),
    }
}

/// Read a u32 array of `CPU_MAX_IDLE_TYPES` elements from guest memory.
fn read_u32_array(mem: &GuestMem, pa: u64, base_offset: usize) -> [u32; CPU_MAX_IDLE_TYPES] {
    std::array::from_fn(|i| mem.read_u32(pa, base_offset + i * 4))
}

/// Read CONFIG_SCHEDSTATS fields from one sched_domain.
///
/// Each load-balance counter (`lb_*`) is a per-idle-type array with
/// `CPU_MAX_IDLE_TYPES` u32 elements (indexed by `enum cpu_idle_type`);
/// scalar counters (`alb_*`, `sbe_*`, `sbf_*`, `ttwu_*`) are single
/// u32 fields on the sched_domain itself.
fn read_sd_stats(mem: &GuestMem, sd_pa: u64, so: &SchedDomainStatsOffsets) -> SchedDomainStats {
    SchedDomainStats {
        lb_count: read_u32_array(mem, sd_pa, so.sd_lb_count),
        lb_failed: read_u32_array(mem, sd_pa, so.sd_lb_failed),
        lb_balanced: read_u32_array(mem, sd_pa, so.sd_lb_balanced),
        lb_imbalance_load: read_u32_array(mem, sd_pa, so.sd_lb_imbalance_load),
        lb_imbalance_util: read_u32_array(mem, sd_pa, so.sd_lb_imbalance_util),
        lb_imbalance_task: read_u32_array(mem, sd_pa, so.sd_lb_imbalance_task),
        lb_imbalance_misfit: read_u32_array(mem, sd_pa, so.sd_lb_imbalance_misfit),
        lb_gained: read_u32_array(mem, sd_pa, so.sd_lb_gained),
        lb_hot_gained: read_u32_array(mem, sd_pa, so.sd_lb_hot_gained),
        lb_nobusyg: read_u32_array(mem, sd_pa, so.sd_lb_nobusyg),
        lb_nobusyq: read_u32_array(mem, sd_pa, so.sd_lb_nobusyq),
        alb_count: mem.read_u32(sd_pa, so.sd_alb_count),
        alb_failed: mem.read_u32(sd_pa, so.sd_alb_failed),
        alb_pushed: mem.read_u32(sd_pa, so.sd_alb_pushed),
        sbe_count: mem.read_u32(sd_pa, so.sd_sbe_count),
        sbe_balanced: mem.read_u32(sd_pa, so.sd_sbe_balanced),
        sbe_pushed: mem.read_u32(sd_pa, so.sd_sbe_pushed),
        sbf_count: mem.read_u32(sd_pa, so.sd_sbf_count),
        sbf_balanced: mem.read_u32(sd_pa, so.sd_sbf_balanced),
        sbf_pushed: mem.read_u32(sd_pa, so.sd_sbf_pushed),
        ttwu_wake_remote: mem.read_u32(sd_pa, so.sd_ttwu_wake_remote),
        ttwu_move_affine: mem.read_u32(sd_pa, so.sd_ttwu_move_affine),
        ttwu_move_balance: mem.read_u32(sd_pa, so.sd_ttwu_move_balance),
    }
}

/// Read the `sd->name` string from guest memory.
///
/// `sd->name` is a `char *` pointer to a static string in kernel rodata.
/// Rodata lives in the text mapping (`__START_KERNEL_map`), so
/// `text_kva_to_pa_with_base` is tried first. Falls back to direct
/// mapping (`kva_to_pa`) for kernels that place topology name strings
/// differently. `start_kernel_map` is the runtime kernel image base
/// (`__START_KERNEL_map` on x86_64; derived from `TCR_EL1.T1SZ` on
/// aarch64) so VA_BITS=47 hosts read the right rodata bytes.
/// Returns an empty string if the pointer is null or translation
/// fails.
fn read_sd_name(
    mem: &GuestMem,
    sd_pa: u64,
    name_offset: usize,
    page_offset: u64,
    start_kernel_map: u64,
    phys_base: u64,
) -> String {
    let name_kva = mem.read_u64(sd_pa, name_offset);
    if name_kva == 0 {
        return String::new();
    }
    // Try text mapping first (rodata), then direct mapping.
    let text_pa = super::symbols::text_kva_to_pa_with_base(name_kva, start_kernel_map, phys_base);
    let name_pa = if text_pa < mem.size() {
        text_pa
    } else {
        let direct_pa = super::symbols::kva_to_pa(name_kva, page_offset);
        if direct_pa >= mem.size() {
            return String::new();
        }
        direct_pa
    };
    // Domain names are short static strings ("SMT", "MC", "DIE", "NUMA",
    // "PKG", "BOOK", "DRAWER"). Read up to 16 bytes.
    let mut buf = [0u8; 16];
    let n = mem.read_bytes(name_pa, &mut buf);
    let end = buf[..n].iter().position(|&b| b == 0).unwrap_or(n);
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

/// Read the sched_domain tree for one CPU.
///
/// Starts at `rq->sd` (the lowest-level domain), walks `sd->parent`
/// until NULL. Each domain is kmalloc'd and lives in the direct mapping.
///
/// `page_offset` is the runtime `PAGE_OFFSET` for direct-mapping translation;
/// `start_kernel_map` is the runtime kernel image base used by
/// [`read_sd_name`] to translate `sd->name` rodata pointers.
///
/// Returns `None` if `rq->sd` is null (domain not yet built, or CPU
/// offline). Returns an empty `Vec` if the first domain pointer cannot
/// be translated.
///
/// Maximum depth is bounded to 8 levels and a visited-set of domain
/// KVAs breaks `sd->parent` cycles — a corrupted or self-referential
/// chain would otherwise emit the same domain up to MAX_DEPTH times.
pub(crate) fn read_sched_domain_tree(
    mem: &GuestMem,
    rq_pa: u64,
    sd_offsets: &SchedDomainOffsets,
    page_offset: u64,
    start_kernel_map: u64,
    phys_base: u64,
) -> Option<Vec<SchedDomainSnapshot>> {
    const MAX_DEPTH: usize = 8;

    // rq->sd is a pointer (KVA).
    let sd_kva = mem.read_u64(rq_pa, sd_offsets.rq_sd);
    if sd_kva == 0 {
        return None;
    }

    let mut domains = Vec::new();
    let mut current_kva = sd_kva;
    let mut visited: std::collections::HashSet<u64> = std::collections::HashSet::new();

    for _ in 0..MAX_DEPTH {
        if current_kva == 0 {
            break;
        }
        if !visited.insert(current_kva) {
            // Cycle or self-reference: same KVA already emitted. Stop
            // so we do not inflate the tree with duplicate snapshots.
            tracing::warn!(
                sd_kva = format_args!("{current_kva:#x}"),
                "sched_domain cycle detected; truncating tree"
            );
            break;
        }

        // sched_domain is kmalloc'd — lives in direct mapping.
        let sd_pa = super::symbols::kva_to_pa(current_kva, page_offset);
        if sd_pa >= mem.size() {
            break;
        }

        let level = mem.read_u32(sd_pa, sd_offsets.sd_level) as i32;
        let name = read_sd_name(
            mem,
            sd_pa,
            sd_offsets.sd_name,
            page_offset,
            start_kernel_map,
            phys_base,
        );
        let flags = mem.read_u32(sd_pa, sd_offsets.sd_flags) as i32;
        let span_weight = mem.read_u32(sd_pa, sd_offsets.sd_span_weight);

        let stats = sd_offsets
            .stats_offsets
            .as_ref()
            .map(|so| read_sd_stats(mem, sd_pa, so));

        let snap = SchedDomainSnapshot {
            level,
            name,
            flags,
            span_weight,
            balance_interval: mem.read_u32(sd_pa, sd_offsets.sd_balance_interval),
            nr_balance_failed: mem.read_u32(sd_pa, sd_offsets.sd_nr_balance_failed),
            newidle_call: sd_offsets
                .sd_newidle_call
                .map(|off| mem.read_u32(sd_pa, off)),
            newidle_success: sd_offsets
                .sd_newidle_success
                .map(|off| mem.read_u32(sd_pa, off)),
            newidle_ratio: sd_offsets
                .sd_newidle_ratio
                .map(|off| mem.read_u32(sd_pa, off)),
            max_newidle_lb_cost: mem.read_u64(sd_pa, sd_offsets.sd_max_newidle_lb_cost),
            stats,
        };

        domains.push(snap);

        // Follow sd->parent.
        current_kva = mem.read_u64(sd_pa, sd_offsets.sd_parent);
    }

    Some(domains)
}

/// Resolve per-CPU physical addresses for event counter reads.
///
/// Reads `*scx_root` to find the active `scx_sched` struct, then reads
/// the percpu pointer at `percpu_ptr_off` within it. On 6.18+ (and
/// 6.17.7+ stable) this is `scx_sched.pcpu` (pointing to `scx_sched_pcpu`);
/// on 6.16 through 6.17.6 it is `scx_sched.event_stats_cpu` (pointing
/// directly to `scx_event_stats`). Computes each CPU's PA via
/// `__per_cpu_offset`.
///
/// Returns None if `scx_root` is null (no scheduler loaded).
pub(crate) fn resolve_event_pcpu_pas(
    mem: &GuestMem,
    scx_root_pa: u64,
    ev: &ScxEventOffsets,
    per_cpu_offsets: &[u64],
    page_offset: u64,
) -> Option<Vec<u64>> {
    let scx_sched_kva = mem.read_u64(scx_root_pa, 0);
    if scx_sched_kva == 0 {
        return None;
    }

    let scx_sched_pa = super::symbols::kva_to_pa(scx_sched_kva, page_offset);
    let pcpu_kva = mem.read_u64(scx_sched_pa, ev.percpu_ptr_off);
    if pcpu_kva == 0 {
        return None;
    }

    let pas: Vec<u64> = per_cpu_offsets
        .iter()
        .map(|&cpu_off| super::symbols::kva_to_pa(pcpu_kva.wrapping_add(cpu_off), page_offset))
        .collect();

    Some(pas)
}

/// Per-vCPU host thread timing info for gating stall detection.
///
/// When the host is loaded, vCPU threads get preempted and rq_clock
/// cannot advance. Reading per-thread CPU time distinguishes real
/// stalls (vCPU running but clock stuck) from host preemption
/// (vCPU not scheduled, clock can't advance).
pub(crate) struct VcpuTiming {
    /// pthread_t handles for each vCPU, indexed by vCPU ID.
    /// Used with `pthread_getcpuclockid()` + `clock_gettime()`.
    pub pthreads: Vec<libc::pthread_t>,
}

impl VcpuTiming {
    /// Read CPU time for each vCPU thread. Returns `Some(ns)` per vCPU
    /// on success, `None` when the per-thread clock could not be read.
    ///
    /// `None` propagates through `CpuSnapshot::vcpu_cpu_time_ns`.
    /// Downstream stall detection (`evaluate_preempted`) treats a
    /// `None` on either side of a pair as `preempted=false` — the
    /// stall check falls through to `rq_clock` comparison and fires
    /// if progress isn't observed there. This deliberately prefers
    /// spurious alerts (better visibility) over missed stalls (silent
    /// failure) when clock reads are unavailable. The previous bug
    /// did the opposite: a silent `0` collided with `saturating_sub`
    /// to fabricate "no delta", which looked like preemption and
    /// suppressed every stall after the first clock-read failure.
    ///
    /// Emits a one-shot `tracing::warn` per vCPU (debounced via
    /// `reported_err`) naming the failing syscall + errno so a user
    /// can diagnose why stall gating has degraded to "no data".
    fn read_cpu_times(&self, reported_err: &mut [bool]) -> Vec<Option<u64>> {
        self.pthreads
            .iter()
            .enumerate()
            .map(|(vcpu, &pt)| {
                let mut clk: libc::clockid_t = 0;
                let ret = unsafe { libc::pthread_getcpuclockid(pt, &mut clk) };
                if ret != 0 {
                    if let Some(slot) = reported_err.get_mut(vcpu)
                        && !*slot
                    {
                        tracing::warn!(
                            vcpu,
                            ret,
                            errno = std::io::Error::last_os_error().raw_os_error(),
                            "pthread_getcpuclockid failed; stall gating unavailable for this vCPU"
                        );
                        *slot = true;
                    }
                    return None;
                }
                let mut ts = libc::timespec {
                    tv_sec: 0,
                    tv_nsec: 0,
                };
                let ret = unsafe { libc::clock_gettime(clk, &mut ts) };
                if ret != 0 {
                    if let Some(slot) = reported_err.get_mut(vcpu)
                        && !*slot
                    {
                        tracing::warn!(
                            vcpu,
                            ret,
                            errno = std::io::Error::last_os_error().raw_os_error(),
                            "clock_gettime on pthread clock failed; stall gating unavailable for this vCPU"
                        );
                        *slot = true;
                    }
                    return None;
                }
                // CPU-time pthread clocks are cumulative nanoseconds
                // and thus always non-negative; guard anyway so a
                // negative tv_sec or tv_nsec from a hypothetical clock
                // bug doesn't silently wrap through `as u64`.
                if ts.tv_sec < 0 || ts.tv_nsec < 0 {
                    if let Some(slot) = reported_err.get_mut(vcpu)
                        && !*slot
                    {
                        tracing::warn!(
                            vcpu,
                            tv_sec = ts.tv_sec,
                            tv_nsec = ts.tv_nsec,
                            "negative clock_gettime result; stall gating unavailable for this vCPU"
                        );
                        *slot = true;
                    }
                    return None;
                }
                // Re-arm the error latch on a successful read so a
                // transient failure doesn't permanently mute the log.
                if let Some(slot) = reported_err.get_mut(vcpu) {
                    *slot = false;
                }
                Some(ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64)
            })
            .collect()
    }
}

/// Decide whether a vCPU was preempted between two consecutive samples.
///
/// Returns `true` only when BOTH samples produced a valid reading
/// (`Some`) and the delta falls strictly below `threshold_ns`. Any
/// missing reading (`None` on either side) is treated as "no data" and
/// returns `false` — the stuck path must NEVER infer preemption from
/// absent data, otherwise a clock-read failure would silently suppress
/// every subsequent stuck reading (the original bug).
///
/// Uses `saturating_sub` to tolerate non-monotonic reads across clock
/// resolution edges; a non-monotonic sample yields delta=0, which is
/// below any positive threshold, so `preempted=true` — matching the
/// semantics of "the vCPU made no measurable progress".
pub(crate) fn evaluate_preempted(prev: Option<u64>, curr: Option<u64>, threshold_ns: u64) -> bool {
    match (prev, curr) {
        (Some(p), Some(c)) => c.saturating_sub(p) < threshold_ns,
        _ => false,
    }
}

/// Decide whether a CPU is stuck between two consecutive samples.
///
/// "Stuck" means the scheduler made no progress on this CPU: `rq_clock`
/// did not advance AND the CPU was not legitimately quiescent. The two
/// legitimate exemptions — NOHZ idle (both samples show `nr_running==0`)
/// and vCPU preemption (host scheduled the vCPU thread off-CPU, so the
/// vCPU couldn't tick the clock) — are recognized here so callers don't
/// re-derive the predicate.
///
/// Distinct from sched_ext's watchdog stall (`SCX_EXIT_ERROR_STALL`,
/// emitted by the kernel when a runnable task hasn't been scheduled
/// within the watchdog timeout): "stuck" describes a CPU whose
/// rq_clock isn't advancing, while watchdog "stall" describes a
/// task that hasn't run. The two conditions can co-occur but have
/// different root causes and detection paths.
///
/// This helper exists to keep the post-hoc `MonitorSummary::from_samples`
/// path and the reactive `MonitorThresholds::evaluate` path in lock-step:
/// previously each site re-implemented the same four-condition conjunction
/// and drifting one half would let the SysRq-D trigger fire on conditions
/// the post-hoc verdict accepted (or vice versa). Both callers now agree
/// on a single definition of "stuck" by construction.
///
/// `rq_clock == 0` is treated as "never sampled" and returns false —
/// the first sample interval typically reads zero before the kernel
/// writes rq_clock, and a zero-to-zero comparison must not fire a stuck.
pub(crate) fn is_cpu_stuck(
    prev: &super::CpuSnapshot,
    curr: &super::CpuSnapshot,
    preemption_threshold_ns: u64,
) -> bool {
    if curr.rq_clock == 0 || curr.rq_clock != prev.rq_clock {
        return false;
    }
    let idle = curr.nr_running == 0 && prev.nr_running == 0;
    if idle {
        return false;
    }
    let preempted = evaluate_preempted(
        prev.vcpu_cpu_time_ns,
        curr.vcpu_cpu_time_ns,
        preemption_threshold_ns,
    );
    !preempted
}

/// Configuration for reactive SysRq-D dump triggering.
///
/// When provided to `monitor_loop`, the monitor evaluates thresholds
/// inline and pushes a `SIGNAL_VC_DUMP` wake byte through the
/// attached virtio-console on sustained violation. The guest's
/// `hvc0_poll_loop` blocks on `/dev/hvc0`, recognises the byte, and
/// dispatches SysRq-D directly.
pub(crate) struct DumpTrigger {
    /// Thresholds for violation detection.
    pub thresholds: super::MonitorThresholds,
    /// Virtio-console handle. When `Some`, the monitor pushes a wake
    /// byte (`SIGNAL_VC_DUMP`) into the device's RX queue on
    /// sustained violation; the guest wakes within microseconds. When
    /// `None`, the trigger logs a `tracing::warn!` so the missing
    /// transport is visible — there is no longer a SHM-byte fallback.
    /// Production callers always populate this; the field stays
    /// `Option` only so unit tests that exercise the threshold logic
    /// without a fully-wired device can pass `None`.
    pub virtio_con:
        Option<std::sync::Arc<crate::vmm::PiMutex<crate::vmm::virtio_console::VirtioConsole>>>,
}

/// Inputs for per-iteration refresh of `__per_cpu_offset[]` and the
/// physical addresses derived from it.
///
/// The host monitor thread spawns concurrently with the guest BSP
/// entering KVM_RUN — `start_kernel`'s `setup_per_cpu_areas` runs
/// inside the guest, not on the host, so a one-shot read of
/// `__per_cpu_offset[]` at host monitor spawn time hits BSS zeros
/// for every slot. `compute_rq_pas` then wraps `runqueues_kva.add(0)`
/// past zero into the upper-half KVA space, [`super::symbols::kva_to_pa`]
/// subtracts `page_offset` (`0xFFFF_8880_…`) producing a `~131 TB` PA
/// that fails [`GuestMem::read_u64`]'s bounds check, and every
/// scalar read returns silent zero. The result is `rq_clock=0`,
/// `nr_running=0`, etc. for the entire VM lifetime.
///
/// When [`MonitorConfig::rq_refresh`] is set, [`monitor_loop`]
/// re-reads `__per_cpu_offset[]` at the top of every sample
/// iteration, recomputes `rq_pas`, and (when [`Self::event`] is
/// set) recomputes `event_pcpu_pas`. Once the BSP populates
/// `__per_cpu_offset[]` (during `setup_per_cpu_areas` in
/// `start_kernel`) the offsets become non-zero and reads land
/// inside guest DRAM. The transition is silent: early samples
/// observe zeros (consistent with the pre-refresh behaviour) and
/// later samples observe real values once the guest has booted
/// past percpu setup.
pub(crate) struct RqRefresh {
    /// PA of the kernel's `__per_cpu_offset[]` array. Read each
    /// iteration via [`super::symbols::read_per_cpu_offsets`].
    pub pco_pa: u64,
    /// KVA of the `runqueues` percpu symbol. Combined with each
    /// CPU's offset to compute the per-CPU rq KVA, then reduced
    /// to a PA via [`super::symbols::kva_to_pa`].
    pub runqueues_kva: u64,
    /// Link-time KVA of `__per_cpu_start`. Subtracted from
    /// `runqueues_kva` to get the section-relative offset.
    pub per_cpu_start: u64,
    /// Virtual KASLR offset. Per-CPU offset computation needs this
    /// to bridge the link-time and runtime `__per_cpu_start`.
    pub kaslr_offset: u64,
    /// Number of CPUs (entries to read from `__per_cpu_offset[]`).
    pub num_cpus: u32,
    /// PA of the `page_offset_base` symbol (text-mapped). When
    /// `Some`, the monitor re-reads `PAGE_OFFSET` each sample so a
    /// KASLR `CONFIG_RANDOMIZE_MEMORY` value (e.g.
    /// `0xff11_0000_0000_0000`) replaces the
    /// [`super::symbols::DEFAULT_PAGE_OFFSET`] fallback once the
    /// guest kernel finishes randomization (which happens after
    /// the host monitor thread spawns — the same boot race that
    /// motivates the `__per_cpu_offset[]` refresh). When `None`
    /// (or when the read returns zero / fails the bit-63 check),
    /// the per-iteration code keeps using the pre-loop resolved
    /// `page_offset` value.
    pub page_offset_base_pa: Option<u64>,
    /// Optional refresh inputs for `event_pcpu_pas`. When `None`,
    /// the monitor leaves `event_pcpu_pas` at its initial value
    /// (typically `None` because `scx_root` was null at host
    /// monitor spawn time).
    pub event: Option<EventRefresh>,
}

/// Inputs for per-iteration refresh of `event_pcpu_pas`.
///
/// `event_pcpu_pas` derives from `*scx_root -> scx_sched.pcpu`
/// (or `event_stats_cpu` on pre-6.18 kernels) plus
/// `__per_cpu_offset[]`. The `scx_root` deref returns null until a
/// scheduler attaches, so a one-shot pre-loop resolve always
/// observes `None`. Refresh each iteration so the first sample
/// after scheduler attach picks up real PAs.
pub(crate) struct EventRefresh {
    /// PA of the `scx_root` global pointer (text mapping).
    pub scx_root_pa: u64,
    /// BTF-resolved offsets within `scx_sched` and the per-CPU
    /// stats struct. See [`ScxEventOffsets`] for version-specific
    /// indirection.
    pub event_offsets: ScxEventOffsets,
}

/// Override for the scheduler watchdog timeout, written every monitor
/// iteration.
///
/// Two write paths are supported:
/// - 7.1+ (`ScxSched`): deref `*scx_root` to find the runtime
///   `scx_sched` struct, then write at the BTF-resolved offset.
///   Re-derefs each iteration because `scx_sched` is reallocated on
///   scheduler (re)load.
/// - pre-7.1 (`StaticGlobal`): write directly to the PA of the
///   `scx_watchdog_timeout` static global. No deref needed — the
///   address is fixed for the kernel's lifetime.
pub(crate) enum WatchdogOverride {
    /// 7.1+ path: deref `scx_root` -> `scx_sched` -> write at offset.
    ScxSched {
        /// PA of the `scx_root` global pointer (text mapping).
        scx_root_pa: u64,
        /// Byte offset of `watchdog_timeout` within `struct scx_sched`.
        watchdog_offset: usize,
        /// Jiffies value to write.
        jiffies: u64,
        /// PA of `scx_watchdog_interval` global.
        interval_pa: Option<u64>,
        /// PA of `scx_watchdog_timestamp` global.
        timestamp_pa: Option<u64>,
        /// PA of `jiffies_64` global (to read current time).
        jiffies_64_pa: Option<u64>,
    },
    /// Pre-7.1 path: write directly to the static global's PA.
    StaticGlobal {
        /// PA of the `scx_watchdog_timeout` static global (text mapping).
        watchdog_timeout_pa: u64,
        /// Jiffies value to write.
        jiffies: u64,
        /// PA of `scx_watchdog_interval` global.
        interval_pa: Option<u64>,
        /// PA of `scx_watchdog_timestamp` global.
        timestamp_pa: Option<u64>,
        /// PA of `jiffies_64` global.
        jiffies_64_pa: Option<u64>,
    },
}

/// BPF program stats context for the monitor loop.
///
/// Holds the static parameters [`super::bpf_prog::walk_struct_ops_runtime_stats`]
/// needs each sample. The IDR walk re-runs every cycle, which is cheap
/// (`idr_next` is in the dozens for ktstr workloads) and removes the
/// staleness window the prior cached-discovery design opened — newly
/// loaded struct_ops programs surface immediately rather than waiting
/// for a context rebuild.
pub(crate) struct ProgStatsCtx {
    /// Per-CPU offset table (`__per_cpu_offset[]`) seed used to
    /// translate each program's percpu stats pointer into a
    /// concrete KVA. When [`MonitorConfig::rq_refresh`] is `None`
    /// this seed is forwarded verbatim to
    /// [`super::bpf_prog::walk_struct_ops_runtime_stats`] every
    /// sample. When `rq_refresh` is `Some`, the loop overrides
    /// this with a freshly read array each iteration so newly
    /// attached struct_ops programs read post-`setup_per_cpu_areas`
    /// per-CPU bases.
    pub per_cpu_offsets: Vec<u64>,
    /// Paging context ([`WalkContext`]) threaded into
    /// [`super::bpf_prog::walk_struct_ops_runtime_stats`] so per-CPU
    /// `bpf_prog_stats` allocations that fall outside the direct
    /// mapping (vmalloc-backed percpu) translate via a page-table
    /// walk instead of being silently dropped.
    pub walk: WalkContext,
    /// `prog_idr` symbol KVA (kernel BSS). Read each sample to walk
    /// all loaded BPF programs.
    pub prog_idr_kva: u64,
    /// BTF offsets for the `bpf_prog` + related struct fields read
    /// while summing stats.
    pub offsets: super::btf_offsets::BpfProgOffsets,
    /// Runtime kernel image base (`__START_KERNEL_map` on x86_64,
    /// derived `KIMAGE_VADDR` on aarch64). Threaded into
    /// [`super::bpf_prog::walk_struct_ops_runtime_stats`] so
    /// `prog_idr` symbol KVAs translate correctly even on aarch64
    /// kernels with VA_BITS != 48.
    pub start_kernel_map: u64,
    /// Runtime KASLR offset (`phys_base` on x86_64; `0` on aarch64
    /// / non-KASLR boots). Threaded into
    /// [`super::bpf_prog::walk_struct_ops_runtime_stats`] alongside
    /// `start_kernel_map` so the IDR head's PA resolves correctly
    /// on KASLR kernels.
    pub phys_base: u64,
}

/// Samples and optional watchdog observation returned by
/// [`monitor_loop`].
pub(crate) struct MonitorLoopResult {
    /// Per-interval `MonitorSample`s collected across the run.
    pub(crate) samples: Vec<MonitorSample>,
    /// Mid-flight TLV entries drained from the guest. Empty in this
    /// loop variant — the freeze coordinator owns the bulk-port
    /// drain and stashes per-tick parsed entries on its own buffer;
    /// kept on this struct for compatibility with `collect_results`'s
    /// merge.
    pub(crate) drain: crate::vmm::host_comms::BulkDrainResult,
    /// Watchdog read-back, when a watchdog override was installed.
    pub(crate) watchdog_observation: Option<super::WatchdogObservation>,
    /// Live `PAGE_OFFSET` value used by the loop for KVA→PA
    /// translation, captured at the moment the per-iteration
    /// `DATA_VALID` latch fired. Forwarded onto `MonitorReport` so
    /// callers can prove the KASLR-randomized base was observed
    /// (rather than the static `DEFAULT_PAGE_OFFSET` fallback).
    /// 0 means the latch never fired during the run.
    pub(crate) page_offset: u64,
    /// vCPU preemption threshold (ns) derived once from the guest
    /// kernel's CONFIG_HZ at monitor-thread start and propagated
    /// here so the post-run [`super::MonitorReport`] consumer does
    /// not re-read the vmlinux file to recompute the same value.
    /// Mirrors [`MonitorConfig::preemption_threshold_ns`] after the
    /// loop's `0 → super::vcpu_preemption_threshold_ns(None)`
    /// fallback has resolved a concrete value.
    pub(crate) preemption_threshold_ns: u64,
}

/// Configuration for the monitor sampling loop.
///
/// Bundles the parameters that `monitor_loop` needs beyond the
/// required `mem`, `rq_pas`, `offsets`, `interval`, `kill`, and `run_start`.
pub(crate) struct MonitorConfig<'a> {
    /// Per-CPU physical addresses of `scx_sched_pcpu` (or
    /// `scx_event_stats` on pre-6.18 kernels). When `rq_refresh` is
    /// `None` and `event_offsets` exist, each sample includes event
    /// counters drawn from these PAs. Production callers pass
    /// `None` here and supply a [`RqRefresh`] whose
    /// [`RqRefresh::event`] is `Some(EventRefresh)` so the loop can
    /// recompute the PAs against the live `__per_cpu_offset[]` and
    /// observe schedulers that attach mid-run.
    pub event_pcpu_pas: Option<&'a [u64]>,
    /// Reactive dump configuration. When a sustained threshold violation is
    /// detected, writes the dump request flag to guest SHM to trigger a
    /// SysRq-D dump inside the guest.
    pub dump_trigger: Option<&'a DumpTrigger>,
    /// Optional watchdog timeout override to install before sampling
    /// begins; read back into `WatchdogObservation` after the loop.
    pub watchdog_override: Option<&'a WatchdogOverride>,
    /// Optional per-vCPU timing context for preemption accounting.
    pub vcpu_timing: Option<&'a VcpuTiming>,
    /// Optional host-side perf-counter capture. When present, every
    /// monitor sample reads `cycles`/`instructions`/`cache_misses`/
    /// `branch_misses` per vCPU into [`super::CpuSnapshot::vcpu_perf`].
    /// `None` skips the per-vCPU PMU capture.
    pub perf_capture: Option<&'a super::perf_counters::PerfCountersCapture>,
    /// Preemption threshold in nanoseconds used for stall detection.
    /// Pass 0 to derive it from the guest kernel's CONFIG_HZ.
    pub preemption_threshold_ns: u64,
    /// Optional BPF program statistics context; when present, each
    /// sample includes per-program exec counters.
    pub prog_stats_ctx: Option<&'a ProgStatsCtx>,
    /// Runtime `PAGE_OFFSET` for direct-mapping KVA translation. Used by
    /// sched_domain tree walking to translate `rq->sd` and `sd->parent`
    /// pointers.
    pub page_offset: u64,
    /// Runtime kernel image base (`__START_KERNEL_map` on x86_64,
    /// `KIMAGE_VADDR` on aarch64). Threaded into [`read_sd_name`] so
    /// the rodata-string fast path translates `sd->name` correctly on
    /// aarch64 hosts where the base depends on `VA_BITS` (16 KB-granule
    /// kernels with VA_BITS=47 vs the 48-bit default).
    pub start_kernel_map: u64,
    /// Runtime KASLR offset (x86_64 `phys_base`; `0` on aarch64 and
    /// non-KASLR boots). Forwarded into helpers that translate
    /// kernel-text/data symbols so KASLR'd kernels resolve symbols
    /// correctly. Resolved by
    /// [`super::symbols::resolve_phys_base`] at monitor-thread
    /// start, then threaded through here.
    pub phys_base: u64,
    /// Optional per-iteration refresh of `__per_cpu_offset[]` and the
    /// addresses derived from it. When set, [`monitor_loop`] re-reads
    /// the per-CPU offset table each sample, overriding the static
    /// `rq_pas` and `event_pcpu_pas` arguments. See [`RqRefresh`] for
    /// the boot-race rationale. When `None`, the static arrays are
    /// used unchanged (test path; production callers always populate
    /// this).
    pub rq_refresh: Option<&'a RqRefresh>,
    /// Optional boot-complete eventfd. When set, [`monitor_loop`]
    /// blocks on this eventfd (alongside `kill_evt` and a 5 s
    /// timeout) BEFORE entering the sample loop, so the first
    /// sample observes a fully booted guest. The eventfd is fired
    /// by the freeze coordinator's bulk-drain dispatch when the
    /// guest publishes a CRC-valid
    /// [`crate::vmm::wire::MSG_TYPE_SYS_RDY`] TLV frame on the
    /// virtio-console bulk port, which happens after the guest's
    /// `ktstr_guest_init` completes `mount_filesystems()`. By that
    /// point `setup_per_cpu_areas` and KASLR randomization have
    /// long since completed (both happen during kernel boot,
    /// strictly before userspace init runs) — both prerequisites
    /// for the per-iteration `__per_cpu_offset[]` /
    /// `page_offset_base` reads to land in DRAM. When `None` (test
    /// path or eventfd-create failure), the pre-loop wait is
    /// skipped entirely; the per-iteration `data_valid` gate in
    /// the sample loop remains as defense-in-depth against
    /// pre-boot zeros.
    pub sys_rdy: Option<&'a EventFd>,
    /// Optional scheduler-attach watchdog reset. When `Some`, the
    /// loop reads `*scx_root` each iteration via
    /// [`WatchdogReset::scx_root_pa`] and, on the first 0 →
    /// non-zero transition, stores `(now - run_start +
    /// workload_duration).as_nanos()` into the shared atomic so
    /// the host-side VM watchdog can recompute its hard deadline
    /// from the moment a scheduler attaches instead of from VM
    /// boot. The value is encoded as nanoseconds since
    /// `run_start` (passed to `monitor_loop`) so the watchdog
    /// can decode it as `run_start + Duration::from_nanos(value)`.
    /// Storing `0` is reserved as the "no reset requested"
    /// sentinel — the watchdog ignores zero. The reset is
    /// bounded above by the watchdog's original
    /// `timeout`-derived deadline at the consumer side, so a
    /// late attach cannot extend past the outer kill timer.
    /// `None` (test path or kernels with no `scx_root` symbol)
    /// disables attach detection.
    pub watchdog_reset: Option<WatchdogReset<'a>>,
}

/// Inputs the monitor needs to push a watchdog-reset deadline when
/// a scheduler attaches.
///
/// All three fields move together — none of them are useful alone —
/// so they're bundled to keep [`MonitorConfig`] flat. Construction
/// is owned by [`crate::vmm::freeze_coord`] inside `start_monitor`,
/// where `scx_root_pa` is the text-mapped PA of the kernel's
/// `scx_root` global, `workload_duration` flows from
/// [`crate::vmm::KtstrVm::workload_duration`] (which mirrors the
/// test entry's `duration`), and `reset_ns` is shared with the
/// watchdog thread.
pub(crate) struct WatchdogReset<'a> {
    /// PA of the `scx_root` global pointer (text mapping). The
    /// loop reads `mem.read_u64(scx_root_pa, 0)` each iteration
    /// and detects the 0 → non-zero edge.
    pub scx_root_pa: u64,
    /// Workload time budget the watchdog should reset to. Encoded
    /// as nanoseconds since `run_start` (added to `Instant::now()
    /// - run_start` at attach time and stored into [`reset_ns`]).
    pub workload_duration: Duration,
    /// Shared atomic written once on the first scheduler-attach
    /// observation. The watchdog thread reads this each tick and,
    /// when non-zero, uses `run_start +
    /// Duration::from_nanos(value)` as its hard deadline (capped
    /// at the original `timeout`-derived deadline).
    pub reset_ns: &'a AtomicU64,
}

/// Run the monitor loop, sampling all CPUs at the given interval
/// until `kill` is set. Returns a [`MonitorLoopResult`] containing
/// the collected per-interval samples and — when a watchdog override
/// was installed — the post-run `WatchdogObservation` read-back.
///
/// `rq_pas` is the static seed for per-CPU `struct rq` PAs and is
/// used unchanged when `cfg.rq_refresh` is `None` (the test path
/// uses this; production passes `&[]` and supplies a
/// [`RqRefresh`] that recomputes `rq_pas` per iteration to dodge
/// the host-monitor / guest-BSP boot race documented on
/// [`RqRefresh`]).
///
/// The cadence is driven by a `CLOCK_MONOTONIC` `timerfd` armed at
/// `interval`; an external `kill_evt` write breaks out of the
/// `epoll_wait` immediately rather than waiting up to one full
/// interval. `kill` (the atomic) and `kill_evt` (the eventfd) carry
/// the same shutdown signal: external setters should flip both so
/// the wait returns within microseconds of the flip and the
/// `kill.load(Acquire)` re-check at the top of the loop body
/// observes the new state.
#[allow(clippy::too_many_arguments)]
pub(crate) fn monitor_loop(
    mem: &GuestMem,
    rq_pas: &[u64],
    offsets: &KernelOffsets,
    interval: Duration,
    kill: &AtomicBool,
    kill_evt: &EventFd,
    run_start: Instant,
    cfg: &MonitorConfig<'_>,
) -> MonitorLoopResult {
    let dump_trigger = cfg.dump_trigger;
    let watchdog_override = cfg.watchdog_override;
    let vcpu_timing = cfg.vcpu_timing;
    let perf_capture = cfg.perf_capture;
    let preemption_threshold_ns = cfg.preemption_threshold_ns;
    let prog_stats_ctx = cfg.prog_stats_ctx;
    // Mutable so the per-iteration refresh can update it once
    // `page_offset_base` becomes readable. Initialized from the
    // pre-loop resolved value, which on KASLR-randomized kernels
    // (CONFIG_RANDOMIZE_MEMORY) is the DEFAULT_PAGE_OFFSET fallback
    // because the symbol was zero at host monitor spawn time. The
    // refresh below re-reads `page_offset_base` and overrides
    // `page_offset` once the guest writes the real KASLR value
    // (e.g. 0xff11_0000_0000_0000). Without this override, every
    // `kva_to_pa(kva, page_offset)` produces a PA off by the
    // KASLR delta (~13.5 GB) — the recomputed `rq_pas` land
    // outside the 256 MB DRAM region and `read_u64`
    // bounds-rejects to zero, so `rq_clock` reads as 0 forever.
    let mut page_offset = cfg.page_offset;
    let start_kernel_map = cfg.start_kernel_map;
    let phys_base = cfg.phys_base;
    let rq_refresh = cfg.rq_refresh;
    let preemption_threshold_ns = if preemption_threshold_ns > 0 {
        preemption_threshold_ns
    } else {
        super::vcpu_preemption_threshold_ns(None)
    };
    // Per-CPU PA buffers maintained across sample iterations. When
    // `rq_refresh` is `Some`, every iteration recomputes them from
    // a freshly read `__per_cpu_offset[]` to dodge the boot race
    // documented on [`RqRefresh`]: the host monitor thread spawns
    // concurrently with the guest BSP entering KVM_RUN, so a
    // one-shot read at thread start hits BSS zeros and
    // `compute_rq_pas` wraps to a ~131 TB PA that
    // [`GuestMem::read_u64`] silently bounds-rejects to 0. When
    // `rq_refresh` is `None` (test path), the static `rq_pas` /
    // `cfg.event_pcpu_pas` / `prog_stats_ctx.per_cpu_offsets`
    // arguments are used as-is for every sample.
    let mut rq_pas_buf: Vec<u64> = rq_pas.to_vec();
    let mut event_pcpu_pas_buf: Option<Vec<u64>> = cfg.event_pcpu_pas.map(|s| s.to_vec());
    let mut per_cpu_offsets_buf: Vec<u64> = prog_stats_ctx
        .map(|c| c.per_cpu_offsets.clone())
        .unwrap_or_default();
    // Diagnostic counter for the refresh path. Emits per_cpu_offset[],
    // rq_pas[], page_offset, runqueues_kva, mem.size, plus a direct
    // rq_clock read, a phys_base probe, and a 16-byte hex dump for
    // both the FIRST 5 iterations (catches initial state) AND the
    // last 5 iterations of an assumed ~300-sample run (catches the
    // post-guest-boot state). The early window distinguishes wrong
    // pco_pa / wrong page_offset / wrong BTF rq_clock; the late
    // window distinguishes "guest never wrote the BSS" from "writes
    // are invisible to host monitor". Removed once probe-reload lands.
    let mut diag_iter: u32 = 0;
    // Previous `page_offset_base` read, used by the stability gate
    // below. The bit-63 + canonical-half check alone is too loose:
    // mid-decompression garbage in the bzImage region can satisfy
    // bit 63 while pointing nowhere useful (observed:
    // 0xfedfe68cfedfe680 latched and corrupted page_offset before
    // the guest finished kernel decompression). Requiring two
    // consecutive reads to match — combined with a 4 KiB-page
    // alignment check below — rejects transient garbage and only
    // accepts the stable post-KASLR value.
    let mut prev_pob: u64 = 0;
    // Latched once `__per_cpu_offset[0]` becomes non-zero AND the
    // KASLR-randomized `page_offset_base` has been observed (i.e.
    // the guest finished `setup_per_cpu_areas` and KASLR
    // randomization). Until both are visible, the recomputed
    // `rq_pas` either wrap to ~131 TB (pre-percpu) or land outside
    // the 256 MB DRAM region (pre-KASLR), so every downstream
    // walker — `read_event_stats`, `read_rq_schedstat`,
    // `read_sched_domain_tree`, `walk_struct_ops_runtime_stats` —
    // chases pointers through silent-zero returns. The chase is
    // O(n) per phantom node and turns the 100 ms timerfd cadence
    // into multi-second iterations. Skipping the expensive walks
    // until `data_valid` flips to true keeps pre-boot iterations
    // microsecond-cheap so post-boot samples land on the timerfd
    // schedule rather than minutes after the BSP brought up the
    // scheduler. Latches monotonically — once the guest has
    // valid data, we never go back to the cheap path even if a
    // subsequent read transiently returns zero.
    //
    // Initialized `true` when `rq_refresh` is `None` because the
    // test path passes pre-resolved `rq_pas` and `event_pcpu_pas`
    // arguments — there is no boot race to gate against and the
    // walks must run from the first iteration.
    let mut data_valid: bool = rq_refresh.is_none();
    // `page_offset` value captured at the instant `data_valid`
    // latched, forwarded onto `MonitorReport.page_offset` via
    // `MonitorLoopResult.page_offset`. Stays 0 when the latch
    // never fires (guest never booted far enough, monitor was
    // never started, or VM died during the pre-sample boot
    // wait); externally-visible signal that the KASLR-randomized
    // base was actually observed during the run rather than the
    // static `DEFAULT_PAGE_OFFSET` fallback. Distinct from the
    // mutable `page_offset` local because that local always
    // carries SOME value (initialized from `cfg.page_offset`),
    // so it cannot itself signal "no observation made."
    let mut latched_page_offset: u64 = 0;
    // Authoritative CPU count: `rq_refresh` overrides `rq_pas.len()`
    // because the static seed slice may be empty in production
    // (where the boot-race fix sources every PA from the refresh).
    let num_cpus: usize = rq_refresh
        .map(|r| r.num_cpus as usize)
        .unwrap_or(rq_pas.len());
    let mut samples: Vec<MonitorSample> = Vec::new();
    // Reactive threshold trackers — reuse the post-hoc
    // `SustainedViolationTracker` so "sustained for N samples"
    // means the same thing to the reactive SysRq-D dump and to
    // `MonitorThresholds::evaluate` running over the full sample vec.
    let mut imbalance_tracker = super::SustainedViolationTracker::default();
    let mut dsq_tracker = super::SustainedViolationTracker::default();
    let mut stall_trackers: Vec<super::SustainedViolationTracker> =
        vec![super::SustainedViolationTracker::default(); num_cpus];
    let mut dump_requested = false;
    let mut cpus: Vec<CpuSnapshot> = Vec::with_capacity(num_cpus);
    let mut perf_read_err_reported = false;
    let mut vcpu_timing_err_reported: Vec<bool> = vcpu_timing
        .map(|vt| vec![false; vt.pthreads.len()])
        .unwrap_or_default();
    let shm_entries: Vec<crate::vmm::wire::ShmEntry> = Vec::new();
    let mut watchdog_observation: Option<super::WatchdogObservation> = None;
    // Once-only latch for the scheduler-attach watchdog reset.
    // The first time the loop observes `*scx_root != 0` (a
    // scheduler attached and the runtime `scx_sched` struct is
    // live in guest memory), encode `(now - run_start +
    // workload_duration)` as nanoseconds and store it into
    // `cfg.watchdog_reset.reset_ns` so the watchdog thread can
    // recompute its hard deadline from the attach moment rather
    // than VM boot. `false` means "not yet observed"; flipped to
    // `true` after a successful store so a later transient
    // null-read or scheduler reload does not stomp the attach
    // moment with a fresh `Instant::now()`.
    let mut watchdog_reset_signaled = false;

    // Cadence + wake plumbing. `tick_tfd` is a periodic
    // `CLOCK_MONOTONIC` timerfd that fires every `interval` so the
    // sampling cadence matches the previous `thread::sleep(interval)`
    // contract. `kill_evt` is the shutdown wake — external setters
    // (collect_results, vCPU shutdown classifier, panic hook) write
    // to it so the wait returns within microseconds of the kill
    // flip rather than waiting up to one full interval.
    //
    // EpollEvent data is the source fd so the post-wait dispatch can
    // tell which fd fired without reading the timerfd counter (which
    // we drain unconditionally on the timer branch).
    let tick_tfd = match TimerFd::new() {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!(err = %e, "monitor: timerfd_create failed");
            return MonitorLoopResult {
                samples,
                drain: crate::vmm::host_comms::BulkDrainResult {
                    entries: shm_entries,
                },
                watchdog_observation,
                page_offset: latched_page_offset,
                preemption_threshold_ns,
            };
        }
    };
    let mut tick_tfd = tick_tfd;
    if let Err(e) = tick_tfd.reset(interval, Some(interval)) {
        tracing::warn!(err = %e, "monitor: timerfd_settime failed");
        return MonitorLoopResult {
            samples,
            drain: crate::vmm::host_comms::BulkDrainResult {
                entries: shm_entries,
            },
            watchdog_observation,
            page_offset: latched_page_offset,
            preemption_threshold_ns,
        };
    }
    let epoll = match Epoll::new() {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!(err = %e, "monitor: epoll_create1 failed");
            return MonitorLoopResult {
                samples,
                drain: crate::vmm::host_comms::BulkDrainResult {
                    entries: shm_entries,
                },
                watchdog_observation,
                page_offset: latched_page_offset,
                preemption_threshold_ns,
            };
        }
    };
    let tick_fd = tick_tfd.as_raw_fd();
    let kill_fd = kill_evt.as_raw_fd();
    if let Err(e) = epoll.ctl(
        ControlOperation::Add,
        tick_fd,
        EpollEvent::new(EventSet::IN, tick_fd as u64),
    ) {
        tracing::warn!(err = %e, "monitor: epoll_ctl add timerfd failed");
        return MonitorLoopResult {
            samples,
            drain: crate::vmm::host_comms::BulkDrainResult {
                entries: shm_entries,
            },
            watchdog_observation,
            page_offset: latched_page_offset,
            preemption_threshold_ns,
        };
    }
    if let Err(e) = epoll.ctl(
        ControlOperation::Add,
        kill_fd,
        EpollEvent::new(EventSet::IN, kill_fd as u64),
    ) {
        tracing::warn!(err = %e, "monitor: epoll_ctl add kill_evt failed");
        return MonitorLoopResult {
            samples,
            drain: crate::vmm::host_comms::BulkDrainResult {
                entries: shm_entries,
            },
            watchdog_observation,
            page_offset: latched_page_offset,
            preemption_threshold_ns,
        };
    }
    let mut epoll_buf = [EpollEvent::default(); 2];

    // Pre-sample boot-complete wait. When `cfg.sys_rdy` is set
    // (production path) we register the eventfd alongside `kill_evt`
    // on a dedicated epoll instance and block for up to 5 s. The
    // eventfd is fired by the freeze coordinator's bulk-drain
    // dispatch when the guest publishes a CRC-valid
    // [`crate::vmm::wire::MSG_TYPE_SYS_RDY`] TLV frame on the
    // virtio-console bulk port — sent by `ktstr_guest_init` after
    // `mount_filesystems()`. By the time the SYS_RDY frame reaches
    // the host, `setup_per_cpu_areas` has populated
    // `__per_cpu_offset[]` and the KASLR randomizer has populated
    // `page_offset_base` (both kernel-boot prerequisites that
    // complete strictly before userspace init runs), so the first
    // sample iteration's refresh produces in-DRAM PAs.
    //
    // Three exit conditions:
    //   1. sys_rdy fires: proceed to the sample loop (normal path).
    //   2. kill_evt fires: VM died before booting. Skip the sample
    //      loop entirely, returning the empty MonitorLoopResult.
    //   3. 5 s timeout: the guest never published SYS_RDY (e.g.
    //      KTSTR_EXIT cmdline triggered an early exit before
    //      `ktstr_guest_init` reached the SYS_RDY emission, or the
    //      bulk port wasn't yet open when the guest tried to send).
    //      Best-effort fall through to the sample loop; reads will
    //      still hit the per-iteration refresh which tolerates
    //      pre-boot zeros via the existing `data_valid` gate.
    if let Some(sys_rdy) = cfg.sys_rdy {
        let boot_epoll = match Epoll::new() {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(err = %e, "monitor: boot epoll_create1 failed");
                return MonitorLoopResult {
                    samples,
                    drain: crate::vmm::host_comms::BulkDrainResult {
                        entries: shm_entries,
                    },
                    watchdog_observation,
                    page_offset: latched_page_offset,
                    preemption_threshold_ns,
                };
            }
        };
        let boot_fd = sys_rdy.as_raw_fd();
        if let Err(e) = boot_epoll.ctl(
            ControlOperation::Add,
            boot_fd,
            EpollEvent::new(EventSet::IN, boot_fd as u64),
        ) {
            tracing::warn!(err = %e, "monitor: epoll_ctl add sys_rdy failed");
        }
        if let Err(e) = boot_epoll.ctl(
            ControlOperation::Add,
            kill_fd,
            EpollEvent::new(EventSet::IN, kill_fd as u64),
        ) {
            tracing::warn!(err = %e, "monitor: epoll_ctl add kill_evt (boot wait) failed");
        }
        let mut boot_buf = [EpollEvent::default(); 2];
        // 5 s ceiling: a healthy guest emits SYS_RDY within ~3 s
        // of boot; longer is a stuck guest. The fallthrough path
        // (no SYS_RDY) is gated by the per-iteration `data_valid`
        // latch in the sample loop below, so a missed SYS_RDY does
        // not produce phantom-zero walks — it just delays the first
        // valid sample. Tighter timeout means VM teardown does not
        // wait on this thread joining when the test exits without
        // sending SYS_RDY (e.g. early-init crash).
        let timeout_ms: i32 = 5_000;
        let mut killed = false;
        match boot_epoll.wait(timeout_ms, &mut boot_buf) {
            Ok(n) => {
                for ev in &boot_buf[..n] {
                    if ev.fd() == kill_fd {
                        killed = true;
                    }
                }
            }
            Err(e) => {
                tracing::warn!(err = %e, "monitor: boot epoll_wait failed");
            }
        }
        if killed || kill.load(Ordering::Acquire) {
            return MonitorLoopResult {
                samples,
                drain: crate::vmm::host_comms::BulkDrainResult {
                    entries: shm_entries,
                },
                watchdog_observation,
                page_offset: latched_page_offset,
                preemption_threshold_ns,
            };
        }
    }

    loop {
        if kill.load(Ordering::Acquire) {
            break;
        }
        // Refresh `page_offset` from `page_offset_base` BEFORE the
        // watchdog write below. The watchdog override dereferences
        // `*scx_root` and translates the resulting `scx_sched_kva`
        // to a PA via `kva_to_pa(_, page_offset)`. On
        // KASLR-randomized kernels the pre-loop `page_offset` fell
        // back to DEFAULT_PAGE_OFFSET (the symbol read returned
        // zero before guest randomization completed); using that
        // stale default here would write to a wrong PA and the
        // post-write read-back would observe zero — surfacing as
        // the `watchdog_timeout_override_lands_in_guest_memory`
        // assertion failure (expected_jiffies != observed_jiffies).
        // Hoisting this refresh above both the watchdog write and
        // the rq_pas / event_pcpu_pas recompute block ensures every
        // KVA-to-PA translation in the iteration sees the same
        // live `page_offset`. Three-gate accept on the read: bit 63
        // (canonical upper-half signal), 4 KiB page alignment
        // (kernel PAGE_OFFSET is page-aligned by construction), and
        // stability vs the previous read (rejects mid-decompression
        // garbage that briefly satisfied the bit-63 check).
        let mut page_offset_resolved = false;
        if let Some(refresh) = rq_refresh {
            if let Some(pob_pa) = refresh.page_offset_base_pa {
                let val = mem.read_u64(pob_pa, 0);
                let pob_aligned = (val & 0xFFF) == 0;
                let pob_stable = prev_pob == val;
                prev_pob = val;
                if val & (1u64 << 63) != 0 && pob_aligned && pob_stable {
                    page_offset = val;
                    page_offset_resolved = true;
                }
            } else {
                // No symbol available (aarch64 / stripped vmlinux):
                // pre-loop `page_offset` is the best we have. Treat
                // the page-offset side as already resolved.
                page_offset_resolved = true;
            }
        }
        // Scheduler-attach watchdog reset. Read `*scx_root` and,
        // on the first 0 → non-zero transition, store the encoded
        // attach-moment deadline so the watchdog thread can
        // recompute its hard deadline from the attach moment
        // (instead of from VM boot, which wastes the budget on
        // boot + BPF verifier time). Runs each iteration until
        // the latch fires; the read itself is cheap (one bounded
        // [`GuestMem::read_u64`]) and `scx_root_pa` is text-mapped
        // so the read is valid throughout the run regardless of
        // the per-iteration `data_valid` gate above. Fires
        // independently of `watchdog_override`: a kernel without
        // a resolvable `scx_sched.watchdog_timeout` BTF field
        // still gets a correct outer kill timer.
        if let Some(reset) = cfg.watchdog_reset.as_ref() {
            let scx_sched_kva = mem.read_u64(reset.scx_root_pa, 0);
            if scx_sched_kva != 0 && !watchdog_reset_signaled {
                let elapsed = run_start.elapsed();
                let target_ns = elapsed
                    .as_nanos()
                    .saturating_add(reset.workload_duration.as_nanos());
                let encoded = u64::try_from(target_ns).unwrap_or(u64::MAX).max(1);
                reset.reset_ns.store(encoded, Ordering::Release);
                watchdog_reset_signaled = true;
            }
        }
        if let Some(wd) = watchdog_override {
            let (write_pa, write_offset, wd_jiffies) = match wd {
                WatchdogOverride::ScxSched {
                    scx_root_pa,
                    watchdog_offset,
                    jiffies,
                    ..
                } => {
                    let sch_kva = mem.read_u64(*scx_root_pa, 0);
                    if sch_kva == 0 {
                        (None, 0, *jiffies)
                    } else {
                        let sch_pa = super::symbols::kva_to_pa(sch_kva, page_offset);
                        (Some(sch_pa), *watchdog_offset, *jiffies)
                    }
                }
                WatchdogOverride::StaticGlobal {
                    watchdog_timeout_pa,
                    jiffies,
                    ..
                } => (Some(*watchdog_timeout_pa), 0, *jiffies),
            };
            let (interval_pa, _timestamp_pa, _jiffies_64_pa) = match wd {
                WatchdogOverride::ScxSched {
                    interval_pa,
                    timestamp_pa,
                    jiffies_64_pa,
                    ..
                }
                | WatchdogOverride::StaticGlobal {
                    interval_pa,
                    timestamp_pa,
                    jiffies_64_pa,
                    ..
                } => (*interval_pa, *timestamp_pa, *jiffies_64_pa),
            };
            if let Some(pa) = write_pa {
                mem.write_u64(pa, write_offset, wd_jiffies);
                if let Some(intv_pa) = interval_pa {
                    let intv = std::cmp::max(wd_jiffies / 2, 1);
                    mem.write_u64(intv_pa, 0, intv);
                }
                if watchdog_observation.is_none() {
                    let observed = mem.read_u64(pa, write_offset);
                    watchdog_observation = Some(super::WatchdogObservation {
                        expected_jiffies: wd_jiffies,
                        observed_jiffies: observed,
                    });
                }
            }
        }
        // Per-iteration refresh of `__per_cpu_offset[]` and the
        // PAs derived from it. Boot-race rationale: see [`RqRefresh`]
        // doc. Until the guest BSP runs `setup_per_cpu_areas`, every
        // entry is BSS zero and the recomputed `rq_pas` wrap to PAs
        // that fail [`GuestMem::read_u64`]'s bounds check, so
        // `read_rq_stats` returns zero-filled `CpuSnapshot`s. Once
        // the BSP has populated the array, real PAs land inside
        // guest DRAM and subsequent samples observe live counters.
        if let Some(refresh) = rq_refresh {
            // `page_offset` was already refreshed at the top of this
            // iteration (before the watchdog write). `page_offset_resolved`
            // carries the result into the data_valid latch below.
            let fresh = super::symbols::read_per_cpu_offsets(mem, refresh.pco_pa, refresh.num_cpus);
            // Latch `data_valid` once the guest has populated the
            // percpu offset table AND we've observed (or accepted
            // the absence of) the KASLR base. Both halves are
            // necessary — see the `data_valid` declaration for the
            // rationale on why pre-validity walks are catastrophic.
            // `__per_cpu_offset[]` is filled by the guest's
            // `setup_per_cpu_areas` one slot at a time; observing
            // only `[0] != 0` would latch the gate the moment the
            // BSP's slot is initialised, while later AP slots still
            // hold BSS zero. Walks for those APs would compute
            // `runqueues_kva + 0`, wrap to a non-DRAM PA, and
            // silently read zeros — the exact failure mode the gate
            // exists to prevent. Require every slot to be populated
            // (and the slice to be non-empty so a degenerate
            // `num_cpus == 0` cannot vacuously pass).
            if !data_valid
                && page_offset_resolved
                && !fresh.is_empty()
                && fresh.iter().all(|&v| v != 0)
            {
                data_valid = true;
                latched_page_offset = page_offset;
                eprintln!(
                    "DATA_VALID latched at iter={} page_offset={:#x} pco0={:#x}",
                    diag_iter,
                    page_offset,
                    fresh.first().copied().unwrap_or(0),
                );
            }
            rq_pas_buf = super::symbols::compute_rq_pas(
                refresh.runqueues_kva,
                &fresh,
                page_offset,
                refresh.per_cpu_start,
                refresh.kaslr_offset,
            );
            // `event_pcpu_pas` requires both fresh `__per_cpu_offset[]`
            // AND a non-null `*scx_root` (a scheduler must be
            // attached). Until the scheduler attaches, `scx_sched_kva`
            // is zero and `resolve_event_pcpu_pas` returns `None` —
            // event counters stay absent for that sample.
            event_pcpu_pas_buf = refresh.event.as_ref().and_then(|ev| {
                resolve_event_pcpu_pas(mem, ev.scx_root_pa, &ev.event_offsets, &fresh, page_offset)
            });
            per_cpu_offsets_buf = fresh;

            // Iteration counter — used by the DATA_VALID latch
            // eprintln and the FINAL diagnostic after the loop. We
            // dropped the per-iteration MONITOR/PHYS_BASE/HEX dumps
            // because nextest's stderr capture made the 15
            // eprintlns slower than the timerfd cadence and starved
            // the loop down to a handful of samples per run.
            // Saturating to avoid overflow on long runs.
            diag_iter = diag_iter.saturating_add(1);
        }

        // Pre-validity short circuit: skip every guest-memory walk
        // until both `__per_cpu_offset[]` and `page_offset` are
        // resolved. See the `data_valid` declaration for why
        // walking through silent-zero returns turns 100 ms iterations
        // into seconds. When invalid we leave `cpus` empty so the
        // sample push below records a degenerate sample rather than
        // a sample full of phantom zeros — `dump_trigger`'s
        // `!cpus.is_empty()` gate already short-circuits in that
        // case, and `MonitorThresholds::evaluate` tolerates
        // cpu-less samples.
        if data_valid {
            cpus.clear();
            cpus.extend(rq_pas_buf.iter().map(|&pa| read_rq_stats(mem, pa, offsets)));

            // Overlay event counters if available.
            if let (Some(pcpu_pas), Some(ev)) =
                (event_pcpu_pas_buf.as_deref(), &offsets.event_offsets)
            {
                for (i, cpu) in cpus.iter_mut().enumerate() {
                    if let Some(&pcpu_pa) = pcpu_pas.get(i) {
                        cpu.event_counters = Some(read_event_stats(mem, pcpu_pa, ev));
                    }
                }
            }

            // Overlay schedstat fields if available.
            if let Some(ss) = &offsets.schedstat_offsets {
                for (i, cpu) in cpus.iter_mut().enumerate() {
                    if let Some(&rq_pa) = rq_pas_buf.get(i) {
                        cpu.schedstat = Some(read_rq_schedstat(mem, rq_pa, ss));
                    }
                }
            }

            // Overlay sched domain tree if available.
            if let Some(sd) = &offsets.sched_domain_offsets {
                for (i, cpu) in cpus.iter_mut().enumerate() {
                    if let Some(&rq_pa) = rq_pas_buf.get(i) {
                        cpu.sched_domains = read_sched_domain_tree(
                            mem,
                            rq_pa,
                            sd,
                            page_offset,
                            start_kernel_map,
                            phys_base,
                        );
                    }
                }
            }

            // Stamp vCPU CPU times into the per-CPU snapshots. Reactive
            // stuck detection below reads these via `is_cpu_stuck`; the
            // post-hoc `MonitorThresholds::evaluate` path reads them off
            // the pushed samples.
            if let Some(vt) = vcpu_timing {
                let times = vt.read_cpu_times(&mut vcpu_timing_err_reported);
                for (i, cpu) in cpus.iter_mut().enumerate() {
                    if let Some(&t) = times.get(i) {
                        cpu.vcpu_cpu_time_ns = t;
                    }
                }
            }
        }

        // Read per-vCPU PMU counters (cycles / instructions /
        // cache-misses / branch-misses) into each snapshot. Errors
        // here are surfaced as `vcpu_perf = None` for that sample;
        // we don't fail the monitor over a transient read error.
        if let Some(pc) = perf_capture {
            match pc.read_all() {
                Ok(samples) => {
                    for (i, cpu) in cpus.iter_mut().enumerate() {
                        if let Some(s) = samples.get(i) {
                            cpu.vcpu_perf = Some(*s);
                        }
                    }
                }
                Err(e) => {
                    if !perf_read_err_reported {
                        tracing::warn!(
                            err = %e,
                            "perf counter read failed; vcpu_perf will be None until next successful sample"
                        );
                        perf_read_err_reported = true;
                    }
                }
            }
        }

        // Inline threshold evaluation for reactive dump. Each check
        // mirrors `MonitorThresholds::evaluate`: the same
        // `SustainedViolationTracker`, the same `is_cpu_stuck`
        // predicate, the same `imbalance_ratio`/`local_dsq_depth`
        // reads. Any drift would let the reactive SysRq-D trigger
        // fire on conditions the post-hoc verdict accepts (or vice
        // versa).
        if let Some(trigger) = dump_trigger
            && !dump_requested
            && !cpus.is_empty()
        {
            let t = &trigger.thresholds;
            let sample_idx = samples.len();

            // Imbalance check — use the shared sample method so the
            // min_nr.max(1)/max_nr calculation matches post-hoc.
            let tmp_sample = MonitorSample {
                elapsed_ms: 0,
                cpus: cpus.clone(),
                prog_stats: None,
            };
            let ratio = tmp_sample.imbalance_ratio();
            imbalance_tracker.record(ratio > t.max_imbalance_ratio, ratio, sample_idx);

            // DSQ depth check.
            let worst_dsq = cpus.iter().map(|c| c.local_dsq_depth).max().unwrap_or(0);
            dsq_tracker.record(
                worst_dsq > t.max_local_dsq_depth,
                worst_dsq as f64,
                sample_idx,
            );

            // Stuck check: per-CPU sustained window. Delegate to
            // `is_cpu_stuck` so reactive and post-hoc stuck paths
            // cannot drift — the predicate owns the idle + preempted
            // exemptions. `vcpu_cpu_time_ns` is already stamped into
            // `cpus[i]` (and into the last pushed sample) above, so the
            // helper sees the same vCPU timing the post-hoc path sees.
            if t.fail_on_stall
                && let Some(prev) = samples.last()
            {
                let n = prev.cpus.len().min(cpus.len()).min(stall_trackers.len());
                for i in 0..n {
                    let is_stall = is_cpu_stuck(&prev.cpus[i], &cpus[i], preemption_threshold_ns);
                    stall_trackers[i].record(is_stall, cpus[i].rq_clock as f64, sample_idx);
                }
            }
            let sustained = imbalance_tracker.sustained(t.sustained_samples)
                || dsq_tracker.sustained(t.sustained_samples)
                || stall_trackers
                    .iter()
                    .any(|s| s.sustained(t.sustained_samples));

            if sustained {
                // Push the `SIGNAL_VC_DUMP` wake byte through the
                // virtio-console RX queue via
                // [`crate::vmm::host_comms::request_dump`]. The
                // guest's `hvc0_poll_loop` blocks on `/dev/hvc0`,
                // recognises the byte, and dispatches SysRq-D
                // directly — no SHM control byte involved.
                if let Some(ref vc) = trigger.virtio_con {
                    crate::vmm::host_comms::request_dump(vc);
                } else {
                    // No virtio-console attached — every production
                    // path through `start_monitor` always populates
                    // `trigger.virtio_con` (the device is built
                    // unconditionally because `ktstr.kconfig`
                    // mandates `CONFIG_VIRTIO_CONSOLE=y`). Surface
                    // the missing handle so a host-side caller that
                    // omits the device sees the wake go nowhere
                    // instead of falling back silently.
                    tracing::warn!(
                        "dump_trigger: no virtio_console attached; \
                         SIGNAL_VC_DUMP not delivered"
                    );
                }
                dump_requested = true;
            }
        }

        // Gate the BPF struct_ops IDR walk on `data_valid` for the
        // same reason as the rq/sched_domain walks above: it chases
        // pointers through guest memory, and pre-validity reads
        // return silent zeros that look like phantom IDR nodes.
        // Skip when invalid; emit `None` so the sample push records
        // a degenerate entry without minutes of phantom traversal.
        let prog_stats = if data_valid {
            prog_stats_ctx.map(|ctx| {
                super::bpf_prog::walk_struct_ops_runtime_stats(
                    mem,
                    ctx.walk,
                    ctx.prog_idr_kva,
                    &ctx.offsets,
                    &per_cpu_offsets_buf,
                    ctx.start_kernel_map,
                    ctx.phys_base,
                )
            })
        } else {
            None
        };

        samples.push(MonitorSample {
            elapsed_ms: run_start.elapsed().as_millis() as u64,
            cpus: cpus.clone(),
            prog_stats,
        });

        // Block until the next tick or a kill_evt write. -1 timeout
        // is OK because both fds carry hard wakes — a missing
        // kill_evt write means the sampler keeps running on the
        // timerfd cadence and the kill atomic check at the top of
        // the loop body still terminates within one tick.
        match epoll.wait(-1, &mut epoll_buf) {
            Ok(n) => {
                for ev in &epoll_buf[..n] {
                    if ev.fd() == tick_fd {
                        // Drain the timerfd counter so the next
                        // epoll_wait blocks again. The expiry count
                        // is informational only — we always run one
                        // sampling pass per loop iteration whether
                        // 1 or 5 ticks accumulated, matching the
                        // legacy `thread::sleep(interval)` cadence.
                        let _ = tick_tfd.wait();
                    }
                    // kill_fd has nothing to drain at the EventFd
                    // level — `kill.load(Acquire)` at the top of
                    // the loop is the source of truth and the
                    // EventFd counter never gets read.
                }
            }
            Err(e) => {
                if e.raw_os_error() != Some(libc::EINTR) {
                    tracing::warn!(err = %e, "monitor: epoll_wait failed");
                    break;
                }
            }
        }
    }
    // Final post-loop diagnostic — captures post-boot state once,
    // after the kill flag flips. Distinguishes "guest never wrote
    // __per_cpu_offset[]" from "writes invisible to host monitor"
    // by re-reading the same PAs the early-window diag covered.
    if let Some(refresh) = rq_refresh {
        let fresh = super::symbols::read_per_cpu_offsets(mem, refresh.pco_pa, refresh.num_cpus);
        let valid_samples = samples
            .iter()
            .filter(|s| s.cpus.iter().any(|c| c.rq_clock > 0))
            .count();
        let pob_pa_live = refresh.page_offset_base_pa.unwrap_or(0);
        let pob_val = if pob_pa_live != 0 {
            mem.read_u64(pob_pa_live, 0)
        } else {
            0
        };
        eprintln!(
            "FINAL DIAG iters={iters} samples={samples} valid={valid} data_valid={dv} \
             page_offset={po:#x} phys_base={pb:#x} start_kernel_map={skm:#x} \
             pco_pa={pco_pa:#x} pob_pa={pob_pa:#x} pob_val={pob_val:#x} \
             pco0={pco0:#x} pco1={pco1:#x} runqueues_kva={rq:#x} \
             mem_size={ms:#x} rq_pa0={rq0:#x} kaslr_off={ko:#x}",
            iters = diag_iter,
            samples = samples.len(),
            valid = valid_samples,
            dv = data_valid,
            po = page_offset,
            pb = phys_base,
            skm = cfg.start_kernel_map,
            pco_pa = refresh.pco_pa,
            pob_pa = pob_pa_live,
            pob_val = pob_val,
            pco0 = fresh.first().copied().unwrap_or(0),
            pco1 = fresh.get(1).copied().unwrap_or(0),
            rq = refresh.runqueues_kva,
            ms = mem.size(),
            ko = refresh.kaslr_offset,
            rq0 = rq_pas_buf.first().copied().unwrap_or(0),
        );
        if let Some(&rq0_pa) = rq_pas_buf.first() {
            let raw_clock = mem.read_u64(rq0_pa, offsets.rq_clock);
            let raw_nr_running = mem.read_u32(rq0_pa, offsets.rq_nr_running);
            // Sanity: read first 8 bytes at rq0_pa (should be nr_running if offset 0 is correct)
            let raw_first_8 = mem.read_u64(rq0_pa, 0);
            // Read pco_pa itself to verify __per_cpu_offset[0] value
            let pco0_verify = mem.read_u64(refresh.pco_pa, 0);
            eprintln!(
                "FINAL DIAG rq0: clock_off={} nr_off={} raw_clock={raw_clock} raw_nr={raw_nr_running} \
                 raw_first_8={raw_first_8:#x} pco0_verify={pco0_verify:#x} rq0_pa={rq0_pa:#x} pco_pa={:#x}",
                offsets.rq_clock, offsets.rq_nr_running, refresh.pco_pa,
            );
        }
    }
    let shm_result = crate::vmm::host_comms::BulkDrainResult {
        entries: shm_entries,
    };
    MonitorLoopResult {
        samples,
        drain: shm_result,
        watchdog_observation,
        page_offset: latched_page_offset,
        preemption_threshold_ns,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::thread::JoinHandleExt;

    const THRESHOLD_NS: u64 = 10_000_000;

    #[test]
    fn evaluate_preempted_both_none_is_not_preempted() {
        assert!(!evaluate_preempted(None, None, THRESHOLD_NS));
    }

    #[test]
    fn evaluate_preempted_first_read_failed_is_not_preempted() {
        // Prev missing: we have no baseline, so the stall path must
        // fire if other conditions match. Treating this as preempted
        // would mask every first-sample stall.
        assert!(!evaluate_preempted(None, Some(1_000_000_000), THRESHOLD_NS));
    }

    #[test]
    fn evaluate_preempted_current_read_failed_is_not_preempted() {
        // Curr missing: likewise no evidence of preemption. The bug
        // this replaces would have returned `saturating_sub(big, 0)
        // < threshold` = false, or `saturating_sub(0, big)` = 0 <
        // threshold = true — both wrong.
        assert!(!evaluate_preempted(Some(1_000_000_000), None, THRESHOLD_NS));
    }

    #[test]
    fn evaluate_preempted_delta_below_threshold_is_preempted() {
        // 1ms delta with 10ms threshold: vCPU barely ran — preempted.
        assert!(evaluate_preempted(
            Some(1_000_000_000),
            Some(1_001_000_000),
            THRESHOLD_NS,
        ));
    }

    #[test]
    fn evaluate_preempted_delta_at_threshold_is_not_preempted() {
        // Exactly 10ms delta: not below threshold, so running, not preempted.
        assert!(!evaluate_preempted(
            Some(1_000_000_000),
            Some(1_010_000_000),
            THRESHOLD_NS,
        ));
    }

    #[test]
    fn evaluate_preempted_delta_above_threshold_is_not_preempted() {
        assert!(!evaluate_preempted(
            Some(1_000_000_000),
            Some(2_000_000_000),
            THRESHOLD_NS,
        ));
    }

    #[test]
    fn evaluate_preempted_non_monotonic_treated_as_no_progress() {
        // saturating_sub of a reverse-going clock yields 0 < threshold,
        // so "no measurable progress" maps to preempted=true. Documented
        // invariant: never unwinds into a false not-preempted when the
        // clock read jitters backwards.
        assert!(evaluate_preempted(
            Some(1_000_000_000),
            Some(999_000_000),
            THRESHOLD_NS,
        ));
    }

    #[test]
    fn evaluate_preempted_zero_threshold_never_preempted() {
        // Degenerate case: with threshold=0, nothing is strictly below,
        // so preempted is always false (stall path always fires).
        assert!(!evaluate_preempted(Some(100), Some(100), 0));
        assert!(!evaluate_preempted(Some(100), Some(200), 0));
    }

    /// Test helper: build a fresh `EventFd` for the `kill_evt`
    /// parameter. Tests that flip the `kill` atomic from a helper
    /// thread don't strictly need to write to this — the timerfd
    /// tick (typically 10 ms in tests) wakes `epoll_wait` within
    /// one interval — but production callers MUST write so the
    /// shutdown latency stays in the microsecond range.
    fn test_kill_evt() -> vmm_sys_util::eventfd::EventFd {
        vmm_sys_util::eventfd::EventFd::new(vmm_sys_util::eventfd::EFD_NONBLOCK)
            .expect("create kill EventFd")
    }

    fn test_config() -> MonitorConfig<'static> {
        MonitorConfig {
            event_pcpu_pas: None,
            dump_trigger: None,
            watchdog_override: None,
            vcpu_timing: None,
            perf_capture: None,
            preemption_threshold_ns: 0,
            prog_stats_ctx: None,
            page_offset: 0,
            start_kernel_map: super::super::symbols::START_KERNEL_MAP,
            phys_base: 0,
            rq_refresh: None,
            sys_rdy: None,
            watchdog_reset: None,
        }
    }

    fn test_offsets() -> KernelOffsets {
        KernelOffsets {
            rq_nr_running: 8,
            rq_clock: 16,
            rq_scx: 100,
            scx_rq_nr_running: 4,
            scx_rq_local_dsq: 20,
            scx_rq_flags: 8,
            dsq_nr: 0,
            event_offsets: None,
            schedstat_offsets: None,
            sched_domain_offsets: None,
            watchdog_offsets: None,
        }
    }

    /// Test helper: build a virtio_console handle wrapped in
    /// `Arc<PiMutex<...>>` so DumpTrigger tests can pass a real
    /// device and inspect its `pending_rx_bytes()` after the loop
    /// runs. Without DRIVER_OK on the queue the wake byte stays in
    /// `port0_pending_rx`, so the test can assert on byte equality
    /// without driving a full guest handshake.
    fn test_virtio_console()
    -> std::sync::Arc<crate::vmm::PiMutex<crate::vmm::virtio_console::VirtioConsole>> {
        std::sync::Arc::new(crate::vmm::PiMutex::new(
            crate::vmm::virtio_console::VirtioConsole::new(),
        ))
    }

    /// Build a byte buffer simulating a struct rq with the given field values.
    fn make_rq_buffer(
        offsets: &KernelOffsets,
        nr_running: u32,
        scx_nr: u32,
        dsq_nr: u32,
        clock: u64,
        flags: u32,
    ) -> Vec<u8> {
        let size = offsets.rq_scx + offsets.scx_rq_local_dsq + offsets.dsq_nr + 8;
        let mut buf = vec![0u8; size];

        buf[offsets.rq_nr_running..offsets.rq_nr_running + 4]
            .copy_from_slice(&nr_running.to_ne_bytes());
        buf[offsets.rq_clock..offsets.rq_clock + 8].copy_from_slice(&clock.to_ne_bytes());

        let scx_base = offsets.rq_scx;
        buf[scx_base + offsets.scx_rq_nr_running..scx_base + offsets.scx_rq_nr_running + 4]
            .copy_from_slice(&scx_nr.to_ne_bytes());
        buf[scx_base + offsets.scx_rq_flags..scx_base + offsets.scx_rq_flags + 4]
            .copy_from_slice(&flags.to_ne_bytes());

        let dsq_base = scx_base + offsets.scx_rq_local_dsq;
        buf[dsq_base + offsets.dsq_nr..dsq_base + offsets.dsq_nr + 4]
            .copy_from_slice(&dsq_nr.to_ne_bytes());
        buf
    }

    #[test]
    fn read_rq_stats_known_values() {
        let offsets = test_offsets();
        let buf = make_rq_buffer(&offsets, 5, 3, 7, 999_000, 0x1);
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let snap = read_rq_stats(&mem, 0, &offsets);
        assert_eq!(snap.nr_running, 5);
        assert_eq!(snap.scx_nr_running, 3);
        assert_eq!(snap.local_dsq_depth, 7);
        assert_eq!(snap.rq_clock, 999_000);
        assert_eq!(snap.scx_flags, 0x1);
    }

    #[test]
    fn read_rq_stats_all_zeros() {
        let offsets = test_offsets();
        let buf = make_rq_buffer(&offsets, 0, 0, 0, 0, 0);
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let snap = read_rq_stats(&mem, 0, &offsets);
        assert_eq!(snap.nr_running, 0);
        assert_eq!(snap.scx_nr_running, 0);
        assert_eq!(snap.local_dsq_depth, 0);
        assert_eq!(snap.rq_clock, 0);
        assert_eq!(snap.scx_flags, 0);
    }

    #[test]
    fn read_rq_stats_max_values() {
        let offsets = test_offsets();
        let buf = make_rq_buffer(&offsets, u32::MAX, u32::MAX, u32::MAX, u64::MAX, u32::MAX);
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let snap = read_rq_stats(&mem, 0, &offsets);
        assert_eq!(snap.nr_running, u32::MAX);
        assert_eq!(snap.scx_nr_running, u32::MAX);
        assert_eq!(snap.local_dsq_depth, u32::MAX);
        assert_eq!(snap.rq_clock, u64::MAX);
        assert_eq!(snap.scx_flags, u32::MAX);
    }

    #[test]
    fn read_u32_out_of_bounds() {
        let buf = [0xFFu8; 8];
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        // PA 6 + 4 bytes = 10 > 8, out of bounds
        assert_eq!(mem.read_u32(6, 0), 0);
        // Exactly at boundary: PA 4, offset 0 => addr 4, 4+4=8 == size, not >
        assert_eq!(mem.read_u32(4, 0), u32::from_ne_bytes([0xFF; 4]));
        // One past: PA 5, offset 0 => addr 5, 5+4=9 > 8
        assert_eq!(mem.read_u32(5, 0), 0);
    }

    #[test]
    fn read_u64_out_of_bounds() {
        let buf = [0xFFu8; 16];
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        // PA 10 + 8 = 18 > 16
        assert_eq!(mem.read_u64(10, 0), 0);
        // Exactly at boundary: PA 8, 8+8=16 == size
        assert_eq!(mem.read_u64(8, 0), u64::from_ne_bytes([0xFF; 8]));
        // One past
        assert_eq!(mem.read_u64(9, 0), 0);
    }

    #[test]
    fn monitor_loop_kill_immediately() {
        let offsets = test_offsets();
        let buf = make_rq_buffer(&offsets, 1, 1, 1, 100, 0);
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let kill = AtomicBool::new(true);
        let kill_evt = test_kill_evt();
        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[0],
            &offsets,
            Duration::from_millis(10),
            &kill,
            &kill_evt,
            Instant::now(),
            &test_config(),
        );
        assert!(samples.is_empty());
    }

    #[test]
    fn monitor_loop_one_iteration() {
        let offsets = test_offsets();
        let buf = make_rq_buffer(&offsets, 2, 1, 3, 500, 0);
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let kill = std::sync::Arc::new(AtomicBool::new(false));
        let kill_evt = test_kill_evt();

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(50));
                kill.store(true, Ordering::Release);
            })
        };

        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[0],
            &offsets,
            Duration::from_millis(10),
            &kill,
            &kill_evt,
            Instant::now(),
            &test_config(),
        );
        handle.join().unwrap();

        assert!(!samples.is_empty());
        assert_eq!(samples[0].cpus.len(), 1);
        assert_eq!(samples[0].cpus[0].nr_running, 2);
        assert_eq!(samples[0].cpus[0].scx_nr_running, 1);
        assert_eq!(samples[0].cpus[0].local_dsq_depth, 3);
        assert_eq!(samples[0].cpus[0].rq_clock, 500);
    }

    #[test]
    fn two_cpu_independent_reads() {
        let offsets = test_offsets();
        let buf0 = make_rq_buffer(&offsets, 10, 5, 2, 1000, 0x1);
        let buf1 = make_rq_buffer(&offsets, 20, 15, 8, 2000, 0x2);

        // Concatenate into a single memory region; CPU 1's rq starts after CPU 0's.
        let pa1 = buf0.len() as u64;
        let mut combined = buf0;
        combined.extend_from_slice(&buf1);

        // SAFETY: combined is a live local buffer (Vec<u8> or stack
        // array) whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(combined.as_ptr() as *mut u8, combined.len() as u64) };

        let snap0 = read_rq_stats(&mem, 0, &offsets);
        let snap1 = read_rq_stats(&mem, pa1, &offsets);

        assert_eq!(snap0.nr_running, 10);
        assert_eq!(snap0.scx_nr_running, 5);
        assert_eq!(snap0.local_dsq_depth, 2);
        assert_eq!(snap0.rq_clock, 1000);
        assert_eq!(snap0.scx_flags, 0x1);

        assert_eq!(snap1.nr_running, 20);
        assert_eq!(snap1.scx_nr_running, 15);
        assert_eq!(snap1.local_dsq_depth, 8);
        assert_eq!(snap1.rq_clock, 2000);
        assert_eq!(snap1.scx_flags, 0x2);
    }

    #[test]
    fn read_u32_nonzero_pa_and_offset() {
        // Check that PA + offset are combined correctly.
        let mut buf = [0u8; 32];
        // Place 0xDEADBEEF at byte 20 (PA=12, offset=8).
        buf[20..24].copy_from_slice(&0xDEADBEEFu32.to_ne_bytes());
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        assert_eq!(mem.read_u32(12, 8), 0xDEADBEEF);
    }

    #[test]
    fn read_u64_nonzero_pa_and_offset() {
        let mut buf = [0u8; 32];
        // Place value at byte 16 (PA=10, offset=6).
        buf[16..24].copy_from_slice(&0x0123456789ABCDEFu64.to_ne_bytes());
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        assert_eq!(mem.read_u64(10, 6), 0x0123456789ABCDEF);
    }

    #[test]
    fn monitor_loop_multi_cpu() {
        let offsets = test_offsets();
        let buf0 = make_rq_buffer(&offsets, 3, 2, 1, 100, 0);
        let buf1 = make_rq_buffer(&offsets, 7, 5, 4, 200, 0);
        let pa1 = buf0.len() as u64;
        let mut combined = buf0;
        combined.extend_from_slice(&buf1);

        // SAFETY: combined is a live local buffer (Vec<u8> or stack
        // array) whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(combined.as_ptr() as *mut u8, combined.len() as u64) };
        let kill = std::sync::Arc::new(AtomicBool::new(false));
        let kill_evt = test_kill_evt();

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(50));
                kill.store(true, Ordering::Release);
            })
        };

        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[0, pa1],
            &offsets,
            Duration::from_millis(10),
            &kill,
            &kill_evt,
            Instant::now(),
            &test_config(),
        );
        handle.join().unwrap();

        assert!(!samples.is_empty());
        // Each sample should have 2 CPUs.
        for s in &samples {
            assert_eq!(s.cpus.len(), 2);
        }
        // CPU 0 values
        assert_eq!(samples[0].cpus[0].nr_running, 3);
        assert_eq!(samples[0].cpus[0].scx_nr_running, 2);
        // CPU 1 values
        assert_eq!(samples[0].cpus[1].nr_running, 7);
        assert_eq!(samples[0].cpus[1].scx_nr_running, 5);
    }

    #[test]
    fn monitor_loop_elapsed_ms_progresses() {
        let offsets = test_offsets();
        let buf = make_rq_buffer(&offsets, 1, 1, 1, 100, 0);
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let kill = std::sync::Arc::new(AtomicBool::new(false));
        let kill_evt = test_kill_evt();

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(200));
                kill.store(true, Ordering::Release);
            })
        };

        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[0],
            &offsets,
            Duration::from_millis(10),
            &kill,
            &kill_evt,
            Instant::now(),
            &test_config(),
        );
        handle.join().unwrap();

        assert!(
            samples.len() >= 2,
            "need at least 2 samples, got {}",
            samples.len()
        );
        // elapsed_ms must be monotonically non-decreasing.
        for w in samples.windows(2) {
            assert!(
                w[1].elapsed_ms >= w[0].elapsed_ms,
                "elapsed_ms went backwards: {} -> {}",
                w[0].elapsed_ms,
                w[1].elapsed_ms
            );
        }
        // Last sample should have elapsed > 0.
        assert!(samples.last().unwrap().elapsed_ms > 0);
    }

    fn test_event_offsets() -> ScxEventOffsets {
        ScxEventOffsets {
            percpu_ptr_off: 0,
            event_stats_off: 0,
            ev_select_cpu_fallback: 0,
            ev_dispatch_local_dsq_offline: 8,
            ev_dispatch_keep_last: 16,
            ev_enq_skip_exiting: 24,
            ev_enq_skip_migration_disabled: 32,
            ev_reenq_immed: None,
            ev_reenq_local_repeat: None,
            ev_refill_slice_dfl: None,
            ev_bypass_duration: None,
            ev_bypass_dispatch: None,
            ev_bypass_activate: None,
            ev_insert_not_owned: None,
            ev_sub_bypass_dispatch: None,
        }
    }

    /// Build a byte buffer simulating a scx_sched_pcpu with event_stats.
    fn make_event_stats_buffer(
        ev: &ScxEventOffsets,
        fallback: i64,
        offline: i64,
        keep_last: i64,
        skip_exit: i64,
        skip_mig: i64,
    ) -> Vec<u8> {
        let size = ev.event_stats_off + ev.ev_enq_skip_migration_disabled + 8;
        let mut buf = vec![0u8; size];
        let base = ev.event_stats_off;
        buf[base + ev.ev_select_cpu_fallback..base + ev.ev_select_cpu_fallback + 8]
            .copy_from_slice(&fallback.to_ne_bytes());
        buf[base + ev.ev_dispatch_local_dsq_offline..base + ev.ev_dispatch_local_dsq_offline + 8]
            .copy_from_slice(&offline.to_ne_bytes());
        buf[base + ev.ev_dispatch_keep_last..base + ev.ev_dispatch_keep_last + 8]
            .copy_from_slice(&keep_last.to_ne_bytes());
        buf[base + ev.ev_enq_skip_exiting..base + ev.ev_enq_skip_exiting + 8]
            .copy_from_slice(&skip_exit.to_ne_bytes());
        buf[base + ev.ev_enq_skip_migration_disabled..base + ev.ev_enq_skip_migration_disabled + 8]
            .copy_from_slice(&skip_mig.to_ne_bytes());
        buf
    }

    #[test]
    fn read_event_stats_known_values() {
        let ev = test_event_offsets();
        let buf = make_event_stats_buffer(&ev, 42, 7, 100, 3, 5);
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let stats = read_event_stats(&mem, 0, &ev);
        assert_eq!(stats.select_cpu_fallback, 42);
        assert_eq!(stats.dispatch_local_dsq_offline, 7);
        assert_eq!(stats.dispatch_keep_last, 100);
        assert_eq!(stats.enq_skip_exiting, 3);
        assert_eq!(stats.enq_skip_migration_disabled, 5);
    }

    #[test]
    fn read_event_stats_zeros() {
        let ev = test_event_offsets();
        let buf = make_event_stats_buffer(&ev, 0, 0, 0, 0, 0);
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let stats = read_event_stats(&mem, 0, &ev);
        assert_eq!(stats.select_cpu_fallback, 0);
        assert_eq!(stats.dispatch_local_dsq_offline, 0);
    }

    #[test]
    fn read_event_stats_optional_fields() {
        let mut ev = test_event_offsets();
        // Place bypass_activate at offset 40 (after the 5 mandatory fields).
        ev.ev_bypass_activate = Some(40);
        let mut buf = [0u8; 48];
        let val: i64 = 999;
        buf[40..48].copy_from_slice(&val.to_ne_bytes());
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let stats = read_event_stats(&mem, 0, &ev);
        assert_eq!(stats.bypass_activate, 999);
        // Fields without offsets remain 0.
        assert_eq!(stats.reenq_immed, 0);
        assert_eq!(stats.bypass_duration, 0);
        assert_eq!(stats.sub_bypass_dispatch, 0);
    }

    #[test]
    fn read_i64_roundtrip() {
        let val: i64 = -12345;
        let buf = val.to_ne_bytes();
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        assert_eq!(mem.read_i64(0, 0), -12345);
    }

    #[test]
    fn write_u8_and_read_u8() {
        let mut buf = [0u8; 16];
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        mem.write_u8(0, 5, 0xAB);
        assert_eq!(mem.read_u8(0, 5), 0xAB);
        assert_eq!(buf[5], 0xAB);
    }

    #[test]
    fn write_u8_out_of_bounds() {
        let mut buf = [0u8; 4];
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        // Should not panic or write.
        mem.write_u8(4, 0, 0xFF);
        assert_eq!(buf, [0u8; 4]);
    }

    #[test]
    fn write_u64_and_read_u64() {
        let mut buf = [0u8; 32];
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        mem.write_u64(0, 8, 0xDEAD_BEEF_CAFE_1234);
        assert_eq!(mem.read_u64(0, 8), 0xDEAD_BEEF_CAFE_1234);
        assert_eq!(
            u64::from_ne_bytes(buf[8..16].try_into().unwrap()),
            0xDEAD_BEEF_CAFE_1234
        );
    }

    #[test]
    fn write_u64_out_of_bounds() {
        let mut buf = [0u8; 8];
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        // addr 1 + 8 = 9 > 8, out of bounds
        mem.write_u64(1, 0, 0xFF);
        assert_eq!(buf, [0u8; 8]);
    }

    #[test]
    fn write_u64_at_boundary() {
        let mut buf = [0u8; 16];
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        // PA 8 + 8 = 16 == size, should succeed
        mem.write_u64(8, 0, 0x0123_4567_89AB_CDEF);
        assert_eq!(mem.read_u64(8, 0), 0x0123_4567_89AB_CDEF);
    }

    #[test]
    fn read_u8_out_of_bounds() {
        let buf = [0xFFu8; 4];
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        assert_eq!(mem.read_u8(4, 0), 0);
        assert_eq!(mem.read_u8(3, 0), 0xFF);
    }

    #[test]
    fn read_rq_stats_has_no_event_counters() {
        let offsets = test_offsets();
        let buf = make_rq_buffer(&offsets, 1, 1, 1, 100, 0);
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let snap = read_rq_stats(&mem, 0, &offsets);
        assert!(snap.event_counters.is_none());
    }

    #[test]
    fn monitor_loop_with_event_counters() {
        let ev = test_event_offsets();
        let mut offsets = test_offsets();
        offsets.event_offsets = Some(ev.clone());

        let rq_buf = make_rq_buffer(&offsets, 1, 1, 1, 100, 0);
        let ev_buf = make_event_stats_buffer(&ev, 10, 20, 30, 40, 50);

        let rq_pa = 0u64;
        let ev_pa = rq_buf.len() as u64;
        let mut combined = rq_buf;
        combined.extend_from_slice(&ev_buf);

        // SAFETY: combined is a live local buffer (Vec<u8> or stack
        // array) whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(combined.as_ptr() as *mut u8, combined.len() as u64) };
        let kill = std::sync::Arc::new(AtomicBool::new(false));
        let kill_evt = test_kill_evt();

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(30));
                kill.store(true, Ordering::Release);
            })
        };

        let ev_pas = vec![ev_pa];
        let cfg = MonitorConfig {
            event_pcpu_pas: Some(&ev_pas),
            ..test_config()
        };
        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[rq_pa],
            &offsets,
            Duration::from_millis(10),
            &kill,
            &kill_evt,
            Instant::now(),
            &cfg,
        );
        handle.join().unwrap();

        assert!(!samples.is_empty());
        let counters = samples[0].cpus[0].event_counters.as_ref().unwrap();
        assert_eq!(counters.select_cpu_fallback, 10);
        assert_eq!(counters.dispatch_local_dsq_offline, 20);
        assert_eq!(counters.dispatch_keep_last, 30);
        assert_eq!(counters.enq_skip_exiting, 40);
        assert_eq!(counters.enq_skip_migration_disabled, 50);
    }

    #[test]
    fn monitor_loop_no_event_counters_when_none() {
        let offsets = test_offsets();
        let buf = make_rq_buffer(&offsets, 1, 1, 1, 100, 0);
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let kill = std::sync::Arc::new(AtomicBool::new(false));
        let kill_evt = test_kill_evt();

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(30));
                kill.store(true, Ordering::Release);
            })
        };

        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[0],
            &offsets,
            Duration::from_millis(10),
            &kill,
            &kill_evt,
            Instant::now(),
            &test_config(),
        );
        handle.join().unwrap();

        assert!(!samples.is_empty());
        assert!(samples[0].cpus[0].event_counters.is_none());
    }

    /// `rq_refresh` recomputes `rq_pas` from `__per_cpu_offset[]` each
    /// iteration. Build a buffer where:
    ///   - byte 0..16 = per-CPU offset table for 2 CPUs
    ///     (offsets are PAs of the per-CPU rq buffers minus
    ///     `runqueues_kva`; we set `runqueues_kva = 0` so the offset
    ///     IS the PA).
    ///   - rq buffers follow the offset table.
    ///     With `page_offset = 0`, `kva_to_pa(off + 0, 0) = off` so the
    ///     recomputed `rq_pas[cpu]` equals the offset slot.
    #[test]
    fn monitor_loop_rq_refresh_drives_pas() {
        let offsets = test_offsets();
        let pco_size: u64 = 16; // 2 CPUs * 8 bytes
        let rq0_buf = make_rq_buffer(&offsets, 11, 1, 1, 7777, 0);
        let rq1_buf = make_rq_buffer(&offsets, 22, 2, 2, 8888, 0);
        let rq0_pa = pco_size;
        let rq1_pa = pco_size + rq0_buf.len() as u64;

        let mut combined = vec![0u8; pco_size as usize];
        combined[0..8].copy_from_slice(&rq0_pa.to_ne_bytes());
        combined[8..16].copy_from_slice(&rq1_pa.to_ne_bytes());
        combined.extend_from_slice(&rq0_buf);
        combined.extend_from_slice(&rq1_buf);

        // SAFETY: combined is a live local buffer (Vec<u8> or stack
        // array) whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(combined.as_ptr() as *mut u8, combined.len() as u64) };
        let kill = std::sync::Arc::new(AtomicBool::new(false));
        let kill_evt = test_kill_evt();

        let refresh = RqRefresh {
            pco_pa: 0,
            runqueues_kva: 0,
            num_cpus: 2,
            page_offset_base_pa: None,
            event: None,
            per_cpu_start: 0,
            kaslr_offset: 0,
        };

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(30));
                kill.store(true, Ordering::Release);
            })
        };

        let cfg = MonitorConfig {
            rq_refresh: Some(&refresh),
            ..test_config()
        };
        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[],
            &offsets,
            Duration::from_millis(10),
            &kill,
            &kill_evt,
            Instant::now(),
            &cfg,
        );
        handle.join().unwrap();

        assert!(!samples.is_empty());
        assert_eq!(samples[0].cpus.len(), 2);
        assert_eq!(samples[0].cpus[0].rq_clock, 7777);
        assert_eq!(samples[0].cpus[0].nr_running, 11);
        assert_eq!(samples[0].cpus[1].rq_clock, 8888);
        assert_eq!(samples[0].cpus[1].nr_running, 22);
    }

    /// Boot-race regression test. Same layout as the refresh test, but
    /// the per-CPU offset table is all-zero (mirrors host monitor
    /// spawn time before guest BSP runs `setup_per_cpu_areas`). With
    /// `runqueues_kva = 0` and `page_offset = 0`, the recomputed PAs
    /// are 0. The real boot race produces phantom-zero snapshots
    /// because the recomputed PA falls outside guest memory; the
    /// `data_valid` gate inside `monitor_loop` (which requires every
    /// `__per_cpu_offset[]` slot to be non-zero before walking guest
    /// memory) prevents those phantom-zero walks. We can't reproduce
    /// the OOB silent-zero in a single-region `GuestMem` test, but
    /// we CAN verify that the loop tolerates a transient empty
    /// offset table without panicking AND that the gate keeps
    /// `cpus` empty for as long as the offsets are invalid. The
    /// assertion that `cpus` stays empty is the regression check —
    /// if the gate ever loosened (e.g. accepted partial zero
    /// offsets), the `read_rq_stats` walks would resume on stale
    /// zeros and surface as populated `cpus` with bogus counters.
    #[test]
    fn monitor_loop_rq_refresh_zero_offsets_no_panic() {
        let offsets = test_offsets();
        // 16 zero bytes simulating BSS __per_cpu_offset[2].
        #[allow(clippy::useless_vec)]
        let mem_buf = vec![0u8; 64];
        // SAFETY: mem_buf is a live local buffer whose backing
        // storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(mem_buf.as_ptr() as *mut u8, mem_buf.len() as u64) };
        let kill = std::sync::Arc::new(AtomicBool::new(false));
        let kill_evt = test_kill_evt();

        let refresh = RqRefresh {
            pco_pa: 0,
            runqueues_kva: 0,
            num_cpus: 2,
            page_offset_base_pa: None,
            event: None,
            per_cpu_start: 0,
            kaslr_offset: 0,
        };

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(30));
                kill.store(true, Ordering::Release);
            })
        };

        let cfg = MonitorConfig {
            rq_refresh: Some(&refresh),
            ..test_config()
        };
        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[],
            &offsets,
            Duration::from_millis(10),
            &kill,
            &kill_evt,
            Instant::now(),
            &cfg,
        );
        handle.join().unwrap();

        // Loop survives the zero-offset state without panicking and
        // pushes one sample per timerfd tick. The `data_valid` gate
        // inside `monitor_loop` keeps `cpus` empty until every slot
        // of `__per_cpu_offset[]` is non-zero — which never happens
        // here because the buffer stays zeroed for the entire run.
        // Asserting `cpus.is_empty()` for every sample pins the gate:
        // if it ever loosened (e.g. accepted partial zero offsets),
        // the walks would resume on zero PAs and `cpus` would carry
        // bogus counters.
        assert!(!samples.is_empty());
        for s in &samples {
            assert!(
                s.cpus.is_empty(),
                "data_valid gate should keep cpus empty when __per_cpu_offset[] is all-zero"
            );
        }
    }

    #[test]
    fn resolve_event_pcpu_pas_null_scx_root() {
        let ev = test_event_offsets();
        // scx_root pointer is 0 (null) — no scheduler loaded.
        let buf = [0u8; 64];
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let result = resolve_event_pcpu_pas(&mem, 0, &ev, &[0, 0x4000], 0);
        assert!(result.is_none());
    }

    #[test]
    fn monitor_loop_with_watchdog_override() {
        let offsets = test_offsets();
        // Layout:
        //   [rq_buf]
        //   [scx_root pointer slot @ scx_root_pa] (holds scx_sched KVA)
        //   [scx_sched struct @ sch_pa, with watchdog_timeout at watchdog_offset]
        // The monitor derefs *scx_root_pa -> KVA, translates via PAGE_OFFSET -> PA,
        // then writes jiffies at sch_pa + watchdog_offset.
        let rq_buf = make_rq_buffer(&offsets, 1, 1, 1, 100, 0);
        let scx_root_pa = rq_buf.len() as u64;
        let sch_pa = scx_root_pa + 8;
        let watchdog_offset: usize = 16;
        let page_offset = super::super::symbols::DEFAULT_PAGE_OFFSET;
        let scx_sched_kva = page_offset.wrapping_add(sch_pa);

        // Buffer = rq_buf | 8 bytes (scx_root slot) | 64 bytes (scx_sched stub).
        let mut combined = rq_buf;
        combined.extend_from_slice(&scx_sched_kva.to_ne_bytes());
        combined.extend_from_slice(&[0u8; 64]);

        // SAFETY: combined is a live local buffer (Vec<u8> or stack
        // array) whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(combined.as_mut_ptr(), combined.len() as u64) };
        let kill = std::sync::Arc::new(AtomicBool::new(false));
        let kill_evt = test_kill_evt();

        let wd = WatchdogOverride::ScxSched {
            scx_root_pa,
            watchdog_offset,
            jiffies: 99999,
            interval_pa: None,
            timestamp_pa: None,
            jiffies_64_pa: None,
        };

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(30));
                kill.store(true, Ordering::Release);
            })
        };

        let cfg = MonitorConfig {
            watchdog_override: Some(&wd),
            page_offset,
            ..test_config()
        };
        let MonitorLoopResult {
            samples,
            watchdog_observation,
            ..
        } = monitor_loop(
            &mem,
            &[0],
            &offsets,
            Duration::from_millis(10),
            &kill,
            &kill_evt,
            Instant::now(),
            &cfg,
        );
        handle.join().unwrap();

        assert!(!samples.is_empty());
        // Check the watchdog value was written at sch_pa + watchdog_offset.
        let write_pa = sch_pa as usize + watchdog_offset;
        let written = u64::from_ne_bytes(combined[write_pa..write_pa + 8].try_into().unwrap());
        assert_eq!(written, 99999);
        // Check monitor_loop recorded the observation.
        let obs = watchdog_observation.expect("watchdog_observation should be Some after write");
        assert_eq!(obs.expected_jiffies, 99999);
        assert_eq!(obs.observed_jiffies, 99999);
    }

    #[test]
    fn monitor_loop_watchdog_override_skipped_when_scx_root_null() {
        let offsets = test_offsets();
        // Layout: rq_buf | scx_root slot = 0 (no scheduler loaded).
        let rq_buf = make_rq_buffer(&offsets, 1, 1, 1, 100, 0);
        let scx_root_pa = rq_buf.len() as u64;
        let mut combined = rq_buf;
        combined.extend_from_slice(&[0u8; 8]); // scx_root = null
        // Extra space in case of accidental write via garbage deref.
        combined.extend_from_slice(&[0u8; 128]);

        // SAFETY: combined is a live local buffer (Vec<u8> or stack
        // array) whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(combined.as_mut_ptr(), combined.len() as u64) };
        let kill = std::sync::Arc::new(AtomicBool::new(false));
        let kill_evt = test_kill_evt();
        let wd = WatchdogOverride::ScxSched {
            scx_root_pa,
            watchdog_offset: 16,
            jiffies: 0xDEADBEEF,
            interval_pa: None,
            timestamp_pa: None,
            jiffies_64_pa: None,
        };

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(30));
                kill.store(true, Ordering::Release);
            })
        };

        let cfg = MonitorConfig {
            watchdog_override: Some(&wd),
            ..test_config()
        };
        let MonitorLoopResult {
            watchdog_observation,
            ..
        } = monitor_loop(
            &mem,
            &[0],
            &offsets,
            Duration::from_millis(10),
            &kill,
            &kill_evt,
            Instant::now(),
            &cfg,
        );
        handle.join().unwrap();

        // No write should have happened: buffer is all zeros past rq_buf.
        assert!(
            combined[scx_root_pa as usize..].iter().all(|&b| b == 0),
            "no write should occur when scx_root is null"
        );
        // No observation should have been recorded.
        assert!(
            watchdog_observation.is_none(),
            "watchdog_observation should be None when scx_root is null"
        );
    }

    #[test]
    fn monitor_loop_watchdog_static_global_writes_directly() {
        let offsets = test_offsets();
        let rq_buf = make_rq_buffer(&offsets, 1, 1, 1, 100, 0);
        let watchdog_pa = rq_buf.len() as u64;

        let mut combined = rq_buf;
        combined.extend_from_slice(&[0u8; 8]);

        // SAFETY: combined is a live local buffer (Vec<u8> or stack
        // array) whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(combined.as_mut_ptr(), combined.len() as u64) };
        let kill = std::sync::Arc::new(AtomicBool::new(false));
        let kill_evt = test_kill_evt();

        let wd = WatchdogOverride::StaticGlobal {
            watchdog_timeout_pa: watchdog_pa,
            jiffies: 77777,
            interval_pa: None,
            timestamp_pa: None,
            jiffies_64_pa: None,
        };

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(30));
                kill.store(true, Ordering::Release);
            })
        };

        let cfg = MonitorConfig {
            watchdog_override: Some(&wd),
            ..test_config()
        };
        let MonitorLoopResult {
            samples,
            watchdog_observation,
            ..
        } = monitor_loop(
            &mem,
            &[0],
            &offsets,
            Duration::from_millis(10),
            &kill,
            &kill_evt,
            Instant::now(),
            &cfg,
        );
        handle.join().unwrap();

        assert!(!samples.is_empty());
        let written = u64::from_ne_bytes(
            combined[watchdog_pa as usize..watchdog_pa as usize + 8]
                .try_into()
                .unwrap(),
        );
        assert_eq!(written, 77777);
        let obs = watchdog_observation.expect("watchdog_observation should be Some");
        assert_eq!(obs.expected_jiffies, 77777);
        assert_eq!(obs.observed_jiffies, 77777);
    }

    #[test]
    fn monitor_loop_dump_trigger_fires_on_imbalance() {
        let offsets = test_offsets();
        // Two rq buffers: CPU0 = 1 task, CPU1 = 20 tasks -> ratio=20 >> threshold.
        let buf0 = make_rq_buffer(&offsets, 1, 1, 1, 100, 0);
        let buf1 = make_rq_buffer(&offsets, 20, 20, 1, 200, 0);
        let pa1 = buf0.len() as u64;
        let mut combined = buf0;
        combined.extend_from_slice(&buf1);
        // Append SHM region (carried for diagnostics; no longer
        // touched by the dump path).
        combined.extend(vec![0u8; 64]);

        // SAFETY: combined is a live local buffer (Vec<u8> or stack
        // array) whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(combined.as_mut_ptr(), combined.len() as u64) };
        let kill = std::sync::Arc::new(AtomicBool::new(false));
        let kill_evt = test_kill_evt();

        let virtio_con = test_virtio_console();
        let trigger = DumpTrigger {
            thresholds: super::super::MonitorThresholds {
                max_imbalance_ratio: 2.0,
                sustained_samples: 2,
                fail_on_stall: false,
                ..Default::default()
            },
            virtio_con: Some(virtio_con.clone()),
        };

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(200));
                kill.store(true, Ordering::Release);
            })
        };

        let cfg = MonitorConfig {
            dump_trigger: Some(&trigger),
            ..test_config()
        };
        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[0, pa1],
            &offsets,
            Duration::from_millis(10),
            &kill,
            &kill_evt,
            Instant::now(),
            &cfg,
        );
        handle.join().unwrap();

        assert!(!samples.is_empty());
        // Check `SIGNAL_VC_DUMP` was queued for the guest. Without
        // DRIVER_OK on the queue the byte stays in `port0_pending_rx`
        // — that's the observable for tests.
        let pending = virtio_con.lock().pending_rx_bytes();
        assert!(
            pending.contains(&crate::vmm::virtio_console::SIGNAL_VC_DUMP),
            "imbalance threshold violation should queue SIGNAL_VC_DUMP; \
             pending RX bytes: {pending:?}"
        );
    }

    #[test]
    fn monitor_loop_dump_trigger_stall_with_sustained_window() {
        // Reactive stall path: stuck rq_clock with nr_running>0 triggers
        // dump after sustained_samples consecutive stall pairs.
        let offsets = test_offsets();
        // Single CPU: nr_running=2 (busy), rq_clock stuck at 5000.
        // Need a second CPU with a different clock value so samples
        // differ (otherwise all-same-clock triggers the uninitialized
        // check in from_samples, though monitor_loop's reactive path
        // doesn't use from_samples — it checks inline).
        let buf = make_rq_buffer(&offsets, 2, 1, 1, 5000, 0);
        let mut combined = buf;
        combined.extend(vec![0u8; 64]);

        // SAFETY: combined is a live local buffer (Vec<u8> or stack
        // array) whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(combined.as_mut_ptr(), combined.len() as u64) };
        let kill = std::sync::Arc::new(AtomicBool::new(false));
        let kill_evt = test_kill_evt();

        let virtio_con = test_virtio_console();
        let trigger = DumpTrigger {
            thresholds: super::super::MonitorThresholds {
                max_imbalance_ratio: 100.0,
                max_local_dsq_depth: 10000,
                fail_on_stall: true,
                sustained_samples: 2,
                ..Default::default()
            },
            virtio_con: Some(virtio_con.clone()),
        };

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(200));
                kill.store(true, Ordering::Release);
            })
        };

        let cfg = MonitorConfig {
            dump_trigger: Some(&trigger),
            ..test_config()
        };
        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[0],
            &offsets,
            Duration::from_millis(10),
            &kill,
            &kill_evt,
            Instant::now(),
            &cfg,
        );
        handle.join().unwrap();

        // Should have enough samples for 2+ stall pairs.
        assert!(
            samples.len() >= 3,
            "need >= 3 samples for 2 stall pairs, got {}",
            samples.len()
        );
        // Dump should have fired due to sustained stall.
        let pending = virtio_con.lock().pending_rx_bytes();
        assert!(
            pending.contains(&crate::vmm::virtio_console::SIGNAL_VC_DUMP),
            "stall should trigger SIGNAL_VC_DUMP after sustained_samples=2; \
             pending RX bytes: {pending:?}"
        );
    }

    #[test]
    fn monitor_loop_dump_trigger_idle_cpu_no_stall() {
        // Reactive path: nr_running==0 (idle) with stuck rq_clock should
        // NOT trigger the dump, even with fail_on_stall=true.
        let offsets = test_offsets();
        // CPU idle: nr_running=0, rq_clock stuck at 5000.
        let buf = make_rq_buffer(&offsets, 0, 0, 0, 5000, 0);
        let mut combined = buf;
        combined.extend(vec![0u8; 64]);

        // SAFETY: combined is a live local buffer (Vec<u8> or stack
        // array) whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(combined.as_mut_ptr(), combined.len() as u64) };
        let kill = std::sync::Arc::new(AtomicBool::new(false));
        let kill_evt = test_kill_evt();

        let virtio_con = test_virtio_console();
        let trigger = DumpTrigger {
            thresholds: super::super::MonitorThresholds {
                max_imbalance_ratio: 100.0,
                max_local_dsq_depth: 10000,
                fail_on_stall: true,
                sustained_samples: 1,
                ..Default::default()
            },
            virtio_con: Some(virtio_con.clone()),
        };

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(200));
                kill.store(true, Ordering::Release);
            })
        };

        let cfg = MonitorConfig {
            dump_trigger: Some(&trigger),
            ..test_config()
        };
        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[0],
            &offsets,
            Duration::from_millis(10),
            &kill,
            &kill_evt,
            Instant::now(),
            &cfg,
        );
        handle.join().unwrap();

        assert!(
            samples.len() >= 2,
            "need >= 2 samples, got {}",
            samples.len()
        );
        // Dump should NOT have fired — idle CPU is exempt.
        let pending = virtio_con.lock().pending_rx_bytes();
        assert!(
            !pending.contains(&crate::vmm::virtio_console::SIGNAL_VC_DUMP),
            "idle CPU should not queue SIGNAL_VC_DUMP; pending RX bytes: {pending:?}"
        );
    }

    #[test]
    fn monitor_loop_vcpu_timing_preempted_no_stall() {
        // Sleeping thread: CPU time stays near zero between samples.
        // rq_clock stuck + CPU time not advancing = preempted, suppress stall.
        // 30ms interval gives margin on loaded hosts. Explicit threshold
        // (10ms) avoids host CONFIG_HZ dependency.
        let offsets = test_offsets();
        let buf = make_rq_buffer(&offsets, 2, 1, 1, 5000, 0);
        let mut combined = buf;
        combined.extend(vec![0u8; 64]);

        // SAFETY: combined is a live local buffer (Vec<u8> or stack
        // array) whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(combined.as_mut_ptr(), combined.len() as u64) };
        let kill = std::sync::Arc::new(AtomicBool::new(false));
        let kill_evt = test_kill_evt();

        let sleeper_kill = std::sync::Arc::new(AtomicBool::new(false));
        let sleeper_kill_clone = sleeper_kill.clone();
        let sleeper = std::thread::Builder::new()
            .name("vcpu-sleeper".into())
            .spawn(move || {
                while !sleeper_kill_clone.load(Ordering::Acquire) {
                    std::thread::sleep(Duration::from_millis(100));
                }
            })
            .unwrap();

        let pt = sleeper.as_pthread_t() as libc::pthread_t;
        let vcpu_timing = VcpuTiming { pthreads: vec![pt] };

        let virtio_con = test_virtio_console();
        let trigger = DumpTrigger {
            thresholds: super::super::MonitorThresholds {
                max_imbalance_ratio: 100.0,
                max_local_dsq_depth: 10000,
                fail_on_stall: true,
                sustained_samples: 1,
                ..Default::default()
            },
            virtio_con: Some(virtio_con.clone()),
        };

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(150));
                kill.store(true, Ordering::Release);
            })
        };

        let cfg = MonitorConfig {
            dump_trigger: Some(&trigger),
            vcpu_timing: Some(&vcpu_timing),
            preemption_threshold_ns: 10_000_000,
            ..test_config()
        };
        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[0],
            &offsets,
            Duration::from_millis(30),
            &kill,
            &kill_evt,
            Instant::now(),
            &cfg,
        );
        handle.join().unwrap();
        sleeper_kill.store(true, Ordering::Release);
        let _ = sleeper.join();

        assert!(
            samples.len() >= 2,
            "need >= 2 samples, got {}",
            samples.len()
        );
        let pending = virtio_con.lock().pending_rx_bytes();
        assert!(
            !pending.contains(&crate::vmm::virtio_console::SIGNAL_VC_DUMP),
            "preempted vCPU should not queue SIGNAL_VC_DUMP; pending RX bytes: {pending:?}"
        );
    }

    #[test]
    fn monitor_loop_vcpu_timing_running_stall_fires() {
        // Busy-spinning thread: accumulates CPU time every interval.
        // 30ms interval ensures spinner clears the 10ms preemption
        // threshold with margin.
        // rq_clock stuck + CPU time advancing = real stall. Explicit
        // threshold (10ms) avoids host CONFIG_HZ dependency (CONFIG_HZ=250
        // gives 40ms threshold, which would mask 30ms of spin time).
        let offsets = test_offsets();
        let buf = make_rq_buffer(&offsets, 2, 1, 1, 5000, 0);
        let mut combined = buf;
        combined.extend(vec![0u8; 64]);

        // SAFETY: combined is a live local buffer (Vec<u8> or stack
        // array) whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(combined.as_mut_ptr(), combined.len() as u64) };
        let kill = std::sync::Arc::new(AtomicBool::new(false));
        let kill_evt = test_kill_evt();

        let spinner_kill = std::sync::Arc::new(AtomicBool::new(false));
        let spinner_kill_clone = spinner_kill.clone();
        let spinner = std::thread::Builder::new()
            .name("vcpu-spinner".into())
            .spawn(move || {
                while !spinner_kill_clone.load(Ordering::Relaxed) {
                    std::hint::spin_loop();
                }
            })
            .unwrap();

        let pt = spinner.as_pthread_t() as libc::pthread_t;
        let vcpu_timing = VcpuTiming { pthreads: vec![pt] };

        let virtio_con = test_virtio_console();
        let trigger = DumpTrigger {
            thresholds: super::super::MonitorThresholds {
                max_imbalance_ratio: 100.0,
                max_local_dsq_depth: 10000,
                fail_on_stall: true,
                sustained_samples: 2,
                ..Default::default()
            },
            virtio_con: Some(virtio_con.clone()),
        };

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(200));
                kill.store(true, Ordering::Release);
            })
        };

        let cfg = MonitorConfig {
            dump_trigger: Some(&trigger),
            vcpu_timing: Some(&vcpu_timing),
            preemption_threshold_ns: 10_000_000,
            ..test_config()
        };
        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[0],
            &offsets,
            Duration::from_millis(30),
            &kill,
            &kill_evt,
            Instant::now(),
            &cfg,
        );
        handle.join().unwrap();
        spinner_kill.store(true, Ordering::Release);
        let _ = spinner.join();

        assert!(
            samples.len() >= 3,
            "need >= 3 samples for 2 stall pairs, got {}",
            samples.len()
        );
        let pending = virtio_con.lock().pending_rx_bytes();
        assert!(
            pending.contains(&crate::vmm::virtio_console::SIGNAL_VC_DUMP),
            "real stall (vCPU running, clock stuck, nr_running>0) should queue \
             SIGNAL_VC_DUMP; pending RX bytes: {pending:?}"
        );
    }

    #[test]
    fn reactive_and_evaluate_stall_consistency() {
        // Check that the reactive path (monitor_loop with dump_trigger)
        // and the post-hoc path (evaluate) agree on stall detection.
        // Build a scenario where stall fires: stuck rq_clock, nr_running>0,
        // sustained_samples=2.
        // Two CPUs: cpu0 stuck (rq_clock=5000), cpu1 has a different
        // clock value from cpu0 in each sample, so data_looks_valid
        // sees non-identical clocks.
        let offsets = test_offsets();
        let buf0 = make_rq_buffer(&offsets, 2, 1, 1, 5000, 0);
        let buf1 = make_rq_buffer(&offsets, 1, 1, 1, 9000, 0);
        let pa1 = buf0.len() as u64;
        let mut combined = buf0;
        combined.extend_from_slice(&buf1);
        combined.extend(vec![0u8; 64]);

        // SAFETY: combined is a live local buffer (Vec<u8> or stack
        // array) whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(combined.as_mut_ptr(), combined.len() as u64) };
        let kill = std::sync::Arc::new(AtomicBool::new(false));
        let kill_evt = test_kill_evt();

        let thresholds = super::super::MonitorThresholds {
            max_imbalance_ratio: 100.0,
            max_local_dsq_depth: 10000,
            fail_on_stall: true,
            sustained_samples: 2,
            ..Default::default()
        };

        let virtio_con = test_virtio_console();
        let trigger = DumpTrigger {
            thresholds,
            virtio_con: Some(virtio_con.clone()),
        };

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(200));
                kill.store(true, Ordering::Release);
            })
        };

        let cfg = MonitorConfig {
            dump_trigger: Some(&trigger),
            ..test_config()
        };
        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[0, pa1],
            &offsets,
            Duration::from_millis(10),
            &kill,
            &kill_evt,
            Instant::now(),
            &cfg,
        );
        handle.join().unwrap();

        assert!(
            samples.len() >= 3,
            "need >= 3 samples, got {}",
            samples.len()
        );

        // Reactive path result: check if dump fired (via the
        // SIGNAL_VC_DUMP wake byte queued for the guest).
        let pending = virtio_con.lock().pending_rx_bytes();
        let reactive_stall = pending.contains(&crate::vmm::virtio_console::SIGNAL_VC_DUMP);

        // Post-hoc evaluate path on the same samples.
        let summary = super::super::MonitorSummary::from_samples(&samples);
        let report = super::super::MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let verdict = thresholds.evaluate(&report);

        // Both paths should agree: stall detected on cpu0.
        assert!(reactive_stall, "reactive path should detect stall");
        assert!(
            !verdict.passed,
            "evaluate should detect stall: {:?}",
            verdict.details
        );
        assert!(
            verdict.details.iter().any(|d| d.contains("rq_clock stall")),
            "evaluate details should mention stall: {:?}",
            verdict.details
        );
    }

    #[test]
    fn reactive_and_evaluate_idle_consistency() {
        // Both reactive and evaluate should agree: idle CPU is exempt.
        let offsets = test_offsets();
        // nr_running=0, rq_clock stuck.
        let buf = make_rq_buffer(&offsets, 0, 0, 0, 5000, 0);
        let mut combined = buf;
        combined.extend(vec![0u8; 64]);

        // SAFETY: combined is a live local buffer (Vec<u8> or stack
        // array) whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(combined.as_mut_ptr(), combined.len() as u64) };
        let kill = std::sync::Arc::new(AtomicBool::new(false));
        let kill_evt = test_kill_evt();

        let thresholds = super::super::MonitorThresholds {
            max_imbalance_ratio: 100.0,
            max_local_dsq_depth: 10000,
            fail_on_stall: true,
            sustained_samples: 1,
            ..Default::default()
        };

        let virtio_con = test_virtio_console();
        let trigger = DumpTrigger {
            thresholds,
            virtio_con: Some(virtio_con.clone()),
        };

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(200));
                kill.store(true, Ordering::Release);
            })
        };

        let cfg = MonitorConfig {
            dump_trigger: Some(&trigger),
            ..test_config()
        };
        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[0],
            &offsets,
            Duration::from_millis(10),
            &kill,
            &kill_evt,
            Instant::now(),
            &cfg,
        );
        handle.join().unwrap();

        assert!(
            samples.len() >= 2,
            "need >= 2 samples, got {}",
            samples.len()
        );

        // Reactive: dump should NOT fire — no SIGNAL_VC_DUMP queued.
        let pending = virtio_con.lock().pending_rx_bytes();
        assert!(
            !pending.contains(&crate::vmm::virtio_console::SIGNAL_VC_DUMP),
            "reactive: idle CPU should not queue SIGNAL_VC_DUMP; pending RX: {pending:?}"
        );

        // Evaluate: from_samples should not detect stuck.
        let summary = super::super::MonitorSummary::from_samples(&samples);
        assert!(
            !summary.stuck_detected,
            "from_samples: idle CPU should not flag stuck"
        );

        // Evaluate verdict: should pass (no stall on idle CPU).
        // Note: evaluate may pass via data_looks_valid returning false
        // (all-same clocks with single CPU) — that's consistent behavior.
        let report = super::super::MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let verdict = thresholds.evaluate(&report);
        assert!(
            verdict.passed,
            "evaluate: idle CPU should pass: {:?}",
            verdict.details
        );
    }

    fn test_schedstat_offsets() -> super::super::btf_offsets::SchedstatOffsets {
        super::super::btf_offsets::SchedstatOffsets {
            rq_sched_info: 200,
            sched_info_run_delay: 8,
            sched_info_pcount: 0,
            rq_yld_count: 300,
            rq_sched_count: 304,
            rq_sched_goidle: 308,
            rq_ttwu_count: 312,
            rq_ttwu_local: 316,
        }
    }

    /// Build a byte buffer simulating a struct rq with schedstat fields.
    #[allow(clippy::too_many_arguments)]
    fn make_schedstat_buffer(
        ss: &super::super::btf_offsets::SchedstatOffsets,
        run_delay: u64,
        pcount: u64,
        yld_count: u32,
        sched_count: u32,
        sched_goidle: u32,
        ttwu_count: u32,
        ttwu_local: u32,
    ) -> Vec<u8> {
        let size = ss.rq_ttwu_local + 4 + 8;
        let mut buf = vec![0u8; size];

        let si_base = ss.rq_sched_info;
        buf[si_base + ss.sched_info_pcount..si_base + ss.sched_info_pcount + 8]
            .copy_from_slice(&pcount.to_ne_bytes());
        buf[si_base + ss.sched_info_run_delay..si_base + ss.sched_info_run_delay + 8]
            .copy_from_slice(&run_delay.to_ne_bytes());

        buf[ss.rq_yld_count..ss.rq_yld_count + 4].copy_from_slice(&yld_count.to_ne_bytes());
        buf[ss.rq_sched_count..ss.rq_sched_count + 4].copy_from_slice(&sched_count.to_ne_bytes());
        buf[ss.rq_sched_goidle..ss.rq_sched_goidle + 4]
            .copy_from_slice(&sched_goidle.to_ne_bytes());
        buf[ss.rq_ttwu_count..ss.rq_ttwu_count + 4].copy_from_slice(&ttwu_count.to_ne_bytes());
        buf[ss.rq_ttwu_local..ss.rq_ttwu_local + 4].copy_from_slice(&ttwu_local.to_ne_bytes());
        buf
    }

    #[test]
    fn read_rq_schedstat_known_values() {
        let ss = test_schedstat_offsets();
        let buf = make_schedstat_buffer(&ss, 50000, 10, 3, 100, 20, 80, 40);
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let stats = read_rq_schedstat(&mem, 0, &ss);
        assert_eq!(stats.run_delay, 50000);
        assert_eq!(stats.pcount, 10);
        assert_eq!(stats.yld_count, 3);
        assert_eq!(stats.sched_count, 100);
        assert_eq!(stats.sched_goidle, 20);
        assert_eq!(stats.ttwu_count, 80);
        assert_eq!(stats.ttwu_local, 40);
    }

    #[test]
    fn read_rq_schedstat_zeros() {
        let ss = test_schedstat_offsets();
        let buf = make_schedstat_buffer(&ss, 0, 0, 0, 0, 0, 0, 0);
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let stats = read_rq_schedstat(&mem, 0, &ss);
        assert_eq!(stats.run_delay, 0);
        assert_eq!(stats.pcount, 0);
        assert_eq!(stats.yld_count, 0);
        assert_eq!(stats.sched_count, 0);
        assert_eq!(stats.sched_goidle, 0);
        assert_eq!(stats.ttwu_count, 0);
        assert_eq!(stats.ttwu_local, 0);
    }

    #[test]
    fn monitor_loop_with_schedstat_overlay() {
        let ss = test_schedstat_offsets();
        let mut offsets = test_offsets();
        offsets.schedstat_offsets = Some(ss.clone());

        // Build a buffer that contains both rq fields and schedstat fields.
        // The rq buffer must be large enough to cover schedstat offsets.
        let rq_size = ss.rq_ttwu_local + 4 + 8;
        let mut buf = vec![0u8; rq_size];

        // Write rq base fields.
        buf[offsets.rq_nr_running..offsets.rq_nr_running + 4].copy_from_slice(&2u32.to_ne_bytes());
        buf[offsets.rq_clock..offsets.rq_clock + 8].copy_from_slice(&500u64.to_ne_bytes());

        // Write schedstat fields.
        let si_base = ss.rq_sched_info;
        buf[si_base + ss.sched_info_run_delay..si_base + ss.sched_info_run_delay + 8]
            .copy_from_slice(&12345u64.to_ne_bytes());
        buf[si_base + ss.sched_info_pcount..si_base + ss.sched_info_pcount + 8]
            .copy_from_slice(&7u64.to_ne_bytes());
        buf[ss.rq_sched_count..ss.rq_sched_count + 4].copy_from_slice(&42u32.to_ne_bytes());

        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let kill = std::sync::Arc::new(AtomicBool::new(false));
        let kill_evt = test_kill_evt();

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(30));
                kill.store(true, Ordering::Release);
            })
        };

        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[0],
            &offsets,
            Duration::from_millis(10),
            &kill,
            &kill_evt,
            Instant::now(),
            &test_config(),
        );
        handle.join().unwrap();

        assert!(!samples.is_empty());
        let ss_snap = samples[0].cpus[0].schedstat.as_ref().unwrap();
        assert_eq!(ss_snap.run_delay, 12345);
        assert_eq!(ss_snap.pcount, 7);
        assert_eq!(ss_snap.sched_count, 42);
    }

    #[test]
    fn monitor_loop_no_schedstat_when_none() {
        let offsets = test_offsets();
        assert!(offsets.schedstat_offsets.is_none());

        let buf = make_rq_buffer(&offsets, 1, 1, 1, 100, 0);
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let kill = std::sync::Arc::new(AtomicBool::new(false));
        let kill_evt = test_kill_evt();

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(30));
                kill.store(true, Ordering::Release);
            })
        };

        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[0],
            &offsets,
            Duration::from_millis(10),
            &kill,
            &kill_evt,
            Instant::now(),
            &test_config(),
        );
        handle.join().unwrap();

        assert!(!samples.is_empty());
        assert!(samples[0].cpus[0].schedstat.is_none());
    }

    fn test_sched_domain_offsets() -> SchedDomainOffsets {
        // Synthetic offsets for a sched_domain struct.
        // Layout: parent(0) level(8) flags(12) name(16) span_weight(24)
        //         balance_interval(28) nr_balance_failed(32)
        //         newidle_call(36) newidle_success(40) newidle_ratio(44)
        //         max_newidle_lb_cost(48)
        //         [stats at 56+]
        SchedDomainOffsets {
            rq_sd: 400,
            sd_parent: 0,
            sd_level: 8,
            sd_flags: 12,
            sd_name: 16,
            sd_span_weight: 24,
            sd_balance_interval: 28,
            sd_nr_balance_failed: 32,
            sd_newidle_call: Some(36),
            sd_newidle_success: Some(40),
            sd_newidle_ratio: Some(44),
            sd_max_newidle_lb_cost: 48,
            stats_offsets: Some(test_sd_stats_offsets()),
        }
    }

    fn test_sd_stats_offsets() -> SchedDomainStatsOffsets {
        SchedDomainStatsOffsets {
            sd_lb_count: 56,
            sd_lb_failed: 68,
            sd_lb_balanced: 80,
            sd_lb_imbalance_load: 92,
            sd_lb_imbalance_util: 104,
            sd_lb_imbalance_task: 116,
            sd_lb_imbalance_misfit: 128,
            sd_lb_gained: 140,
            sd_lb_hot_gained: 152,
            sd_lb_nobusyg: 164,
            sd_lb_nobusyq: 176,
            sd_alb_count: 188,
            sd_alb_failed: 192,
            sd_alb_pushed: 196,
            sd_sbe_count: 200,
            sd_sbe_balanced: 204,
            sd_sbe_pushed: 208,
            sd_sbf_count: 212,
            sd_sbf_balanced: 216,
            sd_sbf_pushed: 220,
            sd_ttwu_wake_remote: 224,
            sd_ttwu_move_affine: 228,
            sd_ttwu_move_balance: 232,
        }
    }

    /// Build a synthetic sched_domain buffer with known values.
    /// `parent_kva`: KVA of parent domain (0 = no parent).
    /// `name_kva`: KVA of name string (0 = no name).
    /// Returns a buffer representing one sched_domain struct.
    #[allow(clippy::too_many_arguments)]
    fn make_sd_buffer(
        sd: &SchedDomainOffsets,
        parent_kva: u64,
        level: i32,
        flags: i32,
        name_kva: u64,
        span_weight: u32,
        balance_interval: u32,
        newidle_call: u32,
        lb_count_0: u32,
        alb_pushed: u32,
        ttwu_wake_remote: u32,
    ) -> Vec<u8> {
        // Size must cover the highest offset used.
        let so = sd.stats_offsets.as_ref().unwrap();
        let size = so.sd_ttwu_move_balance + 4 + 8;
        let mut buf = vec![0u8; size];

        buf[sd.sd_parent..sd.sd_parent + 8].copy_from_slice(&parent_kva.to_ne_bytes());
        buf[sd.sd_level..sd.sd_level + 4].copy_from_slice(&level.to_ne_bytes());
        buf[sd.sd_flags..sd.sd_flags + 4].copy_from_slice(&flags.to_ne_bytes());
        buf[sd.sd_name..sd.sd_name + 8].copy_from_slice(&name_kva.to_ne_bytes());
        buf[sd.sd_span_weight..sd.sd_span_weight + 4].copy_from_slice(&span_weight.to_ne_bytes());
        buf[sd.sd_balance_interval..sd.sd_balance_interval + 4]
            .copy_from_slice(&balance_interval.to_ne_bytes());
        if let Some(off) = sd.sd_newidle_call {
            buf[off..off + 4].copy_from_slice(&newidle_call.to_ne_bytes());
        }
        buf[so.sd_lb_count..so.sd_lb_count + 4].copy_from_slice(&lb_count_0.to_ne_bytes());
        buf[so.sd_alb_pushed..so.sd_alb_pushed + 4].copy_from_slice(&alb_pushed.to_ne_bytes());
        buf[so.sd_ttwu_wake_remote..so.sd_ttwu_wake_remote + 4]
            .copy_from_slice(&ttwu_wake_remote.to_ne_bytes());
        buf
    }

    #[test]
    fn read_sched_domain_tree_null_sd() {
        // rq->sd is null — should return None.
        let sd_off = test_sched_domain_offsets();
        let buf = vec![0u8; 512];
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let result = read_sched_domain_tree(&mem, 0, &sd_off, 0, 0, 0);
        assert!(result.is_none());
    }

    #[test]
    fn read_sched_domain_tree_single_domain() {
        let sd_off = test_sched_domain_offsets();

        // Build: rq at PA 0 with rq->sd pointing to a domain.
        // Domain at some offset in the buffer, parent=0 (no parent).
        // page_offset=0 so KVA == PA for testing.
        let sd_pa: u64 = 1024;
        let name_pa: u64 = 2048;

        let sd_buf = make_sd_buffer(&sd_off, 0, 0, 0x42, name_pa, 4, 64, 15, 10, 3, 7);
        let name_bytes = b"SMT\0";

        // Build combined buffer: rq region + sd region + name region.
        let total_size = (name_pa as usize) + 16;
        let mut buf = vec![0u8; total_size];

        // Write rq->sd pointer (KVA == PA since page_offset=0).
        buf[sd_off.rq_sd..sd_off.rq_sd + 8].copy_from_slice(&sd_pa.to_ne_bytes());

        // Write sched_domain at sd_pa.
        buf[sd_pa as usize..sd_pa as usize + sd_buf.len()].copy_from_slice(&sd_buf);

        // Write name string.
        buf[name_pa as usize..name_pa as usize + name_bytes.len()].copy_from_slice(name_bytes);

        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let domains = read_sched_domain_tree(&mem, 0, &sd_off, 0, 0, 0).unwrap();

        assert_eq!(domains.len(), 1);
        assert_eq!(domains[0].level, 0);
        assert_eq!(domains[0].name, "SMT");
        assert_eq!(domains[0].flags, 0x42);
        assert_eq!(domains[0].span_weight, 4);
        assert_eq!(domains[0].balance_interval, 64);
        assert_eq!(domains[0].newidle_call, Some(15));
        let stats = domains[0].stats.as_ref().unwrap();
        assert_eq!(stats.lb_count[0], 10);
        assert_eq!(stats.alb_pushed, 3);
        assert_eq!(stats.ttwu_wake_remote, 7);
    }

    #[test]
    fn read_sched_domain_tree_two_levels() {
        let sd_off = test_sched_domain_offsets();

        // page_offset=0 so KVA == PA.
        let sd0_pa: u64 = 1024;
        let sd1_pa: u64 = 2048;
        let name0_pa: u64 = 3072;
        let name1_pa: u64 = 3088;

        // Domain 0 (SMT, level 0) -> parent = Domain 1
        let sd0_buf = make_sd_buffer(&sd_off, sd1_pa, 0, 0x10, name0_pa, 2, 32, 8, 5, 1, 2);
        // Domain 1 (MC, level 1) -> parent = 0 (top)
        let sd1_buf = make_sd_buffer(&sd_off, 0, 1, 0x20, name1_pa, 8, 128, 22, 20, 4, 10);

        let total_size = 3104;
        let mut buf = vec![0u8; total_size];

        // rq->sd -> domain 0
        buf[sd_off.rq_sd..sd_off.rq_sd + 8].copy_from_slice(&sd0_pa.to_ne_bytes());
        buf[sd0_pa as usize..sd0_pa as usize + sd0_buf.len()].copy_from_slice(&sd0_buf);
        buf[sd1_pa as usize..sd1_pa as usize + sd1_buf.len()].copy_from_slice(&sd1_buf);
        buf[name0_pa as usize..name0_pa as usize + 4].copy_from_slice(b"SMT\0");
        buf[name1_pa as usize..name1_pa as usize + 3].copy_from_slice(b"MC\0");

        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let domains = read_sched_domain_tree(&mem, 0, &sd_off, 0, 0, 0).unwrap();

        assert_eq!(domains.len(), 2);
        // First = lowest level (SMT).
        assert_eq!(domains[0].level, 0);
        assert_eq!(domains[0].name, "SMT");
        assert_eq!(domains[0].span_weight, 2);
        assert_eq!(domains[0].balance_interval, 32);
        assert_eq!(domains[0].newidle_call, Some(8));
        let s0 = domains[0].stats.as_ref().unwrap();
        assert_eq!(s0.lb_count[0], 5);
        // Second = higher level (MC).
        assert_eq!(domains[1].level, 1);
        assert_eq!(domains[1].name, "MC");
        assert_eq!(domains[1].span_weight, 8);
        assert_eq!(domains[1].balance_interval, 128);
        assert_eq!(domains[1].newidle_call, Some(22));
        let s1 = domains[1].stats.as_ref().unwrap();
        assert_eq!(s1.lb_count[0], 20);
        assert_eq!(s1.alb_pushed, 4);
        assert_eq!(s1.ttwu_wake_remote, 10);
    }

    #[test]
    fn read_sched_domain_tree_self_reference_breaks_cycle() {
        let sd_off = test_sched_domain_offsets();

        // Self-referential: sd->parent == sd. With the visited-set
        // cycle check, the walker emits sd exactly once and stops on
        // the next iteration rather than emitting the same domain
        // MAX_DEPTH times.
        let sd_pa: u64 = 1024;
        let sd_buf = make_sd_buffer(&sd_off, sd_pa, 0, 0, 0, 1, 0, 0, 0, 0, 0);

        let total_size = sd_pa as usize + sd_buf.len();
        let mut buf = vec![0u8; total_size];
        buf[sd_off.rq_sd..sd_off.rq_sd + 8].copy_from_slice(&sd_pa.to_ne_bytes());
        buf[sd_pa as usize..sd_pa as usize + sd_buf.len()].copy_from_slice(&sd_buf);

        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let domains = read_sched_domain_tree(&mem, 0, &sd_off, 0, 0, 0).unwrap();

        assert_eq!(
            domains.len(),
            1,
            "self-referential sd should produce exactly one snapshot"
        );
    }

    #[test]
    fn read_sched_domain_tree_max_depth_bound_on_long_chain() {
        let sd_off = test_sched_domain_offsets();
        // Per-struct size covers make_sd_buffer's layout
        // (sd_ttwu_move_balance=232 + 4 bytes for that u32 + 8 bytes
        // guard = 244) rounded up to the next multiple of 8. The
        // readers go through `GuestMem::read_u32` / `read_u64`, which
        // call `read_volatile_bytes<N>` — per-byte volatile reads that
        // are always 1-aligned and recompose the integer via
        // `from_ne_bytes`, so misaligned PAs are safe. The 8-alignment
        // baked into this stride is therefore no longer load-bearing
        // for correctness; it remains because every real kernel
        // sched_domain is `__randomize_layout`-aligned past u64 and
        // matching that stride keeps the fixture realistic.
        //
        // Byte-wise volatile has no atomicity guarantee: a concurrent
        // guest-side write mid-read can produce a torn integer. The
        // monitor treats its samples as best-effort, so tearing is
        // acceptable here. `from_ne_bytes` uses host endianness;
        // x86_64/aarch64 guests and hosts share little-endian, so
        // the recomposed value matches the guest's stored value.
        //
        // Each sched_domain lives at a distinct PA and points at the
        // next via sd->parent, forming an acyclic chain longer than
        // MAX_DEPTH so the depth bound — not the visited set — is
        // what stops the walk.
        const SD_SIZE: u64 = 248;
        const CHAIN_LEN: usize = 10;
        let first_pa: u64 = 1024;

        let pa = |i: usize| first_pa + (i as u64) * SD_SIZE;
        let total_size = pa(CHAIN_LEN) as usize;
        let mut buf = vec![0u8; total_size];

        // rq->sd -> first domain.
        buf[sd_off.rq_sd..sd_off.rq_sd + 8].copy_from_slice(&pa(0).to_ne_bytes());

        for i in 0..CHAIN_LEN {
            let parent_kva = if i + 1 == CHAIN_LEN { 0 } else { pa(i + 1) };
            let sd_buf = make_sd_buffer(&sd_off, parent_kva, i as i32, 0, 0, 1, 0, 0, 0, 0, 0);
            let start = pa(i) as usize;
            buf[start..start + sd_buf.len()].copy_from_slice(&sd_buf);
        }

        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let domains = read_sched_domain_tree(&mem, 0, &sd_off, 0, 0, 0).unwrap();

        assert_eq!(
            domains.len(),
            8,
            "acyclic chain of {CHAIN_LEN} levels must truncate at MAX_DEPTH=8"
        );
        // Sanity: the emitted levels are the first 8 in order.
        for (i, snap) in domains.iter().enumerate() {
            assert_eq!(snap.level, i as i32, "level mismatch at index {i}");
        }
    }

    #[test]
    fn read_sched_domain_tree_out_of_bounds_pa() {
        let sd_off = test_sched_domain_offsets();

        // rq->sd points to a KVA that translates to a PA beyond guest memory.
        let bad_kva: u64 = 0xFFFF_FFFF_FFFF_0000;
        let mut buf = vec![0u8; 512];
        buf[sd_off.rq_sd..sd_off.rq_sd + 8].copy_from_slice(&bad_kva.to_ne_bytes());

        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        // page_offset=0 -> PA = bad_kva which is > buf.len().
        let domains = read_sched_domain_tree(&mem, 0, &sd_off, 0, 0, 0);

        // Should return Some(empty vec) — non-null sd but untranslatable.
        assert!(domains.is_some());
        assert!(domains.unwrap().is_empty());
    }

    #[test]
    fn read_sched_domain_tree_newidle_none() {
        // 6.16 kernel: newidle_call/success/ratio are absent.
        // Other fields (level, name, span_weight, balance_interval) must
        // still populate correctly.
        let mut sd_off = test_sched_domain_offsets();
        sd_off.sd_newidle_call = None;
        sd_off.sd_newidle_success = None;
        sd_off.sd_newidle_ratio = None;

        let sd_pa: u64 = 1024;
        let name_pa: u64 = 2048;

        let sd_buf = make_sd_buffer(&sd_off, 0, 0, 0x42, name_pa, 4, 64, 0, 10, 3, 7);
        let name_bytes = b"SMT\0";

        let total_size = (name_pa as usize) + 16;
        let mut buf = vec![0u8; total_size];

        buf[sd_off.rq_sd..sd_off.rq_sd + 8].copy_from_slice(&sd_pa.to_ne_bytes());
        buf[sd_pa as usize..sd_pa as usize + sd_buf.len()].copy_from_slice(&sd_buf);
        buf[name_pa as usize..name_pa as usize + name_bytes.len()].copy_from_slice(name_bytes);

        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let domains = read_sched_domain_tree(&mem, 0, &sd_off, 0, 0, 0).unwrap();

        assert_eq!(domains.len(), 1);
        assert_eq!(domains[0].level, 0);
        assert_eq!(domains[0].name, "SMT");
        assert_eq!(domains[0].flags, 0x42);
        assert_eq!(domains[0].span_weight, 4);
        assert_eq!(domains[0].balance_interval, 64);
        assert_eq!(domains[0].newidle_call, None);
        assert_eq!(domains[0].newidle_success, None);
        assert_eq!(domains[0].newidle_ratio, None);
        let stats = domains[0].stats.as_ref().unwrap();
        assert_eq!(stats.lb_count[0], 10);
        assert_eq!(stats.alb_pushed, 3);
        assert_eq!(stats.ttwu_wake_remote, 7);
    }

    #[test]
    fn read_u32_array_known_values() {
        let mut buf = [0u8; 16];
        buf[0..4].copy_from_slice(&10u32.to_ne_bytes());
        buf[4..8].copy_from_slice(&20u32.to_ne_bytes());
        buf[8..12].copy_from_slice(&30u32.to_ne_bytes());
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let arr = read_u32_array(&mem, 0, 0);
        assert_eq!(arr, [10, 20, 30]);
    }

    /// End-to-end BTF-resolved offsets → reader readback.
    ///
    /// Parses `KernelOffsets` from the real test vmlinux (skipping when
    /// no cached test kernel is available), writes known values at the
    /// BTF-resolved field offsets in a synthetic byte buffer, and asserts
    /// `read_rq_stats` returns exactly those values. Catches drift
    /// between BTF parsing and reader field arithmetic.
    #[test]
    fn btf_offsets_couple_with_rq_reader() {
        let path = match crate::monitor::find_test_vmlinux() {
            Some(p) => p,
            None => skip!("no test vmlinux available"),
        };
        let offsets = crate::test_support::require_kernel_offsets(&path);

        let max_scalar_off = offsets.rq_clock + 8;
        let max_scx_off = offsets.rq_scx + offsets.scx_rq_local_dsq + offsets.dsq_nr + 4;
        let max_flags_off = offsets.rq_scx + offsets.scx_rq_flags + 4;
        let size = max_scalar_off.max(max_scx_off).max(max_flags_off) + 64;
        let mut buf = vec![0u8; size];

        let nr_running: u32 = 0xDEAD_BEEF;
        let scx_nr: u32 = 0x1234_5678;
        let dsq_depth: u32 = 0x0BAD_F00D;
        let clock: u64 = 0xCAFE_BABE_1357_9BDF;
        let flags: u32 = 0xA5A5_A5A5;

        buf[offsets.rq_nr_running..offsets.rq_nr_running + 4]
            .copy_from_slice(&nr_running.to_ne_bytes());
        buf[offsets.rq_clock..offsets.rq_clock + 8].copy_from_slice(&clock.to_ne_bytes());
        let scx_nr_off = offsets.rq_scx + offsets.scx_rq_nr_running;
        buf[scx_nr_off..scx_nr_off + 4].copy_from_slice(&scx_nr.to_ne_bytes());
        let scx_flags_off = offsets.rq_scx + offsets.scx_rq_flags;
        buf[scx_flags_off..scx_flags_off + 4].copy_from_slice(&flags.to_ne_bytes());
        let dsq_off = offsets.rq_scx + offsets.scx_rq_local_dsq + offsets.dsq_nr;
        buf[dsq_off..dsq_off + 4].copy_from_slice(&dsq_depth.to_ne_bytes());

        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let snap = read_rq_stats(&mem, 0, &offsets);
        assert_eq!(snap.nr_running, nr_running);
        assert_eq!(snap.scx_nr_running, scx_nr);
        assert_eq!(snap.local_dsq_depth, dsq_depth);
        assert_eq!(snap.rq_clock, clock);
        assert_eq!(snap.scx_flags, flags);
    }

    // ---- GuestMem write/resolve coverage --------------------------
    //
    // Pin the bounds-check semantics of `write_scalar` (the sole
    // gateway for `write_u8` / `write_u32` / `write_u64`), the
    // size-vs-offset interaction documented on `GuestMem::new`, and
    // the multi-region `resolve_ptr` routing that backs every read
    // and write on a NUMA layout.

    #[test]
    fn write_u32_at_boundary_writes_full_word() {
        let mut buf = [0u8; 16];
        // SAFETY: buf outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        // PA 12 + 4 = 16 == size: write_scalar's `>` bound is not
        // crossed, so the full word lands at the very end of the
        // mapping.
        mem.write_u32(12, 0, 0xDEAD_BEEF);
        assert_eq!(mem.read_u32(12, 0), 0xDEAD_BEEF);
        assert_eq!(
            u32::from_ne_bytes(buf[12..16].try_into().unwrap()),
            0xDEAD_BEEF
        );
    }

    #[test]
    fn write_u32_one_past_boundary_is_noop() {
        let mut buf = [0u8; 16];
        // SAFETY: buf outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        // PA 13 + 4 = 17 > 16: write must drop silently.
        mem.write_u32(13, 0, 0xFFFF_FFFF);
        assert_eq!(buf, [0u8; 16]);
    }

    #[test]
    fn write_u8_at_boundary_writes_last_byte() {
        let mut buf = [0u8; 4];
        // SAFETY: buf outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        // PA 3 + 1 = 4 == size.
        mem.write_u8(3, 0, 0xAB);
        assert_eq!(buf[3], 0xAB);
    }

    #[test]
    fn write_scalar_offset_arg_is_added_to_pa() {
        // `write_scalar` computes `addr = pa + offset`, so a write
        // through (pa, offset) must land at the same byte as a write
        // through (pa+offset, 0). Pin this so the offset path can
        // never silently drift from the pa path.
        let mut buf = [0u8; 32];
        // SAFETY: buf outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        mem.write_u64(8, 16, 0x0123_4567_89AB_CDEF);
        assert_eq!(mem.read_u64(24, 0), 0x0123_4567_89AB_CDEF);
        assert_eq!(
            u64::from_ne_bytes(buf[24..32].try_into().unwrap()),
            0x0123_4567_89AB_CDEF
        );
    }

    #[test]
    fn write_scalar_offset_only_out_of_bounds_is_noop() {
        // pa fits but pa+offset crosses the boundary. The bounds
        // check must combine pa and offset, not check them
        // independently.
        let mut buf = [0xCCu8; 8];
        // SAFETY: buf outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        // pa=4 (in-bounds), offset=4, addr=8, addr+8=16 > 8.
        mem.write_u64(4, 4, 0xFFFF_FFFF_FFFF_FFFF);
        assert_eq!(buf, [0xCCu8; 8]);
    }

    #[test]
    fn guest_mem_new_size_smaller_than_write_offset_is_noop() {
        // Construct a GuestMem reporting size N over a backing
        // buffer of >= N bytes, then attempt a write at offset > N.
        // The bounds check uses the *declared* size, so the write
        // must drop and the backing bytes past `size` must remain
        // untouched.
        let mut backing = [0u8; 32];
        let declared_size: u64 = 8;
        // SAFETY: backing outlives the GuestMem use; declared_size
        // is <= backing.len() so reads/writes stay within the
        // allocated buffer when they are accepted.
        let mem = unsafe { GuestMem::new(backing.as_mut_ptr(), declared_size) };
        // Write at byte 16 (well past declared size 8) — must noop.
        mem.write_u64(16, 0, 0xDEAD_BEEF_CAFE_1234);
        assert_eq!(backing, [0u8; 32]);
        // The same write inside declared bounds must succeed.
        mem.write_u64(0, 0, 0xDEAD_BEEF_CAFE_1234);
        assert_eq!(
            u64::from_ne_bytes(backing[0..8].try_into().unwrap()),
            0xDEAD_BEEF_CAFE_1234
        );
        // Bytes past the declared size remain untouched.
        assert_eq!(backing[8..32], [0u8; 24]);
    }

    #[test]
    fn resolve_ptr_multi_region_routes_to_correct_region() {
        // Two distinct host buffers (separate allocations) wired
        // into a single GuestMem with a gap between their DRAM
        // offsets. Reads and writes must route to the right host
        // buffer based on offset; otherwise multi-region NUMA
        // layouts would silently corrupt or read stale data.
        let mut buf0 = [0xAAu8; 64];
        let mut buf1 = [0xBBu8; 64];
        let regions = vec![
            MemRegion {
                host_ptr: buf0.as_mut_ptr(),
                offset: 0,
                size: 64,
            },
            MemRegion {
                host_ptr: buf1.as_mut_ptr(),
                offset: 1024, // gap from 64..1024
                size: 64,
            },
        ];
        // SAFETY: buf0 and buf1 outlive the GuestMem use; each
        // region's host_ptr addresses a 64-byte mapping.
        let mem = unsafe { GuestMem::from_regions_for_test(regions) };

        // Reads within region 0 see buf0's contents.
        assert_eq!(mem.read_u8(0, 0), 0xAA);
        assert_eq!(mem.read_u8(63, 0), 0xAA);
        // Reads within region 1 see buf1's contents.
        assert_eq!(mem.read_u8(1024, 0), 0xBB);
        assert_eq!(mem.read_u8(1087, 0), 0xBB);
        // Reads in the gap return 0 (resolve_ptr -> None ->
        // read_scalar returns zeroed bytes).
        assert_eq!(mem.read_u8(64, 0), 0);
        assert_eq!(mem.read_u8(512, 0), 0);
        assert_eq!(mem.read_u8(1023, 0), 0);

        // A write in region 1 hits buf1 and leaves buf0 alone.
        mem.write_u32(1024, 0, 0x1234_5678);
        assert_eq!(buf0, [0xAAu8; 64]);
        assert_eq!(
            u32::from_ne_bytes(buf1[0..4].try_into().unwrap()),
            0x1234_5678
        );

        // A write in the gap is a no-op (resolve_ptr -> None).
        // Re-zero buf1[60..64] then attempt a write into the gap and
        // verify both buffers remain untouched at their unaffected
        // byte ranges.
        let buf0_snapshot = buf0;
        let buf1_snapshot = buf1;
        mem.write_u32(900, 0, 0xFFFF_FFFF);
        assert_eq!(buf0, buf0_snapshot);
        assert_eq!(buf1, buf1_snapshot);
    }

    #[test]
    fn resolve_ptr_multi_region_read_ring_volatile_routes_correctly() {
        // Multi-region GuestMem reads route byte-by-byte via
        // `mem.read_u8`. With a region whose data sits inside
        // region 1 (past the end of region 0), each `read_u8` must
        // resolve to region 1's host pointer — not stale bytes past
        // region 0's end.
        let mut buf0 = [0u8; 64];
        let mut buf1 = [0u8; 64];
        // Plant a known pattern at the start of region 1.
        buf1[0..8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let regions = vec![
            MemRegion {
                host_ptr: buf0.as_mut_ptr(),
                offset: 0,
                size: 64,
            },
            MemRegion {
                host_ptr: buf1.as_mut_ptr(),
                offset: 1024,
                size: 64,
            },
        ];
        // SAFETY: backing buffers outlive the GuestMem use.
        let mem = unsafe { GuestMem::from_regions_for_test(regions) };

        // Byte-by-byte reads from region 1's first 8 bytes must
        // return the planted pattern, not bytes from region 0.
        for (i, expected) in [1, 2, 3, 4, 5, 6, 7, 8].iter().enumerate() {
            assert_eq!(mem.read_u8(1024 + i as u64, 0), *expected);
        }
    }

    #[test]
    fn resolve_ptr_offset_at_exact_region_end_is_out_of_region() {
        // resolve_ptr's `local < r.size` check is strict: an offset
        // equal to a region's end must fall outside that region.
        // For a single-region GuestMem this means offset == size
        // resolves to None.
        let mut buf = [0xCCu8; 16];
        // SAFETY: buf outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        // read_scalar's bounds check (`addr + N > size`) catches
        // this for >= 1-byte reads, returning zero. The exact-end
        // u8 read at offset 16 has addr=16, addr+1=17 > 16, so the
        // outer check returns 0 before resolve_ptr is reached.
        assert_eq!(mem.read_u8(16, 0), 0);
    }

    /// Unified bounds-check pin for `write_u8` and `read_u8`: a
    /// single GuestMem of declared size N is exercised at three
    /// load-bearing positions:
    /// 1. last valid offset (N-1) — write must land and a read
    ///    back must observe the written byte;
    /// 2. one past the end (N) — write must be a silent no-op
    ///    (no panic, no out-of-bounds write to the backing buffer);
    /// 3. one past the end (N) — read must return 0 (the
    ///    `read_scalar` `addr + N as u64 > self.size` arm fires
    ///    before `resolve_ptr` is consulted).
    ///
    /// `write_scalar`'s bound is `addr + N as u64 > self.size`. For
    /// the 1-byte path (N=1) the boundary is `addr == size - 1`
    /// inclusive, `addr == size` exclusive. Pinning all three
    /// positions on one fixture catches a regression that flips
    /// the `>` to `>=` (which would reject the last valid byte) or
    /// drops the bound entirely (which would scribble past the
    /// declared mapping).
    #[test]
    fn write_u8_and_read_u8_bounds_at_declared_size() {
        const SIZE: u64 = 8;
        let mut buf = [0u8; SIZE as usize];
        // SAFETY: buf outlives the GuestMem use; declared_size
        // matches the backing allocation so even an accepted write
        // at the last valid offset stays inside `buf`.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), SIZE) };

        // (1) Last valid offset (SIZE - 1). pa=SIZE-1, offset=0,
        //     addr=SIZE-1, addr+1 = SIZE which is NOT > SIZE, so
        //     the write is accepted.
        mem.write_u8(SIZE - 1, 0, 0xAB);
        assert_eq!(
            mem.read_u8(SIZE - 1, 0),
            0xAB,
            "write at last valid offset must round-trip via read_u8"
        );
        assert_eq!(
            buf[(SIZE - 1) as usize],
            0xAB,
            "write at last valid offset must land in the backing byte"
        );

        // (2) Past the end (SIZE). pa=SIZE, offset=0, addr=SIZE,
        //     addr+1 = SIZE+1 > SIZE → bound trips, write is a
        //     silent no-op. Snapshot the buffer to confirm no other
        //     bytes moved either.
        let snapshot = buf;
        mem.write_u8(SIZE, 0, 0xFF);
        assert_eq!(
            buf, snapshot,
            "write past the end must be a silent no-op — no byte of \
             the backing buffer may change"
        );

        // (3) Read past the end. addr=SIZE, addr+1 > SIZE → returns 0.
        assert_eq!(
            mem.read_u8(SIZE, 0),
            0,
            "read past the end must return 0, not stale memory"
        );
        // One byte further is also out of bounds.
        assert_eq!(
            mem.read_u8(SIZE + 1, 0),
            0,
            "read several bytes past the end must also return 0"
        );
    }

    /// Direct exercise of [`GuestMem::walk_4level`] for a simple
    /// 4 KiB-page KVA. The walk descends PGD -> PUD -> PMD -> PTE
    /// with no PS bit set, returning a PA whose page-aligned base is
    /// the leaf PTE's payload and whose low 12 bits come from the
    /// KVA's in-page offset. Pinned at this layer (rather than
    /// going through `translate_kva`) so a regression in the leaf
    /// walker is observable even if the TLB / dispatch wrapper
    /// hides it.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn walk_4level_simple_4kib_page() {
        let kva: u64 = 0xFFFF_8880_0000_5678;
        let pgd_idx = (kva >> 39) & 0x1FF;
        let pud_idx = (kva >> 30) & 0x1FF;
        let pmd_idx = (kva >> 21) & 0x1FF;
        let pte_idx = (kva >> 12) & 0x1FF;
        let page_off = kva & 0xFFF;

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

        // 0x63 = PRESENT|RW|USER|ACCESSED|DIRTY — bit 0 (PRESENT) is
        // the only bit walk_4level inspects on intermediate entries;
        // the rest match the kernel's typical page-table flags so a
        // future tightening of the walker's flag checks still passes.
        write_entry(&mut buf, pgd_pa, pgd_idx, pud_pa | 0x63);
        write_entry(&mut buf, pud_pa, pud_idx, pmd_pa | 0x63);
        write_entry(&mut buf, pmd_pa, pmd_idx, pte_pa | 0x63);
        write_entry(&mut buf, pte_pa, pte_idx, data_pa | 0x63);

        // SAFETY: buf outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let pa = mem.walk_4level(pgd_pa, Kva(kva));
        assert_eq!(
            pa,
            Some(data_pa | page_off),
            "4-level walk must compose leaf PTE base with KVA in-page offset"
        );
    }

    /// `walk_4level` must honour a PMD entry with the PS bit (bit 7)
    /// set as a 2 MiB huge-page leaf rather than descending into a
    /// PTE table. Verifies the address composition: bits [51:21] from
    /// the PMD entry, bits [20:0] from the KVA. Catches a regression
    /// that drops the PS check or applies the wrong base mask.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn walk_4level_pmd_ps_bit_2mib_huge_page() {
        // Pick a KVA that's NOT 2 MiB-aligned so the in-page offset
        // (bits 20:0) is non-zero; the leaf composition is then
        // observable in the returned PA.
        let kva: u64 = 0xFFFF_8880_0020_3456;
        let pgd_idx = (kva >> 39) & 0x1FF;
        let pud_idx = (kva >> 30) & 0x1FF;
        let pmd_idx = (kva >> 21) & 0x1FF;
        let in_page_off_2m = kva & 0x1F_FFFF;

        let pgd_pa: u64 = 0x10000;
        let pud_pa: u64 = pgd_pa + 0x1000;
        let pmd_pa: u64 = pud_pa + 0x1000;
        // 2 MiB-aligned huge page base.
        let huge_base: u64 = 0x20_0000;

        let size = (pmd_pa + 0x1000) as usize;
        let mut buf = vec![0u8; size];

        let write_entry = |buf: &mut Vec<u8>, base: u64, idx: u64, val: u64| {
            let off = (base + idx * 8) as usize;
            buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
        };

        write_entry(&mut buf, pgd_pa, pgd_idx, pud_pa | 0x63);
        write_entry(&mut buf, pud_pa, pud_idx, pmd_pa | 0x63);
        // 0xE3 = PRESENT|RW|USER|ACCESSED|DIRTY|PS. Bit 7 (0x80)
        // flags the entry as a 2 MiB block leaf; walk_4level returns
        // before reading any PTE table.
        write_entry(&mut buf, pmd_pa, pmd_idx, huge_base | 0xE3);

        // SAFETY: buf outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let pa = mem.walk_4level(pgd_pa, Kva(kva));
        assert_eq!(
            pa,
            Some(huge_base | in_page_off_2m),
            "PMD PS-bit leaf must splice 2 MiB base with KVA bits [20:0]"
        );
    }

    /// `walk_4level` must return `None` when an intermediate level's
    /// entry has the PRESENT bit clear. Pinning at the PUD layer
    /// (not the PGD root) verifies the descent unwinds cleanly mid-
    /// walk, not just at the very first read.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn walk_4level_unmapped_pud_returns_none() {
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

        // PGD entry present, points at PUD.
        write_entry(&mut buf, pgd_pa, pgd_idx, pud_pa | 0x63);
        // PUD entry with PRESENT (bit 0) clear — non-zero payload so
        // the test fails loudly if the walker mistakes "non-zero
        // entry" for "present".
        write_entry(&mut buf, pud_pa, pud_idx, 0x3000);

        // SAFETY: buf outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        assert_eq!(
            mem.walk_4level(pgd_pa, Kva(kva)),
            None,
            "missing PRESENT bit at any level must return None"
        );
    }

    /// `walk_5level` must descend PML5 -> P4D and finish through the
    /// inner 4-level walk. Verifies bits [56:48] of the KVA index
    /// the PML5 entry while bits [47:39] continue to index the next
    /// level (which `walk_4level` labels "PGD" and treats as the
    /// new root). Catches a regression that drops the PML5 step or
    /// mis-decodes which KVA bits select the PML5 entry.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn walk_5level_descent_to_4kib_page() {
        // KVA with non-zero bits in BOTH the PML5 (56:48) and the
        // 4-level region (47:0) so an off-by-one in either decode
        // shows up.
        let kva: u64 = 0xFF42_8881_0000_5678;
        let pml5_idx = (kva >> 48) & 0x1FF;
        let pgd_idx = (kva >> 39) & 0x1FF;
        let pud_idx = (kva >> 30) & 0x1FF;
        let pmd_idx = (kva >> 21) & 0x1FF;
        let pte_idx = (kva >> 12) & 0x1FF;
        let page_off = kva & 0xFFF;

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

        write_entry(&mut buf, pml5_pa, pml5_idx, p4d_pa | 0x63);
        // P4D is read by walk_4level as the new "PGD", indexed by
        // KVA bits 47:39 (pgd_idx).
        write_entry(&mut buf, p4d_pa, pgd_idx, pud_pa | 0x63);
        write_entry(&mut buf, pud_pa, pud_idx, pmd_pa | 0x63);
        write_entry(&mut buf, pmd_pa, pmd_idx, pte_pa | 0x63);
        write_entry(&mut buf, pte_pa, pte_idx, data_pa | 0x63);

        // SAFETY: buf outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let pa = mem.walk_5level(pml5_pa, Kva(kva));
        assert_eq!(
            pa,
            Some(data_pa | page_off),
            "5-level walk must descend PML5->P4D->PUD->PMD->PTE and \
             splice the leaf PTE base with the KVA in-page offset"
        );
    }

    /// `walk_4level` must honour a PUD entry with the PS bit (bit 7)
    /// set as a 1 GiB huge-page leaf rather than descending into a
    /// PMD table. Verifies the address composition: bits [51:30] from
    /// the PUD entry, bits [29:0] from the KVA. Catches a regression
    /// that drops the PS check at the PUD level or applies the wrong
    /// 1 GiB base mask. Pinned in addition to the 2 MiB PMD test
    /// because the two leaf branches use different masks
    /// (0x000F_FFFF_C000_0000 vs 0x000F_FFFF_FFE0_0000) and a
    /// regression in either is undetectable from the other.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn walk_4level_pud_ps_bit_1gib_huge_page() {
        // KVA that's NOT 1 GiB-aligned so the in-page offset
        // (bits 29:0) is non-zero; the leaf composition is then
        // observable in the returned PA.
        let kva: u64 = 0xFFFF_8880_4123_4567;
        let pgd_idx = (kva >> 39) & 0x1FF;
        let pud_idx = (kva >> 30) & 0x1FF;
        let in_page_off_1g = kva & 0x3FFF_FFFF;

        let pgd_pa: u64 = 0x10000;
        let pud_pa: u64 = pgd_pa + 0x1000;
        // 1 GiB-aligned huge page base.
        let huge_base: u64 = 0x4000_0000;

        let size = (pud_pa + 0x1000) as usize;
        let mut buf = vec![0u8; size];

        let write_entry = |buf: &mut Vec<u8>, base: u64, idx: u64, val: u64| {
            let off = (base + idx * 8) as usize;
            buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
        };

        write_entry(&mut buf, pgd_pa, pgd_idx, pud_pa | 0x63);
        // 0xE3 = PRESENT|RW|USER|ACCESSED|DIRTY|PS. Bit 7 (0x80)
        // flags the entry as a 1 GiB block leaf; walk_4level returns
        // before reading any PMD/PTE table.
        write_entry(&mut buf, pud_pa, pud_idx, huge_base | 0xE3);

        // SAFETY: buf outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let pa = mem.walk_4level(pgd_pa, Kva(kva));
        assert_eq!(
            pa,
            Some(huge_base | in_page_off_1g),
            "PUD PS-bit leaf must splice 1 GiB base with KVA bits [29:0]"
        );
    }

    /// `walk_5level` must honour a PML5 entry with the PS bit (bit 7)
    /// set as a 256 TiB huge-page leaf rather than descending into a
    /// P4D table. Verifies the address composition: bits [51:48] from
    /// the PML5 entry, bits [47:0] from the KVA. The existing source
    /// comment notes Intel SDM Vol 3A currently reserves PS=0 at this
    /// level, but the walker pins the bit as load-bearing for any
    /// future / non-Intel implementation that enables it — drop the
    /// check and a `walk_5level` returning `walk_4level(p4d_pa, kva)`
    /// on a PS-bit-set entry would mis-decode the leaf and hand
    /// `walk_4level` a P4D PA that doesn't exist.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn walk_5level_pml5_ps_bit_256tib_huge_page() {
        // KVA whose low 48 bits are non-zero so the in-page offset is
        // observable in the composed leaf PA. Top bits (63:57) carry
        // the canonical kernel-half sign extension; bits 56:48 select
        // the PML5 entry; bits 47:0 splice in as the in-leaf offset.
        let kva: u64 = 0xFF42_0001_2345_6789;
        let pml5_idx = (kva >> 48) & 0x1FF;
        // 256 TiB in-page offset: low 48 bits.
        let in_page_off_256t = kva & 0x0000_FFFF_FFFF_FFFF;

        let pml5_pa: u64 = 0x10000;
        // Base mask clears the low 48 bits per the source comment;
        // pick a value whose bits 51:48 are non-zero so the splice
        // is testable.
        let huge_base: u64 = 0x000A_0000_0000_0000;

        let size = (pml5_pa + 0x1000) as usize;
        let mut buf = vec![0u8; size];

        let write_entry = |buf: &mut Vec<u8>, base: u64, idx: u64, val: u64| {
            let off = (base + idx * 8) as usize;
            buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
        };

        // 0xE3 = PRESENT|RW|USER|ACCESSED|DIRTY|PS. Bit 7 (0x80)
        // flags the entry as a 256 TiB block leaf; walk_5level
        // returns before recursing into walk_4level.
        write_entry(&mut buf, pml5_pa, pml5_idx, huge_base | 0xE3);

        // SAFETY: buf outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let pa = mem.walk_5level(pml5_pa, Kva(kva));
        assert_eq!(
            pa,
            Some(huge_base | in_page_off_256t),
            "PML5 PS-bit leaf must splice 256 TiB base with KVA bits [47:0]"
        );
    }

    /// `walk_5level` must return `None` when the PML5 entry has the
    /// PRESENT bit clear, without recursing into `walk_4level`.
    /// Pinning at the new top level (the layer 5-level paging adds
    /// over 4-level) verifies the early-exit covers the PML5 step
    /// itself; the `walk_4level_unmapped_pud_returns_none` test
    /// already covers absent intermediate levels below.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn walk_5level_unmapped_pml5_returns_none() {
        let kva: u64 = 0xFF42_8881_0000_5000;
        let pml5_idx = (kva >> 48) & 0x1FF;

        let pml5_pa: u64 = 0x10000;
        let size = (pml5_pa + 0x1000) as usize;
        let mut buf = vec![0u8; size];

        let write_entry = |buf: &mut Vec<u8>, base: u64, idx: u64, val: u64| {
            let off = (base + idx * 8) as usize;
            buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
        };

        // PML5 entry with PRESENT (bit 0) clear — non-zero payload
        // so the test fails loudly if the walker mistakes "non-zero
        // entry" for "present" at the new top level.
        write_entry(&mut buf, pml5_pa, pml5_idx, 0x3000);

        // SAFETY: buf outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        assert_eq!(
            mem.walk_5level(pml5_pa, Kva(kva)),
            None,
            "missing PRESENT bit at the PML5 root must return None \
             without recursing into walk_4level"
        );
    }

    /// `write_bytes_at` must return 0 — and not write anything — when
    /// `pa + offset` overflows `u64`. The `checked_add` guard is the
    /// only barrier between an attacker-controlled offset and the
    /// downstream `write_bytes` bounds-clip; if it's dropped, an
    /// overflowed addr would alias zero and silently overwrite the
    /// start of guest DRAM. Pinning the contract here keeps that
    /// regression visible without booting a VM.
    #[test]
    fn write_bytes_at_offset_overflow_returns_zero() {
        let mut buf = [0u8; 64];
        let snapshot = buf;
        // SAFETY: buf outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

        // pa + offset overflow: pa = u64::MAX, offset = 1.
        let n = mem.write_bytes_at(u64::MAX, 1, &[0xAA; 8]);
        assert_eq!(n, 0, "wraparound add must return 0 bytes written");
        assert_eq!(
            buf, snapshot,
            "overflow must not mutate any byte of the backing buffer"
        );

        // Same input that would alias offset 0 if the guard were
        // missing (pa = u64::MAX - 7, offset = 8 -> wraps to 0).
        let n = mem.write_bytes_at(u64::MAX - 7, 8, &[0xBB; 4]);
        assert_eq!(n, 0, "wraparound to 0 must return 0 bytes written");
        assert_eq!(buf, snapshot, "no aliasing into low DRAM is permitted");
    }
}
