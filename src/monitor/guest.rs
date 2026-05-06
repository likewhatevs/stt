//! Host-side kernel memory accessor for a running guest VM.
//!
//! Provides read/write access to kernel variables and structures in
//! guest physical memory. Resolves symbols from the vmlinux ELF,
//! handles address translation (text mapping, direct mapping, vmalloc),
//! and caches paging configuration.
//!
//! Scalar reads and writes use volatile semantics (the guest kernel
//! modifies memory concurrently). Bulk byte reads delegate to
//! `GuestMem::read_bytes` which uses `copy_nonoverlapping`;
//! `read_kva_bytes_chunked` adds page-boundary chunking on top so
//! large vmalloc'd reads (BTF blobs) translate once per page rather
//! than once per byte.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};

use super::Kva;
use super::reader::{Aarch64WalkParams, GuestMem};
use super::symbols::{
    kva_to_pa, resolve_page_offset_with_tcr, resolve_pgtable_l5, resolve_phys_base,
    start_kernel_map_for_tcr, text_kva_to_pa_with_base,
};

/// Host-side accessor for kernel memory in a running guest VM.
///
/// Resolves ELF symbols and paging configuration once at construction.
/// Subsequent reads use cached state.
///
/// Address translation modes:
/// - **Text/data/bss symbols**: `kva - __START_KERNEL_map`. Used for
///   statically-linked kernel variables.
/// - **Direct mapping**: `kva - PAGE_OFFSET`. Used for SLAB allocations,
///   per-CPU data, physically contiguous memory.
/// - **Vmalloc/vmap**: Page table walk via CR3. Used for BPF maps,
///   vmalloc'd memory, module text.
pub struct GuestKernel<'a> {
    mem: &'a GuestMem,
    symbols: HashMap<String, u64>,
    page_offset: u64,
    cr3_pa: u64,
    /// 5-level paging flag — true when the guest uses 5-level page tables (LA57).
    l5: bool,
    /// Cached TCR_EL1 register (aarch64). Drives the page-table walker's
    /// granule and VA-width decoding. Always 0 on x86_64 where the
    /// register does not exist; the walker ignores the field on that
    /// arch.
    ///
    /// **Immutability**: `TCR_EL1` is written once during early MMU
    /// bring-up (`__cpu_setup` in `arch/arm64/mm/proc.S`) and is
    /// never modified afterward. The host therefore caches both the
    /// raw register and the decoded [`Aarch64WalkParams`]
    /// at construction without any invalidation path; a future
    /// suspend/resume or kexec sequence that re-runs `__cpu_setup`
    /// would require rebuilding both fields.
    tcr_el1: u64,
    /// Decoded aarch64 page-table walk parameters derived once from
    /// [`Self::tcr_el1`]. Cached so each `read_kva_*` translate uses
    /// the cached path (`GuestMem::translate_kva_with_aarch64_params`)
    /// instead of re-decoding `T1SZ`/`TG1`/`va_width`/`levels_below`/
    /// `descaddrmask` per call.
    ///
    /// `None` on x86_64 (TCR_EL1 does not exist) and on aarch64
    /// when `tcr_el1` decodes to an invalid configuration (the
    /// failure modes [`Aarch64WalkParams::from_tcr_el1`] reports).
    /// Translates fall back to the per-call decode path
    /// ([`GuestMem::translate_kva`]) when cached params are absent.
    /// See [`Self::tcr_el1`] for the immutability contract.
    aarch64_params: Option<Aarch64WalkParams>,
    /// Kernel image base (`__START_KERNEL_map` on x86_64, `KIMAGE_VADDR`
    /// on aarch64). Resolved at construction time:
    /// - x86_64: the compile-time constant
    ///   [`crate::monitor::symbols::START_KERNEL_MAP`].
    /// - aarch64: derived from `tcr_el1` via
    ///   [`crate::monitor::symbols::start_kernel_map_for_tcr`],
    ///   reading both T1SZ (VA width) and TG1 (granule) so kernels
    ///   built with `VA_BITS=47` (16 KB granule, e.g. Apple Silicon)
    ///   and 52-bit-VA configurations both translate symbol KVAs to
    ///   the right PAs.
    start_kernel_map: u64,
    /// Kernel runtime `phys_base` (x86_64) used by every text/data
    /// symbol translation: `pa = (kva - start_kernel_map) + phys_base`
    /// (`arch/x86/mm/physaddr.c:15-32`). On a non-KASLR kernel this is
    /// `0` and the formula collapses to the historical
    /// `pa = kva - start_kernel_map`. On a KASLR kernel the value
    /// carries the post-randomization PA of the kernel image. Resolved
    /// at construction time via [`resolve_phys_base`] which walks the
    /// guest page tables (CR3 → PA chase, no `phys_base` involvement)
    /// to find the symbol's PA, breaking the chicken-and-egg with the
    /// text-symbol translation.
    ///
    /// On aarch64 the symbol is absent and the field stays `0`. The
    /// existing aarch64 derivation (`text_kva_to_pa_with_base` with
    /// `phys_base = 0` plus `start_kernel_map_for_tcr`-derived base)
    /// remains correct for non-KASLR aarch64 boots; KASLR-aware
    /// aarch64 support is a follow-up that resolves
    /// `kimage_voffset` from a symbol read via the same page-table
    /// walk.
    phys_base: u64,
}

/// Decode `tcr_el1` into [`Aarch64WalkParams`]; returns `None` on
/// x86_64 (the params struct is unused there) and on aarch64 when
/// the register decodes to an invalid configuration. Pulled out of
/// `GuestKernel::new` so it can be `cfg`-gated cleanly without
/// wrapping the call site in `#[cfg]`.
#[cfg(target_arch = "aarch64")]
fn decode_aarch64_params(tcr_el1: u64) -> Option<Aarch64WalkParams> {
    Aarch64WalkParams::from_tcr_el1(tcr_el1)
}

#[cfg(not(target_arch = "aarch64"))]
fn decode_aarch64_params(_tcr_el1: u64) -> Option<Aarch64WalkParams> {
    None
}

#[allow(dead_code)]
impl<'a> GuestKernel<'a> {
    /// Create from GuestMem and vmlinux path.
    ///
    /// Parses the ELF symbol table and resolves paging configuration
    /// from guest memory.
    ///
    /// `cr3_pa` is the BSP's CR3 register value (KVM_GET_SREGS on
    /// x86_64; TTBR1_EL1 on aarch64), masked to a PA. Used to walk
    /// the page tables for `phys_base` resolution: with KASLR
    /// enabled the kernel image's runtime PA cannot be derived from
    /// the `init_top_pgt` symbol (that derivation requires
    /// `phys_base` itself), so we use the live CR3 from the running
    /// vCPU instead. `0` is accepted as a bootstrap value — the
    /// resulting page-table walk will fail and `phys_base` falls
    /// back to `0` (the non-KASLR value).
    ///
    /// `tcr_el1` is the guest's TCR_EL1 register value, used by the
    /// aarch64 page-table walker to determine the granule (4 KB / 16 KB
    /// / 64 KB) and high-half VA width. Callers on aarch64 should read
    /// it once via `KVM_GET_ONE_REG` from any vCPU after the kernel
    /// finished its boot-time MMU configuration. Pass 0 on x86_64 where
    /// the register does not exist.
    ///
    /// On aarch64, fails with `Err` when `tcr_el1 == 0` or when the
    /// register's T1SZ / TG1 fields cannot be decoded (T1SZ=0 means
    /// the high half is disabled; TG1=0b00 is reserved). The kernel
    /// image base (`KIMAGE_VADDR`) depends on `VA_BITS_MIN`, which is
    /// only knowable from `TCR_EL1.T1SZ` plus `TCR_EL1.TG1`; without
    /// it the symbol-PA translation defaults to the 48-bit VA layout
    /// and reads the wrong bytes for 47-bit kernels (16 KB granule,
    /// e.g. Apple Silicon). Callers in retry contexts (the freeze
    /// coordinator's lazy-retry loops) must keep polling until
    /// `tcr_el1` has been populated by the BSP loop.
    pub fn new(mem: &'a GuestMem, vmlinux: &Path, tcr_el1: u64, cr3_pa: u64) -> Result<Self> {
        let data = std::fs::read(vmlinux)
            .with_context(|| format!("read vmlinux: {}", vmlinux.display()))?;
        let elf = goblin::elf::Elf::parse(&data).context("parse vmlinux ELF")?;

        // Filter on `st_shndx == SHN_UNDEF` (== 0 per ELF spec)
        // rather than `st_value == 0`. SHN_UNDEF marks linker
        // placeholders and imports — those have no defining section
        // and must be skipped. A `st_value == 0` filter would also
        // drop legitimate defined symbols whose section offset is 0
        // (the percpu case, fixed identically in
        // [`crate::vmm::freeze_coord::snapshot::VmlinuxSymbolCache::from_path`]
        // at the user-watchpoint site). Keeping a defined symbol
        // whose KVA happens to be 0 is safe — downstream resolvers
        // reject `kva == 0` so a 0-valued defined symbol surfaces a
        // diagnostic instead of being silently absent.
        const SHN_UNDEF: usize = 0;
        let mut symbols = HashMap::new();
        for sym in elf.syms.iter() {
            if sym.st_shndx == SHN_UNDEF {
                continue;
            }
            if let Some(name) = elf.strtab.get_at(sym.st_name)
                && !name.is_empty()
            {
                symbols.insert(name.to_string(), sym.st_value);
            }
        }

        // Resolve the kernel image base (`__START_KERNEL_map` on
        // x86_64, `KIMAGE_VADDR` on aarch64). On x86_64 this is the
        // compile-time constant; on aarch64 it depends on
        // `VA_BITS_MIN`, derived from `tcr_el1` so VA_BITS=47 kernels
        // (16 KB granule, e.g. Apple Silicon) translate symbols
        // correctly. The aarch64 derivation requires both T1SZ and
        // TG1; both come out of `start_kernel_map_for_tcr`.
        let start_kernel_map = start_kernel_map_for_tcr(tcr_el1).ok_or_else(|| {
            anyhow::anyhow!("could not derive kernel image base from tcr_el1=0x{tcr_el1:x}")
        })?;

        // Resolve paging state using the same logic as KernelSymbols.
        let kern_syms = super::symbols::KernelSymbols::from_vmlinux(vmlinux)?;
        // `__pgtable_l5_enabled` is set by `__startup_64` BEFORE
        // `phys_base` is randomized (the L5 mode is needed to build
        // the bootstrap page tables themselves), so reading it with
        // `phys_base = 0` is correct on x86_64 KASLR boots. The bit
        // is reflected in CR4.LA57 by hardware; using the symbol
        // here matches the historical path.
        let l5_bootstrap = resolve_pgtable_l5(mem, &kern_syms, start_kernel_map, 0);
        // CR3 from KVM SREGS carries PCID/PCD/PWT in bits [11:0]; mask
        // to a clean PA before walking. The walker also masks
        // descriptor entries internally but the initial CR3 it
        // receives is what we hand it, so do the mask here.
        let walk_cr3 = cr3_pa & !0xFFFu64;
        // Resolve `phys_base` by walking the page tables. The walker
        // returns raw PAs from PTE entries — no `phys_base` is
        // consumed during the walk itself, breaking the
        // chicken-and-egg that `text_kva_to_pa_with_base` would
        // otherwise create. A `None` result (symbol absent on
        // aarch64, walk fails on a still-booting guest) defaults to
        // `0`, which is the non-KASLR / aarch64 value and produces
        // the historical translation behaviour.
        let phys_base = resolve_phys_base(mem, &kern_syms, walk_cr3, l5_bootstrap, tcr_el1)
            .unwrap_or(0);

        // Re-resolve l5 with the live `phys_base` so a future
        // toolchain that sets the L5 flag after `phys_base`
        // randomization (currently no such kernel exists, but
        // documenting the assumption here) reads the right PA. With
        // `phys_base == 0` this re-read is identical to the
        // bootstrap.
        let l5 = if phys_base == 0 {
            l5_bootstrap
        } else {
            resolve_pgtable_l5(mem, &kern_syms, start_kernel_map, phys_base)
        };

        let page_offset =
            resolve_page_offset_with_tcr(mem, &kern_syms, start_kernel_map, tcr_el1, phys_base);

        // Cache the decoded aarch64 walk parameters once. On x86_64
        // the helper's `from_tcr_el1` returns None; the cache stays
        // unset and translates use the x86 walk path (which doesn't
        // consume params). On aarch64, decode failures mean the
        // walker would also reject the configuration mid-walk —
        // surfacing as `None` here keeps the cached path consistent
        // with the per-call path (both bail).
        let aarch64_params = decode_aarch64_params(tcr_el1);

        Ok(Self {
            mem,
            symbols,
            page_offset,
            cr3_pa: walk_cr3,
            l5,
            tcr_el1,
            aarch64_params,
            start_kernel_map,
            phys_base,
        })
    }

    /// Construct a `GuestKernel` for unit tests, bypassing the
    /// vmlinux ELF parse and paging-state resolution.
    ///
    /// Cross-module tests (e.g. `monitor::dump::tests`,
    /// `monitor::bpf_map::tests`) need to drive the production
    /// read paths over synthetic guest memory. Those tests cannot
    /// construct a `GuestKernel` via `::new` (no real vmlinux on
    /// hand) and the bare struct fields are private to this module
    /// so field-literal construction is unavailable. This
    /// constructor takes an explicit symbol map and pre-resolved
    /// paging state and stitches them into a `GuestKernel` whose
    /// public methods behave identically to one produced by
    /// `::new`.
    ///
    /// `symbols` typically carries entries the rest of the test
    /// stack will look up (e.g. `map_idr` for the BPF map walker;
    /// `init_top_pgt` is unused in synthetic-memory tests). Pass
    /// `cr3_pa = 0` and `page_offset = DEFAULT_PAGE_OFFSET` for
    /// direct-mapped synthetic buffers; callers that build a
    /// page-table buffer pass their `cr3_pa` instead.
    #[cfg(test)]
    pub(crate) fn new_for_test(
        mem: &'a GuestMem,
        symbols: HashMap<String, u64>,
        page_offset: u64,
        cr3_pa: u64,
        l5: bool,
    ) -> Self {
        Self {
            mem,
            symbols,
            page_offset,
            cr3_pa,
            l5,
            tcr_el1: 0,
            aarch64_params: None,
            // Tests construct synthetic memory layouts assuming the
            // compile-time constant. None of the test fixtures exercise
            // the VA_BITS=47 path; production aarch64 paths come
            // through `::new` where the value is derived from
            // `tcr_el1`.
            start_kernel_map: super::symbols::START_KERNEL_MAP,
            // Tests don't exercise KASLR; `phys_base = 0` reproduces
            // the historical translation: pa = kva - start_kernel_map.
            phys_base: 0,
        }
    }

    /// Look up a kernel symbol KVA by name.
    pub fn symbol_kva(&self, name: &str) -> Option<u64> {
        self.symbols.get(name).copied()
    }

    /// Guest physical memory reference.
    pub fn mem(&self) -> &GuestMem {
        self.mem
    }

    /// Runtime PAGE_OFFSET (resolved from guest memory).
    pub fn page_offset(&self) -> u64 {
        self.page_offset
    }

    /// Physical address of the top-level page table.
    pub fn cr3_pa(&self) -> u64 {
        self.cr3_pa
    }

    /// Whether the guest uses 5-level paging.
    pub fn l5(&self) -> bool {
        self.l5
    }

    /// Cached TCR_EL1 register (aarch64). Always 0 on x86_64. Use with
    /// [`super::reader::GuestMem::translate_kva`] to drive the
    /// granule-agnostic aarch64 page-table walker.
    pub fn tcr_el1(&self) -> u64 {
        self.tcr_el1
    }

    /// Cached aarch64 page-table walk parameters decoded from
    /// [`Self::tcr_el1`]. `None` on x86_64 and on aarch64 when the
    /// TCR decode fails (uninitialised register, reserved
    /// encoding). Hot-path consumers feed it into
    /// [`super::reader::GuestMem::translate_kva_with_aarch64_params`]
    /// to skip the per-call decode.
    pub fn aarch64_walk_params(&self) -> Option<&Aarch64WalkParams> {
        self.aarch64_params.as_ref()
    }

    /// Bundle of the four paging fields ([`super::reader::WalkContext`])
    /// threaded through every host-side KVA translation: `cr3_pa`,
    /// `page_offset`, `l5`, `tcr_el1`. Replaces the four-parameter fan
    /// at every call site that walks guest memory through this
    /// kernel handle.
    pub fn walk_context(&self) -> super::reader::WalkContext {
        super::reader::WalkContext {
            cr3_pa: self.cr3_pa,
            page_offset: self.page_offset,
            l5: self.l5,
            tcr_el1: self.tcr_el1,
        }
    }

    /// Resolved kernel image base (`__START_KERNEL_map` on x86_64,
    /// `KIMAGE_VADDR` on aarch64). Use [`Self::text_kva_to_pa`] for
    /// the actual translation; this accessor exists for callers that
    /// need to forward the base into helpers (e.g. cross-module
    /// readers that build their own translation).
    pub fn start_kernel_map(&self) -> u64 {
        self.start_kernel_map
    }

    /// Resolved kernel runtime `phys_base` (x86_64 KASLR offset).
    /// `0` on non-KASLR x86_64 boots and on aarch64. Forwarded into
    /// helpers that build their own
    /// [`super::symbols::text_kva_to_pa_with_base`] call (e.g.
    /// reader-side bootstraps that don't carry a `GuestKernel`
    /// handle directly).
    pub fn phys_base(&self) -> u64 {
        self.phys_base
    }

    /// Translate a kernel text/data/bss symbol VA to a DRAM-relative
    /// offset using the runtime kernel image base + KASLR `phys_base`
    /// resolved at construction time. Wraps
    /// [`super::symbols::text_kva_to_pa_with_base`] with the cached
    /// `start_kernel_map` and `phys_base` so callers don't have to
    /// re-derive them. On aarch64 with VA_BITS=47 (16 KB granule,
    /// e.g. Apple Silicon) the cached base is the right one; a
    /// constant-based helper would translate to the wrong offset on
    /// those hosts.
    pub fn text_kva_to_pa(&self, kva: u64) -> u64 {
        text_kva_to_pa_with_base(kva, self.start_kernel_map, self.phys_base)
    }

    // ---------------------------------------------------------------
    // Text/data/bss symbol reads (statically-linked kernel variables)
    // ---------------------------------------------------------------

    /// Read a u32 from a kernel text/data/bss symbol.
    ///
    /// Translates via the runtime kernel image base
    /// ([`Self::start_kernel_map`]), not PAGE_OFFSET.
    pub fn read_symbol_u32(&self, name: &str) -> Result<u32> {
        let kva = self.require_symbol(name)?;
        let pa = self.text_kva_to_pa(kva);
        Ok(self.mem.read_u32(pa, 0))
    }

    /// Read a u64 from a kernel text/data/bss symbol.
    pub fn read_symbol_u64(&self, name: &str) -> Result<u64> {
        let kva = self.require_symbol(name)?;
        let pa = self.text_kva_to_pa(kva);
        Ok(self.mem.read_u64(pa, 0))
    }

    /// Read bytes from a kernel text/data/bss symbol.
    pub fn read_symbol_bytes(&self, name: &str, len: usize) -> Result<Vec<u8>> {
        let kva = self.require_symbol(name)?;
        let pa = self.text_kva_to_pa(kva);
        let mut buf = vec![0u8; len];
        self.mem.read_bytes(pa, &mut buf);
        Ok(buf)
    }

    /// Write a u64 to a kernel text/data/bss symbol.
    pub fn write_symbol_u64(&self, name: &str, val: u64) -> Result<()> {
        let kva = self.require_symbol(name)?;
        let pa = self.text_kva_to_pa(kva);
        self.mem.write_u64(pa, 0, val);
        Ok(())
    }

    // ---------------------------------------------------------------
    // Direct mapping reads (SLAB, per-CPU, physmem)
    // ---------------------------------------------------------------

    /// Read a u64 from a direct-mapped kernel virtual address.
    ///
    /// Translates via `kva - PAGE_OFFSET`.
    pub fn read_direct_u64(&self, kva: u64) -> u64 {
        let pa = kva_to_pa(kva, self.page_offset);
        self.mem.read_u64(pa, 0)
    }

    /// Read a u32 from a direct-mapped kernel virtual address.
    pub fn read_direct_u32(&self, kva: u64) -> u32 {
        let pa = kva_to_pa(kva, self.page_offset);
        self.mem.read_u32(pa, 0)
    }

    /// Read bytes from a direct-mapped kernel virtual address.
    pub fn read_direct_bytes(&self, kva: u64, len: usize) -> Vec<u8> {
        let pa = kva_to_pa(kva, self.page_offset);
        let mut buf = vec![0u8; len];
        self.mem.read_bytes(pa, &mut buf);
        buf
    }

    // ---------------------------------------------------------------
    // Vmalloc/vmap reads (page table walk)
    // ---------------------------------------------------------------

    /// Translate a vmalloc'd KVA using the cached aarch64 walk
    /// parameters when available; falls back to the per-call
    /// `tcr_el1` decode path on aarch64 when the cache is absent
    /// (TCR not yet populated) or on x86_64 (where the cache is
    /// unused). Centralizes the "use the cache when possible"
    /// pattern so every `read_kva_*` helper benefits without
    /// duplicating the dispatch.
    fn translate_kva_cached(&self, kva: u64) -> Option<u64> {
        match self.aarch64_params.as_ref() {
            Some(params) => {
                self.mem
                    .translate_kva_with_aarch64_params(self.cr3_pa, Kva(kva), self.l5, params)
            }
            None => self
                .mem
                .translate_kva(self.cr3_pa, Kva(kva), self.l5, self.tcr_el1),
        }
    }

    /// Read a u32 from a vmalloc'd kernel virtual address.
    ///
    /// Translates via page table walk. Returns `None` if unmapped.
    pub fn read_kva_u32(&self, kva: u64) -> Option<u32> {
        let pa = self.translate_kva_cached(kva)?;
        Some(self.mem.read_u32(pa, 0))
    }

    /// Read a u64 from a vmalloc'd kernel virtual address.
    pub fn read_kva_u64(&self, kva: u64) -> Option<u64> {
        let pa = self.translate_kva_cached(kva)?;
        Some(self.mem.read_u64(pa, 0))
    }

    /// Read bytes from a vmalloc'd kernel virtual address range,
    /// chunking at page boundaries.
    ///
    /// Pays one [`super::reader::GuestMem::translate_kva`] call plus
    /// one bulk [`super::reader::GuestMem::read_bytes`] per 4 KiB
    /// page rather than per byte; required for reads above ~hundreds
    /// of bytes (e.g. a BPF program's BTF blob is typically tens of
    /// KB, vmlinux BTF up to several MB).
    ///
    /// Returns `None` when any page in the requested range fails to
    /// translate **or** when a chunk's bulk read returns fewer bytes
    /// than the chunk's expected length (DRAM end before the chunk
    /// completes); the all-or-nothing contract lets callers (e.g. the
    /// BTF-blob loader) treat any non-`None` return as a fully
    /// populated buffer.
    pub fn read_kva_bytes_chunked(&self, kva: u64, len: usize) -> Option<Vec<u8>> {
        let mut buf = vec![0u8; len];
        // 4 KiB chunking is conservative — never straddles a leaf
        // entry regardless of the kernel's page granule (4 KiB,
        // 16 KiB, or 64 KiB). A 16 KiB-granule kernel still maps
        // every 4 KiB sub-window of one PTE into the same
        // contiguous PA range, so chunking finer than the granule
        // pays an extra translate but never produces a torn read;
        // chunking COARSER than 4 KiB on a 4 KiB-granule kernel
        // would walk past the end of one PTE in a single chunk.
        const PAGE: u64 = 4096;
        let mut consumed: u64 = 0;
        let total = len as u64;
        while consumed < total {
            let cur_kva = kva.wrapping_add(consumed);
            let pa = self.translate_kva_cached(cur_kva)?;
            // Bytes remaining in the [`MemRegion`] that contains `pa`.
            // `GuestMem::read_bytes` clamps to this internally
            // (`copy_len = avail.min(region_avail)`), so any chunk
            // extending past it would silently short-return — and
            // the wrapper's `n != chunk_len_us` check would surface
            // that as `None`, masking a NUMA layout where the bytes
            // are physically present but split across two regions.
            // Cap `chunk_len` to `region_avail` so the chunk stays
            // within the resolved region; the next iteration's
            // `translate_kva_cached` resolves the post-boundary KVA
            // into the next region's PA and the loop continues.
            let region_avail = self.mem.region_avail(pa);
            if region_avail == 0 {
                // Translator returned a PA that resolves to no region.
                // Treat as unmapped — same all-or-nothing contract as
                // the outer `?` on a translate failure.
                return None;
            }
            // Advance to the next page boundary so the next translate
            // lands on a fresh resolved page.
            let page_end = (cur_kva & !(PAGE - 1)).wrapping_add(PAGE);
            let mut chunk_len = (page_end - cur_kva).min(total - consumed).min(region_avail);

            // Greedy contiguity merge: walk forward translating the
            // next page; if its PA equals `pa + chunk_len` (consecutive
            // VAs map to consecutive PAs — the common case for slab /
            // physically-contiguous allocations and any bulk read
            // covering one large physical span), extend the current
            // chunk's read instead of starting a new translate+read
            // pair on the next iteration. Each merge saves one
            // `read_bytes` call and one `copy_nonoverlapping` set-up;
            // a multi-megabyte BTF blob in the direct map collapses
            // from `len/PAGE` reads into a single `read_bytes`.
            //
            // The merge MUST also stop at the current region's
            // boundary: contiguous PAs can cross [`MemRegion`] borders
            // in multi-region NUMA layouts, where `read_bytes` would
            // silently clamp to `region_avail` and the wrapper would
            // turn the short return into `None`. `chunk_len` is
            // capped by `region_avail` so a merge step never extends
            // past the region containing the start `pa`; the next
            // outer iteration translates the post-boundary KVA into
            // the new region's PA and resumes there.
            //
            // Loop terminates when (a) we hit the requested total,
            // (b) the next page fails to translate (the outer loop
            // will surface the failure on the next iteration),
            // (c) the next page's PA is non-contiguous, or (d) the
            // current chunk has filled the start region.
            loop {
                let next_kva = cur_kva.wrapping_add(chunk_len);
                if chunk_len >= total - consumed {
                    break;
                }
                if chunk_len >= region_avail {
                    break;
                }
                // Probe the next page's translate. A None here is
                // not necessarily fatal — the outer loop's next
                // iteration will issue the same translate and bail
                // through `?` if the page is unmapped. We just stop
                // merging.
                let Some(next_pa) = self.translate_kva_cached(next_kva) else {
                    break;
                };
                if next_pa != pa.wrapping_add(chunk_len) {
                    break;
                }
                let next_page_end = (next_kva & !(PAGE - 1)).wrapping_add(PAGE);
                let next_chunk = (next_page_end - next_kva)
                    .min(total - consumed - chunk_len)
                    .min(region_avail - chunk_len);
                chunk_len = chunk_len.wrapping_add(next_chunk);
            }

            let chunk_len_us = chunk_len as usize;
            let dst = &mut buf[consumed as usize..consumed as usize + chunk_len_us];
            // `read_bytes` returns the actual count copied; a short
            // read means the page crosses end-of-DRAM. Honour the
            // doc's all-or-nothing contract: callers (e.g. the
            // BTF-blob loader) can't make sense of a partial buffer
            // — a short BTF blob would simply fail `Btf::from_bytes`
            // anyway, so collapse the partial-success case into None
            // up front.
            let n = self.mem.read_bytes(pa, dst);
            if n != chunk_len_us {
                return None;
            }
            consumed += chunk_len;
        }
        Some(buf)
    }

    // ---------------------------------------------------------------
    // Internal helpers
    // ---------------------------------------------------------------

    fn require_symbol(&self, name: &str) -> Result<u64> {
        self.symbols
            .get(name)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("symbol '{}' not found in vmlinux", name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::symbols::START_KERNEL_MAP;

    // Since GuestKernel::new() requires a real vmlinux, we test the
    // methods by constructing GuestKernel manually (bypassing ::new).
    // Page table walk tests are in bpf_map/tests.rs.

    #[test]
    fn text_kva_to_pa_and_read() {
        let start_kernel_map: u64 = START_KERNEL_MAP;
        let sym_kva = start_kernel_map + 0x1000;
        let pa = text_kva_to_pa_with_base(sym_kva, start_kernel_map, 0);
        assert_eq!(pa, 0x1000);

        let mut buf = vec![0u8; 0x2000];
        buf[0x1000..0x1004].copy_from_slice(&42u32.to_ne_bytes());
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        assert_eq!(mem.read_u32(pa, 0), 42);
    }

    #[test]
    fn direct_mapping_read() {
        use crate::monitor::symbols::DEFAULT_PAGE_OFFSET;
        // KVA = PAGE_OFFSET + dram_offset.
        // kva_to_pa returns dram_offset.
        let page_offset = DEFAULT_PAGE_OFFSET;
        let dram_offset = 0x2000u64;
        let kva = page_offset.wrapping_add(dram_offset);
        let pa = kva_to_pa(kva, page_offset);
        assert_eq!(pa, dram_offset);

        let mut buf = vec![0u8; 0x3000];
        buf[dram_offset as usize..dram_offset as usize + 8]
            .copy_from_slice(&0xDEAD_BEEF_1234_5678u64.to_ne_bytes());
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        assert_eq!(mem.read_u64(pa, 0), 0xDEAD_BEEF_1234_5678);
    }

    #[test]
    fn require_symbol_found() {
        // Build a GuestKernel manually (bypassing ::new) for unit testing.
        let buf = [0u8; 64];
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let mut symbols = HashMap::new();
        symbols.insert("test_sym".to_string(), 0xFFFF_FFFF_8000_1000u64);
        let kernel = GuestKernel {
            mem: &mem,
            symbols,
            page_offset: 0xFFFF_8880_0000_0000,
            cr3_pa: 0,
            l5: false,
            tcr_el1: 0,
            aarch64_params: None,
            start_kernel_map: START_KERNEL_MAP,
            phys_base: 0,
        };
        assert_eq!(kernel.symbol_kva("test_sym"), Some(0xFFFF_FFFF_8000_1000));
        assert_eq!(kernel.symbol_kva("missing"), None);
        assert!(kernel.require_symbol("test_sym").is_ok());
        assert!(kernel.require_symbol("missing").is_err());
    }

    #[test]
    fn read_symbol_u32_from_guest() {
        let start_kernel_map: u64 = START_KERNEL_MAP;
        let sym_kva = start_kernel_map + 0x100;
        // PA = 0x100
        let mut buf = vec![0u8; 0x200];
        buf[0x100..0x104].copy_from_slice(&99u32.to_ne_bytes());

        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let mut symbols = HashMap::new();
        symbols.insert("my_counter".to_string(), sym_kva);
        let kernel = GuestKernel {
            mem: &mem,
            symbols,
            page_offset: 0xFFFF_8880_0000_0000,
            cr3_pa: 0,
            l5: false,
            tcr_el1: 0,
            aarch64_params: None,
            start_kernel_map: START_KERNEL_MAP,
            phys_base: 0,
        };
        assert_eq!(kernel.read_symbol_u32("my_counter").unwrap(), 99);
    }

    #[test]
    fn read_symbol_u64_from_guest() {
        let start_kernel_map: u64 = START_KERNEL_MAP;
        let sym_kva = start_kernel_map + 0x100;
        let mut buf = vec![0u8; 0x200];
        buf[0x100..0x108].copy_from_slice(&0x1234_5678_ABCD_EF00u64.to_ne_bytes());

        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let mut symbols = HashMap::new();
        symbols.insert("my_u64".to_string(), sym_kva);
        let kernel = GuestKernel {
            mem: &mem,
            symbols,
            page_offset: 0xFFFF_8880_0000_0000,
            cr3_pa: 0,
            l5: false,
            tcr_el1: 0,
            aarch64_params: None,
            start_kernel_map: START_KERNEL_MAP,
            phys_base: 0,
        };
        assert_eq!(
            kernel.read_symbol_u64("my_u64").unwrap(),
            0x1234_5678_ABCD_EF00
        );
    }

    #[test]
    fn read_symbol_bytes_from_guest() {
        let start_kernel_map: u64 = START_KERNEL_MAP;
        let sym_kva = start_kernel_map + 0x100;
        let mut buf = vec![0u8; 0x200];
        buf[0x100..0x105].copy_from_slice(&[1, 2, 3, 4, 5]);

        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let mut symbols = HashMap::new();
        symbols.insert("my_bytes".to_string(), sym_kva);
        let kernel = GuestKernel {
            mem: &mem,
            symbols,
            page_offset: 0xFFFF_8880_0000_0000,
            cr3_pa: 0,
            l5: false,
            tcr_el1: 0,
            aarch64_params: None,
            start_kernel_map: START_KERNEL_MAP,
            phys_base: 0,
        };
        assert_eq!(
            kernel.read_symbol_bytes("my_bytes", 5).unwrap(),
            vec![1, 2, 3, 4, 5]
        );
    }

    #[test]
    fn read_symbol_missing_returns_error() {
        let buf = [0u8; 64];
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let kernel = GuestKernel {
            mem: &mem,
            symbols: HashMap::new(),
            page_offset: 0xFFFF_8880_0000_0000,
            cr3_pa: 0,
            l5: false,
            tcr_el1: 0,
            aarch64_params: None,
            start_kernel_map: START_KERNEL_MAP,
            phys_base: 0,
        };
        assert!(kernel.read_symbol_u32("nonexistent").is_err());
        assert!(kernel.read_symbol_u64("nonexistent").is_err());
        assert!(kernel.read_symbol_bytes("nonexistent", 4).is_err());
    }

    #[test]
    fn write_symbol_u64_to_guest() {
        let start_kernel_map: u64 = START_KERNEL_MAP;
        let sym_kva = start_kernel_map + 0x100;
        let mut buf = vec![0u8; 0x200];

        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let mut symbols = HashMap::new();
        symbols.insert("my_var".to_string(), sym_kva);
        let kernel = GuestKernel {
            mem: &mem,
            symbols,
            page_offset: 0xFFFF_8880_0000_0000,
            cr3_pa: 0,
            l5: false,
            tcr_el1: 0,
            aarch64_params: None,
            start_kernel_map: START_KERNEL_MAP,
            phys_base: 0,
        };
        kernel.write_symbol_u64("my_var", 0xCAFE_BABE).unwrap();
        assert_eq!(kernel.read_symbol_u64("my_var").unwrap(), 0xCAFE_BABE);
    }

    #[test]
    fn direct_mapping_methods() {
        use crate::monitor::symbols::DEFAULT_PAGE_OFFSET;
        let page_offset = DEFAULT_PAGE_OFFSET;
        let dram_offset = 0x200u64;
        // Direct mapping KVA = PAGE_OFFSET + dram_offset.
        let kva = page_offset.wrapping_add(dram_offset);
        let mut buf = vec![0u8; 0x300];
        buf[dram_offset as usize..dram_offset as usize + 4].copy_from_slice(&77u32.to_ne_bytes());
        buf[dram_offset as usize + 8..dram_offset as usize + 16]
            .copy_from_slice(&0xAAAA_BBBBu64.to_ne_bytes());

        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let kernel = GuestKernel {
            mem: &mem,
            symbols: HashMap::new(),
            page_offset,
            cr3_pa: 0,
            l5: false,
            tcr_el1: 0,
            aarch64_params: None,
            start_kernel_map: START_KERNEL_MAP,
            phys_base: 0,
        };
        assert_eq!(kernel.read_direct_u32(kva), 77);
        assert_eq!(kernel.read_direct_u64(kva + 8), 0xAAAA_BBBB);
        assert_eq!(&kernel.read_direct_bytes(kva, 4), &77u32.to_ne_bytes());
    }

    #[test]
    fn accessors_return_resolved_state() {
        let buf = [0u8; 64];
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let kernel = GuestKernel {
            mem: &mem,
            symbols: HashMap::new(),
            page_offset: 0x1234,
            cr3_pa: 0x5678,
            l5: true,
            tcr_el1: 0,
            aarch64_params: None,
            start_kernel_map: START_KERNEL_MAP,
            phys_base: 0,
        };
        assert_eq!(kernel.page_offset(), 0x1234);
        assert_eq!(kernel.cr3_pa(), 0x5678);
        assert!(kernel.l5());
        assert!(std::ptr::eq(kernel.mem(), &mem));
    }

    #[test]
    fn new_parses_vmlinux_symbols() {
        let path = match crate::monitor::find_test_vmlinux() {
            Some(p) => p,
            None => return,
        };
        // find_test_vmlinux may return /sys/kernel/btf/vmlinux (raw BTF,
        // not an ELF), which GuestKernel cannot parse.
        if path.starts_with("/sys/") {
            skip!("vmlinux is raw BTF (not ELF), cannot parse symbols");
        }
        // Allocate a buffer large enough for kernel-text-mapped reads.
        // GuestKernel::new reads page_offset_base and pgtable_l5_enabled
        // from guest memory; a zeroed buffer causes safe fallbacks.
        let mut buf = vec![0u8; 64 << 20];
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = match GuestKernel::new(&mem, &path, 0, 0) {
            Ok(k) => k,
            Err(e) => {
                // init_top_pgt missing in some kernel configs.
                skip!("GuestKernel::new failed: {e}");
            }
        };
        assert!(
            kernel.symbol_kva("runqueues").is_some(),
            "symbol map should contain runqueues"
        );
        assert_ne!(
            kernel.symbol_kva("runqueues").unwrap(),
            0,
            "runqueues address should be nonzero"
        );
    }

    /// `read_kva_bytes_chunked` must not extend its greedy
    /// PA-contiguity merge past a [`super::reader::MemRegion`]
    /// boundary. Multi-region NUMA layouts produce contiguous
    /// PA ranges that span two distinct host mappings; the
    /// underlying [`super::reader::GuestMem::read_bytes`] silently
    /// clamps each call to the resolved region's
    /// `region_avail`, so a merge that crossed the boundary
    /// would short-return and the wrapper's `n != chunk_len_us`
    /// check would surface that as `None`. The fix caps each
    /// chunk by the start PA's `region_avail`; the next outer
    /// iteration translates the post-boundary KVA into the
    /// next region's PA and resumes there.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn read_kva_bytes_chunked_crosses_numa_region_boundary() {
        // Two adjacent regions backed by separate host buffers,
        // with data pages A and B at contiguous PAs (0x4000 →
        // 0x5000) straddling the boundary. Without the
        // `region_avail` cap on the greedy merge, a single
        // `read_bytes` call would short-return at the boundary
        // and the wrapper would surface that as `None`.
        //
        // Region 0 [0x0000..0x5000): PML4@0, PDPT@0x1000,
        // PD@0x2000, PT@0x3000, data A@0x4000.
        // Region 1 [0x5000..0x9000): data B@0x5000.
        //
        // Every page-table base is 4-KiB-aligned. walk_4level
        // masks descriptors with `ADDR_MASK =
        // 0x000F_FFFF_FFFF_F000`; a sub-page-aligned base would
        // round down to the page boundary and route the next
        // table read to the wrong page.
        const REGION0_SIZE: usize = 0x5000;
        const REGION1_SIZE: usize = 0x4000;
        let mut buf0 = vec![0u8; REGION0_SIZE];
        let mut buf1 = vec![0u8; REGION1_SIZE];

        let kva: u64 = 0xFFFF_8880_0000_5000;
        let pml4_idx = (kva >> 39) & 0x1FF;
        let pdpt_idx = (kva >> 30) & 0x1FF;
        let pd_idx = (kva >> 21) & 0x1FF;
        let pt_idx = (kva >> 12) & 0x1FF;

        let pml4_pa: u64 = 0x0000;
        let pdpt_pa: u64 = 0x1000;
        let pd_pa: u64 = 0x2000;
        let pt_pa: u64 = 0x3000;
        let data_pa_a: u64 = 0x4000;
        let data_pa_b: u64 = 0x5000;

        let write_entry = |buf: &mut Vec<u8>, base: u64, idx: u64, val: u64| {
            let off = (base + idx * 8) as usize;
            buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
        };
        // 0x63 = present | rw | user | accessed | dirty.
        write_entry(&mut buf0, pml4_pa, pml4_idx, pdpt_pa | 0x63);
        write_entry(&mut buf0, pdpt_pa, pdpt_idx, pd_pa | 0x63);
        write_entry(&mut buf0, pd_pa, pd_idx, pt_pa | 0x63);
        write_entry(&mut buf0, pt_pa, pt_idx, data_pa_a | 0x63);
        write_entry(&mut buf0, pt_pa, pt_idx + 1, data_pa_b | 0x63);

        // Stamp distinct bytes per page so the assertion can prove
        // both regions were read in order.
        for b in &mut buf0[data_pa_a as usize..data_pa_a as usize + 0x1000] {
            *b = 0xAA;
        }
        for b in &mut buf1[..0x1000] {
            *b = 0xBB;
        }

        use crate::monitor::reader::MemRegion;
        let regions = vec![
            MemRegion {
                host_ptr: buf0.as_mut_ptr(),
                offset: 0,
                size: REGION0_SIZE as u64,
            },
            MemRegion {
                host_ptr: buf1.as_mut_ptr(),
                offset: REGION0_SIZE as u64,
                size: REGION1_SIZE as u64,
            },
        ];
        // SAFETY: buf0 and buf1 outlive the GuestMem use; each
        // region's host_ptr addresses its full mapping.
        let mem = unsafe { GuestMem::from_regions_for_test(regions) };

        let kernel = GuestKernel {
            mem: &mem,
            symbols: HashMap::new(),
            page_offset: 0xFFFF_8880_0000_0000,
            cr3_pa: pml4_pa,
            l5: false,
            tcr_el1: 0,
            aarch64_params: None,
            start_kernel_map: START_KERNEL_MAP,
            phys_base: 0,
        };

        // 0x2000 spans both pages. region_avail at PA 0x4000 is
        // 0x1000 (REGION0_SIZE - data_pa_a), so the first chunk
        // caps there and the next outer iteration translates into
        // region 1.
        let buf = kernel
            .read_kva_bytes_chunked(kva, 0x2000)
            .expect("multi-region read must succeed; greedy merge across boundary was the bug");
        assert_eq!(buf.len(), 0x2000);
        for &b in &buf[..0x1000] {
            assert_eq!(b, 0xAA, "first page bytes from region 0");
        }
        for &b in &buf[0x1000..] {
            assert_eq!(b, 0xBB, "second page bytes from region 1");
        }
    }

    /// `GuestKernel::new` must reject `tcr_el1 == 0` on aarch64
    /// because `start_kernel_map_for_tcr` cannot derive the kernel
    /// image base without a populated TCR_EL1 (T1SZ for VA_BITS,
    /// TG1 for granule). Silently falling back to the 48-bit
    /// constant would mis-read every symbol on a 47-bit kernel
    /// (16 KB granule, e.g. Apple Silicon).
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn guest_kernel_rejects_tcr_zero_aarch64() {
        let path = match crate::monitor::find_test_vmlinux() {
            Some(p) => p,
            None => return,
        };
        // find_test_vmlinux may return /sys/kernel/btf/vmlinux (raw BTF,
        // not an ELF), which GuestKernel cannot parse; the goblin
        // parse failure would mask the tcr_el1 check we want to
        // exercise. Skip in that case.
        if path.starts_with("/sys/") {
            skip!("vmlinux is raw BTF (not ELF), cannot parse symbols");
        }
        let mut buf = vec![0u8; 64 << 20];
        // SAFETY: buf outlives mem.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let result = GuestKernel::new(&mem, &path, 0, 0);
        let err = result.expect_err("tcr_el1=0 must be rejected on aarch64");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("tcr_el1"),
            "error message must mention tcr_el1; got: {msg}"
        );
    }
}
