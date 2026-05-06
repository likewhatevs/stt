//! Kernel symbol resolution and address translation.
//!
//! Parses a vmlinux ELF to extract symbol addresses (`runqueues`,
//! `__per_cpu_offset`, `page_offset_base`, etc.) and provides
//! functions for translating kernel virtual addresses to DRAM-relative
//! offsets (for GuestMem) via the text mapping and direct mapping.

use anyhow::{Context, Result};
use std::path::Path;

/// Kernel text mapping base (non-KASLR).
/// Used to convert kernel data/bss symbol VAs to guest-memory offsets
/// for the bootstrap read of `page_offset_base`.
///
/// x86-64: `__START_KERNEL_map` = 0xffff_ffff_8000_0000. Constant
/// across paging modes (4-level / 5-level both place the kernel
/// image at the same VA).
///
/// aarch64: VA-width-dependent. The default constant assumes the
/// 48-bit VA layout (`VA_BITS=48`, T1SZ=16): `KIMAGE_VADDR =
/// _PAGE_END(48) + SZ_2G = 0xffff_8000_8000_0000`. Kernels with
/// `VA_BITS=47` (16 KB granule, e.g. Apple Silicon) place the
/// image at `0xffff_c000_8000_0000`. Production aarch64 paths
/// must derive the runtime base from `tcr_el1` via
/// [`start_kernel_map_for_tcr`]; the constant is the test-only
/// fallback and the bootstrap value before TCR_EL1 is read.
#[cfg(target_arch = "x86_64")]
pub(crate) const START_KERNEL_MAP: u64 = 0xffff_ffff_8000_0000;
#[cfg(target_arch = "aarch64")]
pub(crate) const START_KERNEL_MAP: u64 = 0xffff_8000_8000_0000;

/// Default PAGE_OFFSET (non-KASLR), 48-bit VA layout.
///
/// x86-64 4-level paging: 0xffff_8880_0000_0000.
/// aarch64 48-bit VA: -(1 << 48) = 0xffff_0000_0000_0000.
///
/// **The aarch64 value is hardcoded for the 48-bit VA layout** — the
/// 47-bit layout (16 KiB granule, Apple Silicon style) places
/// `PAGE_OFFSET` at `-(1 << 47) = 0xffff_8000_0000_0000`. For
/// runtime correctness on those kernels, use
/// [`default_page_offset_for_tcr`], which decodes `TCR_EL1.T1SZ`
/// the same way [`start_kernel_map_for_tcr`] does. The const stays
/// for tests (synthetic guest memory layouts that pin a known
/// PAGE_OFFSET) and for the bootstrap window before `tcr_el1` is
/// populated; production paths funnel through
/// [`resolve_page_offset`] which prefers the live
/// `page_offset_base` symbol value over either fallback.
#[cfg(target_arch = "x86_64")]
pub(crate) const DEFAULT_PAGE_OFFSET: u64 = 0xffff_8880_0000_0000;
#[cfg(target_arch = "aarch64")]
pub(crate) const DEFAULT_PAGE_OFFSET: u64 = 0xffff_0000_0000_0000;

/// Derive the runtime `PAGE_OFFSET` from `TCR_EL1`.
///
/// On aarch64 the kernel sets `PAGE_OFFSET = -(1 << VA_BITS)` using
/// the *compile-time* `VA_BITS` (`arch/arm64/include/asm/memory.h`).
/// `TCR_EL1` only exposes `vabits_actual = 64 - T1SZ`, which equals
/// the compile-time `VA_BITS` in every configuration EXCEPT
/// `CONFIG_ARM64_VA_BITS_52=y` running on hardware without FEAT_LVA
/// (T1SZ=16, vabits_actual=48, VA_BITS=52). This function uses
/// `vabits_actual` as a proxy for the compile-time value and
/// therefore returns `0xffff_0000_0000_0000` (`-(1 << 48)`) on such
/// kernels, while the live `PAGE_OFFSET` is `0xfff0_0000_0000_0000`
/// (`-(1 << 52)`); the kernel additionally adjusts `memstart_addr`
/// (`arch/arm64/mm/init.c:arm64_memblock_init`) to compensate.
/// Direct-mapped translations using the value this function returns
/// will be wrong on a `VA_BITS=52` kernel with reduced runtime VA.
/// On x86_64 the register does not exist and the value is constant
/// ([`DEFAULT_PAGE_OFFSET`]); the `tcr_el1` argument is ignored.
///
/// Returns the [`DEFAULT_PAGE_OFFSET`] fallback when:
/// - `tcr_el1 == 0` (BSP loop has not yet read the register),
/// - `T1SZ == 0` (high half disabled),
/// - `T1SZ > 60` (would underflow `1 << va_bits`).
///
/// The fallback matches the 48-bit VA layout — the wrong value for
/// 47-bit and 52-bit kernels, but only reachable when the live
/// `page_offset_base` symbol is also absent. `page_offset_base` is
/// x86_64-only (`arch/x86/mm/kaslr.c`); aarch64 kernels lack the
/// symbol entirely, so this fallback IS the production path on
/// aarch64. ktstr.kconfig does not enable `CONFIG_ARM64_VA_BITS_52`,
/// so the 52-bit-mismatch case is dormant for current usage. See
/// [`resolve_page_offset`] for the preference chain.
pub(crate) fn default_page_offset_for_tcr(tcr_el1: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        let _ = tcr_el1;
        DEFAULT_PAGE_OFFSET
    }
    #[cfg(target_arch = "aarch64")]
    {
        if tcr_el1 == 0 {
            return DEFAULT_PAGE_OFFSET;
        }
        let t1sz = (tcr_el1 >> 16) & 0x3F;
        if t1sz == 0 || t1sz > 60 {
            return DEFAULT_PAGE_OFFSET;
        }
        let va_bits: u32 = 64u32 - t1sz as u32;
        // PAGE_OFFSET = -(1 << VA_BITS) using two's complement
        // wrap on the unsigned `0u64.wrapping_sub(...)`.
        0u64.wrapping_sub(1u64 << va_bits)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = tcr_el1;
        DEFAULT_PAGE_OFFSET
    }
}

/// ELF sections [`KernelSymbols::from_vmlinux`] reads directly.
///
/// The cached-vmlinux strip pipeline
/// ([`crate::cache::strip_vmlinux_debug`]) preserves these bytes
/// verbatim via its keep-list predicate. Removing an entry here
/// without also removing the reader will quietly break symbol
/// resolution on cache-booted runs.
pub(crate) const VMLINUX_KEEP_SECTIONS: &[&[u8]] = &[
    b".symtab", // symbol table — source of every kernel address this module resolves
    b".strtab", // names for .symtab entries
    b".bss",    // already SHT_NOBITS; holds scx_root (uninitialized global)
];

/// ELF data sections whose **addresses** (via `.symtab` `st_value`) are
/// consumed here as guest-memory offsets — the vmlinux file contributes
/// the symbol-to-address mapping only. The **runtime bytes** for these
/// sections come from the live guest, not from the vmlinux image, so
/// the strip pipeline can drop the file-backed contents safely.
///
/// Concretely, the strip pipeline rewrites these sections as
/// `SHT_NOBITS` with zero-length data so symbol-table entries whose
/// `st_shndx` points here survive `Builder::delete_orphans`. How the
/// stored `st_value` becomes a guest PA differs by section:
///
/// - `.data`: `st_value` is a kernel VA. `kva_to_pa` (direct mapping)
///   or `text_kva_to_pa_with_base` (text mapping) translates it to a PA
///   that [`super::reader::GuestMem`] reads directly.
/// - `.data..percpu`: `sh_addr = 0` on this section, so `st_value` is
///   a section-relative offset, NOT a KVA. The per-CPU KVA for CPU
///   `n` is `st_value + __per_cpu_offset[n]`; [`compute_rq_pas`]
///   performs that add and then calls `kva_to_pa` for each CPU.
///   Calling `kva_to_pa` on the raw `st_value` would translate the
///   wrong address.
pub(crate) const VMLINUX_ZERO_DATA_SECTIONS: &[&[u8]] = &[
    b".data",         // init_top_pgt, map_idr, prog_idr, scx_watchdog_timeout
    b".data..percpu", // runqueues (per-CPU runqueue template)
];

/// Kernel symbol addresses extracted from vmlinux ELF.
#[derive(Debug, Clone)]
pub(crate) struct KernelSymbols {
    /// `.data..percpu` section-relative offset of the `runqueues`
    /// per-CPU variable. Per-CPU symbols carry section offsets (not
    /// kernel virtual addresses) in the vmlinux symtab because the
    /// `.data..percpu` section has `sh_addr=0`. The kernel virtual
    /// address for CPU `n` is `runqueues + per_cpu_offset[n]`; see
    /// [`compute_rq_pas`].
    pub runqueues: u64,
    /// Kernel virtual address of the `__per_cpu_offset` array.
    pub per_cpu_offset: u64,
    /// Kernel virtual address of `page_offset_base`. None when the
    /// symbol is absent (non-KASLR kernel without the variable).
    /// The runtime value must be read from guest memory via
    /// `resolve_page_offset`.
    pub page_offset_base_kva: Option<u64>,
    /// Kernel virtual address of `phys_base`
    /// (`arch/x86/kernel/head_64.S`). Holds the runtime physical
    /// address offset between the kernel image's compile-time
    /// VA (`__START_KERNEL_map`) and its load PA: the kernel sets
    /// `phys_base = load_delta = __START_KERNEL_map + p2v_offset`
    /// (see `arch/x86/boot/startup/map_kernel.c:__startup_64`).
    /// Used by `__phys_addr` (`arch/x86/mm/physaddr.c:15-32`):
    /// `pa = (kva - __START_KERNEL_map) + phys_base`.
    ///
    /// On a non-KASLR build `phys_base == 0` and the formula
    /// collapses to `pa = kva - __START_KERNEL_map`. With KASLR
    /// enabled, the kernel is loaded at a randomized PA above
    /// `0x10_0000` (the `LOAD_PHYSICAL_ADDR` floor), so `phys_base`
    /// holds the post-randomization PA of the kernel image. The
    /// monitor walks the page tables to translate this symbol's KVA
    /// into a PA without a circular dependency on `phys_base`
    /// itself; the resolved value then feeds every subsequent
    /// text/data symbol translation.
    ///
    /// `None` when the symbol is absent (aarch64 kernels do not
    /// define `phys_base`; their analogue is `kimage_voffset` which
    /// the boot-time `start_kernel_map_for_tcr` derivation already
    /// covers).
    pub phys_base_kva: Option<u64>,
    /// Kernel virtual address of `scx_root` (pointer to active scx_sched).
    /// None when the symbol is absent: pre-6.16 kernels with sched_ext
    /// (older `scx_ops` API predates `scx_root`), and kernels built
    /// without sched_ext.
    pub scx_root: Option<u64>,
    /// Kernel virtual address of `scx_tasks` — the global LIST_HEAD
    /// (`kernel/sched/ext.c:47`) every scx-managed task is linked
    /// into via `task_struct.scx.tasks_node`. The host walker uses
    /// this anchor to enumerate every task owned by an scx_sched
    /// across ALL CPUs in one walk, surviving the per-rq
    /// runnable_list drain that `scx_bypass`
    /// (`kernel/sched/ext.c:5304-5404`) triggers during scheduler
    /// teardown — `scx_tasks` outlives runnable_list because tasks
    /// only leave it via `sched_ext_dead`
    /// (`kernel/sched/ext.c:3792`). `None` when the symbol is
    /// absent (kernel without sched_ext, or stripped vmlinux).
    pub scx_tasks: Option<u64>,
    /// Kernel virtual address of the top-level page table.
    /// `init_top_pgt` (older kernels) or `swapper_pg_dir` (newer kernels).
    /// Used to derive CR3 for page table walks when KVM SREGS are unavailable.
    pub init_top_pgt: Option<u64>,
    /// Kernel virtual address of `__pgtable_l5_enabled` (u32).
    /// 0 = 4-level paging, 1 = 5-level paging (LA57 active).
    /// None if the symbol is absent (CONFIG_PGTABLE_LEVELS < 5).
    pub pgtable_l5_enabled: Option<u64>,
    /// Kernel virtual address of `prog_idr` (BPF program IDR).
    /// None if the symbol is absent.
    pub prog_idr: Option<u64>,
    /// Kernel virtual address of `scx_watchdog_timeout` (static global).
    /// Present on pre-7.1 kernels where the watchdog timeout is a
    /// file-scope static rather than a field on `struct scx_sched`.
    /// None on 7.1+ or when the symbol is absent.
    pub scx_watchdog_timeout: Option<u64>,
    /// Kernel virtual address of `scx_watchdog_timestamp` (file-scope
    /// static `unsigned long` declared at `kernel/sched/ext.c:94`).
    /// Updated to `jiffies` by `scx_watchdog_workfn`
    /// (`kernel/sched/ext.c:3383`) each time the workqueue runs;
    /// `scx_tick` (`kernel/sched/ext.c:3409`) reads it via `READ_ONCE`
    /// and fires `SCX_EXIT_ERROR_STALL` when
    /// `jiffies > timestamp + root->watchdog_timeout`. Reading it at
    /// scan time gives the dual-snapshot path the same global stall
    /// signal `scx_tick` checks, regardless of whether any task is
    /// stuck on a per-rq runnable_list. None when the symbol is absent
    /// (kernel without sched_ext or stripped vmlinux). Lives in
    /// `.data` (file-scope static), so resolution uses
    /// [`text_kva_to_pa_with_base`].
    pub scx_watchdog_timestamp: Option<u64>,
    /// Kernel virtual address of `jiffies_64` (`u64` global maintained
    /// by the timer subsystem). Used by the dual-snapshot freeze
    /// coordinator to compare each runnable task's `p->scx.runnable_at`
    /// against the current jiffies value, mirroring the kernel's
    /// `check_rq_for_timeouts` walk. None when the symbol is absent
    /// (CONFIG_64BIT=n on a host that emits only the legacy `jiffies`
    /// alias, or a stripped vmlinux that lost the symbol).
    pub jiffies_64: Option<u64>,
    /// `.data..percpu` section-relative offset of the
    /// `kernel_cpustat` per-CPU variable. The per-CPU KVA for CPU
    /// `n` is `kernel_cpustat + __per_cpu_offset[n]`; same percpu-
    /// addressing rule as [`Self::runqueues`]. Used by the failure
    /// dump to read each CPU's `cpustat[CPUTIME_*]` counters.
    /// `None` when the symbol is absent from a stripped vmlinux —
    /// per-CPU CPU-time capture is then skipped.
    pub kernel_cpustat: Option<u64>,
    /// `.data..percpu` section-relative offset of the `kstat`
    /// per-CPU variable (`struct kernel_stat`). Used by the failure
    /// dump to read each CPU's `irqs_sum` and `softirqs[]`. `None`
    /// when absent.
    pub kstat: Option<u64>,
    /// `.data..percpu` section-relative offset of the `tick_cpu_sched`
    /// per-CPU variable (`struct tick_sched`). Used by the failure
    /// dump to read each CPU's `iowait_sleeptime`. `None` when
    /// absent — kernels without `CONFIG_NO_HZ_COMMON` omit this
    /// symbol; the dump path then skips that field per
    /// [`super::btf_offsets::CpuTimeOffsets::tick_sched_iowait_sleeptime`].
    pub tick_cpu_sched: Option<u64>,
    /// Kernel virtual address of `node_data[]` (declared in
    /// `include/linux/numa.h` as `extern struct pglist_data *node_data[];`).
    /// On a NUMA build the array holds `MAX_NUMNODES` `pglist_data *`
    /// pointers; on a UMA build the symbol may be absent. Used by the
    /// per-node NUMA event walker to reach each node's `pglist_data`.
    /// `None` when the symbol is absent (UMA build, stripped vmlinux,
    /// or kernel built without `CONFIG_NUMA`).
    ///
    /// Stub-stage `dead_code` suppression: the consumer
    /// ([`crate::vmm::capture_numa::build`]) is wired but returns
    /// `None` until the implementer fills in the walker. Removing
    /// this attribute is the natural marker for "the producer has
    /// landed" — drop it the moment a real reader appears.
    #[allow(dead_code)]
    pub node_data: Option<u64>,
}

impl KernelSymbols {
    /// Parse a vmlinux ELF and extract symbol addresses for kernel
    /// monitoring.
    ///
    /// The `page_offset_base` symbol KVA is stored but NOT dereferenced
    /// here — call `resolve_page_offset` with a `GuestMem` after the
    /// guest kernel has booted to read the runtime value.
    pub fn from_vmlinux(path: &Path) -> Result<Self> {
        let data =
            std::fs::read(path).with_context(|| format!("read vmlinux: {}", path.display()))?;
        Self::from_vmlinux_bytes(&data)
    }

    /// Same as [`Self::from_vmlinux`] but accepts pre-read vmlinux
    /// ELF bytes, avoiding a redundant `std::fs::read`.
    pub fn from_vmlinux_bytes(data: &[u8]) -> Result<Self> {
        let elf = goblin::elf::Elf::parse(data).context("parse vmlinux ELF")?;
        Self::from_elf(&elf)
    }

    /// Same as [`Self::from_vmlinux_bytes`] but accepts a pre-parsed
    /// `goblin::elf::Elf`, avoiding a redundant ELF parse when the
    /// caller already holds one. The `Elf` borrows from the underlying
    /// vmlinux bytes; the caller must keep those bytes alive for the
    /// duration of this call but no longer — the returned
    /// `KernelSymbols` carries owned `u64` values only.
    pub fn from_elf(elf: &goblin::elf::Elf<'_>) -> Result<Self> {
        // SHN_UNDEF = 0 (ELF spec): undefined symbols (linker
        // placeholders) carry st_shndx == 0 and must be skipped
        // here. We DO NOT filter `st_value != 0` because the cached-
        // vmlinux strip pipeline rewrites `.data..percpu` sh_addr to
        // 0, leaving percpu st_value as a section-relative offset.
        // A percpu symbol legitimately at offset 0 (e.g. the first
        // entry in `.data..percpu`) would be silently dropped by an
        // st_value filter, masquerading as an absent symbol.
        const SHN_UNDEF: u16 = 0;
        let sym_addr = |name: &str| -> Option<u64> {
            elf.syms
                .iter()
                .find(|s| s.st_shndx as u16 != SHN_UNDEF && elf.strtab.get_at(s.st_name) == Some(name))
                .map(|s| s.st_value)
        };

        let runqueues = sym_addr("runqueues").context("symbol 'runqueues' not found in vmlinux")?;

        let per_cpu_offset = sym_addr("__per_cpu_offset")
            .context("symbol '__per_cpu_offset' not found in vmlinux")?;

        let page_offset_base_kva = sym_addr("page_offset_base");
        // x86_64-only KASLR randomization base; absent on aarch64
        // kernels (their kimage_voffset analogue is derived from
        // `start_kernel_map_for_tcr`, not a static symbol).
        let phys_base_kva = sym_addr("phys_base");

        let scx_root = sym_addr("scx_root");
        // scx_tasks is `static LIST_HEAD(scx_tasks)` in
        // kernel/sched/ext.c:47. Static globals carry lowercase 't'
        // in `nm` output but are still present in `.symtab` —
        // sym_addr resolves them by name, no st_bind filtering.
        let scx_tasks = sym_addr("scx_tasks");

        let init_top_pgt = sym_addr("init_top_pgt").or_else(|| sym_addr("swapper_pg_dir"));

        let pgtable_l5_enabled = sym_addr("__pgtable_l5_enabled");

        let prog_idr = sym_addr("prog_idr");

        let scx_watchdog_timeout = sym_addr("scx_watchdog_timeout");

        // scx_watchdog_timestamp is a file-scope static
        // (`static unsigned long scx_watchdog_timestamp` in
        // kernel/sched/ext.c) — like other static globals it appears in
        // .symtab with lowercase 'd' in `nm` output but is still
        // resolvable by name. Absent on kernels without sched_ext.
        let scx_watchdog_timestamp = sym_addr("scx_watchdog_timestamp");

        let jiffies_64 = sym_addr("jiffies_64");

        // Per-CPU CPU-time / softirq / IRQ / iowait_sleeptime
        // symbols. All three are `.data..percpu` (section-relative
        // offsets, NOT KVAs); resolution per CPU adds
        // `__per_cpu_offset[cpu]` like every other percpu read.
        // Each is optional: a stripped vmlinux without
        // `kernel_cpustat`/`kstat` skips the corresponding capture
        // leg without failing the dump (CPU-time is best-effort
        // diagnostics, not a precondition for the rest of the
        // freeze-coordinator output).
        let kernel_cpustat = sym_addr("kernel_cpustat");
        let kstat = sym_addr("kstat");
        let tick_cpu_sched = sym_addr("tick_cpu_sched");

        // node_data is the `extern struct pglist_data *node_data[]`
        // global; absent on UMA builds and on kernels built without
        // CONFIG_NUMA. Walker gates capture on Some.
        let node_data = sym_addr("node_data");

        Ok(Self {
            runqueues,
            per_cpu_offset,
            page_offset_base_kva,
            phys_base_kva,
            scx_root,
            scx_tasks,
            init_top_pgt,
            pgtable_l5_enabled,
            prog_idr,
            scx_watchdog_timeout,
            scx_watchdog_timestamp,
            jiffies_64,
            kernel_cpustat,
            kstat,
            tick_cpu_sched,
            node_data,
        })
    }
}

/// Read the runtime value of PAGE_OFFSET from guest memory.
///
/// Preference chain:
/// 1. Live `page_offset_base` symbol value (modern kernels with
///    `CONFIG_RANDOMIZE_MEMORY`).
/// 2. [`DEFAULT_PAGE_OFFSET`] (48-bit VA hardcoded) as the
///    fallback when the symbol is absent — wrong for 47-bit
///    kernels but only reachable on stripped vmlinux. Modern
///    kernels with `CONFIG_RANDOMIZE_MEMORY=y` always export
///    the symbol.
///
/// `start_kernel_map` is the runtime base resolved by the caller
/// (x86_64: [`START_KERNEL_MAP`]; aarch64: derived from `tcr_el1`
/// via [`start_kernel_map_for_tcr`]).
///
/// For the TCR-aware fallback path that picks the right
/// `-(1 << VA_BITS)` for 47-bit kernels, see
/// [`resolve_page_offset_with_tcr`].
// Production callers replaced by [`resolve_page_offset_with_tcr`];
// preserved for tests that pin the `tcr_el1 = 0` x86_64 path.
#[allow(dead_code)]
pub(crate) fn resolve_page_offset(
    mem: &super::reader::GuestMem,
    symbols: &KernelSymbols,
    start_kernel_map: u64,
) -> u64 {
    resolve_page_offset_with_tcr(mem, symbols, start_kernel_map, 0, 0)
}

/// Like [`resolve_page_offset`] but uses
/// [`default_page_offset_for_tcr`] as the fallback so 47-bit
/// kernels (16 KiB granule, Apple Silicon) read the right
/// `-(1 << 47)` value when `page_offset_base` is absent.
/// `tcr_el1` is the guest's register value; pass `0` on x86_64.
///
/// On aarch64 the `page_offset_base` symbol is absent
/// (x86_64-only — see `arch/x86/mm/kaslr.c`) so the fallback IS
/// the production path on aarch64. The fallback uses
/// `vabits_actual` from `TCR_EL1` as a proxy for the compile-time
/// `VA_BITS`; see [`default_page_offset_for_tcr`] for the
/// `CONFIG_ARM64_VA_BITS_52` ambiguity that ktstr.kconfig avoids
/// by not enabling that config.
pub(crate) fn resolve_page_offset_with_tcr(
    mem: &super::reader::GuestMem,
    symbols: &KernelSymbols,
    start_kernel_map: u64,
    tcr_el1: u64,
    phys_base: u64,
) -> u64 {
    let Some(pob_kva) = symbols.page_offset_base_kva else {
        return default_page_offset_for_tcr(tcr_el1);
    };
    let pob_pa = text_kva_to_pa_with_base(pob_kva, start_kernel_map, phys_base);
    let val = mem.read_u64(pob_pa, 0);
    // Valid PAGE_OFFSET has bit 63 set (upper-half virtual address).
    // Kernels with CONFIG_RANDOMIZE_MEMORY use values like
    // 0xff11000000000000 that are below the traditional canonical
    // boundary (0xffff800000000000), so check bit 63 instead.
    if val & (1u64 << 63) != 0 {
        val
    } else {
        default_page_offset_for_tcr(tcr_el1)
    }
}

/// Resolve the kernel's runtime `phys_base` value via a page-table
/// walk.
///
/// Breaks the chicken-and-egg between text-symbol PA translation
/// and KASLR: every text/data translate normally needs `phys_base`
/// (`pa = (kva - start_kernel_map) + phys_base`) but `phys_base`
/// itself lives in `.data` whose PA we'd need `phys_base` to find.
/// The page table walker takes a CR3 (already a PA, read from KVM
/// SREGS) and produces PAs directly from PTE entries — no
/// `phys_base` involved — so we can walk the symbol's KVA to a PA,
/// then read the live `phys_base` value.
///
/// `cr3_pa` is the BSP's CR3 (KVM_GET_SREGS, masked to a PA per
/// Intel SDM §4.5: bits [11:0] hold PCID/PCD/PWT control bits and
/// must be cleared). `l5` selects the walker variant; resolve
/// once via [`resolve_pgtable_l5`] BEFORE the first
/// `resolve_phys_base` call (the L5 read uses `phys_base = 0` and
/// is therefore correct only when KASLR is disabled — for the
/// boot-time pgtable mode probe that requirement is met because
/// `__pgtable_l5_enabled` is zeroed at compile time on every
/// non-LA57 build, and on LA57 builds the value is set by
/// `__startup_64` BEFORE `phys_base` is randomized).
///
/// Returns `None` when:
/// - `phys_base` symbol is absent (aarch64, stripped vmlinux);
/// - the page-table walk for the symbol KVA fails (CR3 not yet
///   populated, page tables not yet initialised, or the symbol's
///   KVA is unmapped — none of which should happen post-boot but
///   the walker returns `None` defensively).
///
/// On a non-KASLR kernel the resolved value is `0`; with KASLR
/// enabled it carries the post-randomization PA of the kernel
/// image. Either result is correct: the
/// [`text_kva_to_pa_with_base`] formula collapses to
/// `pa = kva - start_kernel_map` when `phys_base == 0`.
pub(crate) fn resolve_phys_base(
    mem: &super::reader::GuestMem,
    symbols: &KernelSymbols,
    cr3_pa: u64,
    l5: bool,
    tcr_el1: u64,
) -> Option<u64> {
    let kva = symbols.phys_base_kva?;
    // CR3 carries PCID/PCD/PWT control bits in [11:0]; mask them off
    // before treating the value as a PA. The walker's `ADDR_MASK`
    // already does this internally for descriptor entries, but the
    // initial CR3 we're handed by KVM SREGS is the raw register
    // value.
    let cr3_pa_masked = cr3_pa & !0xFFFu64;
    let pa = mem.translate_kva(cr3_pa_masked, super::Kva(kva), l5, tcr_el1)?;
    Some(mem.read_u64(pa, 0))
}

/// Read the runtime value of `__pgtable_l5_enabled` from guest memory.
///
/// Returns `true` when the guest kernel uses 5-level paging (LA57),
/// `false` when the symbol is absent or the value is 0.
/// `start_kernel_map` is the runtime kernel image base used to
/// translate the symbol KVA — see [`resolve_page_offset`].
/// `phys_base` is the kernel's runtime KASLR offset; pass `0`
/// during the boot-time bootstrap (the L5 mode is set by
/// `__startup_64` BEFORE `phys_base` is randomized, so a
/// `phys_base = 0` read still finds a populated value at the
/// expected PA on x86_64 KASLR boots).
pub(crate) fn resolve_pgtable_l5(
    mem: &super::reader::GuestMem,
    symbols: &KernelSymbols,
    start_kernel_map: u64,
    phys_base: u64,
) -> bool {
    let Some(kva) = symbols.pgtable_l5_enabled else {
        return false;
    };
    let pa = text_kva_to_pa_with_base(kva, start_kernel_map, phys_base);
    mem.read_u32(pa, 0) != 0
}

/// Translate a kernel virtual address in the direct mapping
/// (PAGE_OFFSET region) to a DRAM-relative offset for GuestMem.
///
/// On both x86_64 and aarch64, the direct mapping maps DRAM offset 0
/// at PAGE_OFFSET: `kva = page_offset + dram_offset`. On aarch64 the
/// kernel's `__phys_to_virt(gpa)` is `(gpa - PHYS_OFFSET) | PAGE_OFFSET`,
/// and `PHYS_OFFSET = memstart_addr = DRAM_START`, so
/// `kva = dram_offset | PAGE_OFFSET = PAGE_OFFSET + dram_offset`
/// (the `|` is equivalent to `+` since the operands don't overlap).
/// Subtracting PAGE_OFFSET recovers the DRAM offset directly.
pub(crate) fn kva_to_pa(kva: u64, page_offset: u64) -> u64 {
    kva.wrapping_sub(page_offset)
}

/// Translate a kernel text/data symbol VA to a DRAM-relative offset
/// for GuestMem.
///
/// Kernel text and data symbols (.text, .data, .bss) are mapped via
/// `__START_KERNEL_map` (x86_64) / `KIMAGE_VADDR` (aarch64), not
/// the direct mapping. The kernel's `__phys_addr` formula
/// (`arch/x86/mm/physaddr.c:15-32`) is
/// `pa = (kva - __START_KERNEL_map) + phys_base`, and the aarch64
/// equivalent (`__kimg_to_phys(addr) = addr - kimage_voffset`) is
/// `pa = (kva - KIMAGE_VADDR) + DRAM_START` because
/// `kimage_voffset = KIMAGE_VADDR - phys_base` and on aarch64
/// `phys_base = DRAM_START` for non-KASLR builds.
///
/// On x86_64 the runtime `phys_base` value comes from the kernel's
/// `phys_base` static, set by `__startup_64`
/// (`arch/x86/boot/startup/map_kernel.c:122`). Without KASLR
/// `phys_base == 0` and the formula collapses to
/// `pa = kva - __START_KERNEL_map`. With KASLR the kernel image
/// loads at a randomized PA above `LOAD_PHYSICAL_ADDR` (16 MiB
/// floor) and `phys_base` carries that PA so the formula above
/// resolves text/data symbols correctly.
///
/// On aarch64 callers pass `phys_base = DRAM_START`; the aarch64
/// build does not export a `phys_base` symbol. With KASLR on
/// aarch64 the kernel image still maps to `KIMAGE_VADDR` (the
/// virtual address is fixed by the linker); the post-KASLR
/// physical-side delta is captured by the kernel's
/// `kimage_voffset` runtime variable, which we do not resolve
/// here — `start_kernel_map_for_tcr` already returns the runtime
/// VA base, and `phys_base = DRAM_START` cancels out the
/// PHYS_OFFSET term identically to the original formula. Aarch64
/// KASLR support is therefore a follow-up that resolves
/// `kimage_voffset` from a parallel page-table walk; the current
/// callers pass `phys_base = DRAM_START` and the result is correct
/// for non-KASLR aarch64 boots.
///
/// `start_kernel_map` is the runtime kernel image base — pass
/// [`START_KERNEL_MAP`] on x86_64 and the value derived via
/// [`start_kernel_map_for_tcr`] on aarch64.
///
/// `phys_base` is the kernel's runtime `phys_base` (x86_64) or
/// `DRAM_START` (aarch64). The boot-time bootstrap value is
/// `0` (x86_64) / `DRAM_START` (aarch64); production callers
/// pass the resolved value once the BSP has established the
/// kernel image mapping (see [`resolve_phys_base`]).
///
/// Most callers funnel through
/// [`super::guest::GuestKernel::text_kva_to_pa`] which carries
/// both the resolved base and `phys_base` alongside the symbol
/// map.
pub(crate) fn text_kva_to_pa_with_base(kva: u64, start_kernel_map: u64, phys_base: u64) -> u64 {
    kva.wrapping_sub(start_kernel_map).wrapping_add(phys_base)
}

/// Derive the aarch64 kernel image base (`KIMAGE_VADDR`) from
/// `TCR_EL1`.
///
/// Mirrors the kernel's `KIMAGE_VADDR = _PAGE_END(VA_BITS_MIN) + SZ_2G`
/// (`arch/arm64/include/asm/memory.h`), where `VA_BITS_MIN` depends
/// on the compile-time `VA_BITS` and the granule. The runtime
/// values reachable from `TCR_EL1`:
/// - `T1SZ` (bits [21:16]) → `VA_BITS_runtime = 64 - T1SZ`
/// - `TG1` (bits [31:30]) → granule (0b01=16 KB, 0b10=4 KB,
///   0b11=64 KB; 0b00 reserved)
///
/// `VA_BITS_MIN` reconstruction (`asm/memory.h:56-64`):
/// - if compile-time `VA_BITS <= 48`: `VA_BITS_MIN = VA_BITS`
///   (per `#else #define VA_BITS_MIN (VA_BITS)`)
/// - else (compile-time VA_BITS=52): `VA_BITS_MIN=47` on 16 KB
///   granule, `VA_BITS_MIN=48` otherwise
///
/// **Compile-time-vs-runtime ambiguity (KNOWN LIMITATION):**
/// `TCR_EL1` exposes `vabits_actual = 64 - T1SZ`, NOT the
/// compile-time `VA_BITS`. They match in every case EXCEPT
/// `CONFIG_ARM64_VA_BITS_52=y` running on hardware that lacks
/// FEAT_LVA — there, `VA_BITS=52` (compile) but
/// `vabits_actual=48` (T1SZ=16). On 16 KB pages this matters:
/// the kernel still uses `VA_BITS_MIN=47` (compile-time rule),
/// placing `KIMAGE_VADDR` at `0xFFFF_C000_8000_0000`, but this
/// function — seeing only `T1SZ=16` — returns `0xFFFF_8000_8000_0000`
/// (the 48-bit answer). Callers translating text/data symbols on
/// such a kernel will read the wrong PAs. ktstr.kconfig does not
/// enable `CONFIG_ARM64_VA_BITS_52`, so the ambiguity is dormant
/// for current ktstr usage; a future user that pins that config
/// must disambiguate from a kernel symbol KVA (e.g. `_text`'s high
/// bits) since `TCR_EL1` alone is insufficient.
///
/// Examples (each mapping `_PAGE_END(VA_BITS_MIN) + SZ_2G`):
/// - `T1SZ=16` (48-bit, 4 KB): assumes VA_BITS=48 →
///   `VA_BITS_MIN=48` → `0xFFFF_8000_8000_0000`
/// - `T1SZ=17` (47-bit, 16 KB Apple Silicon): VA_BITS=47 →
///   `VA_BITS_MIN=47` → `0xFFFF_C000_8000_0000`
/// - `T1SZ=12` (52-bit, 16 KB): VA_BITS=52, runtime activates →
///   `VA_BITS_MIN=47` → `0xFFFF_C000_8000_0000`
/// - `T1SZ=12` (52-bit, 4 KB): VA_BITS=52, runtime activates →
///   `VA_BITS_MIN=48` → `0xFFFF_8000_8000_0000`
///
/// Returns `None` when `tcr_el1 == 0` (BSP loop has not yet read
/// the register), when `T1SZ == 0` (high-half disabled), when TG1
/// holds the reserved `0b00` encoding, or when the derived
/// `VA_BITS` would underflow `_PAGE_END`. Callers in retry loops
/// should treat `None` as "TCR not ready, retry".
///
/// On x86_64 `tcr_el1` is ignored and the compile-time
/// [`START_KERNEL_MAP`] is returned wrapped in `Some` — the kernel
/// image lives at the same VA regardless of paging mode (4-level /
/// 5-level), so the derivation is degenerate.
pub(crate) fn start_kernel_map_for_tcr(tcr_el1: u64) -> Option<u64> {
    #[cfg(target_arch = "x86_64")]
    {
        let _ = tcr_el1;
        Some(START_KERNEL_MAP)
    }
    #[cfg(target_arch = "aarch64")]
    {
        if tcr_el1 == 0 {
            return None;
        }
        let t1sz = (tcr_el1 >> 16) & 0x3F;
        if t1sz == 0 {
            return None;
        }
        let tg1 = (tcr_el1 >> 30) & 0x3;
        // TG1=0b00 is reserved per Arm ARM D17.2.139; without a
        // valid granule we cannot determine `VA_BITS_MIN` for the
        // 52-bit branch.
        if tg1 == 0 {
            return None;
        }
        let va_bits_runtime: u32 = (64u32).saturating_sub(t1sz as u32);
        let is_16k_granule = tg1 == 0b01;
        // VA_BITS_MIN reconstruction. The kernel header derives this
        // from the compile-time `VA_BITS`, but `TCR_EL1` only exposes
        // `vabits_actual` (= `va_bits_runtime`). The two diverge ONLY
        // on `CONFIG_ARM64_VA_BITS_52=y` kernels running on hardware
        // without FEAT_LVA: VA_BITS=52 (compile) but T1SZ=16
        // (runtime falls back to 48). The 16 KB-granule case returns
        // a wrong base in that scenario (see the function-level doc
        // for the full disambiguation requirement). All other
        // configurations agree: when `va_bits_runtime <= 48` the
        // kernel was almost always compiled with that exact VA_BITS,
        // and the compile-time rule reduces to `VA_BITS_MIN = VA_BITS
        // = va_bits_runtime`. ktstr.kconfig leaves
        // `CONFIG_ARM64_VA_BITS_52` unset, so the ambiguous branch
        // is unreachable for current ktstr usage.
        let va_bits_min: u32 = if va_bits_runtime <= 48 {
            va_bits_runtime
        } else if is_16k_granule {
            47
        } else {
            48
        };
        // _PAGE_END(va) = -(1 << (va - 1)). Reject va==0 (would
        // shift by -1 / underflow) and va > 64 (shift overflow);
        // the bounded derivation above guarantees va_bits_min in
        // [1, 48] on legitimate inputs.
        if va_bits_min == 0 || va_bits_min > 64 {
            return None;
        }
        let page_end = 0u64.wrapping_sub(1u64 << (va_bits_min - 1));
        Some(page_end.wrapping_add(0x8000_0000))
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = tcr_el1;
        compile_error!("unsupported architecture for start_kernel_map_for_tcr")
    }
}

/// Read the `__per_cpu_offset` array from guest memory.
/// Returns per-CPU offsets for each CPU (index = CPU number).
///
/// Each u64 element is read via [`GuestMem::read_u64`], which uses
/// per-byte `read_volatile` internally and is alignment-safe even
/// when `per_cpu_offset_pa` is not 8-aligned. The previous raw
/// `std::ptr::read_volatile(*const u64)` implementation was UB on
/// misaligned addresses (the same alignment gap that reader.rs
/// closes via its GuestMem wrapper). GuestMem also bounds-checks
/// each read against its
/// mapped size, so reads past the end of guest memory return 0
/// instead of faulting.
pub(crate) fn read_per_cpu_offsets(
    mem: &super::reader::GuestMem,
    per_cpu_offset_pa: u64,
    num_cpus: u32,
) -> Vec<u64> {
    (0..num_cpus)
        .map(|cpu| mem.read_u64(per_cpu_offset_pa + (cpu as u64) * 8, 0))
        .collect()
}

/// Compute the physical address of each CPU's `struct rq`.
///
/// Each CPU's rq is at `runqueues_kva + per_cpu_offset[cpu]` in kernel
/// virtual space; subtracting PAGE_OFFSET yields the guest physical address.
///
/// `per_cpu_offsets` slots that are zero produce a PA at
/// `kva_to_pa(runqueues_kva, page_offset)`, which on a typical
/// `runqueues` percpu offset (small, far below `page_offset`)
/// wraps via `wrapping_sub` into the upper-half KVA region — far
/// outside any guest DRAM region. [`super::reader::GuestMem::read_u64`]
/// silently bounds-rejects such reads to zero, so callers get
/// zero-filled `CpuSnapshot`s when this function is fed a stale
/// (BSS-zero) per-CPU offset table. Callers that need to dodge
/// the host-monitor / guest-BSP boot race should refresh the
/// offset table per sample (see
/// [`super::reader::RqRefresh`]).
pub(crate) fn compute_rq_pas(
    runqueues_kva: u64,
    per_cpu_offsets: &[u64],
    page_offset: u64,
) -> Vec<u64> {
    per_cpu_offsets
        .iter()
        .map(|&offset| kva_to_pa(runqueues_kva.wrapping_add(offset), page_offset))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_runqueues_symbol() {
        let path = match crate::monitor::find_test_vmlinux() {
            Some(p) => p,
            None => return,
        };
        // find_test_vmlinux may return /sys/kernel/btf/vmlinux (raw BTF,
        // not an ELF), which KernelSymbols cannot parse.
        if path.starts_with("/sys/") {
            skip!("vmlinux is raw BTF (not ELF), cannot parse symbols");
        }
        let syms = KernelSymbols::from_vmlinux(&path).unwrap();
        assert_ne!(syms.runqueues, 0);
        assert_ne!(syms.per_cpu_offset, 0);
        // runqueues is a per-cpu symbol — its st_value is a section-
        // relative offset within .data..percpu (sh_addr=0), not a
        // kernel VA. per_cpu_offset is a kernel-VA data symbol
        // and is what should land in the upper half.
        assert!(syms.per_cpu_offset > 0xffff_0000_0000_0000);
    }

    #[test]
    fn kva_to_pa_basic() {
        // KVA = PAGE_OFFSET + dram_offset (kernel's __phys_to_virt
        // subtracts PHYS_OFFSET then ORs PAGE_OFFSET, producing
        // PAGE_OFFSET + dram_offset for small offsets).
        let page_offset = DEFAULT_PAGE_OFFSET;
        let dram_kva = page_offset.wrapping_add(0x10_0000);
        assert_eq!(kva_to_pa(dram_kva, page_offset), 0x10_0000);
        assert_eq!(kva_to_pa(page_offset, page_offset), 0);
    }

    #[test]
    fn compute_rq_pas_two_cpus() {
        let page_offset = DEFAULT_PAGE_OFFSET;
        let runqueues = page_offset.wrapping_add(0x20_0000);
        let offsets = vec![0, 0x4_0000]; // CPU 0 at base, CPU 1 at +256KB
        let pas = compute_rq_pas(runqueues, &offsets, page_offset);
        assert_eq!(pas[0], 0x20_0000);
        assert_eq!(pas[1], 0x24_0000);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn start_kernel_map_for_tcr_returns_none_on_zero() {
        // tcr_el1 = 0 means the BSP loop has not yet read the
        // register; the derivation cannot proceed and the result
        // must be None so retry loops keep polling. On x86_64
        // tcr_el1 has no meaning and the function returns the
        // compile-time START_KERNEL_MAP unconditionally — see
        // start_kernel_map_for_tcr_x86_64_constant.
        assert_eq!(start_kernel_map_for_tcr(0), None);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn start_kernel_map_for_tcr_x86_64_constant() {
        // x86_64 kernel image base does not depend on TCR_EL1
        // (the register does not exist); the derivation returns
        // the compile-time constant for any input.
        assert_eq!(start_kernel_map_for_tcr(0x12345), Some(START_KERNEL_MAP));
        assert_eq!(start_kernel_map_for_tcr(u64::MAX), Some(START_KERNEL_MAP));
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn start_kernel_map_for_tcr_aarch64_48bit() {
        // VA_BITS=48, T1SZ=16, TG1=0b10 (4 KB granule).
        // VA_BITS_MIN=48 → KIMAGE_VADDR = -(1<<47) + 0x8000_0000
        //               = 0xFFFF_8000_8000_0000.
        let tcr = (0b10u64 << 30) | (16u64 << 16);
        assert_eq!(start_kernel_map_for_tcr(tcr), Some(0xFFFF_8000_8000_0000));
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn start_kernel_map_for_tcr_aarch64_47bit_16k() {
        // VA_BITS=47, T1SZ=17, TG1=0b01 (16 KB granule, Apple Silicon).
        // VA_BITS_MIN=47 → KIMAGE_VADDR = -(1<<46) + 0x8000_0000
        //               = 0xFFFF_C000_8000_0000.
        let tcr = (0b01u64 << 30) | (17u64 << 16);
        assert_eq!(start_kernel_map_for_tcr(tcr), Some(0xFFFF_C000_8000_0000));
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn start_kernel_map_for_tcr_aarch64_52bit_4k() {
        // VA_BITS=52, T1SZ=12, TG1=0b10 (4 KB granule, runtime 52-bit).
        // The kernel image base still uses VA_BITS_MIN=48 (the 4 KB
        // branch in asm/memory.h:60), so the result matches the
        // 48-bit layout regardless of T1SZ=12 activating 52-bit VA.
        let tcr = (0b10u64 << 30) | (12u64 << 16);
        assert_eq!(start_kernel_map_for_tcr(tcr), Some(0xFFFF_8000_8000_0000));
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn start_kernel_map_for_tcr_aarch64_52bit_16k() {
        // VA_BITS=52, T1SZ=12, TG1=0b01 (16 KB granule, runtime 52-bit).
        // VA_BITS_MIN=47 (the 16 KB branch in asm/memory.h:58).
        let tcr = (0b01u64 << 30) | (12u64 << 16);
        assert_eq!(start_kernel_map_for_tcr(tcr), Some(0xFFFF_C000_8000_0000));
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn start_kernel_map_for_tcr_aarch64_rejects_reserved_tg1() {
        // TG1=0b00 is reserved per Arm ARM D17.2.139; without a
        // valid granule the VA_BITS_MIN derivation cannot pick the
        // right branch for VA_BITS>48 kernels.
        let tcr = (0b00u64 << 30) | (16u64 << 16);
        assert_eq!(start_kernel_map_for_tcr(tcr), None);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn start_kernel_map_for_tcr_aarch64_rejects_t1sz_zero() {
        // T1SZ=0 means the high half is disabled; nothing useful
        // to derive. TG1=0b10 (4 KB) so only T1SZ is the failure.
        let tcr = 0b10u64 << 30;
        assert_eq!(start_kernel_map_for_tcr(tcr), None);
    }

    /// `CONFIG_ARM64_VA_BITS_52=y` running on hardware without
    /// FEAT_LVA: T1SZ=16 (vabits_actual=48) but the kernel was
    /// compiled with VA_BITS=52, so VA_BITS_MIN follows the
    /// compile-time rule (47 on 16 KB pages, 48 otherwise). The
    /// runtime-only signal in `TCR_EL1` cannot distinguish this
    /// from a plain VA_BITS=48 build, so the function returns the
    /// 48-bit answer in both cases. This test pins the (known,
    /// dormant) incorrect behaviour for the 16 KB-page sub-case
    /// so a future fix that disambiguates via a kernel symbol
    /// KVA can be detected as a behaviour change. ktstr.kconfig
    /// does not enable `CONFIG_ARM64_VA_BITS_52`, so this code
    /// path is unreachable for current ktstr usage; the test
    /// documents the limitation, it does not assert correctness
    /// for that compile-time configuration.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn start_kernel_map_for_tcr_aarch64_va_bits_52_reduced_runtime_returns_48bit() {
        // VA_BITS=52 (compile), 16 KB pages, T1SZ=16 (HW fallback
        // to 48-bit VA). Correct VA_BITS_MIN=47 → KIMAGE_VADDR
        // 0xFFFF_C000_8000_0000, but the function only sees T1SZ=16
        // and returns the 48-bit answer.
        let tcr_52_16k_reduced = (0b01u64 << 30) | (16u64 << 16);
        assert_eq!(
            start_kernel_map_for_tcr(tcr_52_16k_reduced),
            Some(0xFFFF_8000_8000_0000),
            "TCR_EL1 alone cannot distinguish VA_BITS=48 from \
             VA_BITS=52+16K with HW fallback; 0xFFFF_C000_8000_0000 \
             would be correct for VA_BITS=52+16K"
        );
        // VA_BITS=52 (compile), 4 KB pages, T1SZ=16. Correct
        // VA_BITS_MIN=48 → KIMAGE_VADDR 0xFFFF_8000_8000_0000.
        // Function returns the same answer; this case happens to
        // be correct because 4 KB / 64 KB granules use VA_BITS_MIN=48
        // regardless of compile-time VA_BITS.
        let tcr_52_4k_reduced = (0b10u64 << 30) | (16u64 << 16);
        assert_eq!(
            start_kernel_map_for_tcr(tcr_52_4k_reduced),
            Some(0xFFFF_8000_8000_0000)
        );
    }

    #[test]
    fn from_vmlinux_nonexistent() {
        let path = std::path::Path::new("/nonexistent/vmlinux");
        assert!(KernelSymbols::from_vmlinux(path).is_err());
    }

    #[test]
    fn read_per_cpu_offsets_zero_cpus() {
        use crate::monitor::reader::GuestMem;
        // With num_cpus=0, should return an empty vec without any reads.
        let mut buf = [0u8; 64];
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let result = read_per_cpu_offsets(&mem, 0, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn read_per_cpu_offsets_known_buffer() {
        use crate::monitor::reader::GuestMem;
        // Buffer with 3 known u64 offsets at PA 0.
        let mut buf = [0u8; 24];
        buf[0..8].copy_from_slice(&0x1000u64.to_ne_bytes());
        buf[8..16].copy_from_slice(&0x2000u64.to_ne_bytes());
        buf[16..24].copy_from_slice(&0x3000u64.to_ne_bytes());
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let result = read_per_cpu_offsets(&mem, 0, 3);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 0x1000);
        assert_eq!(result[1], 0x2000);
        assert_eq!(result[2], 0x3000);
    }

    #[test]
    fn read_per_cpu_offsets_nonzero_pa() {
        use crate::monitor::reader::GuestMem;
        // Place offsets at PA=16 (skip 16 bytes of padding).
        let mut buf = [0u8; 40]; // 16 padding + 3*8 offsets
        buf[16..24].copy_from_slice(&0xAAu64.to_ne_bytes());
        buf[24..32].copy_from_slice(&0xBBu64.to_ne_bytes());
        buf[32..40].copy_from_slice(&0xCCu64.to_ne_bytes());
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let result = read_per_cpu_offsets(&mem, 16, 3);
        assert_eq!(result, vec![0xAA, 0xBB, 0xCC]);
    }

    #[test]
    fn read_per_cpu_offsets_misaligned_pa() {
        use crate::monitor::reader::GuestMem;
        // Regression for the alignment UB fix: a non-8-aligned PA
        // (PA=1 here) must not cause misaligned-u64 UB. Byte-wise
        // volatile reads through GuestMem make this safe.
        let mut buf = [0u8; 32];
        buf[1..9].copy_from_slice(&0x1122_3344_5566_7788u64.to_ne_bytes());
        buf[9..17].copy_from_slice(&0x99AA_BBCC_DDEE_FF00u64.to_ne_bytes());
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let result = read_per_cpu_offsets(&mem, 1, 2);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 0x1122_3344_5566_7788);
        assert_eq!(result[1], 0x99AA_BBCC_DDEE_FF00);
    }

    #[test]
    fn read_per_cpu_offsets_out_of_bounds_returns_zero() {
        use crate::monitor::reader::GuestMem;
        // Asking for more CPUs than the buffer can hold: GuestMem's
        // bounds check yields 0 for each out-of-range read rather
        // than faulting or reading garbage past the mapped region.
        let mut buf = [0u8; 16];
        buf[0..8].copy_from_slice(&0x1111u64.to_ne_bytes());
        buf[8..16].copy_from_slice(&0x2222u64.to_ne_bytes());
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let result = read_per_cpu_offsets(&mem, 0, 4);
        assert_eq!(result, vec![0x1111, 0x2222, 0, 0]);
    }

    #[test]
    fn text_kva_to_pa_with_base_basic() {
        // phys_base=0 (non-KASLR / aarch64-after-DRAM-cancel): formula
        // collapses to `kva - start_kernel_map`.
        assert_eq!(
            text_kva_to_pa_with_base(START_KERNEL_MAP + 0x10_0000, START_KERNEL_MAP, 0),
            0x10_0000
        );
        assert_eq!(
            text_kva_to_pa_with_base(START_KERNEL_MAP, START_KERNEL_MAP, 0),
            0
        );
        // phys_base != 0 (KASLR): the offset shifts every text symbol
        // PA by the post-randomization kernel image PA.
        assert_eq!(
            text_kva_to_pa_with_base(START_KERNEL_MAP + 0x10_0000, START_KERNEL_MAP, 0x4000_0000),
            0x4010_0000
        );
    }

    #[test]
    fn kva_to_pa_wrapping() {
        // KVA < page_offset wraps around via wrapping_sub.
        let page_offset = DEFAULT_PAGE_OFFSET;
        let kva = 0x0000_0000_0001_0000u64;
        let pa = kva_to_pa(kva, page_offset);
        assert_eq!(pa, kva.wrapping_sub(page_offset));
    }

    #[test]
    fn compute_rq_pas_empty_offsets() {
        let page_offset = DEFAULT_PAGE_OFFSET;
        let runqueues = page_offset.wrapping_add(0x20_0000);
        let pas = compute_rq_pas(runqueues, &[], page_offset);
        assert!(pas.is_empty());
    }

    #[test]
    fn compute_rq_pas_single_cpu() {
        let page_offset = DEFAULT_PAGE_OFFSET;
        let runqueues = page_offset.wrapping_add(0x20_0000);
        let pas = compute_rq_pas(runqueues, &[0], page_offset);
        assert_eq!(pas.len(), 1);
        assert_eq!(pas[0], 0x20_0000);
    }

    #[test]
    fn resolve_page_offset_with_symbol() {
        use crate::monitor::reader::GuestMem;

        // Simulate page_offset_base at KVA = START_KERNEL_MAP + 0x1000
        // -> PA = 0x1000
        let pob_kva = START_KERNEL_MAP + 0x1000;
        let expected_page_offset = 0xffff_8880_0000_0000u64;

        let mut buf = [0u8; 0x2000];
        // Write the runtime value at PA 0x1000
        buf[0x1000..0x1008].copy_from_slice(&expected_page_offset.to_ne_bytes());

        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let symbols = KernelSymbols {
            runqueues: 0,
            per_cpu_offset: 0,
            page_offset_base_kva: Some(pob_kva),
            phys_base_kva: None,
            scx_root: None,
            scx_tasks: None,
            init_top_pgt: None,
            pgtable_l5_enabled: None,
            prog_idr: None,
            scx_watchdog_timeout: None,
            scx_watchdog_timestamp: None,
            jiffies_64: None,
            kernel_cpustat: None,
            kstat: None,
            tick_cpu_sched: None,
            node_data: None,
        };

        assert_eq!(
            resolve_page_offset(&mem, &symbols, START_KERNEL_MAP),
            expected_page_offset
        );
    }

    #[test]
    fn resolve_page_offset_without_symbol() {
        use crate::monitor::reader::GuestMem;

        let buf = [0u8; 64];
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let symbols = KernelSymbols {
            runqueues: 0,
            per_cpu_offset: 0,
            page_offset_base_kva: None,
            phys_base_kva: None,
            scx_root: None,
            scx_tasks: None,
            init_top_pgt: None,
            pgtable_l5_enabled: None,
            prog_idr: None,
            scx_watchdog_timeout: None,
            scx_watchdog_timestamp: None,
            jiffies_64: None,
            kernel_cpustat: None,
            kstat: None,
            tick_cpu_sched: None,
            node_data: None,
        };

        assert_eq!(
            resolve_page_offset(&mem, &symbols, START_KERNEL_MAP),
            DEFAULT_PAGE_OFFSET
        );
    }

    #[test]
    fn resolve_page_offset_zero_value_falls_back() {
        use crate::monitor::reader::GuestMem;

        // page_offset_base exists but the guest hasn't written a value yet (all zeros)
        let pob_kva = START_KERNEL_MAP + 0x100;
        let buf = [0u8; 0x200];
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let symbols = KernelSymbols {
            runqueues: 0,
            per_cpu_offset: 0,
            page_offset_base_kva: Some(pob_kva),
            phys_base_kva: None,
            scx_root: None,
            scx_tasks: None,
            init_top_pgt: None,
            pgtable_l5_enabled: None,
            prog_idr: None,
            scx_watchdog_timeout: None,
            scx_watchdog_timestamp: None,
            jiffies_64: None,
            kernel_cpustat: None,
            kstat: None,
            tick_cpu_sched: None,
            node_data: None,
        };

        assert_eq!(
            resolve_page_offset(&mem, &symbols, START_KERNEL_MAP),
            DEFAULT_PAGE_OFFSET
        );
    }

    #[test]
    fn resolve_page_offset_garbage_value_falls_back() {
        use crate::monitor::reader::GuestMem;

        // page_offset_base exists but contains a non-canonical garbage value
        let pob_kva = START_KERNEL_MAP + 0x1000;
        let mut buf = [0u8; 0x2000];
        let garbage: u64 = 0x1234_5678_DEAD_BEEF;
        buf[0x1000..0x1008].copy_from_slice(&garbage.to_ne_bytes());

        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let symbols = KernelSymbols {
            runqueues: 0,
            per_cpu_offset: 0,
            page_offset_base_kva: Some(pob_kva),
            phys_base_kva: None,
            scx_root: None,
            scx_tasks: None,
            init_top_pgt: None,
            pgtable_l5_enabled: None,
            prog_idr: None,
            scx_watchdog_timeout: None,
            scx_watchdog_timestamp: None,
            jiffies_64: None,
            kernel_cpustat: None,
            kstat: None,
            tick_cpu_sched: None,
            node_data: None,
        };

        assert_eq!(
            resolve_page_offset(&mem, &symbols, START_KERNEL_MAP),
            DEFAULT_PAGE_OFFSET
        );
    }

    #[test]
    fn resolve_page_offset_randomized_memory() {
        use crate::monitor::reader::GuestMem;

        // CONFIG_RANDOMIZE_MEMORY produces PAGE_OFFSET values like
        // 0xff11000000000000 that are below the traditional canonical
        // boundary but have bit 63 set.
        let pob_kva = START_KERNEL_MAP + 0x1000;
        let randomized_page_offset = 0xff11_0000_0000_0000u64;

        let mut buf = [0u8; 0x2000];
        buf[0x1000..0x1008].copy_from_slice(&randomized_page_offset.to_ne_bytes());

        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let symbols = KernelSymbols {
            runqueues: 0,
            per_cpu_offset: 0,
            page_offset_base_kva: Some(pob_kva),
            phys_base_kva: None,
            scx_root: None,
            scx_tasks: None,
            init_top_pgt: None,
            pgtable_l5_enabled: None,
            prog_idr: None,
            scx_watchdog_timeout: None,
            scx_watchdog_timestamp: None,
            jiffies_64: None,
            kernel_cpustat: None,
            kstat: None,
            tick_cpu_sched: None,
            node_data: None,
        };

        assert_eq!(
            resolve_page_offset(&mem, &symbols, START_KERNEL_MAP),
            randomized_page_offset
        );
    }

    #[test]
    fn resolve_pgtable_l5_enabled() {
        use crate::monitor::reader::GuestMem;

        let l5_kva = START_KERNEL_MAP + 0x1000;
        let mut buf = [0u8; 0x2000];
        // Write __pgtable_l5_enabled = 1 at PA 0x1000.
        buf[0x1000..0x1004].copy_from_slice(&1u32.to_ne_bytes());

        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let symbols = KernelSymbols {
            runqueues: 0,
            per_cpu_offset: 0,
            page_offset_base_kva: None,
            phys_base_kva: None,
            scx_root: None,
            scx_tasks: None,
            init_top_pgt: None,
            pgtable_l5_enabled: Some(l5_kva),
            prog_idr: None,
            scx_watchdog_timeout: None,
            scx_watchdog_timestamp: None,
            jiffies_64: None,
            kernel_cpustat: None,
            kstat: None,
            tick_cpu_sched: None,
            node_data: None,
        };

        assert!(resolve_pgtable_l5(&mem, &symbols, START_KERNEL_MAP, 0));
    }

    #[test]
    fn resolve_pgtable_l5_disabled() {
        use crate::monitor::reader::GuestMem;

        let l5_kva = START_KERNEL_MAP + 0x1000;
        let mut buf = [0u8; 0x2000];
        // Write __pgtable_l5_enabled = 0 at PA 0x1000.
        buf[0x1000..0x1004].copy_from_slice(&0u32.to_ne_bytes());

        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let symbols = KernelSymbols {
            runqueues: 0,
            per_cpu_offset: 0,
            page_offset_base_kva: None,
            phys_base_kva: None,
            scx_root: None,
            scx_tasks: None,
            init_top_pgt: None,
            pgtable_l5_enabled: Some(l5_kva),
            prog_idr: None,
            scx_watchdog_timeout: None,
            scx_watchdog_timestamp: None,
            jiffies_64: None,
            kernel_cpustat: None,
            kstat: None,
            tick_cpu_sched: None,
            node_data: None,
        };

        assert!(!resolve_pgtable_l5(&mem, &symbols, START_KERNEL_MAP, 0));
    }

    #[test]
    fn resolve_pgtable_l5_absent_symbol() {
        use crate::monitor::reader::GuestMem;

        let buf = [0u8; 64];
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let symbols = KernelSymbols {
            runqueues: 0,
            per_cpu_offset: 0,
            page_offset_base_kva: None,
            phys_base_kva: None,
            scx_root: None,
            scx_tasks: None,
            init_top_pgt: None,
            pgtable_l5_enabled: None,
            prog_idr: None,
            scx_watchdog_timeout: None,
            scx_watchdog_timestamp: None,
            jiffies_64: None,
            kernel_cpustat: None,
            kstat: None,
            tick_cpu_sched: None,
            node_data: None,
        };

        assert!(!resolve_pgtable_l5(&mem, &symbols, START_KERNEL_MAP, 0));
    }

    /// `default_page_offset_for_tcr` derives `-(1 << VA_BITS)` from
    /// `TCR_EL1.T1SZ`. Pin every legitimate VA_BITS encoding plus
    /// the unset / out-of-range fallback so a regression in the
    /// bit math surfaces here. On x86_64 the function is constant
    /// (the register doesn't exist) so all encodings collapse to
    /// [`DEFAULT_PAGE_OFFSET`].
    #[test]
    fn default_page_offset_for_tcr_derives_va_bits() {
        #[cfg(target_arch = "x86_64")]
        {
            // x86_64: tcr_el1 ignored, always returns the constant.
            assert_eq!(default_page_offset_for_tcr(0), DEFAULT_PAGE_OFFSET);
            assert_eq!(default_page_offset_for_tcr(0x12345), DEFAULT_PAGE_OFFSET);
            assert_eq!(default_page_offset_for_tcr(u64::MAX), DEFAULT_PAGE_OFFSET);
        }
        #[cfg(target_arch = "aarch64")]
        {
            // VA_BITS=48 (T1SZ=16): -(1 << 48) = 0xffff_0000_0000_0000.
            let tcr_48 = 16u64 << 16;
            assert_eq!(default_page_offset_for_tcr(tcr_48), 0xffff_0000_0000_0000);
            // VA_BITS=47 (T1SZ=17, 16 KiB granule, Apple Silicon):
            // -(1 << 47) = 0xffff_8000_0000_0000.
            let tcr_47 = 17u64 << 16;
            assert_eq!(default_page_offset_for_tcr(tcr_47), 0xffff_8000_0000_0000);
            // VA_BITS=52 (T1SZ=12): -(1 << 52) = 0xfff0_0000_0000_0000.
            let tcr_52 = 12u64 << 16;
            assert_eq!(default_page_offset_for_tcr(tcr_52), 0xfff0_0000_0000_0000);
            // tcr_el1 == 0 (BSP loop hasn't read register): fallback.
            assert_eq!(default_page_offset_for_tcr(0), DEFAULT_PAGE_OFFSET);
            // T1SZ == 0 (high half disabled): fallback.
            let tcr_t1sz_0 = 0u64;
            assert_eq!(default_page_offset_for_tcr(tcr_t1sz_0), DEFAULT_PAGE_OFFSET);
            // T1SZ > 60 (out of range): fallback.
            let tcr_t1sz_63 = 63u64 << 16;
            assert_eq!(
                default_page_offset_for_tcr(tcr_t1sz_63),
                DEFAULT_PAGE_OFFSET
            );
        }
    }
}
