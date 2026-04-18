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
/// x86-64: `__START_KERNEL_map` = 0xffff_ffff_8000_0000.
/// aarch64 48-bit VA: `KIMAGE_VADDR` = _PAGE_END(48) + SZ_2G
///   = 0xffff_8000_8000_0000.
#[cfg(target_arch = "x86_64")]
pub(crate) const START_KERNEL_MAP: u64 = 0xffff_ffff_8000_0000;
#[cfg(target_arch = "aarch64")]
pub(crate) const START_KERNEL_MAP: u64 = 0xffff_8000_8000_0000;

/// Default PAGE_OFFSET (non-KASLR).
///
/// x86-64 4-level paging: 0xffff_8880_0000_0000.
/// aarch64 48-bit VA: -(1 << 48) = 0xffff_0000_0000_0000.
#[cfg(target_arch = "x86_64")]
pub(crate) const DEFAULT_PAGE_OFFSET: u64 = 0xffff_8880_0000_0000;
#[cfg(target_arch = "aarch64")]
pub(crate) const DEFAULT_PAGE_OFFSET: u64 = 0xffff_0000_0000_0000;

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
    /// Kernel virtual address of `scx_root` (pointer to active scx_sched).
    /// None when the symbol is absent: pre-6.16 kernels with sched_ext
    /// (older `scx_ops` API predates `scx_root`), and kernels built
    /// without sched_ext.
    pub scx_root: Option<u64>,
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
        let elf = goblin::elf::Elf::parse(&data).context("parse vmlinux ELF")?;

        let sym_addr = |name: &str| -> Option<u64> {
            elf.syms
                .iter()
                .find(|s| s.st_value != 0 && elf.strtab.get_at(s.st_name) == Some(name))
                .map(|s| s.st_value)
        };

        let runqueues = sym_addr("runqueues").context("symbol 'runqueues' not found in vmlinux")?;

        let per_cpu_offset = sym_addr("__per_cpu_offset")
            .context("symbol '__per_cpu_offset' not found in vmlinux")?;

        let page_offset_base_kva = sym_addr("page_offset_base");

        let scx_root = sym_addr("scx_root");

        let init_top_pgt = sym_addr("init_top_pgt").or_else(|| sym_addr("swapper_pg_dir"));

        let pgtable_l5_enabled = sym_addr("__pgtable_l5_enabled");

        let prog_idr = sym_addr("prog_idr");

        let scx_watchdog_timeout = sym_addr("scx_watchdog_timeout");

        Ok(Self {
            runqueues,
            per_cpu_offset,
            page_offset_base_kva,
            scx_root,
            init_top_pgt,
            pgtable_l5_enabled,
            prog_idr,
            scx_watchdog_timeout,
        })
    }
}

/// Read the runtime value of PAGE_OFFSET from guest memory.
///
/// If the vmlinux contains a `page_offset_base` symbol, converts its
/// KVA to a guest physical address via `__START_KERNEL_map` (the kernel
/// text mapping), then reads the u64 stored there by the guest kernel.
///
/// Falls back to the compile-time default (0xffff888000000000, x86-64
/// 4-level paging) when the symbol is absent.
pub(crate) fn resolve_page_offset(mem: &super::reader::GuestMem, symbols: &KernelSymbols) -> u64 {
    let Some(pob_kva) = symbols.page_offset_base_kva else {
        return DEFAULT_PAGE_OFFSET;
    };
    let pob_pa = text_kva_to_pa(pob_kva);
    let val = mem.read_u64(pob_pa, 0);
    // Valid PAGE_OFFSET has bit 63 set (upper-half virtual address).
    // Kernels with CONFIG_RANDOMIZE_MEMORY use values like
    // 0xff11000000000000 that are below the traditional canonical
    // boundary (0xffff800000000000), so check bit 63 instead.
    if val & (1u64 << 63) != 0 {
        val
    } else {
        DEFAULT_PAGE_OFFSET
    }
}

/// Read the runtime value of `__pgtable_l5_enabled` from guest memory.
///
/// Returns `true` when the guest kernel uses 5-level paging (LA57),
/// `false` when the symbol is absent or the value is 0.
pub(crate) fn resolve_pgtable_l5(mem: &super::reader::GuestMem, symbols: &KernelSymbols) -> bool {
    let Some(kva) = symbols.pgtable_l5_enabled else {
        return false;
    };
    let pa = text_kva_to_pa(kva);
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
/// the direct mapping. The kernel's `__kimg_to_phys(addr)` is
/// `addr - kimage_voffset`, where `kimage_voffset = map_base - phys_base`.
///
/// On x86_64: `phys_base = 0`, so GPA = `VA - __START_KERNEL_map`,
/// and DRAM starts at GPA 0, so DRAM offset = GPA.
/// On aarch64: `phys_base = DRAM_START = 0x4000_0000`, so
/// `kimage_voffset = KIMAGE_VADDR - 0x4000_0000`, and
/// GPA = `VA - KIMAGE_VADDR + 0x4000_0000`. DRAM offset =
/// `GPA - DRAM_START = VA - KIMAGE_VADDR`. The two cancel.
///
/// Both cases require `nokaslr` on the guest cmdline.
pub(crate) fn text_kva_to_pa(kva: u64) -> u64 {
    kva.wrapping_sub(START_KERNEL_MAP)
}

/// Read the `__per_cpu_offset` array from guest memory.
/// Returns per-CPU offsets for each CPU (index = CPU number).
///
/// # Safety
///
/// `host_base` must point to the start of a guest memory region at least
/// `per_cpu_offset_pa + num_cpus * 8` bytes long. The memory at each
/// offset must contain a valid `u64` written by the guest kernel.
pub(crate) unsafe fn read_per_cpu_offsets(
    host_base: *const u8,
    per_cpu_offset_pa: u64,
    num_cpus: u32,
) -> Vec<u64> {
    let mut offsets = Vec::with_capacity(num_cpus as usize);
    for cpu in 0..num_cpus {
        let addr = per_cpu_offset_pa + (cpu as u64) * 8;
        let ptr = unsafe { host_base.add(addr as usize) as *const u64 };
        let val = unsafe { std::ptr::read_volatile(ptr) };
        offsets.push(val);
    }
    offsets
}

/// Compute the physical address of each CPU's `struct rq`.
///
/// Each CPU's rq is at `runqueues_kva + per_cpu_offset[cpu]` in kernel
/// virtual space; subtracting PAGE_OFFSET yields the guest physical address.
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
            eprintln!("ktstr: SKIP: vmlinux is raw BTF (not ELF), cannot parse symbols");
            return;
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
    fn from_vmlinux_nonexistent() {
        let path = std::path::Path::new("/nonexistent/vmlinux");
        assert!(KernelSymbols::from_vmlinux(path).is_err());
    }

    #[test]
    fn read_per_cpu_offsets_zero_cpus() {
        // With num_cpus=0, should return an empty vec without any reads.
        let buf = [0u8; 64];
        let result = unsafe { read_per_cpu_offsets(buf.as_ptr(), 0, 0) };
        assert!(result.is_empty());
    }

    #[test]
    fn read_per_cpu_offsets_known_buffer() {
        // Buffer with 3 known u64 offsets at PA 0.
        let offsets: [u64; 3] = [0x1000, 0x2000, 0x3000];
        let buf: &[u8] = unsafe { std::slice::from_raw_parts(offsets.as_ptr() as *const u8, 24) };
        let result = unsafe { read_per_cpu_offsets(buf.as_ptr(), 0, 3) };
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 0x1000);
        assert_eq!(result[1], 0x2000);
        assert_eq!(result[2], 0x3000);
    }

    #[test]
    fn read_per_cpu_offsets_nonzero_pa() {
        // Place offsets at PA=16 (skip 16 bytes of padding).
        let mut buf = [0u8; 40]; // 16 padding + 3*8 offsets
        let vals: [u64; 3] = [0xAA, 0xBB, 0xCC];
        buf[16..40]
            .copy_from_slice(unsafe { std::slice::from_raw_parts(vals.as_ptr() as *const u8, 24) });
        let result = unsafe { read_per_cpu_offsets(buf.as_ptr(), 16, 3) };
        assert_eq!(result, vec![0xAA, 0xBB, 0xCC]);
    }

    #[test]
    fn text_kva_to_pa_basic() {
        assert_eq!(text_kva_to_pa(START_KERNEL_MAP + 0x10_0000), 0x10_0000);
        assert_eq!(text_kva_to_pa(START_KERNEL_MAP), 0);
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

        let mem = GuestMem::new(buf.as_mut_ptr(), buf.len() as u64);
        let symbols = KernelSymbols {
            runqueues: 0,
            per_cpu_offset: 0,
            page_offset_base_kva: Some(pob_kva),
            scx_root: None,
            init_top_pgt: None,
            pgtable_l5_enabled: None,
            prog_idr: None,
            scx_watchdog_timeout: None,
        };

        assert_eq!(resolve_page_offset(&mem, &symbols), expected_page_offset);
    }

    #[test]
    fn resolve_page_offset_without_symbol() {
        use crate::monitor::reader::GuestMem;

        let buf = [0u8; 64];
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let symbols = KernelSymbols {
            runqueues: 0,
            per_cpu_offset: 0,
            page_offset_base_kva: None,
            scx_root: None,
            init_top_pgt: None,
            pgtable_l5_enabled: None,
            prog_idr: None,
            scx_watchdog_timeout: None,
        };

        assert_eq!(resolve_page_offset(&mem, &symbols), DEFAULT_PAGE_OFFSET);
    }

    #[test]
    fn resolve_page_offset_zero_value_falls_back() {
        use crate::monitor::reader::GuestMem;

        // page_offset_base exists but the guest hasn't written a value yet (all zeros)
        let pob_kva = START_KERNEL_MAP + 0x100;
        let buf = [0u8; 0x200];
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let symbols = KernelSymbols {
            runqueues: 0,
            per_cpu_offset: 0,
            page_offset_base_kva: Some(pob_kva),
            scx_root: None,
            init_top_pgt: None,
            pgtable_l5_enabled: None,
            prog_idr: None,
            scx_watchdog_timeout: None,
        };

        assert_eq!(resolve_page_offset(&mem, &symbols), DEFAULT_PAGE_OFFSET);
    }

    #[test]
    fn resolve_page_offset_garbage_value_falls_back() {
        use crate::monitor::reader::GuestMem;

        // page_offset_base exists but contains a non-canonical garbage value
        let pob_kva = START_KERNEL_MAP + 0x1000;
        let mut buf = [0u8; 0x2000];
        let garbage: u64 = 0x1234_5678_DEAD_BEEF;
        buf[0x1000..0x1008].copy_from_slice(&garbage.to_ne_bytes());

        let mem = GuestMem::new(buf.as_mut_ptr(), buf.len() as u64);
        let symbols = KernelSymbols {
            runqueues: 0,
            per_cpu_offset: 0,
            page_offset_base_kva: Some(pob_kva),
            scx_root: None,
            init_top_pgt: None,
            pgtable_l5_enabled: None,
            prog_idr: None,
            scx_watchdog_timeout: None,
        };

        assert_eq!(resolve_page_offset(&mem, &symbols), DEFAULT_PAGE_OFFSET);
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

        let mem = GuestMem::new(buf.as_mut_ptr(), buf.len() as u64);
        let symbols = KernelSymbols {
            runqueues: 0,
            per_cpu_offset: 0,
            page_offset_base_kva: Some(pob_kva),
            scx_root: None,
            init_top_pgt: None,
            pgtable_l5_enabled: None,
            prog_idr: None,
            scx_watchdog_timeout: None,
        };

        assert_eq!(resolve_page_offset(&mem, &symbols), randomized_page_offset);
    }

    #[test]
    fn resolve_pgtable_l5_enabled() {
        use crate::monitor::reader::GuestMem;

        let l5_kva = START_KERNEL_MAP + 0x1000;
        let mut buf = [0u8; 0x2000];
        // Write __pgtable_l5_enabled = 1 at PA 0x1000.
        buf[0x1000..0x1004].copy_from_slice(&1u32.to_ne_bytes());

        let mem = GuestMem::new(buf.as_mut_ptr(), buf.len() as u64);
        let symbols = KernelSymbols {
            runqueues: 0,
            per_cpu_offset: 0,
            page_offset_base_kva: None,
            scx_root: None,
            init_top_pgt: None,
            pgtable_l5_enabled: Some(l5_kva),
            prog_idr: None,
            scx_watchdog_timeout: None,
        };

        assert!(resolve_pgtable_l5(&mem, &symbols));
    }

    #[test]
    fn resolve_pgtable_l5_disabled() {
        use crate::monitor::reader::GuestMem;

        let l5_kva = START_KERNEL_MAP + 0x1000;
        let mut buf = [0u8; 0x2000];
        // Write __pgtable_l5_enabled = 0 at PA 0x1000.
        buf[0x1000..0x1004].copy_from_slice(&0u32.to_ne_bytes());

        let mem = GuestMem::new(buf.as_mut_ptr(), buf.len() as u64);
        let symbols = KernelSymbols {
            runqueues: 0,
            per_cpu_offset: 0,
            page_offset_base_kva: None,
            scx_root: None,
            init_top_pgt: None,
            pgtable_l5_enabled: Some(l5_kva),
            prog_idr: None,
            scx_watchdog_timeout: None,
        };

        assert!(!resolve_pgtable_l5(&mem, &symbols));
    }

    #[test]
    fn resolve_pgtable_l5_absent_symbol() {
        use crate::monitor::reader::GuestMem;

        let buf = [0u8; 64];
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let symbols = KernelSymbols {
            runqueues: 0,
            per_cpu_offset: 0,
            page_offset_base_kva: None,
            scx_root: None,
            init_top_pgt: None,
            pgtable_l5_enabled: None,
            prog_idr: None,
            scx_watchdog_timeout: None,
        };

        assert!(!resolve_pgtable_l5(&mem, &symbols));
    }
}
