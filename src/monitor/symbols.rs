use anyhow::{Context, Result};
use object::{Object, ObjectSymbol};
use std::path::Path;

/// x86-64 kernel text mapping base (non-KASLR).
/// Used to convert kernel data/bss symbol VAs to physical addresses
/// for the bootstrap read of `page_offset_base`.
const START_KERNEL_MAP: u64 = 0xffff_ffff_8000_0000;

/// Default PAGE_OFFSET for x86-64 4-level paging (non-KASLR).
const DEFAULT_PAGE_OFFSET: u64 = 0xffff_8880_0000_0000;

/// Kernel symbol addresses extracted from vmlinux ELF.
#[derive(Debug, Clone)]
pub struct KernelSymbols {
    /// Kernel virtual address of the `runqueues` per-CPU variable.
    pub runqueues: u64,
    /// Kernel virtual address of the `__per_cpu_offset` array.
    pub per_cpu_offset: u64,
    /// Kernel virtual address of `page_offset_base`. None when the
    /// symbol is absent (non-KASLR kernel without the variable).
    /// The runtime value must be read from guest memory via
    /// `resolve_page_offset`.
    pub page_offset_base_kva: Option<u64>,
    /// Kernel virtual address of `scx_root` (pointer to active scx_sched).
    /// None if the symbol is absent (kernel without sched_ext).
    pub scx_root: Option<u64>,
    /// Kernel virtual address of `scx_watchdog_timeout`.
    /// None if the symbol is absent (kernel without sched_ext).
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
        let elf = object::File::parse(&*data).context("parse vmlinux ELF")?;

        let runqueues = elf
            .symbol_by_name("runqueues")
            .context("symbol 'runqueues' not found in vmlinux")?
            .address();

        let per_cpu_offset = elf
            .symbol_by_name("__per_cpu_offset")
            .context("symbol '__per_cpu_offset' not found in vmlinux")?
            .address();

        let page_offset_base_kva = elf.symbol_by_name("page_offset_base").map(|s| s.address());

        let scx_root = elf.symbol_by_name("scx_root").map(|s| s.address());
        let scx_watchdog_timeout = elf
            .symbol_by_name("scx_watchdog_timeout")
            .map(|s| s.address());

        Ok(Self {
            runqueues,
            per_cpu_offset,
            page_offset_base_kva,
            scx_root,
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
    // Valid PAGE_OFFSET must be in the kernel's upper-half canonical
    // address range. Zero or non-canonical values indicate the guest
    // hasn't initialized the variable yet.
    if val >= 0xffff_8000_0000_0000 {
        val
    } else {
        DEFAULT_PAGE_OFFSET
    }
}

/// Translate a kernel virtual address in the direct mapping
/// (PAGE_OFFSET region) to guest physical address.
pub fn kva_to_pa(kva: u64, page_offset: u64) -> u64 {
    kva.wrapping_sub(page_offset)
}

/// Translate a kernel text/data symbol VA to guest physical address.
///
/// Kernel text and data symbols (.text, .data, .bss) are mapped via
/// `__START_KERNEL_map`, not the direct mapping. Their PA is
/// `VA - __START_KERNEL_map`.
pub fn text_kva_to_pa(kva: u64) -> u64 {
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
pub unsafe fn read_per_cpu_offsets(
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
pub fn compute_rq_pas(runqueues_kva: u64, per_cpu_offsets: &[u64], page_offset: u64) -> Vec<u64> {
    per_cpu_offsets
        .iter()
        .map(|&offset| kva_to_pa(runqueues_kva.wrapping_add(offset), page_offset))
        .collect()
}

/// Write `scx_watchdog_timeout` in guest memory.
///
/// `scx_watchdog_timeout` is a kernel data symbol (static unsigned long),
/// so its PA is derived via `__START_KERNEL_map`, not PAGE_OFFSET.
///
/// Returns `true` if the write succeeded, `false` if the symbol address
/// was absent.
#[allow(dead_code)]
pub(crate) fn write_watchdog_timeout(
    mem: &super::reader::GuestMem,
    symbols: &KernelSymbols,
    val: u64,
) -> bool {
    let Some(kva) = symbols.scx_watchdog_timeout else {
        return false;
    };
    let pa = text_kva_to_pa(kva);
    mem.write_u64(pa, 0, val);
    true
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
            return;
        }
        let syms = KernelSymbols::from_vmlinux(&path).unwrap();
        assert_ne!(syms.runqueues, 0);
        assert_ne!(syms.per_cpu_offset, 0);
        // runqueues should be in kernel VA space
        assert!(syms.runqueues > 0xffff_0000_0000_0000);
    }

    #[test]
    fn kva_to_pa_basic() {
        let page_offset = 0xffff_8880_0000_0000_u64;
        assert_eq!(kva_to_pa(0xffff_8880_0010_0000, page_offset), 0x10_0000);
        assert_eq!(kva_to_pa(page_offset, page_offset), 0);
    }

    #[test]
    fn compute_rq_pas_two_cpus() {
        let runqueues = 0xffff_8880_0020_0000_u64;
        let offsets = vec![0, 0x4_0000]; // CPU 0 at base, CPU 1 at +256KB
        let page_offset = 0xffff_8880_0000_0000_u64;
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
        let page_offset = 0xffff_8880_0000_0000u64;
        let kva = 0x0000_0000_0001_0000u64;
        let pa = kva_to_pa(kva, page_offset);
        assert_eq!(pa, kva.wrapping_sub(page_offset));
    }

    #[test]
    fn compute_rq_pas_empty_offsets() {
        let pas = compute_rq_pas(0xffff_8880_0020_0000, &[], 0xffff_8880_0000_0000);
        assert!(pas.is_empty());
    }

    #[test]
    fn compute_rq_pas_single_cpu() {
        let runqueues = 0xffff_8880_0020_0000u64;
        let page_offset = 0xffff_8880_0000_0000u64;
        let pas = compute_rq_pas(runqueues, &[0], page_offset);
        assert_eq!(pas.len(), 1);
        assert_eq!(pas[0], 0x20_0000);
    }

    #[test]
    fn write_watchdog_timeout_writes_value() {
        use crate::monitor::reader::GuestMem;

        // scx_watchdog_timeout is a kernel data symbol: PA via text mapping.
        let watchdog_kva = START_KERNEL_MAP + 0x1000;
        let symbols = KernelSymbols {
            runqueues: 0,
            per_cpu_offset: 0,
            page_offset_base_kva: None,
            scx_root: None,
            scx_watchdog_timeout: Some(watchdog_kva),
        };

        let mut buf = [0u8; 0x2000];
        let mem = GuestMem::new(buf.as_mut_ptr(), buf.len() as u64);

        assert!(write_watchdog_timeout(&mem, &symbols, 30_000));
        // PA = watchdog_kva - START_KERNEL_MAP = 0x1000
        assert_eq!(mem.read_u64(0x1000, 0), 30_000);
    }

    #[test]
    fn write_watchdog_timeout_returns_false_when_absent() {
        use crate::monitor::reader::GuestMem;

        let symbols = KernelSymbols {
            runqueues: 0,
            per_cpu_offset: 0,
            page_offset_base_kva: None,
            scx_root: None,
            scx_watchdog_timeout: None,
        };

        let buf = [0u8; 64];
        let mem = GuestMem::new(buf.as_ptr(), buf.len() as u64);

        assert!(!write_watchdog_timeout(&mem, &symbols, 30_000));
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
            scx_watchdog_timeout: None,
        };

        assert_eq!(resolve_page_offset(&mem, &symbols), expected_page_offset);
    }

    #[test]
    fn resolve_page_offset_without_symbol() {
        use crate::monitor::reader::GuestMem;

        let buf = [0u8; 64];
        let mem = GuestMem::new(buf.as_ptr(), buf.len() as u64);
        let symbols = KernelSymbols {
            runqueues: 0,
            per_cpu_offset: 0,
            page_offset_base_kva: None,
            scx_root: None,
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
        let mem = GuestMem::new(buf.as_ptr(), buf.len() as u64);
        let symbols = KernelSymbols {
            runqueues: 0,
            per_cpu_offset: 0,
            page_offset_base_kva: Some(pob_kva),
            scx_root: None,
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
            scx_watchdog_timeout: None,
        };

        assert_eq!(resolve_page_offset(&mem, &symbols), DEFAULT_PAGE_OFFSET);
    }
}
