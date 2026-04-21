//! Host-side kernel memory accessor for a running guest VM.
//!
//! Provides read/write access to kernel variables and structures in
//! guest physical memory. Resolves symbols from the vmlinux ELF,
//! handles address translation (text mapping, direct mapping, vmalloc),
//! and caches paging configuration.
//!
//! Scalar reads and writes use volatile semantics (the guest kernel
//! modifies memory concurrently). Bulk byte reads differ:
//! `read_symbol_bytes` and `read_direct_bytes` delegate to
//! `GuestMem::read_bytes` which uses `copy_nonoverlapping`;
//! `read_kva_bytes` does per-byte volatile reads across page
//! boundaries.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};

use super::reader::GuestMem;
use super::symbols::{kva_to_pa, resolve_page_offset, resolve_pgtable_l5, text_kva_to_pa};

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
}

#[allow(dead_code)]
impl<'a> GuestKernel<'a> {
    /// Create from GuestMem and vmlinux path.
    ///
    /// Parses the ELF symbol table and resolves paging configuration
    /// from guest memory. Requires `init_top_pgt` (or `swapper_pg_dir`)
    /// for page table walks.
    pub fn new(mem: &'a GuestMem, vmlinux: &Path) -> Result<Self> {
        let data = std::fs::read(vmlinux)
            .with_context(|| format!("read vmlinux: {}", vmlinux.display()))?;
        let elf = goblin::elf::Elf::parse(&data).context("parse vmlinux ELF")?;

        let mut symbols = HashMap::new();
        for sym in elf.syms.iter() {
            if let Some(name) = elf.strtab.get_at(sym.st_name)
                && !name.is_empty()
                && sym.st_value != 0
            {
                symbols.insert(name.to_string(), sym.st_value);
            }
        }

        // Resolve paging state using the same logic as KernelSymbols.
        let kern_syms = super::symbols::KernelSymbols::from_vmlinux(vmlinux)?;
        let init_top_pgt_kva = kern_syms
            .init_top_pgt
            .ok_or_else(|| anyhow::anyhow!("init_top_pgt symbol not found in vmlinux"))?;
        let cr3_pa = text_kva_to_pa(init_top_pgt_kva);
        let page_offset = resolve_page_offset(mem, &kern_syms);
        let l5 = resolve_pgtable_l5(mem, &kern_syms);

        Ok(Self {
            mem,
            symbols,
            page_offset,
            cr3_pa,
            l5,
        })
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

    // ---------------------------------------------------------------
    // Text/data/bss symbol reads (statically-linked kernel variables)
    // ---------------------------------------------------------------

    /// Read a u32 from a kernel text/data/bss symbol.
    ///
    /// Translates via `__START_KERNEL_map` (not PAGE_OFFSET).
    pub fn read_symbol_u32(&self, name: &str) -> Result<u32> {
        let kva = self.require_symbol(name)?;
        let pa = text_kva_to_pa(kva);
        Ok(self.mem.read_u32(pa, 0))
    }

    /// Read a u64 from a kernel text/data/bss symbol.
    pub fn read_symbol_u64(&self, name: &str) -> Result<u64> {
        let kva = self.require_symbol(name)?;
        let pa = text_kva_to_pa(kva);
        Ok(self.mem.read_u64(pa, 0))
    }

    /// Read bytes from a kernel text/data/bss symbol.
    pub fn read_symbol_bytes(&self, name: &str, len: usize) -> Result<Vec<u8>> {
        let kva = self.require_symbol(name)?;
        let pa = text_kva_to_pa(kva);
        let mut buf = vec![0u8; len];
        self.mem.read_bytes(pa, &mut buf);
        Ok(buf)
    }

    /// Write a u64 to a kernel text/data/bss symbol.
    pub fn write_symbol_u64(&self, name: &str, val: u64) -> Result<()> {
        let kva = self.require_symbol(name)?;
        let pa = text_kva_to_pa(kva);
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

    /// Read a u32 from a vmalloc'd kernel virtual address.
    ///
    /// Translates via page table walk. Returns `None` if unmapped.
    pub fn read_kva_u32(&self, kva: u64) -> Option<u32> {
        let pa = self.mem.translate_kva(self.cr3_pa, kva, self.l5)?;
        Some(self.mem.read_u32(pa, 0))
    }

    /// Read a u64 from a vmalloc'd kernel virtual address.
    pub fn read_kva_u64(&self, kva: u64) -> Option<u64> {
        let pa = self.mem.translate_kva(self.cr3_pa, kva, self.l5)?;
        Some(self.mem.read_u64(pa, 0))
    }

    /// Read bytes from a vmalloc'd kernel virtual address range.
    ///
    /// Reads byte-by-byte across page boundaries. Returns `None`
    /// if any page is unmapped.
    pub fn read_kva_bytes(&self, kva: u64, len: usize) -> Option<Vec<u8>> {
        let mut buf = vec![0u8; len];
        for (i, byte) in buf.iter_mut().enumerate() {
            let pa = self
                .mem
                .translate_kva(self.cr3_pa, kva + i as u64, self.l5)?;
            *byte = self.mem.read_u8(pa, 0);
        }
        Some(buf)
    }

    /// Write a u8 to a vmalloc'd kernel virtual address.
    /// Returns false if the address is unmapped.
    pub fn write_kva_u8(&self, kva: u64, val: u8) -> bool {
        let Some(pa) = self.mem.translate_kva(self.cr3_pa, kva, self.l5) else {
            return false;
        };
        self.mem.write_u8(pa, 0, val);
        true
    }

    /// Write bytes to a vmalloc'd kernel virtual address range.
    /// Writes byte-by-byte across page boundaries. Returns false
    /// if any page is unmapped.
    pub fn write_kva_bytes(&self, kva: u64, data: &[u8]) -> bool {
        for (i, &byte) in data.iter().enumerate() {
            let Some(pa) = self.mem.translate_kva(self.cr3_pa, kva + i as u64, self.l5) else {
                return false;
            };
            self.mem.write_u8(pa, 0, byte);
        }
        true
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
    // Page table walk tests are in bpf_map.rs.

    #[test]
    fn text_kva_to_pa_and_read() {
        let start_kernel_map: u64 = START_KERNEL_MAP;
        let sym_kva = start_kernel_map + 0x1000;
        let pa = text_kva_to_pa(sym_kva);
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
        // SAFETY: mem outlives kernel because buf is on the stack in this test.
        let mem_ref: &GuestMem = unsafe { &*(&mem as *const GuestMem) };
        let mut symbols = HashMap::new();
        symbols.insert("test_sym".to_string(), 0xFFFF_FFFF_8000_1000u64);
        let kernel = GuestKernel {
            mem: mem_ref,
            symbols,
            page_offset: 0xFFFF_8880_0000_0000,
            cr3_pa: 0,
            l5: false,
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
        let mem_ref: &GuestMem = unsafe { &*(&mem as *const GuestMem) };
        let mut symbols = HashMap::new();
        symbols.insert("my_counter".to_string(), sym_kva);
        let kernel = GuestKernel {
            mem: mem_ref,
            symbols,
            page_offset: 0xFFFF_8880_0000_0000,
            cr3_pa: 0,
            l5: false,
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
        let mem_ref: &GuestMem = unsafe { &*(&mem as *const GuestMem) };
        let mut symbols = HashMap::new();
        symbols.insert("my_u64".to_string(), sym_kva);
        let kernel = GuestKernel {
            mem: mem_ref,
            symbols,
            page_offset: 0xFFFF_8880_0000_0000,
            cr3_pa: 0,
            l5: false,
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
        let mem_ref: &GuestMem = unsafe { &*(&mem as *const GuestMem) };
        let mut symbols = HashMap::new();
        symbols.insert("my_bytes".to_string(), sym_kva);
        let kernel = GuestKernel {
            mem: mem_ref,
            symbols,
            page_offset: 0xFFFF_8880_0000_0000,
            cr3_pa: 0,
            l5: false,
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
        let mem_ref: &GuestMem = unsafe { &*(&mem as *const GuestMem) };
        let kernel = GuestKernel {
            mem: mem_ref,
            symbols: HashMap::new(),
            page_offset: 0xFFFF_8880_0000_0000,
            cr3_pa: 0,
            l5: false,
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
        let mem_ref: &GuestMem = unsafe { &*(&mem as *const GuestMem) };
        let mut symbols = HashMap::new();
        symbols.insert("my_var".to_string(), sym_kva);
        let kernel = GuestKernel {
            mem: mem_ref,
            symbols,
            page_offset: 0xFFFF_8880_0000_0000,
            cr3_pa: 0,
            l5: false,
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
        let mem_ref: &GuestMem = unsafe { &*(&mem as *const GuestMem) };
        let kernel = GuestKernel {
            mem: mem_ref,
            symbols: HashMap::new(),
            page_offset,
            cr3_pa: 0,
            l5: false,
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
        let mem_ref: &GuestMem = unsafe { &*(&mem as *const GuestMem) };
        let kernel = GuestKernel {
            mem: mem_ref,
            symbols: HashMap::new(),
            page_offset: 0x1234,
            cr3_pa: 0x5678,
            l5: true,
        };
        assert_eq!(kernel.page_offset(), 0x1234);
        assert_eq!(kernel.cr3_pa(), 0x5678);
        assert!(kernel.l5());
        assert!(std::ptr::eq(kernel.mem(), mem_ref));
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
        // Allocate a buffer large enough for text_kva_to_pa reads.
        // GuestKernel::new reads page_offset_base and pgtable_l5_enabled
        // from guest memory; a zeroed buffer causes safe fallbacks.
        let mut buf = vec![0u8; 64 << 20];
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = match GuestKernel::new(&mem, &path) {
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
}
