use anyhow::{Context, Result};
use kvm_bindings::{kvm_regs, kvm_sregs};
use kvm_ioctls::VcpuFd;
use vm_memory::{Address, Bytes, GuestAddress, GuestMemoryMmap};

use crate::vmm::kvm::{
    BOOT_PARAMS_ADDR, CMDLINE_ADDR, CMDLINE_MAX, E820_RAM, EBDA_START, HIMEM_START,
    KERNEL_LOAD_ADDR, MMIO_GAP_END, MMIO_GAP_START, STARTUP64_OFFSET,
};

// Page table addresses (identity-mapped, 2MB pages)
// Firecracker/libkrun/CH all use these same addresses.
const PML4_START: u64 = 0x9000;
const PDPTE_START: u64 = 0xa000;
const PDE_START: u64 = 0xb000;

// GDT/IDT — Firecracker layout
const GDT_OFFSET: u64 = 0x500;
const IDT_OFFSET: u64 = 0x520;

// Stack — Firecracker uses 0x8ff0, we match
const BOOT_STACK: u64 = 0x8ff0;

// CR/EFER bits
const CR0_PE: u64 = 0x1;
const CR0_PG: u64 = 0x8000_0000;
const CR4_PAE: u64 = 0x20;
const EFER_LME: u64 = 0x100;
const EFER_LMA: u64 = 0x400;

// MSR indices — Firecracker/libkrun boot MSR set
const MSR_IA32_SYSENTER_CS: u32 = 0x174;
const MSR_IA32_SYSENTER_ESP: u32 = 0x175;
const MSR_IA32_SYSENTER_EIP: u32 = 0x176;
const MSR_STAR: u32 = 0xc000_0081;
const MSR_LSTAR: u32 = 0xc000_0082;
const MSR_CSTAR: u32 = 0xc000_0083;
const MSR_SYSCALL_MASK: u32 = 0xc000_0084;
const MSR_KERNEL_GS_BASE: u32 = 0xc000_0102;
const MSR_IA32_TSC: u32 = 0x10;
const MSR_IA32_MISC_ENABLE: u32 = 0x1a0;
const MSR_IA32_MISC_ENABLE_FAST_STRING: u64 = 0x1;
const MSR_MTRR_DEF_TYPE: u32 = 0x2ff;
const MTRR_ENABLE: u64 = 1 << 11;
const MTRR_MEM_TYPE_WB: u64 = 0x6;

// LAPIC register offsets within kvm_lapic_state.regs[]
const APIC_LVT0: usize = 0x350;
const APIC_LVT1: usize = 0x360;
// Delivery mode values (unshifted) — match Firecracker/libkrun/CH apicdef.h
const APIC_MODE_NMI: u32 = 0x4;
const APIC_MODE_EXTINT: u32 = 0x7;

/// Result of loading a kernel.
pub struct KernelLoadResult {
    /// Entry point address.
    pub entry: u64,
    /// Setup header from the bzImage.
    pub setup_header: Option<linux_loader::loader::bootparam::setup_header>,
}

/// Load a bzImage kernel into guest memory.
/// Returns the entry point and setup header for boot_params construction.
pub fn load_kernel(
    guest_mem: &GuestMemoryMmap,
    kernel_path: &std::path::Path,
) -> Result<KernelLoadResult> {
    use linux_loader::loader::{KernelLoader, bzimage::BzImage};
    use std::fs::File;

    let mut kernel_file = File::open(kernel_path)
        .with_context(|| format!("open kernel: {}", kernel_path.display()))?;

    let result = BzImage::load(
        guest_mem,
        Some(GuestAddress(KERNEL_LOAD_ADDR)),
        &mut kernel_file,
        Some(GuestAddress(0)),
    )
    .context("load bzImage")?;

    let setup_header = result
        .setup_header
        .context("bzImage missing setup_header")?;

    // The 64-bit entry point is at code32_start + startup_64 offset.
    let entry = result.kernel_load.raw_value() + STARTUP64_OFFSET;

    Ok(KernelLoadResult {
        entry,
        setup_header: Some(setup_header),
    })
}

/// Write the kernel command line into guest memory.
pub fn write_cmdline(guest_mem: &GuestMemoryMmap, cmdline: &str) -> Result<()> {
    anyhow::ensure!(
        cmdline.len() < CMDLINE_MAX,
        "cmdline too long ({} > {})",
        cmdline.len(),
        CMDLINE_MAX
    );
    let mut buf = cmdline.as_bytes().to_vec();
    buf.push(0); // null terminator
    guest_mem
        .write_slice(&buf, GuestAddress(CMDLINE_ADDR))
        .context("write cmdline to guest memory")?;
    Ok(())
}

/// Write boot parameters (zero page) into guest memory.
/// Uses the setup_header from the actual bzImage when available.
///
/// When `shm_size > 0`, the last high-memory E820 entry is shortened by
/// `shm_size` bytes so the SHM region at the top of guest physical memory
/// is an E820 gap (no entry covers it).
#[allow(clippy::field_reassign_with_default)]
pub fn write_boot_params(
    guest_mem: &GuestMemoryMmap,
    cmdline: &str,
    memory_mb: u32,
    initrd_addr: Option<u64>,
    initrd_size: Option<u32>,
    hdr: Option<&linux_loader::loader::bootparam::setup_header>,
    shm_size: u64,
) -> Result<()> {
    use linux_loader::loader::bootparam::{boot_e820_entry, boot_params};

    let mut params = boot_params::default();

    // Use the setup_header from the bzImage if available
    if let Some(h) = hdr {
        params.hdr = *h;
    }
    // Override fields we control regardless of source
    params.hdr.type_of_loader = 0xff;
    params.hdr.boot_flag = 0xaa55;
    params.hdr.header = 0x5372_6448; // "HdrS"
    params.hdr.cmd_line_ptr = CMDLINE_ADDR as u32;
    params.hdr.cmdline_size = cmdline.len() as u32;
    params.hdr.kernel_alignment = 0x100_0000;

    // Initrd
    if let (Some(addr), Some(size)) = (initrd_addr, initrd_size) {
        params.hdr.ramdisk_image = addr as u32;
        params.hdr.ramdisk_size = size;
    }

    // ACPI RSDP address — tell the kernel where to find it
    params.acpi_rsdp_addr = 0x000E_0000;

    // E820 memory map — Firecracker pattern:
    // Entry 0: low memory (0 to EBDA)
    // Entry 1+: high memory (1MB to end, split at MMIO gap if needed)
    //
    // When shm_size > 0, the SHM region occupies the top of physical memory
    // and must not appear in E820. The last high-memory entry is reduced.
    let mem_size = (memory_mb as u64) << 20;
    let usable_size = mem_size - shm_size;

    let mut e820_idx = 0;

    // Low memory: 0 to EBDA (640K - 1K)
    params.e820_table[e820_idx] = boot_e820_entry {
        addr: 0,
        size: EBDA_START,
        type_: E820_RAM,
    };
    e820_idx += 1;

    // High memory: 1MB onwards.
    //
    // Three cases:
    //   usable_size <= MMIO_GAP_START (3GB): fits below gap
    //   usable_size <= MMIO_GAP_END (4GB):   extends into gap, memory in gap is lost
    //   usable_size >  MMIO_GAP_END:         extends beyond gap, split into two entries
    if usable_size <= MMIO_GAP_START {
        params.e820_table[e820_idx] = boot_e820_entry {
            addr: HIMEM_START,
            size: usable_size - HIMEM_START,
            type_: E820_RAM,
        };
        e820_idx += 1;
    } else {
        // Below MMIO gap
        params.e820_table[e820_idx] = boot_e820_entry {
            addr: HIMEM_START,
            size: MMIO_GAP_START - HIMEM_START,
            type_: E820_RAM,
        };
        e820_idx += 1;
        // Above MMIO gap (only if memory extends beyond it)
        if usable_size > MMIO_GAP_END {
            params.e820_table[e820_idx] = boot_e820_entry {
                addr: MMIO_GAP_END,
                size: usable_size - MMIO_GAP_END,
                type_: E820_RAM,
            };
            e820_idx += 1;
        }
    }
    params.e820_entries = e820_idx as u8;

    // Write to guest memory
    guest_mem
        .write_obj(params, GuestAddress(BOOT_PARAMS_ADDR))
        .context("write boot_params")?;

    Ok(())
}

/// Set up identity-mapped page tables for 64-bit long mode.
/// Maps the first 1GB with 2MB pages.
/// All 4 reference implementations use this same layout.
pub fn setup_page_tables(guest_mem: &GuestMemoryMmap) -> Result<()> {
    // PML4[0] -> PDPTE
    guest_mem
        .write_obj(PDPTE_START | 0x03u64, GuestAddress(PML4_START))
        .context("write PML4")?;

    // PDPTE[0] -> PDE
    guest_mem
        .write_obj(PDE_START | 0x03u64, GuestAddress(PDPTE_START))
        .context("write PDPTE")?;

    // 512 × 2MB pages covering [0, 1GB)
    for i in 0u64..512 {
        guest_mem
            .write_obj((i << 21) | 0x83u64, GuestAddress(PDE_START + i * 8))
            .context("write PDE")?;
    }

    Ok(())
}

/// GDT entry helper: pack a segment descriptor.
/// Matches Firecracker gdt.rs:gdt_entry exactly.
fn gdt_entry(flags: u16, base: u32, limit: u32) -> u64 {
    ((u64::from(base) & 0xff00_0000u64) << (56 - 24))
        | ((u64::from(flags) & 0x0000_f0ffu64) << 40)
        | ((u64::from(limit) & 0x000f_0000u64) << (48 - 16))
        | ((u64::from(base) & 0x00ff_ffffu64) << 16)
        | (u64::from(limit) & 0x0000_ffffu64)
}

/// Convert a GDT entry to a kvm_segment.
/// Matches Firecracker gdt.rs:kvm_segment_from_gdt — handles G-flag limit
/// scaling and sets the unusable field.
fn kvm_segment_from_gdt(entry: u64, table_index: u8) -> kvm_bindings::kvm_segment {
    let g = ((entry >> 55) & 0x1) as u8;
    let present = ((entry >> 47) & 0x1) as u8;
    let raw_limit = ((entry & 0xffff) | (((entry >> 48) & 0xf) << 16)) as u32;
    // When G=1 (4K granularity), scale the 20-bit limit to a full 32-bit value.
    let limit = match g {
        0 => raw_limit,
        _ => (raw_limit << 12) | 0xFFF,
    };
    kvm_bindings::kvm_segment {
        base: ((entry >> 16) & 0xff_ffff) | (((entry >> 56) & 0xff) << 24),
        limit,
        selector: u16::from(table_index) << 3,
        type_: ((entry >> 40) & 0xf) as u8,
        present,
        dpl: ((entry >> 45) & 0x3) as u8,
        db: ((entry >> 54) & 0x1) as u8,
        s: ((entry >> 44) & 0x1) as u8,
        l: ((entry >> 53) & 0x1) as u8,
        g,
        avl: ((entry >> 52) & 0x1) as u8,
        padding: 0,
        unusable: match present {
            0 => 1,
            _ => 0,
        },
    }
}

// x2APIC enable bit in IA32_APIC_BASE MSR (bit 10).
const X2APIC_ENABLE_BIT: u64 = 1 << 10;

/// Set up GDT, segment registers, and page tables for 64-bit long mode.
/// GDT layout matches Firecracker/libkrun exactly:
///   \[0\] NULL, \[1\] CODE (0xa09b), \[2\] DATA (0xc093), \[3\] TSS (0x808b)
///
/// When `x2apic` is true, sets the x2APIC enable bit in IA32_APIC_BASE
/// so the LAPIC operates in x2APIC mode (required for APIC IDs > 254).
pub fn setup_sregs(guest_mem: &GuestMemoryMmap, vcpu: &VcpuFd, x2apic: bool) -> Result<()> {
    let gdt_table: [u64; 4] = [
        gdt_entry(0, 0, 0),            // NULL
        gdt_entry(0xa09b, 0, 0xfffff), // CODE — 64-bit, present, DPL0, L=1
        gdt_entry(0xc093, 0, 0xfffff), // DATA — present, DPL0, G=1, DB=1
        gdt_entry(0x808b, 0, 0xfffff), // TSS
    ];

    for (i, &entry) in gdt_table.iter().enumerate() {
        guest_mem
            .write_obj(entry, GuestAddress(GDT_OFFSET + (i as u64) * 8))
            .context("write GDT entry")?;
    }

    // IDT (empty)
    guest_mem
        .write_obj(0u64, GuestAddress(IDT_OFFSET))
        .context("write IDT")?;

    setup_page_tables(guest_mem)?;

    let mut sregs: kvm_sregs = vcpu.get_sregs().context("get sregs")?;

    let code_seg = kvm_segment_from_gdt(gdt_table[1], 1);
    let data_seg = kvm_segment_from_gdt(gdt_table[2], 2);
    let tss_seg = kvm_segment_from_gdt(gdt_table[3], 3);

    sregs.gdt.base = GDT_OFFSET;
    sregs.gdt.limit = (std::mem::size_of_val(&gdt_table) as u16) - 1;
    sregs.idt.base = IDT_OFFSET;
    sregs.idt.limit = (std::mem::size_of::<u64>() as u16) - 1;

    sregs.cs = code_seg;
    sregs.ds = data_seg;
    sregs.es = data_seg;
    sregs.fs = data_seg;
    sregs.gs = data_seg;
    sregs.ss = data_seg;
    sregs.tr = tss_seg;

    // 64-bit long mode — set explicitly, not OR'd with host bits.
    // Firecracker: cr0 |= PE, cr0 |= PG (but starts from get_sregs).
    // We set exactly what's needed — no host NW/CD bits leaking through.
    sregs.cr0 = CR0_PE | CR0_PG;
    sregs.cr4 = CR4_PAE;
    sregs.efer = EFER_LME | EFER_LMA;
    sregs.cr3 = PML4_START;

    if x2apic {
        sregs.apic_base |= X2APIC_ENABLE_BIT;
    }

    vcpu.set_sregs(&sregs).context("set sregs")?;
    Ok(())
}

/// Set up general-purpose registers for the BSP.
/// Matches Firecracker/libkrun: RIP=entry, RSP=RBP=0x8ff0, RSI=boot_params,
/// RFLAGS=0x2.
pub fn setup_regs(vcpu: &VcpuFd, boot_ip: u64) -> Result<()> {
    let regs = kvm_regs {
        rflags: 0x2, // bit 1 always set
        rip: boot_ip,
        rsp: BOOT_STACK,
        rbp: BOOT_STACK,
        rsi: BOOT_PARAMS_ADDR, // Linux ABI: RSI points to boot_params
        ..Default::default()
    };
    vcpu.set_regs(&regs).context("set regs")?;
    Ok(())
}

/// Set up FPU — all 4 reference implementations use fcw=0x37f, mxcsr=0x1f80.
pub fn setup_fpu(vcpu: &VcpuFd) -> Result<()> {
    let fpu = kvm_bindings::kvm_fpu {
        fcw: 0x37f,
        mxcsr: 0x1f80,
        ..Default::default()
    };
    vcpu.set_fpu(&fpu).context("set fpu")?;
    Ok(())
}

/// Default boot MSR entries — Firecracker/libkrun pattern.
/// These MSRs are required for correct kernel boot and syscall infrastructure.
fn default_msr_entries() -> Vec<kvm_bindings::kvm_msr_entry> {
    vec![
        kvm_bindings::kvm_msr_entry {
            index: MSR_IA32_SYSENTER_CS,
            data: 0,
            ..Default::default()
        },
        kvm_bindings::kvm_msr_entry {
            index: MSR_IA32_SYSENTER_ESP,
            data: 0,
            ..Default::default()
        },
        kvm_bindings::kvm_msr_entry {
            index: MSR_IA32_SYSENTER_EIP,
            data: 0,
            ..Default::default()
        },
        kvm_bindings::kvm_msr_entry {
            index: MSR_STAR,
            data: 0,
            ..Default::default()
        },
        kvm_bindings::kvm_msr_entry {
            index: MSR_CSTAR,
            data: 0,
            ..Default::default()
        },
        kvm_bindings::kvm_msr_entry {
            index: MSR_KERNEL_GS_BASE,
            data: 0,
            ..Default::default()
        },
        kvm_bindings::kvm_msr_entry {
            index: MSR_SYSCALL_MASK,
            data: 0,
            ..Default::default()
        },
        kvm_bindings::kvm_msr_entry {
            index: MSR_LSTAR,
            data: 0,
            ..Default::default()
        },
        kvm_bindings::kvm_msr_entry {
            index: MSR_IA32_TSC,
            data: 0,
            ..Default::default()
        },
        kvm_bindings::kvm_msr_entry {
            index: MSR_IA32_MISC_ENABLE,
            data: MSR_IA32_MISC_ENABLE_FAST_STRING,
            ..Default::default()
        },
        kvm_bindings::kvm_msr_entry {
            index: MSR_MTRR_DEF_TYPE,
            data: MTRR_ENABLE | MTRR_MEM_TYPE_WB,
            ..Default::default()
        },
    ]
}

/// Set boot MSRs with optional extra entries.
/// Extra entries override defaults when they share the same MSR index;
/// entries with new indices are appended.
pub fn setup_msrs(vcpu: &VcpuFd, extra: Option<&[kvm_bindings::kvm_msr_entry]>) -> Result<()> {
    let mut entries = default_msr_entries();

    if let Some(extras) = extra {
        for extra_entry in extras {
            if let Some(existing) = entries.iter_mut().find(|e| e.index == extra_entry.index) {
                existing.data = extra_entry.data;
            } else {
                entries.push(*extra_entry);
            }
        }
    }

    let msrs = kvm_bindings::Msrs::from_entries(&entries).context("create MSR entries")?;
    let written = vcpu.set_msrs(&msrs).context("set boot MSRs")?;
    anyhow::ensure!(
        written == entries.len(),
        "set_msrs: wrote {written}/{} MSRs",
        entries.len()
    );
    Ok(())
}

/// Read a 32-bit LAPIC register from the kvm_lapic_state regs array.
fn get_klapic_reg(lapic: &kvm_bindings::kvm_lapic_state, reg_offset: usize) -> u32 {
    u32::from_le_bytes([
        lapic.regs[reg_offset] as u8,
        lapic.regs[reg_offset + 1] as u8,
        lapic.regs[reg_offset + 2] as u8,
        lapic.regs[reg_offset + 3] as u8,
    ])
}

/// Write a 32-bit LAPIC register into the kvm_lapic_state regs array.
fn set_klapic_reg(lapic: &mut kvm_bindings::kvm_lapic_state, reg_offset: usize, value: u32) {
    let bytes = value.to_le_bytes();
    lapic.regs[reg_offset] = bytes[0] as i8;
    lapic.regs[reg_offset + 1] = bytes[1] as i8;
    lapic.regs[reg_offset + 2] = bytes[2] as i8;
    lapic.regs[reg_offset + 3] = bytes[3] as i8;
}

/// Set the delivery mode in an LVT register value, preserving other bits.
/// Matches Firecracker/libkrun/CH: `(reg & !0x700) | (mode << 8)`.
fn set_apic_delivery_mode(reg: u32, mode: u32) -> u32 {
    (reg & !0x700) | (mode << 8)
}

/// LAPIC LVT mask bit (bit 16) — inhibits interrupt delivery.
const APIC_LVT_MASKED: u32 = 1 << 16;

/// Set LAPIC LVT0 and LVT1.
///
/// BSP: LVT0 = ExtINT (PIC pass-through), LVT1 = NMI.
/// APs: both LVTs masked — matches KVM's kvm_lapic_reset behavior where
/// all LVTs start masked and only the BSP's LVT0 is unmasked (via the
/// KVM_X86_QUIRK_LINT0_REENABLED quirk).
pub fn setup_lapic(vcpu: &VcpuFd, is_bsp: bool) -> Result<()> {
    let mut lapic = vcpu.get_lapic().context("get lapic")?;

    let lvt0 = get_klapic_reg(&lapic, APIC_LVT0);
    let lvt1 = get_klapic_reg(&lapic, APIC_LVT1);

    if is_bsp {
        set_klapic_reg(
            &mut lapic,
            APIC_LVT0,
            set_apic_delivery_mode(lvt0, APIC_MODE_EXTINT),
        );
        set_klapic_reg(
            &mut lapic,
            APIC_LVT1,
            set_apic_delivery_mode(lvt1, APIC_MODE_NMI),
        );
    } else {
        set_klapic_reg(&mut lapic, APIC_LVT0, lvt0 | APIC_LVT_MASKED);
        set_klapic_reg(&mut lapic, APIC_LVT1, lvt1 | APIC_LVT_MASKED);
    }

    vcpu.set_lapic(&lapic).context("set lapic")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_mem(mb: u32) -> GuestMemoryMmap {
        GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), (mb as usize) << 20)]).unwrap()
    }

    #[test]
    fn gdt_entry_null() {
        assert_eq!(gdt_entry(0, 0, 0), 0);
    }

    #[test]
    fn gdt_entry_code_segment() {
        let e = gdt_entry(0xa09b, 0, 0xfffff);
        assert_ne!(e, 0);
        let seg = kvm_segment_from_gdt(e, 1);
        assert_eq!(seg.selector, 8); // index 1 << 3
        assert_eq!(seg.present, 1);
        assert_eq!(seg.l, 1); // 64-bit
        assert_eq!(seg.g, 1); // granularity
        assert_eq!(seg.limit, 0xffff_ffff); // G=1 scales 0xfffff to 4GB
        assert_eq!(seg.unusable, 0);
    }

    #[test]
    fn gdt_entry_data_segment() {
        let e = gdt_entry(0xc093, 0, 0xfffff);
        let seg = kvm_segment_from_gdt(e, 2);
        assert_eq!(seg.selector, 16);
        assert_eq!(seg.present, 1);
        assert_eq!(seg.s, 1); // non-system
        assert_eq!(seg.g, 1);
        assert_eq!(seg.limit, 0xffff_ffff); // G=1 scales 0xfffff to 4GB
    }

    #[test]
    fn write_cmdline_basic() {
        let mem = test_mem(16);
        write_cmdline(&mem, "console=ttyS0").unwrap();
        let mut buf = vec![0u8; 14];
        mem.read_slice(&mut buf, GuestAddress(CMDLINE_ADDR))
            .unwrap();
        assert_eq!(&buf[..13], b"console=ttyS0");
        assert_eq!(buf[13], 0); // null terminated
    }

    #[test]
    fn write_cmdline_too_long() {
        let mem = test_mem(16);
        let long = "x".repeat(CMDLINE_MAX + 1);
        assert!(write_cmdline(&mem, &long).is_err());
    }

    #[test]
    fn setup_page_tables_writes() {
        let mem = test_mem(16);
        setup_page_tables(&mem).unwrap();
        let pml4: u64 = mem.read_obj(GuestAddress(PML4_START)).unwrap();
        assert_eq!(pml4 & !0xfff, PDPTE_START);
        assert_eq!(pml4 & 0x3, 0x3);
        let pdpte: u64 = mem.read_obj(GuestAddress(PDPTE_START)).unwrap();
        assert_eq!(pdpte & !0xfff, PDE_START);
        let pde0: u64 = mem.read_obj(GuestAddress(PDE_START)).unwrap();
        assert_eq!(pde0 & !0xfff, 0);
        assert_eq!(pde0 & 0x83, 0x83);
    }

    #[test]
    fn write_boot_params_basic() {
        let mem = test_mem(16);
        write_boot_params(&mem, "console=ttyS0", 16, None, None, None, 0).unwrap();
        use linux_loader::loader::bootparam::boot_params;
        let params: boot_params = mem.read_obj(GuestAddress(BOOT_PARAMS_ADDR)).unwrap();
        let header = { params.hdr.header };
        let cmd_line_ptr = { params.hdr.cmd_line_ptr };
        let e820_entries = { params.e820_entries };
        assert_eq!(header, 0x5372_6448);
        assert_eq!(cmd_line_ptr, CMDLINE_ADDR as u32);
        assert_eq!(e820_entries, 2);
    }

    #[test]
    fn write_boot_params_with_initrd() {
        let mem = test_mem(16);
        write_boot_params(
            &mem,
            "console=ttyS0",
            16,
            Some(0x200000),
            Some(4096),
            None,
            0,
        )
        .unwrap();
        use linux_loader::loader::bootparam::boot_params;
        let params: boot_params = mem.read_obj(GuestAddress(BOOT_PARAMS_ADDR)).unwrap();
        let ramdisk_image = { params.hdr.ramdisk_image };
        let ramdisk_size = { params.hdr.ramdisk_size };
        assert_eq!(ramdisk_image, 0x200000);
        assert_eq!(ramdisk_size, 4096);
    }

    #[test]
    fn setup_sregs_with_kvm() {
        use crate::vmm::kvm::KtstrKvm;
        use crate::vmm::topology::Topology;
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let vm = KtstrKvm::new(topo, 64, false).unwrap();
        setup_sregs(&vm.guest_mem, &vm.vcpus[0], false).unwrap();
        let sregs = vm.vcpus[0].get_sregs().unwrap();
        assert_eq!(sregs.cr3, PML4_START);
        assert_ne!(sregs.cr0 & CR0_PE, 0);
        assert_ne!(sregs.cr0 & CR0_PG, 0);
        assert_ne!(sregs.efer & EFER_LMA, 0);
    }

    #[test]
    fn setup_regs_with_kvm() {
        use crate::vmm::kvm::KtstrKvm;
        use crate::vmm::topology::Topology;
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let vm = KtstrKvm::new(topo, 64, false).unwrap();
        setup_sregs(&vm.guest_mem, &vm.vcpus[0], false).unwrap();
        setup_regs(&vm.vcpus[0], KERNEL_LOAD_ADDR).unwrap();
        let regs = vm.vcpus[0].get_regs().unwrap();
        assert_eq!(regs.rip, KERNEL_LOAD_ADDR);
        assert_eq!(regs.rsi, BOOT_PARAMS_ADDR);
        assert_eq!(regs.rsp, BOOT_STACK);
    }

    #[test]
    fn setup_fpu_with_kvm() {
        use crate::vmm::kvm::KtstrKvm;
        use crate::vmm::topology::Topology;
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let vm = KtstrKvm::new(topo, 64, false).unwrap();
        setup_fpu(&vm.vcpus[0]).unwrap();
        let fpu = vm.vcpus[0].get_fpu().unwrap();
        assert_eq!(fpu.fcw, 0x37f);
    }

    #[test]
    fn setup_msrs_with_kvm() {
        use crate::vmm::kvm::KtstrKvm;
        use crate::vmm::topology::Topology;
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let vm = KtstrKvm::new(topo, 64, false).unwrap();
        setup_msrs(&vm.vcpus[0], None).unwrap();
        // Verify MISC_ENABLE was set
        let mut msrs = kvm_bindings::Msrs::from_entries(&[kvm_bindings::kvm_msr_entry {
            index: MSR_IA32_MISC_ENABLE,
            ..Default::default()
        }])
        .unwrap();
        let read = vm.vcpus[0].get_msrs(&mut msrs).unwrap();
        assert_eq!(read, 1);
        let data = msrs.as_slice()[0].data;
        assert_ne!(
            data & MSR_IA32_MISC_ENABLE_FAST_STRING,
            0,
            "FAST_STRING should be set"
        );
    }

    #[test]
    fn setup_msrs_with_extra_override() {
        use crate::vmm::kvm::KtstrKvm;
        use crate::vmm::topology::Topology;
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let vm = KtstrKvm::new(topo, 64, false).unwrap();
        // Override MISC_ENABLE to disable FAST_STRING
        let extra = [kvm_bindings::kvm_msr_entry {
            index: MSR_IA32_MISC_ENABLE,
            data: 0,
            ..Default::default()
        }];
        setup_msrs(&vm.vcpus[0], Some(&extra)).unwrap();
        let mut msrs = kvm_bindings::Msrs::from_entries(&[kvm_bindings::kvm_msr_entry {
            index: MSR_IA32_MISC_ENABLE,
            ..Default::default()
        }])
        .unwrap();
        let read = vm.vcpus[0].get_msrs(&mut msrs).unwrap();
        assert_eq!(read, 1);
        assert_eq!(
            msrs.as_slice()[0].data,
            0,
            "MISC_ENABLE should be overridden to 0"
        );
    }

    #[test]
    fn setup_msrs_with_extra_append() {
        use crate::vmm::kvm::KtstrKvm;
        use crate::vmm::topology::Topology;
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let vm = KtstrKvm::new(topo, 64, false).unwrap();
        // Append a new MSR (IA32_EFER = 0xC0000080)
        let extra = [kvm_bindings::kvm_msr_entry {
            index: 0xC000_0080,
            data: 0,
            ..Default::default()
        }];
        // Should succeed without error — the extra MSR is appended
        setup_msrs(&vm.vcpus[0], Some(&extra)).unwrap();
    }

    #[test]
    fn default_msr_entries_count() {
        let entries = default_msr_entries();
        assert_eq!(entries.len(), 11, "should have 11 default MSR entries");
    }

    #[test]
    fn setup_lapic_bsp() {
        use crate::vmm::kvm::KtstrKvm;
        use crate::vmm::topology::Topology;
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let vm = KtstrKvm::new(topo, 64, false).unwrap();
        setup_lapic(&vm.vcpus[0], true).unwrap();
        let lapic = vm.vcpus[0].get_lapic().unwrap();
        let lvt0 = get_klapic_reg(&lapic, APIC_LVT0);
        assert_eq!(
            lvt0 & 0x700,
            APIC_MODE_EXTINT << 8,
            "BSP LVT0 should be ExtINT mode"
        );
        let lvt1 = get_klapic_reg(&lapic, APIC_LVT1);
        assert_eq!(
            lvt1 & 0x700,
            APIC_MODE_NMI << 8,
            "BSP LVT1 should be NMI mode"
        );
    }

    #[test]
    fn setup_lapic_ap_masked() {
        use crate::vmm::kvm::KtstrKvm;
        use crate::vmm::topology::Topology;
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let vm = KtstrKvm::new(topo, 64, false).unwrap();
        setup_lapic(&vm.vcpus[1], false).unwrap();
        let lapic = vm.vcpus[1].get_lapic().unwrap();
        let lvt0 = get_klapic_reg(&lapic, APIC_LVT0);
        assert_ne!(lvt0 & APIC_LVT_MASKED, 0, "AP LVT0 should be masked");
        let lvt1 = get_klapic_reg(&lapic, APIC_LVT1);
        assert_ne!(lvt1 & APIC_LVT_MASKED, 0, "AP LVT1 should be masked");
    }

    #[test]
    fn e820_small_mem_two_entries() {
        let mem = test_mem(512);
        write_boot_params(&mem, "console=ttyS0", 512, None, None, None, 0).unwrap();
        use linux_loader::loader::bootparam::boot_params;
        let params: boot_params = mem.read_obj(GuestAddress(BOOT_PARAMS_ADDR)).unwrap();
        let entries = { params.e820_entries };
        assert_eq!(entries, 2, "should have low + high memory entries");
        let addr0 = { params.e820_table[0].addr };
        let size0 = { params.e820_table[0].size };
        assert_eq!(addr0, 0);
        assert_eq!(size0, 0x9FC00);
        let addr1 = { params.e820_table[1].addr };
        let size1 = { params.e820_table[1].size };
        assert_eq!(addr1, 0x10_0000);
        assert_eq!(size1, (512 << 20) - 0x10_0000);
    }

    #[test]
    fn e820_large_mem_splits_around_mmio_gap() {
        // 5 GB: memory extends beyond the MMIO gap.
        let mem = test_mem(5120);
        write_boot_params(&mem, "console=ttyS0", 5120, None, None, None, 0).unwrap();
        use linux_loader::loader::bootparam::boot_params;
        let params: boot_params = mem.read_obj(GuestAddress(BOOT_PARAMS_ADDR)).unwrap();
        let entries = { params.e820_entries };
        assert_eq!(entries, 3, "5GB: low + high-below-gap + high-above-gap");
        let addr0 = { params.e820_table[0].addr };
        assert_eq!(addr0, 0);
        let addr1 = { params.e820_table[1].addr };
        let size1 = { params.e820_table[1].size };
        assert_eq!(addr1, HIMEM_START);
        assert_eq!(size1, MMIO_GAP_START - HIMEM_START);
        let addr2 = { params.e820_table[2].addr };
        let size2 = { params.e820_table[2].size };
        assert_eq!(addr2, MMIO_GAP_END);
        assert_eq!(size2, (5120u64 << 20) - MMIO_GAP_END);
    }

    #[test]
    fn e820_4gb_no_above_gap_entry() {
        // 4 GB exactly equals MMIO_GAP_END — no memory above the gap.
        let mem = test_mem(4096);
        write_boot_params(&mem, "console=ttyS0", 4096, None, None, None, 0).unwrap();
        use linux_loader::loader::bootparam::boot_params;
        let params: boot_params = mem.read_obj(GuestAddress(BOOT_PARAMS_ADDR)).unwrap();
        let entries = { params.e820_entries };
        assert_eq!(entries, 2, "4GB: low + high-below-gap (no above-gap)");
        let size1 = { params.e820_table[1].size };
        assert_eq!(size1, MMIO_GAP_START - HIMEM_START);
    }

    #[test]
    fn e820_exact_3gb_two_entries() {
        let mem = test_mem(3072);
        write_boot_params(&mem, "console=ttyS0", 3072, None, None, None, 0).unwrap();
        use linux_loader::loader::bootparam::boot_params;
        let params: boot_params = mem.read_obj(GuestAddress(BOOT_PARAMS_ADDR)).unwrap();
        let entries = { params.e820_entries };
        assert_eq!(entries, 2, "3GB: low + high (no MMIO split needed)");
    }

    #[test]
    fn cr0_no_host_bits() {
        use crate::vmm::kvm::KtstrKvm;
        use crate::vmm::topology::Topology;
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let vm = KtstrKvm::new(topo, 64, false).unwrap();
        setup_sregs(&vm.guest_mem, &vm.vcpus[0], false).unwrap();
        let sregs = vm.vcpus[0].get_sregs().unwrap();
        // CR0 should be exactly PE|PG — no NW/CD bits from host
        assert_eq!(
            sregs.cr0,
            CR0_PE | CR0_PG,
            "CR0 should be exactly PE|PG, got {:#x}",
            sregs.cr0
        );
    }

    #[test]
    fn boot_stack_matches_firecracker() {
        assert_eq!(
            BOOT_STACK, 0x8ff0,
            "stack should match Firecracker's 0x8ff0"
        );
    }

    #[test]
    fn setup_sregs_x2apic_disabled() {
        use crate::vmm::kvm::KtstrKvm;
        use crate::vmm::topology::Topology;
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let vm = KtstrKvm::new(topo, 64, false).unwrap();
        setup_sregs(&vm.guest_mem, &vm.vcpus[0], false).unwrap();
        let sregs = vm.vcpus[0].get_sregs().unwrap();
        assert_eq!(
            sregs.apic_base & X2APIC_ENABLE_BIT,
            0,
            "x2APIC bit should not be set when x2apic=false"
        );
    }

    #[test]
    fn setup_sregs_x2apic_enabled() {
        use crate::vmm::kvm::KtstrKvm;
        use crate::vmm::topology::Topology;
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let vm = KtstrKvm::new(topo, 64, false).unwrap();
        setup_sregs(&vm.guest_mem, &vm.vcpus[0], true).unwrap();
        let sregs = vm.vcpus[0].get_sregs().unwrap();
        assert_ne!(
            sregs.apic_base & X2APIC_ENABLE_BIT,
            0,
            "x2APIC bit should be set when x2apic=true"
        );
    }

    // -- boundary condition tests --

    #[test]
    fn write_cmdline_at_max_minus_one() {
        let mem = test_mem(16);
        let cmdline = "x".repeat(CMDLINE_MAX - 1);
        assert!(write_cmdline(&mem, &cmdline).is_ok());
    }

    #[test]
    fn write_cmdline_at_max_fails() {
        let mem = test_mem(16);
        let cmdline = "x".repeat(CMDLINE_MAX);
        assert!(write_cmdline(&mem, &cmdline).is_err());
    }

    #[test]
    fn write_cmdline_empty() {
        let mem = test_mem(16);
        assert!(write_cmdline(&mem, "").is_ok());
    }

    #[test]
    fn e820_at_mmio_boundary() {
        // Memory exactly at MMIO gap start (3 GB) — should produce 2 entries
        let mem = test_mem(3072); // 3 GB
        let params = write_boot_params(&mem, "console=ttyS0", 3072, None, None, None, 0);
        assert!(params.is_ok());
    }

    #[test]
    fn e820_above_mmio_gap() {
        // Memory above MMIO gap (5 GB) — should produce 3 entries
        let mem = test_mem(5120); // 5 GB
        let params = write_boot_params(&mem, "console=ttyS0", 5120, None, None, None, 0);
        assert!(params.is_ok());
    }

    // -- SHM region E820 gap tests --

    #[test]
    fn e820_shm_reduces_high_memory() {
        let mem = test_mem(512);
        let shm_size: u64 = 64 * 1024; // 64 KB
        write_boot_params(&mem, "console=ttyS0", 512, None, None, None, shm_size).unwrap();
        use linux_loader::loader::bootparam::boot_params;
        let params: boot_params = mem.read_obj(GuestAddress(BOOT_PARAMS_ADDR)).unwrap();
        let entries = { params.e820_entries };
        assert_eq!(entries, 2, "shm: low + high (no MMIO split)");
        let size1 = { params.e820_table[1].size };
        // High memory should be total - HIMEM_START - shm_size.
        assert_eq!(size1, (512 << 20) - HIMEM_START - shm_size);
    }

    #[test]
    fn e820_shm_zero_unchanged() {
        // shm_size=0 should produce identical results to the non-shm case.
        let mem_a = test_mem(512);
        let mem_b = test_mem(512);
        write_boot_params(&mem_a, "console=ttyS0", 512, None, None, None, 0).unwrap();
        write_boot_params(&mem_b, "console=ttyS0", 512, None, None, None, 0).unwrap();
        use linux_loader::loader::bootparam::boot_params;
        let pa: boot_params = mem_a.read_obj(GuestAddress(BOOT_PARAMS_ADDR)).unwrap();
        let pb: boot_params = mem_b.read_obj(GuestAddress(BOOT_PARAMS_ADDR)).unwrap();
        assert_eq!(pa.e820_entries, pb.e820_entries);
        let size_a = { pa.e820_table[1].size };
        let size_b = { pb.e820_table[1].size };
        assert_eq!(size_a, size_b);
    }

    #[test]
    fn e820_shm_large_mem_reduces_above_gap() {
        // 5 GB with 1 MB SHM: usable = 5119 MB, well above MMIO_GAP_END.
        let mem = test_mem(5120);
        let shm_size: u64 = 1 << 20; // 1 MB
        write_boot_params(&mem, "console=ttyS0", 5120, None, None, None, shm_size).unwrap();
        use linux_loader::loader::bootparam::boot_params;
        let params: boot_params = mem.read_obj(GuestAddress(BOOT_PARAMS_ADDR)).unwrap();
        let entries = { params.e820_entries };
        assert_eq!(entries, 3);
        // The above-gap entry should be reduced by shm_size.
        let size2 = { params.e820_table[2].size };
        assert_eq!(size2, (5120u64 << 20) - MMIO_GAP_END - shm_size);
    }
}
