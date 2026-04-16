/// ACPI 2.0 table generation for SMP topology via zerocopy packed structs.
///
/// Generates RSDP rev 2 -> XSDT -> {FADT, MADT, SRAT, SLIT}.
/// RSDT with 32-bit pointers is also provided as a fallback.
/// FADT rev 6 with legacy hardware (PIC, PIT, ISA serial).
/// Per-CPU APIC type: Local APIC (type 0) for apic_id < 255,
/// x2APIC (type 9) for apic_id >= 255.
use anyhow::{Context, Result};
use vm_memory::{Bytes, GuestAddress, GuestMemoryMmap};
use zerocopy::IntoBytes;

use super::topology::apic_id;
use crate::vmm::topology::Topology;

// RSDP at fixed address in BIOS ROM area — firmware scans for it here.
const RSDP_ADDR: u64 = 0x000E_0000;
const RSDP_SIZE: u64 = 36;

/// Addresses and sizes of all ACPI tables after dynamic placement.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct AcpiLayout {
    pub dsdt_addr: u64,
    pub dsdt_size: u64,
    pub madt_addr: u64,
    pub madt_size: u64,
    pub fadt_addr: u64,
    pub fadt_size: u64,
    pub srat_addr: u64,
    pub srat_size: u64,
    pub slit_addr: u64,
    pub slit_size: u64,
    pub rsdt_addr: u64,
    pub rsdt_size: u64,
    pub xsdt_addr: u64,
    pub xsdt_size: u64,
    pub rsdp_addr: u64,
    pub rsdp_size: u64,
}

// FADT flags
const FADT_F_PWR_BUTTON: u32 = 1 << 4;
const FADT_F_SLP_BUTTON: u32 = 1 << 5;

// IOAPIC
const IOAPIC_ADDR: u32 = 0xFEC0_0000;
const IOAPIC_ID: u8 = 0;

// Local APIC
const LAPIC_ADDR: u32 = 0xFEE0_0000;

// ---------------------------------------------------------------------------
// Packed structs — field offsets verified by zerocopy at compile time
// ---------------------------------------------------------------------------

/// ACPI SDT header (36 bytes). Shared prefix for RSDT, XSDT, FADT, MADT,
/// SRAT, SLIT.
#[repr(C, packed)]
#[derive(Clone, Copy, Default, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
struct SdtHeader {
    signature: [u8; 4],
    length: u32,
    revision: u8,
    checksum: u8,
    oem_id: [u8; 6],
    oem_table_id: [u8; 8],
    oem_revision: u32,
    creator_id: [u8; 4],
    creator_revision: u32,
}

impl SdtHeader {
    fn new(sig: &[u8; 4], length: u32, revision: u8) -> Self {
        Self {
            signature: *sig,
            length,
            revision,
            oem_id: *b"KTSTR\0",
            oem_table_id: {
                let mut id = [0u8; 8];
                let prefix = b"KTSR";
                id[..prefix.len()].copy_from_slice(prefix);
                id[prefix.len()..prefix.len() + sig.len()].copy_from_slice(sig);
                id
            },
            oem_revision: 1,
            creator_id: *b"KTSR",
            creator_revision: 1,
            ..Default::default()
        }
    }
}

/// RSDP rev 2 (36 bytes).
#[repr(C, packed)]
#[derive(Clone, Copy, Default, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
struct Rsdp {
    signature: [u8; 8],
    checksum: u8,
    oem_id: [u8; 6],
    revision: u8,
    rsdt_address: u32,
    length: u32,
    xsdt_address: u64,
    extended_checksum: u8,
    _reserved: [u8; 3],
}

/// MADT header (44 bytes = 36 SDT + 8 MADT-specific).
#[repr(C, packed)]
#[derive(Clone, Copy, Default, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
struct MadtHeader {
    sdt: SdtHeader,
    local_apic_address: u32,
    flags: u32,
}

/// Local APIC entry (type 0, 8 bytes).
#[repr(C, packed)]
#[derive(Clone, Copy, Default, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
struct MadtLocalApic {
    entry_type: u8,
    length: u8,
    processor_id: u8,
    apic_id: u8,
    flags: u32,
}

/// x2APIC entry (type 9, 16 bytes).
#[repr(C, packed)]
#[derive(Clone, Copy, Default, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
struct MadtX2Apic {
    entry_type: u8,
    length: u8,
    _reserved: u16,
    x2apic_id: u32,
    flags: u32,
    processor_uid: u32,
}

/// IOAPIC entry (type 1, 12 bytes).
#[repr(C, packed)]
#[derive(Clone, Copy, Default, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
struct MadtIoApic {
    entry_type: u8,
    length: u8,
    io_apic_id: u8,
    _reserved: u8,
    io_apic_address: u32,
    gsi_base: u32,
}

/// Interrupt Source Override (type 2, 10 bytes).
#[repr(C, packed)]
#[derive(Clone, Copy, Default, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
struct MadtIso {
    entry_type: u8,
    length: u8,
    bus: u8,
    source: u8,
    gsi: u32,
    flags: u16,
}

/// Local APIC NMI (type 4, 6 bytes).
#[repr(C, packed)]
#[derive(Clone, Copy, Default, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
struct MadtLapicNmi {
    entry_type: u8,
    length: u8,
    processor_id: u8,
    flags: u16,
    lint: u8,
}

/// x2APIC NMI (type 0x0A, 12 bytes).
#[repr(C, packed)]
#[derive(Clone, Copy, Default, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
struct MadtX2ApicNmi {
    entry_type: u8,
    length: u8,
    flags: u16,
    processor_uid: u32,
    lint: u8,
    _reserved: [u8; 3],
}

/// SRAT CPU affinity: ProcessorLocalX2ApicAffinity (type 2, 24 bytes).
#[repr(C, packed)]
#[derive(Clone, Copy, Default, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
struct SratCpuAffinity {
    entry_type: u8,
    length: u8,
    _reserved0: u16,
    proximity_domain: u32,
    x2apic_id: u32,
    flags: u32,
    clock_domain: u32,
    _reserved1: u32,
}

/// SRAT memory affinity (type 1, 40 bytes).
#[repr(C, packed)]
#[derive(Clone, Copy, Default, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
struct SratMemAffinity {
    entry_type: u8,
    length: u8,
    proximity_domain_lo: u16, // low 16 bits (we put full u32 split)
    proximity_domain_hi: u16, // high 16 bits
    _reserved0: u16,
    base_address: u64,
    address_length: u64,
    _reserved1: u32,
    flags: u32,
    _reserved2: u64,
}

// ---------------------------------------------------------------------------
// Checksum
// ---------------------------------------------------------------------------

fn acpi_checksum(data: &[u8]) -> u8 {
    let sum: u8 = data.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
    (!sum).wrapping_add(1)
}

/// Apply checksum to a byte buffer at the SDT header checksum offset (byte 9).
fn set_sdt_checksum(buf: &mut [u8]) {
    buf[9] = 0;
    buf[9] = acpi_checksum(buf);
}

// ---------------------------------------------------------------------------
// Table generation
// ---------------------------------------------------------------------------

/// Write ACPI tables to guest memory with dynamic layout.
///
/// Tables are packed contiguously starting after the RSDP (which is at a
/// fixed address). Order: DSDT, MADT, FADT, SRAT, SLIT, RSDT, XSDT, RSDP.
///
/// When `shm_size > 0`, SRAT memory affinity for the last NUMA node is reduced
/// so the SHM region at the top of guest physical memory is excluded.
pub fn setup_acpi(
    mem: &GuestMemoryMmap,
    topo: &Topology,
    memory_mb: u32,
    shm_size: u64,
) -> Result<AcpiLayout> {
    let num_cpus = topo.total_cpus();
    let num_numa_nodes = topo.numa_nodes;

    // Compute table sizes.
    let dsdt_size: u64 = 36;

    let madt_size = compute_madt_size(topo) as u64;

    let fadt_size: u64 = 276;

    // SRAT: one CPU affinity entry per vCPU, one memory affinity entry per NUMA node.
    let srat_size: u64 =
        (48 + std::mem::size_of::<SratCpuAffinity>() as u32 * num_cpus
            + std::mem::size_of::<SratMemAffinity>() as u32 * num_numa_nodes) as u64;

    // SLIT: NxN distance matrix where N = NUMA node count.
    let n = num_numa_nodes as u64;
    let slit_size: u64 = 36 + 8 + n * n;

    // RSDT: 36-byte header + 4 x 32-bit pointers
    let rsdt_size: u64 = 36 + 16;
    // XSDT: 36-byte header + 4 x 64-bit pointers
    let xsdt_size: u64 = 36 + 32;

    // Pack tables contiguously after RSDP.
    let mut cursor = RSDP_ADDR + RSDP_SIZE;

    let dsdt_addr = cursor;
    cursor += dsdt_size;

    let madt_addr = cursor;
    cursor += madt_size;

    let fadt_addr = cursor;
    cursor += fadt_size;

    let srat_addr = cursor;
    cursor += srat_size;

    let slit_addr = cursor;
    cursor += slit_size;

    let rsdt_addr = cursor;
    cursor += rsdt_size;

    let xsdt_addr = cursor;

    let layout = AcpiLayout {
        dsdt_addr,
        dsdt_size,
        madt_addr,
        madt_size,
        fadt_addr,
        fadt_size,
        srat_addr,
        srat_size,
        slit_addr,
        slit_size,
        rsdt_addr,
        rsdt_size,
        xsdt_addr,
        xsdt_size,
        rsdp_addr: RSDP_ADDR,
        rsdp_size: RSDP_SIZE,
    };

    write_dsdt(mem, dsdt_addr)?;
    write_madt(mem, topo, madt_addr)?;
    write_fadt(mem, &layout)?;
    write_srat(mem, topo, memory_mb, shm_size, srat_addr)?;
    write_slit(mem, topo, slit_addr)?;
    write_rsdt(mem, &layout)?;
    write_xsdt(mem, &layout)?;
    write_rsdp(mem, &layout)?;
    Ok(layout)
}

fn write_rsdp(mem: &GuestMemoryMmap, layout: &AcpiLayout) -> Result<()> {
    let mut rsdp = Rsdp {
        signature: *b"RSD PTR ",
        oem_id: *b"KTSTR\0",
        revision: 2,
        rsdt_address: layout.rsdt_addr as u32,
        length: 36,
        xsdt_address: layout.xsdt_addr,
        ..Default::default()
    };
    rsdp.checksum = acpi_checksum(&rsdp.as_bytes()[..20]);
    rsdp.extended_checksum = acpi_checksum(rsdp.as_bytes());
    mem.write_slice(rsdp.as_bytes(), GuestAddress(RSDP_ADDR))
        .context("write RSDP")?;
    Ok(())
}

fn write_rsdt(mem: &GuestMemoryMmap, layout: &AcpiLayout) -> Result<()> {
    let len = 36 + 16;
    let mut buf = vec![0u8; len];
    let hdr = SdtHeader::new(b"RSDT", len as u32, 1);
    buf[..36].copy_from_slice(hdr.as_bytes());
    let entries: [u32; 4] = [
        layout.fadt_addr as u32,
        layout.madt_addr as u32,
        layout.srat_addr as u32,
        layout.slit_addr as u32,
    ];
    buf[36..52].copy_from_slice(entries.as_bytes());
    set_sdt_checksum(&mut buf);
    mem.write_slice(&buf, GuestAddress(layout.rsdt_addr))
        .context("write RSDT")?;
    Ok(())
}

fn write_xsdt(mem: &GuestMemoryMmap, layout: &AcpiLayout) -> Result<()> {
    let len = 36 + 32;
    let mut buf = vec![0u8; len];
    let hdr = SdtHeader::new(b"XSDT", len as u32, 1);
    buf[..36].copy_from_slice(hdr.as_bytes());
    let entries: [u64; 4] = [
        layout.fadt_addr,
        layout.madt_addr,
        layout.srat_addr,
        layout.slit_addr,
    ];
    buf[36..68].copy_from_slice(entries.as_bytes());
    set_sdt_checksum(&mut buf);
    mem.write_slice(&buf, GuestAddress(layout.xsdt_addr))
        .context("write XSDT")?;
    Ok(())
}

fn write_dsdt(mem: &GuestMemoryMmap, addr: u64) -> Result<()> {
    let mut buf = vec![0u8; 36];
    let hdr = SdtHeader::new(b"DSDT", 36, 2);
    buf[..36].copy_from_slice(hdr.as_bytes());
    set_sdt_checksum(&mut buf);
    mem.write_slice(&buf, GuestAddress(addr))
        .context("write DSDT")?;
    Ok(())
}

fn write_fadt(mem: &GuestMemoryMmap, layout: &AcpiLayout) -> Result<()> {
    let mut buf = vec![0u8; 276];
    let hdr = SdtHeader::new(b"FACP", 276, 6);
    buf[..36].copy_from_slice(hdr.as_bytes());
    // DSDT pointer at offset 40 (32-bit, legacy)
    buf[40..44].copy_from_slice(&(layout.dsdt_addr as u32).to_le_bytes());
    // X_DSDT at offset 140 (64-bit)
    buf[140..148].copy_from_slice(&layout.dsdt_addr.to_le_bytes());
    let flags = FADT_F_PWR_BUTTON | FADT_F_SLP_BUTTON;
    buf[112..116].copy_from_slice(&flags.to_le_bytes());
    buf[131] = 5;
    set_sdt_checksum(&mut buf);
    mem.write_slice(&buf, GuestAddress(layout.fadt_addr))
        .context("write FADT")?;
    Ok(())
}

fn write_srat(
    mem: &GuestMemoryMmap,
    topo: &Topology,
    memory_mb: u32,
    shm_size: u64,
    addr: u64,
) -> Result<()> {
    let num_cpus = topo.total_cpus();
    let num_numa_nodes = topo.numa_nodes;

    let len = 48
        + std::mem::size_of::<SratCpuAffinity>() as u32 * num_cpus
        + std::mem::size_of::<SratMemAffinity>() as u32 * num_numa_nodes;
    let mut buf = vec![0u8; len as usize];

    let hdr = SdtHeader::new(b"SRAT", len, 3);
    buf[..36].copy_from_slice(hdr.as_bytes());
    buf[36..40].copy_from_slice(&1u32.to_le_bytes());

    let mut offset = 48;

    // CPU affinity: each vCPU maps to the NUMA node that owns its LLC.
    for cpu_id in 0..num_cpus {
        let (llc_id, _, _) = topo.decompose(cpu_id);
        let node_id = topo.numa_node_of(llc_id);
        let entry = SratCpuAffinity {
            entry_type: 2,
            length: std::mem::size_of::<SratCpuAffinity>() as u8,
            proximity_domain: node_id,
            x2apic_id: apic_id(topo, cpu_id),
            flags: 1,
            ..Default::default()
        };
        let bytes = entry.as_bytes();
        buf[offset..offset + bytes.len()].copy_from_slice(bytes);
        offset += bytes.len();
    }

    // Memory affinity: one entry per NUMA node, memory split evenly.
    let mem_bytes = (memory_mb as u64) << 20;
    let usable_bytes = mem_bytes - shm_size;
    let per_node = usable_bytes / num_numa_nodes as u64;
    for node in 0..num_numa_nodes {
        let base = node as u64 * per_node;
        let length = if node == num_numa_nodes - 1 {
            usable_bytes - base
        } else {
            per_node
        };
        let entry = SratMemAffinity {
            entry_type: 1,
            length: std::mem::size_of::<SratMemAffinity>() as u8,
            proximity_domain_lo: node as u16,
            proximity_domain_hi: (node >> 16) as u16,
            base_address: base,
            address_length: length,
            flags: 1,
            ..Default::default()
        };
        let bytes = entry.as_bytes();
        buf[offset..offset + bytes.len()].copy_from_slice(bytes);
        offset += bytes.len();
    }

    set_sdt_checksum(&mut buf);
    mem.write_slice(&buf, GuestAddress(addr))
        .context("write SRAT")?;
    Ok(())
}

fn write_slit(mem: &GuestMemoryMmap, topo: &Topology, addr: u64) -> Result<()> {
    let n = topo.numa_nodes as u64;
    let len = 36 + 8 + n * n;
    let mut buf = vec![0u8; len as usize];

    let hdr = SdtHeader::new(b"SLIT", len as u32, 1);
    buf[..36].copy_from_slice(hdr.as_bytes());
    buf[36..44].copy_from_slice(&n.to_le_bytes());
    let matrix_start = 44;
    for i in 0..n {
        for j in 0..n {
            buf[matrix_start + (i * n + j) as usize] = if i == j { 10 } else { 20 };
        }
    }

    set_sdt_checksum(&mut buf);
    mem.write_slice(&buf, GuestAddress(addr))
        .context("write SLIT")?;
    Ok(())
}

/// Determine whether a given APIC ID should use x2APIC (type 9) or
/// Local APIC (type 0). APIC ID < 255 uses type 0, >= 255 uses type 9.
fn use_x2apic_entry(apic_id: u32) -> bool {
    apic_id >= 255
}

/// Compute MADT total size for a given topology.
fn compute_madt_size(topo: &Topology) -> u32 {
    let num_cpus = topo.total_cpus();
    let mut cpu_entries_size: u32 = 0;
    let mut has_x2apic = false;
    let mut has_lapic = false;
    for cpu_id in 0..num_cpus {
        if use_x2apic_entry(apic_id(topo, cpu_id)) {
            cpu_entries_size += std::mem::size_of::<MadtX2Apic>() as u32;
            has_x2apic = true;
        } else {
            cpu_entries_size += std::mem::size_of::<MadtLocalApic>() as u32;
            has_lapic = true;
        }
    }
    let nmi_size: u32 = if has_lapic {
        std::mem::size_of::<MadtLapicNmi>() as u32
    } else {
        0
    } + if has_x2apic {
        std::mem::size_of::<MadtX2ApicNmi>() as u32
    } else {
        0
    };
    std::mem::size_of::<MadtHeader>() as u32
        + cpu_entries_size
        + std::mem::size_of::<MadtIoApic>() as u32
        + std::mem::size_of::<MadtIso>() as u32
        + nmi_size
}

fn write_madt(mem: &GuestMemoryMmap, topo: &Topology, addr: u64) -> Result<()> {
    let num_cpus = topo.total_cpus();

    let mut has_x2apic = false;
    let mut has_lapic = false;
    for cpu_id in 0..num_cpus {
        if use_x2apic_entry(apic_id(topo, cpu_id)) {
            has_x2apic = true;
        } else {
            has_lapic = true;
        }
    }

    let len = compute_madt_size(topo);
    let mut buf = vec![0u8; len as usize];

    // MADT header
    let hdr = MadtHeader {
        sdt: SdtHeader::new(b"APIC", len, 3),
        local_apic_address: LAPIC_ADDR,
        flags: 1, // PCAT_COMPAT
    };
    buf[..std::mem::size_of::<MadtHeader>()].copy_from_slice(hdr.as_bytes());

    let mut offset = std::mem::size_of::<MadtHeader>();

    // CPU entries
    for cpu_id in 0..num_cpus {
        let id = apic_id(topo, cpu_id);
        if use_x2apic_entry(id) {
            let entry = MadtX2Apic {
                entry_type: 9,
                length: std::mem::size_of::<MadtX2Apic>() as u8,
                x2apic_id: id,
                flags: 1,
                processor_uid: cpu_id,
                ..Default::default()
            };
            let bytes = entry.as_bytes();
            buf[offset..offset + bytes.len()].copy_from_slice(bytes);
            offset += bytes.len();
        } else {
            let entry = MadtLocalApic {
                entry_type: 0,
                length: std::mem::size_of::<MadtLocalApic>() as u8,
                processor_id: cpu_id as u8,
                apic_id: id as u8,
                flags: 1,
            };
            let bytes = entry.as_bytes();
            buf[offset..offset + bytes.len()].copy_from_slice(bytes);
            offset += bytes.len();
        }
    }

    // IOAPIC
    let ioapic = MadtIoApic {
        entry_type: 1,
        length: std::mem::size_of::<MadtIoApic>() as u8,
        io_apic_id: IOAPIC_ID,
        io_apic_address: IOAPIC_ADDR,
        gsi_base: 0,
        ..Default::default()
    };
    let bytes = ioapic.as_bytes();
    buf[offset..offset + bytes.len()].copy_from_slice(bytes);
    offset += bytes.len();

    // Interrupt Source Override: IRQ0 -> GSI 2
    let iso = MadtIso {
        entry_type: 2,
        length: std::mem::size_of::<MadtIso>() as u8,
        bus: 0,
        source: 0,
        gsi: 2,
        flags: 0,
    };
    let bytes = iso.as_bytes();
    buf[offset..offset + bytes.len()].copy_from_slice(bytes);
    offset += bytes.len();

    // NMI entries
    if has_lapic {
        let nmi = MadtLapicNmi {
            entry_type: 4,
            length: std::mem::size_of::<MadtLapicNmi>() as u8,
            processor_id: 0xFF,
            flags: 0,
            lint: 1,
        };
        let bytes = nmi.as_bytes();
        buf[offset..offset + bytes.len()].copy_from_slice(bytes);
        offset += bytes.len();
    }
    if has_x2apic {
        let nmi = MadtX2ApicNmi {
            entry_type: 0x0A,
            length: std::mem::size_of::<MadtX2ApicNmi>() as u8,
            flags: 0,
            processor_uid: 0xFFFF_FFFF,
            lint: 1,
            _reserved: [0; 3],
        };
        let bytes = nmi.as_bytes();
        buf[offset..offset + bytes.len()].copy_from_slice(bytes);
    }

    set_sdt_checksum(&mut buf);
    mem.write_slice(&buf, GuestAddress(addr))
        .context("write MADT")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_mem(mb: u32) -> GuestMemoryMmap {
        GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), (mb as usize) << 20)]).unwrap()
    }

    fn read_table(mem: &GuestMemoryMmap, addr: u64) -> Vec<u8> {
        let mut len_bytes = [0u8; 4];
        mem.read_slice(&mut len_bytes, GuestAddress(addr + 4))
            .unwrap();
        let len = u32::from_le_bytes(len_bytes) as usize;
        let mut buf = vec![0u8; len];
        mem.read_slice(&mut buf, GuestAddress(addr)).unwrap();
        buf
    }

    fn read_madt(mem: &GuestMemoryMmap, layout: &AcpiLayout) -> Vec<u8> {
        read_table(mem, layout.madt_addr)
    }

    fn walk_madt_entries(madt: &[u8]) -> Vec<(u8, u8, &[u8])> {
        let hdr_size = std::mem::size_of::<MadtHeader>();
        let mut entries = Vec::new();
        let mut offset = hdr_size;
        while offset < madt.len() {
            let entry_type = madt[offset];
            let entry_len = madt[offset + 1];
            entries.push((
                entry_type,
                entry_len,
                &madt[offset..offset + entry_len as usize],
            ));
            offset += entry_len as usize;
        }
        entries
    }

    // -- Struct size compile-time assertions --
    const _: () = assert!(std::mem::size_of::<SdtHeader>() == 36);
    const _: () = assert!(std::mem::size_of::<Rsdp>() == 36);
    const _: () = assert!(std::mem::size_of::<MadtHeader>() == 44);
    const _: () = assert!(std::mem::size_of::<MadtLocalApic>() == 8);
    const _: () = assert!(std::mem::size_of::<MadtX2Apic>() == 16);
    const _: () = assert!(std::mem::size_of::<MadtIoApic>() == 12);
    const _: () = assert!(std::mem::size_of::<MadtIso>() == 10);
    const _: () = assert!(std::mem::size_of::<MadtLapicNmi>() == 6);
    const _: () = assert!(std::mem::size_of::<MadtX2ApicNmi>() == 12);
    const _: () = assert!(std::mem::size_of::<SratCpuAffinity>() == 24);
    const _: () = assert!(std::mem::size_of::<SratMemAffinity>() == 40);

    #[test]
    fn rsdp_signature_and_checksum() {
        let mem = test_mem(16);
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
        let mut rsdp = [0u8; 20];
        mem.read_slice(&mut rsdp, GuestAddress(l.rsdp_addr))
            .unwrap();
        assert_eq!(&rsdp[..8], b"RSD PTR ");
        let sum: u8 = rsdp.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
        assert_eq!(sum, 0, "RSDP checksum must be zero");
    }

    #[test]
    fn rsdt_signature_and_checksum() {
        let mem = test_mem(16);
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
        let rsdt = read_table(&mem, l.rsdt_addr);
        assert_eq!(&rsdt[..4], b"RSDT");
        let sum: u8 = rsdt.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
        assert_eq!(sum, 0, "RSDT checksum must be zero");
    }

    #[test]
    fn madt_signature_and_checksum() {
        let mem = test_mem(16);
        let topo = Topology {
            llcs: 2,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
        let madt = read_madt(&mem, &l);
        assert_eq!(&madt[..4], b"APIC");
        let sum: u8 = madt.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
        assert_eq!(sum, 0, "MADT checksum must be zero");
    }

    #[test]
    fn madt_has_correct_cpu_count() {
        let mem = test_mem(16);
        let topo = Topology {
            llcs: 2,
            cores_per_llc: 4,
            threads_per_core: 2,
            numa_nodes: 1,
        };
        let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
        let madt = read_madt(&mem, &l);
        let entries = walk_madt_entries(&madt);
        let cpu_count = entries
            .iter()
            .filter(|(t, _, _)| *t == 0 || *t == 9)
            .count();
        assert_eq!(cpu_count, 16);
    }

    #[test]
    fn madt_apic_ids_match_topology() {
        let mem = test_mem(16);
        let topo = Topology {
            llcs: 2,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
        let madt = read_madt(&mem, &l);
        let entries = walk_madt_entries(&madt);
        let mut cpu_idx = 0u32;
        for (entry_type, _, data) in &entries {
            match *entry_type {
                0 => {
                    assert_eq!(data[3] as u32, apic_id(&topo, cpu_idx));
                    cpu_idx += 1;
                }
                9 => {
                    assert_eq!(
                        u32::from_le_bytes(data[4..8].try_into().unwrap()),
                        apic_id(&topo, cpu_idx)
                    );
                    cpu_idx += 1;
                }
                _ => {}
            }
        }
    }

    #[test]
    fn madt_has_ioapic() {
        let mem = test_mem(16);
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
        let madt = read_madt(&mem, &l);
        let entries = walk_madt_entries(&madt);
        let ioapic = entries.iter().find(|(t, _, _)| *t == 1);
        assert!(ioapic.is_some());
        let (_, _, data) = ioapic.unwrap();
        assert_eq!(
            u32::from_le_bytes(data[4..8].try_into().unwrap()),
            IOAPIC_ADDR
        );
    }

    #[test]
    fn rsdp_points_to_rsdt() {
        let mem = test_mem(16);
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
        let mut rsdp = [0u8; 20];
        mem.read_slice(&mut rsdp, GuestAddress(l.rsdp_addr))
            .unwrap();
        assert_eq!(
            u32::from_le_bytes(rsdp[16..20].try_into().unwrap()),
            l.rsdt_addr as u32
        );
    }

    #[test]
    fn rsdt_table_pointers() {
        let mem = test_mem(16);
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
        let mut entry = [0u8; 4];
        mem.read_slice(&mut entry, GuestAddress(l.rsdt_addr + 36))
            .unwrap();
        assert_eq!(u32::from_le_bytes(entry), l.fadt_addr as u32);
        mem.read_slice(&mut entry, GuestAddress(l.rsdt_addr + 40))
            .unwrap();
        assert_eq!(u32::from_le_bytes(entry), l.madt_addr as u32);
        mem.read_slice(&mut entry, GuestAddress(l.rsdt_addr + 44))
            .unwrap();
        assert_eq!(u32::from_le_bytes(entry), l.srat_addr as u32);
        mem.read_slice(&mut entry, GuestAddress(l.rsdt_addr + 48))
            .unwrap();
        assert_eq!(u32::from_le_bytes(entry), l.slit_addr as u32);
    }

    #[test]
    fn madt_has_iso_irq0_gsi2() {
        let mem = test_mem(16);
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
        let madt = read_madt(&mem, &l);
        let entries = walk_madt_entries(&madt);
        let iso = entries.iter().find(|(t, _, _)| *t == 2).unwrap();
        assert_eq!(iso.2[3], 0);
        assert_eq!(u32::from_le_bytes(iso.2[4..8].try_into().unwrap()), 2);
    }

    #[test]
    fn madt_has_nmi() {
        let mem = test_mem(16);
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
        let madt = read_madt(&mem, &l);
        let entries = walk_madt_entries(&madt);
        assert!(entries.iter().any(|(t, _, _)| *t == 4 || *t == 0x0A));
    }

    #[test]
    fn small_topology_uses_lapic_entries() {
        let mem = test_mem(16);
        let topo = Topology {
            llcs: 2,
            cores_per_llc: 4,
            threads_per_core: 2,
            numa_nodes: 1,
        };
        let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
        let madt = read_madt(&mem, &l);
        let entries = walk_madt_entries(&madt);
        assert_eq!(entries.iter().filter(|(t, _, _)| *t == 9).count(), 0);
        assert_eq!(entries.iter().filter(|(t, _, _)| *t == 0).count(), 16);
        assert!(entries.iter().any(|(t, _, _)| *t == 4));
        assert!(!entries.iter().any(|(t, _, _)| *t == 0x0A));
    }

    #[test]
    fn large_topology_uses_mixed_entries() {
        let mem = test_mem(16);
        let topo = Topology {
            llcs: 14,
            cores_per_llc: 9,
            threads_per_core: 2,
            numa_nodes: 1,
        };
        let mut has_low = false;
        let mut has_high = false;
        for cpu_id in 0..topo.total_cpus() {
            let id = apic_id(&topo, cpu_id);
            if id < 255 {
                has_low = true;
            } else {
                has_high = true;
            }
        }
        assert!(has_low && has_high);
        let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
        let madt = read_madt(&mem, &l);
        let entries = walk_madt_entries(&madt);
        let lapic_count = entries.iter().filter(|(t, _, _)| *t == 0).count();
        let x2apic_count = entries.iter().filter(|(t, _, _)| *t == 9).count();
        assert!(lapic_count > 0);
        assert!(x2apic_count > 0);
        assert_eq!(lapic_count + x2apic_count, topo.total_cpus() as usize);
        assert!(entries.iter().any(|(t, _, _)| *t == 4));
        assert!(entries.iter().any(|(t, _, _)| *t == 0x0A));
    }

    #[test]
    fn x2apic_nmi_fields_correct() {
        let mem = test_mem(16);
        let topo = Topology {
            llcs: 14,
            cores_per_llc: 9,
            threads_per_core: 2,
            numa_nodes: 1,
        };
        let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
        let madt = read_madt(&mem, &l);
        let entries = walk_madt_entries(&madt);
        let (_, len, data) = entries.iter().find(|(t, _, _)| *t == 0x0A).unwrap();
        assert_eq!(*len, 12);
        assert_eq!(u16::from_le_bytes(data[2..4].try_into().unwrap()), 0);
        assert_eq!(
            u32::from_le_bytes(data[4..8].try_into().unwrap()),
            0xFFFF_FFFF
        );
        assert_eq!(data[8], 1);
    }

    #[test]
    fn lapic_nmi_fields_correct() {
        let mem = test_mem(16);
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
        let madt = read_madt(&mem, &l);
        let entries = walk_madt_entries(&madt);
        let (_, len, data) = entries.iter().find(|(t, _, _)| *t == 4).unwrap();
        assert_eq!(*len, 6);
        assert_eq!(data[2], 0xFF);
        assert_eq!(u16::from_le_bytes(data[3..5].try_into().unwrap()), 0);
        assert_eq!(data[5], 1);
    }

    #[test]
    fn madt_checksum_representative_topologies() {
        let topos = [
            (1, 1, 1),   // degenerate single CPU
            (2, 1, 1),   // minimal multi-LLC
            (3, 3, 1),   // odd non-power-of-2
            (1, 1, 2),   // minimal SMT
            (2, 4, 2),   // standard multi-LLC with SMT
            (7, 5, 3),   // all dimensions non-power-of-2
            (15, 16, 1), // large scale no SMT
            (14, 9, 2),  // large with SMT, mixed LAPIC/x2APIC
            (2, 128, 1), // x2APIC boundary (max APIC ID = 255)
            (14, 18, 1), // large no SMT, mixed LAPIC/x2APIC
        ];
        for (llcs, cores, threads) in topos {
            let mem = test_mem(16);
            let topo = Topology {
                llcs,
                cores_per_llc: cores,
                threads_per_core: threads,
                numa_nodes: 1,
            };
            let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
            let madt = read_madt(&mem, &l);
            let sum: u8 = madt.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
            assert_eq!(
                sum, 0,
                "MADT checksum failed for {llcs}l/{cores}c/{threads}t"
            );
            let entries = walk_madt_entries(&madt);
            let cpu_count = entries
                .iter()
                .filter(|(t, _, _)| *t == 0 || *t == 9)
                .count();
            assert_eq!(cpu_count, topo.total_cpus() as usize);
            assert!(entries.iter().any(|(t, _, _)| *t == 4 || *t == 0x0A));
        }
    }

    #[test]
    fn cpu_entry_type_matches_apic_id() {
        for (llcs, cores, threads) in [(1, 4, 1), (2, 2, 2), (15, 8, 2), (14, 9, 2)] {
            let mem = test_mem(16);
            let topo = Topology {
                llcs,
                cores_per_llc: cores,
                threads_per_core: threads,
                numa_nodes: 1,
            };
            let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
            let madt = read_madt(&mem, &l);
            let entries = walk_madt_entries(&madt);
            let mut cpu_idx = 0u32;
            for (entry_type, _, data) in &entries {
                match *entry_type {
                    0 => {
                        let id = data[3] as u32;
                        assert!(id < 255);
                        assert_eq!(id, apic_id(&topo, cpu_idx));
                        cpu_idx += 1;
                    }
                    9 => {
                        let id = u32::from_le_bytes(data[4..8].try_into().unwrap());
                        assert!(id >= 255);
                        assert_eq!(id, apic_id(&topo, cpu_idx));
                        cpu_idx += 1;
                    }
                    _ => {}
                }
            }
        }
    }

    #[test]
    fn madt_entry_lengths_valid() {
        let mem = test_mem(16);
        let topo = Topology {
            llcs: 14,
            cores_per_llc: 9,
            threads_per_core: 2,
            numa_nodes: 1,
        };
        let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
        let madt = read_madt(&mem, &l);
        let entries = walk_madt_entries(&madt);
        for (entry_type, entry_len, _) in &entries {
            let expected = match *entry_type {
                0 => 8,
                1 => 12,
                2 => 10,
                4 => 6,
                9 => 16,
                0x0A => 12,
                t => panic!("unexpected MADT entry type {t}"),
            };
            assert_eq!(*entry_len, expected);
        }
    }

    #[test]
    fn madt_total_length_matches_entries() {
        let mem = test_mem(16);
        let topo = Topology {
            llcs: 14,
            cores_per_llc: 9,
            threads_per_core: 2,
            numa_nodes: 1,
        };
        let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
        let madt = read_madt(&mem, &l);
        let declared_len = u32::from_le_bytes(madt[4..8].try_into().unwrap()) as usize;
        assert_eq!(declared_len, madt.len());
        let entries = walk_madt_entries(&madt);
        let entries_size: usize = entries.iter().map(|(_, l, _)| *l as usize).sum();
        assert_eq!(
            std::mem::size_of::<MadtHeader>() + entries_size,
            declared_len
        );
    }

    #[test]
    fn cpu_flags_enabled() {
        let mem = test_mem(16);
        let topo = Topology {
            llcs: 2,
            cores_per_llc: 2,
            threads_per_core: 2,
            numa_nodes: 1,
        };
        let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
        let madt = read_madt(&mem, &l);
        let entries = walk_madt_entries(&madt);
        for (entry_type, _, data) in &entries {
            match *entry_type {
                0 => assert_eq!(u32::from_le_bytes(data[4..8].try_into().unwrap()) & 1, 1),
                9 => assert_eq!(u32::from_le_bytes(data[8..12].try_into().unwrap()) & 1, 1),
                _ => {}
            }
        }
    }

    #[test]
    fn rsdp_rev2_structure() {
        let mem = test_mem(16);
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
        let mut rsdp = [0u8; 36];
        mem.read_slice(&mut rsdp, GuestAddress(l.rsdp_addr))
            .unwrap();
        assert_eq!(&rsdp[..8], b"RSD PTR ");
        assert_eq!(rsdp[15], 2);
        assert_eq!(
            u32::from_le_bytes(rsdp[16..20].try_into().unwrap()),
            l.rsdt_addr as u32
        );
        assert_eq!(u32::from_le_bytes(rsdp[20..24].try_into().unwrap()), 36);
        assert_eq!(
            u64::from_le_bytes(rsdp[24..32].try_into().unwrap()),
            l.xsdt_addr
        );
        let sum20: u8 = rsdp[..20].iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
        assert_eq!(sum20, 0);
        let sum36: u8 = rsdp.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
        assert_eq!(sum36, 0);
    }

    #[test]
    fn xsdt_signature_and_checksum() {
        let mem = test_mem(16);
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
        let xsdt = read_table(&mem, l.xsdt_addr);
        assert_eq!(&xsdt[..4], b"XSDT");
        assert_eq!(xsdt.len(), 68);
        let sum: u8 = xsdt.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
        assert_eq!(sum, 0);
    }

    #[test]
    fn xsdt_table_pointers() {
        let mem = test_mem(16);
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
        let mut entry = [0u8; 8];
        mem.read_slice(&mut entry, GuestAddress(l.xsdt_addr + 36))
            .unwrap();
        assert_eq!(u64::from_le_bytes(entry), l.fadt_addr);
        mem.read_slice(&mut entry, GuestAddress(l.xsdt_addr + 44))
            .unwrap();
        assert_eq!(u64::from_le_bytes(entry), l.madt_addr);
        mem.read_slice(&mut entry, GuestAddress(l.xsdt_addr + 52))
            .unwrap();
        assert_eq!(u64::from_le_bytes(entry), l.srat_addr);
        mem.read_slice(&mut entry, GuestAddress(l.xsdt_addr + 60))
            .unwrap();
        assert_eq!(u64::from_le_bytes(entry), l.slit_addr);
    }

    #[test]
    fn fadt_signature_and_checksum() {
        let mem = test_mem(16);
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
        let mut fadt = [0u8; 276];
        mem.read_slice(&mut fadt, GuestAddress(l.fadt_addr))
            .unwrap();
        assert_eq!(&fadt[..4], b"FACP");
        assert_eq!(u32::from_le_bytes(fadt[4..8].try_into().unwrap()), 276);
        assert_eq!(fadt[8], 6);
        let sum: u8 = fadt.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
        assert_eq!(sum, 0);
    }

    #[test]
    fn fadt_hw_reduced_flags() {
        let mem = test_mem(16);
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
        let mut fadt = [0u8; 276];
        mem.read_slice(&mut fadt, GuestAddress(l.fadt_addr))
            .unwrap();
        let flags = u32::from_le_bytes(fadt[112..116].try_into().unwrap());
        assert_eq!(flags & (1 << 20), 0, "HW_REDUCED_ACPI must not be set");
        assert_ne!(flags & FADT_F_PWR_BUTTON, 0);
        assert_ne!(flags & FADT_F_SLP_BUTTON, 0);
    }

    #[test]
    fn fadt_minor_version() {
        let mem = test_mem(16);
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
        let mut fadt = [0u8; 276];
        mem.read_slice(&mut fadt, GuestAddress(l.fadt_addr))
            .unwrap();
        assert_eq!(fadt[131], 5);
    }

    #[test]
    fn fadt_dsdt_pointers() {
        let mem = test_mem(16);
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
        let mut fadt = [0u8; 276];
        mem.read_slice(&mut fadt, GuestAddress(l.fadt_addr))
            .unwrap();
        assert_eq!(
            u32::from_le_bytes(fadt[40..44].try_into().unwrap()),
            l.dsdt_addr as u32
        );
        assert_eq!(
            u64::from_le_bytes(fadt[140..148].try_into().unwrap()),
            l.dsdt_addr
        );
    }

    #[test]
    fn dsdt_signature_and_checksum() {
        let mem = test_mem(16);
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
        let mut dsdt = [0u8; 36];
        mem.read_slice(&mut dsdt, GuestAddress(l.dsdt_addr))
            .unwrap();
        assert_eq!(&dsdt[..4], b"DSDT");
        assert_eq!(u32::from_le_bytes(dsdt[4..8].try_into().unwrap()), 36);
        let sum: u8 = dsdt.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
        assert_eq!(sum, 0, "DSDT checksum must be zero");
    }

    #[test]
    fn rsdp_points_to_xsdt() {
        let mem = test_mem(16);
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
        let mut rsdp = [0u8; 36];
        mem.read_slice(&mut rsdp, GuestAddress(l.rsdp_addr))
            .unwrap();
        assert_eq!(
            u64::from_le_bytes(rsdp[24..32].try_into().unwrap()),
            l.xsdt_addr
        );
    }

    fn walk_srat_entries(srat: &[u8]) -> Vec<(u8, u8, &[u8])> {
        let hdr_size = 48; // 36-byte SDT + 12-byte SRAT-specific
        let mut entries = Vec::new();
        let mut offset = hdr_size;
        while offset < srat.len() {
            let entry_type = srat[offset];
            let entry_len = srat[offset + 1] as usize;
            entries.push((
                entry_type,
                entry_len as u8,
                &srat[offset..offset + entry_len],
            ));
            offset += entry_len;
        }
        entries
    }

    #[test]
    fn srat_cpu_affinity_multi_numa() {
        for (numa_nodes, llcs, cores, threads) in
            [(2, 4, 2, 1), (2, 4, 2, 2), (4, 8, 1, 1), (3, 6, 2, 2)]
        {
            let mem = test_mem(16);
            let topo = Topology {
                llcs,
                cores_per_llc: cores,
                threads_per_core: threads,
                numa_nodes,
            };
            let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
            let srat = read_table(&mem, l.srat_addr);
            let entries = walk_srat_entries(&srat);
            let mut cpu_idx = 0u32;
            for (entry_type, _, data) in &entries {
                if *entry_type == 2 {
                    let prox_domain = u32::from_le_bytes(data[4..8].try_into().unwrap());
                    let (llc_id, _, _) = topo.decompose(cpu_idx);
                    let expected_node = topo.numa_node_of(llc_id);
                    assert_eq!(
                        prox_domain, expected_node,
                        "cpu {cpu_idx}: proximity_domain {prox_domain} != expected {expected_node} \
                         (topo: {numa_nodes}n/{llcs}l/{cores}c/{threads}t)"
                    );
                    let x2apic = u32::from_le_bytes(data[8..12].try_into().unwrap());
                    assert_eq!(
                        x2apic,
                        apic_id(&topo, cpu_idx),
                        "cpu {cpu_idx}: x2apic_id mismatch"
                    );
                    cpu_idx += 1;
                }
            }
            assert_eq!(cpu_idx, topo.total_cpus());
        }
    }

    #[test]
    fn srat_memory_split_multi_numa() {
        for (numa_nodes, llcs) in [(2, 4), (3, 6), (4, 8)] {
            let mem = test_mem(16);
            let topo = Topology {
                llcs,
                cores_per_llc: 2,
                threads_per_core: 1,
                numa_nodes,
            };
            let mem_bytes = 256u64 << 20;
            let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
            let srat = read_table(&mem, l.srat_addr);
            let entries = walk_srat_entries(&srat);
            let mem_entries: Vec<_> = entries.iter().filter(|(t, _, _)| *t == 1).collect();
            assert_eq!(mem_entries.len(), numa_nodes as usize);

            let mut prev_end: u64 = 0;
            let mut total: u64 = 0;
            for (i, (_, _, data)) in mem_entries.iter().enumerate() {
                let prox_lo = u16::from_le_bytes(data[2..4].try_into().unwrap()) as u32;
                let prox_hi = u16::from_le_bytes(data[4..6].try_into().unwrap()) as u32;
                let prox_domain = (prox_hi << 16) | prox_lo;
                assert_eq!(
                    prox_domain, i as u32,
                    "node {i}: proximity_domain {prox_domain} != {i} \
                     (topo: {numa_nodes}n/{llcs}l)"
                );
                let base = u64::from_le_bytes(data[8..16].try_into().unwrap());
                let length = u64::from_le_bytes(data[16..24].try_into().unwrap());
                assert_eq!(
                    base, prev_end,
                    "node {i}: base {base:#x} != prev_end {prev_end:#x} \
                     (topo: {numa_nodes}n/{llcs}l)"
                );
                assert!(length > 0, "node {i}: zero-length memory region");
                prev_end = base + length;
                total += length;
            }
            assert_eq!(
                total, mem_bytes,
                "total memory mismatch for {numa_nodes}n/{llcs}l"
            );
        }
    }

    #[test]
    fn srat_memory_split_with_shm_multi_numa() {
        for (numa_nodes, llcs, shm_size) in
            [(2, 4, 64 * 1024u64), (3, 6, 1 << 20), (4, 8, 512 * 1024)]
        {
            let mem = test_mem(16);
            let topo = Topology {
                llcs,
                cores_per_llc: 2,
                threads_per_core: 1,
                numa_nodes,
            };
            let mem_bytes = 256u64 << 20;
            let expected_usable = mem_bytes - shm_size;
            let l = setup_acpi(&mem, &topo, 256, shm_size).unwrap();
            let srat = read_table(&mem, l.srat_addr);
            let entries = walk_srat_entries(&srat);
            let total: u64 = entries
                .iter()
                .filter(|(t, _, _)| *t == 1)
                .map(|(_, _, data)| u64::from_le_bytes(data[16..24].try_into().unwrap()))
                .sum();
            assert_eq!(
                total, expected_usable,
                "shm_size={shm_size}: total {total} != expected {expected_usable} \
                 (topo: {numa_nodes}n/{llcs}l)"
            );
        }
    }

    #[test]
    fn slit_distance_matrix_multi_numa() {
        for (numa_nodes, llcs) in [(2, 2), (3, 3), (4, 4), (2, 4), (2, 6), (3, 9)] {
            let mem = test_mem(16);
            let topo = Topology {
                llcs,
                cores_per_llc: 1,
                threads_per_core: 1,
                numa_nodes,
            };
            let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
            let slit = read_table(&mem, l.slit_addr);
            assert_eq!(&slit[..4], b"SLIT", "SLIT signature mismatch");
            let n = u64::from_le_bytes(slit[36..44].try_into().unwrap());
            assert_eq!(n, numa_nodes as u64);
            let matrix_start = 44;
            for i in 0..n {
                for j in 0..n {
                    let dist = slit[matrix_start + (i * n + j) as usize];
                    if i == j {
                        assert_eq!(dist, 10, "diagonal ({i},{j}) != 10");
                    } else {
                        assert_eq!(dist, 20, "off-diagonal ({i},{j}) != 20");
                    }
                }
            }
        }
    }

    #[test]
    fn srat_slit_checksum_multi_numa() {
        for (numa_nodes, llcs, cores, threads) in
            [(2, 2, 2, 1), (2, 4, 2, 2), (3, 3, 1, 1), (4, 8, 4, 2)]
        {
            let mem = test_mem(16);
            let topo = Topology {
                llcs,
                cores_per_llc: cores,
                threads_per_core: threads,
                numa_nodes,
            };
            let l = setup_acpi(&mem, &topo, 256, 0).unwrap();
            let srat = read_table(&mem, l.srat_addr);
            let srat_sum: u8 = srat.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
            assert_eq!(
                srat_sum, 0,
                "SRAT checksum failed for {numa_nodes}n/{llcs}l/{cores}c/{threads}t"
            );
            let slit = read_table(&mem, l.slit_addr);
            let slit_sum: u8 = slit.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
            assert_eq!(
                slit_sum, 0,
                "SLIT checksum failed for {numa_nodes}n/{llcs}l/{cores}c/{threads}t"
            );
        }
    }

    #[test]
    fn srat_memory_split_remainder() {
        // 257 MB / 3 nodes: per_node = floor(269_484_032 / 3) = 89_828_010.
        // Last node absorbs remainder: 269_484_032 - 2*89_828_010 = 89_828_012.
        let memory_mb = 257u32;
        let mem = test_mem(memory_mb);
        let topo = Topology {
            llcs: 3,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 3,
        };
        let mem_bytes = (memory_mb as u64) << 20;
        let per_node = mem_bytes / 3;
        let l = setup_acpi(&mem, &topo, memory_mb, 0).unwrap();
        let srat = read_table(&mem, l.srat_addr);
        let entries = walk_srat_entries(&srat);
        let mem_entries: Vec<_> = entries.iter().filter(|(t, _, _)| *t == 1).collect();
        assert_eq!(mem_entries.len(), 3);
        let mut total: u64 = 0;
        for (i, (_, _, data)) in mem_entries.iter().enumerate() {
            let length = u64::from_le_bytes(data[16..24].try_into().unwrap());
            if i < 2 {
                assert_eq!(
                    length, per_node,
                    "node {i}: expected {per_node}, got {length}"
                );
            } else {
                let expected_last = mem_bytes - 2 * per_node;
                assert_eq!(
                    length, expected_last,
                    "last node: expected {expected_last}, got {length}"
                );
                assert!(
                    length > per_node,
                    "last node should be larger due to remainder"
                );
            }
            total += length;
        }
        assert_eq!(total, mem_bytes);
    }

    #[test]
    fn srat_shm_reduces_last_node_memory() {
        let mem = test_mem(16);
        let topo = Topology {
            llcs: 2,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let shm_size: u64 = 64 * 1024;
        let l = setup_acpi(&mem, &topo, 256, shm_size).unwrap();
        let srat = read_table(&mem, l.srat_addr);
        let entries = walk_srat_entries(&srat);
        let total_mem: u64 = entries
            .iter()
            .filter(|(t, _, _)| *t == 1)
            .map(|(_, _, data)| u64::from_le_bytes(data[16..24].try_into().unwrap()))
            .sum();
        let expected = (256u64 << 20) - shm_size;
        assert_eq!(total_mem, expected);
    }
}
