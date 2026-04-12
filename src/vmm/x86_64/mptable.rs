/// MP table setup for SMP boot.
/// The kernel reads this to discover CPUs and their APIC IDs.
/// Uses topology-aware APIC IDs for multi-socket support.
use anyhow::{Context, Result};
use vm_memory::{Address, Bytes, GuestAddress, GuestMemoryMmap};

use super::topology::apic_id;
use crate::vmm::topology::Topology;

const MPTABLE_START: u64 = 0x9fc00;

// MP table signatures
const SMP_MAGIC: [u8; 4] = *b"_MP_";
const MPC_MAGIC: [u8; 4] = *b"PCMP";

// MP table entry types
const MP_PROCESSOR: u8 = 0;
const MP_BUS: u8 = 1;
const MP_IOAPIC: u8 = 2;
const MP_INTSRC: u8 = 3;
const MP_LINTSRC: u8 = 4;

// CPU flags
const CPU_ENABLED: u8 = 0x01;
const CPU_BSP: u8 = 0x02;

// Versions/constants
const APIC_VERSION: u8 = 0x14;
const IO_APIC_ID: u8 = 0xfe;
const IO_APIC_ADDR: u32 = 0xfec0_0000;

/// Write an MP table describing the given topology into guest memory.
/// Each CPU entry uses the topology-computed APIC ID so the kernel
/// sees the correct socket/core/thread structure.
pub fn setup_mptable(mem: &GuestMemoryMmap, topo: &Topology) -> Result<()> {
    let num_cpus = topo.total_cpus();
    let mut addr = GuestAddress(MPTABLE_START);

    // MP Floating Pointer Structure (16 bytes)
    let mpf_size = 16u64;
    let mpc_start = addr.raw_value() + mpf_size;

    let mut mpf = [0u8; 16];
    mpf[0..4].copy_from_slice(&SMP_MAGIC);
    // Physical address of MPC table
    mpf[4..8].copy_from_slice(&(mpc_start as u32).to_le_bytes());
    mpf[8] = 1; // length (in 16-byte units)
    mpf[9] = 4; // spec revision
    // feature1 = 0: custom MP configuration table present
    // feature2 bit 7: IMCR present → use APIC mode (required for SMP)
    mpf[12] = 0x80;
    // Checksum computed after all fields set
    let cksum = mpf.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
    mpf[10] = (!cksum).wrapping_add(1);
    mem.write_slice(&mpf, addr).context("write mpf")?;
    addr = addr.unchecked_add(mpf_size);

    // MPC Table Header (44 bytes)
    // We'll write the header last (need the total length)
    let header_addr = addr;
    let header_size = 44u64;
    addr = addr.unchecked_add(header_size);

    // CPU entries (20 bytes each)
    let cpu_entry_size = 20u64;
    for cpu_id in 0..num_cpus {
        // MP table spec uses 8-bit APIC IDs. For topologies with IDs > 255,
        // the kernel uses ACPI MADT (which handles x2APIC) as authoritative.
        let apic_id = apic_id(topo, cpu_id) as u8;
        let mut entry = [0u8; 20];
        entry[0] = MP_PROCESSOR;
        entry[1] = apic_id;
        entry[2] = APIC_VERSION;
        entry[3] = CPU_ENABLED | if cpu_id == 0 { CPU_BSP } else { 0 };
        // CPU signature (stepping)
        entry[4..8].copy_from_slice(&0x0600u32.to_le_bytes());
        // Feature flags (FPU + APIC)
        entry[8..12].copy_from_slice(&0x0201u32.to_le_bytes());
        mem.write_slice(&entry, addr).context("write mpc_cpu")?;
        addr = addr.unchecked_add(cpu_entry_size);
    }

    // Bus entry (8 bytes)
    let bus_entry_size = 8u64;
    let mut bus = [0u8; 8];
    bus[0] = MP_BUS;
    bus[1] = 0; // bus ID
    bus[2..8].copy_from_slice(b"ISA   ");
    mem.write_slice(&bus, addr).context("write mpc_bus")?;
    addr = addr.unchecked_add(bus_entry_size);

    // IOAPIC entry (8 bytes)
    let ioapic_entry_size = 8u64;
    let mut ioapic = [0u8; 8];
    ioapic[0] = MP_IOAPIC;
    ioapic[1] = IO_APIC_ID;
    ioapic[2] = APIC_VERSION;
    ioapic[3] = 0x01; // enabled
    ioapic[4..8].copy_from_slice(&IO_APIC_ADDR.to_le_bytes());
    mem.write_slice(&ioapic, addr).context("write mpc_ioapic")?;
    addr = addr.unchecked_add(ioapic_entry_size);

    // Interrupt source entries (8 bytes each) — 24 legacy GSI IRQs (0..23)
    const NUM_IRQS: u32 = 24;
    let intsrc_entry_size = 8u64;
    for irq in 0u8..NUM_IRQS as u8 {
        let mut intsrc = [0u8; 8];
        intsrc[0] = MP_INTSRC;
        intsrc[1] = 0; // INT type
        intsrc[2] = 0; // flags
        intsrc[3] = 0;
        intsrc[4] = 0; // bus ID
        intsrc[5] = irq; // bus IRQ
        intsrc[6] = IO_APIC_ID; // dest APIC
        intsrc[7] = irq; // dest APIC INTIN
        mem.write_slice(&intsrc, addr).context("write mpc_intsrc")?;
        addr = addr.unchecked_add(intsrc_entry_size);
    }

    // Local interrupt source entries (8 bytes each) — LINT0 + LINT1
    let lintsrc_entry_size = 8u64;
    for lint in 0u8..2 {
        let mut lintsrc = [0u8; 8];
        lintsrc[0] = MP_LINTSRC;
        lintsrc[1] = if lint == 0 { 3 } else { 1 }; // ExtINT or NMI
        lintsrc[6] = 0xff; // dest APIC (all)
        lintsrc[7] = lint; // dest APIC LINTIN
        mem.write_slice(&lintsrc, addr)
            .context("write mpc_lintsrc")?;
        addr = addr.unchecked_add(lintsrc_entry_size);
    }

    // Now write the MPC table header
    let table_end = addr;
    let table_len = (table_end.raw_value() - header_addr.raw_value() - header_size) as u16;

    let mut header = [0u8; 44];
    header[0..4].copy_from_slice(&MPC_MAGIC);
    // Base table length (includes header + entries)
    let total_len = (header_size as u16) + table_len;
    header[4..6].copy_from_slice(&total_len.to_le_bytes());
    header[6] = 4; // spec revision
    // checksum at [7] — computed last
    header[8..16].copy_from_slice(b"KTSTR   "); // OEM ID
    header[16..28].copy_from_slice(b"000000000000"); // product ID
    // OEM table pointer [28..32] = 0
    // OEM table size [32..34] = 0
    let entry_count = num_cpus + 1 + 1 + NUM_IRQS + 2; // cpus + bus + ioapic + intsrcs + lintsrcs
    header[34..36].copy_from_slice(&(entry_count as u16).to_le_bytes());
    // Local APIC address
    header[36..40].copy_from_slice(&0xfee0_0000u32.to_le_bytes());
    // Extended table length [40..42] = 0
    // Extended table checksum [42] = 0

    // Compute header checksum
    // Need to include all entries in the checksum
    let entries_start = header_addr.unchecked_add(header_size);
    let entries_len = (table_end.raw_value() - entries_start.raw_value()) as usize;
    let mut entry_bytes = vec![0u8; entries_len];
    mem.read_slice(&mut entry_bytes, entries_start)
        .context("read entries for checksum")?;

    let mut cksum: u8 = 0;
    for &b in &header {
        cksum = cksum.wrapping_add(b);
    }
    for &b in &entry_bytes {
        cksum = cksum.wrapping_add(b);
    }
    header[7] = (!cksum).wrapping_add(1);

    mem.write_slice(&header, header_addr)
        .context("write mpc_table header")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_mem(mb: u32) -> GuestMemoryMmap {
        GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), (mb as usize) << 20)]).unwrap()
    }

    #[test]
    fn mptable_single_cpu() {
        let mem = test_mem(16);
        let topo = Topology {
            sockets: 1,
            cores_per_socket: 1,
            threads_per_core: 1,
        };
        setup_mptable(&mem, &topo).unwrap();
        // Verify MP floating pointer magic
        let mut magic = [0u8; 4];
        mem.read_slice(&mut magic, GuestAddress(MPTABLE_START))
            .unwrap();
        assert_eq!(&magic, b"_MP_");
    }

    #[test]
    fn mptable_multi_socket() {
        let mem = test_mem(16);
        let topo = Topology {
            sockets: 2,
            cores_per_socket: 2,
            threads_per_core: 2,
        };
        setup_mptable(&mem, &topo).unwrap();
        let mut magic = [0u8; 4];
        mem.read_slice(&mut magic, GuestAddress(MPTABLE_START))
            .unwrap();
        assert_eq!(&magic, b"_MP_");

        // Verify MPC table magic
        let mut mpc_magic = [0u8; 4];
        mem.read_slice(&mut mpc_magic, GuestAddress(MPTABLE_START + 16))
            .unwrap();
        assert_eq!(&mpc_magic, b"PCMP");
    }

    #[test]
    fn mptable_mpf_checksum() {
        let mem = test_mem(16);
        let topo = Topology {
            sockets: 1,
            cores_per_socket: 2,
            threads_per_core: 1,
        };
        setup_mptable(&mem, &topo).unwrap();
        let mut mpf = [0u8; 16];
        mem.read_slice(&mut mpf, GuestAddress(MPTABLE_START))
            .unwrap();
        let sum: u8 = mpf.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
        assert_eq!(sum, 0, "MPF checksum must be zero");
    }

    #[test]
    fn mptable_header_checksum() {
        let mem = test_mem(16);
        let topo = Topology {
            sockets: 1,
            cores_per_socket: 4,
            threads_per_core: 1,
        };
        setup_mptable(&mem, &topo).unwrap();

        // Read header to get table length
        let header_addr = GuestAddress(MPTABLE_START + 16);
        let mut len_bytes = [0u8; 2];
        mem.read_slice(&mut len_bytes, header_addr.unchecked_add(4))
            .unwrap();
        let table_len = u16::from_le_bytes(len_bytes) as usize;

        // Read entire table and verify checksum
        let mut table = vec![0u8; table_len];
        mem.read_slice(&mut table, header_addr).unwrap();
        let sum: u8 = table.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
        assert_eq!(sum, 0, "MPC table checksum must be zero");
    }

    #[test]
    fn mptable_cpu_apic_ids_match_topology() {
        let mem = test_mem(16);
        let topo = Topology {
            sockets: 2,
            cores_per_socket: 2,
            threads_per_core: 1,
        };
        setup_mptable(&mem, &topo).unwrap();

        // CPU entries start at MPTABLE_START + 16 (mpf) + 44 (header)
        let cpu_start = GuestAddress(MPTABLE_START + 16 + 44);
        for i in 0..topo.total_cpus() {
            let entry_addr = cpu_start.unchecked_add(i as u64 * 20);
            let mut entry = [0u8; 20];
            mem.read_slice(&mut entry, entry_addr).unwrap();
            assert_eq!(entry[0], MP_PROCESSOR, "entry type should be processor");
            let id = entry[1];
            let expected = apic_id(&topo, i) as u8;
            assert_eq!(id, expected, "CPU {i}: APIC ID {id} != expected {expected}");
        }
    }

    #[test]
    fn mptable_bsp_flagged() {
        let mem = test_mem(16);
        let topo = Topology {
            sockets: 2,
            cores_per_socket: 2,
            threads_per_core: 1,
        };
        setup_mptable(&mem, &topo).unwrap();

        let cpu_start = GuestAddress(MPTABLE_START + 16 + 44);
        // CPU 0 should be BSP
        let mut entry0 = [0u8; 20];
        mem.read_slice(&mut entry0, cpu_start).unwrap();
        assert_ne!(entry0[3] & CPU_BSP, 0, "CPU 0 should be BSP");
        assert_ne!(entry0[3] & CPU_ENABLED, 0, "CPU 0 should be enabled");

        // CPU 1 should not be BSP
        let mut entry1 = [0u8; 20];
        mem.read_slice(&mut entry1, cpu_start.unchecked_add(20))
            .unwrap();
        assert_eq!(entry1[3] & CPU_BSP, 0, "CPU 1 should not be BSP");
        assert_ne!(entry1[3] & CPU_ENABLED, 0, "CPU 1 should be enabled");
    }

    #[test]
    fn mptable_large_topology_240_cpus() {
        let mem = test_mem(2048);
        let topo = Topology {
            sockets: 15,
            cores_per_socket: 8,
            threads_per_core: 2,
        };
        assert_eq!(topo.total_cpus(), 240);
        // Should succeed — MP table uses u8 APIC IDs but max here is < 255
        assert!(setup_mptable(&mem, &topo).is_ok());
    }

    #[test]
    fn mptable_large_topology() {
        let mem = test_mem(4096);
        let topo = Topology {
            sockets: 14,
            cores_per_socket: 9,
            threads_per_core: 2,
        };
        assert_eq!(topo.total_cpus(), 252);
        assert!(setup_mptable(&mem, &topo).is_ok());
    }
}
