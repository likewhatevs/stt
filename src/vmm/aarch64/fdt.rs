use anyhow::{Context, Result};
use vm_fdt::{FdtReserveEntry, FdtWriter};

use crate::vmm::aarch64::topology::mpidr_to_fdt_reg;
use crate::vmm::kvm::{
    DRAM_START, FDT_MAX_SIZE, GIC_DIST_BASE, GIC_DIST_SIZE, GIC_REDIST_BASE,
    GIC_REDIST_SIZE_PER_CPU, SERIAL_IRQ, SERIAL_MMIO_BASE, SERIAL_MMIO_SIZE, SERIAL2_IRQ,
    SERIAL2_MMIO_BASE, VIRTIO_CONSOLE_IRQ, VIRTIO_CONSOLE_MMIO_BASE,
};
use crate::vmm::virtio_console;

/// GIC phandle — unique identifier referenced by interrupt-parent properties.
const GIC_PHANDLE: u32 = 1;

/// SPI interrupt type (shared peripheral interrupt).
const GIC_SPI: u32 = 0;

/// PPI interrupt type (private peripheral interrupt).
const GIC_PPI: u32 = 1;

/// IRQ_TYPE_EDGE_RISING for SPI devices driven by irqfd.
/// Edge-triggered avoids the need for resamplefd: KVM sets
/// pending_latch on injection, auto-clears after delivery.
const IRQ_TYPE_EDGE_RISING: u32 = 1;

/// IRQ_TYPE_LEVEL_LOW for timer PPIs (active-low per arm spec).
const IRQ_TYPE_LEVEL_LOW: u32 = 8;

/// Generate a Flattened Device Tree blob for the guest.
///
/// `mpidrs` contains the MPIDR_EL1 values read from KVM for each vCPU.
/// The FDT cpu node `reg` properties use the affinity fields from these
/// values, ensuring the FDT matches KVM's actual MPIDR assignment.
pub fn create_fdt(
    mpidrs: &[u64],
    memory_mb: u32,
    cmdline: &str,
    initrd_addr: Option<u64>,
    initrd_size: Option<u32>,
    shm_size: u64,
) -> Result<Vec<u8>> {
    // SHM base address (top of guest DRAM minus SHM size).
    let shm_base = if shm_size > 0 {
        let mem_size = (memory_mb as u64) << 20;
        Some(DRAM_START + mem_size - shm_size)
    } else {
        None
    };

    // Reserve the SHM region via /memreserve/ so the kernel adds it to
    // memblock.reserved (preventing allocation) while the full DRAM range
    // in the /memory node keeps it in memblock.memory (ensuring /dev/mem
    // maps it with Normal cacheable attributes, not Device-nGnRnE).
    let reserves = if let Some(base) = shm_base {
        vec![FdtReserveEntry::new(base, shm_size).context("SHM reserve entry")?]
    } else {
        vec![]
    };
    let mut fdt = FdtWriter::new_with_mem_reserv(&reserves).context("create FDT writer")?;

    let root = fdt.begin_node("").context("begin root node")?;
    fdt.property_string("compatible", "linux,dummy-virt")
        .context("root compatible")?;
    fdt.property_u32("#address-cells", 2)
        .context("root #address-cells")?;
    fdt.property_u32("#size-cells", 2)
        .context("root #size-cells")?;
    fdt.property_u32("interrupt-parent", GIC_PHANDLE)
        .context("root interrupt-parent")?;

    // /chosen — bootargs, stdout, initrd
    write_chosen(&mut fdt, cmdline, initrd_addr, initrd_size)?;

    // /memory — guest physical RAM
    write_memory(&mut fdt, memory_mb)?;

    // /reserved-memory — SHM region marked reserved (no kernel
    // allocation) but kept in memblock.memory so /dev/mem maps it
    // with Normal cacheable attributes. The /memory node above
    // covers full DRAM including SHM — that is what makes
    // pfn_is_map_memory() return true for SHM pages.
    if let Some(base) = shm_base {
        write_reserved_memory(&mut fdt, base, shm_size)?;
    }

    // /cpus — one node per vCPU with MPIDR from KVM
    write_cpus(&mut fdt, mpidrs)?;

    // /intc — GICv3
    let num_cpus = mpidrs.len() as u32;
    write_gic(&mut fdt, num_cpus)?;

    // /serial — two ns16550a UARTs with edge-triggered SPI interrupts via irqfd
    write_serial(&mut fdt, SERIAL_MMIO_BASE, "serial0", SERIAL_IRQ)?;
    write_serial(&mut fdt, SERIAL2_MMIO_BASE, "serial1", SERIAL2_IRQ)?;

    // /virtio_mmio — virtio-console
    write_virtio_mmio(
        &mut fdt,
        VIRTIO_CONSOLE_MMIO_BASE,
        virtio_console::VIRTIO_MMIO_SIZE,
        VIRTIO_CONSOLE_IRQ,
    )?;

    // /timer — arm generic timer
    write_timer(&mut fdt)?;

    // /psci — power state coordination interface
    write_psci(&mut fdt)?;

    fdt.end_node(root).context("end root node")?;
    let dtb = fdt.finish().context("finish FDT")?;

    anyhow::ensure!(
        dtb.len() as u64 <= FDT_MAX_SIZE,
        "FDT too large: {} bytes (max {})",
        dtb.len(),
        FDT_MAX_SIZE,
    );

    Ok(dtb)
}

/// Compute FDT load address: placed at the end of usable guest RAM.
///
/// The FDT is placed in the last `FDT_MAX_SIZE` bytes of the usable
/// DRAM region. The address must be 8-byte aligned (FDT spec requirement).
pub fn fdt_address(memory_mb: u32, shm_size: u64) -> u64 {
    let mem_size = (memory_mb as u64) << 20;
    let dram_end = DRAM_START + mem_size;
    let usable_end = dram_end - shm_size;
    (usable_end - FDT_MAX_SIZE) & !7
}

fn write_chosen(
    fdt: &mut FdtWriter,
    cmdline: &str,
    initrd_addr: Option<u64>,
    initrd_size: Option<u32>,
) -> Result<()> {
    let chosen = fdt.begin_node("chosen").context("begin chosen")?;
    fdt.property_string("bootargs", cmdline)
        .context("bootargs")?;
    fdt.property_string("stdout-path", &format!("/serial0@{:x}", SERIAL_MMIO_BASE))
        .context("stdout-path")?;

    if let (Some(addr), Some(size)) = (initrd_addr, initrd_size) {
        fdt.property_u64("linux,initrd-start", addr)
            .context("initrd-start")?;
        fdt.property_u64("linux,initrd-end", addr + size as u64)
            .context("initrd-end")?;
    }

    fdt.end_node(chosen).context("end chosen")?;
    Ok(())
}

fn write_memory(fdt: &mut FdtWriter, memory_mb: u32) -> Result<()> {
    // Full DRAM range including SHM. arm64 phys_mem_access_prot()
    // maps addresses outside memblock.memory as Device-nGnRnE, which
    // faults on paired loads (memcpy STP). Including SHM here puts
    // its pages in memblock.memory so /dev/mem maps them cacheable.
    // /reserved-memory prevents kernel allocation from the SHM region.
    let mem_size = (memory_mb as u64) << 20;

    let name = format!("memory@{DRAM_START:x}");
    let mem = fdt.begin_node(&name).context("begin memory")?;
    fdt.property_string("device_type", "memory")
        .context("memory device_type")?;
    // reg = <addr_hi addr_lo size_hi size_lo>
    fdt.property_array_u32(
        "reg",
        &[
            (DRAM_START >> 32) as u32,
            DRAM_START as u32,
            (mem_size >> 32) as u32,
            mem_size as u32,
        ],
    )
    .context("memory reg")?;
    fdt.end_node(mem).context("end memory")?;
    Ok(())
}

fn write_reserved_memory(fdt: &mut FdtWriter, shm_base: u64, shm_size: u64) -> Result<()> {
    let rsv = fdt
        .begin_node("reserved-memory")
        .context("begin reserved-memory")?;
    fdt.property_u32("#address-cells", 2)
        .context("reserved-memory #address-cells")?;
    fdt.property_u32("#size-cells", 2)
        .context("reserved-memory #size-cells")?;
    fdt.property_null("ranges")
        .context("reserved-memory ranges")?;

    // SHM child node: reserved but NOT no-map. Without no-map the
    // kernel keeps the region in memblock.memory (linear map) so
    // /dev/mem maps it with Normal cacheable attributes.
    let name = format!("shm@{shm_base:x}");
    let shm = fdt.begin_node(&name).context("begin shm node")?;
    fdt.property_array_u32(
        "reg",
        &[
            (shm_base >> 32) as u32,
            shm_base as u32,
            (shm_size >> 32) as u32,
            shm_size as u32,
        ],
    )
    .context("shm reg")?;
    fdt.end_node(shm).context("end shm node")?;

    fdt.end_node(rsv).context("end reserved-memory")?;
    Ok(())
}

fn write_cpus(fdt: &mut FdtWriter, mpidrs: &[u64]) -> Result<()> {
    let cpus = fdt.begin_node("cpus").context("begin cpus")?;
    fdt.property_u32("#address-cells", 1)
        .context("cpus #address-cells")?;
    fdt.property_u32("#size-cells", 0)
        .context("cpus #size-cells")?;

    for (cpu_id, &mpidr) in mpidrs.iter().enumerate() {
        let reg = mpidr_to_fdt_reg(mpidr) as u32;
        let name = format!("cpu@{cpu_id}");
        let cpu = fdt.begin_node(&name).context("begin cpu")?;
        fdt.property_string("device_type", "cpu")
            .context("cpu device_type")?;
        fdt.property_string("compatible", "arm,arm-v8")
            .context("cpu compatible")?;
        fdt.property_string("enable-method", "psci")
            .context("cpu enable-method")?;
        fdt.property_u32("reg", reg).context("cpu reg")?;
        fdt.end_node(cpu).context("end cpu")?;
    }

    fdt.end_node(cpus).context("end cpus")?;
    Ok(())
}

fn write_gic(fdt: &mut FdtWriter, num_cpus: u32) -> Result<()> {
    let redist_size = num_cpus as u64 * GIC_REDIST_SIZE_PER_CPU;

    let intc = fdt
        .begin_node(&format!("intc@{GIC_DIST_BASE:x}"))
        .context("begin intc")?;
    fdt.property_string("compatible", "arm,gic-v3")
        .context("intc compatible")?;
    fdt.property_null("interrupt-controller")
        .context("interrupt-controller")?;
    fdt.property_u32("#interrupt-cells", 3)
        .context("#interrupt-cells")?;
    fdt.property_phandle(GIC_PHANDLE).context("intc phandle")?;
    // reg: distributor region, then redistributor region
    fdt.property_array_u32(
        "reg",
        &[
            (GIC_DIST_BASE >> 32) as u32,
            GIC_DIST_BASE as u32,
            (GIC_DIST_SIZE >> 32) as u32,
            GIC_DIST_SIZE as u32,
            (GIC_REDIST_BASE >> 32) as u32,
            GIC_REDIST_BASE as u32,
            (redist_size >> 32) as u32,
            redist_size as u32,
        ],
    )
    .context("intc reg")?;
    fdt.property_u32("#address-cells", 2)
        .context("intc #address-cells")?;
    fdt.property_u32("#size-cells", 2)
        .context("intc #size-cells")?;
    fdt.property_null("ranges").context("intc ranges")?;

    fdt.end_node(intc).context("end intc")?;
    Ok(())
}

fn write_serial(fdt: &mut FdtWriter, base: u64, alias: &str, irq: u32) -> Result<()> {
    let name = format!("{alias}@{base:x}");
    let serial = fdt.begin_node(&name).context("begin serial")?;
    fdt.property_string("compatible", "ns16550a")
        .context("serial compatible")?;
    fdt.property_array_u32(
        "reg",
        &[
            (base >> 32) as u32,
            base as u32,
            (SERIAL_MMIO_SIZE >> 32) as u32,
            SERIAL_MMIO_SIZE as u32,
        ],
    )
    .context("serial reg")?;
    // Edge-triggered SPI: irqfd writes the eventfd once per interrupt,
    // KVM sets pending_latch and auto-clears after delivery. No
    // resamplefd needed. FDT cell 1 is the SPI number (intid - 32).
    fdt.property_array_u32("interrupts", &[GIC_SPI, irq - 32, IRQ_TYPE_EDGE_RISING])
        .context("serial interrupts")?;
    fdt.property_u32("interrupt-parent", GIC_PHANDLE)
        .context("serial interrupt-parent")?;
    fdt.property_u32("clock-frequency", 1843200)
        .context("serial clock-frequency")?;
    fdt.property_u32("reg-shift", 0)
        .context("serial reg-shift")?;
    fdt.property_u32("reg-io-width", 1)
        .context("serial reg-io-width")?;
    fdt.end_node(serial).context("end serial")?;
    Ok(())
}

fn write_virtio_mmio(fdt: &mut FdtWriter, base: u64, size: u64, irq: u32) -> Result<()> {
    let name = format!("virtio_mmio@{base:x}");
    let node = fdt.begin_node(&name).context("begin virtio_mmio")?;
    fdt.property_string("compatible", "virtio,mmio")
        .context("virtio_mmio compatible")?;
    fdt.property_array_u32(
        "reg",
        &[
            (base >> 32) as u32,
            base as u32,
            (size >> 32) as u32,
            size as u32,
        ],
    )
    .context("virtio_mmio reg")?;
    fdt.property_array_u32("interrupts", &[GIC_SPI, irq - 32, IRQ_TYPE_EDGE_RISING])
        .context("virtio_mmio interrupts")?;
    fdt.property_u32("interrupt-parent", GIC_PHANDLE)
        .context("virtio_mmio interrupt-parent")?;
    fdt.end_node(node).context("end virtio_mmio")?;
    Ok(())
}

fn write_timer(fdt: &mut FdtWriter) -> Result<()> {
    let timer = fdt.begin_node("timer").context("begin timer")?;
    fdt.property_string("compatible", "arm,armv8-timer")
        .context("timer compatible")?;
    fdt.property_null("always-on").context("timer always-on")?;
    // Four PPI interrupts: secure phys, non-secure phys, virt, hyp phys.
    // Standard values from QEMU/Firecracker virt machine.
    fdt.property_array_u32(
        "interrupts",
        &[
            GIC_PPI,
            13,
            IRQ_TYPE_LEVEL_LOW, // secure physical timer
            GIC_PPI,
            14,
            IRQ_TYPE_LEVEL_LOW, // non-secure physical timer
            GIC_PPI,
            11,
            IRQ_TYPE_LEVEL_LOW, // virtual timer
            GIC_PPI,
            10,
            IRQ_TYPE_LEVEL_LOW, // hypervisor physical timer
        ],
    )
    .context("timer interrupts")?;
    fdt.end_node(timer).context("end timer")?;
    Ok(())
}

fn write_psci(fdt: &mut FdtWriter) -> Result<()> {
    let psci = fdt.begin_node("psci").context("begin psci")?;
    fdt.property_string("compatible", "arm,psci-0.2")
        .context("psci compatible")?;
    fdt.property_string("method", "hvc")
        .context("psci method")?;
    fdt.end_node(psci).context("end psci")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vmm::topology::Topology;

    fn default_topo() -> Topology {
        Topology {
            sockets: 1,
            cores_per_socket: 2,
            threads_per_core: 1,
        }
    }

    /// Generate fake MPIDRs for testing (bit 31 set, linear Aff0).
    fn fake_mpidrs(count: u32) -> Vec<u64> {
        (0..count).map(|i| (1u64 << 31) | i as u64).collect()
    }

    #[test]
    fn create_fdt_minimal() {
        let topo = default_topo();
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let dtb = create_fdt(&mpidrs, 256, "console=ttyS0", None, None, 0);
        assert!(dtb.is_ok(), "FDT creation failed: {:?}", dtb.err());
        let dtb = dtb.unwrap();
        assert_eq!(&dtb[..4], &[0xd0, 0x0d, 0xfe, 0xed]);
    }

    #[test]
    fn create_fdt_with_initrd() {
        let topo = default_topo();
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let dtb = create_fdt(
            &mpidrs,
            256,
            "console=ttyS0",
            Some(0x4020_0000),
            Some(0x10_0000),
            0,
        );
        assert!(dtb.is_ok());
    }

    #[test]
    fn create_fdt_with_shm() {
        let topo = default_topo();
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let dtb = create_fdt(&mpidrs, 256, "console=ttyS0", None, None, 0x10_0000);
        assert!(dtb.is_ok());
    }

    #[test]
    fn create_fdt_smp() {
        let topo = Topology {
            sockets: 2,
            cores_per_socket: 4,
            threads_per_core: 2,
        };
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let dtb = create_fdt(&mpidrs, 1024, "console=ttyS0", None, None, 0);
        assert!(dtb.is_ok());
    }

    #[test]
    fn fdt_address_aligned() {
        let addr = fdt_address(256, 0);
        assert_eq!(addr % 8, 0, "FDT address must be 8-byte aligned");
    }

    #[test]
    fn fdt_address_with_shm() {
        let addr_no_shm = fdt_address(256, 0);
        let addr_with_shm = fdt_address(256, 0x10_0000);
        assert!(addr_with_shm < addr_no_shm);
    }
}
