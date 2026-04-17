use anyhow::{Context, Result};
use vm_fdt::{FdtReserveEntry, FdtWriter};

use crate::vmm::aarch64::topology::mpidr_to_fdt_reg;
use crate::vmm::kvm::{
    DRAM_START, FDT_MAX_SIZE, GIC_DIST_BASE, GIC_DIST_SIZE, GIC_REDIST_BASE,
    GIC_REDIST_SIZE_PER_CPU, SERIAL_IRQ, SERIAL_MMIO_BASE, SERIAL_MMIO_SIZE, SERIAL2_IRQ,
    SERIAL2_MMIO_BASE, VIRTIO_CONSOLE_IRQ, VIRTIO_CONSOLE_MMIO_BASE,
};
use crate::vmm::topology::Topology;
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
///
/// When `topo.llcs > 1` and `hw_cache_level >= 2`, DT cache nodes
/// are emitted so the guest kernel discovers per-LLC cache domains
/// via `next-level-cache` phandle chains in `cache_setup_of_node`.
///
/// `guest_l1_unified` indicates the host's L1 cache is unified (from
/// sysfs). When true, CPU nodes get `cache-unified` so
/// `of_count_cache_leaves` returns 1 leaf instead of 2, matching
/// CLIDR_EL1's Ctype1=Unified (1 leaf).
///
/// When `topo.numa_nodes > 1`, NUMA topology is described via:
/// - `numa-node-id` properties on cpu and memory nodes
/// - per-NUMA-node memory nodes with disjoint address ranges
/// - a `distance-map` node with `numa-distance-map-v1` compatible
#[allow(clippy::too_many_arguments)]
pub fn create_fdt(
    topo: &Topology,
    mpidrs: &[u64],
    memory_mb: u32,
    cmdline: &str,
    initrd_addr: Option<u64>,
    initrd_size: Option<u32>,
    shm_size: u64,
    hw_cache_level: u32,
    guest_l1_unified: bool,
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

    // /memory — guest physical RAM. When numa_nodes > 1, one memory
    // node per NUMA node with disjoint address ranges and numa-node-id.
    write_memory(&mut fdt, topo, memory_mb, shm_size)?;

    // /reserved-memory — SHM region marked reserved (no kernel
    // allocation) but kept in memblock.memory so /dev/mem maps it
    // with Normal cacheable attributes. The /memory node(s) above
    // cover full DRAM including SHM — that is what makes
    // pfn_is_map_memory() return true for SHM pages.
    if let Some(base) = shm_base {
        write_reserved_memory(&mut fdt, base, shm_size)?;
    }

    // /cpus — one node per vCPU with MPIDR from KVM, plus cache
    // topology nodes when the topology has multiple LLCs.
    write_cpus(&mut fdt, topo, mpidrs, hw_cache_level, guest_l1_unified)?;

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

    // /distance-map — NUMA distance matrix (only for multi-NUMA)
    if topo.numa_nodes > 1 {
        write_distance_map(&mut fdt, topo)?;
    }

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

fn write_memory(fdt: &mut FdtWriter, topo: &Topology, memory_mb: u32, shm_size: u64) -> Result<()> {
    let mem_size = (memory_mb as u64) << 20;

    if topo.numa_nodes <= 1 {
        // Single-NUMA: one memory node covering full DRAM including SHM.
        // arm64 phys_mem_access_prot() maps addresses outside
        // memblock.memory as Device-nGnRnE, which faults on paired
        // loads (memcpy STP). Including SHM here puts its pages in
        // memblock.memory so /dev/mem maps them cacheable.
        // /reserved-memory prevents kernel allocation from the SHM region.
        let name = format!("memory@{DRAM_START:x}");
        let mem = fdt.begin_node(&name).context("begin memory")?;
        fdt.property_string("device_type", "memory")
            .context("memory device_type")?;
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
    } else {
        // Multi-NUMA: one memory node per NUMA node, each with
        // numa-node-id. When explicit per-node memory is configured,
        // use it; otherwise split evenly. The last node extends to
        // cover the SHM region. SHM must be in memblock.memory so
        // arm64 phys_mem_access_prot() maps it cacheable via
        // /dev/mem. /reserved-memory prevents kernel allocation
        // from SHM while keeping it in the linear map.
        let usable_bytes = mem_size - shm_size;
        let n = topo.numa_nodes as u64;
        let uniform_per_node = (usable_bytes / n) & !0xFFF; // page-align down (4 KiB)

        let mut base_offset: u64 = 0;
        for node in 0..topo.numa_nodes {
            let base = DRAM_START + base_offset;
            let length = match topo.node_memory_mb(node) {
                Some(mb) => {
                    let node_bytes = (mb as u64) << 20;
                    if node == topo.numa_nodes - 1 {
                        node_bytes + shm_size
                    } else {
                        node_bytes
                    }
                }
                None => {
                    if node == topo.numa_nodes - 1 {
                        mem_size - base_offset
                    } else {
                        uniform_per_node
                    }
                }
            };
            let name = format!("memory@{base:x}");
            let mem = fdt.begin_node(&name).context("begin memory")?;
            fdt.property_string("device_type", "memory")
                .context("memory device_type")?;
            fdt.property_array_u32(
                "reg",
                &[
                    (base >> 32) as u32,
                    base as u32,
                    (length >> 32) as u32,
                    length as u32,
                ],
            )
            .context("memory reg")?;
            fdt.property_u32("numa-node-id", node)
                .context("memory numa-node-id")?;
            fdt.end_node(mem).context("end memory")?;
            base_offset += length;
        }
    }

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

/// Phandle base for cache nodes. GIC uses phandle 1.
/// Grouped by LLC: each LLC's chain occupies `chain_depth` consecutive
/// phandles starting at `BASE + llc * chain_depth`.
const CACHE_PHANDLE_BASE: u32 = GIC_PHANDLE + 1;

fn write_cpus(
    fdt: &mut FdtWriter,
    topo: &Topology,
    mpidrs: &[u64],
    hw_cache_level: u32,
    guest_l1_unified: bool,
) -> Result<()> {
    let cpus = fdt.begin_node("cpus").context("begin cpus")?;
    fdt.property_u32("#address-cells", 1)
        .context("cpus #address-cells")?;
    fdt.property_u32("#size-cells", 0)
        .context("cpus #size-cells")?;

    // cache_setup_of_node() walks next-level-cache once per non-L1
    // hardware cache level. With N levels from CLIDR_EL1, the chain
    // needs N-1 hops (L1 leaves stay at the CPU node). Each LLC gets
    // its own chain so CPUs sharing an LLC share the same phandles.
    // cache_leaves_are_shared() compares fw_token pointers set by
    // cache_setup_of_node() — shared phandles produce shared IDs.
    let chain_depth = if topo.llcs > 1 && hw_cache_level >= 2 {
        (hw_cache_level - 1) as usize
    } else {
        0
    };

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
        let (llc_id, _, _) = topo.decompose(cpu_id as u32);
        if topo.numa_nodes > 1 {
            let node_id = topo.numa_node_of(llc_id);
            fdt.property_u32("numa-node-id", node_id)
                .context("cpu numa-node-id")?;
        }
        if chain_depth > 0 {
            let first_phandle = CACHE_PHANDLE_BASE + llc_id * chain_depth as u32;
            fdt.property_u32("next-level-cache", first_phandle)
                .context("cpu next-level-cache")?;
            // When the host L1 is unified (single leaf in CLIDR),
            // of_count_cache_leaves defaults to 2 (separate I/D).
            // cache-unified reduces the OF count to 1 to match CLIDR.
            if guest_l1_unified {
                fdt.property_null("cache-unified")
                    .context("cpu cache-unified")?;
            }
        }
        fdt.end_node(cpu).context("end cpu")?;
    }

    // Cache node chains: for each LLC, create `chain_depth` nodes
    // at levels 2..=hw_cache_level. Each non-terminal node chains
    // to the next via next-level-cache. The terminal node is the
    // LLC boundary — CPUs sharing it are in the same LLC domain.
    for llc in 0..topo.llcs {
        for d in 0..chain_depth {
            let phandle = CACHE_PHANDLE_BASE + llc * chain_depth as u32 + d as u32;
            let level = (d + 2) as u32;
            let name = format!("l{level}-cache{llc}");
            let cache = fdt.begin_node(&name).context("begin cache")?;
            fdt.property_string("compatible", "cache")
                .context("cache compatible")?;
            fdt.property_u32("cache-level", level)
                .context("cache-level")?;
            fdt.property_null("cache-unified")
                .context("cache-unified")?;
            fdt.property_phandle(phandle).context("cache phandle")?;
            if d + 1 < chain_depth {
                fdt.property_u32("next-level-cache", phandle + 1)
                    .context("cache next-level-cache")?;
            }
            fdt.end_node(cache).context("end cache")?;
        }
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

/// Write a `distance-map` node with `numa-distance-map-v1` compatible.
///
/// The kernel parses `distance-matrix` as a flat array of (nodea, nodeb,
/// distance) triples. Distances come from `topo.distance()`, defaulting
/// to 10 (local) / 20 (remote).
fn write_distance_map(fdt: &mut FdtWriter, topo: &Topology) -> Result<()> {
    let dm = fdt
        .begin_node("distance-map")
        .context("begin distance-map")?;
    fdt.property_string("compatible", "numa-distance-map-v1")
        .context("distance-map compatible")?;

    // Build flat (nodea, nodeb, distance) triples for the full NxN matrix.
    let n = topo.numa_nodes;
    let mut matrix = Vec::with_capacity((n * n * 3) as usize);
    for i in 0..n {
        for j in 0..n {
            matrix.push(i);
            matrix.push(j);
            matrix.push(topo.distance(i, j) as u32);
        }
    }
    fdt.property_array_u32("distance-matrix", &matrix)
        .context("distance-matrix")?;

    fdt.end_node(dm).context("end distance-map")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_topo() -> Topology {
        Topology {
            llcs: 1,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
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
        let dtb = create_fdt(
            &topo,
            &mpidrs,
            256,
            "console=ttyS0",
            None,
            None,
            0,
            0,
            false,
        );
        assert!(dtb.is_ok(), "FDT creation failed: {:?}", dtb.err());
        let dtb = dtb.unwrap();
        assert_eq!(&dtb[..4], &[0xd0, 0x0d, 0xfe, 0xed]);
    }

    #[test]
    fn create_fdt_with_initrd() {
        let topo = default_topo();
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let dtb = create_fdt(
            &topo,
            &mpidrs,
            256,
            "console=ttyS0",
            Some(0x4020_0000),
            Some(0x10_0000),
            0,
            0,
            false,
        );
        assert!(dtb.is_ok());
    }

    #[test]
    fn create_fdt_with_shm() {
        let topo = default_topo();
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let dtb = create_fdt(
            &topo,
            &mpidrs,
            256,
            "console=ttyS0",
            None,
            None,
            0x10_0000,
            0,
            false,
        );
        assert!(dtb.is_ok());
    }

    #[test]
    fn create_fdt_smp() {
        let topo = Topology {
            llcs: 2,
            cores_per_llc: 4,
            threads_per_core: 2,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let dtb = create_fdt(
            &topo,
            &mpidrs,
            1024,
            "console=ttyS0",
            None,
            None,
            0,
            2,
            false,
        );
        assert!(dtb.is_ok());
    }

    #[test]
    fn create_fdt_multi_numa() {
        let topo = Topology {
            llcs: 4,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 2,
            nodes: None,
            distances: None,
        };
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let dtb = create_fdt(
            &topo,
            &mpidrs,
            512,
            "console=ttyS0",
            None,
            None,
            0,
            2,
            false,
        );
        assert!(dtb.is_ok(), "FDT creation failed: {:?}", dtb.err());
    }

    #[test]
    fn create_fdt_multi_numa_with_shm() {
        let topo = Topology {
            llcs: 4,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 2,
            nodes: None,
            distances: None,
        };
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let dtb = create_fdt(
            &topo,
            &mpidrs,
            512,
            "console=ttyS0",
            None,
            None,
            0x10_0000,
            2,
            false,
        );
        assert!(dtb.is_ok());
    }

    #[test]
    fn create_fdt_three_numa_nodes() {
        let topo = Topology {
            llcs: 6,
            cores_per_llc: 4,
            threads_per_core: 2,
            numa_nodes: 3,
            nodes: None,
            distances: None,
        };
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let dtb = create_fdt(
            &topo,
            &mpidrs,
            1024,
            "console=ttyS0",
            None,
            None,
            0,
            2,
            false,
        );
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

    // -----------------------------------------------------------------------
    // Minimal DTB parser for content validation tests.
    //
    // FDT binary format (big-endian):
    //   Header: magic (0xd00dfeed), totalsize, off_dt_struct, off_dt_strings, ...
    //   Structure block: stream of tokens:
    //     FDT_BEGIN_NODE (1): u32 token, null-terminated name, pad to 4-byte
    //     FDT_END_NODE   (2): u32 token
    //     FDT_PROP       (3): u32 token, u32 len, u32 nameoff, [len bytes data], pad
    //     FDT_NOP        (4): u32 token
    //     FDT_END        (9): u32 token
    // -----------------------------------------------------------------------

    const FDT_MAGIC: u32 = 0xd00dfeed;
    const FDT_BEGIN_NODE: u32 = 1;
    const FDT_END_NODE: u32 = 2;
    const FDT_PROP: u32 = 3;
    const FDT_END: u32 = 9;

    fn read_be32(dtb: &[u8], off: usize) -> u32 {
        u32::from_be_bytes(dtb[off..off + 4].try_into().unwrap())
    }

    /// Walk the DTB structure block and collect (node_path, prop_name, prop_data)
    /// tuples. Only descends into nodes; does not interpret property values.
    fn parse_dtb_props(dtb: &[u8]) -> Vec<(String, String, Vec<u8>)> {
        assert_eq!(read_be32(dtb, 0), FDT_MAGIC, "not a valid DTB");
        let off_struct = read_be32(dtb, 8) as usize;
        let off_strings = read_be32(dtb, 12) as usize;

        let mut pos = off_struct;
        let mut path_stack: Vec<String> = Vec::new();
        let mut results = Vec::new();

        loop {
            let token = read_be32(dtb, pos);
            pos += 4;
            match token {
                FDT_BEGIN_NODE => {
                    // Read null-terminated node name.
                    let name_start = pos;
                    while dtb[pos] != 0 {
                        pos += 1;
                    }
                    let name = std::str::from_utf8(&dtb[name_start..pos])
                        .unwrap()
                        .to_string();
                    pos += 1; // skip null
                    pos = (pos + 3) & !3; // align to 4
                    // Skip the root node (empty name) to avoid a leading
                    // "/" separator in join()-ed paths.
                    if !name.is_empty() {
                        path_stack.push(name);
                    }
                }
                FDT_END_NODE => {
                    path_stack.pop();
                }
                FDT_PROP => {
                    let len = read_be32(dtb, pos) as usize;
                    pos += 4;
                    let nameoff = read_be32(dtb, pos) as usize;
                    pos += 4;
                    let data = dtb[pos..pos + len].to_vec();
                    pos += len;
                    pos = (pos + 3) & !3; // align to 4

                    // Read property name from strings table.
                    let str_start = off_strings + nameoff;
                    let mut str_end = str_start;
                    while dtb[str_end] != 0 {
                        str_end += 1;
                    }
                    let prop_name = std::str::from_utf8(&dtb[str_start..str_end])
                        .unwrap()
                        .to_string();

                    let node_path = path_stack.join("/");
                    results.push((node_path, prop_name, data));
                }
                FDT_END => break,
                _ => {} // FDT_NOP or unknown — skip
            }
        }
        results
    }

    /// Extract a u32 property value from parsed props.
    fn prop_u32(props: &[(String, String, Vec<u8>)], node: &str, name: &str) -> Option<u32> {
        props.iter().find_map(|(n, p, d)| {
            if n == node && p == name && d.len() == 4 {
                Some(u32::from_be_bytes(d[..4].try_into().unwrap()))
            } else {
                None
            }
        })
    }

    /// Extract a u32-array property from parsed props.
    fn prop_u32_array(
        props: &[(String, String, Vec<u8>)],
        node: &str,
        name: &str,
    ) -> Option<Vec<u32>> {
        props.iter().find_map(|(n, p, d)| {
            if n == node && p == name && d.len() % 4 == 0 {
                Some(
                    d.chunks_exact(4)
                        .map(|c| u32::from_be_bytes(c.try_into().unwrap()))
                        .collect(),
                )
            } else {
                None
            }
        })
    }

    #[test]
    fn parse_dtb_props_paths_no_leading_slash() {
        let topo = default_topo();
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let dtb = create_fdt(
            &topo,
            &mpidrs,
            256,
            "console=ttyS0",
            None,
            None,
            0,
            0,
            false,
        )
        .unwrap();
        let props = parse_dtb_props(&dtb);

        // Top-level node paths must not start with "/".
        let cpus_prop = props
            .iter()
            .find(|(n, p, _)| n == "cpus" && p == "#address-cells");
        assert!(cpus_prop.is_some(), "expected path 'cpus', not '/cpus'");

        // Nested node paths use "/" as separator without leading slash.
        let cpu0_prop = props
            .iter()
            .find(|(n, p, _)| n == "cpus/cpu@0" && p == "device_type");
        assert!(
            cpu0_prop.is_some(),
            "expected path 'cpus/cpu@0', not '/cpus/cpu@0'"
        );

        // No path should start with "/".
        for (path, _, _) in &props {
            assert!(
                !path.starts_with('/'),
                "path {path:?} must not start with '/'"
            );
        }
    }

    fn verify_cpu_numa_node_ids(topo: &Topology, props: &[(String, String, Vec<u8>)]) {
        for cpu_id in 0..topo.total_cpus() {
            let node_path = format!("cpus/cpu@{cpu_id}");
            let numa_id = prop_u32(props, &node_path, "numa-node-id")
                .unwrap_or_else(|| panic!("cpu {cpu_id}: missing numa-node-id"));
            let (llc_id, _, _) = topo.decompose(cpu_id);
            let expected = topo.numa_node_of(llc_id);
            assert_eq!(
                numa_id, expected,
                "cpu {cpu_id}: numa-node-id {numa_id} != expected {expected}"
            );
        }
    }

    #[test]
    fn fdt_cpu_numa_node_ids() {
        // No-SMT variant: 4 LLCs, 2 cores/LLC, 1 thread, 2 NUMA nodes.
        let topo = Topology {
            llcs: 4,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 2,
            nodes: None,
            distances: None,
        };
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let dtb = create_fdt(
            &topo,
            &mpidrs,
            512,
            "console=ttyS0",
            None,
            None,
            0,
            2,
            false,
        )
        .unwrap();
        verify_cpu_numa_node_ids(&topo, &parse_dtb_props(&dtb));

        // SMT variant: sibling threads share the same LLC and must get
        // the same numa-node-id.
        let topo_smt = Topology {
            llcs: 4,
            cores_per_llc: 2,
            threads_per_core: 2,
            numa_nodes: 2,
            nodes: None,
            distances: None,
        };
        let mpidrs_smt = fake_mpidrs(topo_smt.total_cpus());
        let dtb_smt = create_fdt(
            &topo_smt,
            &mpidrs_smt,
            512,
            "console=ttyS0",
            None,
            None,
            0,
            2,
            false,
        )
        .unwrap();
        verify_cpu_numa_node_ids(&topo_smt, &parse_dtb_props(&dtb_smt));
    }

    #[test]
    fn fdt_single_numa_no_numa_props() {
        let topo = Topology {
            llcs: 2,
            cores_per_llc: 4,
            threads_per_core: 2,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let dtb = create_fdt(
            &topo,
            &mpidrs,
            256,
            "console=ttyS0",
            None,
            None,
            0,
            2,
            false,
        )
        .unwrap();
        let props = parse_dtb_props(&dtb);

        // CPU nodes must NOT have numa-node-id when numa_nodes == 1.
        for cpu_id in 0..topo.total_cpus() {
            let node_path = format!("cpus/cpu@{cpu_id}");
            assert!(
                prop_u32(&props, &node_path, "numa-node-id").is_none(),
                "cpu {cpu_id}: numa-node-id must be absent for single-NUMA"
            );
        }

        // distance-map node must not exist.
        let has_distance_map = props
            .iter()
            .any(|(n, _, _)| n == "distance-map" || n.starts_with("distance-map/"));
        assert!(
            !has_distance_map,
            "distance-map node must not exist for single-NUMA"
        );
    }

    /// Verify multi-NUMA memory nodes: numa-node-id, reg, contiguity, total size.
    fn verify_memory_nodes(
        topo: &Topology,
        props: &[(String, String, Vec<u8>)],
        memory_mb: u32,
        shm_size: u64,
    ) {
        let mem_size = (memory_mb as u64) << 20;
        let usable_bytes = mem_size - shm_size;
        let n = topo.numa_nodes as u64;
        let per_node = (usable_bytes / n) & !0xFFF;

        let mut prev_end: Option<u64> = None;
        let mut total_size: u64 = 0;

        for node in 0..topo.numa_nodes {
            let base = DRAM_START + node as u64 * per_node;
            let node_name = format!("memory@{base:x}");

            // Verify numa-node-id.
            let numa_id = prop_u32(props, &node_name, "numa-node-id")
                .unwrap_or_else(|| panic!("memory node {node}: missing numa-node-id"));
            assert_eq!(numa_id, node, "memory node {node}: wrong numa-node-id");

            // Verify reg property: [base_hi, base_lo, size_hi, size_lo].
            let reg = prop_u32_array(props, &node_name, "reg")
                .unwrap_or_else(|| panic!("memory node {node}: missing reg"));
            assert_eq!(reg.len(), 4, "memory node {node}: reg must have 4 cells");

            let reg_base = ((reg[0] as u64) << 32) | reg[1] as u64;
            assert_eq!(reg_base, base, "memory node {node}: wrong base address");

            let reg_size = ((reg[2] as u64) << 32) | reg[3] as u64;
            let expected_size = if node == topo.numa_nodes - 1 {
                mem_size - node as u64 * per_node
            } else {
                per_node
            };
            assert_eq!(reg_size, expected_size, "memory node {node}: wrong size");

            // F4: contiguity — node N+1 base == node N base + node N size.
            if let Some(prev) = prev_end {
                assert_eq!(
                    reg_base, prev,
                    "memory node {node}: not contiguous (base {reg_base:#x} != prev end {prev:#x})"
                );
            }
            prev_end = Some(reg_base + reg_size);

            total_size += reg_size;
        }

        // F5: total memory across all nodes == mem_size.
        assert_eq!(
            total_size, mem_size,
            "total memory {total_size:#x} != mem_size {mem_size:#x}"
        );
    }

    #[test]
    fn fdt_memory_nodes_multi_numa() {
        let topo = Topology {
            llcs: 4,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 2,
            nodes: None,
            distances: None,
        };
        let mpidrs = fake_mpidrs(topo.total_cpus());

        // No-SHM case.
        let memory_mb: u32 = 512;
        let dtb = create_fdt(
            &topo,
            &mpidrs,
            memory_mb,
            "console=ttyS0",
            None,
            None,
            0,
            2,
            false,
        )
        .unwrap();
        verify_memory_nodes(&topo, &parse_dtb_props(&dtb), memory_mb, 0);

        // F3: with-SHM case — last node absorbs SHM region.
        let shm_size: u64 = 0x10_0000;
        let dtb_shm = create_fdt(
            &topo,
            &mpidrs,
            memory_mb,
            "console=ttyS0",
            None,
            None,
            shm_size,
            2,
            false,
        )
        .unwrap();
        verify_memory_nodes(&topo, &parse_dtb_props(&dtb_shm), memory_mb, shm_size);
    }

    #[test]
    fn fdt_distance_map() {
        let topo = Topology {
            llcs: 6,
            cores_per_llc: 4,
            threads_per_core: 2,
            numa_nodes: 3,
            nodes: None,
            distances: None,
        };
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let dtb = create_fdt(
            &topo,
            &mpidrs,
            1024,
            "console=ttyS0",
            None,
            None,
            0,
            2,
            false,
        )
        .unwrap();
        let props = parse_dtb_props(&dtb);

        let matrix = prop_u32_array(&props, "distance-map", "distance-matrix")
            .expect("missing distance-matrix property");

        let n = topo.numa_nodes;
        // NxN matrix of (nodea, nodeb, distance) triples.
        assert_eq!(
            matrix.len(),
            (n * n * 3) as usize,
            "distance-matrix length: expected {} triples",
            n * n,
        );

        let mut idx = 0;
        for i in 0..n {
            for j in 0..n {
                assert_eq!(matrix[idx], i, "triple ({i},{j}): wrong nodea");
                assert_eq!(matrix[idx + 1], j, "triple ({i},{j}): wrong nodeb");
                let expected_dist = if i == j { 10 } else { 20 };
                assert_eq!(
                    matrix[idx + 2],
                    expected_dist,
                    "triple ({i},{j}): distance {} != expected {expected_dist}",
                    matrix[idx + 2],
                );
                idx += 3;
            }
        }
    }

    #[test]
    fn fdt_cache_topology_multi_llc() {
        // 2 LLCs, 2 cores each, hw_cache_level=3 (L1/L2/L3).
        // chain_depth = 2: CPU -> L2 node -> L3 node per LLC.
        let topo = Topology {
            llcs: 2,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let dtb = create_fdt(
            &topo,
            &mpidrs,
            512,
            "console=ttyS0",
            None,
            None,
            0,
            3,
            false,
        )
        .unwrap();
        let props = parse_dtb_props(&dtb);

        // Each CPU must have next-level-cache pointing to its LLC's L2 node.
        for cpu_id in 0..topo.total_cpus() {
            let node_path = format!("cpus/cpu@{cpu_id}");
            let nlc = prop_u32(&props, &node_path, "next-level-cache");
            assert!(
                nlc.is_some(),
                "cpu {cpu_id}: missing next-level-cache phandle"
            );
        }

        // CPUs in the same LLC must share the same phandle.
        let cpu0_nlc = prop_u32(&props, "cpus/cpu@0", "next-level-cache").unwrap();
        let cpu1_nlc = prop_u32(&props, "cpus/cpu@1", "next-level-cache").unwrap();
        assert_eq!(
            cpu0_nlc, cpu1_nlc,
            "CPU 0 and 1 (same LLC) must share phandle"
        );

        // CPUs in different LLCs must have different phandles.
        let cpu2_nlc = prop_u32(&props, "cpus/cpu@2", "next-level-cache").unwrap();
        assert_ne!(
            cpu0_nlc, cpu2_nlc,
            "CPU 0 and 2 (different LLC) must differ"
        );

        // L2 cache nodes must exist with correct properties.
        for llc in 0..2u32 {
            let l2_path = format!("cpus/l2-cache{llc}");
            assert_eq!(
                prop_u32(&props, &l2_path, "cache-level"),
                Some(2),
                "L2 cache{llc}: wrong cache-level"
            );
            // L2 must chain to L3 via next-level-cache.
            let l2_nlc = prop_u32(&props, &l2_path, "next-level-cache");
            assert!(l2_nlc.is_some(), "L2 cache{llc}: missing next-level-cache");

            let l3_path = format!("cpus/l3-cache{llc}");
            assert_eq!(
                prop_u32(&props, &l3_path, "cache-level"),
                Some(3),
                "L3 cache{llc}: wrong cache-level"
            );
            // L3 must NOT have next-level-cache (terminal).
            assert!(
                prop_u32(&props, &l3_path, "next-level-cache").is_none(),
                "L3 cache{llc}: should not have next-level-cache"
            );
        }

        // L3 nodes for different LLCs must have different phandles.
        let l3_0_phandle = prop_u32(&props, "cpus/l3-cache0", "phandle").unwrap();
        let l3_1_phandle = prop_u32(&props, "cpus/l3-cache1", "phandle").unwrap();
        assert_ne!(
            l3_0_phandle, l3_1_phandle,
            "L3 phandles must differ per LLC"
        );
    }

    #[test]
    fn fdt_no_cache_nodes_single_llc() {
        // Single LLC: no cache nodes should be emitted.
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 4,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let dtb = create_fdt(
            &topo,
            &mpidrs,
            256,
            "console=ttyS0",
            None,
            None,
            0,
            3,
            false,
        )
        .unwrap();
        let props = parse_dtb_props(&dtb);

        // CPU 0 must NOT have next-level-cache.
        assert!(
            prop_u32(&props, "cpus/cpu@0", "next-level-cache").is_none(),
            "single-LLC: cpu should not have next-level-cache"
        );

        // No cache nodes should exist.
        let has_cache = props
            .iter()
            .any(|(n, _, _)| n.contains("cache") && n.starts_with("cpus/l"));
        assert!(!has_cache, "single-LLC: no cache nodes expected");
    }
}
