use anyhow::{Context, Result};
use kvm_bindings::{
    KVM_DEV_ARM_VGIC_CTRL_INIT, KVM_DEV_ARM_VGIC_GRP_ADDR, KVM_DEV_ARM_VGIC_GRP_CTRL,
    KVM_DEV_ARM_VGIC_GRP_NR_IRQS, KVM_IRQ_ROUTING_IRQCHIP, KVM_VGIC_V3_ADDR_TYPE_DIST,
    KVM_VGIC_V3_ADDR_TYPE_REDIST, KvmIrqRouting, kvm_create_device, kvm_device_attr,
    kvm_device_type_KVM_DEV_TYPE_ARM_VGIC_V3, kvm_irq_routing_entry,
    kvm_irq_routing_entry__bindgen_ty_1, kvm_irq_routing_irqchip, kvm_userspace_memory_region,
};
use kvm_ioctls::{Cap, DeviceFd, Kvm, VcpuFd, VmFd};
use vm_memory::{GuestAddress, GuestMemory, GuestMemoryMmap};

use crate::vmm::topology::Topology;

// ---------------------------------------------------------------------------
// Memory layout — devices below DRAM, guest RAM above
// ---------------------------------------------------------------------------

/// Start of guest DRAM. All device MMIO regions live below this address.
pub(crate) const DRAM_START: u64 = 0x4000_0000; // 1 GB

/// GICv3 distributor MMIO base address.
pub(crate) const GIC_DIST_BASE: u64 = 0x0800_0000;

/// GICv3 distributor size: 64 KB.
pub(crate) const GIC_DIST_SIZE: u64 = 0x1_0000;

/// GICv3 redistributor base: immediately after the distributor.
/// Each redistributor occupies 128 KB (two 64 KB frames: RD_base + SGI_base).
pub(crate) const GIC_REDIST_BASE: u64 = GIC_DIST_BASE + GIC_DIST_SIZE;

/// Size per redistributor: 128 KB (RD_base + SGI_base).
pub(crate) const GIC_REDIST_SIZE_PER_CPU: u64 = 0x2_0000;

/// ns16550a serial MMIO base address. SPI 33.
pub(crate) const SERIAL_MMIO_BASE: u64 = 0x0900_0000;

/// ns16550a serial MMIO size: 4 KB (one page, 8 registers x 4 bytes).
pub(crate) const SERIAL_MMIO_SIZE: u64 = 0x1000;

/// Second serial for application output. SPI 34.
pub(crate) const SERIAL2_MMIO_BASE: u64 = SERIAL_MMIO_BASE + SERIAL_MMIO_SIZE;

/// Virtio-console MMIO base. Placed after the two serial regions.
pub(crate) const VIRTIO_CONSOLE_MMIO_BASE: u64 = SERIAL2_MMIO_BASE + SERIAL_MMIO_SIZE;

/// SPI interrupt for virtio-console. SPI 35.
pub(crate) const VIRTIO_CONSOLE_IRQ: u32 = 35;

/// Kernel Image load address. 2 MB aligned per arm64 boot protocol.
/// Relative to DRAM_START — the kernel is loaded at DRAM_START + text_offset,
/// but the PE loader base address must be DRAM_START (2 MB aligned).
pub(crate) const KERNEL_LOAD_ADDR: u64 = DRAM_START;

/// FDT maximum size: 2 MB. FDT is placed at the end of usable DRAM.
pub(crate) const FDT_MAX_SIZE: u64 = 0x20_0000;

/// Maximum command line length.
pub(crate) const CMDLINE_MAX: usize = 4096;

/// SPI interrupt numbers for the two serial ports.
/// GICv3 SPIs start at IRQ 32. These map to intid = 32 + N.
pub(crate) const SERIAL_IRQ: u32 = 33;
pub(crate) const SERIAL2_IRQ: u32 = 34;

/// Number of IRQs for the GIC. Must be a multiple of 32 and >= 64.
/// 128 covers SPIs 0-95, sufficient for serial + headroom.
const GIC_NR_IRQS: u32 = 128;

/// A KVM virtual machine with configured topology (aarch64).
#[allow(dead_code)]
pub struct KtstrKvm {
    pub kvm: Kvm,
    pub vm_fd: VmFd,
    pub vcpus: Vec<VcpuFd>,
    pub guest_mem: GuestMemoryMmap,
    pub topology: Topology,
    pub has_immediate_exit: bool,
    /// GICv3 device fd — held to keep the device alive.
    gic_fd: DeviceFd,
    /// Whether hugepages were requested at construction time.
    /// Stored so deferred memory allocation uses the same backing.
    use_hugepages: bool,
    /// Performance mode flag. Stored so deferred memory allocation
    /// can check hugepage availability fresh when memory_mb was
    /// unknown at construction time.
    performance_mode: bool,
}

impl KtstrKvm {
    /// Create a new KVM VM with the given topology and memory size.
    pub fn new(topo: Topology, memory_mb: u32, performance_mode: bool) -> Result<Self> {
        Self::new_inner(topo, Some(memory_mb), false, performance_mode)
    }

    /// Create a new KVM VM with hugepage-backed guest memory.
    pub fn new_with_hugepages(
        topo: Topology,
        memory_mb: u32,
        performance_mode: bool,
    ) -> Result<Self> {
        Self::new_inner(topo, Some(memory_mb), true, performance_mode)
    }

    /// Create a KVM VM without allocating guest memory.
    ///
    /// Sets up /dev/kvm, VM fd, vCPUs, and GICv3 — none of which depend
    /// on guest memory size. Memory is allocated later via
    /// [`allocate_and_register_memory`].
    pub fn new_deferred(
        topo: Topology,
        use_hugepages: bool,
        performance_mode: bool,
    ) -> Result<Self> {
        Self::new_inner(topo, None, use_hugepages, performance_mode)
    }

    /// Allocate guest memory and register it with KVM.
    ///
    /// Must be called exactly once on a VM created with `new_deferred`.
    /// Replaces the placeholder guest memory with a real allocation of
    /// `memory_mb` megabytes at DRAM_START. Re-checks hugepage
    /// availability when performance_mode is set, since memory_mb was
    /// unknown at construction time and `use_hugepages` may have been
    /// false.
    pub fn allocate_and_register_memory(&mut self, memory_mb: u32) -> Result<()> {
        let mem_size = (memory_mb as u64) << 20;
        let use_hugepages = self.use_hugepages
            || (self.performance_mode
                && crate::vmm::host_topology::hugepages_free()
                    >= crate::vmm::host_topology::hugepages_needed(memory_mb));
        let guest_mem = if use_hugepages {
            crate::vmm::allocate_hugepage_memory(mem_size as usize, GuestAddress(DRAM_START))?
        } else {
            GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(DRAM_START), mem_size as usize)])
                .context("allocate guest memory")?
        };

        let host_addr = guest_mem
            .get_host_address(GuestAddress(DRAM_START))
            .context("get host address for guest memory")? as u64;
        let mem_region = kvm_userspace_memory_region {
            slot: 0,
            guest_phys_addr: DRAM_START,
            memory_size: mem_size,
            userspace_addr: host_addr,
            flags: 0,
        };
        unsafe {
            self.vm_fd
                .set_user_memory_region(mem_region)
                .context("set user memory region")?;
        }

        self.guest_mem = guest_mem;
        Ok(())
    }

    fn new_inner(
        topo: Topology,
        memory_mb: Option<u32>,
        use_hugepages: bool,
        performance_mode: bool,
    ) -> Result<Self> {
        let kvm = Kvm::new().context("open /dev/kvm")?;

        let has_immediate_exit = kvm.check_extension(Cap::ImmediateExit);

        // Create VM with EINTR retry (Firecracker pattern).
        let vm_fd = {
            let mut attempts = 0;
            loop {
                match kvm.create_vm() {
                    Ok(fd) => break fd,
                    Err(e) if e.errno() == libc::EINTR && attempts < 5 => {
                        attempts += 1;
                        std::thread::sleep(std::time::Duration::from_micros(1 << attempts));
                    }
                    Err(e) => return Err(e).context("create VM"),
                }
            }
        };

        // Allocate guest memory at DRAM_START. When memory_mb is None
        // (deferred mode), use a 1-page placeholder — the real allocation
        // happens later via allocate_and_register_memory().
        let guest_mem = match memory_mb {
            Some(mb) => {
                let mem_size = (mb as u64) << 20;
                let mem = if use_hugepages {
                    crate::vmm::allocate_hugepage_memory(
                        mem_size as usize,
                        GuestAddress(DRAM_START),
                    )?
                } else {
                    GuestMemoryMmap::<()>::from_ranges(&[(
                        GuestAddress(DRAM_START),
                        mem_size as usize,
                    )])
                    .context("allocate guest memory")?
                };

                let host_addr =
                    mem.get_host_address(GuestAddress(DRAM_START))
                        .context("get host address for guest memory")? as u64;
                let mem_region = kvm_userspace_memory_region {
                    slot: 0,
                    guest_phys_addr: DRAM_START,
                    memory_size: mem_size,
                    userspace_addr: host_addr,
                    flags: 0,
                };
                unsafe {
                    vm_fd
                        .set_user_memory_region(mem_region)
                        .context("set user memory region")?;
                }
                mem
            }
            None => {
                // Placeholder: 1 page. Not registered with KVM — no guest
                // code runs until allocate_and_register_memory() is called.
                GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(DRAM_START), 4096)])
                    .context("allocate placeholder guest memory")?
            }
        };

        // Create vCPUs. On aarch64, vCPUs must exist before GIC init.
        let total = topo.total_cpus();
        let mut vcpus = Vec::with_capacity(total as usize);

        let mut kvi = kvm_bindings::kvm_vcpu_init::default();
        vm_fd
            .get_preferred_target(&mut kvi)
            .context("get preferred target")?;
        kvi.features[0] |= 1 << kvm_bindings::KVM_ARM_VCPU_PSCI_0_2;

        for cpu_id in 0..total {
            let vcpu = vm_fd
                .create_vcpu(cpu_id as u64)
                .with_context(|| format!("create vCPU {cpu_id}"))?;

            let mut vcpu_kvi = kvi;
            if cpu_id != 0 {
                vcpu_kvi.features[0] |= 1 << kvm_bindings::KVM_ARM_VCPU_POWER_OFF;
            }
            vcpu.vcpu_init(&vcpu_kvi)
                .with_context(|| format!("init vCPU {cpu_id}"))?;

            vcpus.push(vcpu);
        }

        // Create GICv3 via KVM_CREATE_DEVICE.
        let gic_fd = Self::create_gic(&vm_fd, total)?;

        // Set up GSI routing so irqfd works with the GICv3 device.
        // Map GSI N -> irqchip 0, pin N for the serial SPI IRQs.
        Self::setup_gsi_routing(&vm_fd)?;

        Ok(KtstrKvm {
            kvm,
            vm_fd,
            vcpus,
            guest_mem,
            topology: topo,
            has_immediate_exit,
            gic_fd,
            use_hugepages,
            performance_mode,
        })
    }

    /// Create and initialize a GICv3 interrupt controller.
    fn create_gic(vm_fd: &VmFd, num_cpus: u32) -> Result<DeviceFd> {
        let mut gic_device = kvm_create_device {
            type_: kvm_device_type_KVM_DEV_TYPE_ARM_VGIC_V3,
            fd: 0,
            flags: 0,
        };
        let gic_fd = vm_fd
            .create_device(&mut gic_device)
            .context("create GICv3 device")?;

        // Set number of IRQs.
        let nr_irqs: u32 = GIC_NR_IRQS;
        let nr_irqs_attr = kvm_device_attr {
            group: KVM_DEV_ARM_VGIC_GRP_NR_IRQS,
            attr: 0,
            addr: &nr_irqs as *const u32 as u64,
            flags: 0,
        };
        gic_fd
            .set_device_attr(&nr_irqs_attr)
            .context("set GIC nr_irqs")?;

        // Set distributor address.
        let dist_addr: u64 = GIC_DIST_BASE;
        let dist_attr = kvm_device_attr {
            group: KVM_DEV_ARM_VGIC_GRP_ADDR,
            attr: KVM_VGIC_V3_ADDR_TYPE_DIST as u64,
            addr: &dist_addr as *const u64 as u64,
            flags: 0,
        };
        gic_fd
            .set_device_attr(&dist_attr)
            .context("set GIC distributor address")?;

        // Set redistributor address.
        let redist_addr: u64 = GIC_REDIST_BASE;
        let redist_size = num_cpus as u64 * GIC_REDIST_SIZE_PER_CPU;
        anyhow::ensure!(
            GIC_REDIST_BASE + redist_size <= DRAM_START,
            "GIC redistributor region (ends at {:#x}) overlaps DRAM at {:#x} for {} CPUs",
            GIC_REDIST_BASE + redist_size,
            DRAM_START,
            num_cpus,
        );
        let redist_attr = kvm_device_attr {
            group: KVM_DEV_ARM_VGIC_GRP_ADDR,
            attr: KVM_VGIC_V3_ADDR_TYPE_REDIST as u64,
            addr: &redist_addr as *const u64 as u64,
            flags: 0,
        };
        gic_fd
            .set_device_attr(&redist_attr)
            .context("set GIC redistributor address")?;

        // Initialize the GIC.
        let init_attr = kvm_device_attr {
            group: KVM_DEV_ARM_VGIC_GRP_CTRL,
            attr: KVM_DEV_ARM_VGIC_CTRL_INIT as u64,
            addr: 0,
            flags: 0,
        };
        gic_fd
            .set_device_attr(&init_attr)
            .context("init GIC device")?;

        Ok(gic_fd)
    }

    /// Set up GSI routing for irqfd.
    ///
    /// With GICv3 via KVM_CREATE_DEVICE, there is no default IRQ routing.
    /// We must explicitly route GSI numbers to GIC SPI pins via
    /// KVM_SET_GSI_ROUTING before register_irqfd will deliver interrupts.
    fn setup_gsi_routing(vm_fd: &VmFd) -> Result<()> {
        let irqs = [SERIAL_IRQ, SERIAL2_IRQ, VIRTIO_CONSOLE_IRQ];
        let mut routing = KvmIrqRouting::new(irqs.len()).context("create KvmIrqRouting")?;
        for (i, &irq) in irqs.iter().enumerate() {
            routing.as_mut_slice()[i] = kvm_irq_routing_entry {
                gsi: irq,
                type_: KVM_IRQ_ROUTING_IRQCHIP,
                flags: 0,
                pad: 0,
                u: kvm_irq_routing_entry__bindgen_ty_1 {
                    irqchip: kvm_irq_routing_irqchip {
                        irqchip: 0,    // GIC device index
                        pin: irq - 32, // SPI pin (0-based); KVM adds 32 to get intid
                    },
                },
            };
        }
        vm_fd
            .set_gsi_routing(&routing)
            .context("set GSI routing for serial IRQs")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_vm_basic() {
        let topo = Topology {
            sockets: 1,
            cores_per_socket: 2,
            threads_per_core: 1,
        };
        let vm = KtstrKvm::new(topo, 128, false);
        assert!(vm.is_ok(), "VM creation failed: {:?}", vm.err());
        let vm = vm.unwrap();
        assert_eq!(vm.vcpus.len(), 2);
    }

    #[test]
    fn create_vm_multi_socket() {
        let topo = Topology {
            sockets: 2,
            cores_per_socket: 2,
            threads_per_core: 2,
        };
        let vm = KtstrKvm::new(topo, 256, false);
        assert!(
            vm.is_ok(),
            "multi-socket VM creation failed: {:?}",
            vm.err()
        );
        let vm = vm.unwrap();
        assert_eq!(vm.vcpus.len(), 8);
    }

    #[test]
    fn create_vm_single_cpu() {
        let topo = Topology {
            sockets: 1,
            cores_per_socket: 1,
            threads_per_core: 1,
        };
        let vm = KtstrKvm::new(topo, 64, false);
        assert!(vm.is_ok());
        assert_eq!(vm.unwrap().vcpus.len(), 1);
    }

    #[test]
    fn memory_size_correct() {
        use vm_memory::GuestMemoryRegion;
        let topo = Topology {
            sockets: 1,
            cores_per_socket: 1,
            threads_per_core: 1,
        };
        let vm = KtstrKvm::new(topo, 256, false).unwrap();
        let total: u64 = vm.guest_mem.iter().map(|r| r.len()).sum();
        assert_eq!(total, 256 << 20);
    }

    #[test]
    fn memory_starts_at_dram() {
        use vm_memory::GuestMemoryRegion;
        let topo = Topology {
            sockets: 1,
            cores_per_socket: 1,
            threads_per_core: 1,
        };
        let vm = KtstrKvm::new(topo, 64, false).unwrap();
        let region = vm.guest_mem.iter().next().unwrap();
        assert_eq!(
            region.start_addr(),
            GuestAddress(DRAM_START),
            "guest memory must start at DRAM_START"
        );
    }

    #[test]
    fn immediate_exit_cap_detected() {
        let topo = Topology {
            sockets: 1,
            cores_per_socket: 1,
            threads_per_core: 1,
        };
        let vm = KtstrKvm::new(topo, 64, false).unwrap();
        assert!(vm.has_immediate_exit);
    }

    #[test]
    fn gic_redist_does_not_overlap_dram() {
        // Maximum vCPUs that fit: (DRAM_START - GIC_REDIST_BASE) / GIC_REDIST_SIZE_PER_CPU
        let max_cpus = (DRAM_START - GIC_REDIST_BASE) / GIC_REDIST_SIZE_PER_CPU;
        assert!(
            max_cpus >= 128,
            "layout should support at least 128 vCPUs, got {max_cpus}"
        );
    }

    #[test]
    fn devices_below_dram() {
        const { assert!(GIC_DIST_BASE < DRAM_START) };
        const { assert!(GIC_REDIST_BASE < DRAM_START) };
        const { assert!(SERIAL_MMIO_BASE < DRAM_START) };
        const { assert!(SERIAL2_MMIO_BASE < DRAM_START) };
        const { assert!(VIRTIO_CONSOLE_MMIO_BASE < DRAM_START) };
        const {
            assert!(
                VIRTIO_CONSOLE_MMIO_BASE + crate::vmm::virtio_console::VIRTIO_MMIO_SIZE
                    <= DRAM_START
            )
        };
    }
}
