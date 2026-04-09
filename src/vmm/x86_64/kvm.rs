use anyhow::{Context, Result};
use kvm_bindings::{
    KVM_CAP_SPLIT_IRQCHIP, KVM_CAP_X2APIC_API, KVM_CAP_X86_DISABLE_EXITS, KVM_PIT_SPEAKER_DUMMY,
    KVM_X2APIC_API_DISABLE_BROADCAST_QUIRK, KVM_X2APIC_API_USE_32BIT_IDS,
    KVM_X86_DISABLE_EXITS_PAUSE, kvm_enable_cap, kvm_pit_config, kvm_userspace_memory_region,
};
use kvm_ioctls::{Cap, Kvm, VcpuFd, VmFd};
use vm_memory::{GuestAddress, GuestMemory, GuestMemoryMmap};

use super::topology::{generate_cpuid, max_apic_id};
use crate::vmm::topology::Topology;

/// Physical address where the kernel is loaded.
pub(crate) const KERNEL_LOAD_ADDR: u64 = 0x100000; // 1 MB

/// Physical address of boot parameters (zero page).
pub(crate) const BOOT_PARAMS_ADDR: u64 = 0x7000;

/// Physical address of the kernel command line.
pub(crate) const CMDLINE_ADDR: u64 = 0x20000;

/// Maximum command line length.
pub(crate) const CMDLINE_MAX: usize = 4096;

// ---- Memory layout constants shared by boot.rs and acpi.rs ----

/// End of Extended BIOS Data Area (640K - 1K).
pub(crate) const EBDA_START: u64 = 0x9FC00;

/// Start of high memory (1 MB).
pub(crate) const HIMEM_START: u64 = 0x10_0000;

/// Start of PCI MMIO gap (3 GB). Memory below this is usable RAM.
pub(crate) const MMIO_GAP_START: u64 = 0xC000_0000;

/// End of PCI MMIO gap (4 GB). Memory above this resumes as RAM.
pub(crate) const MMIO_GAP_END: u64 = 0x1_0000_0000;

/// E820 memory type: usable RAM.
pub(crate) const E820_RAM: u32 = 1;

/// Offset from code32_start to 64-bit entry point in bzImage.
pub(crate) const STARTUP64_OFFSET: u64 = 0x200;

/// TSS address — same as Firecracker/libkrun.
const KVM_TSS_ADDRESS: u64 = 0xfffb_d000;

/// Identity map address — placed immediately after the 3-page TSS region.
/// KVM requires this to be set before creating vCPUs on x86_64.
const KVM_IDENTITY_MAP_ADDRESS: u64 = KVM_TSS_ADDRESS + 3 * 4096;

/// IOAPIC supports 24 input pins (IRQ 0-23).
const NUM_IOAPIC_PINS: u64 = 24;

/// APIC IDs above this require x2APIC mode (8-bit xAPIC limit).
const MAX_XAPIC_ID: u32 = 254;

/// Required KVM capabilities — Firecracker checks these 14.
const REQUIRED_CAPS: &[Cap] = &[
    Cap::Irqchip,
    Cap::Ioeventfd,
    Cap::Irqfd,
    Cap::UserMemory,
    Cap::SetTssAddr,
    Cap::Pit2,
    Cap::PitState2,
    Cap::AdjustClock,
    Cap::Debugregs,
    Cap::MpState,
    Cap::VcpuEvents,
    Cap::Xcrs,
    Cap::Xsave,
    Cap::ExtCpuid,
];

/// A KVM virtual machine with configured topology.
#[allow(dead_code)] // kvm/vm_fd/topology fields are held for Drop (fd lifetime)
pub struct SttKvm {
    pub kvm: Kvm,
    pub vm_fd: VmFd,
    pub vcpus: Vec<VcpuFd>,
    pub guest_mem: GuestMemoryMmap,
    pub topology: Topology,
    /// Whether KVM supports the immediate_exit mechanism (KVM_CAP_IMMEDIATE_EXIT).
    pub has_immediate_exit: bool,
    /// Split IRQ chip mode: LAPIC in kernel, PIC/IOAPIC emulated in userspace.
    /// Enabled when any APIC ID exceeds the 8-bit xAPIC limit (254).
    pub split_irqchip: bool,
}

impl SttKvm {
    /// Create a new KVM VM with the given topology and memory size.
    pub fn new(topo: Topology, memory_mb: u32, performance_mode: bool) -> Result<Self> {
        Self::new_inner(topo, memory_mb, false, performance_mode)
    }

    /// Create a new KVM VM with hugepage-backed guest memory.
    pub fn new_with_hugepages(
        topo: Topology,
        memory_mb: u32,
        performance_mode: bool,
    ) -> Result<Self> {
        Self::new_inner(topo, memory_mb, true, performance_mode)
    }

    fn new_inner(
        topo: Topology,
        memory_mb: u32,
        use_hugepages: bool,
        performance_mode: bool,
    ) -> Result<Self> {
        let kvm = Kvm::new().context("open /dev/kvm")?;

        // Check required capabilities (Firecracker pattern)
        for &cap in REQUIRED_CAPS {
            anyhow::ensure!(
                kvm.check_extension(cap),
                "KVM missing required capability: {:?}",
                cap
            );
        }

        let has_immediate_exit = kvm.check_extension(Cap::ImmediateExit);

        // Create VM with EINTR retry (Firecracker: up to 5 attempts, exponential backoff)
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

        // TSS (required on x86_64 before creating vCPUs)
        vm_fd
            .set_tss_address(KVM_TSS_ADDRESS as usize)
            .context("set TSS")?;

        // Identity map — one page after the 3-page TSS region.
        // Must be set before creating vCPUs.
        vm_fd
            .set_identity_map_address(KVM_IDENTITY_MAP_ADDRESS)
            .context("set identity map address")?;

        // Determine whether any APIC ID exceeds the 8-bit xAPIC limit.
        // If so, use split IRQ chip (LAPIC-only in kernel) + x2APIC API.
        let max_apic_id = max_apic_id(&topo);
        let split_irqchip = max_apic_id > MAX_XAPIC_ID;

        if split_irqchip {
            // Split IRQ chip: only LAPIC is emulated in kernel.
            // PIC and IOAPIC are not created — userspace handles them.
            let mut cap = kvm_enable_cap {
                cap: KVM_CAP_SPLIT_IRQCHIP,
                ..Default::default()
            };
            cap.args[0] = NUM_IOAPIC_PINS;
            vm_fd.enable_cap(&cap).context("enable split IRQ chip")?;

            // Enable x2APIC API for 32-bit destination IDs and correct
            // broadcast behavior with APIC IDs > 254.
            let mut cap = kvm_enable_cap {
                cap: KVM_CAP_X2APIC_API,
                ..Default::default()
            };
            cap.args[0] =
                (KVM_X2APIC_API_USE_32BIT_IDS | KVM_X2APIC_API_DISABLE_BROADCAST_QUIRK) as u64;
            vm_fd.enable_cap(&cap).context("enable x2APIC API")?;
        } else {
            // Full IRQ chip (PIC + IOAPIC + LAPIC) — must exist before KVM_CREATE_VCPU
            vm_fd.create_irq_chip().context("create IRQ chip")?;

            // PIT (timer) with dummy speaker port.
            // Only created with full IRQ chip — PIT routes through the in-kernel
            // IOAPIC (IRQ 0 -> GSI 2). With split IRQ chip there is no in-kernel
            // IOAPIC, so PIT creation fails.
            let pit_config = kvm_pit_config {
                flags: KVM_PIT_SPEAKER_DUMMY,
                ..Default::default()
            };
            vm_fd.create_pit2(pit_config).context("create PIT")?;
        }

        // Disable PAUSE VM exits to reduce vmexit overhead during spinlocks.
        // Only PAUSE — HLT exits must remain enabled (BSP shutdown detection).
        if performance_mode {
            let mut cap = kvm_enable_cap {
                cap: KVM_CAP_X86_DISABLE_EXITS,
                ..Default::default()
            };
            cap.args[0] = KVM_X86_DISABLE_EXITS_PAUSE as u64;
            if let Err(e) = vm_fd.enable_cap(&cap) {
                eprintln!(
                    "performance_mode: WARNING: KVM_CAP_X86_DISABLE_EXITS not supported: {e}"
                );
            }
        }

        // Allocate guest memory
        let mem_size = (memory_mb as u64) << 20;
        let guest_mem = if use_hugepages {
            crate::vmm::allocate_hugepage_memory(mem_size as usize, GuestAddress(0))?
        } else {
            GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), mem_size as usize)])
                .context("allocate guest memory")?
        };

        // Register memory with KVM
        let host_addr = guest_mem
            .get_host_address(GuestAddress(0))
            .context("get host address for guest memory")? as u64;
        let mem_region = kvm_userspace_memory_region {
            slot: 0,
            guest_phys_addr: 0,
            memory_size: mem_size,
            userspace_addr: host_addr,
            flags: 0,
        };
        unsafe {
            vm_fd
                .set_user_memory_region(mem_region)
                .context("set user memory region")?;
        }

        // Fetch host CPUID once, reuse for all vCPUs (Firecracker pattern).
        let base_cpuid = kvm
            .get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
            .context("get_supported_cpuid")?;

        // Create vCPUs with topology-specific CPUID
        let total = topo.total_cpus();
        let mut vcpus = Vec::with_capacity(total as usize);
        for cpu_id in 0..total {
            let vcpu = vm_fd
                .create_vcpu(cpu_id as u64)
                .with_context(|| format!("create vCPU {cpu_id}"))?;

            let cpuid_entries =
                generate_cpuid(base_cpuid.as_slice(), &topo, cpu_id, performance_mode);
            let cpuid = kvm_bindings::CpuId::from_entries(&cpuid_entries).context("build CpuId")?;
            vcpu.set_cpuid2(&cpuid)
                .with_context(|| format!("set CPUID for vCPU {cpu_id}"))?;

            vcpus.push(vcpu);
        }

        Ok(SttKvm {
            kvm,
            vm_fd,
            vcpus,
            guest_mem,
            topology: topo,
            has_immediate_exit,
            split_irqchip,
        })
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
        let vm = SttKvm::new(topo, 128, false);
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
        let vm = SttKvm::new(topo, 256, false);
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
        let vm = SttKvm::new(topo, 64, false);
        assert!(vm.is_ok());
        assert_eq!(vm.unwrap().vcpus.len(), 1);
    }

    #[test]
    fn create_vm_large_topology() {
        let topo = Topology {
            sockets: 4,
            cores_per_socket: 4,
            threads_per_core: 2,
        };
        let vm = SttKvm::new(topo, 512, false);
        assert!(vm.is_ok(), "large topology failed: {:?}", vm.err());
        assert_eq!(vm.unwrap().vcpus.len(), 32);
    }

    #[test]
    fn create_vm_odd_topology() {
        let topo = Topology {
            sockets: 3,
            cores_per_socket: 3,
            threads_per_core: 1,
        };
        let vm = SttKvm::new(topo, 128, false);
        assert!(vm.is_ok(), "odd topology failed: {:?}", vm.err());
        assert_eq!(vm.unwrap().vcpus.len(), 9);
    }

    #[test]
    fn memory_size_correct() {
        use vm_memory::GuestMemoryRegion;
        let topo = Topology {
            sockets: 1,
            cores_per_socket: 1,
            threads_per_core: 1,
        };
        let vm = SttKvm::new(topo, 256, false).unwrap();
        let total: u64 = vm.guest_mem.iter().map(|r| r.len()).sum();
        assert_eq!(total, 256 << 20);
    }

    #[test]
    fn tss_address_matches_firecracker() {
        assert_eq!(KVM_TSS_ADDRESS, 0xfffb_d000);
    }

    #[test]
    fn identity_map_follows_tss() {
        assert_eq!(KVM_IDENTITY_MAP_ADDRESS, KVM_TSS_ADDRESS + 3 * 4096);
        assert_eq!(KVM_IDENTITY_MAP_ADDRESS, 0xfffc_0000);
    }

    #[test]
    fn required_caps_non_empty() {
        assert!(!REQUIRED_CAPS.is_empty());
        assert!(REQUIRED_CAPS.len() >= 14);
    }

    #[test]
    fn small_topology_uses_full_irqchip() {
        let topo = Topology {
            sockets: 2,
            cores_per_socket: 4,
            threads_per_core: 2,
        };
        // max APIC ID = apic_id(15) = 1<<3 | 3<<1 | 1 = 15, well under 254
        assert!(max_apic_id(&topo) <= MAX_XAPIC_ID);
        let vm = SttKvm::new(topo, 256, false).unwrap();
        assert!(!vm.split_irqchip, "small topology should use full IRQ chip");
    }

    #[test]
    fn large_topology_uses_split_irqchip() {
        // 15 sockets x 8 cores x 2 threads = 240 vCPUs
        // max APIC ID = apic_id(239) = 14<<4 | 7<<1 | 1 = 239, under 254
        // So try bigger: 14 sockets x 9 cores x 2 threads = 252 vCPUs
        // core_bits = bits_needed(9) = 4, thread_bits = 1, core_shift = 5
        // max APIC ID = apic_id(251) = 13<<5 | 8<<1 | 1 = 433
        let topo = Topology {
            sockets: 14,
            cores_per_socket: 9,
            threads_per_core: 2,
        };
        assert!(
            max_apic_id(&topo) > MAX_XAPIC_ID,
            "max APIC ID {} should exceed {}",
            max_apic_id(&topo),
            MAX_XAPIC_ID,
        );
        let vm = SttKvm::new(topo, 4096, false);
        assert!(vm.is_ok(), "split IRQ chip VM failed: {:?}", vm.err());
        let vm = vm.unwrap();
        assert!(vm.split_irqchip, "large topology should use split IRQ chip");
        assert_eq!(vm.vcpus.len(), 252);
    }

    #[test]
    fn split_irqchip_boundary() {
        // Find a topology that is exactly at the boundary.
        // 8 sockets x 8 cores x 2 threads: core_shift = 4, max APIC ID = 7<<4 | 7<<1 | 1 = 127
        let small = Topology {
            sockets: 8,
            cores_per_socket: 8,
            threads_per_core: 2,
        };
        assert!(
            max_apic_id(&small) <= MAX_XAPIC_ID,
            "8s/8c/2t max APIC ID {} should be <= 254",
            max_apic_id(&small),
        );
        let vm = SttKvm::new(small, 2048, false).unwrap();
        assert!(!vm.split_irqchip);

        // 15 sockets x 8 cores x 2 threads: core_shift = 4, max APIC ID = 14<<4 | 7<<1 | 1 = 239
        let still_small = Topology {
            sockets: 15,
            cores_per_socket: 8,
            threads_per_core: 2,
        };
        assert!(
            max_apic_id(&still_small) <= MAX_XAPIC_ID,
            "15s/8c/2t max APIC ID {} should be <= 254",
            max_apic_id(&still_small),
        );
        let vm = SttKvm::new(still_small, 4096, false).unwrap();
        assert!(!vm.split_irqchip);
    }

    #[test]
    fn immediate_exit_cap_detected() {
        let topo = Topology {
            sockets: 1,
            cores_per_socket: 1,
            threads_per_core: 1,
        };
        let vm = SttKvm::new(topo, 64, false).unwrap();
        // KVM_CAP_IMMEDIATE_EXIT is available since Linux 4.12.
        assert!(vm.has_immediate_exit);
    }

    #[test]
    fn performance_mode_succeeds() {
        let topo = Topology {
            sockets: 1,
            cores_per_socket: 2,
            threads_per_core: 1,
        };
        let vm = SttKvm::new(topo, 128, true);
        assert!(
            vm.is_ok(),
            "performance_mode VM creation failed: {:?}",
            vm.err()
        );
    }

    #[test]
    fn performance_mode_does_not_affect_vcpu_count() {
        let topo = Topology {
            sockets: 2,
            cores_per_socket: 2,
            threads_per_core: 2,
        };
        let vm_normal = SttKvm::new(topo, 256, false).unwrap();
        let vm_perf = SttKvm::new(topo, 256, true).unwrap();
        assert_eq!(vm_normal.vcpus.len(), vm_perf.vcpus.len());
    }
}
