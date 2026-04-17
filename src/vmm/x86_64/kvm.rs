use anyhow::{Context, Result};
use kvm_bindings::{
    KVM_CAP_HALT_POLL, KVM_CAP_SPLIT_IRQCHIP, KVM_CAP_X2APIC_API, KVM_CAP_X86_DISABLE_EXITS,
    KVM_CLOCK_TSC_STABLE, KVM_PIT_SPEAKER_DUMMY, KVM_X2APIC_API_DISABLE_BROADCAST_QUIRK,
    KVM_X2APIC_API_USE_32BIT_IDS, KVM_X86_DISABLE_EXITS_HLT, KVM_X86_DISABLE_EXITS_PAUSE,
    kvm_enable_cap, kvm_pit_config,
};
use kvm_ioctls::{Cap, Kvm, VcpuFd, VmFd};
use vm_memory::{GuestAddress, GuestMemoryMmap};

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

/// Virtio-console MMIO base: start of the MMIO gap.
pub(crate) const VIRTIO_CONSOLE_MMIO_BASE: u64 = MMIO_GAP_START;

/// IRQ for virtio-console (GSI routed through IOAPIC/LAPIC).
/// Uses IRQ 5 — available with full IRQ chip. With split IRQ chip
/// (no IOAPIC), MSI would be needed; not supported for now.
pub(crate) const VIRTIO_CONSOLE_IRQ: u32 = 5;

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

/// Per-VM halt poll interval (nanoseconds) for non-performance_mode VMs.
/// Matches the x86 kernel default (KVM_HALT_POLL_NS_DEFAULT in
/// arch/x86/include/asm/kvm_host.h). Set to 0 for overcommitted
/// topologies where halt polling wastes host CPU time.
const HALT_POLL_NS: u64 = 200_000;

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
pub struct KtstrKvm {
    pub kvm: Kvm,
    pub vm_fd: VmFd,
    pub vcpus: Vec<VcpuFd>,
    pub guest_mem: GuestMemoryMmap,
    pub topology: Topology,
    /// Whether KVM supports the immediate_exit mechanism (KVM_CAP_IMMEDIATE_EXIT).
    pub has_immediate_exit: bool,
    /// Split IRQ chip mode: LAPIC in kernel, PIC/IOAPIC emulated in userspace.
    /// Enabled when any APIC ID exceeds the 8-bit xAPIC limit (254).
    pub(crate) split_irqchip: bool,
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
    /// Sets up /dev/kvm, VM fd, TSS, identity map, IRQ chip, vCPUs, and
    /// CPUID — none of which depend on guest memory size. Memory is
    /// allocated later via [`allocate_and_register_memory`].
    pub fn new_deferred(
        topo: Topology,
        use_hugepages: bool,
        performance_mode: bool,
    ) -> Result<Self> {
        Self::new_inner(topo, None, use_hugepages, performance_mode)
    }

    /// Allocate guest memory and register it with KVM.
    ///
    /// Should be called exactly once on a VM created with
    /// `new_deferred`; calling twice unconditionally replaces the
    /// backing memory. Replaces the placeholder guest memory with a
    /// real allocation of `memory_mb` megabytes. Re-checks hugepage
    /// availability when performance_mode is set, since memory_mb was
    /// unknown at construction time and `use_hugepages` may have been
    /// false.
    pub fn allocate_and_register_memory(&mut self, memory_mb: u32) -> Result<()> {
        self.guest_mem = crate::vmm::allocate_and_register_guest_memory(
            &self.vm_fd,
            memory_mb,
            GuestAddress(0),
            self.use_hugepages,
            self.performance_mode,
        )?;
        Ok(())
    }

    fn new_inner(
        topo: Topology,
        memory_mb: Option<u32>,
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

        let vm_fd = crate::vmm::create_vm_with_retry(&kvm)?;

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

        // Disable PAUSE and HLT VM exits in performance mode.
        // Two separate enable_cap calls: kvm_disable_exits() uses |=
        // (additive), so multiple calls accumulate. Separate calls
        // ensure PAUSE succeeds unconditionally even if HLT is rejected.
        //
        // PAUSE: reduces vmexit overhead during guest spinlocks.
        //        Unconditionally allowed by KVM.
        // HLT:   eliminates the most frequent exit type during boot/idle.
        //        BSP shutdown uses I8042 reset (port 0x64, 0xFE via
        //        reboot=k) and VcpuExit::Shutdown, not VcpuExit::Hlt.
        //        KVM blocks HLT disable when mitigate_smt_rsb is active
        //        (host has X86_BUG_SMT_RSB and cpu_smt_possible()).
        if performance_mode {
            let mut cap = kvm_enable_cap {
                cap: KVM_CAP_X86_DISABLE_EXITS,
                ..Default::default()
            };

            // 1. PAUSE — always allowed.
            cap.args[0] = KVM_X86_DISABLE_EXITS_PAUSE as u64;
            if let Err(e) = vm_fd.enable_cap(&cap) {
                eprintln!(
                    "performance_mode: WARNING: \
                     KVM_CAP_X86_DISABLE_EXITS (PAUSE) not supported: {e}"
                );
            }

            // 2. HLT — may fail on mitigate_smt_rsb hosts.
            cap.args[0] = KVM_X86_DISABLE_EXITS_HLT as u64;
            if let Err(e) = vm_fd.enable_cap(&cap) {
                eprintln!(
                    "performance_mode: WARNING: \
                     KVM_CAP_X86_DISABLE_EXITS (HLT) rejected: {e}"
                );
            }
        }

        // Set per-VM halt poll interval. Skipped in performance_mode:
        // KVM_HINTS_REALTIME enables guest haltpoll cpuidle, which writes
        // MSR_KVM_POLL_CONTROL=0 per-vCPU (arch_haltpoll_enable →
        // kvm_disable_host_haltpoll), disabling host halt polling via
        // kvm_arch_no_poll(). KVM_CAP_HALT_POLL is redundant there.
        //
        // When vCPUs exceed online host CPUs (overcommit), halt polling
        // wastes host CPU time — disable it.
        if !performance_mode {
            let host_cpus = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) };
            let poll_ns: u64 = if host_cpus > 0 && topo.total_cpus() <= host_cpus as u32 {
                HALT_POLL_NS
            } else {
                0
            };
            let mut cap = kvm_enable_cap {
                cap: KVM_CAP_HALT_POLL,
                ..Default::default()
            };
            cap.args[0] = poll_ns;
            if let Err(e) = vm_fd.enable_cap(&cap) {
                eprintln!(
                    "kvm: WARNING: KVM_CAP_HALT_POLL not supported ({e}), using kernel default"
                );
            }
        }

        let guest_mem = crate::vmm::allocate_initial_guest_memory(
            &vm_fd,
            memory_mb,
            GuestAddress(0),
            use_hugepages,
            performance_mode,
        )?;

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

        // Check TSC stability via KVM_GET_CLOCK. An unstable TSC
        // (missing KVM_CLOCK_TSC_STABLE) means kvmclock falls back to
        // host-side timekeeping per-vCPU, adding overhead to
        // clock_gettime and degrading timer accuracy. Common in nested
        // virtualization where the L0 hypervisor does not expose
        // constant TSC to L1.
        //
        // Only checked in performance_mode: non-perf tests use binary
        // pass/fail (cpuset, starvation) where timing precision doesn't
        // affect results.
        //
        // A get→set→get roundtrip is required: use_master_clock
        // starts false and is only evaluated by
        // pvclock_update_vm_gtod_copy(). That function is called by
        // kvm_vm_ioctl_set_clock() but NOT by kvm_vm_ioctl_get_clock()
        // or vCPU creation. Without the set_clock() call, get_clock()
        // always returns flags=0 regardless of actual TSC stability.
        //
        // Flags must be cleared before set_clock(): get_clock() may
        // set KVM_CLOCK_REALTIME, and set_clock() applies a realtime
        // adjustment when that flag is present (x86.c:7209-7215),
        // double-counting elapsed time. KVM_CLOCK_TSC_STABLE and
        // KVM_CLOCK_HOST_TSC are output-only and ignored by set_clock().
        if performance_mode {
            match vm_fd.get_clock() {
                Ok(clock) => {
                    let mut set_data = clock;
                    set_data.flags = 0;
                    if let Err(e) = vm_fd.set_clock(&set_data) {
                        eprintln!(
                            "performance_mode: WARNING: KVM_SET_CLOCK failed ({e}), \
                             cannot check TSC stability"
                        );
                    } else {
                        match vm_fd.get_clock() {
                            Ok(clock2) => {
                                if clock2.flags & KVM_CLOCK_TSC_STABLE == 0 {
                                    eprintln!(
                                        "performance_mode: WARNING: TSC not stable \
                                         (KVM_CLOCK_TSC_STABLE not set), \
                                         timing measurements may have higher variance \
                                         (nested virt?)."
                                    );
                                }
                            }
                            Err(e) => {
                                eprintln!(
                                    "performance_mode: WARNING: KVM_GET_CLOCK failed ({e}), \
                                     cannot check TSC stability"
                                );
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!(
                        "performance_mode: WARNING: KVM_GET_CLOCK failed ({e}), \
                         cannot check TSC stability"
                    );
                }
            }
        }

        Ok(KtstrKvm {
            kvm,
            vm_fd,
            vcpus,
            guest_mem,
            topology: topo,
            has_immediate_exit,
            split_irqchip,
            use_hugepages,
            performance_mode,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vm_memory::GuestMemory;

    #[test]
    fn create_vm_basic() {
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let vm = KtstrKvm::new(topo, 128, false);
        assert!(vm.is_ok(), "VM creation failed: {:?}", vm.err());
        let vm = vm.unwrap();
        assert_eq!(vm.vcpus.len(), 2);
    }

    #[test]
    fn create_vm_multi_llc() {
        let topo = Topology {
            llcs: 2,
            cores_per_llc: 2,
            threads_per_core: 2,
            numa_nodes: 1,
        };
        let vm = KtstrKvm::new(topo, 256, false);
        assert!(vm.is_ok(), "multi-LLC VM creation failed: {:?}", vm.err());
        let vm = vm.unwrap();
        assert_eq!(vm.vcpus.len(), 8);
    }

    #[test]
    fn create_vm_single_cpu() {
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let vm = KtstrKvm::new(topo, 64, false);
        assert!(vm.is_ok());
        assert_eq!(vm.unwrap().vcpus.len(), 1);
    }

    #[test]
    fn create_vm_large_topology() {
        let topo = Topology {
            llcs: 4,
            cores_per_llc: 4,
            threads_per_core: 2,
            numa_nodes: 1,
        };
        let vm = KtstrKvm::new(topo, 512, false);
        assert!(vm.is_ok(), "large topology failed: {:?}", vm.err());
        assert_eq!(vm.unwrap().vcpus.len(), 32);
    }

    #[test]
    fn create_vm_odd_topology() {
        let topo = Topology {
            llcs: 3,
            cores_per_llc: 3,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let vm = KtstrKvm::new(topo, 128, false);
        assert!(vm.is_ok(), "odd topology failed: {:?}", vm.err());
        assert_eq!(vm.unwrap().vcpus.len(), 9);
    }

    #[test]
    fn memory_size_correct() {
        use vm_memory::GuestMemoryRegion;
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let vm = KtstrKvm::new(topo, 256, false).unwrap();
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
            llcs: 2,
            cores_per_llc: 4,
            threads_per_core: 2,
            numa_nodes: 1,
        };
        // max APIC ID = apic_id(15) = 1<<3 | 3<<1 | 1 = 15, well under 254
        assert!(max_apic_id(&topo) <= MAX_XAPIC_ID);
        let vm = KtstrKvm::new(topo, 256, false).unwrap();
        assert!(!vm.split_irqchip, "small topology should use full IRQ chip");
    }

    #[test]
    fn large_topology_uses_split_irqchip() {
        // 15 LLCs x 8 cores x 2 threads = 240 vCPUs
        // max APIC ID = apic_id(239) = 14<<4 | 7<<1 | 1 = 239, under 254
        // So try bigger: 14 LLCs x 9 cores x 2 threads = 252 vCPUs
        // core_bits = bits_needed(9) = 4, thread_bits = 1, core_shift = 5
        // max APIC ID = apic_id(251) = 13<<5 | 8<<1 | 1 = 433
        let topo = Topology {
            llcs: 14,
            cores_per_llc: 9,
            threads_per_core: 2,
            numa_nodes: 1,
        };
        assert!(
            max_apic_id(&topo) > MAX_XAPIC_ID,
            "max APIC ID {} should exceed {}",
            max_apic_id(&topo),
            MAX_XAPIC_ID,
        );
        let vm = match KtstrKvm::new(topo, 4096, false) {
            Ok(v) => v,
            Err(e) => {
                // Some hosts reject 252-vCPU VMs (EEXIST from
                // KVM_CREATE_VCPU when split irqchip + x2APIC
                // interact with host KVM limitations). The APIC ID
                // assertion above validates the split irqchip logic;
                // skip the VM creation test on those hosts.
                eprintln!("skipping large_topology VM creation: {e:#}");
                return;
            }
        };
        assert!(vm.split_irqchip, "large topology should use split IRQ chip");
        assert_eq!(vm.vcpus.len(), 252);
    }

    #[test]
    fn split_irqchip_boundary() {
        // Find a topology that is exactly at the boundary.
        // 8 LLCs x 8 cores x 2 threads: core_shift = 4, max APIC ID = 7<<4 | 7<<1 | 1 = 127
        let small = Topology {
            llcs: 8,
            cores_per_llc: 8,
            threads_per_core: 2,
            numa_nodes: 1,
        };
        assert!(
            max_apic_id(&small) <= MAX_XAPIC_ID,
            "8l/8c/2t max APIC ID {} should be <= 254",
            max_apic_id(&small),
        );
        let vm = KtstrKvm::new(small, 2048, false).unwrap();
        assert!(!vm.split_irqchip);

        // 15 LLCs x 8 cores x 2 threads: core_shift = 4, max APIC ID = 14<<4 | 7<<1 | 1 = 239
        let still_small = Topology {
            llcs: 15,
            cores_per_llc: 8,
            threads_per_core: 2,
            numa_nodes: 1,
        };
        assert!(
            max_apic_id(&still_small) <= MAX_XAPIC_ID,
            "15l/8c/2t max APIC ID {} should be <= 254",
            max_apic_id(&still_small),
        );
        let vm = KtstrKvm::new(still_small, 4096, false).unwrap();
        assert!(!vm.split_irqchip);
    }

    #[test]
    fn immediate_exit_cap_detected() {
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let vm = KtstrKvm::new(topo, 64, false).unwrap();
        // KVM_CAP_IMMEDIATE_EXIT is available since Linux 4.12.
        assert!(vm.has_immediate_exit);
    }

    #[test]
    fn performance_mode_succeeds() {
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let vm = KtstrKvm::new(topo, 128, true);
        assert!(
            vm.is_ok(),
            "performance_mode VM creation failed: {:?}",
            vm.err()
        );
    }

    #[test]
    fn performance_mode_does_not_affect_vcpu_count() {
        let topo = Topology {
            llcs: 2,
            cores_per_llc: 2,
            threads_per_core: 2,
            numa_nodes: 1,
        };
        let vm_normal = KtstrKvm::new(topo, 256, false).unwrap();
        let vm_perf = KtstrKvm::new(topo, 256, true).unwrap();
        assert_eq!(vm_normal.vcpus.len(), vm_perf.vcpus.len());
    }

    #[test]
    fn halt_poll_ns_constant() {
        assert_eq!(HALT_POLL_NS, 200_000);
    }

    #[test]
    fn non_perf_mode_succeeds_with_halt_poll() {
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let vm = KtstrKvm::new(topo, 128, false);
        assert!(
            vm.is_ok(),
            "non-perf VM with halt poll failed: {:?}",
            vm.err()
        );
    }

    #[test]
    fn disable_exits_hlt_bit_value() {
        // KVM_X86_DISABLE_EXITS_HLT is bit 1 (value 2) in the kernel ABI.
        assert_eq!(KVM_X86_DISABLE_EXITS_HLT, 2);
    }

    #[test]
    fn disable_exits_pause_and_hlt_no_overlap() {
        assert_ne!(
            KVM_X86_DISABLE_EXITS_PAUSE, KVM_X86_DISABLE_EXITS_HLT,
            "PAUSE and HLT bits must be distinct"
        );
        assert_eq!(
            KVM_X86_DISABLE_EXITS_PAUSE & KVM_X86_DISABLE_EXITS_HLT,
            0,
            "PAUSE and HLT bits must not overlap"
        );
    }

    #[test]
    fn tsc_stability_check_roundtrip() {
        // Verify the get→set→get roundtrip succeeds with
        // performance_mode=true (which enables the TSC check).
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let vm = KtstrKvm::new(topo, 64, true).unwrap();
        let clock = vm.vm_fd.get_clock().unwrap();
        let mut set_data = clock;
        set_data.flags = 0;
        vm.vm_fd.set_clock(&set_data).unwrap();
        let clock2 = vm.vm_fd.get_clock().unwrap();
        // On bare-metal with invariant TSC, KVM_CLOCK_TSC_STABLE
        // should be set after the roundtrip forces
        // pvclock_update_vm_gtod_copy. In nested virt it may not be.
        // Either way, the roundtrip must not fail.
        let _ = clock2.flags & KVM_CLOCK_TSC_STABLE;
    }

    #[test]
    fn performance_mode_with_hlt_disable_succeeds() {
        // performance_mode issues two separate enable_cap calls:
        // PAUSE (always succeeds) then HLT (may be rejected by
        // mitigate_smt_rsb). Either way, VM creation must succeed.
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let vm = KtstrKvm::new(topo, 128, true);
        assert!(
            vm.is_ok(),
            "performance_mode with HLT disable failed: {:?}",
            vm.err()
        );
    }
}
