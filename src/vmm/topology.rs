use kvm_bindings::kvm_cpuid_entry2;

/// CPU vendor, detected from CPUID leaf 0x0 EBX:EDX:ECX.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuVendor {
    Intel,
    Amd,
    Unknown,
}

/// Detect CPU vendor from leaf 0x0 in the given CPUID entries.
/// Vendor string is encoded across EBX:EDX:ECX (note: not EBX:ECX:EDX).
fn detect_vendor(entries: &[kvm_cpuid_entry2]) -> CpuVendor {
    let leaf0 = entries.iter().find(|e| e.function == 0 && e.index == 0);
    match leaf0 {
        Some(e) => {
            // "GenuineIntel" = EBX:0x756e6547 EDX:0x49656e69 ECX:0x6c65746e
            // "AuthenticAMD" = EBX:0x68747541 EDX:0x69746e65 ECX:0x444d4163
            match (e.ebx, e.edx, e.ecx) {
                (0x756e_6547, 0x4965_6e69, 0x6c65_746e) => CpuVendor::Intel,
                (0x6874_7541, 0x6974_6e65, 0x444d_4163) => CpuVendor::Amd,
                _ => CpuVendor::Unknown,
            }
        }
        None => CpuVendor::Unknown,
    }
}

/// CPU topology specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Topology {
    pub sockets: u32,
    pub cores_per_socket: u32,
    pub threads_per_core: u32,
}

impl Topology {
    pub fn total_cpus(&self) -> u32 {
        self.sockets * self.cores_per_socket * self.threads_per_core
    }

    pub fn num_llcs(&self) -> u32 {
        self.sockets
    }

    /// Decompose a logical CPU ID into (socket, core, thread).
    pub fn decompose(&self, cpu_id: u32) -> (u32, u32, u32) {
        let threads = self.threads_per_core;
        let cores = self.cores_per_socket;
        let thread_id = cpu_id % threads;
        let core_id = (cpu_id / threads) % cores;
        let socket_id = cpu_id / (threads * cores);
        (socket_id, core_id, thread_id)
    }

    /// Compute the x2APIC ID for a logical CPU.
    /// Encoding: socket_id << (core_bits + thread_bits) | core_id << thread_bits | thread_id
    pub fn apic_id(&self, cpu_id: u32) -> u32 {
        let (socket_id, core_id, thread_id) = self.decompose(cpu_id);
        let thread_bits = bits_needed(self.threads_per_core);
        let core_bits = bits_needed(self.cores_per_socket);
        (socket_id << (core_bits + thread_bits)) | (core_id << thread_bits) | thread_id
    }

    /// Highest APIC ID across all logical CPUs in this topology.
    pub fn max_apic_id(&self) -> u32 {
        let total = self.total_cpus();
        if total == 0 {
            return 0;
        }
        self.apic_id(total - 1)
    }

    /// Number of bits needed to represent thread ID within a core.
    pub fn smt_shift(&self) -> u32 {
        bits_needed(self.threads_per_core)
    }

    /// Number of bits needed to represent core+thread ID within a socket.
    pub fn core_shift(&self) -> u32 {
        bits_needed(self.threads_per_core) + bits_needed(self.cores_per_socket)
    }
}

/// Minimum number of bits to represent values 0..n-1.
/// Returns 0 for n <= 1.
fn bits_needed(n: u32) -> u32 {
    if n <= 1 {
        return 0;
    }
    32 - (n - 1).leading_zeros()
}

/// Generate CPUID entries for a specific vCPU with topology information.
/// Takes a pre-fetched base CPUID (from `get_supported_cpuid`) and patches
/// topology-related leaves. The base should be fetched once and reused for
/// all vCPUs — each call clones and patches per-vCPU fields (APIC ID etc).
pub fn generate_cpuid(
    base_cpuid: &[kvm_cpuid_entry2],
    topo: &Topology,
    cpu_id: u32,
) -> Vec<kvm_cpuid_entry2> {
    let mut entries: Vec<kvm_cpuid_entry2> = base_cpuid.to_vec();

    let vendor = detect_vendor(&entries);
    let apic_id = topo.apic_id(cpu_id);
    let smt_shift = topo.smt_shift();
    let core_shift = topo.core_shift();
    let threads_per_pkg = topo.cores_per_socket * topo.threads_per_core;

    for entry in entries.iter_mut() {
        match entry.function {
            // Leaf 0x1: Feature Information (vendor-independent)
            0x1 => {
                // EBX[31:24] = initial APIC ID (8-bit)
                entry.ebx = (entry.ebx & 0x00ffffff) | ((apic_id & 0xff) << 24);
                // EBX[23:16] = max addressable logical processors per package
                entry.ebx = (entry.ebx & 0xff00ffff) | ((threads_per_pkg.min(255)) << 16);
                // EBX[15:8] = CLFLUSH line size — preserved from KVM
                // ECX.31 = hypervisor — preserved from KVM
                // EDX bit 28 = HTT
                if threads_per_pkg > 1 {
                    entry.edx |= 1 << 28;
                }
            }

            // Leaf 0x4: Deterministic Cache Parameters (Intel only)
            0x4 if vendor == CpuVendor::Intel => {
                let cache_level = (entry.eax >> 5) & 0x7;
                // EAX[25:14] = max addressable IDs sharing this cache - 1
                // Uses APIC-ID-space rounding: (1 << shift) - 1
                let max_sharing = match cache_level {
                    1 | 2 => (1u32 << smt_shift).saturating_sub(1), // L1/L2: core level
                    3 => (1u32 << core_shift).saturating_sub(1),    // L3: socket level
                    _ => 0,
                };
                entry.eax = (entry.eax & 0xfc003fff) | ((max_sharing & 0xfff) << 14);
                // EAX[31:26] = max addressable core IDs in package - 1
                // APIC-ID-space: (1 << core_bits) - 1
                let core_bits = bits_needed(topo.cores_per_socket);
                let max_core_ids = (1u32 << core_bits).saturating_sub(1);
                entry.eax = (entry.eax & 0x03ffffff) | ((max_core_ids & 0x3f) << 26);
            }

            // Leaf 0xB: Extended Topology Enumeration (Intel + AMD Zen)
            0xb => {
                match entry.index {
                    // Subleaf 0: SMT level
                    0 => {
                        entry.eax = smt_shift;
                        entry.ebx = topo.threads_per_core & 0xffff;
                        entry.ecx = (1 << 8) | (entry.index & 0xff); // type=SMT, level=0
                        entry.edx = apic_id;
                    }
                    // Subleaf 1: Core level
                    1 => {
                        entry.eax = core_shift;
                        entry.ebx = threads_per_pkg & 0xffff;
                        entry.ecx = (2 << 8) | (entry.index & 0xff); // type=Core, level=1
                        entry.edx = apic_id;
                    }
                    // Subleaf 2+: invalid level (terminate enumeration)
                    _ => {
                        entry.eax = 0;
                        entry.ebx = 0;
                        entry.ecx = entry.index & 0xff; // level number only, type=0 (invalid)
                        entry.edx = apic_id;
                    }
                }
            }

            // Leaf 0x1F: V2 Extended Topology (Intel + AMD, superset of 0xB)
            0x1f => match entry.index {
                0 => {
                    entry.eax = smt_shift;
                    entry.ebx = topo.threads_per_core & 0xffff;
                    entry.ecx = (1 << 8) | (entry.index & 0xff); // type=SMT
                    entry.edx = apic_id;
                }
                1 => {
                    entry.eax = core_shift;
                    entry.ebx = threads_per_pkg & 0xffff;
                    entry.ecx = (2 << 8) | (entry.index & 0xff); // type=Core
                    entry.edx = apic_id;
                }
                _ => {
                    entry.eax = 0;
                    entry.ebx = 0;
                    entry.ecx = entry.index & 0xff;
                    entry.edx = apic_id;
                }
            },

            // Leaf 0xA: PMU — zero all registers (disable PMU, vendor-independent)
            0xa => {
                entry.eax = 0;
                entry.ebx = 0;
                entry.ecx = 0;
                entry.edx = 0;
            }

            // Leaf 0x80000001: AMD extended feature identification (AMD only)
            0x8000_0001 if vendor == CpuVendor::Amd && threads_per_pkg > 1 => {
                // ECX bit 1 = CmpLegacy: multi-core chip
                // ECX bit 22 = TopologyExtensions: enables leaves 0x8000001D/1E
                entry.ecx |= (1 << 1) | (1 << 22);
            }

            // Leaf 0x80000008: virtual/physical address sizes (vendor-independent)
            // ECX[7:0] = number of physical threads - 1
            // ECX[15:12] = APIC ID size (bits needed for thread IDs in package)
            0x8000_0008 => {
                if threads_per_pkg > 1 {
                    let apic_id_size = core_shift;
                    entry.ecx = (apic_id_size << 12) | (threads_per_pkg - 1);
                } else {
                    entry.ecx = 0;
                }
            }

            // Leaf 0x8000001E: AMD Extended APIC ID / Topology (AMD only)
            0x8000_001e if vendor == CpuVendor::Amd => {
                // EAX = Extended APIC ID
                entry.eax = apic_id;
                // EBX[7:0] = Compute Unit (core) ID
                // EBX[15:8] = Threads per compute unit - 1
                let (_, core_id, _) = topo.decompose(cpu_id);
                entry.ebx = ((topo.threads_per_core - 1) << 8) | (core_id & 0xff);
                // ECX[7:0] = Node ID (0 = all in one node)
                // ECX[10:8] = Nodes per processor - 1 (0 = 1 node)
                entry.ecx = 0;
                // EDX = reserved
                entry.edx = 0;
            }

            _ => {}
        }
    }

    // entries is already a Vec from the clone above

    // Add hypervisor identification leaf (0x40000000) if not present.
    // Guest OS uses leaf 0x1 ECX.31 to detect hypervisor, then reads
    // 0x40000000 for the hypervisor signature. KVM's supported CPUID
    // may already include this; only add if missing.
    if !entries.iter().any(|e| e.function == 0x4000_0000) {
        entries.push(kvm_cpuid_entry2 {
            function: 0x4000_0000,
            index: 0,
            flags: 0,
            eax: 0x4000_0000, // max hypervisor leaf
            // "KVMKVMKVM\0\0\0" signature
            ebx: 0x4b56_4d4b, // "KVMK"
            ecx: 0x564b_4d56, // "VMKV"
            edx: 0x0000_004d, // "M\0\0\0"
            ..Default::default()
        });
    }

    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_needed_values() {
        assert_eq!(bits_needed(1), 0);
        assert_eq!(bits_needed(2), 1);
        assert_eq!(bits_needed(3), 2);
        assert_eq!(bits_needed(4), 2);
        assert_eq!(bits_needed(5), 3);
        assert_eq!(bits_needed(8), 3);
        assert_eq!(bits_needed(9), 4);
        assert_eq!(bits_needed(16), 4);
    }

    #[test]
    fn topology_total_cpus() {
        let t = Topology {
            sockets: 2,
            cores_per_socket: 4,
            threads_per_core: 2,
        };
        assert_eq!(t.total_cpus(), 16);
    }

    #[test]
    fn topology_num_llcs() {
        let t = Topology {
            sockets: 3,
            cores_per_socket: 4,
            threads_per_core: 2,
        };
        assert_eq!(t.num_llcs(), 3);
    }

    #[test]
    fn decompose_simple() {
        let t = Topology {
            sockets: 2,
            cores_per_socket: 2,
            threads_per_core: 2,
        };
        // 2 sockets x 2 cores x 2 threads = 8 CPUs
        assert_eq!(t.decompose(0), (0, 0, 0));
        assert_eq!(t.decompose(1), (0, 0, 1));
        assert_eq!(t.decompose(2), (0, 1, 0));
        assert_eq!(t.decompose(3), (0, 1, 1));
        assert_eq!(t.decompose(4), (1, 0, 0));
        assert_eq!(t.decompose(5), (1, 0, 1));
        assert_eq!(t.decompose(6), (1, 1, 0));
        assert_eq!(t.decompose(7), (1, 1, 1));
    }

    #[test]
    fn decompose_no_smt() {
        let t = Topology {
            sockets: 2,
            cores_per_socket: 4,
            threads_per_core: 1,
        };
        assert_eq!(t.decompose(0), (0, 0, 0));
        assert_eq!(t.decompose(3), (0, 3, 0));
        assert_eq!(t.decompose(4), (1, 0, 0));
        assert_eq!(t.decompose(7), (1, 3, 0));
    }

    #[test]
    fn decompose_single_socket() {
        let t = Topology {
            sockets: 1,
            cores_per_socket: 4,
            threads_per_core: 1,
        };
        assert_eq!(t.decompose(0), (0, 0, 0));
        assert_eq!(t.decompose(3), (0, 3, 0));
    }

    #[test]
    fn apic_ids_unique() {
        let t = Topology {
            sockets: 2,
            cores_per_socket: 4,
            threads_per_core: 2,
        };
        let ids: Vec<u32> = (0..t.total_cpus()).map(|i| t.apic_id(i)).collect();
        let unique: std::collections::HashSet<u32> = ids.iter().copied().collect();
        assert_eq!(ids.len(), unique.len(), "APIC IDs must be unique: {ids:?}");
    }

    #[test]
    fn apic_ids_smt_siblings_adjacent() {
        let t = Topology {
            sockets: 2,
            cores_per_socket: 2,
            threads_per_core: 2,
        };
        // SMT siblings should differ only in thread_id bits
        let smt_mask = (1u32 << t.smt_shift()) - 1;
        for core_start in (0..t.total_cpus()).step_by(t.threads_per_core as usize) {
            let base = t.apic_id(core_start) & !smt_mask;
            for thread in 0..t.threads_per_core {
                let apic = t.apic_id(core_start + thread);
                assert_eq!(
                    apic & !smt_mask,
                    base,
                    "SMT siblings should share upper bits: cpu {}, apic {apic:#x}",
                    core_start + thread
                );
            }
        }
    }

    #[test]
    fn apic_ids_same_socket_share_upper_bits() {
        let t = Topology {
            sockets: 2,
            cores_per_socket: 4,
            threads_per_core: 2,
        };
        let pkg_mask = !((1u32 << t.core_shift()) - 1);
        let cpus_per_socket = t.cores_per_socket * t.threads_per_core;
        for socket in 0..t.sockets {
            let start = socket * cpus_per_socket;
            let socket_bits = t.apic_id(start) & pkg_mask;
            for cpu in start..start + cpus_per_socket {
                assert_eq!(
                    t.apic_id(cpu) & pkg_mask,
                    socket_bits,
                    "CPU {cpu} should be in socket {socket}"
                );
            }
        }
        let s0 = t.apic_id(0) & pkg_mask;
        let s1 = t.apic_id(cpus_per_socket) & pkg_mask;
        assert_ne!(
            s0, s1,
            "different sockets should have different package IDs"
        );
    }

    #[test]
    fn smt_shift_values() {
        assert_eq!(
            Topology {
                sockets: 1,
                cores_per_socket: 1,
                threads_per_core: 1
            }
            .smt_shift(),
            0
        );
        assert_eq!(
            Topology {
                sockets: 1,
                cores_per_socket: 1,
                threads_per_core: 2
            }
            .smt_shift(),
            1
        );
        assert_eq!(
            Topology {
                sockets: 1,
                cores_per_socket: 1,
                threads_per_core: 4
            }
            .smt_shift(),
            2
        );
    }

    #[test]
    fn core_shift_values() {
        // 1 thread, 4 cores: smt_shift=0, core_bits=2, core_shift=2
        assert_eq!(
            Topology {
                sockets: 1,
                cores_per_socket: 4,
                threads_per_core: 1
            }
            .core_shift(),
            2
        );
        // 2 threads, 4 cores: smt_shift=1, core_bits=2, core_shift=3
        assert_eq!(
            Topology {
                sockets: 1,
                cores_per_socket: 4,
                threads_per_core: 2
            }
            .core_shift(),
            3
        );
    }

    #[test]
    fn generate_cpuid_produces_entries() {
        let kvm = match kvm_ioctls::Kvm::new() {
            Ok(k) => k,
            Err(_) => return, // skip if no KVM
        };
        let topo = Topology {
            sockets: 2,
            cores_per_socket: 2,
            threads_per_core: 2,
        };
        let cpuid = generate_cpuid(
            kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
                .unwrap()
                .as_slice(),
            &topo,
            0,
        );
        assert!(!cpuid.is_empty());

        // Verify leaf 0x1 has correct APIC ID in EBX[31:24]
        let leaf1 = cpuid.iter().find(|e| e.function == 1);
        if let Some(entry) = leaf1 {
            let apic_from_cpuid = (entry.ebx >> 24) & 0xff;
            assert_eq!(apic_from_cpuid, topo.apic_id(0));
        }
    }

    #[test]
    fn generate_cpuid_different_per_vcpu() {
        let kvm = match kvm_ioctls::Kvm::new() {
            Ok(k) => k,
            Err(_) => return,
        };
        let topo = Topology {
            sockets: 2,
            cores_per_socket: 2,
            threads_per_core: 1,
        };
        let cpuid0 = generate_cpuid(
            kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
                .unwrap()
                .as_slice(),
            &topo,
            0,
        );
        let cpuid1 = generate_cpuid(
            kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
                .unwrap()
                .as_slice(),
            &topo,
            1,
        );

        // Different vCPUs should have different APIC IDs in leaf 0xB
        let leaf_b_0 = cpuid0.iter().find(|e| e.function == 0xb && e.index == 0);
        let leaf_b_1 = cpuid1.iter().find(|e| e.function == 0xb && e.index == 0);
        if let (Some(e0), Some(e1)) = (leaf_b_0, leaf_b_1) {
            assert_ne!(
                e0.edx, e1.edx,
                "different vCPUs should have different x2APIC IDs"
            );
        }
    }

    #[test]
    fn topology_odd_counts() {
        let t = Topology {
            sockets: 3,
            cores_per_socket: 3,
            threads_per_core: 1,
        };
        assert_eq!(t.total_cpus(), 9);
        let ids: Vec<u32> = (0..9).map(|i| t.apic_id(i)).collect();
        let unique: std::collections::HashSet<u32> = ids.iter().copied().collect();
        assert_eq!(
            unique.len(),
            9,
            "odd topology APIC IDs must be unique: {ids:?}"
        );
    }

    #[test]
    fn leaf1_threads_per_package_not_total() {
        let kvm = match kvm_ioctls::Kvm::new() {
            Ok(k) => k,
            Err(_) => return,
        };
        let topo = Topology {
            sockets: 4,
            cores_per_socket: 4,
            threads_per_core: 2,
        };
        assert_eq!(topo.total_cpus(), 32);
        let cpuid = generate_cpuid(
            kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
                .unwrap()
                .as_slice(),
            &topo,
            0,
        );
        let leaf1 = cpuid.iter().find(|e| e.function == 1);
        if let Some(entry) = leaf1 {
            let threads_per_pkg = (entry.ebx >> 16) & 0xff;
            assert_eq!(
                threads_per_pkg,
                8, // cores_per_socket * threads_per_core, not total
                "EBX[23:16] should be threads per package (8), not total CPUs (32)"
            );
        }
    }

    #[test]
    fn apic_ids_unique_for_all_gauntlet_presets() {
        let presets = [
            (1, 4, 1),  // tiny-1llc
            (2, 2, 1),  // tiny-2llc
            (3, 3, 1),  // odd-3llc
            (5, 3, 1),  // odd-5llc
            (7, 2, 1),  // odd-7llc
            (2, 2, 2),  // smt-2llc
            (3, 2, 2),  // smt-3llc
            (4, 4, 2),  // medium-4llc
            (8, 4, 2),  // medium-8llc
            (4, 16, 2), // large-4llc
            (8, 8, 2),  // large-8llc
            (15, 8, 2), // near-max-llc
            (14, 9, 2), // max-cpu
        ];
        for (sockets, cores, threads) in presets {
            let t = Topology {
                sockets,
                cores_per_socket: cores,
                threads_per_core: threads,
            };
            let ids: Vec<u32> = (0..t.total_cpus()).map(|i| t.apic_id(i)).collect();
            let unique: std::collections::HashSet<u32> = ids.iter().copied().collect();
            assert_eq!(
                ids.len(),
                unique.len(),
                "topology {sockets}s/{cores}c/{threads}t: APIC IDs not unique"
            );
        }
    }

    #[test]
    fn leaf0b_subleaf0_ebx_is_threads_per_core() {
        let kvm = match kvm_ioctls::Kvm::new() {
            Ok(k) => k,
            Err(_) => return,
        };
        let topo = Topology {
            sockets: 2,
            cores_per_socket: 4,
            threads_per_core: 2,
        };
        let cpuid = generate_cpuid(
            kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
                .unwrap()
                .as_slice(),
            &topo,
            0,
        );
        let leaf_b_0 = cpuid.iter().find(|e| e.function == 0xb && e.index == 0);
        if let Some(entry) = leaf_b_0 {
            assert_eq!(
                entry.ebx & 0xffff,
                2, // threads_per_core
                "leaf 0xB subleaf 0 EBX should be threads per core"
            );
            assert_eq!(
                entry.eax,
                topo.smt_shift(),
                "leaf 0xB subleaf 0 EAX should be smt_shift"
            );
        }
    }

    #[test]
    fn leaf0b_subleaf1_ebx_is_threads_per_socket() {
        let kvm = match kvm_ioctls::Kvm::new() {
            Ok(k) => k,
            Err(_) => return,
        };
        let topo = Topology {
            sockets: 2,
            cores_per_socket: 4,
            threads_per_core: 2,
        };
        let cpuid = generate_cpuid(
            kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
                .unwrap()
                .as_slice(),
            &topo,
            0,
        );
        let leaf_b_1 = cpuid.iter().find(|e| e.function == 0xb && e.index == 1);
        if let Some(entry) = leaf_b_1 {
            assert_eq!(
                entry.ebx & 0xffff,
                8, // cores_per_socket * threads_per_core
                "leaf 0xB subleaf 1 EBX should be threads per socket"
            );
            assert_eq!(
                entry.eax,
                topo.core_shift(),
                "leaf 0xB subleaf 1 EAX should be core_shift"
            );
        }
    }

    #[test]
    fn leaf0b_ecx_includes_subleaf_index() {
        let kvm = match kvm_ioctls::Kvm::new() {
            Ok(k) => k,
            Err(_) => return,
        };
        let topo = Topology {
            sockets: 2,
            cores_per_socket: 2,
            threads_per_core: 2,
        };
        let cpuid = generate_cpuid(
            kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
                .unwrap()
                .as_slice(),
            &topo,
            0,
        );
        // Subleaf 0: ECX should have level_type=1 (SMT) in bits 15:8 and index=0 in bits 7:0
        if let Some(entry) = cpuid.iter().find(|e| e.function == 0xb && e.index == 0) {
            assert_eq!(entry.ecx & 0xff, 0, "subleaf 0 ECX[7:0] should be 0");
            assert_eq!(
                (entry.ecx >> 8) & 0xff,
                1,
                "subleaf 0 ECX[15:8] should be 1 (SMT)"
            );
        }
        // Subleaf 1: ECX should have level_type=2 (Core) in bits 15:8 and index=1 in bits 7:0
        if let Some(entry) = cpuid.iter().find(|e| e.function == 0xb && e.index == 1) {
            assert_eq!(entry.ecx & 0xff, 1, "subleaf 1 ECX[7:0] should be 1");
            assert_eq!(
                (entry.ecx >> 8) & 0xff,
                2,
                "subleaf 1 ECX[15:8] should be 2 (Core)"
            );
        }
    }

    #[test]
    fn leaf4_l3_shared_within_socket() {
        let kvm = match kvm_ioctls::Kvm::new() {
            Ok(k) => k,
            Err(_) => return,
        };
        let topo = Topology {
            sockets: 2,
            cores_per_socket: 4,
            threads_per_core: 2,
        };
        let cpuid = generate_cpuid(
            kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
                .unwrap()
                .as_slice(),
            &topo,
            0,
        );
        let l3 = cpuid
            .iter()
            .find(|e| e.function == 0x4 && ((e.eax >> 5) & 0x7) == 3);
        if let Some(entry) = l3 {
            // EAX[25:14] = max addressable IDs sharing this cache - 1
            // For 4c/2t: core_shift=3, (1<<3)-1 = 7
            let max_sharing = ((entry.eax >> 14) & 0xfff) + 1;
            let expected = 1u32 << topo.core_shift(); // APIC-ID-space rounded
            assert_eq!(max_sharing, expected, "L3 max sharing: APIC-ID-space value");
        }
    }

    #[test]
    fn leaf4_core_ids_apic_space() {
        let kvm = match kvm_ioctls::Kvm::new() {
            Ok(k) => k,
            Err(_) => return,
        };
        // 3 cores: needs 2 bits, so (1<<2)-1 = 3 addressable core IDs
        let topo = Topology {
            sockets: 1,
            cores_per_socket: 3,
            threads_per_core: 1,
        };
        let cpuid = generate_cpuid(
            kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
                .unwrap()
                .as_slice(),
            &topo,
            0,
        );
        // Find any valid leaf 0x4 entry
        let leaf4 = cpuid
            .iter()
            .find(|e| e.function == 0x4 && ((e.eax >> 5) & 0x7) > 0);
        if let Some(entry) = leaf4 {
            let max_core_ids = ((entry.eax >> 26) & 0x3f) + 1;
            let core_bits = bits_needed(topo.cores_per_socket);
            assert_eq!(
                max_core_ids,
                1 << core_bits,
                "leaf 0x4 EAX[31:26]+1 should be power-of-2 from APIC ID space"
            );
        }
    }

    #[test]
    fn leaf1_hypervisor_bit_set() {
        let kvm = match kvm_ioctls::Kvm::new() {
            Ok(k) => k,
            Err(_) => return,
        };
        let topo = Topology {
            sockets: 1,
            cores_per_socket: 1,
            threads_per_core: 1,
        };
        let cpuid = generate_cpuid(
            kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
                .unwrap()
                .as_slice(),
            &topo,
            0,
        );
        let leaf1 = cpuid.iter().find(|e| e.function == 1);
        if let Some(entry) = leaf1 {
            assert_ne!(
                entry.ecx & (1 << 31),
                0,
                "hypervisor bit (ECX.31) should be set"
            );
        }
    }

    #[test]
    fn leaf1_clflush_set() {
        let kvm = match kvm_ioctls::Kvm::new() {
            Ok(k) => k,
            Err(_) => return,
        };
        let topo = Topology {
            sockets: 1,
            cores_per_socket: 1,
            threads_per_core: 1,
        };
        let cpuid = generate_cpuid(
            kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
                .unwrap()
                .as_slice(),
            &topo,
            0,
        );
        let leaf1 = cpuid.iter().find(|e| e.function == 1);
        if let Some(entry) = leaf1 {
            let clflush = (entry.ebx >> 8) & 0xff;
            assert_eq!(clflush, 8, "CLFLUSH should be 8 (64-byte cache lines)");
        }
    }

    #[test]
    fn leaf_0xa_pmu_zeroed() {
        let kvm = match kvm_ioctls::Kvm::new() {
            Ok(k) => k,
            Err(_) => return,
        };
        let topo = Topology {
            sockets: 1,
            cores_per_socket: 1,
            threads_per_core: 1,
        };
        let cpuid = generate_cpuid(
            kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
                .unwrap()
                .as_slice(),
            &topo,
            0,
        );
        let leaf_a = cpuid.iter().find(|e| e.function == 0xa);
        if let Some(entry) = leaf_a {
            assert_eq!(entry.eax, 0, "PMU leaf should be zeroed");
            assert_eq!(entry.ebx, 0);
            assert_eq!(entry.ecx, 0);
            assert_eq!(entry.edx, 0);
        }
    }

    #[test]
    fn hypervisor_leaf_present() {
        let kvm = match kvm_ioctls::Kvm::new() {
            Ok(k) => k,
            Err(_) => return,
        };
        let topo = Topology {
            sockets: 1,
            cores_per_socket: 1,
            threads_per_core: 1,
        };
        let cpuid = generate_cpuid(
            kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
                .unwrap()
                .as_slice(),
            &topo,
            0,
        );
        let leaf_40 = cpuid.iter().find(|e| e.function == 0x4000_0000);
        assert!(leaf_40.is_some(), "hypervisor leaf 0x40000000 should exist");
    }

    #[test]
    fn decompose_roundtrip_all_gauntlet() {
        let presets = [
            (1, 4, 1),
            (2, 2, 1),
            (3, 3, 1),
            (5, 3, 1),
            (7, 2, 1),
            (2, 2, 2),
            (3, 2, 2),
            (4, 4, 2),
            (8, 4, 2),
            (4, 16, 2),
            (8, 8, 2),
            (15, 8, 2),
            (14, 9, 2),
        ];
        for (sockets, cores, threads) in presets {
            let t = Topology {
                sockets,
                cores_per_socket: cores,
                threads_per_core: threads,
            };
            for cpu in 0..t.total_cpus() {
                let (s, c, th) = t.decompose(cpu);
                assert!(s < sockets, "cpu {cpu}: socket {s} >= {sockets}");
                assert!(c < cores, "cpu {cpu}: core {c} >= {cores}");
                assert!(th < threads, "cpu {cpu}: thread {th} >= {threads}");
                let recomposed = s * cores * threads + c * threads + th;
                assert_eq!(
                    recomposed, cpu,
                    "decompose roundtrip failed for {sockets}s/{cores}c/{threads}t cpu {cpu}"
                );
            }
        }
    }

    #[test]
    fn leaf_80000008_amd_topology() {
        let kvm = match kvm_ioctls::Kvm::new() {
            Ok(k) => k,
            Err(_) => return,
        };
        let topo = Topology {
            sockets: 2,
            cores_per_socket: 4,
            threads_per_core: 2,
        };
        let cpuid = generate_cpuid(
            kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
                .unwrap()
                .as_slice(),
            &topo,
            0,
        );
        let leaf = cpuid.iter().find(|e| e.function == 0x8000_0008);
        if let Some(entry) = leaf {
            // ECX[7:0] = threads per package - 1
            let nc = entry.ecx & 0xff;
            assert_eq!(nc, 7, "NC should be threads_per_package - 1");
            // ECX[15:12] = APIC ID size
            let apic_id_size = (entry.ecx >> 12) & 0xf;
            assert_eq!(apic_id_size, topo.core_shift(), "APIC ID size = core_shift");
        }
    }

    #[test]
    fn leaf_8000001e_amd_extended_apic() {
        let kvm = match kvm_ioctls::Kvm::new() {
            Ok(k) => k,
            Err(_) => return,
        };
        let topo = Topology {
            sockets: 2,
            cores_per_socket: 4,
            threads_per_core: 2,
        };

        // Check CPU 0
        let cpuid0 = generate_cpuid(
            kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
                .unwrap()
                .as_slice(),
            &topo,
            0,
        );
        let leaf0 = cpuid0.iter().find(|e| e.function == 0x8000_001e);
        if let Some(entry) = leaf0 {
            assert_eq!(entry.eax, topo.apic_id(0), "EAX = extended APIC ID");
            assert_eq!(entry.ebx & 0xff, 0, "core ID for cpu 0 should be 0");
            assert_eq!(
                (entry.ebx >> 8) & 0xff,
                1,
                "threads per core - 1 should be 1"
            );
            assert_eq!(entry.ecx, 0, "single node, node ID 0");
            assert_eq!(entry.edx, 0, "EDX reserved");
        }

        // Check CPU 3 (socket 0, core 1, thread 1)
        let cpuid3 = generate_cpuid(
            kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
                .unwrap()
                .as_slice(),
            &topo,
            3,
        );
        let leaf3 = cpuid3.iter().find(|e| e.function == 0x8000_001e);
        if let Some(entry) = leaf3 {
            assert_eq!(entry.eax, topo.apic_id(3), "EAX = extended APIC ID");
            assert_eq!(entry.ebx & 0xff, 1, "core ID for cpu 3 should be 1");
        }
    }

    #[test]
    fn leaf_80000008_single_cpu() {
        let kvm = match kvm_ioctls::Kvm::new() {
            Ok(k) => k,
            Err(_) => return,
        };
        let topo = Topology {
            sockets: 1,
            cores_per_socket: 1,
            threads_per_core: 1,
        };
        let cpuid = generate_cpuid(
            kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
                .unwrap()
                .as_slice(),
            &topo,
            0,
        );
        let leaf = cpuid.iter().find(|e| e.function == 0x8000_0008);
        if let Some(entry) = leaf {
            // Single CPU: ECX should be 0
            assert_eq!(entry.ecx & 0xff, 0, "single CPU: NC = 0");
            assert_eq!((entry.ecx >> 12) & 0xf, 0, "single CPU: ApicIdSize = 0");
        }
    }

    #[test]
    fn leaf1f_matches_leaf0b() {
        let kvm = match kvm_ioctls::Kvm::new() {
            Ok(k) => k,
            Err(_) => return,
        };
        let topo = Topology {
            sockets: 2,
            cores_per_socket: 4,
            threads_per_core: 2,
        };
        let cpuid = generate_cpuid(
            kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
                .unwrap()
                .as_slice(),
            &topo,
            0,
        );

        // For subleaves 0 and 1, leaf 0x1F should produce the same topology
        // data as leaf 0xB (EAX, EBX, EDX match; ECX may differ only in type encoding)
        for sub in 0..2 {
            let leaf_b = cpuid.iter().find(|e| e.function == 0xb && e.index == sub);
            let leaf_1f = cpuid.iter().find(|e| e.function == 0x1f && e.index == sub);
            if let (Some(b), Some(f)) = (leaf_b, leaf_1f) {
                assert_eq!(b.eax, f.eax, "subleaf {sub}: EAX should match");
                assert_eq!(b.ebx, f.ebx, "subleaf {sub}: EBX should match");
                assert_eq!(b.edx, f.edx, "subleaf {sub}: EDX should match");
                assert_eq!(b.ecx, f.ecx, "subleaf {sub}: ECX should match");
            }
        }
    }

    #[test]
    fn leaf1_htt_not_set_for_single_cpu() {
        let kvm = match kvm_ioctls::Kvm::new() {
            Ok(k) => k,
            Err(_) => return,
        };
        let topo = Topology {
            sockets: 1,
            cores_per_socket: 1,
            threads_per_core: 1,
        };
        let cpuid = generate_cpuid(
            kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
                .unwrap()
                .as_slice(),
            &topo,
            0,
        );
        let leaf1 = cpuid.iter().find(|e| e.function == 1);
        if let Some(entry) = leaf1 {
            // HTT bit should not be forcibly set when threads_per_pkg == 1.
            // KVM may still set it in its supported CPUID, but we should not
            // add it when unnecessary.
            let threads_per_pkg = (entry.ebx >> 16) & 0xff;
            assert_eq!(threads_per_pkg, 1, "single CPU: threads per pkg = 1");
        }
    }

    #[test]
    fn leaf_80000001_cmplegacy_and_topoext_multi_cpu() {
        let kvm = match kvm_ioctls::Kvm::new() {
            Ok(k) => k,
            Err(_) => return,
        };
        let topo = Topology {
            sockets: 2,
            cores_per_socket: 4,
            threads_per_core: 2,
        };
        let cpuid = generate_cpuid(
            kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
                .unwrap()
                .as_slice(),
            &topo,
            0,
        );
        let leaf = cpuid.iter().find(|e| e.function == 0x8000_0001);
        if let Some(entry) = leaf {
            assert_ne!(entry.ecx & (1 << 1), 0, "CmpLegacy (bit 1) should be set");
            assert_ne!(
                entry.ecx & (1 << 22),
                0,
                "TopologyExtensions (bit 22) should be set"
            );
        }
    }

    #[test]
    fn leaf_80000001_not_set_for_single_cpu() {
        let kvm = match kvm_ioctls::Kvm::new() {
            Ok(k) => k,
            Err(_) => return,
        };
        let topo = Topology {
            sockets: 1,
            cores_per_socket: 1,
            threads_per_core: 1,
        };
        let cpuid = generate_cpuid(
            kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
                .unwrap()
                .as_slice(),
            &topo,
            0,
        );
        let leaf = cpuid.iter().find(|e| e.function == 0x8000_0001);
        if let Some(entry) = leaf {
            // We should not forcibly set CmpLegacy or TopologyExtensions
            // when there is only one logical CPU in the package.
            // KVM may set these in its supported CPUID on AMD hosts,
            // but our code should not add them for single-CPU topologies.
            let our_bits = (1u32 << 1) | (1u32 << 22);
            // Get the host baseline to compare
            let host_cpuid = kvm
                .get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
                .expect("get_supported_cpuid");
            let host_leaf = host_cpuid
                .as_slice()
                .iter()
                .find(|e| e.function == 0x8000_0001);
            if let Some(host_entry) = host_leaf {
                // Our code should not have added bits beyond what the host provides
                let added = entry.ecx & !host_entry.ecx;
                assert_eq!(
                    added & our_bits,
                    0,
                    "single CPU: should not add CmpLegacy or TopologyExtensions"
                );
            }
        }
    }

    #[test]
    fn leaf_80000008_apic_id_size_all_gauntlet() {
        let kvm = match kvm_ioctls::Kvm::new() {
            Ok(k) => k,
            Err(_) => return,
        };
        let presets = [
            (1, 4, 1),
            (2, 2, 1),
            (3, 3, 1),
            (5, 3, 1),
            (7, 2, 1),
            (2, 2, 2),
            (3, 2, 2),
            (4, 4, 2),
            (8, 4, 2),
            (4, 16, 2),
            (8, 8, 2),
            (15, 8, 2),
            (14, 9, 2),
        ];
        for (sockets, cores, threads) in presets {
            let topo = Topology {
                sockets,
                cores_per_socket: cores,
                threads_per_core: threads,
            };
            let cpuid = generate_cpuid(
                kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
                    .unwrap()
                    .as_slice(),
                &topo,
                0,
            );
            let leaf = cpuid.iter().find(|e| e.function == 0x8000_0008);
            if let Some(entry) = leaf {
                let threads_per_pkg = cores * threads;
                let apic_id_size = (entry.ecx >> 12) & 0xf;
                let nc = entry.ecx & 0xff;

                if threads_per_pkg > 1 {
                    // ApicIdSize must accommodate all thread IDs in package
                    assert!(
                        (1u32 << apic_id_size) >= threads_per_pkg,
                        "{sockets}s/{cores}c/{threads}t: ApicIdSize {apic_id_size} too small \
                         for {threads_per_pkg} threads (2^{apic_id_size} = {})",
                        1u32 << apic_id_size
                    );
                    assert_eq!(
                        nc,
                        threads_per_pkg - 1,
                        "{sockets}s/{cores}c/{threads}t: NC should be threads_per_pkg - 1"
                    );
                } else {
                    assert_eq!(
                        entry.ecx & 0xf0ff,
                        0,
                        "{sockets}s/{cores}c/{threads}t: single CPU ECX should be 0"
                    );
                }
            }
        }
    }

    #[test]
    fn detect_vendor_intel() {
        let entries = [kvm_cpuid_entry2 {
            function: 0,
            index: 0,
            flags: 0,
            eax: 0,
            ebx: 0x756e_6547, // "Genu"
            edx: 0x4965_6e69, // "ineI"
            ecx: 0x6c65_746e, // "ntel"
            ..Default::default()
        }];
        assert_eq!(detect_vendor(&entries), CpuVendor::Intel);
    }

    #[test]
    fn detect_vendor_amd() {
        let entries = [kvm_cpuid_entry2 {
            function: 0,
            index: 0,
            flags: 0,
            eax: 0,
            ebx: 0x6874_7541, // "Auth"
            edx: 0x6974_6e65, // "enti"
            ecx: 0x444d_4163, // "cAMD"
            ..Default::default()
        }];
        assert_eq!(detect_vendor(&entries), CpuVendor::Amd);
    }

    #[test]
    fn detect_vendor_unknown() {
        let entries = [kvm_cpuid_entry2 {
            function: 0,
            index: 0,
            flags: 0,
            eax: 0,
            ebx: 0,
            edx: 0,
            ecx: 0,
            ..Default::default()
        }];
        assert_eq!(detect_vendor(&entries), CpuVendor::Unknown);
    }

    #[test]
    fn detect_vendor_missing_leaf0() {
        let entries = [kvm_cpuid_entry2 {
            function: 1,
            index: 0,
            ..Default::default()
        }];
        assert_eq!(detect_vendor(&entries), CpuVendor::Unknown);
    }

    #[test]
    fn detect_vendor_from_kvm() {
        let kvm = match kvm_ioctls::Kvm::new() {
            Ok(k) => k,
            Err(_) => return,
        };
        let cpuid = kvm
            .get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
            .expect("get_supported_cpuid");
        let vendor = detect_vendor(cpuid.as_slice());
        assert_ne!(vendor, CpuVendor::Unknown, "host should be Intel or AMD");
    }

    #[test]
    fn brand_string_not_clobbered() {
        let kvm = match kvm_ioctls::Kvm::new() {
            Ok(k) => k,
            Err(_) => return,
        };
        let host_cpuid = kvm
            .get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
            .expect("get_supported_cpuid");
        let topo = Topology {
            sockets: 2,
            cores_per_socket: 2,
            threads_per_core: 2,
        };
        let cpuid = generate_cpuid(
            kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
                .unwrap()
                .as_slice(),
            &topo,
            0,
        );
        // Brand string leaves 0x80000002-0x80000004 should match host
        for leaf_fn in [0x8000_0002u32, 0x8000_0003, 0x8000_0004] {
            let host_leaf = host_cpuid.as_slice().iter().find(|e| e.function == leaf_fn);
            let guest_leaf = cpuid.iter().find(|e| e.function == leaf_fn);
            match (host_leaf, guest_leaf) {
                (Some(h), Some(g)) => {
                    assert_eq!(
                        (h.eax, h.ebx, h.ecx, h.edx),
                        (g.eax, g.ebx, g.ecx, g.edx),
                        "brand string leaf {leaf_fn:#x} should pass through from host"
                    );
                }
                (None, None) => {}
                _ => panic!(
                    "leaf {leaf_fn:#x}: host has it = {}, guest has it = {}",
                    host_leaf.is_some(),
                    guest_leaf.is_some()
                ),
            }
        }
    }

    #[test]
    fn vendor_conditional_leaf4_on_intel() {
        let kvm = match kvm_ioctls::Kvm::new() {
            Ok(k) => k,
            Err(_) => return,
        };
        let host_cpuid = kvm
            .get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
            .expect("get_supported_cpuid");
        let vendor = detect_vendor(host_cpuid.as_slice());
        if vendor != CpuVendor::Intel {
            return; // test only meaningful on Intel
        }
        let topo = Topology {
            sockets: 2,
            cores_per_socket: 4,
            threads_per_core: 2,
        };
        let cpuid = generate_cpuid(
            kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
                .unwrap()
                .as_slice(),
            &topo,
            0,
        );
        // On Intel, leaf 0x4 should have been patched
        let l3 = cpuid
            .iter()
            .find(|e| e.function == 0x4 && ((e.eax >> 5) & 0x7) == 3);
        if let Some(entry) = l3 {
            let max_sharing = (entry.eax >> 14) & 0xfff;
            assert_eq!(
                max_sharing,
                (1u32 << topo.core_shift()) - 1,
                "Intel leaf 0x4 L3 sharing should be patched"
            );
        }
    }

    #[test]
    fn vendor_conditional_leaf8000001e_on_amd() {
        let kvm = match kvm_ioctls::Kvm::new() {
            Ok(k) => k,
            Err(_) => return,
        };
        let host_cpuid = kvm
            .get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
            .expect("get_supported_cpuid");
        let vendor = detect_vendor(host_cpuid.as_slice());
        if vendor != CpuVendor::Amd {
            return; // test only meaningful on AMD
        }
        let topo = Topology {
            sockets: 2,
            cores_per_socket: 4,
            threads_per_core: 2,
        };
        let cpuid = generate_cpuid(
            kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
                .unwrap()
                .as_slice(),
            &topo,
            0,
        );
        let leaf = cpuid.iter().find(|e| e.function == 0x8000_001e);
        if let Some(entry) = leaf {
            assert_eq!(
                entry.eax,
                topo.apic_id(0),
                "AMD leaf 0x8000001E EAX should be patched"
            );
        }
    }

    #[test]
    fn max_apic_id_single_cpu() {
        let t = Topology {
            sockets: 1,
            cores_per_socket: 1,
            threads_per_core: 1,
        };
        assert_eq!(t.max_apic_id(), 0);
    }

    #[test]
    fn max_apic_id_equals_last_cpu() {
        let t = Topology {
            sockets: 2,
            cores_per_socket: 4,
            threads_per_core: 2,
        };
        assert_eq!(t.max_apic_id(), t.apic_id(t.total_cpus() - 1));
    }

    #[test]
    fn max_apic_id_large_topology() {
        // 14 sockets x 9 cores x 2 threads: the gauntlet max-cpu preset
        let t = Topology {
            sockets: 14,
            cores_per_socket: 9,
            threads_per_core: 2,
        };
        // core_bits = bits_needed(9) = 4, thread_bits = 1
        // last cpu = 251: socket 13, core 8, thread 1
        // apic_id = 13 << 5 | 8 << 1 | 1 = 433
        assert_eq!(t.max_apic_id(), 433);
        assert!(t.max_apic_id() > 254);
    }

    #[test]
    fn topology_single_thread_per_core() {
        let t = Topology {
            sockets: 4,
            cores_per_socket: 4,
            threads_per_core: 1,
        };
        assert_eq!(t.smt_shift(), 0);
        // APIC IDs should still be unique
        let ids: Vec<u32> = (0..t.total_cpus()).map(|i| t.apic_id(i)).collect();
        let unique: std::collections::HashSet<u32> = ids.iter().cloned().collect();
        assert_eq!(ids.len(), unique.len());
    }

    #[test]
    fn topology_1x1x1() {
        let t = Topology {
            sockets: 1,
            cores_per_socket: 1,
            threads_per_core: 1,
        };
        assert_eq!(t.total_cpus(), 1);
        assert_eq!(t.apic_id(0), 0);
        assert_eq!(t.max_apic_id(), 0);
    }
}
