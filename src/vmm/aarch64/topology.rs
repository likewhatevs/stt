use anyhow::{Context, Result};
use kvm_bindings::{
    KVM_REG_ARM64, KVM_REG_ARM64_SYSREG, KVM_REG_ARM64_SYSREG_CRM_MASK,
    KVM_REG_ARM64_SYSREG_CRM_SHIFT, KVM_REG_ARM64_SYSREG_CRN_MASK, KVM_REG_ARM64_SYSREG_CRN_SHIFT,
    KVM_REG_ARM64_SYSREG_OP0_MASK, KVM_REG_ARM64_SYSREG_OP0_SHIFT, KVM_REG_ARM64_SYSREG_OP1_MASK,
    KVM_REG_ARM64_SYSREG_OP1_SHIFT, KVM_REG_ARM64_SYSREG_OP2_MASK, KVM_REG_ARM64_SYSREG_OP2_SHIFT,
    KVM_REG_SIZE_U64,
};
use kvm_ioctls::VcpuFd;

use crate::vmm::topology::Topology;

/// MPIDR_EL1 register ID for KVM_GET_ONE_REG / KVM_SET_ONE_REG.
/// Encoded as system register (3, 0, 0, 0, 5) per the kernel's
/// arch/arm64/include/uapi/asm/kvm.h.
pub const MPIDR_EL1: u64 = KVM_REG_ARM64
    | KVM_REG_SIZE_U64
    | KVM_REG_ARM64_SYSREG as u64
    | ((3u64 << KVM_REG_ARM64_SYSREG_OP0_SHIFT) & KVM_REG_ARM64_SYSREG_OP0_MASK as u64)
    | ((0u64 << KVM_REG_ARM64_SYSREG_OP1_SHIFT) & KVM_REG_ARM64_SYSREG_OP1_MASK as u64)
    | ((0u64 << KVM_REG_ARM64_SYSREG_CRN_SHIFT) & KVM_REG_ARM64_SYSREG_CRN_MASK as u64)
    | ((0u64 << KVM_REG_ARM64_SYSREG_CRM_SHIFT) & KVM_REG_ARM64_SYSREG_CRM_MASK as u64)
    | ((5u64 << KVM_REG_ARM64_SYSREG_OP2_SHIFT) & KVM_REG_ARM64_SYSREG_OP2_MASK as u64);

/// Mask for the affinity fields used in FDT cpu node `reg` property.
/// Bits [23:0] of MPIDR: Aff0 [7:0], Aff1 [15:8], Aff2 [23:16].
pub const MPIDR_AFF_MASK: u64 = 0xFF_FFFF;

/// Read MPIDR_EL1 from a vCPU after vcpu_init.
pub fn read_mpidr(vcpu: &VcpuFd) -> Result<u64> {
    let mut buf = [0u8; 8];
    vcpu.get_one_reg(MPIDR_EL1, &mut buf)
        .context("read MPIDR_EL1")?;
    Ok(u64::from_le_bytes(buf))
}

/// Read MPIDRs for all vCPUs.
pub fn read_mpidrs(vcpus: &[VcpuFd]) -> Result<Vec<u64>> {
    vcpus.iter().map(read_mpidr).collect()
}

/// Extract the FDT `reg` value from an MPIDR: affinity fields only.
pub fn mpidr_to_fdt_reg(mpidr: u64) -> u64 {
    mpidr & MPIDR_AFF_MASK
}

/// Compute MPIDR affinity encoding from topology decomposition.
/// Aff0 = thread, Aff1 = core, Aff2 = socket.
/// This matches KVM's default MPIDR assignment for linearly-created vCPUs.
pub fn mpidr_from_topology(topo: &Topology, cpu_id: u32) -> u64 {
    let (socket, core, thread) = topo.decompose(cpu_id);
    let aff0 = thread as u64;
    let aff1 = core as u64;
    let aff2 = socket as u64;
    (1u64 << 31) | (aff2 << 16) | (aff1 << 8) | aff0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mpidr_el1_reg_id() {
        // MPIDR_EL1 = sys_reg(3, 0, 0, 0, 5):
        //   KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM64_SYSREG
        //   | (3 << 14) | (0 << 11) | (0 << 7) | (0 << 3) | 5
        assert_eq!(
            MPIDR_EL1, 0x6030_0000_0013_C005,
            "MPIDR_EL1 register ID encoding"
        );
    }

    #[test]
    fn mpidr_from_topology_single() {
        let t = Topology {
            sockets: 1,
            cores_per_socket: 1,
            threads_per_core: 1,
        };
        let mpidr = mpidr_from_topology(&t, 0);
        assert_eq!(mpidr & MPIDR_AFF_MASK, 0);
        assert_ne!(mpidr & (1 << 31), 0, "bit 31 must be set");
    }

    #[test]
    fn mpidr_from_topology_multi() {
        let t = Topology {
            sockets: 2,
            cores_per_socket: 4,
            threads_per_core: 2,
        };
        // cpu 0: socket 0, core 0, thread 0
        let m0 = mpidr_from_topology(&t, 0);
        assert_eq!(m0 & 0xFF, 0, "aff0 (thread) = 0");
        assert_eq!((m0 >> 8) & 0xFF, 0, "aff1 (core) = 0");
        assert_eq!((m0 >> 16) & 0xFF, 0, "aff2 (socket) = 0");

        // cpu 1: socket 0, core 0, thread 1
        let m1 = mpidr_from_topology(&t, 1);
        assert_eq!(m1 & 0xFF, 1, "aff0 (thread) = 1");
        assert_eq!((m1 >> 8) & 0xFF, 0, "aff1 (core) = 0");

        // cpu 2: socket 0, core 1, thread 0
        let m2 = mpidr_from_topology(&t, 2);
        assert_eq!(m2 & 0xFF, 0, "aff0 (thread) = 0");
        assert_eq!((m2 >> 8) & 0xFF, 1, "aff1 (core) = 1");

        // cpu 8: socket 1, core 0, thread 0
        let m8 = mpidr_from_topology(&t, 8);
        assert_eq!(m8 & 0xFF, 0, "aff0 (thread) = 0");
        assert_eq!((m8 >> 8) & 0xFF, 0, "aff1 (core) = 0");
        assert_eq!((m8 >> 16) & 0xFF, 1, "aff2 (socket) = 1");
    }

    #[test]
    fn mpidr_to_fdt_reg_masks() {
        let mpidr = (1u64 << 31) | (2 << 16) | (3 << 8) | 1;
        let reg = mpidr_to_fdt_reg(mpidr);
        assert_eq!(reg, (2 << 16) | (3 << 8) | 1);
        assert_eq!(reg & (1 << 31), 0, "bit 31 should be masked out");
    }

    #[test]
    fn mpidr_unique_all_gauntlet_presets() {
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
            (4, 8, 1),
            (8, 8, 1),
            (4, 32, 1),
            (8, 16, 1),
            (15, 16, 1),
            (14, 18, 1),
        ];
        for (sockets, cores, threads) in presets {
            let t = Topology {
                sockets,
                cores_per_socket: cores,
                threads_per_core: threads,
            };
            let mpidrs: Vec<u64> = (0..t.total_cpus())
                .map(|i| mpidr_from_topology(&t, i))
                .collect();
            let unique: std::collections::HashSet<u64> = mpidrs.iter().copied().collect();
            assert_eq!(
                mpidrs.len(),
                unique.len(),
                "topology {sockets}s/{cores}c/{threads}t: MPIDRs not unique"
            );
        }
    }

    #[test]
    fn mpidr_bit31_always_set() {
        let t = Topology {
            sockets: 2,
            cores_per_socket: 4,
            threads_per_core: 2,
        };
        for cpu in 0..t.total_cpus() {
            let mpidr = mpidr_from_topology(&t, cpu);
            assert_ne!(mpidr & (1 << 31), 0, "cpu {cpu}: MPIDR bit 31 must be set");
        }
    }

    #[test]
    fn mpidr_aff_mask_covers_three_levels() {
        assert_eq!(MPIDR_AFF_MASK, 0xFF_FFFF);
        assert_eq!(MPIDR_AFF_MASK & 0xFF, 0xFF, "Aff0 fully covered");
        assert_eq!((MPIDR_AFF_MASK >> 8) & 0xFF, 0xFF, "Aff1 fully covered");
        assert_eq!((MPIDR_AFF_MASK >> 16) & 0xFF, 0xFF, "Aff2 fully covered");
    }

    #[test]
    fn decompose_matches_mpidr_fields() {
        let t = Topology {
            sockets: 3,
            cores_per_socket: 5,
            threads_per_core: 2,
        };
        for cpu in 0..t.total_cpus() {
            let (socket, core, thread) = t.decompose(cpu);
            let mpidr = mpidr_from_topology(&t, cpu);
            assert_eq!(mpidr & 0xFF, thread as u64, "cpu {cpu}: aff0 = thread");
            assert_eq!((mpidr >> 8) & 0xFF, core as u64, "cpu {cpu}: aff1 = core");
            assert_eq!(
                (mpidr >> 16) & 0xFF,
                socket as u64,
                "cpu {cpu}: aff2 = socket"
            );
        }
    }
}
