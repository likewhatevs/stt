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

/// Maximum cache level supported by arm64 CLIDR_EL1 (7 Ctype fields).
const MAX_CACHE_LEVEL: u32 = 7;

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

/// CLIDR_EL1 register ID for KVM_GET_ONE_REG / KVM_SET_ONE_REG.
/// Encoded as system register (3, 1, 0, 0, 1).
const CLIDR_EL1: u64 = KVM_REG_ARM64
    | KVM_REG_SIZE_U64
    | KVM_REG_ARM64_SYSREG as u64
    | ((3u64 << KVM_REG_ARM64_SYSREG_OP0_SHIFT) & KVM_REG_ARM64_SYSREG_OP0_MASK as u64)
    | ((1u64 << KVM_REG_ARM64_SYSREG_OP1_SHIFT) & KVM_REG_ARM64_SYSREG_OP1_MASK as u64)
    | ((0u64 << KVM_REG_ARM64_SYSREG_CRN_SHIFT) & KVM_REG_ARM64_SYSREG_CRN_MASK as u64)
    | ((0u64 << KVM_REG_ARM64_SYSREG_CRM_SHIFT) & KVM_REG_ARM64_SYSREG_CRM_MASK as u64)
    | ((1u64 << KVM_REG_ARM64_SYSREG_OP2_SHIFT) & KVM_REG_ARM64_SYSREG_OP2_MASK as u64);

// CLIDR_EL1 Ctype field values.
const CLIDR_CTYPE_NO_CACHE: u64 = 0;
const CLIDR_CTYPE_INSTRUCTION: u64 = 1;
const CLIDR_CTYPE_DATA: u64 = 2;
const CLIDR_CTYPE_SEPARATE: u64 = 3;
const CLIDR_CTYPE_UNIFIED: u64 = 4;

// CLIDR_EL1 field positions.
const CLIDR_CTYPE_BITS: u32 = 3;
const CLIDR_LOC_SHIFT: u32 = 24;

/// Return true if the host's L1 cache is unified (from sysfs).
///
/// When the host L1 is unified, the CLIDR Ctype1 field is Unified (1
/// leaf). The DT's `of_count_cache_leaves` defaults to 2 for CPU nodes
/// without cache properties, so the CPU node needs `cache-unified` to
/// reduce the OF count to 1.
pub fn host_l1_is_unified() -> bool {
    let cache_dir = "/sys/devices/system/cpu/cpu0/cache";
    let entries = match std::fs::read_dir(cache_dir) {
        Ok(e) => e,
        Err(_) => return false,
    };
    let mut has_data = false;
    let mut has_instruction = false;
    let mut has_unified = false;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("index") {
            continue;
        }
        let level_path = entry.path().join("level");
        let type_path = entry.path().join("type");
        let Ok(level_str) = std::fs::read_to_string(&level_path) else {
            continue;
        };
        let Ok(level) = level_str.trim().parse::<u32>() else {
            continue;
        };
        if level != 1 {
            continue;
        }
        let Ok(type_str) = std::fs::read_to_string(&type_path) else {
            continue;
        };
        match type_str.trim() {
            "Data" => has_data = true,
            "Instruction" => has_instruction = true,
            "Unified" => has_unified = true,
            _ => {}
        }
    }
    has_unified && !has_data && !has_instruction
}

/// Build a CLIDR_EL1 value from the host's sysfs cache topology.
///
/// Reads /sys/devices/system/cpu/cpu0/cache/index*/level and type to
/// determine Ctype fields for each cache level. Sets LoC to the
/// highest level found.
fn build_clidr_from_sysfs() -> u64 {
    let cache_dir = "/sys/devices/system/cpu/cpu0/cache";
    let entries = match std::fs::read_dir(cache_dir) {
        Ok(e) => e,
        Err(_) => return 0,
    };

    // Collect cache types per level.
    let mut level_types: [u8; MAX_CACHE_LEVEL as usize + 1] = [0; MAX_CACHE_LEVEL as usize + 1];
    // Bit flags: 1=Data, 2=Instruction, 4=Unified
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("index") {
            continue;
        }
        let level_path = entry.path().join("level");
        let type_path = entry.path().join("type");
        let Ok(level_str) = std::fs::read_to_string(&level_path) else {
            continue;
        };
        let Ok(level) = level_str.trim().parse::<u32>() else {
            continue;
        };
        if level == 0 || level > MAX_CACHE_LEVEL {
            continue;
        }
        let Ok(type_str) = std::fs::read_to_string(&type_path) else {
            continue;
        };
        match type_str.trim() {
            "Data" => level_types[level as usize] |= 1,
            "Instruction" => level_types[level as usize] |= 2,
            "Unified" => level_types[level as usize] |= 4,
            _ => {}
        }
    }

    let mut clidr: u64 = 0;
    let mut max_level: u32 = 0;
    for level in 1..=MAX_CACHE_LEVEL {
        let flags = level_types[level as usize];
        if flags == 0 {
            break;
        }
        let ctype = if flags & 4 != 0 {
            CLIDR_CTYPE_UNIFIED
        } else if flags & 1 != 0 && flags & 2 != 0 {
            CLIDR_CTYPE_SEPARATE
        } else if flags & 2 != 0 {
            CLIDR_CTYPE_INSTRUCTION
        } else if flags & 1 != 0 {
            CLIDR_CTYPE_DATA
        } else {
            CLIDR_CTYPE_NO_CACHE
        };
        let shift = CLIDR_CTYPE_BITS * (level - 1);
        clidr |= ctype << shift;
        max_level = level;
    }

    // Set LoC (Level of Coherence) to the highest cache level.
    clidr |= (max_level as u64) << CLIDR_LOC_SHIFT;

    clidr
}

/// Merge sysfs-derived Ctype and LoC fields into an existing CLIDR_EL1
/// value, preserving LoUU, LoUIS, ICB, and Ttype fields from the
/// original.
fn merge_clidr(current: u64, sysfs: u64) -> u64 {
    // Ctype fields: bits [20:0] (7 levels x 3 bits = 21 bits).
    // LoC field: bits [26:24].
    const CTYPE_MASK: u64 = 0x001F_FFFF;
    const LOC_MASK: u64 = 0x0700_0000;
    const REPLACE_MASK: u64 = CTYPE_MASK | LOC_MASK;
    (current & !REPLACE_MASK) | (sysfs & REPLACE_MASK)
}

/// Override CLIDR_EL1 on each vCPU to match the host's real cache
/// topology from sysfs.
///
/// KVM's `reset_clidr` (since host kernel 6.3) fabricates CLIDR_EL1
/// from CTR_EL0 flags, which can report fewer cache levels than the
/// host actually has. The DT is built from sysfs and may describe
/// more levels. When CLIDR and DT disagree on cache leaf counts,
/// `cache_setup_of_node` fails and the guest sees no cache topology.
///
/// This reads the current (possibly fabricated) CLIDR from vCPU 0,
/// replaces only the Ctype and LoC fields with values from sysfs,
/// preserves LoUU/LoUIS/ICB/Ttype, and writes back to all vCPUs.
/// On pre-6.3 kernels where CLIDR already passes through the real
/// value, the write is effectively a no-op.
pub fn override_clidr(vcpus: &[VcpuFd]) -> Result<()> {
    let sysfs_clidr = build_clidr_from_sysfs();
    if sysfs_clidr == 0 {
        tracing::warn!("no cache info from sysfs, skipping CLIDR override");
        return Ok(());
    }

    let mut cur_clidr_bytes = [0u8; 8];
    if let Err(e) = vcpus[0].get_one_reg(CLIDR_EL1, &mut cur_clidr_bytes) {
        tracing::warn!("failed to read CLIDR_EL1, skipping override: {e}");
        return Ok(());
    }
    let cur_clidr = u64::from_le_bytes(cur_clidr_bytes);
    let new_clidr = merge_clidr(cur_clidr, sysfs_clidr);

    if new_clidr != cur_clidr {
        let new_bytes = new_clidr.to_le_bytes();
        for (i, vcpu) in vcpus.iter().enumerate() {
            vcpu.set_one_reg(CLIDR_EL1, &new_bytes)
                .with_context(|| format!("set CLIDR_EL1 on vCPU {i}"))?;
        }
        tracing::debug!(
            cur = format_args!("{cur_clidr:#x}"),
            new = format_args!("{new_clidr:#x}"),
            "CLIDR_EL1 override applied",
        );
    }

    Ok(())
}

/// Return the highest cache level from the host's sysfs. Used to
/// determine the DT cache chain depth for multi-LLC topologies.
pub fn host_cache_levels() -> u32 {
    let mut max_level: u32 = 0;
    let cache_dir = "/sys/devices/system/cpu/cpu0/cache";
    let entries = match std::fs::read_dir(cache_dir) {
        Ok(e) => e,
        Err(_) => return 0,
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("index") {
            continue;
        }
        let level_path = entry.path().join("level");
        if let Ok(s) = std::fs::read_to_string(&level_path)
            && let Ok(level) = s.trim().parse::<u32>()
            && level > max_level
        {
            max_level = level;
        }
    }
    max_level
}

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
/// Aff0 = thread, Aff1 = core, Aff2 = LLC.
/// This matches KVM's default MPIDR assignment for linearly-created vCPUs.
pub fn mpidr_from_topology(topo: &Topology, cpu_id: u32) -> u64 {
    let (llc, core, thread) = topo.decompose(cpu_id);
    let aff0 = thread as u64;
    let aff1 = core as u64;
    let aff2 = llc as u64;
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
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let mpidr = mpidr_from_topology(&t, 0);
        assert_eq!(mpidr & MPIDR_AFF_MASK, 0);
        assert_ne!(mpidr & (1 << 31), 0, "bit 31 must be set");
    }

    #[test]
    fn mpidr_from_topology_multi() {
        let t = Topology {
            llcs: 2,
            cores_per_llc: 4,
            threads_per_core: 2,
            numa_nodes: 1,
        };
        // cpu 0: LLC 0, core 0, thread 0
        let m0 = mpidr_from_topology(&t, 0);
        assert_eq!(m0 & 0xFF, 0, "aff0 (thread) = 0");
        assert_eq!((m0 >> 8) & 0xFF, 0, "aff1 (core) = 0");
        assert_eq!((m0 >> 16) & 0xFF, 0, "aff2 (LLC) = 0");

        // cpu 1: LLC 0, core 0, thread 1
        let m1 = mpidr_from_topology(&t, 1);
        assert_eq!(m1 & 0xFF, 1, "aff0 (thread) = 1");
        assert_eq!((m1 >> 8) & 0xFF, 0, "aff1 (core) = 0");

        // cpu 2: LLC 0, core 1, thread 0
        let m2 = mpidr_from_topology(&t, 2);
        assert_eq!(m2 & 0xFF, 0, "aff0 (thread) = 0");
        assert_eq!((m2 >> 8) & 0xFF, 1, "aff1 (core) = 1");

        // cpu 8: LLC 1, core 0, thread 0
        let m8 = mpidr_from_topology(&t, 8);
        assert_eq!(m8 & 0xFF, 0, "aff0 (thread) = 0");
        assert_eq!((m8 >> 8) & 0xFF, 0, "aff1 (core) = 0");
        assert_eq!((m8 >> 16) & 0xFF, 1, "aff2 (LLC) = 1");
    }

    #[test]
    fn mpidr_to_fdt_reg_masks() {
        let mpidr = (1u64 << 31) | (2 << 16) | (3 << 8) | 1;
        let reg = mpidr_to_fdt_reg(mpidr);
        assert_eq!(reg, (2 << 16) | (3 << 8) | 1);
        assert_eq!(reg & (1 << 31), 0, "bit 31 should be masked out");
    }

    #[test]
    fn mpidr_unique_representative_topologies() {
        let topos = [
            (1, 1, 1),   // degenerate single CPU
            (2, 1, 1),   // minimal multi-LLC
            (3, 3, 1),   // odd non-power-of-2
            (1, 1, 2),   // minimal SMT
            (2, 4, 2),   // standard multi-LLC with SMT
            (7, 5, 3),   // all dimensions non-power-of-2
            (15, 16, 1), // large scale no SMT
            (14, 9, 2),  // large with SMT
            (255, 1, 1), // max LLCs before Aff2 overflow
            (1, 255, 1), // max cores before Aff1 overflow
            (4, 32, 1),  // many cores, multi-LLC
        ];
        for (llcs, cores, threads) in topos {
            let t = Topology {
                llcs,
                cores_per_llc: cores,
                threads_per_core: threads,
                numa_nodes: 1,
            };
            let mpidrs: Vec<u64> = (0..t.total_cpus())
                .map(|i| mpidr_from_topology(&t, i))
                .collect();
            let unique: std::collections::HashSet<u64> = mpidrs.iter().copied().collect();
            assert_eq!(
                mpidrs.len(),
                unique.len(),
                "topology {llcs}l/{cores}c/{threads}t: MPIDRs not unique"
            );
        }
    }

    #[test]
    fn mpidr_bit31_always_set() {
        let t = Topology {
            llcs: 2,
            cores_per_llc: 4,
            threads_per_core: 2,
            numa_nodes: 1,
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
            llcs: 3,
            cores_per_llc: 5,
            threads_per_core: 2,
            numa_nodes: 1,
        };
        for cpu in 0..t.total_cpus() {
            let (llc, core, thread) = t.decompose(cpu);
            let mpidr = mpidr_from_topology(&t, cpu);
            assert_eq!(mpidr & 0xFF, thread as u64, "cpu {cpu}: aff0 = thread");
            assert_eq!((mpidr >> 8) & 0xFF, core as u64, "cpu {cpu}: aff1 = core");
            assert_eq!((mpidr >> 16) & 0xFF, llc as u64, "cpu {cpu}: aff2 = LLC");
        }
    }

    #[test]
    fn host_cache_levels_reads_sysfs() {
        let level = host_cache_levels();
        assert!(
            level >= 1,
            "host_cache_levels should detect at least 1 cache level, got {level}"
        );
    }

    #[test]
    fn clidr_el1_reg_id() {
        // CLIDR_EL1 = sys_reg(3, 1, 0, 0, 1):
        //   KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM64_SYSREG
        //   | (3 << 14) | (1 << 11) | (0 << 7) | (0 << 3) | 1
        assert_eq!(
            CLIDR_EL1, 0x6030_0000_0013_C801,
            "CLIDR_EL1 register ID encoding"
        );
    }

    #[test]
    fn build_clidr_from_sysfs_nonzero() {
        let clidr = build_clidr_from_sysfs();
        assert_ne!(clidr, 0, "sysfs should produce a non-zero CLIDR");
        // L1 must be present.
        let ctype1 = clidr & 0x7;
        assert_ne!(ctype1, 0, "L1 Ctype must be non-zero");
        // LoC must be at least 1.
        let loc = (clidr >> CLIDR_LOC_SHIFT) & 0x7;
        assert!(loc >= 1, "LoC must be >= 1, got {loc}");
    }

    #[test]
    fn merge_clidr_replaces_ctype_and_loc() {
        // current: LoUU=2 [29:27], LoUIS=1 [23:21], Ctype1=Unified(4), LoC=1
        let current: u64 = (2 << 27) | (1 << 21) | (1 << 24) | 4;
        // sysfs: Ctype1=Separate(3), Ctype2=Unified(4), LoC=2
        let sysfs: u64 = (2 << 24) | (4 << 3) | 3;
        let merged = merge_clidr(current, sysfs);

        // Ctype and LoC from sysfs.
        assert_eq!(merged & 0x001F_FFFF, sysfs & 0x001F_FFFF);
        assert_eq!((merged >> 24) & 0x7, 2, "LoC from sysfs");
        // LoUIS and LoUU preserved from current.
        assert_eq!((merged >> 21) & 0x7, 1, "LoUIS preserved");
        assert_eq!((merged >> 27) & 0x7, 2, "LoUU preserved");
    }

    #[test]
    fn merge_clidr_identity_when_equal() {
        let val = 0x0000_0000_0200_0023_u64;
        assert_eq!(merge_clidr(val, val), val);
    }
}
