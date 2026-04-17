use std::io::Seek;

use anyhow::{Context, Result};
use kvm_bindings::{KVM_REG_ARM_CORE, KVM_REG_ARM64, KVM_REG_SIZE_U64};
use kvm_ioctls::VcpuFd;
use vm_memory::{Address, GuestAddress, GuestMemoryMmap};

use crate::vmm::kvm::{CMDLINE_MAX, KERNEL_LOAD_ADDR};

/// Result of loading a kernel image.
pub struct KernelLoadResult {
    /// Entry point address (kernel_load from PE loader).
    pub entry: u64,
    /// End of the kernel image in guest physical memory.
    pub kernel_end: u64,
}

/// Gzip magic bytes.
const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];

/// Load an aarch64 kernel into guest memory.
///
/// Accepts both raw PE Image files and gzip-compressed vmlinuz files.
/// Compressed kernels (identified by gzip magic `1f 8b`) are decompressed
/// in memory before loading via the PE loader.
pub fn load_kernel(
    guest_mem: &GuestMemoryMmap,
    kernel_path: &std::path::Path,
) -> Result<KernelLoadResult> {
    use linux_loader::loader::{KernelLoader, pe::PE};
    use std::fs::File;
    use std::io::Read;

    let mut kernel_file = File::open(kernel_path)
        .with_context(|| format!("open kernel: {}", kernel_path.display()))?;

    // Read the first 2 bytes to detect gzip compression.
    let mut magic = [0u8; 2];
    kernel_file
        .read_exact(&mut magic)
        .context("read kernel magic")?;
    kernel_file
        .seek(std::io::SeekFrom::Start(0))
        .context("seek kernel to start")?;

    if magic == GZIP_MAGIC {
        // Decompress gzip kernel into a Cursor for the PE loader.
        let mut decoder = flate2::read::GzDecoder::new(kernel_file);
        let mut decompressed = Vec::new();
        decoder
            .read_to_end(&mut decompressed)
            .context("decompress gzip kernel")?;
        let mut cursor = std::io::Cursor::new(decompressed);
        let result = PE::load(
            guest_mem,
            Some(GuestAddress(KERNEL_LOAD_ADDR)),
            &mut cursor,
            None,
        )
        .context("load decompressed aarch64 Image")?;
        Ok(KernelLoadResult {
            entry: result.kernel_load.raw_value(),
            kernel_end: result.kernel_end,
        })
    } else {
        let result = PE::load(
            guest_mem,
            Some(GuestAddress(KERNEL_LOAD_ADDR)),
            &mut kernel_file,
            None,
        )
        .context("load aarch64 Image")?;
        Ok(KernelLoadResult {
            entry: result.kernel_load.raw_value(),
            kernel_end: result.kernel_end,
        })
    }
}

/// Validate that a kernel command line fits within the maximum length.
///
/// On aarch64 the kernel reads the command line from the FDT /chosen
/// node's bootargs property. No separate memory write is needed.
pub fn validate_cmdline(cmdline: &str) -> Result<()> {
    anyhow::ensure!(
        cmdline.len() < CMDLINE_MAX,
        "cmdline too long ({} > {})",
        cmdline.len(),
        CMDLINE_MAX
    );
    Ok(())
}

/// KVM register IDs for aarch64 core registers.
///
/// The encoding follows the KVM_REG_ARM_CORE format:
///   KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM_CORE | (offset / 4)
///
/// The offset is into struct kvm_regs (user_pt_regs) defined in
/// arch/arm64/include/uapi/asm/kvm.h.
const REG_CORE_BASE: u64 = KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM_CORE as u64;

/// Register ID for x0 (regs.regs[0]).
const REG_X0: u64 = REG_CORE_BASE;

/// Register ID for PC (regs.pc, offset = 256/4 = 64 in u32 units).
/// In user_pt_regs: regs[0..31] at offsets 0..248, sp at 248, pc at
/// 256, pstate at 264. 256 bytes / 4 = 64 u32-offset.
const REG_PC: u64 = REG_CORE_BASE | (256 / 4);

/// Register ID for pstate (regs.pstate, offset = 264/4 = 66).
const REG_PSTATE: u64 = REG_CORE_BASE | (264 / 4);

/// PSR mode bits for EL1h (EL1, SP_EL1).
const PSTATE_MODE_EL1H: u64 = 0x5;

/// PSR D/A/I/F mask bits — mask debug, SError, IRQ, FIQ exceptions.
const PSTATE_DAIF_MASK: u64 = 0x3C0;

/// Set up vCPU registers for the BSP.
///
/// Per the arm64 boot protocol (Documentation/arm64/booting.rst):
/// - x0 = physical address of the FDT
/// - PC = kernel entry point
/// - pstate = EL1h with DAIF masked
/// - All other registers are undefined (kernel does not depend on them)
pub fn setup_regs(vcpu: &VcpuFd, entry: u64, fdt_addr: u64) -> Result<()> {
    // Set PC to kernel entry point.
    vcpu.set_one_reg(REG_PC, &entry.to_le_bytes())
        .context("set PC")?;

    // Set x0 to FDT address.
    vcpu.set_one_reg(REG_X0, &fdt_addr.to_le_bytes())
        .context("set x0 (FDT address)")?;

    // Set pstate to EL1h with all exceptions masked.
    let pstate: u64 = PSTATE_MODE_EL1H | PSTATE_DAIF_MASK;
    vcpu.set_one_reg(REG_PSTATE, &pstate.to_le_bytes())
        .context("set pstate")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_load_result_fields() {
        let r = KernelLoadResult {
            entry: 0x28_0000,
            kernel_end: 0x40_0000,
        };
        assert_eq!(r.entry, 0x28_0000);
        assert_eq!(r.kernel_end, 0x40_0000);
    }

    #[test]
    fn write_cmdline_basic() {
        validate_cmdline("console=ttyAMA0").unwrap();
    }

    #[test]
    fn write_cmdline_too_long() {
        let long = "x".repeat(CMDLINE_MAX + 1);
        assert!(validate_cmdline(&long).is_err());
    }

    #[test]
    fn reg_ids_follow_encoding() {
        // Verify register ID encoding matches the KVM ABI.
        // x0 is at offset 0 in user_pt_regs.
        assert_eq!(REG_X0 & 0xFFFF, 0);
        // PC is at byte offset 256 -> u32 offset 64.
        assert_eq!(REG_PC & 0xFFFF, 64);
        // pstate is at byte offset 264 -> u32 offset 66.
        assert_eq!(REG_PSTATE & 0xFFFF, 66);
    }

    #[test]
    fn pstate_el1h_value() {
        let pstate = PSTATE_MODE_EL1H | PSTATE_DAIF_MASK;
        // EL1h = 0x5, DAIF = 0x3C0 -> 0x3C5
        assert_eq!(pstate, 0x3C5);
    }

    #[test]
    fn setup_regs_on_real_vcpu() {
        use crate::vmm::topology::Topology;
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let vm = crate::vmm::kvm::KtstrKvm::new(topo, 64, false).unwrap();
        let result = setup_regs(&vm.vcpus[0], 0x28_0000, 0x4000_0000);
        assert!(result.is_ok(), "setup_regs failed: {:?}", result.err());

        // Verify registers were set.
        let mut pc_buf = [0u8; 8];
        vm.vcpus[0].get_one_reg(REG_PC, &mut pc_buf).unwrap();
        assert_eq!(u64::from_le_bytes(pc_buf), 0x28_0000);

        let mut x0_buf = [0u8; 8];
        vm.vcpus[0].get_one_reg(REG_X0, &mut x0_buf).unwrap();
        assert_eq!(u64::from_le_bytes(x0_buf), 0x4000_0000);

        let mut pstate_buf = [0u8; 8];
        vm.vcpus[0]
            .get_one_reg(REG_PSTATE, &mut pstate_buf)
            .unwrap();
        assert_eq!(u64::from_le_bytes(pstate_buf), 0x3C5);
    }
}
