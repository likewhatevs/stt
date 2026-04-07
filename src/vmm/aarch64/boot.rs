use anyhow::Result;
use kvm_ioctls::VcpuFd;
use vm_memory::GuestMemoryMmap;

/// Result of loading a kernel image.
pub struct KernelLoadResult {
    pub entry: u64,
}

/// Load a kernel Image into guest memory.
pub fn load_kernel(
    _guest_mem: &GuestMemoryMmap,
    _kernel_path: &std::path::Path,
) -> Result<KernelLoadResult> {
    todo!("aarch64 kernel loading")
}

/// Write the kernel command line into guest memory.
pub fn write_cmdline(_guest_mem: &GuestMemoryMmap, _cmdline: &str) -> Result<()> {
    todo!("aarch64 cmdline setup")
}

/// Set up vCPU registers for the BSP.
pub fn setup_regs(_vcpu: &VcpuFd, _entry: u64) -> Result<()> {
    todo!("aarch64 register setup")
}
