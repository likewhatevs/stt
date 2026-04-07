use anyhow::Result;
use kvm_ioctls::{VcpuFd, VmFd};
use vm_memory::GuestMemoryMmap;

use crate::vmm::topology::Topology;

/// A KVM virtual machine with configured topology (aarch64).
#[allow(dead_code)]
pub struct SttKvm {
    pub vm_fd: VmFd,
    pub vcpus: Vec<VcpuFd>,
    pub guest_mem: GuestMemoryMmap,
    pub topology: Topology,
    pub has_immediate_exit: bool,
}

impl SttKvm {
    pub fn new(_topo: Topology, _memory_mb: u32) -> Result<Self> {
        todo!("aarch64 KVM VM creation")
    }

    pub fn new_with_hugepages(_topo: Topology, _memory_mb: u32) -> Result<Self> {
        todo!("aarch64 KVM VM creation with hugepages")
    }
}
