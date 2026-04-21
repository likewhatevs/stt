use anyhow::{Context, Result};
use vm_memory::mmap::{GuestRegionMmap, MmapRegion};
use vm_memory::{GuestAddress, GuestMemory, GuestMemoryMmap};

use super::topology::Topology;

/// Owns a VA reservation created via `mmap(PROT_NONE)`. Drop calls
/// `munmap` on the entire reservation, releasing all MAP_FIXED
/// sub-mappings within it.
pub(crate) struct ReservationGuard {
    addr: *mut libc::c_void,
    size: usize,
}

unsafe impl Send for ReservationGuard {}
unsafe impl Sync for ReservationGuard {}

impl Drop for ReservationGuard {
    fn drop(&mut self) {
        if !self.addr.is_null() && self.addr != libc::MAP_FAILED {
            unsafe {
                libc::munmap(self.addr, self.size);
            }
        }
    }
}

/// Result of `NumaMemoryLayout::allocate_and_register`.
pub(crate) struct AllocatedMemory {
    pub guest_mem: GuestMemoryMmap,
    pub reservation: ReservationGuard,
}

/// Per-NUMA-node guest physical address range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeRegion {
    /// NUMA node index (0-based), matching the index into
    /// `Topology::nodes`. Nodes with `memory_mb == 0` are omitted.
    pub node_id: u32,
    /// Guest physical address where this node's memory starts.
    pub gpa_start: u64,
    /// Size in bytes.
    pub size: u64,
    /// KVM memory slot index for this region.
    pub slot: u32,
}

/// Per-node GPA layout with per-node MAP_FIXED mmaps within a
/// contiguous VA reservation.
///
/// A PROT_NONE VA reservation covers the total memory range. Each
/// node's sub-range is replaced via MAP_FIXED with a real
/// PROT_READ|PROT_WRITE mapping, individually mbind'd and
/// registered as a separate KVM memory slot. The `ReservationGuard`
/// owns the VA range and munmaps it on drop.
///
/// Contiguity is maintained by the VA reservation: all node regions
/// occupy adjacent sub-ranges of the same contiguous VA.
#[derive(Debug, Clone)]
pub struct NumaMemoryLayout {
    /// Per-node regions sorted by ascending GPA. Regions are
    /// contiguous: `regions[i+1].gpa_start == regions[i].gpa_start +
    /// regions[i].size`.
    regions: Vec<NodeRegion>,
}

impl NumaMemoryLayout {
    /// Compute per-node GPA ranges from a topology and total memory.
    ///
    /// `dram_base`: GPA where guest RAM starts (0 on x86_64,
    /// `DRAM_START` on aarch64).
    ///
    /// `total_memory_mb`: total guest memory in MiB. For `with_nodes`
    /// topologies, must equal the sum of all `NumaNode::memory_mb`.
    /// For uniform topologies, memory is divided evenly across
    /// `numa_nodes` nodes.
    pub fn compute(topo: &Topology, total_memory_mb: u32, dram_base: u64) -> Result<Self> {
        let total_bytes = (total_memory_mb as u64) << 20;
        let numa_nodes = topo.numa_nodes;

        match topo.nodes {
            Some(nodes) => {
                let node_total_mb: u32 = nodes.iter().map(|n| n.memory_mb).sum();
                anyhow::ensure!(
                    total_memory_mb == node_total_mb,
                    "total_memory_mb ({total_memory_mb}) must equal \
                     sum of node memory_mb ({node_total_mb})"
                );

                let mut regions = Vec::with_capacity(numa_nodes as usize);
                let mut gpa = dram_base;

                for (i, node) in nodes.iter().enumerate() {
                    let size = (node.memory_mb as u64) << 20;
                    if size == 0 {
                        continue;
                    }
                    regions.push(NodeRegion {
                        node_id: i as u32,
                        gpa_start: gpa,
                        size,
                        slot: regions.len() as u32,
                    });
                    gpa += size;
                }

                anyhow::ensure!(
                    !regions.is_empty(),
                    "at least one node must have non-zero memory"
                );

                Ok(Self { regions })
            }
            None => {
                if numa_nodes <= 1 {
                    let region = NodeRegion {
                        node_id: 0,
                        gpa_start: dram_base,
                        size: total_bytes,
                        slot: 0,
                    };
                    return Ok(Self {
                        regions: vec![region],
                    });
                }

                let per_node_mb = total_memory_mb / numa_nodes;
                let mut regions = Vec::with_capacity(numa_nodes as usize);
                let mut gpa = dram_base;
                for i in 0..numa_nodes {
                    let mb = if i == numa_nodes - 1 {
                        total_memory_mb - per_node_mb * (numa_nodes - 1)
                    } else {
                        per_node_mb
                    };
                    let size = (mb as u64) << 20;
                    regions.push(NodeRegion {
                        node_id: i,
                        gpa_start: gpa,
                        size,
                        slot: i,
                    });
                    gpa += size;
                }

                Ok(Self { regions })
            }
        }
    }

    /// Per-node regions sorted by ascending GPA.
    pub fn regions(&self) -> &[NodeRegion] {
        &self.regions
    }

    /// Total guest memory in bytes (sum of all node regions).
    pub fn total_bytes(&self) -> u64 {
        self.regions.iter().map(|r| r.size).sum()
    }

    /// GPA where guest DRAM starts (first region's start address).
    pub fn dram_base(&self) -> u64 {
        self.regions[0].gpa_start
    }

    /// GPA immediately after the last node's memory.
    #[allow(dead_code)]
    pub fn end_gpa(&self) -> u64 {
        let last = self.regions.last().unwrap();
        last.gpa_start + last.size
    }

    /// Whether this layout has exactly one region.
    #[allow(dead_code)]
    pub fn is_single_region(&self) -> bool {
        self.regions.len() == 1
    }

    /// Next available KVM slot index (after all node regions).
    #[allow(dead_code)]
    pub fn next_slot(&self) -> u32 {
        self.regions.last().map_or(0, |r| r.slot + 1)
    }

    /// Reserve contiguous VA, per-node MAP_FIXED mmap, register per-node
    /// KVM memory slots, and return the multi-region `GuestMemoryMmap`
    /// with a `ReservationGuard` that owns the VA range.
    ///
    /// Each node gets its own MAP_FIXED mmap within the reserved VA.
    /// The `MmapRegion` wrappers have `owned=false` (via `build_raw`),
    /// so their Drop is a no-op. The `ReservationGuard` munmaps the
    /// entire reservation on drop, releasing all sub-mappings.
    pub fn allocate_and_register(
        &self,
        vm_fd: &kvm_ioctls::VmFd,
        use_hugepages: bool,
        performance_mode: bool,
    ) -> Result<AllocatedMemory> {
        let total = self.total_bytes() as usize;
        let memory_mb = (total >> 20) as u32;

        let use_hugepages = use_hugepages
            || (performance_mode
                && super::host_topology::hugepages_free()
                    >= super::host_topology::hugepages_needed(memory_mb));

        // Step 1: Reserve contiguous VA with PROT_NONE.
        let reservation = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                total,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        anyhow::ensure!(
            reservation != libc::MAP_FAILED,
            "mmap VA reservation failed: {}",
            std::io::Error::last_os_error()
        );

        let guard = ReservationGuard {
            addr: reservation,
            size: total,
        };

        let mut guest_regions: Vec<GuestRegionMmap> = Vec::with_capacity(self.regions.len());

        for region in &self.regions {
            let offset = (region.gpa_start - self.dram_base()) as usize;
            let node_size = region.size as usize;
            let node_addr = unsafe { (reservation as *mut u8).add(offset) as *mut libc::c_void };

            // Step 2: Per-node MAP_FIXED mmap.
            let mut flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED;
            if use_hugepages {
                flags |= libc::MAP_HUGETLB | libc::MAP_HUGE_2MB;
            }

            let node_ptr = unsafe {
                libc::mmap(
                    node_addr,
                    node_size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    flags,
                    -1,
                    0,
                )
            };
            anyhow::ensure!(
                node_ptr != libc::MAP_FAILED,
                "MAP_FIXED mmap for node {} failed: {}",
                region.node_id,
                std::io::Error::last_os_error()
            );

            // Step 5: Wrap as vm-memory types. build_raw sets owned=false.
            let mmap_region = unsafe {
                MmapRegion::build_raw(
                    node_ptr as *mut u8,
                    node_size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                )
                .with_context(|| format!("build MmapRegion for node {}", region.node_id))?
            };
            let guest_region = GuestRegionMmap::new(mmap_region, GuestAddress(region.gpa_start))
                .ok_or_else(|| {
                    anyhow::anyhow!("GuestRegionMmap overflow for node {}", region.node_id)
                })?;
            guest_regions.push(guest_region);

            // Step 7: Register KVM memory slot.
            let mem_region = kvm_bindings::kvm_userspace_memory_region {
                slot: region.slot,
                guest_phys_addr: region.gpa_start,
                memory_size: region.size,
                userspace_addr: node_ptr as u64,
                flags: 0,
            };
            unsafe {
                vm_fd.set_user_memory_region(mem_region).with_context(|| {
                    format!(
                        "set KVM memory slot {} for node {}",
                        region.slot, region.node_id
                    )
                })?;
            }
        }

        // Step 6: Build multi-region GuestMemoryMmap.
        let guest_mem = GuestMemoryMmap::from_regions(guest_regions)
            .context("create multi-region GuestMemoryMmap")?;

        Ok(AllocatedMemory {
            guest_mem,
            reservation: guard,
        })
    }

    /// Bind each node's region to the corresponding host NUMA node(s),
    /// then pre-fault pages.
    ///
    /// `host_nodes` is indexed by guest node_id. Entries beyond the
    /// slice length or empty entries are skipped (e.g. CXL nodes on
    /// non-NUMA hosts).
    ///
    /// Ordering: mbind before MADV_POPULATE_WRITE ensures pages are
    /// allocated on the target node rather than the faulting CPU's node.
    pub fn mbind_regions(&self, guest_mem: &GuestMemoryMmap, host_nodes: &[Vec<usize>]) {
        for region in &self.regions {
            let idx = region.node_id as usize;
            if idx >= host_nodes.len() {
                continue;
            }
            let nodes = &host_nodes[idx];
            if nodes.is_empty() {
                continue;
            }
            let ptr = match guest_mem.get_host_address(GuestAddress(region.gpa_start)) {
                Ok(addr) => addr,
                Err(_) => continue,
            };

            // Step 3: Per-node mbind (before any page faults).
            super::host_topology::mbind_to_nodes(ptr, region.size as usize, nodes);

            // Step 4: Pre-fault after mbind.
            let ret = unsafe {
                libc::madvise(
                    ptr as *mut libc::c_void,
                    region.size as usize,
                    libc::MADV_POPULATE_WRITE,
                )
            };
            if ret != 0 {
                eprintln!(
                    "performance_mode: WARNING: MADV_POPULATE_WRITE for node {} failed: {}",
                    region.node_id,
                    std::io::Error::last_os_error()
                );
            }
        }
    }

    /// Find the node region containing a GPA.
    ///
    /// Regions are sorted by `gpa_start`, so this uses binary search.
    #[allow(dead_code)]
    pub fn region_for_gpa(&self, gpa: u64) -> Option<&NodeRegion> {
        let idx = self
            .regions
            .partition_point(|r| r.gpa_start <= gpa)
            .checked_sub(1)?;
        let r = &self.regions[idx];
        if gpa < r.gpa_start + r.size {
            Some(r)
        } else {
            None
        }
    }

    /// Node region by node_id.
    #[allow(dead_code)]
    pub fn region_for_node(&self, node_id: u32) -> Option<&NodeRegion> {
        self.regions.iter().find(|r| r.node_id == node_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vmm::topology::{NumaNode, Topology};

    #[test]
    fn uniform_single_region() {
        let topo = Topology::new(1, 2, 4, 2);
        let layout = NumaMemoryLayout::compute(&topo, 256, 0).unwrap();
        assert!(layout.is_single_region());
        assert_eq!(layout.total_bytes(), 256 << 20);
        assert_eq!(layout.regions().len(), 1);
        assert_eq!(layout.regions()[0].node_id, 0);
        assert_eq!(layout.regions()[0].gpa_start, 0);
        assert_eq!(layout.regions()[0].size, 256 << 20);
        assert_eq!(layout.regions()[0].slot, 0);
        assert_eq!(layout.next_slot(), 1);
    }

    #[test]
    fn uniform_multi_numa_splits_evenly() {
        let topo = Topology::new(2, 4, 2, 1);
        let layout = NumaMemoryLayout::compute(&topo, 512, 0).unwrap();
        assert_eq!(layout.regions().len(), 2);
        assert_eq!(layout.regions()[0].node_id, 0);
        assert_eq!(layout.regions()[0].size, 256 << 20);
        assert_eq!(layout.regions()[0].slot, 0);
        assert_eq!(layout.regions()[1].node_id, 1);
        assert_eq!(layout.regions()[1].gpa_start, 256 << 20);
        assert_eq!(layout.regions()[1].size, 256 << 20);
        assert_eq!(layout.regions()[1].slot, 1);
    }

    #[test]
    fn uniform_multi_numa_remainder() {
        let topo = Topology::new(3, 3, 2, 1);
        let layout = NumaMemoryLayout::compute(&topo, 100, 0).unwrap();
        assert_eq!(layout.regions().len(), 3);
        let sizes: Vec<u64> = layout.regions().iter().map(|r| r.size).collect();
        assert_eq!(sizes[0], 33 << 20);
        assert_eq!(sizes[1], 33 << 20);
        assert_eq!(sizes[2], 34 << 20);
        assert_eq!(layout.total_bytes(), 100 << 20);
    }

    static TWO_NODES: [NumaNode; 2] = [NumaNode::new(2, 256), NumaNode::new(2, 256)];

    #[test]
    fn with_nodes_two_regions() {
        let topo = Topology::with_nodes(4, 2, &TWO_NODES);
        let layout = NumaMemoryLayout::compute(&topo, 512, 0).unwrap();
        assert!(!layout.is_single_region());
        assert_eq!(layout.regions().len(), 2);

        let r0 = &layout.regions()[0];
        assert_eq!(r0.node_id, 0);
        assert_eq!(r0.gpa_start, 0);
        assert_eq!(r0.size, 256 << 20);
        assert_eq!(r0.slot, 0);

        let r1 = &layout.regions()[1];
        assert_eq!(r1.node_id, 1);
        assert_eq!(r1.gpa_start, 256 << 20);
        assert_eq!(r1.size, 256 << 20);
        assert_eq!(r1.slot, 1);

        assert_eq!(layout.total_bytes(), 512 << 20);
        assert_eq!(layout.end_gpa(), 512 << 20);
        assert_eq!(layout.next_slot(), 2);
    }

    static ASYM_NODES: [NumaNode; 2] = [NumaNode::new(1, 128), NumaNode::new(3, 384)];

    #[test]
    fn asymmetric_node_memory() {
        let topo = Topology::with_nodes(2, 1, &ASYM_NODES);
        let layout = NumaMemoryLayout::compute(&topo, 512, 0).unwrap();
        assert_eq!(layout.regions().len(), 2);
        assert_eq!(layout.regions()[0].size, 128 << 20);
        assert_eq!(layout.regions()[1].size, 384 << 20);
        assert_eq!(layout.regions()[1].gpa_start, 128 << 20);
    }

    static CXL_NODES: [NumaNode; 3] = [
        NumaNode::new(2, 256),
        NumaNode::new(2, 256),
        NumaNode::new(0, 128),
    ];

    #[test]
    fn cxl_memory_only_node() {
        let topo = Topology::with_nodes(4, 1, &CXL_NODES);
        let layout = NumaMemoryLayout::compute(&topo, 640, 0).unwrap();
        assert_eq!(layout.regions().len(), 3);

        assert_eq!(layout.regions()[0].node_id, 0);
        assert_eq!(layout.regions()[1].node_id, 1);
        assert_eq!(layout.regions()[2].node_id, 2);
        assert_eq!(layout.regions()[2].size, 128 << 20);
    }

    static CXL_ZERO_MEM: [NumaNode; 3] = [
        NumaNode::new(2, 256),
        NumaNode::new(0, 0),
        NumaNode::new(2, 256),
    ];

    #[test]
    fn cxl_zero_memory_node_skipped() {
        let topo = Topology::with_nodes(4, 1, &CXL_ZERO_MEM);
        let layout = NumaMemoryLayout::compute(&topo, 512, 0).unwrap();
        assert_eq!(layout.regions().len(), 2);
        assert_eq!(layout.regions()[0].node_id, 0);
        assert_eq!(layout.regions()[1].node_id, 2);
    }

    #[test]
    fn aarch64_dram_base() {
        let topo = Topology::with_nodes(4, 2, &TWO_NODES);
        let dram_base = 0x4000_0000u64;
        let layout = NumaMemoryLayout::compute(&topo, 512, dram_base).unwrap();
        assert_eq!(layout.dram_base(), dram_base);
        assert_eq!(layout.regions()[0].gpa_start, dram_base);
        assert_eq!(layout.regions()[1].gpa_start, dram_base + (256 << 20));
        assert_eq!(layout.end_gpa(), dram_base + (512 << 20));
    }

    #[test]
    fn memory_mismatch_error() {
        let topo = Topology::with_nodes(4, 2, &TWO_NODES);
        let err = NumaMemoryLayout::compute(&topo, 1024, 0).unwrap_err();
        assert!(format!("{err}").contains("must equal"), "got: {err}");
    }

    #[test]
    fn region_for_gpa_lookup() {
        let topo = Topology::with_nodes(4, 2, &TWO_NODES);
        let layout = NumaMemoryLayout::compute(&topo, 512, 0).unwrap();

        let r = layout.region_for_gpa(0).unwrap();
        assert_eq!(r.node_id, 0);

        let r = layout.region_for_gpa((256 << 20) - 1).unwrap();
        assert_eq!(r.node_id, 0);

        let r = layout.region_for_gpa(256 << 20).unwrap();
        assert_eq!(r.node_id, 1);

        assert!(layout.region_for_gpa(512 << 20).is_none());
    }

    #[test]
    fn region_for_gpa_with_dram_base() {
        let dram_base = 0x4000_0000u64;
        let topo = Topology::with_nodes(4, 2, &TWO_NODES);
        let layout = NumaMemoryLayout::compute(&topo, 512, dram_base).unwrap();

        assert!(layout.region_for_gpa(0).is_none());
        assert_eq!(layout.region_for_gpa(dram_base).unwrap().node_id, 0);
        assert_eq!(
            layout
                .region_for_gpa(dram_base + (256 << 20))
                .unwrap()
                .node_id,
            1
        );
    }

    #[test]
    fn region_for_node_lookup() {
        let topo = Topology::with_nodes(4, 2, &TWO_NODES);
        let layout = NumaMemoryLayout::compute(&topo, 512, 0).unwrap();

        assert_eq!(layout.region_for_node(0).unwrap().gpa_start, 0);
        assert_eq!(layout.region_for_node(1).unwrap().gpa_start, 256 << 20);
        assert!(layout.region_for_node(5).is_none());
    }

    #[test]
    fn slot_assignment_contiguous() {
        let topo = Topology::with_nodes(4, 1, &CXL_NODES);
        let layout = NumaMemoryLayout::compute(&topo, 640, 0).unwrap();
        for (i, r) in layout.regions().iter().enumerate() {
            assert_eq!(r.slot, i as u32);
        }
    }

    #[test]
    fn single_node_with_nodes() {
        static ONE: [NumaNode; 1] = [NumaNode::new(4, 512)];
        let topo = Topology::with_nodes(2, 1, &ONE);
        let layout = NumaMemoryLayout::compute(&topo, 512, 0).unwrap();
        assert!(layout.is_single_region());
        assert_eq!(layout.regions()[0].size, 512 << 20);
    }

    #[test]
    fn allocate_register_single_region() {
        let topo = Topology::new(1, 1, 1, 1);
        let layout = NumaMemoryLayout::compute(&topo, 64, 0).unwrap();

        let kvm = kvm_ioctls::Kvm::new().unwrap();
        let vm_fd = kvm.create_vm().unwrap();

        let alloc = layout.allocate_and_register(&vm_fd, false, false).unwrap();

        use vm_memory::GuestMemoryRegion;
        let total: u64 = alloc.guest_mem.iter().map(|r| r.len()).sum();
        assert_eq!(total, 64 << 20);
        assert_eq!(alloc.guest_mem.iter().count(), 1);
    }

    #[test]
    fn allocate_register_multi_node_per_region() {
        let topo = Topology::with_nodes(4, 2, &TWO_NODES);
        let layout = NumaMemoryLayout::compute(&topo, 512, 0).unwrap();

        let kvm = kvm_ioctls::Kvm::new().unwrap();
        let vm_fd = kvm.create_vm().unwrap();

        let alloc = layout.allocate_and_register(&vm_fd, false, false).unwrap();

        use vm_memory::GuestMemoryRegion;
        let total: u64 = alloc.guest_mem.iter().map(|r| r.len()).sum();
        assert_eq!(total, 512 << 20);
        // Per-node MAP_FIXED: one GuestMemoryMmap region per node.
        assert_eq!(alloc.guest_mem.iter().count(), 2);
    }

    #[test]
    fn contiguous_host_va() {
        let topo = Topology::with_nodes(4, 2, &TWO_NODES);
        let layout = NumaMemoryLayout::compute(&topo, 512, 0).unwrap();

        let kvm = kvm_ioctls::Kvm::new().unwrap();
        let vm_fd = kvm.create_vm().unwrap();

        let alloc = layout.allocate_and_register(&vm_fd, false, false).unwrap();

        let base = alloc.guest_mem.get_host_address(GuestAddress(0)).unwrap();
        let mid = alloc
            .guest_mem
            .get_host_address(GuestAddress(256 << 20))
            .unwrap();
        let offset = unsafe { mid.offset_from(base) };
        assert_eq!(offset, (256isize << 20));
    }

    #[test]
    fn cross_region_write_read() {
        let topo = Topology::with_nodes(4, 2, &TWO_NODES);
        let layout = NumaMemoryLayout::compute(&topo, 512, 0).unwrap();

        let kvm = kvm_ioctls::Kvm::new().unwrap();
        let vm_fd = kvm.create_vm().unwrap();

        let alloc = layout.allocate_and_register(&vm_fd, false, false).unwrap();

        use vm_memory::Bytes;

        let boundary = (256u64 << 20) - 4;
        let data: [u8; 8] = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE];
        alloc
            .guest_mem
            .write_slice(&data, GuestAddress(boundary))
            .unwrap();

        let mut readback = [0u8; 8];
        alloc
            .guest_mem
            .read_slice(&mut readback, GuestAddress(boundary))
            .unwrap();
        assert_eq!(data, readback);
    }

    #[test]
    fn uniform_multi_numa_allocate() {
        let topo = Topology::new(2, 2, 2, 1);
        let layout = NumaMemoryLayout::compute(&topo, 128, 0).unwrap();
        assert_eq!(layout.regions().len(), 2);

        let kvm = kvm_ioctls::Kvm::new().unwrap();
        let vm_fd = kvm.create_vm().unwrap();

        let alloc = layout.allocate_and_register(&vm_fd, false, false).unwrap();

        use vm_memory::GuestMemoryRegion;
        let total: u64 = alloc.guest_mem.iter().map(|r| r.len()).sum();
        assert_eq!(total, 128 << 20);
        // Uniform multi-NUMA: one region per node.
        assert_eq!(alloc.guest_mem.iter().count(), 2);
    }

    #[test]
    fn reservation_guard_munmaps_on_drop() {
        let topo = Topology::new(1, 1, 1, 1);
        let layout = NumaMemoryLayout::compute(&topo, 64, 0).unwrap();

        let kvm = kvm_ioctls::Kvm::new().unwrap();
        let vm_fd = kvm.create_vm().unwrap();

        let alloc = layout.allocate_and_register(&vm_fd, false, false).unwrap();

        let addr = alloc.reservation.addr;
        let size = alloc.reservation.size;
        assert!(!addr.is_null());
        assert_eq!(size, 64 << 20);
        // Drop releases the VA reservation.
        drop(alloc);
    }

    #[test]
    fn three_node_allocation() {
        let topo = Topology::with_nodes(4, 1, &CXL_NODES);
        let layout = NumaMemoryLayout::compute(&topo, 640, 0).unwrap();
        assert_eq!(layout.regions().len(), 3);

        let kvm = kvm_ioctls::Kvm::new().unwrap();
        let vm_fd = kvm.create_vm().unwrap();

        let alloc = layout.allocate_and_register(&vm_fd, false, false).unwrap();

        use vm_memory::GuestMemoryRegion;
        assert_eq!(alloc.guest_mem.iter().count(), 3);
        let total: u64 = alloc.guest_mem.iter().map(|r| r.len()).sum();
        assert_eq!(total, 640 << 20);
    }
}
