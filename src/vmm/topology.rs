/// CPU topology specification with NUMA memory topology.
///
/// Models the hierarchy: NUMA nodes → LLCs → cores → threads.
///
/// Each NUMA node owns a contiguous range of LLCs and a memory region.
/// When `nodes` is `None` (the default), memory and LLCs are distributed
/// uniformly across `numa_nodes` synthetic nodes with 10/20 distances.
///
/// Use [`new`](Self::new) for the simple uniform case, or
/// [`with_nodes`](Self::with_nodes) for explicit per-node configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Topology {
    /// Total number of last-level caches across the whole VM; must be
    /// a multiple of `numa_nodes` when `nodes` is `None`.
    pub llcs: u32,
    /// Physical cores grouped into each LLC.
    pub cores_per_llc: u32,
    /// Hardware threads exposed per core (`1` = no SMT, `2` = SMT-2).
    pub threads_per_core: u32,
    /// Number of NUMA nodes.
    pub numa_nodes: u32,
    /// Per-node configuration. When `None`, LLCs and memory are
    /// distributed uniformly. When `Some`, the slice length must
    /// equal `numa_nodes` and the sum of all `NumaNode::llcs` must
    /// equal `self.llcs`.
    pub nodes: Option<&'static [NumaNode]>,
    /// Inter-node distance matrix. When `None`, distances default to
    /// 10 (local) / 20 (remote). When `Some`, the matrix dimension
    /// must equal `numa_nodes`.
    pub distances: Option<&'static NumaDistance>,
}

/// Per-NUMA-node configuration.
///
/// `llcs = 0` models a CXL memory-only node: the node has RAM but no
/// CPUs or LLCs. Such nodes appear in the SRAT memory affinity table
/// and SLIT distance matrix but contribute no CPU affinity entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NumaNode {
    /// Number of LLCs owned by this node. Zero means memory-only (CXL).
    pub llcs: u32,
    /// Memory attached to this node in MiB.
    pub memory_mb: u32,
    /// HMAT access latency in nanoseconds. `None` uses the default
    /// (100ns for CPU-bearing, 300ns for memory-only). Emitted as
    /// an SLLBI access_latency entry in the HMAT table by
    /// `write_hmat` in `x86_64/acpi.rs`. x86_64 only — aarch64 does
    /// not expose an HMAT-equivalent to the guest.
    pub latency_ns: Option<u32>,
    /// HMAT read bandwidth in MB/s. `None` uses the default
    /// (51200 MB/s for CPU-bearing, 20480 MB/s for memory-only).
    /// Emitted as an SLLBI access_bandwidth entry in the HMAT table
    /// by `write_hmat` in `x86_64/acpi.rs`. x86_64 only.
    pub bandwidth_mbs: Option<u32>,
    /// HMAT Type 2 memory-side cache. `None` means no cache entry
    /// is emitted for this node. x86_64 only.
    pub mem_side_cache: Option<MemSideCache>,
}

/// HMAT Type 2 memory-side cache descriptor.
///
/// Models a hardware cache between the CPU and memory on this node
/// (e.g. CXL HDM decoder cache, HBM cache). Emitted as an HMAT
/// Memory Side Cache Information Structure.
///
/// `associativity` and `write_policy` occupy 4-bit nibbles in the
/// HMAT cache_attributes field. Values above 15 are invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemSideCache {
    /// Cache size in bytes.
    pub size: u64,
    /// Cache associativity (0=none, 1=direct-mapped, 2=complex).
    /// Must be <= 15 (4-bit field).
    pub associativity: u8,
    /// Write policy (0=none, 1=write-back, 2=write-through).
    /// Must be <= 15 (4-bit field).
    pub write_policy: u8,
    /// Cache line size in bytes.
    pub line_size: u16,
}

impl MemSideCache {
    /// Const constructor with validation.
    ///
    /// Panics if `associativity > 15` or `write_policy > 15`
    /// (4-bit HMAT nibble fields).
    pub const fn new(size: u64, associativity: u8, write_policy: u8, line_size: u16) -> Self {
        assert!(
            associativity <= 15,
            "MemSideCache: associativity must fit in 4 bits (0-15)"
        );
        assert!(
            write_policy <= 15,
            "MemSideCache: write_policy must fit in 4 bits (0-15)"
        );
        Self {
            size,
            associativity,
            write_policy,
            line_size,
        }
    }

    /// Non-panicking validation.
    pub fn validate(&self) -> Result<(), String> {
        if self.associativity > 15 {
            return Err(format!(
                "associativity {} exceeds 4-bit maximum (15)",
                self.associativity
            ));
        }
        if self.write_policy > 15 {
            return Err(format!(
                "write_policy {} exceeds 4-bit maximum (15)",
                self.write_policy
            ));
        }
        Ok(())
    }
}

impl NumaNode {
    /// Const constructor.
    pub const fn new(llcs: u32, memory_mb: u32) -> Self {
        Self {
            llcs,
            memory_mb,
            latency_ns: None,
            bandwidth_mbs: None,
            mem_side_cache: None,
        }
    }

    /// Const constructor with HMAT attributes.
    pub const fn with_hmat(llcs: u32, memory_mb: u32, latency_ns: u32, bandwidth_mbs: u32) -> Self {
        Self {
            llcs,
            memory_mb,
            latency_ns: Some(latency_ns),
            bandwidth_mbs: Some(bandwidth_mbs),
            mem_side_cache: None,
        }
    }

    /// Attach a memory-side cache descriptor.
    pub const fn with_cache(mut self, cache: MemSideCache) -> Self {
        self.mem_side_cache = Some(cache);
        self
    }

    /// Whether this is a memory-only node (CXL: has RAM but no CPUs).
    pub const fn is_memory_only(&self) -> bool {
        self.llcs == 0
    }
}

/// NxN inter-NUMA-node distance matrix.
///
/// Stored as a flat row-major array. ACPI SLIT requires diagonal = 10
/// and off-diagonal > 10. ktstr additionally enforces symmetry.
///
/// Construct via [`NumaDistance::new`] which validates all invariants
/// at const-eval time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NumaDistance {
    /// Number of NUMA nodes (matrix is `n x n`).
    n: u32,
    /// Row-major distance values. Length must be `n * n`.
    entries: &'static [u8],
}

impl NumaDistance {
    /// Const constructor with full validation.
    ///
    /// Panics if:
    /// - `n == 0`
    /// - `entries.len() != n * n`
    /// - any diagonal entry is not 10
    /// - any off-diagonal entry is not > 10
    /// - the matrix is not symmetric
    pub const fn new(n: u32, entries: &'static [u8]) -> Self {
        assert!(n > 0, "NumaDistance: n must be > 0");
        let expected = (n as usize) * (n as usize);
        assert!(
            entries.len() == expected,
            "NumaDistance: entries.len() must equal n * n"
        );
        Self::validate_entries(n, entries);
        Self { n, entries }
    }

    const fn validate_entries(n: u32, entries: &[u8]) {
        let dim = n as usize;
        let mut i = 0;
        while i < dim {
            let mut j = 0;
            while j < dim {
                let idx = i * dim + j;
                if i == j {
                    assert!(
                        entries[idx] == 10,
                        "NumaDistance: diagonal entry must be 10"
                    );
                } else {
                    assert!(
                        entries[idx] > 10,
                        "NumaDistance: off-diagonal entry must be > 10"
                    );
                    let sym_idx = j * dim + i;
                    assert!(
                        entries[idx] == entries[sym_idx],
                        "NumaDistance: matrix must be symmetric"
                    );
                }
                j += 1;
            }
            i += 1;
        }
    }

    /// Non-panicking validation.
    pub fn validate(&self) -> Result<(), String> {
        if self.n == 0 {
            return Err("n must be > 0".into());
        }
        let expected = (self.n as usize) * (self.n as usize);
        if self.entries.len() != expected {
            return Err(format!(
                "entries.len() ({}) must equal n * n ({})",
                self.entries.len(),
                expected
            ));
        }
        let dim = self.n as usize;
        for i in 0..dim {
            for j in 0..dim {
                let v = self.entries[i * dim + j];
                if i == j {
                    if v != 10 {
                        return Err(format!("diagonal entry [{i}][{j}] is {v}, must be 10"));
                    }
                } else {
                    if v <= 10 {
                        return Err(format!(
                            "off-diagonal entry [{i}][{j}] is {v}, must be > 10"
                        ));
                    }
                    let sym = self.entries[j * dim + i];
                    if v != sym {
                        return Err(format!("asymmetric: [{i}][{j}]={v} != [{j}][{i}]={sym}"));
                    }
                }
            }
        }
        Ok(())
    }

    /// Matrix dimension (number of NUMA nodes).
    pub const fn dimension(&self) -> u32 {
        self.n
    }

    /// Distance from node `i` to node `j`.
    pub const fn distance(&self, i: u32, j: u32) -> u8 {
        self.entries[(i as usize) * (self.n as usize) + (j as usize)]
    }

    /// Raw row-major entries.
    pub const fn entries(&self) -> &[u8] {
        self.entries
    }
}

/// Formats as `{numa}n{llcs}l{cores}c{threads}t` — e.g. `1n2l4c2t` =
/// 1 NUMA node, 2 LLCs, 4 cores/LLC, 2 threads/core.
impl std::fmt::Display for Topology {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}n{}l{}c{}t",
            self.numa_nodes, self.llcs, self.cores_per_llc, self.threads_per_core,
        )
    }
}

impl Topology {
    /// Fallback topology used by
    /// [`Payload::topology`](crate::test_support::Payload::topology)
    /// for binary-kind payloads that have no scheduler-side topology
    /// opinion. Matches the inline default previously baked into
    /// [`KtstrTestEntry::DEFAULT`](crate::test_support::KtstrTestEntry::DEFAULT):
    /// 1 NUMA node / 1 LLC / 2 cores / 1 thread (2 CPUs total), the
    /// smallest VM shape that runs the harness meaningfully.
    pub const DEFAULT_FOR_PAYLOAD: Topology = Topology {
        llcs: 1,
        cores_per_llc: 2,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };

    /// Validated const constructor for uniform topologies.
    ///
    /// Produces a topology where LLCs and memory are distributed evenly
    /// across NUMA nodes, with default 10/20 distances.
    ///
    /// Invariants:
    /// - All fields must be > 0.
    /// - `llcs` must be divisible by `numa_nodes`.
    /// - Total CPU count must not overflow `u32`.
    ///
    /// See [`validate`](Self::validate) for a non-panicking alternative.
    pub const fn new(
        numa_nodes: u32,
        llcs: u32,
        cores_per_llc: u32,
        threads_per_core: u32,
    ) -> Self {
        assert!(llcs > 0, "invalid Topology: llcs must be > 0");
        assert!(
            cores_per_llc > 0,
            "invalid Topology: cores_per_llc must be > 0"
        );
        assert!(
            threads_per_core > 0,
            "invalid Topology: threads_per_core must be > 0"
        );
        assert!(numa_nodes > 0, "invalid Topology: numa_nodes must be > 0");
        assert!(
            llcs.is_multiple_of(numa_nodes),
            "invalid Topology: llcs must be divisible by numa_nodes"
        );
        let cpus_per_llc = match cores_per_llc.checked_mul(threads_per_core) {
            Some(v) => v,
            None => panic!("invalid Topology: total CPU count overflows u32"),
        };
        match llcs.checked_mul(cpus_per_llc) {
            Some(_) => {}
            None => panic!("invalid Topology: total CPU count overflows u32"),
        };
        Topology {
            llcs,
            cores_per_llc,
            threads_per_core,
            numa_nodes,
            nodes: None,
            distances: None,
        }
    }

    /// Const constructor with explicit per-node configuration.
    ///
    /// Total LLC count is computed from the sum of `NumaNode::llcs`
    /// across all nodes. Memory-only nodes (llcs=0) are permitted.
    ///
    /// Panics if:
    /// - `nodes` is empty
    /// - `cores_per_llc == 0`
    /// - `threads_per_core == 0`
    /// - node LLC sum overflows `u32`
    /// - a CPU-bearing node (`llcs > 0`) has `memory_mb == 0`
    /// - total CPU count overflows `u32`
    /// - no node has LLCs (at least one must have `llcs > 0`)
    pub const fn with_nodes(
        cores_per_llc: u32,
        threads_per_core: u32,
        nodes: &'static [NumaNode],
    ) -> Self {
        assert!(
            !nodes.is_empty(),
            "invalid Topology: nodes must not be empty"
        );
        assert!(
            cores_per_llc > 0,
            "invalid Topology: cores_per_llc must be > 0"
        );
        assert!(
            threads_per_core > 0,
            "invalid Topology: threads_per_core must be > 0"
        );

        let mut llcs: u32 = 0;
        let mut i = 0;
        while i < nodes.len() {
            llcs = match llcs.checked_add(nodes[i].llcs) {
                Some(v) => v,
                None => panic!("invalid Topology: node LLC sum overflows u32"),
            };
            assert!(
                !(nodes[i].llcs > 0 && nodes[i].memory_mb == 0),
                "invalid Topology: CPU-bearing node has zero memory"
            );
            i += 1;
        }
        assert!(llcs > 0, "invalid Topology: total LLCs must be > 0");

        let cpus_per_llc = match cores_per_llc.checked_mul(threads_per_core) {
            Some(v) => v,
            None => panic!("invalid Topology: total CPU count overflows u32"),
        };
        match llcs.checked_mul(cpus_per_llc) {
            Some(_) => {}
            None => panic!("invalid Topology: total CPU count overflows u32"),
        };

        Topology {
            llcs,
            cores_per_llc,
            threads_per_core,
            numa_nodes: nodes.len() as u32,
            nodes: Some(nodes),
            distances: None,
        }
    }

    /// Attach a distance matrix. Panics if dimension doesn't match.
    pub const fn with_distances(mut self, distances: &'static NumaDistance) -> Self {
        assert!(
            distances.n == self.numa_nodes,
            "invalid Topology: NumaDistance dimension must equal numa_nodes"
        );
        self.distances = Some(distances);
        self
    }

    /// Non-panicking validation.
    ///
    /// Returns `Ok(())` if all invariants hold, or `Err` with a
    /// description of the first violated invariant.
    pub fn validate(&self) -> Result<(), String> {
        if self.llcs == 0 {
            return Err("llcs must be > 0".into());
        }
        if self.cores_per_llc == 0 {
            return Err("cores_per_llc must be > 0".into());
        }
        if self.threads_per_core == 0 {
            return Err("threads_per_core must be > 0".into());
        }
        if self.numa_nodes == 0 {
            return Err("numa_nodes must be > 0".into());
        }
        if self
            .cores_per_llc
            .checked_mul(self.threads_per_core)
            .and_then(|x| self.llcs.checked_mul(x))
            .is_none()
        {
            return Err("total CPU count overflows u32".into());
        }
        match &self.nodes {
            None => {
                if !self.llcs.is_multiple_of(self.numa_nodes) {
                    return Err(format!(
                        "llcs ({}) must be divisible by numa_nodes ({})",
                        self.llcs, self.numa_nodes,
                    ));
                }
            }
            Some(nodes) => {
                if nodes.len() != self.numa_nodes as usize {
                    return Err(format!(
                        "nodes.len() ({}) must equal numa_nodes ({})",
                        nodes.len(),
                        self.numa_nodes,
                    ));
                }
                let llc_sum: u32 = nodes.iter().map(|n| n.llcs).sum();
                if llc_sum != self.llcs {
                    return Err(format!(
                        "sum of node LLCs ({llc_sum}) must equal total llcs ({})",
                        self.llcs,
                    ));
                }
                for (i, node) in nodes.iter().enumerate() {
                    if node.llcs > 0 && node.memory_mb == 0 {
                        return Err(format!("node {i} has {} LLCs but zero memory", node.llcs,));
                    }
                }
            }
        }
        if let Some(d) = &self.distances {
            if d.n != self.numa_nodes {
                return Err(format!(
                    "NumaDistance dimension ({}) must equal numa_nodes ({})",
                    d.n, self.numa_nodes,
                ));
            }
            d.validate()?;
        }
        Ok(())
    }

    /// Total vCPU count = `llcs * cores_per_llc * threads_per_core`.
    pub fn total_cpus(&self) -> u32 {
        self.llcs * self.cores_per_llc * self.threads_per_core
    }

    /// Number of LLC domains in the topology.
    pub fn num_llcs(&self) -> u32 {
        self.llcs
    }

    /// Number of NUMA nodes in the topology.
    pub fn num_numa_nodes(&self) -> u32 {
        self.numa_nodes
    }

    /// LLCs owned by NUMA node `node_id`.
    ///
    /// With explicit nodes, returns `nodes[node_id].llcs`.
    /// With uniform distribution, returns `llcs / numa_nodes`.
    pub fn llcs_in_node(&self, node_id: u32) -> u32 {
        match &self.nodes {
            Some(nodes) => nodes[node_id as usize].llcs,
            None => self.llcs / self.numa_nodes,
        }
    }

    /// LLCs per NUMA node (uniform distribution only).
    ///
    /// Panics if the topology uses explicit nodes (use `llcs_in_node()`
    /// instead).
    pub fn llcs_per_numa_node(&self) -> u32 {
        assert!(
            self.nodes.is_none(),
            "llcs_per_numa_node() requires uniform topology; use llcs_in_node() instead"
        );
        assert!(self.numa_nodes > 0, "numa_nodes must be > 0");
        assert!(
            self.llcs.is_multiple_of(self.numa_nodes),
            "llcs ({}) must be divisible by numa_nodes ({})",
            self.llcs,
            self.numa_nodes,
        );
        self.llcs / self.numa_nodes
    }

    /// NUMA node that owns the given LLC index.
    ///
    /// With explicit nodes, walks the node list to find the owning node.
    /// With uniform distribution, computes `llc_id / llcs_per_node`.
    ///
    /// Out-of-bounds `llc_id` (>= total LLCs): with explicit nodes,
    /// saturates to the last node index; with uniform distribution, no
    /// bounds check — returns `llc_id / llcs_per_node`, which may
    /// exceed `numa_nodes - 1`.
    pub fn numa_node_of(&self, llc_id: u32) -> u32 {
        match &self.nodes {
            Some(nodes) => {
                let mut cumulative: u32 = 0;
                for (i, node) in nodes.iter().enumerate() {
                    cumulative += node.llcs;
                    if llc_id < cumulative {
                        return i as u32;
                    }
                }
                (nodes.len() - 1) as u32
            }
            None => {
                let per_node = self.llcs / self.numa_nodes;
                llc_id / per_node
            }
        }
    }

    /// First LLC index owned by NUMA node `node_id`.
    ///
    /// Panics if `node_id > numa_nodes` for explicit-node topologies
    /// (the walk would index past the end of the node slice). Uniform
    /// topologies do not bounds-check and return `node_id *
    /// llcs_per_node` for any input.
    pub fn first_llc_in_node(&self, node_id: u32) -> u32 {
        match &self.nodes {
            Some(nodes) => {
                let mut offset: u32 = 0;
                for i in 0..node_id as usize {
                    offset += nodes[i].llcs;
                }
                offset
            }
            None => {
                let per_node = self.llcs / self.numa_nodes;
                node_id * per_node
            }
        }
    }

    /// Memory in MiB for NUMA node `node_id`.
    ///
    /// With explicit nodes, returns `nodes[node_id].memory_mb`.
    /// With uniform distribution, returns `None` (caller must divide
    /// total memory evenly).
    pub fn node_memory_mb(&self, node_id: u32) -> Option<u32> {
        self.nodes.map(|nodes| nodes[node_id as usize].memory_mb)
    }

    /// Total memory across all explicit nodes, or `None` for uniform.
    pub fn total_node_memory_mb(&self) -> Option<u32> {
        self.nodes
            .map(|nodes| nodes.iter().map(|n| n.memory_mb).sum())
    }

    /// Distance from node `i` to node `j`.
    ///
    /// Returns the explicit distance if a matrix is attached,
    /// otherwise 10 for local and 20 for remote.
    pub fn distance(&self, i: u32, j: u32) -> u8 {
        match &self.distances {
            Some(d) => d.distance(i, j),
            None => {
                if i == j {
                    10
                } else {
                    20
                }
            }
        }
    }

    /// Whether any node is memory-only (CXL).
    pub fn has_memory_only_nodes(&self) -> bool {
        self.nodes
            .is_some_and(|nodes| nodes.iter().any(|n| n.is_memory_only()))
    }

    /// Number of nodes that have CPUs (non-memory-only).
    pub fn cpu_bearing_nodes(&self) -> u32 {
        match &self.nodes {
            Some(nodes) => nodes.iter().filter(|n| !n.is_memory_only()).count() as u32,
            None => self.numa_nodes,
        }
    }

    /// Decompose a logical CPU ID into (llc, core, thread).
    pub fn decompose(&self, cpu_id: u32) -> (u32, u32, u32) {
        let threads = self.threads_per_core;
        let cores = self.cores_per_llc;
        let thread_id = cpu_id % threads;
        let core_id = (cpu_id / threads) % cores;
        let llc_id = cpu_id / (threads * cores);
        (llc_id, core_id, thread_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_total_cpus() {
        let t = Topology {
            llcs: 2,
            cores_per_llc: 4,
            threads_per_core: 2,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        assert_eq!(t.total_cpus(), 16);
    }

    #[test]
    fn topology_num_llcs() {
        let t = Topology {
            llcs: 3,
            cores_per_llc: 4,
            threads_per_core: 2,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        assert_eq!(t.num_llcs(), 3);
    }

    #[test]
    fn decompose_simple() {
        let t = Topology {
            llcs: 2,
            cores_per_llc: 2,
            threads_per_core: 2,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
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
            llcs: 2,
            cores_per_llc: 4,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        assert_eq!(t.decompose(0), (0, 0, 0));
        assert_eq!(t.decompose(3), (0, 3, 0));
        assert_eq!(t.decompose(4), (1, 0, 0));
        assert_eq!(t.decompose(7), (1, 3, 0));
    }

    #[test]
    fn decompose_single_llc() {
        let t = Topology {
            llcs: 1,
            cores_per_llc: 4,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        assert_eq!(t.decompose(0), (0, 0, 0));
        assert_eq!(t.decompose(3), (0, 3, 0));
    }

    #[test]
    fn numa_node_of_single_node() {
        let t = Topology {
            llcs: 4,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        for llc in 0..4 {
            assert_eq!(t.numa_node_of(llc), 0);
        }
    }

    #[test]
    fn numa_node_of_two_nodes() {
        let t = Topology {
            llcs: 4,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 2,
            nodes: None,
            distances: None,
        };
        assert_eq!(t.llcs_per_numa_node(), 2);
        assert_eq!(t.numa_node_of(0), 0);
        assert_eq!(t.numa_node_of(1), 0);
        assert_eq!(t.numa_node_of(2), 1);
        assert_eq!(t.numa_node_of(3), 1);
    }

    #[test]
    fn numa_node_of_out_of_bounds_explicit_saturates() {
        static TWO: [NumaNode; 2] = [NumaNode::new(2, 512), NumaNode::new(2, 512)];
        let t = Topology::with_nodes(2, 1, &TWO);
        // Total LLCs = 4. Any llc_id >= 4 must saturate to last node (1).
        assert_eq!(t.numa_node_of(4), 1);
        assert_eq!(t.numa_node_of(999), 1);
    }

    #[test]
    fn numa_node_of_out_of_bounds_uniform_no_check() {
        // Uniform: numa_nodes=2, llcs=4 => llcs_per_node=2.
        // Per documented behavior: no bounds check, returns llc_id/2.
        let t = Topology::new(2, 4, 2, 1);
        assert_eq!(t.numa_node_of(100), 50);
        assert!(t.numa_node_of(100) > t.numa_nodes - 1);
    }

    #[test]
    fn num_numa_nodes() {
        let t = Topology {
            llcs: 6,
            cores_per_llc: 4,
            threads_per_core: 2,
            numa_nodes: 3,
            nodes: None,
            distances: None,
        };
        assert_eq!(t.num_numa_nodes(), 3);
        assert_eq!(t.llcs_per_numa_node(), 2);
    }

    #[test]
    #[should_panic(expected = "llcs_per_numa_node() requires uniform topology")]
    fn llcs_per_numa_node_panics_on_explicit_nodes() {
        static EXPLICIT: [NumaNode; 2] = [NumaNode::new(2, 512), NumaNode::new(2, 512)];
        let t = Topology::with_nodes(2, 1, &EXPLICIT);
        t.llcs_per_numa_node();
    }

    #[test]
    fn new_valid() {
        let t = Topology::new(2, 4, 2, 2);
        assert_eq!(t.numa_nodes, 2);
        assert_eq!(t.llcs, 4);
        assert_eq!(t.cores_per_llc, 2);
        assert_eq!(t.threads_per_core, 2);
        assert!(t.nodes.is_none());
        assert!(t.distances.is_none());
    }

    #[test]
    fn new_single_everything() {
        let t = Topology::new(1, 1, 1, 1);
        assert_eq!(t.total_cpus(), 1);
    }

    #[test]
    fn validate_valid() {
        let t = Topology {
            llcs: 4,
            cores_per_llc: 2,
            threads_per_core: 2,
            numa_nodes: 2,
            nodes: None,
            distances: None,
        };
        assert!(t.validate().is_ok());
    }

    #[test]
    fn validate_zero_llcs() {
        let t = Topology {
            llcs: 0,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let err = t.validate().unwrap_err();
        assert!(err.contains("llcs must be > 0"), "got: {err}");
    }

    #[test]
    fn validate_zero_cores() {
        let t = Topology {
            llcs: 1,
            cores_per_llc: 0,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let err = t.validate().unwrap_err();
        assert!(err.contains("cores_per_llc must be > 0"), "got: {err}");
    }

    #[test]
    fn validate_zero_threads() {
        let t = Topology {
            llcs: 1,
            cores_per_llc: 2,
            threads_per_core: 0,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let err = t.validate().unwrap_err();
        assert!(err.contains("threads_per_core must be > 0"), "got: {err}");
    }

    #[test]
    fn validate_zero_numa_nodes() {
        let t = Topology {
            llcs: 1,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 0,
            nodes: None,
            distances: None,
        };
        let err = t.validate().unwrap_err();
        assert!(err.contains("numa_nodes must be > 0"), "got: {err}");
    }

    #[test]
    fn validate_llcs_not_divisible_by_numa() {
        let t = Topology {
            llcs: 3,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 2,
            nodes: None,
            distances: None,
        };
        let err = t.validate().unwrap_err();
        assert!(err.contains("divisible"), "got: {err}");
    }

    #[test]
    #[should_panic(expected = "invalid Topology")]
    fn new_panics_zero_numa() {
        Topology::new(0, 2, 1, 1);
    }

    #[test]
    #[should_panic(expected = "invalid Topology")]
    fn new_panics_zero_llcs() {
        Topology::new(1, 0, 2, 1);
    }

    #[test]
    #[should_panic(expected = "invalid Topology")]
    fn new_panics_zero_cores() {
        Topology::new(1, 1, 0, 1);
    }

    #[test]
    #[should_panic(expected = "invalid Topology")]
    fn new_panics_zero_threads() {
        Topology::new(1, 1, 2, 0);
    }

    #[test]
    #[should_panic(expected = "invalid Topology")]
    fn new_panics_indivisible() {
        Topology::new(2, 3, 2, 1);
    }

    #[test]
    #[should_panic(expected = "invalid Topology")]
    fn new_panics_overflow() {
        Topology::new(1, 65536, 65536, 2);
    }

    #[test]
    fn validate_overflow() {
        let t = Topology {
            llcs: 65536,
            cores_per_llc: 65536,
            threads_per_core: 2,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let err = t.validate().unwrap_err();
        assert!(err.contains("overflows"), "got: {err}");
    }

    #[test]
    fn display_format() {
        let t = Topology {
            llcs: 2,
            cores_per_llc: 4,
            threads_per_core: 2,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        assert_eq!(t.to_string(), "1n2l4c2t");
    }

    #[test]
    fn display_format_multi_numa() {
        let t = Topology {
            llcs: 4,
            cores_per_llc: 8,
            threads_per_core: 2,
            numa_nodes: 2,
            nodes: None,
            distances: None,
        };
        assert_eq!(t.to_string(), "2n4l8c2t");
    }

    // -- NumaNode tests --

    #[test]
    fn numa_node_memory_only() {
        let n = NumaNode::new(0, 1024);
        assert!(n.is_memory_only());
    }

    #[test]
    fn numa_node_with_cpus() {
        let n = NumaNode::new(2, 512);
        assert!(!n.is_memory_only());
    }

    // -- NumaDistance tests --

    #[test]
    fn numa_distance_single_node() {
        static D: NumaDistance = NumaDistance::new(1, &[10]);
        assert_eq!(D.dimension(), 1);
        assert_eq!(D.distance(0, 0), 10);
    }

    #[test]
    fn numa_distance_two_nodes() {
        static D: NumaDistance = NumaDistance::new(2, &[10, 20, 20, 10]);
        assert_eq!(D.dimension(), 2);
        assert_eq!(D.distance(0, 0), 10);
        assert_eq!(D.distance(0, 1), 20);
        assert_eq!(D.distance(1, 0), 20);
        assert_eq!(D.distance(1, 1), 10);
    }

    #[test]
    fn numa_distance_three_nodes_varied_weights() {
        static D: NumaDistance = NumaDistance::new(3, &[10, 20, 30, 20, 10, 40, 30, 40, 10]);
        assert_eq!(D.distance(0, 2), 30);
        assert_eq!(D.distance(1, 2), 40);
    }

    #[test]
    fn numa_distance_validate_ok() {
        let d = NumaDistance {
            n: 2,
            entries: &[10, 20, 20, 10],
        };
        assert!(d.validate().is_ok());
    }

    #[test]
    fn numa_distance_validate_bad_diagonal() {
        let d = NumaDistance {
            n: 2,
            entries: &[11, 20, 20, 10],
        };
        let err = d.validate().unwrap_err();
        assert!(err.contains("diagonal"), "got: {err}");
    }

    #[test]
    fn numa_distance_validate_bad_offdiag() {
        let d = NumaDistance {
            n: 2,
            entries: &[10, 10, 10, 10],
        };
        let err = d.validate().unwrap_err();
        assert!(err.contains("off-diagonal"), "got: {err}");
    }

    #[test]
    fn numa_distance_validate_asymmetric() {
        let d = NumaDistance {
            n: 2,
            entries: &[10, 20, 30, 10],
        };
        let err = d.validate().unwrap_err();
        assert!(err.contains("asymmetric"), "got: {err}");
    }

    #[test]
    fn numa_distance_validate_wrong_size() {
        let d = NumaDistance {
            n: 2,
            entries: &[10, 20, 20],
        };
        let err = d.validate().unwrap_err();
        assert!(err.contains("n * n"), "got: {err}");
    }

    // -- with_nodes tests --

    static TWO_NODES: [NumaNode; 2] = [NumaNode::new(2, 512), NumaNode::new(2, 512)];

    #[test]
    fn with_nodes_basic() {
        let t = Topology::with_nodes(4, 2, &TWO_NODES);
        assert_eq!(t.numa_nodes, 2);
        assert_eq!(t.llcs, 4);
        assert_eq!(t.total_cpus(), 32);
        assert!(t.nodes.is_some());
    }

    #[test]
    fn with_nodes_numa_node_of() {
        let t = Topology::with_nodes(4, 2, &TWO_NODES);
        assert_eq!(t.numa_node_of(0), 0);
        assert_eq!(t.numa_node_of(1), 0);
        assert_eq!(t.numa_node_of(2), 1);
        assert_eq!(t.numa_node_of(3), 1);
    }

    static ASYMMETRIC_NODES: [NumaNode; 2] = [NumaNode::new(1, 256), NumaNode::new(3, 768)];

    #[test]
    fn with_nodes_asymmetric_llcs() {
        let t = Topology::with_nodes(2, 1, &ASYMMETRIC_NODES);
        assert_eq!(t.llcs_in_node(0), 1);
        assert_eq!(t.llcs_in_node(1), 3);
        assert_eq!(t.numa_node_of(0), 0);
        assert_eq!(t.numa_node_of(1), 1);
        assert_eq!(t.numa_node_of(2), 1);
        assert_eq!(t.numa_node_of(3), 1);
        assert_eq!(t.first_llc_in_node(0), 0);
        assert_eq!(t.first_llc_in_node(1), 1);
    }

    #[test]
    fn with_nodes_memory() {
        let t = Topology::with_nodes(2, 1, &ASYMMETRIC_NODES);
        assert_eq!(t.node_memory_mb(0), Some(256));
        assert_eq!(t.node_memory_mb(1), Some(768));
        assert_eq!(t.total_node_memory_mb(), Some(1024));
    }

    // -- CXL memory-only node tests --

    static CXL_NODES: [NumaNode; 3] = [
        NumaNode::new(2, 512),
        NumaNode::new(2, 512),
        NumaNode::new(0, 1024),
    ];

    #[test]
    fn cxl_memory_only_node() {
        let t = Topology::with_nodes(4, 1, &CXL_NODES);
        assert_eq!(t.numa_nodes, 3);
        assert!(t.has_memory_only_nodes());
        assert_eq!(t.cpu_bearing_nodes(), 2);
        assert_eq!(t.llcs_in_node(2), 0);
        assert_eq!(t.node_memory_mb(2), Some(1024));
    }

    #[test]
    fn cxl_first_llc_in_memory_only_node() {
        let t = Topology::with_nodes(4, 1, &CXL_NODES);
        assert_eq!(t.first_llc_in_node(0), 0);
        assert_eq!(t.first_llc_in_node(1), 2);
        assert_eq!(t.first_llc_in_node(2), 4);
    }

    static CXL_MIDDLE: [NumaNode; 3] = [
        NumaNode::new(2, 512),
        NumaNode::new(0, 256),
        NumaNode::new(2, 512),
    ];

    #[test]
    fn numa_node_of_cxl_middle_node() {
        let t = Topology::with_nodes(4, 1, &CXL_MIDDLE);
        assert_eq!(t.numa_nodes, 3);
        assert_eq!(t.numa_node_of(0), 0);
        assert_eq!(t.numa_node_of(1), 0);
        assert_eq!(t.numa_node_of(2), 2);
        assert_eq!(t.numa_node_of(3), 2);
        assert!(t.has_memory_only_nodes());
        assert_eq!(t.cpu_bearing_nodes(), 2);
    }

    #[test]
    fn first_llc_in_cxl_middle_node() {
        let t = Topology::with_nodes(4, 1, &CXL_MIDDLE);
        assert_eq!(t.first_llc_in_node(0), 0);
        assert_eq!(t.first_llc_in_node(1), 2);
        assert_eq!(t.first_llc_in_node(2), 2);
        assert_eq!(t.llcs_in_node(1), 0);
        assert_eq!(t.llcs_in_node(2), 2);
    }

    static CXL_FIRST: [NumaNode; 3] = [
        NumaNode::new(0, 256),
        NumaNode::new(2, 512),
        NumaNode::new(2, 512),
    ];

    #[test]
    fn cxl_first_node_numa_node_of() {
        let t = Topology::with_nodes(4, 1, &CXL_FIRST);
        assert_eq!(t.numa_node_of(0), 1);
        assert_eq!(t.numa_node_of(1), 1);
        assert_eq!(t.numa_node_of(2), 2);
        assert_eq!(t.numa_node_of(3), 2);
        assert_eq!(t.first_llc_in_node(0), 0);
        assert_eq!(t.first_llc_in_node(1), 0);
        assert_eq!(t.first_llc_in_node(2), 2);
    }

    static MULTI_CXL: [NumaNode; 4] = [
        NumaNode::new(2, 512),
        NumaNode::new(0, 256),
        NumaNode::new(0, 256),
        NumaNode::new(2, 512),
    ];

    #[test]
    fn multiple_consecutive_cxl_nodes() {
        let t = Topology::with_nodes(4, 1, &MULTI_CXL);
        assert_eq!(t.numa_nodes, 4);
        assert_eq!(t.cpu_bearing_nodes(), 2);
        assert_eq!(t.numa_node_of(0), 0);
        assert_eq!(t.numa_node_of(1), 0);
        assert_eq!(t.numa_node_of(2), 3);
        assert_eq!(t.numa_node_of(3), 3);
    }

    static ASYMMETRIC_HEAVY: [NumaNode; 2] = [NumaNode::new(1, 256), NumaNode::new(7, 1792)];

    #[test]
    fn highly_asymmetric_llcs() {
        let t = Topology::with_nodes(2, 1, &ASYMMETRIC_HEAVY);
        assert_eq!(t.numa_node_of(0), 0);
        assert_eq!(t.numa_node_of(1), 1);
        assert_eq!(t.numa_node_of(7), 1);
        assert_eq!(t.llcs_in_node(0), 1);
        assert_eq!(t.llcs_in_node(1), 7);
    }

    static CXL_DIST: NumaDistance = NumaDistance::new(3, &[10, 20, 30, 20, 10, 25, 30, 25, 10]);

    #[test]
    fn distance_with_cxl_middle() {
        let t = Topology::with_nodes(4, 1, &CXL_MIDDLE).with_distances(&CXL_DIST);
        assert_eq!(t.distance(1, 2), 25);
        assert_eq!(t.distance(0, 1), 20);
        assert!(t.validate().is_ok());
    }

    // -- with_distances tests --

    static DIST_2: NumaDistance = NumaDistance::new(2, &[10, 20, 20, 10]);

    #[test]
    fn with_distances() {
        let t = Topology::new(2, 4, 2, 1).with_distances(&DIST_2);
        assert_eq!(t.distance(0, 0), 10);
        assert_eq!(t.distance(0, 1), 20);
        assert_eq!(t.distance(1, 0), 20);
        assert_eq!(t.distance(1, 1), 10);
    }

    #[test]
    fn default_distances() {
        let t = Topology::new(2, 4, 2, 1);
        assert_eq!(t.distance(0, 0), 10);
        assert_eq!(t.distance(0, 1), 20);
    }

    static DIST_3: NumaDistance = NumaDistance::new(3, &[10, 20, 30, 20, 10, 25, 30, 25, 10]);

    #[test]
    fn with_nodes_and_distances() {
        let t = Topology::with_nodes(4, 1, &CXL_NODES).with_distances(&DIST_3);
        assert_eq!(t.distance(0, 2), 30);
        assert_eq!(t.distance(1, 2), 25);
        assert!(t.validate().is_ok());
    }

    // -- Validation with nodes --

    #[test]
    fn validate_with_nodes_ok() {
        let t = Topology::with_nodes(4, 2, &TWO_NODES);
        assert!(t.validate().is_ok());
    }

    #[test]
    fn validate_with_nodes_llc_mismatch() {
        static BAD: [NumaNode; 2] = [NumaNode::new(1, 256), NumaNode::new(1, 256)];
        let t = Topology {
            llcs: 4,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 2,
            nodes: Some(&BAD),
            distances: None,
        };
        let err = t.validate().unwrap_err();
        assert!(err.contains("sum of node LLCs"), "got: {err}");
    }

    #[test]
    fn validate_with_nodes_count_mismatch() {
        let t = Topology {
            llcs: 4,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 3,
            nodes: Some(&TWO_NODES),
            distances: None,
        };
        let err = t.validate().unwrap_err();
        assert!(err.contains("nodes.len()"), "got: {err}");
    }

    #[test]
    fn validate_distance_dimension_mismatch() {
        static BAD_DIST: NumaDistance = NumaDistance::new(1, &[10]);
        let t = Topology {
            llcs: 4,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 2,
            nodes: None,
            distances: Some(&BAD_DIST),
        };
        let err = t.validate().unwrap_err();
        assert!(err.contains("dimension"), "got: {err}");
    }

    #[test]
    fn validate_cpu_node_zero_memory() {
        static BAD: [NumaNode; 2] = [NumaNode::new(2, 0), NumaNode::new(2, 512)];
        let t = Topology {
            llcs: 4,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 2,
            nodes: Some(&BAD),
            distances: None,
        };
        let err = t.validate().unwrap_err();
        assert!(err.contains("zero memory"), "got: {err}");
    }

    #[test]
    fn uniform_no_node_memory() {
        let t = Topology::new(2, 4, 2, 1);
        assert!(t.node_memory_mb(0).is_none());
        assert!(t.total_node_memory_mb().is_none());
    }

    // -- const construction smoke tests --

    const _CONST_TOPO: Topology = Topology::new(1, 2, 4, 2);

    static _CONST_NODES: [NumaNode; 2] = [NumaNode::new(1, 256), NumaNode::new(1, 256)];
    const _CONST_WITH_NODES: Topology = Topology::with_nodes(4, 2, &_CONST_NODES);

    static _CONST_DIST: NumaDistance = NumaDistance::new(2, &[10, 20, 20, 10]);
    const _CONST_WITH_DIST: Topology = Topology::new(2, 2, 4, 2).with_distances(&_CONST_DIST);

    #[test]
    fn const_construction_valid() {
        assert!(_CONST_TOPO.validate().is_ok());
        assert!(_CONST_WITH_NODES.validate().is_ok());
        assert!(_CONST_WITH_DIST.validate().is_ok());
    }

    // -- single node edge case --

    #[test]
    fn single_node_topology() {
        let t = Topology::new(1, 1, 1, 1);
        assert_eq!(t.total_cpus(), 1);
        assert_eq!(t.numa_node_of(0), 0);
        assert_eq!(t.distance(0, 0), 10);
        assert!(!t.has_memory_only_nodes());
        assert_eq!(t.cpu_bearing_nodes(), 1);
    }

    #[test]
    fn first_llc_in_node_uniform() {
        let t = Topology::new(2, 4, 2, 1);
        assert_eq!(t.first_llc_in_node(0), 0);
        assert_eq!(t.first_llc_in_node(1), 2);
    }

    #[test]
    fn first_llc_in_node_at_numa_nodes_returns_total() {
        // Documented behavior: node_id == numa_nodes does NOT panic for
        // explicit nodes; the walk sums all llcs and returns the total.
        let t = Topology::with_nodes(2, 1, &TWO_NODES);
        assert_eq!(t.first_llc_in_node(2), 4);
    }

    #[test]
    #[should_panic]
    fn first_llc_in_node_above_numa_nodes_panics() {
        // Documented behavior: node_id > numa_nodes indexes past the
        // end of the node slice on the walk, panicking.
        let t = Topology::with_nodes(2, 1, &TWO_NODES);
        let _ = t.first_llc_in_node(3);
    }
}
