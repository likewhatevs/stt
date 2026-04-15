/// CPU topology specification.
///
/// Models the hierarchy: NUMA nodes → LLCs → cores → threads.
/// `llcs` is the total LLC count across all NUMA nodes.
/// `numa_nodes` groups LLCs into NUMA proximity domains.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Topology {
    pub llcs: u32,
    pub cores_per_llc: u32,
    pub threads_per_core: u32,
    pub numa_nodes: u32,
}

impl Topology {
    pub fn total_cpus(&self) -> u32 {
        self.llcs * self.cores_per_llc * self.threads_per_core
    }

    pub fn num_llcs(&self) -> u32 {
        self.llcs
    }

    pub fn num_numa_nodes(&self) -> u32 {
        self.numa_nodes
    }

    /// LLCs per NUMA node (uniform distribution).
    ///
    /// Requires `numa_nodes > 0` and `llcs % numa_nodes == 0`.
    pub fn llcs_per_numa_node(&self) -> u32 {
        debug_assert!(self.numa_nodes > 0, "numa_nodes must be > 0");
        debug_assert!(
            self.llcs.is_multiple_of(self.numa_nodes),
            "llcs ({}) must be divisible by numa_nodes ({})",
            self.llcs,
            self.numa_nodes,
        );
        self.llcs / self.numa_nodes
    }

    /// NUMA node that owns the given LLC index.
    ///
    /// Requires `numa_nodes > 0` and `llcs % numa_nodes == 0`.
    pub fn numa_node_of(&self, llc_id: u32) -> u32 {
        let per_node = self.llcs_per_numa_node();
        llc_id / per_node
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
        };
        // 2 LLCs x 2 cores x 2 threads = 8 CPUs
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
        };
        assert_eq!(t.llcs_per_numa_node(), 2);
        assert_eq!(t.numa_node_of(0), 0);
        assert_eq!(t.numa_node_of(1), 0);
        assert_eq!(t.numa_node_of(2), 1);
        assert_eq!(t.numa_node_of(3), 1);
    }

    #[test]
    fn num_numa_nodes() {
        let t = Topology {
            llcs: 6,
            cores_per_llc: 4,
            threads_per_core: 2,
            numa_nodes: 3,
        };
        assert_eq!(t.num_numa_nodes(), 3);
        assert_eq!(t.llcs_per_numa_node(), 2);
    }
}
