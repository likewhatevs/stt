/// CPU topology specification.
///
/// Models the hierarchy: NUMA nodes → LLCs → cores → threads.
/// `llcs` is the total LLC count across all NUMA nodes.
/// `numa_nodes` groups LLCs into NUMA proximity domains.
///
/// Use [`new`](Self::new) for validated construction, or
/// [`validate`](Self::validate) to check a struct-literal value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Topology {
    /// Total number of last-level caches across the whole VM; must be
    /// a multiple of `numa_nodes`.
    pub llcs: u32,
    /// Physical cores grouped into each LLC.
    pub cores_per_llc: u32,
    /// Hardware threads exposed per core (`1` = no SMT, `2` = SMT-2).
    pub threads_per_core: u32,
    /// Number of NUMA nodes; LLCs partition evenly across them.
    pub numa_nodes: u32,
}

/// Formats as `NnNlNcNt` — e.g. `1n2l4c2t` (1 NUMA node, 2 LLCs, 4 cores, 2 threads).
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
    /// Validated const constructor. Panics on invalid values.
    ///
    /// Invariants:
    /// - All fields must be > 0.
    /// - `llcs` must be divisible by `numa_nodes` (uniform LLC distribution).
    /// - Total CPU count (`llcs * cores_per_llc * threads_per_core`) must
    ///   not overflow `u32`.
    ///
    /// An invalid topology is a programmer error, not a runtime condition.
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
        // Overflow check.
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
        }
    }

    /// Non-panicking validation.
    ///
    /// Returns `Ok(())` if all invariants hold, or `Err` with a
    /// description of the first violated invariant.
    ///
    /// Checks the same invariants as [`new`](Self::new).
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
        if !self.llcs.is_multiple_of(self.numa_nodes) {
            return Err(format!(
                "llcs ({}) must be divisible by numa_nodes ({})",
                self.llcs, self.numa_nodes,
            ));
        }
        if self
            .cores_per_llc
            .checked_mul(self.threads_per_core)
            .and_then(|x| self.llcs.checked_mul(x))
            .is_none()
        {
            return Err("total CPU count overflows u32".into());
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

    #[test]
    fn new_valid() {
        let t = Topology::new(2, 4, 2, 2);
        assert_eq!(t.numa_nodes, 2);
        assert_eq!(t.llcs, 4);
        assert_eq!(t.cores_per_llc, 2);
        assert_eq!(t.threads_per_core, 2);
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
        };
        assert_eq!(t.to_string(), "2n4l8c2t");
    }
}
