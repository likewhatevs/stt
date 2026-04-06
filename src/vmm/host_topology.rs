//! Host CPU topology discovery for performance_mode.
//!
//! Wraps [`TestTopology`](crate::topology::TestTopology) for LLC-aware
//! vCPU pinning and host resource validation.

use anyhow::{Context, Result};

/// A physical LLC group on the host, identified by its cache ID.
#[derive(Debug, Clone)]
pub struct LlcGroup {
    /// CPUs sharing this LLC.
    pub cpus: Vec<usize>,
}

/// Host CPU topology: LLC groups and online CPU set.
#[derive(Debug, Clone)]
pub struct HostTopology {
    /// LLC groups indexed by their order of discovery.
    pub llc_groups: Vec<LlcGroup>,
    /// All online CPUs.
    pub online_cpus: Vec<usize>,
}

/// Pinning plan: maps each vCPU index to a host CPU.
#[derive(Debug)]
pub struct PinningPlan {
    /// vcpu_index -> host_cpu
    pub assignments: Vec<(u32, usize)>,
}

impl HostTopology {
    /// Read host topology from sysfs via [`TestTopology::from_system()`].
    pub fn from_sysfs() -> Result<Self> {
        let topo = crate::topology::TestTopology::from_system()
            .context("read host topology from sysfs")?;
        let online_cpus = topo.all_cpus().to_vec();
        let llc_groups = topo
            .llcs()
            .iter()
            .map(|llc| LlcGroup {
                cpus: llc.cpus().to_vec(),
            })
            .collect();
        Ok(Self {
            llc_groups,
            online_cpus,
        })
    }

    /// Total available host CPUs.
    pub fn total_cpus(&self) -> usize {
        self.online_cpus.len()
    }

    /// Compute a pinning plan that maps virtual sockets to physical LLC groups.
    ///
    /// Each virtual socket's vCPUs are assigned to cores within a single LLC.
    /// Returns an error if the host cannot satisfy the topology.
    pub fn compute_pinning(&self, sockets: u32, cores: u32, threads: u32) -> Result<PinningPlan> {
        let vcpus_per_socket = cores * threads;
        let total_vcpus = sockets * vcpus_per_socket;

        anyhow::ensure!(
            total_vcpus as usize <= self.total_cpus(),
            "performance_mode: need {} vCPUs but only {} host CPUs available",
            total_vcpus,
            self.total_cpus(),
        );

        anyhow::ensure!(
            sockets as usize <= self.llc_groups.len(),
            "performance_mode: need {} LLCs for {} virtual sockets, \
             but host has {} LLC groups",
            sockets,
            sockets,
            self.llc_groups.len(),
        );

        let mut assignments = Vec::with_capacity(total_vcpus as usize);
        let mut used_cpus = std::collections::HashSet::new();

        for socket in 0..sockets {
            let llc = &self.llc_groups[socket as usize];
            let available: Vec<usize> = llc
                .cpus
                .iter()
                .copied()
                .filter(|c| !used_cpus.contains(c))
                .collect();

            anyhow::ensure!(
                available.len() >= vcpus_per_socket as usize,
                "performance_mode: LLC group {} has {} available CPUs, \
                 need {} for virtual socket {}",
                socket,
                available.len(),
                vcpus_per_socket,
                socket,
            );

            for vcpu_in_socket in 0..vcpus_per_socket {
                let vcpu_id = socket * vcpus_per_socket + vcpu_in_socket;
                let host_cpu = available[vcpu_in_socket as usize];
                used_cpus.insert(host_cpu);
                assignments.push((vcpu_id, host_cpu));
            }
        }

        Ok(PinningPlan { assignments })
    }
}

/// Parse a CPU list string like "0-3,5,7-9" into individual CPU IDs.
fn parse_cpu_list(s: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    for part in s.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start, end)) = part.split_once('-') {
            if let (Ok(s), Ok(e)) = (start.parse::<usize>(), end.parse::<usize>()) {
                cpus.extend(s..=e);
            }
        } else if let Ok(cpu) = part.parse::<usize>() {
            cpus.push(cpu);
        }
    }
    cpus
}

/// Number of free 2MB hugepages on the host.
pub fn hugepages_free() -> u64 {
    std::fs::read_to_string("/sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

/// Estimate the number of 2MB hugepages needed for a given memory size in MB.
pub fn hugepages_needed(memory_mb: u32) -> u64 {
    // 2MB per hugepage.
    (memory_mb as u64).div_ceil(2)
}

/// Estimate current host CPU load by checking /proc/stat.
/// Returns (busy_cpus, total_cpus) as a rough estimate.
pub fn host_load_estimate() -> Option<(usize, usize)> {
    // Count processes in R state from /proc/stat.
    let stat = std::fs::read_to_string("/proc/stat").ok()?;
    let procs_running = stat
        .lines()
        .find(|l| l.starts_with("procs_running "))?
        .split_whitespace()
        .nth(1)?
        .parse::<usize>()
        .ok()?;
    let online = std::fs::read_to_string("/sys/devices/system/cpu/online").ok()?;
    let total = parse_cpu_list(&online).len();
    Some((procs_running, total))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cpu_list_range() {
        assert_eq!(parse_cpu_list("0-3"), vec![0, 1, 2, 3]);
    }

    #[test]
    fn parse_cpu_list_single() {
        assert_eq!(parse_cpu_list("5"), vec![5]);
    }

    #[test]
    fn parse_cpu_list_mixed() {
        assert_eq!(parse_cpu_list("0-2,5,7-9"), vec![0, 1, 2, 5, 7, 8, 9]);
    }

    #[test]
    fn parse_cpu_list_empty() {
        assert!(parse_cpu_list("").is_empty());
    }

    #[test]
    fn parse_cpu_list_whitespace() {
        assert_eq!(parse_cpu_list("0-3\n"), vec![0, 1, 2, 3]);
    }

    #[test]
    fn host_topology_from_sysfs() {
        let topo = HostTopology::from_sysfs();
        assert!(topo.is_ok(), "should read host topology: {:?}", topo.err());
        let topo = topo.unwrap();
        assert!(!topo.online_cpus.is_empty());
        assert!(!topo.llc_groups.is_empty());
    }

    #[test]
    fn pinning_plan_simple() {
        let topo = HostTopology::from_sysfs().unwrap();
        if topo.total_cpus() < 2 {
            return; // skip on single-CPU hosts
        }
        let plan = topo.compute_pinning(1, 2, 1);
        assert!(plan.is_ok(), "pinning should succeed: {:?}", plan.err());
        let plan = plan.unwrap();
        assert_eq!(plan.assignments.len(), 2);
        // All assigned CPUs should be distinct.
        let cpus: Vec<usize> = plan.assignments.iter().map(|a| a.1).collect();
        let unique: std::collections::HashSet<usize> = cpus.iter().copied().collect();
        assert_eq!(cpus.len(), unique.len());
    }

    #[test]
    fn pinning_plan_oversubscribed() {
        let topo = HostTopology::from_sysfs().unwrap();
        let too_many = topo.total_cpus() as u32 + 1;
        let plan = topo.compute_pinning(1, too_many, 1);
        assert!(plan.is_err());
    }

    #[test]
    fn hugepages_needed_values() {
        assert_eq!(hugepages_needed(2), 1);
        assert_eq!(hugepages_needed(4), 2);
        assert_eq!(hugepages_needed(2048), 1024);
        assert_eq!(hugepages_needed(3), 2);
    }

    #[test]
    fn hugepages_free_runs() {
        // Should not panic. Returns 0 if hugepages not configured.
        let _ = hugepages_free();
    }

    #[test]
    fn host_load_estimate_runs() {
        let result = host_load_estimate();
        // Should succeed on any Linux host.
        assert!(result.is_some());
        let (running, total) = result.unwrap();
        assert!(total > 0);
        // At least this test is running.
        assert!(running >= 1);
    }

    // -- parse_cpu_list edge cases --

    #[test]
    fn parse_cpu_list_trailing_comma() {
        assert_eq!(parse_cpu_list("0,1,2,"), vec![0, 1, 2]);
    }

    #[test]
    fn parse_cpu_list_leading_comma() {
        assert_eq!(parse_cpu_list(",0,1"), vec![0, 1]);
    }

    #[test]
    fn parse_cpu_list_single_zero() {
        assert_eq!(parse_cpu_list("0"), vec![0]);
    }

    #[test]
    fn parse_cpu_list_large_ids() {
        assert_eq!(parse_cpu_list("127,255"), vec![127, 255]);
    }

    #[test]
    fn parse_cpu_list_reversed_range() {
        // "5-3" parses as start=5, end=3 — 5..=3 is empty.
        assert!(parse_cpu_list("5-3").is_empty());
    }

    #[test]
    fn parse_cpu_list_non_numeric() {
        // Garbage is silently ignored.
        assert!(parse_cpu_list("abc").is_empty());
    }

    // -- synthetic topology mapping tests --

    /// Helper: build a synthetic HostTopology with the given LLC groups.
    fn synthetic_topo(groups: Vec<Vec<usize>>) -> HostTopology {
        let all_cpus: Vec<usize> = groups.iter().flatten().copied().collect();
        let llc_groups = groups.into_iter().map(|cpus| LlcGroup { cpus }).collect();
        HostTopology {
            llc_groups,
            online_cpus: all_cpus,
        }
    }

    #[test]
    fn mapping_single_socket_single_llc() {
        // 1 LLC with 4 CPUs, request 1 socket x 2 cores x 1 thread.
        let topo = synthetic_topo(vec![vec![0, 1, 2, 3]]);
        let plan = topo.compute_pinning(1, 2, 1).unwrap();
        assert_eq!(plan.assignments.len(), 2);
        assert_eq!(plan.assignments[0], (0, 0));
        assert_eq!(plan.assignments[1], (1, 1));
    }

    #[test]
    fn mapping_two_sockets_two_llcs() {
        // 2 LLCs, each with 4 CPUs. Request 2s2c1t.
        let topo = synthetic_topo(vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]);
        let plan = topo.compute_pinning(2, 2, 1).unwrap();
        assert_eq!(plan.assignments.len(), 4);
        // Socket 0 vCPUs (0,1) should map to LLC 0 CPUs (0,1).
        assert_eq!(plan.assignments[0], (0, 0));
        assert_eq!(plan.assignments[1], (1, 1));
        // Socket 1 vCPUs (2,3) should map to LLC 1 CPUs (4,5).
        assert_eq!(plan.assignments[2], (2, 4));
        assert_eq!(plan.assignments[3], (3, 5));
    }

    #[test]
    fn mapping_with_smt() {
        // 1 LLC with 8 CPUs, request 1s2c2t = 4 vCPUs.
        let topo = synthetic_topo(vec![vec![0, 1, 2, 3, 4, 5, 6, 7]]);
        let plan = topo.compute_pinning(1, 2, 2).unwrap();
        assert_eq!(plan.assignments.len(), 4);
        // All 4 vCPUs map to distinct CPUs within the same LLC.
        let cpus: Vec<usize> = plan.assignments.iter().map(|a| a.1).collect();
        let unique: std::collections::HashSet<usize> = cpus.iter().copied().collect();
        assert_eq!(cpus.len(), unique.len());
    }

    #[test]
    fn mapping_exact_fit() {
        // 2 LLCs with exactly 2 CPUs each, request 2s2c1t = 4 total.
        let topo = synthetic_topo(vec![vec![0, 1], vec![2, 3]]);
        let plan = topo.compute_pinning(2, 2, 1).unwrap();
        assert_eq!(plan.assignments.len(), 4);
    }

    #[test]
    fn mapping_error_too_many_vcpus() {
        // 1 LLC with 2 CPUs, request 4 vCPUs.
        let topo = synthetic_topo(vec![vec![0, 1]]);
        let err = topo.compute_pinning(1, 4, 1).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("4 vCPUs") && msg.contains("2 host CPUs"),
            "error should mention CPU counts: {msg}",
        );
    }

    #[test]
    fn mapping_error_too_many_sockets() {
        // 1 LLC, request 2 sockets.
        let topo = synthetic_topo(vec![vec![0, 1, 2, 3]]);
        let err = topo.compute_pinning(2, 1, 1).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("2 LLCs") && msg.contains("1 LLC groups"),
            "error should mention LLC count mismatch: {msg}",
        );
    }

    #[test]
    fn mapping_error_llc_too_small() {
        // 2 LLCs: first has 4 CPUs, second has only 1. Request 2s2c1t.
        let topo = synthetic_topo(vec![vec![0, 1, 2, 3], vec![4]]);
        let err = topo.compute_pinning(2, 2, 1).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("LLC group 1") && msg.contains("1 available"),
            "error should identify the undersized LLC: {msg}",
        );
    }

    #[test]
    fn mapping_no_cross_llc_sharing() {
        // Verify vCPUs in different sockets never share an LLC's CPUs.
        let topo = synthetic_topo(vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9, 10, 11]]);
        let plan = topo.compute_pinning(3, 2, 1).unwrap();
        // Socket 0 should only use CPUs 0-3, socket 1 only 4-7, socket 2 only 8-11.
        for (vcpu_id, host_cpu) in &plan.assignments {
            let socket = vcpu_id / 2; // 2 vCPUs per socket
            let llc_start = socket as usize * 4;
            let llc_end = llc_start + 3;
            assert!(
                *host_cpu >= llc_start && *host_cpu <= llc_end,
                "vCPU {vcpu_id} (socket {socket}) pinned to CPU {host_cpu}, \
                 expected range {llc_start}..={llc_end}",
            );
        }
    }

    #[test]
    fn mapping_all_assignments_unique() {
        let topo = synthetic_topo(vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]);
        let plan = topo.compute_pinning(2, 4, 1).unwrap();
        let cpus: Vec<usize> = plan.assignments.iter().map(|a| a.1).collect();
        let unique: std::collections::HashSet<usize> = cpus.iter().copied().collect();
        assert_eq!(
            cpus.len(),
            unique.len(),
            "all host CPU assignments must be unique: {:?}",
            cpus,
        );
    }

    #[test]
    fn mapping_vcpu_ids_sequential() {
        let topo = synthetic_topo(vec![vec![0, 1, 2, 3]]);
        let plan = topo.compute_pinning(1, 4, 1).unwrap();
        let vcpu_ids: Vec<u32> = plan.assignments.iter().map(|a| a.0).collect();
        assert_eq!(vcpu_ids, vec![0, 1, 2, 3]);
    }

    #[test]
    fn mapping_single_vcpu() {
        let topo = synthetic_topo(vec![vec![42]]);
        let plan = topo.compute_pinning(1, 1, 1).unwrap();
        assert_eq!(plan.assignments.len(), 1);
        assert_eq!(plan.assignments[0], (0, 42));
    }

    // -- sysfs-based tests with real host topology --

    #[test]
    fn sysfs_llc_groups_cover_all_cpus() {
        let topo = HostTopology::from_sysfs().unwrap();
        let llc_cpus: Vec<usize> = topo
            .llc_groups
            .iter()
            .flat_map(|g| g.cpus.iter().copied())
            .collect();
        for cpu in &topo.online_cpus {
            assert!(
                llc_cpus.contains(cpu),
                "CPU {} is online but not in any LLC group",
                cpu,
            );
        }
    }

    #[test]
    fn sysfs_total_cpus_matches_online() {
        let topo = HostTopology::from_sysfs().unwrap();
        assert_eq!(topo.total_cpus(), topo.online_cpus.len());
    }

    #[test]
    fn sysfs_llc_groups_nonempty() {
        let topo = HostTopology::from_sysfs().unwrap();
        for (i, group) in topo.llc_groups.iter().enumerate() {
            assert!(
                !group.cpus.is_empty(),
                "LLC group {} should have at least one CPU",
                i,
            );
        }
    }

    #[test]
    fn sysfs_pinning_respects_llc_boundaries() {
        let topo = HostTopology::from_sysfs().unwrap();
        if topo.llc_groups.len() < 2 || topo.total_cpus() < 4 {
            return; // need at least 2 LLCs with 2+ CPUs each
        }
        let min_llc_size = topo.llc_groups.iter().map(|g| g.cpus.len()).min().unwrap();
        if min_llc_size < 2 {
            return;
        }
        let plan = topo.compute_pinning(2, 2, 1).unwrap();
        // Socket 0 vCPUs should be in LLC group 0.
        for (vcpu_id, host_cpu) in &plan.assignments {
            let socket = vcpu_id / 2;
            let llc = &topo.llc_groups[socket as usize];
            assert!(
                llc.cpus.contains(host_cpu),
                "vCPU {} mapped to CPU {} which is not in LLC group {}",
                vcpu_id,
                host_cpu,
                socket,
            );
        }
    }

    // -- hugepages_needed edge cases --

    #[test]
    fn hugepages_needed_boundary() {
        assert_eq!(hugepages_needed(1), 1); // 1 MB -> ceil(1/2) = 1
        assert_eq!(hugepages_needed(0), 0);
    }

    #[test]
    fn hugepages_needed_exact_multiple() {
        assert_eq!(hugepages_needed(1024), 512);
    }
}
