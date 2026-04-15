//! Host CPU topology discovery for performance_mode.
//!
//! Wraps [`TestTopology`](crate::topology::TestTopology) for LLC-aware
//! vCPU pinning and host resource validation.

use anyhow::{Context, Result};

/// Resource contention error — LLC slots or CPUs unavailable.
/// Downcast via `anyhow::Error::downcast_ref::<ResourceContention>()`
/// to distinguish from fatal errors.
#[derive(Debug)]
pub struct ResourceContention {
    pub reason: String,
}

impl std::fmt::Display for ResourceContention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.reason)
    }
}

impl std::error::Error for ResourceContention {}

/// A physical LLC group on the host, identified by its cache ID.
#[derive(Debug, Clone)]
pub struct LlcGroup {
    /// CPUs sharing this LLC.
    pub cpus: Vec<usize>,
}

/// Host CPU topology: LLC groups, NUMA nodes, and online CPU set.
#[derive(Debug, Clone)]
pub struct HostTopology {
    /// LLC groups indexed by their order of discovery.
    pub llc_groups: Vec<LlcGroup>,
    /// All online CPUs.
    pub online_cpus: Vec<usize>,
    /// NUMA node ID for each online CPU, indexed by CPU ID.
    /// CPUs not in the map default to node 0.
    pub cpu_to_node: std::collections::HashMap<usize, usize>,
}

/// Pinning plan: maps each vCPU index to a host CPU, plus a dedicated
/// CPU for service threads (monitor, watchdog).
#[derive(Debug)]
pub struct PinningPlan {
    /// vcpu_index -> host_cpu
    pub assignments: Vec<(u32, usize)>,
    /// Dedicated host CPU for monitor/watchdog threads. Set when
    /// `reserve_service_cpu` is true in `compute_pinning`.
    pub service_cpu: Option<usize>,
    /// Host LLC group indices used by this plan, sorted.
    pub llc_indices: Vec<usize>,
    /// Held flock fds for resource reservation. Dropped when the plan
    /// (and the KtstrVm holding it) is dropped, releasing all locks.
    #[allow(dead_code)]
    pub(crate) locks: Vec<std::os::fd::OwnedFd>,
}

impl HostTopology {
    /// Read host topology from sysfs via [`TestTopology::from_system()`](crate::topology::TestTopology::from_system).
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
        let cpu_to_node = discover_cpu_numa_nodes(&online_cpus);
        Ok(Self {
            llc_groups,
            online_cpus,
            cpu_to_node,
        })
    }

    /// Maximum cores per LLC group on the host.
    pub fn max_cores_per_llc(&self) -> usize {
        self.llc_groups
            .iter()
            .map(|g| g.cpus.len())
            .max()
            .unwrap_or(0)
    }

    /// NUMA nodes used by the given set of host CPUs.
    pub fn numa_nodes_for_cpus(&self, cpus: &[usize]) -> Vec<usize> {
        let mut nodes: Vec<usize> = cpus
            .iter()
            .map(|c| self.cpu_to_node.get(c).copied().unwrap_or(0))
            .collect();
        nodes.sort_unstable();
        nodes.dedup();
        nodes
    }

    /// Total available host CPUs.
    pub fn total_cpus(&self) -> usize {
        self.online_cpus.len()
    }

    /// NUMA node for a host LLC group, determined by majority vote of
    /// its CPUs' NUMA assignments. Returns 0 when the map is empty
    /// (single-node systems).
    pub fn llc_numa_node(&self, llc_idx: usize) -> usize {
        let group = &self.llc_groups[llc_idx];
        let mut counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for &cpu in &group.cpus {
            let node = self.cpu_to_node.get(&cpu).copied().unwrap_or(0);
            *counts.entry(node).or_insert(0) += 1;
        }
        counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(node, _)| node)
            .unwrap_or(0)
    }

    /// Compute a pinning plan that maps virtual LLCs to physical LLC groups.
    ///
    /// Each virtual LLC's vCPUs are assigned to cores within a single physical LLC.
    /// `llc_offset` rotates the starting LLC group so concurrent VMs pin to
    /// different physical cores. When `reserve_service_cpu` is true, one
    /// additional host CPU is reserved for service threads (monitor, watchdog).
    ///
    /// When `topo.numa_nodes > 1`, virtual LLCs are grouped by guest NUMA
    /// node and each group is placed on host LLCs within the same physical
    /// NUMA node. Falls back to sequential placement when the host lacks
    /// enough NUMA-aligned LLCs.
    ///
    /// Returns an error if the host cannot satisfy the topology.
    pub fn compute_pinning(
        &self,
        topo: &super::topology::Topology,
        reserve_service_cpu: bool,
        llc_offset: usize,
    ) -> Result<PinningPlan> {
        let cores = topo.cores_per_llc;
        let threads = topo.threads_per_core;
        let llcs = topo.llcs;
        let vcpus_per_llc = cores * threads;
        let total_vcpus = llcs * vcpus_per_llc;
        let total_needed = total_vcpus as usize + if reserve_service_cpu { 1 } else { 0 };

        anyhow::ensure!(
            total_needed <= self.total_cpus(),
            "performance_mode: need {} CPUs ({} vCPUs + {} service) \
             but only {} host CPUs available",
            total_needed,
            total_vcpus,
            if reserve_service_cpu { 1 } else { 0 },
            self.total_cpus(),
        );

        let num_llcs = self.llc_groups.len();
        anyhow::ensure!(
            llcs as usize <= num_llcs,
            "performance_mode: need {} LLCs for {} virtual LLCs, \
             but host has {} LLC groups",
            llcs,
            llcs,
            num_llcs,
        );

        // Build the virtual-to-host LLC index mapping. When numa_nodes > 1,
        // try to place each guest NUMA node's LLCs on host LLCs within
        // the same physical NUMA node.
        let llc_order = self.numa_aware_llc_order(topo.numa_nodes, llcs, llc_offset);

        let mut assignments = Vec::with_capacity(total_vcpus as usize);
        let mut used_cpus = std::collections::HashSet::new();

        for llc in 0..llcs {
            let llc_idx = llc_order[llc as usize];
            let group = &self.llc_groups[llc_idx];
            let available: Vec<usize> = group
                .cpus
                .iter()
                .copied()
                .filter(|c| !used_cpus.contains(c))
                .collect();

            anyhow::ensure!(
                available.len() >= vcpus_per_llc as usize,
                "performance_mode: LLC group {} has {} available CPUs, \
                 need {} for virtual LLC {}",
                llc_idx,
                available.len(),
                vcpus_per_llc,
                llc,
            );

            for vcpu_in_llc in 0..vcpus_per_llc {
                let vcpu_id = llc * vcpus_per_llc + vcpu_in_llc;
                let host_cpu = available[vcpu_in_llc as usize];
                used_cpus.insert(host_cpu);
                assignments.push((vcpu_id, host_cpu));
            }
        }

        let service_cpu = if reserve_service_cpu {
            let cpu = self
                .online_cpus
                .iter()
                .copied()
                .find(|c| !used_cpus.contains(c));
            anyhow::ensure!(
                cpu.is_some(),
                "performance_mode: no free host CPU for service threads \
                 after assigning {} vCPUs",
                total_vcpus,
            );
            cpu
        } else {
            None
        };

        // Deduplicate LLC indices (multiple virtual LLCs may map to the
        // same host LLC at different offsets, but that's prevented by the
        // used_cpus check above — each virtual LLC consumes distinct CPUs).
        let mut llc_indices = llc_order;
        llc_indices.sort_unstable();
        llc_indices.dedup();

        Ok(PinningPlan {
            assignments,
            service_cpu,
            llc_indices,
            locks: Vec::new(),
        })
    }

    /// Build the virtual LLC to host LLC index mapping.
    ///
    /// When `numa_nodes <= 1`, falls back to sequential offset mapping
    /// (original behavior). When `numa_nodes > 1`, groups host LLCs by
    /// their physical NUMA node and assigns each guest NUMA node's LLCs
    /// to host LLCs on the same physical node. Falls back to sequential
    /// mapping when the host lacks enough NUMA-aligned LLCs.
    fn numa_aware_llc_order(&self, numa_nodes: u32, llcs: u32, llc_offset: usize) -> Vec<usize> {
        let num_host_llcs = self.llc_groups.len();

        if numa_nodes <= 1 || self.cpu_to_node.is_empty() {
            // Sequential offset mapping (original behavior).
            return (0..llcs as usize)
                .map(|i| (i + llc_offset) % num_host_llcs)
                .collect();
        }

        let llcs_per_node = llcs / numa_nodes;

        // Group host LLC indices by their physical NUMA node.
        let mut host_node_llcs: std::collections::BTreeMap<usize, Vec<usize>> =
            std::collections::BTreeMap::new();
        for idx in 0..num_host_llcs {
            let node = self.llc_numa_node(idx);
            host_node_llcs.entry(node).or_default().push(idx);
        }

        // Collect host NUMA nodes that have enough LLCs for one guest
        // NUMA node's worth.
        let eligible_nodes: Vec<(usize, &Vec<usize>)> = host_node_llcs
            .iter()
            .filter(|(_, llcs_vec)| llcs_vec.len() >= llcs_per_node as usize)
            .map(|(&node, llcs_vec)| (node, llcs_vec))
            .collect();

        // Need at least numa_nodes distinct host NUMA nodes with enough
        // LLCs each.
        if eligible_nodes.len() < numa_nodes as usize {
            // Fall back to sequential offset mapping.
            return (0..llcs as usize)
                .map(|i| (i + llc_offset) % num_host_llcs)
                .collect();
        }

        // Assign guest NUMA nodes to host NUMA nodes, rotating by
        // llc_offset to spread concurrent VMs.
        let mut order = Vec::with_capacity(llcs as usize);
        let node_offset = llc_offset / llcs_per_node.max(1) as usize;
        for guest_node in 0..numa_nodes {
            let host_idx = (guest_node as usize + node_offset) % eligible_nodes.len();
            let (_, host_llcs) = &eligible_nodes[host_idx];
            let within_offset = llc_offset % host_llcs.len();
            for i in 0..llcs_per_node as usize {
                let llc_idx = host_llcs[(i + within_offset) % host_llcs.len()];
                order.push(llc_idx);
            }
        }

        order
    }
}

/// Lock mode for LLC reservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlcLockMode {
    /// Exclusive access to the entire LLC (performance_mode tests).
    /// Blocks while any shared or exclusive holder exists.
    Exclusive,
    /// Shared access to the LLC (non-perf pinned tests).
    /// Multiple shared holders coexist; blocked by exclusive.
    Shared,
}

/// Resource lock acquisition outcome.
#[derive(Debug)]
pub enum LockOutcome {
    /// Locks acquired; PinningPlan.locks holds the fds.
    Acquired {
        llc_offset: usize,
        locks: Vec<std::os::fd::OwnedFd>,
    },
    /// Resources busy.
    Unavailable(String),
}

/// Open a lock file and attempt flock with LOCK_NB.
/// Returns the OwnedFd on success, None on EWOULDBLOCK, or
/// propagates other errors.
fn try_flock(path: &str, mode: i32) -> Result<Option<std::os::fd::OwnedFd>> {
    use std::os::fd::FromRawFd;
    let cpath = std::ffi::CString::new(path).unwrap();
    let fd = unsafe {
        libc::open(
            cpath.as_ptr(),
            libc::O_CREAT | libc::O_RDWR,
            0o666 as libc::mode_t,
        )
    };
    if fd < 0 {
        let err = std::io::Error::last_os_error();
        anyhow::bail!("open {path}: {err}");
    }
    let rc = unsafe { libc::flock(fd, mode | libc::LOCK_NB) };
    if rc == 0 {
        Ok(Some(unsafe { std::os::fd::OwnedFd::from_raw_fd(fd) }))
    } else {
        let err = std::io::Error::last_os_error();
        unsafe { libc::close(fd) };
        if err.raw_os_error() == Some(libc::EWOULDBLOCK) {
            Ok(None)
        } else {
            anyhow::bail!("flock {path}: {err}");
        }
    }
}

/// Acquire resource locks for a pinning plan (non-blocking).
///
/// **LLC locks** (`/tmp/ktstr-llc-{N}.lock`):
/// - `Exclusive`: `flock(LOCK_EX | LOCK_NB)` — sole access to the LLC.
/// - `Shared`: `flock(LOCK_SH | LOCK_NB)` — multiple holders coexist.
///
/// **CPU locks** (`/tmp/ktstr-cpu-{C}.lock`):
/// - Always `flock(LOCK_EX | LOCK_NB)` — exclusive per CPU.
/// - Skipped for `Exclusive` LLC mode (the LLC lock already provides
///   exclusivity over all CPUs in the group).
///
/// Single non-blocking attempt. Returns `LockOutcome::Unavailable`
/// immediately when any resource is busy. Callers rely on nextest
/// retry backoff for contention resolution.
pub fn acquire_resource_locks(
    plan: &PinningPlan,
    llc_indices: &[usize],
    llc_mode: LlcLockMode,
) -> Result<LockOutcome> {
    match try_acquire_all(plan, llc_indices, llc_mode) {
        Ok(locks) => Ok(LockOutcome::Acquired {
            llc_offset: llc_indices.first().copied().unwrap_or(0),
            locks,
        }),
        Err(reason) => Ok(LockOutcome::Unavailable(reason)),
    }
}

/// Try to acquire all resource locks atomically (all-or-nothing).
/// Returns the held fds on success, or an error string describing
/// which resource was busy.
fn try_acquire_all(
    plan: &PinningPlan,
    llc_indices: &[usize],
    llc_mode: LlcLockMode,
) -> std::result::Result<Vec<std::os::fd::OwnedFd>, String> {
    let flock_mode = match llc_mode {
        LlcLockMode::Exclusive => libc::LOCK_EX,
        LlcLockMode::Shared => libc::LOCK_SH,
    };
    let mut locks = Vec::new();

    // Lock LLC files.
    for &llc_idx in llc_indices {
        let path = format!("/tmp/ktstr-llc-{llc_idx}.lock");
        match try_flock(&path, flock_mode) {
            Ok(Some(fd)) => locks.push(fd),
            Ok(None) => return Err(format!("LLC {llc_idx} busy")),
            Err(e) => return Err(format!("LLC {llc_idx}: {e}")),
        }
    }

    // Per-CPU locks: skip for exclusive LLC mode (the LLC lock covers
    // all CPUs in the group).
    if llc_mode != LlcLockMode::Exclusive {
        for &(_vcpu, host_cpu) in &plan.assignments {
            let path = format!("/tmp/ktstr-cpu-{host_cpu}.lock");
            match try_flock(&path, libc::LOCK_EX) {
                Ok(Some(fd)) => locks.push(fd),
                Ok(None) => return Err(format!("CPU {host_cpu} busy")),
                Err(e) => return Err(format!("CPU {host_cpu}: {e}")),
            }
        }
        if let Some(cpu) = plan.service_cpu {
            let path = format!("/tmp/ktstr-cpu-{cpu}.lock");
            match try_flock(&path, libc::LOCK_EX) {
                Ok(Some(fd)) => locks.push(fd),
                Ok(None) => return Err(format!("service CPU {cpu} busy")),
                Err(e) => return Err(format!("service CPU {cpu}: {e}")),
            }
        }
    }

    Ok(locks)
}

/// Acquire exclusive CPU locks for a non-perf VM (non-blocking).
///
/// Tries to flock `count` consecutive CPU files starting from offset 0,
/// stepping by 1 if any CPU in the window is busy. Returns the held
/// fds on success, or `ResourceContention` when no window is available.
///
/// When `host_topo` is provided, also acquires `LOCK_SH` on the LLC lock
/// files containing the acquired CPUs. This prevents a perf VM from
/// grabbing exclusive LLC access while non-perf VMs hold CPUs in that LLC.
///
/// `total_host_cpus` bounds the search space. Single non-blocking pass;
/// callers rely on nextest retry backoff for contention resolution.
pub fn acquire_cpu_locks(
    count: usize,
    total_host_cpus: usize,
    host_topo: Option<&HostTopology>,
) -> Result<Vec<std::os::fd::OwnedFd>> {
    if count == 0 {
        return Ok(Vec::new());
    }

    let mut offset = 0;
    while offset + count <= total_host_cpus {
        match try_acquire_cpu_window(offset, count) {
            Ok(mut locks) => {
                // Acquire shared LLC locks so perf VMs cannot take
                // exclusive access to LLCs we are using.
                if let Some(topo) = host_topo {
                    let cpus: Vec<usize> = (offset..offset + count).collect();
                    match acquire_llc_shared_locks(topo, &cpus) {
                        Ok(llc_locks) => locks.extend(llc_locks),
                        Err(_) => {
                            // LLC lock busy — drop CPU locks and try next window.
                            drop(locks);
                            offset += 1;
                            continue;
                        }
                    }
                }
                return Ok(locks);
            }
            Err(_) => {
                offset += 1;
            }
        }
    }

    Err(anyhow::Error::new(ResourceContention {
        reason: format!("no {count} consecutive CPUs available"),
    }))
}

/// Acquire `LOCK_SH` on LLC lock files for the LLCs containing `cpus`.
fn acquire_llc_shared_locks(
    topo: &HostTopology,
    cpus: &[usize],
) -> std::result::Result<Vec<std::os::fd::OwnedFd>, String> {
    let mut llc_indices: Vec<usize> = Vec::new();
    for &cpu in cpus {
        for (idx, group) in topo.llc_groups.iter().enumerate() {
            if group.cpus.contains(&cpu) && !llc_indices.contains(&idx) {
                llc_indices.push(idx);
            }
        }
    }
    let mut locks = Vec::new();
    for &llc_idx in &llc_indices {
        let path = format!("/tmp/ktstr-llc-{llc_idx}.lock");
        match try_flock(&path, libc::LOCK_SH) {
            Ok(Some(fd)) => locks.push(fd),
            Ok(None) => return Err(format!("LLC {llc_idx} exclusively held")),
            Err(e) => return Err(format!("LLC {llc_idx}: {e}")),
        }
    }
    Ok(locks)
}

/// Try to flock CPUs [offset..offset+count) exclusively.
/// Returns all fds on success, or an error string on any busy CPU.
fn try_acquire_cpu_window(
    offset: usize,
    count: usize,
) -> std::result::Result<Vec<std::os::fd::OwnedFd>, String> {
    let mut locks = Vec::with_capacity(count);
    for cpu in offset..offset + count {
        let path = format!("/tmp/ktstr-cpu-{cpu}.lock");
        match try_flock(&path, libc::LOCK_EX) {
            Ok(Some(fd)) => locks.push(fd),
            Ok(None) => return Err(format!("CPU {cpu} busy")),
            Err(e) => return Err(format!("CPU {cpu}: {e}")),
        }
    }
    Ok(locks)
}

/// Discover the NUMA node for each CPU by reading
/// `/sys/devices/system/cpu/cpuN/node*` symlinks.
/// Returns a map from CPU ID to NUMA node ID. On single-node systems
/// or when sysfs is unavailable, returns an empty map (callers default
/// to node 0).
fn discover_cpu_numa_nodes(online_cpus: &[usize]) -> std::collections::HashMap<usize, usize> {
    let mut map = std::collections::HashMap::new();
    for &cpu in online_cpus {
        let cpu_dir = format!("/sys/devices/system/cpu/cpu{cpu}");
        let Ok(entries) = std::fs::read_dir(&cpu_dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if let Some(node_str) = name_str.strip_prefix("node")
                && let Ok(node_id) = node_str.parse::<usize>()
            {
                map.insert(cpu, node_id);
                break;
            }
        }
    }
    map
}

/// Bind a memory region to specific NUMA nodes using `mbind(MPOL_BIND)`.
/// `nodes` is the set of NUMA node IDs. Logs a warning on error
/// (single-node systems, missing capabilities).
pub fn mbind_to_nodes(addr: *mut u8, len: usize, nodes: &[usize]) {
    if nodes.is_empty() || len == 0 {
        return;
    }
    let max_node = nodes.iter().copied().max().unwrap_or(0);
    // nodemask is a bitmask: bit N = node N. Size in unsigned longs.
    let mask_bits = max_node + 2; // mbind maxnode is 1-indexed
    let mask_longs = mask_bits.div_ceil(usize::BITS as usize);
    let mut nodemask = vec![0usize; mask_longs];
    for &node in nodes {
        nodemask[node / (usize::BITS as usize)] |= 1 << (node % (usize::BITS as usize));
    }

    const MPOL_BIND: i32 = 2;
    let rc = unsafe {
        libc::syscall(
            libc::SYS_mbind,
            addr as *mut libc::c_void,
            len,
            MPOL_BIND,
            nodemask.as_ptr(),
            mask_bits as libc::c_ulong,
            0u32, // flags: 0 = policy applies to future allocations only, existing pages not moved
        )
    };
    if rc == 0 {
        eprintln!(
            "performance_mode: mbind {} MB to NUMA node(s) {:?}",
            len >> 20,
            nodes,
        );
    } else {
        let err = std::io::Error::last_os_error();
        eprintln!(
            "performance_mode: WARNING: mbind to node(s) {:?} failed: {err}",
            nodes,
        );
    }
}

use crate::topology::parse_cpu_list_lenient;

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
    let total = parse_cpu_list_lenient(&online).len();
    Some((procs_running, total))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vmm::topology::Topology;

    #[test]
    fn parse_cpu_list_range() {
        assert_eq!(parse_cpu_list_lenient("0-3"), vec![0, 1, 2, 3]);
    }

    #[test]
    fn parse_cpu_list_single() {
        assert_eq!(parse_cpu_list_lenient("5"), vec![5]);
    }

    #[test]
    fn parse_cpu_list_mixed() {
        assert_eq!(
            parse_cpu_list_lenient("0-2,5,7-9"),
            vec![0, 1, 2, 5, 7, 8, 9]
        );
    }

    #[test]
    fn parse_cpu_list_empty() {
        assert!(parse_cpu_list_lenient("").is_empty());
    }

    #[test]
    fn parse_cpu_list_whitespace() {
        assert_eq!(parse_cpu_list_lenient("0-3\n"), vec![0, 1, 2, 3]);
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
        let plan = topo.compute_pinning(&Topology::new(1, 1, 2, 1), false, 0);
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
        let plan = topo.compute_pinning(&Topology::new(1, 1, too_many, 1), false, 0);
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
        assert_eq!(parse_cpu_list_lenient("0,1,2,"), vec![0, 1, 2]);
    }

    #[test]
    fn parse_cpu_list_leading_comma() {
        assert_eq!(parse_cpu_list_lenient(",0,1"), vec![0, 1]);
    }

    #[test]
    fn parse_cpu_list_single_zero() {
        assert_eq!(parse_cpu_list_lenient("0"), vec![0]);
    }

    #[test]
    fn parse_cpu_list_large_ids() {
        assert_eq!(parse_cpu_list_lenient("127,255"), vec![127, 255]);
    }

    #[test]
    fn parse_cpu_list_reversed_range() {
        // "5-3" parses as start=5, end=3 — 5..=3 is empty.
        assert!(parse_cpu_list_lenient("5-3").is_empty());
    }

    #[test]
    fn parse_cpu_list_non_numeric() {
        // Garbage is silently ignored.
        assert!(parse_cpu_list_lenient("abc").is_empty());
    }

    // -- synthetic topology mapping tests --

    /// Helper: build a synthetic HostTopology with the given LLC groups.
    /// Assigns each LLC group to a NUMA node matching the group index.
    fn synthetic_topo(groups: Vec<Vec<usize>>) -> HostTopology {
        let all_cpus: Vec<usize> = groups.iter().flatten().copied().collect();
        let mut cpu_to_node = std::collections::HashMap::new();
        for (node, group) in groups.iter().enumerate() {
            for &cpu in group {
                cpu_to_node.insert(cpu, node);
            }
        }
        let llc_groups = groups.into_iter().map(|cpus| LlcGroup { cpus }).collect();
        HostTopology {
            llc_groups,
            online_cpus: all_cpus,
            cpu_to_node,
        }
    }

    #[test]
    fn mapping_single_llc() {
        // 1 LLC with 4 CPUs, request 1 LLC x 2 cores x 1 thread.
        let topo = synthetic_topo(vec![vec![0, 1, 2, 3]]);
        let plan = topo
            .compute_pinning(&Topology::new(1, 1, 2, 1), false, 0)
            .unwrap();
        assert_eq!(plan.assignments.len(), 2);
        assert_eq!(plan.assignments[0], (0, 0));
        assert_eq!(plan.assignments[1], (1, 1));
    }

    #[test]
    fn mapping_two_llcs() {
        // 2 LLCs, each with 4 CPUs. Request 2l2c1t.
        let topo = synthetic_topo(vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]);
        let plan = topo
            .compute_pinning(&Topology::new(1, 2, 2, 1), false, 0)
            .unwrap();
        assert_eq!(plan.assignments.len(), 4);
        // LLC 0 vCPUs (0,1) should map to LLC group 0 CPUs (0,1).
        assert_eq!(plan.assignments[0], (0, 0));
        assert_eq!(plan.assignments[1], (1, 1));
        // LLC 1 vCPUs (2,3) should map to LLC group 1 CPUs (4,5).
        assert_eq!(plan.assignments[2], (2, 4));
        assert_eq!(plan.assignments[3], (3, 5));
    }

    #[test]
    fn mapping_with_smt() {
        // 1 LLC with 8 CPUs, request 1l2c2t = 4 vCPUs.
        let topo = synthetic_topo(vec![vec![0, 1, 2, 3, 4, 5, 6, 7]]);
        let plan = topo
            .compute_pinning(&Topology::new(1, 1, 2, 2), false, 0)
            .unwrap();
        assert_eq!(plan.assignments.len(), 4);
        // All 4 vCPUs map to distinct CPUs within the same LLC.
        let cpus: Vec<usize> = plan.assignments.iter().map(|a| a.1).collect();
        let unique: std::collections::HashSet<usize> = cpus.iter().copied().collect();
        assert_eq!(cpus.len(), unique.len());
    }

    #[test]
    fn mapping_exact_fit() {
        // 2 LLCs with exactly 2 CPUs each, request 2l2c1t = 4 total.
        let topo = synthetic_topo(vec![vec![0, 1], vec![2, 3]]);
        let plan = topo
            .compute_pinning(&Topology::new(1, 2, 2, 1), false, 0)
            .unwrap();
        assert_eq!(plan.assignments.len(), 4);
    }

    #[test]
    fn mapping_error_too_many_vcpus() {
        // 1 LLC with 2 CPUs, request 4 vCPUs.
        let topo = synthetic_topo(vec![vec![0, 1]]);
        let err = topo
            .compute_pinning(&Topology::new(1, 1, 4, 1), false, 0)
            .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("4 vCPUs") && msg.contains("2 host CPUs"),
            "error should mention CPU counts: {msg}",
        );
    }

    #[test]
    fn mapping_error_too_many_llcs() {
        // 1 LLC, request 2 LLCs.
        let topo = synthetic_topo(vec![vec![0, 1, 2, 3]]);
        let err = topo
            .compute_pinning(&Topology::new(1, 2, 1, 1), false, 0)
            .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("2 LLCs") && msg.contains("1 LLC groups"),
            "error should mention LLC count mismatch: {msg}",
        );
    }

    #[test]
    fn mapping_error_llc_too_small() {
        // 2 LLCs: first has 4 CPUs, second has only 1. Request 2l2c1t.
        let topo = synthetic_topo(vec![vec![0, 1, 2, 3], vec![4]]);
        let err = topo
            .compute_pinning(&Topology::new(1, 2, 2, 1), false, 0)
            .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("LLC group 1") && msg.contains("1 available"),
            "error should identify the undersized LLC: {msg}",
        );
    }

    #[test]
    fn mapping_no_cross_llc_sharing() {
        // Verify vCPUs in different LLCs never share an LLC's CPUs.
        let topo = synthetic_topo(vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9, 10, 11]]);
        let plan = topo
            .compute_pinning(&Topology::new(1, 3, 2, 1), false, 0)
            .unwrap();
        // LLC 0 should only use CPUs 0-3, LLC 1 only 4-7, LLC 2 only 8-11.
        for (vcpu_id, host_cpu) in &plan.assignments {
            let llc_idx = vcpu_id / 2; // 2 vCPUs per LLC
            let llc_start = llc_idx as usize * 4;
            let llc_end = llc_start + 3;
            assert!(
                *host_cpu >= llc_start && *host_cpu <= llc_end,
                "vCPU {vcpu_id} (LLC {llc_idx}) pinned to CPU {host_cpu}, \
                 expected range {llc_start}..={llc_end}",
            );
        }
    }

    #[test]
    fn mapping_all_assignments_unique() {
        let topo = synthetic_topo(vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]);
        let plan = topo
            .compute_pinning(&Topology::new(1, 2, 4, 1), false, 0)
            .unwrap();
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
        let plan = topo
            .compute_pinning(&Topology::new(1, 1, 4, 1), false, 0)
            .unwrap();
        let vcpu_ids: Vec<u32> = plan.assignments.iter().map(|a| a.0).collect();
        assert_eq!(vcpu_ids, vec![0, 1, 2, 3]);
    }

    #[test]
    fn mapping_single_vcpu() {
        let topo = synthetic_topo(vec![vec![42]]);
        let plan = topo
            .compute_pinning(&Topology::new(1, 1, 1, 1), false, 0)
            .unwrap();
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
        let plan = topo
            .compute_pinning(&Topology::new(1, 2, 2, 1), false, 0)
            .unwrap();
        // LLC 0 vCPUs should be in LLC group 0.
        for (vcpu_id, host_cpu) in &plan.assignments {
            let llc_idx = vcpu_id / 2;
            let group = &topo.llc_groups[llc_idx as usize];
            assert!(
                group.cpus.contains(host_cpu),
                "vCPU {} mapped to CPU {} which is not in LLC group {}",
                vcpu_id,
                host_cpu,
                llc_idx,
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

    // -- service CPU reservation tests --

    #[test]
    fn reserve_service_cpu_picks_unpinned() {
        // 4 CPUs in one LLC, request 2 vCPUs + service CPU.
        let topo = synthetic_topo(vec![vec![0, 1, 2, 3]]);
        let plan = topo
            .compute_pinning(&Topology::new(1, 1, 2, 1), true, 0)
            .unwrap();
        assert_eq!(plan.assignments.len(), 2);
        let service = plan.service_cpu.expect("service_cpu should be set");
        // Service CPU must not overlap with any vCPU assignment.
        let vcpu_cpus: std::collections::HashSet<usize> =
            plan.assignments.iter().map(|a| a.1).collect();
        assert!(
            !vcpu_cpus.contains(&service),
            "service CPU {service} must not be assigned to a vCPU",
        );
    }

    #[test]
    fn reserve_service_cpu_false_returns_none() {
        let topo = synthetic_topo(vec![vec![0, 1, 2, 3]]);
        let plan = topo
            .compute_pinning(&Topology::new(1, 1, 2, 1), false, 0)
            .unwrap();
        assert!(plan.service_cpu.is_none());
    }

    #[test]
    fn reserve_service_cpu_exact_fit() {
        // 3 CPUs total, request 2 vCPUs + 1 service = exact fit.
        let topo = synthetic_topo(vec![vec![0, 1, 2]]);
        let plan = topo
            .compute_pinning(&Topology::new(1, 1, 2, 1), true, 0)
            .unwrap();
        assert!(plan.service_cpu.is_some());
    }

    #[test]
    fn reserve_service_cpu_insufficient_fails() {
        // 2 CPUs, request 2 vCPUs + 1 service = 3 needed, only 2 available.
        let topo = synthetic_topo(vec![vec![0, 1]]);
        let err = topo
            .compute_pinning(&Topology::new(1, 1, 2, 1), true, 0)
            .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("3 CPUs") && msg.contains("2 host CPUs"),
            "error should mention CPU shortage: {msg}",
        );
    }

    #[test]
    fn reserve_service_cpu_multi_llc() {
        // 2 LLCs with 3 CPUs each, request 2l2c1t + service = 5 CPUs needed.
        let topo = synthetic_topo(vec![vec![0, 1, 2], vec![3, 4, 5]]);
        let plan = topo
            .compute_pinning(&Topology::new(1, 2, 2, 1), true, 0)
            .unwrap();
        let service = plan.service_cpu.unwrap();
        let vcpu_cpus: std::collections::HashSet<usize> =
            plan.assignments.iter().map(|a| a.1).collect();
        assert!(!vcpu_cpus.contains(&service));
    }

    // -- NUMA node discovery tests --

    #[test]
    fn sysfs_cpu_to_node_populated() {
        let topo = HostTopology::from_sysfs().unwrap();
        // On any Linux host, at least some CPUs should have NUMA info.
        // On single-node systems the map may map everything to node 0.
        if !topo.cpu_to_node.is_empty() {
            for (&cpu, &node) in &topo.cpu_to_node {
                assert!(
                    topo.online_cpus.contains(&cpu),
                    "NUMA mapping for CPU {cpu} but not in online set",
                );
                // NUMA node IDs are typically small (0-N).
                assert!(node < 1024, "unexpected NUMA node ID {node} for CPU {cpu}");
            }
        }
    }

    #[test]
    fn numa_nodes_for_cpus_synthetic() {
        let topo = synthetic_topo(vec![vec![0, 1], vec![2, 3]]);
        assert_eq!(topo.numa_nodes_for_cpus(&[0, 1]), vec![0]);
        assert_eq!(topo.numa_nodes_for_cpus(&[2, 3]), vec![1]);
        let mut nodes = topo.numa_nodes_for_cpus(&[0, 2]);
        nodes.sort();
        assert_eq!(nodes, vec![0, 1]);
    }

    #[test]
    fn numa_nodes_for_cpus_empty() {
        let topo = synthetic_topo(vec![vec![0, 1]]);
        assert!(topo.numa_nodes_for_cpus(&[]).is_empty());
    }

    #[test]
    fn max_cores_per_llc_synthetic() {
        let topo = synthetic_topo(vec![vec![0, 1, 2, 3], vec![4, 5]]);
        assert_eq!(topo.max_cores_per_llc(), 4);
    }

    #[test]
    fn max_cores_per_llc_uniform() {
        let topo = synthetic_topo(vec![vec![0, 1, 2], vec![3, 4, 5]]);
        assert_eq!(topo.max_cores_per_llc(), 3);
    }

    #[test]
    fn mbind_to_nodes_empty_is_noop() {
        // Should not panic.
        mbind_to_nodes(std::ptr::null_mut(), 0, &[]);
        mbind_to_nodes(std::ptr::null_mut(), 4096, &[]);
    }

    // -- NUMA-aware pinning tests --

    /// Build a synthetic topology with explicit NUMA node assignment.
    /// `groups` is a list of (numa_node, cpus) pairs.
    fn synthetic_topo_numa(groups: Vec<(usize, Vec<usize>)>) -> HostTopology {
        let all_cpus: Vec<usize> = groups
            .iter()
            .flat_map(|(_, cpus)| cpus.iter().copied())
            .collect();
        let mut cpu_to_node = std::collections::HashMap::new();
        for (node, cpus) in &groups {
            for &cpu in cpus {
                cpu_to_node.insert(cpu, *node);
            }
        }
        let llc_groups = groups
            .into_iter()
            .map(|(_, cpus)| LlcGroup { cpus })
            .collect();
        HostTopology {
            llc_groups,
            online_cpus: all_cpus,
            cpu_to_node,
        }
    }

    #[test]
    fn llc_numa_node_synthetic() {
        // 4 LLCs: 0,1 on node 0; 2,3 on node 1.
        let topo = synthetic_topo_numa(vec![
            (0, vec![0, 1]),
            (0, vec![2, 3]),
            (1, vec![4, 5]),
            (1, vec![6, 7]),
        ]);
        assert_eq!(topo.llc_numa_node(0), 0);
        assert_eq!(topo.llc_numa_node(1), 0);
        assert_eq!(topo.llc_numa_node(2), 1);
        assert_eq!(topo.llc_numa_node(3), 1);
    }

    #[test]
    fn numa_pinning_two_nodes() {
        // Host: 4 LLCs, 2 per NUMA node. LLCs 0,1 on node 0; LLCs 2,3 on node 1.
        // Guest: 2 NUMA nodes, 4 LLCs (2 per node), 2 cores each.
        let topo = synthetic_topo_numa(vec![
            (0, vec![0, 1, 2, 3]),
            (0, vec![4, 5, 6, 7]),
            (1, vec![8, 9, 10, 11]),
            (1, vec![12, 13, 14, 15]),
        ]);
        let plan = topo
            .compute_pinning(&Topology::new(2, 4, 2, 1), false, 0)
            .unwrap();
        assert_eq!(plan.assignments.len(), 8);

        // Guest NUMA node 0 (vLLCs 0,1) should map to host LLCs on the
        // same physical NUMA node.
        let node_0_cpus: Vec<usize> = plan
            .assignments
            .iter()
            .filter(|(vcpu, _)| *vcpu < 4) // vLLC 0,1 = vCPUs 0-3
            .map(|(_, cpu)| *cpu)
            .collect();
        let node_0_host_nodes = topo.numa_nodes_for_cpus(&node_0_cpus);
        assert_eq!(
            node_0_host_nodes.len(),
            1,
            "guest NUMA 0 LLCs should all be on one host NUMA node, got {:?}",
            node_0_host_nodes,
        );

        // Guest NUMA node 1 (vLLCs 2,3) should map to host LLCs on the
        // same physical NUMA node.
        let node_1_cpus: Vec<usize> = plan
            .assignments
            .iter()
            .filter(|(vcpu, _)| *vcpu >= 4) // vLLC 2,3 = vCPUs 4-7
            .map(|(_, cpu)| *cpu)
            .collect();
        let node_1_host_nodes = topo.numa_nodes_for_cpus(&node_1_cpus);
        assert_eq!(
            node_1_host_nodes.len(),
            1,
            "guest NUMA 1 LLCs should all be on one host NUMA node, got {:?}",
            node_1_host_nodes,
        );

        // The two guest NUMA nodes should map to different host NUMA nodes.
        assert_ne!(
            node_0_host_nodes[0], node_1_host_nodes[0],
            "guest NUMA nodes should map to different host NUMA nodes",
        );
    }

    #[test]
    fn numa_pinning_fallback_insufficient_nodes() {
        // Host: 4 LLCs all on NUMA node 0. Guest wants 2 NUMA nodes.
        // Should fall back to sequential mapping.
        let topo = synthetic_topo_numa(vec![
            (0, vec![0, 1]),
            (0, vec![2, 3]),
            (0, vec![4, 5]),
            (0, vec![6, 7]),
        ]);
        let plan = topo
            .compute_pinning(&Topology::new(2, 4, 2, 1), false, 0)
            .unwrap();
        assert_eq!(plan.assignments.len(), 8);
        // All assignments should be valid (sequential fallback).
        let cpus: Vec<usize> = plan.assignments.iter().map(|a| a.1).collect();
        let unique: std::collections::HashSet<usize> = cpus.iter().copied().collect();
        assert_eq!(cpus.len(), unique.len());
    }

    #[test]
    fn numa_pinning_single_node_unchanged() {
        // numa_nodes=1 should behave identically to the original sequential
        // mapping regardless of host NUMA layout.
        let topo = synthetic_topo_numa(vec![(0, vec![0, 1, 2, 3]), (1, vec![4, 5, 6, 7])]);
        let plan = topo
            .compute_pinning(&Topology::new(1, 2, 2, 1), false, 0)
            .unwrap();
        assert_eq!(plan.assignments.len(), 4);
        // Sequential: vLLC 0 -> host LLC 0, vLLC 1 -> host LLC 1.
        assert_eq!(plan.assignments[0], (0, 0));
        assert_eq!(plan.assignments[1], (1, 1));
        assert_eq!(plan.assignments[2], (2, 4));
        assert_eq!(plan.assignments[3], (3, 5));
    }

    #[test]
    fn numa_pinning_three_nodes() {
        // Host: 6 LLCs, 2 per NUMA node (nodes 0,1,2).
        // Guest: 3 NUMA nodes, 6 LLCs.
        let topo = synthetic_topo_numa(vec![
            (0, vec![0, 1]),
            (0, vec![2, 3]),
            (1, vec![4, 5]),
            (1, vec![6, 7]),
            (2, vec![8, 9]),
            (2, vec![10, 11]),
        ]);
        let plan = topo
            .compute_pinning(&Topology::new(3, 6, 1, 1), false, 0)
            .unwrap();
        assert_eq!(plan.assignments.len(), 6);

        // Each guest NUMA node's vCPUs should be on one host NUMA node.
        for guest_node in 0..3u32 {
            let start = guest_node * 2;
            let end = start + 2;
            let cpus: Vec<usize> = plan
                .assignments
                .iter()
                .filter(|(vcpu, _)| *vcpu >= start && *vcpu < end)
                .map(|(_, cpu)| *cpu)
                .collect();
            let nodes = topo.numa_nodes_for_cpus(&cpus);
            assert_eq!(
                nodes.len(),
                1,
                "guest NUMA {} should be on one host NUMA node, got {:?}",
                guest_node,
                nodes,
            );
        }
    }

    #[test]
    fn numa_pinning_with_service_cpu() {
        // 2 NUMA nodes, 4 LLCs, request 2 NUMA nodes + service CPU.
        let topo = synthetic_topo_numa(vec![
            (0, vec![0, 1, 2, 3]),
            (0, vec![4, 5, 6, 7]),
            (1, vec![8, 9, 10, 11]),
            (1, vec![12, 13, 14, 15]),
        ]);
        let plan = topo
            .compute_pinning(&Topology::new(2, 4, 2, 1), true, 0)
            .unwrap();
        assert_eq!(plan.assignments.len(), 8);
        let service = plan.service_cpu.expect("service_cpu should be set");
        let vcpu_cpus: std::collections::HashSet<usize> =
            plan.assignments.iter().map(|a| a.1).collect();
        assert!(
            !vcpu_cpus.contains(&service),
            "service CPU {service} must not overlap vCPU assignments",
        );
    }

    #[test]
    fn llc_numa_node_empty_map() {
        // Empty cpu_to_node should default to node 0.
        let topo = HostTopology {
            llc_groups: vec![LlcGroup { cpus: vec![0, 1] }],
            online_cpus: vec![0, 1],
            cpu_to_node: std::collections::HashMap::new(),
        };
        assert_eq!(topo.llc_numa_node(0), 0);
    }
}
