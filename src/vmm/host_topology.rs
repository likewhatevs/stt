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
    #[allow(dead_code)] // RAII: flock fds released on Drop, not read after construction.
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
        let cpu_to_node: std::collections::HashMap<usize, usize> = topo
            .llcs()
            .iter()
            .flat_map(|llc| llc.cpus().iter().map(|&cpu| (cpu, llc.numa_node())))
            .collect();
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
    /// Falls back to sequential offset mapping when any of these hold:
    /// `numa_nodes == 0` (avoids divide-by-zero), `numa_nodes == 1`
    /// (no NUMA-awareness needed), `cpu_to_node` is empty (no NUMA
    /// map available), `llcs < numa_nodes` (base-per-node would be 0
    /// and leave guest nodes empty), or the host lacks enough
    /// NUMA-aligned LLCs.
    ///
    /// Otherwise, distributes `llcs` across `numa_nodes` guest nodes:
    /// the first `llcs % numa_nodes` guest nodes receive
    /// `base + 1 = ceil(llcs / numa_nodes)` LLCs each; the rest
    /// receive `base = floor(llcs / numa_nodes)` LLCs. This preserves
    /// the remainder that floor-only division would silently drop
    /// (e.g. `llcs=5, numa_nodes=2` yields counts 3+2 = 5).
    /// Eligibility requires each host NUMA node to supply at least
    /// `ceil(llcs / numa_nodes)` (the max any single guest node will
    /// claim) — stricter than the prior floor-based check, so the
    /// "+1" guest nodes always land on a node with capacity.
    fn numa_aware_llc_order(&self, numa_nodes: u32, llcs: u32, llc_offset: usize) -> Vec<usize> {
        let num_host_llcs = self.llc_groups.len();

        // Sequential fallback used by the degenerate cases below.
        let sequential_fallback = || -> Vec<usize> {
            (0..llcs as usize)
                .map(|i| (i + llc_offset) % num_host_llcs)
                .collect()
        };

        // Defensive: zero NUMA nodes would divide-by-zero below. Also
        // handles the single-node case (no NUMA-awareness needed) and
        // the "cpu_to_node map unavailable" case.
        if numa_nodes == 0 || numa_nodes == 1 || self.cpu_to_node.is_empty() {
            return sequential_fallback();
        }

        // If the guest has fewer LLCs than NUMA nodes, a per-node base
        // of 0 would leave some guest nodes empty. Fall back rather
        // than silently dropping those nodes' LLCs.
        if llcs < numa_nodes {
            return sequential_fallback();
        }

        // Distribute LLCs across guest NUMA nodes. Integer division
        // alone drops the remainder (e.g. llcs=5, numa_nodes=2 gave
        // 2 per node = 4 LLCs assigned, 5th dropped). Fix: the first
        // `remainder` nodes get `base + 1`, the rest get `base`.
        let base_per_node = (llcs / numa_nodes) as usize;
        let remainder = (llcs % numa_nodes) as usize;
        // Ceiling-per-node — the largest count any single guest node
        // will claim. Host NUMA nodes must supply at least this many
        // to remain eligible.
        let max_per_node = base_per_node + if remainder > 0 { 1 } else { 0 };

        // Group host LLC indices by their physical NUMA node.
        let mut host_node_llcs: std::collections::BTreeMap<usize, Vec<usize>> =
            std::collections::BTreeMap::new();
        for idx in 0..num_host_llcs {
            let node = self.llc_numa_node(idx);
            host_node_llcs.entry(node).or_default().push(idx);
        }

        // Collect host NUMA nodes that can supply the ceiling (max)
        // per-node count — so any guest node can land there regardless
        // of whether it's one of the `remainder` "+1" nodes.
        let eligible_nodes: Vec<(usize, &Vec<usize>)> = host_node_llcs
            .iter()
            .filter(|(_, llcs_vec)| llcs_vec.len() >= max_per_node)
            .map(|(&node, llcs_vec)| (node, llcs_vec))
            .collect();

        // Need at least numa_nodes distinct host NUMA nodes with enough
        // LLCs each.
        if eligible_nodes.len() < numa_nodes as usize {
            return sequential_fallback();
        }

        // Assign guest NUMA nodes to host NUMA nodes, rotating by
        // llc_offset to spread concurrent VMs.
        let mut order = Vec::with_capacity(llcs as usize);
        let node_offset = llc_offset / max_per_node.max(1);
        for guest_node in 0..numa_nodes as usize {
            let host_idx = (guest_node + node_offset) % eligible_nodes.len();
            let (_, host_llcs) = &eligible_nodes[host_idx];
            let within_offset = llc_offset % host_llcs.len();
            // First `remainder` guest nodes get `base + 1` LLCs; rest
            // get `base`. Total assigned == llcs (remainder preserved).
            let count = if guest_node < remainder {
                base_per_node + 1
            } else {
                base_per_node
            };
            for i in 0..count {
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
    /// Returns unavailable when any shared or exclusive holder exists.
    Exclusive,
    /// Shared access to the LLC (non-perf pinned tests).
    /// Multiple shared holders coexist; returns unavailable when
    /// exclusive holder exists.
    #[allow(dead_code)]
    Shared,
}

/// Resource lock acquisition outcome.
#[derive(Debug)]
pub enum LockOutcome {
    /// All locks acquired successfully.
    Acquired {
        /// LLC offset consumed; read only by the locking test fixtures.
        #[allow(dead_code)]
        llc_offset: usize,
        locks: Vec<std::os::fd::OwnedFd>,
    },
    /// Resources busy. The inner string carries the diagnostic reason
    /// surfaced to test fixtures; production callers only match the
    /// variant tag.
    Unavailable(#[allow(dead_code)] String),
}

/// Requested sharing mode for [`try_flock`]. Translated to the
/// corresponding non-blocking [`rustix::fs::FlockOperation`] internally;
/// callers never see the libc-specific constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlockMode {
    /// Exclusive (`LOCK_EX`) — sole access to the lock file.
    Exclusive,
    /// Shared (`LOCK_SH`) — multiple holders can coexist.
    Shared,
}

/// Open a lock file and attempt flock with LOCK_NB.
/// Returns the OwnedFd on success, None on EWOULDBLOCK, or
/// propagates other errors. Uses rustix so file descriptor ownership
/// and errno handling are not open-coded per call.
fn try_flock(path: &str, mode: FlockMode) -> Result<Option<std::os::fd::OwnedFd>> {
    use rustix::fs::{FlockOperation, Mode, OFlags, flock, open};

    let fd = open(
        path,
        OFlags::CREATE | OFlags::RDWR,
        Mode::from_raw_mode(0o666),
    )
    .map_err(|e| anyhow::anyhow!("open {path}: {e}"))?;
    let op = match mode {
        FlockMode::Exclusive => FlockOperation::NonBlockingLockExclusive,
        FlockMode::Shared => FlockOperation::NonBlockingLockShared,
    };
    match flock(&fd, op) {
        Ok(()) => Ok(Some(fd)),
        Err(e) if e == rustix::io::Errno::WOULDBLOCK => Ok(None),
        Err(e) => anyhow::bail!("flock {path}: {e}"),
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

/// Try to acquire all resource locks (all-or-nothing).
/// Returns the held fds on success, or an error string describing
/// which resource was busy.
fn try_acquire_all(
    plan: &PinningPlan,
    llc_indices: &[usize],
    llc_mode: LlcLockMode,
) -> std::result::Result<Vec<std::os::fd::OwnedFd>, String> {
    let flock_mode = match llc_mode {
        LlcLockMode::Exclusive => FlockMode::Exclusive,
        LlcLockMode::Shared => FlockMode::Shared,
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
            match try_flock(&path, FlockMode::Exclusive) {
                Ok(Some(fd)) => locks.push(fd),
                Ok(None) => return Err(format!("CPU {host_cpu} busy")),
                Err(e) => return Err(format!("CPU {host_cpu}: {e}")),
            }
        }
        if let Some(cpu) = plan.service_cpu {
            let path = format!("/tmp/ktstr-cpu-{cpu}.lock");
            match try_flock(&path, FlockMode::Exclusive) {
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
        reason: format!(
            "no {count} consecutive CPUs available\n  \
             hint: pass --no-perf-mode or set KTSTR_NO_PERF_MODE=1 to run without CPU reservation"
        ),
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
        match try_flock(&path, FlockMode::Shared) {
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
        match try_flock(&path, FlockMode::Exclusive) {
            Ok(Some(fd)) => locks.push(fd),
            Ok(None) => return Err(format!("CPU {cpu} busy")),
            Err(e) => return Err(format!("CPU {cpu}: {e}")),
        }
    }
    Ok(locks)
}

/// Bind a memory region to specific NUMA nodes using `mbind(MPOL_BIND)`.
/// `nodes` is the set of NUMA node IDs. Logs a warning on error
/// (single-node systems, missing capabilities).
pub fn mbind_to_nodes(addr: *mut u8, len: usize, nodes: &[usize]) {
    if nodes.is_empty() || len == 0 {
        return;
    }
    let node_set: std::collections::BTreeSet<usize> = nodes.iter().copied().collect();
    let (nodemask, maxnode) = crate::workload::build_nodemask(&node_set);

    let rc = unsafe {
        libc::syscall(
            libc::SYS_mbind,
            addr as *mut libc::c_void,
            len,
            libc::MPOL_BIND,
            nodemask.as_ptr(),
            maxnode,
            0u32,
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

    /// Collect the distinct host NUMA node IDs the given CPUs belong
    /// to. Tests that assert "these N CPUs all live on one NUMA node"
    /// (or span two) route through this helper so the CPU → node
    /// lookup and the single-CPU default stay in one place rather
    /// than duplicating the same closure across every assertion
    /// site.
    fn numa_nodes_for_cpus(
        topo: &HostTopology,
        cpus: &[usize],
    ) -> std::collections::BTreeSet<usize> {
        cpus.iter()
            .map(|c| topo.cpu_to_node.get(c).copied().unwrap_or(0))
            .collect()
    }

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
        // `hugepages_free` returns 0 (not Err / not panic) when
        // `/sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages`
        // is absent, so this smoke test is safe to run on any host
        // regardless of hugetlbfs configuration. Only the 2 MiB
        // pool is consulted (matches the exact path the
        // implementation opens); other hugepage sizes are not
        // read here.
        let _ = hugepages_free();
    }

    #[test]
    fn host_load_estimate_runs() {
        let result = host_load_estimate();
        // `host_load_estimate` reads `/proc/stat` (scanning for
        // the `procs_running` line) and
        // `/sys/devices/system/cpu/online`. Both are mandatory on
        // any Linux kernel with CONFIG_PROC_FS + CONFIG_SYSFS, so
        // `Some(_)` is guaranteed when the test runs on a Linux
        // host.
        assert!(result.is_some());
        let (running, total) = result.unwrap();
        assert!(total > 0);
        // `running` is the `procs_running` counter from
        // `/proc/stat` — number of processes currently in state
        // `R`. This test thread itself is running at observation
        // time, so the floor is 1.
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
        // All host CPUs consumed.
        let assigned: std::collections::HashSet<usize> =
            plan.assignments.iter().map(|a| a.1).collect();
        let all_cpus: std::collections::HashSet<usize> = topo.online_cpus.iter().copied().collect();
        assert_eq!(assigned, all_cpus, "exact fit must consume all host CPUs");
        // No duplicates (unique count == total count).
        assert_eq!(
            assigned.len(),
            plan.assignments.len(),
            "all assignments must be unique",
        );
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
        let service = plan.service_cpu.expect("service_cpu should be set");
        // vCPUs consume CPUs 0,1. The only remaining CPU is 2.
        assert_eq!(service, 2, "service CPU should be the only remaining CPU");
        // Service CPU must not overlap with vCPU assignments.
        let vcpu_cpus: std::collections::HashSet<usize> =
            plan.assignments.iter().map(|a| a.1).collect();
        assert!(
            !vcpu_cpus.contains(&service),
            "service CPU {service} must not overlap vCPU assignments",
        );
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
        // Empty `nodes` slice short-circuits before the `mbind(2)`
        // syscall, so neither a null pointer nor a non-zero size
        // reaches the kernel. Guards against a regression where a
        // caller passing `&[]` would either fault on the null ptr
        // or silently mbind the "all nodes" default set.
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
        let node_0_host_nodes = numa_nodes_for_cpus(&topo, &node_0_cpus);
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
        let node_1_host_nodes = numa_nodes_for_cpus(&topo, &node_1_cpus);
        assert_eq!(
            node_1_host_nodes.len(),
            1,
            "guest NUMA 1 LLCs should all be on one host NUMA node, got {:?}",
            node_1_host_nodes,
        );

        // The two guest NUMA nodes should map to different host NUMA nodes.
        assert_ne!(
            node_0_host_nodes.iter().next(),
            node_1_host_nodes.iter().next(),
            "guest NUMA nodes should map to different host NUMA nodes",
        );
    }

    #[test]
    fn numa_aware_llc_order_uneven_llcs_preserves_remainder() {
        // With llcs=5 and numa_nodes=2, naive integer division would
        // yield llcs_per_node=2 → only 4 entries in `order`, dropping
        // the remainder LLC. The implementation distributes the
        // remainder across the first `llcs % numa_nodes` guest
        // nodes: node 0 → 3 LLCs, node 1 → 2 LLCs, total 5.
        //
        // Host: 6 LLCs, 3 per NUMA node — satisfies the ceiling
        // eligibility check (each host node must supply max_per_node=3).
        let topo = synthetic_topo_numa(vec![
            (0, vec![0, 1]),
            (0, vec![2, 3]),
            (0, vec![4, 5]),
            (1, vec![6, 7]),
            (1, vec![8, 9]),
            (1, vec![10, 11]),
        ]);
        let order = topo.numa_aware_llc_order(2, 5, 0);
        assert_eq!(
            order.len(),
            5,
            "uneven llc distribution must preserve all LLCs, got {order:?}"
        );
        // First 3 entries belong to the host's first eligible NUMA
        // node; last 2 to the second.
        let first_three_nodes: std::collections::BTreeSet<usize> = order[..3]
            .iter()
            .map(|&idx| topo.llc_numa_node(idx))
            .collect();
        let last_two_nodes: std::collections::BTreeSet<usize> = order[3..]
            .iter()
            .map(|&idx| topo.llc_numa_node(idx))
            .collect();
        assert_eq!(
            first_three_nodes.len(),
            1,
            "first 3 LLCs must share a host node"
        );
        assert_eq!(
            last_two_nodes.len(),
            1,
            "last 2 LLCs must share a host node"
        );
        assert_ne!(
            first_three_nodes, last_two_nodes,
            "two guest NUMA nodes must map to distinct host nodes"
        );
    }

    #[test]
    fn numa_aware_llc_order_zero_numa_nodes_is_safe() {
        // numa_nodes=0 would divide-by-zero on `llcs / numa_nodes`.
        // Falls back to sequential mapping instead.
        let topo = synthetic_topo_numa(vec![(0, vec![0, 1]), (0, vec![2, 3])]);
        let order = topo.numa_aware_llc_order(0, 2, 0);
        assert_eq!(
            order.len(),
            2,
            "zero-numa fallback must still produce an order"
        );
    }

    #[test]
    fn numa_aware_llc_order_fewer_llcs_than_nodes_falls_back() {
        // llcs < numa_nodes would give base_per_node = 0 and leave
        // some guest nodes empty. Falls back to sequential mapping
        // so all requested LLCs land.
        let topo = synthetic_topo_numa(vec![
            (0, vec![0, 1]),
            (0, vec![2, 3]),
            (1, vec![4, 5]),
            (1, vec![6, 7]),
        ]);
        let order = topo.numa_aware_llc_order(4, 2, 0);
        assert_eq!(
            order.len(),
            2,
            "fewer-llcs-than-nodes fallback must still produce 2 entries"
        );
    }

    #[test]
    fn numa_pinning_fallback_insufficient_nodes() {
        // Host: 4 LLCs all on NUMA node 0; guest requests 2 NUMA
        // nodes. The host cannot distribute LLCs across distinct
        // nodes (there is only one), so `compute_pinning` falls
        // back to the single-node sequential mapping rather than
        // erroring. The fallback still produces 8 unique host-CPU
        // assignments so the VM boots on the same host memory
        // throughout.
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
            let nodes = numa_nodes_for_cpus(&topo, &cpus);
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

    // -- llc_offset pinning tests --

    #[test]
    fn pinning_offset_single_llc_wraps() {
        // 1 host LLC with 4 CPUs, request 1l2c1t, offset 1.
        // (0 + 1) % 1 = 0 — wraps back to the only LLC.
        let topo = synthetic_topo(vec![vec![0, 1, 2, 3]]);
        let plan = topo
            .compute_pinning(&Topology::new(1, 1, 2, 1), false, 1)
            .unwrap();
        assert_eq!(plan.assignments.len(), 2);
        assert_eq!(plan.assignments[0], (0, 0));
        assert_eq!(plan.assignments[1], (1, 1));
    }

    #[test]
    fn pinning_offset_two_llcs_shifts() {
        // 2 host LLCs, request 2l2c1t, offset 1.
        // vLLC 0 -> (0+1)%2 = host LLC 1 (CPUs 4,5).
        // vLLC 1 -> (1+1)%2 = host LLC 0 (CPUs 0,1).
        let topo = synthetic_topo(vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]);
        let plan = topo
            .compute_pinning(&Topology::new(1, 2, 2, 1), false, 1)
            .unwrap();
        assert_eq!(plan.assignments.len(), 4);
        assert_eq!(plan.assignments[0], (0, 4));
        assert_eq!(plan.assignments[1], (1, 5));
        assert_eq!(plan.assignments[2], (2, 0));
        assert_eq!(plan.assignments[3], (3, 1));
    }

    #[test]
    fn pinning_offset_wraps_modulo() {
        // 2 host LLCs, request 2l2c1t, offset 2.
        // (0+2)%2 = 0, (1+2)%2 = 1 — same as offset 0.
        let topo = synthetic_topo(vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]);
        let plan = topo
            .compute_pinning(&Topology::new(1, 2, 2, 1), false, 2)
            .unwrap();
        assert_eq!(plan.assignments.len(), 4);
        assert_eq!(plan.assignments[0], (0, 0));
        assert_eq!(plan.assignments[1], (1, 1));
        assert_eq!(plan.assignments[2], (2, 4));
        assert_eq!(plan.assignments[3], (3, 5));
    }

    #[test]
    fn pinning_offset_three_llcs_partial() {
        // 3 host LLCs (4 CPUs each), request 2l2c1t, offset 1.
        // vLLC 0 -> (0+1)%3 = host LLC 1 (CPUs 4,5).
        // vLLC 1 -> (1+1)%3 = host LLC 2 (CPUs 8,9).
        let topo = synthetic_topo(vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9, 10, 11]]);
        let plan = topo
            .compute_pinning(&Topology::new(1, 2, 2, 1), false, 1)
            .unwrap();
        assert_eq!(plan.assignments.len(), 4);
        assert_eq!(plan.assignments[0], (0, 4));
        assert_eq!(plan.assignments[1], (1, 5));
        assert_eq!(plan.assignments[2], (2, 8));
        assert_eq!(plan.assignments[3], (3, 9));
    }

    #[test]
    fn pinning_offset_large_wraps() {
        // 3 host LLCs, request 1l2c1t, offset 5.
        // (0 + 5) % 3 = 2 — maps to host LLC 2 (CPUs 8,9).
        let topo = synthetic_topo(vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9, 10, 11]]);
        let plan = topo
            .compute_pinning(&Topology::new(1, 1, 2, 1), false, 5)
            .unwrap();
        assert_eq!(plan.assignments.len(), 2);
        assert_eq!(plan.assignments[0], (0, 8));
        assert_eq!(plan.assignments[1], (1, 9));
    }

    #[test]
    fn pinning_offset_numa_within_rotation() {
        // 4 host LLCs across 2 NUMA nodes, offset 1.
        // node_offset = 1/2 = 0 (no node rotation).
        // within_offset = 1 % 2 = 1 (rotate within each node).
        // Guest node 0 → host node 0: LLCs [1, 0].
        // Guest node 1 → host node 1: LLCs [3, 2].
        // LLC order: [1, 0, 3, 2].
        let topo = synthetic_topo_numa(vec![
            (0, vec![0, 1, 2, 3]),
            (0, vec![4, 5, 6, 7]),
            (1, vec![8, 9, 10, 11]),
            (1, vec![12, 13, 14, 15]),
        ]);
        let plan = topo
            .compute_pinning(&Topology::new(2, 4, 2, 1), false, 1)
            .unwrap();
        assert_eq!(plan.assignments.len(), 8);
        // vLLC 0 → host LLC 1 (CPUs 4,5).
        assert_eq!(plan.assignments[0], (0, 4));
        assert_eq!(plan.assignments[1], (1, 5));
        // vLLC 1 → host LLC 0 (CPUs 0,1).
        assert_eq!(plan.assignments[2], (2, 0));
        assert_eq!(plan.assignments[3], (3, 1));
        // vLLC 2 → host LLC 3 (CPUs 12,13).
        assert_eq!(plan.assignments[4], (4, 12));
        assert_eq!(plan.assignments[5], (5, 13));
        // vLLC 3 → host LLC 2 (CPUs 8,9).
        assert_eq!(plan.assignments[6], (6, 8));
        assert_eq!(plan.assignments[7], (7, 9));
    }

    #[test]
    fn pinning_offset_numa_node_rotation() {
        // 4 host LLCs across 2 NUMA nodes, offset 2.
        // node_offset = 2/2 = 1 (rotates guest→host node mapping).
        // within_offset = 2 % 2 = 0 (no within-node rotation).
        // Guest node 0 → host node 1: LLCs [2, 3].
        // Guest node 1 → host node 0: LLCs [0, 1].
        // LLC order: [2, 3, 0, 1].
        let topo = synthetic_topo_numa(vec![
            (0, vec![0, 1, 2, 3]),
            (0, vec![4, 5, 6, 7]),
            (1, vec![8, 9, 10, 11]),
            (1, vec![12, 13, 14, 15]),
        ]);
        let plan = topo
            .compute_pinning(&Topology::new(2, 4, 2, 1), false, 2)
            .unwrap();
        assert_eq!(plan.assignments.len(), 8);
        // vLLC 0 → host LLC 2 (CPUs 8,9).
        assert_eq!(plan.assignments[0], (0, 8));
        assert_eq!(plan.assignments[1], (1, 9));
        // vLLC 1 → host LLC 3 (CPUs 12,13).
        assert_eq!(plan.assignments[2], (2, 12));
        assert_eq!(plan.assignments[3], (3, 13));
        // vLLC 2 → host LLC 0 (CPUs 0,1).
        assert_eq!(plan.assignments[4], (4, 0));
        assert_eq!(plan.assignments[5], (5, 1));
        // vLLC 3 → host LLC 1 (CPUs 4,5).
        assert_eq!(plan.assignments[6], (6, 4));
        assert_eq!(plan.assignments[7], (7, 5));
    }

    #[test]
    fn pinning_offset_with_service_cpu() {
        // 2 host LLCs, offset 1, reserve_service_cpu=true.
        // LLC order: [1, 0]. vCPUs consume 4,5,0,1.
        // Service CPU: first online_cpus entry not in {0,1,4,5} → 2.
        let topo = synthetic_topo(vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]);
        let plan = topo
            .compute_pinning(&Topology::new(1, 2, 2, 1), true, 1)
            .unwrap();
        assert_eq!(plan.assignments.len(), 4);
        assert_eq!(plan.assignments[0], (0, 4));
        assert_eq!(plan.assignments[1], (1, 5));
        assert_eq!(plan.assignments[2], (2, 0));
        assert_eq!(plan.assignments[3], (3, 1));
        let service = plan.service_cpu.expect("service_cpu should be set");
        assert_eq!(service, 2);
        let vcpu_cpus: std::collections::HashSet<usize> =
            plan.assignments.iter().map(|a| a.1).collect();
        assert!(!vcpu_cpus.contains(&service));
    }

    #[test]
    fn pinning_offset_numa_combined_rotation() {
        // 4 host LLCs across 2 NUMA nodes, offset 3.
        // node_offset = 3/2 = 1 (rotates node mapping).
        // within_offset = 3 % 2 = 1 (rotates within each node).
        // Guest node 0 → host node 1: LLCs [3, 2].
        // Guest node 1 → host node 0: LLCs [1, 0].
        // LLC order: [3, 2, 1, 0].
        let topo = synthetic_topo_numa(vec![
            (0, vec![0, 1, 2, 3]),
            (0, vec![4, 5, 6, 7]),
            (1, vec![8, 9, 10, 11]),
            (1, vec![12, 13, 14, 15]),
        ]);
        let plan = topo
            .compute_pinning(&Topology::new(2, 4, 2, 1), false, 3)
            .unwrap();
        assert_eq!(plan.assignments.len(), 8);
        // vLLC 0 → host LLC 3 (CPUs 12,13).
        assert_eq!(plan.assignments[0], (0, 12));
        assert_eq!(plan.assignments[1], (1, 13));
        // vLLC 1 → host LLC 2 (CPUs 8,9).
        assert_eq!(plan.assignments[2], (2, 8));
        assert_eq!(plan.assignments[3], (3, 9));
        // vLLC 2 → host LLC 1 (CPUs 4,5).
        assert_eq!(plan.assignments[4], (4, 4));
        assert_eq!(plan.assignments[5], (5, 5));
        // vLLC 3 → host LLC 0 (CPUs 0,1).
        assert_eq!(plan.assignments[6], (6, 0));
        assert_eq!(plan.assignments[7], (7, 1));
    }

    // -- resource lock tests --

    /// Clean up a lock file. Best-effort; ignores errors.
    fn cleanup_lock(path: &str) {
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn resource_lock_exclusive_acquires() {
        let path = "/tmp/ktstr-test-flock-excl-acquires.lock";
        cleanup_lock(path);
        let fd = try_flock(path, FlockMode::Exclusive).expect("open should succeed");
        assert!(fd.is_some(), "exclusive lock on fresh file should succeed");
        cleanup_lock(path);
    }

    #[test]
    fn resource_lock_shared_acquires() {
        let path = "/tmp/ktstr-test-flock-shared-acquires.lock";
        cleanup_lock(path);
        let fd = try_flock(path, FlockMode::Shared).expect("open should succeed");
        assert!(fd.is_some(), "shared lock on fresh file should succeed");
        cleanup_lock(path);
    }

    #[test]
    fn resource_lock_exclusive_contention() {
        let path = "/tmp/ktstr-test-flock-excl-contention.lock";
        cleanup_lock(path);
        let holder = try_flock(path, FlockMode::Exclusive)
            .expect("open should succeed")
            .expect("first lock should succeed");
        let second = try_flock(path, FlockMode::Exclusive).expect("open should succeed");
        assert!(
            second.is_none(),
            "second exclusive lock while held should return None",
        );
        drop(holder);
        cleanup_lock(path);
    }

    #[test]
    fn resource_lock_shared_coexist() {
        let path = "/tmp/ktstr-test-flock-shared-coexist.lock";
        cleanup_lock(path);
        let h1 = try_flock(path, FlockMode::Shared)
            .expect("open should succeed")
            .expect("first shared lock should succeed");
        let h2 = try_flock(path, FlockMode::Shared)
            .expect("open should succeed")
            .expect("second shared lock should succeed");
        // Both held simultaneously.
        drop(h1);
        drop(h2);
        cleanup_lock(path);
    }

    #[test]
    fn resource_lock_exclusive_blocks_shared() {
        let path = "/tmp/ktstr-test-flock-excl-blocks-sh.lock";
        cleanup_lock(path);
        let holder = try_flock(path, FlockMode::Exclusive)
            .expect("open should succeed")
            .expect("exclusive lock should succeed");
        let shared = try_flock(path, FlockMode::Shared).expect("open should succeed");
        assert!(
            shared.is_none(),
            "shared lock should fail while exclusive is held",
        );
        drop(holder);
        cleanup_lock(path);
    }

    #[test]
    fn resource_lock_shared_blocks_exclusive() {
        let path = "/tmp/ktstr-test-flock-sh-blocks-excl.lock";
        cleanup_lock(path);
        let holder = try_flock(path, FlockMode::Shared)
            .expect("open should succeed")
            .expect("shared lock should succeed");
        let excl = try_flock(path, FlockMode::Exclusive).expect("open should succeed");
        assert!(
            excl.is_none(),
            "exclusive lock should fail while shared is held",
        );
        drop(holder);
        cleanup_lock(path);
    }

    #[test]
    fn resource_lock_release_on_drop() {
        let path = "/tmp/ktstr-test-flock-release-drop.lock";
        cleanup_lock(path);
        {
            let _holder = try_flock(path, FlockMode::Exclusive)
                .expect("open should succeed")
                .expect("lock should succeed");
        }
        // After drop, the lock should be available again.
        let fd = try_flock(path, FlockMode::Exclusive)
            .expect("open should succeed")
            .expect("lock should be available after drop");
        drop(fd);
        cleanup_lock(path);
    }

    #[test]
    fn resource_lock_exclusive_success() {
        // Use high LLC indices to avoid collision with real locks.
        let plan = PinningPlan {
            assignments: vec![(0, 90100), (1, 90101)],
            service_cpu: None,
            llc_indices: vec![90100],
            locks: Vec::new(),
        };
        let llc_indices = &[90100usize];
        cleanup_lock("/tmp/ktstr-llc-90100.lock");
        let outcome = acquire_resource_locks(&plan, llc_indices, LlcLockMode::Exclusive).unwrap();
        match outcome {
            LockOutcome::Acquired { llc_offset, locks } => {
                assert_eq!(llc_offset, 90100);
                // Exclusive mode: only LLC locks, no per-CPU locks.
                assert_eq!(locks.len(), 1);
            }
            LockOutcome::Unavailable(reason) => {
                panic!("expected Acquired, got Unavailable: {reason}");
            }
        }
        cleanup_lock("/tmp/ktstr-llc-90100.lock");
    }

    #[test]
    fn resource_lock_shared_includes_cpu_locks() {
        let plan = PinningPlan {
            assignments: vec![(0, 90200), (1, 90201)],
            service_cpu: None,
            llc_indices: vec![90200],
            locks: Vec::new(),
        };
        let llc_indices = &[90200usize];
        cleanup_lock("/tmp/ktstr-llc-90200.lock");
        cleanup_lock("/tmp/ktstr-cpu-90200.lock");
        cleanup_lock("/tmp/ktstr-cpu-90201.lock");

        let outcome = acquire_resource_locks(&plan, llc_indices, LlcLockMode::Shared).unwrap();
        match outcome {
            LockOutcome::Acquired { locks, .. } => {
                // Shared mode: 1 LLC lock + 2 CPU locks = 3 total.
                assert_eq!(locks.len(), 3);
            }
            LockOutcome::Unavailable(reason) => {
                panic!("expected Acquired, got Unavailable: {reason}");
            }
        }
        cleanup_lock("/tmp/ktstr-llc-90200.lock");
        cleanup_lock("/tmp/ktstr-cpu-90200.lock");
        cleanup_lock("/tmp/ktstr-cpu-90201.lock");
    }

    #[test]
    fn resource_lock_shared_with_service_cpu() {
        let plan = PinningPlan {
            assignments: vec![(0, 90300)],
            service_cpu: Some(90301),
            llc_indices: vec![90300],
            locks: Vec::new(),
        };
        let llc_indices = &[90300usize];
        cleanup_lock("/tmp/ktstr-llc-90300.lock");
        cleanup_lock("/tmp/ktstr-cpu-90300.lock");
        cleanup_lock("/tmp/ktstr-cpu-90301.lock");

        let outcome = acquire_resource_locks(&plan, llc_indices, LlcLockMode::Shared).unwrap();
        match outcome {
            LockOutcome::Acquired { locks, .. } => {
                // 1 LLC lock + 1 assignment CPU lock + 1 service CPU lock = 3.
                assert_eq!(locks.len(), 3);
            }
            LockOutcome::Unavailable(reason) => {
                panic!("expected Acquired, got Unavailable: {reason}");
            }
        }
        cleanup_lock("/tmp/ktstr-llc-90300.lock");
        cleanup_lock("/tmp/ktstr-cpu-90300.lock");
        cleanup_lock("/tmp/ktstr-cpu-90301.lock");
    }

    #[test]
    fn resource_lock_exclusive_skips_cpu_locks() {
        // Exclusive LLC mode should NOT acquire per-CPU locks.
        let plan = PinningPlan {
            assignments: vec![(0, 90400), (1, 90401)],
            service_cpu: Some(90402),
            llc_indices: vec![90400],
            locks: Vec::new(),
        };
        let llc_indices = &[90400usize];
        cleanup_lock("/tmp/ktstr-llc-90400.lock");

        let outcome = acquire_resource_locks(&plan, llc_indices, LlcLockMode::Exclusive).unwrap();
        match outcome {
            LockOutcome::Acquired { locks, .. } => {
                // Exclusive: only 1 LLC lock, no CPU locks.
                assert_eq!(locks.len(), 1);
            }
            LockOutcome::Unavailable(reason) => {
                panic!("expected Acquired, got Unavailable: {reason}");
            }
        }
        cleanup_lock("/tmp/ktstr-llc-90400.lock");
    }

    #[test]
    fn resource_lock_contention_returns_unavailable() {
        // Hold an exclusive lock, then try to acquire the same LLC.
        let plan = PinningPlan {
            assignments: vec![(0, 90500)],
            service_cpu: None,
            llc_indices: vec![90500],
            locks: Vec::new(),
        };
        let llc_indices = &[90500usize];
        cleanup_lock("/tmp/ktstr-llc-90500.lock");

        let holder = try_flock("/tmp/ktstr-llc-90500.lock", FlockMode::Exclusive)
            .unwrap()
            .unwrap();

        let outcome = acquire_resource_locks(&plan, llc_indices, LlcLockMode::Exclusive).unwrap();
        match outcome {
            LockOutcome::Unavailable(reason) => {
                assert!(
                    reason.contains("90500"),
                    "reason should identify the busy LLC: {reason}",
                );
            }
            LockOutcome::Acquired { .. } => {
                panic!("expected Unavailable while lock is held");
            }
        }
        drop(holder);
        cleanup_lock("/tmp/ktstr-llc-90500.lock");
    }

    #[test]
    fn resource_lock_all_or_nothing() {
        // Two LLC indices: hold the second one, verify the first is
        // released when the second fails (all-or-nothing semantics).
        let plan = PinningPlan {
            assignments: vec![(0, 90600), (1, 90601)],
            service_cpu: None,
            llc_indices: vec![90600, 90601],
            locks: Vec::new(),
        };
        let llc_indices = &[90600usize, 90601];
        cleanup_lock("/tmp/ktstr-llc-90600.lock");
        cleanup_lock("/tmp/ktstr-llc-90601.lock");

        let holder = try_flock("/tmp/ktstr-llc-90601.lock", FlockMode::Exclusive)
            .unwrap()
            .unwrap();

        let outcome = acquire_resource_locks(&plan, llc_indices, LlcLockMode::Exclusive).unwrap();
        assert!(
            matches!(outcome, LockOutcome::Unavailable(_)),
            "should fail when second LLC is busy",
        );

        // LLC 90600 should be released (all-or-nothing). Verify by
        // acquiring it successfully.
        let reacquire = try_flock("/tmp/ktstr-llc-90600.lock", FlockMode::Exclusive)
            .unwrap()
            .expect("LLC 90600 should be released after all-or-nothing failure");
        drop(reacquire);
        drop(holder);
        cleanup_lock("/tmp/ktstr-llc-90600.lock");
        cleanup_lock("/tmp/ktstr-llc-90601.lock");
    }

    #[test]
    fn resource_lock_shared_cpu_contention() {
        // Shared LLC mode: hold a CPU lock, verify acquire fails.
        let plan = PinningPlan {
            assignments: vec![(0, 90700)],
            service_cpu: None,
            llc_indices: vec![90700],
            locks: Vec::new(),
        };
        let llc_indices = &[90700usize];
        cleanup_lock("/tmp/ktstr-llc-90700.lock");
        cleanup_lock("/tmp/ktstr-cpu-90700.lock");

        let holder = try_flock("/tmp/ktstr-cpu-90700.lock", FlockMode::Exclusive)
            .unwrap()
            .unwrap();

        let outcome = acquire_resource_locks(&plan, llc_indices, LlcLockMode::Shared).unwrap();
        assert!(
            matches!(outcome, LockOutcome::Unavailable(_)),
            "should fail when CPU lock is held",
        );

        // LLC lock should be released (all-or-nothing).
        let reacquire = try_flock("/tmp/ktstr-llc-90700.lock", FlockMode::Shared)
            .unwrap()
            .expect("LLC 90700 should be released after CPU contention");
        drop(reacquire);
        drop(holder);
        cleanup_lock("/tmp/ktstr-llc-90700.lock");
        cleanup_lock("/tmp/ktstr-cpu-90700.lock");
    }

    #[test]
    fn resource_lock_empty_llc_indices() {
        // Empty llc_indices: LLC lock loop iterates zero times.
        // Exclusive mode skips CPU locks. Result: Acquired with
        // llc_offset 0 and empty locks vec.
        let plan = PinningPlan {
            assignments: vec![(0, 90800)],
            service_cpu: None,
            llc_indices: vec![],
            locks: Vec::new(),
        };
        let outcome = acquire_resource_locks(&plan, &[], LlcLockMode::Exclusive).unwrap();
        match outcome {
            LockOutcome::Acquired { llc_offset, locks } => {
                assert_eq!(llc_offset, 0);
                assert!(locks.is_empty());
            }
            LockOutcome::Unavailable(reason) => {
                panic!("expected Acquired, got Unavailable: {reason}");
            }
        }
    }

    #[test]
    fn resource_lock_service_cpu_contention() {
        // Shared mode: LLC and assignment CPU locks succeed, but
        // service CPU is held → Unavailable. All prior locks released.
        let plan = PinningPlan {
            assignments: vec![(0, 90900)],
            service_cpu: Some(90901),
            llc_indices: vec![90850],
            locks: Vec::new(),
        };
        let llc_indices = &[90850usize];
        cleanup_lock("/tmp/ktstr-llc-90850.lock");
        cleanup_lock("/tmp/ktstr-cpu-90900.lock");
        cleanup_lock("/tmp/ktstr-cpu-90901.lock");

        // Hold the service CPU lock.
        let holder = try_flock("/tmp/ktstr-cpu-90901.lock", FlockMode::Exclusive)
            .unwrap()
            .unwrap();

        let outcome = acquire_resource_locks(&plan, llc_indices, LlcLockMode::Shared).unwrap();
        match &outcome {
            LockOutcome::Unavailable(reason) => {
                assert!(
                    reason.contains("service CPU") && reason.contains("90901"),
                    "reason should mention service CPU 90901: {reason}",
                );
            }
            LockOutcome::Acquired { .. } => {
                panic!("expected Unavailable when service CPU is held");
            }
        }

        // All prior locks should be released (all-or-nothing).
        let reacquire_llc = try_flock("/tmp/ktstr-llc-90850.lock", FlockMode::Shared)
            .unwrap()
            .expect("LLC 90850 should be released after service CPU contention");
        let reacquire_cpu = try_flock("/tmp/ktstr-cpu-90900.lock", FlockMode::Exclusive)
            .unwrap()
            .expect("CPU 90900 should be released after service CPU contention");
        drop(reacquire_llc);
        drop(reacquire_cpu);
        drop(holder);
        cleanup_lock("/tmp/ktstr-llc-90850.lock");
        cleanup_lock("/tmp/ktstr-cpu-90900.lock");
        cleanup_lock("/tmp/ktstr-cpu-90901.lock");
    }

    #[test]
    fn cpu_lock_window_success() {
        for c in 91300..91303 {
            cleanup_lock(&format!("/tmp/ktstr-cpu-{c}.lock"));
        }
        let locks = try_acquire_cpu_window(91300, 3).unwrap();
        assert_eq!(locks.len(), 3);
        for c in 91300..91303 {
            cleanup_lock(&format!("/tmp/ktstr-cpu-{c}.lock"));
        }
    }

    #[test]
    fn cpu_lock_window_contention_all_or_nothing() {
        cleanup_lock("/tmp/ktstr-cpu-91400.lock");
        cleanup_lock("/tmp/ktstr-cpu-91401.lock");

        let holder = try_flock("/tmp/ktstr-cpu-91400.lock", FlockMode::Exclusive)
            .unwrap()
            .unwrap();

        let result = try_acquire_cpu_window(91400, 2);
        assert!(result.is_err(), "should fail when first CPU is held");

        // Hold 91401 instead — 91400 acquires then drops on failure.
        drop(holder);

        let holder2 = try_flock("/tmp/ktstr-cpu-91401.lock", FlockMode::Exclusive)
            .unwrap()
            .unwrap();
        let result2 = try_acquire_cpu_window(91400, 2);
        assert!(result2.is_err(), "should fail when second CPU is held");

        // 91400 was acquired then dropped (all-or-nothing). Verify
        // it's available.
        let reacquire = try_flock("/tmp/ktstr-cpu-91400.lock", FlockMode::Exclusive)
            .unwrap()
            .expect("CPU 91400 should be released after all-or-nothing");
        drop(reacquire);
        drop(holder2);
        cleanup_lock("/tmp/ktstr-cpu-91400.lock");
        cleanup_lock("/tmp/ktstr-cpu-91401.lock");
    }

    #[test]
    fn cpu_lock_zero_count() {
        let locks = acquire_cpu_locks(0, 4, None).unwrap();
        assert!(locks.is_empty());
    }

    #[test]
    fn cpu_lock_contention_slides_window() {
        // Hold CPU at offset 91500, verify next window succeeds
        // via try_acquire_cpu_window (unit-level sliding test).
        for c in 91500..91503 {
            cleanup_lock(&format!("/tmp/ktstr-cpu-{c}.lock"));
        }

        let holder = try_flock("/tmp/ktstr-cpu-91500.lock", FlockMode::Exclusive)
            .unwrap()
            .unwrap();

        let result = try_acquire_cpu_window(91500, 2);
        assert!(result.is_err(), "window starting at held CPU should fail");

        let locks = try_acquire_cpu_window(91501, 2).unwrap();
        assert_eq!(locks.len(), 2);

        drop(locks);
        drop(holder);
        for c in 91500..91503 {
            cleanup_lock(&format!("/tmp/ktstr-cpu-{c}.lock"));
        }
    }

    #[test]
    fn cpu_lock_acquire_success() {
        // acquire_cpu_locks starts at offset 0. Use total_host_cpus
        // large enough that a free 3-CPU window always exists even
        // when other parallel tests hold some low-index CPU locks.
        let locks = acquire_cpu_locks(3, 100, None).unwrap();
        assert_eq!(locks.len(), 3);
    }

    #[test]
    fn cpu_lock_acquire_slides_past_held() {
        // Hold CPU 0. acquire_cpu_locks(count=2, total=100) slides
        // past window [0,1] and finds a free window. Headroom of
        // 100 CPUs absorbs parallel test interference.
        cleanup_lock("/tmp/ktstr-cpu-0.lock");
        let holder = try_flock("/tmp/ktstr-cpu-0.lock", FlockMode::Exclusive)
            .unwrap()
            .unwrap();

        let locks = acquire_cpu_locks(2, 100, None).unwrap();
        assert_eq!(locks.len(), 2);

        drop(locks);
        drop(holder);
        cleanup_lock("/tmp/ktstr-cpu-0.lock");
    }

    #[test]
    fn cpu_lock_acquire_no_windows_fit() {
        // count > total_host_cpus: loop condition never satisfied,
        // returns ResourceContention without touching any files.
        let err = acquire_cpu_locks(2, 0, None).unwrap_err();
        assert!(
            err.downcast_ref::<ResourceContention>().is_some(),
            "error should be ResourceContention: {err}",
        );
    }

    #[test]
    fn cpu_lock_acquire_with_llc_shared() {
        // Place the LLC group at Vec index 92000 so the lock file
        // is /tmp/ktstr-llc-92000.lock, avoiding collision with
        // production LLC lock files at low indices.
        let mut llc_groups: Vec<LlcGroup> =
            (0..92000).map(|_| LlcGroup { cpus: Vec::new() }).collect();
        llc_groups.push(LlcGroup {
            cpus: (0..100).collect(),
        });
        let topo = HostTopology {
            llc_groups,
            online_cpus: (0..100).collect(),
            cpu_to_node: std::collections::HashMap::new(),
        };
        cleanup_lock("/tmp/ktstr-llc-92000.lock");

        let locks = acquire_cpu_locks(2, 100, Some(&topo)).unwrap();
        // 2 CPU locks + 1 shared LLC lock = 3.
        assert_eq!(locks.len(), 3);

        // The LLC lock is shared — another shared should coexist.
        let shared2 = try_flock("/tmp/ktstr-llc-92000.lock", FlockMode::Shared)
            .unwrap()
            .expect("second shared LLC should coexist");
        // Exclusive should fail while shared is held.
        let excl = try_flock("/tmp/ktstr-llc-92000.lock", FlockMode::Exclusive).unwrap();
        assert!(
            excl.is_none(),
            "exclusive LLC should fail while shared is held",
        );

        drop(shared2);
        drop(locks);
        cleanup_lock("/tmp/ktstr-llc-92000.lock");
    }

    #[test]
    fn cpu_lock_llc_shared_protection() {
        // Tests acquire_llc_shared_locks directly: verifies shared lock
        // acquired, shared coexistence, and exclusive blocking.
        let mut llc_groups: Vec<LlcGroup> =
            (0..92100).map(|_| LlcGroup { cpus: Vec::new() }).collect();
        llc_groups.push(LlcGroup {
            cpus: vec![91200, 91201],
        });
        let topo = HostTopology {
            llc_groups,
            online_cpus: vec![91200, 91201],
            cpu_to_node: std::collections::HashMap::new(),
        };
        cleanup_lock("/tmp/ktstr-llc-92100.lock");

        let cpus = vec![91200usize, 91201];
        let llc_locks = acquire_llc_shared_locks(&topo, &cpus).unwrap();
        assert_eq!(llc_locks.len(), 1);

        let shared2 = try_flock("/tmp/ktstr-llc-92100.lock", FlockMode::Shared)
            .unwrap()
            .expect("second shared LLC should coexist");
        let excl = try_flock("/tmp/ktstr-llc-92100.lock", FlockMode::Exclusive).unwrap();
        assert!(
            excl.is_none(),
            "exclusive LLC should fail while shared is held",
        );

        drop(shared2);
        drop(llc_locks);
        cleanup_lock("/tmp/ktstr-llc-92100.lock");
    }
}
