//! CPU topology abstraction.
//!
//! [`TestTopology`] reads sysfs to discover CPUs, LLCs, and NUMA nodes.
//! Provides cpuset generation methods used by [`CpusetMode`](crate::scenario::CpusetMode)
//! and [`CpusetSpec`](crate::scenario::ops::CpusetSpec).
//!
//! See the [Scenarios](https://likewhatevs.github.io/ktstr/guide/concepts/scenarios.html)
//! chapter for how topology drives cpuset partitioning.

use anyhow::{Context, Result, bail};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

/// Information about a last-level cache domain.
#[derive(Debug, Clone)]
pub struct LlcInfo {
    cpus: Vec<usize>,
    numa_node: usize,
    cache_size_kb: Option<u64>,
    /// core_id -> sorted list of CPU IDs (SMT siblings).
    cores: BTreeMap<usize, Vec<usize>>,
}

impl LlcInfo {
    /// Sorted list of CPU IDs in this LLC domain.
    pub fn cpus(&self) -> &[usize] {
        &self.cpus
    }
    /// NUMA node containing this LLC.
    pub fn numa_node(&self) -> usize {
        self.numa_node
    }
    /// LLC cache size in KiB when sysfs reported it, else `None`.
    pub fn cache_size_kb(&self) -> Option<u64> {
        self.cache_size_kb
    }
    /// Per-core sibling map: `core_id -> sorted list of CPU IDs that
    /// are SMT siblings of that core`.
    pub fn cores(&self) -> &BTreeMap<usize, Vec<usize>> {
        &self.cores
    }
    /// Number of physical cores in the LLC; falls back to the CPU
    /// count when core-group data is unavailable.
    pub fn num_cores(&self) -> usize {
        if self.cores.is_empty() {
            self.cpus.len()
        } else {
            self.cores.len()
        }
    }
}

/// Per-node memory information (total and free KiB).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeMemInfo {
    /// Total memory in KiB.
    pub total_kb: u64,
    /// Free memory in KiB.
    pub free_kb: u64,
}

impl NodeMemInfo {
    /// Used memory in KiB (`total_kb - free_kb`).
    pub fn used_kb(&self) -> u64 {
        self.total_kb.saturating_sub(self.free_kb)
    }
}

/// CPU topology abstraction for test configuration.
///
/// Provides LLC-aware CPU partitioning, cpuset generation, NUMA
/// distance queries, and per-node memory introspection. Built from
/// sysfs (`from_system()`), a VM spec (`from_spec()`), or synthetic
/// parameters (test-only).
#[derive(Debug, Clone)]
pub struct TestTopology {
    cpus: Vec<usize>,
    llcs: Vec<LlcInfo>,
    numa_nodes: BTreeSet<usize>,
    /// Flat row-major NxN distance matrix. Dimension equals
    /// `numa_nodes.len()`, rows ordered by ascending node ID.
    /// Default 10/20 when sysfs distances are unavailable.
    numa_distances: Vec<u8>,
    /// Per-node memory info, keyed by NUMA node ID.
    node_mem: BTreeMap<usize, NodeMemInfo>,
    /// NUMA nodes that have memory but no CPUs (CXL memory-only).
    memory_only_nodes: BTreeSet<usize>,
}

/// Parse a CPU list string (e.g., "0-3,5,7-9") into a sorted vec of CPU IDs.
///
/// Returns an error if any element is not a valid integer or range.
/// For lenient parsing that skips invalid entries, use
/// [`parse_cpu_list_lenient`].
pub fn parse_cpu_list(s: &str) -> Result<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in s.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((lo, hi)) = part.split_once('-') {
            let lo: usize = lo.parse()?;
            let hi: usize = hi.parse()?;
            cpus.extend(lo..=hi);
        } else {
            cpus.push(part.parse()?);
        }
    }
    cpus.sort();
    Ok(cpus)
}

/// Parse a CPU list string, silently skipping invalid entries.
///
/// Unlike [`parse_cpu_list`], this never fails — non-numeric elements
/// and reversed ranges are ignored.
pub fn parse_cpu_list_lenient(s: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    for part in s.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((lo, hi)) = part.split_once('-') {
            if let (Ok(lo), Ok(hi)) = (lo.parse::<usize>(), hi.parse::<usize>()) {
                cpus.extend(lo..=hi);
            }
        } else if let Ok(cpu) = part.parse::<usize>() {
            cpus.push(cpu);
        }
    }
    cpus
}

/// Find the sysfs index of the highest-level (last-level) cache for a CPU.
///
/// Iterates `/sys/devices/system/cpu/cpuN/cache/indexM/level` entries and
/// returns the index with the largest level value.
fn find_llc_index(cpu: usize) -> Result<usize> {
    let cache_dir = format!("/sys/devices/system/cpu/cpu{cpu}/cache");
    let mut max_level = 0usize;
    let mut llc_index = 0usize;
    for entry in fs::read_dir(&cache_dir).context("read cache dir")? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("index") {
            continue;
        }
        let level_path = entry.path().join("level");
        if let Ok(level_str) = fs::read_to_string(&level_path)
            && let Ok(level) = level_str.trim().parse::<usize>()
            && level > max_level
        {
            max_level = level;
            llc_index = name
                .strip_prefix("index")
                .unwrap_or("0")
                .parse()
                .unwrap_or(0);
        }
    }
    Ok(llc_index)
}

/// Read the LLC cache ID for a CPU from sysfs.
///
/// Prefers the `id` file when available (x86_64 always has it).
/// Falls back to the lowest CPU in the LLC's `shared_cpu_list`,
/// which is unique per LLC group. The previous fallback used the
/// cache index number, which is the same for every CPU and
/// collapsed all LLCs into one group.
fn read_llc_id(cpu: usize) -> Result<usize> {
    let llc_index = find_llc_index(cpu)?;
    let id_path = format!("/sys/devices/system/cpu/cpu{cpu}/cache/index{llc_index}/id");
    if let Ok(id_str) = fs::read_to_string(&id_path)
        && let Ok(id) = id_str.trim().parse::<usize>()
    {
        return Ok(id);
    }
    // Fallback: use the lowest CPU in shared_cpu_list as a stable
    // group identifier. Each LLC group has a unique minimum CPU.
    let shared_path =
        format!("/sys/devices/system/cpu/cpu{cpu}/cache/index{llc_index}/shared_cpu_list");
    if let Ok(shared_str) = fs::read_to_string(&shared_path) {
        let siblings = parse_cpu_list_lenient(shared_str.trim());
        if let Some(&min_cpu) = siblings.iter().min() {
            return Ok(min_cpu);
        }
    }
    Ok(0)
}

/// Read the NUMA node ID for a CPU from sysfs.
fn read_numa_node(cpu: usize) -> Result<usize> {
    let node_dir = format!("/sys/devices/system/cpu/cpu{cpu}");
    for entry in fs::read_dir(&node_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("node")
            && let Some(id_str) = name.strip_prefix("node")
            && let Ok(id) = id_str.parse::<usize>()
        {
            return Ok(id);
        }
    }
    Ok(0)
}

/// Read the LLC cache size in KB for a CPU from sysfs.
fn read_llc_cache_size(cpu: usize) -> Option<u64> {
    let llc_index = find_llc_index(cpu).ok()?;
    let size_path = format!("/sys/devices/system/cpu/cpu{cpu}/cache/index{llc_index}/size");
    let size_str = fs::read_to_string(&size_path).ok()?;
    parse_cache_size(size_str.trim())
}

/// Parse a cache size string like "32768K" or "32M" into KB.
fn parse_cache_size(s: &str) -> Option<u64> {
    let s = s.trim();
    if let Some(kb) = s.strip_suffix('K') {
        kb.parse().ok()
    } else if let Some(mb) = s.strip_suffix('M') {
        mb.parse::<u64>().ok().map(|v| v * 1024)
    } else {
        // Bare number: assume bytes, convert to KB
        s.parse::<u64>().ok().map(|v| v / 1024)
    }
}

/// Read the core_id for a CPU from sysfs.
fn read_core_id(cpu: usize) -> Option<usize> {
    let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/core_id");
    fs::read_to_string(&path)
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

/// Read per-node memory info from `/sys/devices/system/node/nodeN/meminfo`.
///
/// Parses `MemTotal` and `MemFree` lines; returns `None` if the file
/// is missing or unparseable.
fn read_node_meminfo(node: usize) -> Option<NodeMemInfo> {
    let path = format!("/sys/devices/system/node/node{node}/meminfo");
    let content = fs::read_to_string(path).ok()?;
    let mut total_kb = None;
    let mut free_kb = None;
    for line in content.lines() {
        if let Some(rest) = line.strip_suffix("kB").map(str::trim_end) {
            if rest.contains("MemTotal") {
                total_kb = rest
                    .rsplit_once(char::is_whitespace)
                    .and_then(|(_, v)| v.parse().ok());
            } else if rest.contains("MemFree") {
                free_kb = rest
                    .rsplit_once(char::is_whitespace)
                    .and_then(|(_, v)| v.parse().ok());
            }
        }
    }
    Some(NodeMemInfo {
        total_kb: total_kb?,
        free_kb: free_kb?,
    })
}

/// Read NUMA distance row from `/sys/devices/system/node/nodeN/distance`.
///
/// Returns the space-separated distance values as a `Vec<u8>`.
fn read_node_distances(node: usize) -> Option<Vec<u8>> {
    let path = format!("/sys/devices/system/node/node{node}/distance");
    let content = fs::read_to_string(path).ok()?;
    let values: Option<Vec<u8>> = content.split_whitespace().map(|s| s.parse().ok()).collect();
    values
}

/// Read the cpulist for a NUMA node. Returns `true` if the node has
/// no CPUs (memory-only / CXL).
fn is_node_memory_only(node: usize) -> bool {
    let path = format!("/sys/devices/system/node/node{node}/cpulist");
    match fs::read_to_string(path) {
        Ok(s) => s.trim().is_empty(),
        Err(_) => false,
    }
}

impl TestTopology {
    /// Discover topology from sysfs (reads `/sys/devices/system/cpu/`).
    pub fn from_system() -> Result<Self> {
        let online_str =
            fs::read_to_string("/sys/devices/system/cpu/online").context("read online cpus")?;
        let online_cpus = parse_cpu_list(&online_str)?;
        if online_cpus.is_empty() {
            bail!("no online CPUs found");
        }

        let mut cpus = BTreeSet::new();
        let mut llc_map: BTreeMap<usize, LlcInfo> = BTreeMap::new();
        let mut numa_nodes = BTreeSet::new();

        // First pass: collect cache size per LLC (read once per LLC, not per CPU).
        let mut llc_cache_sizes: BTreeMap<usize, Option<u64>> = BTreeMap::new();

        for &cpu_id in &online_cpus {
            if !Path::new(&format!("/sys/devices/system/cpu/cpu{cpu_id}")).exists() {
                continue;
            }
            cpus.insert(cpu_id);
            let llc_id = read_llc_id(cpu_id).unwrap_or(0);
            let node_id = read_numa_node(cpu_id).unwrap_or(0);
            let core_id = read_core_id(cpu_id);
            numa_nodes.insert(node_id);
            llc_cache_sizes
                .entry(llc_id)
                .or_insert_with(|| read_llc_cache_size(cpu_id));
            llc_map
                .entry(llc_id)
                .and_modify(|info| {
                    info.cpus.push(cpu_id);
                    if let Some(cid) = core_id {
                        info.cores.entry(cid).or_default().push(cpu_id);
                    }
                })
                .or_insert_with(|| {
                    let mut cores = BTreeMap::new();
                    if let Some(cid) = core_id {
                        cores.insert(cid, vec![cpu_id]);
                    }
                    LlcInfo {
                        cpus: vec![cpu_id],
                        numa_node: node_id,
                        cache_size_kb: llc_cache_sizes.get(&llc_id).copied().flatten(),
                        cores,
                    }
                });
        }
        for info in llc_map.values_mut() {
            info.cpus.sort();
            for siblings in info.cores.values_mut() {
                siblings.sort();
            }
        }

        // Discover additional NUMA nodes from /sys/devices/system/node/
        // (catches memory-only nodes that have no CPUs).
        if let Ok(entries) = fs::read_dir("/sys/devices/system/node") {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if let Some(id_str) = name.strip_prefix("node")
                    && let Ok(id) = id_str.parse::<usize>()
                {
                    numa_nodes.insert(id);
                }
            }
        }

        let n = numa_nodes.len();
        let node_ids: Vec<usize> = numa_nodes.iter().copied().collect();

        // Read per-node memory info.
        let mut node_mem = BTreeMap::new();
        for &nid in &node_ids {
            if let Some(mi) = read_node_meminfo(nid) {
                node_mem.insert(nid, mi);
            }
        }

        // Identify memory-only nodes.
        let mut memory_only_nodes = BTreeSet::new();
        for &nid in &node_ids {
            if is_node_memory_only(nid) {
                memory_only_nodes.insert(nid);
            }
        }

        // Build distance matrix. Try sysfs first, fall back to 10/20.
        let numa_distances = {
            let mut matrix = Vec::with_capacity(n * n);
            let mut from_sysfs = true;
            for &nid in &node_ids {
                if let Some(row) = read_node_distances(nid) {
                    if row.len() == n {
                        matrix.extend_from_slice(&row);
                    } else {
                        from_sysfs = false;
                        break;
                    }
                } else {
                    from_sysfs = false;
                    break;
                }
            }
            if !from_sysfs || matrix.len() != n * n {
                matrix.clear();
                matrix.resize(n * n, 0);
                for i in 0..n {
                    for j in 0..n {
                        matrix[i * n + j] = if i == j { 10 } else { 20 };
                    }
                }
            }
            matrix
        };

        Ok(Self {
            cpus: cpus.into_iter().collect(),
            llcs: llc_map.into_values().collect(),
            numa_nodes,
            numa_distances,
            node_mem,
            memory_only_nodes,
        })
    }

    /// Total number of CPUs.
    pub fn total_cpus(&self) -> usize {
        self.cpus.len()
    }
    /// Number of last-level caches.
    pub fn num_llcs(&self) -> usize {
        self.llcs.len()
    }
    /// Number of NUMA nodes.
    pub fn num_numa_nodes(&self) -> usize {
        self.numa_nodes.len()
    }
    /// NUMA node IDs as a `BTreeSet`.
    pub fn numa_node_ids(&self) -> &BTreeSet<usize> {
        &self.numa_nodes
    }
    /// All LLC domains.
    pub fn llcs(&self) -> &[LlcInfo] {
        &self.llcs
    }
    /// All CPU IDs, sorted.
    pub fn all_cpus(&self) -> &[usize] {
        &self.cpus
    }
    /// All CPU IDs as a `BTreeSet`.
    pub fn all_cpuset(&self) -> BTreeSet<usize> {
        self.cpus.iter().copied().collect()
    }

    /// CPUs available for workload placement. When the topology has
    /// more than 2 CPUs, the last CPU is reserved for the root cgroup
    /// (cgroup 0); with 2 or fewer CPUs, every CPU is returned.
    pub fn usable_cpus(&self) -> &[usize] {
        if self.cpus.len() > 2 {
            &self.cpus[..self.cpus.len() - 1]
        } else {
            &self.cpus
        }
    }
    /// Usable CPUs as a `BTreeSet`.
    pub fn usable_cpuset(&self) -> BTreeSet<usize> {
        self.usable_cpus().iter().copied().collect()
    }
    /// CPUs belonging to LLC at index `idx`.
    pub fn cpus_in_llc(&self, idx: usize) -> &[usize] {
        &self.llcs[idx].cpus
    }
    /// CPUs in LLC `idx` as a `BTreeSet`.
    pub fn llc_aligned_cpuset(&self, idx: usize) -> BTreeSet<usize> {
        self.llcs[idx].cpus.iter().copied().collect()
    }
    /// CPUs in all LLCs belonging to NUMA node `node` as a `BTreeSet`.
    pub fn numa_aligned_cpuset(&self, node: usize) -> BTreeSet<usize> {
        self.llcs
            .iter()
            .filter(|llc| llc.numa_node() == node)
            .flat_map(|llc| llc.cpus())
            .copied()
            .collect()
    }

    /// NUMA nodes covered by the given CPU set.
    pub fn numa_nodes_for_cpuset(&self, cpus: &BTreeSet<usize>) -> BTreeSet<usize> {
        self.llcs
            .iter()
            .filter(|llc| llc.cpus.iter().any(|c| cpus.contains(c)))
            .map(|llc| llc.numa_node)
            .collect()
    }

    /// Per-node memory info. Returns `None` when the node ID is not
    /// present or meminfo is unavailable.
    pub fn node_meminfo(&self, node_id: usize) -> Option<&NodeMemInfo> {
        self.node_mem.get(&node_id)
    }

    /// Inter-node NUMA distance. Returns 255 when either node ID is
    /// not present, matching the kernel's unreachable distance.
    pub fn numa_distance(&self, from: usize, to: usize) -> u8 {
        let n = self.numa_nodes.len();
        let Some(from_idx) = self.numa_nodes.iter().position(|&id| id == from) else {
            return 255;
        };
        let Some(to_idx) = self.numa_nodes.iter().position(|&id| id == to) else {
            return 255;
        };
        self.numa_distances[from_idx * n + to_idx]
    }

    /// Whether the node is memory-only (has RAM but no CPUs). Typical
    /// for CXL-attached memory tiers.
    pub fn is_memory_only(&self, node_id: usize) -> bool {
        self.memory_only_nodes.contains(&node_id)
    }

    /// One `BTreeSet` of CPUs per LLC.
    pub fn split_by_llc(&self) -> Vec<BTreeSet<usize>> {
        self.llcs
            .iter()
            .map(|l| l.cpus.iter().copied().collect())
            .collect()
    }

    /// Generate `n` cpusets with `overlap_frac` overlap between adjacent sets.
    pub fn overlapping_cpusets(&self, n: usize, overlap_frac: f64) -> Vec<BTreeSet<usize>> {
        let total = self.cpus.len();
        if n == 0 || total == 0 {
            return vec![];
        }
        let base = total / n;
        let overlap = ((base as f64) * overlap_frac).ceil() as usize;
        let stride = if base > overlap { base - overlap } else { 1 };
        (0..n)
            .map(|i| {
                let start = (i * stride) % total;
                (0..base.max(1))
                    .map(|j| self.cpus[(start + j) % total])
                    .collect()
            })
            .collect()
    }

    /// Format a CPU set as a compact range string (e.g. `"0-3,5,7-9"`).
    pub fn cpuset_string(cpus: &BTreeSet<usize>) -> String {
        if cpus.is_empty() {
            return String::new();
        }
        let sorted: Vec<usize> = cpus.iter().copied().collect();
        let mut ranges = Vec::new();
        let (mut start, mut end) = (sorted[0], sorted[0]);
        for &cpu in &sorted[1..] {
            if cpu == end + 1 {
                end = cpu;
            } else {
                ranges.push(if start == end {
                    format!("{start}")
                } else {
                    format!("{start}-{end}")
                });
                start = cpu;
                end = cpu;
            }
        }
        ranges.push(if start == end {
            format!("{start}")
        } else {
            format!("{start}-{end}")
        });
        ranges.join(",")
    }

    /// Build a topology from a VM spec (NUMA nodes x LLCs x cores x threads).
    ///
    /// Parameters are big-to-little: NUMA nodes, LLCs, cores per LLC,
    /// threads per core. Multiple LLCs can share a NUMA node when
    /// `numa_nodes < llcs`. CPUs are numbered sequentially. Used as a
    /// fallback when sysfs is incomplete inside a guest VM.
    pub fn from_spec(numa_nodes: u32, llcs: u32, cores: u32, threads: u32) -> Self {
        assert!(numa_nodes > 0, "numa_nodes must be > 0");
        assert!(
            llcs.is_multiple_of(numa_nodes),
            "llcs ({llcs}) must be divisible by numa_nodes ({numa_nodes})"
        );
        let total = (llcs * cores * threads) as usize;
        let cpus_per_llc = (cores * threads) as usize;
        let cpus: Vec<usize> = (0..total).collect();
        let llcs_per_node = (llcs / numa_nodes) as usize;
        let llc_infos: Vec<LlcInfo> = (0..llcs as usize)
            .map(|l| {
                let start = l * cpus_per_llc;
                let end = start + cpus_per_llc;
                let mut core_map = BTreeMap::new();
                for c in 0..cores as usize {
                    let base = start + c * threads as usize;
                    let siblings: Vec<usize> = (base..base + threads as usize).collect();
                    core_map.insert(c, siblings);
                }
                LlcInfo {
                    cpus: (start..end).collect(),
                    numa_node: l / llcs_per_node,
                    cache_size_kb: None,
                    cores: core_map,
                }
            })
            .collect();
        let n = numa_nodes as usize;
        let numa_node_set: BTreeSet<usize> = (0..n).collect();
        let mut distances = vec![0u8; n * n];
        for i in 0..n {
            for j in 0..n {
                distances[i * n + j] = if i == j { 10 } else { 20 };
            }
        }
        Self {
            cpus,
            llcs: llc_infos,
            numa_nodes: numa_node_set,
            numa_distances: distances,
            node_mem: BTreeMap::new(),
            memory_only_nodes: BTreeSet::new(),
        }
    }

    #[cfg(test)]
    pub fn synthetic(num_cpus: usize, num_llcs: usize) -> Self {
        let cpus: Vec<usize> = (0..num_cpus).collect();
        let per_llc = num_cpus / num_llcs;
        let llcs: Vec<LlcInfo> = (0..num_llcs)
            .map(|i| {
                let start = i * per_llc;
                let end = if i == num_llcs - 1 {
                    num_cpus
                } else {
                    (i + 1) * per_llc
                };
                LlcInfo {
                    cpus: (start..end).collect(),
                    numa_node: i,
                    cache_size_kb: None,
                    cores: BTreeMap::new(),
                }
            })
            .collect();
        let n = num_llcs;
        let numa_nodes: BTreeSet<usize> = (0..n).collect();
        let mut distances = vec![0u8; n * n];
        for i in 0..n {
            for j in 0..n {
                distances[i * n + j] = if i == j { 10 } else { 20 };
            }
        }
        Self {
            cpus,
            llcs,
            numa_nodes,
            numa_distances: distances,
            node_mem: BTreeMap::new(),
            memory_only_nodes: BTreeSet::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpuset_string_empty() {
        assert_eq!(TestTopology::cpuset_string(&BTreeSet::new()), "");
    }

    #[test]
    fn cpuset_string_single() {
        assert_eq!(TestTopology::cpuset_string(&[3].into_iter().collect()), "3");
    }

    #[test]
    fn cpuset_string_range() {
        assert_eq!(
            TestTopology::cpuset_string(&[0, 1, 2, 3].into_iter().collect()),
            "0-3"
        );
    }

    #[test]
    fn cpuset_string_gaps() {
        assert_eq!(
            TestTopology::cpuset_string(&[0, 1, 3, 5, 6, 7].into_iter().collect()),
            "0-1,3,5-7"
        );
    }

    #[test]
    fn synthetic_topology() {
        let t = TestTopology::synthetic(8, 2);
        assert_eq!(t.total_cpus(), 8);
        assert_eq!(t.num_llcs(), 2);
        assert_eq!(t.cpus_in_llc(0), &[0, 1, 2, 3]);
        assert_eq!(t.cpus_in_llc(1), &[4, 5, 6, 7]);
    }

    #[test]
    fn overlapping_cpusets_basic() {
        let t = TestTopology::synthetic(8, 1);
        let sets = t.overlapping_cpusets(2, 0.5);
        assert_eq!(sets.len(), 2);
        for s in &sets {
            assert_eq!(s.len(), 4);
        }
        let overlap: BTreeSet<usize> = sets[0].intersection(&sets[1]).copied().collect();
        assert!(!overlap.is_empty());
    }

    #[test]
    fn overlapping_cpusets_no_overlap() {
        let t = TestTopology::synthetic(8, 1);
        let sets = t.overlapping_cpusets(2, 0.0);
        assert_eq!(sets.len(), 2);
        let overlap: BTreeSet<usize> = sets[0].intersection(&sets[1]).copied().collect();
        assert!(overlap.is_empty());
    }

    #[test]
    fn split_by_llc() {
        let t = TestTopology::synthetic(8, 2);
        let splits = t.split_by_llc();
        assert_eq!(splits.len(), 2);
        assert_eq!(splits[0], [0, 1, 2, 3].into_iter().collect());
        assert_eq!(splits[1], [4, 5, 6, 7].into_iter().collect());
    }

    #[test]
    fn llc_aligned_cpuset() {
        let t = TestTopology::synthetic(8, 2);
        assert_eq!(t.llc_aligned_cpuset(0), [0, 1, 2, 3].into_iter().collect());
        assert_eq!(t.llc_aligned_cpuset(1), [4, 5, 6, 7].into_iter().collect());
    }

    #[test]
    fn from_spec_single_llc() {
        let t = TestTopology::from_spec(1, 1, 4, 2);
        assert_eq!(t.total_cpus(), 8);
        assert_eq!(t.num_llcs(), 1);
        assert_eq!(t.num_numa_nodes(), 1);
        assert_eq!(t.all_cpus(), &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(t.cpus_in_llc(0), &[0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn from_spec_multi_llc() {
        let t = TestTopology::from_spec(1, 2, 4, 2);
        assert_eq!(t.total_cpus(), 16);
        assert_eq!(t.num_llcs(), 2);
        assert_eq!(t.num_numa_nodes(), 1);
        assert_eq!(t.cpus_in_llc(0), &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(t.cpus_in_llc(1), &[8, 9, 10, 11, 12, 13, 14, 15]);
    }

    #[test]
    fn from_spec_no_smt() {
        let t = TestTopology::from_spec(1, 2, 2, 1);
        assert_eq!(t.total_cpus(), 4);
        assert_eq!(t.num_llcs(), 2);
        assert_eq!(t.cpus_in_llc(0), &[0, 1]);
        assert_eq!(t.cpus_in_llc(1), &[2, 3]);
    }

    #[test]
    fn from_spec_minimal() {
        let t = TestTopology::from_spec(1, 1, 1, 1);
        assert_eq!(t.total_cpus(), 1);
        assert_eq!(t.num_llcs(), 1);
        assert_eq!(t.all_cpus(), &[0]);
    }

    #[test]
    fn from_spec_multi_numa() {
        let t = TestTopology::from_spec(2, 4, 4, 2);
        assert_eq!(t.total_cpus(), 32);
        assert_eq!(t.num_llcs(), 4);
        assert_eq!(t.num_numa_nodes(), 2);
        // LLCs 0,1 -> NUMA node 0; LLCs 2,3 -> NUMA node 1
        assert_eq!(t.llcs()[0].numa_node(), 0);
        assert_eq!(t.llcs()[1].numa_node(), 0);
        assert_eq!(t.llcs()[2].numa_node(), 1);
        assert_eq!(t.llcs()[3].numa_node(), 1);
    }

    #[test]
    fn overlapping_cpusets_zero_n() {
        let t = TestTopology::synthetic(8, 1);
        assert!(t.overlapping_cpusets(0, 0.5).is_empty());
    }

    #[test]
    fn synthetic_single_llc() {
        let t = TestTopology::synthetic(4, 1);
        assert_eq!(t.num_llcs(), 1);
        assert_eq!(t.total_cpus(), 4);
        assert_eq!(t.num_numa_nodes(), 1);
        assert_eq!(t.all_cpus(), &[0, 1, 2, 3]);
    }

    #[test]
    fn synthetic_many_llcs() {
        let t = TestTopology::synthetic(16, 4);
        assert_eq!(t.num_llcs(), 4);
        for i in 0..4 {
            assert_eq!(t.cpus_in_llc(i).len(), 4);
        }
    }

    #[test]
    fn cpuset_string_two_ranges() {
        assert_eq!(
            TestTopology::cpuset_string(&[0, 1, 2, 5, 6, 7].into_iter().collect()),
            "0-2,5-7"
        );
    }

    #[test]
    fn cpuset_string_all_isolated() {
        assert_eq!(
            TestTopology::cpuset_string(&[1, 3, 5].into_iter().collect()),
            "1,3,5"
        );
    }

    #[test]
    fn cpuset_string_large_range() {
        let cpus: BTreeSet<usize> = (0..128).collect();
        assert_eq!(TestTopology::cpuset_string(&cpus), "0-127");
    }

    #[test]
    fn overlapping_cpusets_single_set() {
        let t = TestTopology::synthetic(8, 1);
        let sets = t.overlapping_cpusets(1, 0.5);
        assert_eq!(sets.len(), 1);
        assert_eq!(sets[0].len(), 8);
    }

    #[test]
    fn split_by_llc_single() {
        let t = TestTopology::synthetic(4, 1);
        let splits = t.split_by_llc();
        assert_eq!(splits.len(), 1);
        assert_eq!(splits[0].len(), 4);
    }

    /// Regression test for the split_by_llc bug: topology(2,4,1) must
    /// produce 2 disjoint LLC sets covering all 8 CPUs. Before the fix,
    /// from_system() on AMD hosts returned 1 LLC because CPUID leaf
    /// 0x8000001D was not patched, and the test panicked indexing
    /// llc_sets[1].
    #[test]
    fn split_by_llc_two_llc_regression() {
        let t = TestTopology::from_spec(1, 2, 4, 1);
        assert_eq!(t.total_cpus(), 8);
        assert_eq!(t.num_llcs(), 2);

        let splits = t.split_by_llc();
        assert_eq!(splits.len(), 2, "2-LLC topology must produce 2 LLC sets");

        // Sets must be disjoint
        let overlap: BTreeSet<usize> = splits[0].intersection(&splits[1]).copied().collect();
        assert!(
            overlap.is_empty(),
            "LLC sets must be disjoint: overlap={overlap:?}"
        );

        // Union must cover all CPUs
        let union: BTreeSet<usize> = splits[0].union(&splits[1]).copied().collect();
        assert_eq!(union, t.all_cpuset(), "LLC sets must cover all CPUs");

        // Each set has 4 CPUs (4 cores per LLC, 1 thread)
        assert_eq!(splits[0].len(), 4);
        assert_eq!(splits[1].len(), 4);

        // Verify exact contents
        assert_eq!(splits[0], [0, 1, 2, 3].into_iter().collect());
        assert_eq!(splits[1], [4, 5, 6, 7].into_iter().collect());
    }

    #[test]
    fn usable_cpus_reserves_last() {
        let t = TestTopology::synthetic(8, 2);
        assert_eq!(t.usable_cpus().len(), 7);
        assert!(!t.usable_cpus().contains(&7));
    }

    #[test]
    fn usable_cpus_small_no_reserve() {
        let t = TestTopology::synthetic(2, 1);
        assert_eq!(t.usable_cpus().len(), 2);
    }

    #[test]
    fn usable_cpus_single_cpu() {
        let t = TestTopology::synthetic(1, 1);
        assert_eq!(t.usable_cpus().len(), 1);
    }

    #[test]
    fn parse_cpu_list_simple() {
        assert_eq!(parse_cpu_list("0,1,2,3").unwrap(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn parse_cpu_list_range() {
        assert_eq!(parse_cpu_list("0-3").unwrap(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn parse_cpu_list_mixed() {
        assert_eq!(
            parse_cpu_list("0-2,5,7-9").unwrap(),
            vec![0, 1, 2, 5, 7, 8, 9]
        );
    }

    #[test]
    fn parse_cpu_list_empty() {
        assert!(parse_cpu_list("").unwrap().is_empty());
    }

    #[test]
    fn parse_cpu_list_whitespace() {
        assert_eq!(parse_cpu_list("  0 , 1 , 2  ").unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn from_spec_large() {
        let t = TestTopology::from_spec(1, 4, 8, 2);
        assert_eq!(t.total_cpus(), 64);
        assert_eq!(t.num_llcs(), 4);
        assert_eq!(t.num_numa_nodes(), 1);
    }

    #[test]
    fn llc_info_accessors() {
        let t = TestTopology::synthetic(8, 2);
        let llcs = t.llcs();
        assert_eq!(llcs.len(), 2);
        assert_eq!(llcs[0].cpus(), &[0, 1, 2, 3]);
        assert_eq!(llcs[0].numa_node(), 0);
        assert_eq!(llcs[1].cpus(), &[4, 5, 6, 7]);
        assert_eq!(llcs[1].numa_node(), 1);
    }

    #[test]
    fn from_spec_cores_populated() {
        let t = TestTopology::from_spec(1, 2, 4, 2);
        let llc0 = &t.llcs()[0];
        assert_eq!(llc0.num_cores(), 4);
        assert_eq!(llc0.cores().len(), 4);
        assert_eq!(llc0.cores()[&0], vec![0, 1]);
        assert_eq!(llc0.cores()[&1], vec![2, 3]);
        assert_eq!(llc0.cores()[&2], vec![4, 5]);
        assert_eq!(llc0.cores()[&3], vec![6, 7]);
        let llc1 = &t.llcs()[1];
        assert_eq!(llc1.cores()[&0], vec![8, 9]);
    }

    #[test]
    fn from_spec_no_smt_cores() {
        let t = TestTopology::from_spec(1, 1, 4, 1);
        let llc = &t.llcs()[0];
        assert_eq!(llc.num_cores(), 4);
        assert_eq!(llc.cores()[&0], vec![0]);
        assert_eq!(llc.cores()[&3], vec![3]);
    }

    #[test]
    fn parse_cache_size_formats() {
        assert_eq!(parse_cache_size("32768K"), Some(32768));
        assert_eq!(parse_cache_size("32M"), Some(32768));
        assert_eq!(parse_cache_size("65536"), Some(64));
    }

    #[test]
    fn num_cores_from_cores_map() {
        let llc = LlcInfo {
            cpus: vec![0, 1, 2, 3],
            numa_node: 0,
            cache_size_kb: None,
            cores: BTreeMap::from([(0, vec![0, 1]), (1, vec![2, 3])]),
        };
        assert_eq!(llc.num_cores(), 2);
    }

    #[test]
    fn num_cores_fallback_to_cpus() {
        let llc = LlcInfo {
            cpus: vec![0, 1, 2, 3],
            numa_node: 0,
            cache_size_kb: None,
            cores: BTreeMap::new(),
        };
        assert_eq!(llc.num_cores(), 4);
    }

    #[test]
    fn parse_cpu_list_lenient_simple() {
        assert_eq!(parse_cpu_list_lenient("0,1,2,3"), vec![0, 1, 2, 3]);
    }

    #[test]
    fn parse_cpu_list_lenient_range() {
        assert_eq!(parse_cpu_list_lenient("0-3"), vec![0, 1, 2, 3]);
    }

    #[test]
    fn parse_cpu_list_lenient_mixed() {
        assert_eq!(
            parse_cpu_list_lenient("0-2,5,7-9"),
            vec![0, 1, 2, 5, 7, 8, 9]
        );
    }

    #[test]
    fn parse_cpu_list_lenient_empty() {
        assert!(parse_cpu_list_lenient("").is_empty());
    }

    #[test]
    fn parse_cpu_list_lenient_skips_garbage() {
        assert_eq!(parse_cpu_list_lenient("0,abc,2,xyz-3,4"), vec![0, 2, 4]);
    }

    #[test]
    fn parse_cpu_list_lenient_whitespace() {
        assert_eq!(parse_cpu_list_lenient("  0 , 1 , 2  "), vec![0, 1, 2]);
    }

    #[test]
    fn cache_size_bare_number() {
        // Bare number without suffix is treated as bytes, converted to KB.
        assert_eq!(parse_cache_size("1024"), Some(1));
    }

    #[test]
    fn cache_size_empty_string() {
        assert_eq!(parse_cache_size(""), None);
    }

    #[test]
    fn cache_size_whitespace_only() {
        assert_eq!(parse_cache_size("   "), None);
    }

    #[test]
    fn numa_aligned_cpuset_two_nodes() {
        // 2 NUMA nodes, 4 LLCs (2 per NUMA), 4 cores, 1 thread
        // LLCs 0,1 -> NUMA 0 (CPUs 0-7), LLCs 2,3 -> NUMA 1 (CPUs 8-15)
        // Total = 4 * 4 * 1 = 16 CPUs per NUMA pair = each LLC has 4 CPUs
        // NUMA 0: LLCs 0,1 → CPUs 0-3, 4-7 = 0-7
        // NUMA 1: LLCs 2,3 → CPUs 8-11, 12-15 = 8-15 (but only 16 CPUs)
        //
        // from_spec(2, 4, 4, 1) → 4 LLCs × 4 cores × 1 thread = 16 CPUs
        let t = TestTopology::from_spec(2, 4, 4, 1);
        assert_eq!(t.total_cpus(), 16);
        assert_eq!(t.num_numa_nodes(), 2);
        assert_eq!(t.num_llcs(), 4);

        let node0: BTreeSet<usize> = t.numa_aligned_cpuset(0);
        let node1: BTreeSet<usize> = t.numa_aligned_cpuset(1);

        // NUMA 0: LLCs 0,1 each with 4 CPUs → CPUs 0-7
        let expected0: BTreeSet<usize> = (0..8).collect();
        assert_eq!(node0, expected0);

        // NUMA 1: LLCs 2,3 each with 4 CPUs → CPUs 8-15
        let expected1: BTreeSet<usize> = (8..16).collect();
        assert_eq!(node1, expected1);
    }

    // -- proptest --

    proptest::proptest! {
        #[test]
        fn prop_parse_cpu_list_never_panics(s in "\\PC{0,30}") {
            let _ = parse_cpu_list(&s);
        }

        #[test]
        fn prop_parse_cpu_list_single_cpu(cpu in 0usize..256) {
            let result = parse_cpu_list(&cpu.to_string()).unwrap();
            assert_eq!(result, vec![cpu]);
        }

        #[test]
        fn prop_parse_cpu_list_range_sorted(lo in 0usize..128, span in 1usize..64) {
            let hi = lo + span;
            let result = parse_cpu_list(&format!("{lo}-{hi}")).unwrap();
            assert_eq!(result.len(), span + 1);
            assert_eq!(*result.first().unwrap(), lo);
            assert_eq!(*result.last().unwrap(), hi);
            // Must be sorted.
            for w in result.windows(2) {
                assert!(w[0] <= w[1]);
            }
        }

        #[test]
        fn prop_parse_cpu_list_lenient_never_panics(s in "\\PC{0,30}") {
            let _ = parse_cpu_list_lenient(&s);
        }

        #[test]
        fn prop_parse_cpu_list_lenient_superset_of_strict(
            lo in 0usize..64,
            hi in 64usize..128,
        ) {
            let s = format!("{lo}-{hi}");
            let strict = parse_cpu_list(&s).unwrap();
            let lenient = parse_cpu_list_lenient(&s);
            assert_eq!(strict, lenient);
        }

        #[test]
        fn prop_parse_cpu_list_roundtrip(
            cpus in proptest::collection::btree_set(0usize..256, 1..16),
        ) {
            // Format as comma-separated list, parse back, compare.
            let s: String = cpus.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(",");
            let parsed = parse_cpu_list(&s).unwrap();
            let roundtrip: std::collections::BTreeSet<usize> = parsed.into_iter().collect();
            assert_eq!(cpus, roundtrip);
        }
    }

    #[test]
    fn numa_node_ids_synthetic() {
        let t = TestTopology::synthetic(8, 2);
        assert_eq!(*t.numa_node_ids(), [0, 1].into_iter().collect());
    }

    #[test]
    fn numa_nodes_for_cpuset_single_node() {
        let t = TestTopology::from_spec(2, 4, 4, 1);
        let cpuset: BTreeSet<usize> = (0..4).collect(); // LLC 0, NUMA 0
        assert_eq!(t.numa_nodes_for_cpuset(&cpuset), [0].into_iter().collect());
    }

    #[test]
    fn numa_nodes_for_cpuset_both_nodes() {
        let t = TestTopology::from_spec(2, 4, 4, 1);
        let cpuset: BTreeSet<usize> = [0, 8].into_iter().collect(); // NUMA 0 + NUMA 1
        assert_eq!(
            t.numa_nodes_for_cpuset(&cpuset),
            [0, 1].into_iter().collect()
        );
    }

    #[test]
    fn numa_nodes_for_cpuset_empty() {
        let t = TestTopology::from_spec(2, 4, 4, 1);
        assert!(t.numa_nodes_for_cpuset(&BTreeSet::new()).is_empty());
    }

    // -- NUMA distance tests --

    #[test]
    fn from_spec_numa_distance_local() {
        let t = TestTopology::from_spec(2, 4, 4, 1);
        assert_eq!(t.numa_distance(0, 0), 10);
        assert_eq!(t.numa_distance(1, 1), 10);
    }

    #[test]
    fn from_spec_numa_distance_remote() {
        let t = TestTopology::from_spec(2, 4, 4, 1);
        assert_eq!(t.numa_distance(0, 1), 20);
        assert_eq!(t.numa_distance(1, 0), 20);
    }

    #[test]
    fn from_spec_numa_distance_single_node() {
        let t = TestTopology::from_spec(1, 2, 4, 1);
        assert_eq!(t.numa_distance(0, 0), 10);
    }

    #[test]
    fn numa_distance_invalid_node() {
        let t = TestTopology::from_spec(2, 4, 4, 1);
        assert_eq!(t.numa_distance(0, 99), 255);
        assert_eq!(t.numa_distance(99, 0), 255);
    }

    #[test]
    fn synthetic_distances_default() {
        let t = TestTopology::synthetic(8, 2);
        assert_eq!(t.numa_distance(0, 0), 10);
        assert_eq!(t.numa_distance(0, 1), 20);
        assert_eq!(t.numa_distance(1, 0), 20);
    }

    // -- node_meminfo tests --

    #[test]
    fn node_meminfo_used_kb() {
        let mi = NodeMemInfo {
            total_kb: 1024,
            free_kb: 256,
        };
        assert_eq!(mi.used_kb(), 768);
    }

    #[test]
    fn node_meminfo_used_kb_saturates() {
        let mi = NodeMemInfo {
            total_kb: 0,
            free_kb: 100,
        };
        assert_eq!(mi.used_kb(), 0);
    }

    #[test]
    fn from_spec_no_meminfo() {
        let t = TestTopology::from_spec(2, 4, 4, 1);
        assert!(t.node_meminfo(0).is_none());
        assert!(t.node_meminfo(1).is_none());
    }

    #[test]
    fn synthetic_no_meminfo() {
        let t = TestTopology::synthetic(8, 2);
        assert!(t.node_meminfo(0).is_none());
    }

    // -- is_memory_only tests --

    #[test]
    fn from_spec_not_memory_only() {
        let t = TestTopology::from_spec(2, 4, 4, 1);
        assert!(!t.is_memory_only(0));
        assert!(!t.is_memory_only(1));
    }

    #[test]
    fn is_memory_only_nonexistent_node() {
        let t = TestTopology::from_spec(2, 4, 4, 1);
        assert!(!t.is_memory_only(99));
    }
}
