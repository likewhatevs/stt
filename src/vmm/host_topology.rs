//! Host CPU topology discovery for performance_mode.
//!
//! Wraps [`TestTopology`](crate::topology::TestTopology) for LLC-aware
//! vCPU pinning and host resource validation.

use anyhow::{Context, Result};

// Advisory flock primitives live in `crate::flock` so both LLC +
// per-CPU coordination here and per-cache-entry coordination in
// `crate::cache` share one `try_flock` implementation (with a single
// `O_CLOEXEC` source of truth) plus one `HolderInfo` /proc/locks
// parser. Re-importing the names keeps existing in-module call sites
// (production + `super::*` tests) compiling unchanged.
use crate::flock::{FlockMode, try_flock};

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
    /// LLC indices grouped by their NUMA node. Memoized at construction
    /// time from `llc_groups + cpu_to_node` so repeated NUMA-aware
    /// placement queries (perf-mode rotation, `--cpu-cap` consolidation
    /// PLAN) don't re-walk every LLC's CPU list on every call. Access
    /// via [`HostTopology::host_llcs_by_numa_node`]. `BTreeMap` (not
    /// `HashMap`) for deterministic iteration order — two ktstr
    /// invocations on the same host MUST produce identical LLC
    /// selections so their ACQUIRE phases converge on the same indices.
    pub(crate) host_node_llcs: std::collections::BTreeMap<usize, Vec<usize>>,
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
        let llc_groups: Vec<LlcGroup> = topo
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
        let host_node_llcs = Self::compute_host_node_llcs(&llc_groups, &cpu_to_node);
        Ok(Self {
            llc_groups,
            online_cpus,
            cpu_to_node,
            host_node_llcs,
        })
    }

    /// Build a synthetic `HostTopology` from `(cpu_list, node_id)`
    /// pairs for tests. One pair per LLC group; within a pair the
    /// `cpu_list` becomes the group's CPUs and the `node_id` is the
    /// NUMA node every CPU in that group is assigned to.
    /// `online_cpus` is the flattened concatenation of every group's
    /// CPUs in input order; `cpu_to_node` is built by broadcasting
    /// each group's node over its CPUs; `host_node_llcs` goes through
    /// the same [`compute_host_node_llcs`] path production uses, so
    /// tests never diverge from the sysfs-derived memoization.
    ///
    /// Intended for test fixtures that want a deterministic in-memory
    /// topology without stubbing `/sys/devices/system/cpu/*`.
    /// Previously this logic was duplicated across three helper
    /// functions (`synthetic_topo`, `synthetic_topo_numa`,
    /// `synth_host_topo`) — consolidated here so the
    /// `HostTopology` invariant is maintained in one place. The
    /// `#[cfg(test)]` gate keeps the symbol out of release builds.
    #[cfg(test)]
    pub(crate) fn new_for_tests(groups: &[(Vec<usize>, usize)]) -> Self {
        let llc_groups: Vec<LlcGroup> = groups
            .iter()
            .map(|(cpus, _)| LlcGroup { cpus: cpus.clone() })
            .collect();
        let cpu_to_node: std::collections::HashMap<usize, usize> = groups
            .iter()
            .flat_map(|(cpus, node)| cpus.iter().map(move |&cpu| (cpu, *node)))
            .collect();
        let online_cpus: Vec<usize> = groups
            .iter()
            .flat_map(|(cpus, _)| cpus.iter().copied())
            .collect();
        let host_node_llcs = HostTopology::compute_host_node_llcs(&llc_groups, &cpu_to_node);
        HostTopology {
            llc_groups,
            online_cpus,
            cpu_to_node,
            host_node_llcs,
        }
    }

    /// Compute the memoized `host_node_llcs` map from `llc_groups` +
    /// `cpu_to_node`. Uses the same majority-vote NUMA-assignment rule
    /// as [`llc_numa_node`], so the memoized map and the one-off query
    /// method never disagree. Separate fn (not inlined) so
    /// `from_sysfs` and synthetic-test constructors share one path.
    fn compute_host_node_llcs(
        llc_groups: &[LlcGroup],
        cpu_to_node: &std::collections::HashMap<usize, usize>,
    ) -> std::collections::BTreeMap<usize, Vec<usize>> {
        let mut node_llcs: std::collections::BTreeMap<usize, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (idx, group) in llc_groups.iter().enumerate() {
            // Majority-vote NUMA node for this LLC — matches
            // `llc_numa_node` exactly. We inline the logic here rather
            // than calling the method because we don't yet have `self`.
            let mut counts: std::collections::HashMap<usize, usize> =
                std::collections::HashMap::new();
            for &cpu in &group.cpus {
                let node = cpu_to_node.get(&cpu).copied().unwrap_or(0);
                *counts.entry(node).or_insert(0) += 1;
            }
            let node = counts
                .into_iter()
                .max_by_key(|&(_, count)| count)
                .map(|(node, _)| node)
                .unwrap_or(0);
            node_llcs.entry(node).or_default().push(idx);
        }
        // Within-node LLC ordering: ascending llc_idx. Callers that
        // walk `host_node_llcs[node]` rely on this for deterministic
        // output — two ktstr invocations with identical topology see
        // the same walk order.
        for llcs in node_llcs.values_mut() {
            llcs.sort_unstable();
        }
        node_llcs
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

    // ------------------------------------------------------------------
    // Shared NUMA-placement primitives
    // ------------------------------------------------------------------
    //
    // Used by the existing perf-mode pinning path
    // ([`numa_aware_llc_order`]) AND the `--cpu-cap` consolidation
    // PLAN phase. Both callers implement DIFFERENT selection algorithms
    // on top of these queries:
    //
    // - Perf-mode distributes virtual NUMA nodes across host NUMA
    //   nodes with modulo rotation; uses primitive 1 + 2 (group-by-node
    //   + eligibility-by-capacity). No distance lookup.
    // - Consolidation seeds from a scored LLC list then greedily
    //   expands within the seed's node, spilling to nearest-by-distance
    //   when needed; uses primitive 1 + 2 + 3.
    //
    // Kept as small orthogonal queries rather than a single mega-selector
    // — the two algorithms genuinely do different things, but they both
    // need the same three topology lookups.

    /// Memoized map of NUMA node → LLC indices on that node. Returned
    /// by reference so callers can iterate without cloning; `BTreeMap`
    /// gives deterministic iteration so two invocations on identical
    /// topologies produce identical walks.
    ///
    /// In-tree callers currently reach the same data via
    /// [`numa_nodes_sorted_by_distance`] and [`numa_nodes_with_capacity`]
    /// — both iterate `host_node_llcs` internally — so this accessor
    /// has no direct consumer today. Kept as a stable handle for
    /// future callers (e.g. a planned `ktstr topo --json` NUMA
    /// section) and downstream tooling that wants the raw map.
    #[allow(dead_code)]
    pub(crate) fn host_llcs_by_numa_node(&self) -> &std::collections::BTreeMap<usize, Vec<usize>> {
        &self.host_node_llcs
    }

    /// Return every NUMA node that has `>= min_llcs` LLCs, paired with
    /// that node's LLC-index slice. Callers filter through this when
    /// their algorithm requires per-node capacity guarantees (perf-mode
    /// passes `ceil(llcs/numa_nodes)` so any guest node can land on any
    /// host node; consolidation passes 1 so every node with at least
    /// one free LLC is a valid spill candidate). Iteration order
    /// follows the underlying `BTreeMap` — ascending by node id.
    pub(crate) fn numa_nodes_with_capacity(&self, min_llcs: usize) -> Vec<(usize, &Vec<usize>)> {
        self.host_node_llcs
            .iter()
            .filter(|(_, llcs)| llcs.len() >= min_llcs)
            .map(|(&node, llcs)| (node, llcs))
            .collect()
    }

    /// Return NUMA node ids sorted by distance from `anchor` ascending,
    /// with unreachable nodes (distance 255 per Linux convention)
    /// demoted to the end. Caller supplies the distance lookup via
    /// `distance_fn` so this primitive stays independent of any
    /// specific distance source — consolidation threads
    /// `TestTopology::numa_distance` through a closure, while callers
    /// without a distance matrix can pass
    /// `|from, to| if from == to { 10 } else { 20 }` for a trivial
    /// near/far split.
    ///
    /// `anchor` is included in the output (distance to self = 10 on
    /// the Linux convention, sorting first). Nodes without any LLCs
    /// on this host are skipped — spilling to an empty node has no
    /// value.
    pub(crate) fn numa_nodes_sorted_by_distance(
        &self,
        anchor: usize,
        distance_fn: impl Fn(usize, usize) -> u8,
    ) -> Vec<usize> {
        let mut nodes: Vec<(usize, u8)> = self
            .host_node_llcs
            .keys()
            .map(|&node| (node, distance_fn(anchor, node)))
            .collect();
        // Sort: unreachable (255) last; among reachable, ascending
        // distance; ties broken by ascending node id via the stable
        // sort applied over a pre-sorted (BTreeMap-ordered) input.
        nodes.sort_by(|a, b| {
            let a_unreachable = a.1 == 255;
            let b_unreachable = b.1 == 255;
            match (a_unreachable, b_unreachable) {
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                _ => a.1.cmp(&b.1),
            }
        });
        nodes.into_iter().map(|(node, _)| node).collect()
    }

    /// NUMA node for a host LLC group, determined by majority vote of
    /// its CPUs' NUMA assignments. Returns 0 when the map is empty
    /// (single-node systems).
    ///
    /// Production callers pre-compute the node-to-LLC mapping once at
    /// [`HostTopology::from_sysfs`] via
    /// [`compute_host_node_llcs`](Self::compute_host_node_llcs)
    /// (memoized in [`host_node_llcs`](Self::host_node_llcs)); use
    /// [`host_llcs_by_numa_node`](Self::host_llcs_by_numa_node) to
    /// iterate the pre-built map. This method stays exposed for
    /// external callers (future `ktstr locks` NUMA column + any
    /// downstream tooling that needs a single-LLC lookup) and
    /// synthetic-topology tests that assert per-LLC node assignment.
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
    ///
    /// Implementation composes [`host_llcs_by_numa_node`] +
    /// [`numa_nodes_with_capacity`] — the same group-by-node + eligibility
    /// queries the `--cpu-cap` consolidation PLAN phase uses. The two
    /// callers' SELECTION algorithms differ (perf-mode does modulo
    /// rotation of guest onto host nodes; consolidation does
    /// score-driven greedy expansion), but the underlying topology
    /// lookups are the same.
    pub(crate) fn numa_aware_llc_order(
        &self,
        numa_nodes: u32,
        llcs: u32,
        llc_offset: usize,
    ) -> Vec<usize> {
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

        // Collect host NUMA nodes that can supply the ceiling (max)
        // per-node count — so any guest node can land there regardless
        // of whether it's one of the `remainder` "+1" nodes. Shared
        // primitive: `numa_nodes_with_capacity` filters the memoized
        // group-by-node map.
        let eligible_nodes = self.numa_nodes_with_capacity(max_per_node);

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

/// Default path prefix for LLC flock files. Production binds to
/// `/tmp/ktstr-llc-`; test builds can override via [`llc_lock_path`]
/// (see the `#[cfg(test)]` hook).
const LLC_LOCK_PREFIX: &str = "/tmp/ktstr-llc-";

/// Default path prefix for per-CPU flock files. Production binds to
/// `/tmp/ktstr-cpu-`; test builds can override via [`cpu_lock_path`]
/// (see the `#[cfg(test)]` hook).
const CPU_LOCK_PREFIX: &str = "/tmp/ktstr-cpu-";

#[cfg(test)]
thread_local! {
    /// Thread-local override for [`LLC_LOCK_PREFIX`]. Tests set this
    /// to a per-test tempdir so the acquire path operates on its
    /// own lockfile pool instead of padding the `LlcGroup` vector
    /// to 90,000+ entries just to avoid collision with production
    /// indices at 0..<host-llcs>. See tests `acquire_llc_plan_*`
    /// that build a small synth topo and point the prefix at a
    /// `TempDir`.
    static LLC_LOCK_PREFIX_OVERRIDE: std::cell::RefCell<Option<String>> =
        const { std::cell::RefCell::new(None) };

    /// Thread-local override for [`CPU_LOCK_PREFIX`]. Symmetric
    /// with [`LLC_LOCK_PREFIX_OVERRIDE`] for the per-CPU path.
    static CPU_LOCK_PREFIX_OVERRIDE: std::cell::RefCell<Option<String>> =
        const { std::cell::RefCell::new(None) };
}

/// Compose the LLC lockfile path for `llc_idx`. Production always
/// returns `{LLC_LOCK_PREFIX}{llc_idx}.lock`; tests can override the
/// prefix via [`LLC_LOCK_PREFIX_OVERRIDE`] to keep their lockfile
/// pool out of the real `/tmp/ktstr-llc-*` namespace.
fn llc_lock_path(llc_idx: usize) -> String {
    #[cfg(test)]
    {
        if let Some(p) = LLC_LOCK_PREFIX_OVERRIDE.with(|p| p.borrow().clone()) {
            return format!("{p}{llc_idx}.lock");
        }
    }
    format!("{LLC_LOCK_PREFIX}{llc_idx}.lock")
}

/// Compose the per-CPU lockfile path for `cpu`. Symmetric with
/// [`llc_lock_path`] — production binds to the hardcoded prefix;
/// tests can override via [`CPU_LOCK_PREFIX_OVERRIDE`].
fn cpu_lock_path(cpu: usize) -> String {
    #[cfg(test)]
    {
        if let Some(p) = CPU_LOCK_PREFIX_OVERRIDE.with(|p| p.borrow().clone()) {
            return format!("{p}{cpu}.lock");
        }
    }
    format!("{CPU_LOCK_PREFIX}{cpu}.lock")
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
        let path = llc_lock_path(llc_idx);
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
            let path = cpu_lock_path(host_cpu);
            match try_flock(&path, FlockMode::Exclusive) {
                Ok(Some(fd)) => locks.push(fd),
                Ok(None) => return Err(format!("CPU {host_cpu} busy")),
                Err(e) => return Err(format!("CPU {host_cpu}: {e}")),
            }
        }
        if let Some(cpu) = plan.service_cpu {
            let path = cpu_lock_path(cpu);
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

// ===========================================================================
// --cpu-cap PLAN pipeline — CpuCap / LlcSnapshot / LlcPlan + discover/plan/acquire
// ===========================================================================
//
// Entry point [`acquire_llc_plan`] is the single non-perf-mode
// reservation path: kernel builds and no-perf-mode VMs both call it
// with or without `--cpu-cap N`. `--cpu-cap` is a CPU-count budget:
// the planner reserves exactly N host CPUs by walking whole LLCs in
// contention- / NUMA-aware order and partial-taking the last LLC
// so `plan.cpus.len() == N`. The flock is per-LLC even when the
// last LLC is only partially used — coordination with concurrent
// ktstr peers is unchanged at LLC granularity. When `--cpu-cap`
// is absent the planner defaults to 30% of the calling process's
// sched_getaffinity cpuset (see [`default_cpu_budget`] and
// [`host_allowed_cpus`]) — not 30% of the host's online CPU count,
// because a CI runner whose parent cgroup pins ktstr to a 4-CPU
// subset must plan within THAT subset or sched_setaffinity on the
// resulting mask produces an empty effective set.
// Perf-mode never reaches this path; it stays on
// [`acquire_resource_locks`] for its `LOCK_EX` reservation contract.
//
// The pipeline has three phases: discover (snapshot holders per
// LLC), plan (NUMA-aware, consolidation-aware selection, filtered
// to the process's allowed cpuset), acquire (non-blocking `LOCK_SH`
// on each selected LLC). One TOCTOU retry absorbs the window between
// the discover snapshot and the non-blocking acquire; the second
// discover's /proc/locks read IS the backoff, so no sleep is needed
// between attempts.

/// Return the CPUs the calling process is allowed to run on, per
/// `sched_getaffinity(2)` with a `/proc/self/status` Cpus_allowed_list
/// fallback. Every consumer of the `--cpu-cap` pipeline plans against
/// this set instead of `HostTopology::online_cpus` so
/// `sched_setaffinity` on the plan's CPU list never produces an empty
/// effective mask under a cgroup-restricted runner (CI hosts, systemd
/// slices, sudo -u under a limited cpuset).
///
/// Returns an empty vec only when BOTH the syscall AND procfs fail —
/// a pathological host that can't enumerate its own affinity. Callers
/// treat that as a bail reason, not a fallback "every CPU" permission:
/// guessing on a misconfigured host is worse than failing visibly.
///
/// Tests override the return value via [`ALLOWED_CPUS_OVERRIDE`] so
/// the 30% default and allowed-cpu filtering are deterministic in
/// unit tests regardless of the CI runner's real cpuset.
pub(crate) fn host_allowed_cpus() -> Vec<usize> {
    #[cfg(test)]
    {
        if let Some(override_set) = ALLOWED_CPUS_OVERRIDE.with(|p| p.borrow().clone()) {
            return override_set;
        }
    }
    if let Some(cpus) = crate::host_state::read_affinity(0) {
        return cpus.into_iter().map(|c| c as usize).collect();
    }
    if let Ok(raw) = std::fs::read_to_string("/proc/self/status") {
        for line in raw.lines() {
            if let Some(v) = line.strip_prefix("Cpus_allowed_list:")
                && let Some(parsed) = crate::host_state::parse_cpu_list(v.trim())
            {
                return parsed.into_iter().map(|c| c as usize).collect();
            }
        }
    }
    Vec::new()
}

#[cfg(test)]
thread_local! {
    /// Test-only override for [`host_allowed_cpus`]. Set via
    /// [`AllowedCpusGuard`] to make 30%-of-allowed calculations and
    /// plan filtering deterministic in unit tests. Mirrors the
    /// [`LLC_LOCK_PREFIX_OVERRIDE`] pattern.
    pub(crate) static ALLOWED_CPUS_OVERRIDE: std::cell::RefCell<Option<Vec<usize>>> =
        const { std::cell::RefCell::new(None) };
}

/// Default CPU budget when `--cpu-cap` is not set: 30% of the
/// allowed-CPU count, rounded up, with a min-1 floor for small or
/// degenerate hosts. 30% leaves enough headroom for concurrent peers
/// (tests, builds) while still reserving a non-trivial slice; the
/// min-1 floor prevents returning 0 on a 1- or 2-CPU host, where
/// ceil(×0.30) ≥ 1 anyway — the `.max(1)` is defense in depth for
/// future ratio tweaks.
fn default_cpu_budget(allowed_cpus: usize) -> usize {
    allowed_cpus.saturating_mul(30).div_ceil(100).max(1)
}

/// Parsed `--cpu-cap N` value. N is a CPU count: the planner reserves
/// exactly N host CPUs by walking whole LLCs in contention- /
/// NUMA-aware order (filtered to the calling process's allowed
/// cpuset) and partial-taking the last LLC so `plan.cpus.len() == N`.
/// The flock set is still per-LLC (the last LLC is flocked whole
/// even when only a prefix of its CPUs enters `plan.cpus`).
/// Bounded to `1..=usize::MAX` at the constructor — a cap of 0 is
/// nonsensical (reserving zero CPUs is just "don't run") and
/// rejected upstream by the CLI layer, but we enforce the bound in
/// the type system via [`NonZeroUsize`] so callers can
/// `CpuCap::new(...)?` without a follow-up bounds check.
///
/// The runtime upper bound — "don't exceed the process's allowed
/// CPU count" — is enforced at acquire time via
/// [`CpuCap::effective_count`] because the allowed set is not known
/// until [`host_allowed_cpus`] reads `sched_getaffinity`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuCap {
    n: std::num::NonZeroUsize,
}

impl CpuCap {
    /// Construct from a raw `usize` CPU count. Returns `Err` on `0`;
    /// `usize::MAX` is accepted here and clamped later by
    /// [`effective_count`].
    pub fn new(n: usize) -> Result<Self> {
        std::num::NonZeroUsize::new(n)
            .map(|n| CpuCap { n })
            .ok_or_else(|| anyhow::anyhow!("--cpu-cap must be ≥ 1 CPU (got 0)"))
    }

    /// Three-tier resolution: explicit CLI flag wins over env var,
    /// which wins over "not set". Returns `None` when neither is
    /// present, meaning "use the 30% default of the allowed CPU set"
    /// (see [`default_cpu_budget`]).
    ///
    /// Env var is `KTSTR_CPU_CAP` (integer ≥ 1, CPU count). An empty
    /// or unset env var is treated as absent; a non-numeric value
    /// OR the numeric value `0` is an error — `KTSTR_CPU_CAP=0`
    /// flows through `CpuCap::new(0)` which rejects with "--cpu-cap
    /// must be ≥ 1 CPU (got 0)". Zero is not a silent fallback to
    /// "no cap"; it surfaces as a parse-time error so typos and
    /// scripting mistakes don't accidentally disable the resource
    /// contract.
    pub fn resolve(cli_flag: Option<usize>) -> Result<Option<CpuCap>> {
        if let Some(n) = cli_flag {
            return Ok(Some(CpuCap::new(n)?));
        }
        match std::env::var("KTSTR_CPU_CAP") {
            Ok(s) if s.is_empty() => Ok(None),
            Ok(s) => {
                let n: usize = s
                    .parse()
                    .with_context(|| format!("KTSTR_CPU_CAP is not a valid integer: {s:?}"))?;
                Ok(Some(CpuCap::new(n)?))
            }
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(std::env::VarError::NotUnicode(raw)) => {
                anyhow::bail!(
                    "KTSTR_CPU_CAP contains non-UTF-8 bytes ({} bytes): {raw:?}. \
                     Set an integer value or unset.",
                    raw.len(),
                )
            }
        }
    }

    /// Runtime-bounded cap: returns the inner count unless it exceeds
    /// `allowed_cpus` (the calling process's sched_getaffinity cpuset
    /// count), in which case a [`ResourceContention`] error steers
    /// the caller toward an actionable message. This check lives at
    /// acquire time — not at construction — because the allowed set
    /// is not known until [`host_allowed_cpus`] reads the syscall.
    pub fn effective_count(&self, allowed_cpus: usize) -> Result<usize> {
        let n = self.n.get();
        if n > allowed_cpus {
            return Err(anyhow::Error::new(ResourceContention {
                reason: format!(
                    "--cpu-cap N = {n} exceeds the {allowed_cpus} CPUs this \
                     process is allowed on (from sched_getaffinity / \
                     Cpus_allowed_list). Pick a cap ≤ {allowed_cpus}, release \
                     the cgroup/taskset constraint restricting this process, \
                     or omit --cpu-cap to use the 30% default of the allowed \
                     set."
                ),
            }));
        }
        Ok(n)
    }
}

/// Per-LLC discover snapshot: identity + current holder set.
/// Constructed by [`discover_llc_snapshots`] before the PLAN phase.
/// `pub(crate)` because the `ktstr locks` observational command
/// renders holder lists via the same snapshot structure; external
/// callers have no reason to construct one.
#[derive(Debug, Clone)]
pub(crate) struct LlcSnapshot {
    /// Host LLC index — matches [`HostTopology::llc_groups`] ordering.
    pub(crate) llc_idx: usize,
    /// Canonical `/tmp/ktstr-llc-{N}.lock` path. Stored so the ACQUIRE
    /// phase doesn't re-format the string per LLC.
    pub(crate) lockfile_path: std::path::PathBuf,
    /// Processes currently holding this LLC's flock (any mode). Empty
    /// when no peer holds the lock. Derived from a single `/proc/locks`
    /// read shared across every LLC in the discover phase.
    pub(crate) holders: Vec<crate::flock::HolderInfo>,
    /// `holders.len()`, cached so the PLAN sort can access it without
    /// re-traversing the holder list per candidate.
    pub(crate) holder_count: usize,
}

/// Output of [`acquire_llc_plan`]: the concrete LLC reservation plus
/// every piece of diagnostic context a downstream consumer could
/// want.
///
/// `mems` is the union of NUMA nodes containing the selected CPUs —
/// `BuildSandbox::try_create` writes this to the child cgroup's
/// `cpuset.mems` so memory allocations respect the same NUMA locality
/// the CPU reservation already implies.
///
/// `locks` holds the RAII file descriptors whose `OwnedFd::drop`
/// releases the kernel-side flock; the field is `pub(crate)` because
/// direct manipulation from outside the crate would defeat the drop
/// guarantee.
#[derive(Debug)]
pub struct LlcPlan {
    /// Selected host LLC indices, sorted ASCENDING. Acquire order
    /// matches this slice — two callers with the same target see the
    /// same ordering and converge on the same one-wins-the-others-retry
    /// livelock-proof sequence.
    pub locked_llcs: Vec<usize>,
    /// Flattened host CPU list, sized exactly `target_cpus`. The last
    /// locked LLC may contribute only a prefix of its allowed CPUs.
    /// Preserves LLC ordering: CPUs from `locked_llcs[0]` come
    /// before CPUs from `locked_llcs[1]`, etc.
    pub cpus: Vec<usize>,
    /// Union of NUMA nodes hosting the locked LLCs. When the plan
    /// spans > 1 node (cross-node spill — seed node exhausted, plan
    /// spilled to nearest-by-distance neighbors), `mems`
    /// contains every node — not just the seed node's.
    pub mems: std::collections::BTreeSet<usize>,
    /// Per-LLC discovery trail. Preserved through the lifetime of the
    /// plan so error-formatting (via `acquire_llc_plan`'s final
    /// fresh snapshot) and future `ktstr locks` rendering don't
    /// re-probe `/proc/locks`. In-tree consumers currently re-read
    /// the snapshot only on the TOCTOU failure path; the field is
    /// kept populated so downstream tooling can inspect the
    /// plan-at-acquire holder set without a second pass.
    #[allow(dead_code)]
    pub(crate) snapshot: Vec<LlcSnapshot>,
    /// RAII flock holders. Dropped when the plan goes out of scope,
    /// releasing each LLC's `LOCK_SH` in declared order.
    #[allow(dead_code)] // RAII only — Drop releases flocks, no reads.
    pub(crate) locks: Vec<std::os::fd::OwnedFd>,
}

/// Total wall-clock budget for PLAN + ACQUIRE under the consolidation
/// path. Each DISCOVER + PLAN + ACQUIRE attempt is essentially
/// non-blocking; the TOCTOU retry absorbs at most one racing peer.
/// No sleep is needed between the first and second attempt — the
/// second DISCOVER's `/proc/locks` read IS the backoff. If two attempts
/// fail, the contention is persistent and the caller should
/// nextest-retry / operator-wait.
const ACQUIRE_MAX_TOCTOU_RETRIES: u32 = 1;

/// DISCOVER phase — read-only LLC snapshot.
///
/// For every LLC on the host: stat the canonical lockfile (materializing
/// it with `O_CREAT | O_CLOEXEC | 0o666` if absent so subsequent
/// ACQUIRE has a stable inode), then parse one `/proc/locks` read to
/// populate every snapshot's holder list in a single pass. No flock
/// acquires — DISCOVER never contends.
///
/// `mountinfo` is the `/proc/self/mountinfo` text read once per
/// `acquire_llc_plan` invocation at [`acquire_llc_plan_with_acquire_fn`]
/// and threaded through here so a host with N LLCs pays for exactly
/// one mountinfo read per DISCOVER pass (DISCOVER is called twice
/// on the TOCTOU-exhausted diagnostic path, hence caching at the
/// plan level rather than per snapshot walk).
///
/// Returns `Ok(snapshots)` on success. Propagates opening + stat
/// errors so a missing `/tmp` or permission failure surfaces
/// actionably.
fn discover_llc_snapshots(topo: &HostTopology, mountinfo: &str) -> Result<Vec<LlcSnapshot>> {
    let mut snapshots: Vec<LlcSnapshot> = Vec::with_capacity(topo.llc_groups.len());
    for llc_idx in 0..topo.llc_groups.len() {
        let path = std::path::PathBuf::from(llc_lock_path(llc_idx));
        // Ensure the lockfile inode exists so `read_holders_with_mountinfo`
        // can key /proc/locks lookups on it. Deliberately takes no
        // flock — DISCOVER is observational. Also runs the NFS/FUSE
        // reject check inside `materialize`, so a misconfigured
        // `/tmp` mount surfaces here instead of silently at ACQUIRE
        // time.
        crate::flock::materialize(&path)?;
        let holders =
            crate::flock::read_holders_with_mountinfo(&path, mountinfo).unwrap_or_default();
        let holder_count = holders.len();
        snapshots.push(LlcSnapshot {
            llc_idx,
            lockfile_path: path,
            holders,
            holder_count,
        });
    }
    Ok(snapshots)
}

/// PLAN phase — NUMA-aware placement over discover snapshots.
///
/// Composite sort driven by three ordered keys:
///   1. Consolidation — prefer LLCs already holding peers.
///   2. NUMA locality — after seeding on the highest-scored LLC's
///      node, greedily fill the seed node before spilling.
///   3. LLC index ASC — tiebreak + final ACQUIRE ordering for livelock
///      safety.
///
/// `target_cpus` is the exact number of allowed CPUs the plan
/// reserves. The walk selects whole LLCs (filtered to their
/// allowed-CPU overlap) until the accumulated contribution meets
/// the budget. The LAST selected LLC may contribute more allowed
/// CPUs than the remaining budget needs; the materialization layer
/// at [`acquire_llc_plan_with_acquire_fn`] takes only the needed
/// prefix of that LLC's allowed CPUs into `plan.cpus`. The flock
/// is always held at LLC granularity — coordination with concurrent
/// ktstr peers happens per-LLC, regardless of how many of the LLC's
/// CPUs are consumed here. LLCs whose CPUs are all outside
/// `allowed` are skipped entirely — locking one would never
/// contribute a schedulable CPU to `plan.cpus`.
///
/// Distance fallback: callers without a distance matrix pass a closure
/// that returns `10` for equal nodes and `20` otherwise — primitive 3
/// keeps the spill order reasonable even on hosts whose
/// `/sys/devices/system/node/*/distance` is unavailable.
fn plan_from_snapshots(
    snapshots: &[LlcSnapshot],
    target_cpus: usize,
    topo: &HostTopology,
    allowed: &std::collections::BTreeSet<usize>,
    distance_fn: impl Fn(usize, usize) -> u8,
) -> Vec<usize> {
    if target_cpus == 0 {
        return Vec::new();
    }

    // Allowed-CPU count contributed by each LLC. An LLC with zero
    // overlap contributes no schedulable CPUs to `plan.cpus`, so
    // reserving it adds a useless flock and no planning value — drop
    // those up front so every subsequent walk only considers
    // candidates that can actually carry budget.
    let llc_allowed_cpus = |idx: usize| -> usize {
        topo.llc_groups[idx]
            .cpus
            .iter()
            .filter(|c| allowed.contains(c))
            .count()
    };
    let total_allowed_in_llcs: usize = (0..snapshots.len()).map(llc_allowed_cpus).sum();
    if target_cpus >= total_allowed_in_llcs {
        // Budget ≥ sum of per-LLC contributions: select every LLC
        // that has at least one allowed CPU, in ascending order.
        // Short-circuits the scoring walk when the cap degenerates
        // to "reserve everything we can schedule on."
        let mut all: Vec<usize> = (0..snapshots.len())
            .filter(|&idx| llc_allowed_cpus(idx) > 0)
            .collect();
        all.sort_unstable();
        return all;
    }

    // Step a: partition + sort. Only LLCs with at least one allowed
    // CPU are eligible — locking an out-of-cpuset LLC is useless.
    // Consolidation candidates first (holder_count DESC, llc_idx ASC);
    // fresh candidates after, sorted by llc_idx ASC. A single
    // composite sort would do the same work but the two-partition
    // form is easier to read and lets future "prefer consolidation
    // only if score ≥ threshold" tweaks slot in.
    let eligible = |s: &&LlcSnapshot| -> bool { llc_allowed_cpus(s.llc_idx) > 0 };
    let mut consolidation: Vec<&LlcSnapshot> = snapshots
        .iter()
        .filter(|s| s.holder_count > 0)
        .filter(eligible)
        .collect();
    let mut fresh: Vec<&LlcSnapshot> = snapshots
        .iter()
        .filter(|s| s.holder_count == 0)
        .filter(eligible)
        .collect();
    consolidation.sort_by(|a, b| {
        b.holder_count
            .cmp(&a.holder_count)
            .then(a.llc_idx.cmp(&b.llc_idx))
    });
    fresh.sort_by_key(|s| s.llc_idx);
    let ranked: Vec<&LlcSnapshot> = consolidation.into_iter().chain(fresh).collect();
    if ranked.is_empty() {
        // No LLC on this host overlaps the caller's allowed cpuset.
        // Bail upstream handles this as ResourceContention; here we
        // just return empty so the caller can surface the diagnostic.
        return Vec::new();
    }

    // Step b: seed. Highest-scored eligible LLC; its NUMA node
    // anchors the greedy expansion.
    let seed = ranked[0];
    let seed_node = topo.llc_numa_node(seed.llc_idx);

    // Step c–d: walk seed-node LLCs first, then spill to
    // nearest-by-distance nodes. Primitives 1 + 3 drive the node
    // ordering; the per-node LLC lists come from primitive 1. Within
    // each node, we still honour the composite score by walking
    // `ranked` and skipping LLCs not on the current target node.
    // Accumulation is by allowed-CPU contribution — an LLC with 4
    // CPUs of which 2 are in `allowed` counts as 2 toward the
    // budget and the other 2 never appear in `plan.cpus`.
    let node_order = topo.numa_nodes_sorted_by_distance(seed_node, distance_fn);
    let mut selected: Vec<usize> = Vec::new();
    let mut picked: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut accumulated: usize = 0;
    for node in node_order {
        if accumulated >= target_cpus {
            break;
        }
        // Ranked walk, taking every candidate on this node in
        // score-order until we've filled `target_cpus` or exhausted
        // the node.
        for snap in &ranked {
            if accumulated >= target_cpus {
                break;
            }
            if picked.contains(&snap.llc_idx) {
                continue;
            }
            if topo.llc_numa_node(snap.llc_idx) != node {
                continue;
            }
            selected.push(snap.llc_idx);
            picked.insert(snap.llc_idx);
            accumulated += llc_allowed_cpus(snap.llc_idx);
        }
    }

    // Step e: livelock-proof acquire order — ascending index.
    selected.sort_unstable();
    selected
}

/// ACQUIRE phase — non-blocking `LOCK_SH` on every selected LLC.
///
/// All-or-nothing. A single `EWOULDBLOCK` releases every held fd (via
/// `drop(locks)`) and returns `Ok(None)` so the caller re-runs
/// discover + plan with a fresh snapshot. Non-retryable errors
/// (unexpected errno, path open failures) propagate unchanged.
fn try_acquire_llc_plan_locks(
    selected: &[usize],
    snapshots: &[LlcSnapshot],
) -> Result<Option<Vec<std::os::fd::OwnedFd>>> {
    let mut locks: Vec<std::os::fd::OwnedFd> = Vec::with_capacity(selected.len());
    for &idx in selected {
        let snap = snapshots
            .iter()
            .find(|s| s.llc_idx == idx)
            .expect("selected index must come from snapshots — plan invariant");
        match crate::flock::try_flock(&snap.lockfile_path, FlockMode::Shared)? {
            Some(fd) => locks.push(fd),
            None => {
                // Drop previously-held fds so the peer racing us sees
                // a consistent post-bail state, then signal "retry".
                drop(locks);
                return Ok(None);
            }
        }
    }
    Ok(Some(locks))
}

/// Entry point for the `--cpu-cap` PLAN pipeline.
///
/// Runs DISCOVER → PLAN → ACQUIRE with at most one TOCTOU retry. On
/// success returns an [`LlcPlan`] holding the selected LLCs, their
/// flattened CPUs (intersected with the calling process's allowed
/// cpuset), the derived `mems` set, the diagnostic snapshot, and the
/// RAII flock handles.
///
/// `cpu_cap == None` means "reserve 30% of the allowed-CPU set" (see
/// [`default_cpu_budget`]). `cpu_cap == Some(cap)` where
/// `cap > allowed_cpus` errors at acquire time via
/// [`CpuCap::effective_count`]. The allowed-CPU set comes from
/// [`host_allowed_cpus`] — `sched_getaffinity(0)` with a procfs
/// fallback — so plans are always schedulable under cgroup-restricted
/// runners (CI hosts, systemd slices, sudo under a limited cpuset).
///
/// Consolidation uses the host distance matrix from [`TestTopology`]
/// so spill order matches actual NUMA cost. Hosts whose
/// `/sys/devices/system/node/*/distance` failed to parse degrade to a
/// numerically-adjacent ordering via the distance closure (`10` for
/// same-node, `20` for cross-node).
pub fn acquire_llc_plan(
    topo: &HostTopology,
    test_topo: &crate::topology::TestTopology,
    cpu_cap: Option<CpuCap>,
) -> Result<LlcPlan> {
    acquire_llc_plan_with_acquire_fn(topo, test_topo, cpu_cap, try_acquire_llc_plan_locks)
}

/// Parameterized form of [`acquire_llc_plan`] that takes the
/// ACQUIRE closure as a seam. Production calls this with
/// [`try_acquire_llc_plan_locks`] (non-blocking `LOCK_SH` per LLC);
/// tests can pass a closure that returns `Ok(None)` on attempt 0 and
/// forwards on attempt 1 to simulate a peer winning the first race,
/// or an attempt-counting closure that always fails to exercise the
/// retry-exhausted error path.
///
/// `acquire_fn` receives `(selected, snapshots)` and returns
/// `Ok(Some(locks))` on success, `Ok(None)` to trigger a retry, or
/// propagates hard errors unchanged. Production closure is the
/// free-standing [`try_acquire_llc_plan_locks`]; the test closure
/// can track its own attempt counter via interior mutability
/// ([`std::cell::Cell`], `Mutex`, atomic int).
///
/// The outer loop body — DISCOVER, PLAN, retry budget, final
/// holder diagnostics — is shared between both entry points so the
/// test seam exercises the exact retry-and-diagnose sequence
/// production uses, not a parallel implementation.
fn acquire_llc_plan_with_acquire_fn<F>(
    topo: &HostTopology,
    test_topo: &crate::topology::TestTopology,
    cpu_cap: Option<CpuCap>,
    mut acquire_fn: F,
) -> Result<LlcPlan>
where
    F: FnMut(&[usize], &[LlcSnapshot]) -> Result<Option<Vec<std::os::fd::OwnedFd>>>,
{
    // Resolve the calling process's allowed cpuset. Plans must fit
    // inside this set — sched_setaffinity against a mask outside the
    // process's cgroup cpuset either fails outright or produces an
    // empty effective set (the vCPU thread then cannot run). Reading
    // the syscall ONCE here and threading it through means every
    // TOCTOU retry sees the same baseline; a cgroup change mid-plan
    // is a host-reconfiguration event the retry budget does not
    // attempt to absorb.
    let allowed_vec = host_allowed_cpus();
    if allowed_vec.is_empty() {
        anyhow::bail!(
            "acquire_llc_plan: could not determine the calling process's \
             allowed CPU set (both sched_getaffinity and \
             /proc/self/status Cpus_allowed_list failed). Cannot plan a \
             reservation without knowing which CPUs are schedulable."
        );
    }
    let allowed: std::collections::BTreeSet<usize> = allowed_vec.iter().copied().collect();
    let allowed_cpus = allowed.len();

    let target_cpus = match cpu_cap {
        Some(cap) => cap.effective_count(allowed_cpus)?,
        None => default_cpu_budget(allowed_cpus),
    };
    if target_cpus == 0 {
        // Defense in depth. `default_cpu_budget` has a `.max(1)`
        // floor and `effective_count` on a `NonZeroUsize` cap can
        // never return 0, but surfacing this as an explicit bail
        // catches future regressions (e.g. someone wires a signed
        // integer into the budget math) instead of silently
        // producing a plan with no locks.
        anyhow::bail!("acquire_llc_plan: CPU budget resolved to zero");
    }

    // Read /proc/self/mountinfo ONCE per acquire_llc_plan invocation.
    // Every DISCOVER pass re-uses this text to derive per-LLC
    // /proc/locks needles (major:minor:inode). Without this cache, a
    // host with N LLCs would re-read mountinfo N× per DISCOVER pass,
    // and DISCOVER itself runs up to twice (once per attempt + once
    // on the retry-exhausted diagnostic path). Mount points are
    // effectively static during a plan acquisition — a bind mount
    // changing under us mid-acquire is a host-reconfiguration event
    // that invalidates every parallel acquirer anyway, not something
    // we need to re-read to observe.
    let mountinfo = crate::flock::read_mountinfo()?;

    let mut attempt: u32 = 0;
    loop {
        let snapshots = discover_llc_snapshots(topo, &mountinfo)?;
        let selected = plan_from_snapshots(&snapshots, target_cpus, topo, &allowed, |from, to| {
            test_topo.numa_distance(from, to)
        });
        if selected.is_empty() {
            // Every LLC's CPU set lies outside the allowed cpuset —
            // sysfs disagrees with sched_getaffinity. This is a host
            // misconfiguration (stale sysfs after hotplug, cgroup
            // pinned to a CPU range the kernel no longer reports in
            // llc_groups, etc.). Bail with actionable text rather
            // than looping through retries that cannot change the
            // outcome.
            anyhow::bail!(
                "acquire_llc_plan: no host LLC overlaps the process's \
                 {allowed_cpus}-CPU allowed set — sysfs LLC groups and \
                 sched_getaffinity disagree. Check for a stale \
                 /sys/devices/system/cpu view or a cgroup cpuset that \
                 excludes every LLC."
            );
        }
        match acquire_fn(&selected, &snapshots)? {
            Some(locks) => {
                // Success — materialize cpus + mems from the selected
                // indices, intersecting each LLC's CPU list with
                // `allowed` so `plan.cpus` never contains a CPU the
                // process cannot run on, and TRUNCATING at exactly
                // `target_cpus` so the last-LLC overshoot
                // contributes only the prefix the budget needs. The
                // full LLC is still flocked (the coordination unit
                // is per-LLC), but the CPUs beyond `target_cpus`
                // never appear in `plan.cpus` — sched_setaffinity
                // masks and cgroup cpuset.cpus writes reflect the
                // exact budget. `mems` collects the NUMA nodes of
                // CPUs that actually appear in `plan.cpus`; an LLC
                // that contributes a partial slice on a cross-node
                // split only registers the nodes of its
                // actually-used CPUs.
                let mut cpus: Vec<usize> = Vec::new();
                let mut mems: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
                'outer: for &idx in &selected {
                    let group = &topo.llc_groups[idx];
                    for &cpu in &group.cpus {
                        if !allowed.contains(&cpu) {
                            continue;
                        }
                        if cpus.len() >= target_cpus {
                            break 'outer;
                        }
                        cpus.push(cpu);
                        let node = topo.cpu_to_node.get(&cpu).copied().unwrap_or(0);
                        mems.insert(node);
                    }
                }
                return Ok(LlcPlan {
                    locked_llcs: selected,
                    cpus,
                    mems,
                    snapshot: snapshots,
                    locks,
                });
            }
            None => {
                if attempt >= ACQUIRE_MAX_TOCTOU_RETRIES {
                    // Rebuild holder diagnostics from a FRESH read so
                    // the error points at the peer that actually won.
                    let final_snapshots = discover_llc_snapshots(topo, &mountinfo)?;
                    let holders: Vec<String> = final_snapshots
                        .iter()
                        .filter(|s| !s.holders.is_empty())
                        .map(|s| {
                            format!(
                                "LLC {}: {}",
                                s.llc_idx,
                                crate::flock::format_holder_list(&s.holders)
                            )
                        })
                        .collect();
                    let holder_text = if holders.is_empty() {
                        "<none recorded>".to_string()
                    } else {
                        holders.join("; ")
                    };
                    return Err(anyhow::Error::new(ResourceContention {
                        reason: format!(
                            "acquire_llc_plan: could not reserve {target_cpus} \
                             CPU(s) after {retries} TOCTOU retry; holders: \
                             {holder_text}. Run `ktstr locks --json` to see \
                             every ktstr lock on this host.",
                            retries = ACQUIRE_MAX_TOCTOU_RETRIES + 1,
                        ),
                    }));
                }
                attempt += 1;
            }
        }
    }
}

/// Parallelism hint for `make -j{N}` when running under an
/// [`LlcPlan`] reservation. Returns the flattened host-CPU count
/// (`plan.cpus.len()`), clamped to at least 1 so a pathological empty
/// plan still produces a runnable command.
///
/// Rationale: without this hint, `make -j$(nproc)` fans gcc
/// children across every online CPU, defeating the --cpu-cap
/// reservation — the build escapes the cgroup cpuset in scheduling
/// terms even though the kernel enforces CPU membership. Passing
/// `plan.cpus.len()` to make keeps gcc's parallel width aligned with
/// the reserved capacity.
pub fn make_jobs_for_plan(plan: &LlcPlan) -> usize {
    plan.cpus.len().max(1)
}

/// Render selected LLC indices for user-facing warning text.
///
/// Format is compact and stable: `[0 (node 0), 2 (node 1)]` when the
/// host exposes NUMA information, `[0, 2]` on degraded hosts whose
/// `cpu_to_node` map is empty. Used by
/// [`warn_if_cross_node_spill`] to render the `ktstr: reserving LLCs
/// …` message when an `--cpu-cap` plan spills across nodes.
pub fn format_llc_list(locked: &[usize], topo: &HostTopology) -> String {
    let parts: Vec<String> = locked
        .iter()
        .map(|&idx| {
            if topo.cpu_to_node.is_empty() {
                idx.to_string()
            } else {
                let node = topo.llc_numa_node(idx);
                format!("{idx} (node {node})")
            }
        })
        .collect();
    format!("[{}]", parts.join(", "))
}

/// Emit the cross-node spill warning when an `--cpu-cap` plan's
/// `mems` set spans more than one NUMA node. No-op for single-node
/// plans.
///
/// `eprintln!`, not `tracing::warn!`: this is user-visible
/// UX feedback (the operator picked a cap that couldn't fit in one
/// NUMA node), not operational instrumentation. Fires at most once
/// per plan — there is nothing in the plan lifecycle that causes a
/// re-trigger. Single-node plans (including single-socket hosts and
/// caps that fit within a single node) never emit.
///
/// Placement: called by `kernel_build_pipeline` and friends right
/// after [`acquire_llc_plan`] returns, before the sandbox mount.
/// Extracting this into a helper rather than inlining at the call
/// site lets tests capture stderr and assert the format without
/// poking into the orchestrator internals.
pub fn warn_if_cross_node_spill(plan: &LlcPlan, topo: &HostTopology) {
    if should_warn_cross_node(&plan.mems) {
        eprintln!(
            "ktstr: reserving LLCs {list} across {n} NUMA nodes \
             (preferred single-node contiguous unavailable). Build \
             will run; memory-access latency may be higher.",
            list = format_llc_list(&plan.locked_llcs, topo),
            n = plan.mems.len(),
        );
    }
}

/// Pure predicate backing [`warn_if_cross_node_spill`]. Returns
/// `true` when the plan spans more than one NUMA node
/// (`mems.len() > 1`); the warning suppression for single-node
/// plans follows directly from this.
///
/// Split out so tests can pin the polarity of the single-node /
/// multi-node decision without capturing stderr. A refactor that
/// accidentally flipped the comparison (`>= 1` or `== 1`) would
/// either warn on every plan (noise) or never warn (silent cost),
/// both of which the test suite catches here before the stderr
/// capture layer sees it.
fn should_warn_cross_node(mems: &std::collections::BTreeSet<usize>) -> bool {
    mems.len() > 1
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
        let path = llc_lock_path(llc_idx);
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
        let path = cpu_lock_path(cpu);
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

    // ─── SYNTHETIC-TOPOLOGY OFFSET CONVENTION ────────────────────
    //
    // Tests in this module that touch real `/tmp/ktstr-llc-*.lock`
    // files choose LLC indices in the 90000..=99999 range to avoid
    // collision with any real host's LLC count (modern server
    // sockets top out around 1024 LLCs). Per-test offsets are
    // subdivided by 100:
    //   90000-90999: acquire_resource_locks / per-CPU path tests
    //   91000-91999: acquire_cpu_locks tests
    //   92000-92999: reserved
    //   93000-93999: acquire_llc_plan (none-cap, EX-peer, SH-peer)
    //                — each sub-test picks its own sub-range
    //                (93000-93099, 93100-93199, …) so leaked state
    //                from a panicking prior test doesn't cross-
    //                contaminate.
    //   94000-94099: acquire_llc_plan cross-node spill (mems union
    //                invariant, I1).
    //   94100-99999: reserved for future LLC-level tests.
    // Tests that build a HostTopology in memory but do NOT touch
    // real /tmp paths use small indices (0, 1, 2, …) because no
    // cross-process collision is possible.
    //
    // When adding a new test that flocks under /tmp, pick an
    // unused 100-entry sub-range in 90000-99999 and document the
    // claim in a comment at the test site so the next author
    // doesn't accidentally re-use it.
    // ─────────────────────────────────────────────────────────────

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

    /// Backwards-compat helper: builds a synthetic HostTopology from
    /// LLC-group CPU lists, assigning each group to a NUMA node equal
    /// to its positional index (LLC 0 → node 0, LLC 1 → node 1, …).
    /// Delegates to [`HostTopology::new_for_tests`].
    ///
    /// Kept as a thin wrapper so the many existing call sites that
    /// pass only CPU lists (no explicit NUMA info) don't have to
    /// thread node ids through their parameter lists.
    fn synthetic_topo(groups: Vec<Vec<usize>>) -> HostTopology {
        let tagged: Vec<(Vec<usize>, usize)> = groups
            .into_iter()
            .enumerate()
            .map(|(node, cpus)| (cpus, node))
            .collect();
        HostTopology::new_for_tests(&tagged)
    }

    #[test]
    fn compute_pinning_single_llc() {
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
    fn compute_pinning_two_llcs() {
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
    fn compute_pinning_with_smt() {
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
    fn compute_pinning_exact_fit() {
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
    fn compute_pinning_error_too_many_vcpus() {
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
    fn compute_pinning_error_too_many_llcs() {
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
    fn compute_pinning_error_llc_too_small() {
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
    fn compute_pinning_no_cross_llc_sharing() {
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
    fn compute_pinning_all_assignments_unique() {
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
    fn compute_pinning_vcpu_ids_sequential() {
        let topo = synthetic_topo(vec![vec![0, 1, 2, 3]]);
        let plan = topo
            .compute_pinning(&Topology::new(1, 1, 4, 1), false, 0)
            .unwrap();
        let vcpu_ids: Vec<u32> = plan.assignments.iter().map(|a| a.0).collect();
        assert_eq!(vcpu_ids, vec![0, 1, 2, 3]);
    }

    #[test]
    fn compute_pinning_single_vcpu() {
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
    fn compute_pinning_service_cpu_picks_unpinned() {
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
    fn compute_pinning_service_cpu_false_returns_none() {
        let topo = synthetic_topo(vec![vec![0, 1, 2, 3]]);
        let plan = topo
            .compute_pinning(&Topology::new(1, 1, 2, 1), false, 0)
            .unwrap();
        assert!(plan.service_cpu.is_none());
    }

    #[test]
    fn compute_pinning_service_cpu_exact_fit() {
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
    fn compute_pinning_service_cpu_insufficient_fails() {
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
    fn compute_pinning_service_cpu_multi_llc() {
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

    /// Backwards-compat helper: builds a synthetic HostTopology from
    /// `(numa_node, cpu_list)` pairs. Delegates to
    /// [`HostTopology::new_for_tests`], flipping the tuple order to
    /// `(cpus, node)` so the underlying constructor presents a
    /// consistent `(cpus, node)` shape to callers that build pairs
    /// directly.
    fn synthetic_topo_numa(groups: Vec<(usize, Vec<usize>)>) -> HostTopology {
        let tagged: Vec<(Vec<usize>, usize)> = groups
            .into_iter()
            .map(|(node, cpus)| (cpus, node))
            .collect();
        HostTopology::new_for_tests(&tagged)
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
    fn compute_pinning_numa_two_nodes() {
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
    fn compute_pinning_numa_fallback_insufficient_nodes() {
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
    fn compute_pinning_numa_single_node_unchanged() {
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
    fn compute_pinning_numa_three_nodes() {
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
    fn compute_pinning_numa_with_service_cpu() {
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
        // Empty cpu_to_node should default to node 0 (implicit via
        // the unwrap_or(0) in `llc_numa_node`).
        //
        // Construct manually rather than via `new_for_tests` because
        // this test pins behavior when `cpu_to_node` is EMPTY — our
        // fixture helper always populates node ids. Emptying
        // `cpu_to_node` after a `new_for_tests` call is clearer than
        // a special-case seam.
        let mut topo = HostTopology::new_for_tests(&[(vec![0, 1], 0)]);
        topo.cpu_to_node.clear();
        topo.host_node_llcs.clear();
        assert_eq!(topo.llc_numa_node(0), 0);
    }

    // -- llc_offset pinning tests --

    #[test]
    fn compute_pinning_offset_single_llc_wraps() {
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
    fn compute_pinning_offset_two_llcs_shifts() {
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
    fn compute_pinning_offset_wraps_modulo() {
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
    fn compute_pinning_offset_three_llcs_partial() {
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
    fn compute_pinning_offset_large_wraps() {
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
    fn compute_pinning_offset_numa_within_rotation() {
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
    fn compute_pinning_offset_numa_node_rotation() {
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
    fn compute_pinning_offset_with_service_cpu() {
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
    fn compute_pinning_offset_numa_combined_rotation() {
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
        let locks = match acquire_cpu_locks(3, 100, None) {
            Ok(l) => l,
            Err(e) if e.downcast_ref::<ResourceContention>().is_some() => {
                panic!("{e}");
            }
            Err(e) => panic!("{e:#}"),
        };
        assert_eq!(locks.len(), 3);
    }

    #[test]
    fn cpu_lock_acquire_slides_past_held() {
        cleanup_lock("/tmp/ktstr-cpu-0.lock");
        let holder = try_flock("/tmp/ktstr-cpu-0.lock", FlockMode::Exclusive)
            .unwrap()
            .unwrap();

        let locks = match acquire_cpu_locks(2, 100, None) {
            Ok(l) => l,
            Err(e) if e.downcast_ref::<ResourceContention>().is_some() => {
                drop(holder);
                cleanup_lock("/tmp/ktstr-cpu-0.lock");
                panic!("{e}");
            }
            Err(e) => panic!("{e:#}"),
        };
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
        // Uses a per-test lockfile prefix so the LLC group can sit
        // at index 0 instead of padding to 92000. The production
        // `acquire_cpu_locks` path threads through `llc_lock_path`,
        // which honors the test-only prefix override.
        let _prefix = LlcLockPrefixGuard::new();
        let cpu_prefix_dir = tempfile::TempDir::new().expect("tempdir");
        let cpu_prefix = format!("{}/cpu-", cpu_prefix_dir.path().display());
        CPU_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = Some(cpu_prefix));

        struct CpuPrefixGuard;
        impl Drop for CpuPrefixGuard {
            fn drop(&mut self) {
                CPU_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = None);
            }
        }
        let _cpu_prefix = CpuPrefixGuard;

        let topo = HostTopology::new_for_tests(&[((0..100).collect(), 0)]);

        let locks = match acquire_cpu_locks(2, 100, Some(&topo)) {
            Ok(l) => l,
            Err(e) if e.downcast_ref::<ResourceContention>().is_some() => {
                panic!("{e}");
            }
            Err(e) => panic!("{e:#}"),
        };
        assert_eq!(locks.len(), 3);

        // The LLC lock is shared — another shared should coexist.
        let llc_path = llc_lock_path(0);
        let shared2 = try_flock(&llc_path, FlockMode::Shared)
            .unwrap()
            .expect("second shared LLC should coexist");
        // Exclusive should fail while shared is held.
        let excl = try_flock(&llc_path, FlockMode::Exclusive).unwrap();
        assert!(
            excl.is_none(),
            "exclusive LLC should fail while shared is held",
        );

        drop(shared2);
        drop(locks);
        drop(cpu_prefix_dir);
    }

    #[test]
    fn cpu_lock_llc_shared_protection() {
        // Tests acquire_llc_shared_locks directly: verifies shared lock
        // acquired, shared coexistence, and exclusive blocking.
        // Uses a per-test lockfile prefix so the LLC group can sit
        // at index 0 with real CPU ids (no 92100-entry padding).
        let _prefix = LlcLockPrefixGuard::new();
        let topo = HostTopology::new_for_tests(&[(vec![91200, 91201], 0)]);

        let cpus = vec![91200usize, 91201];
        let llc_locks = acquire_llc_shared_locks(&topo, &cpus).unwrap();
        assert_eq!(llc_locks.len(), 1);

        let llc_path = llc_lock_path(0);
        let shared2 = try_flock(&llc_path, FlockMode::Shared)
            .unwrap()
            .expect("second shared LLC should coexist");
        let excl = try_flock(&llc_path, FlockMode::Exclusive).unwrap();
        assert!(
            excl.is_none(),
            "exclusive LLC should fail while shared is held",
        );

        drop(shared2);
        drop(llc_locks);
    }

    /// RAII guard for a per-test LLC lockfile path prefix. Installs
    /// a `{tempdir}/llc-` prefix into [`LLC_LOCK_PREFIX_OVERRIDE`]
    /// on construction and unsets it on Drop. Two parallel tests
    /// using this guard each get their own tempdir, so their
    /// `acquire_llc_plan` lockfiles can't collide. Eliminates the
    /// 90K+ empty `LlcGroup` padding that earlier tests used to
    /// sidestep collision with real host LLC indices.
    ///
    /// Uses [`tempfile::TempDir`] so cleanup runs via RAII on panic
    /// — a panicking test can't leak `/tmp` lockfiles into other
    /// test runs.
    struct LlcLockPrefixGuard {
        _dir: tempfile::TempDir,
    }

    impl LlcLockPrefixGuard {
        fn new() -> Self {
            let dir = tempfile::TempDir::new().expect("tempdir");
            let prefix = format!("{}/llc-", dir.path().display());
            LLC_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = Some(prefix));
            LlcLockPrefixGuard { _dir: dir }
        }
    }

    impl Drop for LlcLockPrefixGuard {
        fn drop(&mut self) {
            LLC_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = None);
        }
    }

    /// RAII guard for a per-test override of
    /// [`host_allowed_cpus`]'s return value via
    /// [`ALLOWED_CPUS_OVERRIDE`]. Lets tests pin the 30%-default and
    /// allowed-cpu filtering math to a known input regardless of
    /// what the CI runner's real sched_getaffinity returns. Unset on
    /// Drop so a panicking test cannot leak state across the suite.
    struct AllowedCpusGuard;

    impl AllowedCpusGuard {
        fn new(cpus: Vec<usize>) -> Self {
            ALLOWED_CPUS_OVERRIDE.with(|p| *p.borrow_mut() = Some(cpus));
            AllowedCpusGuard
        }
    }

    impl Drop for AllowedCpusGuard {
        fn drop(&mut self) {
            ALLOWED_CPUS_OVERRIDE.with(|p| *p.borrow_mut() = None);
        }
    }

    /// `acquire_llc_plan` with `cpu_cap: None` reserves exactly 30%
    /// of the allowed-CPU set (ceiling), walking whole LLCs and
    /// partial-taking the last LLC's CPUs when the budget falls
    /// mid-LLC. On a 10-CPU host split across 5 LLCs (2 CPUs each)
    /// where every CPU is in the allowed set: ceil(10 * 0.30) = 3
    /// CPUs → flock 2 LLCs (the first LLC's 2 CPUs + 1 CPU from
    /// the second), `plan.cpus` holds exactly 3 CPUs.
    ///
    /// Uses a per-test lockfile prefix via [`LlcLockPrefixGuard`] so
    /// the `LlcGroup` vector can be a small 5-entry topology rather
    /// than padding to host-LLC-count slots. Production path runs
    /// through [`llc_lock_path`] which honors the test-only override.
    /// Uses [`AllowedCpusGuard`] to pin the allowed-CPU set so the
    /// 30%-default math is deterministic regardless of the CI
    /// runner's real sched_getaffinity.
    #[test]
    fn acquire_llc_plan_none_cap_reserves_thirty_percent_cpus() {
        let _prefix = LlcLockPrefixGuard::new();
        let _allowed = AllowedCpusGuard::new(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let topo = HostTopology::new_for_tests(&[
            (vec![0, 1], 0),
            (vec![2, 3], 0),
            (vec![4, 5], 0),
            (vec![6, 7], 0),
            (vec![8, 9], 0),
        ]);

        // synthetic() needs >= num_cpus >= num_llcs; the distance
        // function is never invoked with target_cpus >= sum-of-allowed
        // (the planner's short-circuit at plan_from_snapshots), so
        // the TestTopology's shape doesn't matter beyond "valid".
        let test_topo = crate::topology::TestTopology::synthetic(4, 1);

        let plan = acquire_llc_plan(&topo, &test_topo, None)
            .expect("clean pool must allow SH on every selected LLC");
        // 30% of 10 CPUs = ceil(3.0) = 3 CPUs. 2-CPU LLCs: LLC 0
        // contributes 2, LLC 1 contributes 1 (partial-take), total
        // exactly 3.
        assert_eq!(
            plan.locked_llcs.len(),
            2,
            "budget of 3 CPUs flocks 2 LLCs (2 CPUs + 1 partial): {:?}",
            plan.locked_llcs,
        );
        assert_eq!(
            plan.cpus.len(),
            3,
            "plan.cpus is truncated to exactly the budget: {:?}",
            plan.cpus,
        );
        assert_eq!(plan.locks.len(), 2, "one fd per selected LLC");
    }

    /// `acquire_llc_plan` bails with `ResourceContention` when ANY
    /// target LLC is held `LOCK_EX` by a peer (the path a perf-mode
    /// VM takes via `acquire_resource_locks`). The error chain must
    /// name the busy LLC index so an operator running `fuser
    /// {lockfile}` can trace the holder.
    ///
    /// Scope: pins ONE attempt of the EX-blocks-SH invariant — a
    /// single DISCOVER round-trip. The full Tier-1 / Tier-2
    /// coordination contract (retry budget, TOCTOU recovery,
    /// holder-diagnostic freshness) is covered piecewise by the
    /// retry-budget pin and the coexistence test; this test's
    /// narrow claim is: when EX is held, SH fails fast with an
    /// actionable error that names the LLC.
    #[test]
    fn acquire_llc_plan_bails_on_exclusive_peer() {
        let _prefix = LlcLockPrefixGuard::new();
        let _allowed = AllowedCpusGuard::new(vec![0]);
        let topo = HostTopology::new_for_tests(&[(vec![0], 0)]);

        // Peer holds EX on LLC 0's lockfile through the overridden
        // prefix. Acquire the path after setting the prefix so the
        // held lock matches what the planner will try to acquire.
        let busy_path = llc_lock_path(0);
        let _peer_ex = try_flock(&busy_path, FlockMode::Exclusive)
            .unwrap()
            .expect("peer EX must acquire on clean pool");

        let test_topo = crate::topology::TestTopology::synthetic(4, 1);
        let err = acquire_llc_plan(&topo, &test_topo, None)
            .expect_err("EX peer must block SH acquisition of the only LLC");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("LLC 0"),
            "error must name the busy LLC index so fuser can trace: {rendered}",
        );
        // Verify the ResourceContention tag survives so callers
        // pattern-matching on the error type (for nextest-retry
        // routing) see it correctly.
        assert!(
            err.downcast_ref::<ResourceContention>().is_some(),
            "error must downcast to ResourceContention for retry routing: {rendered}",
        );

        drop(_peer_ex);
    }

    /// Two no-perf-mode peers coexist: both acquire `acquire_llc_plan`
    /// successfully because `LOCK_SH` is reentrant. The contract says
    /// "shared holders coexist; exclusive blocks" — this pins the
    /// shared-coexistence half, complementing the EX-blocks-SH test
    /// above.
    #[test]
    fn acquire_llc_plan_coexists_with_shared_peer() {
        let _prefix = LlcLockPrefixGuard::new();
        let _allowed = AllowedCpusGuard::new(vec![0]);
        let topo = HostTopology::new_for_tests(&[(vec![0], 0)]);
        let shared_path = llc_lock_path(0);

        // First peer: SH. Simulates an already-running no-perf-mode VM.
        let _peer_sh = try_flock(&shared_path, FlockMode::Shared)
            .unwrap()
            .expect("peer SH must acquire on clean pool");

        let test_topo = crate::topology::TestTopology::synthetic(4, 1);
        let plan = acquire_llc_plan(&topo, &test_topo, None)
            .expect("second SH caller must coexist with the first");
        assert_eq!(
            plan.locks.len(),
            topo.llc_groups.len(),
            "second SH caller must acquire one fd per LLC group",
        );
    }

    // ---------------------------------------------------------------
    // CpuCap — construction, env resolution, acquire-time bounding
    // ---------------------------------------------------------------

    /// Serialize KTSTR_CPU_CAP env-var mutation across test threads.
    /// std::env::set_var is process-wide (unsafe in edition 2024);
    /// parallel tests would race if each mutated the same variable
    /// without coordination. Every env-touching test below takes
    /// this mutex for the duration of the test body.
    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        use std::sync::{Mutex, OnceLock};
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        // `lock().unwrap()` would panic on poison from an earlier
        // panicking test, cascading failures. Recover by taking the
        // inner guard — the test that panicked already failed; the
        // current test's env cleanup still runs.
        ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// RAII guard for scoped `std::env::set_var` mutation inside a
    /// test. On construction sets the variable to `value`; on Drop
    /// removes it regardless of whether the test body panicked or
    /// returned early. Pairs with [`env_lock`] — callers take the
    /// mutex first, then mint the guard, so two env-touching tests
    /// never observe each other's intermediate state.
    ///
    /// Replaces the bare `unsafe { set_var(..) } ... unsafe {
    /// remove_var(..) }` pairs that appeared in every env-set test:
    /// an early return or panic between the set and the remove used
    /// to leak the env var into subsequent tests serialized on the
    /// same mutex. `Drop` closes that leak.
    struct EnvGuard {
        name: &'static str,
    }

    impl EnvGuard {
        /// Set `name=value` under the assumed-held `env_lock` mutex.
        /// The caller must have taken `env_lock()` before calling
        /// this constructor — `EnvGuard` does NOT take the mutex
        /// itself because some tests need to interleave multiple
        /// guards (e.g. set, read, remove, re-set) within a single
        /// lock scope.
        fn set(name: &'static str, value: &str) -> Self {
            // SAFETY: caller holds the env_lock mutex; edition 2024
            // set_var is unsafe-marked because it races with reads
            // from other threads, but the mutex serializes every
            // env-touching test so no other test is reading
            // concurrently.
            unsafe {
                std::env::set_var(name, value);
            }
            EnvGuard { name }
        }

        /// Remove `name` under the assumed-held `env_lock` mutex.
        /// Symmetric helper for tests that want to start from a
        /// known-unset state without first creating a set-and-drop
        /// guard.
        fn remove(name: &'static str) -> Self {
            // SAFETY: caller holds the env_lock mutex; see set().
            unsafe {
                std::env::remove_var(name);
            }
            EnvGuard { name }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            // SAFETY: guard lifetime is bounded by env_lock held by
            // the test that constructed it. Drop runs before the
            // mutex guard is released, so the remove_var happens
            // under the same mutex as the matching set_var.
            unsafe {
                std::env::remove_var(self.name);
            }
        }
    }

    /// `CpuCap::new(0)` must reject with the "≥ 1 (got 0)" message.
    /// Zero is a scripting-mistake sentinel — silent acceptance would
    /// disable the resource contract.
    #[test]
    fn cpu_cap_new_rejects_zero() {
        let err = CpuCap::new(0).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("≥ 1"), "msg={msg}");
        assert!(msg.contains("got 0"), "msg={msg}");
    }

    /// `CpuCap::new(1)` succeeds — minimum legal cap.
    #[test]
    fn cpu_cap_new_accepts_one() {
        let cap = CpuCap::new(1).expect("cap of 1 must succeed");
        assert_eq!(cap.effective_count(4).unwrap(), 1);
    }

    /// `CpuCap::new(usize::MAX)` is accepted at construction time
    /// and clamped later by `effective_count`. Pins the contract
    /// that construction never consults the host.
    #[test]
    fn cpu_cap_new_accepts_usize_max() {
        let cap = CpuCap::new(usize::MAX).expect("MAX accepted at construction");
        // Actual clamping surfaces at effective_count; see
        // `cpu_cap_effective_count_exceeds_host` below.
        assert!(cap.effective_count(usize::MAX).is_ok());
    }

    /// `effective_count` returns the inner value when it fits.
    #[test]
    fn cpu_cap_effective_count_fits() {
        let cap = CpuCap::new(3).unwrap();
        assert_eq!(cap.effective_count(4).unwrap(), 3);
        assert_eq!(cap.effective_count(3).unwrap(), 3);
    }

    /// `effective_count` when cap exceeds the allowed-CPU count
    /// returns a `ResourceContention` error naming both numbers, so
    /// the operator can fix the flag without re-running `ktstr topo`.
    #[test]
    fn cpu_cap_effective_count_exceeds_host() {
        let cap = CpuCap::new(8).unwrap();
        let err = cap.effective_count(4).expect_err("8 > 4 must error");
        let msg = format!("{err:#}");
        assert!(msg.contains("8"), "msg must name requested cap: {msg}");
        assert!(msg.contains("4"), "msg must name allowed-CPU count: {msg}");
        // Must downcast to ResourceContention for nextest-retry
        // routing per the Tier-1/Tier-2 contract.
        assert!(
            err.downcast_ref::<ResourceContention>().is_some(),
            "must be a ResourceContention for retry routing: {msg}",
        );
    }

    /// `effective_count` at the boundary: cap == allowed_cpus is OK.
    #[test]
    fn cpu_cap_effective_count_at_host_boundary() {
        let cap = CpuCap::new(4).unwrap();
        assert_eq!(cap.effective_count(4).unwrap(), 4);
    }

    /// CLI flag supplied → wins over env var. `resolve(Some(N))`
    /// ignores `KTSTR_CPU_CAP` entirely. Pins the precedence
    /// contract documented on `CpuCap::resolve`.
    #[test]
    fn cpu_cap_resolve_cli_wins_over_env() {
        let _lock = env_lock();
        let _env = EnvGuard::set("KTSTR_CPU_CAP", "99");
        let cap = CpuCap::resolve(Some(3)).unwrap().expect("CLI flag set");
        assert_eq!(cap.effective_count(4).unwrap(), 3, "CLI wins");
    }

    /// No CLI flag, no env var → `None` (the 30%-of-allowed default
    /// is applied at acquire time — `resolve` never synthesizes a
    /// cap here).
    #[test]
    fn cpu_cap_resolve_no_cli_no_env_returns_none() {
        let _lock = env_lock();
        let _env = EnvGuard::remove("KTSTR_CPU_CAP");
        assert!(CpuCap::resolve(None).unwrap().is_none());
    }

    /// Env var set to a valid integer, no CLI flag → resolves to
    /// that value.
    #[test]
    fn cpu_cap_resolve_env_set() {
        let _lock = env_lock();
        let _env = EnvGuard::set("KTSTR_CPU_CAP", "2");
        let cap = CpuCap::resolve(None)
            .expect("resolve must succeed")
            .expect("env-set cap must yield Some");
        assert_eq!(cap.effective_count(8).unwrap(), 2);
    }

    /// Env var set to the empty string → treated as absent
    /// (matches `Ok(s) if s.is_empty()` arm).
    #[test]
    fn cpu_cap_resolve_empty_env_is_absent() {
        let _lock = env_lock();
        let _env = EnvGuard::set("KTSTR_CPU_CAP", "");
        assert!(CpuCap::resolve(None).unwrap().is_none());
    }

    /// Env var set to a non-numeric value → parse error with the
    /// variable name in the message.
    #[test]
    fn cpu_cap_resolve_non_numeric_env_errors() {
        let _lock = env_lock();
        let _env = EnvGuard::set("KTSTR_CPU_CAP", "not-a-number");
        let err = CpuCap::resolve(None).expect_err("non-numeric must error");
        let msg = format!("{err:#}");
        assert!(msg.contains("KTSTR_CPU_CAP"), "msg={msg}");
    }

    /// Env var set to `"0"` flows through `CpuCap::new(0)` and
    /// surfaces the same "--cpu-cap must be ≥ 1 (got 0)" error.
    /// Regression guard: typos like `KTSTR_CPU_CAP=0` must NOT
    /// silently fall back to "no cap".
    #[test]
    fn cpu_cap_resolve_zero_env_rejected() {
        let _lock = env_lock();
        let _env = EnvGuard::set("KTSTR_CPU_CAP", "0");
        let err = CpuCap::resolve(None).expect_err("zero must error");
        let msg = format!("{err:#}");
        assert!(msg.contains("≥ 1"), "msg={msg}");
        assert!(msg.contains("got 0"), "msg={msg}");
    }

    /// CLI flag of 0 is the same rejection path as env var of 0 —
    /// both feed `CpuCap::new(0)`. Pins that precedence doesn't
    /// let a valid env var "save" an invalid CLI zero.
    #[test]
    fn cpu_cap_resolve_zero_cli_rejected_even_with_valid_env() {
        let _lock = env_lock();
        let _env = EnvGuard::set("KTSTR_CPU_CAP", "2");
        let err = CpuCap::resolve(Some(0)).expect_err("cli=0 must error");
        let msg = format!("{err:#}");
        assert!(msg.contains("≥ 1"), "msg={msg}");
    }

    /// `EnvGuard::set` applies the value, and `Drop` removes the
    /// variable even if the test body panics mid-scope. Pins the
    /// RAII contract so a refactor that accidentally drops the
    /// Drop impl leaks env state across tests.
    #[test]
    fn env_guard_set_and_drop_removes_variable() {
        let _lock = env_lock();
        let probe = "KTSTR_CPU_CAP_ENV_GUARD_TEST";
        {
            let _env = EnvGuard::set(probe, "abc");
            assert_eq!(
                std::env::var(probe).ok().as_deref(),
                Some("abc"),
                "set must apply immediately",
            );
        }
        // Drop ran — variable must be gone.
        assert!(
            std::env::var(probe).is_err(),
            "EnvGuard::drop must remove the variable",
        );
    }

    // ---------------------------------------------------------------
    // NUMA primitives — host_llcs_by_numa_node / with_capacity /
    // sorted_by_distance
    // ---------------------------------------------------------------

    /// Backwards-compat helper: forwards to
    /// [`HostTopology::new_for_tests`]. Kept so existing tests that
    /// reference `synth_host_topo` don't need to be renamed in lock-
    /// step with the consolidation — the single authoritative
    /// constructor is `new_for_tests`, this and
    /// [`synthetic_topo`] / [`synthetic_topo_numa`] are thin adapters
    /// over it.
    fn synth_host_topo(groups: &[(Vec<usize>, usize)]) -> HostTopology {
        HostTopology::new_for_tests(groups)
    }

    /// Single-node host: one entry in host_llcs_by_numa_node with
    /// every LLC index in ascending order.
    #[test]
    fn host_llcs_by_numa_node_single_node() {
        let topo = synth_host_topo(&[(vec![0, 1], 0), (vec![2, 3], 0), (vec![4, 5], 0)]);
        let map = topo.host_llcs_by_numa_node();
        assert_eq!(map.len(), 1, "single-node host has one entry");
        assert_eq!(map.get(&0), Some(&vec![0, 1, 2]));
    }

    /// Dual-node host: two entries, each with its own LLC indices
    /// in ascending order.
    #[test]
    fn host_llcs_by_numa_node_dual_node() {
        let topo = synth_host_topo(&[
            (vec![0, 1], 0),
            (vec![2, 3], 1),
            (vec![4, 5], 0),
            (vec![6, 7], 1),
        ]);
        let map = topo.host_llcs_by_numa_node();
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&0), Some(&vec![0, 2]));
        assert_eq!(map.get(&1), Some(&vec![1, 3]));
    }

    /// Asymmetric: node 0 has 3 LLCs, node 1 has 1 LLC.
    /// `numa_nodes_with_capacity(2)` returns only node 0.
    #[test]
    fn numa_nodes_with_capacity_asymmetric() {
        let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 0), (vec![2], 0), (vec![3], 1)]);
        let cap2: Vec<usize> = topo
            .numa_nodes_with_capacity(2)
            .into_iter()
            .map(|(node, _)| node)
            .collect();
        assert_eq!(cap2, vec![0], "only node 0 has ≥ 2 LLCs");
        let cap1: Vec<usize> = topo
            .numa_nodes_with_capacity(1)
            .into_iter()
            .map(|(node, _)| node)
            .collect();
        assert_eq!(cap1, vec![0, 1], "both nodes have ≥ 1 LLC");
    }

    /// `numa_nodes_with_capacity` with min_llcs > every node's
    /// count returns empty — no candidates.
    #[test]
    fn numa_nodes_with_capacity_over_max_returns_empty() {
        let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 1)]);
        assert!(topo.numa_nodes_with_capacity(99).is_empty());
    }

    /// `numa_nodes_sorted_by_distance` with identity closure:
    /// anchor == node → 10, else 20. Anchor sorts first; remaining
    /// nodes preserve BTreeMap ascending order (stable sort over
    /// equal distances).
    #[test]
    fn numa_nodes_sorted_by_distance_identity_closure() {
        let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 1), (vec![2], 2)]);
        let order =
            topo.numa_nodes_sorted_by_distance(1, |from, to| if from == to { 10 } else { 20 });
        // Anchor node 1 first; nodes 0 and 2 tied at distance 20,
        // stable over BTreeMap-ascending order.
        assert_eq!(order[0], 1, "anchor node first");
        assert_eq!(
            &order[1..],
            &[0, 2],
            "tied-distance nodes in ascending order"
        );
    }

    /// `numa_nodes_sorted_by_distance` demotes unreachable nodes
    /// (distance 255 per Linux convention) to the end even when
    /// the node has LLCs. Pins the unreachable-last contract.
    #[test]
    fn numa_nodes_sorted_by_distance_unreachable_demoted() {
        let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 1), (vec![2], 2)]);
        // Node 2 unreachable from anchor 0, node 1 at distance 20.
        let order = topo.numa_nodes_sorted_by_distance(0, |from, to| match (from, to) {
            (0, 0) => 10,
            (0, 1) => 20,
            (0, 2) => 255,
            _ => 20,
        });
        assert_eq!(order, vec![0, 1, 2]);
        // The key invariant: unreachable at end even though its
        // numeric id (2) would naturally sort mid-range.
        assert_eq!(*order.last().unwrap(), 2, "unreachable node is last");
    }

    /// `numa_nodes_sorted_by_distance` skips nodes not in
    /// host_node_llcs — a node with no LLCs is excluded entirely.
    /// "Nodes without any LLCs on this host are skipped — spilling
    /// to an empty node has no value" per the doc.
    #[test]
    fn numa_nodes_sorted_by_distance_skips_empty_nodes() {
        // Only node 0 has LLCs. Anchor 99 never appears in output.
        let topo = synth_host_topo(&[(vec![0], 0)]);
        let order = topo.numa_nodes_sorted_by_distance(99, |_, _| 20);
        assert_eq!(order, vec![0], "only node 0 is in host_node_llcs");
    }

    // ---------------------------------------------------------------
    // acquire_llc_plan — cap semantics (host-integration-light)
    // ---------------------------------------------------------------

    /// `acquire_llc_plan` with `cpu_cap == Some(cap)` and
    /// `cap > allowed-CPU count` fails at `effective_count` with a
    /// `ResourceContention` — before any /tmp side-effects. Pins
    /// that over-cap fails cleanly without touching the lock pool.
    /// The test pins a 2-CPU allowed set and caps at 3 CPUs, the
    /// minimum pair that exercises the "N > allowed" branch.
    #[test]
    fn acquire_llc_plan_rejects_cap_over_allowed_cpus() {
        let _allowed = AllowedCpusGuard::new(vec![0, 1]);
        // Two real LLC groups (one CPU each), cap of 3 CPUs.
        let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 0)]);
        let test_topo = crate::topology::TestTopology::synthetic(4, 1);
        let cap = CpuCap::new(3).unwrap();
        let err = acquire_llc_plan(&topo, &test_topo, Some(cap))
            .expect_err("cap > allowed_cpus must error");
        assert!(
            err.downcast_ref::<ResourceContention>().is_some(),
            "must be ResourceContention: {err:#}"
        );
    }

    // ---------------------------------------------------------------
    // BuildSandbox supplementary coverage lives in
    // src/vmm/cgroup_sandbox.rs's mod tests — see
    // `cpuset_sets_equal_identity`, `cpuset_sets_equal_narrower_effective`,
    // `sandbox_degraded_display_text` (includes RootCgroupRefused),
    // `parent_controllers_include_missing_file`, and
    // `read_cpuset_effective_missing_file_returns_none`. The
    // try_create RootCgroupRefused guard requires a test-only seam
    // over `read_self_cgroup_path` which doesn't exist yet — tracked
    // for a future iteration; the variant's Display is already
    // covered.
    // ---------------------------------------------------------------

    // ---------------------------------------------------------------
    // Deadlock guards — plan_from_snapshots produces ascending
    // llc_idx for livelock-proof acquire order
    // ---------------------------------------------------------------

    /// `plan_from_snapshots` returns selected LLC indices in
    /// ascending order — pinned at step e of the algorithm. Two
    /// concurrent callers with the same target see the same
    /// sequence, so their `try_acquire_llc_plan_locks` walk each
    /// flock in the same order. Reverse-order acquire would
    /// deadlock if one caller grabbed LLC N first while another
    /// grabbed LLC 0 first and they competed for each other's
    /// next targets. Ascending order eliminates that possibility.
    ///
    /// The expected output `[0, 2, 3]` catches TWO independent
    /// regressions at once:
    ///   1. Consolidation dropped (filter on `holder_count > 0`
    ///      removed). Output would become `[0, 1, 2]` because the
    ///      fresh LLCs at indices 0 and 1 would rank equal to LLC
    ///      2 without the consolidation preference.
    ///   2. Final `sort_unstable` dropped. Output would preserve
    ///      the interior walk order, typically `[2, 3, 0]` once
    ///      consolidation promoted the peer-held LLCs.
    ///
    /// Either regression fails this test. See
    /// `plan_from_snapshots_always_ascending_across_target_range`
    /// for the broader property-based guard.
    #[test]
    fn plan_from_snapshots_returns_ascending_indices() {
        let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 0), (vec![2], 0), (vec![3], 0)]);
        // Synthetic snapshots — holder_count higher on "later"
        // LLCs so consolidation score would put them first if the
        // algorithm didn't re-sort ascending at the end.
        let snapshots: Vec<LlcSnapshot> = (0..4)
            .map(|idx| LlcSnapshot {
                llc_idx: idx,
                lockfile_path: std::path::PathBuf::from(format!("/tmp/ktstr-llc-{idx}.lock")),
                holders: Vec::new(),
                holder_count: if idx >= 2 { 5 } else { 0 },
            })
            .collect();
        let allowed: std::collections::BTreeSet<usize> = (0..4).collect();
        let selected = plan_from_snapshots(
            &snapshots,
            3,
            &topo,
            &allowed,
            |_, _| 10, // everything same-node
        );
        // Step e of plan_from_snapshots is
        // `selected.sort_unstable()` — guarantees ascending llc_idx
        // regardless of consolidation score or seed ordering. Two
        // concurrent callers with the same snapshots see the same
        // acquire order, eliminating reverse-order deadlock.
        assert_eq!(selected, vec![0, 2, 3], "step e sorts ascending");
    }

    /// `plan_from_snapshots` with `target_cpus >= sum of allowed
    /// CPUs across every LLC` short-circuits to "select every LLC
    /// with at least one allowed CPU" in ascending order. Pins the
    /// saturation-case behaviour: the CPU budget covers or exceeds
    /// the total schedulable capacity, so the walk picks every
    /// eligible LLC without running the scoring pass.
    #[test]
    fn plan_from_snapshots_target_ge_all_selects_every_llc() {
        let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 1), (vec![2], 2)]);
        let snapshots: Vec<LlcSnapshot> = (0..3)
            .map(|idx| LlcSnapshot {
                llc_idx: idx,
                lockfile_path: std::path::PathBuf::from(format!("/tmp/ktstr-llc-{idx}.lock")),
                holders: Vec::new(),
                holder_count: 0,
            })
            .collect();
        let allowed: std::collections::BTreeSet<usize> = (0..3).collect();
        let selected = plan_from_snapshots(&snapshots, 3, &topo, &allowed, |_, _| 10);
        assert_eq!(selected, vec![0, 1, 2]);
        let selected_over = plan_from_snapshots(&snapshots, 999, &topo, &allowed, |_, _| 10);
        assert_eq!(selected_over, vec![0, 1, 2], "target > len clamps");
    }

    /// `plan_from_snapshots` with `target == 0` returns empty —
    /// early return in the algorithm. Pins the degenerate case
    /// so a future "optimization" that assumes selected[0] exists
    /// fails here first.
    #[test]
    fn plan_from_snapshots_target_zero_returns_empty() {
        let topo = synth_host_topo(&[(vec![0], 0)]);
        let snapshots: Vec<LlcSnapshot> = vec![LlcSnapshot {
            llc_idx: 0,
            lockfile_path: std::path::PathBuf::from("/tmp/ktstr-llc-0.lock"),
            holders: Vec::new(),
            holder_count: 0,
        }];
        let allowed: std::collections::BTreeSet<usize> = [0].into_iter().collect();
        let selected = plan_from_snapshots(&snapshots, 0, &topo, &allowed, |_, _| 10);
        assert!(selected.is_empty());
    }

    /// `plan_from_snapshots` prefers LLCs with `holder_count > 0`
    /// over fresh LLCs on the same NUMA node — the consolidation
    /// half of the composite sort ("consolidation candidates
    /// first, then fresh candidates"). Two same-node LLCs,
    /// holder_count [0, 5],
    /// target=1 → must pick the holder=5 LLC (index 1), not the
    /// fresh one (index 0). A future bug that flipped the partition
    /// order (fresh-first) or dropped the holder_count tiebreaker
    /// would pick LLC 0 instead and fail this test.
    ///
    /// Distinct from `plan_from_snapshots_returns_ascending_indices`
    /// which only asserted the post-sort ordering — that test
    /// accepted EITHER consolidation ordering because its output
    /// happened to be ascending in both cases. This one rejects
    /// the non-consolidation output.
    #[test]
    fn plan_from_snapshots_prefers_higher_holder_count() {
        let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 0)]);
        let snapshots: Vec<LlcSnapshot> = vec![
            LlcSnapshot {
                llc_idx: 0,
                lockfile_path: std::path::PathBuf::from("/tmp/ktstr-llc-0.lock"),
                holders: Vec::new(),
                holder_count: 0,
            },
            LlcSnapshot {
                llc_idx: 1,
                lockfile_path: std::path::PathBuf::from("/tmp/ktstr-llc-1.lock"),
                holders: Vec::new(),
                holder_count: 5,
            },
        ];
        // Same-node distance closure so placement doesn't bias by
        // NUMA — isolates the consolidation preference signal.
        let allowed: std::collections::BTreeSet<usize> = (0..2).collect();
        let selected = plan_from_snapshots(&snapshots, 1, &topo, &allowed, |_, _| 10);
        assert_eq!(
            selected,
            vec![1],
            "target=1 with holders [0,5] must pick LLC 1 \
             (consolidation preference), not LLC 0 (fresh)"
        );
    }

    /// Invariant-based ascending-order property: for every target
    /// in 1..=snapshots.len(), `selected.windows(2)` all satisfy
    /// `w[0] < w[1]`. This pins the step-e sort_unstable invariant
    /// independent of the consolidation / node-spill traversal —
    /// a future refactor that restructures the inner walk but
    /// forgets the final sort will fail this test at SOME target,
    /// not just the specific one `_returns_ascending_indices` pins.
    #[test]
    fn plan_from_snapshots_always_ascending_across_target_range() {
        let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 1), (vec![2], 0), (vec![3], 1)]);
        // Mixed holder_counts so consolidation ordering varies.
        let snapshots: Vec<LlcSnapshot> = vec![
            LlcSnapshot {
                llc_idx: 0,
                lockfile_path: std::path::PathBuf::from("/tmp/ktstr-llc-0.lock"),
                holders: Vec::new(),
                holder_count: 3,
            },
            LlcSnapshot {
                llc_idx: 1,
                lockfile_path: std::path::PathBuf::from("/tmp/ktstr-llc-1.lock"),
                holders: Vec::new(),
                holder_count: 0,
            },
            LlcSnapshot {
                llc_idx: 2,
                lockfile_path: std::path::PathBuf::from("/tmp/ktstr-llc-2.lock"),
                holders: Vec::new(),
                holder_count: 7,
            },
            LlcSnapshot {
                llc_idx: 3,
                lockfile_path: std::path::PathBuf::from("/tmp/ktstr-llc-3.lock"),
                holders: Vec::new(),
                holder_count: 1,
            },
        ];
        let allowed: std::collections::BTreeSet<usize> = (0..4).collect();
        // Each LLC has 1 CPU, so target_cpus == #LLCs to select. The
        // ascending-order invariant is agnostic to CPU-count vs
        // LLC-count semantics — the post-step-e sort holds regardless.
        for target_cpus in 1..=snapshots.len() {
            let selected = plan_from_snapshots(&snapshots, target_cpus, &topo, &allowed, |_, _| 10);
            assert_eq!(
                selected.len(),
                target_cpus,
                "target_cpus={target_cpus} must produce {target_cpus} selections, got {selected:?}"
            );
            assert!(
                selected.windows(2).all(|w| w[0] < w[1]),
                "target_cpus={target_cpus}: selection {selected:?} is not strictly ascending",
            );
        }
    }

    /// `make_jobs_for_plan` returns `plan.cpus.len().max(1)` so the
    /// `-jN` hint to make matches the reserved CPU count — gcc
    /// doesn't fan out beyond the cgroup budget.
    #[test]
    fn make_jobs_for_plan_matches_cpu_count() {
        let plan = LlcPlan {
            locked_llcs: vec![0, 1],
            cpus: vec![0, 1, 2, 3],
            mems: std::collections::BTreeSet::new(),
            snapshot: Vec::new(),
            locks: Vec::new(),
        };
        assert_eq!(make_jobs_for_plan(&plan), 4);
    }

    /// Edge: empty `plan.cpus` must yield `1`, never `0` — `make
    /// -j0` on GNU make produces unbounded parallelism, exactly
    /// the pathology the cap is supposed to prevent. The `.max(1)`
    /// floor pins this.
    #[test]
    fn make_jobs_for_plan_empty_cpus_floors_to_one() {
        let plan = LlcPlan {
            locked_llcs: Vec::new(),
            cpus: Vec::new(),
            mems: std::collections::BTreeSet::new(),
            snapshot: Vec::new(),
            locks: Vec::new(),
        };
        assert_eq!(
            make_jobs_for_plan(&plan),
            1,
            "empty-cpus must floor to 1, not 0 — -j0 is unbounded",
        );
    }

    /// `format_llc_list` renders LLC indices with per-entry NUMA
    /// node annotation when `cpu_to_node` is populated. Two
    /// locked LLCs on different nodes → "0 (node 0), 2 (node 1)".
    #[test]
    fn format_llc_list_with_numa_info() {
        let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 0), (vec![2], 1), (vec![3], 1)]);
        let rendered = format_llc_list(&[0, 2], &topo);
        assert!(
            rendered.contains("0 (node 0)"),
            "must annotate LLC 0 with its node: {rendered}",
        );
        assert!(
            rendered.contains("2 (node 1)"),
            "must annotate LLC 2 with its node: {rendered}",
        );
        // Full bracket form — enforces "[...]" wrapping so the
        // warning message reads naturally.
        assert_eq!(rendered, "[0 (node 0), 2 (node 1)]");
    }

    /// `format_llc_list` single-LLC case — no comma, no cross-node
    /// spill, bracket-wrapped. Pins the rendering shape for the
    /// warning that fires on non-spilling plans (which don't
    /// actually emit the cross-node warning, but the helper may
    /// still be called by future tooling).
    #[test]
    fn format_llc_list_single_llc() {
        let topo = synth_host_topo(&[(vec![0], 0)]);
        let rendered = format_llc_list(&[0], &topo);
        assert_eq!(rendered, "[0 (node 0)]");
    }

    /// `format_llc_list` on a degraded host with empty
    /// `cpu_to_node` drops the `(node N)` annotation per the doc
    /// ("[0, 2] on degraded hosts whose cpu_to_node map is empty").
    /// Synth helper populates cpu_to_node — mimic the degraded
    /// case by clearing it before calling.
    #[test]
    fn format_llc_list_without_numa_info() {
        let mut topo = synth_host_topo(&[(vec![0], 0), (vec![1], 0)]);
        topo.cpu_to_node.clear();
        let rendered = format_llc_list(&[0, 1], &topo);
        assert_eq!(
            rendered, "[0, 1]",
            "degraded-host form drops node annotation"
        );
    }

    /// `should_warn_cross_node` polarity pin: empty set or
    /// single-node set → false; two or more nodes → true.
    /// Splits the decision out of the eprintln! side-channel so
    /// regression tests can assert the condition without capturing
    /// stderr.
    #[test]
    fn should_warn_cross_node_polarity() {
        use std::collections::BTreeSet;
        let empty: BTreeSet<usize> = BTreeSet::new();
        assert!(
            !should_warn_cross_node(&empty),
            "empty mems must NOT warn (degenerate plan with no NUMA info)",
        );
        let single: BTreeSet<usize> = [0].into_iter().collect();
        assert!(
            !should_warn_cross_node(&single),
            "single-node plan must NOT warn — the whole point of the cap \
             is to fit on one node when possible",
        );
        let dual: BTreeSet<usize> = [0, 1].into_iter().collect();
        assert!(
            should_warn_cross_node(&dual),
            "two-node plan MUST warn — operator picked a cap that \
             couldn't fit on one node and deserves to hear about it",
        );
        let triple: BTreeSet<usize> = [0, 1, 2].into_iter().collect();
        assert!(
            should_warn_cross_node(&triple),
            "three-node plan MUST warn — same rationale as dual",
        );
    }

    /// `warn_if_cross_node_spill` end-to-end pin: a multi-node plan
    /// produces the formatted warning (non-empty side effect
    /// observable via the pure predicate). A single-node plan is
    /// a no-op (predicate returns false → eprintln! is skipped).
    /// Pins the coupling between the predicate and the side-
    /// effecting wrapper so a refactor that dropped the predicate
    /// call (e.g. inlined an incorrect comparison) would fail.
    #[test]
    fn warn_if_cross_node_spill_predicate_gates_stderr() {
        let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 1)]);
        let multi_plan = LlcPlan {
            locked_llcs: vec![0, 1],
            cpus: vec![0, 1],
            mems: [0usize, 1].into_iter().collect(),
            snapshot: Vec::new(),
            locks: Vec::new(),
        };
        assert!(should_warn_cross_node(&multi_plan.mems));
        // Call the wrapper — it produces stderr output but we rely
        // on the predicate gate above to verify the "will fire" half.
        // Directly capturing stderr in-process is fragile across
        // test runners; the predicate test pins the decision.
        warn_if_cross_node_spill(&multi_plan, &topo);

        let single_plan = LlcPlan {
            locked_llcs: vec![0],
            cpus: vec![0],
            mems: [0usize].into_iter().collect(),
            snapshot: Vec::new(),
            locks: Vec::new(),
        };
        assert!(!should_warn_cross_node(&single_plan.mems));
        // No-op call — predicate returns false, eprintln! is skipped.
        warn_if_cross_node_spill(&single_plan, &topo);
    }

    /// `CpuCap::new(1).effective_count(0)` errors: `n=1 > host=0`.
    /// Degenerate "host has zero LLCs" edge — unlikely on a real
    /// machine but critical to pin the boundary so a future bug
    /// that flipped the comparison to `n >= host_llcs` (rejecting
    /// cap == total) OR `n > host_llcs - 1` (overflow on 0) fails
    /// here first.
    #[test]
    fn cpu_cap_effective_count_on_zero_llc_host() {
        let cap = CpuCap::new(1).unwrap();
        let err = cap.effective_count(0).expect_err("1 > 0 must error");
        assert!(
            err.downcast_ref::<ResourceContention>().is_some(),
            "must be ResourceContention for retry routing",
        );
    }

    /// Multi-process concurrent `acquire_llc_plan`: a child process
    /// holds `LOCK_SH` on one LLC's lockfile via `flock(1)` (SHELL
    /// utility), then the parent calls `acquire_llc_plan` with a
    /// cap forcing the planner to consolidate onto an LLC that has
    /// holders. The consolidation invariant (`holder_count DESC`
    /// ordering in `plan_from_snapshots`) requires the parent's
    /// plan to include the child's LLC.
    ///
    /// Uses `flock(1)` + `sleep 10` rather than Rust fork() so the
    /// holder is a different process (different pid, different OFD)
    /// than the test thread — proving the /proc/locks cross-process
    /// enumeration path is exercised.
    ///
    /// `flock(1)` is expected on every Linux host that runs ktstr
    /// tests (it's in util-linux, part of the minimum viable CI
    /// image). If it's absent the test short-circuits rather than
    /// failing — the invariant is real but the test infrastructure
    /// depends on a userspace utility.
    #[test]
    fn acquire_llc_plan_consolidates_on_peer_held_llc() {
        let _prefix = LlcLockPrefixGuard::new();
        // 2 LLCs on the same node so NUMA-locality doesn't bias
        // against consolidation.
        let topo = HostTopology::new_for_tests(&[(vec![0], 0), (vec![1], 0)]);

        // Child process holds SH on LLC 1's lockfile via flock(1),
        // sleeping long enough for the parent's acquire to complete
        // inside the same SH window.
        let target_lock = llc_lock_path(1);
        // Ensure the lockfile exists so flock(1) opens the right
        // inode (not a fresh one that /proc/locks would attribute
        // to the flock(1) pid on a different inode than the parent
        // sees).
        crate::flock::materialize(&target_lock).expect("materialize lockfile");

        let child = std::process::Command::new("flock")
            .args(["-s", "-n", &target_lock, "sleep", "2"])
            .spawn();
        let mut child = match child {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // flock(1) missing — skip rather than fail.
                eprintln!(
                    "acquire_llc_plan_consolidates_on_peer_held_llc: \
                     flock(1) not available, skipping ({e})"
                );
                return;
            }
            Err(e) => panic!("spawn flock(1): {e}"),
        };

        // Brief sleep to let the child acquire SH before the parent
        // reads /proc/locks in discover. 50ms is well past util-linux's
        // exec + flock NB path and short enough that the child's
        // `sleep 2` still covers the parent's acquire window.
        std::thread::sleep(std::time::Duration::from_millis(50));

        let test_topo = crate::topology::TestTopology::synthetic(2, 1);
        let cap = CpuCap::new(1).expect("cap=1 valid");
        let plan = acquire_llc_plan(&topo, &test_topo, Some(cap))
            .expect("SH is reentrant — parent SH must coexist with child SH");

        // Consolidation picked LLC 1 (the one with a holder) over
        // LLC 0 (fresh). The `holder_count DESC` ordering in
        // `plan_from_snapshots` makes this deterministic.
        assert_eq!(
            plan.locked_llcs,
            vec![1],
            "cap=1 with child holding SH on LLC 1 must pick LLC 1 \
             (consolidation over fresh LLC 0); got {:?}",
            plan.locked_llcs,
        );

        drop(plan);
        // Child exits naturally after sleep 2; reap it so we don't
        // leave zombies.
        let _ = child.wait();
    }

    /// `ACQUIRE_MAX_TOCTOU_RETRIES` pins the retry budget at 1 —
    /// one DISCOVER + at most one retry DISCOVER (two total
    /// attempts). The second DISCOVER's /proc/locks read IS the
    /// backoff; more retries just amplify livelock risk without
    /// adding coordination signal. Regression guard against a
    /// future "just retry harder" tweak.
    #[test]
    fn acquire_max_toctou_retries_pinned_at_one() {
        assert_eq!(
            ACQUIRE_MAX_TOCTOU_RETRIES, 1,
            "retry budget must be 1 — higher values amplify livelock",
        );
    }

    /// TOCTOU retry SUCCESS path via the acquire-fn seam: attempt 0
    /// returns `Ok(None)` (simulating a peer holding EX during the
    /// first ACQUIRE), attempt 1 returns `Ok(Some(Vec::new()))`
    /// (peer released, shared acquire succeeds). The outer
    /// `acquire_llc_plan_with_acquire_fn` must re-run DISCOVER +
    /// PLAN and retry — not propagate the first `None` upward.
    ///
    /// Uses two real LLC groups with empty CPU lists so
    /// `discover_llc_snapshots` succeeds without touching any real
    /// `/tmp` lockfile (the seam consumes the snapshots instead of
    /// handing off to the real flock code). LLC indices 93500/93501
    /// are in the reserved 93000-99999 test range per the module's
    /// SYNTHETIC-TOPOLOGY OFFSET CONVENTION.
    #[test]
    fn acquire_llc_plan_retry_succeeds_on_attempt_one() {
        let _allowed = AllowedCpusGuard::new(vec![93500, 93501]);
        let topo = synth_host_topo(&[(vec![93500], 0), (vec![93501], 0)]);
        // Clean the lockfile paths DISCOVER materializes so the
        // test is idempotent across re-runs.
        cleanup_lock("/tmp/ktstr-llc-0.lock");
        cleanup_lock("/tmp/ktstr-llc-1.lock");

        let test_topo = crate::topology::TestTopology::synthetic(2, 1);
        let counter = std::cell::Cell::new(0u32);
        let plan =
            acquire_llc_plan_with_acquire_fn(&topo, &test_topo, None, |_selected, _snapshots| {
                let n = counter.get();
                counter.set(n + 1);
                if n == 0 {
                    // Attempt 0: simulate peer winning EX race.
                    Ok(None)
                } else {
                    // Attempt 1: peer released, acquire succeeds
                    // with an empty fd set (production would have
                    // actual OwnedFd values; the LlcPlan RAII
                    // contract is exercised elsewhere).
                    Ok(Some(Vec::new()))
                }
            })
            .expect("retry on attempt 1 must succeed");
        // Attempt 1 produced locks (empty vec is fine — the plan
        // constructor accepts any Vec<OwnedFd>).
        assert_eq!(counter.get(), 2, "acquire_fn called exactly twice");
        // 30% of 2 allowed CPUs = ceil(0.6) = 1 CPU → pick 1 LLC
        // (seed-node first: LLC 0). `selected` holds only LLC 0;
        // the second LLC stays unlocked.
        assert_eq!(plan.locked_llcs, vec![0]);

        cleanup_lock("/tmp/ktstr-llc-0.lock");
        cleanup_lock("/tmp/ktstr-llc-1.lock");
    }

    /// TOCTOU retry EXHAUSTED path via the acquire-fn seam: every
    /// attempt returns `Ok(None)`. After
    /// `ACQUIRE_MAX_TOCTOU_RETRIES + 1` attempts, the outer loop
    /// bails with a `ResourceContention` whose message names the
    /// retry count.
    ///
    /// Pins: (a) the retry budget is respected — the acquire
    /// closure is called exactly `ACQUIRE_MAX_TOCTOU_RETRIES + 1`
    /// times before the error is returned; (b) the error surfaces
    /// as `ResourceContention` for nextest-retry routing; (c) the
    /// holder diagnostic block runs (the final DISCOVER read).
    #[test]
    fn acquire_llc_plan_retry_exhausted_bails_with_resource_contention() {
        let _allowed = AllowedCpusGuard::new(vec![93600]);
        let topo = synth_host_topo(&[(vec![93600], 0)]);
        cleanup_lock("/tmp/ktstr-llc-0.lock");
        let test_topo = crate::topology::TestTopology::synthetic(1, 1);

        let counter = std::cell::Cell::new(0u32);
        let err =
            acquire_llc_plan_with_acquire_fn(&topo, &test_topo, None, |_selected, _snapshots| {
                counter.set(counter.get() + 1);
                Ok(None)
            })
            .expect_err("every attempt returns None — must bail after retries");

        // The retry budget consumes exactly ACQUIRE_MAX_TOCTOU_RETRIES
        // + 1 acquire-fn calls. Attempt index 0 is the first
        // acquire; attempt reaches MAX before incrementing, so the
        // failure occurs on call MAX+1.
        assert_eq!(
            counter.get(),
            ACQUIRE_MAX_TOCTOU_RETRIES + 1,
            "acquire_fn called exactly ACQUIRE_MAX_TOCTOU_RETRIES + 1 times",
        );

        assert!(
            err.downcast_ref::<ResourceContention>().is_some(),
            "must downcast to ResourceContention for retry routing: {err:#}",
        );
        let msg = format!("{err:#}");
        assert!(
            msg.contains("TOCTOU retry"),
            "message must name the retry outcome: {msg}",
        );

        cleanup_lock("/tmp/ktstr-llc-0.lock");
    }

    /// `plan_from_snapshots` MUST-CONSOLIDATE invariant: on a
    /// single-node host where every fresh LLC is ascending, the
    /// single peer-held LLC at index 3 MUST be selected over any
    /// lower-index fresh LLC when target=1. A future refactor that
    /// accidentally flipped the partition order (fresh-first) or
    /// dropped the `holder_count > 0` filter would pick LLC 0
    /// instead and fail this test.
    ///
    /// Complements `plan_from_snapshots_prefers_higher_holder_count`
    /// (same-node, two LLCs) by proving the peer-held LLC wins
    /// even when it sits at the TAIL of the ascending fresh order,
    /// not just adjacent — the `holder_count > 0` partition MUST
    /// override the fresh-LLC ordering.
    #[test]
    fn plan_from_snapshots_consolidation_overrides_fresh_ordering() {
        let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 0), (vec![2], 0), (vec![3], 0)]);
        let snapshots: Vec<LlcSnapshot> = (0..4)
            .map(|idx| LlcSnapshot {
                llc_idx: idx,
                lockfile_path: std::path::PathBuf::from(format!("/tmp/ktstr-llc-{idx}.lock")),
                holders: Vec::new(),
                holder_count: if idx == 3 { 5 } else { 0 },
            })
            .collect();
        let allowed: std::collections::BTreeSet<usize> = (0..4).collect();
        let selected = plan_from_snapshots(&snapshots, 1, &topo, &allowed, |_, _| 10);
        assert_eq!(
            selected,
            vec![3],
            "target=1 with peer-held LLC 3 must pick LLC 3, not the \
             lowest-index fresh LLC 0 — consolidation overrides fresh",
        );
    }

    /// `plan_from_snapshots` NUMA-locality invariant: a single-node
    /// fit (target ≤ seed-node capacity) must NEVER spill. 4 LLCs
    /// split 2+2 across nodes 0/1, all fresh, target=2 → selected
    /// must be both LLCs on the seed node. A future refactor that
    /// accidentally spanned both nodes (e.g. by iterating every
    /// node's LLCs before checking selected.len()) would fail here.
    ///
    /// Walk seed node first, exhaust it
    /// before spilling to nearest-by-distance nodes. This test
    /// pins that the seed-node-fits-fully short-circuit works.
    #[test]
    fn plan_from_snapshots_single_node_fit_no_spill() {
        // LLCs 0,1 on node 0; LLCs 2,3 on node 1. CPUs disjoint so
        // synth_host_topo populates cpu_to_node cleanly.
        let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 0), (vec![2], 1), (vec![3], 1)]);
        // All fresh so neither node has a consolidation signal —
        // isolates the NUMA-locality bias.
        let snapshots: Vec<LlcSnapshot> = (0..4)
            .map(|idx| LlcSnapshot {
                llc_idx: idx,
                lockfile_path: std::path::PathBuf::from(format!("/tmp/ktstr-llc-{idx}.lock")),
                holders: Vec::new(),
                holder_count: 0,
            })
            .collect();
        // Canonical distance: same-node 10, cross-node 20.
        let allowed: std::collections::BTreeSet<usize> = (0..4).collect();
        let selected = plan_from_snapshots(&snapshots, 2, &topo, &allowed, |from, to| {
            if from == to { 10 } else { 20 }
        });
        assert_eq!(
            selected,
            vec![0, 1],
            "target=2 must stay on seed node 0 (LLCs 0,1); seed-node \
             capacity (2) covers the request, no spill to node 1 allowed",
        );
    }

    /// `plan_from_snapshots` tie-break invariant: when every
    /// consolidation score is identical (all holder_count=5),
    /// selection tiebreaks on `llc_idx ASC`. target=2 on 4 equal
    /// LLCs → selected == [0, 1]. A future refactor that made the
    /// consolidation sort unstable, or that used `sort_by_key`
    /// without the secondary ASC tiebreak, would pick a non-
    /// deterministic pair and fail this test.
    ///
    /// The `holder_count DESC, llc_idx ASC` composite key — the
    /// second key is mandatory for cross-run determinism.
    #[test]
    fn plan_from_snapshots_equal_scores_tiebreak_ascending() {
        let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 0), (vec![2], 0), (vec![3], 0)]);
        let snapshots: Vec<LlcSnapshot> = (0..4)
            .map(|idx| LlcSnapshot {
                llc_idx: idx,
                lockfile_path: std::path::PathBuf::from(format!("/tmp/ktstr-llc-{idx}.lock")),
                holders: Vec::new(),
                holder_count: 5,
            })
            .collect();
        let allowed: std::collections::BTreeSet<usize> = (0..4).collect();
        let selected = plan_from_snapshots(&snapshots, 2, &topo, &allowed, |_, _| 10);
        assert_eq!(
            selected,
            vec![0, 1],
            "equal consolidation scores must tiebreak on llc_idx ASC \
             — selected={selected:?}",
        );
    }

    /// `default_cpu_budget` math: 30% rounded UP with min-1 floor.
    /// Covers the small-host edge (1 CPU → 1 CPU budget), the
    /// rounding boundary (3 CPUs → ceil(0.9) = 1 CPU), the
    /// non-trivial case (10 CPUs → 3 CPUs), and the large case
    /// (100 CPUs → 30 CPUs). Zero-input is pinned at min-1 for
    /// defense in depth even though production callers bail
    /// upstream on empty allowed sets.
    #[test]
    fn default_cpu_budget_30_percent_rounded_up_min_one() {
        assert_eq!(default_cpu_budget(0), 1, "min-1 floor");
        assert_eq!(default_cpu_budget(1), 1, "ceil(0.3) = 1");
        assert_eq!(default_cpu_budget(3), 1, "ceil(0.9) = 1");
        assert_eq!(default_cpu_budget(4), 2, "ceil(1.2) = 2");
        assert_eq!(default_cpu_budget(10), 3, "ceil(3.0) = 3");
        assert_eq!(default_cpu_budget(100), 30, "exact 30%");
    }

    /// `acquire_llc_plan` bails with a diagnostic when the allowed
    /// CPU set has no overlap with ANY host LLC — a misconfigured
    /// host where sysfs and sched_getaffinity disagree. Pins the
    /// plan_from_snapshots-returns-empty → bail path so a future
    /// refactor that silently produces an empty plan surfaces as a
    /// test failure rather than an "no-op" VM boot.
    #[test]
    fn acquire_llc_plan_bails_when_no_llc_overlaps_allowed() {
        let _prefix = LlcLockPrefixGuard::new();
        // Allowed CPUs {100, 101} don't overlap ANY of the host's
        // LLCs (CPUs 0, 1). plan_from_snapshots returns empty →
        // acquire_llc_plan bails with the no-overlap diagnostic.
        let _allowed = AllowedCpusGuard::new(vec![100, 101]);
        let topo = HostTopology::new_for_tests(&[(vec![0], 0), (vec![1], 0)]);
        let test_topo = crate::topology::TestTopology::synthetic(4, 1);
        let err = acquire_llc_plan(&topo, &test_topo, None)
            .expect_err("no LLC overlap must bail, not silently run");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no host LLC overlaps"),
            "err must name the no-overlap condition: {msg}"
        );
    }

    /// Allowed-cpu filter invariant — LLCs whose CPUs are entirely
    /// outside the allowed set MUST NOT appear in `selected`, even
    /// when their consolidation score would otherwise promote them.
    ///
    /// Four LLCs, two CPUs each. Allowed set = {0, 1, 4, 5} —
    /// contains every CPU of LLCs 0 and 2, NONE of LLCs 1 or 3.
    /// target_cpus=3 → planner picks LLC 0 (2 allowed CPUs,
    /// accumulated 2 < 3 keeps walking) then LLC 2 (1 more CPU is
    /// enough to cover the budget once materialization
    /// partial-takes; the plan_from_snapshots walk itself stops
    /// once accumulated ≥ target, which here fires at accumulated
    /// == 4 ≥ 3). `selected` is [0, 2]; LLCs 1 and 3 must stay
    /// out of the list.
    ///
    /// Regresses any refactor that drops the eligibility filter —
    /// e.g. a cleaner that collapses the `filter(eligible)` pass
    /// into the sort closure would produce a plan containing an
    /// LLC with zero schedulable CPUs, which sched_setaffinity on
    /// the resulting mask would reject.
    #[test]
    fn plan_from_snapshots_filters_llcs_outside_allowed_set() {
        let topo = synth_host_topo(&[
            (vec![0, 1], 0),
            (vec![2, 3], 0),
            (vec![4, 5], 0),
            (vec![6, 7], 0),
        ]);
        let snapshots: Vec<LlcSnapshot> = (0..4)
            .map(|idx| LlcSnapshot {
                llc_idx: idx,
                lockfile_path: std::path::PathBuf::from(format!("/tmp/ktstr-llc-{idx}.lock")),
                holders: Vec::new(),
                holder_count: 0,
            })
            .collect();
        let allowed: std::collections::BTreeSet<usize> = [0, 1, 4, 5].into_iter().collect();
        let selected = plan_from_snapshots(&snapshots, 3, &topo, &allowed, |_, _| 10);
        assert_eq!(
            selected,
            vec![0, 2],
            "planner must skip LLCs 1 and 3 (no allowed-CPU overlap) \
             and pick LLCs 0 and 2 whose CPUs are fully in allowed; \
             got {selected:?}"
        );
    }

    /// Partial-take on the last selected LLC — when the budget
    /// falls mid-LLC, `plan.cpus` contains only the budget-needed
    /// prefix of that LLC's allowed CPUs, not the whole LLC. Two
    /// 4-CPU LLCs, cpu_cap = 5 → LLC 0 contributes 4 CPUs, LLC 1
    /// contributes 1 CPU, `plan.cpus.len() == 5`, both LLCs are
    /// flocked. Regresses any refactor that reverts to the
    /// round-up-whole-LLC policy (which would produce 8 CPUs,
    /// over-reserving).
    #[test]
    fn acquire_llc_plan_partial_take_last_llc_matches_exact_budget() {
        let _prefix = LlcLockPrefixGuard::new();
        let _allowed = AllowedCpusGuard::new(vec![0, 1, 2, 3, 4, 5, 6, 7]);
        let topo = HostTopology::new_for_tests(&[(vec![0, 1, 2, 3], 0), (vec![4, 5, 6, 7], 0)]);
        let test_topo = crate::topology::TestTopology::synthetic(4, 1);
        let cap = CpuCap::new(5).expect("cap=5 valid");
        let plan = acquire_llc_plan(&topo, &test_topo, Some(cap))
            .expect("clean pool must allow SH on both LLCs");

        assert_eq!(
            plan.locked_llcs,
            vec![0, 1],
            "budget of 5 CPUs crosses LLC boundary — both must be flocked"
        );
        assert_eq!(
            plan.cpus.len(),
            5,
            "plan.cpus is EXACTLY the budget, not rounded up: {:?}",
            plan.cpus,
        );
        // Partial-take is deterministic: first LLC fully, then the
        // ordered prefix of the second.
        assert_eq!(plan.cpus, vec![0, 1, 2, 3, 4]);
    }

    /// Partial-LLC allowed overlap — an LLC that contains SOME
    /// allowed CPUs is still selectable, and its contribution to
    /// the CPU budget is the size of the intersection, not the
    /// full LLC. Two LLCs with 2 CPUs each; allowed = {0, 2} (one
    /// CPU from each LLC). target_cpus=2 → both LLCs must be
    /// selected (each contributes 1 allowed CPU, total 2 meets the
    /// budget).
    #[test]
    fn plan_from_snapshots_partial_llc_overlap_counted_correctly() {
        let topo = synth_host_topo(&[(vec![0, 1], 0), (vec![2, 3], 0)]);
        let snapshots: Vec<LlcSnapshot> = (0..2)
            .map(|idx| LlcSnapshot {
                llc_idx: idx,
                lockfile_path: std::path::PathBuf::from(format!("/tmp/ktstr-llc-{idx}.lock")),
                holders: Vec::new(),
                holder_count: 0,
            })
            .collect();
        let allowed: std::collections::BTreeSet<usize> = [0, 2].into_iter().collect();
        let selected = plan_from_snapshots(&snapshots, 2, &topo, &allowed, |_, _| 10);
        assert_eq!(
            selected,
            vec![0, 1],
            "target_cpus=2 with 1 allowed CPU per LLC must pick \
             BOTH LLCs — each contributes 1, total 2 meets budget"
        );
    }

    /// Full `LlcPlan.mems` invariant (I1) — on a cross-node spill,
    /// `mems` MUST equal the union of NUMA nodes hosting every
    /// selected LLC. 4 LLCs split 2+2 across nodes 0/1, cap=3
    /// forces exactly one LLC from node 1 to spill after node 0
    /// exhausts. Assert `locked_llcs.len() == 3` AND
    /// `mems == {0, 1}`.
    ///
    /// Without this guard, a broken mems computation could produce
    /// an empty set (cgroup cpuset.mems write rejects → SIGKILL on
    /// mem alloc), OR the wrong nodes (forcing cross-socket
    /// allocation that defeats the LLC reservation).
    ///
    /// Uses a per-test lockfile prefix via [`LlcLockPrefixGuard`] so
    /// the topology can use small indices (0..4) instead of padding
    /// to 94004 entries to avoid colliding with production LLC
    /// lockfile paths.
    #[test]
    fn acquire_llc_plan_cross_node_spill_mems_union() {
        let _prefix = LlcLockPrefixGuard::new();
        let _allowed = AllowedCpusGuard::new(vec![0, 1, 2, 3]);
        // LLC 0,1 on node 0 (CPUs 0,1); LLC 2,3 on node 1 (CPUs 2,3).
        let topo =
            HostTopology::new_for_tests(&[(vec![0], 0), (vec![1], 0), (vec![2], 1), (vec![3], 1)]);

        let test_topo = crate::topology::TestTopology::synthetic(4, 2);
        // Each LLC has 1 CPU, so cap=3 CPUs → exactly 3 LLCs.
        let cap = CpuCap::new(3).expect("cap=3 valid");
        let plan = acquire_llc_plan(&topo, &test_topo, Some(cap))
            .expect("clean pool must allow 3-CPU acquisition");

        assert_eq!(
            plan.locked_llcs.len(),
            3,
            "cap=3 CPUs with 1-CPU LLCs must reserve exactly 3 LLCs, got {:?}",
            plan.locked_llcs,
        );
        assert_eq!(
            plan.mems.len(),
            2,
            "3 LLCs split across 2 nodes → mems must span BOTH nodes; \
             got {:?} (locked_llcs={:?})",
            plan.mems,
            plan.locked_llcs,
        );
        assert!(
            plan.mems.contains(&0) && plan.mems.contains(&1),
            "mems must contain BOTH node 0 and node 1 after cross-node \
             spill; got {:?}",
            plan.mems,
        );
    }
}
