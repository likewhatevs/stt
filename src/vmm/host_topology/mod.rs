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

/// Diffuse a pid across `[0, max_start)` so adjacent pids do not
/// land on adjacent offsets. Used by [`acquire_cpu_locks`] to
/// pick a starting window.
///
/// Bare `pid % max_start` collapses adjacent pids onto adjacent
/// offsets (Linux's pid allocator walks `pid_max` sequentially),
/// which is the worst spread shape for the common batch-spawn
/// case: nextest forks N test processes back-to-back, every pid
/// lands within a small contiguous range, every `pid % max_start`
/// lands within an equally small contiguous slice of the offset
/// space, and they all probe overlapping windows on the first
/// pass. SipHash13 avalanche on the pid bytes diffuses adjacent
/// pids across the whole `[0, max_start)` range, so the
/// walk-around-once loop in [`acquire_cpu_locks`] has a fair
/// chance of finding a free window without burning the entire
/// lockfile pool.
///
/// The hash key is the SipHash13 default (`SipHasher13::new()`,
/// equivalent to `new_with_keys(0, 0)`) — a per-run random key
/// would defeat reproducibility for unit-test fixtures and for
/// any future debug logging that wants to confirm "pid X picks
/// offset Y for window N".
///
/// Caller invariant: `max_start >= 1`. Panics on `max_start == 0`
/// (modulo-by-zero); callers must enforce this upstream — the
/// `count > total_host_cpus` early-bail in [`acquire_cpu_locks`]
/// is the production guarantee.
fn pid_window_offset(pid: u32, max_start: usize) -> usize {
    use siphasher::sip::SipHasher13;
    use std::hash::Hasher;
    let mut hasher = SipHasher13::new();
    hasher.write(&pid.to_le_bytes());
    (hasher.finish() as usize) % max_start
}

/// Acquire exclusive CPU locks for a non-perf VM (non-blocking).
///
/// Tries to flock `count` consecutive CPU files starting from a
/// pid-derived offset, stepping by 1 if any CPU in the window is busy
/// and wrapping around once the high end of the search range is
/// exhausted so the lower windows (those below `start_offset`) are
/// also probed before giving up. Returns the held fds on success, or
/// `ResourceContention` when no window is available.
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

    // No window can fit if the request exceeds the host. Bail before
    // entering the search loop so the modular arithmetic below has a
    // non-zero domain.
    if count > total_host_cpus {
        return Err(anyhow::Error::new(ResourceContention {
            reason: format!(
                "no {count} consecutive CPUs available on a {total_host_cpus}-CPU host\n  \
                 hint: pass --no-perf-mode or set KTSTR_NO_PERF_MODE=1 to run without CPU reservation"
            ),
        }));
    }

    // Spread peers across the lockfile pool: start at a pid-derived
    // offset so two ktstr invocations launching simultaneously don't
    // both probe CPU 0 first. `max_start` is the count of valid
    // window-start positions in `[0, total_host_cpus - count]`
    // (inclusive), giving `total_host_cpus - count + 1` candidates.
    // The `count > total_host_cpus` early-bail above guarantees
    // `max_start >= 1`, so the modulo never divides by zero.
    let max_start = total_host_cpus - count + 1;
    let start_offset = pid_window_offset(std::process::id(), max_start);
    // Walk every candidate window exactly once, wrapping around so a
    // peer holding the high end never starves the low end. Visit
    // order is `start_offset, start_offset+1, ..., max_start-1,
    // 0, 1, ..., start_offset-1`.
    for step in 0..max_start {
        let offset = (start_offset + step) % max_start;
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
                            continue;
                        }
                    }
                }
                return Ok(locks);
            }
            Err(_) => continue,
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
// LLC, filtered to the process's allowed cpuset), plan (NUMA-aware,
// consolidation-aware selection), acquire (non-blocking `LOCK_SH`
// on each selected LLC). Up to ACQUIRE_MAX_TOCTOU_RETRIES retries
// absorb the window between the discover snapshot and the
// non-blocking acquire; between retries the loop sleeps for an
// ascending micro-budget (TOCTOU_RETRY_DELAYS) so a peer that
// raced us has time to drop its fds before the next snapshot.
// If every retry fails, the contention is persistent and the
// caller falls back to nextest-retry / operator-wait.

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
    if let Some(cpus) = crate::cpu_util::read_affinity(0) {
        return cpus.into_iter().map(|c| c as usize).collect();
    }
    if let Ok(raw) = std::fs::read_to_string("/proc/self/status") {
        for line in raw.lines() {
            if let Some(v) = line.strip_prefix("Cpus_allowed_list:")
                && let Some(parsed) = crate::cpu_util::parse_cpu_list(v.trim())
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

/// Maximum TOCTOU retry budget for the DISCOVER → PLAN → ACQUIRE
/// pipeline. Production sees up to `RETRIES + 1 = 4` attempts: one
/// initial DISCOVER and three retries. Between retries the caller
/// sleeps for an ascending micro-budget (10ms, 50ms, 200ms — see
/// [`TOCTOU_RETRY_DELAYS`]) so two peers that initially raced on the
/// same LLC have time to drop their fds before the next snapshot.
/// Without the sleep the second DISCOVER often sees the same holder
/// state and bails on a transient race; the in-process micro-sleep
/// absorbs that without paying the nextest-retry cost.
const ACQUIRE_MAX_TOCTOU_RETRIES: u32 = 3;

/// Per-retry sleep durations between DISCOVER attempts. Indexed by
/// the retry index: after attempt 0 fails the loop sleeps
/// `TOCTOU_RETRY_DELAYS[0]`, after attempt 1 fails it sleeps
/// `TOCTOU_RETRY_DELAYS[1]`, etc. Length must equal
/// [`ACQUIRE_MAX_TOCTOU_RETRIES`] — there are exactly that many
/// sleeps before the final attempt that can still bail.
const TOCTOU_RETRY_DELAYS: [std::time::Duration; ACQUIRE_MAX_TOCTOU_RETRIES as usize] = [
    std::time::Duration::from_millis(10),
    std::time::Duration::from_millis(50),
    std::time::Duration::from_millis(200),
];

/// DISCOVER phase — read-only LLC snapshot.
///
/// Walks ONLY the LLCs whose CPUs overlap `allowed` (the calling
/// process's `sched_getaffinity` cpuset). LLCs entirely outside the
/// cpuset are skipped — locking one would never contribute a
/// schedulable CPU to `plan.cpus`, and on a heavily-pinned runner
/// (CI cgroup with N out of M CPUs allowed) skipping them avoids
/// O(host_llcs - allowed_llcs) lockfile materializations and
/// /proc/locks lookups per attempt. The PLAN phase still receives a
/// snapshot vector indexed by `LlcSnapshot.llc_idx`, not by
/// position, so a sparse snapshot set works without any further
/// adjustment downstream.
///
/// For every selected LLC: stat the canonical lockfile (materializing
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
fn discover_llc_snapshots(
    topo: &HostTopology,
    allowed: &std::collections::BTreeSet<usize>,
    mountinfo: &str,
) -> Result<Vec<LlcSnapshot>> {
    let mut snapshots: Vec<LlcSnapshot> = Vec::with_capacity(topo.llc_groups.len());
    for llc_idx in 0..topo.llc_groups.len() {
        // Skip LLCs whose CPUs are entirely outside the calling
        // process's allowed cpuset — they cannot contribute a
        // schedulable CPU to `plan.cpus`, and locking one would just
        // pay for a lockfile + /proc/locks pass without coordination
        // value. The sparse snapshot vector keeps llc_idx as the
        // identity key, so PLAN's index-based iteration is
        // unaffected.
        if !topo.llc_groups[llc_idx]
            .cpus
            .iter()
            .any(|c| allowed.contains(c))
        {
            continue;
        }
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
/// Runs DISCOVER → PLAN → ACQUIRE with up to
/// [`ACQUIRE_MAX_TOCTOU_RETRIES`] retries (each separated by a
/// per-retry sleep from [`TOCTOU_RETRY_DELAYS`]). On
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
        return Err(ResourceContention {
            reason: "could not determine allowed CPU set \
                     (sched_getaffinity and /proc/self/status both failed)"
                .into(),
        }
        .into());
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
        return Err(ResourceContention {
            reason: "CPU budget resolved to zero".into(),
        }
        .into());
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
    let mountinfo = crate::flock::read_mountinfo().map_err(|e| ResourceContention {
        reason: format!("read /proc/self/mountinfo: {e}"),
    })?;

    let mut attempt: u32 = 0;
    loop {
        let snapshots =
            discover_llc_snapshots(topo, &allowed, &mountinfo).map_err(|e| ResourceContention {
                reason: format!("discover LLC snapshots: {e}"),
            })?;
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
            return Err(ResourceContention {
                reason: format!(
                    "no host LLC overlaps the process's \
                     {allowed_cpus}-CPU allowed set — sysfs LLC groups \
                     and sched_getaffinity disagree"
                ),
            }
            .into());
        }
        match acquire_fn(&selected, &snapshots).map_err(|e| ResourceContention {
            reason: format!("acquire LLC locks: {e}"),
        })? {
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
                    let final_snapshots = discover_llc_snapshots(topo, &allowed, &mountinfo)?;
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
                             CPU(s) after {attempts} attempts; holders: \
                             {holder_text}. Run `ktstr locks --json` to see \
                             every ktstr lock on this host.",
                            attempts = ACQUIRE_MAX_TOCTOU_RETRIES + 1,
                        ),
                    }));
                }
                // Sleep between attempts so a racing peer has time
                // to drop its fds before the next DISCOVER. Indexed
                // by `attempt` (0..RETRIES) — see TOCTOU_RETRY_DELAYS.
                std::thread::sleep(TOCTOU_RETRY_DELAYS[attempt as usize]);
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
///
/// # Safety
///
/// The caller must ensure that `addr` points to a valid mmap'd region
/// of at least `len` bytes. The kernel will read this range via the
/// `mbind(2)` syscall to set its NUMA memory policy; passing a stale,
/// unmapped, or out-of-bounds pointer is undefined behavior from the
/// process's perspective (the syscall itself returns EFAULT, but the
/// surrounding Rust contract is violated).
///
/// When `nodes.is_empty()` or `len == 0`, the function short-circuits
/// without dereferencing `addr`, so a null or dangling pointer is
/// permitted in those cases.
pub unsafe fn mbind_to_nodes(addr: *mut u8, len: usize, nodes: &[usize]) {
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
mod tests;
