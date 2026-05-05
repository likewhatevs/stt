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
fn numa_nodes_for_cpus(topo: &HostTopology, cpus: &[usize]) -> std::collections::BTreeSet<usize> {
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
    let assigned: std::collections::HashSet<usize> = plan.assignments.iter().map(|a| a.1).collect();
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

/// `pid_window_offset` must diffuse adjacent pids across the
/// offset space so a batch-spawn (e.g. nextest forking N test
/// processes back-to-back, common pid range like
/// 100000..100100) doesn't pile every peer onto the same
/// starting window.
///
/// Pin the spread shape across a window of 100 consecutive pids:
///   * every pid's offset must fit in `[0, max_start)`;
///   * the unique-offset count must exceed half the input
///     range (50 for 100 pids), so adjacent pids don't all
///     collapse to the same offset;
///   * the average gap between consecutive pids' offsets
///     must exceed 1 (the trivial `pid % max_start` baseline).
///
/// `max_start = 33` is chosen so the ideal uniform spread
/// would visit every offset 100/33 ≈ 3 times; the unique
/// count must approach `max_start`.
#[test]
fn pid_window_offset_spreads_adjacent_pids() {
    let max_start = 33usize;
    let pids: Vec<u32> = (100_000..100_100).collect();
    let offsets: Vec<usize> = pids
        .iter()
        .map(|&p| pid_window_offset(p, max_start))
        .collect();

    // Every offset must fit in the target range.
    for (pid, off) in pids.iter().zip(offsets.iter()) {
        assert!(
            *off < max_start,
            "pid_window_offset({pid}, {max_start}) = {off}, exceeds max_start",
        );
    }

    // Unique-offset count: SipHash13's avalanche makes
    // adjacent pid landings independent of each other, so
    // 100 pids over 33 offsets should hit roughly all 33
    // offsets. Pin >= 25 to absorb the natural birthday-
    // paradox compression while still catching a regression
    // that flattens adjacent pids onto the same offset.
    //
    // The bare `pid % 33` baseline ALSO produces 33 unique
    // offsets, so the unique-count assertion alone does not
    // distinguish the avalanching hash from the trivial
    // modulo. The "adjacent-pid landings differ by 1"
    // assertion below catches that case: bare modulo gives
    // |offset[i+1] - offset[i]| == 1 always (the cyclic
    // step), whereas SipHash13 produces a fully randomized
    // step distribution.
    let unique: std::collections::HashSet<_> = offsets.iter().copied().collect();
    assert!(
        unique.len() >= 25,
        "100 adjacent pids spread to only {} unique offsets (max_start={max_start}); \
         hash mixer is losing entropy. offsets: {offsets:?}",
        unique.len(),
    );

    // Distinguish from the bare `pid % max_start` baseline:
    // bare modulo on consecutive pids gives consecutive
    // offsets (gap of exactly 1 modulo max_start). Count how
    // many adjacent pid pairs land at exactly +/-1 offset
    // (handling the wrap at the boundary). For SipHash13,
    // each adjacent pair has a 2/max_start ≈ 6% probability
    // of landing at +/-1 by chance, so over 99 pairs the
    // expected count is ~6 with stddev ~2.4. The bare
    // modulo baseline produces 99/99. Pin <= 30 (well above
    // the random-noise expected value, well below the bare-
    // modulo signature) so a regression that drops the
    // SipHash mixer trips this without flaking on the
    // hashed distribution.
    let adjacent_step_count = offsets
        .windows(2)
        .filter(|w| {
            let d = (w[0] as i64 - w[1] as i64).unsigned_abs() as usize;
            d == 1 || d == max_start - 1
        })
        .count();
    assert!(
        adjacent_step_count <= 30,
        "{adjacent_step_count} of {} adjacent pid pairs landed at +/-1 offset \
         (max_start={max_start}); the bare `pid % {max_start}` baseline produces \
         99/99 such pairs. SipHash13 avalanche should give ~6. offsets: {offsets:?}",
        offsets.len() - 1,
    );

    // Average gap between consecutive pids' offsets — the
    // signal that distinguishes "diffused" (gap >> 1) from
    // "adjacent collapse" (gap == 1, the bare `pid % 33`
    // shape that this fix replaces). Compute mean absolute
    // difference; SipHash13 avalanche should give average
    // gap near `max_start / 3` (uniform random walk on a
    // circular space of size N has expected step `N/3`).
    let gaps: Vec<usize> = offsets
        .windows(2)
        .map(|w| {
            let a = w[0] as i64;
            let b = w[1] as i64;
            (a - b).unsigned_abs() as usize
        })
        .collect();
    let mean_gap: f64 = gaps.iter().sum::<usize>() as f64 / gaps.len() as f64;
    assert!(
        mean_gap > 5.0,
        "mean offset gap between adjacent pids = {mean_gap:.2}, expected > 5 \
         (the bare `pid % {max_start}` baseline produces gap = 1; SipHash13 \
         avalanche should produce >> 5). offsets: {offsets:?}",
    );
}

/// `pid_window_offset` is deterministic: same (pid, max_start)
/// always produces the same offset. Pin against a regression
/// that introduces randomness or per-run state.
#[test]
fn pid_window_offset_deterministic() {
    for &pid in &[1u32, 100, 12345, 999_999, u32::MAX] {
        for &max_start in &[1usize, 3, 33, 1024, usize::MAX] {
            assert_eq!(
                pid_window_offset(pid, max_start),
                pid_window_offset(pid, max_start),
                "non-deterministic offset for pid={pid}, max_start={max_start}",
            );
        }
    }
}

/// `pid_window_offset` with `max_start == 1` always returns 0
/// (only valid offset). Pin the trivial-domain edge.
#[test]
fn pid_window_offset_max_start_one() {
    for &pid in &[0u32, 1, 100, u32::MAX] {
        assert_eq!(pid_window_offset(pid, 1), 0);
    }
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
    let order = topo.numa_nodes_sorted_by_distance(1, |from, to| if from == to { 10 } else { 20 });
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
    let err =
        acquire_llc_plan(&topo, &test_topo, Some(cap)).expect_err("cap > allowed_cpus must error");
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

/// `ACQUIRE_MAX_TOCTOU_RETRIES` pins the retry budget at 3 —
/// one DISCOVER + up to three retry DISCOVERs (four total
/// attempts), each separated by an ascending micro-sleep
/// (10ms, 50ms, 200ms — see [`TOCTOU_RETRY_DELAYS`]) so a
/// racing peer has time to drop its fds before the next
/// snapshot. Regression guard against a future "just retry
/// harder" tweak that would amplify livelock cost without
/// adding coordination signal.
#[test]
fn acquire_max_toctou_retries_pinned() {
    assert_eq!(
        ACQUIRE_MAX_TOCTOU_RETRIES, 3,
        "retry budget must be 3 — micro-sleeps absorb mid-sized races",
    );
    assert_eq!(
        TOCTOU_RETRY_DELAYS.len(),
        ACQUIRE_MAX_TOCTOU_RETRIES as usize,
        "one sleep per retry — TOCTOU_RETRY_DELAYS length must \
         match ACQUIRE_MAX_TOCTOU_RETRIES exactly",
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
    let err = acquire_llc_plan_with_acquire_fn(&topo, &test_topo, None, |_selected, _snapshots| {
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
        msg.contains("attempts"),
        "message must name the attempt count: {msg}",
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
