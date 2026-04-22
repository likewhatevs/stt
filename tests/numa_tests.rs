use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{
    CgroupDef, CpusetSpec, HoldSpec, Step, execute_steps, execute_steps_with,
};
use ktstr::test_support::{KtstrTestEntry, NumaDistance, NumaNode, Topology, TopologyConstraints};
use ktstr::workload::{MemPolicy, MpolFlags};

// All NUMA tests use auto_repro: false — these verify topology plumbing,
// not scheduler behavior, so BPF crash probes add no diagnostic value.

// ---------------------------------------------------------------------------
// Multi-NUMA boot: 2 NUMA nodes, uniform memory, default 10/20 distances
// ---------------------------------------------------------------------------

fn scenario_multi_numa_boot(ctx: &Ctx) -> Result<AssertResult> {
    let topo = &ctx.topo;
    assert!(
        topo.num_numa_nodes() >= 2,
        "expected >= 2 NUMA nodes, got {}",
        topo.num_numa_nodes()
    );
    assert_eq!(topo.numa_distance(0, 0), 10);
    assert_eq!(topo.numa_distance(1, 1), 10);
    assert_eq!(topo.numa_distance(0, 1), 20);
    assert_eq!(topo.numa_distance(1, 0), 20);

    for &nid in topo.numa_node_ids() {
        let mi = topo
            .node_meminfo(nid)
            .unwrap_or_else(|| panic!("node {nid} missing meminfo"));
        assert!(mi.total_kb > 0, "node {nid} has zero memory");
    }

    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_MULTI_NUMA_BOOT: KtstrTestEntry = KtstrTestEntry {
    name: "numa_multi_node_boot",
    func: scenario_multi_numa_boot,
    topology: Topology::new(2, 4, 2, 1),
    constraints: TopologyConstraints {
        min_numa_nodes: 2,
        max_numa_nodes: Some(2),
        ..TopologyConstraints::DEFAULT
    },
    auto_repro: false,
    duration: std::time::Duration::from_secs(3),
    ..KtstrTestEntry::DEFAULT
};

// ---------------------------------------------------------------------------
// CXL memory-only node: node 2 has llcs=0 (no CPUs), only memory.
// Manual distributed_slice: #[ktstr_test] cannot express with_nodes/with_distances.
// ---------------------------------------------------------------------------

static CXL_NODES: [NumaNode; 3] = [
    NumaNode::new(2, 256),
    NumaNode::new(2, 256),
    NumaNode::new(0, 128),
];

static CXL_DIST: NumaDistance = NumaDistance::new(3, &[10, 20, 30, 20, 10, 25, 30, 25, 10]);

fn scenario_cxl_memory_only(ctx: &Ctx) -> Result<AssertResult> {
    let topo = &ctx.topo;
    assert_eq!(topo.num_numa_nodes(), 3, "expected 3 NUMA nodes");

    assert!(topo.is_memory_only(2), "node 2 must be memory-only");
    assert!(!topo.is_memory_only(0), "node 0 has CPUs");
    assert!(!topo.is_memory_only(1), "node 1 has CPUs");

    assert_eq!(topo.numa_distance(0, 2), 30);
    assert_eq!(topo.numa_distance(1, 2), 25);

    let mi = topo.node_meminfo(2).expect("CXL node 2 must have meminfo");
    assert!(mi.total_kb > 0, "CXL node must have memory");

    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_CXL_MEM_ONLY: KtstrTestEntry = KtstrTestEntry {
    name: "numa_cxl_memory_only_node",
    func: scenario_cxl_memory_only,
    topology: Topology::with_nodes(2, 1, &CXL_NODES).with_distances(&CXL_DIST),
    constraints: TopologyConstraints {
        min_numa_nodes: 3,
        max_numa_nodes: Some(3),
        ..TopologyConstraints::DEFAULT
    },
    memory_mb: 640,
    auto_repro: false,
    duration: std::time::Duration::from_secs(3),
    ..KtstrTestEntry::DEFAULT
};

// ---------------------------------------------------------------------------
// MemPolicy::Bind + min_page_locality assertion
// ---------------------------------------------------------------------------

fn scenario_mempolicy_bind_locality(ctx: &Ctx) -> Result<AssertResult> {
    let checks = ktstr::assert::Assert::default_checks().min_page_locality(0.5);
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_0")
                .workers(ctx.workers_per_cgroup)
                .with_cpuset(CpusetSpec::Numa(0))
                .mem_policy(MemPolicy::bind([0])),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps_with(ctx, steps, Some(&checks))
}

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_MEMPOLICY_BIND: KtstrTestEntry = KtstrTestEntry {
    name: "numa_mempolicy_bind_locality",
    func: scenario_mempolicy_bind_locality,
    topology: Topology::new(2, 4, 2, 1),
    constraints: TopologyConstraints {
        min_numa_nodes: 2,
        max_numa_nodes: Some(2),
        ..TopologyConstraints::DEFAULT
    },
    auto_repro: false,
    duration: std::time::Duration::from_secs(5),
    ..KtstrTestEntry::DEFAULT
};

// ---------------------------------------------------------------------------
// vmstat cross-node migration tracking
// ---------------------------------------------------------------------------

fn scenario_vmstat_migration(ctx: &Ctx) -> Result<AssertResult> {
    let checks = ktstr::assert::Assert::default_checks().max_cross_node_migration_ratio(0.5);
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_0")
                .workers(ctx.workers_per_cgroup)
                .with_cpuset(CpusetSpec::Numa(0))
                .mem_policy(MemPolicy::bind([0])),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps_with(ctx, steps, Some(&checks))
}

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_VMSTAT_MIGRATION: KtstrTestEntry = KtstrTestEntry {
    name: "numa_vmstat_migration_tracking",
    func: scenario_vmstat_migration,
    topology: Topology::new(2, 4, 2, 1),
    constraints: TopologyConstraints {
        min_numa_nodes: 2,
        max_numa_nodes: Some(2),
        ..TopologyConstraints::DEFAULT
    },
    auto_repro: false,
    duration: std::time::Duration::from_secs(5),
    ..KtstrTestEntry::DEFAULT
};

// ---------------------------------------------------------------------------
// MemPolicy::Interleave round-robins allocations across BOTH nodes, so the
// expected page locality on the binding-cpuset half is roughly 50%. The
// scenario pins workers to node 0 via cpuset but interleaves their pages
// across nodes 0 and 1, so the locality assertion sets a low minimum.
// Exercises the round-robin nodemask path that Bind/Preferred don't reach.
//
// `MpolFlags::STATIC_NODES` is required because without it the kernel
// silently intersects the interleave nodemask with the task's cpuset —
// which would degenerate to "interleave across {0} only" and defeat the
// cross-node intent.
// ---------------------------------------------------------------------------

fn scenario_mempolicy_interleave_cross_node(ctx: &Ctx) -> Result<AssertResult> {
    let checks = ktstr::assert::Assert::default_checks().min_page_locality(0.3);
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_0")
                .workers(ctx.workers_per_cgroup)
                .with_cpuset(CpusetSpec::Numa(0))
                .mem_policy(MemPolicy::interleave([0, 1]))
                .mpol_flags(MpolFlags::STATIC_NODES),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps_with(ctx, steps, Some(&checks))
}

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_MEMPOLICY_INTERLEAVE: KtstrTestEntry = KtstrTestEntry {
    name: "numa_mempolicy_interleave_cross_node",
    func: scenario_mempolicy_interleave_cross_node,
    topology: Topology::new(2, 4, 2, 1),
    constraints: TopologyConstraints {
        min_numa_nodes: 2,
        max_numa_nodes: Some(2),
        ..TopologyConstraints::DEFAULT
    },
    auto_repro: false,
    duration: std::time::Duration::from_secs(5),
    ..KtstrTestEntry::DEFAULT
};

// ---------------------------------------------------------------------------
// MemPolicy::PreferredMany lists multiple nodes that the kernel may use for
// allocation, falling back across the set rather than a single fallback path.
// Workers are pinned to node 0; the policy prefers nodes 0 and 1, so most
// pages should land on node 0 with the minority spilling to node 1.
// Exercises the MPOL_PREFERRED_MANY (kernel 5.15+) path that
// `MemPolicy::Preferred` cannot express.
//
// `MpolFlags::STATIC_NODES` is required so node 1 stays in the preferred
// set — without it the kernel narrows the nodemask to the cpuset (node 0
// only), collapsing the PreferredMany semantics into a single-node
// preferred that this test is not exercising.
// ---------------------------------------------------------------------------

fn scenario_mempolicy_preferred_many_locality(ctx: &Ctx) -> Result<AssertResult> {
    let checks = ktstr::assert::Assert::default_checks().min_page_locality(0.5);
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_0")
                .workers(ctx.workers_per_cgroup)
                .with_cpuset(CpusetSpec::Numa(0))
                .mem_policy(MemPolicy::preferred_many([0, 1]))
                .mpol_flags(MpolFlags::STATIC_NODES),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps_with(ctx, steps, Some(&checks))
}

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_MEMPOLICY_PREFERRED_MANY: KtstrTestEntry = KtstrTestEntry {
    name: "numa_mempolicy_preferred_many_locality",
    func: scenario_mempolicy_preferred_many_locality,
    topology: Topology::new(2, 4, 2, 1),
    constraints: TopologyConstraints {
        min_numa_nodes: 2,
        max_numa_nodes: Some(2),
        ..TopologyConstraints::DEFAULT
    },
    auto_repro: false,
    duration: std::time::Duration::from_secs(5),
    ..KtstrTestEntry::DEFAULT
};
