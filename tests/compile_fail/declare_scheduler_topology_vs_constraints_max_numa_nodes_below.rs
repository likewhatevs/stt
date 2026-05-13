// Pins the `max_numa_nodes` cross-field check arm. `topology`
// declares 4 NUMA nodes (2 LLCs per NUMA node = 8 LLCs total) but
// `constraints.max_numa_nodes = Some(1)` excludes any host that
// has more than 1 NUMA node.
use ktstr::declare_scheduler;
#[allow(unused_imports)]
use ktstr::test_support::TopologyConstraints;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    topology = (4, 8, 2, 1),
    constraints = TopologyConstraints {
        max_numa_nodes: Some(1),
        ..TopologyConstraints::DEFAULT
    },
});

fn main() {}
