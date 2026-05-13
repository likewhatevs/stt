// Pins the `min_numa_nodes` cross-field check arm. `topology`
// declares 1 NUMA node but `constraints.min_numa_nodes = 4`
// excludes any host that doesn't have at least 4 NUMA nodes.
use ktstr::declare_scheduler;
#[allow(unused_imports)]
use ktstr::test_support::TopologyConstraints;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    topology = (1, 2, 4, 1),
    constraints = TopologyConstraints {
        min_numa_nodes: 4,
        ..TopologyConstraints::DEFAULT
    },
});

fn main() {}
