// Pins the `max_numa_nodes` cross-field check arm against the
// default topology. Default numa_nodes = 1; `max_numa_nodes =
// Some(0)` excludes any host with 1 or more NUMA nodes.
use ktstr::declare_scheduler;
#[allow(unused_imports)]
use ktstr::test_support::TopologyConstraints;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    constraints = TopologyConstraints {
        max_numa_nodes: Some(0),
        ..TopologyConstraints::DEFAULT
    },
});

fn main() {}
