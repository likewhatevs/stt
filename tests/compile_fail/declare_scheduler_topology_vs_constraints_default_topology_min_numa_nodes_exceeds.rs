// Pins the `min_numa_nodes` cross-field check arm against the
// default topology (when `topology` field is omitted). Default
// numa_nodes = 1; `constraints.min_numa_nodes = 2` excludes any
// host with fewer than 2 NUMA nodes.
use ktstr::declare_scheduler;
#[allow(unused_imports)]
use ktstr::test_support::TopologyConstraints;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    constraints = TopologyConstraints {
        min_numa_nodes: 2,
        ..TopologyConstraints::DEFAULT
    },
});

fn main() {}
