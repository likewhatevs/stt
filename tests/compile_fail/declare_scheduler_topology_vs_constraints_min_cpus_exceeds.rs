// Pins the `min_cpus` cross-field check arm. `topology` declares
// total_cpus = 2*4*1 = 8 but `constraints.min_cpus = 100`
// excludes any host that doesn't have at least 100 CPUs.
use ktstr::declare_scheduler;
#[allow(unused_imports)]
use ktstr::test_support::TopologyConstraints;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    topology = (1, 2, 4, 1),
    constraints = TopologyConstraints {
        min_cpus: 100,
        ..TopologyConstraints::DEFAULT
    },
});

fn main() {}
