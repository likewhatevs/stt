// Pins the `max_cpus` cross-field check arm. `topology` declares
// total_cpus = 2*4*1 = 8 but `constraints.max_cpus = Some(4)`
// excludes any host that has more than 4 CPUs — a common gotcha
// when a test author sets a small max_cpus on a topology larger
// than expected.
use ktstr::declare_scheduler;
#[allow(unused_imports)]
use ktstr::test_support::TopologyConstraints;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    topology = (1, 2, 4, 1),
    constraints = TopologyConstraints {
        max_cpus: Some(4),
        ..TopologyConstraints::DEFAULT
    },
});

fn main() {}
