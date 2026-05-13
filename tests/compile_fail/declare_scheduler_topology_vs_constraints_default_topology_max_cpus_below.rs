// Pins the `max_cpus` cross-field check arm against the default
// topology. Default total_cpus = 2; `max_cpus = Some(1)` excludes
// any host with 2 or more CPUs.
use ktstr::declare_scheduler;
#[allow(unused_imports)]
use ktstr::test_support::TopologyConstraints;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    constraints = TopologyConstraints {
        max_cpus: Some(1),
        ..TopologyConstraints::DEFAULT
    },
});

fn main() {}
