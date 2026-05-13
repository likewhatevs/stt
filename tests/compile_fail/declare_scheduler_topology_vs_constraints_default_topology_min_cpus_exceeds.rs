// Pins the `min_cpus` cross-field check arm against the default
// topology. Default total_cpus = 1 * 2 * 1 = 2; `min_cpus = 3`
// exceeds the default total — every gauntlet preset rejects.
use ktstr::declare_scheduler;
#[allow(unused_imports)]
use ktstr::test_support::TopologyConstraints;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    constraints = TopologyConstraints {
        min_cpus: 3,
        ..TopologyConstraints::DEFAULT
    },
});

fn main() {}
