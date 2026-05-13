// Pins the `max_llcs` cross-field check arm. `topology` declares
// 4 LLCs but `constraints.max_llcs = Some(2)` excludes any host
// with at least 4 LLCs — every gauntlet preset rejects the test
// at runtime.
use ktstr::declare_scheduler;
#[allow(unused_imports)]
use ktstr::test_support::TopologyConstraints;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    topology = (1, 4, 2, 1),
    constraints = TopologyConstraints {
        max_llcs: Some(2),
        ..TopologyConstraints::DEFAULT
    },
});

fn main() {}
