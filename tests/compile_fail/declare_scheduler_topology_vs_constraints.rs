// Pins the macro-time topology-vs-constraints consistency check.
// `topology = (1, 2, 4, 1)` declares a 2-LLC VM, but
// `constraints.min_llcs = 100` requires every preset to provide at
// least 100 LLCs — every gauntlet preset would reject this test at
// runtime and the test would never execute. The macro catches the
// inconsistency at expand time with a targeted diagnostic.
use ktstr::declare_scheduler;
#[allow(unused_imports)]
use ktstr::test_support::TopologyConstraints;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    topology = (1, 2, 4, 1),
    constraints = TopologyConstraints {
        min_llcs: 100,
        ..TopologyConstraints::DEFAULT
    },
});

fn main() {}
