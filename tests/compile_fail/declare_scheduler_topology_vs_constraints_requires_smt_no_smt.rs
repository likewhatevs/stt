// Pins the `requires_smt` cross-field check arm. `topology`
// declares threads_per_core = 1 (no SMT) but
// `constraints.requires_smt = true` excludes any host without
// SMT — semantically distinct from LLC/CPU/NUMA count gates.
use ktstr::declare_scheduler;
#[allow(unused_imports)]
use ktstr::test_support::TopologyConstraints;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    topology = (1, 2, 4, 1),
    constraints = TopologyConstraints {
        requires_smt: true,
        ..TopologyConstraints::DEFAULT
    },
});

fn main() {}
