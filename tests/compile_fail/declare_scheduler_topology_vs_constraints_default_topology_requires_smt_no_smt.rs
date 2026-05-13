// Pins the `requires_smt` cross-field check arm against the
// default topology. Default threads_per_core = 1 (no SMT);
// `requires_smt = true` excludes any host without SMT.
use ktstr::declare_scheduler;
#[allow(unused_imports)]
use ktstr::test_support::TopologyConstraints;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    constraints = TopologyConstraints {
        requires_smt: true,
        ..TopologyConstraints::DEFAULT
    },
});

fn main() {}
