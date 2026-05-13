// Pins macro-time rejection of `constraints = <call>()`. The emitted
// `pub static` requires a const-evaluable expression; a non-const
// helper call fails with a deep, confusing const-eval diagnostic at
// the spread site. The macro must catch this at expand time and
// point at the const-eligible shapes.
use ktstr::declare_scheduler;
use ktstr::test_support::TopologyConstraints;

fn build_constraints() -> TopologyConstraints {
    TopologyConstraints::DEFAULT
}

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    constraints = build_constraints(),
});

fn main() {}
