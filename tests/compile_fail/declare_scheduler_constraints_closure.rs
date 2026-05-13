// Pins the catchall arm of `validate_constraints_expr`. A closure
// expression `|| TopologyConstraints::DEFAULT` parses as
// `Expr::Closure`, which is not const-eligible and falls into the
// catchall arm. The diagnostic is the shorter base message
// (without the call-specific hint).
use ktstr::declare_scheduler;
#[allow(unused_imports)]
use ktstr::test_support::TopologyConstraints;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    constraints = || TopologyConstraints::DEFAULT,
});

fn main() {}
