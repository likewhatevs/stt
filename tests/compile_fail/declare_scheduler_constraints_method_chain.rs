// Pins the MethodCall arm of `validate_constraints_expr`. A method
// chain like `TopologyConstraints::DEFAULT.clone()` parses as
// `Expr::MethodCall` and must be rejected at expand time with the
// call-hint version of the diagnostic.
use ktstr::declare_scheduler;
#[allow(unused_imports)]
use ktstr::test_support::TopologyConstraints;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    constraints = TopologyConstraints::DEFAULT.clone(),
});

fn main() {}
