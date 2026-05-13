// `validate_constraints_expr` Call arm must recurse into
// `Some(...)` constructor args. Without arg recursion, the outer
// PascalCase shape acceptance would let `Some(non_const_call())`
// slip past the const-eval guard.
use ktstr::declare_scheduler;
#[allow(unused_imports)]
use ktstr::test_support::TopologyConstraints;

fn my_helper() -> u32 {
    4
}

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    constraints = TopologyConstraints {
        max_llcs: Some(my_helper()),
        ..TopologyConstraints::DEFAULT
    },
});

fn main() {}
