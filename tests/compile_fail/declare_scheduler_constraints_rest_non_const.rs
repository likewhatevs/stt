// `validate_constraints_expr` must recurse into the struct
// literal's `..rest` spread. Without recursion, a non-const
// expression in the rest slot (`..build_helper()`) would slip past
// the outer Struct shape and surface as a deep const-eval failure
// at the spread site.
use ktstr::declare_scheduler;
#[allow(unused_imports)]
use ktstr::test_support::TopologyConstraints;

fn build_helper() -> TopologyConstraints {
    TopologyConstraints::DEFAULT
}

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    constraints = TopologyConstraints {
        min_llcs: 1,
        ..build_helper()
    },
});

fn main() {}
