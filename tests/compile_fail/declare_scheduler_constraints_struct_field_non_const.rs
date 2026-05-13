// `validate_constraints_expr` must recurse into struct-literal
// field values. Without the recursion, an outer
// `TopologyConstraints { .. }` shape would accept any non-const
// expression inside a field — defeating the const-eval guard.
use ktstr::declare_scheduler;
#[allow(unused_imports)]
use ktstr::test_support::TopologyConstraints;

fn build_value() -> u32 {
    4
}

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    constraints = TopologyConstraints {
        min_llcs: build_value(),
        ..TopologyConstraints::DEFAULT
    },
});

fn main() {}
