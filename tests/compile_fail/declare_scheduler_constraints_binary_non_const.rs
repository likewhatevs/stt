// `validate_constraints_expr` Binary arm must recurse into BOTH
// operands. Without recursion, `min_llcs: build_value() + 1` would
// slip past the outer Binary acceptance.
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
        min_llcs: build_value() + 1,
        ..TopologyConstraints::DEFAULT
    },
});

fn main() {}
