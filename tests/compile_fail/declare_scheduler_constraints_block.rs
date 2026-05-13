// `validate_constraints_expr` Block expressions are rejected
// outright. Block contents can contain let-bindings and arbitrary
// statements that the validator can't walk reliably; operators who
// want a single value can drop the braces (write `100` not
// `{ 100 }`).
use ktstr::declare_scheduler;
#[allow(unused_imports)]
use ktstr::test_support::TopologyConstraints;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    constraints = TopologyConstraints {
        min_llcs: { 100 },
        ..TopologyConstraints::DEFAULT
    },
});

fn main() {}
