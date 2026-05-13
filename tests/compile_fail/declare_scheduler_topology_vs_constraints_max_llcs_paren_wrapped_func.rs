// Pins the call.func unwrap_parens hardening. A user who wraps the
// function path itself (`(Some)(N)`) for clarity must still flow
// through the topology-vs-constraints cross-field check; otherwise
// the macro would silently accept infeasible constraints.
#[allow(unused_imports)]
use ktstr::declare_scheduler;
#[allow(unused_imports)]
use ktstr::test_support::TopologyConstraints;

declare_scheduler!(MAX_LLCS_PAREN_WRAPPED_FUNC, {
    name = "max_llcs_paren_wrapped_func",
    binary = "scx-paren-func",
    topology = (1, 4, 2, 1),
    constraints = TopologyConstraints {
        max_llcs: (Some)(2),
        ..TopologyConstraints::DEFAULT
    },
});

fn main() {}
