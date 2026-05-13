// Pins the unwrap_parens hardening in `u64_from_option_some_lit`.
// A paren-wrapped `(Some(N))` must still flow through the
// topology-vs-constraints cross-field check; otherwise the macro
// would silently accept infeasible constraints when the operator
// wraps the literal for clarity.
#[allow(unused_imports)]
use ktstr::declare_scheduler;
#[allow(unused_imports)]
use ktstr::test_support::TopologyConstraints;

declare_scheduler!(MAX_LLCS_PAREN_WRAPPED_SOME, {
    name = "max_llcs_paren_wrapped_some",
    binary = "scx-paren-some",
    topology = (1, 4, 2, 1),
    constraints = TopologyConstraints {
        max_llcs: (Some(2)),
        ..TopologyConstraints::DEFAULT
    },
});

fn main() {}
