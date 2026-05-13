// Pins the `max_llcs` cross-field check arm against the default
// topology (when `topology` field is omitted). Default llcs = 1;
// `constraints.max_llcs = Some(0)` excludes any host with 1 or
// more LLCs.
use ktstr::declare_scheduler;
#[allow(unused_imports)]
use ktstr::test_support::TopologyConstraints;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    constraints = TopologyConstraints {
        max_llcs: Some(0),
        ..TopologyConstraints::DEFAULT
    },
});

fn main() {}
