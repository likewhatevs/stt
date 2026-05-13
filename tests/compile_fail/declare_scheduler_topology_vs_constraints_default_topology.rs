// Pins that the topology-vs-constraints consistency check fires
// even when `topology` is omitted. Without an explicit `topology`,
// the runtime falls back to `Scheduler::new`'s default
// (numa_nodes=1, llcs=1, cores_per_llc=2, threads_per_core=1,
// total_cpus=2). `constraints.min_llcs = 100` requires every
// preset to provide at least 100 LLCs — every gauntlet preset
// would reject this test at runtime and the test would never
// execute. The macro catches the inconsistency at expand time
// against the default topology.
use ktstr::declare_scheduler;
#[allow(unused_imports)]
use ktstr::test_support::TopologyConstraints;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    constraints = TopologyConstraints {
        min_llcs: 100,
        ..TopologyConstraints::DEFAULT
    },
});

fn main() {}
