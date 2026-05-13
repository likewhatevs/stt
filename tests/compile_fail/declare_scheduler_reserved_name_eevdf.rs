// Rejects `name = "eevdf"` — distinct from
// `declare_scheduler_reserved_eevdf.rs`, which exercises the const
// identifier reservation. Both axes (const ident + string name) are
// reserved so user code cannot shadow the built-in
// `Scheduler::EEVDF` baseline.
use ktstr::declare_scheduler;

declare_scheduler!(MY_SCHED, {
    name = "eevdf",
    binary = "scx_my_sched",
});

fn main() {}
