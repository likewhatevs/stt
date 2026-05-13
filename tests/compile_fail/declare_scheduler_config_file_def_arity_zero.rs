// config_file_def tuple-arity check fires on 0-tuple `()` just as
// it does on 3-tuple. Boundary coverage alongside the existing
// wrong_arity (3-tuple) fixture.
use ktstr::declare_scheduler;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    config_file_def = (),
});

fn main() {}
