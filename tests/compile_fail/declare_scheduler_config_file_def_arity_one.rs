// config_file_def tuple-arity check fires on 1-tuple `(a,)`
// boundary just as it does on 3-tuple.
use ktstr::declare_scheduler;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    config_file_def = ("only-one",),
});

fn main() {}
