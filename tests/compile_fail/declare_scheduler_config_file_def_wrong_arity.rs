// Pins config_file_def arity check: must be exactly 2 elements.
// 3-tuple (or 0/1) is rejected with the actual arity in the
// diagnostic.
use ktstr::declare_scheduler;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    config_file_def = ("arg", "guest", "extra"),
});

fn main() {}
