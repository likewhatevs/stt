// Pins config_file_def shape check: must be a 2-tuple, not a
// bare string. The diagnostic names the expected shape with an
// example.
use ktstr::declare_scheduler;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    config_file_def = "single-string-not-a-tuple",
});

fn main() {}
