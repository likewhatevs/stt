// Pins config_file_def per-element string-literal check: the
// guest_path position (element 1) must be a string literal.
use ktstr::declare_scheduler;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    config_file_def = ("--config {file}", 42),
});

fn main() {}
