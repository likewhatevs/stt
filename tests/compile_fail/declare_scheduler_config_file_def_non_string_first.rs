// Pins config_file_def per-element string-literal check: the
// arg_template position (element 0) must be a string literal.
use ktstr::declare_scheduler;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    config_file_def = (42, "/include-files/cfg.json"),
});

fn main() {}
