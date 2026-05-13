// Pins config_file_def empty-string check on arg_template
// position (element 0). Empty arg_template produces a malformed
// scheduler invocation.
use ktstr::declare_scheduler;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    config_file_def = ("", "/include-files/cfg.json"),
});

fn main() {}
