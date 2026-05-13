// Pins config_file_def `{file}` placeholder check: arg_template
// must contain the `{file}` substring. Without it, the runtime
// substitution path has no placeholder to anchor the guest path
// at, breaking dispatch.
use ktstr::declare_scheduler;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    config_file_def = ("--config", "/include-files/cfg.json"),
});

fn main() {}
