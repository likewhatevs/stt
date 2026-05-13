// Pins config_file_def absolute-path check on guest_path
// position (element 1). The framework `mkdir -p`s the parent
// before writing the config; a relative path breaks that
// invariant.
use ktstr::declare_scheduler;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    config_file_def = ("--config {file}", "relative-cfg.json"),
});

fn main() {}
