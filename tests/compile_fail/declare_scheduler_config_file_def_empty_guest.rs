// Pins config_file_def empty-string check on guest_path
// position (element 1). Empty guest_path breaks the `mkdir -p`
// invariant.
use ktstr::declare_scheduler;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    config_file_def = ("--config {file}", ""),
});

fn main() {}
