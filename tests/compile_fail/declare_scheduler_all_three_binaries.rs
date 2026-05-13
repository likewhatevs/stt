// Mutual exclusion: all three scheduler sources set at once is rejected
// with the pick-exactly-one diagnostic.
use ktstr::declare_scheduler;

declare_scheduler!(ALL_THREE_SOURCES, {
    name = "all_three_sources",
    binary = "scx_foo",
    binary_path = "/usr/local/bin/scx_foo",
    kernel_builtin_enable = ["echo 1 > /proc/sys/kernel/sched_autogroup_enabled"],
    kernel_builtin_disable = ["echo 0 > /proc/sys/kernel/sched_autogroup_enabled"],
});

fn main() {}
