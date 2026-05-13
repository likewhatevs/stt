// `kernel_builtin_disable` must be an array literal `[..]`. Same
// shape rule as `kernel_builtin_enable`.
use ktstr::declare_scheduler;

declare_scheduler!(KERNEL_BUILTIN_DISABLE_NON_ARRAY, {
    name = "kernel_builtin_disable_non_array",
    kernel_builtin_enable = ["echo 1 > /proc/sys/kernel/sched_autogroup_enabled"],
    kernel_builtin_disable = "echo 0 > /proc/sys/kernel/sched_autogroup_enabled",
});

fn main() {}
