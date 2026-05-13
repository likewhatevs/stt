// `kernel_builtin_enable` must be an array literal `[..]`. A bare
// string (or any non-array shape) is rejected per `expect_array`.
use ktstr::declare_scheduler;

declare_scheduler!(KERNEL_BUILTIN_ENABLE_NON_ARRAY, {
    name = "kernel_builtin_enable_non_array",
    kernel_builtin_enable = "echo 1 > /proc/sys/kernel/sched_autogroup_enabled",
    kernel_builtin_disable = ["echo 0 > /proc/sys/kernel/sched_autogroup_enabled"],
});

fn main() {}
