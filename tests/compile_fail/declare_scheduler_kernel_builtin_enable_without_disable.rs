// `kernel_builtin_enable` and `kernel_builtin_disable` are paired —
// setting one without the other is always a typo. The macro rejects
// with a hint to add the missing field.
use ktstr::declare_scheduler;

declare_scheduler!(KERNEL_BUILTIN_ENABLE_ONLY_SET, {
    name = "kernel_builtin_enable_only_set",
    kernel_builtin_enable = ["echo 1 > /proc/sys/kernel/sched_autogroup_enabled"],
});

fn main() {}
