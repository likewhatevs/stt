// Reverse of the enable-only case: setting `kernel_builtin_disable`
// without `kernel_builtin_enable` is rejected with a hint to add the
// missing field.
use ktstr::declare_scheduler;

declare_scheduler!(KERNEL_BUILTIN_DISABLE_ONLY_SET, {
    name = "kernel_builtin_disable_only_set",
    kernel_builtin_disable = ["echo 0 > /proc/sys/kernel/sched_autogroup_enabled"],
});

fn main() {}
