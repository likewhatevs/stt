// `echo > /path` writes an empty value — valid shell but useless. The
// macro rejects so the operator provides a real value to write.
use ktstr::declare_scheduler;

declare_scheduler!(KERNEL_BUILTIN_EMPTY_VALUE, {
    name = "kernel_builtin_empty_value",
    kernel_builtin_enable = ["echo  > /proc/sys/kernel/sched_autogroup_enabled"],
    kernel_builtin_disable = ["echo 0 > /proc/sys/kernel/sched_autogroup_enabled"],
});

fn main() {}
