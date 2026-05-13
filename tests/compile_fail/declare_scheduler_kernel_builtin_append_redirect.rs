// The guest interpreter only handles single-`>` truncating writes. An
// append (`>>`) silently no-ops at runtime, so the macro rejects it
// at expand time on either `kernel_builtin_enable` or `kernel_builtin_disable`.
use ktstr::declare_scheduler;

declare_scheduler!(KERNEL_BUILTIN_APPEND_REDIRECT, {
    name = "kernel_builtin_append_redirect",
    kernel_builtin_enable = ["echo 1 >> /proc/sys/kernel/sched_autogroup_enabled"],
    kernel_builtin_disable = ["echo 0 > /proc/sys/kernel/sched_autogroup_enabled"],
});

fn main() {}
