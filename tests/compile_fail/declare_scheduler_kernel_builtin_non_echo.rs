// The guest interpreter accepts only `echo VALUE > /path` (plus blank
// lines and `#` comments). Other shell syntax (sysctl -w, pipes, `;`,
// etc.) silently no-ops at runtime. The macro catches them at expand
// time so the operator doesn't ship a no-op KernelBuiltin.
use ktstr::declare_scheduler;

declare_scheduler!(KERNEL_BUILTIN_NON_ECHO, {
    name = "kernel_builtin_non_echo",
    kernel_builtin_enable = ["sysctl -w kernel.sched_autogroup_enabled=1"],
    kernel_builtin_disable = ["echo 0 > /proc/sys/kernel/sched_autogroup_enabled"],
});

fn main() {}
