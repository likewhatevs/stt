// Mutual exclusion: `binary_path` (Path) and the `kernel_builtin_*`
// pair (KernelBuiltin) cannot stack.
use ktstr::declare_scheduler;

declare_scheduler!(BINARY_PATH_AND_KERNEL_BUILTIN, {
    name = "binary_path_and_kernel_builtin",
    binary_path = "/usr/local/bin/scx_foo",
    kernel_builtin_enable = ["echo 1 > /proc/sys/kernel/sched_autogroup_enabled"],
    kernel_builtin_disable = ["echo 0 > /proc/sys/kernel/sched_autogroup_enabled"],
});

fn main() {}
