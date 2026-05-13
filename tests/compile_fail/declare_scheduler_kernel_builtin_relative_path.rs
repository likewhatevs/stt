// The guest interpreter writes via `std::fs::write`, which resolves
// relative paths against the guest init's cwd. The macro requires
// absolute paths so the operator's intent is unambiguous.
use ktstr::declare_scheduler;

declare_scheduler!(KERNEL_BUILTIN_RELATIVE_PATH, {
    name = "kernel_builtin_relative_path",
    kernel_builtin_enable = ["echo 1 > relative/path"],
    kernel_builtin_disable = ["echo 0 > /proc/sys/kernel/sched_autogroup_enabled"],
});

fn main() {}
