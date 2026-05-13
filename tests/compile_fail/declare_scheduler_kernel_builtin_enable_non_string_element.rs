// `kernel_builtin_enable` elements must each be string literals.
// Non-string elements are rejected per `expect_str_lit_element`.
use ktstr::declare_scheduler;

declare_scheduler!(KERNEL_BUILTIN_ENABLE_NON_STRING_ELEMENT, {
    name = "kernel_builtin_enable_non_string_element",
    kernel_builtin_enable = [42, "echo 1 > /proc/sys/kernel/sched_autogroup_enabled"],
    kernel_builtin_disable = ["echo 0 > /proc/sys/kernel/sched_autogroup_enabled"],
});

fn main() {}
