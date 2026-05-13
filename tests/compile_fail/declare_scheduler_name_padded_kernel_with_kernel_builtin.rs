// Pins the trim-before-reservation hardening for the KernelBuiltin
// `name = "kernel"` collision check. Whitespace-padded variants
// (`"  kernel  "`) bypassed the prior bare-to_lowercase() check
// because the padding survived the lowercase pass. Now symmetric
// with the inline `name` arm reservation: trim+lowercase normalizes
// both whitespace and case so the variant-label collision check
// holds regardless of how the user formatted the literal.
use ktstr::declare_scheduler;

declare_scheduler!(NAME_PADDED_KERNEL_WITH_KERNEL_BUILTIN, {
    name = "  Kernel  ",
    kernel_builtin_enable = ["echo 1 > /proc/sys/kernel/sched_autogroup_enabled"],
    kernel_builtin_disable = ["echo 0 > /proc/sys/kernel/sched_autogroup_enabled"],
});

fn main() {}
