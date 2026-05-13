// String-name reservation: when the `kernel_builtin_*` pair is set,
// the variant's display_name is the literal `"kernel"` — a
// `name = "kernel"` value collides with that label in failure dumps
// and sidecar comparisons. Case-insensitive, matching the existing
// reservation of `"eevdf"` / `"kernel_default"`.
use ktstr::declare_scheduler;

declare_scheduler!(NAME_KERNEL_WITH_KERNEL_BUILTIN, {
    name = "Kernel",
    kernel_builtin_enable = ["echo 1 > /proc/sys/kernel/sched_autogroup_enabled"],
    kernel_builtin_disable = ["echo 0 > /proc/sys/kernel/sched_autogroup_enabled"],
});

fn main() {}
