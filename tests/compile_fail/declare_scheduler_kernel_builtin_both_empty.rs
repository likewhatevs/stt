// Both arrays empty is functionally identical to the kernel-default
// baseline but registers as KernelBuiltin — silently misleading. The
// macro suggests `Scheduler::EEVDF` instead.
use ktstr::declare_scheduler;

declare_scheduler!(KERNEL_BUILTIN_BOTH_EMPTY, {
    name = "kernel_builtin_both_empty",
    kernel_builtin_enable = [],
    kernel_builtin_disable = [],
});

fn main() {}
