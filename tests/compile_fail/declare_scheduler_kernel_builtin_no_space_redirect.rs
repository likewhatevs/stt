// Macro must match guest interpreter exactly. `exec_shell_line` at
// src/vmm/rust_init.rs uses `split_once(" > ")` (literal space-
// greater-space substring). `echo 1>/path` (no spaces around `>`)
// passes a permissive split but the runtime silently no-ops it.
// Macro rejects at expand time to prevent that runtime mismatch.
use ktstr::declare_scheduler;

declare_scheduler!(KERNEL_BUILTIN_NO_SPACE_REDIRECT, {
    name = "kernel_builtin_no_space_redirect",
    kernel_builtin_enable = ["echo 1>/proc/sys/kernel/sched_autogroup_enabled"],
    kernel_builtin_disable = ["echo 0 > /proc/sys/kernel/sched_autogroup_enabled"],
});

fn main() {}
