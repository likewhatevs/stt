// Pins the `name = "kernel_default"` reservation — second of the
// two reserved string names recognized by declare_scheduler!.
// Sister fixture to `declare_scheduler_reserved_name_eevdf.rs`
// (covers the `Scheduler::EEVDF` arm); this one covers the
// `Payload::KERNEL_DEFAULT` arm. A regression that dropped
// `"kernel_default"` from the reserved set would silently re-allow
// this form and shadow the built-in `Payload::KERNEL_DEFAULT`
// baseline.
use ktstr::declare_scheduler;

declare_scheduler!(MY_SCHED, {
    name = "kernel_default",
    binary = "scx_my_sched",
});

fn main() {}
