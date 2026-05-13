// Pins case-insensitive enforcement of the `name = "kernel_default"`
// reservation — `"KERNEL_DEFAULT"` (uppercase) must be rejected just
// like the lowercase form. Sister fixture to
// `declare_scheduler_reserved_name_kernel_default.rs` (lowercase axis)
// and `declare_scheduler_reserved_name_eevdf_upper.rs` (uppercase
// EEVDF axis). A future regression that drops `to_lowercase()` from
// the declare_scheduler! validator would silently re-allow this form
// and this fixture would start compiling unexpectedly.
use ktstr::declare_scheduler;

declare_scheduler!(MY_SCHED, {
    name = "KERNEL_DEFAULT",
    binary = "scx_my_sched",
});

fn main() {}
