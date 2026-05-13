// Pins case-insensitive enforcement of the `name = "eevdf"`
// reservation — `"EEVDF"` (uppercase) must be rejected just like the
// lowercase form. Sister fixture to
// `declare_scheduler_reserved_name_eevdf.rs` (lowercase axis).
// A future regression that drops `to_lowercase()` from the
// declare_scheduler! validator would silently re-allow this form
// and this fixture would start compiling unexpectedly.
use ktstr::declare_scheduler;

declare_scheduler!(MY_SCHED, {
    name = "EEVDF",
    binary = "scx_my_sched",
});

fn main() {}
