// Empty `binary_path` would resolve as `SchedulerSpec::Path("")` and
// fail at runtime inside `resolve_scheduler`. Reject at macro-expand
// time so the diagnostic lands on the literal.
use ktstr::declare_scheduler;

declare_scheduler!(EMPTY_BINARY_PATH, {
    name = "empty_binary_path",
    binary_path = "",
});

fn main() {}
