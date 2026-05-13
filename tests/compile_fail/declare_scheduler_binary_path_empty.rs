// Empty `binary_path` would resolve as `SchedulerSpec::Path("")` and
// fail at runtime inside `resolve_scheduler`. Reject at macro-expand
// time so the diagnostic lands on the literal.
use ktstr::declare_scheduler;

declare_scheduler!(BINARY_PATH_EMPTY, {
    name = "binary_path_empty",
    binary_path = "",
});

fn main() {}
