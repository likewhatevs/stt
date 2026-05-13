// An invalid Rust visibility token (`extern` is not a valid
// visibility keyword) must be rejected at macro-expand time via
// syn::Visibility::parse's natural failure.
use ktstr::declare_scheduler;

declare_scheduler!(extern MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
});

fn main() {}
