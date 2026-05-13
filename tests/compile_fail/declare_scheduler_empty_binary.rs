// Sister fixture to `declare_scheduler_empty_name.rs` for the
// `binary` field. An empty binary string flows into
// `SchedulerSpec::Discover("")` and fails confusingly at runtime
// inside `build_and_find_binary("")`; the macro must reject it
// at expand time so the error surfaces at the call site.
use ktstr::declare_scheduler;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "",
});

fn main() {}
