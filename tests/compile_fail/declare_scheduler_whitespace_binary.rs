// Pins the whitespace-only binary rejection. Previously the
// `lit.is_empty()` check accepted `" "` (single space) which flowed
// to runtime as `SchedulerSpec::Discover(" ")` and failed
// confusingly inside `build_and_find_binary(" ")`. The
// `lit.trim().is_empty()` check rejects empty AND whitespace-only.
use ktstr::declare_scheduler;

declare_scheduler!(WHITESPACE_BINARY, {
    name = "whitespace_binary",
    binary = "   ",
});

fn main() {}
