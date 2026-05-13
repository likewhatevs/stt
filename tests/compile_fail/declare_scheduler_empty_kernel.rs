// Pins macro-time rejection of empty `kernels = [..]` entries. An
// empty string parses as `CacheKey("")` and fails confusingly at
// verifier runtime with "cache key not found"; the macro must reject
// it at expand time so the diagnostic lands on the literal.
use ktstr::declare_scheduler;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    kernels = [""],
});

fn main() {}
