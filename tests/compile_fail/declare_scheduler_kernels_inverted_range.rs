// Pins the `KernelId.validate()` macro-time branch. `"6.20..6.14"`
// parses as `Range { start: "6.20", end: "6.14" }`, which the
// validator rejects as an inverted range. The macro bails via the
// `parsed.validate()` branch (distinct from the empty-kernel
// branch and the dot-dot CacheKey heuristic).
use ktstr::declare_scheduler;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    kernels = ["6.20..6.14"],
});

fn main() {}
