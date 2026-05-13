// Pins macro-time validation of `kernels = [..]` entries through the
// same `KernelId::parse` + `validate` the verifier uses at runtime.
// `"abc..def"` parses as `CacheKey` (neither endpoint is
// version-shaped, so the Range arm rejects it), but the `..` substring
// reveals the user almost certainly meant a version range that got
// typo'd. Without macro-time validation the verifier fails late with
// a confusing "cache key not found" error; with it, the diagnostic
// surfaces at the call site.
use ktstr::declare_scheduler;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    kernels = ["abc..def"],
});

fn main() {}
