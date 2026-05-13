// Tilde-prefixed paths are not expanded at compile time or by the
// runtime — `path.exists()` checks the literal `~/foo` against the
// filesystem, which never matches. Reject up-front with the fix.
use ktstr::declare_scheduler;

declare_scheduler!(BINARY_PATH_TILDE, {
    name = "binary_path_tilde",
    binary_path = "~/scx_tilde_sched",
});

fn main() {}
