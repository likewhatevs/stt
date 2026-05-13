// Relative paths are ambiguous between "sibling file" and "discover-by-
// name" intent. The macro rejects with a hint to use `binary = "..."`
// for discovery or an absolute path for explicit files.
use ktstr::declare_scheduler;

declare_scheduler!(BINARY_PATH_RELATIVE, {
    name = "binary_path_relative",
    binary_path = "scx_relative_sched",
});

fn main() {}
