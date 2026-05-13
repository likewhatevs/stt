// `binary_path` must be a string literal. Non-string values are
// rejected with the standard string-literal diagnostic.
use ktstr::declare_scheduler;

declare_scheduler!(BINARY_PATH_NON_STRING, {
    name = "binary_path_non_string",
    binary_path = 42,
});

fn main() {}
