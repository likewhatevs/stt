// Symmetric to the whitespace_binary fixture: a whitespace-only
// `name` previously passed the post-loop is_empty() check and flowed
// to runtime where sidecar lookups would fail. Inline trim-empty
// rejection lands the caret on the offending literal.
use ktstr::declare_scheduler;

declare_scheduler!(WHITESPACE_NAME, {
    name = "   ",
    binary = "scx_whitespace_name",
});

fn main() {}
