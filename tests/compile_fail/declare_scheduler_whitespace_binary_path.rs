// Family-symmetry sibling of declare_scheduler_whitespace_{name,binary}.rs.
// Whitespace-only `binary_path` is routed through `is_visibly_empty`
// (same as empty + invisible), so the rejection diagnostic is the
// unified "visible character" message. Pinned for matrix exhaustiveness
// across the empty/whitespace/invisible × name/binary/binary_path
// validator family.
use ktstr::declare_scheduler;

declare_scheduler!(WHITESPACE_BINARY_PATH, {
    name = "whitespace_binary_path",
    binary_path = "   ",
});

fn main() {}
