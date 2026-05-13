// Sibling of declare_scheduler_invisible_{name,binary}.rs. Pins the
// invisible-char rejection in the `binary_path` arm — previously
// bare `is_empty()` allowed Cf-category invisibles through and the
// downstream "must be absolute" check fired with a confusing
// root-cause-obscuring diagnostic.
use ktstr::declare_scheduler;

declare_scheduler!(INVISIBLE_BINARY_PATH, {
    name = "invisible_binary_path",
    binary_path = "\u{200B}\u{FEFF}",
});

fn main() {}
