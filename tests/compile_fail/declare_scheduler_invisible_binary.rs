// Sibling of declare_scheduler_invisible_name.rs. Pins the
// invisible-char rejection in the `binary` arm.
use ktstr::declare_scheduler;

declare_scheduler!(INVISIBLE_BINARY, {
    name = "invisible_binary",
    binary = "\u{200B}\u{2060}",
});

fn main() {}
