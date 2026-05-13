// Pins the Cf-category invisible-char rejection in the `name` arm.
// `str::trim` strips Unicode White_Space but not Cf-category
// invisibles like ZERO WIDTH SPACE (U+200B). A literal containing
// only such chars would slip past the prior trim().is_empty() check
// and surface as a confusing sidecar-lookup failure at runtime.
use ktstr::declare_scheduler;

declare_scheduler!(INVISIBLE_NAME, {
    name = "\u{200B}\u{FEFF}",
    binary = "scx_invisible_name",
});

fn main() {}
