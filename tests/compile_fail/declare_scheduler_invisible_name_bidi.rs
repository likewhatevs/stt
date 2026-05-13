// Pins the bidi-mark rejection extension to is_visibly_empty. Covers
// LEFT-TO-RIGHT OVERRIDE (U+202D) + RIGHT-TO-LEFT ISOLATE (U+2067) —
// representative Cf-category bidi chars in the 0x202A-E and 0x2066-9
// ranges. Common copy-paste hazards from RTL-language documentation.
use ktstr::declare_scheduler;

declare_scheduler!(INVISIBLE_NAME_BIDI, {
    name = "\u{202D}\u{2067}",
    binary = "scx_invisible_name_bidi",
});

fn main() {}
