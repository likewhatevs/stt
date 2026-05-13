// Pins the SOFT HYPHEN (U+00AD) rejection extension to is_visibly_empty.
// SHY is the most common Cf-category invisible in browser-wrapped
// text (line-break hyphenation hint). Without explicit coverage the
// initial 5-char allowlist would silently accept it.
use ktstr::declare_scheduler;

declare_scheduler!(INVISIBLE_NAME_SHY, {
    name = "\u{00AD}",
    binary = "scx_invisible_name_shy",
});

fn main() {}
