// #62: an unrecognized constructor inside #[default_check(...)]
// must fail to compile — the macro prepends
// `::ktstr::test_support::Check::` so a typo like
// `nonexistent_check(...)` resolves to
// `::ktstr::test_support::Check::nonexistent_check(...)`, which
// doesn't exist in the `Check` API, producing a rustc E0599 "no
// function named ... in `Check`" against the generated const that
// pins both `Check::` forms (bare + qualified) to the same
// constructor surface.
use ktstr::Payload;

#[derive(Payload)]
#[payload(binary = "bad_check_bin")]
#[default_check(nonexistent_check("metric", 1.0))]
#[allow(dead_code)]
struct BadCheckPayload;

fn main() {}
