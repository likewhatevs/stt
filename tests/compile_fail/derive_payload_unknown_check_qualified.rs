// A Check-prefixed constructor that isn't real also fails to
// compile. When the user writes `Check::nonexistent_check(...)`, the
// macro sees the prefix via `expr_has_check_prefix`, skips its
// implicit `::ktstr::test_support::Check::` prepend, and emits the
// user's path verbatim — the same E0599 "no function named ... in
// `Check`" still fires. Pairs with `derive_payload_unknown_check.rs`
// (bare form); both must bail, neither may silently resolve to
// something else.
use ktstr::Payload;
use ktstr::test_support::Check;

#[derive(Payload)]
#[payload(binary = "bad_qualified_check_bin")]
#[default_check(Check::nonexistent_check("metric", 1.0))]
#[allow(dead_code)]
struct BadQualifiedCheckPayload;

fn main() {}
