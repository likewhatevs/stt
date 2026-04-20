use ktstr::Payload as PayloadDerive;

// A struct named exactly `Payload` would strip-suffix to the empty
// string and produce an unnameable const. The derive macro rejects
// this at expansion time. Import the derive macro under an alias so
// the struct name below doesn't collide with the `Payload` trait /
// type brought in by ktstr::test_support::Payload.
#[derive(PayloadDerive)]
#[payload(binary = "x")]
#[allow(dead_code)]
struct Payload;

fn main() {}
