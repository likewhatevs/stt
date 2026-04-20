use ktstr::Payload;

// `TargetValue` as a bare identifier with no argument is ambiguous —
// it needs a float literal, e.g. `TargetValue(50.0)`. The derive
// macro must reject this at expansion time with an actionable error.
#[derive(Payload)]
#[payload(binary = "x")]
#[metric(name = "m", polarity = TargetValue)]
#[allow(dead_code)]
struct BareTargetValuePayload;

fn main() {}
