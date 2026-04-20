use ktstr::Payload;

#[derive(Payload)]
#[payload(name = "no_binary")]
#[allow(dead_code)]
struct NoBinaryPayload;

fn main() {}
