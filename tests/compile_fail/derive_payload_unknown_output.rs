use ktstr::Payload;

#[derive(Payload)]
#[payload(binary = "x", output = yaml)]
#[allow(dead_code)]
struct YamlPayload;

fn main() {}
