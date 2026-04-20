use ktstr::Payload;

#[derive(Payload)]
#[payload(binary = "x")]
#[allow(dead_code)]
enum NotAStruct {
    A,
    B,
}

fn main() {}
