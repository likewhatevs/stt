#[derive(ktstr::Scheduler)]
#[scheduler(name = "bad")]
#[allow(dead_code)]
enum NonUnitFlag {
    Good,
    Bad(u32),
}

fn main() {}
