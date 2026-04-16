#[derive(ktstr::Scheduler)]
#[scheduler(name = "bad", bogus = "value")]
#[allow(dead_code)]
enum UnknownAttrFlag {}

fn main() {}
