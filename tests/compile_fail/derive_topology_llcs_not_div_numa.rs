#[derive(ktstr::Scheduler)]
#[scheduler(name = "bad", topology(2, 3, 2, 1))]
#[allow(dead_code)]
enum LlcsNotDivFlag {}

fn main() {}
