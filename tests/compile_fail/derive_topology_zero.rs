#[derive(ktstr::Scheduler)]
#[scheduler(name = "bad", topology(0, 1, 2, 1))]
#[allow(dead_code)]
enum TopoZeroFlag {}

fn main() {}
