#[derive(ktstr::Scheduler)]
#[scheduler(name = "bad", kargs = [42])]
#[allow(dead_code)]
enum KargsNonStringFlag {}

fn main() {}
