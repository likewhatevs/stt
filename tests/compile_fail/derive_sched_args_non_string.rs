#[derive(ktstr::Scheduler)]
#[scheduler(name = "bad", sched_args = [42])]
#[allow(dead_code)]
enum SchedArgsNonStringFlag {}

fn main() {}
