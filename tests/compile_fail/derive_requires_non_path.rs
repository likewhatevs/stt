#[derive(ktstr::Scheduler)]
#[scheduler(name = "bad")]
#[allow(dead_code)]
enum RequiresNonPathFlag {
    Alpha,
    #[flag(requires = [42])]
    Beta,
}

fn main() {}
