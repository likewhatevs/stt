#[derive(ktstr::Scheduler)]
#[scheduler(name = "bad")]
#[allow(dead_code)]
enum FlagArgsNonStringFlag {
    #[flag(args = [42])]
    Bad,
}

fn main() {}
