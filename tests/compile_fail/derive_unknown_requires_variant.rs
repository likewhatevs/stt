#[derive(ktstr::Scheduler)]
#[scheduler(name = "bad")]
#[allow(dead_code)]
enum UnknownReqFlag {
    Alpha,
    #[flag(requires = [NonExistent])]
    Beta,
}

fn main() {}
